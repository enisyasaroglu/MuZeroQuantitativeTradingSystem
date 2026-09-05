"""
Two things live here, deliberately kept separate:

1. A shared input/output/interface contract every optimisation method
   satisfies -- OptimizationInputs, OptimizationResult,
   PortfolioOptimizerProtocol.
2. Portfolio: owns current weights and rebalance timing/caching. This is
   the responsibility currently living INSIDE MultiAssetTradingEnv's
   _current_pso_template()/_pso_template_cache -- see the PSO/portfolio
   architecture review for why that's a boundary violation once more
   than one method exists. Portfolio calls an injected optimizer; it
   implements no optimisation logic itself.

NOT wired into multi_asset_env.py in this change -- that environment is
existing, working, tested code, and swapping its internal PSO-caching
logic for this class is a separate, deliberate follow-up step, not a
side-effect of adding this file.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
import numpy as np

from src.portfolio.constraints import PortfolioConstraints


@dataclass
class OptimizationInputs:
    """
    Optional-field bundle so one call signature serves methods with
    different data needs, rather than forcing every method to accept
    parameters it can't use -- Equal Weight needs none of these; CVaR
    needs returns_sample, which Mean-Variance doesn't.
    """
    expected_returns: Optional[np.ndarray] = None   # shape (n_assets,)
    cov_matrix: Optional[np.ndarray] = None          # shape (n_assets, n_assets)
    returns_sample: Optional[np.ndarray] = None      # shape (n_obs, n_assets) -- CVaR/robust methods


@dataclass
class OptimizationResult:
    weights: np.ndarray
    method_name: str
    converged: bool
    # For exact convex solvers, True means "solver reported optimality".
    # For metaheuristics (PSO/ABC/ACO), True means "ran its full
    # iteration budget" -- NOT "reached the global optimum". These are
    # different guarantees; diagnostics below is where that distinction
    # should actually be recorded, not collapsed into this one flag.
    diagnostics: dict = field(default_factory=dict)


@runtime_checkable
class PortfolioOptimizerProtocol(Protocol):
    """
    Structural interface (typing.Protocol, not an ABC) so methods with
    genuinely different internals -- e.g. HRP, which ignores
    expected_returns entirely -- can satisfy this shape without being
    forced into a shared base class. Black-Litterman deliberately does
    NOT implement this: it PRODUCES an expected_returns estimate, it
    doesn't consume one, so it belongs on a separate, smaller interface
    that a Mean-Variance implementation can optionally read from.
    """
    def optimize(
        self, inputs: OptimizationInputs, constraints: PortfolioConstraints
    ) -> OptimizationResult:
        ...


class Portfolio:
    """Owns current weights and WHEN to call the optimizer. Implements no
    optimisation logic itself -- `optimizer` is any object satisfying
    PortfolioOptimizerProtocol, injected at construction."""

    def __init__(
        self,
        n_assets: int,
        optimizer: PortfolioOptimizerProtocol,
        constraints: PortfolioConstraints,
        rebalance_every: int = 20,
        lookback: int = 60,
    ):
        self.n_assets = n_assets
        self.optimizer = optimizer
        self.constraints = constraints
        self.rebalance_every = rebalance_every
        self.lookback = lookback

        self.current_weights = np.zeros(n_assets)
        self._cached_target = None
        self.last_result: Optional[OptimizationResult] = None

    def reset(self):
        """Call once per episode, alongside env.reset()."""
        self.current_weights = np.zeros(self.n_assets)
        self._cached_target = None
        self.last_result = None

    def propose_weights(self, t: int, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Returns the target weight vector for step t WITHOUT committing it
        to current_weights. Callers should compute turnover/cost against
        the pre-commit current_weights (see turnover() below), THEN call
        commit(target) once the transition is actually taken -- mirrors
        how MultiAssetTradingEnv.step() already sequences this for PSO.

        returns_matrix: (n_timesteps, n_assets) RAW realised returns
        (not an observation panel) -- same requirement PSO's own
        rolling_rebalance_weights already documents.
        """
        if t < self.lookback:
            return self.current_weights.copy()

        should_rebalance = (
            self._cached_target is None
            or (t - self.lookback) % self.rebalance_every == 0
        )
        if should_rebalance:
            window = returns_matrix[t - self.lookback : t]
            inputs = OptimizationInputs(
                expected_returns=window.mean(axis=0),
                cov_matrix=np.cov(window, rowvar=False),
                returns_sample=window,
            )
            result = self.optimizer.optimize(inputs, self.constraints)
            self._cached_target = result.weights
            self.last_result = result

        return self._cached_target

    def turnover(self, target_weights: np.ndarray) -> float:
        """L1 distance between the current (pre-commit) weights and a
        proposed target -- the same turnover definition already used in
        multi_asset_env.py's transaction-cost model."""
        return float(np.abs(np.asarray(target_weights) - self.current_weights).sum())

    def commit(self, weights: np.ndarray):
        """Advances current_weights to `weights` -- call AFTER computing
        turnover()/cost against the pre-commit value."""
        self.current_weights = np.asarray(weights, dtype=np.float64)