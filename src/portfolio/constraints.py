"""
Shared portfolio constraints: the feasible set every optimisation method
(PSO today; Mean-Variance, HRP, Risk Parity, CVaR, Robust, etc. later)
reads from, rather than each method hardcoding its own bounds -- this is
what closes the gap flagged in the PSO/portfolio architecture review:
PSOPortfolioOptimizer previously had no constraints parameter at all,
just a fixed no-short/full-investment projection baked into the method.

Scoped to what's actually needed today: long-only-by-default bounds, an
optional cardinality limit, and full-investment. NOT included:
sector/group exposure limits, turnover limits, leverage -- add these
when a method actually needs them, not speculatively.

project() is a HEURISTIC REPAIR, not an exact projection. Projecting
onto a box + simplex + cardinality feasible set exactly is combinatorial
in general (choosing WHICH assets to zero out to stay optimal is itself
an optimisation problem). The heuristic here -- clip to bounds, keep the
largest max_assets weights, zero the rest, renormalise -- is standard
practice for cardinality-constrained metaheuristics, and is exactly the
role a per-iteration repair step plays in the broader
cardinality-constrained-portfolio literature the Swarm Intelligence
dissertation cites. It is NOT guaranteed to satisfy max_weight exactly
after renormalisation when max_assets forces a small feasible set --
documented in project()'s body, not silently swept under the rug.

Cardinality selection ranks assets by RAW (signed) weight value, which
is appropriate for the long-only default (min_weight >= 0) used
throughout this project. A long-short use case would need
magnitude-based ranking instead -- not implemented, since nothing
currently in scope needs it.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class PortfolioConstraints:
    min_weight: float = 0.0            # long-only by default (no short-selling)
    max_weight: float = 1.0            # no per-asset concentration limit by default
    max_assets: Optional[int] = None   # cardinality limit; None = unconstrained
    full_investment: bool = True       # weights must sum to 1

    def __post_init__(self):
        if self.min_weight > self.max_weight:
            raise ValueError(f"min_weight ({self.min_weight}) cannot exceed max_weight ({self.max_weight}).")
        if self.max_assets is not None and self.max_assets <= 0:
            raise ValueError("max_assets must be a positive integer when set.")

    def project(self, weights: np.ndarray) -> np.ndarray:
        """Heuristic repair of an arbitrary weight vector into this
        constraint set. See module docstring for what "heuristic" means
        here and why an exact projection isn't attempted."""
        weights = np.asarray(weights, dtype=np.float64).copy()
        n = len(weights)

        weights = np.clip(weights, self.min_weight, self.max_weight)

        keep_idx = None
        if self.max_assets is not None and self.max_assets < n:
            keep_idx = np.argsort(weights)[::-1][: self.max_assets]
            mask = np.zeros(n, dtype=bool)
            mask[keep_idx] = True
            weights = np.where(mask, weights, 0.0)

        if self.full_investment:
            total = weights.sum()
            if total <= 1e-12:
                # Degenerate: nothing survived clipping/cardinality selection.
                # Fall back to equal weight over whichever assets cardinality
                # allows (or all assets, if unconstrained).
                fallback = np.zeros(n)
                allowed = np.arange(n) if keep_idx is None else keep_idx
                fallback[allowed] = 1.0 / len(allowed)
                return fallback
            weights = weights / total
            # NOTE: renormalising after a cardinality cut can push an
            # individual weight back above max_weight -- e.g. max_weight=0.3
            # with max_assets=2 cannot be jointly satisfied while also
            # summing to 1 (2*0.3=0.6 < 1). Callers setting both should
            # ensure max_assets * max_weight >= 1.

        return weights

    def is_feasible(self, weights: np.ndarray, tol: float = 1e-6) -> bool:
        """Checks (does not repair) whether `weights` satisfies this
        constraint set, within numerical tolerance `tol`."""
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights < self.min_weight - tol) or np.any(weights > self.max_weight + tol):
            return False
        if self.max_assets is not None and np.sum(weights > tol) > self.max_assets:
            return False
        if self.full_investment and abs(weights.sum() - 1.0) > tol:
            return False
        return True