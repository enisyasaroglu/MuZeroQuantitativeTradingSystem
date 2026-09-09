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
largest max_assets weights, then iteratively redistribute any
sum-to-one shortfall/excess only among assets still strictly inside
their bounds -- is standard practice for cardinality-constrained
metaheuristics, and is exactly the role a per-iteration repair step
plays in the broader cardinality-constrained-portfolio literature the
Swarm Intelligence dissertation cites. It is NOT guaranteed to reach
exact full investment when max_assets * max_weight < 1 -- that
combination is genuinely infeasible (e.g. max_assets=2, max_weight=0.3
can sum to at most 0.6), and the loop correctly stops (via the
`if not np.any(free): break` guard) rather than looping forever or
violating max_weight to force a sum of 1.

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
        mask = None
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

            # Iteratively redistribute excess/shortfall to satisfy bounds
            # and full investment, only among assets still strictly inside
            # their bounds. TOLERANCE: 1e-6, not 1e-9 -- 1e-9 is close
            # enough to float64 epsilon that, for some inputs, the
            # remaining gap after a redistribution step could shrink
            # asymptotically without ever closing within tolerance,
            # spinning the loop for its full 100-iteration budget on
            # every call (confirmed directly: a multi-asset PSO run hung
            # with the interpreter stack sitting inside this exact loop).
            allowed_mask = np.ones(n, dtype=bool) if keep_idx is None else mask
            converged = False
            for _ in range(100):
                total = weights.sum()
                if abs(total - 1.0) < 1e-6:
                    converged = True
                    break
                diff = 1.0 - total
                if diff > 0:
                    free = allowed_mask & (weights < self.max_weight - 1e-12)
                else:
                    free = allowed_mask & (weights > self.min_weight + 1e-12)

                if not np.any(free):
                    break  # genuinely infeasible combination -- see module docstring

                weights[free] += diff / np.sum(free)
                weights[allowed_mask] = np.clip(weights[allowed_mask], self.min_weight, self.max_weight)

            if not converged:
                # Explicit fallback so this loop can NEVER silently exhaust
                # its iteration budget without terminating: plain
                # renormalisation over the allowed assets is not a perfect
                # constraint satisfaction, but it guarantees termination
                # and a sum very close to 1, rather than hanging.
                total = weights[allowed_mask].sum()
                if total > 1e-12:
                    weights[allowed_mask] = weights[allowed_mask] / total * 1.0

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