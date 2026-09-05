"""
Equal-weight portfolio construction: 1/N across all assets.

The trivial baseline, and a genuinely competitive one -- DeMiguel,
Garlappi & Uppal (2009) is the well-known empirical result that naive
1/N frequently matches or beats "optimised" portfolios out-of-sample,
because it needs no expected-return or covariance ESTIMATE at all (zero
estimation error), unlike every other method on this list. Worth
treating as a first-class benchmark, not a placeholder.

Ignores expected_returns, cov_matrix, and returns_sample entirely by
design -- Group C from the portfolio-methods architecture review: same
interface shape as PSO/Mean-Variance, deliberately narrower inputs.
"""
import numpy as np

from src.portfolio.portfolio import OptimizationResult


class EqualWeightOptimizer:
    def optimize(self, inputs, constraints) -> OptimizationResult:
        # n_assets is read from whichever input field happens to be
        # populated -- none of their ECONOMIC content is used, only
        # their shape, since this method has no notion of return/risk.
        n_assets = None
        for field in (inputs.expected_returns, inputs.cov_matrix, inputs.returns_sample):
            if field is not None:
                n_assets = len(field) if field.ndim == 1 else field.shape[-1]
                break
        if n_assets is None:
            raise ValueError(
                "EqualWeightOptimizer needs at least one OptimizationInputs "
                "field populated, to determine n_assets."
            )

        raw_weights = np.full(n_assets, 1.0 / n_assets)
        # Passed through constraints.project() for consistency with every
        # other method, even though 1/n satisfies the DEFAULT constraint
        # set trivially -- a custom max_assets/max_weight could still make
        # raw 1/n infeasible, and this keeps that case handled correctly
        # rather than silently assumed away.
        weights = constraints.project(raw_weights)

        return OptimizationResult(
            weights=weights,
            method_name="EqualWeight",
            converged=True,  # deterministic, closed-form -- always "converged"
            diagnostics={},
        )