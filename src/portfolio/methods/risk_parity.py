"""
Risk Parity (Equal Risk Contribution): each asset contributes equally to
total portfolio variance, rather than being weighted by expected return
at all. Group B from the portfolio-methods architecture review -- same
PortfolioOptimizerProtocol shape as Mean-Variance/PSO, but DELIBERATELY
IGNORES expected_returns entirely, same as EqualWeightOptimizer and (once
built) HRP.

No closed-form solution exists in general (unlike Mean-Variance) --
solved numerically via scipy.optimize.minimize (SLSQP), matching what
was proposed when this method was sequenced. Introduces scipy as a
genuinely new dependency here -- see the requirements.txt note below;
NOT adding it there would repeat the exact missing-dependency bug
already found twice this session (finrl, hmmlearn).

HONEST CAVEAT, same shape as Mean-Variance's: constraints.project() is
applied to the FINAL numerical solution, not enforced throughout the
search itself. Under the default constraints (long-only, full
investment, no cardinality) this is immaterial. Under a tight
cardinality limit, post-hoc projection is not equivalent to solving the
constrained ERC problem directly -- same limitation as Mean-Variance,
for the same reason: unlike PSO, this isn't a population-based search
that can enforce constraints at every step.
"""
import numpy as np
from scipy.optimize import minimize

from src.portfolio.portfolio import OptimizationResult


class RiskParityOptimizer:
    def __init__(self, max_iter: int = 500, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _risk_contributions(weights, cov_matrix):
        """RC_i = w_i * (Sigma @ w)_i -- asset i's contribution to total
        portfolio variance w'Sigma w. Sums to total portfolio variance
        exactly, by Euler's theorem for this homogeneous-degree-2 form."""
        portfolio_var = weights @ cov_matrix @ weights
        marginal = cov_matrix @ weights
        return weights * marginal, portfolio_var

    def _objective(self, weights, cov_matrix):
        """Sum of squared deviations between each asset's risk
        contribution and the equal-contribution target
        (total_variance / n) -- zero exactly at the ERC solution."""
        rc, port_var = self._risk_contributions(weights, cov_matrix)
        n = len(weights)
        target = port_var / n
        return np.sum((rc - target) ** 2)

    def optimize(self, inputs, constraints) -> OptimizationResult:
        if inputs.cov_matrix is None:
            raise ValueError("RiskParityOptimizer requires cov_matrix in OptimizationInputs.")

        sigma = np.asarray(inputs.cov_matrix, dtype=np.float64)
        n = sigma.shape[0]

        x0 = np.full(n, 1.0 / n)  # equal weight -- a neutral, reasonable starting point
        # Small epsilon above min_weight is a numerical-stability nicety
        # (avoids the optimizer stepping into an exact-zero weight early,
        # which flattens the local gradient), not a mathematical
        # requirement of the ERC objective itself.
        bounds = [(max(constraints.min_weight, 1e-8), constraints.max_weight) for _ in range(n)]
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}] if constraints.full_investment else []

        result = minimize(
            self._objective, x0, args=(sigma,), method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        weights = constraints.project(result.x)

        return OptimizationResult(
            weights=weights,
            method_name="RiskParity",
            converged=bool(result.success),
            diagnostics={"objective_value": float(result.fun), "n_iter": result.nit},
        )