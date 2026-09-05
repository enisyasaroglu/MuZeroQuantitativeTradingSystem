"""
Classical Markowitz Mean-Variance portfolio optimisation.

Maximises w'mu - (lambda/2) w'Sigma w subject to full investment
(sum(w)=1), using the closed-form unconstrained solution:
    w* = (1/lambda) * Sigma^-1 * mu, renormalised to sum to 1

then passes the result through constraints.project() for any
bound/cardinality constraints, exactly like every other method in this
package (PSO, EqualWeight).

WHY CLOSED-FORM RATHER THAN A QP SOLVER: avoids adding a new dependency
(cvxpy) for what this method needs to serve right now -- (1) a
general-purpose benchmark optimizer, and (2) the "convert posterior
returns to weights" step Black-Litterman's own sanity check depends on.
A true constrained QP solve is a real, separate upgrade path if
projection-after-the-fact ever proves insufficient -- not implemented
here because nothing yet requires it.

HONEST CAVEAT: closed-form-then-project is NOT equivalent to solving the
constrained problem directly. Under loose constraints (the default:
long-only, full-investment, no cardinality) the two agree, because the
unconstrained optimum already lands inside the feasible set. Under a
TIGHT constraint (e.g. a cardinality limit), projection can move the
answer away from what a true constrained solve would find -- this is
precisely the scenario PSO's IN-LOOP projection handles better than a
single-shot closed-form-then-project ever could, which is a large part
of why constrained portfolio problems motivate metaheuristics in the
first place.
"""
import numpy as np

from src.portfolio.portfolio import OptimizationResult


class MeanVarianceOptimizer:
    def __init__(self, risk_aversion: float = 1.0, ridge: float = 1e-8):
        """
        risk_aversion (lambda): higher values weight risk more heavily
            relative to return. 1.0 is a simple, uncalibrated default.
        ridge: small value added to the covariance diagonal before
            inversion (Sigma + ridge*I), guarding against a singular or
            near-singular covariance estimate -- e.g. a short lookback
            window, or two near-perfectly-correlated assets.
        """
        self.risk_aversion = risk_aversion
        self.ridge = ridge

    def optimize(self, inputs, constraints) -> OptimizationResult:
        if inputs.expected_returns is None or inputs.cov_matrix is None:
            raise ValueError("MeanVarianceOptimizer requires expected_returns and cov_matrix in OptimizationInputs.")

        mu = np.asarray(inputs.expected_returns, dtype=np.float64)
        sigma = np.asarray(inputs.cov_matrix, dtype=np.float64)
        n = len(mu)

        sigma_reg = sigma + self.ridge * np.eye(n)
        sigma_inv = np.linalg.pinv(sigma_reg)  # pinv, not inv: safe even if ridge alone doesn't fix conditioning

        raw_weights = (1.0 / self.risk_aversion) * (sigma_inv @ mu)

        # Renormalise BEFORE constraint projection -- the closed-form
        # solution's scale depends on risk_aversion and isn't guaranteed
        # to sum to 1 on its own. Doing this explicitly (rather than
        # relying on constraints.project()'s own renormalisation) keeps
        # "the closed-form MVO weight" well-defined on its own, since the
        # Black-Litterman sanity check depends on this exact quantity.
        total = raw_weights.sum()
        if abs(total) > 1e-12:
            raw_weights = raw_weights / total

        weights = constraints.project(raw_weights)

        return OptimizationResult(
            weights=weights,
            method_name="MeanVariance",
            converged=True,  # closed-form -- always "converged"; see module
                              # docstring for what this does NOT guarantee
                              # once constraints stop being loose
            diagnostics={"risk_aversion": self.risk_aversion},
        )