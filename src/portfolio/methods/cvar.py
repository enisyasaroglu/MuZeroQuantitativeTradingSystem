"""
CVaR (Conditional Value at Risk) portfolio optimisation, via the
Rockafellar & Uryasev (2000) linear-programming formulation.

Unlike Mean-Variance/Risk Parity, CVaR needs a SAMPLE of realised
returns (OptimizationInputs.returns_sample), not a covariance matrix --
covariance implicitly assumes a roughly Gaussian, symmetric risk
distribution, which says nothing about tail-loss severity. CVaR is
computed directly from the empirical loss distribution instead, so it
captures fat tails / skew that a Mean-Variance objective is blind to by
construction. This is exactly the distinction raised when
OptimizationInputs was first designed: CVaR needing a different input
shape than Mean-Variance is why that bundle has optional fields rather
than one fixed signature.

FORMULATION: minimise CVaR_alpha(w) = zeta + (1/((1-alpha)*T)) *
sum_t [L_t - zeta]_+, where L_t = -w'r_t (portfolio loss at scenario t)
and zeta is a free decision variable (the LP's optimal zeta is exactly
the portfolio's Value-at-Risk at level alpha, as a by-product, not
something separately estimated). Made linear by introducing one
auxiliary variable u_t >= max(0, L_t - zeta) per return scenario:
    minimise  zeta + (1/((1-alpha)*T)) * sum(u_t)
    s.t.      u_t >= -w'r_t - zeta   for every t
              u_t >= 0
              sum(w) = 1  (if full_investment)
              min_weight <= w_i <= max_weight

Solved via scipy.optimize.linprog (method="highs") -- an exact LP
solver, not a metaheuristic or closed-form approximation. Chosen over
adding cvxpy as a new dependency: scipy is already required (Risk
Parity's SLSQP solve, HRP's clustering), and this problem is genuinely a
plain LP once the auxiliary u_t variables are introduced -- no need for
a more general convex-solver library just to express it. No new
requirements.txt entry needed.

Cardinality (max_assets) is NOT expressible as a linear constraint (it's
an integer/combinatorial restriction -- a true cardinality-constrained
CVaR problem is a mixed-integer LP, out of scope here). constraints.
project() is applied to the LP's solution afterward for that case, with
the same honest caveat as mean_variance.py/risk_parity.py: this is NOT
equivalent to solving the cardinality-constrained problem directly.
"""
import numpy as np
from scipy.optimize import linprog

from src.portfolio.portfolio import OptimizationResult


class CVaROptimizer:
    def __init__(self, alpha: float = 0.95):
        """
        alpha: confidence level. 0.95 means optimising the average loss
        in the worst 5% of scenarios in returns_sample -- the standard
        choice in the Rockafellar-Uryasev literature, not calibrated
        for this project specifically.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be strictly between 0 and 1.")
        self.alpha = alpha

    def optimize(self, inputs, constraints) -> OptimizationResult:
        if inputs.returns_sample is None:
            raise ValueError("CVaROptimizer requires returns_sample in OptimizationInputs.")

        returns = np.asarray(inputs.returns_sample, dtype=np.float64)
        T, n = returns.shape

        # Variable vector: [w_1..w_n, zeta, u_1..u_T]
        n_vars = n + 1 + T

        c = np.zeros(n_vars)
        c[n] = 1.0                                 # zeta
        c[n + 1:] = 1.0 / ((1.0 - self.alpha) * T)  # u_t coefficients

        # Inequality constraints: for each t, -r_t . w - zeta - u_t <= 0
        A_ub = np.zeros((T, n_vars))
        A_ub[:, :n] = -returns
        A_ub[:, n] = -1.0
        A_ub[np.arange(T), n + 1 + np.arange(T)] = -1.0
        b_ub = np.zeros(T)

        bounds = [(constraints.min_weight, constraints.max_weight)] * n  # w_i
        bounds.append((None, None))       # zeta: unbounded -- it's an estimated VaR, not itself constrained
        bounds.extend([(0.0, None)] * T)  # u_t >= 0

        A_eq, b_eq = None, None
        if constraints.full_investment:
            A_eq = np.zeros((1, n_vars))
            A_eq[0, :n] = 1.0
            b_eq = np.array([1.0])

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")

        if result.success:
            raw_weights = result.x[:n]
            var_estimate = float(result.x[n])
        else:
            # LP infeasible/failed (e.g. contradictory bounds) -- fall
            # back to equal weight rather than propagating a garbage
            # solution or crashing. Recorded explicitly in diagnostics,
            # not silently swallowed.
            raw_weights = np.full(n, 1.0 / n)
            var_estimate = None

        weights = constraints.project(raw_weights)

        return OptimizationResult(
            weights=weights,
            method_name="CVaR",
            converged=bool(result.success),
            diagnostics={
                "alpha": self.alpha,
                "var_estimate": var_estimate,
                "cvar_estimate": float(result.fun) if result.success else None,
                "solver_status": result.message,
            },
        )