"""
Black-Litterman: NOT a PortfolioOptimizerProtocol member. It produces a
posterior EXPECTED RETURN VECTOR (and posterior covariance) by
Bayesian-blending a market-equilibrium prior with investor views -- it
does not itself produce portfolio weights. Converting its output into
weights is Mean-Variance's job, matching the standard institutional
Black-Litterman -> Mean-Variance pipeline, and matching the earlier
architectural decision that BL doesn't belong on the same interface as
PSO/MVO/HRP since it produces one of THEIR inputs, not a weight vector.

MARKET-CAP WEIGHTS -- AN HONEST GAP: the classical formulation computes
the equilibrium prior as pi = lambda * Sigma @ w_market, where w_market
is the TRUE market-cap-weighted portfolio. This project has no
market-cap data anywhere in its pipeline (only OHLCV). w_market defaults
to EQUAL WEIGHT here, as an explicit, documented proxy -- not a claim
this is a real market portfolio.

SANITY CHECK (the reason empty views are supported at all): with zero
views, the posterior return vector must equal the prior EXACTLY -- a
direct algebraic property of the BL update (the view-dependent terms
vanish when there are no views), not something tuned to match. Running
that zero-view output through MeanVarianceOptimizer (same risk_aversion)
must then recover market_weights exactly, since MVO's closed-form
solution w*=(1/lambda)*Sigma^-1*mu with mu=lambda*Sigma@w reduces
algebraically to w. If either property fails, the prior/posterior math
has a bug -- see tests/test_black_litterman.py.
"""
import numpy as np


class BlackLittermanEstimator:
    def __init__(self, risk_aversion: float = 1.0, tau: float = 0.025):
        """
        risk_aversion (lambda): should match whatever MeanVarianceOptimizer
            this estimator's output feeds, so the sanity check's algebra
            actually holds.
        tau: confidence in the EQUILIBRIUM PRIOR relative to views
            (smaller = more confidence in the prior). 0.025 is a
            commonly-cited illustrative default in the BL literature,
            not calibrated for this project.
        """
        self.risk_aversion = risk_aversion
        self.tau = tau

    def estimate(self, cov_matrix: np.ndarray, market_weights: np.ndarray = None,
                 P: np.ndarray = None, Q: np.ndarray = None, omega: np.ndarray = None):
        """
        Returns (posterior_returns, posterior_cov).

        market_weights: defaults to equal weight if not given (see
            module docstring).
        P, Q, omega: standard BL view matrices. All three None (default)
            means NO VIEWS -- posterior_returns equals the prior exactly.
            P: (k, n), each row a view expressed as an asset combination.
            Q: (k,), the view's return value. omega: (k, k), view
            uncertainty (smaller = more confident).
        """
        sigma = np.asarray(cov_matrix, dtype=np.float64)
        n = sigma.shape[0]

        if market_weights is None:
            market_weights = np.full(n, 1.0 / n)
        else:
            market_weights = np.asarray(market_weights, dtype=np.float64)

        pi = self.risk_aversion * (sigma @ market_weights)  # equilibrium prior

        if P is None or Q is None or omega is None:
            return pi, sigma

        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)

        tau_sigma_inv = np.linalg.pinv(self.tau * sigma)
        omega_inv = np.linalg.pinv(omega)

        precision = tau_sigma_inv + P.T @ omega_inv @ P
        precision_inv = np.linalg.pinv(precision)

        posterior_returns = precision_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        posterior_cov = sigma + precision_inv

        return posterior_returns, posterior_cov