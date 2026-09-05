import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.mean_variance import MeanVarianceOptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def test_recovers_market_weights_when_mu_equals_lambda_sigma_w():
    """The algebraic identity the Black-Litterman sanity check depends
    on: w* = (1/lambda) Sigma^-1 mu, with mu = lambda * Sigma @ w, must
    reduce exactly to w."""
    rng = np.random.RandomState(0)
    n = 4
    A = rng.normal(size=(n, n))
    sigma = A @ A.T + np.eye(n) * 0.01  # guaranteed positive-definite
    w_true = np.array([0.4, 0.3, 0.2, 0.1])
    risk_aversion = 2.0
    mu = risk_aversion * (sigma @ w_true)

    opt = MeanVarianceOptimizer(risk_aversion=risk_aversion)
    result = opt.optimize(OptimizationInputs(expected_returns=mu, cov_matrix=sigma), PortfolioConstraints())

    assert np.allclose(result.weights, w_true, atol=1e-6)


def test_weights_sum_to_one():
    rng = np.random.RandomState(1)
    n = 5
    A = rng.normal(size=(n, n))
    sigma = A @ A.T + np.eye(n) * 0.01
    mu = rng.uniform(0.0001, 0.001, size=n)

    result = MeanVarianceOptimizer().optimize(
        OptimizationInputs(expected_returns=mu, cov_matrix=sigma), PortfolioConstraints()
    )
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_missing_inputs_raises():
    opt = MeanVarianceOptimizer()
    try:
        opt.optimize(OptimizationInputs(expected_returns=np.zeros(3)), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_singular_covariance_does_not_crash():
    """A rank-deficient covariance (e.g. two identical assets) should be
    handled via ridge + pinv, not raise a LinAlgError."""
    sigma = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],  # identical to row 0 -- singular
        [0.0, 0.0, 1.0],
    ])
    mu = np.array([0.001, 0.001, 0.002])
    result = MeanVarianceOptimizer().optimize(
        OptimizationInputs(expected_returns=mu, cov_matrix=sigma), PortfolioConstraints()
    )
    assert np.all(np.isfinite(result.weights))


def test_respects_constraints_projection():
    rng = np.random.RandomState(2)
    n = 4
    A = rng.normal(size=(n, n))
    sigma = A @ A.T + np.eye(n) * 0.01
    mu = rng.uniform(0.0001, 0.001, size=n)
    constraints = PortfolioConstraints(max_assets=2)

    result = MeanVarianceOptimizer().optimize(
        OptimizationInputs(expected_returns=mu, cov_matrix=sigma), constraints
    )
    assert constraints.is_feasible(result.weights)