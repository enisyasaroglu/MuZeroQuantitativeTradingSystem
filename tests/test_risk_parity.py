import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.risk_parity import RiskParityOptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def test_diagonal_covariance_gives_inverse_volatility_weights():
    """For a DIAGONAL covariance, the exact ERC solution is known in
    closed form: w_i proportional to 1/sigma_i -- a strong, specific
    check, not just 'weights sum to 1'."""
    variances = np.array([0.01, 0.04, 0.09])
    cov = np.diag(variances)

    result = RiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())

    inv_vol = 1.0 / np.sqrt(variances)
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(result.weights, expected, atol=1e-3)


def test_equal_variances_zero_correlation_gives_equal_weight():
    cov = np.eye(4) * 0.02
    result = RiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    assert np.allclose(result.weights, 0.25, atol=1e-3)


def test_risk_contributions_are_actually_equal_at_solution():
    """The defining property of Risk Parity, checked directly rather
    than inferred from the weights alone."""
    rng = np.random.RandomState(0)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T + np.eye(4) * 0.01

    result = RiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    rc, _ = RiskParityOptimizer._risk_contributions(result.weights, cov)
    assert np.allclose(rc, rc[0], atol=1e-3)


def test_weights_sum_to_one():
    cov = np.eye(3) * 0.01
    result = RiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_missing_cov_matrix_raises():
    try:
        RiskParityOptimizer().optimize(OptimizationInputs(expected_returns=np.zeros(3)), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ignores_expected_returns_if_provided():
    """Weights must be identical regardless of expected_returns, since
    Risk Parity never reads it."""
    cov = np.eye(3) * 0.01
    r1 = RiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    r2 = RiskParityOptimizer().optimize(
        OptimizationInputs(cov_matrix=cov, expected_returns=np.array([1, -5, 3])), PortfolioConstraints()
    )
    assert np.allclose(r1.weights, r2.weights, atol=1e-6)