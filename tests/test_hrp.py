import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.hierarchical_risk_parity import (
    HierarchicalRiskParityOptimizer, _cov_to_corr, _correlation_distance, _cluster_variance,
)
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def test_cov_to_corr_diagonal_is_one():
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    corr = _cov_to_corr(cov)
    assert np.allclose(np.diag(corr), 1.0)


def test_correlation_distance_zero_for_perfect_correlation():
    corr = np.array([[1.0, 1.0], [1.0, 1.0]])
    dist = _correlation_distance(corr)
    assert np.allclose(dist, 0.0, atol=1e-10)


def test_correlation_distance_max_for_perfect_anticorrelation():
    corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
    dist = _correlation_distance(corr)
    assert np.isclose(dist[0, 1], 1.0)


def test_cluster_variance_matches_inverse_variance_portfolio_by_hand():
    """Direct check of the specific quantity recursive bisection
    compares -- not just 'runs without error'."""
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])  # zero correlation, easy to hand-verify
    inv_var = np.array([1 / 0.04, 1 / 0.01])
    w = inv_var / inv_var.sum()
    expected = w @ cov @ w
    assert np.isclose(_cluster_variance(cov, [0, 1]), expected)


def test_two_uncorrelated_assets_favour_lower_variance():
    """A basic sanity direction, not an exact closed-form match (HRP has
    no simple closed form even for n=2 the way Risk Parity does) --
    the lower-variance asset should get MORE weight."""
    cov = np.array([[0.01, 0.0], [0.0, 0.09]])  # asset 0 much less volatile
    result = HierarchicalRiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    assert result.weights[0] > result.weights[1]


def test_weights_sum_to_one():
    rng = np.random.RandomState(0)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + np.eye(5) * 0.01
    result = HierarchicalRiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_single_asset_gets_full_weight():
    cov = np.array([[0.02]])
    result = HierarchicalRiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    assert np.allclose(result.weights, [1.0])


def test_missing_cov_matrix_raises():
    try:
        HierarchicalRiskParityOptimizer().optimize(OptimizationInputs(expected_returns=np.zeros(3)), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ignores_expected_returns_if_provided():
    cov = np.eye(3) * 0.01
    r1 = HierarchicalRiskParityOptimizer().optimize(OptimizationInputs(cov_matrix=cov), PortfolioConstraints())
    r2 = HierarchicalRiskParityOptimizer().optimize(
        OptimizationInputs(cov_matrix=cov, expected_returns=np.array([1, -5, 3])), PortfolioConstraints()
    )
    assert np.allclose(r1.weights, r2.weights, atol=1e-6)