import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.black_litterman import BlackLittermanEstimator
from src.portfolio.methods.mean_variance import MeanVarianceOptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def _random_cov(n, seed):
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n) * 0.01


def test_zero_views_returns_prior_exactly():
    sigma = _random_cov(4, seed=0)
    market_weights = np.array([0.4, 0.3, 0.2, 0.1])
    risk_aversion = 1.5

    bl = BlackLittermanEstimator(risk_aversion=risk_aversion)
    posterior_returns, posterior_cov = bl.estimate(sigma, market_weights=market_weights)

    expected_prior = risk_aversion * (sigma @ market_weights)
    assert np.allclose(posterior_returns, expected_prior, atol=1e-10)
    assert np.allclose(posterior_cov, sigma, atol=1e-10)


def test_full_pipeline_zero_views_recovers_market_weights_through_mvo():
    """THE requested sanity check: BL (no views) -> MVO must recover the
    assumed market_weights exactly. Catches a bug in the prior/posterior
    math, or in how MVO consumes it."""
    sigma = _random_cov(4, seed=1)
    market_weights = np.array([0.25, 0.25, 0.25, 0.25])
    risk_aversion = 2.0

    bl = BlackLittermanEstimator(risk_aversion=risk_aversion)
    posterior_returns, posterior_cov = bl.estimate(sigma, market_weights=market_weights)

    mvo = MeanVarianceOptimizer(risk_aversion=risk_aversion)
    result = mvo.optimize(
        OptimizationInputs(expected_returns=posterior_returns, cov_matrix=posterior_cov),
        PortfolioConstraints(),
    )

    assert np.allclose(result.weights, market_weights, atol=1e-6)


def test_defaults_to_equal_weight_market_proxy():
    sigma = _random_cov(3, seed=2)
    bl = BlackLittermanEstimator(risk_aversion=1.0)
    posterior_returns, _ = bl.estimate(sigma)  # no market_weights given
    expected_prior = 1.0 * (sigma @ np.full(3, 1 / 3))
    assert np.allclose(posterior_returns, expected_prior, atol=1e-10)


def test_views_shift_posterior_away_from_prior():
    """With a real view supplied, the posterior must differ from the
    zero-view prior -- confirms the view-blending terms actually do
    something, rather than being silently ignored."""
    sigma = _random_cov(3, seed=3)
    market_weights = np.full(3, 1 / 3)
    bl = BlackLittermanEstimator(risk_aversion=1.0, tau=0.05)

    prior_only, _ = bl.estimate(sigma, market_weights=market_weights)

    # Single relative view: asset 0 outperforms asset 1 by 5%
    P = np.array([[1.0, -1.0, 0.0]])
    Q = np.array([0.05])
    omega = np.array([[0.0001]])  # high confidence

    with_view, _ = bl.estimate(sigma, market_weights=market_weights, P=P, Q=Q, omega=omega)
    assert not np.allclose(prior_only, with_view, atol=1e-4)