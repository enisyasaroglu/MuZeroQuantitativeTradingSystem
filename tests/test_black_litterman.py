import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.black_litterman import BlackLittermanOptimizerAdapter
from src.portfolio.methods.momentum_views import MomentumViewGenerator
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def _random_cov(n, seed):
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + np.eye(n) * 0.01


def test_no_view_generator_recovers_market_weights_exactly():
    sigma = _random_cov(4, seed=0)
    market_weights = np.array([0.4, 0.3, 0.2, 0.1])
    risk_aversion = 2.0

    adapter = BlackLittermanOptimizerAdapter(
        view_generator=None, risk_aversion=risk_aversion, market_weights=market_weights
    )
    result = adapter.optimize(OptimizationInputs(cov_matrix=sigma), PortfolioConstraints())
    assert np.allclose(result.weights, market_weights, atol=1e-6)


def test_with_momentum_views_shifts_away_from_market_weights():
    rng = np.random.RandomState(1)
    n = 3
    sigma = _random_cov(n, seed=1)
    market_weights = np.full(n, 1 / n)

    T = 60
    weak = rng.normal(0.0, 0.01, T)
    strong = weak + 0.01
    flat = rng.normal(0.0, 0.01, T)
    returns = np.stack([weak, strong, flat], axis=1)

    adapter = BlackLittermanOptimizerAdapter(
        view_generator=MomentumViewGenerator(tilt_strength=5.0),
        risk_aversion=1.0,
        market_weights=market_weights,
    )
    result = adapter.optimize(
        OptimizationInputs(cov_matrix=sigma, returns_sample=returns), PortfolioConstraints()
    )
    assert not np.allclose(result.weights, market_weights, atol=1e-3)


def test_missing_returns_sample_raises_when_view_generator_set():
    adapter = BlackLittermanOptimizerAdapter(view_generator=MomentumViewGenerator())
    sigma = _random_cov(3, seed=2)
    try:
        adapter.optimize(OptimizationInputs(cov_matrix=sigma), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_missing_cov_matrix_always_raises():
    adapter = BlackLittermanOptimizerAdapter()
    try:
        adapter.optimize(OptimizationInputs(), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_weights_sum_to_one_with_views():
    rng = np.random.RandomState(3)
    n = 4
    sigma = _random_cov(n, seed=3)
    returns = rng.normal(0.0005, 0.01, size=(50, n))

    adapter = BlackLittermanOptimizerAdapter(view_generator=MomentumViewGenerator())
    result = adapter.optimize(
        OptimizationInputs(cov_matrix=sigma, returns_sample=returns), PortfolioConstraints()
    )
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_respects_constraints():
    rng = np.random.RandomState(4)
    n = 5
    sigma = _random_cov(n, seed=4)
    returns = rng.normal(0.0005, 0.01, size=(50, n))
    constraints = PortfolioConstraints(max_assets=2)

    adapter = BlackLittermanOptimizerAdapter(view_generator=MomentumViewGenerator())
    result = adapter.optimize(
        OptimizationInputs(cov_matrix=sigma, returns_sample=returns), constraints
    )
    assert constraints.is_feasible(result.weights)