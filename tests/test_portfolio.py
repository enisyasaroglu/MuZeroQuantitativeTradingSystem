import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import Portfolio, OptimizationResult


class _FixedOptimizer:
    """Test double: always returns fixed weights, records call count so
    tests can assert exactly when Portfolio decided to rebalance."""
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.call_count = 0

    def optimize(self, inputs, constraints):
        self.call_count += 1
        return OptimizationResult(weights=self.weights, method_name="fixed", converged=True)


def _fake_returns(n_timesteps, n_assets, seed=0):
    return np.random.RandomState(seed).normal(0, 0.01, size=(n_timesteps, n_assets))


def test_no_rebalance_before_lookback():
    opt = _FixedOptimizer([0.5, 0.5])
    p = Portfolio(2, opt, PortfolioConstraints(), lookback=60, rebalance_every=20)
    p.propose_weights(t=10, returns_matrix=_fake_returns(100, 2))
    assert opt.call_count == 0


def test_rebalances_on_schedule():
    opt = _FixedOptimizer([0.5, 0.5])
    p = Portfolio(2, opt, PortfolioConstraints(), lookback=60, rebalance_every=20)
    returns = _fake_returns(150, 2)

    p.propose_weights(t=60, returns_matrix=returns)
    assert opt.call_count == 1
    p.propose_weights(t=70, returns_matrix=returns)  # not due yet
    assert opt.call_count == 1
    p.propose_weights(t=80, returns_matrix=returns)  # (80-60)%20==0 -> due
    assert opt.call_count == 2


def test_turnover_uses_precommit_weights():
    opt = _FixedOptimizer([1.0, 0.0])
    p = Portfolio(2, opt, PortfolioConstraints(), lookback=60, rebalance_every=20)
    target = p.propose_weights(t=60, returns_matrix=_fake_returns(100, 2))
    assert p.turnover(target) == 1.0
    p.commit(target)
    assert p.turnover(target) == 0.0


def test_reset_clears_cache_and_weights():
    opt = _FixedOptimizer([0.5, 0.5])
    p = Portfolio(2, opt, PortfolioConstraints(), lookback=60, rebalance_every=20)
    returns = _fake_returns(100, 2)
    p.commit(p.propose_weights(t=60, returns_matrix=returns))
    assert opt.call_count == 1

    p.reset()
    assert np.allclose(p.current_weights, [0.0, 0.0])
    p.propose_weights(t=60, returns_matrix=returns)
    assert opt.call_count == 2  # cache cleared -> rebalances again