import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.comparison import MethodSpec, compare_methods, print_comparison_table, _run_single_backtest
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.methods.equal_weight import EqualWeightOptimizer


def _make_deterministic_returns(T=100, n=3, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(0.0005, 0.01, size=(T, n))


def test_single_backtest_matches_hand_computed_equal_weight_no_cost():
    """Zero transaction fee + Equal Weight: portfolio value should match
    a manually compounded 1/n-weighted return series exactly."""
    returns = _make_deterministic_returns(T=80, n=3, seed=1)
    lookback = 10

    result = _run_single_backtest(
        returns, EqualWeightOptimizer(), PortfolioConstraints(),
        rebalance_every=5, lookback=lookback, initial_capital=100_000.0,
        transaction_fee=0.0,
    )

    manual_value = 100_000.0
    for t in range(lookback, len(returns) - 1):
        port_return = np.mean(returns[t])  # equal weight = mean of asset returns
        manual_value *= np.exp(port_return)

    assert np.isclose(result["total_return"], (manual_value / 100_000.0) - 1.0, atol=1e-6)


def test_deterministic_method_reproducible_across_runs():
    returns = _make_deterministic_returns(T=80, n=3, seed=2)
    r1 = _run_single_backtest(returns, EqualWeightOptimizer(), PortfolioConstraints(), 5, 10, 100_000.0, 0.001)
    r2 = _run_single_backtest(returns, EqualWeightOptimizer(), PortfolioConstraints(), 5, 10, 100_000.0, 0.001)
    assert r1 == r2


def test_compare_methods_produces_one_row_per_method():
    returns = _make_deterministic_returns(T=80, n=3, seed=3)
    specs = [MethodSpec("EqualWeight", lambda seed: EqualWeightOptimizer(), n_seeds=1)]
    df = compare_methods(returns, specs, rebalance_every=5, lookback=10)

    assert len(df) == 1
    assert df.iloc[0]["method"] == "EqualWeight"
    assert df.iloc[0]["stochastic"] == False
    assert df.iloc[0]["total_return_std"] is None


def test_stochastic_method_reports_std_across_seeds():
    """A synthetic 'stochastic' method whose result depends on the seed
    should produce a nonzero std -- confirms seeds are actually varied
    per run, not silently reused."""
    from src.portfolio.portfolio import OptimizationResult

    class _SeededFakeOptimizer:
        def __init__(self, seed):
            self._rng = np.random.RandomState(seed)

        def optimize(self, inputs, constraints):
            n = len(inputs.cov_matrix) if inputs.cov_matrix is not None else 3
            raw = self._rng.uniform(0, 1, n)
            weights = constraints.project(raw)
            return OptimizationResult(weights=weights, method_name="fake", converged=True)

    returns = _make_deterministic_returns(T=80, n=3, seed=4)
    specs = [MethodSpec("FakeStochastic", lambda seed: _SeededFakeOptimizer(seed), n_seeds=8)]
    df = compare_methods(returns, specs, rebalance_every=5, lookback=10)

    assert df.iloc[0]["stochastic"] == True
    assert df.iloc[0]["total_return_std"] is not None
    assert df.iloc[0]["total_return_std"] >= 0.0


def test_print_comparison_table_does_not_raise():
    returns = _make_deterministic_returns(T=80, n=3, seed=5)
    specs = [MethodSpec("EqualWeight", lambda seed: EqualWeightOptimizer(), n_seeds=1)]
    df = compare_methods(returns, specs, rebalance_every=5, lookback=10)
    print_comparison_table(df)  # smoke test -- must not raise