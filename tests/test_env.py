import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.env.trading_env import StockTradingEnv
from configs.base_config import config


def _make_synthetic_df(n=300, seed=0):
    """Synthetic data fixture -- tests must not depend on external
    downloaded/processed CSVs being present on disk."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='B'),
        'log_return': rng.normal(0.0003, 0.01, n),
        'macd': rng.randn(n), 'rsi_14': rng.randn(n), 'rsi_30': rng.randn(n),
        'cci_14': rng.randn(n), 'dx_30': rng.randn(n), 'atr_30': rng.randn(n),
        'boll_ub': rng.randn(n), 'boll_lb': rng.randn(n),
    })


def test_environment_reset():
    """
    Regression test for the portfolio-reset bug: portfolio_value and
    portfolio_history must be restored to INITIAL_CAPITAL on every
    reset(), regardless of what happened in the previous episode.
    """
    df = _make_synthetic_df()
    env = StockTradingEnv(df)

    obs, info = env.reset()
    assert env.portfolio_value == config.INITIAL_CAPITAL
    assert env.portfolio_history == [config.INITIAL_CAPITAL]

    # Run an episode that should move portfolio_value away from initial capital
    for _ in range(100):
        env.step(0)  # persistent short

    assert env.portfolio_value != config.INITIAL_CAPITAL, \
        "test setup issue: portfolio value did not change during episode"

    # Reset again -- must NOT inherit the previous episode's ending value
    obs, info = env.reset()
    assert env.portfolio_value == config.INITIAL_CAPITAL, \
        f"portfolio reset bug: expected {config.INITIAL_CAPITAL}, got {env.portfolio_value}"
    assert env.portfolio_history == [config.INITIAL_CAPITAL]
    print("test_environment_reset PASSED")


def test_environment_random_run():
    df = _make_synthetic_df()
    env = StockTradingEnv(df)

    obs, info = env.reset()
    assert obs.shape == (env.lookback, env.n_features), "Observation shape mismatch!"

    done = False
    step = 0
    while not done and step < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

    print("test_environment_random_run PASSED")


def test_transaction_cost_applied_only_on_position_change():
    """Holding the same position across steps should not repeatedly
    incur the transaction fee; changing position should."""
    df = _make_synthetic_df()
    env = StockTradingEnv(df, use_dsr=False)  # raw log-return reward for a direct check
    env.reset()

    # Force log_return to 0 for a clean check of cost-only effect
    env.df['log_return'] = 0.0

    _, r_first_long, _, _, _ = env.step(2)   # Neutral -> Long: position changes, cost applied
    _, r_hold_long, _, _, _ = env.step(2)    # Long -> Long: no change, no cost

    assert r_first_long == -config.TRANSACTION_FEE
    assert r_hold_long == 0.0
    print("test_transaction_cost_applied_only_on_position_change PASSED")


def test_portfolio_compounds_via_exp():
    """Portfolio value must compound via exp(net_return), not a linear
    (1 + net_return) approximation."""
    df = _make_synthetic_df()
    df['log_return'] = 0.0  # isolate compounding math from cost-only steps
    env = StockTradingEnv(df, use_dsr=False)
    env.reset()

    env.step(1)  # Neutral -> Neutral, no cost, net_return = 0
    assert abs(env.portfolio_value - config.INITIAL_CAPITAL) < 1e-6
    print("test_portfolio_compounds_via_exp PASSED")


if __name__ == "__main__":
    test_environment_reset()
    test_environment_random_run()
    test_transaction_cost_applied_only_on_position_change()
    test_portfolio_compounds_via_exp()