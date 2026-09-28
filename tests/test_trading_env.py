import numpy as np
import pandas as pd
import pytest
from configs.base_config import config
from src.env.trading_env import StockTradingEnv


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 300
    cols = ['macd', 'rsi_14', 'rsi_30', 'cci_14', 'dx_30', 'atr_30', 'boll_ub', 'boll_lb']
    data = {c: rng.standard_normal(n) for c in cols}
    data['date'] = pd.date_range('2020-01-01', periods=n)
    data['log_return'] = rng.normal(0.0005, 0.01, n)
    return pd.DataFrame(data)


def test_reset_observation_shape_and_capital(df):
    env = StockTradingEnv(df)
    obs, _ = env.reset()
    assert obs.shape == (config.LOOKBACK_WINDOW, env.n_features + 1)  # +1 = position column
    assert obs.shape == env.observation_space.shape
    assert env.portfolio_value == config.INITIAL_CAPITAL


def test_reset_restores_capital_after_an_episode(df):
    env = StockTradingEnv(df)
    env.reset()
    for _ in range(100):
        env.step(0)
    assert env.portfolio_value != config.INITIAL_CAPITAL  # precondition: state really changed
    env.reset()
    assert env.portfolio_value == config.INITIAL_CAPITAL
    assert env.portfolio_history == [config.INITIAL_CAPITAL]


def test_dsr_is_called_exactly_once_per_step(df):
    env = StockTradingEnv(df, use_dsr=True)
    env.reset()
    calls = []
    original_step = env.dsr.step

    def spy(x):
        calls.append(x)
        return original_step(x)

    env.dsr.step = spy
    for _ in range(10):
        env.step(2)
    assert len(calls) == 10