import numpy as np
import pandas as pd
from configs.base_config import config
from src.env.trading_env import StockTradingEnv  # adjust to your real path
import pytest

LONG, FLAT = 2, 1

def make_env():
    n = config.LOOKBACK_WINDOW + 20
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, n)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "log_return": r,
        "log_return_norm": (r - r.mean()) / r.std(),
        "rsi_14": rng.normal(0, 1, n),
    })
    return StockTradingEnv(df, use_dsr=False)  # window_size=None -> deterministic

def test_position_is_visible_and_explains_cost():
    assert config.TRANSACTION_FEE > 0

    a = make_env(); a.reset()
    obs_a, *_ = a.step(LONG)              # holding LONG
    _, _, _, _, info_a = a.step(LONG)     # LONG -> LONG

    b = make_env(); b.reset()
    obs_b, *_ = b.step(FLAT)              # holding FLAT
    _, _, _, _, info_b = b.step(LONG)     # FLAT -> LONG

    # Same date, same action: reward differs by exactly one fee (passes today)
    assert np.isclose(info_a["net_return"] - info_b["net_return"], config.TRANSACTION_FEE)

    # The agent must be able to see why (fails today: observations are identical)
    assert not np.array_equal(obs_a, obs_b)
    
def test_position_feature_at_reset_is_flat():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (config.LOOKBACK_WINDOW, env.n_features + 1)
    assert env.observation_space.contains(obs)
    assert np.all(obs[:, -1] == 0.0)

@pytest.mark.parametrize("action,expected", [(0, -1.0), (1, 0.0), (2, 1.0)])
def test_position_feature_reflects_action_just_taken(action, expected):
    env = make_env(); env.reset()
    obs, *_ = env.step(action)
    assert np.all(obs[:, -1] == expected)

def test_market_features_are_unchanged_by_position():
    a = make_env(); a.reset(); obs_a, *_ = a.step(LONG)
    b = make_env(); b.reset(); obs_b, *_ = b.step(FLAT)
    assert np.array_equal(obs_a[:, :-1], obs_b[:, :-1])   # only the position column differs