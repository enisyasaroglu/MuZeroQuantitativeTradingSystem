"""
Tests for the rule-based strategies.

The two most important checks are the same for every strategy:

* No look-ahead: the actions for the first part of a series must not
  change when more data is added afterwards.
* The one-day lag: the action for day t comes from the position wanted
  after day t-1.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)


def random_returns(n=700, seed=0):
    """Build a reproducible series of small random log returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n))


def all_strategies():
    """Create one instance of every strategy, including short-selling variants."""
    return [
        MomentumStrategy(),
        MomentumStrategy(allow_short=True),
        MeanReversionStrategy(),
        MeanReversionStrategy(allow_short=True),
        TrendFollowingStrategy(),
    ]


@pytest.mark.parametrize("strategy", all_strategies(), ids=lambda s: s.name)
def test_no_lookahead(strategy):
    """Adding later data must not change earlier actions."""
    returns = random_returns()
    full = strategy.actions(returns)
    partial = strategy.actions(returns.iloc[:450])
    assert np.array_equal(full[:450], partial)


@pytest.mark.parametrize("strategy", all_strategies(), ids=lambda s: s.name)
def test_action_comes_from_previous_days_position(strategy):
    """The action for day t equals the target position of day t-1, plus one."""
    returns = random_returns()
    target = strategy.target_positions(returns).to_numpy()
    actions = strategy.actions(returns)
    assert actions[0] == 1
    assert np.array_equal(actions[1:], target[:-1] + 1)


@pytest.mark.parametrize("strategy", all_strategies(), ids=lambda s: s.name)
def test_actions_are_valid(strategy):
    """Every action is 0, 1 or 2 and there is one per row."""
    returns = random_returns()
    actions = strategy.actions(returns)
    assert len(actions) == len(returns)
    assert set(np.unique(actions)) <= {0, 1, 2}


def test_long_only_strategies_never_short():
    """Without allow_short, no action is 0."""
    returns = random_returns()
    for strategy in (MomentumStrategy(), MeanReversionStrategy(), TrendFollowingStrategy()):
        assert 0 not in set(strategy.actions(returns))


def test_momentum_long_in_uptrend_and_cash_in_downtrend():
    """A steady rise gives long positions and a steady fall gives cash."""
    up = pd.Series(np.full(300, 0.002))
    down = pd.Series(np.full(300, -0.002))
    strategy = MomentumStrategy()
    assert strategy.actions(up)[-1] == 2
    assert strategy.actions(down)[-1] == 1
    assert MomentumStrategy(allow_short=True).actions(down)[-1] == 0


def test_momentum_is_neutral_until_enough_history():
    """Before lookback days of history, momentum stays in cash."""
    up = pd.Series(np.full(300, 0.002))
    actions = MomentumStrategy(lookback=126, skip=5).actions(up)
    assert set(actions[:126]) == {1}


def test_mean_reversion_buys_a_sharp_drop_and_exits_later():
    """After a sudden fall the strategy goes long, then leaves once it recovers.

    The returns are flat, then one sharp drop, then a steady recovery that
    brings the price back above its recent average. The exit rule needs
    that recovery, so the data provides it on purpose instead of hoping
    that random noise does.
    """
    values = np.zeros(200)
    values[100] = -0.05
    values[101:121] = 0.004
    strategy = MeanReversionStrategy()
    actions = strategy.actions(pd.Series(values))
    assert 2 in set(actions[101:104])
    assert set(actions[125:200]) == {1}


def test_trend_following_rides_a_rise_and_exits_in_a_fall():
    """The strategy is long during a steady rise and out after a long fall."""
    values = np.concatenate([np.full(100, 0.005), np.full(60, -0.01)])
    actions = TrendFollowingStrategy().actions(pd.Series(values))
    assert actions[99] == 2
    assert actions[150] == 1


def test_invalid_parameters_are_rejected():
    """Parameters that make no sense raise an error."""
    with pytest.raises(ValueError):
        MomentumStrategy(lookback=5, skip=5)
    with pytest.raises(ValueError):
        MeanReversionStrategy(window=1)
    with pytest.raises(ValueError):
        TrendFollowingStrategy(entry_window=1)


def test_bad_input_is_rejected():
    """Missing values and empty series are not accepted."""
    with pytest.raises(ValueError):
        MomentumStrategy().actions(pd.Series([0.01, np.nan, 0.02]))
    with pytest.raises(ValueError):
        MomentumStrategy().actions(pd.Series([], dtype=float))