"""
Tests on synthetic sine-wave prices.

A sine-wave price has no long-term drift, and each day's return can be
predicted from the two days before it. That makes it a good check of the
data, the environment and the learning code, separate from the real market:

* Staying in cash must earn exactly nothing.
* Buying and holding cannot profit, because the price only oscillates.
* A trader who knows the rule must earn a great deal.
* A learning agent should earn clearly more than buy-and-hold. If it does
  not, the learning code or the reward scale is the problem. If it does,
  the learning code works, and any failure on real prices means the
  market data holds little to learn.

The data is built in memory. Nothing is read from or written to disk.

The slow test trains an agent for several minutes. It runs only when the
environment variable RUN_SLOW_TESTS is set to 1:

    RUN_SLOW_TESTS=1 pytest tests/test_sin_wave.py -v -s -k learns
"""

import os
import random

import numpy as np
import pandas as pd
import pytest
import torch

from configs.base_config import config
from src.env.trading_env import StockTradingEnv

PERIOD = 20
AMPLITUDE = 0.05
N_ROWS = 1500


def make_sine_df(n=N_ROWS, period=PERIOD, amplitude=AMPLITUDE):
    """Build a DataFrame of sine-wave log returns in the processed-data format.

    The log price is amplitude * sin(2 * pi * t / period) and the log
    return is its day-to-day change. The only feature is the normalised
    log return, as in the real data.

    Args:
        n: Number of rows.
        period: Length of one full cycle, in days.
        amplitude: Height of the swings in log-price units.

    Returns:
        A DataFrame with date, tic, log_return and log_return_norm columns.
    """
    t = np.arange(n + 1)
    log_price = amplitude * np.sin(2 * np.pi * t / period)
    returns = np.diff(log_price)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-01", periods=n),
            "tic": "SINE",
            "log_return": returns,
            "log_return_norm": (returns - returns.mean()) / returns.std(),
        }
    )


def play(env, action_fn):
    """Run one full pass over the environment's data.

    Args:
        env: A StockTradingEnv that uses its full dataset (no window size).
        action_fn: A function taking the observation and returning an
            action: 0 for short, 1 for neutral, 2 for long.

    Returns:
        The list of portfolio values, starting with the initial capital.
    """
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(int(action_fn(obs)))
        done = terminated or truncated
    return list(env.portfolio_history)


def test_sine_series_has_no_drift():
    """Returns over any stretch sum to a change in a bounded log price."""
    returns = make_sine_df()["log_return"].to_numpy()
    assert abs(returns.sum()) <= 2 * AMPLITUDE + 1e-9
    assert abs(returns[:500].sum()) <= 2 * AMPLITUDE + 1e-9
    assert np.abs(returns).max() < 0.02


def test_cash_earns_exactly_zero():
    """Staying neutral from the start costs and earns nothing."""
    env = StockTradingEnv(make_sine_df(), use_dsr=False)
    history = play(env, lambda obs: 1)
    assert history[-1] == pytest.approx(config.INITIAL_CAPITAL)


def test_buy_and_hold_cannot_profit_from_sine():
    """The price only oscillates, so holding earns almost nothing either way."""
    env = StockTradingEnv(make_sine_df(), use_dsr=False)
    history = play(env, lambda obs: 2)
    total = history[-1] / history[0] - 1.0
    assert abs(total) < 0.12


def test_oracle_earns_far_more_than_buy_and_hold():
    """A trader who knows the rule must earn a great deal, even after costs.

    Every return in a sine series follows from the previous two:
    r[t] = 2 * cos(2 * pi / period) * r[t-1] - r[t-2].
    The oracle predicts the next return with this rule and goes long if
    the prediction is positive, short otherwise.
    """
    df = make_sine_df()
    returns = df["log_return"].to_numpy()
    factor = 2.0 * np.cos(2.0 * np.pi / PERIOD)
    env = StockTradingEnv(df, use_dsr=False)

    def oracle(obs):
        t = env.current_step
        predicted = factor * returns[t - 1] - returns[t - 2]
        return 2 if predicted > 0 else 0

    history = play(env, oracle)
    assert history[-1] / history[0] > 100.0


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Slow test (several minutes). Set RUN_SLOW_TESTS=1 to run it.",
)
def test_ppo_learns_the_sine_rule():
    """A PPO agent trained on the first part of the sine series should earn
    clearly more than buy-and-hold on the remainder.

    This is an experiment as much as a test. If it fails, the result is
    informative: it says the learning code or the reward scale cannot
    extract a pattern that is known to exist.
    """
    from src.agents.ppo.ppo_agent import PPOAgent

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    df = make_sine_df()
    train_df = df.iloc[:1000].reset_index(drop=True)
    eval_df = df.iloc[1000:].reset_index(drop=True)

    train_env = StockTradingEnv(train_df, use_dsr=False, window_size=252)
    agent = PPOAgent(
        obs_shape=train_env.observation_space.shape,
        action_dim=train_env.action_space.n,
    )

    num_episodes = 400
    episodes_per_update = 8
    memory = {"states": [], "actions": [], "log_probs": [],
              "values": [], "rewards": [], "dones": []}

    for episode in range(1, num_episodes + 1):
        state, _ = train_env.reset(seed=0 if episode == 1 else None)
        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            memory["states"].append(state)
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["values"].append(value)
            memory["rewards"].append(reward)
            memory["dones"].append(done)
            state = next_state
        if episode % episodes_per_update == 0:
            agent.update(memory)
            memory = {k: [] for k in memory}

    eval_env = StockTradingEnv(eval_df, use_dsr=False)
    history = play(
        eval_env,
        lambda obs: agent.select_action(obs, deterministic=True)[0],
    )
    agent_return = history[-1] / history[0] - 1.0

    hold_env = StockTradingEnv(eval_df, use_dsr=False)
    hold_history = play(hold_env, lambda obs: 2)
    hold_return = hold_history[-1] / hold_history[0] - 1.0

    print(f"PPO return {agent_return:+.2%}, buy-and-hold {hold_return:+.2%}")
    assert agent_return > 0.25, (
        f"PPO earned {agent_return:+.2%} against buy-and-hold "
        f"{hold_return:+.2%}; it did not learn the sine rule."
    )