"""
Does MuZero learn a pattern that is known to exist?

The price is a sine wave with no long-term drift, and each day's return
can be predicted from the two days before it. The agent trains on the
first 1,000 rows and is evaluated on the remaining 500. Buy-and-hold earns
about nothing on this data, so a profit can only come from learning the
pattern. The evaluation covers whole cycles, so any constant position
earns the same small loss from the entry fee. The action counts are
therefore printed as well: they show what the agent really did.

This checks the whole MuZero loop: the search, the learned model, the
replay buffer and the updates. The test is slow and runs only when the
environment variable RUN_SLOW_TESTS is set to 1. The number of training
episodes can be changed with MUZERO_SINE_EPISODES (default 300):

    RUN_SLOW_TESTS=1 pytest tests/test_sin_wave_muzero.py -v -s
    MUZERO_SINE_EPISODES=600 RUN_SLOW_TESTS=1 pytest tests/test_sin_wave_muzero.py -v -s

This file is self-contained: it builds its own data and does not depend
on any other test file.
"""

import os
import random
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from configs.muzero_config import MuZeroConfig
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.env.trading_env import StockTradingEnv
from src.utils.replay_buffer import ReplayBuffer
from src.utils.schedules import temperature_schedule

PERIOD = 20
AMPLITUDE = 0.05
N_ROWS = 1500
REWARD_SCALE = float(os.environ.get("MUZERO_REWARD_SCALE", "100"))


def make_sine_df(n=N_ROWS, period=PERIOD, amplitude=AMPLITUDE):
    """Build a DataFrame of sine-wave log returns in the processed-data format.

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


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Slow test (several minutes). Set RUN_SLOW_TESTS=1 to run it.",
)
def test_muzero_learns_the_sine_rule():
    """A MuZero agent trained on the first part of the sine series should
    earn clearly more than buy-and-hold on the remainder.

    The training budget is small, so a failure first suggests raising
    MUZERO_SINE_EPISODES before looking for a bug.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    df = make_sine_df()
    train_df = df.iloc[:1000].reset_index(drop=True)
    eval_df = df.iloc[1000:].reset_index(drop=True)

    num_episodes = int(os.environ.get("MUZERO_SINE_EPISODES", "300"))
    cfg = replace(MuZeroConfig(), num_simulations=20, num_episodes=num_episodes)

    train_env = StockTradingEnv(train_df, use_dsr=False, window_size=cfg.episode_length)
    agent = MuZeroAgent(cfg, train_env.observation_space.shape)
    buffer = ReplayBuffer(
        capacity=cfg.replay_buffer_capacity,
        batch_size=cfg.batch_size,
        unroll_steps=cfg.unroll_steps,
        discount=cfg.discount_factor,
        td_steps=cfg.td_steps,
    )

    for episode in range(1, num_episodes + 1):
        obs, _ = train_env.reset(seed=0 if episode == 1 else None)
        temperature = temperature_schedule(episode, num_episodes)
        game = {"obs": [], "actions": [], "rewards": [], "policies": [], "values": []}
        done = False
        while not done:
            action, policy, value = agent.select_action(obs, temperature=temperature)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            game["obs"].append(obs)
            game["actions"].append(action)
            game["rewards"].append(reward* REWARD_SCALE)  # Scale rewards to avoid numerical issues 
            game["policies"].append(policy)
            game["values"].append(value)
            obs = next_obs
        buffer.save_game(game)

        if buffer.size() >= cfg.min_buffer_size:
            sums = {"value": 0.0, "policy": 0.0, "reward": 0.0, "entropy": 0.0}
            for _ in range(cfg.updates_per_episode):
                _, parts = agent.update(buffer.sample_batch(), k_steps=cfg.unroll_steps)
                for key in sums:
                    sums[key] += parts[key]
            if episode % 50 == 0:
                n = cfg.updates_per_episode
                print(f"episode {episode}: " + ", ".join(
                    f"{key} {total / n:.5f}" for key, total in sums.items()))

    os.makedirs("runs", exist_ok=True)
    torch.save(agent.network.state_dict(), os.path.join("runs", "sine_muzero_diag.pth"))
                
    chosen = []

    def trained_action(obs):
        action, _, _ = agent.select_action(
            obs, temperature=0.0, add_exploration_noise=False
        )
        chosen.append(action)
        return action

    eval_env = StockTradingEnv(eval_df, use_dsr=False)
    history = play(eval_env, trained_action)
    agent_return = history[-1] / history[0] - 1.0

    hold_env = StockTradingEnv(eval_df, use_dsr=False)
    hold_history = play(hold_env, lambda obs: 2)
    hold_return = hold_history[-1] / hold_history[0] - 1.0

    print(f"MuZero ({num_episodes} episodes) return {agent_return:+.2%}, "
          f"buy-and-hold {hold_return:+.2%}")
    print("action counts short/neutral/long:", np.bincount(chosen, minlength=3).tolist())
    assert agent_return > 0.25, (
        f"MuZero earned {agent_return:+.2%} against buy-and-hold "
        f"{hold_return:+.2%}; it did not learn the sine rule."
    )