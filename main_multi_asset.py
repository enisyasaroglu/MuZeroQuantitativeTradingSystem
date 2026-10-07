"""
Training entry point for the multi-asset environment.

The multi-asset environment offers a discrete menu of portfolio templates
as its actions (see src/env/multi_asset_env.py), so MuZeroAgent and
PPOAgent are reused unchanged. Only the number of actions differs from
the single-asset case: it is taken from env.n_actions instead of being
fixed at 3.

Every run writes its models and a configuration record into --run-dir,
so a result can always be traced back to the settings that produced it.
Note that evaluate.py works with the single-asset environment only, so
the models trained here cannot be evaluated with it yet.

Usage:
    python3 src/pipeline/processor.py --multi-asset
    python3 main_multi_asset.py --agent muzero
    python3 main_multi_asset.py --agent ppo --episodes 50 --run-dir runs/ma_ppo
"""

import argparse
import json
import os
import random
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from configs.base_config import config as project_config
from configs.muzero_config import MuZeroConfig
from src.env.multi_asset_env import MultiAssetTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.agents.ppo.ppo_agent import PPOAgent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.schedules import temperature_schedule, entropy_coef_schedule
from utils.dashboard_logger import QuantRLLogger

# Maximum number of games kept for MuZero. Each stored game keeps every
# observation window of an episode, so memory grows quickly with long
# episodes. This is a cautious cap; raise it only after checking memory.
MUZERO_BUFFER_GAMES = 50


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, record):
    """Write a dictionary to a JSON file, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=4, default=str)


def load_multi_asset_split(split="train"):
    """Load data/processed/multi_asset_<split>_data.csv as a DataFrame."""
    path = os.path.join("data", "processed", f"multi_asset_{split}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it with "
            "`python3 src/pipeline/processor.py --multi-asset`."
        )
    return pd.read_csv(path)


def env_summary(env):
    """Facts about the environment that must match between training and use."""
    return {
        "n_actions": int(env.n_actions),
        "n_assets": int(env.n_assets),
        "obs_shape": list(env.observation_space.shape),
    }


def train_muzero(num_episodes, num_simulations, run_dir):
    """Train a MuZero agent on the multi-asset environment.

    num_episodes: number of training episodes (full passes over the data).
    num_simulations: search simulations per decision, or None to use the
        value in MuZeroConfig.
    run_dir: folder that receives the checkpoints and the config records.
    """
    set_seed(project_config.SEED)
    df = load_multi_asset_split("train")

    # Raw returns are used as the reward (no Differential Sharpe Ratio).
    # With DSR, doing nothing (action 0, all cash) becomes the best choice
    # once the running average return turns negative, which freezes the
    # agent. This is the same choice made in main_muzero.py.
    env = MultiAssetTradingEnv(df, use_dsr=False)

    overrides = {"action_space_dim": env.n_actions, "num_episodes": num_episodes}
    if num_simulations is not None:
        overrides["num_simulations"] = num_simulations
    cfg = replace(MuZeroConfig(), **overrides)

    agent = MuZeroAgent(cfg, env.observation_space.shape)
    buffer = ReplayBuffer(
        capacity=MUZERO_BUFFER_GAMES,
        batch_size=cfg.batch_size,
        unroll_steps=cfg.unroll_steps,
        discount=cfg.discount_factor,
        td_steps=cfg.td_steps,
    )

    run_record = {"agent": "muzero", **env_summary(env), **asdict(cfg)}
    write_json(os.path.join(run_dir, "run_config.json"), run_record)

    logger = QuantRLLogger(
        agent_name="MuZero",
        asset_symbol=f"Multi-Asset ({env.n_assets})",
        total_episodes=num_episodes,
    )
    logger.start_dashboard()

    try:
        for episode in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            temperature = temperature_schedule(episode, num_episodes)
            game = {"obs": [], "actions": [], "rewards": [],
                    "policies": [], "values": []}

            portfolio_history = [env.portfolio_value]
            daily_returns = []

            while not done:
                action, policy, value = agent.select_action(obs, temperature=temperature)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                game["obs"].append(obs)
                game["actions"].append(action)
                game["rewards"].append(reward)
                game["policies"].append(policy)
                game["values"].append(value)

                portfolio_history.append(env.portfolio_value)
                daily_returns.append(info["net_return"])
                obs = next_obs

            buffer.save_game(game)

            # Training starts once enough complete games are stored. The
            # buffer counts games, not steps, so this uses min_buffer_size
            # (a number of games) and not the batch size.
            policy_loss, value_loss, policy_entropy = 0.0, 0.0, 0.0
            if buffer.size() >= cfg.min_buffer_size:
                totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
                for _ in range(cfg.updates_per_episode):
                    batch = buffer.sample_batch()
                    _, parts = agent.update(batch, k_steps=cfg.unroll_steps)
                    for key in totals:
                        totals[key] += float(parts[key])
                policy_loss = totals["policy"] / cfg.updates_per_episode
                value_loss = totals["value"] / cfg.updates_per_episode
                policy_entropy = totals["entropy"] / cfg.updates_per_episode

            logger.log_cycle(
                cycle=episode,
                portfolio_values=portfolio_history,
                daily_returns=daily_returns,
                policy_loss=policy_loss,
                value_loss=value_loss,
                policy_entropy=policy_entropy,
            )

            if episode % cfg.checkpoint_every == 0:
                path = os.path.join(run_dir, f"muzero_multiasset_checkpoint_{episode}.pth")
                os.makedirs(run_dir, exist_ok=True)
                torch.save(agent.network.state_dict(), path)
                write_json(path.replace(".pth", "_config.json"),
                           {"episode": episode, **run_record})
    finally:
        logger.stop_dashboard()


def train_ppo(num_episodes, run_dir, checkpoint_every=10):
    """Train a PPO agent on the multi-asset environment.

    One update is made after each episode, using that episode only.

    num_episodes: number of training episodes.
    run_dir: folder that receives the checkpoints and the config record.
    checkpoint_every: save a checkpoint every this many episodes.
    """
    set_seed(project_config.SEED)
    df = load_multi_asset_split("train")

    # Raw returns as the reward, for the same reason as in train_muzero.
    env = MultiAssetTradingEnv(df, use_dsr=False)

    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=env.n_actions)

    run_record = {"agent": "ppo", "num_episodes": num_episodes,
                  "seed": project_config.SEED, **env_summary(env)}
    write_json(os.path.join(run_dir, "run_config.json"), run_record)

    logger = QuantRLLogger(
        agent_name="PPO",
        asset_symbol=f"Multi-Asset ({env.n_assets})",
        total_episodes=num_episodes,
    )
    logger.start_dashboard()

    try:
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            memory = {"states": [], "actions": [], "log_probs": [],
                      "rewards": [], "dones": []}
            portfolio_history = [env.portfolio_value]
            daily_returns = []
            done = False

            agent.entropy_coef = entropy_coef_schedule(episode, num_episodes)

            while not done:
                action, log_prob, _ = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                memory["states"].append(state)
                memory["actions"].append(action)
                memory["log_probs"].append(log_prob)
                memory["rewards"].append(reward)
                memory["dones"].append(done)

                portfolio_history.append(env.portfolio_value)
                daily_returns.append(info["net_return"])
                state = next_state

            # update() returns exactly three numbers. They are unpacked
            # directly so that any change to that shape fails loudly
            # instead of silently logging zeros.
            policy_loss, value_loss, policy_entropy = agent.update(memory)

            logger.log_cycle(
                cycle=episode,
                portfolio_values=portfolio_history,
                daily_returns=daily_returns,
                policy_loss=policy_loss,
                value_loss=value_loss,
                policy_entropy=policy_entropy,
            )

            if episode % checkpoint_every == 0:
                os.makedirs(run_dir, exist_ok=True)
                torch.save(agent.policy.state_dict(),
                           os.path.join(run_dir, f"ppo_multiasset_checkpoint_{episode}.pth"))
    finally:
        logger.stop_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an agent on the multi-asset environment.")
    parser.add_argument("--agent", choices=["muzero", "ppo"], default="ppo")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes (default: 20 for MuZero, 50 for PPO).")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MuZero search simulations per decision (MuZero only).")
    parser.add_argument("--run-dir", default=None,
                        help="Output folder (default: runs/multi_asset_<agent>).")
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join("runs", f"multi_asset_{args.agent}")

    if args.agent == "muzero":
        train_muzero(args.episodes or 20, args.simulations, run_dir)
    else:
        train_ppo(args.episodes or 50, run_dir)