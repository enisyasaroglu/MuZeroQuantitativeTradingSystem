"""
Evaluation script: runs one or more agents on a held-out data split and
reports the standardised financial metric set (total return, CAGR,
Sharpe, Sortino, max drawdown, Calmar, win rate), matching the tables
in the dissertation's Results & Evaluation chapter.

Usage:
    python evaluate.py --agent muzero --split test --checkpoint src/models/muzero_checkpoint_20.pth
    python evaluate.py --agent all --split val

Evaluation intentionally uses RAW net log-return as the environment
reward (use_dsr=False), not the shaped DSR training signal -- financial
metrics should reflect true realised portfolio performance, not the
reward-shaping used during training.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from configs.base_config import config as base_config
from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.agents.baselines.ppo_agent import PPOAgent
from src.utils.metrics import compute_all_metrics

ACTION_NAMES = {0: "Short", 1: "Neutral", 2: "Long"}


def load_split(split: str) -> pd.DataFrame:
    path = os.path.join("data", "processed", f"{split}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pipeline/processor.py` first "
            f"to generate train/val/test_data.csv."
        )
    return pd.read_csv(path)


def run_episode(env: StockTradingEnv, action_fn):
    """
    Runs one full pass over the split. action_fn(obs) -> int action.
    Returns (portfolio_history, action_counts dict).
    """
    obs, _ = env.reset()
    done = False
    action_counts = {0: 0, 1: 0, 2: 0}

    while not done:
        action = action_fn(obs)
        action_counts[int(action)] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.portfolio_history, action_counts


def action_distribution_pct(action_counts):
    total = sum(action_counts.values())
    if total == 0:
        return {ACTION_NAMES[a]: 0.0 for a in ACTION_NAMES}
    return {ACTION_NAMES[a]: 100.0 * c / total for a, c in action_counts.items()}


def evaluate_muzero(df, checkpoint_path=None):
    env = StockTradingEnv(df, use_dsr=False)
    cfg = MuZeroConfig()
    agent = MuZeroAgent(cfg, env.observation_space.shape)
    if checkpoint_path:
        agent.network.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
        print(f"Loaded MuZero checkpoint: {checkpoint_path}")

    def action_fn(obs):
        action, _, _ = agent.select_action(obs)
        return action

    return run_episode(env, action_fn)


def evaluate_ppo(df, checkpoint_path=None):
    env = StockTradingEnv(df, use_dsr=False)
    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=env.action_space.n)
    if checkpoint_path:
        agent.policy.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
        print(f"Loaded PPO checkpoint: {checkpoint_path}")

    def action_fn(obs):
        action, _, _ = agent.select_action(obs)
        return action

    return run_episode(env, action_fn)


def evaluate_buy_and_hold(df):
    env = StockTradingEnv(df, use_dsr=False)
    return run_episode(env, action_fn=lambda obs: 2)  # always Long


def print_report(name, portfolio_history, action_counts):
    metrics = compute_all_metrics(portfolio_history)
    dist = action_distribution_pct(action_counts)

    print(f"\n=== {name} ===")
    print(f"  Final Portfolio Value : £{portfolio_history[-1]:,.2f}")
    print(f"  Total Return          : {metrics['total_return']:.2%}")
    print(f"  CAGR                  : {metrics['cagr']:.2%}")
    print(f"  Sharpe Ratio          : {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio         : {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown          : {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio          : {metrics['calmar_ratio']:.2f}")
    print(f"  Win Rate              : {metrics['win_rate']:.2%}")
    print(f"  Action Distribution   : Short {dist['Short']:.1f}% | "
          f"Neutral {dist['Neutral']:.1f}% | Long {dist['Long']:.1f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trading agents on a held-out split.")
    parser.add_argument("--agent", choices=["muzero", "ppo", "buy_and_hold", "all"], default="all")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pth checkpoint (muzero/ppo only)")
    args = parser.parse_args()

    df = load_split(args.split)
    print(f"Evaluating on '{args.split}' split ({len(df)} rows).")

    results = {}
    if args.agent in ("muzero", "all"):
        history, counts = evaluate_muzero(df, args.checkpoint if args.agent == "muzero" else None)
        results["MuZero"] = print_report("MuZero", history, counts)

    if args.agent in ("ppo", "all"):
        history, counts = evaluate_ppo(df, args.checkpoint if args.agent == "ppo" else None)
        results["PPO"] = print_report("PPO", history, counts)

    if args.agent in ("buy_and_hold", "all"):
        history, counts = evaluate_buy_and_hold(df)
        results["Buy-and-Hold"] = print_report("Buy-and-Hold", history, counts)

    out_dir = os.path.join("logs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_results.csv")
    pd.DataFrame(results).T.to_csv(out_path)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()