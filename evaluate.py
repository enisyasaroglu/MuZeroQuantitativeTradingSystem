"""
Evaluation script: runs one or more agents on a held-out data split and
reports the standardised financial metric set (total return, CAGR,
Sharpe, Sortino, max drawdown, Calmar, win rate), matching the tables
in the dissertation's Results & Evaluation chapter.
"""

import argparse
import glob
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/macOS plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from configs.base_config import config as base_config
from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.agents.ppo.ppo_agent import PPOAgent
from src.utils.dashboard_logger import QuantRLLogger
from src.utils.metrics import compute_all_metrics, annualized_volatility, worst_single_period_loss

ACTION_NAMES = {0: "Short", 1: "Neutral", 2: "Long"}
POSITION_EXPOSURE_PCT = {0: -100.0, 1: 0.0, 2: 100.0}


def load_split(split: str) -> pd.DataFrame:
    path = os.path.join("data", "processed", f"{split}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pipeline/processor.py` first "
            f"to generate train/val/test_data.csv."
        )
    return pd.read_csv(path)


def find_latest_checkpoint(prefix: str):
    """Finds the checkpoint with the highest episode number matching
    src/checkpoints/{prefix}_*.pth, or None if none exist."""
    files = glob.glob(os.path.join("src", "checkpoints", f"{prefix}_*.pth"))
    if not files:
        return None

    def extract_ep(f):
        try:
            filename = os.path.basename(f)
            ep_str = filename.replace(f"{prefix}_", "").replace(".pth", "")
            return int(ep_str)
        except ValueError:
            return -1

    return max(files, key=extract_ep)


def run_episode(env: StockTradingEnv, action_fn):
    """
    Runs one full pass over the split. action_fn(obs) -> int action.

    Returns (portfolio_history, action_counts, position_history, step_indices).
    """
    obs, _ = env.reset()
    done = False
    action_counts = {0: 0, 1: 0, 2: 0}
    position_history = []
    step_indices = []

    while not done:
        step_indices.append(env.current_step)
        action = action_fn(obs)
        action_counts[int(action)] += 1
        position_history.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.portfolio_history, action_counts, position_history, step_indices


def action_distribution_pct(action_counts):
    total = sum(action_counts.values())
    if total == 0:
        return {ACTION_NAMES[a]: 0.0 for a in ACTION_NAMES}
    return {ACTION_NAMES[a]: 100.0 * c / total for a, c in action_counts.items()}


def evaluate_muzero(df, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No MuZero checkpoint found ({checkpoint_path}). Aborting rather than "
            f"evaluating an untrained network. Train one first with `python main_muzero.py`."
        )

    env = StockTradingEnv(df, use_dsr=True)
    cfg = MuZeroConfig()
    
    # Robust instantiation across config parameter order
    try:
        agent = MuZeroAgent(cfg, env.observation_space.shape)
    except TypeError:
        try:
            agent = MuZeroAgent(obs_shape=env.observation_space.shape, cfg=cfg)
        except TypeError:
            agent = MuZeroAgent(cfg=cfg)

    device = getattr(agent, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Robust loading across network structure variations
    if hasattr(agent, 'network'):
        agent.network.load_state_dict(checkpoint)
    elif hasattr(agent, 'load_checkpoint'):
        agent.load_checkpoint(checkpoint_path)
    else:
        agent.load_state_dict(checkpoint)

    print(f"Loaded MuZero checkpoint: {checkpoint_path}")

    def action_fn(obs):
        try:
            res = agent.select_action(obs, temperature=0.0, add_exploration_noise=False)
        except TypeError:
            try:
                res = agent.select_action(obs, eval_mode=True)
            except TypeError:
                res = agent.select_action(obs)
        return res[0] if isinstance(res, (tuple, list)) else res

    return run_episode(env, action_fn)


def evaluate_ppo(df, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No PPO checkpoint found ({checkpoint_path}). Aborting rather than "
            f"evaluating an untrained network. Train one first with `python main_ppo.py`."
        )

    env = StockTradingEnv(df, use_dsr=True)
    action_dim = getattr(env, 'action_dim', getattr(env, 'n_actions', getattr(getattr(env, 'action_space', None), 'n', 3)))
    
    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=action_dim)
    device = getattr(agent, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if hasattr(agent, 'policy'):
        agent.policy.load_state_dict(checkpoint)
    else:
        agent.load_state_dict(checkpoint)

    print(f"Loaded PPO checkpoint: {checkpoint_path}")

    def action_fn(obs):
        try:
            res = agent.select_action(obs, deterministic=True)
        except TypeError:
            res = agent.select_action(obs)
        return res[0] if isinstance(res, (tuple, list)) else res

    return run_episode(env, action_fn)


def evaluate_buy_and_hold(df):
    env = StockTradingEnv(df, use_dsr=False)
    return run_episode(env, action_fn=lambda obs: 2)  # always Long


def export_equity_and_positions(name, df, portfolio_history, position_history, step_indices,
                                  split, out_dir="logs/evaluation"):
    """
    Saves the FULL per-step equity curve and position/exposure history to CSV.
    Ensures all output fields match step count N exactly.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_steps = len(position_history)

    # Date alignment with safety fallback
    if 'date' in df.columns:
        dates = df['date'].iloc[step_indices].reset_index(drop=True)
    elif 'Date' in df.columns:
        dates = df['Date'].iloc[step_indices].reset_index(drop=True)
    else:
        dates = pd.Series(step_indices)

    # Align portfolio values to step count N (drops initial capital at index 0)
    if len(portfolio_history) == n_steps + 1:
        port_vals = portfolio_history[1:]
    else:
        port_vals = portfolio_history[-n_steps:]

    out_df = pd.DataFrame({
        "date": dates,
        "portfolio_value": port_vals,
        "position_action": position_history,
        "position_label": [ACTION_NAMES[a] for a in position_history],
        "exposure_pct": [POSITION_EXPOSURE_PCT[a] for a in position_history],
    })

    out_path = os.path.join(out_dir, f"{split}_{name.lower().replace(' ', '_')}_equity.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved full equity/position history: {out_path}")
    return out_df


def plot_equity_comparison(equity_dfs: dict, split, out_dir="logs/evaluation"):
    plt.figure(figsize=(11, 5))
    for name, edf in equity_dfs.items():
        plt.plot(pd.to_datetime(edf['date']), edf['portfolio_value'], label=name, linewidth=1.6)
    plt.title(f"Equity Curve Comparison ({split} split)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_equity_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved equity curve comparison: {out_path}")


def plot_exposure_over_time(name, equity_df, split, out_dir="logs/evaluation"):
    plt.figure(figsize=(11, 2.5))
    plt.step(pd.to_datetime(equity_df['date']), equity_df['exposure_pct'], where='post', linewidth=1.2)
    plt.title(f"{name} — Exposure Over Time ({split} split)")
    plt.xlabel("Date")
    plt.ylabel("Exposure (%)")
    plt.ylim(-110, 110)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_{name.lower().replace(' ', '_')}_exposure.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved exposure chart: {out_path}")


def print_report(name, portfolio_history, action_counts):
    metrics = compute_all_metrics(portfolio_history)
    dist = action_distribution_pct(action_counts)

    logger = QuantRLLogger(agent_name=name, asset_symbol="")
    logger.print_evaluation_report(name, metrics, dist, portfolio_history[-1])

    vol = annualized_volatility(portfolio_history)
    worst_day = worst_single_period_loss(portfolio_history)
    print(f"  Annualised Volatility : {vol:.2%}")
    print(f"  Worst Single-Step Loss: {worst_day:.2%}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trading agents on a held-out split.")
    parser.add_argument("--agent", choices=["muzero", "ppo", "buy_and_hold", "all"], default="all")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", default=None,
                         help="Path to a .pth checkpoint. Only applied when --agent is muzero or "
                              "ppo specifically -- with --agent all, use --muzero-ckpt/--ppo-ckpt "
                              "instead, or let both auto-detect.")
    parser.add_argument("--muzero-ckpt", default=None, help="Override MuZero checkpoint path.")
    parser.add_argument("--ppo-ckpt", default=None, help="Override PPO checkpoint path.")
    args = parser.parse_args()

    df = load_split(args.split)
    print(f"Evaluating on '{args.split}' split ({len(df)} rows).")

    muzero_ckpt = (
        args.muzero_ckpt
        or (args.checkpoint if args.agent == "muzero" else None)
        or find_latest_checkpoint("muzero_checkpoint")
    )
    ppo_ckpt = (
        args.ppo_ckpt
        or (args.checkpoint if args.agent == "ppo" else None)
        or find_latest_checkpoint("ppo_checkpoint")
    )

    results = {}
    equity_dfs = {}

    if args.agent in ("muzero", "all"):
        history, counts, positions, steps = evaluate_muzero(df, muzero_ckpt)
        results["MuZero"] = print_report("MuZero", history, counts)
        equity_dfs["MuZero"] = export_equity_and_positions("MuZero", df, history, positions, steps, args.split)
        plot_exposure_over_time("MuZero", equity_dfs["MuZero"], args.split)

    if args.agent in ("ppo", "all"):
        history, counts, positions, steps = evaluate_ppo(df, ppo_ckpt)
        results["PPO"] = print_report("PPO", history, counts)
        equity_dfs["PPO"] = export_equity_and_positions("PPO", df, history, positions, steps, args.split)
        plot_exposure_over_time("PPO", equity_dfs["PPO"], args.split)

    if args.agent in ("buy_and_hold", "all"):
        history, counts, positions, steps = evaluate_buy_and_hold(df)
        results["Buy-and-Hold"] = print_report("Buy-and-Hold", history, counts)
        equity_dfs["Buy-and-Hold"] = export_equity_and_positions(
            "Buy-and-Hold", df, history, positions, steps, args.split
        )

    if len(equity_dfs) > 1:
        plot_equity_comparison(equity_dfs, args.split)

    out_dir = os.path.join("logs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_results.csv")
    pd.DataFrame(results).T.to_csv(out_path)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()