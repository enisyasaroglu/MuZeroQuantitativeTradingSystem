"""
Evaluation script -- rewritten from scratch.

Three modes:
    --mode single       One checkpoint each for MuZero/PPO, one fixed
                         split (default: --split test). Full report,
                         equity/position export, comparison table.
    --mode progression   Evaluates a SEQUENCE of saved checkpoints for
                         ONE agent on the same fixed split -- shows
                         whether performance improves over training,
                         rather than only ever looking at the final
                         checkpoint.
    --mode walk-forward  Evaluates ONE checkpoint for ONE agent across
                         several ROLLING historical windows drawn from
                         out-of-sample data -- shows whether the
                         strategy's edge is consistent across regimes,
                         or an artifact of the single fixed test window.
                         This is walk-forward EVALUATION of an existing
                         checkpoint, not walk-forward RE-TRAINING (which
                         would retrain on each rolling window before
                         testing the next -- a real, larger, deliberately
                         deferred future step).

DETERMINISTIC EVALUATION: MuZero always called with temperature=0.0,
add_exploration_noise=False. PPO always called with deterministic=True.
No fallback chains -- a signature mismatch raises immediately.

Reward is always raw net log-return (use_dsr=False) -- financial
metrics reflect true realised performance, never the DSR training signal.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich import box

from configs.base_config import config as base_config
from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent, load_muzero_checkpoint
from src.agents.ppo.ppo_agent import PPOAgent
from src.utils.dashboard_logger import QuantRLLogger
from src.utils.metrics import compute_all_metrics, annualized_volatility, worst_single_period_loss
from src.risk.risk_manager import RiskManager

console = Console()

ACTION_NAMES = {0: "Short", 1: "Neutral", 2: "Long"}
POSITION_EXPOSURE_PCT = {0: -100.0, 1: 0.0, 2: 100.0}


# --------------------------------------------------------------------------
# Data / checkpoint loading
# --------------------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    path = os.path.join("data", "processed", f"{split}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pipeline/processor.py` first."
        )
    return pd.read_csv(path)


def load_val_and_test_combined() -> pd.DataFrame:
    """
    val_data.csv and test_data.csv are both normalised against the SAME
    training-split statistics (processor.py's normalize() fits mean/std
    on train only, then applies identically to val and test) -- so
    concatenating them is numerically consistent, not two differently-
    scaled series stitched together. Assumes split_data()'s embargo gap
    between val and test still holds as originally designed; not
    re-verified this round.
    """
    val_df = load_split("val")
    test_df = load_split("test")
    return pd.concat([val_df, test_df], ignore_index=True)


def find_latest_checkpoint(prefix: str):
    files = glob.glob(os.path.join("src", "checkpoints", f"{prefix}_*.pth"))
    if not files:
        return None

    def extract_ep(f):
        try:
            return int(f.split("_")[-1].replace(".pth", ""))
        except ValueError:
            return -1

    return max(files, key=extract_ep)


def list_checkpoints(prefix: str):
    """All checkpoints matching the prefix, as [(episode, path), ...],
    sorted ascending by episode. Used by progression mode."""
    files = glob.glob(os.path.join("src", "checkpoints", f"{prefix}_*.pth"))

    def extract_ep(f):
        try:
            return int(f.split("_")[-1].replace(".pth", ""))
        except ValueError:
            return -1

    pairs = [(extract_ep(f), f) for f in files]
    pairs = [p for p in pairs if p[0] >= 0]
    return sorted(pairs, key=lambda p: p[0])


def _evenly_spaced_indices(n: int, k: int):
    """k evenly spaced indices from range(n), always including 0 and
    n-1 when k>=2 and n>k. Returns all indices if k<=0 or n<=k."""
    if k <= 0 or n <= k:
        return list(range(n))
    if k == 1:
        return [n - 1]
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


# --------------------------------------------------------------------------
# Agent construction (load once, reuse across many evaluation calls)
# --------------------------------------------------------------------------

def _load_muzero_agent(checkpoint_path, obs_shape):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No MuZero checkpoint found ({checkpoint_path}). "
            f"Train one first with `python main_muzero.py`."
        )
    cfg = MuZeroConfig()
    agent = MuZeroAgent(cfg, obs_shape)
    load_muzero_checkpoint(agent, checkpoint_path)
    return agent


def _load_ppo_agent(checkpoint_path, obs_shape, action_dim):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No PPO checkpoint found ({checkpoint_path}). "
            f"Train one first with `python main_ppo.py`."
        )
    agent = PPOAgent(obs_shape=obs_shape, action_dim=action_dim)
    agent.policy.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    agent.policy.eval()
    return agent


# --------------------------------------------------------------------------
# Core episode runner
# --------------------------------------------------------------------------

def run_episode(env: StockTradingEnv, action_fn, risk_manager: RiskManager = None):
    """
    Returns (portfolio_history, action_counts, position_history,
    step_indices, risk_overrides).
    """
    obs, _ = env.reset()
    done = False
    action_counts = {0: 0, 1: 0, 2: 0}
    position_history = []
    step_indices = []

    if risk_manager is not None:
        risk_manager.reset(initial_value=env.portfolio_value)

    while not done:
        step_indices.append(env.current_step)
        proposed_action = action_fn(obs)
        action = (
            risk_manager.filter_action(proposed_action, env.portfolio_value)
            if risk_manager is not None else proposed_action
        )
        action_counts[int(action)] += 1
        position_history.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    risk_overrides = risk_manager.triggered_count if risk_manager is not None else 0
    return env.portfolio_history, action_counts, position_history, step_indices, risk_overrides


def evaluate_on_df(df, agent_kind, agent, risk_manager: RiskManager = None):
    """
    agent_kind: "muzero" | "ppo" | "buy_and_hold". agent is ignored for
    buy_and_hold. Single point through which every mode (single split,
    progression, walk-forward) runs an evaluation episode.
    """
    env = StockTradingEnv(df, use_dsr=False)

    if agent_kind == "buy_and_hold":
        action_fn = lambda obs: 2
    elif agent_kind == "muzero":
        action_fn = lambda obs: agent.select_action(obs, temperature=0.0, add_exploration_noise=False)[0]
    elif agent_kind == "ppo":
        action_fn = lambda obs: agent.select_action(obs, deterministic=True)[0]
    else:
        raise ValueError(f"Unknown agent_kind: {agent_kind}")

    return run_episode(env, action_fn, risk_manager=risk_manager)


# --------------------------------------------------------------------------
# Shared helpers: metrics printing, export, plotting
# --------------------------------------------------------------------------

def action_distribution_pct(action_counts):
    total = sum(action_counts.values())
    if total == 0:
        return {ACTION_NAMES[a]: 0.0 for a in ACTION_NAMES}
    return {ACTION_NAMES[a]: 100.0 * c / total for a, c in action_counts.items()}


def count_position_changes(position_history):
    return sum(1 for i in range(1, len(position_history)) if position_history[i] != position_history[i - 1])


def export_equity_and_positions(name, df, portfolio_history, position_history, step_indices,
                                  split, out_dir="logs/evaluation"):
    os.makedirs(out_dir, exist_ok=True)
    # Align by the SHORTEST of the three -- the terminal env step can
    # append one more entry to position_history/step_indices than there
    # are aligned dates/portfolio values remaining.
    n = min(len(step_indices), len(position_history), len(portfolio_history) - 1)

    dates = df['date'].iloc[step_indices[:n]].reset_index(drop=True)
    positions = position_history[:n]

    out_df = pd.DataFrame({
        "date": dates,
        "portfolio_value": portfolio_history[1:1 + n],
        "position_action": positions,
        "position_label": [ACTION_NAMES[a] for a in positions],
        "exposure_pct": [POSITION_EXPOSURE_PCT[a] for a in positions],
    })

    out_path = os.path.join(out_dir, f"{split}_{name.lower().replace(' ', '_')}_equity.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved equity/position history: {out_path}")
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


def print_report(name, portfolio_history, action_counts, position_history, risk_overrides=0):
    metrics = compute_all_metrics(portfolio_history)
    dist = action_distribution_pct(action_counts)

    logger = QuantRLLogger(agent_name=name, asset_symbol="")
    logger.print_evaluation_report(name, metrics, dist, portfolio_history[-1])

    vol = annualized_volatility(portfolio_history)
    worst_day = worst_single_period_loss(portfolio_history)
    changes = count_position_changes(position_history)
    print(f"  Annualised Volatility : {vol:.2%}")
    print(f"  Worst Single-Step Loss: {worst_day:.2%}")
    print(f"  Position Changes      : {changes} (over {len(position_history)} steps)")
    if risk_overrides:
        print(f"  Risk Overrides        : {risk_overrides} steps forced flat")

    return metrics


def print_comparison_table(results: dict):
    """Direct answer to 'what's the difference between them' -- every
    agent's headline metrics side by side, not in separate panels."""
    table = Table(title="Agent Comparison", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Metric", style="bold white")
    for name in results:
        table.add_column(name, justify="right")

    rows = [
        ("Total Return", "total_return", "{:+.2%}"),
        ("CAGR", "cagr", "{:+.2%}"),
        ("Sharpe", "sharpe_ratio", "{:.2f}"),
        ("Sortino", "sortino_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar", "calmar_ratio", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.2%}"),
    ]
    for label, key, fmt in rows:
        table.add_row(label, *[fmt.format(m[key]) for m in results.values()])
    console.print(table)


# --------------------------------------------------------------------------
# Mode: single
# --------------------------------------------------------------------------

def run_single_mode(args):
    df = load_split(args.split)
    print(f"Evaluating on '{args.split}' split ({len(df)} rows).")

    muzero_ckpt = (
        args.muzero_ckpt or (args.checkpoint if args.agent == "muzero" else None)
        or find_latest_checkpoint("muzero_checkpoint")
    )
    ppo_ckpt = (
        args.ppo_ckpt or (args.checkpoint if args.agent == "ppo" else None)
        or find_latest_checkpoint("ppo_checkpoint")
    )

    template_env = StockTradingEnv(df, use_dsr=False)
    obs_shape = template_env.observation_space.shape
    action_dim = template_env.action_space.n

    results = {}
    equity_dfs = {}

    if args.agent in ("muzero", "all"):
        mz_risk = RiskManager(max_drawdown=args.max_drawdown, cooldown_steps=args.cooldown_steps) if args.max_drawdown else None
        agent = _load_muzero_agent(muzero_ckpt, obs_shape)
        print(f"Loaded MuZero checkpoint: {muzero_ckpt}")
        history, counts, positions, steps, overrides = evaluate_on_df(df, "muzero", agent, risk_manager=mz_risk)
        results["MuZero"] = print_report("MuZero", history, counts, positions, risk_overrides=overrides)
        equity_dfs["MuZero"] = export_equity_and_positions("MuZero", df, history, positions, steps, args.split)
        plot_exposure_over_time("MuZero", equity_dfs["MuZero"], args.split)

    if args.agent in ("ppo", "all"):
        ppo_risk = RiskManager(max_drawdown=args.max_drawdown, cooldown_steps=args.cooldown_steps) if args.max_drawdown else None
        agent = _load_ppo_agent(ppo_ckpt, obs_shape, action_dim)
        print(f"Loaded PPO checkpoint: {ppo_ckpt}")
        history, counts, positions, steps, overrides = evaluate_on_df(df, "ppo", agent, risk_manager=ppo_risk)
        results["PPO"] = print_report("PPO", history, counts, positions, risk_overrides=overrides)
        equity_dfs["PPO"] = export_equity_and_positions("PPO", df, history, positions, steps, args.split)
        plot_exposure_over_time("PPO", equity_dfs["PPO"], args.split)

    if args.agent in ("buy_and_hold", "all"):
        history, counts, positions, steps, _ = evaluate_on_df(df, "buy_and_hold", None)
        results["Buy-and-Hold"] = print_report("Buy-and-Hold", history, counts, positions)
        equity_dfs["Buy-and-Hold"] = export_equity_and_positions("Buy-and-Hold", df, history, positions, steps, args.split)

    if len(results) > 1:
        print_comparison_table(results)
    if len(equity_dfs) > 1:
        plot_equity_comparison(equity_dfs, args.split)

    out_dir = os.path.join("logs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).T.to_csv(os.path.join(out_dir, f"{args.split}_results.csv"))
    print(f"\nSaved results to logs/evaluation/{args.split}_results.csv")


# --------------------------------------------------------------------------
# Mode: progression (does performance improve over training?)
# --------------------------------------------------------------------------

def run_progression_mode(args):
    if args.agent not in ("muzero", "ppo"):
        raise ValueError("--mode progression requires --agent muzero or --agent ppo (not all/buy_and_hold).")

    df = load_split(args.split)
    prefix = "muzero_checkpoint" if args.agent == "muzero" else "ppo_checkpoint"
    checkpoints = list_checkpoints(prefix)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found matching src/checkpoints/{prefix}_*.pth")

    idx = _evenly_spaced_indices(len(checkpoints), args.max_checkpoints)
    checkpoints = [checkpoints[i] for i in idx]
    print(f"Evaluating {len(checkpoints)} checkpoints on '{args.split}' split "
          f"(episodes: {[ep for ep, _ in checkpoints]})")

    template_env = StockTradingEnv(df, use_dsr=False)
    obs_shape = template_env.observation_space.shape
    action_dim = template_env.action_space.n

    bh_history, *_ = evaluate_on_df(df, "buy_and_hold", None)
    bh_metrics = compute_all_metrics(bh_history)

    rows = []
    for ep, path in checkpoints:
        agent = (_load_muzero_agent(path, obs_shape) if args.agent == "muzero"
                 else _load_ppo_agent(path, obs_shape, action_dim))
        history, counts, positions, *_ = evaluate_on_df(df, args.agent, agent)
        metrics = compute_all_metrics(history)
        dist = action_distribution_pct(counts)
        rows.append({
            "episode": ep, **metrics,
            "pct_short": dist["Short"], "pct_neutral": dist["Neutral"], "pct_long": dist["Long"],
            "position_changes": count_position_changes(positions),
        })
        print(f"  Episode {ep:>5}: Total Return {metrics['total_return']:+.2%}, "
              f"Sharpe {metrics['sharpe_ratio']:.2f}")

    prog_df = pd.DataFrame(rows)
    out_dir = os.path.join("logs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_{args.agent}_progression.csv")
    prog_df.to_csv(out_path, index=False)
    print(f"\nSaved progression table: {out_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(prog_df["episode"], prog_df["total_return"], marker="o")
    ax1.axhline(bh_metrics["total_return"], color="green", linestyle="--", label="Buy-and-Hold")
    ax1.set_ylabel("Total Return")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(prog_df["episode"], prog_df["sharpe_ratio"], marker="o", color="orange")
    ax2.axhline(bh_metrics["sharpe_ratio"], color="green", linestyle="--", label="Buy-and-Hold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_xlabel("Training Episode")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{args.agent.upper()} Performance vs. Training Episode ({args.split} split)")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{args.split}_{args.agent}_progression.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved progression chart: {plot_path}")


# --------------------------------------------------------------------------
# Mode: walk-forward (is the edge consistent across regimes?)
# --------------------------------------------------------------------------

def build_walk_forward_windows(df, lookback, window_length, step):
    """
    Each window = df.iloc[start : start + lookback + window_length].
    The first `lookback` rows build the initial observation (mirroring
    exactly how train/val/test's own boundaries already work); the
    remaining `window_length` rows are the period actually traded and
    evaluated. Windows roll forward by `step` rows each time.
    """
    windows = []
    start = 0
    while start + lookback + window_length <= len(df):
        windows.append(df.iloc[start: start + lookback + window_length].reset_index(drop=True))
        start += step
    return windows


def run_walk_forward_mode(args):
    if args.agent not in ("muzero", "ppo"):
        raise ValueError("--mode walk-forward requires --agent muzero or --agent ppo.")

    source_df = load_split("test") if args.wf_source == "test" else load_val_and_test_combined()
    lookback = base_config.LOOKBACK_WINDOW

    windows = build_walk_forward_windows(source_df, lookback, args.wf_window_length, args.wf_step)
    if not windows:
        raise ValueError(
            f"No walk-forward windows fit: source has {len(source_df)} rows, "
            f"need at least {lookback + args.wf_window_length}. "
            f"Try --wf-source val_test or a smaller --wf-window-length."
        )
    print(f"Built {len(windows)} walk-forward windows "
          f"(lookback={lookback}, window_length={args.wf_window_length}, step={args.wf_step}, "
          f"source={args.wf_source}, {len(source_df)} total rows).")

    template_env = StockTradingEnv(source_df, use_dsr=False)
    obs_shape = template_env.observation_space.shape
    action_dim = template_env.action_space.n

    checkpoint = (
        (args.muzero_ckpt if args.agent == "muzero" else args.ppo_ckpt)
        or args.checkpoint
        or find_latest_checkpoint(f"{args.agent}_checkpoint")
    )
    agent = (_load_muzero_agent(checkpoint, obs_shape) if args.agent == "muzero"
             else _load_ppo_agent(checkpoint, obs_shape, action_dim))
    print(f"Loaded {args.agent} checkpoint: {checkpoint}")

    rows = []
    for i, wdf in enumerate(windows):
        agent_hist, *_ = evaluate_on_df(wdf, args.agent, agent)
        bh_hist, *_ = evaluate_on_df(wdf, "buy_and_hold", None)  # matched baseline, same exact window
        m = compute_all_metrics(agent_hist)
        bh_m = compute_all_metrics(bh_hist)
        rows.append({
            "window": i,
            "start_date": wdf['date'].iloc[lookback],
            "end_date": wdf['date'].iloc[-1],
            "agent_return": m["total_return"], "agent_sharpe": m["sharpe_ratio"],
            "bh_return": bh_m["total_return"], "bh_sharpe": bh_m["sharpe_ratio"],
            "beat_bh": m["total_return"] > bh_m["total_return"],
        })
        print(f"  Window {i} [{wdf['date'].iloc[lookback]} → {wdf['date'].iloc[-1]}]: "
              f"agent {m['total_return']:+.2%} (Sharpe {m['sharpe_ratio']:.2f}) vs. "
              f"B&H {bh_m['total_return']:+.2%} (Sharpe {bh_m['sharpe_ratio']:.2f})")

    wf_df = pd.DataFrame(rows)
    out_dir = os.path.join("logs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.agent}_walk_forward.csv")
    wf_df.to_csv(out_path, index=False)

    win_rate = wf_df["beat_bh"].mean()
    print(f"\n=== Walk-Forward Summary ({len(windows)} windows) ===")
    print(f"  Beat Buy-and-Hold in {wf_df['beat_bh'].sum()}/{len(windows)} windows ({win_rate:.0%})")
    print(f"  Mean Sharpe   — agent: {wf_df['agent_sharpe'].mean():.2f}  |  B&H: {wf_df['bh_sharpe'].mean():.2f}")
    print(f"  Median Sharpe — agent: {wf_df['agent_sharpe'].median():.2f}  |  B&H: {wf_df['bh_sharpe'].median():.2f}")
    print(f"  Std of agent Sharpe across windows: {wf_df['agent_sharpe'].std():.2f} "
          f"(higher = less consistent across regimes)")
    print(f"Saved: {out_path}")

    x = np.arange(len(windows))
    width = 0.35
    plt.figure(figsize=(max(10, len(windows) * 0.8), 5))
    plt.bar(x - width / 2, wf_df["agent_sharpe"], width, label=args.agent.upper())
    plt.bar(x + width / 2, wf_df["bh_sharpe"], width, label="Buy-and-Hold")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, [d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d) for d in
                   pd.to_datetime(wf_df["start_date"])], rotation=45, ha="right")
    plt.ylabel("Sharpe Ratio")
    plt.title(f"{args.agent.upper()} vs. Buy-and-Hold — Sharpe per Walk-Forward Window")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{args.agent}_walk_forward.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved chart: {plot_path}")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trading agents.")
    parser.add_argument("--mode", choices=["single", "progression", "walk-forward"], default="single")
    parser.add_argument("--agent", choices=["muzero", "ppo", "buy_and_hold", "all"], default="all")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--muzero-ckpt", default=None)
    parser.add_argument("--ppo-ckpt", default=None)
    parser.add_argument("--max-drawdown", type=float, default=None,
                         help="Single mode only: enable the RiskManager kill-switch overlay.")
    parser.add_argument("--cooldown-steps", type=int, default=5)
    parser.add_argument("--max-checkpoints", type=int, default=10,
                         help="Progression mode: max checkpoints to evaluate (evenly spaced).")
    parser.add_argument("--wf-source", choices=["test", "val_test"], default="test",
                         help="Walk-forward mode: data to roll windows across.")
    parser.add_argument("--wf-window-length", type=int, default=60,
                         help="Walk-forward mode: trading days evaluated per window.")
    parser.add_argument("--wf-step", type=int, default=30,
                         help="Walk-forward mode: days to roll forward between windows.")
    args = parser.parse_args()

    if args.mode == "single":
        run_single_mode(args)
    elif args.mode == "progression":
        run_progression_mode(args)
    elif args.mode == "walk-forward":
        run_walk_forward_mode(args)


if __name__ == "__main__":
    main()