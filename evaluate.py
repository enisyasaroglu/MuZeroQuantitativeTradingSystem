"""
Evaluation script: plays one or more agents over a held-out split in the
trading environment and reports the standard financial metrics.

Examples
--------
    # Baselines only
    python3 evaluate.py --agent all --split val

    # A MuZero checkpoint plus the baselines
    python3 evaluate.py --agent all --split val \
        --muzero-ckpt runs/run_001/muzero_checkpoint_500.pth

    # PPO only
    python3 evaluate.py --agent ppo --split val --ppo-ckpt runs/ppo_001/ppo_final.pth

Rules this script follows
-------------------------
* Checkpoints are never guessed: pass --muzero-ckpt / --ppo-ckpt explicitly.
* MuZero is evaluated with the config saved next to its checkpoint.
* Evaluation uses raw net log-returns (no DSR), deterministic actions
  (MuZero temperature 0, no noise; PPO argmax).
* The default split is 'val'. Every run is appended to
  logs/evaluation/evaluation_log.csv, so test-set use is always on record.
"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, fields
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from configs.base_config import config
from configs.muzero_config import MuZeroConfig
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.agents.ppo.ppo_agent import PPOAgent
from src.env.trading_env import StockTradingEnv
from src.risk.risk_manager import RiskManager
from src.utils.dashboard_logger import QuantRLLogger
from src.utils.metrics import compute_all_metrics

ACTION_NAMES = {0: "Short", 1: "Neutral", 2: "Long"}
EXPOSURE_PCT = {0: -100.0, 1: 0.0, 2: 100.0}
DEFAULT_OUT_DIR = os.path.join("logs", "evaluation")
SMA_WINDOW = 50
BASELINES = ["buy_and_hold", "cash", "sma"]


# Small helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def json_safe(value):
    """Make nested values JSON-friendly (NaN/inf become null)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_split(split: str):
    path = os.path.join("data", "processed", f"{split}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python3 src/pipeline/processor.py` first."
        )
    return pd.read_csv(path), path


# Playing a split
@dataclass
class EpisodeResult:
    portfolio_history: list
    positions: list
    step_indices: list
    risk_overrides: int


def play_split(env, action_fn, risk_manager=None) -> EpisodeResult:
    """Run one full chronological pass. action_fn(obs) -> action in {0,1,2}."""
    obs, _ = env.reset()
    if risk_manager is not None:
        risk_manager.reset(initial_value=env.portfolio_value)

    positions, step_indices = [], []
    done = False
    while not done:
        step_indices.append(env.current_step)
        action = int(action_fn(obs))
        if risk_manager is not None:
            action = int(risk_manager.filter_action(action, env.portfolio_value))
        positions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    overrides = risk_manager.triggered_count if risk_manager is not None else 0
    return EpisodeResult(list(env.portfolio_history), positions, step_indices, overrides)


# Agents: each builder returns (display_name, action_fn, metadata)
def check_input_width(state_dict, key, env, label):
    """Fail early, readably, if the checkpoint was trained on other features."""
    expected = state_dict[key].shape[1]
    actual = env.observation_space.shape[1]
    if expected != actual:
        raise ValueError(
            f"{label} checkpoint expects {expected} input columns but the data "
            f"gives {actual}. Features now: {env.feature_cols} (plus position). "
            "Regenerate the data or use a matching checkpoint."
        )


def load_muzero_config(ckpt: str) -> MuZeroConfig:
    """Use the config saved next to the checkpoint, so evaluation runs with
    the settings (search simulations, discount, ...) the model trained with."""
    cfg_path = ckpt.replace(".pth", "_config.json")
    if not os.path.exists(cfg_path):
        print(f"  WARNING: no saved config next to {ckpt}; using current defaults.")
        return MuZeroConfig()
    with open(cfg_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    known = {fld.name for fld in fields(MuZeroConfig)}
    kwargs = {k: v for k, v in saved.items() if k in known}
    missing = sorted(known - set(kwargs))
    if missing:
        print(f"  WARNING: saved config lacks {missing}; defaults used for these.")
    return MuZeroConfig(**kwargs)


def build_muzero(env, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    check_input_width(state_dict, "representation.encoder.rnn.weight_ih_l0", env, "MuZero")
    cfg = load_muzero_config(ckpt)

    agent = MuZeroAgent(cfg, env.observation_space.shape)
    agent.network.load_state_dict(state_dict)
    agent.network.eval()
    print(f"  Loaded {ckpt} ({cfg.num_simulations} simulations per decision)")

    def action_fn(obs):
        action, _, _ = agent.select_action(
            obs, temperature=0.0, add_exploration_noise=False
        )
        return action

    meta = {"checkpoint": ckpt, "checkpoint_sha256": file_sha256(ckpt), "config": asdict(cfg)}
    return "MuZero", action_fn, meta


def build_ppo(env, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    check_input_width(state_dict, "encoder.rnn.weight_ih_l0", env, "PPO")

    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=env.action_space.n)
    agent.policy.load_state_dict(state_dict)
    agent.policy.eval()
    print(f"  Loaded {ckpt}")

    def action_fn(obs):
        action, _, _ = agent.select_action(obs, deterministic=True)
        return action

    meta = {"checkpoint": ckpt, "checkpoint_sha256": file_sha256(ckpt)}
    return "PPO", action_fn, meta


def build_buy_and_hold():
    return "Buy-and-Hold", (lambda obs: 2), {}


def build_cash():
    return "Cash", (lambda obs: 1), {}


def build_sma(env, df, window):
    """Long when yesterday's close was above its moving average, else cash.

    Timing: at row t the agent has seen closes up to row t-1 and earns the
    return of row t. So the signal for row t is (price > SMA) evaluated at
    row t-1. A relative price path is rebuilt from the raw log returns; its
    arbitrary starting level does not affect the signal.
    """
    if window <= 0:
        raise ValueError("SMA window must be greater than zero.")
    log_returns = pd.to_numeric(df["log_return"], errors="coerce")
    if log_returns.isna().any():
        raise ValueError("SMA baseline found NaN values in 'log_return'.")

    price = np.exp(log_returns.cumsum())
    above = (price > price.rolling(window).mean()).shift(1, fill_value=False)
    long_signal = above.to_numpy()

    def action_fn(obs):
        return 2 if long_signal[env.current_step] else 1

    return f"SMA({window})", action_fn, {"window": window}


def build_agent(key, env, df, args):
    if key == "muzero":
        return build_muzero(env, args.muzero_ckpt)
    if key == "ppo":
        return build_ppo(env, args.ppo_ckpt)
    if key == "buy_and_hold":
        return build_buy_and_hold()
    if key == "cash":
        return build_cash()
    if key == "sma":
        return build_sma(env, df, args.sma_window)
    raise ValueError(f"Unknown agent: {key}")


# Reporting and exports
def report(name, result):
    metrics = compute_all_metrics(result.portfolio_history)
    counts = np.bincount(result.positions, minlength=3)
    dist = {ACTION_NAMES[a]: 100.0 * counts[a] / counts.sum() for a in ACTION_NAMES}

    QuantRLLogger(agent_name=name, asset_symbol="").print_evaluation_report(
        name, metrics, dist, result.portfolio_history[-1]
    )
    print(f"  Annualised Volatility : {metrics['annualized_volatility']:.2%}")
    print(f"  Worst Single-Step Loss: {metrics['worst_single_period_loss']:.2%}")
    if result.risk_overrides:
        print(f"  Risk Overrides        : {result.risk_overrides} steps forced flat")
    return metrics, dist


def build_equity_frame(df, result):
    n = len(result.positions)
    return pd.DataFrame(
        {
            "date": df["date"].iloc[result.step_indices].reset_index(drop=True),
            "portfolio_value": result.portfolio_history[1 : 1 + n],
            "position_action": result.positions,
            "position_label": [ACTION_NAMES[a] for a in result.positions],
            "exposure_pct": [EXPOSURE_PCT[a] for a in result.positions],
        }
    )


def plot_exposure(name, frame, split, out_dir):
    plt.figure(figsize=(11, 2.5))
    plt.step(pd.to_datetime(frame["date"]), frame["exposure_pct"], where="post", linewidth=1.2)
    plt.title(f"{name} - Exposure Over Time ({split} split)")
    plt.xlabel("Date")
    plt.ylabel("Exposure (%)")
    plt.ylim(-110, 110)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{split}_{slug(name)}_exposure.png"), dpi=150)
    plt.close()


def plot_comparison(frames, split, out_dir):
    plt.figure(figsize=(11, 5))
    for name, frame in frames.items():
        plt.plot(pd.to_datetime(frame["date"]), frame["portfolio_value"], label=name, linewidth=1.6)
    plt.title(f"Equity Curve Comparison ({split} split)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{split}_equity_comparison.png"), dpi=150)
    plt.close()


def write_run_record(path, record):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(record), handle, indent=4)


def append_log(out_dir, row):
    path = os.path.join(out_dir, "evaluation_log.csv")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# Command-line 
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trading agents on a held-out split.")
    parser.add_argument("--agent", default="all",
                        choices=["muzero", "ppo", "sma", "buy_and_hold", "cash", "all"],
                        help="'all' = baselines plus every agent whose checkpoint you pass.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--muzero-ckpt", default=None)
    parser.add_argument("--ppo-ckpt", default=None)
    parser.add_argument("--sma-window", type=int, default=SMA_WINDOW)
    parser.add_argument("--max-drawdown", type=float, default=None,
                        help="Optional RiskManager kill-switch for MuZero/PPO, e.g. 0.15.")
    parser.add_argument("--cooldown-steps", type=int, default=5)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.agent == "muzero" and not args.muzero_ckpt:
        parser.error("--muzero-ckpt is required with --agent muzero")
    if args.agent == "ppo" and not args.ppo_ckpt:
        parser.error("--ppo-ckpt is required with --agent ppo")
    return args


def resolve_agents(args):
    if args.agent != "all":
        return [args.agent]
    agents = []
    if args.muzero_ckpt:
        agents.append("muzero")
    if args.ppo_ckpt:
        agents.append("ppo")
    return agents + BASELINES


def make_risk_manager(key, args):
    if args.max_drawdown and key in ("muzero", "ppo"):
        return RiskManager(max_drawdown=args.max_drawdown, cooldown_steps=args.cooldown_steps)
    return None


def main():
    args = parse_args()
    set_seed(config.SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    df, data_path = load_split(args.split)
    data_sha = file_sha256(data_path)
    agents = resolve_agents(args)

    probe = StockTradingEnv(df, use_dsr=False)
    print(f"Evaluating on '{args.split}' split: {len(df)} rows, "
          f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Traded days start at {df['date'].iloc[probe.lookback]} "
          f"(the first {probe.lookback} rows are lookback only).")
    print(f"Features ({len(probe.feature_cols)}): {probe.feature_cols}")
    print(f"Agents: {agents}")
    if args.split == "test":
        print("NOTE: test-set run. It will be recorded in evaluation_log.csv.")

    table, frames = {}, {}
    for key in agents:
        env = StockTradingEnv(df, use_dsr=False)
        print(f"\n=== {key} ===")
        name, action_fn, meta = build_agent(key, env, df, args)
        result = play_split(env, action_fn, make_risk_manager(key, args))
        metrics, dist = report(name, result)

        frame = build_equity_frame(df, result)
        frames[name] = frame
        table[name] = {k: v for k, v in metrics.items() if k != "rolling_drawdown_series"}

        stem = f"{args.split}_{slug(name)}"
        frame.to_csv(os.path.join(args.out_dir, f"{stem}_equity.csv"), index=False)
        if not args.no_plots:
            plot_exposure(name, frame, args.split, args.out_dir)

        write_run_record(
            os.path.join(args.out_dir, f"{stem}_run.json"),
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "agent": name,
                "split": args.split,
                "data_file": data_path,
                "data_sha256": data_sha,
                "rows": len(df),
                "feature_cols": probe.feature_cols,
                "obs_shape": list(probe.observation_space.shape),
                "traded_start": str(frame["date"].iloc[0]),
                "traded_end": str(frame["date"].iloc[-1]),
                "traded_steps": len(frame),
                "risk_overlay_max_drawdown": args.max_drawdown,
                "action_distribution_pct": dist,
                "metrics": table[name],
                "agent_meta": meta,
            },
        )
        append_log(args.out_dir, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "split": args.split,
            "agent": name,
            "checkpoint": meta.get("checkpoint", ""),
            "data_sha256_12": data_sha[:12],
            "traded_start": str(frame["date"].iloc[0]),
            "traded_end": str(frame["date"].iloc[-1]),
            "total_return": table[name]["total_return"],
            "sharpe_ratio": table[name]["sharpe_ratio"],
            "max_drawdown": table[name]["max_drawdown"],
        })

    if len(frames) > 1 and not args.no_plots:
        plot_comparison(frames, args.split, args.out_dir)

    out_path = os.path.join(args.out_dir, f"{args.split}_results_{args.agent}.csv")
    pd.DataFrame(table).T.to_csv(out_path)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()