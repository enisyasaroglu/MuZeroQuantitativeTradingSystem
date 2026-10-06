"""
Evaluation script: runs one or more agents on a held-out data split and
reports the standardised financial metric set.

Supported agents:
    - muzero
    - ppo
    - sma
    - buy_and_hold
    - all

SMA baseline:
    - 50-period simple moving average
    - Long when price > SMA
    - Neutral when price <= SMA
    - Never shorts

Evaluation intentionally uses RAW net log-return as the environment reward
(use_dsr=False), not the shaped DSR training signal.

Evaluation is deterministic:
    - MuZero: temperature=0.0, no exploration noise
    - PPO: deterministic=True

Risk overlay:
    --max-drawdown enables the optional RiskManager kill-switch on
    MuZero and PPO only.

Buy-and-Hold and SMA remain pure benchmark strategies.
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

from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent, load_muzero_checkpoint
from src.agents.ppo.ppo_agent import PPOAgent
from src.utils.dashboard_logger import QuantRLLogger
from src.utils.metrics import (
    compute_all_metrics,
    annualized_volatility,
    worst_single_period_loss,
)
from src.risk.risk_manager import RiskManager
from src.strategies import BuyAndHoldStrategy, SMAStrategy


ACTION_NAMES = {
    0: "Short",
    1: "Neutral",
    2: "Long",
}

POSITION_EXPOSURE_PCT = {
    0: -100.0,
    1: 0.0,
    2: 100.0,
}

SMA_WINDOW = 50


def load_split(split: str) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "processed",
        f"{split}_data.csv",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run "
            "`python src/pipeline/processor.py` first "
            "to generate train/val/test_data.csv."
        )

    return pd.read_csv(path)


def find_latest_checkpoint(prefix: str):
    """
    Find the checkpoint with the highest episode number matching:

        checkpoints/{prefix}_*.pth
    """

    files = glob.glob(
        os.path.join(
            "checkpoints",
            f"{prefix}_*.pth",
        )
    )

    if not files:
        return None

    def extract_ep(filepath):
        try:
            return int(
                filepath.split("_")[-1].replace(".pth", "")
            )
        except ValueError:
            return -1

    return max(files, key=extract_ep)


def run_episode(
    env: StockTradingEnv,
    action_fn,
    risk_manager: RiskManager = None,
):
    """
    Run one complete pass over the split.

    action_fn(obs) -> int action

    Returns:
        portfolio_history
        action_counts
        position_history
        step_indices
        risk_overrides
    """

    obs, _ = env.reset()

    done = False

    action_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    position_history = []
    step_indices = []

    if risk_manager is not None:
        risk_manager.reset(
            initial_value=env.portfolio_value
        )

    while not done:

        step_indices.append(
            env.current_step
        )

        proposed_action = action_fn(obs)

        if risk_manager is not None:
            action = risk_manager.filter_action(
                proposed_action,
                env.portfolio_value,
            )
        else:
            action = proposed_action

        action_counts[int(action)] += 1
        position_history.append(int(action))

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

    risk_overrides = (
        risk_manager.triggered_count
        if risk_manager is not None
        else 0
    )

    return (
        env.portfolio_history,
        action_counts,
        position_history,
        step_indices,
        risk_overrides,
    )


def evaluate_muzero(
    df,
    checkpoint_path,
    risk_manager: RiskManager = None,
):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No MuZero checkpoint found ({checkpoint_path}). "
            "Aborting rather than evaluating an untrained network. "
            "Train one first with `python main_muzero.py`."
        )

    env = StockTradingEnv(
        df,
        use_dsr=False,
    )

    cfg = MuZeroConfig()

    agent = MuZeroAgent(
        cfg,
        env.observation_space.shape,
    )

    load_muzero_checkpoint(
        agent,
        checkpoint_path,
    )

    print(
        f"Loaded MuZero checkpoint: {checkpoint_path}"
    )

    def action_fn(obs):
        action, _, _ = agent.select_action(
            obs,
            temperature=0.0,
            add_exploration_noise=False,
        )
        return action

    return run_episode(
        env,
        action_fn,
        risk_manager=risk_manager,
    )


def evaluate_ppo(
    df,
    checkpoint_path,
    risk_manager: RiskManager = None,
):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No PPO checkpoint found ({checkpoint_path}). "
            "Aborting rather than evaluating an untrained network. "
            "Train one first with `python main_ppo.py`."
        )

    env = StockTradingEnv(
        df,
        use_dsr=False,
    )

    agent = PPOAgent(
        obs_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
    )

    agent.policy.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=agent.device,
        )
    )

    agent.policy.eval()

    print(
        f"Loaded PPO checkpoint: {checkpoint_path}"
    )

    def action_fn(obs):
        action, _, _ = agent.select_action(
            obs,
            deterministic=True,
        )
        return action

    return run_episode(
        env,
        action_fn,
        risk_manager=risk_manager,
    )


def evaluate_buy_and_hold(df):
    """
    Pure Buy-and-Hold benchmark.

    Always maintains a Long position.

    No risk overlay is applied.
    """

    env = StockTradingEnv(
        df,
        use_dsr=False,
    )

    return run_episode(
        env,
        action_fn=lambda obs: 2,
    )


def reconstruct_price_from_log_returns(df):
    """
    Reconstruct a relative price index from log returns.

    The processed dataset intentionally removes raw OHLC prices,
    but retains raw `log_return`.

    Given:

        log_return_t = log(P_t / P_{t-1})

    cumulative exponentiation reconstructs the price path
    up to an arbitrary constant.

    The arbitrary starting price does not affect SMA signals.
    """

    if "log_return" not in df.columns:
        raise ValueError(
            "SMA evaluation requires the processed dataframe "
            "to contain the raw 'log_return' column."
        )

    log_returns = pd.to_numeric(
        df["log_return"],
        errors="coerce",
    )

    if log_returns.isna().any():
        raise ValueError(
            "SMA evaluation found NaN values in 'log_return'. "
            "Clean the processed test data before evaluation."
        )

    relative_price = np.exp(
        log_returns.cumsum()
    )

    return pd.Series(
        relative_price,
        index=df.index,
        name="close",
    )


def evaluate_sma(df, window=SMA_WINDOW):
    """
    Evaluate a Simple Moving Average baseline.

    Strategy:
        price > SMA(window) -> Long
        price <= SMA(window) -> Neutral

    No shorting.

    The processed dataset does not contain raw close prices,
    so a relative price series is reconstructed from raw log returns.
    """

    if window <= 0:
        raise ValueError(
            "SMA window must be greater than zero."
        )

    # Reconstruct relative price from raw log returns.
    sma_df = df.copy()

    sma_df["close"] = reconstruct_price_from_log_returns(
        sma_df
    )

    strategy = SMAStrategy(
        window=window
    )

    signals = strategy.generate_signals(
        sma_df
    )

    env = StockTradingEnv(
        df,
        use_dsr=False,
    )

    def action_fn(obs):
        current_step = env.current_step

        signal = int(
            signals.iloc[current_step]["signal"]
        )

        # SMAStrategy:
        #   0 = Neutral
        #   1 = Long
        #
        # Environment:
        #   0 = Short
        #   1 = Neutral
        #   2 = Long

        if signal == 1:
            return 2

        return 1

    print(
        f"SMA strategy: {window}-period SMA"
    )

    return run_episode(
        env,
        action_fn,
    )


def action_distribution_pct(action_counts):
    total = sum(
        action_counts.values()
    )

    if total == 0:
        return {
            ACTION_NAMES[a]: 0.0
            for a in ACTION_NAMES
        }

    return {
        ACTION_NAMES[a]: 100.0 * count / total
        for a, count in action_counts.items()
    }


def export_equity_and_positions(
    name,
    df,
    portfolio_history,
    position_history,
    step_indices,
    split,
    out_dir="logs/evaluation",
):
    """
    Save complete per-step portfolio and position history.
    """

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    n = min(
        len(step_indices),
        len(position_history),
        len(portfolio_history) - 1,
    )

    dates = (
        df["date"]
        .iloc[step_indices[:n]]
        .reset_index(drop=True)
    )

    trimmed_positions = position_history[:n]

    out_df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": portfolio_history[
                1:1 + n
            ],
            "position_action": trimmed_positions,
            "position_label": [
                ACTION_NAMES[a]
                for a in trimmed_positions
            ],
            "exposure_pct": [
                POSITION_EXPOSURE_PCT[a]
                for a in trimmed_positions
            ],
        }
    )

    filename = (
        f"{split}_"
        f"{name.lower().replace(' ', '_')}"
        f"_equity.csv"
    )

    out_path = os.path.join(
        out_dir,
        filename,
    )

    out_df.to_csv(
        out_path,
        index=False,
    )

    print(
        f"  Saved full equity/position history: "
        f"{out_path}"
    )

    return out_df


def plot_equity_comparison(
    equity_dfs: dict,
    split,
    out_dir="logs/evaluation",
):
    """
    Plot all evaluated equity curves together.
    """

    plt.figure(
        figsize=(11, 5)
    )

    for name, edf in equity_dfs.items():

        plt.plot(
            pd.to_datetime(edf["date"]),
            edf["portfolio_value"],
            label=name,
            linewidth=1.6,
        )

    plt.title(
        f"Equity Curve Comparison ({split} split)"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    out_path = os.path.join(
        out_dir,
        f"{split}_equity_comparison.png",
    )

    plt.savefig(
        out_path,
        dpi=150,
    )

    plt.close()

    print(
        f"  Saved equity curve comparison: "
        f"{out_path}"
    )


def plot_exposure_over_time(
    name,
    equity_df,
    split,
    out_dir="logs/evaluation",
):
    """
    Plot position/exposure over time for one strategy.
    """

    plt.figure(
        figsize=(11, 2.5)
    )

    plt.step(
        pd.to_datetime(equity_df["date"]),
        equity_df["exposure_pct"],
        where="post",
        linewidth=1.2,
    )

    plt.title(
        f"{name} — Exposure Over Time ({split} split)"
    )

    plt.xlabel("Date")
    plt.ylabel("Exposure (%)")

    plt.ylim(
        -110,
        110,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    out_path = os.path.join(
        out_dir,
        f"{split}_"
        f"{name.lower().replace(' ', '_')}"
        f"_exposure.png",
    )

    plt.savefig(
        out_path,
        dpi=150,
    )

    plt.close()

    print(
        f"  Saved exposure chart: "
        f"{out_path}"
    )


def print_report(
    name,
    portfolio_history,
    action_counts,
    risk_overrides=0,
):
    metrics = compute_all_metrics(
        portfolio_history
    )

    dist = action_distribution_pct(
        action_counts
    )

    logger = QuantRLLogger(
        agent_name=name,
        asset_symbol="",
    )

    logger.print_evaluation_report(
        name,
        metrics,
        dist,
        portfolio_history[-1],
    )

    vol = annualized_volatility(
        portfolio_history
    )

    worst_day = worst_single_period_loss(
        portfolio_history
    )

    print(
        f"  Annualised Volatility : {vol:.2%}"
    )

    print(
        f"  Worst Single-Step Loss: {worst_day:.2%}"
    )

    if risk_overrides:
        print(
            f"  Risk Overrides        : "
            f"{risk_overrides} steps forced flat"
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trading agents on a held-out split."
        )
    )

    parser.add_argument(
        "--agent",
        choices=[
            "muzero",
            "ppo",
            "sma",
            "buy_and_hold",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a .pth checkpoint. "
            "Only applied when --agent is muzero "
            "or ppo specifically."
        ),
    )

    parser.add_argument(
        "--muzero-ckpt",
        default=None,
        help="Override MuZero checkpoint path.",
    )

    parser.add_argument(
        "--ppo-ckpt",
        default=None,
        help="Override PPO checkpoint path.",
    )

    parser.add_argument(
        "--sma-window",
        type=int,
        default=SMA_WINDOW,
        help=(
            "SMA lookback window. "
            f"Default: {SMA_WINDOW}."
        ),
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help=(
            "Enable RiskManager kill-switch "
            "overlay, e.g. 0.15 for 15%%. "
            "Applied to MuZero/PPO only."
        ),
    )

    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=5,
        help=(
            "Steps to stay forced-flat after "
            "a kill-switch trigger."
        ),
    )

    args = parser.parse_args()

    df = load_split(
        args.split
    )

    print(
        f"Evaluating on '{args.split}' split "
        f"({len(df)} rows)."
    )

    muzero_ckpt = (
        args.muzero_ckpt
        or (
            args.checkpoint
            if args.agent == "muzero"
            else None
        )
        or find_latest_checkpoint(
            "muzero_checkpoint"
        )
    )

    ppo_ckpt = (
        args.ppo_ckpt
        or (
            args.checkpoint
            if args.agent == "ppo"
            else None
        )
        or find_latest_checkpoint(
            "ppo_checkpoint"
        )
    )

    results = {}
    equity_dfs = {}

    # MuZero
    if args.agent in (
        "muzero",
        "all",
    ):

        mz_risk = (
            RiskManager(
                max_drawdown=args.max_drawdown,
                cooldown_steps=args.cooldown_steps,
            )
            if args.max_drawdown
            else None
        )

        (
            history,
            counts,
            positions,
            steps,
            overrides,
        ) = evaluate_muzero(
            df,
            muzero_ckpt,
            risk_manager=mz_risk,
        )

        results["MuZero"] = print_report(
            "MuZero",
            history,
            counts,
            risk_overrides=overrides,
        )

        equity_dfs["MuZero"] = (
            export_equity_and_positions(
                "MuZero",
                df,
                history,
                positions,
                steps,
                args.split,
            )
        )

        plot_exposure_over_time(
            "MuZero",
            equity_dfs["MuZero"],
            args.split,
        )

    # PPO
    if args.agent in (
        "ppo",
        "all",
    ):

        ppo_risk = (
            RiskManager(
                max_drawdown=args.max_drawdown,
                cooldown_steps=args.cooldown_steps,
            )
            if args.max_drawdown
            else None
        )

        (
            history,
            counts,
            positions,
            steps,
            overrides,
        ) = evaluate_ppo(
            df,
            ppo_ckpt,
            risk_manager=ppo_risk,
        )

        results["PPO"] = print_report(
            "PPO",
            history,
            counts,
            risk_overrides=overrides,
        )

        equity_dfs["PPO"] = (
            export_equity_and_positions(
                "PPO",
                df,
                history,
                positions,
                steps,
                args.split,
            )
        )

        plot_exposure_over_time(
            "PPO",
            equity_dfs["PPO"],
            args.split,
        )

    # SMA
    if args.agent in (
        "sma",
        "all",
    ):

        (
            history,
            counts,
            positions,
            steps,
            _,
        ) = evaluate_sma(
            df,
            window=args.sma_window,
        )

        sma_name = (
            f"SMA({args.sma_window})"
        )

        results[sma_name] = print_report(
            sma_name,
            history,
            counts,
        )

        equity_dfs[sma_name] = (
            export_equity_and_positions(
                sma_name,
                df,
                history,
                positions,
                steps,
                args.split,
            )
        )

        plot_exposure_over_time(
            sma_name,
            equity_dfs[sma_name],
            args.split,
        )

    # Buy & Hold
    if args.agent in (
        "buy_and_hold",
        "all",
    ):

        (
            history,
            counts,
            positions,
            steps,
            _,
        ) = evaluate_buy_and_hold(
            df
        )

        results["Buy-and-Hold"] = (
            print_report(
                "Buy-and-Hold",
                history,
                counts,
            )
        )

        equity_dfs["Buy-and-Hold"] = (
            export_equity_and_positions(
                "Buy-and-Hold",
                df,
                history,
                positions,
                steps,
                args.split,
            )
        )
 
    # Comparison chart
    if len(equity_dfs) > 1:
        plot_equity_comparison(
            equity_dfs,
            args.split,
        )

    # Save results
    out_dir = os.path.join(
        "logs",
        "evaluation",
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    out_path = os.path.join(
        out_dir,
        f"{args.split}_results.csv",
    )

    pd.DataFrame(
        results
    ).T.to_csv(
        out_path
    )

    print(
        f"\nSaved results to {out_path}"
    )


if __name__ == "__main__":
    main()