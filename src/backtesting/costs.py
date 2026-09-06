"""
Fair, apples-to-apples comparison across every implemented portfolio
construction method (Equal Weight, Mean-Variance, Risk Parity, HRP,
CVaR, PSO, Robust/resampled, Black-Litterman with/without momentum
views).

"FAIR" means, specifically:
  - Same data: one raw returns matrix, every method sees identical
    history.
  - Same mechanics: every method wrapped in its own Portfolio instance
    (src/portfolio/portfolio.py), but with IDENTICAL constraints,
    rebalance cadence, and lookback window across all of them. Turnover
    and transaction cost use the exact exp(net_return) log-compounding
    and TRANSACTION_FEE * turnover convention already established in
    trading_env.py / multi_asset_env.py -- NOT a fourth reimplementation
    of portfolio compounding.
  - Same metrics: src.utils.metrics.compute_all_metrics -- the same,
    already-tested module, not a new one. (evaluate.py's own
    compute_metrics() was previously found to silently diverge from this
    module in ddof and drawdown sign convention -- reusing this module
    directly, rather than writing a fifth metrics implementation, is a
    deliberate choice not to repeat that mistake here.)
  - Stochastic methods get multiple seeded runs, reported as mean AND
    std (n_runs=20 by default, matching the "N=20-30 runs" standard
    already applied to the companion Swarm Intelligence PSO/ABC/ACO
    comparison). Deterministic methods (Equal Weight, Mean-Variance,
    Risk Parity, HRP, CVaR, both Black-Litterman variants) run exactly
    once -- running them N times would produce a std of exactly zero
    and waste compute, not add rigor.

DELIBERATELY NOT BUILT: automatic train/val/test looping. The caller
picks which returns_matrix (one split) to compare methods on; comparing
across splits means calling compare_methods() again with a different
matrix, using the already-separate per-split CSVs -- add real looping
later if a specific need shows up, not assumed necessary now.

Minor, separately-noted divergence from multi_asset_env.py: that
environment's termination check (current_step >= n_dates - 1) stops one
index earlier than strictly necessary, so its episodes never process
the very last available return. This harness processes the full
available range instead (range(lookback, n_dates)), since nothing here
requires stopping early. Low-priority, likely immaterial to results --
flagged for awareness, not fixed in multi_asset_env.py as a side effect
of building this file.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from configs.base_config import config
from src.portfolio.portfolio import Portfolio
from src.portfolio.constraints import PortfolioConstraints
from src.utils.metrics import compute_all_metrics

console = Console()


@dataclass
class MethodSpec:
    name: str
    optimizer_factory: Callable[[Optional[int]], object]  # seed (or None) -> optimizer instance
    stochastic: bool = False
    n_runs: int = 20  # only used when stochastic=True


def run_backtest(returns_matrix: np.ndarray, optimizer, constraints: PortfolioConstraints,
                  rebalance_every: int, lookback: int,
                  transaction_fee: float = None, initial_capital: float = None) -> np.ndarray:
    """
    Walks forward through returns_matrix exactly once, using Portfolio
    to decide weights and the same exp(net_return) compounding /
    TRANSACTION_FEE*turnover cost convention used everywhere else in
    this project. Returns the portfolio value history (length
    n_dates - lookback + 1, including the starting capital).
    """
    transaction_fee = config.TRANSACTION_FEE if transaction_fee is None else transaction_fee
    initial_capital = config.INITIAL_CAPITAL if initial_capital is None else initial_capital

    n_dates, n_assets = returns_matrix.shape
    portfolio = Portfolio(n_assets=n_assets, optimizer=optimizer, constraints=constraints,
                          rebalance_every=rebalance_every, lookback=lookback)
    portfolio.reset()

    portfolio_value = initial_capital
    history = [portfolio_value]

    for t in range(lookback, n_dates):
        target_weights = portfolio.propose_weights(t, returns_matrix)
        asset_returns = returns_matrix[t]

        gross_return = float(target_weights @ asset_returns)
        turnover = portfolio.turnover(target_weights)
        cost = transaction_fee * turnover
        net_return = gross_return - cost

        portfolio_value *= np.exp(net_return)
        history.append(portfolio_value)

        portfolio.commit(target_weights)

    return np.array(history)


def compare_methods(returns_matrix: np.ndarray, method_specs: List[MethodSpec],
                     constraints: PortfolioConstraints = None,
                     rebalance_every: int = 20, lookback: int = 60,
                     base_seed: int = 0):
    """
    Runs every MethodSpec through run_backtest with IDENTICAL
    constraints/rebalance_every/lookback, on the SAME returns_matrix.

    Returns (summary_df, raw_results):
      summary_df: one row per method, metric cells formatted as
        "X.XX%" for deterministic methods or "X.XX% \u00b1 Y.YY%" for
        stochastic ones -- readable directly, and the spread is never
        hidden or silently dropped for a stochastic method.
      raw_results: {method_name: [per-run metrics dict, ...]} -- the
        full, unformatted numeric data behind the summary, for anyone
        wanting to compute additional statistics or plot the
        distribution across seeds rather than just read mean/std.
    """
    constraints = constraints or PortfolioConstraints()
    raw_results = {}
    summary_rows = []

    metric_keys = ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                   "max_drawdown", "calmar_ratio", "win_rate"]
    pct_metrics = {"total_return", "cagr", "max_drawdown", "win_rate"}

    for spec in method_specs:
        n_runs = spec.n_runs if spec.stochastic else 1
        per_run_metrics = []

        for i in range(n_runs):
            seed = (base_seed + i) if spec.stochastic else None
            optimizer = spec.optimizer_factory(seed)
            history = run_backtest(returns_matrix, optimizer, constraints, rebalance_every, lookback)
            per_run_metrics.append(compute_all_metrics(history))

        raw_results[spec.name] = per_run_metrics

        row = {"Method": spec.name, "Runs": n_runs}
        for key in metric_keys:
            values = np.array([m[key] for m in per_run_metrics])
            scale = 100.0 if key in pct_metrics else 1.0
            suffix = "%" if key in pct_metrics else ""
            mean_val = values.mean() * scale

            if n_runs > 1:
                std_val = values.std(ddof=1) * scale
                row[key] = f"{mean_val:.2f}{suffix} \u00b1 {std_val:.2f}{suffix}"
            else:
                row[key] = f"{mean_val:.2f}{suffix}"

        summary_rows.append(row)

    return pd.DataFrame(summary_rows), raw_results


def print_comparison_table(summary_df: pd.DataFrame):
    table = Table(title="Portfolio Method Comparison", box=box.ROUNDED,
                  header_style="bold cyan", title_justify="center")
    table.add_column("Method", style="bold white")
    table.add_column("Runs", justify="right")
    for col in ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                "max_drawdown", "calmar_ratio", "win_rate"]:
        table.add_column(col.replace("_", " ").title(), justify="right")

    for _, row in summary_df.iterrows():
        table.add_row(row["Method"], str(row["Runs"]),
                      row["total_return"], row["cagr"], row["sharpe_ratio"],
                      row["sortino_ratio"], row["max_drawdown"], row["calmar_ratio"],
                      row["win_rate"])
    console.print(table)


def default_method_specs(n_stochastic_runs: int = 20) -> List[MethodSpec]:
    """Every method built this session, ready to run in one call."""
    from src.portfolio.methods.equal_weight import EqualWeightOptimizer
    from src.portfolio.methods.mean_variance import MeanVarianceOptimizer
    from src.portfolio.methods.risk_parity import RiskParityOptimizer
    from src.portfolio.methods.hierarchical_risk_parity import HierarchicalRiskParityOptimizer
    from src.portfolio.methods.cvar import CVaROptimizer
    from src.portfolio.methods.pso_optimizer import PSOPortfolioOptimizer, PSOOptimizerAdapter
    from src.portfolio.methods.robust_portfolio_optimisation import RobustPortfolioOptimizer
    from src.portfolio.methods.black_litterman import BlackLittermanOptimizerAdapter
    from src.portfolio.methods.momentum_views import MomentumViewGenerator

    return [
        MethodSpec("EqualWeight", lambda seed: EqualWeightOptimizer()),
        MethodSpec("MeanVariance", lambda seed: MeanVarianceOptimizer()),
        MethodSpec("RiskParity", lambda seed: RiskParityOptimizer()),
        MethodSpec("HRP", lambda seed: HierarchicalRiskParityOptimizer()),
        MethodSpec("CVaR", lambda seed: CVaROptimizer()),
        MethodSpec("PSO", lambda seed: PSOOptimizerAdapter(
            pso=PSOPortfolioOptimizer(n_particles=30, n_iterations=40, random_state=seed)),
            stochastic=True, n_runs=n_stochastic_runs),
        MethodSpec("RobustResampled", lambda seed: RobustPortfolioOptimizer(
            n_resamples=100, random_state=seed),
            stochastic=True, n_runs=n_stochastic_runs),
        MethodSpec("BlackLitterman", lambda seed: BlackLittermanOptimizerAdapter(view_generator=None)),
        MethodSpec("BlackLitterman+Momentum", lambda seed: BlackLittermanOptimizerAdapter(
            view_generator=MomentumViewGenerator())),
    ]


if __name__ == "__main__":
    import os
    data_path = os.path.join("data", "processed", "multi_asset_train_data.csv")
    if not os.path.exists(data_path):
        console.print(f"[bold red]{data_path} not found.[/bold red]")
        # NOTE: the current processor.py has no --multi-asset CLI flag
        # (main_multi_asset.py's own docstring still references one from
        # before the project reset -- stale, not reflecting current code).
        # process_multi_asset() must be called directly for now.
        console.print("[dim]Generate it first, e.g.:\n"
                      "  from src.pipeline.data_fetcher import DataFetcher\n"
                      "  from src.pipeline.processor import DataProcessor\n"
                      "  raw = DataFetcher().fetch_data(['SPY','QQQ','GLD'], config.START_DATE, config.END_DATE)\n"
                      "  train, val, test = DataProcessor().process_multi_asset(raw)\n"
                      "  train.to_csv('data/processed/multi_asset_train_data.csv', index=False)[/dim]")
    else:
        df = pd.read_csv(data_path)
        tickers = sorted(df['tic'].unique())
        raw_return_wide = df.pivot(index='date', columns='tic', values='log_return')[tickers]
        returns_matrix = raw_return_wide.values.astype(np.float64)

        summary_df, raw_results = compare_methods(returns_matrix, default_method_specs())
        print_comparison_table(summary_df)