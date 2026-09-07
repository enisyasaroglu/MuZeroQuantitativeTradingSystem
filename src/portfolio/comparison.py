"""
Fair-comparison harness for portfolio optimisation methods.

Answers the question this project's whole portfolio-methods sequence has
been building toward: given the SAME data, SAME constraints, SAME
transaction-cost model, and SAME evaluation metrics, how do Equal
Weight, Mean-Variance, Risk Parity, HRP, CVaR, PSO, Robust (both
variants), and Black-Litterman (both variants) actually perform against
each other on this project's data -- not "PSO beat ABC/ACO on its own
terms" (the un-cross-checked comparison flagged at the very start of
this review, in the companion Swarm Intelligence project), but a
genuinely controlled comparison across every method built here.

DELIBERATELY NOT built into src/backtesting/ (costs.py/engine.py/
metrics.py/report.py, confirmed empty): this is scoped SPECIFICALLY to
comparing PortfolioOptimizerProtocol methods walking forward over a
returns panel. A general backtesting engine also needs to serve RL-agent
evaluation (select_action() loops, checkpoint loading -- already
implemented separately, in evaluate.py), which this harness makes no
attempt to unify with speculatively. If src/backtesting/ is filled in
later with a real, specified need, this walk-forward loop is a natural
piece to fold into it then -- not guessed at now.

MECHANICS -- reuses, does not reimplement:
  - Portfolio (portfolio.py) for rebalance cadence/caching, exactly as
    MultiAssetTradingEnv uses it -- a method configured with
    rebalance_every=20 is only actually re-solved every 20 steps here
    too, not every single step.
  - exp(net_return) log-compounding, matching trading_env.py/
    multi_asset_env.py's convention exactly -- not a third compounding
    formula.
  - TRANSACTION_FEE * turnover cost, matching multi_asset_env.py's
    model exactly.
  - src/utils/metrics.py's compute_all_metrics() for every reported
    metric -- not a fourth reimplementation of Sharpe/drawdown, the
    specific failure mode already found and fixed once between
    evaluate.py and utils/metrics.py's silently-diverging ddof/sign
    conventions.

STOCHASTIC VS DETERMINISTIC METHODS -- handled explicitly, not
uniformly averaged: PSO and the resampled Robust optimizer both accept
a random_state and are run over N independent seeds, reporting mean +/-
std. Every other method here (Equal Weight, Mean-Variance, Risk Parity,
HRP, CVaR, both Black-Litterman variants, the box-uncertainty variant)
is exactly deterministic given the same inputs -- run ONCE. Running a
deterministic method N times and reporting a std of 0.00 would be
theatre, not rigor; the output table labels which rows are which rather
than presenting both the same way.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable
from rich.console import Console
from rich.table import Table
from rich import box

from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import Portfolio
from src.utils.metrics import compute_all_metrics

console = Console()


@dataclass
class MethodSpec:
    name: str
    optimizer_factory: Callable[[int], object]  # seed -> PortfolioOptimizerProtocol instance
    n_seeds: int = 1  # 1 => deterministic, run once. >1 => stochastic, mean +/- std over n_seeds.


def _run_single_backtest(returns_matrix: np.ndarray, optimizer, constraints: PortfolioConstraints,
                          rebalance_every: int, lookback: int, initial_capital: float,
                          transaction_fee: float) -> dict:
    """
    One full walk-forward pass over returns_matrix for one optimizer
    instance. Mirrors MultiAssetTradingEnv.step()'s compounding/cost
    mechanics exactly, but without that environment's action-menu/RL
    machinery -- the optimizer's proposed weights are followed directly,
    every step, with no discrete action space in between.
    """
    n_dates, n_assets = returns_matrix.shape
    portfolio = Portfolio(n_assets, optimizer, constraints, rebalance_every, lookback)
    portfolio.reset()

    portfolio_value = initial_capital
    portfolio_history = [initial_capital]
    turnovers = []

    for t in range(lookback, n_dates - 1):
        target_weights = portfolio.propose_weights(t, returns_matrix)
        turnover = portfolio.turnover(target_weights)
        turnovers.append(turnover)

        asset_returns = returns_matrix[t]
        gross_return = float(target_weights @ asset_returns)
        cost = transaction_fee * turnover
        net_return = gross_return - cost

        portfolio_value *= np.exp(net_return)
        portfolio_history.append(portfolio_value)
        portfolio.commit(target_weights)

    metrics = compute_all_metrics(portfolio_history)
    metrics["mean_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    return metrics


def compare_methods(returns_matrix: np.ndarray, method_specs: list,
                     constraints: PortfolioConstraints = None, rebalance_every: int = 20,
                     lookback: int = 60, initial_capital: float = 100_000.0,
                     transaction_fee: float = 0.001) -> pd.DataFrame:
    """
    Runs every MethodSpec in method_specs over the SAME returns_matrix,
    SAME constraints, SAME rebalance/lookback/cost settings. Returns a
    DataFrame, one row per method: "{metric}_mean" always populated,
    "{metric}_std" populated only when n_seeds > 1 (None otherwise) --
    the caller can tell a single deterministic value from an aggregated
    stochastic one directly from the DataFrame, not just the printed table.
    """
    constraints = constraints or PortfolioConstraints()
    rows = []

    for spec in method_specs:
        run_results = [
            _run_single_backtest(
                returns_matrix, spec.optimizer_factory(seed), constraints,
                rebalance_every, lookback, initial_capital, transaction_fee,
            )
            for seed in range(spec.n_seeds)
        ]

        row = {"method": spec.name, "n_seeds": spec.n_seeds, "stochastic": spec.n_seeds > 1}
        for key in run_results[0].keys():
            values = np.array([r[key] for r in run_results])
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if spec.n_seeds > 1 else None
        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(results_df: pd.DataFrame,
                            metric_order=("total_return", "sharpe_ratio", "max_drawdown",
                                          "calmar_ratio", "mean_turnover")):
    """Renders results_df as a Rich table. Stochastic methods print as
    'mean +/- std'; deterministic methods print a plain value -- visually
    distinct so a single lucky seed is never mistaken for a method's
    expected performance."""
    table = Table(title="Portfolio Method Comparison", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Method", style="bold white")
    table.add_column("Seeds", justify="right")
    for m in metric_order:
        table.add_column(m, justify="right")

    for _, row in results_df.iterrows():
        cells = [row["method"], str(row["n_seeds"])]
        for m in metric_order:
            mean_val = row.get(f"{m}_mean")
            std_val = row.get(f"{m}_std")
            if mean_val is None:
                cells.append("—")
            elif std_val is not None:
                cells.append(f"{mean_val:.4f} ± {std_val:.4f}")
            else:
                cells.append(f"{mean_val:.4f}")
        table.add_row(*cells)

    console.print(table)


def build_default_method_specs(n_seeds_stochastic: int = 20) -> list:
    """
    One MethodSpec per optimizer built so far. Imports done lazily inside
    this function (not at module top) so importing comparison.py's core
    (MethodSpec/compare_methods/print_comparison_table) doesn't pull in
    every optimizer module unless this convenience function is actually
    called.
    """
    from src.portfolio.methods.equal_weight import EqualWeightOptimizer
    from src.portfolio.methods.mean_variance import MeanVarianceOptimizer
    from src.portfolio.methods.risk_parity import RiskParityOptimizer
    from src.portfolio.methods.hierarchical_risk_parity import HierarchicalRiskParityOptimizer
    from src.portfolio.methods.cvar import CVaROptimizer
    from src.portfolio.methods.pso_optimizer import PSOOptimizerAdapter
    from src.portfolio.methods.robust_portfolio_optimisation import (
        RobustPortfolioOptimizer, BoxUncertaintyOptimizerAdapter,
    )
    from src.portfolio.methods.black_litterman import BlackLittermanOptimizerAdapter
    from src.portfolio.methods.momentum_views import MomentumViewGenerator

    return [
        MethodSpec("EqualWeight", lambda seed: EqualWeightOptimizer(), n_seeds=1),
        MethodSpec("MeanVariance", lambda seed: MeanVarianceOptimizer(), n_seeds=1),
        MethodSpec("RiskParity", lambda seed: RiskParityOptimizer(), n_seeds=1),
        MethodSpec("HRP", lambda seed: HierarchicalRiskParityOptimizer(), n_seeds=1),
        MethodSpec("CVaR", lambda seed: CVaROptimizer(), n_seeds=1),
        MethodSpec("PSO", lambda seed: PSOOptimizerAdapter(random_state=seed), n_seeds=n_seeds_stochastic),
        MethodSpec("RobustResampled", lambda seed: RobustPortfolioOptimizer(random_state=seed), n_seeds=n_seeds_stochastic),
        MethodSpec("RobustBoxUncertainty", lambda seed: BoxUncertaintyOptimizerAdapter(), n_seeds=1),
        MethodSpec("BlackLitterman", lambda seed: BlackLittermanOptimizerAdapter(view_generator=None), n_seeds=1),
        MethodSpec("BlackLitterman+Momentum",
                   lambda seed: BlackLittermanOptimizerAdapter(view_generator=MomentumViewGenerator()),
                   n_seeds=1),
    ]


if __name__ == "__main__":
    import argparse
    import os
    from configs.base_config import config

    parser = argparse.ArgumentParser(description="Compare portfolio optimisation methods fairly.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=60)
    args = parser.parse_args()

    data_path = os.path.join("data", "processed", f"multi_asset_{args.split}_data.csv")
    if not os.path.exists(data_path):
        console.print(f"[bold red]{data_path} not found.[/bold red]")
        console.print(
            "[dim]Generate it with DataProcessor.process_multi_asset() on a multi-ticker "
            "DataFrame from DataFetcher.fetch_data(['SPY','QQQ','GLD'], ...).[/dim]"
        )
        raise SystemExit(1)

    df = pd.read_csv(data_path)
    tickers = sorted(df['tic'].unique())
    # Pivot to a (T, n_assets) RAW log-return panel, mirroring
    # MultiAssetTradingEnv.__init__'s own construction of self.raw_returns
    # exactly -- not a second, independently-written pivot.
    returns_matrix = df.pivot(index='date', columns='tic', values='log_return')[tickers].values

    console.print(f"[dim]Comparing methods on '{args.split}' split ({returns_matrix.shape[0]} rows, "
                  f"{len(tickers)} assets: {tickers})...[/dim]\n")

    method_specs = build_default_method_specs(n_seeds_stochastic=args.seeds)
    results_df = compare_methods(
        returns_matrix, method_specs,
        constraints=PortfolioConstraints(),
        rebalance_every=args.rebalance_every,
        lookback=args.lookback,
        initial_capital=config.INITIAL_CAPITAL,
        transaction_fee=config.TRANSACTION_FEE,
    )
    print_comparison_table(results_df)
    