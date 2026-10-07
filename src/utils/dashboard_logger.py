"""
Console logger for reinforcement learning runs.

Shows a live table of the most recent training episodes and prints a
one-off report for a finished evaluation. Performance figures come from
the same functions as the evaluation code (src/utils/metrics.py), so a
number means the same thing everywhere.

Each training episode is one random window of the training data, played
with exploration switched on. The figures in the live table are
therefore results of training episodes. They are not validation or test
results.
"""

import csv
import os
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.utils.metrics import max_drawdown, sharpe_ratio, total_return, win_rate

console = Console()

HISTORY_FIELDS = [
    "cycle",
    "portfolio_value",
    "pnl_pct",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "policy_loss",
    "value_loss",
    "policy_entropy",
]


class QuantRLLogger:
    """Live training table, per-episode history and evaluation report."""

    def __init__(self, agent_name, asset_symbol, total_episodes=0, log_path=None):
        """Create a logger.

        Args:
            agent_name: Name shown in the banner, for example "MuZero".
            asset_symbol: Asset or universe shown in the banner.
            total_episodes: Planned number of episodes (0 if unknown).
            log_path: Optional CSV file. If given, the full episode
                history is saved there when the dashboard stops.
        """
        self.agent_name = agent_name.upper()
        self.asset_symbol = asset_symbol.upper()
        self.total_episodes = total_episodes
        self.log_path = log_path
        self.start_time = time.time()
        self.history = []
        self.live = None

    def print_banner(self):
        """Print the header with the agent, asset and planned episodes."""
        subtitle = f"Agent: {self.agent_name}  |  Asset: {self.asset_symbol}"
        if self.total_episodes > 0:
            subtitle += f"  |  Planned Episodes: {self.total_episodes}"

        panel = Panel(
            f"[bold cyan]{subtitle}[/bold cyan]",
            title="[bold white]TRAINING DASHBOARD[/bold white]",
            border_style="cyan",
            expand=False,
        )
        console.print("\n")
        console.print(panel)
        console.print()

    def generate_table(self, last_n=10):
        """Build the table of the most recent training episodes.

        Args:
            last_n: How many recent episodes to show.

        Returns:
            A rich Table.
        """
        table = Table(
            title=f"Recent Training Episodes ({self.agent_name})",
            border_style="dim",
        )

        table.add_column("Episode", justify="right", style="cyan", no_wrap=True)
        table.add_column("End Value", justify="right", style="green")
        table.add_column("Episode PnL (%)", justify="right")
        table.add_column("Sharpe", justify="right", style="yellow")
        table.add_column("Max Drawdown (%)", justify="right", style="red")
        table.add_column("Win Rate (%)", justify="right", style="magenta")
        table.add_column("Policy Loss", justify="right", style="blue")
        table.add_column("Value Loss", justify="right", style="blue")
        table.add_column("Entropy", justify="right", style="blue")

        for entry in self.history[-last_n:]:
            pnl = entry["pnl_pct"]
            pnl_style = "bold green" if pnl >= 0 else "bold red"
            table.add_row(
                f"#{entry['cycle']}",
                f"{entry['portfolio_value']:,.2f}",
                f"[{pnl_style}]{pnl:+.2f}%[/{pnl_style}]",
                f"{entry['sharpe']:.2f}",
                f"{entry['max_drawdown']:.2f}%",
                f"{entry['win_rate']:.1f}%",
                f"{entry['policy_loss']:.4f}",
                f"{entry['value_loss']:.4f}",
                f"{entry['policy_entropy']:.4f}",
            )

        return table

    def start_dashboard(self):
        """Print the banner and start the live table."""
        self.print_banner()
        self.live = Live(self.generate_table(), console=console, refresh_per_second=4)
        self.live.start()

    def log_cycle(
        self,
        cycle,
        portfolio_values,
        daily_returns=None,
        policy_loss=0.0,
        value_loss=0.0,
        policy_entropy=0.0,
    ):
        """Record one finished training episode and refresh the table.

        Args:
            cycle: Episode number.
            portfolio_values: Portfolio value after every step, starting
                with the initial capital.
            daily_returns: Not used. Kept so existing callers still work.
                All figures are computed from portfolio_values.
            policy_loss: Mean policy loss of this episode's updates.
            value_loss: Mean value loss of this episode's updates.
            policy_entropy: Mean policy entropy of this episode's updates.
        """
        values = [float(v) for v in portfolio_values]

        if len(values) >= 2:
            pnl_pct = total_return(values) * 100.0
            sharpe = sharpe_ratio(values)
            drawdown_pct = max_drawdown(values) * 100.0
            win_pct = win_rate(values) * 100.0
        else:
            pnl_pct = sharpe = drawdown_pct = win_pct = 0.0

        self.history.append(
            {
                "cycle": cycle,
                "portfolio_value": values[-1] if values else 0.0,
                "pnl_pct": pnl_pct,
                "sharpe": sharpe,
                "max_drawdown": drawdown_pct,
                "win_rate": win_pct,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "policy_entropy": policy_entropy,
            }
        )

        if self.live:
            self.live.update(self.generate_table())

    def save_history(self, path):
        """Write every recorded episode to a CSV file.

        Args:
            path: Output file. Missing folders are created.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)

    def stop_dashboard(self):
        """Stop the live table, print a summary and save the history.

        The history is saved only if log_path was given.
        """
        if self.live:
            self.live.stop()
            self.live = None

        elapsed = time.time() - self.start_time
        count = max(len(self.history), 1)
        average = sum(e["pnl_pct"] for e in self.history) / count
        best = max((e["pnl_pct"] for e in self.history), default=0.0)

        summary = (
            f"[bold]Episodes:[/bold] {len(self.history):,}\n"
            f"[bold]Average episode return:[/bold] {average:+.2f}%\n"
            f"[bold]Best episode return:[/bold] {best:+.2f}%\n"
            f"[bold]Duration:[/bold] {elapsed:.1f} seconds"
        )
        console.print("\n")
        console.print(
            Panel(
                summary,
                title="[bold green]TRAINING FINISHED[/bold green]",
                border_style="green",
                expand=False,
            )
        )

        if self.log_path:
            self.save_history(self.log_path)
            console.print(f"Episode history saved to {self.log_path}")

    def print_evaluation_report(self, name, metrics, action_dist, final_value):
        """Print a one-off report for one finished evaluation run.

        Args:
            name: Name of the evaluated agent or strategy.
            metrics: Dictionary from compute_all_metrics().
            action_dist: Dictionary mapping an action name to the
                percentage of steps it was chosen.
            final_value: Portfolio value at the end of the run.
        """
        table = Table(title=f"Evaluation Report: {name}", border_style="dim")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Final Portfolio Value", f"{final_value:,.2f}")
        table.add_row("Total Return", f"{metrics['total_return']:+.2%}")
        table.add_row("CAGR", f"{metrics['cagr']:+.2%}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        table.add_row("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
        table.add_row("Win Rate", f"{metrics['win_rate']:.2%}")
        table.add_row(
            "Action Distribution",
            " | ".join(f"{label} {pct:.1f}%" for label, pct in action_dist.items()),
        )

        console.print(table)