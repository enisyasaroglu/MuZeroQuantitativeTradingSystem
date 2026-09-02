import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

console = Console()


class QuantRLLogger:
    """Console logger for quantitative trading reinforcement learning."""

    def __init__(self, agent_name: str, asset_symbol: str, total_episodes: int = 0):
        self.agent_name = agent_name.upper()
        self.asset_symbol = asset_symbol.upper()
        self.total_episodes = total_episodes
        self.start_time = time.time()
        self.history = []
        self.live = None

    def print_banner(self):
        """Displays summary header."""
        subtitle = f"Agent: {self.agent_name}  |  Asset: {self.asset_symbol}"
        if self.total_episodes > 0:
            subtitle += f"  |  Planned Cycles: {self.total_episodes}"

        panel = Panel(
            f"[bold cyan]{subtitle}[/bold cyan]",
            title="[bold white]QUANTITATIVE TRADING SYSTEM - TRAINING DASHBOARD[/bold white]",
            border_style="cyan",
            expand=False,
        )
        console.print("\n")
        console.print(panel)
        console.print()

    def generate_table(self, last_n: int = 10) -> Table:
        """Generates table object for live updates."""
        table = Table(title=f"Recent Performance ({self.agent_name})", border_style="dim")

        table.add_column("Cycle", justify="right", style="cyan", no_wrap=True)
        table.add_column("Strategy PnL (%)", justify="right")
        table.add_column("Sharpe Ratio", justify="right", style="yellow")
        table.add_column("Max Drawdown (%)", justify="right", style="red")
        table.add_column("Win Rate (%)", justify="right", style="magenta")
        table.add_column("Policy Loss", justify="right", style="blue")
        table.add_column("Value Loss", justify="right", style="blue")

        recent_entries = self.history[-last_n:]
        for entry in recent_entries:
            pnl = entry["pnl_pct"]
            pnl_style = "bold green" if pnl >= 0 else "bold red"
            
            table.add_row(
                f"#{entry['cycle']}",
                f"[{pnl_style}]{pnl:+.2f}%[/{pnl_style}]",
                f"{entry['sharpe']:.2f}",
                f"{entry['max_drawdown']:.2f}%",
                f"{entry['win_rate']:.1f}%",
                f"{entry['policy_loss']:.4f}",
                f"{entry['value_loss']:.4f}"
            )

        return table

    def start_dashboard(self):
        """Starts live in-place console updates."""
        self.print_banner()
        self.live = Live(self.generate_table(), console=console, refresh_per_second=4)
        self.live.start()

    def log_cycle(
        self,
        cycle: int,
        portfolio_values: list,
        daily_returns: list,
        policy_loss: float,
        value_loss: float
    ):
        """Calculates performance metrics dynamically per episode."""
        port_vals = np.array(portfolio_values)
        returns = np.array(daily_returns)

        # 1. PnL Percentage
        pnl_pct = ((port_vals[-1] - port_vals[0]) / port_vals[0]) * 100.0 if len(port_vals) > 1 else 0.0

        # 2. Win Rate Percentage
        pos_days = np.sum(returns > 0)
        win_rate = (pos_days / len(returns)) * 100.0 if len(returns) > 0 else 0.0

        # 3. Max Drawdown Percentage
        peaks = np.maximum.accumulate(port_vals)
        drawdowns = (peaks - port_vals) / peaks
        max_drawdown = np.max(drawdowns) * 100.0 if len(drawdowns) > 0 else 0.0

        # 4. Annualized Sharpe Ratio
        std_ret = np.std(returns)
        mean_ret = np.mean(returns)
        sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252) if std_ret > 0 else 0.0

        self.history.append({
            "cycle": cycle,
            "pnl_pct": pnl_pct,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "policy_loss": policy_loss,
            "value_loss": value_loss
        })

        if self.live:
            self.live.update(self.generate_table())

    def stop_dashboard(self):
        """Stops live screen updating and prints summary banner."""
        if self.live:
            self.live.stop()
            self.live = None

        elapsed = time.time() - self.start_time
        avg_pnl = sum(e["pnl_pct"] for e in self.history) / max(len(self.history), 1)
        best_pnl = max((e["pnl_pct"] for e in self.history), default=0.0)

        summary_text = (
            f"[bold]Total Training Cycles:[/bold] {len(self.history):,}\n"
            f"[bold]Average Strategy Return:[/bold] {avg_pnl:+.2f}%\n"
            f"[bold]Best Cycle Return:[/bold]       {best_pnl:+.2f}%\n"
            f"[bold]Execution Duration:[/bold]    {elapsed:.1f} seconds"
        )
        console.print("\n")
        console.print(Panel(summary_text, title="[bold green]TRAINING SESSION COMPLETED[/bold green]", border_style="green", expand=False))