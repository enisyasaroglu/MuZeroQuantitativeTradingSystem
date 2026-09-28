"""
Drawdown series -- thin wrapper around src/utils/metrics.py's
rolling_drawdown_series(), for the same reason as volatility.py: one
implementation, imported from wherever it's needed, not duplicated.
"""
from src.utils.metrics import rolling_drawdown_series


def drawdown_series(portfolio_values):
    """Drawdown from the running peak at every step. See
    src.utils.metrics.rolling_drawdown_series for the formula."""
    return rolling_drawdown_series(portfolio_values)


def current_drawdown(portfolio_values):
    """The single most recent drawdown value -- how far below the
    running peak the portfolio sits RIGHT NOW, not the historical worst
    (that's max_drawdown, already in compute_all_metrics)."""
    series = drawdown_series(portfolio_values)
    return series[-1] if series else 0.0