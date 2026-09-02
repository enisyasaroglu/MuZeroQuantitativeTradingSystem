"""
Financial performance metrics computed from a portfolio value series.

All functions take a 1-D array-like of portfolio values (NOT returns),
sampled once per environment step, starting at INITIAL_CAPITAL. This
matches what StockTradingEnv.portfolio_history accumulates.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def _returns_from_values(portfolio_values):
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return np.array([])
    return values[1:] / values[:-1] - 1.0


def total_return(portfolio_values):
    values = np.asarray(portfolio_values, dtype=np.float64)
    return (values[-1] / values[0]) - 1.0


def cagr(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Compound Annual Growth Rate."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    n_periods = len(values) - 1
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    ratio = values[-1] / values[0]
    if ratio <= 0:
        return -1.0
    return ratio ** (1.0 / years) - 1.0


def sharpe_ratio(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """Annualised Sharpe Ratio computed from per-step returns."""
    returns = _returns_from_values(portfolio_values)
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """Like Sharpe, but only penalises downside volatility."""
    returns = _returns_from_values(portfolio_values)
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else np.abs(downside[0])
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values):
    """
    Maximum peak-to-trough decline, as a negative fraction (e.g. -0.25 for
    a 25% drawdown).
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    return float(np.min(drawdowns))


def calmar_ratio(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR):
    """CAGR divided by the magnitude of max drawdown."""
    mdd = max_drawdown(portfolio_values)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(cagr(portfolio_values, periods_per_year) / abs(mdd))


def win_rate(portfolio_values):
    """Fraction of steps with a positive return."""
    returns = _returns_from_values(portfolio_values)
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def compute_all_metrics(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """Convenience wrapper returning the standard metric set used
    throughout the evaluation chapter."""
    return {
        "total_return": total_return(portfolio_values),
        "cagr": cagr(portfolio_values, periods_per_year),
        "sharpe_ratio": sharpe_ratio(portfolio_values, periods_per_year, risk_free_rate),
        "sortino_ratio": sortino_ratio(portfolio_values, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(portfolio_values),
        "calmar_ratio": calmar_ratio(portfolio_values, periods_per_year),
        "win_rate": win_rate(portfolio_values),
    }