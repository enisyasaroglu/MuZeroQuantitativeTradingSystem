"""
Performance metrics for a series of portfolio values.

Every function takes a one-dimensional sequence of portfolio values (not
returns), recorded once per step and starting with the initial capital.
This is the format of StockTradingEnv.portfolio_history.

Returns are simple step-to-step returns: value[t] / value[t-1] - 1.
Annualised figures assume 252 trading days per year and a risk-free rate
of zero unless a rate is given. Annualised figures from short windows
(a few months) are noisy, so read them with caution.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def _returns_from_values(portfolio_values):
    """Convert portfolio values to simple step-to-step returns.

    Args:
        portfolio_values: Sequence of portfolio values.

    Returns:
        A NumPy array with one return per step, or an empty array if
        there are fewer than two values.
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return np.array([])
    return values[1:] / values[:-1] - 1.0


def total_return(portfolio_values):
    """Total return over the whole series, as a fraction.

    A result of 0.10 means a gain of 10%. Needs at least one value.

    Args:
        portfolio_values: Sequence of portfolio values.

    Returns:
        Final value divided by first value, minus one.
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    return (values[-1] / values[0]) - 1.0


def cagr(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Compound annual growth rate.

    Args:
        portfolio_values: Sequence of portfolio values.
        periods_per_year: Number of steps in one year.

    Returns:
        The yearly growth rate as a fraction. Returns 0.0 if there are
        fewer than two values, and -1.0 if the final value is zero or
        negative.
    """
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
    """Annualised Sharpe ratio: average excess return per unit of volatility.

    Args:
        portfolio_values: Sequence of portfolio values.
        periods_per_year: Number of steps in one year.
        risk_free_rate: Annual risk-free rate as a fraction.

    Returns:
        The annualised Sharpe ratio. Returns 0.0 if there are fewer than
        two returns, or if the returns do not vary at all (for example
        a portfolio held in cash).
    """
    returns = _returns_from_values(portfolio_values)
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """Annualised Sortino ratio: like Sharpe, but only losses count as risk.

    Args:
        portfolio_values: Sequence of portfolio values.
        periods_per_year: Number of steps in one year.
        risk_free_rate: Annual risk-free rate as a fraction.

    Returns:
        The annualised Sortino ratio. Returns 0.0 if there are fewer than
        two returns or the series is flat. Returns NaN if the portfolio
        made a profit without a single losing step, because the ratio is
        undefined in that case.
    """
    returns = _returns_from_values(portfolio_values)
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    downside_dev = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    if downside_dev < 1e-12:
        if abs(np.mean(excess)) < 1e-12:
            return 0.0
        return float("nan")
    return float(np.mean(excess) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values):
    """Largest fall from a previous peak to a later low.

    Args:
        portfolio_values: Sequence of portfolio values.

    Returns:
        A negative fraction, for example -0.25 for a 25% fall. Returns
        0.0 if the portfolio never fell below an earlier value.
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    return float(np.min(drawdowns))


def calmar_ratio(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Yearly growth rate divided by the size of the maximum drawdown.

    Args:
        portfolio_values: Sequence of portfolio values.
        periods_per_year: Number of steps in one year.

    Returns:
        The Calmar ratio. Returns 0.0 if there was no drawdown at all.
    """
    mdd = max_drawdown(portfolio_values)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(cagr(portfolio_values, periods_per_year) / abs(mdd))


def win_rate(portfolio_values):
    """Share of active steps that made money.

    A step is active if the portfolio value changed. Steps spent entirely
    in cash are left out, otherwise an agent that mostly holds cash would
    look as if it never wins.

    Args:
        portfolio_values: Sequence of portfolio values.

    Returns:
        A fraction between 0 and 1. Returns 0.0 if there were no active
        steps.
    """
    returns = _returns_from_values(portfolio_values)
    active = returns[np.abs(returns) > 1e-12]
    if len(active) == 0:
        return 0.0
    return float(np.mean(active > 0))


def annualized_volatility(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Annualised standard deviation of step returns.

    This is the plain measure of how much the portfolio swings. It is
    reported next to Sharpe and Sortino because two strategies can share
    a Sharpe ratio while swinging by very different amounts.

    Args:
        portfolio_values: Sequence of portfolio values.
        periods_per_year: Number of steps in one year.

    Returns:
        The annualised volatility as a fraction. Returns 0.0 if there are
        fewer than two returns.
    """
    returns = _returns_from_values(portfolio_values)
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def worst_single_period_loss(portfolio_values):
    """Worst return of any single step.

    This differs from the maximum drawdown, which measures the fall from
    a peak and can span many steps.

    Args:
        portfolio_values: Sequence of portfolio values.

    Returns:
        The lowest step return as a fraction (negative for a loss).
        Returns 0.0 if there are fewer than two values.
    """
    returns = _returns_from_values(portfolio_values)
    if len(returns) == 0:
        return 0.0
    return float(np.min(returns))


def rolling_drawdown_series(portfolio_values):
    """Drawdown from the running peak at every step.

    This is the series an underwater chart would plot. Its lowest value
    equals max_drawdown().

    Args:
        portfolio_values: Sequence of portfolio values.

    Returns:
        A list with one value per input value, each zero or negative.
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    return ((values - running_max) / running_max).tolist()


def compute_all_metrics(portfolio_values, periods_per_year=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """Compute the standard set of metrics used in every evaluation report.

    Args:
        portfolio_values: Sequence of portfolio values.
        periods_per_year: Number of steps in one year.
        risk_free_rate: Annual risk-free rate as a fraction.

    Returns:
        A dictionary with total_return, cagr, sharpe_ratio, sortino_ratio,
        max_drawdown, calmar_ratio, win_rate, annualized_volatility,
        worst_single_period_loss and rolling_drawdown_series.
    """
    return {
        "total_return": total_return(portfolio_values),
        "cagr": cagr(portfolio_values, periods_per_year),
        "sharpe_ratio": sharpe_ratio(portfolio_values, periods_per_year, risk_free_rate),
        "sortino_ratio": sortino_ratio(portfolio_values, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(portfolio_values),
        "calmar_ratio": calmar_ratio(portfolio_values, periods_per_year),
        "win_rate": win_rate(portfolio_values),
        "annualized_volatility": annualized_volatility(portfolio_values, periods_per_year),
        "worst_single_period_loss": worst_single_period_loss(portfolio_values),
        "rolling_drawdown_series": rolling_drawdown_series(portfolio_values),
    }