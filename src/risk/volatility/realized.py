"""
Realized volatility of a portfolio value series -- thin wrapper around
src/utils/metrics.py's annualized_volatility(), so the risk/ package has
its own clean import surface without a second, competing
implementation. Reuse, not duplication -- annualized_volatility() is
already correct and tested.
"""
from src.utils.metrics import annualized_volatility


def realized_volatility(portfolio_values, periods_per_year=252):
    """Annualised standard deviation of per-step returns. See
    src.utils.metrics.annualized_volatility for the underlying formula
    and its own docstring."""
    return annualized_volatility(portfolio_values, periods_per_year=periods_per_year)