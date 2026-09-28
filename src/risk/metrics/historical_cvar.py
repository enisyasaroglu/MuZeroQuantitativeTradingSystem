"""
Historical Conditional VaR (Expected Shortfall): the AVERAGE loss in the
tail beyond the VaR threshold, not just the threshold itself -- VaR
tells you the cutoff; CVaR tells you how bad it typically is once you're
past that cutoff. Same Rockafellar-Uryasev tail-averaging concept
already implemented for portfolio OPTIMISATION in
src/portfolio/methods/cvar.py -- applied here to a realized return
series for MEASUREMENT, not fed into an LP solver, but using the same
underlying definition so the two don't quietly diverge.
"""
import numpy as np
from src.risk.metrics.historical_var import historical_var


def historical_cvar(portfolio_values, confidence=0.95):
    """
    Returns CVaR as a NEGATIVE number, same sign convention as
    historical_var. By definition, CVaR is at least as negative as VaR
    (the tail average is at least as bad as the tail's boundary).
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    returns = (values[1:] - values[:-1]) / values[:-1]
    var_threshold = historical_var(portfolio_values, confidence=confidence)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return var_threshold
    return float(np.mean(tail_losses))