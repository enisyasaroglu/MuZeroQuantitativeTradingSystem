"""
Historical Value-at-Risk: the loss threshold not exceeded in
`confidence`% of historical per-step returns. Answers "how bad could a
normal bad day plausibly be", as a complement to RiskManager's
kill-switch (which answers "what do we do once already deep in a bad
stretch") -- a different question in the same risk-management family,
not a replacement for it.

HISTORICAL (empirical), not parametric: computed directly from the
observed return distribution via a percentile, not assumed Gaussian --
consistent with this project's existing preference for empirical/
non-parametric methods over distributional assumptions (e.g. CVaR
portfolio optimisation already uses the empirical Rockafellar-Uryasev
formulation, not a variance-covariance shortcut).
"""
import numpy as np


def historical_var(portfolio_values, confidence=0.95):
    """
    Returns VaR as a NEGATIVE number (a loss), e.g. -0.03 meaning
    "on the worst `1-confidence` fraction of days, losses exceeded 3%".
    Matches the sign convention already used by max_drawdown and
    worst_single_period_loss elsewhere in this project.
    """
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    returns = (values[1:] - values[:-1]) / values[:-1]
    return float(np.percentile(returns, (1 - confidence) * 100))