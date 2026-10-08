"""
Time-series momentum: hold the asset if it has gone up over the past
lookback period, skipping the most recent days, otherwise stay in cash.

The most recent days are skipped because very short-term moves tend to
reverse, which would work against a medium-term trend signal.
"""

import pandas as pd

from .base_strategy import SignalStrategy


class MomentumStrategy(SignalStrategy):
    """Long when the past return (excluding the latest days) is positive."""

    def __init__(self, lookback=126, skip=5, allow_short=False):
        """Create the strategy.

        Args:
            lookback: Number of days from which the past return is measured.
            skip: Number of most recent days left out of that measurement.
            allow_short: If True, go short when the past return is negative.
                If False, stay in cash instead.
        """
        if lookback <= skip or skip < 0:
            raise ValueError("lookback must be greater than skip, and skip cannot be negative.")
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.allow_short = bool(allow_short)
        self.name = f"Momentum({self.lookback},{self.skip})"

    def params(self):
        return {"lookback": self.lookback, "skip": self.skip, "allow_short": self.allow_short}

    def target_positions(self, log_returns):
        returns = self.prepare(log_returns)
        past = returns.shift(self.skip).rolling(self.lookback - self.skip).sum()
        position = pd.Series(0, index=returns.index)
        position[past > 0] = 1
        if self.allow_short:
            position[past < 0] = -1
        return position