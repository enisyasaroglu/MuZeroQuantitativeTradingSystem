"""
Trend following with a channel breakout (the classic Donchian rule):
buy when the price makes a new high over the entry window, and sell when
it makes a new low over the shorter exit window.

Unlike a moving-average rule, it stays in a trade until the price
actually breaks down, so it trades less often.
"""

import numpy as np
import pandas as pd

from .base_strategy import SignalStrategy


class TrendFollowingStrategy(SignalStrategy):
    """Long after a breakout to a new high, cash after a breakdown."""

    def __init__(self, entry_window=55, exit_window=20):
        """Create the strategy.

        Args:
            entry_window: Buy when the price exceeds its highest value
                over this many previous days.
            exit_window: Sell when the price falls below its lowest value
                over this many previous days.
        """
        if entry_window < 2 or exit_window < 2:
            raise ValueError("entry_window and exit_window must be at least 2.")
        self.entry_window = int(entry_window)
        self.exit_window = int(exit_window)
        self.name = f"Trend({self.entry_window},{self.exit_window})"

    def params(self):
        return {"entry_window": self.entry_window, "exit_window": self.exit_window}

    def target_positions(self, log_returns):
        returns = self.prepare(log_returns)
        price = returns.cumsum()
        upper = price.shift(1).rolling(self.entry_window).max().to_numpy()
        lower = price.shift(1).rolling(self.exit_window).min().to_numpy()
        level = price.to_numpy()

        position = np.zeros(len(level))
        state = 0
        for t in range(len(level)):
            if state == 0:
                if not np.isnan(upper[t]) and level[t] > upper[t]:
                    state = 1
            elif not np.isnan(lower[t]) and level[t] < lower[t]:
                state = 0
            position[t] = state

        return pd.Series(position, index=returns.index).astype(int)