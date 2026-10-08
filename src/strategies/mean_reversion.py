"""
Mean reversion: buy when the price is unusually low compared with its
recent average, and sell once it has come back to that average.

The z-score measures how many standard deviations the price is from its
rolling average. A trade is entered below -entry_z and left when the
z-score returns to zero. This tends to lose money in a steady trend,
which is useful to know when it is compared with a trend-following rule.
"""

import numpy as np
import pandas as pd

from .base_strategy import SignalStrategy


class MeanReversionStrategy(SignalStrategy):
    """Long after an unusually low price, flat after it recovers."""

    def __init__(self, window=20, entry_z=2.0, allow_short=False):
        """Create the strategy.

        Args:
            window: Number of days used for the rolling average and spread.
            entry_z: How far below the average (in standard deviations)
                the price must be before buying.
            allow_short: If True, also go short when the price is
                unusually high.
        """
        if window < 2 or entry_z <= 0:
            raise ValueError("window must be at least 2 and entry_z must be positive.")
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.allow_short = bool(allow_short)
        self.name = f"MeanReversion({self.window},{self.entry_z:g})"

    def params(self):
        return {"window": self.window, "entry_z": self.entry_z, "allow_short": self.allow_short}

    def target_positions(self, log_returns):
        returns = self.prepare(log_returns)
        price = returns.cumsum()
        mean = price.rolling(self.window).mean()
        spread = price.rolling(self.window).std(ddof=1).replace(0.0, np.nan)
        z_scores = ((price - mean) / spread).to_numpy()

        position = np.zeros(len(z_scores))
        state = 0
        for t, z in enumerate(z_scores):
            if np.isnan(z):
                state = 0
            elif state == 0:
                if z < -self.entry_z:
                    state = 1
                elif self.allow_short and z > self.entry_z:
                    state = -1
            elif state == 1 and z >= 0.0:
                state = 0
            elif state == -1 and z <= 0.0:
                state = 0
            position[t] = state

        return pd.Series(position, index=returns.index).astype(int)