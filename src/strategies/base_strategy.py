"""
Base class for rule-based strategies that work from log returns.

A strategy decides the position it wants after seeing the data up to and
including day t. The environment pays the action for day t+1 with the
return of day t+1, so the position is applied one day later. This class
applies that one-day lag in one place, so a strategy cannot use the
return it is about to earn.

Positions are -1 (short), 0 (cash) and +1 (long). The environment's
actions are positions plus one: 0 short, 1 neutral, 2 long.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SignalStrategy(ABC):
    """A rule-based strategy driven by a series of log returns."""

    name = "strategy"

    @abstractmethod
    def target_positions(self, log_returns):
        """Return the position wanted after seeing each day's data.

        The value at row t may use log returns up to and including row t.

        Args:
            log_returns: A pandas Series of daily log returns.

        Returns:
            A pandas Series of -1, 0 or +1 with the same length and index.
            Days without enough history must be 0.
        """

    def params(self):
        """Return the strategy's parameters as a dictionary."""
        return {}

    @staticmethod
    def prepare(log_returns):
        """Check a log-return series and return a clean float copy.

        Args:
            log_returns: Anything convertible to a numeric pandas Series.

        Returns:
            A float Series with a fresh integer index.

        Raises:
            ValueError: If the series is empty or contains missing values.
        """
        series = pd.Series(pd.to_numeric(log_returns, errors="coerce"), dtype=float)
        series = series.reset_index(drop=True)
        if series.empty:
            raise ValueError("log_returns cannot be empty.")
        if series.isna().any():
            raise ValueError("log_returns contains missing or non-numeric values.")
        return series

    def actions(self, log_returns):
        """Convert target positions into environment actions, with the lag.

        The action for row t is built from the target position of row
        t-1. The first row is always neutral.

        Args:
            log_returns: A pandas Series of daily log returns.

        Returns:
            A NumPy integer array with one action (0, 1 or 2) per row.
        """
        target = self.target_positions(log_returns).to_numpy(dtype=float)
        lagged = np.concatenate([[0.0], target[:-1]])
        return (lagged + 1.0).astype(int)