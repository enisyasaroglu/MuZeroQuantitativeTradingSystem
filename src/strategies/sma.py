"""
Simple Moving Average (SMA) baseline strategy.

The strategy uses a simple moving average of the closing price:

    Close > SMA -> Long
    Close <= SMA -> Neutral

This is intended as a simple rule-based benchmark for evaluating
the performance of PPO and MuZero.

No short positions are taken.
"""

from __future__ import annotations

import pandas as pd


class SMAStrategy:
    """
    Simple Moving Average crossover strategy.

    Parameters
    ----------
    window : int
        Number of periods used to calculate the SMA.

    Attributes
    ----------
    window : int
        SMA lookback period.
    """

    def __init__(self, window: int = 50) -> None:
        if not isinstance(window, int):
            raise TypeError("window must be an integer")

        if window <= 0:
            raise ValueError("window must be greater than 0")

        self.window = window

    def calculate_sma(self, prices: pd.Series) -> pd.Series:
        """
        Calculate the simple moving average.

        Parameters
        ----------
        prices : pd.Series
            Closing prices.

        Returns
        -------
        pd.Series
            Rolling simple moving average.
        """
        if prices.empty:
            raise ValueError("prices cannot be empty")

        if not pd.api.types.is_numeric_dtype(prices):
            raise TypeError("prices must contain numeric values")

        return prices.rolling(
            window=self.window,
            min_periods=self.window,
        ).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SMA trading signals.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'close' column.

        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with:

            - sma: calculated simple moving average
            - signal: trading signal
                1 = Long
                0 = Neutral
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if "close" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'close' column to calculate SMA"
            )

        result = df.copy()

        result["sma"] = self.calculate_sma(result["close"])

        # Long when price is above the SMA.
        # Neutral otherwise.
        result["signal"] = (
            result["close"] > result["sma"]
        ).astype(int)

        # There is not enough historical data to calculate the SMA
        # during the initial lookback period. Stay neutral rather
        # than generating a signal from incomplete information.
        result.loc[result["sma"].isna(), "signal"] = 0

        return result

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate only the position series.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'close' column.

        Returns
        -------
        pd.Series
            Position for each timestep:

            1 = Long
            0 = Neutral
        """
        signals = self.generate_signals(df)
        return signals["signal"]

    def __repr__(self) -> str:
        return f"SMAStrategy(window={self.window})"