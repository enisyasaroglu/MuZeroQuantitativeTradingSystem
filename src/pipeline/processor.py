import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from configs.base_config import config


class DataProcessor:
    """
    Handles feature engineering, chronological splitting, and normalisation.

    Pipeline order matters and is enforced by the method names below:
    clean -> add_log_returns -> split_data (with embargo) -> add_technical_indicators
    (per split) -> normalize (train stats only).

    Computing technical indicators BEFORE splitting is a common source of
    look-ahead bias in financial ML: indicators like RSI/MACD use rolling
    windows, so an indicator value computed near a split boundary on the
    full, unsplit dataframe is partly derived from data on the other side
    of that boundary. Computing indicators independently per split (after
    splitting) avoids this. stockstats computes indicators statelessly per
    call, so this reordering is safe.
    """

    def __init__(self):
        self.tech_indicators = config.TECH_INDICATORS

    def clean_data(self, df):
        """Basic cleaning: drops duplicates and missing values."""
        df = df.copy()
        df = df.dropna()
        if 'tic' in df.columns:
            df = df.drop_duplicates(subset=['date', 'tic'], keep='last')
        else:
            df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values(by='date').reset_index(drop=True)
        return df

    def add_log_returns(self, df):
        """
        Converts raw Close prices to Log Returns: ln(P_t / P_{t-1}).
        Done before splitting because it only ever looks one row backward
        (not a multi-day rolling window), so it cannot leak across a split
        boundary the way a 14/30-day indicator window can. The first row
        becomes NaN (no t-1) and is dropped.
        """
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna(subset=['log_return'])
        return df.reset_index(drop=True)

    def split_data(self, df):
        """
        Chronological three-way split (train / validation / test) with an
        embargo gap dropped at each boundary. The embargo size matches
        LOOKBACK_WINDOW: this guarantees the first observation any agent
        sees in validation or test is built entirely from validation/test
        period data -- no feature in that first window was computed using
        any training-period (or validation-period, for test) price.
        """
        n = len(df)
        embargo = config.EMBARGO_DAYS

        train_end = int(n * config.TRAIN_SPLIT)
        val_end = train_end + int(n * config.VAL_SPLIT)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end + embargo: val_end].copy()
        test_df = df.iloc[val_end + embargo:].copy()

        print(f"Data Split - Train: {len(train_df)} rows, "
              f"Val: {len(val_df)} rows (after {embargo}-day embargo), "
              f"Test: {len(test_df)} rows (after {embargo}-day embargo)")

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def add_technical_indicators(self, df):
        """
        Calculates indicators (MACD, RSI, CCI, DX, ATR, Bollinger Bands)
        using stockstats. Must be called AFTER split_data() on each split
        independently -- see module docstring.
        """
        if df.empty:
            return df
        df = df.copy()
        stock = Sdf.retype(df.copy())

        for indicator in self.tech_indicators:
            try:
                df[indicator] = stock[indicator].values
            except (KeyError, Exception) as e:
                print(f"Error calculating {indicator}: {e}")

        # Rolling-window indicators produce NaNs for the first ~30 rows of
        # THIS split only (not a cross-split leak -- these rows are simply
        # dropped because there isn't enough history within the split to
        # compute them yet).
        df = df.dropna().reset_index(drop=True)
        return df

    def normalize(self, train_df, val_df, test_df):
        """
        Z-Score Normalisation (Standardisation).
        CRITICAL: statistics (mean/std) are computed on the TRAINING split
        only, then applied unchanged to validation and test. Fitting a
        scaler on val/test data (or on the full dataset before splitting)
        leaks distributional information about the future into the
        features the agent sees during training/evaluation.
        """
        cols_to_norm = [c for c in list(self.tech_indicators) + ['log_return', 'volume']
                         if c in train_df.columns]

        train_mean = train_df[cols_to_norm].mean()
        train_std = train_df[cols_to_norm].std()

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df[cols_to_norm] = (train_df[cols_to_norm] - train_mean) / (train_std + 1e-8)
        if not val_df.empty:
            val_df[cols_to_norm] = (val_df[cols_to_norm] - train_mean) / (train_std + 1e-8)
        if not test_df.empty:
            test_df[cols_to_norm] = (test_df[cols_to_norm] - train_mean) / (train_std + 1e-8)

        return train_df, val_df, test_df

    def process(self, raw_df):
        """
        Full pipeline, in the correct order:
        clean -> log returns -> split (+ embargo) -> indicators per split -> normalize.
        """
        df_clean = self.clean_data(raw_df)
        df_returns = self.add_log_returns(df_clean)

        train_raw, val_raw, test_raw = self.split_data(df_returns)

        train_feat = self.add_technical_indicators(train_raw)
        val_feat = self.add_technical_indicators(val_raw)
        test_feat = self.add_technical_indicators(test_raw)

        train_norm, val_norm, test_norm = self.normalize(train_feat, val_feat, test_feat)
        return train_norm, val_norm, test_norm


if __name__ == "__main__":
    from src.pipeline.fetcher import DataFetcher

    fetcher = DataFetcher()
    raw_df = fetcher.fetch_data([config.TICKER], config.START_DATE, config.END_DATE)

    if raw_df is not None:
        processor = DataProcessor()
        train_norm, val_norm, test_norm = processor.process(raw_df)

        save_path = os.path.join(os.path.dirname(__file__), '../../data/processed')
        os.makedirs(save_path, exist_ok=True)

        train_norm.to_csv(os.path.join(save_path, "train_data.csv"), index=False)
        val_norm.to_csv(os.path.join(save_path, "val_data.csv"), index=False)
        test_norm.to_csv(os.path.join(save_path, "test_data.csv"), index=False)

        print("\nSuccessfully processed and saved train/val/test datasets.")
        print(train_norm[['date', 'log_return'] + list(config.TECH_INDICATORS)].head())