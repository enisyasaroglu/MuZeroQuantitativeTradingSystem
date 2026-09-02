import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

warnings.filterwarnings("ignore")

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from configs.base_config import config


class DataProcessor:
    """
    Handles feature engineering, chronological splitting, and normalization.
    """

    def __init__(self):
        self.tech_indicators = config.TECH_INDICATORS

    def clean_data(self, df):
        df = df.copy()
        df = df.dropna()
        if 'tic' in df.columns:
            df = df.drop_duplicates(subset=['date', 'tic'], keep='last')
        else:
            df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values(by='date').reset_index(drop=True)
        return df

    def add_log_returns(self, df):
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna(subset=['log_return'])
        return df.reset_index(drop=True)

    def split_data(self, df):
        n = len(df)
        embargo = config.EMBARGO_DAYS

        train_end = int(n * config.TRAIN_SPLIT)
        val_end = train_end + int(n * config.VAL_SPLIT)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end + embargo: val_end].copy()
        test_df = df.iloc[val_end + embargo:].copy()

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def add_technical_indicators(self, df):
        if df.empty:
            return df
        df = df.copy()
        stock = Sdf.retype(df.copy())

        for indicator in self.tech_indicators:
            try:
                df[indicator] = stock[indicator].values
            except Exception as e:
                print(f"Warning: Could not compute {indicator}: {e}")

        df = df.dropna().reset_index(drop=True)
        return df

    def normalize(self, train_df, val_df, test_df):
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

    start_time = time.time()
    
    print("\n" + "=" * 80)
    print(" DATA PIPELINE: PROCESSING & FEATURE ENGINEERING")
    print("=" * 80)

    fetcher = DataFetcher()
    raw_df = fetcher.fetch_data([config.TICKER], config.START_DATE, config.END_DATE)

    if raw_df is not None:
        processor = DataProcessor()
        train_df, val_df, test_df = processor.process(raw_df)

        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train_data.csv")
        val_path = os.path.join(save_path, "val_data.csv")
        test_path = os.path.join(save_path, "test_data.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        elapsed_time = time.time() - start_time

        # Print structured metrics report
        print("\nDATASET SPLIT & PROCESSING METRICS")
        print("-" * 80)
        print(f"{'Split':<12} | {'Rows':<8} | {'Start Date':<12} | {'End Date':<12} | {'Nulls':<6}")
        print("-" * 80)

        for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            start_d = pd.to_datetime(split_df['date'].min()).strftime('%Y-%m-%d')
            end_d = pd.to_datetime(split_df['date'].max()).strftime('%Y-%m-%d')
            null_count = split_df.isnull().sum().sum()
            print(f"{name:<12} | {len(split_df):<8} | {start_d:<12} | {end_d:<12} | {null_count:<6}")

        print("-" * 80)
        print(f"Embargo Applied    : {config.EMBARGO_DAYS} trading days between splits")
        print(f"Technical Indicators: {len(config.TECH_INDICATORS)} features generated")
        print(f"Normalization      : Z-Score (fitted on Training split only)")
        print(f"Saved Directory    : {save_path}")
        print(f"Total Process Time : {elapsed_time:.2f} seconds")
        print("-" * 80)

        print("\nPROCESSED FEATURE PREVIEW (TRAINING SET - FIRST 5 ROWS)")
        print("-" * 80)
        preview_cols = ['date', 'log_return'] + list(config.TECH_INDICATORS[:4])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(train_df[preview_cols].head().to_string(index=False))
        print("=" * 80 + "\n")