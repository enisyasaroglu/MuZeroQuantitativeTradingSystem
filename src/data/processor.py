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
    Handles feature engineering and data normalization.
    """

    def __init__(self):
        self.tech_indicators = config.TECH_INDICATORS

    def clean_data(self, df):
        """
        Basic cleaning: drops duplicates and missing values.
        """
        print("Cleaning data...")
        df = df.copy()
        df = df.dropna()
        df = df.drop_duplicates(subset=['date', 'tic'], keep='last')
        # Sort by date is crucial for time-series
        df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        return df

    def add_technical_indicators(self, df):
        """
        Calculates indicators like MACD, RSI using stockstats.
        """
        print(f"Adding technical indicators: {self.tech_indicators}...")
        df = df.copy()
        
        # stockstats requires the dataframe to be wrapped
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()[0]

        for indicator in self.tech_indicators:
            # stockstats computes the column and adds it to the internal df
            # We then extract it back to our main df
            try:
                temp_indicator = stock[indicator]
                # If we have multiple tickers, this logic needs to handle grouping.
                # For single ticker (S&P 500), direct assignment works.
                df[indicator] = temp_indicator.values
            except KeyError as e:
                print(f"Error calculating {indicator}: {e}")
        
        # Fill NaNs created by rolling windows (e.g., first 30 days of RSI will be NaN)
        df = df.dropna()
        df = df.reset_index(drop=True)
        print(f"Data shape after feature engineering: {df.shape}")
        return df

    def add_log_returns(self, df):
        """
        Converts raw Close prices to Log Returns.
        Formula: ln(P_t / P_{t-1})
        """
        # We compute log returns on the 'close' price
        # FinRL data usually has 'close' in lowercase
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # The first row will be NaN because there is no t-1
        df = df.dropna()
        return df

    def add_turbulence(self, df):
        """
        (Optional) Adds a turbulence index to detect market crashes.
        For now, we skip this to keep the 'Crawl' phase simple.
        """
        return df

    def split_data(self, df):
        """
        Splits data into Training and Validation sets strictly by time.
        """
        # Find the index that corresponds to the split percentage
        split_idx = int(len(df) * config.TRAIN_SPLIT)
        
        # Pure time-based split
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        print(f"Data Split - Train: {len(train_df)} rows, Validation: {len(val_df)} rows")
        return train_df, val_df

    def normalize(self, train_df, val_df):
        """
        Z-Score Normalization (Standardization).
        CRITICAL: We fit the scaler ONLY on the Training data to avoid look-ahead bias.
        """
        print("Normalizing data...")
        
        # Columns to normalize: Indicators + Log Return + Volume
        # We do NOT normalize the Date or Ticker columns
        cols_to_norm = list(self.tech_indicators) + ['log_return', 'volume']
        
        # Calculate statistics on TRAINING set
        train_mean = train_df[cols_to_norm].mean()
        train_std = train_df[cols_to_norm].std()
        
        # Apply to Training set
        train_df[cols_to_norm] = (train_df[cols_to_norm] - train_mean) / (train_std + 1e-8)
        
        # Apply same stats to Validation set
        val_df[cols_to_norm] = (val_df[cols_to_norm] - train_mean) / (train_std + 1e-8)
        
        return train_df, val_df

if __name__ == "__main__":
    # Integration Test: Run Fetcher -> Processor
    from src.data.fetcher import DataFetcher
    
    # 1. Fetch
    fetcher = DataFetcher()
    raw_df = fetcher.fetch_data([config.TICKER], config.START_DATE, config.END_DATE)
    
    if raw_df is not None:
        # 2. Process
        processor = DataProcessor()
        
        # Step A: Clean
        df_clean = processor.clean_data(raw_df)
        
        # Step B: Add Features
        df_features = processor.add_technical_indicators(df_clean)
        df_features = processor.add_log_returns(df_features)
        
        # Step C: Split
        train, val = processor.split_data(df_features)
        
        # Step D: Normalize
        train_norm, val_norm = processor.normalize(train, val)
        
        # Save processed data for the Environment to load later
        save_path = os.path.join(os.path.dirname(__file__), '../../data/processed')
        os.makedirs(save_path, exist_ok=True)
        
        train_norm.to_csv(os.path.join(save_path, "train_data.csv"), index=False)
        val_norm.to_csv(os.path.join(save_path, "val_data.csv"), index=False)
        
        print("\nSuccessfully processed and saved train/val datasets.")
        print("Sample Normalized Data (Train):")
        print(train_norm[['date', 'log_return'] + list(config.TECH_INDICATORS)].head())