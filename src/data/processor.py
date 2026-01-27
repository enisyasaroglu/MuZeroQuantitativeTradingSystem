import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def clean_data(self, df):
        """
        Removes NaN and ensures the index is Datetime.
        """
        df = df.dropna()
        df = df.sort_index()
        return df

    def add_technical_indicators(self, df):
        """
        Adds RSI, MACD, etc. 
        IMPORTANT: This must use rolling windows to prevent look-ahead bias.
        """
        # Placeholder: You will use libraries like 'ta' or 'FinRL' here
        # Example: df['rsi'] = ...
        return df

    def split_data(self, df):
        """
        Strict time-series split. NO SHUFFLING.
        """
        split_idx = int(len(df) * self.config.TRAIN_SPLIT)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df

    def normalize(self, df):
        """
        Z-Score normalization.
        CRITICAL: Calculate Mean/Std on TRAIN set only, apply to TEST set.
        """
        # Logic to be implemented in Sprint 1
        pass