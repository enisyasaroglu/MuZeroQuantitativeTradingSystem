import os
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import sys

# Add the project root to the path so we can import from configs
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from configs.base_config import config

class DataFetcher:
    """
    Handles the downloading of raw market data using FinRL-Meta.
    """

    def __init__(self):
        self.save_path = os.path.join(os.path.dirname(__file__), '../../data/raw')
        # Ensure the directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def fetch_data(self, ticker_list, start_date, end_date):
        """
        Downloads OHLCV data from Yahoo Finance.

        Args:
            ticker_list (list): List of stock tickers (e.g., ['AAPL', 'MSFT']) or single index ['^GSPC']
            start_date (str): Start date in 'YYYY-MM-DD'
            end_date (str): End date in 'YYYY-MM-DD'

        Returns:
            pd.DataFrame: The downloaded data with standard FinRL columns.
        """
        print(f"Starting download for {ticker_list} from {start_date} to {end_date}...")

        try:
            # FinRL's YahooDownloader handles the API connection
            df = YahooDownloader(
                start_date=start_date,
                end_date=end_date,
                ticker_list=ticker_list
            ).fetch_data()

            if df is None or df.empty:
                print("Error: No data fetched. Check your internet connection or ticker symbols.")
                return None

            print(f"Successfully fetched {len(df)} rows.")
            
            # Save to CSV for reproducibility (Crucial for your research methodology)
            file_name = f"raw_data_{ticker_list[0]}_{start_date}_{end_date}.csv"
            full_path = os.path.join(self.save_path, file_name)
            df.to_csv(full_path, index=False)
            print(f"Data saved to: {full_path}")
            
            return df

        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            return None

if __name__ == "__main__":
    # Quick test to verify the script works independently
    print("Testing DataFetcher...")
    
    # Use the settings from your config file
    fetcher = DataFetcher()
    
    # We pass the single ticker as a list because FinRL expects a list
    data = fetcher.fetch_data(
        ticker_list=[config.TICKER], 
        start_date=config.START_DATE, 
        end_date=config.END_DATE
    )

    if data is not None:
        print("\nHead of data:")
        print(data.head())
        print("\nData Columns:", data.columns)