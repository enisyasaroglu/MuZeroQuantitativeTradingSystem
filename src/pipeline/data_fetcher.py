import os
import sys
import time
import warnings
import contextlib
import io
import pandas as pd
import numpy as np
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from configs.base_config import config
from src.pipeline.asset_registry import AssetClass, get_tickers
from src.pipeline.validation import validate_multi_asset_frame

console = Console()


class DataFetcher:
    """Handles downloading raw market data using FinRL-Meta, with local caching."""

    def __init__(self):
        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
        os.makedirs(self.save_path, exist_ok=True)

    def fetch_data(self, ticker_list, start_date, end_date, verbose=True):
        """Downloads OHLCV data for one or more tickers, reusing a cached
        file if one already exists for this exact ticker-set/date-range."""
        start_time = time.time()
        # Filename encodes ALL tickers, not just the first -- fetching
        # ["SPY","QQQ","GLD"] and later ["SPY","AAPL"] over the same date
        # range previously collided on the same cache file (both named
        # from ticker_list[0] alone), silently returning the wrong data
        # for whichever call ran second.
        file_name = f"raw_data_{'_'.join(ticker_list)}_{start_date}_{end_date}.csv"
        full_path = os.path.join(self.save_path, file_name)

        if verbose:
            console.print(Panel("[bold white]STAGE 1 · DATA ACQUISITION[/bold white]",
                                box=box.ROUNDED, expand=False, border_style="cyan"))
            header = Table.grid(padding=(0, 2))
            header.add_column(style="bold cyan")
            header.add_column(style="white")
            header.add_row("Target Tickers", str(ticker_list))
            header.add_row("Date Range", f"{start_date}  →  {end_date}")
            header.add_row("Destination", full_path)
            console.print(Panel(header, box=box.ROUNDED, expand=False, border_style="cyan"))
            
        if os.path.exists(full_path):
            df = pd.read_csv(full_path, parse_dates=["date"])
            if verbose:
                console.print(f"[dim]Cached file found — skipping download "
                               f"({len(df):,} rows, {full_path})[/dim]\n")
            return df

        try:
            with console.status("[bold green]Connecting to Yahoo Finance...[/bold green]", spinner="dots"):
                raw_data = yf.download(
                    tickers=ticker_list,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                df_list = []
                if len(ticker_list) == 1:
                    tic = ticker_list[0]
                    temp = raw_data.reset_index()
                    temp.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in temp.columns]
                    temp['tic'] = tic
                    df_list.append(temp)
                else:
                    for tic in ticker_list:
                        if tic in raw_data.columns.levels[1]:
                            temp = raw_data.xs(tic, axis=1, level=1).dropna(how='all').reset_index()
                            temp.columns = [c.lower() for c in temp.columns]
                            temp['tic'] = tic
                            df_list.append(temp)

                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    if 'adj close' in df.columns:
                        df['close'] = df['adj close']
                    df['day'] = pd.to_datetime(df['date']).dt.dayofweek
                    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']
                    df = df[[c for c in expected_cols if c in df.columns]]
                else:
                    df = pd.DataFrame()

            if df is None or df.empty:
                if verbose:
                    console.print("[bold red]FAILED — no data retrieved.[/bold red]")
                return None

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df.to_csv(full_path, index=False)

            if verbose:
                elapsed = time.time() - start_time
                summary = Table.grid(padding=(0, 2))
                summary.add_column(style="bold cyan")
                summary.add_column(style="white")
                summary.add_row("Status", "[bold green]SUCCESS[/bold green]")
                summary.add_row("Rows Downloaded", f"{len(df):,}")
                summary.add_row("Date Range (actual)",
                                 f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
                summary.add_row("File Size", f"{os.path.getsize(full_path) / 1024:.1f} KB")
                summary.add_row("Execution Time", f"{elapsed:.2f}s")
                console.print(Panel(summary, box=box.ROUNDED, border_style="green", expand=False))

            return df

        except Exception as e:
            if verbose:
                console.print(f"[bold red]ERROR — {e}[/bold red]")
            return None

    def fetch_universe(self, asset_class: AssetClass = None, start_date=None, end_date=None, verbose=True):
        """
        Convenience wrapper: resolves tickers from the asset registry
        (optionally filtered to one asset class), fetches them, then
        validates the result. Does not change fetch_data's own contract
        -- this only saves the caller from wiring the registry and
        validation together by hand each time.

        Returns (dataframe, validation_report). Raises ValueError if
        validation finds a problem serious enough that downstream code
        could not safely proceed (see validation.py).
        """
        tickers = get_tickers(asset_class)
        df = self.fetch_data(tickers, start_date or config.START_DATE, end_date or config.END_DATE, verbose=verbose)
        if df is None:
            raise RuntimeError(f"fetch_universe: fetch_data returned no data for {tickers}")

        report = validate_multi_asset_frame(df, expected_tickers=tickers)
        if verbose and report["warnings"]:
            console.print(Panel("\n".join(report["warnings"]),
                                 title="[bold yellow]VALIDATION WARNINGS[/bold yellow]",
                                 border_style="yellow", expand=False))
        return df, report


if __name__ == "__main__":
    import argparse
    from src.pipeline.asset_registry import AssetClass, get_tickers

    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-asset", action="store_true",
                         help="Fetch the full asset registry universe instead of the single config.TICKER.")
    parser.add_argument("--asset-class", type=str, default=None,
                         help="Optional filter, e.g. equity/etf/commodity/index. Only used with --multi-asset.")
    args = parser.parse_args()

    fetcher = DataFetcher()

    if args.multi_asset:
        asset_class = AssetClass(args.asset_class) if args.asset_class else None
        data, report = fetcher.fetch_universe(asset_class=asset_class, verbose=True)
    else:
        data = fetcher.fetch_data(
            ticker_list=[config.TICKER],
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            verbose=True
        )

    if data is not None:
        preview = Table(box=box.ROUNDED, header_style="bold blue", title="Preview")
        for col in data.columns:
            preview.add_column(col, justify="right" if col not in ("date", "tic") else "left")
        for _, row in data.head().iterrows():
            preview.add_row(*[
                pd.to_datetime(row[c]).strftime('%Y-%m-%d') if c == 'date'
                else f"{row[c]:,.2f}" if isinstance(row[c], (float, np.floating))
                else str(row[c])
                for c in data.columns
            ])
        console.print(preview)