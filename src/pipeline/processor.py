import os
import sys
import time
import warnings
import contextlib
import io
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from configs.base_config import config
from src.regime.regime_detector import RegimeDetector

console = Console()


class DataProcessor:
    """Handles feature engineering, chronological splitting, and normalization.

    Pipeline order: clean -> log returns -> split (with embargo) ->
    technical indicators (per split) -> normalize (train stats only) ->
    drop raw OHLC. See individual methods for why each step is ordered
    this way -- the ordering is what prevents look-ahead bias.
    """

    def __init__(self):
        self.tech_indicators = config.TECH_INDICATORS

    def clean_data(self, df):
        df = df.copy().dropna()
        subset_col = ['date', 'tic'] if 'tic' in df.columns else ['date']
        df = df.drop_duplicates(subset=subset_col, keep='last')
        return df.sort_values(by='date').reset_index(drop=True)

    def add_log_returns(self, df):
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        return df.dropna(subset=['log_return']).reset_index(drop=True)

    def split_data(self, df):
        """Chronological three-way split with an embargo GAP inserted
        between each pair of adjacent splits. embargo is added on top of
        each split's configured size, not subtracted from inside it --
        val_end previously omitted the embargo term here, which silently
        shrank the validation window by `embargo` rows below its
        configured 15% share."""
        n = len(df)
        embargo = config.EMBARGO_DAYS
        train_end = int(n * config.TRAIN_SPLIT)
        val_end = train_end + embargo + int(n * config.VAL_SPLIT)

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
            except Exception:
                pass

        return df.dropna().reset_index(drop=True)

    def normalize(self, train_df, val_df, test_df):
        """Z-Score normalisation using TRAINING SET statistics only.

        log_return is deliberately EXCLUDED from cols_to_norm and left
        raw -- trading_env.py's portfolio compounding
        (portfolio_value *= exp(net_return)) needs the true economic
        return, not a z-scored value; np.exp() of a z-score (typically
        in the range ±2 to ±4) rather than a real daily return
        (typically ±0.01-0.05) would produce a multi-x portfolio swing
        from a single step. log_return_norm is added as a SEPARATE
        column for the observation instead, mirroring the pattern
        already documented in trading_env.py's own docstring.
        """
        cols_to_norm = [c for c in list(self.tech_indicators) + ['volume'] if c in train_df.columns]

        train_mean = train_df[cols_to_norm].mean()
        train_std = train_df[cols_to_norm].std()

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        for df in (train_df, val_df, test_df):
            if not df.empty and 'log_return' in df.columns:
                df['log_return_norm'] = (
                    (df['log_return'] - train_df['log_return'].mean())
                    / (train_df['log_return'].std() + 1e-8)
                )

        train_df[cols_to_norm] = (train_df[cols_to_norm] - train_mean) / (train_std + 1e-8)
        if not val_df.empty:
            val_df[cols_to_norm] = (val_df[cols_to_norm] - train_mean) / (train_std + 1e-8)
        if not test_df.empty:
            test_df[cols_to_norm] = (test_df[cols_to_norm] - train_mean) / (train_std + 1e-8)

        return self._drop_raw_ohlc(train_df), self._drop_raw_ohlc(val_df), self._drop_raw_ohlc(test_df)


    def add_regime_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        detector = RegimeDetector(n_states=3, window=20)
        
        # Fit on train log-returns, predict on train and test
        train_regimes, test_regimes = detector.fit_predict_pipeline(
            train_df["log_return"], 
            test_df["log_return"]
        )
        
        # Merge one-hot features back into dataframes
        train_out = train_df.join(train_regimes[["regime_bear", "regime_sideways", "regime_bull"]])
        test_out = test_df.join(test_regimes[["regime_bear", "regime_sideways", "regime_bull"]])
        
        # Forward-fill initial warmup NaN values from rolling window
        return train_out.bfill(), test_out.bfill()
        
    @staticmethod
    def _drop_raw_ohlc(df):
        """Drops raw price-scale columns before the frame reaches
        trading_env.py. open/high/low/close are SPY-scale values (hundreds),
        and `day` is FinRL's integer weekday column -- either would enter
        the observation unnormalised, alongside Z-scored indicators
        (~N(0,1) scale) and log_return_norm, if left in."""
        cols_to_drop = [c for c in ['open', 'high', 'low', 'close', 'day'] if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def process(self, raw_df):
        with console.status("[bold green]Calculating technical indicators & applying embargo...[/bold green]", spinner="dots"):
            df_clean = self.clean_data(raw_df)
            df_returns = self.add_log_returns(df_clean)

            train_raw, val_raw, test_raw = self.split_data(df_returns)

            train_feat = self.add_technical_indicators(train_raw)
            val_feat = self.add_technical_indicators(val_raw)
            test_feat = self.add_technical_indicators(test_raw)

            return self.normalize(train_feat, val_feat, test_feat)
    
    def process_multi_asset(self, raw_df):
        """
        Multi-asset pipeline. Structurally reuses process()'s per-asset
        logic (clean -> log returns -> split -> indicators -> normalize)
        rather than reimplementing it, so a future fix to the single-asset
        path (e.g. the embargo-gap fix in split_data(), or the
        log_return_norm fix in normalize()) can't silently diverge between
        the two paths again.

        Two things a naive multi-asset extension would get wrong, both
        handled explicitly:
          1. Log returns must be computed WITHIN each ticker's own series,
             not across interleaved multi-ticker rows -- a groupby-free
             shift(1) on a long-format frame would take the previous
             ROW's close, which is a different ticker's price at ticker
             boundaries.
          2. Every ticker must end up with an IDENTICAL set of
             post-processing dates, or MultiAssetTradingEnv's df.pivot()
             produces NaN for any (date, ticker) gap. Restricting to the
             date INTERSECTION across all tickers before splitting
             guarantees this by construction; the final loop below then
             checks that guarantee held, rather than assuming it.

        Normalisation is computed PER TICKER (each asset's own train-split
        mean/std), not pooled across assets -- an indicator's or return's
        scale can differ meaningfully between assets with very different
        price/volatility profiles (e.g. GLD vs QQQ), so a pooled mean/std
        would distort the less-volatile asset's Z-scores toward the
        more-volatile asset's scale.
        """
        if 'tic' not in raw_df.columns:
            raise ValueError("process_multi_asset requires a 'tic' column identifying each asset.")

        tickers = sorted(raw_df['tic'].unique())

        # Step 1: clean + log returns, PER TICKER (see docstring point 1).
        per_ticker = {}
        for tic in tickers:
            df_t = raw_df[raw_df['tic'] == tic]
            df_t = self.clean_data(df_t)
            df_t = self.add_log_returns(df_t)
            per_ticker[tic] = df_t

        # Step 2: restrict every ticker to the date INTERSECTION across all
        # tickers (see docstring point 2), before splitting.
        common_dates = set(per_ticker[tickers[0]]['date'])
        for tic in tickers[1:]:
            common_dates &= set(per_ticker[tic]['date'])
        if not common_dates:
            raise ValueError(f"No overlapping dates across tickers {tickers} after cleaning.")

        for tic in tickers:
            df_t = per_ticker[tic]
            per_ticker[tic] = df_t[df_t['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

        # Step 3: split, indicators, normalize -- PER TICKER, reusing the
        # exact same methods the single-asset path uses. Because every
        # ticker was already restricted to identical dates in step 2, and
        # split_data()'s split is purely positional, every ticker's
        # train/val/test boundaries land on the same actual dates.
        train_parts, val_parts, test_parts = [], [], []
        for tic in tickers:
            train_raw, val_raw, test_raw = self.split_data(per_ticker[tic])

            train_feat = self.add_technical_indicators(train_raw)
            val_feat = self.add_technical_indicators(val_raw)
            test_feat = self.add_technical_indicators(test_raw)

            train_norm, val_norm, test_norm = self.normalize(train_feat, val_feat, test_feat)

            train_parts.append(train_norm)
            val_parts.append(val_norm)
            test_parts.append(test_norm)

        train_df = pd.concat(train_parts, ignore_index=True).sort_values(['date', 'tic']).reset_index(drop=True)
        val_df = pd.concat(val_parts, ignore_index=True).sort_values(['date', 'tic']).reset_index(drop=True)
        test_df = pd.concat(test_parts, ignore_index=True).sort_values(['date', 'tic']).reset_index(drop=True)

        # Explicit guarantee, not an assumption: confirm every ticker's
        # per-split indicator computation actually preserved identical
        # date coverage (should hold by construction given step 2, but
        # checked directly here rather than left implicit -- this is what
        # multi_asset_env.py's own NaN assert would otherwise be silently
        # relying on without this check existing anywhere).
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if split_df.empty:
                continue
            counts = split_df.groupby('tic')['date'].nunique()
            if counts.nunique() > 1:
                raise ValueError(
                    f"process_multi_asset: '{name}' split has unequal date "
                    f"coverage per ticker after indicator computation: {counts.to_dict()}"
                )

        return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    from src.pipeline.asset_registry import AssetClass

    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-asset", action="store_true",
                         help="Run the multi-asset pipeline instead of the single config.TICKER.")
    parser.add_argument("--asset-class", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    console.print(Panel("[bold white]STAGE 2 · FEATURE ENGINEERING[/bold white]",
                         box=box.ROUNDED, expand=False, border_style="cyan"))

    from src.pipeline.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    processor = DataProcessor()

    if args.multi_asset:
        asset_class = AssetClass(args.asset_class) if args.asset_class else None
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_df, _ = fetcher.fetch_universe(asset_class=asset_class, verbose=False)
        train_df, val_df, test_df = processor.process_multi_asset(raw_df)
        file_prefix = "multi_asset_"
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_df = fetcher.fetch_data([config.TICKER], config.START_DATE, config.END_DATE, verbose=False)
        train_df, val_df, test_df = (processor.process(raw_df) if raw_df is not None else (None, None, None))
        file_prefix = ""

    if train_df is not None:
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
        os.makedirs(save_path, exist_ok=True)
        train_df.to_csv(os.path.join(save_path, f"{file_prefix}train_data.csv"), index=False)
        val_df.to_csv(os.path.join(save_path, f"{file_prefix}val_data.csv"), index=False)
        test_df.to_csv(os.path.join(save_path, f"{file_prefix}test_data.csv"), index=False)
        elapsed_time = time.time() - start_time

        split_table = Table(title="Dataset Splits", box=box.ROUNDED, header_style="bold green")
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Rows", justify="right")
        split_table.add_column("Share", justify="right")
        split_table.add_column("Start Date", justify="center")
        split_table.add_column("End Date", justify="center")

        n_total = len(train_df) + len(val_df) + len(test_df)
        for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            start_d = pd.to_datetime(split_df['date'].min()).strftime('%Y-%m-%d')
            end_d = pd.to_datetime(split_df['date'].max()).strftime('%Y-%m-%d')
            share = f"{len(split_df) / n_total * 100:.1f}%"
            split_table.add_row(name, f"{len(split_df):,}", share, start_d, end_d)
        console.print(split_table)

        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="bold cyan")
        meta.add_column(style="white")
        meta.add_row("Mode", "Multi-asset" if args.multi_asset else "Single-asset")
        meta.add_row("Embargo", f"{config.EMBARGO_DAYS} trading days (pure gap, not counted in split size)")
        meta.add_row("Features", f"{len(config.TECH_INDICATORS)} technical indicators + log_return_norm")
        meta.add_row("Normalization", "Z-score, fitted on Training set only" + (" (per ticker)" if args.multi_asset else ""))
        meta.add_row("Raw OHLC / day", "Dropped before saving")
        meta.add_row("Processing Time", f"{elapsed_time:.2f}s")
        console.print(Panel(meta, box=box.ROUNDED, border_style="green", expand=False))