"""
Lightweight validation for freshly-fetched multi-asset data.

Scoped ONLY to failure modes already proven to matter in this project:
missing/misaligned dates across tickers (the root cause the NaN guard in
multi_asset_env.py exists to catch, after the fact) and NaN in core
OHLCV columns. This is not a general data-quality framework -- no
outlier detection, no corporate-action handling. Extend this function if
a real need emerges; don't build a class hierarchy around it speculatively.
"""
import pandas as pd


def validate_multi_asset_frame(df: pd.DataFrame, expected_tickers: list) -> dict:
    """
    Returns {"errors": [...], "warnings": [...]}. Raises ValueError only
    when downstream code (DataProcessor, MultiAssetTradingEnv) could not
    safely proceed -- everything else is a warning for the caller to
    inspect.
    """
    report = {"errors": [], "warnings": []}

    missing_tickers = set(expected_tickers) - set(df['tic'].unique())
    if missing_tickers:
        report["errors"].append(f"No data returned for: {sorted(missing_tickers)}")

    core_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    nan_counts = df[core_cols].isna().sum()
    if nan_counts.any():
        report["errors"].append(f"NaN in core columns: {nan_counts[nan_counts > 0].to_dict()}")

    # Date-coverage: flags an asset with materially fewer trading days
    # than the rest of the universe (recently-listed asset, symbol typo
    # returning a truncated series). A warning, not an error -- a
    # genuinely shorter history can be legitimate.
    coverage = df.groupby('tic')['date'].nunique()
    if len(coverage) > 1:
        short = coverage[coverage < coverage.max() * 0.95]
        if not short.empty:
            report["warnings"].append(f"Assets below 95% of max date coverage: {short.to_dict()}")

    if report["errors"]:
        raise ValueError("Multi-asset validation failed:\n" + "\n".join(report["errors"]))

    return report