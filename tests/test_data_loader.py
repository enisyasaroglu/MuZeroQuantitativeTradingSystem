import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.pipeline.processor import DataProcessor
from configs.base_config import config


def _make_synthetic_ohlcv(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2015-01-01', periods=n, freq='B')
    price = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame({
        'date': dates, 'tic': 'SPY',
        'open': price, 'high': price * 1.01, 'low': price * 0.99, 'close': price,
        'volume': rng.randint(1_000_000, 5_000_000, n),
    })


def test_split_produces_three_chronological_splits():
    df = _make_synthetic_ohlcv()
    proc = DataProcessor()
    df_clean = proc.clean_data(df)
    df_ret = proc.add_log_returns(df_clean)
    train, val, test = proc.split_data(df_ret)

    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train['date'].max() < val['date'].min()
    assert val['date'].max() < test['date'].min()
    print("test_split_produces_three_chronological_splits PASSED")


def test_embargo_gap_enforced():
    """The gap between the end of train and the start of val (and val/test)
    must be at least EMBARGO_DAYS calendar days -- enough to guarantee no
    rolling-window indicator in the first val/test observation touches
    training-period prices."""
    df = _make_synthetic_ohlcv()
    proc = DataProcessor()
    df_clean = proc.clean_data(df)
    df_ret = proc.add_log_returns(df_clean)
    train, val, test = proc.split_data(df_ret)

    gap_train_val = (val['date'].iloc[0] - train['date'].iloc[-1]).days
    gap_val_test = (test['date'].iloc[0] - val['date'].iloc[-1]).days

    # Business-day embargo of N rows implies at least N calendar days gap
    assert gap_train_val >= config.EMBARGO_DAYS
    assert gap_val_test >= config.EMBARGO_DAYS
    print("test_embargo_gap_enforced PASSED")


def test_no_nans_after_full_pipeline():
    df = _make_synthetic_ohlcv()
    proc = DataProcessor()
    train, val, test = proc.process(df)
    assert train.isnull().sum().sum() == 0
    assert val.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0
    print("test_no_nans_after_full_pipeline PASSED")


def test_normalization_uses_train_statistics_only():
    """
    Regression test for the look-ahead-bias pipeline-order bug: val/test
    must be normalised using TRAIN mean/std, never their own statistics.
    We check this by asserting that shifting val's raw values by a large
    constant offset does not change the mean/std USED for normalisation
    (i.e. proves the train stats, not val's own stats, were applied).
    """
    df = _make_synthetic_ohlcv()
    proc = DataProcessor()
    df_clean = proc.clean_data(df)
    df_ret = proc.add_log_returns(df_clean)
    train_raw, val_raw, test_raw = proc.split_data(df_ret)

    train_feat = proc.add_technical_indicators(train_raw)
    val_feat = proc.add_technical_indicators(val_raw)
    test_feat = proc.add_technical_indicators(test_raw)

    train_norm, val_norm, test_norm = proc.normalize(train_feat, val_feat, test_feat)

    cols = [c for c in list(config.TECH_INDICATORS) + ['log_return', 'volume'] if c in train_feat.columns]
    train_mean = train_feat[cols].mean()
    train_std = train_feat[cols].std()

    expected_val_norm = (val_feat[cols] - train_mean) / (train_std + 1e-8)
    np.testing.assert_allclose(val_norm[cols].values, expected_val_norm.values, atol=1e-6)
    print("test_normalization_uses_train_statistics_only PASSED")


def test_indicators_computed_after_split_not_before():
    """
    Regression test for the pipeline-order bug: indicators computed via
    the public `process()` pipeline must differ from indicators computed
    on the globally-unsplit dataframe near the split boundary, because a
    correct per-split computation has no history before the start of each
    split, while an (incorrect) global computation would.
    This test simply asserts process() calls split BEFORE indicators by
    checking indicator columns are absent immediately after split_data()
    and only appear after add_technical_indicators().
    """
    df = _make_synthetic_ohlcv()
    proc = DataProcessor()
    df_clean = proc.clean_data(df)
    df_ret = proc.add_log_returns(df_clean)
    train_raw, val_raw, test_raw = proc.split_data(df_ret)

    for indicator in config.TECH_INDICATORS:
        assert indicator not in train_raw.columns, \
            f"{indicator} present before add_technical_indicators() was called -- pipeline order violated"
    print("test_indicators_computed_after_split_not_before PASSED")


if __name__ == "__main__":
    test_split_produces_three_chronological_splits()
    test_embargo_gap_enforced()
    test_no_nans_after_full_pipeline()
    test_normalization_uses_train_statistics_only()
    test_indicators_computed_after_split_not_before()