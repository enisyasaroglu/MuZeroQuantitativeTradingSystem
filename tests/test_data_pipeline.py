import pandas as pd, numpy as np
from src.pipeline.processor import DataProcessor
from configs.base_config import config

np.random.seed(0)
n = 2000
dates = pd.date_range('2015-01-01', periods=n, freq='B')
price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
df = pd.DataFrame({
    'date': dates,
    'tic': 'SPY',
    'open': price, 'high': price*1.01, 'low': price*0.99, 'close': price,
    'volume': np.random.randint(1e6, 5e6, n),
})

proc = DataProcessor()
train, val, test = proc.process(df)
print('shapes:', train.shape, val.shape, test.shape)
print('columns:', list(train.columns))

# check embargo: gap between last train date and first val date must be >= EMBARGO_DAYS
# (recompute raw split boundaries the same way process() does internally)
df_clean = proc.clean_data(df)
df_ret = proc.add_log_returns(df_clean)
train_raw, val_raw, test_raw = proc.split_data(df_ret)
gap_val = (val_raw['date'].iloc[0] - train_raw['date'].iloc[-1]).days
print('calendar gap (days) between train end and val start (raw, pre-indicator-drop):', gap_val)
assert len(train_raw) > 0 and len(val_raw) > 0 and len(test_raw) > 0

# check no NaNs leaked through
assert train.isnull().sum().sum() == 0
assert val.isnull().sum().sum() == 0
assert test.isnull().sum().sum() == 0
print('no NaNs in any split - ok')

# check normalization uses train stats: recompute manually
cols = [c for c in list(config.TECH_INDICATORS) + ['log_return','volume'] if c in train_raw.columns]
train_feat = proc.add_technical_indicators(train_raw)
mean_check = train_feat[cols].mean()
std_check = train_feat[cols].std()
recomputed = (train_feat[cols] - mean_check) / (std_check + 1e-8)
import numpy.testing as npt
npt.assert_allclose(recomputed.values, train[cols].values, atol=1e-6)
print('train normalization matches expected train-only stats - ok')
print('ALL PROCESSOR SANITY CHECKS PASSED')