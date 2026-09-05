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

# Check normalization uses TRAIN statistics only
train_feat = proc.add_technical_indicators(train_raw)
val_feat = proc.add_technical_indicators(val_raw)
test_feat = proc.add_technical_indicators(test_raw)

cols = [
    c
    for c in list(config.TECH_INDICATORS) + ['volume']
    if c in train_feat.columns
]

# Calculate statistics ONLY from the training set
train_mean = train_feat[cols].mean()
train_std = train_feat[cols].std()

# Recompute expected normalization for all splits
expected_train = (train_feat[cols] - train_mean) / (train_std + 1e-8)
expected_val = (val_feat[cols] - train_mean) / (train_std + 1e-8)
expected_test = (test_feat[cols] - train_mean) / (train_std + 1e-8)

import numpy.testing as npt

npt.assert_allclose(
    train[cols].values,
    expected_train.values,
    atol=1e-6
)

npt.assert_allclose(
    val[cols].values,
    expected_val.values,
    atol=1e-6
)

npt.assert_allclose(
    test[cols].values,
    expected_test.values,
    atol=1e-6
)

print('normalization uses train-only statistics - ok')