import numpy as np, pandas as pd
np.random.seed(0)
n = 300
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n, freq='B'),
    'log_return': np.random.normal(0.0004, 0.01, n),
    'macd': np.random.randn(n), 'rsi_14': np.random.randn(n), 'rsi_30': np.random.randn(n),
    'cci_14': np.random.randn(n), 'dx_30': np.random.randn(n), 'atr_30': np.random.randn(n),
    'boll_ub': np.random.randn(n), 'boll_lb': np.random.randn(n),
})
df.to_csv('data/processed/test_data.csv', index=False)
print('synthetic test split written:', df.shape)