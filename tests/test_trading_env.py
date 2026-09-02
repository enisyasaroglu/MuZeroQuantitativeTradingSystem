import pandas as pd, numpy as np
from src.env.trading_env import StockTradingEnv
from configs.base_config import config

np.random.seed(0)
n = 300
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=n),
    'log_return': np.random.normal(0.0005, 0.01, n),
    'macd': np.random.randn(n), 'rsi_14': np.random.randn(n), 'rsi_30': np.random.randn(n),
    'cci_14': np.random.randn(n), 'dx_30': np.random.randn(n), 'atr_30': np.random.randn(n),
    'boll_ub': np.random.randn(n), 'boll_lb': np.random.randn(n),
})

env = StockTradingEnv(df)
obs, info = env.reset()
assert obs.shape == (config.LOOKBACK_WINDOW, env.n_features)
assert env.portfolio_value == config.INITIAL_CAPITAL
print('reset ok, obs shape', obs.shape)

for _ in range(20):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
print('after 20 steps, portfolio value:', info['portfolio_value'])

# Reset-independence test: run episode to loss, then reset, confirm capital restored
for _ in range(200):
    env.step(0)  # force lots of shorting against random data to likely lose money
val_before_reset = env.portfolio_value
obs, info = env.reset()
assert env.portfolio_value == config.INITIAL_CAPITAL, f'reset bug! got {env.portfolio_value}'
print('reset-independence ok. value before reset was', val_before_reset, '-> after reset:', env.portfolio_value)

# DSR single-call check: reward should not be raw net_return when use_dsr=True after warmup
env2 = StockTradingEnv(df, use_dsr=True)
env2.reset()
for _ in range(config.DSR_WARMUP_STEPS + 2):
    obs, r, term, trunc, info = env2.step(2)
print('DSR-shaped reward after warmup (should generally differ from raw net_return):', r, 'vs net_return', info['net_return'])