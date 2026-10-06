import os
import numpy as np
import pandas as pd

n = 1500
t = np.arange(n + 1)
period, amp = 20, 0.05                       # large, perfectly regular swings
log_price = amp * np.sin(2 * np.pi * t / period)
r = np.diff(log_price)                       # log returns, length n

df = pd.DataFrame({
    "date": pd.bdate_range("2015-01-01", periods=n),
    "tic": "SINE",
    "log_return": r,
    "log_return_norm": (r - r.mean()) / r.std(),
})
os.makedirs("data/synthetic", exist_ok=True)
df.to_csv("data/synthetic/sine_train.csv", index=False)
