import numpy as np
from src.utils.metrics import max_drawdown, total_return, sharpe_ratio, compute_all_metrics

# Analytical drawdown check from the dissertation: 100k -> 120k -> 90k
# expected: (90000 - 120000) / 120000 = -25%
values = [100000, 110000, 120000, 105000, 90000, 95000]
mdd = max_drawdown(values)
print('max_drawdown:', mdd, '(expected -0.25)')
assert abs(mdd - (-0.25)) < 1e-9

# total return sanity
tr = total_return([100000, 130000])
print('total_return:', tr, '(expected 0.30)')
assert abs(tr - 0.30) < 1e-9

# Sharpe: zero-variance flat returns -> should not blow up, should be 0
flat = [100000 * (1.0001 ** i) for i in range(50)]
print('sharpe (near-constant growth):', sharpe_ratio(flat))

# random walk sanity
np.random.seed(1)
rw = [100000]
for _ in range(500):
    rw.append(rw[-1] * np.exp(np.random.normal(0.0003, 0.01)))
print('full metrics on random walk:', compute_all_metrics(rw))
print('ALL METRICS SANITY CHECKS PASSED')