"""
Contract checks for the multi-asset environment's step results.

* Every step, including the last, must report net_return in its info.
* The end of a window is a time limit, so it must be reported as
  truncated and not as terminated.
"""

import numpy as np
import pandas as pd

from src.env.multi_asset_env import MultiAssetTradingEnv

TICKERS = ["AAA", "BBB", "CCC"]
N_DATES = 130


def make_panel():
    """Build a small long-format panel with one feature per asset."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=N_DATES)
    rows = []
    for tic in TICKERS:
        returns = rng.normal(0.0003, 0.01, N_DATES)
        for date, r in zip(dates, returns):
            rows.append(
                {"date": date, "tic": tic, "log_return": r, "feature": r * 10}
            )
    return pd.DataFrame(rows)


def run_to_the_end(env):
    """Take action 0 (all cash) until the episode ends; return all step results."""
    env.reset()
    results = []
    done = False
    while not done:
        results.append(env.step(0))
        done = results[-1][2] or results[-1][3]
    return results


def test_every_step_reports_net_return():
    env = MultiAssetTradingEnv(make_panel(), use_dsr=False)
    results = run_to_the_end(env)
    assert all("net_return" in info for _, _, _, _, info in results)


def test_end_of_window_is_truncated_not_terminated():
    env = MultiAssetTradingEnv(make_panel(), use_dsr=False)
    results = run_to_the_end(env)
    _, _, terminated, truncated, _ = results[-1]
    assert truncated is True
    assert terminated is False
    assert not any(r[2] for r in results)