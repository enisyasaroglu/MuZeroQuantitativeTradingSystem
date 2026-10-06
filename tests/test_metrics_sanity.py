import numpy as np
import pytest

from src.utils.metrics import (
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)

SCALAR_METRICS = [
    "total_return", "cagr", "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "calmar_ratio", "win_rate", "annualized_volatility", "worst_single_period_loss",
]


def test_max_drawdown_matches_hand_calculation():
    # Peak 120k, trough 90k: (90000 - 120000) / 120000 = -25%
    values = [100000, 110000, 120000, 105000, 90000, 95000]
    assert max_drawdown(values) == pytest.approx(-0.25, abs=1e-9)


def test_total_return():
    assert total_return([100000, 130000]) == pytest.approx(0.30, abs=1e-9)


def test_sharpe_of_constant_growth_is_zero_not_blowup():
    flat = [100000 * (1.0001 ** i) for i in range(50)]
    assert sharpe_ratio(flat) == 0.0


def test_sortino_matches_textbook_formula():
    rets = np.tile([0.02, -0.01], 100)
    values = 100 * np.concatenate([[1.0], np.cumprod(1 + rets)])
    expected = 0.005 / np.sqrt(0.5 * 0.01 ** 2) * np.sqrt(252)   # about 11.2
    assert sortino_ratio(values) == pytest.approx(expected, rel=1e-6)


def test_win_rate_ignores_cash_days():
    # 2 active days (1 up, 1 down) and 3 flat days -> 50%, not 20%
    values = [100, 100, 101, 101, 100.5, 100.5]
    assert win_rate(values) == pytest.approx(0.5)


def test_flat_portfolio_gives_finite_zeros():
    values = [100000.0] * 50
    m = compute_all_metrics(values)
    for name in SCALAR_METRICS:
        assert np.isfinite(m[name]), name
    assert m["sharpe_ratio"] == 0.0
    assert m["sortino_ratio"] == 0.0
    assert m["max_drawdown"] == 0.0


def test_extreme_single_day_crash_stays_finite():
    values = [100000.0] * 20 + [50000.0] + [50000.0] * 20      # -50% in one day
    m = compute_all_metrics(values)
    for name in SCALAR_METRICS:
        assert np.isfinite(m[name]), name
    assert m["worst_single_period_loss"] == pytest.approx(-0.5)
    assert m["max_drawdown"] == pytest.approx(-0.5)


def test_steady_gains_never_produce_inf():
    # No losing day: Sortino is undefined. NaN is allowed here, inf is not.
    values = 100000 * np.cumprod(np.r_[1.0, np.tile([1.01, 1.02], 50)])
    m = compute_all_metrics(values)
    assert not np.isinf(m["sortino_ratio"])
    assert np.isfinite(m["sharpe_ratio"])


def test_very_short_series_do_not_crash():
    for values in ([100.0], [100.0, 101.0]):
        m = compute_all_metrics(values)
        assert not np.isinf(m["sharpe_ratio"])


def test_random_walk_metrics_are_sane():
    rng = np.random.default_rng(1)
    values = [100000.0]
    for _ in range(500):
        values.append(values[-1] * np.exp(rng.normal(0.0003, 0.01)))
    m = compute_all_metrics(values)
    for name in SCALAR_METRICS:
        assert np.isfinite(m[name]), name
    assert -1.0 < m["max_drawdown"] <= 0.0
    assert m["annualized_volatility"] == pytest.approx(0.16, abs=0.04)   # 0.01 * sqrt(252) ≈ 0.159
    assert 0.3 < m["win_rate"] < 0.7
    assert min(m["rolling_drawdown_series"]) == pytest.approx(m["max_drawdown"])