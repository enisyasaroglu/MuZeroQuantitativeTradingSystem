import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.risk.metrics.volatility import realized_volatility
from src.risk.metrics.drawdown import drawdown_series, current_drawdown
from src.risk.metrics.historical_var import historical_var
from src.risk.metrics.historical_cvar import historical_cvar


def test_volatility_matches_existing_metrics_function():
    from src.utils.metrics import annualized_volatility
    values = [100_000 * (1.001 ** i) for i in range(50)]
    assert realized_volatility(values) == annualized_volatility(values)


def test_drawdown_series_min_equals_current_at_the_trough():
    values = [100, 110, 90, 95, 105]
    series = drawdown_series(values)
    # trough is at index 2 (value=90, peak=110) -> (90-110)/110
    assert np.isclose(series[2], (90 - 110) / 110)


def test_current_drawdown_is_last_series_value():
    values = [100, 120, 80, 100]
    assert current_drawdown(values) == drawdown_series(values)[-1]


def test_current_drawdown_zero_at_new_peak():
    values = [100, 90, 120]
    assert current_drawdown(values) == 0.0


def test_var_is_negative_for_volatile_series():
    rng = np.random.RandomState(0)
    values = [100_000]
    for _ in range(200):
        values.append(values[-1] * (1 + rng.normal(0, 0.02)))
    assert historical_var(values, confidence=0.95) < 0


def test_cvar_at_least_as_negative_as_var():
    rng = np.random.RandomState(1)
    values = [100_000]
    for _ in range(200):
        values.append(values[-1] * (1 + rng.normal(0, 0.02)))
    var = historical_var(values, confidence=0.95)
    cvar = historical_cvar(values, confidence=0.95)
    assert cvar <= var


def test_short_series_returns_zero_not_crash():
    assert historical_var([100_000]) == 0.0
    assert historical_cvar([100_000]) == 0.0
    assert current_drawdown([100_000]) == 0.0