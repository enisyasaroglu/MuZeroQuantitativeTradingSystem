import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from src.portfolio.methods.cvar import CVaROptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


def test_favours_asset_with_less_tail_risk():
    """The point of CVaR over Mean-Variance: two assets can share similar
    day-to-day behaviour, but one carries occasional severe tail losses
    the other doesn't -- CVaR should clearly favour the calmer one."""
    rng = np.random.RandomState(0)
    T = 200
    asset_a = rng.normal(0.0005, 0.01, T)
    asset_b = rng.normal(0.0005, 0.01, T).copy()
    crash_days = rng.choice(T, size=5, replace=False)
    asset_b[crash_days] -= 0.15  # inject a fat left tail
    returns = np.stack([asset_a, asset_b], axis=1)

    result = CVaROptimizer(alpha=0.95).optimize(OptimizationInputs(returns_sample=returns), PortfolioConstraints())
    assert result.weights[0] > result.weights[1]


def test_cvar_estimate_is_at_least_var_estimate():
    """CVaR = VaR + a non-negative average tail excess, by construction
    of the LP (u_t >= 0) -- checked to confirm diagnostics correctly
    extract zeta vs. the full objective, not swapped."""
    rng = np.random.RandomState(1)
    returns = rng.normal(0.0003, 0.02, size=(100, 3))
    result = CVaROptimizer(alpha=0.95).optimize(OptimizationInputs(returns_sample=returns), PortfolioConstraints())
    assert result.diagnostics["cvar_estimate"] >= result.diagnostics["var_estimate"] - 1e-8


def test_weights_sum_to_one():
    rng = np.random.RandomState(2)
    returns = rng.normal(0.0002, 0.015, size=(80, 4))
    result = CVaROptimizer().optimize(OptimizationInputs(returns_sample=returns), PortfolioConstraints())
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_missing_returns_sample_raises():
    try:
        CVaROptimizer().optimize(OptimizationInputs(expected_returns=np.zeros(3)), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_alpha_raises():
    for bad_alpha in (0.0, 1.0, 1.5, -0.1):
        try:
            CVaROptimizer(alpha=bad_alpha)
            assert False, f"expected ValueError for alpha={bad_alpha}"
        except ValueError:
            pass


def test_respects_max_weight_bound_exactly():
    """LP bounds should be honoured EXACTLY, not just approximately via
    post-hoc projection -- constructed so the bound is actually binding."""
    rng = np.random.RandomState(3)
    calm = rng.normal(0.001, 0.005, 100)
    risky = rng.normal(0.001, 0.05, 100)
    returns = np.stack([calm, risky], axis=1)
    constraints = PortfolioConstraints(max_weight=0.6)

    result = CVaROptimizer().optimize(OptimizationInputs(returns_sample=returns), constraints)
    assert result.weights[0] <= 0.6 + 1e-6
    assert constraints.is_feasible(result.weights)


def test_cardinality_enforced_via_projection():
    rng = np.random.RandomState(4)
    returns = rng.normal(0.0005, 0.01, size=(60, 5))
    constraints = PortfolioConstraints(max_assets=2)

    result = CVaROptimizer().optimize(OptimizationInputs(returns_sample=returns), constraints)
    assert constraints.is_feasible(result.weights)
    assert np.sum(result.weights > 1e-9) == 2