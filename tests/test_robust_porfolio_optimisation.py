import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from scipy.stats import norm
from src.portfolio.methods.robust_portfolio_optimisation import BoxUncertaintyEstimator, RobustPortfolioOptimizer
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationInputs


# --- BoxUncertaintyEstimator ---

def test_shrinkage_reduces_expected_return():
    rng = np.random.RandomState(0)
    returns = rng.normal(0.001, 0.02, size=(100, 3))
    raw_mean = returns.mean(axis=0)

    shrunk = BoxUncertaintyEstimator(confidence_level=0.95).shrink_returns(returns)
    assert np.all(shrunk <= raw_mean + 1e-12)


def test_higher_confidence_shrinks_more():
    rng = np.random.RandomState(1)
    returns = rng.normal(0.001, 0.02, size=(100, 3))

    shrunk_low = BoxUncertaintyEstimator(confidence_level=0.60).shrink_returns(returns)
    shrunk_high = BoxUncertaintyEstimator(confidence_level=0.99).shrink_returns(returns)
    assert np.all(shrunk_high <= shrunk_low + 1e-12)


def test_shrinkage_matches_hand_computed_value():
    returns = np.array([[0.01], [0.02], [0.03], [0.00]])  # single asset, T=4
    mu_hat = returns.mean()
    se = returns.std(ddof=1) / np.sqrt(4)
    expected = mu_hat - norm.ppf(0.95) * se

    result = BoxUncertaintyEstimator(confidence_level=0.95).shrink_returns(returns)
    assert np.isclose(result[0], expected)


def test_zero_variance_asset_has_no_shrinkage():
    returns = np.column_stack([
        np.full(20, 0.001),  # constant -- zero variance
        np.random.RandomState(2).normal(0.001, 0.02, 20),
    ])
    shrunk = BoxUncertaintyEstimator().shrink_returns(returns)
    assert np.isclose(shrunk[0], 0.001)


def test_invalid_confidence_level_raises():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            BoxUncertaintyEstimator(confidence_level=bad)
            assert False, f"expected ValueError for confidence_level={bad}"
        except ValueError:
            pass


# --- RobustPortfolioOptimizer ---

def test_returns_sample_required():
    opt = RobustPortfolioOptimizer(n_resamples=10)
    try:
        opt.optimize(OptimizationInputs(expected_returns=np.zeros(3)), PortfolioConstraints())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_weights_sum_to_one():
    rng = np.random.RandomState(3)
    returns = rng.normal(0.0005, 0.015, size=(120, 4))
    opt = RobustPortfolioOptimizer(n_resamples=50, random_state=1)
    result = opt.optimize(OptimizationInputs(returns_sample=returns), PortfolioConstraints())
    assert abs(result.weights.sum() - 1.0) < 1e-6


def test_reproducible_with_fixed_seed():
    rng = np.random.RandomState(4)
    returns = rng.normal(0.0005, 0.015, size=(100, 3))

    result_a = RobustPortfolioOptimizer(n_resamples=30, random_state=7).optimize(
        OptimizationInputs(returns_sample=returns), PortfolioConstraints()
    )
    result_b = RobustPortfolioOptimizer(n_resamples=30, random_state=7).optimize(
        OptimizationInputs(returns_sample=returns), PortfolioConstraints()
    )
    assert np.allclose(result_a.weights, result_b.weights)


def test_respects_cardinality_constraint():
    rng = np.random.RandomState(5)
    returns = rng.normal(0.0005, 0.02, size=(100, 5))
    constraints = PortfolioConstraints(max_assets=2)

    opt = RobustPortfolioOptimizer(n_resamples=30, random_state=1)
    result = opt.optimize(OptimizationInputs(returns_sample=returns), constraints)
    assert constraints.is_feasible(result.weights)
    assert np.sum(result.weights > 1e-9) == 2


def test_handles_degenerate_zero_variance_data_without_crashing():
    """Every bootstrap resample of IDENTICAL rows has zero covariance --
    exercises Mean-Variance's ridge+pinv singular-covariance handling
    (already tested standalone in test_mean_variance.py) through the
    resampling loop, not just directly."""
    returns = np.tile(np.array([0.001, 0.002, -0.001]), (50, 1))
    opt = RobustPortfolioOptimizer(n_resamples=20, random_state=1)
    result = opt.optimize(OptimizationInputs(returns_sample=returns), PortfolioConstraints())
    assert np.all(np.isfinite(result.weights))
    assert abs(result.weights.sum() - 1.0) < 1e-6
    