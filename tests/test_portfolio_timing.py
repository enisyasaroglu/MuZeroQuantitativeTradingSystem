"""
Timing checks for Portfolio.propose_weights.

The weights proposed for day t are earned on the return of day t, so they
may use returns up to day t-1 and nothing later. Changing the return of
day t must therefore leave the proposed weights unchanged, while changing
the return of day t-1 must change them. The second check shows that the
first one could fail.
"""

import numpy as np

from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import OptimizationResult, Portfolio

N_ASSETS = 4
LOOKBACK = 60
T = 100


class TiltOptimizer:
    """Gives more weight to assets with a higher average return in the window."""

    def optimize(self, inputs, constraints):
        scores = np.exp(inputs.expected_returns * 100.0)
        return OptimizationResult(
            weights=scores / scores.sum(),
            method_name="tilt",
            converged=True,
        )


def make_returns():
    """Return a fixed 200-day by 4-asset matrix of random daily returns."""
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, (200, N_ASSETS))


def propose(returns):
    """Return the weights a fresh Portfolio proposes for day T."""
    portfolio = Portfolio(
        N_ASSETS,
        TiltOptimizer(),
        PortfolioConstraints(),
        rebalance_every=20,
        lookback=LOOKBACK,
    )
    return portfolio.propose_weights(T, returns)


def test_weights_for_day_t_ignore_the_return_of_day_t():
    returns = make_returns()
    changed = returns.copy()
    changed[T] += np.array([0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(propose(returns), propose(changed))


def test_weights_do_use_the_return_of_the_previous_day():
    returns = make_returns()
    changed = returns.copy()
    changed[T - 1] += np.array([0.5, 0.0, 0.0, 0.0])
    assert not np.allclose(propose(returns), propose(changed))