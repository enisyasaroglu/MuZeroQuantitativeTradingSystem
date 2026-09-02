"""
Particle Swarm Optimisation for portfolio weight allocation.

This bridges the "Swarm Intelligence for Portfolio Optimisation" project
(PSO / ABC / ACO applied to cardinality-constrained Markowitz mean-variance
optimisation) into this trading system, rather than treating the two
final-year projects as unrelated. Here, PSO computes a periodically
rebalanced weight vector across the multi-asset universe, maximising the
Sharpe Ratio subject to a full-investment, no-short-selling constraint.

Implemented from scratch (matching the Swarm Intelligence project),
following the standard PSO update rule:
    v_{t+1} = w*v_t + c1*r1*(pbest - x_t) + c2*r2*(gbest - x_t)
    x_{t+1} = x_t + v_{t+1}

Two uses in this repo:
  1. `PSOPortfolioOptimizer.optimize(expected_returns, cov_matrix)` produces
     a single static allocation -- used as one of MultiAssetTradingEnv's
     discrete action templates (see src/env/multi_asset_env.py) and as a
     standalone "PSO-Optimal Rebalance" baseline strategy, directly
     comparable to Buy-and-Hold and the RL agents in evaluate.py.
  2. It can be re-run periodically (e.g. every N days, using a trailing
     lookback window's returns) for a simple periodic-rebalancing
     strategy -- see `rolling_rebalance_weights`.
"""

import numpy as np


class PSOPortfolioOptimizer:
    def __init__(self, n_particles: int = 50, n_iterations: int = 100,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 risk_free_rate: float = 0.0, random_state: int = 42,
                 v_max: float = 0.2):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.risk_free_rate = risk_free_rate
        self.rng = np.random.RandomState(random_state)
        # Velocity clamp: without this, velocity is unbounded while
        # position is hard-projected onto the simplex every iteration --
        # a particle with runaway velocity just gets slammed against the
        # simplex boundary each step instead of converging smoothly. This
        # is very likely the mechanism behind the instability documented
        # in the companion Swarm Intelligence project (high inertia
        # "flying past" optima; low inertia collapsing to a
        # two-asset-concentrated corner solution).
        self.v_max = v_max

    @staticmethod
    def _project_to_simplex(x: np.ndarray) -> np.ndarray:
        """
        Enforces the portfolio constraints: no short-selling (weights >= 0)
        and full investment (weights sum to 1). Raw PSO particle positions
        don't naturally satisfy either, so every position is projected
        onto the probability simplex after each update -- clip negative
        weights to zero, then renormalise to sum to 1.
        """
        x = np.clip(x, 0.0, None)
        total = x.sum()
        if total <= 1e-12:
            # Degenerate particle (all weights clipped to ~0): fall back
            # to an equal-weight allocation rather than dividing by zero.
            return np.full_like(x, 1.0 / len(x))
        return x / total

    def _sharpe_fitness(self, weights: np.ndarray, expected_returns: np.ndarray,
                         cov_matrix: np.ndarray) -> float:
        port_return = float(weights @ expected_returns)
        port_variance = float(weights @ cov_matrix @ weights)
        port_std = np.sqrt(max(port_variance, 1e-12))
        return (port_return - self.risk_free_rate) / port_std

    def optimize(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Returns a weight vector (summing to 1, all >= 0) maximising the
        Sharpe Ratio implied by `expected_returns` and `cov_matrix`.
        """
        expected_returns = np.asarray(expected_returns, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        n_assets = len(expected_returns)

        # Initialise particles uniformly on the simplex-ish region, then project.
        positions = self.rng.uniform(0, 1, size=(self.n_particles, n_assets))
        positions = np.array([self._project_to_simplex(p) for p in positions])
        velocities = self.rng.uniform(-0.1, 0.1, size=(self.n_particles, n_assets))

        pbest_positions = positions.copy()
        pbest_scores = np.array([
            self._sharpe_fitness(p, expected_returns, cov_matrix) for p in positions
        ])

        gbest_idx = int(np.argmax(pbest_scores))
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        for _ in range(self.n_iterations):
            r1 = self.rng.uniform(0, 1, size=(self.n_particles, n_assets))
            r2 = self.rng.uniform(0, 1, size=(self.n_particles, n_assets))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (gbest_position - positions)
            )
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            positions = positions + velocities
            positions = np.array([self._project_to_simplex(p) for p in positions])

            scores = np.array([
                self._sharpe_fitness(p, expected_returns, cov_matrix) for p in positions
            ])

            improved = scores > pbest_scores
            pbest_positions[improved] = positions[improved]
            pbest_scores[improved] = scores[improved]

            best_idx = int(np.argmax(pbest_scores))
            if pbest_scores[best_idx] > gbest_score:
                gbest_score = pbest_scores[best_idx]
                gbest_position = pbest_positions[best_idx].copy()

        return gbest_position

    def rolling_rebalance_weights(self, returns_matrix: np.ndarray, lookback: int = 60,
                                    rebalance_every: int = 20) -> np.ndarray:
        """
        Produces a (n_timesteps, n_assets) array of portfolio weights for
        a periodic-rebalancing strategy: every `rebalance_every` steps,
        re-optimise using the trailing `lookback`-step sample mean/
        covariance of returns_matrix (shape: n_timesteps x n_assets), and
        hold that allocation until the next rebalance.

        This is intentionally simple (sample mean/cov over a trailing
        window, no shrinkage or robust covariance estimation) -- a
        reasonable v1 baseline, with more sophisticated estimators noted
        as future work (see README).
        """
        n_timesteps, n_assets = returns_matrix.shape
        weights_over_time = np.full((n_timesteps, n_assets), 1.0 / n_assets)

        current_weights = np.full(n_assets, 1.0 / n_assets)
        for t in range(lookback, n_timesteps):
            if (t - lookback) % rebalance_every == 0:
                window = returns_matrix[t - lookback:t]
                expected_returns = window.mean(axis=0)
                cov_matrix = np.cov(window, rowvar=False)
                current_weights = self.optimize(expected_returns, cov_matrix)
            weights_over_time[t] = current_weights

        return weights_over_time