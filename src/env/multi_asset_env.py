"""
Multi-asset trading environment.

SCOPING DECISION: Uses a DISCRETE MENU of portfolio allocation templates as
the action space:
    0: All-cash (no exposure)
    1: Equal-weight across all assets
    2..N-1: One "single-asset long" template per asset (concentrated bet)
    N: The portfolio optimizer's allocation (recomputed periodically)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from configs.base_config import config
from src.env.rewards import DifferentialSharpeRatio
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.portfolio import Portfolio, PortfolioOptimizerProtocol
from src.portfolio.methods.pso_optimizer import PSOOptimizerAdapter


class MultiAssetTradingEnv(gym.Env):
    """
    df must be a long-format multi-asset panel (columns include 'date',
    'tic', 'log_return', plus technical indicators), produced by
    DataProcessor.process_multi_asset().
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, use_dsr=None, optimizer: PortfolioOptimizerProtocol = None,
                 constraints: PortfolioConstraints = None,
                 rebalance_every: int = 20, lookback_window: int = 60):
        super().__init__()
        self.tickers = sorted(df['tic'].unique())
        self.n_assets = len(self.tickers)
        self.lookback = config.LOOKBACK_WINDOW

        self.feature_cols = [c for c in df.columns if c not in ['date', 'tic', 'log_return']]
        self.n_features = len(self.feature_cols)

        pivoted = {}
        for col in self.feature_cols:
            wide = df.pivot(index='date', columns='tic', values=col)[self.tickers]
            pivoted[col] = wide.values  # (n_dates, n_assets)
        self.dates = sorted(df['date'].unique())
        self.n_dates = len(self.dates)

        # Observation panel (normalised features)
        self.panel = np.stack([pivoted[c] for c in self.feature_cols], axis=-1).astype(np.float32)

        # Raw log-returns panel for P&L calculations
        raw_return_wide = df.pivot(index='date', columns='tic', values='log_return')[self.tickers]
        self.raw_returns = raw_return_wide.values.astype(np.float32)  # (n_dates, n_assets)

        assert not np.isnan(self.panel).any(), "NaN detected in observation panel after pivot."
        assert not np.isnan(self.raw_returns).any(), "NaN detected in raw_returns after pivot."

        self.action_templates = self._build_static_templates()
        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_assets * self.n_features),
            dtype=np.float32,
        )

        self.use_dsr = config.USE_DSR_REWARD if use_dsr is None else use_dsr
        self.dsr = DifferentialSharpeRatio(
            eta=config.DSR_ETA, warmup_steps=config.DSR_WARMUP_STEPS, clip=config.DSR_CLIP,
        )

        self.portfolio = Portfolio(
            n_assets=self.n_assets,
            optimizer=optimizer or PSOOptimizerAdapter(n_particles=30, n_iterations=40),
            constraints=constraints or PortfolioConstraints(),
            rebalance_every=rebalance_every,
            lookback=lookback_window,
        )

        self.current_step = 0
        self.current_weights = None
        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [config.INITIAL_CAPITAL]

    def _build_static_templates(self):
        templates = [np.zeros(self.n_assets)]  # 0: all-cash
        templates.append(np.full(self.n_assets, 1.0 / self.n_assets))  # 1: equal-weight
        for i in range(self.n_assets):  # 2..N: single-asset long
            w = np.zeros(self.n_assets)
            w[i] = 1.0
            templates.append(w)
        return templates

    def _get_action_weights(self, action, t):
        if action < len(self.action_templates):
            return self.action_templates[action]
        return self.portfolio.propose_weights(t, self.raw_returns)

    @property
    def n_actions(self):
        return len(self.action_templates) + 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.current_weights = np.zeros(self.n_assets)
        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [config.INITIAL_CAPITAL]
        self.dsr.reset()
        self.portfolio.reset()
        return self._get_observation(), {}

    def step(self, action):
        terminated = self.current_step >= self.n_dates - 1
        if terminated:
            return self._get_observation(), 0.0, True, False, {'portfolio_value': self.portfolio_value}

        target_weights = self._get_action_weights(int(action), self.current_step)
        asset_log_returns = self.raw_returns[self.current_step]  # (n_assets,)

        # 1. Convert individual log returns to simple returns: R_i = exp(r_i) - 1
        simple_asset_returns = np.exp(asset_log_returns) - 1.0

        # 2. Compute gross simple return of portfolio: R_p = sum(w_i * R_i)
        gross_simple_return = float(target_weights @ simple_asset_returns)

        # 3. Calculate turnover and transaction cost in simple return space
        turnover = float(np.abs(target_weights - self.current_weights).sum())
        fee_rate = getattr(config, 'TRANSACTION_FEE', 0.001)
        cost = fee_rate * turnover

        # 4. Compute net simple portfolio return
        net_simple_return = gross_simple_return - cost

        # Floor total single-step loss at -99% to prevent negative portfolio values
        net_simple_return = max(net_simple_return, -0.99)

        # 5. Update portfolio value cleanly using simple return
        self.portfolio_value *= (1.0 + net_simple_return)
        self.portfolio_history.append(self.portfolio_value)

        # 6. Convert net simple return back to log return for DSR/reward calculations
        net_log_return = np.log(1.0 + net_simple_return)
        reward = self.dsr.step(net_log_return) if self.use_dsr else net_log_return

        self.current_weights = target_weights
        self.portfolio.commit(target_weights)
        self.current_step += 1

        return self._get_observation(), float(reward), terminated, False, {
            'portfolio_value': self.portfolio_value,
            'net_return': net_log_return,
            'weights': target_weights,
        }

    def _get_observation(self):
        window = self.panel[self.current_step - self.lookback: self.current_step]
        return window.reshape(self.lookback, self.n_assets * self.n_features)

    def render(self):
        print(f"Step: {self.current_step}, Weights: {self.current_weights}, "
              f"Portfolio Value: {self.portfolio_value:.2f}")