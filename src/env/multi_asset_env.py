"""
Multi-asset trading environment.

SCOPING DECISION (documented, not hidden): full continuous-weight
portfolio allocation would require MuZero's MCTS to search a continuous
action space. The dissertation's own literature review (Section 2.5.2)
identifies the standard technique for this -- progressive widening -- as
future work, not something implemented in the single-asset system. Rather
than silently pretending this environment supports arbitrary continuous
weights, this version uses a DISCRETE MENU of portfolio allocation
templates as the action space:

    0: All-cash (no exposure)
    1: Equal-weight across all assets
    2..N-1: One "single-asset long" template per asset (concentrated bet)
    N: The portfolio optimizer's allocation (recomputed periodically from
       trailing returns via a Portfolio instance -- see src/portfolio/
       portfolio.py). Defaults to PSO (bridges the Swarm Intelligence
       project's PSO implementation into this system), but this
       environment now accepts ANY optimizer satisfying
       PortfolioOptimizerProtocol -- it is no longer PSO-specific.

This keeps the action space discrete and small, so the EXISTING MuZero
architecture (discrete action embeddings, standard closed-form MCTS
action selection) works unmodified -- no changes needed to mcts.py or the
action-embedding layer in dynamics.py. A genuinely continuous-weight
version (arbitrary points on the simplex, searched via progressive
widening) is a natural next step and is noted in the README as future
work, not claimed as implemented here.

REBALANCE CADENCE/CACHING: previously implemented inline here
(_current_pso_template/_pso_template_cache). Now delegated entirely to
Portfolio (src/portfolio/portfolio.py), so this logic exists in exactly
one place -- an environment for method #2 (Mean-Variance, HRP, ...)
reuses the same Portfolio class rather than reimplementing this pattern
a second time.
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
    'tic', 'log_return', plus the technical indicator columns), as
    produced by DataProcessor.process_multi_asset().

    Observation: for each asset, the same LOOKBACK_WINDOW-day feature
    window used in the single-asset environment, concatenated across
    assets: shape (lookback, n_assets * n_features_per_asset).

    Action: discrete index into a fixed menu of portfolio weight
    templates (see module docstring).
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, use_dsr=None, optimizer: PortfolioOptimizerProtocol = None,
                 constraints: PortfolioConstraints = None,
                 rebalance_every: int = 20, lookback_window: int = 60):
        """
        optimizer: any PortfolioOptimizerProtocol-conforming method.
            Defaults to PSO (via PSOOptimizerAdapter) if not provided --
            preserves this environment's previous default behaviour.
        constraints: PortfolioConstraints applied to the optimizer's
            search. Defaults to PortfolioConstraints() (long-only,
            full-investment, no cardinality limit) if not provided --
            matches the original _project_to_simplex behaviour exactly.
        """
        super().__init__()
        self.tickers = sorted(df['tic'].unique())
        self.n_assets = len(self.tickers)
        self.lookback = config.LOOKBACK_WINDOW

        # Observation feature set EXCLUDES raw 'log_return': it must stay
        # unnormalised for portfolio P&L (see _pivot_raw_returns below),
        # while the observation needs every feature on a comparable
        # (roughly mean-0/std-1) scale. 'log_return_norm', produced by
        # DataProcessor.normalize(), is used in the observation instead.
        # This mirrors the equivalent fix in trading_env.py -- mixing an
        # O(0.01)-scale raw return into an otherwise z-scored observation
        # vector is a milder version of the same feature-scale-mismatch
        # bug that raw OHLC prices caused elsewhere in this pipeline.
        self.feature_cols = [c for c in df.columns if c not in ['date', 'tic', 'log_return']]
        self.n_features = len(self.feature_cols)

        pivoted = {}
        for col in self.feature_cols:
            wide = df.pivot(index='date', columns='tic', values=col)[self.tickers]
            pivoted[col] = wide.values  # (n_dates, n_assets)
        self.dates = sorted(df['date'].unique())
        self.n_dates = len(self.dates)

        # panel[t, a, f] -- the OBSERVATION panel (normalised features only).
        self.panel = np.stack([pivoted[c] for c in self.feature_cols], axis=-1).astype(np.float32)

        # SEPARATE panel of RAW log-returns, used only for portfolio P&L
        # and for the optimizer's expected-return/covariance estimation --
        # both need true return magnitudes, not z-scored ones (z-scoring
        # a return series removes exactly the mean-return signal that a
        # Sharpe-ratio fitness needs to be meaningful, on top of the
        # exp()-compounding blow-up risk this caused in the single-asset
        # environment).
        raw_return_wide = df.pivot(index='date', columns='tic', values='log_return')[self.tickers]
        self.raw_returns = raw_return_wide.values.astype(np.float32)  # (n_dates, n_assets)

        # Guard against silent NaN propagation: df.pivot() inserts NaN for
        # any (date, ticker) combination missing from the input panel. An
        # undetected NaN here reaches portfolio_value *= np.exp(net_return)
        # in step() and permanently NaNs the portfolio with no error raised.
        assert not np.isnan(self.panel).any(), "NaN detected in observation panel after pivot -- check for missing (date, ticker) rows."
        assert not np.isnan(self.raw_returns).any(), "NaN detected in raw_returns after pivot -- check for missing (date, ticker) rows."

        # Action menu: cash, equal-weight, one concentrated template per
        # asset, and the optimizer's template (recomputed periodically via
        # self.portfolio, a Portfolio instance -- see src/portfolio/portfolio.py).
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

        # Rebalance cadence/caching now owned entirely by Portfolio, not
        # reimplemented here. Defaults reproduce this environment's
        # previous PSO-specific behaviour exactly if no optimizer/
        # constraints are supplied.
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
        for i in range(self.n_assets):  # 2..N: concentrated single-asset
            w = np.zeros(self.n_assets)
            w[i] = 1.0
            templates.append(w)
        return templates  # optimizer's template appended dynamically per-episode at index len(templates)

    def _get_action_weights(self, action, t):
        if action < len(self.action_templates):
            return self.action_templates[action]
        # The optimizer's template slot: Portfolio owns rebalance timing
        # and caching, so this environment doesn't decide "is it time to
        # re-optimise" itself anymore.
        return self.portfolio.propose_weights(t, self.raw_returns)

    @property
    def n_actions(self):
        return len(self.action_templates) + 1  # +1 for the optimizer's template

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.current_weights = np.zeros(self.n_assets)  # start all-cash
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
        asset_returns = self.raw_returns[self.current_step]  # (n_assets,) -- RAW returns, not observation panel

        gross_return = float(target_weights @ asset_returns)

        # Transaction cost proportional to total turnover (L1 distance
        # between old and new weights) -- a natural multi-asset
        # generalisation of the single-asset "cost only on position
        # change" rule. Computed against the ENVIRONMENT's current_weights
        # (not self.portfolio.current_weights) because the action menu's
        # static templates (cash/equal-weight/concentrated) can also be
        # chosen -- turnover must reflect a switch AWAY FROM those too,
        # not just switches between successive optimizer allocations.
        turnover = np.abs(target_weights - self.current_weights).sum()
        cost = config.TRANSACTION_FEE * turnover
        net_return = gross_return - cost

        self.portfolio_value *= np.exp(net_return)
        self.portfolio_history.append(self.portfolio_value)

        reward = self.dsr.step(net_return) if self.use_dsr else net_return

        self.current_weights = target_weights
        # Keep Portfolio's own current_weights in sync too, in case a
        # future optimizer method's fitness function wants to read it
        # (e.g. a turnover-penalised variant) -- see Portfolio's own
        # docstring on why current-state-awareness is opt-in, not default.
        self.portfolio.commit(target_weights)
        self.current_step += 1

        return self._get_observation(), reward, terminated, False, {
            'portfolio_value': self.portfolio_value,
            'net_return': net_return,
            'weights': target_weights,
        }

    def _get_observation(self):
        window = self.panel[self.current_step - self.lookback: self.current_step]  # (lookback, n_assets, n_features)
        return window.reshape(self.lookback, self.n_assets * self.n_features)

    def render(self):
        print(f"Step: {self.current_step}, Weights: {self.current_weights}, "
              f"Portfolio Value: {self.portfolio_value:.2f}")