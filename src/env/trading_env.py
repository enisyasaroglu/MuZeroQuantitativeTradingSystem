import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from configs.base_config import config
from src.env.rewards import DifferentialSharpeRatio


class StockTradingEnv(gym.Env):
    """
    A custom trading environment for S&P 500.

    Action Space:
    0: Short (Sell)
    1: Neutral (Hold Cash)
    2: Long (Buy)

    Portfolio value compounds continuously via the exact log-return
    formula: portfolio_value *= exp(net_return). This is financially exact
    for log-returns, unlike the linear approximation
    portfolio_value *= (1 + net_return), which introduces compounding
    error over long episodes.

    Reward is either the Differential Sharpe Ratio (dense, risk-adjusted;
    config.USE_DSR_REWARD = True) or the raw net log-return (used for
    baseline/ablation comparisons and for evaluation, where we want the
    portfolio's true realised return rather than a shaped training signal).
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, mode='train', use_dsr=None, window_size=None):
        """
        window_size: if set, reset() samples a random contiguous slice of
        this many rows from self.df each episode, instead of always
        walking the full sequence from self.lookback to the end. This
        prevents an agent from memorising one fixed historical path
        (confirmed directly: MuZero's training dashboard showed
        near-identical portfolio values recurring across non-adjacent
        episodes, consistent with the policy reproducing the same trade
        sequence on the same dates every time) rather than learning
        features that generalise across regimes.

        window_size=None (default) preserves the ORIGINAL full-sequential
        behaviour -- unchanged for evaluation/backtesting, where walking
        the complete, real chronological sequence is the correct thing to
        do, not an oversight to fix. Only pass window_size for TRAINING.
        """
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.window_size = window_size

        # Action Space: 0=Short, 1=Neutral, 2=Long
        self.action_space = spaces.Discrete(3)

        # Observation feature set EXCLUDES raw 'log_return' -- that column
        # must stay unnormalised for portfolio compounding (see step()
        # below), while the observation needs every feature on a
        # comparable, roughly mean-0/std-1 scale. 'log_return_norm'
        # (added by DataProcessor.normalize()) is used in the observation
        # instead. Mirrors the same fix in multi_asset_env.py's
        # feature_cols.
        self.feature_cols = [c for c in df.columns if c not in ['date', 'tic', 'log_return']]
        self.n_features = len(self.feature_cols)
        self.lookback = config.LOOKBACK_WINDOW

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_features),
            dtype=np.float32
        )

        # use_dsr defaults to config.USE_DSR_REWARD but can be overridden
        # per-instance (e.g. evaluation always uses raw log-return so that
        # reported financial metrics reflect true portfolio performance,
        # not the shaped training signal).
        self.use_dsr = config.USE_DSR_REWARD if use_dsr is None else use_dsr
        self.dsr = DifferentialSharpeRatio(
            eta=config.DSR_ETA,
            warmup_steps=config.DSR_WARMUP_STEPS,
            clip=config.DSR_CLIP,
        )

        # State (all real initialisation happens in reset(); these are
        # placeholders so attributes exist before the first reset() call)
        self.current_step = 0
        self.current_action = 1  # Start Neutral
        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [config.INITIAL_CAPITAL]

    def reset(self, seed=None, options=None):
        """
        Resets to the start of a fresh episode window.

        With window_size set: samples a random contiguous slice
        [start, start+window_size) from self.df, so each episode covers a
        different historical period rather than always the same full
        sequence -- see window_size's docstring in __init__.

        Both current_step AND portfolio_value are reset explicitly --
        an earlier version of this environment reset current_step but
        not portfolio_value, which meant each new episode silently
        inherited whatever capital the previous episode ended with. That
        bug produced anomalous, non-independent validation results across
        checkpoints.
        """
        super().reset(seed=seed)

        if self.window_size is not None:
            max_start = len(self.df) - self.window_size
            if max_start <= self.lookback:
                # Not enough data for a full window -- fall back to the
                # full sequence rather than raising, matching the
                # window_size=None behaviour.
                self._window_start = self.lookback
                self._window_end = len(self.df)
            else:
                self._window_start = self.np_random.integers(self.lookback, max_start + 1)
                self._window_end = self._window_start + self.window_size
        else:
            self._window_start = self.lookback
            self._window_end = len(self.df)

        self.current_step = self._window_start
        self.current_action = 1  # Neutral
        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [config.INITIAL_CAPITAL]
        self.dsr.reset()

        return self._get_observation(), {}

    def step(self, action):
        """
        Executes one time step.
        """
        window_end = getattr(self, '_window_end', len(self.df))
        terminated = self.current_step >= min(window_end, len(self.df)) - 1
        if terminated:
            return self._get_observation(), 0.0, True, False, {
                'portfolio_value': self.portfolio_value,
            }

        # log_return at current_step is ln(P_t / P_{t-1}), i.e. the return
        # realised BY today. We decide the action at t and realise the
        # position's P&L over that same return.
        current_log_return = self.df.iloc[self.current_step]['log_return']

        # Map action 0,1,2 to position multiplier -1, 0, 1
        position_multiplier = int(action) - 1
        gross_return = current_log_return * position_multiplier

        # Transaction cost applied only when position changes
        cost = config.TRANSACTION_FEE if int(action) != int(self.current_action) else 0.0
        net_return = gross_return - cost

        # Portfolio compounding: exact log-return compounding, cost already
        # deducted from net_return before this multiplication.
        self.portfolio_value *= np.exp(net_return)
        self.portfolio_history.append(self.portfolio_value)

        # Reward: DSR (single call per step -- an earlier version of this
        # environment called dsr.step() twice per step, which silently
        # advanced the DSR's internal EMA statistics twice per environment
        # step and corrupted the reward signal) or raw net log-return.
        reward = self.dsr.step(net_return) if self.use_dsr else net_return

        self.current_action = int(action)
        self.current_step += 1

        return self._get_observation(), reward, terminated, False, {
            'portfolio_value': self.portfolio_value,
            'net_return': net_return,
        }

    def _get_observation(self):
        """
        Returns the window of features ending at current_step.
        """
        obs = self.df.iloc[self.current_step - self.lookback: self.current_step][self.feature_cols]
        return obs.values.astype(np.float32)

    def render(self):
        print(f"Step: {self.current_step}, Position: {self.current_action}, "
              f"Portfolio Value: {self.portfolio_value:.2f}")