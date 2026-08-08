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

    def __init__(self, df, mode='train', use_dsr=None):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.mode = mode

        # Action Space: 0=Short, 1=Neutral, 2=Long
        self.action_space = spaces.Discrete(3)

        # Observation Space: (Lookback Window, Number of Features)
        self.feature_cols = [c for c in df.columns if c not in ['date', 'tic']]
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
        Resets the environment to the start of the window AND to the
        initial capital. Both must be reset explicitly -- an earlier
        version of this environment reset current_step but not
        portfolio_value, which meant each new episode silently inherited
        whatever capital the previous episode ended with. That bug
        produced anomalous, non-independent validation results across
        checkpoints.
        """
        super().reset(seed=seed)

        self.current_step = self.lookback
        self.current_action = 1  # Neutral
        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [config.INITIAL_CAPITAL]
        self.dsr.reset()

        return self._get_observation(), {}

    def step(self, action):
        """
        Executes one time step.
        """
        terminated = self.current_step >= len(self.df) - 1
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