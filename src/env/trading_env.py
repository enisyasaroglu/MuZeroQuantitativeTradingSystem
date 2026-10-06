import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from configs.base_config import config
from src.env.rewards import DifferentialSharpeRatio


class StockTradingEnv(gym.Env):
    """
    A custom trading environment for S&P 500.

    Action Space:
        0: Short
        1: Neutral (Cash)
        2: Long

    Portfolio value compounds using exact log returns:
        portfolio_value *= exp(net_return)

    Reward:
        Differential Sharpe Ratio (DSR) when enabled.
        Raw net log-return otherwise.

    Risk Management:
        Maximum drawdown and daily loss thresholds.
        Once either threshold is breached, the agent
        remains in cash for the rest of the episode.

    Training:
        Random contiguous historical windows when
        window_size is specified.

    Evaluation:
        Full chronological sequence when window_size=None.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        mode="train",
        use_dsr=None,
        window_size=None,
        max_drawdown=None,
        max_daily_loss=None,
    ):
        super().__init__()
        
        max_drawdown = config.MAX_DRAWDOWN if max_drawdown is None else max_drawdown
        max_daily_loss = config.MAX_DAILY_LOSS if max_daily_loss is None else max_daily_loss

        # Validate inputs
        if df is None or df.empty:
            raise ValueError("Input dataframe cannot be empty.")

        if max_drawdown <= 0 or max_drawdown > 1:
            raise ValueError(
                "max_drawdown must be in (0, 1]."
            )

        if max_daily_loss <= 0 or max_daily_loss > 1:
            raise ValueError(
                "max_daily_loss must be in (0, 1]."
            )

        if window_size is not None and (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size <= 0
        ):
            raise ValueError(
                "window_size must be a positive integer."
            )

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.window_size = window_size

        self.max_drawdown = float(max_drawdown)
        self.max_daily_loss = float(max_daily_loss)

        # Action space: Short, Neutral, Long
        self.action_space = spaces.Discrete(3)

        # Exclude raw log_return from observations.
        # Keep it unnormalised for portfolio compounding.
        # Use normalised features when available.
        self.feature_cols = [
            col
            for col in self.df.columns
            if col not in ["date", "tic", "log_return"]
        ]

        if not self.feature_cols:
            raise ValueError(
                "No observation features found in dataframe."
            )

        if "log_return" not in self.df.columns:
            raise ValueError(
                "Dataframe must contain a 'log_return' column."
            )

        self.n_features = len(self.feature_cols)
        self.lookback = config.LOOKBACK_WINDOW

        if len(self.df) <= self.lookback:
            raise ValueError(
                "Dataset must contain more rows than "
                "the lookback window."
            )

        if window_size is not None:
            available_steps = len(self.df) - self.lookback

            if window_size > available_steps:
                raise ValueError(
                    "Dataset does not contain enough trading "
                    "steps for the requested window_size."
                )

        # Observation: historical features + current position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, self.n_features + 1),
            dtype=np.float32,
        )

        # Reward configuration
        self.use_dsr = (
            config.USE_DSR_REWARD
            if use_dsr is None
            else use_dsr
        )

        self.dsr = DifferentialSharpeRatio(
            eta=config.DSR_ETA,
            warmup_steps=config.DSR_WARMUP_STEPS,
            clip=config.DSR_CLIP,
        )

        # Environment state
        self.current_step = 0
        self.current_action = 1

        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [self.portfolio_value]

        self.peak_portfolio_value = self.portfolio_value
        self.risk_halted = False

        self._window_start = self.lookback
        self._window_end = len(self.df)

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        If window_size is specified, sample a random
        contiguous window for training.

        Otherwise, use the complete chronological
        sequence for evaluation.
        """
        super().reset(seed=seed)

        if self.window_size is not None:
            max_start = len(self.df) - self.window_size

            self._window_start = int(
                self.np_random.integers(
                    self.lookback,
                    max_start + 1,
                )
            )

            self._window_end = (
                self._window_start + self.window_size
            )
        else:
            self._window_start = self.lookback
            self._window_end = len(self.df)

        # Reset episode state
        self.current_step = self._window_start
        self.current_action = 1

        self.portfolio_value = config.INITIAL_CAPITAL
        self.portfolio_history = [self.portfolio_value]

        self.peak_portfolio_value = self.portfolio_value
        self.risk_halted = False

        self.dsr.reset()

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one trading step.

        The action determines exposure to the next
        realised log return. Risk limits are checked
        after the return has been realised.
        """
        if self.current_step >= self._window_end:
            raise RuntimeError(
                "Episode has terminated. Call reset() "
                "before taking another step."
            )

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Once halted, remain in cash.
        if self.risk_halted:
            action = 1

        # Realised log return for this trading step
        current_log_return = float(
            self.df.iloc[self.current_step]["log_return"]
        )

        if not np.isfinite(current_log_return):
            raise ValueError(
                f"Invalid log return at index "
                f"{self.current_step}."
            )

        # Map actions to portfolio exposure:
        # Short = -1, Neutral = 0, Long = +1
        position_multiplier = action - 1

        gross_return = (
            current_log_return * position_multiplier
        )

        # Charge transaction costs whenever the
        # position changes, including entering cash.
        old_position = self.current_action - 1
        cost = config.TRANSACTION_FEE * abs(position_multiplier - old_position)

        net_return = gross_return - cost

        # Compound using log returns.
        self.portfolio_value *= np.exp(net_return)

        self.portfolio_history.append(
            self.portfolio_value
        )

        # Update running portfolio peak.
        self.peak_portfolio_value = max(
            self.peak_portfolio_value,
            self.portfolio_value,
        )

        # Calculate drawdown from the running peak.
        drawdown = (
            1.0
            - self.portfolio_value
            / self.peak_portfolio_value
        )

        # Calculate realised daily loss.
        daily_loss = max(
            0.0,
            1.0 - np.exp(net_return),
        )

        # Trigger risk protection after the realised
        # return. It applies to subsequent steps.
        triggered = (
            drawdown >= self.max_drawdown
            or daily_loss >= self.max_daily_loss
        )

        if triggered:
            self.risk_halted = True

        # Calculate reward.
        reward = (
            self.dsr.step(net_return)
            if self.use_dsr
            else net_return
        )

        # Update state.
        self.current_action = action
        self.current_step += 1

        # End the episode after the requested number
        # of trading transitions.
        terminated = (
            self.current_step >= self._window_end
        )

        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "net_return": net_return,
            "gross_return": gross_return,
            "transaction_cost": cost,
            "drawdown": drawdown,
            "daily_loss": daily_loss,
            "risk_halted": self.risk_halted,
            "position": position_multiplier,
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _get_observation(self):
        """
        Return the historical observation window.

        Only data up to, but not including, current_step
        is included. The current portfolio position is
        appended as an additional feature.
        """
        window = self.df.iloc[
            self.current_step - self.lookback:
            self.current_step
        ][self.feature_cols]

        features = window.to_numpy(dtype=np.float32)

        position = np.full(
            (self.lookback, 1),
            self.current_action - 1,
            dtype=np.float32,
        )

        observation = np.concatenate(
            [features, position],
            axis=1,
        )

        return observation.astype(np.float32)

    def render(self):
        """Display the current environment state."""
        position_names = {
            -1: "Short",
            0: "Neutral",
            1: "Long",
        }

        position = self.current_action - 1
        position_name = position_names[position]

        print(
            f"Step: {self.current_step} | "
            f"Position: {position_name} | "
            f"Portfolio: {self.portfolio_value:.2f} | "
            f"Drawdown: "
            f"{1 - self.portfolio_value / self.peak_portfolio_value:.2%} | "
            f"Risk Halted: {self.risk_halted}"
        )