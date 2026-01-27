import gymnasium as gym
import numpy as np
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """
    A custom wrapper that follows OpenAI Gym interface.
    This hides the complexity of FinRL from the Agent.
    """
    
    def __init__(self, df, config):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.config = config
        self.current_step = 0
        
        # Action Space: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: [Lookback, Features]
        # Shape: (60 days, N_indicators + Price)
        n_features = len(config.TECH_INDICATORS) + 1 # +1 for Price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.LOOKBACK_WINDOW, n_features), 
            dtype=np.float32
        )

    def reset(self, seed=None):
        """
        Resets the environment to the beginning of the dataset.
        """
        self.current_step = self.config.LOOKBACK_WINDOW
        # Return initial state
        return self._get_observation(), {}

    def step(self, action):
        """
        Executes one step in the simulation.
        """
        self.current_step += 1
        
        # 1. Execute Trade & Calculate Reward (DSR or PnL)
        reward = self._calculate_reward(action)
        
        # 2. Check if done (run out of data or bankrupt)
        done = self.current_step >= len(self.df) - 1
        
        # 3. Get next state
        next_state = self._get_observation()
        
        return next_state, reward, done, False, {}

    def _get_observation(self):
        # Return window: [current_step - lookback : current_step]
        pass

    def _calculate_reward(self, action):
        # Implement Differential Sharpe Ratio logic here later
        return 0.0