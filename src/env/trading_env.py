import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from configs.base_config import config

class StockTradingEnv(gym.Env):
    """
    A custom trading environment for S&P 500.
    
    Action Space:
    0: Short (Sell)
    1: Neutral (Hold Cash)
    2: Long (Buy)
    
    Reward:
    Realized Log Return minus Transaction Costs.
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df, mode='train'):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.mode = mode
        
        # Define Action Space (Discrete)
        # 0=Short, 1=Neutral, 2=Long
        self.action_space = spaces.Discrete(3)
        
        # Define Observation Space
        # Shape: (Lookback Window, Number of Features)
        # We exclude 'date' and 'price' from the features fed to the NN, 
        # but we keep them in self.df for calculation.
        self.feature_cols = [c for c in df.columns if c not in ['date', 'tic']]
        self.n_features = len(self.feature_cols)
        self.lookback = config.LOOKBACK_WINDOW
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback, self.n_features), 
            dtype=np.float32
        )

        # Initialize State
        self.current_step = 0
        self.position = 1 # Start Neutral (1)
        self.cumulative_return = 0.0

    def reset(self, seed=None, options=None):
        """
        Resets the environment index to the beginning of the window.
        """
        super().reset(seed=seed)
        
        # We start at 'lookback' because we need prior data to form the first state
        self.current_step = self.lookback
        self.position = 1  # Reset to Neutral
        self.cumulative_return = 0.0
        
        # Return first observation
        return self._get_observation(), {}

    def step(self, action):
        """
        Executes one time step.
        """
        # 1. Check if we are done (End of Data)
        terminated = self.current_step >= len(self.df) - 1
        if terminated:
            return self._get_observation(), 0.0, True, False, {}

        # 2. Execute Logic
        # Calculate the return of the asset for the NEXT day (t+1)
        # Note: log_return in df is usually ln(P_t / P_{t-1}). 
        # So at step t, the 'log_return' column holds the return achieved *today*.
        # To simulate trading, we decide action at t, and realize return at t+1.
        
        current_log_return = self.df.iloc[self.current_step]['log_return']
        
        # Reward Logic:
        # If Long (2) -> Profit if return is positive
        # If Short (0) -> Profit if return is negative
        # If Neutral (1) -> 0 return (Cash)
        
        # Map action 0,1,2 to -1, 0, 1 multiplier
        position_multiplier = action - 1 
        
        gross_reward = current_log_return * position_multiplier
        
        # Transaction Cost Logic
        # Cost is incurred only if we CHANGE position (e.g. Long -> Short is a flip of size 2)
        # Previous position needs to be stored? 
        # For simplicity in Phase 1: We assume the cost is paid on the *notional* value.
        # If action != prev_action: cost = fee
        
        # For this simplified env, we assume rebalancing every step.
        # But to be precise:
        cost = 0.0
        if action != self.position:
            cost = config.TRANSACTION_FEE
        
        reward = gross_reward - cost
        
        # 3. Update State
        self.position = action
        self.current_step += 1
        self.cumulative_return += reward
        
        # 4. Return tuple
        # (Observation, Reward, Terminated, Truncated, Info)
        return self._get_observation(), reward, terminated, False, {'cumulative_return': self.cumulative_return}

    def _get_observation(self):
        """
        Returns the window of features ending at current_step.
        """
        # Slice: [current - lookback : current]
        obs = self.df.iloc[self.current_step - self.lookback : self.current_step][self.feature_cols]
        return obs.values.astype(np.float32)

    def render(self):
        print(f"Step: {self.current_step}, Position: {self.position}, Cumulative Return: {self.cumulative_return:.4f}")