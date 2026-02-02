import pandas as pd
import sys
import os
import numpy as np

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.env.trading_env import StockTradingEnv

def test_environment_random_run():
    # 1. Load the processed data
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/train_data.csv')
    if not os.path.exists(data_path):
        print("Error: Train data not found. Run processor.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded Data: {df.shape}")

    # 2. Init Environment
    env = StockTradingEnv(df)
    
    # 3. Reset
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    assert obs.shape == (env.lookback, env.n_features), "Observation shape mismatch!"
    
    # 4. Run Loop (Simulate 10 steps)
    print("Running random simulation...")
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 10:
        # Pick random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.5f}, Cumulative={info['cumulative_return']:.5f}")
        
        total_reward += reward
        step += 1
        
        if terminated or truncated:
            done = True

    print("Test Passed: Environment handles Reset and Step correctly.")

if __name__ == "__main__":
    test_environment_random_run()