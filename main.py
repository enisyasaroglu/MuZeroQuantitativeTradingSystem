import pandas as pd
import numpy as np
import torch
import os
from configs.base_config import config
from src.env.trading_env import StockTradingEnv
from src.agents.ppo_agent import PPOAgent

def train():
    # 1. Load Processed Data
    data_path = os.path.join('data', 'processed', 'train_data.csv')
    if not os.path.exists(data_path):
        print("Error: train_data.csv not found. Please run src/data/processor.py first.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Initialize Environment and Agent
    env = StockTradingEnv(df)
    
    # Obs shape: (60, 11), Action dim: 3
    agent = PPOAgent(
        obs_shape=env.observation_space.shape, # (Time steps, features)
        action_dim=env.action_space.n, # (0=Sell, 1=Hold, 2=Buy)
        lr=0.0003,      # Learning Rate
        gamma=0.99,     # Discount Factor
        K_epochs=4      # PPO updates the policy repeats per batch
    )
    
    # 3. Training Loop Settings
    num_episodes = 500  # Start small for testing
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        
        # Memory storage for this episode
        memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }
        
        total_reward = 0
        done = False
        
        # --- Run One Episode ---
        while not done:
            # A. Select Action
            action, log_prob, val = agent.select_action(state)
            
            # B. Execute Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # C. Store in Memory
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            
            # D. Move to next state
            state = next_state
            total_reward += reward
            
        # --- Update Agent (Learning) ---
        # We update the network using the data collected in this episode
        loss = agent.update(memory)
        
        # --- Logging ---
        # Print progress every episode
        print(f"Episode {episode}/{num_episodes} | Total Reward: {total_reward:.4f} | Loss: {loss:.4f}")
        
        # Save the model periodically
        if episode % 10 == 0:
            save_path = os.path.join('src', 'models', f'ppo_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.policy.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()