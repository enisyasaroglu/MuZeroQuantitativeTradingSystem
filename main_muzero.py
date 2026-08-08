import pandas as pd
import numpy as np
import torch
import os
import sys

from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.utils.replay_buffer import ReplayBuffer

def run_muzero():
    # 1. Setup
    config = MuZeroConfig()
    
    # Load Data
    data_path = os.path.join('data', 'processed', 'train_data.csv')
    if not os.path.exists(data_path):
        print("Data not found!")
        return
    df = pd.read_csv(data_path)
    
    env = StockTradingEnv(df)
    obs_shape = env.observation_space.shape # (60, 11)
    
    agent = MuZeroAgent(config, obs_shape)
    
    # Replay Buffer
    buffer = ReplayBuffer(
        capacity=500, 
        batch_size=config.batch_size, 
        unroll_steps=5, 
        discount=config.discount_factor
    )
    
    print(f"Starting MuZero Training on {config.stock_symbol}...")
    print(f"Device: {agent.network.device}")
    
    num_episodes = 20 # Start small, MCTS is slow!
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        # Temp storage for current episode
        game_history = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'policies': [],
            'values': []
        }
        
        step_count = 0
        
        while not done:
            # A. MCTS Planning
            # This is the slow part!
            action, policy, value = agent.select_action(obs)
            
            # B. Execute
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # C. Store Data
            game_history['obs'].append(obs)
            game_history['actions'].append(action)
            game_history['rewards'].append(reward)
            game_history['policies'].append(policy)
            game_history['values'].append(value)
            
            obs = next_obs
            total_reward += reward
            step_count += 1
            
            # Optional: Print progress every 100 steps
            # if step_count % 100 == 0:
            #     print(f"Step {step_count} | Value: {value:.3f}")

        # End of Episode
        buffer.save_game(game_history)
        
        # D. Training Step
        # Only train if we have enough data
        loss = 0
        if buffer.size() >= config.batch_size:
            # Train for K epochs on the buffer
            for _ in range(10): 
                batch = buffer.sample_batch()
                loss, loss_components = agent.update(batch, k_steps=5)
        
        print(f"Episode {episode} | Reward: {total_reward:.4f} | Loss: {loss:.4f} | Steps: {step_count}")
        
        # Save Model
        if episode % 5 == 0:
            save_path = os.path.join('src', 'models', f'muzero_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.network.state_dict(), save_path)

if __name__ == "__main__":
    run_muzero()