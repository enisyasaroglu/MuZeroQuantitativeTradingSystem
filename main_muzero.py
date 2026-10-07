import os
import pandas as pd
import torch
import random
import argparse
import json
import hashlib

from dataclasses import asdict, replace
from configs.base_config import ProjectConfig
from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.schedules import temperature_schedule
from utils.dashboard_logger import QuantRLLogger


def run_muzero(num_episodes=None, num_simulations=None, run_dir="checkpoints"):
    config = MuZeroConfig()
    overrides = {}
    if num_episodes is not None:
        overrides["num_episodes"] = num_episodes
    if num_simulations is not None:
        overrides["num_simulations"] = num_simulations
    if overrides:
        config = replace(config, **overrides)
    project_config = ProjectConfig()
    
    data_path = os.path.join('data', 'processed', 'train_data.csv')
    if not os.path.exists(data_path):
        print("Data not found!")
        return
    df = pd.read_csv(data_path)
    with open(data_path, 'rb') as handle:
        data_sha256 = hashlib.sha256(handle.read()).hexdigest()
    
    env = StockTradingEnv(
        df, 
        use_dsr=False, 
        window_size=config.episode_length
        )
    obs_shape = env.observation_space.shape
    
    agent = MuZeroAgent(config, obs_shape)
    
    buffer = ReplayBuffer(
        capacity=2000,
        batch_size=config.batch_size, 
        unroll_steps=config.unroll_steps,
        discount=config.discount_factor,
        td_steps=config.td_steps
    )
    
    num_episodes = config.num_episodes

    logger = QuantRLLogger(
        agent_name="MuZero",
        asset_symbol=project_config.TICKER,
        total_episodes=num_episodes
    )
    
    logger.start_dashboard()
    
    min_buffer_size = getattr(config, 'min_buffer_size', 1)

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        temperature = temperature_schedule(episode, num_episodes)
        
        portfolio_history = [getattr(env, 'portfolio_value', 10000.0)]
        daily_returns = []
        
        game_history = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'policies': [],
            'values': []
        }
        
        while not done:
            action, policy, value = agent.select_action(obs, temperature=temperature)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            game_history['obs'].append(obs)
            game_history['actions'].append(action)
            game_history['rewards'].append(reward)
            game_history['policies'].append(policy)
            game_history['values'].append(value)
            
            portfolio_history.append(getattr(env, 'portfolio_value', 10000.0))
            daily_returns.append(info.get('net_return', reward))
            
            obs = next_obs

        buffer.save_game(game_history)
        
        policy_loss, value_loss = 0.0, 0.0
        policy_entropy = 0.0
        
        if buffer.size() >= min_buffer_size:
            total_policy_loss, total_value_loss, total_policy_entropy = 0.0, 0.0, 0.0
            num_updates = 10
            successful_updates = 0
            
            for _ in range(num_updates):
                batch = buffer.sample_batch()

                loss_out, loss_components = agent.update(batch, k_steps=config.unroll_steps)
                successful_updates += 1
                
                if isinstance(loss_components, dict):
                    total_policy_loss += float(loss_components.get('policy', loss_out))
                    total_value_loss += float(loss_components.get('value', 0.0))
                    total_policy_entropy += float(loss_components.get('entropy', 0.0))
                elif isinstance(loss_components, (tuple, list)) and len(loss_components) >= 2:
                    total_policy_loss += float(loss_components[0])
                    total_value_loss += float(loss_components[1])
                    total_policy_entropy += float(loss_components[2]) if len(loss_components) > 2 else 0.0
                else:
                    total_policy_loss += float(loss_out)
            
            if successful_updates > 0:
                policy_loss = total_policy_loss / successful_updates
                value_loss = total_value_loss / successful_updates
                policy_entropy = total_policy_entropy / successful_updates
        
        logger.log_cycle(
            cycle=episode,
            portfolio_values=portfolio_history,
            daily_returns=daily_returns,
            policy_loss=policy_loss,
            value_loss=value_loss,
            policy_entropy=policy_entropy
        )
        
        if episode % 5 == 0:
            os.makedirs(run_dir, exist_ok=True)
            save_path = os.path.join(run_dir, f'muzero_checkpoint_{episode}.pth')
            torch.save(agent.network.state_dict(), save_path)
            with open(save_path.replace('.pth', '_config.json'), 'w') as f:
                    json.dump({'episode': episode, 'data_sha256': data_sha256,
                           'feature_cols': list(env.feature_cols), **asdict(config)}, f, indent=4)

    logger.stop_dashboard()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--run-dir", default="checkpoints")
    args = parser.parse_args()
    run_muzero(
        num_episodes=args.episodes,
        num_simulations=args.simulations,
        run_dir=args.run_dir,
    )