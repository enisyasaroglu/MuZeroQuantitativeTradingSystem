import os
import pandas as pd
import torch

from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.schedules import temperature_schedule
from utils.dashboard_logger import QuantRLLogger


def run_muzero():
    config = MuZeroConfig()
    
    data_path = os.path.join('data', 'processed', 'train_data.csv')
    if not os.path.exists(data_path):
        print("Data not found!")
        return
    df = pd.read_csv(data_path)
    
    env = StockTradingEnv(df)
    obs_shape = env.observation_space.shape
    
    agent = MuZeroAgent(config, obs_shape)
    
    buffer = ReplayBuffer(
        capacity=2000,
        batch_size=config.batch_size, 
        unroll_steps=5, 
        discount=config.discount_factor
    )
    
    num_episodes = 500

    logger = QuantRLLogger(
        agent_name="MuZero",
        asset_symbol=getattr(config, 'stock_symbol', 'SPY'),
        total_episodes=num_episodes
    )
    
    # Start the live updating terminal view
    logger.start_dashboard()

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        temperature = temperature_schedule(episode, num_episodes)
        
        # Trajectory tracking for financial quantitative metrics
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
            # True net return from the environment's info dict -- env never
            # sets a `daily_return` attribute, so the previous
            # getattr(env, 'daily_return', reward) call always silently
            # fell back to `reward`, which is DSR-shaped (not the true
            # return) whenever USE_DSR_REWARD is True.
            daily_returns.append(info.get('net_return', reward))
            
            obs = next_obs

        buffer.save_game(game_history)
        
        policy_loss, value_loss = 0.0, 0.0
        if buffer.size() >= config.batch_size:
            total_policy_loss, total_value_loss = 0.0, 0.0
            num_updates = 10
            
            for _ in range(num_updates): 
                batch = buffer.sample_batch()
                loss_out, loss_components = agent.update(batch, k_steps=5)
                
                # Safely parse loss components. MuZeroAgent.update() returns
                # keys 'value'/'policy'/'reward' (not 'value_loss'/
                # 'policy_loss') -- the previous key names here always
                # missed, pinning Value Loss at 0.0 for the entire run.
                if isinstance(loss_components, dict):
                    total_policy_loss += float(loss_components.get('policy', loss_out))
                    total_value_loss += float(loss_components.get('value', 0.0))
                elif isinstance(loss_components, (tuple, list)) and len(loss_components) >= 2:
                    total_policy_loss += float(loss_components[0])
                    total_value_loss += float(loss_components[1])
                else:
                    total_policy_loss += float(loss_out)
            
            policy_loss = total_policy_loss / num_updates
            value_loss = total_value_loss / num_updates
        
        # Update live terminal dashboard
        logger.log_cycle(
            cycle=episode,
            portfolio_values=portfolio_history,
            daily_returns=daily_returns,
            policy_loss=policy_loss,
            value_loss=value_loss
        )
        
        # Save model checkpoint periodically without printing duplicated tables
        if episode % 5 == 0:
            save_path = os.path.join('src', 'models', f'muzero_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.network.state_dict(), save_path)

    # Stop live screen engine and display final summary panel
    logger.stop_dashboard()


if __name__ == "__main__":
    run_muzero()