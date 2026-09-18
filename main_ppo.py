"""
PPO baseline training entry point.

NOTE: an earlier version of this file was an exact byte-for-byte
duplicate of main_muzero.py -- running `python main_ppo.py` actually
trained and checkpointed a MuZero agent under a PPO-labelled filename,
and every PPO figure evaluated from those checkpoints was really MuZero.
This was compounded by a second bug in evaluate.py's `--agent all` mode
never loading a checkpoint for either agent (see evaluate.py), so no
"PPO" result produced by this project before both fixes reflected an
actually-trained PPO policy. This file now genuinely implements PPO
training using PPOAgent, matching the interface already used correctly
in main_multi_asset.py's train_ppo().
"""
import os
import pandas as pd
import torch

from configs.ppo_config import ppo_config
from src.env.trading_env import StockTradingEnv
from src.agents.baselines.ppo_agent import PPOAgent
from utils.dashboard_logger import QuantRLLogger

def run_ppo(num_episodes=500, checkpoint_every=5):
    data_path = os.path.join('data', 'processed', 'train_data.csv')
    if not os.path.exists(data_path):
        print("Data not found!")
        return
    df = pd.read_csv(data_path)

    env = StockTradingEnv(df, window_size=252)
    action_dim = getattr(env, 'action_dim', getattr(env, 'n_actions', getattr(getattr(env, 'action_space', None), 'n', 3)))
    
    # Initialize PPOAgent with explicit configuration object
    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=action_dim, cfg=ppo_config)
    
    logger = QuantRLLogger(
        agent_name="PPO",
        asset_symbol="SPY",
        total_episodes=num_episodes
    )
    
    # Start the live dashboard interface
    logger.start_dashboard()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        memory = {'states': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'dones': []}
        
        # Track full trajectory metrics for quantitative log calculations
        portfolio_history = [getattr(env, 'portfolio_value', 100000.0)]
        daily_returns = []
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['values'].append(value)
            memory['rewards'].append(reward)
            memory['dones'].append(done)

            # Record step metrics. True net return from the info dict --
            # same fix as main_muzero.py; getattr(env, 'daily_return', ...)
            # never resolved since the environment never sets that attribute.
            portfolio_history.append(getattr(env, 'portfolio_value', 100000.0))
            daily_returns.append(info.get('net_return', reward))

            state = next_state

        # Update policy and extract losses
        policy_loss, value_loss, _ = agent.update(memory)

        # Refresh single live table in terminal
        logger.log_cycle(
            cycle=episode,
            portfolio_values=portfolio_history,
            daily_returns=daily_returns,
            policy_loss=policy_loss,
            value_loss=value_loss
        )

        # Save model checkpoints periodically without printing table duplicates
        if episode % checkpoint_every == 0:
            save_path = os.path.join('src', 'checkpoints', f'ppo_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.policy.state_dict(), save_path)

    # Stop live update engine and render session summary panel
    logger.stop_dashboard()


if __name__ == "__main__":
    run_ppo()