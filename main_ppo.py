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

from src.env.trading_env import StockTradingEnv
from src.agents.baselines.ppo_agent import PPOAgent
from src.utils.schedules import entropy_coef_schedule
from utils.dashboard_logger import QuantRLLogger


def run_ppo(num_episodes=500, checkpoint_every=5):
    data_path = os.path.join('data', 'processed', 'train_data.csv')
    if not os.path.exists(data_path):
        print("Data not found!")
        return
    df = pd.read_csv(data_path)

    env = StockTradingEnv(df)
    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=env.action_space.n)

    logger = QuantRLLogger(
        agent_name="PPO",
        asset_symbol="SPY",
        total_episodes=num_episodes
    )
    
    # Start the live dashboard interface
    logger.start_dashboard()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
        
        # Track full trajectory metrics for quantitative log calculations
        portfolio_history = [getattr(env, 'portfolio_value', 10000.0)]
        daily_returns = []
        done = False

        # Anneal the entropy bonus over training
        agent.entropy_coef = entropy_coef_schedule(episode, num_episodes)

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)

            # Record step metrics. True net return from the info dict --
            # same fix as main_muzero.py; getattr(env, 'daily_return', ...)
            # never resolved since the environment never sets that attribute.
            portfolio_history.append(getattr(env, 'portfolio_value', 10000.0))
            daily_returns.append(info.get('net_return', reward))

            state = next_state

        # Update policy and extract losses
        update_result = agent.update(memory)
        if isinstance(update_result, tuple):
            if len(update_result) == 3:
                policy_loss, value_loss, _ = update_result
            elif len(update_result) == 2:
                policy_loss, _ = update_result
                value_loss = 0.0
            else:
                policy_loss, value_loss = update_result[0], update_result[1]
        else:
            policy_loss, value_loss = float(update_result), 0.0

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
            save_path = os.path.join('src', 'models', f'ppo_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.policy.state_dict(), save_path)

    # Stop live update engine and render session summary panel
    logger.stop_dashboard()


if __name__ == "__main__":
    run_ppo()