import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from configs.base_config import config
from configs.ppo_config import ppo_config
from src.env.trading_env import StockTradingEnv
from src.agents.ppo.ppo_agent import PPOAgent
from utils.dashboard_logger import QuantRLLogger
from src.utils.schedules import entropy_coef_schedule

EPISODE_LENGTH = 252        # same as MuZero's episode_length
EPISODES_PER_UPDATE = 8     # about 2,000 steps per PPO update


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_memory():
    return {'states': [], 'actions': [], 'log_probs': [],
            'values': [], 'rewards': [], 'dones': []}


def run_ppo(num_episodes=500, checkpoint_every=5,
            data_path=os.path.join('data', 'processed', 'train_data.csv'),
            checkpoint_dir='checkpoints'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    set_seed(config.SEED)
    df = pd.read_csv(data_path)
    asset = str(df['tic'].iloc[0]) if 'tic' in df.columns else config.TICKER

    env = StockTradingEnv(df, use_dsr=False, window_size=EPISODE_LENGTH)
    agent = PPOAgent(obs_shape=env.observation_space.shape,
                     action_dim=env.action_space.n, cfg=ppo_config)

    logger = QuantRLLogger(agent_name="PPO", asset_symbol=asset,
                           total_episodes=num_episodes)
    logger.start_dashboard()

    memory = new_memory()
    policy_loss = value_loss = policy_entropy = 0.0

    try:
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset(seed=config.SEED if episode == 1 else None)
            agent.entropy_coef = entropy_coef_schedule(episode, num_episodes)

            portfolio_history = [env.portfolio_value]
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

                portfolio_history.append(env.portfolio_value)
                daily_returns.append(info.get('net_return', reward))
                state = next_state

            # One update per EPISODES_PER_UPDATE episodes. The GAE loop
            # handles joined episodes because each ends with done=True.
            if episode % EPISODES_PER_UPDATE == 0:
                policy_loss, value_loss, policy_entropy = agent.update(memory)
                memory = new_memory()

            logger.log_cycle(
                cycle=episode,
                portfolio_values=portfolio_history,
                daily_returns=daily_returns,
                policy_loss=policy_loss,
                value_loss=value_loss,
                policy_entropy=policy_entropy,
            )

            if episode % checkpoint_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(agent.policy.state_dict(),
                           os.path.join(checkpoint_dir, f'ppo_checkpoint_{episode}.pth'))
    finally:
        logger.stop_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "processed", "train_data.csv"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    args = parser.parse_args()
    run_ppo(num_episodes=args.episodes, data_path=args.data,
            checkpoint_dir=args.ckpt_dir)