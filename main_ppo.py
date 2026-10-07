import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_memory():
    return {'states': [], 'actions': [], 'log_probs': [],
            'values': [], 'rewards': [], 'dones': []}


def _settings(obj) -> dict:
    """Plain-data snapshot of a config object, safe to write as JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return {
        k: v for k, v in vars(obj).items()
        if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))
    }


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_config(run_dir, num_episodes, data_path, env) -> None:
    """Record exactly what this run used, so results can be traced later."""
    os.makedirs(run_dir, exist_ok=True)
    record = {
        'agent': 'ppo',
        'num_episodes': num_episodes,
        'episode_length': EPISODE_LENGTH,
        'episodes_per_update': EPISODES_PER_UPDATE,
        'seed': config.SEED,
        'reward': 'raw_net_return',
        'data_path': data_path,
        'data_sha256': _file_sha256(data_path),
        'feature_cols': list(env.feature_cols),
        'obs_shape': list(env.observation_space.shape),
        'ppo_config': _settings(ppo_config),
        'project_config': _settings(config),
    }
    with open(os.path.join(run_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=4)


def run_ppo(num_episodes=500, checkpoint_every=5,
            data_path=os.path.join('data', 'processed', 'train_data.csv'),
            run_dir='checkpoints'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    set_seed(config.SEED)
    df = pd.read_csv(data_path)
    asset = str(df['tic'].iloc[0]) if 'tic' in df.columns else config.TICKER

    env = StockTradingEnv(df, use_dsr=False, window_size=EPISODE_LENGTH)
    agent = PPOAgent(obs_shape=env.observation_space.shape,
                     action_dim=env.action_space.n, cfg=ppo_config)

    write_run_config(run_dir, num_episodes, data_path, env)

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
                os.makedirs(run_dir, exist_ok=True)
                torch.save(agent.policy.state_dict(),
                           os.path.join(run_dir, f'ppo_checkpoint_{episode}.pth'))

        # Train on any episodes collected since the last update, so the
        # final few episodes are not silently discarded.
        if memory['states']:
            agent.update(memory)
            os.makedirs(run_dir, exist_ok=True)
            torch.save(agent.policy.state_dict(),
                       os.path.join(run_dir, 'ppo_final.pth'))
    finally:
        logger.stop_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "processed", "train_data.csv"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--run-dir", "--ckpt-dir", dest="run_dir", default="checkpoints")
    args = parser.parse_args()
    run_ppo(num_episodes=args.episodes, data_path=args.data,
            run_dir=args.run_dir)