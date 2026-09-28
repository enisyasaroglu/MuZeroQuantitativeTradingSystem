"""
Training entry point for the multi-asset environment.

Reuses MuZeroAgent / PPOAgent unmodified: MultiAssetTradingEnv exposes a
discrete action space (a menu of portfolio templates -- see
src/env/multi_asset_env.py for the scoping rationale), so no changes to
the MCTS or network code are needed -- only `action_space_dim` /
`action_dim` need to be set to env.n_actions instead of 3.

Usage:
    python src/pipeline/processor.py --multi-asset
    python main_multi_asset.py --agent muzero
    python main_multi_asset.py --agent ppo
"""

import argparse
import os
import pandas as pd
import torch

from configs.muzero_config import MuZeroConfig
from src.env.multi_asset_env import MultiAssetTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent
from src.agents.ppo.ppo_agent import PPOAgent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.schedules import temperature_schedule, entropy_coef_schedule
from utils.dashboard_logger import QuantRLLogger


def load_multi_asset_split(split="train"):
    path = os.path.join("data", "processed", f"multi_asset_{split}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it with DataProcessor.process_multi_asset() "
            f"on a multi-ticker DataFrame from DataFetcher.fetch_data(['SPY','QQQ','GLD'], ...)."
        )
    return pd.read_csv(path)


def train_muzero(num_episodes=20):
    df = load_multi_asset_split("train")

    # use_dsr=False: with DSR active, "do nothing" (here, action 0 =
    # all-cash) becomes strictly dominant once the running-average return
    # goes negative -- the same proven freeze already fixed in
    # main_muzero.py.
    env = MultiAssetTradingEnv(df, use_dsr=False)

    config = MuZeroConfig()
    config.action_space_dim = env.n_actions

    agent = MuZeroAgent(config, env.observation_space.shape)
    buffer = ReplayBuffer(capacity=500, batch_size=config.batch_size,
                           unroll_steps=config.unroll_steps, discount=config.discount_factor)

    logger = QuantRLLogger(
        agent_name="MuZero",
        asset_symbol=f"Multi-Asset ({env.n_assets})",
        total_episodes=num_episodes
    )
    logger.start_dashboard()

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        temperature = temperature_schedule(episode, num_episodes)
        game_history = {'obs': [], 'actions': [], 'rewards': [], 'policies': [], 'values': []}

        portfolio_history = [getattr(env, 'portfolio_value', 10000.0)]
        daily_returns = []

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

        policy_loss, value_loss, policy_entropy = 0.0, 0.0, 0.0
        if buffer.size() >= config.batch_size:
            total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
            num_updates = 10
            for _ in range(num_updates):
                batch = buffer.sample_batch()
                loss_out, loss_components = agent.update(batch, k_steps=config.unroll_steps)
                if isinstance(loss_components, dict):
                    total_policy_loss += float(loss_components.get('policy', loss_out))
                    total_value_loss += float(loss_components.get('value', 0.0))
                    total_entropy += float(loss_components.get('entropy', 0.0))
                else:
                    total_policy_loss += float(loss_out)
            policy_loss = total_policy_loss / num_updates
            value_loss = total_value_loss / num_updates
            policy_entropy = total_entropy / num_updates

        logger.log_cycle(
            cycle=episode,
            portfolio_values=portfolio_history,
            daily_returns=daily_returns,
            policy_loss=policy_loss,
            value_loss=value_loss,
            policy_entropy=policy_entropy
        )

        if episode % 5 == 0:
            # src/checkpoints, matching every other training script and
            # evaluate.py (was src/models). The "muzero_multiasset_"
            # prefix is deliberately distinct from "muzero_checkpoint_",
            # so evaluate.py's single-asset auto-detection can never
            # pick these up by accident.
            save_path = os.path.join('src', 'checkpoints', f'muzero_multiasset_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.network.state_dict(), save_path)

    logger.stop_dashboard()


def train_ppo(num_episodes=50):
    df = load_multi_asset_split("train")

    # use_dsr=False: same reward fix as train_muzero above and main_ppo.py.
    env = MultiAssetTradingEnv(df, use_dsr=False)

    agent = PPOAgent(obs_shape=env.observation_space.shape, action_dim=env.n_actions)

    logger = QuantRLLogger(
        agent_name="PPO",
        asset_symbol=f"Multi-Asset ({env.n_assets})",
        total_episodes=num_episodes
    )
    logger.start_dashboard()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
        portfolio_history = [getattr(env, 'portfolio_value', 10000.0)]
        daily_returns = []
        done = False

        agent.entropy_coef = entropy_coef_schedule(episode, num_episodes)

        while not done:
            action, log_prob, val = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)

            portfolio_history.append(getattr(env, 'portfolio_value', 10000.0))
            daily_returns.append(info.get('net_return', reward))

            state = next_state

        # Direct 3-tuple unpack, matching main_ppo.py. The previous
        # len()-based fallback silently set value_loss = 0.0 on a
        # mismatched return shape -- the exact mechanism that hid the
        # Value Loss bug earlier. If update()'s return shape ever
        # changes, this should fail loudly instead.
        policy_loss, value_loss, policy_entropy = agent.update(memory)

        logger.log_cycle(
            cycle=episode,
            portfolio_values=portfolio_history,
            daily_returns=daily_returns,
            policy_loss=policy_loss,
            value_loss=value_loss,
            policy_entropy=policy_entropy
        )

        if episode % 10 == 0:
            save_path = os.path.join('src', 'checkpoints', f'ppo_multiasset_checkpoint_{episode}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.policy.state_dict(), save_path)

    logger.stop_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["muzero", "ppo"], default="ppo")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    if args.agent == "muzero":
        train_muzero(num_episodes=args.episodes or 20)
    else:
        train_ppo(num_episodes=args.episodes or 50)