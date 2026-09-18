import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from configs.ppo_config import ppo_config
from src.networks.ppo_networks import PPOActorCritic

class PPOAgent:
    def __init__(self, obs_shape, action_dim, cfg=ppo_config):
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.eps_clip = cfg.eps_clip
        self.K_epochs = cfg.k_epochs
        self.batch_size = cfg.batch_size
        self.entropy_coef = cfg.entropy_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.target_kl = cfg.target_kl

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PPOActorCritic(obs_shape, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        self.policy.eval()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, val = self.policy(state)
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), val.item()

    def evaluate(self, states, actions):
        logits, state_values = self.policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return log_probs, state_values, dist_entropy

    def update(self, memory):
        self.policy.train()

        states = torch.tensor(np.array(memory['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(memory['actions'], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(memory['log_probs'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(memory['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(memory['dones'], dtype=torch.float32).to(self.device)

        # 1. Compute State Values for full trajectory
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze(-1)

        # 2. Compute Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae = 0.0
        trajectory_len = len(rewards)

        for t in reversed(range(trajectory_len)):
            if t == trajectory_len - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Target returns for Critic (unnormalized)
        returns = advantages + values

        # Normalize advantages across trajectory
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        accum_policy_loss = 0.0
        accum_value_loss = 0.0
        accum_entropy = 0.0
        num_updates = 0

        # 3. K_epochs Optimization Loop
        for epoch in range(self.K_epochs):
            permutation = torch.randperm(dataset_size)
            kl_stopped = False

            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = permutation[start_idx:start_idx + self.batch_size]

                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]

                log_probs, state_values, dist_entropy = self.evaluate(b_states, b_actions)
                state_values = state_values.squeeze(-1)

                # Policy ratio & clipped surrogate loss
                ratios = torch.exp(log_probs - b_old_log_probs)

                # Early stopping check via Target KL Divergence
                with torch.no_grad():
                    approx_kl = ((ratios - 1) - torch.log(ratios)).mean().item()
                if approx_kl > 1.5 * self.target_kl:
                    kl_stopped = True
                    break

                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss
                value_loss = self.mse_loss(state_values, b_returns)
                mean_entropy = dist_entropy.mean()

                # Combined Loss
                total_loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * mean_entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                accum_policy_loss += policy_loss.item()
                accum_value_loss += value_loss.item()
                accum_entropy += mean_entropy.item()
                num_updates += 1

            if kl_stopped:
                break

        if num_updates == 0:
            num_updates = 1

        return (
            accum_policy_loss / num_updates,
            accum_value_loss / num_updates,
            accum_entropy / num_updates
        )