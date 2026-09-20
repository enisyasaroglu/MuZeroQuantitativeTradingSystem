import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.networks.ppo_networks import PPOActorCritic


class PPOAgent:
    def __init__(self, obs_shape, action_dim, cfg=None, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, K_epochs=4, entropy_coef=0.02, max_grad_norm=0.5):
        """
        cfg: optional PPOConfig (configs/ppo_config.py). When provided,
        its fields take precedence over the keyword defaults -- this is
        what main_ppo.py's `PPOAgent(..., cfg=ppo_config)` needed and
        previously crashed on, since this constructor had no `cfg`
        parameter at all.

        cfg=None (e.g. main_multi_asset.py's
        `PPOAgent(obs_shape=..., action_dim=...)`, which never passes
        cfg) keeps the keyword defaults exactly as before -- existing
        callers are unaffected.

        cfg.gae_lambda and cfg.target_kl are DECLARED in PPOConfig but
        NOT consumed here: GAE and adaptive-KL early stopping aren't
        implemented in update() below (still the disclosed plain
        discounted-return baseline). Flagged explicitly rather than
        silently ignored, so the config doesn't imply a capability this
        file doesn't have.
        """
        if cfg is not None:
            lr = cfg.learning_rate
            gamma = cfg.gamma
            eps_clip = cfg.eps_clip
            K_epochs = cfg.k_epochs
            entropy_coef = cfg.entropy_coef
            max_grad_norm = cfg.max_grad_norm

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cpu")

        self.policy = PPOActorCritic(obs_shape, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
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
        # Re-enable dropout for the actual gradient step -- select_action()
        # calls self.policy.eval() for deterministic/noise-free inference;
        # without this, dropout stayed permanently off after the first
        # inference call, since nothing else ever switched back to train
        # mode.
        self.policy.train()

        states = torch.tensor(np.array(memory['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(memory['actions'], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(memory['log_probs'], dtype=torch.float32).to(self.device)
        rewards = memory['rewards']
        dones = memory['dones']

        returns = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs, state_values, dist_entropy = self.evaluate(states, actions)
        state_values = state_values.squeeze(-1)

        advantages = returns - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = self.mse_loss(state_values, returns)

        mean_entropy = dist_entropy.mean()

        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * mean_entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping via cfg.max_grad_norm -- declared in
        # PPOConfig but previously unused anywhere. Standard, cheap
        # stability guard; now actually does what the config implies.
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), mean_entropy.item()