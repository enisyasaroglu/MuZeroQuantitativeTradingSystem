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
    def __init__(self, obs_shape, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.device = torch.device("cpu") # Move to "cuda" later if you have GPU setup
        
        # Init Policy
        self.policy = PPOActorCritic(obs_shape, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        """
        Run inference to pick an action. No gradients needed here.
        """
        # Convert state (numpy) to tensor
        # state shape: (60, 11) -> (1, 60, 11)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, val = self.policy(state)
            
            # Create distribution to sample action
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), val.item()

    def update(self, memory):
        """
        The core PPO Learning Step.
        """
        # 1. Convert memory to tensors
        states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions = torch.tensor(memory['actions']).to(self.device)
        old_log_probs = torch.tensor(memory['log_probs']).to(self.device)
        rewards = memory['rewards']
        dones = memory['dones']
        
        # 2. Monte Carlo Estimate of Returns (Bootstrap)
        # In a full implementation, we would use GAE here. 
        # For the "Crawl" phase, we use simple discounted returns.
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize returns for stability (Crucial for Finance!)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # 3. PPO Optimization Loop
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logits, state_values = self.policy(states)
            state_values = state_values.squeeze()
            
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Finding Surrogate Loss
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final Loss: Actor Loss + Critic Loss - Entropy Bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy
            
            # Gradient Step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        return loss.mean().item()