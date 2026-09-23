import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size, unroll_steps, discount, td_steps=10):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.gamma = discount
        self.td_steps = td_steps
        self.buffer = []
        self.position = 0

    def save_game(self, game_history):
        """
        Saves a finished game (episode) to memory.
        game_history: dict with lists for 'obs', 'actions', 'rewards', 'policies', 'values'
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = game_history
        self.position = (self.position + 1) % self.capacity

    def can_sample(self):
        """Returns True only when enough games are stored to sample a full batch."""
        return len(self.buffer) >= self.batch_size

    def sample_batch(self):
        """
        Constructs a batch of unrolled sequences for training.
        """
        if len(self.buffer) == 0:
            return None

        obs_batch, act_batch = [], []
        target_val_batch, target_rew_batch, target_pol_batch = [], [], []
        
        # --- CHANGED: Use replacement when buffer has fewer games than batch_size ---
        if len(self.buffer) < self.batch_size:
            games = random.choices(self.buffer, k=self.batch_size)
        else:
            games = random.sample(self.buffer, self.batch_size)
        
        for game in games:
            game_len = len(game['actions'])
            
            max_start = max(0, game_len - self.unroll_steps - 1)
            start_index = random.randint(0, max_start) if max_start > 0 else 0
            
            obs_batch.append(game['obs'][start_index])
            
            val_seq, rew_seq, pol_seq, act_seq = [], [], [], []
            
            for k in range(self.unroll_steps):
                current_idx = start_index + k
                
                if current_idx < game_len:
                    act_seq.append(game['actions'][current_idx])
                    rew_seq.append(game['rewards'][current_idx])
                    pol_seq.append(game['policies'][current_idx])
                    
                    G = 0.0
                    horizon = min(current_idx + self.td_steps, game_len)
                    for i in range(current_idx, horizon):
                        G += (self.gamma ** (i - current_idx)) * game['rewards'][i]
                    
                    if current_idx + self.td_steps < game_len and 'values' in game:
                        G += (self.gamma ** self.td_steps) * game['values'][current_idx + self.td_steps]
                        
                    val_seq.append(G)
                else:
                    act_seq.append(0)
                    rew_seq.append(0.0)
                    val_seq.append(0.0)
                    action_dim = len(game['policies'][0])
                    pol_seq.append([1.0 / action_dim] * action_dim)
            
            act_batch.append(act_seq)
            target_val_batch.append(val_seq)
            target_rew_batch.append(rew_seq)
            target_pol_batch.append(pol_seq)

        return {
            'observations': obs_batch,
            'actions': act_batch,
            'target_values': target_val_batch,
            'target_rewards': target_rew_batch,
            'target_policies': target_pol_batch
        }
        
    def size(self):
        return len(self.buffer)