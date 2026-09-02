import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity, batch_size, unroll_steps, discount):
        self.capacity = capacity
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.gamma = discount
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

    def sample_batch(self):
        """
        Constructs a batch of sequences for training.
        """
        obs_batch = []
        act_batch = []
        target_val_batch = []
        target_rew_batch = []
        target_pol_batch = []
        
        # Sample N games
        games = random.sample(self.buffer, self.batch_size)
        
        for game in games:
            # Pick a random start step in this game
            # We need at least unroll_steps remaining
            game_len = len(game['actions'])
            start_index = random.randint(0, game_len - self.unroll_steps - 1)
            
            # 1. Observation at start_index
            obs_batch.append(game['obs'][start_index])
            
            # 2. Action Sequence (for Dynamics)
            # From start_index to start_index + K
            actions = game['actions'][start_index : start_index + self.unroll_steps]
            act_batch.append(actions)
            
            # 3. Targets (Value, Reward, Policy)
            # We need targets for EACH step in the unroll (k=0 to K)
            val_seq = []
            rew_seq = []
            pol_seq = []
            
            for k in range(self.unroll_steps):
                current_idx = start_index + k
                
                # Bootstrap Value (Target = Real Return from this point)
                # We calculate return G_t = r_t + gamma * r_{t+1} ...
                # Ideally, we use the pre-calculated 'returns' from the game
                # But here we calculate on the fly for simplicity:
                G = 0
                for i in range(current_idx, min(current_idx + 10, game_len)):
                    G += (self.gamma ** (i - current_idx)) * game['rewards'][i]
                
                val_seq.append(G)
                rew_seq.append(game['rewards'][current_idx])
                pol_seq.append(game['policies'][current_idx])
                
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