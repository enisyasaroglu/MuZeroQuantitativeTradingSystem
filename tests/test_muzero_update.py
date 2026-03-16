import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from configs.muzero_config import MuZeroConfig
from src.agents.muzero.muzero_agent import MuZeroAgent

def test_muzero_update():
    # 1. Setup
    config = MuZeroConfig()
    obs_shape = (60, 11)
    agent = MuZeroAgent(config, obs_shape)
    
    print("--- Testing MuZero Update (Backprop) ---")
    
    # 2. Create Fake Batch (Batch Size = 4, Unroll Steps = 5)
    batch_size = 4
    k_steps = 5
    
    # Observations (Start of the sequence)
    observations = np.random.randn(batch_size, 60, 11).astype(np.float32)
    
    # Actions Sequence (What actually happened for next 5 steps)
    actions = np.random.randint(0, 3, (batch_size, k_steps))
    
    # Targets (What MCTS found)
    # Values: (Batch, K)
    target_values = np.random.randn(batch_size, k_steps).astype(np.float32)
    # Rewards: (Batch, K)
    target_rewards = np.random.randn(batch_size, k_steps).astype(np.float32)
    # Policies: (Batch, K, 3) - Normalized probs
    target_policies = np.random.rand(batch_size, k_steps, 3).astype(np.float32)
    target_policies /= target_policies.sum(axis=2, keepdims=True)
    
    batch = {
        'observations': observations,
        'actions': actions,
        'target_values': target_values,
        'target_rewards': target_rewards,
        'target_policies': target_policies
    }
    
    # 3. Run Update
    initial_loss = agent.update(batch, k_steps=k_steps)
    print(f"Initial Loss: {initial_loss:.4f}")
    
    # 4. Run Again (Check for crash)
    loss_2 = agent.update(batch, k_steps=k_steps)
    print(f"Second Step Loss: {loss_2:.4f}")
    
    assert initial_loss > 0
    print("Success! MuZero Agent can learn from sequences.")

if __name__ == "__main__":
    test_muzero_update()