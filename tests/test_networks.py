import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.networks.ppo_networks import PPOActorCritic

def test_ppo_network_shapes():
    # 1. Define dummy input
    batch_size = 32
    lookback = 60
    n_features = 11
    n_actions = 3
    
    # Random tensor resembling market data
    dummy_input = torch.randn(batch_size, lookback, n_features)
    
    # 2. Initialize Network
    net = PPOActorCritic(obs_shape=(lookback, n_features), action_dim=n_actions)
    
    # 3. Forward Pass
    logits, value = net(dummy_input)
    
    # 4. Check Shapes
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Logits Shape: {logits.shape}") # Should be (32, 3)
    print(f"Value Shape: {value.shape}")   # Should be (32, 1)
    
    assert logits.shape == (batch_size, n_actions)
    assert value.shape == (batch_size, 1)
    
    print("Test Passed: PPO Network handles shapes correctly.")

if __name__ == "__main__":
    test_ppo_network_shapes()