import torch
import sys
import os

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.networks.representation import RepresentationNetwork
from src.networks.dynamics import DynamicsNetwork
from src.networks.prediction import PredictionNetwork

def test_muzero_flow():
    # Parameters: These sets up the rules for our test case.
    batch_size = 4  # Number of parallel samples in the batch.
    lookback = 60   # Number of past time steps in the observation.
    n_features = 11 # Number of features per time step (e.g., OHLCV + indicators).
    hidden_dim = 32 # Size of the hidden state in the networks.
    action_dim = 3  # Number of possible actions (e.g., Buy, Sell, Hold). 
    
    # 1. Initialize Networks
    rep_net = RepresentationNetwork((lookback, n_features), hidden_dim)
    dyn_net = DynamicsNetwork(hidden_dim, action_dim)
    pred_net = PredictionNetwork(hidden_dim, action_dim)
    
    # 2. Fake Data: Create a batch of random observations and actions.
    obs = torch.randn(batch_size, lookback, n_features)
    action = torch.randint(0, action_dim, (batch_size, 1))
    
    print("--- Testing MuZero Flow ---")
    
    # Step A: Representation: Encode the observation into a hidden state.
    hidden_state = rep_net(obs)
    print(f"Representation Output (Hidden State): {hidden_state.shape}")
    assert hidden_state.shape == (batch_size, hidden_dim)
    
    # Step B: Prediction: From the hidden state, predict the policy and value.
    policy, value = pred_net(hidden_state)
    print(f"Prediction Policy Shape: {policy.shape}")
    print(f"Prediction Value Shape: {value.shape}")
    assert policy.shape == (batch_size, action_dim)
    
    # Step C: Dynamics: Given the hidden state and action, predict the next hidden state and reward.
    next_hidden, reward = dyn_net(hidden_state, action)
    print(f"Dynamics Next State: {next_hidden.shape}")
    print(f"Dynamics Reward: {reward.shape}")
    assert next_hidden.shape == (batch_size, hidden_dim)
    assert reward.shape == (batch_size, 1)
    
    # Step D: Predict on Next State: From the next hidden state, predict the next policy and value.
    next_policy, next_value = pred_net(next_hidden)
    assert next_policy.shape == (batch_size, action_dim)
    
    print("\n Success! All MuZero components connect correctly.")

if __name__ == "__main__":
    test_muzero_flow()
