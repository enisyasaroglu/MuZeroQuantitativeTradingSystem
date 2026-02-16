import torch
import torch.nn as nn
from .shared import MLP

class DynamicsNetwork(nn.Module):
    """
    g(s, a): (Hidden State, Action) -> (Next Hidden State, Reward)
    """
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        
        # We need to combine the state vector and the action.
        # Simple approach: One-hot encode action or embed it.
        # Here we use a small embedding layer for the action.
        self.action_embedding = nn.Embedding(num_embeddings=action_dim, embedding_dim=hidden_dim)
        
        # Input to MLP is State + Action_Embedding
        self.dynamics_mlp = MLP(
            input_size=hidden_dim * 2, 
            hidden_sizes=[hidden_dim, hidden_dim], 
            output_size=hidden_dim # Next State
        )
        
        self.reward_mlp = MLP(
            input_size=hidden_dim * 2, 
            hidden_sizes=[64], 
            output_size=1 # Reward is a scalar
        )

    def forward(self, hidden_state, action):
        # hidden_state: (Batch, Hidden_Dim)
        # action: (Batch, 1) -> LongTensor
        
        action_embed = self.action_embedding(action.squeeze(-1)) # (Batch, Hidden_Dim)
        
        # Concatenate State and Action
        x = torch.cat([hidden_state, action_embed], dim=1)
        
        next_hidden_state = self.dynamics_mlp(x)
        reward = self.reward_mlp(x)
        
        # In MuZero, we often normalize the hidden state to keep values stable
        # Using simple min-max or L2 norm is common. 
        # For now, we leave it raw, but we might add Tanh later if unstable.
        return next_hidden_state, reward
