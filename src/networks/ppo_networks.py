import torch
import torch.nn as nn
from .shared import TimeSeriesEncoder, MLP

class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic Network for PPO.
    
    1. Representation: LSTM/GRU encodes (60, 11) -> Hidden Vector
    2. Actor Head: Hidden -> Softmax over 3 actions
    3. Critic Head: Hidden -> Scalar Value
    """
    def __init__(self, obs_shape, action_dim, hidden_dim=128):
        super().__init__()
        
        # obs_shape is (Lookback, Features) e.g., (60, 11)
        self.lookback = obs_shape[0]
        self.features = obs_shape[1]
        
        # 1. Shared Feature Extractor (The "Body")
        self.encoder = TimeSeriesEncoder(
            input_features=self.features,
            hidden_size=hidden_dim,
            use_lstm=True # As per report recommendation
        )
        
        # 2. Actor Head (Policy)
        # Outputs logits for actions (Short, Neutral, Long)
        self.actor = MLP(
            input_size=hidden_dim, 
            hidden_sizes=[64], 
            output_size=action_dim
        )
        
        # 3. Critic Head (Value)
        # Outputs a single scalar (Value of state)
        self.critic = MLP(
            input_size=hidden_dim, 
            hidden_sizes=[64], 
            output_size=1
        )

    def forward(self, x):
        # x: (Batch, Lookback, Features)
        
        # Encode
        embedding = self.encoder(x)
        
        # Heads
        action_logits = self.actor(embedding)
        state_value = self.critic(embedding)
        
        return action_logits, state_value