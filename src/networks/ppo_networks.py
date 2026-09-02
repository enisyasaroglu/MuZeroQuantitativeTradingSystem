import torch
import torch.nn as nn
from .shared import TimeSeriesEncoder, MLP

class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic Network for PPO.
    
    Structure:
    1. Shared Encoder (LSTM): (Batch, 60, 11) -> (Batch, Hidden)
    2. Actor Head (Policy):   (Batch, Hidden) -> (Batch, 3) [Logits]
    3. Critic Head (Value):   (Batch, Hidden) -> (Batch, 1) [Value]
    """
    def __init__(self, obs_shape, action_dim, hidden_dim=64):
        super().__init__()
        
        # obs_shape is (Lookback, Features) e.g., (60, 11)
        self.input_features = obs_shape[1]
        
        # 1. The "Eyes": Reuse the robust TimeSeriesEncoder from shared.py
        self.encoder = TimeSeriesEncoder(
            input_features=self.input_features,
            hidden_size=hidden_dim,
            use_lstm=True 
        )
        
        # 2. The Actor (Policy): Decides what to do
        self.actor = MLP(
            input_size=hidden_dim, 
            hidden_sizes=[64], 
            output_size=action_dim
        )
        
        # 3. The Critic (Value): Estimates how good the situation is
        self.critic = MLP(
            input_size=hidden_dim, 
            hidden_sizes=[64], 
            output_size=1
        )

    def forward(self, x):
        # x: (Batch, Lookback, Features)
        
        embedding = self.encoder(x)
        
        action_logits = self.actor(embedding)
        state_value = self.critic(embedding)
        
        return action_logits, state_value