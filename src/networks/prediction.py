import torch.nn as nn
from src.networks.shared import MLP

class PredictionNetwork(nn.Module):
    """
    f(s): Hidden State -> (Policy Logits, Value)
    """
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        
        # Policy Head: Predicts action probabilities
        self.policy_head = MLP(
            input_size=hidden_dim,
            hidden_sizes=[64],
            output_size=action_dim
        )
        
        # Value Head: Predicts expected return
        self.value_head = MLP(
            input_size=hidden_dim,
            hidden_sizes=[64],
            output_size=1
        )

    def forward(self, hidden_state):
        # hidden_state: (Batch, Hidden_Dim)
        
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        
        return policy_logits, value
