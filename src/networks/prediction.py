import torch.nn as nn
from .shared import MLP

class PredictionNetwork(nn.Module):
    """
    f(s): Hidden State -> (Policy Logits, Value)

    Both heads use ELU rather than ReLU. The hidden state feeding this
    network is tanh-bounded to [-1, 1] (see TimeSeriesEncoder /
    DynamicsNetwork), so roughly half of its incoming activations are
    negative. ReLU zeroes the gradient entirely for negative inputs, which
    would kill gradient flow through a large fraction of this network's
    inputs; ELU stays smooth and non-zero for negative inputs.
    """
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        
        # Policy Head: Predicts action probabilities
        self.policy_head = MLP(
            input_size=hidden_dim,
            hidden_sizes=[64],
            output_size=action_dim,
            activation=nn.ELU,
        )
        
        # Value Head: Predicts expected return
        self.value_head = MLP(
            input_size=hidden_dim,
            hidden_sizes=[64],
            output_size=1,
            activation=nn.ELU,
        )

    def forward(self, hidden_state):
        # hidden_state: (Batch, Hidden_Dim)
        
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        
        return policy_logits, value