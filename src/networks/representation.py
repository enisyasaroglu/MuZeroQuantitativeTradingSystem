import torch.nn as nn
from .shared import TimeSeriesEncoder

class RepresentationNetwork(nn.Module):
    """
    h(x): Observation -> Hidden State
    """
    def __init__(self, obs_shape, hidden_dim):
        super().__init__()
        # obs_shape is (Lookback, Features)
        self.input_features = obs_shape[1]
        
        self.encoder = TimeSeriesEncoder(
            input_features=self.input_features,
            hidden_size=hidden_dim
        )

    def forward(self, observation):
        # observation: (Batch, Lookback, Features)
        # returns: (Batch, Hidden_Dim)
        return self.encoder(observation)
