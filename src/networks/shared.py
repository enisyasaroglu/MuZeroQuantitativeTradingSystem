import torch
import torch.nn as nn

class AbstractNetwork(nn.Module):
    """
    Base class to handle device management uniformly.
    """
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

class MLP(AbstractNetwork):
    """
    Multi-Layer Perceptron.
    Used for the internal processing of Dynamics and Prediction networks.
    """
    def __init__(self, input_size, hidden_sizes, output_size=None, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_size
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
            
        if output_size is not None:
            layers.append(nn.Linear(in_dim, output_size))
            
        self.net = nn.Sequential(*layers)
        self.output_dim = output_size if output_size else hidden_sizes[-1]

    def forward(self, x):
        return self.net(x)

class TimeSeriesEncoder(AbstractNetwork):
    """
    The 'Eyes' of the agent. 
    Encodes the rolling window of market data into a compact hidden state.
    """
    def __init__(self, input_features, hidden_size, num_layers=2, use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
        
        if self.use_lstm:
            self.rnn = nn.LSTM(
                input_size=input_features, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_features, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True
            )

    def forward(self, x):
        # x shape: (Batch, Lookback, Features)
        # RNN Output: (Batch, Lookback, Hidden_Size)
        output, _ = self.rnn(x)
        
        # We take the state at the LAST time step to represent the current market condition
        return output[:, -1, :]