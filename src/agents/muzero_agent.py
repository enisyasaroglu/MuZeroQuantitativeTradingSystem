import torch
import torch.optim as optim
import numpy as np
import sys
import os

# Internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.networks.representation import RepresentationNetwork
from src.networks.dynamics import DynamicsNetwork
from src.networks.prediction import PredictionNetwork
from src.agents.mcts import run_mcts, MinMaxStats

class MuZeroNetwork(torch.nn.Module):
    """
    The 'Brain' of the agent. 
    It groups the three sub-networks into one manageable module.
    """
    def __init__(self, obs_shape, action_dim, hidden_dim, learning_rate):
        super().__init__()
        
        # 1. Initialize the three networks
        self.representation = RepresentationNetwork(obs_shape, hidden_dim)
        self.dynamics = DynamicsNetwork(hidden_dim, action_dim)
        self.prediction = PredictionNetwork(hidden_dim, action_dim)
        
        # 2. Setup Optimizer (Adam is standard for MuZero)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def get_device(self):
        return next(self.parameters()).device

    @property
    def device(self):
        return self.get_device()
    
class MuZeroAgent:
    def __init__(self, config, obs_shape):
        self.config = config
        self.obs_shape = obs_shape
        self.action_dim = config.action_space_dim
        
        # Initialize the network container
        self.network = MuZeroNetwork(
            obs_shape=obs_shape,
            action_dim=self.action_dim,
            hidden_dim=config.hidden_size,
            learning_rate=config.learning_rate
        )
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def select_action(self, observation):
        """
        The Planning Step.
        1. Observe the market (Observation -> Hidden State).
        2. Think (Run MCTS).
        3. Act (Select best action based on visit counts).
        """
        self.network.eval()
        
        # Prepare MinMaxStats to normalize values during this specific search
        min_max_stats = MinMaxStats()
        
        with torch.no_grad():
            # 1. Convert Observation to Tensor
            # Shape: (1, 60, 11)
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # 2. Generate Initial Hidden State (s_0)
            root_state = self.network.representation(obs_tensor)
            
            # 3. Run Monte Carlo Tree Search
            # This builds the tree and returns the root node
            root = run_mcts(self.config, root_state, self.network, min_max_stats)
            
            # 4. Extract Visit Counts (The 'Policy')
            visit_counts = [root.children[a].visit_count if a in root.children else 0 
                            for a in range(self.action_dim)]
            
            sum_visits = sum(visit_counts)
            
            # Safety check for 0 visits (should not happen with Dirichlet noise)
            if sum_visits == 0:
                probs = [1.0 / self.action_dim] * self.action_dim
            else:
                probs = [v / sum_visits for v in visit_counts]
            
            # 5. Select Action
            # During Training: Sample from probability distribution (Exploration)
            # During Deployment: Pick the move with max visits (Exploitation)
            # For now, we use argmax for stability in testing
            action = np.argmax(visit_counts)
            
            return action, probs, root.value()

    def update(self, memory):
        """
        Placeholder for the MuZero Loss Function.
        We will implement this in the next step.
        """
        # print("MuZero Update called - Logic to be implemented")
        return 0.0