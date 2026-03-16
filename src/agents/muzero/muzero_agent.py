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
from src.agents.muzero.mcts import run_mcts, MinMaxStats

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

    def update(self, batch, k_steps=5):
        """
        The MuZero Training Step.
        Unrolls the network for k_steps and calculates loss at each step.
        """
        self.network.train()
        
        # Unpack the batch
        # Observations: (Batch, 60, 11)
        observations = torch.FloatTensor(np.array(batch['observations'])).to(self.device)
        # Actions taken: (Batch, K_steps) -> We need the sequence of actions!
        action_sequence = torch.LongTensor(np.array(batch['actions'])).to(self.device)
        # Targets: (Batch, K_steps, 3) -> [Target_Value, Target_Reward, Target_Policy]
        target_values = torch.FloatTensor(np.array(batch['target_values'])).to(self.device)
        target_rewards = torch.FloatTensor(np.array(batch['target_rewards'])).to(self.device)
        target_policies = torch.FloatTensor(np.array(batch['target_policies'])).to(self.device)

        # 1. Initial Representation (Step 0)
        hidden_state = self.network.representation(observations)
        
        total_loss = 0
        
        # Loop K steps (Unroll)
        for k in range(k_steps):
            # A. Predict (Policy & Value)
            pred_logits, pred_value = self.network.prediction(hidden_state)
            
            # --- Targets for this step ---
            # target_values[:, k] is the value target for step k
            t_val = target_values[:, k].unsqueeze(1)
            t_pol = target_policies[:, k, :]
            t_rew = target_rewards[:, k].unsqueeze(1)

            # --- Calculate Losses ---
            # 1. Value Loss (MSE)
            loss_value = torch.mean((pred_value - t_val) ** 2)
            
            # 2. Policy Loss (Cross Entropy)
            # t_pol is a probability distribution (from MCTS), pred_logits is raw scores
            loss_policy = torch.sum(-t_pol * torch.nn.functional.log_softmax(pred_logits, dim=1), dim=1).mean()
            
            # 3. Reward Loss (Only if k > 0, because Step 0 has no immediate reward)
            loss_reward = 0
            if k > 0:
                # We used Dynamics to get here, so we check if the reward was correct
                # The reward prediction comes from the PREVIOUS step's dynamics call
                # But for simplicity in this implementation, we handle it implicitly via the loop structure.
                # In strict MuZero, the dynamics outputs (next_state, reward).
                # We compare that 'reward' against t_rew.
                pass 
                # (Note: For this "Walk" phase, we trust the Value Loss to capture the reward signal 
                # to keep the gradient flow simple. Adding explicit reward loss is a "Run" phase optimization).

            # Scale gradient: 1/K to prevent exploding gradients during unroll
            scale = 1.0 / k_steps
            total_loss += scale * (loss_value + loss_policy)

            # --- Dynamics Step (Move to k+1) ---
            # We must use the REAL action that was taken to guide the imagination
            if k < k_steps - 1:
                real_action = action_sequence[:, k].unsqueeze(1) # (Batch, 1)
                
                # Predict next state and reward
                hidden_state, pred_reward = self.network.dynamics(hidden_state, real_action)
                
                # Reward Loss (Calculated here for the transition)
                # Compare predicted reward vs real reward
                real_reward_t = target_rewards[:, k+1].unsqueeze(1)
                loss_reward = torch.mean((pred_reward - real_reward_t) ** 2)
                
                total_loss += scale * loss_reward

        # Optimize
        self.network.optimizer.zero_grad()
        total_loss.backward()
        # Gradient Clipping (Essential for RNNs/Unrolling)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.network.optimizer.step()

        return total_loss.item()