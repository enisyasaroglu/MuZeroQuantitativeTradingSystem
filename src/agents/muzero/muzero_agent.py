import torch
import torch.optim as optim
import numpy as np
import sys
import os


def scale_gradient(tensor, scale):
    """
    Scales the gradient flowing through `tensor` by `scale` on the backward
    pass, while leaving the forward value untouched.

    Used between MuZero unroll steps: each application of the dynamics
    network in a K-step unroll multiplies gradients by that step's
    Jacobian on the way back through backprop, so an unscaled K-step chain
    can explode. Scaling by 0.5 at each step damps this while still
    letting a multi-step gradient signal reach the representation network
    -- a full stop-gradient (scale=0) would prevent the dynamics network
    from ever learning to produce hidden states that are useful as the
    STARTING point for future unroll steps, since no loss from step k+1
    onward would ever reach the state produced at step k.
    """
    return tensor * scale + tensor.detach() * (1 - scale)

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

    def update(self, batch, k_steps=None):
        """
        The MuZero Training Step.
        Unrolls the network for k_steps and calculates value/policy/reward
        loss at each step, exactly following Schrittwieser et al. (2020).

        Reward-target alignment: the dynamics network at unroll step k
        performs the transition (s_k, a_k) -> (s_{k+1}, r_k) -- it consumes
        the action taken at step k and predicts the reward received for
        THAT transition. The correct training target is therefore
        target_rewards[:, k], not target_rewards[:, k+1]. Using k+1 trains
        the reward head to predict next step's reward instead of the
        current transition's reward, which is a systematic misalignment:
        the reward loss will never converge because the head is being
        asked to solve a different (and, from its inputs, unsolvable)
        problem. This is diagnosable by inspecting the reward loss curve
        in isolation -- it plateaus near its initial value while value and
        policy loss decrease normally.
        """
        self.network.train()
        cfg = self.config
        k_steps = k_steps or getattr(cfg, "unroll_steps", 5)

        observations = torch.FloatTensor(np.array(batch['observations'])).to(self.device)
        action_sequence = torch.LongTensor(np.array(batch['actions'])).to(self.device)
        target_values = torch.FloatTensor(np.array(batch['target_values'])).to(self.device)
        target_rewards = torch.FloatTensor(np.array(batch['target_rewards'])).to(self.device)
        target_policies = torch.FloatTensor(np.array(batch['target_policies'])).to(self.device)

        value_w = getattr(cfg, "value_loss_weight", 0.25)
        policy_w = getattr(cfg, "policy_loss_weight", 1.0)
        reward_w = getattr(cfg, "reward_loss_weight", 1.0)
        grad_scale = getattr(cfg, "hidden_state_grad_scale", 0.5)
        clip_norm = getattr(cfg, "grad_clip_norm", 5.0)

        # Step 0: initial representation from the real observation.
        hidden_state = self.network.representation(observations)

        total_loss = 0.0
        # Per-component losses are tracked separately (not just summed into
        # total_loss) so that, e.g., a reward-loss regression like the one
        # described above shows up immediately in logs/tests rather than
        # being invisible inside a single combined scalar.
        loss_components = {"value": 0.0, "policy": 0.0, "reward": 0.0}

        scale = 1.0 / k_steps

        for k in range(k_steps):
            pred_logits, pred_value = self.network.prediction(hidden_state)

            t_val = target_values[:, k].unsqueeze(1)
            t_pol = target_policies[:, k, :]

            loss_value = torch.mean((pred_value - t_val) ** 2)
            loss_policy = torch.sum(
                -t_pol * torch.nn.functional.log_softmax(pred_logits, dim=1), dim=1
            ).mean()

            total_loss = total_loss + scale * (value_w * loss_value + policy_w * loss_policy)
            loss_components["value"] += scale * loss_value.item()
            loss_components["policy"] += scale * loss_policy.item()

            if k < k_steps - 1:
                real_action = action_sequence[:, k].unsqueeze(1)  # (Batch, 1)

                # g(s_k, a_k) -> s_{k+1}, r_k
                hidden_state, pred_reward = self.network.dynamics(hidden_state, real_action)

                # Correct target: the reward for THIS transition (step k),
                # not step k+1.
                t_rew = target_rewards[:, k].unsqueeze(1)
                loss_reward = torch.mean((pred_reward - t_rew) ** 2)

                total_loss = total_loss + scale * reward_w * loss_reward
                loss_components["reward"] += scale * loss_reward.item()

                # Damp gradient flowing back into hidden_state before the
                # next unroll step (see scale_gradient docstring).
                hidden_state = scale_gradient(hidden_state, grad_scale)

        self.network.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=clip_norm)
        self.network.optimizer.step()

        return total_loss.item(), loss_components