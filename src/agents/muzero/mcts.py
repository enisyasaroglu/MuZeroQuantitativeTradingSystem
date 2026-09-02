import math
import numpy as np
import torch

class MinMaxStats:
    """
    A utility to normalize Q-values to the range [0, 1].
    Crucial for the UCB formula to work correctly when rewards are unbounded.
    """
    def __init__(self, known_bounds=None):
        self.maximum = known_bounds.max if known_bounds else -float('inf')
        self.minimum = known_bounds.min if known_bounds else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    """
    A node in the search tree. 
    In Open Loop MCTS, this represents a SEQUENCE of actions, not a board state.
    """
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None  # The latent state s_k
        self.reward = 0           # The immediate reward r_k received to get here
        self.prior = prior        # The policy probability P(a)
        self.is_expanded = False

    def value(self):
        """Returns the mean Q-value."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def run_mcts(config, root_state, network, min_max_stats):
    """
    The core Open Loop MCTS loop.
    """
    root = Node(0)
    root.hidden_state = root_state
    root.is_expanded = True
    
    # --- 1. Expand Root ---
    # Get Policy and Value for the current real observation
    policy_logits, value = network.prediction(root_state)
    
    # Softmax to get probabilities
    policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
    
    # Add Dirichlet Noise (Exploration at the root)
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_dim)
    frac = config.root_exploration_fraction
    
    for action_id in range(config.action_space_dim):
        noisy_prior = policy[action_id] * (1 - frac) + noise[action_id] * frac
        root.children[action_id] = Node(noisy_prior)

    # --- 2. Simulation Loop ---
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        actions_taken = []
        
        # A. Selection (Traverse down using UCB)
        while node.is_expanded:
            action, node = select_child(config, node, min_max_stats)
            search_path.append(node)
            actions_taken.append(action)

        # B. Expansion (We reached a leaf)
        parent = search_path[-2]
        action_id = actions_taken[-1]
        
        # Use Dynamics Network to IMAGINE what happens next
        # g(s, a) -> next_s, reward
        action_tensor = torch.tensor([[action_id]]).to(network.device)
        next_state, reward = network.dynamics(parent.hidden_state, action_tensor)
        
        node.hidden_state = next_state
        node.reward = reward.item()
        
        # Predict value of this imagined state
        policy_logits, value = network.prediction(next_state)
        
        # Create children for this new node
        node.is_expanded = True
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        for i in range(config.action_space_dim):
            node.children[i] = Node(policy[i])
            
        # C. Backpropagation
        backpropagate(search_path, value.item(), config.discount_factor, min_max_stats)

    return root

def select_child(config, node, min_max_stats):
    """Selects the child with the highest UCB score."""
    max_score = -float('inf')
    best_action = -1
    best_child = None

    for action, child in node.children.items():
        score = ucb_score(config, node, child, min_max_stats)
        if score > max_score:
            max_score = score
            best_action = action
            best_child = child
            
    return best_action, best_child

def ucb_score(config, parent, child, min_max_stats):
    """The AlphaZero UCB Formula."""
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    
    return prior_score + value_score

def backpropagate(search_path, value, discount, min_max_stats):
    """Updates node values up the tree."""
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        
        # Value for parent = Reward + Discount * Value
        value = node.reward + discount * value