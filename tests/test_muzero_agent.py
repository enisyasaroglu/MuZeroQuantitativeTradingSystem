import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from configs.muzero_config import MuZeroConfig
from src.agents.muzero_agent import MuZeroAgent

def test_inference():
    # 1. Setup
    config = MuZeroConfig()
    obs_shape = (config.window_size, 11) # (60, 11)
    
    print("Initializing MuZero Agent...")
    agent = MuZeroAgent(config, obs_shape)
    
    # 2. Fake Data (Random Market Data)
    dummy_obs = np.random.randn(*obs_shape).astype(np.float32)
    
    print("Running MCTS Inference (This calls Rep, Dyn, Pred, and MCTS)...")
    action, probs, value = agent.select_action(dummy_obs)
    
    # 3. Validation
    print(f"\nAction Selected: {action} (Type: {type(action)})")
    print(f"Policy Probs: {probs}")
    print(f"Root Value: {value:.4f}")
    
    # Logic Checks
    assert action in [0, 1, 2]
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-5

if __name__ == "__main__":
    test_inference()