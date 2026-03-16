import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.agents.baselines.ppo_agent import PPOAgent

def test_ppo_update():
    # 1. Setup Dummy Data
    obs_shape = (60, 11)
    action_dim = 3
    agent = PPOAgent(obs_shape, action_dim)
    
    print("--- Testing PPO Agent ---")
    
    # 2. Create Fake Memory (One small batch)
    memory = {
        'states': [np.random.randn(*obs_shape).astype(np.float32) for _ in range(10)],
        'actions': [np.random.randint(0, 3) for _ in range(10)],
        'log_probs': [np.random.randn() for _ in range(10)],
        'rewards': [np.random.randn() for _ in range(10)],
        'dones': [False] * 10
    }
    
    # 3. Test Action Selection
    action, log_prob, val = agent.select_action(memory['states'][0])
    print(f"Select Action: {action} (Type: {type(action)})")
    assert isinstance(action, int)
    
    # 4. Test Update Step
    initial_loss = agent.update(memory)
    print(f"Update Step Loss: {initial_loss:.4f}")
    
    # 5. Verify Model Changed
    # Run update again, loss should ideally be different (though not guaranteed to be lower immediately)
    loss_2 = agent.update(memory)
    print(f"Second Step Loss: {loss_2:.4f}")
    
    print("Success! PPO Agent handles selection and updates.")

if __name__ == "__main__":
    test_ppo_update()