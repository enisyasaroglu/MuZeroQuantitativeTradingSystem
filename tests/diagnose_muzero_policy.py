"""
Diagnostic v2: distinguishes REPRESENTATION COLLAPSE (hidden states
nearly identical across different real inputs) from MCTS AMPLIFICATION
(a real, input-sensitive raw preference that 150-simulation UCB search
sharpens into near-saturation regardless of exact magnitude). The
previous diagnostic only checked MCTS's OUTPUT after search -- this
checks hidden_state and the raw policy prior BEFORE any search runs.
"""
import os
import pandas as pd
import torch

from configs.muzero_config import MuZeroConfig
from src.env.trading_env import StockTradingEnv
from src.agents.muzero.muzero_agent import MuZeroAgent, load_muzero_checkpoint

df = pd.read_csv(os.path.join("data", "processed", "test_data.csv"))
env = StockTradingEnv(df, use_dsr=False)

cfg = MuZeroConfig()
agent = MuZeroAgent(cfg, env.observation_space.shape)
load_muzero_checkpoint(agent, os.path.join("src", "checkpoints", "muzero_checkpoint_500.pth"))

obs, _ = env.reset()
sample_steps = [0, 30, 60, 90, 120]
samples = []
step = 0
done = False
while not done and step <= max(sample_steps):
    if step in sample_steps:
        samples.append((step, obs.copy(), df['date'].iloc[env.current_step]))
    action, _, _ = agent.select_action(obs, temperature=1.0, add_exploration_noise=False)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1

print(f"{'Step':>5} {'Date':>12} {'hid_mean':>9} {'hid_std':>8} {'raw P(S)':>9} {'raw P(N)':>9} {'raw P(L)':>9} {'raw_val':>8}")
agent.network.eval()
with torch.no_grad():
    for step, sample_obs, date in samples:
        obs_tensor = torch.FloatTensor(sample_obs).unsqueeze(0).to(agent.device)
        hidden_state = agent.network.representation(obs_tensor)
        policy_logits, value = agent.network.prediction(hidden_state)
        raw_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        print(f"{step:>5} {str(date):>12} {hidden_state.mean().item():>9.5f} "
              f"{hidden_state.std().item():>8.5f} {raw_probs[0]:>9.4f} {raw_probs[1]:>9.4f} "
              f"{raw_probs[2]:>9.4f} {value.item():>8.4f}")