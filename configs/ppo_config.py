from dataclasses import dataclass

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4      # Stable learning rate for time-series LSTM
    gamma: float = 0.99              # Standard horizon
    gae_lambda: float = 0.95         # Generalized Advantage Estimation lambda
    
    eps_clip: float = 0.2            # PPO clip ratio
    k_epochs: int = 4                # Epochs per rollout batch
    batch_size: int = 64             # Mini-batch size
    
    value_loss_coef: float = 0.1     # Scaled down to prevent critic overriding policy gradients
    entropy_coef: float = 0.005      # Lower entropy coefficient to reduce action jitter/churn
    max_grad_norm: float = 0.5       # Gradient clipping
    target_kl: float = 0.015         # Target KL divergence limit for early stopping

ppo_config = PPOConfig()