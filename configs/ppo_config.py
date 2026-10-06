from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    # Optimisation
    learning_rate: float = 3e-4
    batch_size: int = 64
    k_epochs: int = 4

    # Discounting and advantage estimation
    gamma: float = 0.99             # keep equal to MuZeroConfig.discount_factor
    gae_lambda: float = 0.95

    # Policy optimisation
    eps_clip: float = 0.2
    target_kl: float = 0.015

    # Loss coefficients
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.005

    # Gradient clipping
    max_grad_norm: float = 0.5


ppo_config = PPOConfig()