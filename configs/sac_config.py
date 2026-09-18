from dataclasses import dataclass

@dataclass
class SACConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005             # Soft target update coefficient
    alpha: float = 0.2             # Initial temperature parameter for entropy
    auto_tune_alpha: bool = True   # Automatically adjust entropy target during training
    
    buffer_size: int = 100000
    batch_size: int = 64
    warmup_steps: int = 1000       # Random steps before training starts

sac_config = SACConfig()