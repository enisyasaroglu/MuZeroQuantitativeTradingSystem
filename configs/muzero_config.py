from dataclasses import dataclass


@dataclass(frozen=True)
class MuZeroConfig:
    """
    Configuration for the MuZero quantitative trading agent.

    All experiment/training parameters should live here rather than
    being hard-coded inside the training script.
    """

    # Reproducibility
    seed: int = 42

    # Network architecture
    latent_state_dim: int = 64
    hidden_size: int = 64
    action_space_dim: int = 3

    # Optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    batch_size: int = 64

    # Discounting
    discount_factor: float = 0.99

    # Loss weights
    value_loss_weight: float = 0.25
    policy_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    entropy_loss_weight: float = 0.05

    # MuZero unroll / target construction
    unroll_steps: int = 5
    td_steps: int = 5

    # Maximum number of environment steps in one training episode.
    episode_length: int = 252

    # Replay buffer
    replay_buffer_capacity: int = 2000

    # Number of games/trajectories that must exist before learning starts.
    min_buffer_size: int = 10

    # Training
    num_episodes: int = 500
    updates_per_episode: int = 10

    # Gradient control
    hidden_state_grad_scale: float = 0.5
    grad_clip_norm: float = 5.0

    # MCTS
    num_simulations: int = 150

    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    pb_c_base: int = 19652
    pb_c_init: float = 1.25

    # Checkpointing
    checkpoint_every: int = 5


muzero_config = MuZeroConfig()