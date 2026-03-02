from dataclasses import dataclass

@dataclass
class MuZeroConfig:
    # 1. Environment & Data Settings
    stock_symbol: str = "SPY"       # S&P 500 ETF or ^GSPC
    window_size: int = 60           # Lookback window T=60 
    start_date: str = "2015-01-01"
    end_date: str = "2021-01-01"    # Training Data
    test_start_date: str = "2021-01-02" # Out of sample [cite: 298]
    
    # Market Mechanics
    transaction_fee: float = 0.001  # 0.1% per trade 
    initial_capital: float = 100000.0

    # 2. Network Architecture
    latent_state_dim: int = 64      # Size of hidden state s_t
    action_space_dim: int = 3       # 0=Short, 1=Neutral, 2=Long [cite: 193]
    hidden_size: int = 64           # Neurons in MLP layers
    
    # 3. Training Hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 64
    discount_factor: float = 0.99 
    weight_decay: float = 1e-4      # L2 Regularization [cite: 292]

    # 4. MCTS Specifics (The "Brain" Settings)
    num_simulations: int = 50       # Reduced from 800 to 50 for speed 
    
    # Root Noise (Exploration)
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    
    # UCB Score Formula Constants (Standard MuZero values)
    pb_c_base: int = 19652
    pb_c_init: float = 1.25