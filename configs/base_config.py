from dataclasses import dataclass

@dataclass
class ProjectConfig:
    # Data Settings
    TICKER: str = "^GSPC"     # S&P 500
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2023-01-01"
    TRAIN_SPLIT: float = 0.8  # First 80% for training
    
    # Environment Settings
    INITIAL_CAPITAL: float = 100000.0
    TRANSACTION_FEE: float = 0.001  # 0.1% per trade
    LOOKBACK_WINDOW: int = 60       # Agent sees past 60 days
    
    # Feature Engineering
    TECH_INDICATORS: tuple = (
        "macd", "rsi_30", "cci_30", "dx_30"
    )

config = ProjectConfig()