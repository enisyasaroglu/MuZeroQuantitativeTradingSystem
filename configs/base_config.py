from dataclasses import dataclass

@dataclass
class ProjectConfig:
    # Data Settings
    TICKER: str = "SPY"     # S&P 500
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2023-01-01"

    # Chronological three-way split. TRAIN_SPLIT + VAL_SPLIT must be < 1.0;
    # the remainder is TEST. EMBARGO_DAYS rows are dropped at each split
    # boundary (matches LOOKBACK_WINDOW) so that the first observation in
    # validation/test cannot contain any feature computed from a rolling
    # window that crosses back into the previous split's price data.
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.15   
    TEST_SPLIT: float = 0.15

    # Environment Settings
    INITIAL_CAPITAL: float = 100000.0
    TRANSACTION_FEE: float = 0.001  # 0.1% per trade
    LOOKBACK_WINDOW: int = 60       # Agent sees past 60 days

    # Reward shaping: Differential Sharpe Ratio (Moody & Wu, 1997).
    # Set USE_DSR_REWARD = False to fall back to raw net log-return, e.g.
    # for baseline/ablation comparisons.
    USE_DSR_REWARD: bool = True
    DSR_ETA: float = 0.01
    DSR_WARMUP_STEPS: int = 5
    DSR_CLIP: float = 10.0
    
    # Risk halt inside the env. 1.0 = disabled (the halt cannot trigger).
    MAX_DRAWDOWN: float = 1.0
    MAX_DAILY_LOSS: float = 1.0

    # Feature Engineering
    # stockstats naming convention:
    # 'macd'    : Moving Average Convergence Divergence
    # 'rsi_14'  : Relative Strength Index (14 days) - short-term
    # 'rsi_30'  : Relative Strength Index (30 days) - medium-term
    # 'cci_14'  : Commodity Channel Index (14 days)
    # 'dx_30'   : Directional Movement Index (30 days)
    # 'atr_30'  : Average True Range (30 days) - volatility
    # 'boll_ub' : Bollinger Band, upper
    # 'boll_lb' : Bollinger Band, lower
    TECH_INDICATORS: tuple = (
        "macd", "rsi_14", "rsi_30", "cci_14", "dx_30", "atr_30", "boll_ub", "boll_lb"
    )
    
    # Reproducibility
    SEED: int = 42
    
    def __post_init__(self):
        if not 0.0 < self.TRAIN_SPLIT + self.VAL_SPLIT < 1.0:
            raise ValueError("TRAIN_SPLIT + VAL_SPLIT must be strictly between 0 and 1")

    @property
    def EMBARGO_DAYS(self) -> int:
        # Must equal the lookback so no rolling window crosses a split.
        return self.LOOKBACK_WINDOW

config = ProjectConfig()