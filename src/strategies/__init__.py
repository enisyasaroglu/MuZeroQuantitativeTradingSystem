"""
Public interface for src.strategies.

Two kinds of strategy live here:

* BuyAndHoldStrategy has a select_action(observation) method, like the
  learning agents.
* The rule-based strategies (SMAStrategy is older and has its own
  interface; MomentumStrategy, MeanReversionStrategy and
  TrendFollowingStrategy) derive from SignalStrategy. They turn a series
  of log returns into environment actions with a one-day lag, so they
  cannot use the return they are about to earn.

SMAStrategy.generate_signals does not apply that lag. Use it only if you
shift its signal by one row yourself, as evaluate.py does.
"""

from .base_strategy import SignalStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .sma import SMAStrategy
from .trend_following import TrendFollowingStrategy

__all__ = [
    "SignalStrategy",
    "BuyAndHoldStrategy",
    "SMAStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
]