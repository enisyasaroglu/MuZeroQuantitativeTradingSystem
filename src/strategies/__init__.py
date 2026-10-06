"""
Public interface for src.strategies.

Both strategies share the select_action(observation, deterministic=...)
-> (action, info) convention, matching MuZeroAgent/PPOAgent's calling
convention -- see each class's own docstring.
"""

from .buy_and_hold import BuyAndHoldStrategy
from .sma import SMAStrategy

__all__ = ["BuyAndHoldStrategy", "SMAStrategy"]