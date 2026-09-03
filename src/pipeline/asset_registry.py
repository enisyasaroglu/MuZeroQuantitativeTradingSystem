"""
Asset universe registry: a small, static, explicit list of what this
project fetches and trades. Deliberately NOT a database or a dynamic
membership system (e.g. "current index constituents") -- unnecessary at
this project's scale, and dynamic universes introduce survivorship-bias
questions that don't apply to a small, manually-curated basket.
"""
from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    """
    Metadata label only -- NOT a dispatch key to different fetch logic.
    Yahoo Finance's OHLCV schema is identical across these classes for
    our purposes. The label exists so future asset-class-specific logic
    (e.g. different cost assumptions, a future crypto-specific provider)
    has somewhere to hook in later, without a redesign to add the hook.
    """
    EQUITY = "equity"
    ETF = "etf"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: AssetClass
    provider: str = "yahoo"  # single provider today -- see fetcher.py


UNIVERSE = [
    Asset("^GSPC", "S&P 500 Index", AssetClass.INDEX),
    Asset("SPY", "SPDR S&P 500 ETF", AssetClass.ETF),
    Asset("QQQ", "Invesco QQQ (Nasdaq-100)", AssetClass.ETF),
    Asset("GLD", "SPDR Gold Shares", AssetClass.COMMODITY),
]


def get_universe(asset_class: AssetClass = None) -> list:
    """Full universe, or filtered to one asset class."""
    if asset_class is None:
        return list(UNIVERSE)
    return [a for a in UNIVERSE if a.asset_class == asset_class]


def get_tickers(asset_class: AssetClass = None) -> list:
    return [a.ticker for a in get_universe(asset_class)]