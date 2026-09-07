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
    Asset("AAPL", "Apple Inc.", AssetClass.EQUITY),
    Asset("MSFT", "Microsoft Corporation", AssetClass.EQUITY),
    Asset("AMZN", "Amazon.com, Inc.", AssetClass.EQUITY),
    Asset("GOOGL", "Alphabet Inc. (Class A)", AssetClass.EQUITY),
    Asset("TSLA", "Tesla, Inc.", AssetClass.EQUITY),
    Asset("BRK-B", "Berkshire Hathaway Inc. (Class B)", AssetClass.EQUITY),
    Asset("NVDA", "NVIDIA Corporation", AssetClass.EQUITY),
    Asset("JPM", "JPMorgan Chase & Co.", AssetClass.EQUITY),
    Asset("V", "Visa Inc.", AssetClass.EQUITY),
    Asset("JNJ", "Johnson & Johnson", AssetClass.EQUITY),
    Asset("WMT", "Walmart Inc.", AssetClass.EQUITY),
    Asset("PG", "Procter & Gamble Company", AssetClass.EQUITY),
    Asset("DIS", "The Walt Disney Company", AssetClass.EQUITY),
    Asset("MA", "Mastercard Incorporated", AssetClass.EQUITY),
    Asset("HD", "The Home Depot, Inc.", AssetClass.EQUITY),
    Asset("BAC", "Bank of America Corporation", AssetClass.EQUITY),
    Asset("XOM", "Exxon Mobil Corporation", AssetClass.EQUITY),
    Asset("VZ", "Verizon Communications Inc.", AssetClass.EQUITY),
    Asset("KO", "The Coca-Cola Company", AssetClass.EQUITY),
    Asset("PFE", "Pfizer Inc.", AssetClass.EQUITY),
    Asset("MRK", "Merck & Co., Inc.", AssetClass.EQUITY),
    Asset("INTC", "Intel Corporation", AssetClass.EQUITY),
    Asset("CSCO", "Cisco Systems, Inc.", AssetClass.EQUITY),
    Asset("ORCL", "Oracle Corporation", AssetClass.EQUITY),
    Asset("PEP", "PepsiCo, Inc.", AssetClass.EQUITY),
    Asset("T", "AT&T Inc.", AssetClass.EQUITY),
    Asset("CVX", "Chevron Corporation", AssetClass.EQUITY),
    Asset("WFC", "Wells Fargo & Company", AssetClass.EQUITY),
    Asset("C", "Citigroup Inc.", AssetClass.EQUITY),
    Asset("BA", "The Boeing Company", AssetClass.EQUITY),
    Asset("MCD", "McDonald's Corporation", AssetClass.EQUITY),
    Asset("NKE", "NIKE, Inc.", AssetClass.EQUITY),
    Asset("IBM", "International Business Machines Corporation", AssetClass.EQUITY),
    Asset("GE", "General Electric Company", AssetClass.EQUITY),
    Asset("GM", "General Motors Company", AssetClass.EQUITY),
    Asset("F", "Ford Motor Company", AssetClass.EQUITY),
    Asset("CAT", "Caterpillar Inc.", AssetClass.EQUITY),
    Asset("MMM", "3M Company", AssetClass.EQUITY),
    Asset("RTX", "Raytheon Technologies Corporation", AssetClass.EQUITY),
    Asset("LMT", "Lockheed Martin Corporation", AssetClass.EQUITY),
    Asset("CVS", "CVS Health Corporation", AssetClass.EQUITY),
    Asset("UNH", "UnitedHealth Group Incorporated", AssetClass.EQUITY),
    Asset("AMGN", "Amgen Inc.", AssetClass.EQUITY),
    Asset("GILD", "Gilead Sciences, Inc.", AssetClass.EQUITY),
    Asset("BIIB", "Biogen Inc.", AssetClass.EQUITY),
    Asset("REGN", "Regeneron Pharmaceuticals, Inc.", AssetClass.EQUITY),
    Asset("VRTX", "Vertex Pharmaceuticals Incorporated", AssetClass.EQUITY),
    Asset("TMO", "Thermo Fisher Scientific Inc.", AssetClass.EQUITY),
    Asset("ABT", "Abbott Laboratories", AssetClass.EQUITY),
    Asset("BABA", "Alibaba Group Holding Limited", AssetClass.EQUITY),
    Asset("JD", "JD.com, Inc.", AssetClass.EQUITY),
    Asset("BIDU", "Baidu, Inc.", AssetClass.EQUITY),
    Asset("TCEHY", "Tencent Holdings Limited", AssetClass.EQUITY),
    Asset("NTES", "NetEase, Inc.", AssetClass.EQUITY),
]


def get_universe(asset_class: AssetClass = None) -> list:
    """Full universe, or filtered to one asset class."""
    if asset_class is None:
        return list(UNIVERSE)
    return [a for a in UNIVERSE if a.asset_class == asset_class]


def get_tickers(asset_class: AssetClass = None) -> list:
    return [a.ticker for a in get_universe(asset_class)]