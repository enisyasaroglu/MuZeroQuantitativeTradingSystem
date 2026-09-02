from abc import ABC, abstractmethod


class MarketDataSource(ABC):

    @abstractmethod
    def fetch(self, symbols, start, end):
        pass
    
class AlpacaSource(MarketDataSource):

    def fetch(self, symbols, start, end):
        # Alpaca-specific implementation
        DATA_SOURCE = "alpaca"
        DATA_SOURCE = "binance"
        DATA_SOURCE = "ibkr"
        

class BinanceSource(MarketDataSource):

    def fetch(self, symbols, start, end):
        # Binance-specific implementation
        ...

class IBKRSource(MarketDataSource):

    def fetch(self, symbols, start, end):
        # IBKR-specific implementation
        ...