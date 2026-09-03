"""
PLANNING / INTERFACE-LEVEL ONLY. No alternative data source is
implemented yet. This package exists so a future implementation has an
agreed shape to fit into, rather than being bolted on ad hoc.

Four categories anticipated: fundamental, macro, news_sentiment, filings.
Each, when built, implements AlternativeDataSource below and lands in
data/raw/alternative/<category>/ and data/processed/alternative/<category>/,
mirroring the existing data/raw/, data/processed/ convention.

WHY AN INTERFACE NOW WITH ZERO IMPLEMENTATIONS: the one design decision
that's expensive to retrofit is how alternative data joins onto the
price/return timeline without look-ahead bias -- e.g. an earnings report
must be joined on its PUBLICATION date, not its fiscal-period-end date.
Fixing that contract now means it's specified once here, not separately
(and inconsistently) per future data source.
"""
from abc import ABC, abstractmethod
import pandas as pd


class AlternativeDataSource(ABC):
    """Minimal contract -- no shared base-class logic, no registration
    system. A real implementation is free to add whatever internal
    structure it needs; this only fixes what the rest of the pipeline
    can rely on."""

    @abstractmethod
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Must return a DataFrame with at minimum:
          - 'publication_date': when this data point became PUBLICLY
            KNOWN (not the period it describes) -- what DataProcessor
            would join on, to prevent look-ahead bias, matching the
            embargo/split-order discipline already used for price data.
          - one or more value columns, source-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def source_name(self) -> str:
        """Short identifier, e.g. 'sec_filings' -- used in cache
        filenames and logging, mirroring DataFetcher's convention."""
        raise NotImplementedError


PLANNED_CATEGORIES = ("fundamental", "macro", "news_sentiment", "filings")