"""Historical bar data and dataset for backtesting."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from bot.core.instruments import Instrument


@dataclass(frozen=True, slots=True)
class BarData:
    """A single OHLCV bar for one instrument at a specific timestamp."""

    instrument: Instrument
    timestamp: datetime  # UTC
    open: Decimal  # noqa: A003
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class HistoricalDataset:
    """Immutable collection of OHLCV bars for one or more instruments."""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars: list[BarData] = list(bars)

    @property
    def instruments(self) -> frozenset[Instrument]:
        """Return the set of instruments present in the dataset."""
        return frozenset(b.instrument for b in self._bars)

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Return (earliest_timestamp, latest_timestamp).

        Raises ``ValueError`` if the dataset is empty.
        """
        timestamps = [b.timestamp for b in self._bars]
        return min(timestamps), max(timestamps)

    @property
    def bar_count(self) -> int:
        """Return the total number of bars in the dataset."""
        return len(self._bars)

    def get_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        """Return bars for *instrument* in [start, end], sorted by timestamp."""
        return sorted(
            [b for b in self._bars if b.instrument == instrument and start <= b.timestamp <= end],
            key=lambda b: b.timestamp,
        )

    def get_all_bars_sorted(self) -> list[BarData]:
        """Return all bars sorted by timestamp, then by instrument symbol."""
        return sorted(
            self._bars,
            key=lambda b: (b.timestamp, b.instrument.symbol),
        )

    def split(
        self,
        train_end: datetime,
    ) -> tuple["HistoricalDataset", "HistoricalDataset"]:
        """Return *(in_sample, out_of_sample)* split at *train_end* inclusive.

        No bar appears in both halves.
        """
        in_sample = [b for b in self._bars if b.timestamp <= train_end]
        out_of_sample = [b for b in self._bars if b.timestamp > train_end]
        return HistoricalDataset(in_sample), HistoricalDataset(out_of_sample)
