"""Market data events: quotes, bars, and trades."""

from dataclasses import dataclass
from decimal import Decimal

from bot.core.instruments import Instrument
from bot.core.signals import Direction
from bot.events.base import BaseEvent

# NOTE: super().__post_init__() is broken with frozen+slots dataclass inheritance
# in Python 3.13 (CPython bug). Use BaseEvent.__post_init__(self) explicitly.


@dataclass(frozen=True, slots=True)
class QuoteEvent(BaseEvent):
    """Best bid/ask quote update for an instrument."""

    instrument: Instrument
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if self.bid <= Decimal(0):
            raise ValueError(f"bid must be positive; got {self.bid}")
        if self.ask <= Decimal(0):
            raise ValueError(f"ask must be positive; got {self.ask}")
        if self.bid > self.ask:
            raise ValueError(f"bid must be <= ask; got bid={self.bid}, ask={self.ask}")

    @property
    def mid(self) -> Decimal:
        """Return mid-point price: (bid + ask) / 2."""
        return (self.bid + self.ask) / Decimal(2)


@dataclass(frozen=True, slots=True)
class BarEvent(BaseEvent):
    """OHLCV bar for a specific time interval."""

    instrument: Instrument
    open: Decimal  # noqa: A003
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    interval_seconds: int

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"requires low <= open <= high; "
                f"got low={self.low}, open={self.open}, high={self.high}"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"requires low <= close <= high; "
                f"got low={self.low}, close={self.close}, high={self.high}"
            )
        if self.volume < Decimal(0):
            raise ValueError(f"volume must be >= 0; got {self.volume}")
        if self.interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive; got {self.interval_seconds}")


@dataclass(frozen=True, slots=True)
class TradeEvent(BaseEvent):
    """A single market trade."""

    instrument: Instrument
    price: Decimal
    quantity: Decimal
    aggressor_side: Direction

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if self.price <= Decimal(0):
            raise ValueError(f"price must be positive; got {self.price}")
        if self.quantity <= Decimal(0):
            raise ValueError(f"quantity must be positive; got {self.quantity}")
