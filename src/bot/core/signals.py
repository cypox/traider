"""Trading signals and forecasts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from bot.core.errors import InvalidForecastError, InvalidSignalError
from bot.core.instruments import Instrument


class Direction(Enum):
    """The directional bias of a signal or position."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True, slots=True)
class Signal:
    """A directional trading signal emitted by a strategy rule."""

    instrument: Instrument
    timestamp: datetime
    direction: Direction
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not (0.0 < self.confidence <= 1.0):
            raise InvalidSignalError(f"confidence must be in (0.0, 1.0]; got {self.confidence}")
        if self.direction == Direction.FLAT:
            raise InvalidSignalError("Strategies must not emit FLAT direction signals")
        if not self.reason:
            raise InvalidSignalError("Signal reason must be non-empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise InvalidSignalError("timestamp must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class Forecast:
    """A normalised combined signal value in the range [-1.0, 1.0].

    Negative values indicate short bias; positive values indicate long bias.
    Zero is neutral. The magnitude represents conviction strength.
    """

    instrument: Instrument
    timestamp: datetime
    value: float
    source: str

    def __post_init__(self) -> None:
        if abs(self.value) > 1.0:
            raise InvalidForecastError(f"Forecast value must be in [-1.0, 1.0]; got {self.value}")
