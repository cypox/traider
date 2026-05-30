"""Portfolio lifecycle events."""

from dataclasses import dataclass, field
from decimal import Decimal

from bot.core.instruments import Instrument
from bot.core.money import Money
from bot.core.positions import ApprovedPosition, TargetPosition
from bot.core.signals import Direction
from bot.events.base import BaseEvent

# NOTE: super().__post_init__() is broken with frozen+slots dataclass inheritance
# in Python 3.13 (CPython bug). Use BaseEvent.__post_init__(self) explicitly.


@dataclass(frozen=True, slots=True)
class TargetPositionEvent(BaseEvent):
    """Desired position produced by portfolio construction."""

    target: TargetPosition


@dataclass(frozen=True, slots=True)
class ApprovedPositionEvent(BaseEvent):
    """Position approved by the risk engine."""

    approved: ApprovedPosition


@dataclass(frozen=True, slots=True)
class FillEvent(BaseEvent):
    """Execution fill report from the broker."""

    order_id: str
    instrument: Instrument
    filled_quantity: Decimal
    fill_price: Money
    commission: Money
    side: Direction = field(kw_only=True)  # LONG = buy, SHORT = sell

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if self.filled_quantity <= Decimal(0):
            raise ValueError(f"filled_quantity must be positive; got {self.filled_quantity}")
