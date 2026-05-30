"""Position-related domain models."""

from dataclasses import dataclass
from decimal import Decimal

from bot.core.instruments import Instrument
from bot.core.money import Money
from bot.core.signals import Direction, Forecast


@dataclass(frozen=True, slots=True)
class Position:
    """Current holding in a single instrument."""

    instrument: Instrument
    quantity: Decimal
    average_price: Money

    @property
    def direction(self) -> Direction:
        """Return the directional bias implied by quantity sign."""
        if self.quantity > Decimal(0):
            return Direction.LONG
        if self.quantity < Decimal(0):
            return Direction.SHORT
        return Direction.FLAT

    @property
    def is_flat(self) -> bool:
        """Return True when quantity is zero."""
        return self.quantity == Decimal(0)

    def market_value(self, current_price: Money) -> Money:
        """Return current_price * quantity."""
        return current_price * self.quantity

    def unrealized_pnl(self, current_price: Money) -> Money:
        """Return (current_price - average_price) * quantity."""
        return (current_price - self.average_price) * self.quantity


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Desired position produced by portfolio construction."""

    instrument: Instrument
    target_quantity: Decimal
    target_notional: Money
    source_forecast: Forecast


@dataclass(frozen=True, slots=True)
class ApprovedPosition:
    """Position approved by the risk engine after reviewing the target."""

    instrument: Instrument
    approved_quantity: Decimal
    original_target: TargetPosition
    risk_notes: str
