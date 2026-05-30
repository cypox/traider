"""Portfolio state: tracks positions and cash as fills arrive."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from bot.core.instruments import Instrument
from bot.core.money import Money
from bot.core.positions import Position
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.portfolio import FillEvent

_logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Immutable point-in-time view of portfolio holdings.

    Obtained via ``PortfolioState.snapshot()``.  Changes to the mutable
    ``PortfolioState`` after the snapshot is taken do not affect this object.
    """

    positions: tuple[Position, ...]
    cash: Money
    timestamp: datetime

    @property
    def total_position_count(self) -> int:
        """Number of open positions (non-flat instruments)."""
        return len(self.positions)

    @property
    def instruments(self) -> frozenset[Instrument]:
        """Set of instruments currently held."""
        return frozenset(p.instrument for p in self.positions)

    def get_position(self, instrument: Instrument) -> Position | None:
        """Return the position for *instrument*, or ``None`` if flat."""
        for p in self.positions:
            if p.instrument == instrument:
                return p
        return None

    def gross_notional(self, prices: dict[Instrument, Money]) -> Money:
        """Sum of ``abs(quantity) * price`` across all positions."""
        total = Money.zero(self.cash.currency)
        for p in self.positions:
            total = total + abs(p.market_value(prices[p.instrument]))
        return total

    def net_notional(self, prices: dict[Instrument, Money]) -> Money:
        """Sum of signed ``quantity * price`` across all positions."""
        total = Money.zero(self.cash.currency)
        for p in self.positions:
            total = total + p.market_value(prices[p.instrument])
        return total

    def total_market_value(self, prices: dict[Instrument, Money]) -> Money:
        """``net_notional(prices) + cash``."""
        return self.net_notional(prices) + self.cash


class PortfolioState:
    """Mutable portfolio state updated by ``FillEvent`` s from the event bus.

    This is the single source of truth for current positions and cash.
    The ``snapshot()`` method returns an immutable copy safe for consumption
    by downstream components.
    """

    def __init__(self, initial_cash: Money, event_bus: EventBus) -> None:
        self._positions: dict[Instrument, Position] = {}
        self._cash: Money = initial_cash
        self._fill_history: list[FillEvent] = []
        event_bus.subscribe(FillEvent, self._on_fill)

    def _on_fill(self, event: FillEvent) -> None:
        """Process a fill: update position and cash."""
        instrument = event.instrument
        qty = event.filled_quantity
        price = event.fill_price
        side = event.side
        signed_qty = qty if side == Direction.LONG else -qty

        existing = self._positions.get(instrument)

        if existing is None:
            self._positions[instrument] = Position(instrument, signed_qty, price)
        else:
            new_qty = existing.quantity + signed_qty

            if new_qty == Decimal(0):
                del self._positions[instrument]
            elif (new_qty > Decimal(0)) == (existing.quantity > Decimal(0)):
                # Same sign as before
                if (signed_qty > Decimal(0)) == (existing.quantity > Decimal(0)):
                    # Adding to position → weighted-average cost
                    avg_amount = (
                        abs(existing.quantity) * existing.average_price.amount + qty * price.amount
                    ) / abs(new_qty)
                    new_avg = Money(avg_amount, price.currency)
                else:
                    # Reducing position (partial close) → keep existing avg price
                    new_avg = existing.average_price
                self._positions[instrument] = Position(instrument, new_qty, new_avg)
            else:
                # Sign flipped (e.g. long → short) → new avg = fill price
                self._positions[instrument] = Position(instrument, new_qty, price)

        # Update cash: buy debits (price*qty + commission), sell credits (price*qty - commission)
        if side == Direction.LONG:
            self._cash = self._cash - price * qty - event.commission
        else:
            self._cash = self._cash + price * qty - event.commission

        self._fill_history.append(event)

        new_pos = self._positions.get(instrument)
        _logger.debug(
            "fill processed",
            symbol=instrument.symbol,
            fill_price=str(price.amount),
            filled_quantity=str(qty),
            side=side.value,
            new_quantity=str(new_pos.quantity) if new_pos is not None else "0",
            new_cash=str(self._cash.amount),
        )

    def snapshot(self) -> PortfolioSnapshot:
        """Return an immutable point-in-time copy of the current state."""
        return PortfolioSnapshot(
            positions=tuple(self._positions.values()),
            cash=self._cash,
            timestamp=datetime.now(UTC),
        )

    def get_position(self, instrument: Instrument) -> Position | None:
        """Return current position for *instrument*, or ``None`` if flat."""
        return self._positions.get(instrument)

    def get_cash(self) -> Money:
        """Return current cash balance."""
        return self._cash

    def fill_count(self) -> int:
        """Return the number of fills processed so far."""
        return len(self._fill_history)
