"""Risk engine: validates and scales target positions before execution."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from bot.core.instruments import Instrument
from bot.core.money import Money
from bot.core.positions import ApprovedPosition
from bot.events.bus import EventBus
from bot.events.diagnostics import DiagnosticsEvent
from bot.events.market import QuoteEvent
from bot.events.portfolio import ApprovedPositionEvent, FillEvent, TargetPositionEvent
from bot.portfolio.state import PortfolioSnapshot, PortfolioState
from bot.risk.config import RiskConfig

_logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RiskState:
    """Immutable snapshot of the risk engine's current monitoring state."""

    peak_equity: Money
    current_equity: Money
    daily_start_equity: Money
    is_halted: bool
    halt_reason: str


class RiskEngine:
    """Validates and scales ``TargetPosition`` s before they reach execution.

    Subscribes to ``TargetPositionEvent``, ``FillEvent``, and ``QuoteEvent``
    on the provided ``EventBus``.  When a target passes all checks it publishes
    an ``ApprovedPositionEvent``; otherwise it logs and silently drops the
    target — never raising exceptions.

    Safety principle: when in doubt, reject rather than approve.
    """

    def __init__(
        self,
        config: RiskConfig,
        event_bus: EventBus,
        portfolio_state: PortfolioState,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._portfolio_state = portfolio_state
        initial_cash = portfolio_state.get_cash()
        self._peak_equity: Money = initial_cash
        self._daily_start_equity: Money = initial_cash
        self._is_halted: bool = False
        self._halt_reason: str = ""
        self._price_cache: dict[Instrument, Money] = {}
        event_bus.subscribe(TargetPositionEvent, self._on_target_position)
        event_bus.subscribe(FillEvent, self._on_fill)
        event_bus.subscribe(QuoteEvent, self._on_quote)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_equity(self) -> Money:
        """Return cash + mark-to-market of all positions with known prices."""
        snapshot = self._portfolio_state.snapshot()
        equity = snapshot.cash
        for pos in snapshot.positions:
            if pos.instrument in self._price_cache:
                price = self._price_cache[pos.instrument]
                equity = equity + price * pos.quantity
        return equity

    def _scale_quantity(self, quantity: Decimal, scale: Decimal) -> Decimal:
        """Scale *quantity* by *scale*, round to integer, preserve sign."""
        sign = Decimal(1) if quantity > Decimal(0) else Decimal(-1)
        return sign * (abs(quantity) * scale).to_integral_value()

    def _compute_gross_exposure(self, snapshot: PortfolioSnapshot) -> Decimal:
        """Sum ``abs(qty * price)`` for all positions with a cached price."""
        total = Decimal(0)
        for pos in snapshot.positions:
            if pos.instrument in self._price_cache:
                price = self._price_cache[pos.instrument]
                total += abs(pos.quantity) * price.amount
        return total

    def _get_existing_notional(
        self,
        snapshot: PortfolioSnapshot,
        instrument: Instrument,
        price: Money,
    ) -> Decimal:
        """Return ``abs(qty * price)`` for an existing position, or zero."""
        pos = snapshot.get_position(instrument)
        if pos is None:
            return Decimal(0)
        return abs(pos.quantity) * price.amount

    def _halt(self, reason: str) -> None:
        """Halt the engine, log at CRITICAL, and publish a DiagnosticsEvent."""
        self._is_halted = True
        self._halt_reason = reason
        _logger.critical("RISK ENGINE HALT", reason=reason)
        self._event_bus.publish(
            DiagnosticsEvent(
                timestamp=datetime.now(UTC),
                source="risk_engine",
                payload={"is_halted": True, "halt_reason": reason},
            )
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_quote(self, event: QuoteEvent) -> None:
        """Cache the mid-price for the quoted instrument."""
        self._price_cache[event.instrument] = Money(event.mid, event.instrument.currency)

    def _on_fill(self, event: FillEvent) -> None:
        """Recompute equity after each fill and check halt conditions."""
        if self._is_halted:
            return

        current_equity = self._compute_equity()

        # Update peak equity if this is a new high-water mark
        if current_equity.amount > self._peak_equity.amount:
            self._peak_equity = current_equity

        # Check daily loss limit
        daily_loss = self._daily_start_equity.amount - current_equity.amount
        if daily_loss > self._config.daily_loss_limit_usd:
            self._halt("daily loss limit exceeded")
            return

        # Check maximum drawdown from peak
        if self._peak_equity.amount > Decimal(0):
            drawdown = (self._peak_equity.amount - current_equity.amount) / self._peak_equity.amount
            if drawdown > self._config.max_drawdown_pct:
                self._halt("max drawdown exceeded")

    def _on_target_position(self, event: TargetPositionEvent) -> None:
        """Validate, scale if needed, and publish an ApprovedPositionEvent."""
        if self._is_halted:
            _logger.warning(
                "risk engine halted, rejecting target position",
                symbol=event.target.instrument.symbol,
            )
            return

        target = event.target
        instrument = target.instrument

        if instrument not in self._price_cache:
            _logger.warning(
                "no price for instrument, rejecting target",
                symbol=instrument.symbol,
            )
            return

        current_price = self._price_cache[instrument]
        target_notional = abs(target.target_quantity) * current_price.amount

        # Check 1: single-position notional limit
        note_parts: list[str] = []
        if target_notional > self._config.max_position_usd:
            scale = self._config.max_position_usd / target_notional
            approved_quantity = self._scale_quantity(target.target_quantity, scale)
            note_parts.append(f"scaled down: position limit {self._config.max_position_usd}")
        else:
            approved_quantity = target.target_quantity

        # Recompute notional after Check 1
        new_instrument_notional = abs(approved_quantity) * current_price.amount

        # Check 2: portfolio gross exposure
        snapshot = self._portfolio_state.snapshot()
        current_gross = self._compute_gross_exposure(snapshot)
        existing_notional = self._get_existing_notional(snapshot, instrument, current_price)
        new_gross = current_gross - existing_notional + new_instrument_notional

        if new_gross > self._config.max_gross_exposure_usd:
            headroom = self._config.max_gross_exposure_usd - (current_gross - existing_notional)
            if headroom <= Decimal(0):
                _logger.warning(
                    "no gross exposure headroom, rejecting target",
                    symbol=instrument.symbol,
                )
                return
            approved_quantity = self._scale_quantity(
                approved_quantity, headroom / new_instrument_notional
            )
            note_parts.append("gross exposure limit applied")

        # Recompute notional and gross after Check 2
        new_instrument_notional = abs(approved_quantity) * current_price.amount
        new_gross = current_gross - existing_notional + new_instrument_notional

        # Check 3: single-instrument concentration
        if new_gross > Decimal(0):
            concentration = new_instrument_notional / new_gross
            if concentration > self._config.max_concentration_pct:
                max_notional = self._config.max_concentration_pct * new_gross
                approved_quantity = self._scale_quantity(
                    approved_quantity, max_notional / new_instrument_notional
                )
                note_parts.append("concentration limit applied")

        # Reject if all checks reduce the quantity to zero
        if approved_quantity == Decimal(0):
            _logger.warning(
                "approved quantity is zero after scaling, rejecting",
                symbol=instrument.symbol,
            )
            return

        notes = " | ".join(note_parts) if note_parts else "approved as-is"
        approved = ApprovedPosition(
            instrument=instrument,
            approved_quantity=approved_quantity,
            original_target=target,
            risk_notes=notes,
        )
        self._event_bus.publish(
            ApprovedPositionEvent(
                timestamp=datetime.now(UTC),
                source="risk_engine",
                approved=approved,
            )
        )
        _logger.debug(
            "position approved",
            symbol=instrument.symbol,
            approved_qty=str(approved_quantity),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_state(self) -> RiskState:
        """Return an immutable snapshot of the current risk engine state."""
        current_equity = self._compute_equity()
        return RiskState(
            peak_equity=self._peak_equity,
            current_equity=current_equity,
            daily_start_equity=self._daily_start_equity,
            is_halted=self._is_halted,
            halt_reason=self._halt_reason,
        )

    def reset_daily(self) -> None:
        """Update the daily-start equity to the current equity level.

        Call this at the start of each trading day to reset the daily-loss
        reference point.
        """
        self._daily_start_equity = self._compute_equity()
