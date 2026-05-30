"""Execution engine: translates approved positions into broker orders."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from bot.core.execution import ExecutionIntent
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.orders import OrderEvent
from bot.events.portfolio import ApprovedPositionEvent
from bot.portfolio.state import PortfolioState
from bot.providers.base import ExecutionProvider

_logger = structlog.get_logger(__name__)


class ExecutionEngine:
    """Translates :class:`ApprovedPositionEvent` messages into broker orders.

    On construction, subscribes to :class:`ApprovedPositionEvent` on the
    provided event bus.  For each approval, computes the delta vs. the current
    position, creates an :class:`ExecutionIntent`, and submits it
    asynchronously via the :class:`ExecutionProvider`.
    """

    def __init__(
        self,
        event_bus: EventBus,
        execution_provider: ExecutionProvider,
        portfolio_state: PortfolioState,
    ) -> None:
        self._event_bus = event_bus
        self._execution_provider = execution_provider
        self._portfolio_state = portfolio_state
        event_bus.subscribe(ApprovedPositionEvent, self._on_approved)

    def _on_approved(self, event: ApprovedPositionEvent) -> None:
        """Handle an approved position: compute delta and schedule async order."""
        approved = event.approved

        current_position = self._portfolio_state.get_position(approved.instrument)
        current_qty = current_position.quantity if current_position else Decimal("0")
        delta_qty = approved.approved_quantity - current_qty

        if abs(delta_qty) < Decimal("1"):
            _logger.debug(
                "no trade needed, delta < 1 share",
                instrument=str(approved.instrument),
                delta=str(delta_qty),
            )
            return

        side = Direction.LONG if delta_qty > Decimal("0") else Direction.SHORT
        reason = approved.risk_notes if approved.risk_notes else "approved by risk engine"
        intent = ExecutionIntent(
            instrument=approved.instrument,
            side=side,
            quantity=abs(delta_qty).quantize(Decimal("1")),
            reason=reason,
            source_approved=approved,
        )

        _logger.info(
            "placing order",
            instrument=approved.instrument.symbol,
            side=side.value,
            quantity=str(intent.quantity),
            reason=intent.reason,
        )

        asyncio.create_task(self._submit_order(intent))  # noqa: RUF006

    async def _submit_order(self, intent: ExecutionIntent) -> None:
        """Submit the order to the execution provider and publish an :class:`OrderEvent`."""
        order_id = await self._execution_provider.place_order(intent)
        self._event_bus.publish(
            OrderEvent(
                intent=intent,
                order_id=order_id,
                timestamp=datetime.now(UTC),
                source="execution-engine",
            )
        )
