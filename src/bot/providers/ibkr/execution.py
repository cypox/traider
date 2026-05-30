"""IBKR execution provider: submits orders and converts fills to FillEvents."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog
from ib_async import Fill, MarketOrder, Trade

from bot.core.execution import ExecutionIntent
from bot.core.money import Money
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.portfolio import FillEvent
from bot.providers.base import ExecutionProvider
from bot.providers.errors import ConnectionError  # noqa: A004
from bot.providers.ibkr.utils import contract_to_instrument, instrument_to_contract

if TYPE_CHECKING:
    from bot.providers.ibkr.market_data import IBKRMarketDataProvider

_logger = structlog.get_logger(__name__)


class IBKRExecutionProvider(ExecutionProvider):
    """Submits orders to IBKR and converts fill callbacks into ``FillEvent`` objects.

    The underlying IB connection is owned by ``IBKRMarketDataProvider`` and
    accessed via its ``ib`` property — this provider never creates its own IB.
    """

    def __init__(
        self,
        market_data_provider: IBKRMarketDataProvider,
        event_bus: EventBus,
    ) -> None:
        self._market_data_provider = market_data_provider
        self._event_bus = event_bus
        self._is_connected: bool = False
        self._open_orders: dict[str, Trade] = {}

    async def connect(self) -> None:
        """Register the fill callback on the already-open IB connection."""
        self._market_data_provider.ib.execDetailsEvent += self._on_exec_details
        self._is_connected = True
        _logger.info("ibkr_execution_provider_ready")

    async def disconnect(self) -> None:
        """Deregister the fill callback."""
        self._market_data_provider.ib.execDetailsEvent -= self._on_exec_details
        self._is_connected = False
        _logger.info("ibkr_execution_provider_disconnected")

    async def place_order(self, intent: ExecutionIntent) -> str:
        """Submit a market order to IBKR and return the broker order ID."""
        if not self._is_connected:
            raise ConnectionError("IBKRExecutionProvider not connected")

        contract = instrument_to_contract(intent.instrument)
        action = "BUY" if intent.side == Direction.LONG else "SELL"

        order = MarketOrder(
            action=action,
            totalQuantity=float(intent.quantity),
            transmit=True,
        )

        trade = self._market_data_provider.ib.placeOrder(contract, order)
        ibkr_order_id = str(trade.order.orderId)
        self._open_orders[ibkr_order_id] = trade

        _logger.info(
            "order_placed",
            symbol=intent.instrument.symbol,
            side=intent.side.name,
            quantity=str(intent.quantity),
            ibkr_order_id=ibkr_order_id,
        )

        return ibkr_order_id

    async def cancel_order(self, order_id: str) -> None:
        """Cancel a previously placed order. Swallows errors (e.g. already filled)."""
        if order_id not in self._open_orders:
            _logger.warning("cancel_unknown_order", order_id=order_id)
            return

        trade = self._open_orders[order_id]
        try:
            self._market_data_provider.ib.cancelOrder(trade.order)
            _logger.info("order_cancelled", order_id=order_id)
        except Exception as exc:
            # IBKR error 10148 = order already filled; swallow all cancel errors.
            _logger.warning(
                "cancel_failed_may_be_filled",
                order_id=order_id,
                error=str(exc),
            )

    def _on_exec_details(self, trade: Trade, fill: Fill) -> None:
        """Handle an execution detail callback from ib_async.

        Called for every full or partial fill. Publishes a ``FillEvent`` to the
        event bus.
        """
        instrument = contract_to_instrument(fill.contract)
        if instrument is None:
            _logger.warning(
                "fill_unrecognised_contract",
                symbol=fill.contract.symbol,
            )
            return

        side = Direction.LONG if fill.execution.side == "BOT" else Direction.SHORT

        if fill.commissionReport is not None:
            commission_amount = Decimal(str(fill.commissionReport.commission))
        else:
            commission_amount = Decimal("0")

        fill_event = FillEvent(
            instrument=instrument,
            filled_quantity=Decimal(str(fill.execution.shares)),
            fill_price=Money(
                Decimal(str(fill.execution.price)),
                instrument.currency,
            ),
            commission=Money(commission_amount, instrument.currency),
            order_id=str(fill.execution.orderId),
            side=side,
            source="ibkr-execution",
            timestamp=datetime.now(tz=UTC),
        )

        self._event_bus.publish(fill_event)

        _logger.info(
            "fill_received",
            symbol=instrument.symbol,
            side=side.name,
            filled_quantity=str(fill.execution.shares),
            fill_price=str(fill.execution.price),
            commission=str(commission_amount),
        )

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def name(self) -> str:
        return "ibkr-execution"
