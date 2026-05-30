"""Integration tests for the execution engine and runner wiring."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.positions import ApprovedPosition, Forecast, TargetPosition
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.diagnostics import DiagnosticsEvent
from bot.events.market import QuoteEvent
from bot.events.orders import OrderEvent
from bot.events.portfolio import ApprovedPositionEvent, FillEvent
from bot.execution.engine import ExecutionEngine
from bot.portfolio.construction import (
    PortfolioConstruction,
    PortfolioConstructionConfig,
    SignalCombinationMethod,
)
from bot.portfolio.state import PortfolioState
from bot.providers.mock.execution import MockExecutionProvider
from bot.providers.mock.market_data import MockMarketDataProvider
from bot.providers.mock.metadata import MockMetadataProvider
from bot.risk.config import RiskConfig
from bot.risk.engine import RiskEngine
from bot.strategies.diagnostic.strategy import DiagnosticStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)
_INITIAL_CASH = Money(Decimal("100000"), "USD")


def _spy() -> Instrument:
    return Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _make_target(instrument: Instrument, qty: Decimal) -> TargetPosition:
    return TargetPosition(
        instrument=instrument,
        target_quantity=qty,
        target_notional=Money(qty * Decimal("100"), "USD"),
        source_forecast=Forecast(instrument=instrument, timestamp=_TS, value=1.0, source="test"),
    )


def _make_approved(
    instrument: Instrument, qty: Decimal, notes: str = "approved"
) -> ApprovedPosition:
    return ApprovedPosition(
        instrument=instrument,
        approved_quantity=qty,
        original_target=_make_target(instrument, qty),
        risk_notes=notes,
    )


def _make_approved_event(approved: ApprovedPosition) -> ApprovedPositionEvent:
    return ApprovedPositionEvent(
        approved=approved,
        timestamp=_TS,
        source="test",
    )


def _build_engine(
    portfolio_state: PortfolioState | None = None,
    mock_exec: MockExecutionProvider | None = None,
) -> tuple[ExecutionEngine, MockExecutionProvider, EventBus, PortfolioState]:
    bus = EventBus()
    if portfolio_state is None:
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
    if mock_exec is None:
        mock_exec = MockExecutionProvider()
    engine = ExecutionEngine(
        event_bus=bus,
        execution_provider=mock_exec,
        portfolio_state=portfolio_state,
    )
    return engine, mock_exec, bus, portfolio_state


# ---------------------------------------------------------------------------
# ExecutionEngine — delta computation and order placement
# ---------------------------------------------------------------------------


class TestExecutionEngineDelta:
    async def test_zero_current_places_buy_for_full_qty(self) -> None:
        """qty=10, current=0 → buy 10."""
        _, mock_exec, bus, _ = _build_engine()
        approved = _make_approved(_spy(), Decimal("10"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)  # let the task execute
        assert len(mock_exec.placed_orders) == 1
        assert mock_exec.placed_orders[0].quantity == Decimal("10")
        assert mock_exec.placed_orders[0].side == Direction.LONG

    async def test_partial_position_places_buy_for_delta(self) -> None:
        """qty=10, current=7 → buy 3."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        mock_exec = MockExecutionProvider()
        ExecutionEngine(bus, mock_exec, portfolio_state)

        # Simulate a fill so portfolio_state knows about the existing 7 shares.
        bus.publish(
            FillEvent(
                order_id="EXISTING",
                instrument=_spy(),
                filled_quantity=Decimal("7"),
                fill_price=Money(Decimal("100"), "USD"),
                commission=Money(Decimal("0"), "USD"),
                side=Direction.LONG,
                timestamp=_TS,
                source="test",
            )
        )

        approved = _make_approved(_spy(), Decimal("10"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert len(mock_exec.placed_orders) == 1
        assert mock_exec.placed_orders[0].quantity == Decimal("3")
        assert mock_exec.placed_orders[0].side == Direction.LONG

    async def test_reduce_to_zero_places_sell(self) -> None:
        """qty=0, current=5 → sell 5."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        mock_exec = MockExecutionProvider()
        ExecutionEngine(bus, mock_exec, portfolio_state)

        bus.publish(
            FillEvent(
                order_id="EXISTING",
                instrument=_spy(),
                filled_quantity=Decimal("5"),
                fill_price=Money(Decimal("100"), "USD"),
                commission=Money(Decimal("0"), "USD"),
                side=Direction.LONG,
                timestamp=_TS,
                source="test",
            )
        )

        approved = _make_approved(_spy(), Decimal("0"), notes="reduce to flat")
        # approved_quantity=0, current=5 → delta=-5 → sell 5
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert len(mock_exec.placed_orders) == 1
        assert mock_exec.placed_orders[0].quantity == Decimal("5")
        assert mock_exec.placed_orders[0].side == Direction.SHORT

    async def test_already_at_target_no_order(self) -> None:
        """qty=5, current=5 → no order (delta=0)."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        mock_exec = MockExecutionProvider()
        ExecutionEngine(bus, mock_exec, portfolio_state)

        bus.publish(
            FillEvent(
                order_id="EXISTING",
                instrument=_spy(),
                filled_quantity=Decimal("5"),
                fill_price=Money(Decimal("100"), "USD"),
                commission=Money(Decimal("0"), "USD"),
                side=Direction.LONG,
                timestamp=_TS,
                source="test",
            )
        )

        approved = _make_approved(_spy(), Decimal("5"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert len(mock_exec.placed_orders) == 0

    async def test_delta_less_than_one_no_order(self) -> None:
        """delta=0.5 < 1 → no order placed."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        mock_exec = MockExecutionProvider()
        ExecutionEngine(bus, mock_exec, portfolio_state)

        # current=10, approved=10.5 → delta=0.5 < 1
        bus.publish(
            FillEvent(
                order_id="EXISTING",
                instrument=_spy(),
                filled_quantity=Decimal("10"),
                fill_price=Money(Decimal("100"), "USD"),
                commission=Money(Decimal("0"), "USD"),
                side=Direction.LONG,
                timestamp=_TS,
                source="test",
            )
        )
        approved = _make_approved(_spy(), Decimal("10.5"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert len(mock_exec.placed_orders) == 0

    async def test_positive_delta_gives_long_side(self) -> None:
        """Positive delta (buying more) → LONG side."""
        _, mock_exec, bus, _ = _build_engine()
        approved = _make_approved(_spy(), Decimal("20"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert mock_exec.placed_orders[0].side == Direction.LONG

    async def test_negative_delta_gives_short_side(self) -> None:
        """Negative delta (selling) → SHORT side."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        mock_exec = MockExecutionProvider()
        ExecutionEngine(bus, mock_exec, portfolio_state)

        bus.publish(
            FillEvent(
                order_id="EXISTING",
                instrument=_spy(),
                filled_quantity=Decimal("20"),
                fill_price=Money(Decimal("100"), "USD"),
                commission=Money(Decimal("0"), "USD"),
                side=Direction.LONG,
                timestamp=_TS,
                source="test",
            )
        )
        approved = _make_approved(_spy(), Decimal("5"), notes="trim position")
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert mock_exec.placed_orders[0].side == Direction.SHORT


# ---------------------------------------------------------------------------
# ExecutionEngine — order event publishing
# ---------------------------------------------------------------------------


class TestExecutionEngineOrderEvent:
    async def test_order_event_published_after_fill(self) -> None:
        _, mock_exec, bus, _ = _build_engine()
        order_events: list[OrderEvent] = []
        bus.subscribe(OrderEvent, order_events.append)

        approved = _make_approved(_spy(), Decimal("10"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert len(order_events) == 1
        assert order_events[0].order_id.startswith("MOCK-ORDER")
        assert order_events[0].intent.quantity == Decimal("10")

    async def test_order_event_source_is_execution_engine(self) -> None:
        _, mock_exec, bus, _ = _build_engine()
        order_events: list[OrderEvent] = []
        bus.subscribe(OrderEvent, order_events.append)

        approved = _make_approved(_spy(), Decimal("5"))
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert order_events[0].source == "execution-engine"

    async def test_empty_risk_notes_uses_fallback_reason(self) -> None:
        """Empty risk_notes should not raise InvalidIntentError."""
        _, mock_exec, bus, _ = _build_engine()
        approved = _make_approved(_spy(), Decimal("10"), notes="")
        bus.publish(_make_approved_event(approved))
        await asyncio.sleep(0)
        assert len(mock_exec.placed_orders) == 1
        assert mock_exec.placed_orders[0].reason == "approved by risk engine"


# ---------------------------------------------------------------------------
# Full wiring test with mock providers
# ---------------------------------------------------------------------------


class TestFullWiring:
    async def test_diagnostics_event_published_on_quote(self) -> None:
        """End-to-end: QuoteEvent → DiagnosticStrategy → DiagnosticsEvent."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)

        instruments = [_spy()]
        mock_meta = MockMetadataProvider(catalog={})
        mock_exec = MockExecutionProvider()
        mock_market = MockMarketDataProvider(bus)

        DiagnosticStrategy(
            event_bus=bus,
            metadata_provider=mock_meta,
            instruments=instruments,
        )

        _risk = RiskEngine(
            config=RiskConfig(
                max_position_usd=Decimal("50000"),
                max_gross_exposure_usd=Decimal("200000"),
                max_drawdown_pct=Decimal("0.10"),
                max_concentration_pct=Decimal("0.50"),
                daily_loss_limit_usd=Decimal("5000"),
            ),
            event_bus=bus,
            portfolio_state=portfolio_state,
        )

        _pc = PortfolioConstruction(
            config=PortfolioConstructionConfig(
                target_annual_vol=0.15,
                vol_lookback_days=60,
                min_forecast_for_trade=0.01,
                combination_method=SignalCombinationMethod.SIMPLE_AVERAGE,
            ),
            event_bus=bus,
            portfolio_state=portfolio_state,
        )

        ExecutionEngine(
            event_bus=bus,
            execution_provider=mock_exec,
            portfolio_state=portfolio_state,
        )

        diagnostics_received: list[DiagnosticsEvent] = []
        bus.subscribe(DiagnosticsEvent, diagnostics_received.append)

        await mock_market.connect()
        await mock_market.subscribe_quotes(instruments)

        mock_market.push_quote(
            QuoteEvent(
                instrument=_spy(),
                bid=Decimal("450"),
                ask=Decimal("451"),
                timestamp=_TS,
                source="mock",
            )
        )

        assert len(diagnostics_received) == 1
        assert diagnostics_received[0].payload["symbol"] == "SPY"

    async def test_full_wiring_no_exception_on_quote(self) -> None:
        """Wiring smoke test: no exception when pushing a QuoteEvent."""
        bus = EventBus()
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)

        instruments = [_spy()]
        mock_meta = MockMetadataProvider(catalog={})
        mock_exec = MockExecutionProvider()
        mock_market = MockMarketDataProvider(bus)

        DiagnosticStrategy(event_bus=bus, metadata_provider=mock_meta, instruments=instruments)

        RiskEngine(
            config=RiskConfig(
                max_position_usd=Decimal("50000"),
                max_gross_exposure_usd=Decimal("200000"),
                max_drawdown_pct=Decimal("0.10"),
                max_concentration_pct=Decimal("0.50"),
                daily_loss_limit_usd=Decimal("5000"),
            ),
            event_bus=bus,
            portfolio_state=portfolio_state,
        )

        PortfolioConstruction(
            config=PortfolioConstructionConfig(
                target_annual_vol=0.15,
                vol_lookback_days=60,
                min_forecast_for_trade=0.01,
                combination_method=SignalCombinationMethod.SIMPLE_AVERAGE,
            ),
            event_bus=bus,
            portfolio_state=portfolio_state,
        )

        ExecutionEngine(
            event_bus=bus,
            execution_provider=mock_exec,
            portfolio_state=portfolio_state,
        )

        await mock_market.connect()
        await mock_market.subscribe_quotes(instruments)

        # Should not raise
        mock_market.push_quote(
            QuoteEvent(
                instrument=_spy(),
                bid=Decimal("450"),
                ask=Decimal("451"),
                timestamp=_TS,
                source="mock",
            )
        )
