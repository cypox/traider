"""Unit tests for src/bot/events/ — event types and EventBus."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from bot.core.execution import ExecutionIntent
from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.positions import ApprovedPosition, TargetPosition
from bot.core.signals import Direction, Forecast, Signal
from bot.events.base import BaseEvent
from bot.events.bus import EventBus
from bot.events.diagnostics import DiagnosticsEvent
from bot.events.market import BarEvent, QuoteEvent, TradeEvent
from bot.events.orders import OrderEvent
from bot.events.portfolio import ApprovedPositionEvent, FillEvent, TargetPositionEvent
from bot.events.signals import ForecastEvent, SignalEvent

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_UTC_NOW = datetime.now(tz=UTC)


def _instr() -> Instrument:
    return Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _signal(instrument: Instrument | None = None) -> Signal:
    return Signal(
        instrument=instrument or _instr(),
        timestamp=_UTC_NOW,
        direction=Direction.LONG,
        confidence=0.8,
        reason="test signal",
    )


def _forecast(instrument: Instrument | None = None) -> Forecast:
    return Forecast(
        instrument=instrument or _instr(),
        timestamp=_UTC_NOW,
        value=0.5,
        source="test_strategy",
    )


def _target(instrument: Instrument | None = None) -> TargetPosition:
    instr = instrument or _instr()
    return TargetPosition(
        instrument=instr,
        target_quantity=Decimal("10"),
        target_notional=Money(Decimal("1000"), "USD"),
        source_forecast=_forecast(instr),
    )


def _approved(instrument: Instrument | None = None) -> ApprovedPosition:
    instr = instrument or _instr()
    return ApprovedPosition(
        instrument=instr,
        approved_quantity=Decimal("10"),
        original_target=_target(instr),
        risk_notes="approved",
    )


def _intent(instrument: Instrument | None = None) -> ExecutionIntent:
    instr = instrument or _instr()
    return ExecutionIntent(
        instrument=instr,
        side=Direction.LONG,
        quantity=Decimal("10"),
        reason="entry",
        source_approved=_approved(instr),
    )


def _make_quote(bid: Decimal = Decimal("100"), ask: Decimal = Decimal("102")) -> QuoteEvent:
    return QuoteEvent(
        timestamp=_UTC_NOW,
        source="feed",
        instrument=_instr(),
        bid=bid,
        ask=ask,
    )


def _make_bar() -> BarEvent:
    return BarEvent(
        timestamp=_UTC_NOW,
        source="feed",
        instrument=_instr(),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("98"),
        close=Decimal("103"),
        volume=Decimal("1000"),
        interval_seconds=60,
    )


# ---------------------------------------------------------------------------
# BaseEvent
# ---------------------------------------------------------------------------


def test_base_event_naive_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        QuoteEvent(
            timestamp=datetime.now(),  # naive — no tzinfo
            source="feed",
            instrument=_instr(),
            bid=Decimal("100"),
            ask=Decimal("101"),
        )


def test_base_event_empty_source_raises() -> None:
    with pytest.raises(ValueError, match="source must be non-empty"):
        QuoteEvent(
            timestamp=_UTC_NOW,
            source="",
            instrument=_instr(),
            bid=Decimal("100"),
            ask=Decimal("101"),
        )


def test_base_event_auto_generates_event_id() -> None:
    e = _make_quote()
    assert e.event_id
    assert len(e.event_id) == 36  # UUID4 string length


def test_base_event_custom_event_id() -> None:
    e = QuoteEvent(
        timestamp=_UTC_NOW,
        source="feed",
        instrument=_instr(),
        bid=Decimal("100"),
        ask=Decimal("101"),
        event_id="custom-id",
    )
    assert e.event_id == "custom-id"


def test_base_event_two_events_have_different_ids() -> None:
    a = _make_quote()
    b = _make_quote()
    assert a.event_id != b.event_id


# ---------------------------------------------------------------------------
# QuoteEvent
# ---------------------------------------------------------------------------


def test_quote_event_valid() -> None:
    q = _make_quote()
    assert q.bid == Decimal("100")
    assert q.ask == Decimal("102")


def test_quote_event_mid() -> None:
    q = _make_quote(bid=Decimal("100"), ask=Decimal("102"))
    assert q.mid == Decimal("101")


def test_quote_event_negative_bid_raises() -> None:
    with pytest.raises(ValueError, match="bid must be positive"):
        _make_quote(bid=Decimal("-1"), ask=Decimal("102"))


def test_quote_event_zero_ask_raises() -> None:
    with pytest.raises(ValueError, match="ask must be positive"):
        _make_quote(bid=Decimal("100"), ask=Decimal("0"))


def test_quote_event_bid_greater_than_ask_raises() -> None:
    with pytest.raises(ValueError, match="bid must be <= ask"):
        _make_quote(bid=Decimal("103"), ask=Decimal("102"))


# ---------------------------------------------------------------------------
# BarEvent
# ---------------------------------------------------------------------------


def test_bar_event_valid() -> None:
    b = _make_bar()
    assert b.close == Decimal("103")


def test_bar_event_low_greater_than_open_raises() -> None:
    with pytest.raises(ValueError, match="low <= open <= high"):
        BarEvent(
            timestamp=_UTC_NOW,
            source="feed",
            instrument=_instr(),
            open=Decimal("97"),  # open < low → invalid
            high=Decimal("105"),
            low=Decimal("98"),
            close=Decimal("103"),
            volume=Decimal("1000"),
            interval_seconds=60,
        )


def test_bar_event_high_less_than_close_raises() -> None:
    with pytest.raises(ValueError, match="low <= close <= high"):
        BarEvent(
            timestamp=_UTC_NOW,
            source="feed",
            instrument=_instr(),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("104"),  # close > high → invalid
            volume=Decimal("1000"),
            interval_seconds=60,
        )


def test_bar_event_negative_volume_raises() -> None:
    with pytest.raises(ValueError, match="volume must be"):
        BarEvent(
            timestamp=_UTC_NOW,
            source="feed",
            instrument=_instr(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("98"),
            close=Decimal("103"),
            volume=Decimal("-1"),
            interval_seconds=60,
        )


def test_bar_event_zero_interval_seconds_raises() -> None:
    with pytest.raises(ValueError, match="interval_seconds must be positive"):
        BarEvent(
            timestamp=_UTC_NOW,
            source="feed",
            instrument=_instr(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("98"),
            close=Decimal("103"),
            volume=Decimal("1000"),
            interval_seconds=0,
        )


# ---------------------------------------------------------------------------
# TradeEvent
# ---------------------------------------------------------------------------


def test_trade_event_valid() -> None:
    t = TradeEvent(
        timestamp=_UTC_NOW,
        source="feed",
        instrument=_instr(),
        price=Decimal("101"),
        quantity=Decimal("5"),
        aggressor_side=Direction.LONG,
    )
    assert t.price == Decimal("101")


def test_trade_event_zero_price_raises() -> None:
    with pytest.raises(ValueError, match="price must be positive"):
        TradeEvent(
            timestamp=_UTC_NOW,
            source="feed",
            instrument=_instr(),
            price=Decimal("0"),
            quantity=Decimal("5"),
            aggressor_side=Direction.LONG,
        )


def test_trade_event_zero_quantity_raises() -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        TradeEvent(
            timestamp=_UTC_NOW,
            source="feed",
            instrument=_instr(),
            price=Decimal("101"),
            quantity=Decimal("0"),
            aggressor_side=Direction.LONG,
        )


# ---------------------------------------------------------------------------
# SignalEvent / ForecastEvent
# ---------------------------------------------------------------------------


def test_signal_event_valid() -> None:
    sig = _signal()
    e = SignalEvent(timestamp=_UTC_NOW, source="strategy", signal=sig)
    assert e.signal is sig


def test_forecast_event_valid() -> None:
    fc = _forecast()
    e = ForecastEvent(timestamp=_UTC_NOW, source="portfolio", forecast=fc)
    assert e.forecast is fc


# ---------------------------------------------------------------------------
# TargetPositionEvent / ApprovedPositionEvent
# ---------------------------------------------------------------------------


def test_target_position_event_valid() -> None:
    tgt = _target()
    e = TargetPositionEvent(timestamp=_UTC_NOW, source="portfolio", target=tgt)
    assert e.target is tgt


def test_approved_position_event_valid() -> None:
    app = _approved()
    e = ApprovedPositionEvent(timestamp=_UTC_NOW, source="risk", approved=app)
    assert e.approved is app


# ---------------------------------------------------------------------------
# FillEvent
# ---------------------------------------------------------------------------


def test_fill_event_valid() -> None:
    instr = _instr()
    e = FillEvent(
        timestamp=_UTC_NOW,
        source="broker",
        order_id="ORD-001",
        instrument=instr,
        filled_quantity=Decimal("5"),
        fill_price=Money(Decimal("101"), "USD"),
        commission=Money(Decimal("1"), "USD"),
        side=Direction.LONG,
    )
    assert e.filled_quantity == Decimal("5")


def test_fill_event_zero_quantity_raises() -> None:
    with pytest.raises(ValueError, match="filled_quantity must be positive"):
        FillEvent(
            timestamp=_UTC_NOW,
            source="broker",
            order_id="ORD-001",
            instrument=_instr(),
            filled_quantity=Decimal("0"),
            fill_price=Money(Decimal("101"), "USD"),
            commission=Money(Decimal("1"), "USD"),
            side=Direction.LONG,
        )


# ---------------------------------------------------------------------------
# OrderEvent
# ---------------------------------------------------------------------------


def test_order_event_valid() -> None:
    e = OrderEvent(
        timestamp=_UTC_NOW,
        source="execution",
        intent=_intent(),
        order_id="ORD-001",
    )
    assert e.order_id == "ORD-001"


# ---------------------------------------------------------------------------
# DiagnosticsEvent
# ---------------------------------------------------------------------------


def test_diagnostics_event_all_primitive_types() -> None:
    e = DiagnosticsEvent(
        timestamp=_UTC_NOW,
        source="monitor",
        payload={"str_val": "hello", "int_val": 42, "float_val": 3.14, "bool_val": True},
    )
    assert e.payload["int_val"] == 42


def test_diagnostics_event_nested_dict_raises() -> None:
    with pytest.raises(ValueError, match="payload values must be"):
        DiagnosticsEvent(
            timestamp=_UTC_NOW,
            source="monitor",
            payload={"key": {"nested": "dict"}},  # type: ignore[dict-item]
        )


def test_diagnostics_event_list_value_raises() -> None:
    with pytest.raises(ValueError, match="payload values must be"):
        DiagnosticsEvent(
            timestamp=_UTC_NOW,
            source="monitor",
            payload={"key": [1, 2, 3]},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


def test_bus_handler_receives_matching_event() -> None:
    bus = EventBus()
    received: list[QuoteEvent] = []

    def handler(event: QuoteEvent) -> None:
        received.append(event)

    bus.subscribe(QuoteEvent, handler)
    event = _make_quote()
    bus.publish(event)
    assert received == [event]


def test_bus_handler_does_not_receive_unrelated_event() -> None:
    bus = EventBus()
    received: list[QuoteEvent] = []

    def handler(event: QuoteEvent) -> None:
        received.append(event)

    bus.subscribe(QuoteEvent, handler)
    bus.publish(_make_bar())  # BarEvent is not a subtype of QuoteEvent
    assert received == []


def test_bus_multiple_handlers_all_receive_event() -> None:
    bus = EventBus()
    results: list[int] = []

    def h1(event: QuoteEvent) -> None:
        results.append(1)

    def h2(event: QuoteEvent) -> None:
        results.append(2)

    bus.subscribe(QuoteEvent, h1)
    bus.subscribe(QuoteEvent, h2)
    bus.publish(_make_quote())
    assert results == [1, 2]


@pytest.mark.filterwarnings("ignore:Remove `format_exc_info`:UserWarning")
def test_bus_failing_handler_does_not_block_remaining() -> None:
    bus = EventBus()
    good_calls: list[int] = []

    def failing_handler(event: QuoteEvent) -> None:
        raise RuntimeError("intentional failure")

    def good_handler(event: QuoteEvent) -> None:
        good_calls.append(1)

    bus.subscribe(QuoteEvent, failing_handler)
    bus.subscribe(QuoteEvent, good_handler)
    bus.publish(_make_quote())
    assert good_calls == [1]


def test_bus_base_event_handler_receives_all_subtypes() -> None:
    bus = EventBus()
    received: list[BaseEvent] = []

    def handler(event: BaseEvent) -> None:
        received.append(event)

    bus.subscribe(BaseEvent, handler)
    q = _make_quote()
    b = _make_bar()
    bus.publish(q)
    bus.publish(b)
    assert received == [q, b]


def test_bus_quote_handler_does_not_receive_bar_event() -> None:
    bus = EventBus()
    received: list[QuoteEvent] = []

    def handler(event: QuoteEvent) -> None:
        received.append(event)

    bus.subscribe(QuoteEvent, handler)
    bus.publish(_make_bar())
    assert received == []


def test_bus_subscriber_count_accurate() -> None:
    bus = EventBus()

    def h1(event: QuoteEvent) -> None:
        pass

    def h2(event: QuoteEvent) -> None:
        pass

    assert bus.subscriber_count(QuoteEvent) == 0
    bus.subscribe(QuoteEvent, h1)
    assert bus.subscriber_count(QuoteEvent) == 1
    bus.subscribe(QuoteEvent, h2)
    assert bus.subscriber_count(QuoteEvent) == 2
    bus.unsubscribe(QuoteEvent, h1)
    assert bus.subscriber_count(QuoteEvent) == 1


def test_bus_unsubscribe_unregistered_is_noop() -> None:
    bus = EventBus()

    def handler(event: QuoteEvent) -> None:
        pass

    def other(event: QuoteEvent) -> None:
        pass

    # event_type never subscribed → handlers is None path
    bus.unsubscribe(QuoteEvent, handler)

    # event_type has handlers but not this one → ValueError suppressed path
    bus.subscribe(QuoteEvent, other)
    bus.unsubscribe(QuoteEvent, handler)  # handler not in list
    assert bus.subscriber_count(QuoteEvent) == 1


def test_bus_clear_removes_all_subscriptions() -> None:
    bus = EventBus()
    received: list[BaseEvent] = []

    def handler(event: QuoteEvent) -> None:
        received.append(event)

    bus.subscribe(QuoteEvent, handler)
    bus.clear()
    assert bus.subscriber_count(QuoteEvent) == 0
    bus.publish(_make_quote())
    assert received == []


def test_bus_handlers_called_in_registration_order() -> None:
    bus = EventBus()
    call_order: list[int] = []

    def h1(event: QuoteEvent) -> None:
        call_order.append(1)

    def h2(event: QuoteEvent) -> None:
        call_order.append(2)

    def h3(event: QuoteEvent) -> None:
        call_order.append(3)

    bus.subscribe(QuoteEvent, h1)
    bus.subscribe(QuoteEvent, h2)
    bus.subscribe(QuoteEvent, h3)
    bus.publish(_make_quote())
    assert call_order == [1, 2, 3]


def test_bus_same_handler_registered_twice_receives_event_twice() -> None:
    bus = EventBus()
    count: list[int] = []

    def handler(event: QuoteEvent) -> None:
        count.append(1)

    bus.subscribe(QuoteEvent, handler)
    bus.subscribe(QuoteEvent, handler)
    bus.publish(_make_quote())
    assert len(count) == 2
