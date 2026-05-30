"""Unit tests for src/bot/providers/ — interfaces, models, and mocks."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from bot.core.execution import ExecutionIntent
from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.positions import ApprovedPosition, TargetPosition
from bot.core.signals import Direction, Forecast
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.providers import errors as _provider_errors
from bot.providers.errors import (
    InstrumentNotFoundError,
    OrderRejectedError,
    ProviderError,
    UnsupportedAssetClassError,
)
from bot.providers.mock.execution import MockExecutionProvider
from bot.providers.mock.market_data import MockMarketDataProvider
from bot.providers.mock.metadata import MockMetadataProvider
from bot.providers.mock.yield_curve import MockYieldCurveProvider
from bot.providers.models import ContractDetails, YieldCurve

_UTC_NOW = datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instr(symbol: str = "SPY") -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        currency="USD",
        exchange="SMART",
    )


def _forecast(instrument: Instrument | None = None) -> Forecast:
    return Forecast(
        instrument=instrument or _instr(),
        timestamp=_UTC_NOW,
        value=0.5,
        source="test",
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
        risk_notes="ok",
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


def _contract_details(instrument: Instrument | None = None) -> ContractDetails:
    return ContractDetails(
        instrument=instrument or _instr(),
        full_name="SPDR S&P 500 ETF",
        coupon=None,
        maturity_date=None,
        face_value=None,
        tick_size=Decimal("0.01"),
        multiplier=Decimal("1"),
    )


def _quote_event(instrument: Instrument | None = None) -> QuoteEvent:
    return QuoteEvent(
        timestamp=_UTC_NOW,
        source="feed",
        instrument=instrument or _instr(),
        bid=Decimal("100"),
        ask=Decimal("101"),
    )


def _bar_event(instrument: Instrument | None = None) -> BarEvent:
    return BarEvent(
        timestamp=_UTC_NOW,
        source="feed",
        instrument=instrument or _instr(),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("98"),
        close=Decimal("103"),
        volume=Decimal("1000"),
        interval_seconds=60,
    )


# ---------------------------------------------------------------------------
# Error hierarchy (import coverage)
# ---------------------------------------------------------------------------


def test_provider_error_hierarchy() -> None:
    assert issubclass(ProviderError, Exception)
    assert issubclass(InstrumentNotFoundError, ProviderError)
    assert issubclass(_provider_errors.ConnectionError, ProviderError)
    assert issubclass(UnsupportedAssetClassError, ProviderError)
    assert issubclass(OrderRejectedError, ProviderError)


# ---------------------------------------------------------------------------
# ContractDetails
# ---------------------------------------------------------------------------


def test_contract_details_valid() -> None:
    instr = _instr()
    details = ContractDetails(
        instrument=instr,
        full_name="SPDR S&P 500 ETF",
        coupon=None,
        maturity_date=None,
        face_value=None,
        tick_size=Decimal("0.01"),
        multiplier=Decimal("1"),
    )
    assert details.instrument is instr
    assert details.coupon is None


def test_contract_details_bond_fields() -> None:
    instr = Instrument(symbol="US10Y", asset_class=AssetClass.BOND, currency="USD", exchange="CME")
    details = ContractDetails(
        instrument=instr,
        full_name="US 10-Year Treasury",
        coupon=Decimal("0.045"),
        maturity_date=date(2034, 6, 30),
        face_value=Money(Decimal("1000"), "USD"),
        tick_size=Decimal("0.0001"),
        multiplier=Decimal("1000"),
    )
    assert details.coupon == Decimal("0.045")
    assert details.maturity_date == date(2034, 6, 30)


# ---------------------------------------------------------------------------
# YieldCurve — construction validation
# ---------------------------------------------------------------------------


def test_yield_curve_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        YieldCurve(
            as_of=_UTC_NOW,
            tenors_days=(30, 90),
            rates=(Decimal("0.04"),),  # only one rate for two tenors
        )


def test_yield_curve_non_increasing_tenors_raises() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        YieldCurve(
            as_of=_UTC_NOW,
            tenors_days=(90, 30),
            rates=(Decimal("0.04"), Decimal("0.05")),
        )


def test_yield_curve_equal_tenors_raises() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        YieldCurve(
            as_of=_UTC_NOW,
            tenors_days=(30, 30),
            rates=(Decimal("0.04"), Decimal("0.05")),
        )


def test_yield_curve_negative_rate_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        YieldCurve(
            as_of=_UTC_NOW,
            tenors_days=(30,),
            rates=(Decimal("-0.01"),),
        )


def test_yield_curve_valid() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(30, 90, 180),
        rates=(Decimal("0.04"), Decimal("0.05"), Decimal("0.06")),
    )
    assert curve.tenors_days == (30, 90, 180)


# ---------------------------------------------------------------------------
# YieldCurve — interpolation
# ---------------------------------------------------------------------------


def _two_point_curve() -> YieldCurve:
    return YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(0, 100),
        rates=(Decimal("0.04"), Decimal("0.06")),
    )


def test_yield_curve_interpolate_exact_min_tenor() -> None:
    curve = _two_point_curve()
    assert curve.interpolate(0) == Decimal("0.04")


def test_yield_curve_interpolate_exact_max_tenor() -> None:
    curve = _two_point_curve()
    assert curve.interpolate(100) == Decimal("0.06")


def test_yield_curve_interpolate_midpoint() -> None:
    curve = _two_point_curve()
    result = curve.interpolate(50)
    assert result == Decimal("0.05")


def test_yield_curve_interpolate_below_minimum_flat() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(30, 90),
        rates=(Decimal("0.04"), Decimal("0.06")),
    )
    assert curve.interpolate(10) == Decimal("0.04")


def test_yield_curve_interpolate_above_maximum_flat() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(30, 90),
        rates=(Decimal("0.04"), Decimal("0.06")),
    )
    assert curve.interpolate(200) == Decimal("0.06")


def test_yield_curve_interpolate_single_tenor_any_days() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(30,),
        rates=(Decimal("0.05"),),
    )
    assert curve.interpolate(1) == Decimal("0.05")
    assert curve.interpolate(365) == Decimal("0.05")


def test_yield_curve_interpolate_exact_middle_tenor() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(0, 50, 100),
        rates=(Decimal("0.04"), Decimal("0.05"), Decimal("0.06")),
    )
    assert curve.interpolate(50) == Decimal("0.05")


# ---------------------------------------------------------------------------
# MockMarketDataProvider
# ---------------------------------------------------------------------------


async def test_mock_market_data_connect_sets_connected() -> None:
    provider = MockMarketDataProvider(bus=EventBus())
    assert not provider.is_connected
    await provider.connect()
    assert provider.is_connected


async def test_mock_market_data_disconnect_clears_connected() -> None:
    provider = MockMarketDataProvider(bus=EventBus())
    await provider.connect()
    await provider.disconnect()
    assert not provider.is_connected


async def test_mock_market_data_name() -> None:
    provider = MockMarketDataProvider(bus=EventBus())
    assert provider.name == "mock-market-data"


async def test_mock_market_data_push_quote_when_subscribed() -> None:
    bus = EventBus()
    provider = MockMarketDataProvider(bus=bus)
    instr = _instr()
    await provider.subscribe_quotes([instr])

    received: list[QuoteEvent] = []

    def handler(event: QuoteEvent) -> None:
        received.append(event)

    bus.subscribe(QuoteEvent, handler)
    event = _quote_event(instr)
    provider.push_quote(event)
    assert received == [event]


async def test_mock_market_data_push_quote_not_subscribed() -> None:
    bus = EventBus()
    provider = MockMarketDataProvider(bus=bus)

    received: list[QuoteEvent] = []

    def handler(event: QuoteEvent) -> None:
        received.append(event)

    bus.subscribe(QuoteEvent, handler)
    provider.push_quote(_quote_event())  # instrument never subscribed
    assert received == []


async def test_mock_market_data_push_bar_when_subscribed() -> None:
    bus = EventBus()
    provider = MockMarketDataProvider(bus=bus)
    instr = _instr()
    await provider.subscribe_bars([instr], interval_seconds=60)

    received: list[BarEvent] = []

    def handler(event: BarEvent) -> None:
        received.append(event)

    bus.subscribe(BarEvent, handler)
    event = _bar_event(instr)
    provider.push_bar(event)
    assert received == [event]


async def test_mock_market_data_push_bar_not_subscribed() -> None:
    bus = EventBus()
    provider = MockMarketDataProvider(bus=bus)

    received: list[BarEvent] = []

    def handler(event: BarEvent) -> None:
        received.append(event)

    bus.subscribe(BarEvent, handler)
    provider.push_bar(_bar_event())  # instrument never subscribed
    assert received == []


# ---------------------------------------------------------------------------
# MockExecutionProvider
# ---------------------------------------------------------------------------


async def test_mock_execution_placed_orders_empty_on_init() -> None:
    provider = MockExecutionProvider()
    assert provider.placed_orders == []


async def test_mock_execution_connect_disconnect() -> None:
    provider = MockExecutionProvider()
    assert not provider.is_connected
    await provider.connect()
    assert provider.is_connected
    await provider.disconnect()
    assert not provider.is_connected


async def test_mock_execution_place_order_first() -> None:
    provider = MockExecutionProvider()
    intent = _intent()
    order_id = await provider.place_order(intent)
    assert order_id == "MOCK-ORDER-001"
    assert provider.placed_orders == [intent]


async def test_mock_execution_place_order_sequential_ids() -> None:
    provider = MockExecutionProvider()
    intent = _intent()
    await provider.place_order(intent)
    second_id = await provider.place_order(intent)
    third_id = await provider.place_order(intent)
    assert second_id == "MOCK-ORDER-002"
    assert third_id == "MOCK-ORDER-003"


async def test_mock_execution_cancel_order() -> None:
    provider = MockExecutionProvider()
    await provider.cancel_order("MOCK-ORDER-001")
    assert provider.cancelled_orders == ["MOCK-ORDER-001"]


# ---------------------------------------------------------------------------
# MockMetadataProvider
# ---------------------------------------------------------------------------


async def test_mock_metadata_known_symbol_get_instrument() -> None:
    instr = _instr()
    details = _contract_details(instr)
    provider = MockMetadataProvider(catalog={"SPY": details})
    result = await provider.get_instrument("SPY", "SMART")
    assert result == instr


async def test_mock_metadata_unknown_symbol_raises() -> None:
    provider = MockMetadataProvider(catalog={})
    with pytest.raises(InstrumentNotFoundError, match="UNKNOWN"):
        await provider.get_instrument("UNKNOWN", "SMART")


async def test_mock_metadata_get_contract_details_known() -> None:
    instr = _instr()
    details = _contract_details(instr)
    provider = MockMetadataProvider(catalog={"SPY": details})
    result = await provider.get_contract_details(instr)
    assert result is details


async def test_mock_metadata_get_contract_details_unknown_raises() -> None:
    provider = MockMetadataProvider(catalog={})
    with pytest.raises(InstrumentNotFoundError, match="SPY"):
        await provider.get_contract_details(_instr())


# ---------------------------------------------------------------------------
# MockYieldCurveProvider
# ---------------------------------------------------------------------------


async def test_mock_yield_curve_always_returns_fixed_curve() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(30, 90),
        rates=(Decimal("0.04"), Decimal("0.05")),
    )
    provider = MockYieldCurveProvider(fixed_curve=curve)
    result = await provider.get_curve(as_of=_UTC_NOW)
    assert result is curve


async def test_mock_yield_curve_ignores_as_of_date() -> None:
    curve = YieldCurve(
        as_of=_UTC_NOW,
        tenors_days=(30,),
        rates=(Decimal("0.05"),),
    )
    provider = MockYieldCurveProvider(fixed_curve=curve)
    different_date = datetime(2020, 1, 1, tzinfo=UTC)
    result = await provider.get_curve(as_of=different_date)
    assert result is curve
