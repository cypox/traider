"""Unit tests for DiagnosticStrategy and Strategy base class."""

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bot.analytics.errors import ConvergenceError
from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.events.bus import EventBus
from bot.events.diagnostics import DiagnosticsEvent
from bot.events.market import BarEvent, QuoteEvent
from bot.events.orders import OrderEvent
from bot.events.signals import SignalEvent
from bot.providers.mock.metadata import MockMetadataProvider
from bot.providers.models import ContractDetails
from bot.strategies.diagnostic.strategy import DiagnosticStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _equity(symbol: str = "SPY") -> Instrument:
    return Instrument(
        symbol=symbol, asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART"
    )


def _bond(symbol: str = "ZN") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.BOND, currency="USD", exchange="SMART")


def _quote(instrument: Instrument, bid: str = "150.00", ask: str = "150.05") -> QuoteEvent:
    return QuoteEvent(
        timestamp=_TS,
        source="mock",
        instrument=instrument,
        bid=Decimal(bid),
        ask=Decimal(ask),
    )


def _bar(instrument: Instrument) -> BarEvent:
    return BarEvent(
        timestamp=_TS,
        source="mock",
        instrument=instrument,
        open=Decimal("150.00"),
        high=Decimal("151.00"),
        low=Decimal("149.50"),
        close=Decimal("150.50"),
        volume=Decimal("1000"),
        interval_seconds=5,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spy() -> Instrument:
    return _equity("SPY")


@pytest.fixture
def zn() -> Instrument:
    return _bond("ZN")


@pytest.fixture
def spy_details(spy: Instrument) -> ContractDetails:
    return ContractDetails(
        instrument=spy,
        full_name="SPDR S&P 500 ETF",
        coupon=None,
        maturity_date=None,
        face_value=None,
        tick_size=Decimal("0.01"),
        multiplier=Decimal("1"),
    )


@pytest.fixture
def zn_details(zn: Instrument) -> ContractDetails:
    return ContractDetails(
        instrument=zn,
        full_name="10-Year T-Note Future",
        coupon=Decimal("0.025"),
        maturity_date=date(2034, 6, 15),
        face_value=Money(Decimal("1000"), "USD"),
        tick_size=Decimal("0.015625"),
        multiplier=Decimal("1000"),
    )


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def metadata_provider(
    spy_details: ContractDetails, zn_details: ContractDetails
) -> MockMetadataProvider:
    return MockMetadataProvider({"SPY": spy_details, "ZN": zn_details})


@pytest.fixture
def strategy(
    event_bus: EventBus,
    metadata_provider: MockMetadataProvider,
    spy: Instrument,
    zn: Instrument,
) -> DiagnosticStrategy:
    return DiagnosticStrategy(
        event_bus=event_bus,
        metadata_provider=metadata_provider,
        instruments=[spy, zn],
    )


# ---------------------------------------------------------------------------
# on_quote — equity (SPY) with metadata loaded
# ---------------------------------------------------------------------------


class TestOnQuoteEquity:
    async def test_publishes_exactly_one_diagnostics_event(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
        metadata_provider: MockMetadataProvider,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(spy))
        assert len(received) == 1

    async def test_payload_contains_quote_fields(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(spy, bid="149.99", ask="150.01"))
        p = received[0].payload
        assert p["symbol"] == "SPY"
        assert p["bid"] == "149.99"
        assert p["ask"] == "150.01"
        assert "mid" in p

    async def test_payload_has_coupon_false_for_equity(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(spy))
        assert received[0].payload["has_coupon"] is False

    async def test_no_signal_event_published(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        signals: list[SignalEvent] = []
        event_bus.subscribe(SignalEvent, signals.append)
        event_bus.publish(_quote(spy))
        assert len(signals) == 0

    async def test_no_order_event_published(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        orders: list[OrderEvent] = []
        event_bus.subscribe(OrderEvent, orders.append)
        event_bus.publish(_quote(spy))
        assert len(orders) == 0


# ---------------------------------------------------------------------------
# on_quote — bond (ZN) with metadata loaded
# ---------------------------------------------------------------------------


class TestOnQuoteBond:
    async def test_payload_has_coupon_true(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(zn))
        assert received[0].payload["has_coupon"] is True

    async def test_payload_has_maturity_true(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(zn))
        assert received[0].payload["has_maturity"] is True

    async def test_no_signal_event_published(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        signals: list[SignalEvent] = []
        event_bus.subscribe(SignalEvent, signals.append)
        event_bus.publish(_quote(zn))
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# on_quote — instrument not in strategy's list
# ---------------------------------------------------------------------------


class TestOnQuoteUnknownInstrument:
    def test_no_diagnostics_event_published(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
    ) -> None:
        unknown = _equity("AAPL")
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(unknown))
        assert len(received) == 0

    def test_no_log_output(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
    ) -> None:
        unknown = _equity("AAPL")
        with patch("bot.strategies.diagnostic.strategy._logger") as mock_logger:
            event_bus.publish(_quote(unknown))
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# on_quote — without metadata (no refresh_metadata called)
# ---------------------------------------------------------------------------


class TestOnQuoteNoMetadata:
    def test_publishes_event_without_metadata_fields(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(spy))
        assert len(received) == 1
        p = received[0].payload
        assert "symbol" in p
        assert "has_coupon" not in p
        assert "full_name" not in p


# ---------------------------------------------------------------------------
# on_bar
# ---------------------------------------------------------------------------


class TestOnBar:
    def test_publishes_exactly_one_diagnostics_event(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_bar(spy))
        assert len(received) == 1

    def test_payload_contains_bar_fields(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_bar(spy))
        p = received[0].payload
        assert p["symbol"] == "SPY"
        assert p["open"] == "150.00"
        assert p["high"] == "151.00"
        assert p["low"] == "149.50"
        assert p["close"] == "150.50"
        assert p["volume"] == "1000"
        assert p["interval_seconds"] == 5

    def test_no_signal_event_published(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        signals: list[SignalEvent] = []
        event_bus.subscribe(SignalEvent, signals.append)
        event_bus.publish(_bar(spy))
        assert len(signals) == 0

    def test_unknown_instrument_skipped(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
    ) -> None:
        unknown = _equity("TSLA")
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_bar(unknown))
        assert len(received) == 0


# ---------------------------------------------------------------------------
# refresh_metadata
# ---------------------------------------------------------------------------


class TestRefreshMetadata:
    async def test_populates_cache_for_known_instruments(
        self,
        strategy: DiagnosticStrategy,
        spy: Instrument,
        zn: Instrument,
        spy_details: ContractDetails,
        zn_details: ContractDetails,
    ) -> None:
        await strategy.refresh_metadata()
        assert strategy._metadata_cache[spy] == spy_details
        assert strategy._metadata_cache[zn] == zn_details

    async def test_unknown_instrument_logs_warning_and_continues(
        self,
        event_bus: EventBus,
        spy: Instrument,
        zn: Instrument,
        spy_details: ContractDetails,
    ) -> None:
        # Only SPY in catalog; ZN will raise InstrumentNotFoundError.
        provider = MockMetadataProvider({"SPY": spy_details})
        strat = DiagnosticStrategy(
            event_bus=event_bus,
            metadata_provider=provider,
            instruments=[spy, zn],
        )
        mock_logger = MagicMock()
        with patch("bot.strategies.diagnostic.strategy._logger", mock_logger):
            await strat.refresh_metadata()
        assert spy in strat._metadata_cache
        assert zn not in strat._metadata_cache
        mock_logger.warning.assert_called_once()

    async def test_missing_instrument_does_not_crash(
        self,
        event_bus: EventBus,
        spy: Instrument,
        zn: Instrument,
    ) -> None:
        provider = MockMetadataProvider({})
        strat = DiagnosticStrategy(
            event_bus=event_bus,
            metadata_provider=provider,
            instruments=[spy, zn],
        )
        # Should not raise even when all instruments are missing.
        await strat.refresh_metadata()
        assert strat._metadata_cache == {}


# ---------------------------------------------------------------------------
# Isolation — 100 quotes must produce zero SignalEvent / OrderEvent
# ---------------------------------------------------------------------------


class TestIsolation:
    async def test_100_quotes_zero_signal_events(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        signals: list[SignalEvent] = []
        event_bus.subscribe(SignalEvent, signals.append)
        for i in range(100):
            bid = Decimal(f"{150 + i}.00")
            ask = bid + Decimal("0.05")
            event_bus.publish(
                QuoteEvent(
                    timestamp=_TS,
                    source="mock",
                    instrument=spy,
                    bid=bid,
                    ask=ask,
                )
            )
        assert len(signals) == 0

    async def test_100_quotes_zero_order_events(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        orders: list[OrderEvent] = []
        event_bus.subscribe(OrderEvent, orders.append)
        for i in range(100):
            bid = Decimal(f"{150 + i}.00")
            ask = bid + Decimal("0.05")
            event_bus.publish(
                QuoteEvent(
                    timestamp=_TS,
                    source="mock",
                    instrument=spy,
                    bid=bid,
                    ask=ask,
                )
            )
        assert len(orders) == 0


# ---------------------------------------------------------------------------
# Bond analytics — new tests for Prompt 7
# ---------------------------------------------------------------------------


class TestOnQuoteBondAnalytics:
    async def test_payload_contains_analytics_fields(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        # Use a realistic price near par (ZN face_value=1000)
        event_bus.publish(_quote(zn, bid="980.00", ask="982.00"))
        p = received[0].payload
        assert "ytm_pct" in p
        assert "dv01_usd" in p
        assert "macaulay_duration_years" in p
        assert "modified_duration_years" in p
        assert "discount_factor" in p
        assert "convexity" in p
        assert "days_to_maturity" in p

    async def test_analytics_values_are_native_types_not_decimal(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(zn, bid="980.00", ask="982.00"))
        p = received[0].payload
        assert isinstance(p["ytm_pct"], float)
        assert isinstance(p["dv01_usd"], float)
        assert isinstance(p["macaulay_duration_years"], float)
        assert isinstance(p["modified_duration_years"], float)
        assert isinstance(p["discount_factor"], float)
        assert isinstance(p["convexity"], float)
        assert isinstance(p["days_to_maturity"], int)

    async def test_discount_bond_positive_ytm_pct(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        # Price well below face_value=1000 → discount bond → ytm > coupon
        event_bus.publish(_quote(zn, bid="950.00", ask="952.00"))
        assert received[0].payload["ytm_pct"] > 0  # type: ignore[operator]

    async def test_no_signal_or_order_events_with_analytics(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        signals: list[SignalEvent] = []
        orders: list[OrderEvent] = []
        event_bus.subscribe(SignalEvent, signals.append)
        event_bus.subscribe(OrderEvent, orders.append)
        event_bus.publish(_quote(zn, bid="980.00", ask="982.00"))
        assert len(signals) == 0
        assert len(orders) == 0

    async def test_matured_bond_skips_analytics_and_warns(
        self,
        event_bus: EventBus,
        spy: Instrument,
        zn: Instrument,
        spy_details: ContractDetails,
    ) -> None:
        matured_details = ContractDetails(
            instrument=zn,
            full_name="Matured T-Note",
            coupon=Decimal("0.025"),
            maturity_date=date(2020, 1, 1),
            face_value=Money(Decimal("1000"), "USD"),
            tick_size=Decimal("0.015625"),
            multiplier=Decimal("1000"),
        )
        provider = MockMetadataProvider({"SPY": spy_details, "ZN": matured_details})
        strat = DiagnosticStrategy(
            event_bus=event_bus,
            metadata_provider=provider,
            instruments=[spy, zn],
        )
        await strat.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        mock_logger = MagicMock()
        with patch("bot.strategies.diagnostic.strategy._logger", mock_logger):
            event_bus.publish(_quote(zn, bid="980.00", ask="982.00"))
        assert len(received) == 1
        assert "ytm_pct" not in received[0].payload
        mock_logger.warning.assert_called_once()

    async def test_analytics_error_is_caught_and_warns(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        zn: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        mock_logger = MagicMock()
        with (
            patch(
                "bot.strategies.diagnostic.strategy.compute_ytm",
                side_effect=ConvergenceError("no convergence"),
            ),
            patch("bot.strategies.diagnostic.strategy._logger", mock_logger),
        ):
            event_bus.publish(_quote(zn, bid="980.00", ask="982.00"))
        assert len(received) == 1
        assert "ytm_pct" not in received[0].payload
        mock_logger.warning.assert_called_once()

    async def test_equity_payload_does_not_contain_ytm(
        self,
        strategy: DiagnosticStrategy,
        event_bus: EventBus,
        spy: Instrument,
    ) -> None:
        await strategy.refresh_metadata()
        received: list[DiagnosticsEvent] = []
        event_bus.subscribe(DiagnosticsEvent, received.append)
        event_bus.publish(_quote(spy))
        assert "ytm_pct" not in received[0].payload
