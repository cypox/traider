"""Integration tests for the IBKR connectivity layer.

All tests use ``unittest.mock`` and do not require a live TWS/Gateway connection.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ib_async import Bond as IbBond
from ib_async import ContractDetails as IbContractDetails
from ib_async import Forex, Future
from ib_async import Stock as IbStock

from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.providers.errors import InstrumentNotFoundError, UnsupportedAssetClassError
from bot.providers.ibkr.market_data import IBKRMarketDataProvider, _instrument_to_contract
from bot.providers.ibkr.metadata import IBKRMetadataProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity(symbol: str = "SPY") -> Instrument:
    return Instrument(
        symbol=symbol, asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART"
    )


def _future(symbol: str = "ES") -> Instrument:
    return Instrument(
        symbol=symbol, asset_class=AssetClass.FUTURE, currency="USD", exchange="GLOBEX"
    )


def _bond(symbol: str = "T") -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.BOND, currency="USD", exchange="SMART")


def _fx(symbol: str = "EURUSD") -> Instrument:
    return Instrument(
        symbol=symbol, asset_class=AssetClass.FX, currency="USD", exchange="IDEALPRO"
    )


def _crypto(symbol: str = "BTC") -> Instrument:
    return Instrument(
        symbol=symbol, asset_class=AssetClass.CRYPTO, currency="USD", exchange="PAXOS"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_ib() -> MagicMock:
    ib = MagicMock()
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.reqMktData = MagicMock()
    ib.reqRealTimeBars = MagicMock()
    ib.reqContractDetailsAsync = AsyncMock()
    return ib


@pytest.fixture
def provider(event_bus: EventBus, mock_ib: MagicMock) -> IBKRMarketDataProvider:
    with patch("bot.providers.ibkr.market_data.IB", return_value=mock_ib):
        p = IBKRMarketDataProvider(
            host="127.0.0.1", port=7497, client_id=1, event_bus=event_bus
        )
    return p


# ---------------------------------------------------------------------------
# _instrument_to_contract
# ---------------------------------------------------------------------------


class TestInstrumentToContract:
    def test_equity_returns_stock(self) -> None:
        result = _instrument_to_contract(_equity())
        assert isinstance(result, IbStock)
        assert result.symbol == "SPY"

    def test_future_returns_future(self) -> None:
        result = _instrument_to_contract(_future())
        assert isinstance(result, Future)
        assert result.symbol == "ES"

    def test_bond_returns_bond(self) -> None:
        result = _instrument_to_contract(_bond())
        assert isinstance(result, IbBond)

    def test_fx_returns_forex(self) -> None:
        result = _instrument_to_contract(_fx())
        assert isinstance(result, Forex)

    def test_crypto_raises_unsupported(self) -> None:
        with pytest.raises(UnsupportedAssetClassError, match="CRYPTO"):
            _instrument_to_contract(_crypto())


# ---------------------------------------------------------------------------
# IBKRMarketDataProvider – connection
# ---------------------------------------------------------------------------


class TestIBKRConnection:
    async def test_connect_calls_connect_async(
        self, provider: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        mock_ib.connectAsync.assert_called_once_with("127.0.0.1", 7497, clientId=1)
        assert provider.is_connected is True

    async def test_connect_live_port_7496_succeeds(
        self, event_bus: EventBus, mock_ib: MagicMock
    ) -> None:
        with patch("bot.providers.ibkr.market_data.IB", return_value=mock_ib):
            p = IBKRMarketDataProvider(
                host="127.0.0.1", port=7496, client_id=1, event_bus=event_bus
            )
        await p.connect()
        assert p.is_connected is True

    async def test_disconnect_calls_sync_disconnect(
        self, provider: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.disconnect()
        mock_ib.disconnect.assert_called_once()
        assert provider.is_connected is False

    def test_name_property(self, provider: IBKRMarketDataProvider) -> None:
        assert provider.name == "ibkr-market-data"

    def test_is_connected_initially_false(self, provider: IBKRMarketDataProvider) -> None:
        assert provider.is_connected is False


# ---------------------------------------------------------------------------
# IBKRMarketDataProvider – subscriptions
# ---------------------------------------------------------------------------


class TestIBKRSubscriptions:
    async def test_subscribe_quotes_registers_handler_once(
        self, provider: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        mock_ib.reqMktData.return_value = MagicMock()
        await provider.subscribe_quotes([_equity()])
        await provider.subscribe_quotes([_equity("AAPL")])
        assert mock_ib.pendingTickersEvent.connect.call_count == 1

    async def test_subscribe_quotes_stores_ticker_mapping(
        self, provider: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        mock_ticker = MagicMock()
        mock_ib.reqMktData.return_value = mock_ticker
        instrument = _equity("MSFT")
        await provider.subscribe_quotes([instrument])
        assert provider._ticker_to_instrument[mock_ticker] is instrument

    async def test_subscribe_bars_valid_interval(
        self, provider: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        mock_ib.reqRealTimeBars.return_value = MagicMock()
        await provider.subscribe_bars([_equity()], interval_seconds=5)
        mock_ib.reqRealTimeBars.assert_called_once()

    async def test_subscribe_bars_invalid_interval_raises(
        self, provider: IBKRMarketDataProvider
    ) -> None:
        with pytest.raises(ValueError, match="5-second"):
            await provider.subscribe_bars([_equity()], interval_seconds=60)


# ---------------------------------------------------------------------------
# Tick buffering
# ---------------------------------------------------------------------------


class TestTickBuffering:
    async def test_bid_alone_no_event(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity()
        ticker = MagicMock()
        ticker.bid = 150.0
        ticker.ask = float("nan")
        provider._ticker_to_instrument[ticker] = instrument
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker})
        assert len(received) == 0

    async def test_ask_alone_no_event(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity()
        ticker = MagicMock()
        ticker.bid = float("nan")
        ticker.ask = 150.05
        provider._ticker_to_instrument[ticker] = instrument
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker})
        assert len(received) == 0

    async def test_negative_bid_discarded(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity()
        ticker = MagicMock()
        ticker.bid = -1.0
        ticker.ask = 150.05
        provider._ticker_to_instrument[ticker] = instrument
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker})
        assert len(received) == 0

    async def test_valid_bid_and_ask_publishes_quote_event(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity("AAPL")
        ticker = MagicMock()
        ticker.bid = 182.50
        ticker.ask = 182.55
        provider._ticker_to_instrument[ticker] = instrument
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker})
        assert len(received) == 1
        assert received[0].bid == Decimal("182.5")
        assert received[0].ask == Decimal("182.55")
        assert received[0].instrument == instrument

    async def test_bid_and_ask_arrive_in_separate_ticks(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity("MSFT")
        ticker_bid = MagicMock()
        ticker_bid.bid = 300.0
        ticker_bid.ask = float("nan")
        ticker_ask = MagicMock()
        ticker_ask.bid = float("nan")
        ticker_ask.ask = 300.05
        provider._ticker_to_instrument[ticker_bid] = instrument
        provider._ticker_to_instrument[ticker_ask] = instrument
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker_bid})
        assert len(received) == 0
        provider._on_pending_tickers({ticker_ask})
        assert len(received) == 1
        assert received[0].bid == Decimal("300")
        assert received[0].ask == Decimal("300.05")

    async def test_unknown_ticker_ignored(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        ticker = MagicMock()
        ticker.bid = 100.0
        ticker.ask = 100.05
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker})
        assert len(received) == 0

    async def test_consecutive_valid_ticks_emit_two_events(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity("NVDA")
        ticker = MagicMock()
        ticker.bid = 500.0
        ticker.ask = 500.05
        provider._ticker_to_instrument[ticker] = instrument
        received: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, received.append)
        provider._on_pending_tickers({ticker})
        provider._on_pending_tickers({ticker})
        assert len(received) == 2


# ---------------------------------------------------------------------------
# Bar update handler
# ---------------------------------------------------------------------------


class TestBarUpdateHandler:
    async def test_handler_publishes_bar_event(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity("SPY")
        handler = provider._make_bar_handler(instrument)
        mock_bar = MagicMock()
        mock_bar.time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
        mock_bar.open_ = 450.0
        mock_bar.high = 451.0
        mock_bar.low = 449.5
        mock_bar.close = 450.5
        mock_bar.volume = 1000.0
        received: list[BarEvent] = []
        event_bus.subscribe(BarEvent, received.append)
        handler([mock_bar], True)
        assert len(received) == 1
        assert received[0].open == Decimal("450.0")
        assert received[0].close == Decimal("450.5")
        assert received[0].instrument == instrument

    async def test_handler_no_new_bar_skips_event(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity("SPY")
        handler = provider._make_bar_handler(instrument)
        received: list[BarEvent] = []
        event_bus.subscribe(BarEvent, received.append)
        handler([MagicMock()], False)
        assert len(received) == 0

    async def test_handler_empty_bars_list_skips_event(
        self, provider: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        instrument = _equity("SPY")
        handler = provider._make_bar_handler(instrument)
        received: list[BarEvent] = []
        event_bus.subscribe(BarEvent, received.append)
        handler([], True)
        assert len(received) == 0


# ---------------------------------------------------------------------------
# IBKRMetadataProvider
# ---------------------------------------------------------------------------


class TestIBKRMetadataProvider:
    def _make_provider(self, mock_ib: MagicMock) -> IBKRMetadataProvider:
        return IBKRMetadataProvider(ib=mock_ib, event_bus=EventBus())

    async def test_get_contract_details_equity(self, mock_ib: MagicMock) -> None:
        cd = IbContractDetails()
        cd.longName = "SPDR S&P 500 ETF"
        cd.minTick = 0.01
        s = IbStock("SPY", "SMART", "USD")
        s.conId = 756733
        cd.contract = s
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        result = await self._make_provider(mock_ib).get_contract_details(_equity("SPY"))
        assert result.full_name == "SPDR S&P 500 ETF"
        assert result.tick_size == Decimal("0.01")
        assert result.coupon is None
        assert result.maturity_date is None
        assert result.multiplier == Decimal("1")

    async def test_get_contract_details_bond(self, mock_ib: MagicMock) -> None:
        cd = IbContractDetails()
        cd.longName = "US 10-Year Treasury Note"
        cd.minTick = 0.015625
        cd.coupon = 4.5
        cd.maturity = "20340615"
        bond = IbBond()
        bond.conId = 12345
        cd.contract = bond
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        result = await self._make_provider(mock_ib).get_contract_details(_bond("T"))
        assert result.coupon == Decimal("4.5")
        assert result.maturity_date == date(2034, 6, 15)
        assert result.face_value == Money(Decimal("1000"), "USD")

    async def test_get_contract_details_empty_raises(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetailsAsync.return_value = []
        with pytest.raises(InstrumentNotFoundError, match="SPY"):
            await self._make_provider(mock_ib).get_contract_details(_equity("SPY"))

    async def test_get_contract_details_deduplicates_by_con_id(
        self, mock_ib: MagicMock
    ) -> None:
        cd1 = IbContractDetails()
        cd1.longName = "SPDR S&P 500 ETF"
        cd1.minTick = 0.01
        s1 = IbStock("SPY", "SMART", "USD")
        s1.conId = 756733
        cd1.contract = s1

        cd2 = IbContractDetails()
        cd2.longName = "SPDR S&P 500 ETF (duplicate)"
        cd2.minTick = 0.01
        s2 = IbStock("SPY", "SMART", "USD")
        s2.conId = 756733  # same conId → duplicate
        cd2.contract = s2

        mock_ib.reqContractDetailsAsync.return_value = [cd1, cd2]
        result = await self._make_provider(mock_ib).get_contract_details(_equity("SPY"))
        assert result.full_name == "SPDR S&P 500 ETF"

    async def test_get_contract_details_multiple_bonds_picks_nearest_maturity(
        self, mock_ib: MagicMock
    ) -> None:
        today = date.today()
        near_year = today.year + 1
        far_year = today.year + 5

        cd_near = IbContractDetails()
        cd_near.longName = "Near Bond"
        cd_near.minTick = 0.01
        cd_near.coupon = 3.0
        cd_near.maturity = f"{near_year}0615"
        b_near = IbBond()
        b_near.conId = 1
        cd_near.contract = b_near

        cd_far = IbContractDetails()
        cd_far.longName = "Far Bond"
        cd_far.minTick = 0.01
        cd_far.coupon = 4.0
        cd_far.maturity = f"{far_year}0615"
        b_far = IbBond()
        b_far.conId = 2
        cd_far.contract = b_far

        # Far bond is listed first — near bond must still be selected.
        mock_ib.reqContractDetailsAsync.return_value = [cd_far, cd_near]
        result = await self._make_provider(mock_ib).get_contract_details(_bond("T"))
        assert result.full_name == "Near Bond"

    async def test_get_instrument_returns_equity(self, mock_ib: MagicMock) -> None:
        cd = IbContractDetails()
        s = IbStock("SPY", "SMART", "USD")
        s.conId = 756733
        cd.contract = s
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        result = await self._make_provider(mock_ib).get_instrument("SPY", "SMART")
        assert result.symbol == "SPY"
        assert result.asset_class == AssetClass.EQUITY

    async def test_get_instrument_empty_raises(self, mock_ib: MagicMock) -> None:
        mock_ib.reqContractDetailsAsync.return_value = []
        with pytest.raises(InstrumentNotFoundError, match="INVALID"):
            await self._make_provider(mock_ib).get_instrument("INVALID", "SMART")
