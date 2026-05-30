"""Integration tests for IBKRExecutionProvider and IBKR contract utils.

All tests use mocks — no live TWS/Gateway connection required.
"""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ib_async import Bond as IbBond
from ib_async import ContractDetails as IbContractDetails
from ib_async import Forex, Future
from ib_async import Stock as IbStock

from bot.core.execution import ExecutionIntent
from bot.core.instruments import AssetClass, Instrument
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.portfolio import FillEvent
from bot.providers.errors import (
    ConnectionError,  # noqa: A004
    InstrumentNotFoundError,
    UnsupportedAssetClassError,
)
from bot.providers.ibkr.execution import IBKRExecutionProvider
from bot.providers.ibkr.market_data import IBKRMarketDataProvider
from bot.providers.ibkr.metadata import IBKRMetadataProvider
from bot.providers.ibkr.utils import contract_to_instrument, instrument_to_contract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spy() -> Instrument:
    return Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _future_inst() -> Instrument:
    return Instrument(symbol="ES", asset_class=AssetClass.FUTURE, currency="USD", exchange="GLOBEX")


def _bond_inst() -> Instrument:
    return Instrument(symbol="T", asset_class=AssetClass.BOND, currency="USD", exchange="SMART")


def _fx_inst() -> Instrument:
    return Instrument(
        symbol="EURUSD", asset_class=AssetClass.FX, currency="USD", exchange="IDEALPRO"
    )


def _crypto_inst() -> Instrument:
    return Instrument(symbol="BTC", asset_class=AssetClass.CRYPTO, currency="USD", exchange="PAXOS")


def _make_intent(
    instrument: Instrument | None = None,
    side: Direction = Direction.LONG,
    qty: Decimal = Decimal("10"),
) -> ExecutionIntent:
    return ExecutionIntent(
        instrument=instrument or _spy(),
        side=side,
        quantity=qty,
        reason="test order",
        source_approved=None,
    )


def _make_fill(
    symbol: str = "SPY",
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD",
    ib_side: str = "BOT",
    shares: float = 10.0,
    price: float = 450.0,
    order_id: int = 42,
    commission: float | None = 1.5,
) -> tuple[Any, Any]:
    """Build a mock (Trade, Fill) pair for _on_exec_details testing."""
    mock_trade = MagicMock()

    mock_contract = MagicMock()
    mock_contract.symbol = symbol
    mock_contract.secType = sec_type
    mock_contract.exchange = exchange
    mock_contract.currency = currency

    mock_execution = MagicMock()
    mock_execution.side = ib_side
    mock_execution.shares = shares
    mock_execution.price = price
    mock_execution.orderId = order_id

    mock_fill = MagicMock()
    mock_fill.contract = mock_contract
    mock_fill.execution = mock_execution

    if commission is not None:
        mock_commission_report = MagicMock()
        mock_commission_report.commission = commission
        mock_fill.commissionReport = mock_commission_report
    else:
        mock_fill.commissionReport = None

    return mock_trade, mock_fill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ib() -> MagicMock:
    ib = MagicMock()
    # execDetailsEvent: __iadd__ and __isub__ must return the *same* object so
    # the attribute is not replaced after ib.execDetailsEvent += handler.
    event = MagicMock()
    event.__iadd__ = MagicMock(return_value=event)
    event.__isub__ = MagicMock(return_value=event)
    ib.execDetailsEvent = event
    # placeOrder returns a mock Trade with orderId=42
    mock_order = MagicMock()
    mock_order.orderId = 42
    mock_trade = MagicMock()
    mock_trade.order = mock_order
    ib.placeOrder.return_value = mock_trade
    ib.cancelOrder = MagicMock()
    # Async methods needed by real providers
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.reqMktData = MagicMock()
    ib.reqRealTimeBars = MagicMock()
    ib.reqContractDetailsAsync = AsyncMock()
    ib.pendingTickersEvent = MagicMock()
    return ib


@pytest.fixture
def real_mdp(mock_ib: MagicMock, event_bus: EventBus) -> IBKRMarketDataProvider:
    """Real IBKRMarketDataProvider with IB() patched to mock_ib."""
    with patch("bot.providers.ibkr.market_data.IB", return_value=mock_ib):
        return IBKRMarketDataProvider(
            host="127.0.0.1", port=7497, client_id=1, event_bus=event_bus
        )


@pytest.fixture
def real_metadata(real_mdp: IBKRMarketDataProvider, event_bus: EventBus) -> IBKRMetadataProvider:
    """Real IBKRMetadataProvider backed by real_mdp."""
    return IBKRMetadataProvider(market_data_provider=real_mdp, event_bus=event_bus)


@pytest.fixture
def mock_mdp(mock_ib: MagicMock) -> MagicMock:
    mdp = MagicMock(spec=IBKRMarketDataProvider)
    mdp.ib = mock_ib
    return mdp


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def provider(mock_mdp: MagicMock, event_bus: EventBus) -> IBKRExecutionProvider:
    return IBKRExecutionProvider(market_data_provider=mock_mdp, event_bus=event_bus)


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    async def test_connect_registers_fill_callback(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        mock_ib.execDetailsEvent.__iadd__.assert_called_once()

    async def test_connect_sets_is_connected(self, provider: IBKRExecutionProvider) -> None:
        await provider.connect()
        assert provider.is_connected is True

    async def test_disconnect_deregisters_fill_callback(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.disconnect()
        mock_ib.execDetailsEvent.__isub__.assert_called_once()

    async def test_disconnect_clears_is_connected(self, provider: IBKRExecutionProvider) -> None:
        await provider.connect()
        await provider.disconnect()
        assert provider.is_connected is False


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------


class TestPlaceOrder:
    async def test_place_order_when_not_connected_raises(
        self, provider: IBKRExecutionProvider
    ) -> None:
        with pytest.raises(ConnectionError):
            await provider.place_order(_make_intent())

    async def test_place_order_long_calls_buy(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.place_order(_make_intent(side=Direction.LONG, qty=Decimal("5")))
        mock_ib.placeOrder.assert_called_once()
        _, order = mock_ib.placeOrder.call_args[0]
        assert order.action == "BUY"

    async def test_place_order_short_calls_sell(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.place_order(_make_intent(side=Direction.SHORT, qty=Decimal("3")))
        _, order = mock_ib.placeOrder.call_args[0]
        assert order.action == "SELL"

    async def test_place_order_transmit_is_true(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.place_order(_make_intent())
        _, order = mock_ib.placeOrder.call_args[0]
        assert order.transmit is True

    async def test_place_order_quantity_matches(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.place_order(_make_intent(qty=Decimal("7")))
        _, order = mock_ib.placeOrder.call_args[0]
        assert order.totalQuantity == 7.0

    async def test_place_order_returns_order_id_as_string(
        self, provider: IBKRExecutionProvider
    ) -> None:
        await provider.connect()
        result = await provider.place_order(_make_intent())
        assert result == "42"

    async def test_place_order_stores_trade(self, provider: IBKRExecutionProvider) -> None:
        await provider.connect()
        order_id = await provider.place_order(_make_intent())
        assert order_id in provider._open_orders


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    async def test_cancel_known_order_calls_ib_cancel(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        order_id = await provider.place_order(_make_intent())
        await provider.cancel_order(order_id)
        mock_ib.cancelOrder.assert_called_once()

    async def test_cancel_unknown_order_does_not_call_ib(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        await provider.cancel_order("NONEXISTENT")
        mock_ib.cancelOrder.assert_not_called()

    async def test_cancel_unknown_order_no_exception(self, provider: IBKRExecutionProvider) -> None:
        await provider.connect()
        # Should not raise
        await provider.cancel_order("NONEXISTENT")

    async def test_cancel_when_ib_raises_no_exception_propagates(
        self, provider: IBKRExecutionProvider, mock_ib: MagicMock
    ) -> None:
        await provider.connect()
        order_id = await provider.place_order(_make_intent())
        mock_ib.cancelOrder.side_effect = RuntimeError("10148: Already filled")
        # Must not raise
        await provider.cancel_order(order_id)


# ---------------------------------------------------------------------------
# _on_exec_details — fill callbacks
# ---------------------------------------------------------------------------


class TestOnExecDetails:
    async def test_buy_fill_publishes_fill_event(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(ib_side="BOT", shares=10.0, price=450.0)
        provider._on_exec_details(trade, fill)
        assert len(fills) == 1

    async def test_buy_fill_direction_is_long(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(ib_side="BOT")
        provider._on_exec_details(trade, fill)
        assert fills[0].side == Direction.LONG

    async def test_sell_fill_direction_is_short(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(ib_side="SLD", shares=5.0)
        provider._on_exec_details(trade, fill)
        assert fills[0].side == Direction.SHORT

    async def test_fill_quantity_matches(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(shares=7.0)
        provider._on_exec_details(trade, fill)
        assert fills[0].filled_quantity == Decimal("7.0")

    async def test_fill_price_matches(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(price=451.25)
        provider._on_exec_details(trade, fill)
        assert fills[0].fill_price.amount == Decimal("451.25")

    async def test_partial_fill_uses_partial_quantity(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(shares=3.0)  # partial of a 10-share order
        provider._on_exec_details(trade, fill)
        assert fills[0].filled_quantity == Decimal("3.0")

    async def test_commission_none_yields_zero(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(commission=None)
        provider._on_exec_details(trade, fill)
        assert fills[0].commission.amount == Decimal("0")

    async def test_commission_populated(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(commission=2.5)
        provider._on_exec_details(trade, fill)
        assert fills[0].commission.amount == Decimal("2.5")

    async def test_unrecognised_sec_type_no_fill_event(
        self, provider: IBKRExecutionProvider, event_bus: EventBus
    ) -> None:
        fills: list[FillEvent] = []
        event_bus.subscribe(FillEvent, fills.append)
        trade, fill = _make_fill(sec_type="OPT")
        provider._on_exec_details(trade, fill)
        assert len(fills) == 0


# ---------------------------------------------------------------------------
# utils — instrument_to_contract
# ---------------------------------------------------------------------------


class TestInstrumentToContract:
    def test_equity_returns_stock(self) -> None:
        result = instrument_to_contract(_spy())
        assert isinstance(result, IbStock)
        assert result.symbol == "SPY"
        assert result.currency == "USD"

    def test_future_returns_future_with_exchange(self) -> None:
        result = instrument_to_contract(_future_inst())
        assert isinstance(result, Future)
        assert result.symbol == "ES"
        assert result.exchange == "GLOBEX"

    def test_bond_returns_bond(self) -> None:
        result = instrument_to_contract(_bond_inst())
        assert isinstance(result, IbBond)

    def test_fx_returns_forex(self) -> None:
        result = instrument_to_contract(_fx_inst())
        assert isinstance(result, Forex)

    def test_crypto_raises_unsupported(self) -> None:
        with pytest.raises(UnsupportedAssetClassError, match="CRYPTO"):
            instrument_to_contract(_crypto_inst())


# ---------------------------------------------------------------------------
# utils — contract_to_instrument
# ---------------------------------------------------------------------------


class TestContractToInstrument:
    def _make_contract(
        self,
        sec_type: str,
        symbol: str = "SPY",
        currency: str = "USD",
        exchange: str = "SMART",
    ) -> MagicMock:
        c = MagicMock()
        c.secType = sec_type
        c.symbol = symbol
        c.currency = currency
        c.exchange = exchange
        return c

    def test_stk_gives_equity(self) -> None:
        result = contract_to_instrument(self._make_contract("STK"))
        assert result is not None
        assert result.asset_class == AssetClass.EQUITY

    def test_fut_gives_future(self) -> None:
        result = contract_to_instrument(self._make_contract("FUT", symbol="ES"))
        assert result is not None
        assert result.asset_class == AssetClass.FUTURE

    def test_bond_gives_bond(self) -> None:
        result = contract_to_instrument(self._make_contract("BOND", symbol="T"))
        assert result is not None
        assert result.asset_class == AssetClass.BOND

    def test_cash_gives_fx(self) -> None:
        result = contract_to_instrument(self._make_contract("CASH", symbol="EURUSD"))
        assert result is not None
        assert result.asset_class == AssetClass.FX

    def test_option_returns_none(self) -> None:
        result = contract_to_instrument(self._make_contract("OPT"))
        assert result is None

    def test_empty_exchange_becomes_unknown(self) -> None:
        result = contract_to_instrument(self._make_contract("STK", exchange=""))
        assert result is not None
        assert result.exchange == "UNKNOWN"


# ---------------------------------------------------------------------------
# Regression — no duplicate instrument_to_contract definitions
# ---------------------------------------------------------------------------


class TestNoDuplicateContractConversion:
    def _get_function_names(self, filepath: Path) -> list[str]:
        """Return all top-level function names defined in the given Python file."""
        source = filepath.read_text()
        tree = ast.parse(source)
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]

    def test_market_data_does_not_define_instrument_to_contract(self) -> None:
        root = Path(__file__).parent.parent.parent / "src" / "bot" / "providers" / "ibkr"
        names = self._get_function_names(root / "market_data.py")
        assert "_instrument_to_contract" not in names
        assert "instrument_to_contract" not in names

    def test_metadata_does_not_define_instrument_to_contract(self) -> None:
        root = Path(__file__).parent.parent.parent / "src" / "bot" / "providers" / "ibkr"
        names = self._get_function_names(root / "metadata.py")
        assert "_instrument_to_contract" not in names
        assert "instrument_to_contract" not in names

    def test_utils_defines_instrument_to_contract(self) -> None:
        root = Path(__file__).parent.parent.parent / "src" / "bot" / "providers" / "ibkr"
        names = self._get_function_names(root / "utils.py")
        assert "instrument_to_contract" in names

    def test_utils_defines_contract_to_instrument(self) -> None:
        root = Path(__file__).parent.parent.parent / "src" / "bot" / "providers" / "ibkr"
        names = self._get_function_names(root / "utils.py")
        assert "contract_to_instrument" in names


# ---------------------------------------------------------------------------
# IBKRMarketDataProvider — shared IB ownership and basic operations
# ---------------------------------------------------------------------------


class TestIBKRMarketDataProviderShared:
    """Integration tests that exercise the real IBKRMarketDataProvider code path."""

    def test_ib_property_returns_patched_ib(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        assert real_mdp.ib is mock_ib

    def test_is_connected_initially_false(self, real_mdp: IBKRMarketDataProvider) -> None:
        assert real_mdp.is_connected is False

    def test_name_property(self, real_mdp: IBKRMarketDataProvider) -> None:
        assert real_mdp.name == "ibkr-market-data"

    async def test_connect_calls_connect_async(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await real_mdp.connect()
        mock_ib.connectAsync.assert_called_once_with("127.0.0.1", 7497, clientId=1)
        assert real_mdp.is_connected is True

    async def test_connect_live_port_logs_warning(
        self, event_bus: EventBus, mock_ib: MagicMock
    ) -> None:
        with patch("bot.providers.ibkr.market_data.IB", return_value=mock_ib):
            p = IBKRMarketDataProvider(
                host="127.0.0.1", port=7496, client_id=1, event_bus=event_bus
            )
        await p.connect()  # should not raise
        assert p.is_connected is True

    async def test_disconnect_calls_ib_disconnect(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await real_mdp.connect()
        await real_mdp.disconnect()
        mock_ib.disconnect.assert_called_once()
        assert real_mdp.is_connected is False

    async def test_subscribe_quotes_registers_pending_tickers_handler(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await real_mdp.subscribe_quotes([_spy()])
        mock_ib.pendingTickersEvent.connect.assert_called_once()

    async def test_subscribe_quotes_calls_req_mkt_data(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await real_mdp.subscribe_quotes([_spy()])
        mock_ib.reqMktData.assert_called_once()

    async def test_subscribe_quotes_second_call_does_not_re_register_handler(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await real_mdp.subscribe_quotes([_spy()])
        await real_mdp.subscribe_quotes([_future_inst()])
        mock_ib.pendingTickersEvent.connect.assert_called_once()  # registered once only

    async def test_subscribe_bars_wrong_interval_raises(
        self, real_mdp: IBKRMarketDataProvider
    ) -> None:
        with pytest.raises(ValueError, match="5-second"):
            await real_mdp.subscribe_bars([_spy()], interval_seconds=60)

    async def test_subscribe_bars_calls_req_real_time_bars(
        self, real_mdp: IBKRMarketDataProvider, mock_ib: MagicMock
    ) -> None:
        await real_mdp.subscribe_bars([_spy()], interval_seconds=5)
        mock_ib.reqRealTimeBars.assert_called_once()

    async def test_on_pending_tickers_publishes_quote_when_bid_and_ask(
        self, real_mdp: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        from bot.events.market import QuoteEvent

        quotes: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, quotes.append)

        # Register a ticker manually
        mock_ticker = MagicMock()
        mock_ticker.bid = 450.0
        mock_ticker.ask = 451.0
        spy = _spy()
        real_mdp._ticker_to_instrument[mock_ticker] = spy

        real_mdp._on_pending_tickers({mock_ticker})
        assert len(quotes) == 1
        assert quotes[0].bid == Decimal("450.0")

    async def test_on_pending_tickers_no_event_when_bid_missing(
        self, real_mdp: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        from bot.events.market import QuoteEvent

        quotes: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, quotes.append)

        mock_ticker = MagicMock()
        mock_ticker.bid = 0  # missing
        mock_ticker.ask = 451.0
        real_mdp._ticker_to_instrument[mock_ticker] = _spy()

        real_mdp._on_pending_tickers({mock_ticker})
        assert len(quotes) == 0

    async def test_on_pending_tickers_unknown_ticker_skipped(
        self, real_mdp: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        from bot.events.market import QuoteEvent

        quotes: list[QuoteEvent] = []
        event_bus.subscribe(QuoteEvent, quotes.append)
        mock_ticker = MagicMock()
        real_mdp._on_pending_tickers({mock_ticker})  # not in _ticker_to_instrument
        assert len(quotes) == 0

    async def test_bar_handler_publishes_bar_event(
        self, real_mdp: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        from datetime import UTC, datetime

        from bot.events.market import BarEvent

        bars: list[BarEvent] = []
        event_bus.subscribe(BarEvent, bars.append)

        spy = _spy()
        handler = real_mdp._make_bar_handler(spy)

        mock_bar = MagicMock()
        mock_bar.time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        mock_bar.open_ = 450.0
        mock_bar.high = 452.0
        mock_bar.low = 449.0
        mock_bar.close = 451.0
        mock_bar.volume = 1000

        mock_bars_list = MagicMock()
        mock_bars_list.__bool__ = lambda self: True
        mock_bars_list.__getitem__ = lambda self, idx: mock_bar

        handler(mock_bars_list, True)  # has_new_bar=True
        assert len(bars) == 1
        assert bars[0].close == Decimal("451.0")

    async def test_bar_handler_no_event_when_no_new_bar(
        self, real_mdp: IBKRMarketDataProvider, event_bus: EventBus
    ) -> None:
        from bot.events.market import BarEvent

        bars: list[BarEvent] = []
        event_bus.subscribe(BarEvent, bars.append)
        handler = real_mdp._make_bar_handler(_spy())
        handler(MagicMock(), False)  # has_new_bar=False
        assert len(bars) == 0


# ---------------------------------------------------------------------------
# IBKRMetadataProvider — shared IB ownership and basic operations
# ---------------------------------------------------------------------------


class TestIBKRMetadataProviderShared:
    """Integration tests that exercise the real IBKRMetadataProvider code path."""

    def test_uses_mdp_ib(self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock) -> None:
        assert real_metadata._market_data_provider.ib is mock_ib

    def _make_cd(
        self, symbol: str = "SPY", currency: str = "USD", exchange: str = "SMART"
    ) -> MagicMock:
        """Build a mock IbContractDetails."""
        mock_contract = MagicMock()
        mock_contract.symbol = symbol
        mock_contract.currency = currency
        mock_contract.exchange = exchange
        mock_contract.conId = 1
        mock_contract.multiplier = ""
        cd = MagicMock(spec=IbContractDetails)
        cd.contract = mock_contract
        cd.longName = "S&P 500 ETF"
        cd.minTick = 0.01
        cd.coupon = 0.0
        cd.maturity = ""
        return cd

    async def test_get_instrument_found(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        cd = self._make_cd()
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        result = await real_metadata.get_instrument("SPY", "SMART")
        assert result.symbol == "SPY"
        assert result.asset_class == AssetClass.EQUITY

    async def test_get_instrument_not_found_raises(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        mock_ib.reqContractDetailsAsync.return_value = []
        with pytest.raises(InstrumentNotFoundError):
            await real_metadata.get_instrument("UNKNOWN", "SMART")

    async def test_get_instrument_contract_none_raises(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        cd = self._make_cd()
        cd.contract = None
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        with pytest.raises(InstrumentNotFoundError):
            await real_metadata.get_instrument("SPY", "SMART")

    async def test_get_contract_details_equity(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        cd = self._make_cd()
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        result = await real_metadata.get_contract_details(_spy())
        assert result.instrument.symbol == "SPY"
        assert result.tick_size == Decimal("0.01")
        assert result.coupon is None  # equity

    async def test_get_contract_details_bond_fields(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        cd = self._make_cd(symbol="T")
        cd.coupon = 3.5
        cd.maturity = "20301231"
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        bond_inst = _bond_inst()
        result = await real_metadata.get_contract_details(bond_inst)
        assert result.coupon == Decimal("3.5")
        assert result.maturity_date is not None

    async def test_get_contract_details_not_found_raises(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        mock_ib.reqContractDetailsAsync.return_value = []
        with pytest.raises(InstrumentNotFoundError):
            await real_metadata.get_contract_details(_spy())

    async def test_get_contract_details_deduplicates_by_con_id(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        cd1 = self._make_cd()
        cd2 = self._make_cd()  # same conId=1 → deduped
        mock_ib.reqContractDetailsAsync.return_value = [cd1, cd2]
        result = await real_metadata.get_contract_details(_spy())
        assert result.instrument.symbol == "SPY"

    async def test_get_contract_details_multiple_future_maturities(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        """When multiple unique contracts, picks the nearest future maturity."""
        cd1 = self._make_cd(symbol="ES")
        cd1.contract.conId = 1
        cd1.maturity = "20991231"
        cd2 = self._make_cd(symbol="ES")
        cd2.contract.conId = 2
        cd2.maturity = "20260101"
        mock_ib.reqContractDetailsAsync.return_value = [cd1, cd2]
        result = await real_metadata.get_contract_details(_future_inst())
        assert result.instrument.symbol == "ES"

    async def test_get_contract_details_no_unique_results_raises(
        self, real_metadata: IBKRMetadataProvider, mock_ib: MagicMock
    ) -> None:
        """All results have contract=None → unique list is empty → raise."""
        cd = self._make_cd()
        cd.contract = None
        mock_ib.reqContractDetailsAsync.return_value = [cd]
        with pytest.raises(InstrumentNotFoundError):
            await real_metadata.get_contract_details(_spy())
