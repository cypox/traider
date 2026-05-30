"""IBKR market data provider via ib_async."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from ib_async import IB, RealTimeBarList, Ticker

from bot.core.instruments import Instrument
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.providers.base import MarketDataProvider
from bot.providers.ibkr.utils import instrument_to_contract

_logger = structlog.get_logger(__name__)


class IBKRMarketDataProvider(MarketDataProvider):
    """Connects to TWS/Gateway via ib_async, subscribes to quotes and real-time bars."""

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        event_bus: EventBus,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._event_bus = event_bus
        self._ib = IB()
        self._is_connected = False
        self._ticker_to_instrument: dict[Ticker, Instrument] = {}
        self._bid_buffer: dict[str, Decimal] = {}
        self._ask_buffer: dict[str, Decimal] = {}
        self._handler_registered = False

    async def connect(self) -> None:
        """Connect to TWS/Gateway."""
        await self._ib.connectAsync(self._host, self._port, clientId=self._client_id)
        self._is_connected = True
        _logger.info(
            "IBKR connected",
            host=self._host,
            port=self._port,
            client_id=self._client_id,
        )
        if self._port in (7496, 4001):
            _logger.warning("LIVE TRADING PORT ACTIVE", host=self._host, port=self._port)

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        self._ib.disconnect()
        self._is_connected = False
        _logger.info("IBKR disconnected")

    @property
    def ib(self) -> IB:
        """Expose the IB connection for sibling providers to share."""
        return self._ib

    async def subscribe_quotes(self, instruments: list[Instrument]) -> None:
        """Subscribe to real-time quotes for the given instruments."""
        if not self._handler_registered:
            self._ib.pendingTickersEvent.connect(self._on_pending_tickers)
            self._handler_registered = True
        for instrument in instruments:
            contract = instrument_to_contract(instrument)
            ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=False)
            self._ticker_to_instrument[ticker] = instrument

    async def subscribe_bars(
        self,
        instruments: list[Instrument],
        interval_seconds: int,
    ) -> None:
        """Subscribe to 5-second real-time bars for the given instruments."""
        if interval_seconds != 5:
            raise ValueError(
                f"IBKR only supports 5-second real-time bars via reqRealTimeBars; "
                f"got interval_seconds={interval_seconds}"
            )
        for instrument in instruments:
            contract = instrument_to_contract(instrument)
            bars = self._ib.reqRealTimeBars(contract, 5, "MIDPOINT", False)
            bars.updateEvent.connect(self._make_bar_handler(instrument))

    def _make_bar_handler(
        self,
        instrument: Instrument,
    ) -> Callable[[RealTimeBarList, bool], None]:
        """Return a closure that converts bar updates to BarEvents."""

        def handler(bars_list: RealTimeBarList, has_new_bar: bool) -> None:
            if not has_new_bar or not bars_list:
                return
            bar = bars_list[-1]
            event = BarEvent(
                timestamp=bar.time,
                source=self.name,
                instrument=instrument,
                open=Decimal(str(bar.open_)),
                high=Decimal(str(bar.high)),
                low=Decimal(str(bar.low)),
                close=Decimal(str(bar.close)),
                volume=Decimal(str(bar.volume)),
                interval_seconds=5,
            )
            self._event_bus.publish(event)

        return handler

    def _on_pending_tickers(self, tickers: set[Ticker]) -> None:
        """Handle pending tickers from IBKR; buffer bid/ask and emit QuoteEvents."""
        for ticker in tickers:
            instrument = self._ticker_to_instrument.get(ticker)
            if instrument is None:
                continue
            symbol = instrument.symbol
            if ticker.bid > 0:
                self._bid_buffer[symbol] = Decimal(str(ticker.bid))
            if ticker.ask > 0:
                self._ask_buffer[symbol] = Decimal(str(ticker.ask))
            if symbol in self._bid_buffer and symbol in self._ask_buffer:
                bid = self._bid_buffer.pop(symbol)
                ask = self._ask_buffer.pop(symbol)
                event = QuoteEvent(
                    timestamp=datetime.now(tz=UTC),
                    source=self.name,
                    instrument=instrument,
                    bid=bid,
                    ask=ask,
                )
                self._event_bus.publish(event)

    @property
    def is_connected(self) -> bool:
        """Return True if connected to TWS/Gateway."""
        return self._is_connected

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "ibkr-market-data"
