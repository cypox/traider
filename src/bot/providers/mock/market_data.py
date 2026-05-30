"""Mock market-data provider for testing."""

from bot.core.instruments import Instrument
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.providers.base import MarketDataProvider


class MockMarketDataProvider(MarketDataProvider):
    """In-process market data provider used in unit tests.

    Call :meth:`push_quote` / :meth:`push_bar` to inject events directly into
    the ``EventBus``; events are only forwarded if the instrument was previously
    subscribed.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._is_connected = False
        self.subscribed_quote_instruments: list[Instrument] = []
        self.subscribed_bar_instruments: list[Instrument] = []

    async def connect(self) -> None:
        self._is_connected = True

    async def disconnect(self) -> None:
        self._is_connected = False

    async def subscribe_quotes(self, instruments: list[Instrument]) -> None:
        self.subscribed_quote_instruments.extend(instruments)

    async def subscribe_bars(self, instruments: list[Instrument], interval_seconds: int) -> None:
        self.subscribed_bar_instruments.extend(instruments)

    def push_quote(self, event: QuoteEvent) -> None:
        """Publish *event* to the bus only if its instrument is subscribed."""
        if event.instrument in self.subscribed_quote_instruments:
            self._bus.publish(event)

    def push_bar(self, event: BarEvent) -> None:
        """Publish *event* to the bus only if its instrument is subscribed."""
        if event.instrument in self.subscribed_bar_instruments:
            self._bus.publish(event)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def name(self) -> str:
        return "mock-market-data"
