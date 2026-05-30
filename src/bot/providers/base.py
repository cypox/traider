"""Abstract base interfaces for all provider types."""

from abc import ABC, abstractmethod
from datetime import datetime

from bot.core.execution import ExecutionIntent
from bot.core.instruments import Instrument
from bot.providers.models import ContractDetails, YieldCurve


class MarketDataProvider(ABC):
    """Source of real-time and historical market data.

    Concrete implementations publish ``QuoteEvent`` and ``BarEvent`` objects to
    the ``EventBus`` they were initialised with.  Consumers subscribe to the bus
    rather than registering callbacks here.
    """

    @abstractmethod
    async def connect(self) -> None:  # pragma: no cover
        ...

    @abstractmethod
    async def disconnect(self) -> None:  # pragma: no cover
        ...

    @abstractmethod
    async def subscribe_quotes(self, instruments: list[Instrument]) -> None:  # pragma: no cover
        ...

    @abstractmethod
    async def subscribe_bars(
        self,
        instruments: list[Instrument],
        interval_seconds: int,
    ) -> None:  # pragma: no cover
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:  # pragma: no cover
        ...

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        ...


class ExecutionProvider(ABC):
    """Sends orders to a broker and receives confirmations."""

    @abstractmethod
    async def connect(self) -> None:  # pragma: no cover
        ...

    @abstractmethod
    async def disconnect(self) -> None:  # pragma: no cover
        ...

    @abstractmethod
    async def place_order(self, intent: ExecutionIntent) -> str:  # pragma: no cover
        """Submit an order.  Returns the broker-assigned order ID."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:  # pragma: no cover
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:  # pragma: no cover
        ...


class MetadataProvider(ABC):
    """Retrieves static instrument and contract metadata."""

    @abstractmethod
    async def get_instrument(self, symbol: str, exchange: str) -> Instrument:  # pragma: no cover
        ...

    @abstractmethod
    async def get_contract_details(  # pragma: no cover
        self, instrument: Instrument
    ) -> ContractDetails: ...


class YieldCurveProvider(ABC):
    """Provides risk-free yield curve data."""

    @abstractmethod
    async def get_curve(self, as_of: datetime) -> YieldCurve:  # pragma: no cover
        ...
