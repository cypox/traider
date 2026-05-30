"""Mock metadata provider for testing."""

from bot.core.instruments import Instrument
from bot.providers.base import MetadataProvider
from bot.providers.errors import InstrumentNotFoundError
from bot.providers.models import ContractDetails


class MockMetadataProvider(MetadataProvider):
    """In-process metadata provider backed by a static catalog.

    Pass a ``dict[str, ContractDetails]`` keyed by symbol at construction time.
    """

    def __init__(self, catalog: dict[str, ContractDetails]) -> None:
        self._catalog = catalog

    async def get_instrument(self, symbol: str, exchange: str) -> Instrument:
        """Return the instrument for *symbol*, or raise :exc:`InstrumentNotFoundError`."""
        if symbol in self._catalog:
            return self._catalog[symbol].instrument
        raise InstrumentNotFoundError(f"instrument not found: {symbol!r}")

    async def get_contract_details(self, instrument: Instrument) -> ContractDetails:
        """Return contract details for *instrument*, or raise :exc:`InstrumentNotFoundError`."""
        if instrument.symbol in self._catalog:
            return self._catalog[instrument.symbol]
        raise InstrumentNotFoundError(f"contract details not found: {instrument.symbol!r}")
