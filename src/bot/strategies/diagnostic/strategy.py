"""Diagnostic strategy for validating the full data pipeline."""

import structlog

from bot.core.instruments import Instrument
from bot.events.bus import EventBus
from bot.events.diagnostics import DiagnosticsEvent
from bot.events.market import BarEvent, QuoteEvent
from bot.providers.base import MetadataProvider
from bot.providers.errors import InstrumentNotFoundError
from bot.providers.models import ContractDetails
from bot.strategies.base import Strategy

_logger = structlog.get_logger(__name__)


class DiagnosticStrategy(Strategy):
    """Validates connectivity, market data delivery, metadata parsing, and event bus.

    This strategy NEVER publishes ``SignalEvent`` or ``OrderEvent`` and NEVER
    calls any ``ExecutionProvider``.  Its sole purpose is observability.
    """

    def __init__(
        self,
        event_bus: EventBus,
        metadata_provider: MetadataProvider,
        instruments: list[Instrument],
    ) -> None:
        super().__init__(event_bus, name="diagnostic")
        self._metadata_provider = metadata_provider
        self._instruments = instruments
        self._metadata_cache: dict[Instrument, ContractDetails] = {}

    def on_quote(self, event: QuoteEvent) -> None:
        """Log quote fields and publish a DiagnosticsEvent."""
        if event.instrument not in self._instruments:
            return

        details = self._metadata_cache.get(event.instrument)
        payload: dict[str, str | int | float | bool] = {
            "symbol": event.instrument.symbol,
            "exchange": event.instrument.exchange,
            "bid": str(event.bid),
            "ask": str(event.ask),
            "mid": str(event.mid),
        }

        if details is not None:
            payload["full_name"] = details.full_name
            payload["tick_size"] = str(details.tick_size)
            payload["multiplier"] = str(details.multiplier)
            payload["has_coupon"] = details.coupon is not None
            payload["has_maturity"] = details.maturity_date is not None
            _logger.info(
                "quote received",
                strategy="diagnostic",
                symbol=event.instrument.symbol,
                exchange=event.instrument.exchange,
                bid=str(event.bid),
                ask=str(event.ask),
                mid=str(event.mid),
                full_name=details.full_name,
                tick_size=str(details.tick_size),
                multiplier=str(details.multiplier),
                has_coupon=details.coupon is not None,
                has_maturity=details.maturity_date is not None,
            )
        else:
            _logger.info(
                "quote received",
                strategy="diagnostic",
                symbol=event.instrument.symbol,
                exchange=event.instrument.exchange,
                bid=str(event.bid),
                ask=str(event.ask),
                mid=str(event.mid),
            )

        self._event_bus.publish(
            DiagnosticsEvent(
                timestamp=event.timestamp,
                source=self.name,
                payload=payload,
            )
        )

    def on_bar(self, event: BarEvent) -> None:
        """Log bar fields and publish a DiagnosticsEvent."""
        if event.instrument not in self._instruments:
            return

        payload: dict[str, str | int | float | bool] = {
            "symbol": event.instrument.symbol,
            "open": str(event.open),
            "high": str(event.high),
            "low": str(event.low),
            "close": str(event.close),
            "volume": str(event.volume),
            "interval_seconds": event.interval_seconds,
        }
        _logger.info(
            "bar received",
            strategy="diagnostic",
            symbol=event.instrument.symbol,
            open=str(event.open),
            high=str(event.high),
            low=str(event.low),
            close=str(event.close),
            volume=str(event.volume),
            interval_seconds=event.interval_seconds,
        )
        self._event_bus.publish(
            DiagnosticsEvent(
                timestamp=event.timestamp,
                source=self.name,
                payload=payload,
            )
        )

    async def refresh_metadata(self) -> None:
        """Populate the metadata cache for all configured instruments."""
        for instrument in self._instruments:
            try:
                details = await self._metadata_provider.get_contract_details(instrument)
                self._metadata_cache[instrument] = details
            except InstrumentNotFoundError:
                _logger.warning(
                    "instrument not found during metadata refresh",
                    strategy="diagnostic",
                    symbol=instrument.symbol,
                    exchange=instrument.exchange,
                )
