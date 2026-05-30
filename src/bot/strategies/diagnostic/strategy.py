"""Diagnostic strategy for validating the full data pipeline."""

from datetime import date
from decimal import Decimal

import structlog

from bot.analytics.errors import AnalyticsError
from bot.analytics.fixed_income import (
    compute_convexity,
    compute_discount_factor,
    compute_dv01,
    compute_macaulay_duration,
    compute_modified_duration,
    compute_ytm,
)
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

            if (
                details.coupon is not None
                and details.maturity_date is not None
                and details.face_value is not None
            ):
                days_to_maturity = (details.maturity_date - date.today()).days
                if days_to_maturity <= 0:
                    _logger.warning(
                        "bond matured or expiring today, skipping analytics",
                        strategy="diagnostic",
                        symbol=event.instrument.symbol,
                    )
                else:
                    price_pct = event.mid * Decimal("100") / details.face_value.amount
                    try:
                        ytm = compute_ytm(
                            face_value=details.face_value.amount,
                            coupon_rate=details.coupon,
                            price_pct=price_pct,
                            days_to_maturity=days_to_maturity,
                        )
                        discount_factor = compute_discount_factor(ytm, days_to_maturity)
                        macaulay = compute_macaulay_duration(
                            face_value=details.face_value.amount,
                            coupon_rate=details.coupon,
                            ytm=ytm,
                            days_to_maturity=days_to_maturity,
                        )
                        modified = compute_modified_duration(macaulay, ytm)
                        dv01 = compute_dv01(details.face_value.amount, modified, price_pct)
                        convexity = compute_convexity(
                            face_value=details.face_value.amount,
                            coupon_rate=details.coupon,
                            ytm=ytm,
                            days_to_maturity=days_to_maturity,
                        )
                        _logger.info(
                            "bond analytics",
                            strategy="diagnostic",
                            symbol=event.instrument.symbol,
                            days_to_maturity=days_to_maturity,
                            ytm_pct=float(ytm * 100),
                            discount_factor=float(discount_factor),
                            macaulay_duration_years=float(macaulay),
                            modified_duration_years=float(modified),
                            dv01_usd=float(dv01),
                            convexity=float(convexity),
                        )
                        payload["days_to_maturity"] = days_to_maturity
                        payload["ytm_pct"] = float(ytm * 100)
                        payload["discount_factor"] = float(discount_factor)
                        payload["macaulay_duration_years"] = float(macaulay)
                        payload["modified_duration_years"] = float(modified)
                        payload["dv01_usd"] = float(dv01)
                        payload["convexity"] = float(convexity)
                    except AnalyticsError as exc:
                        _logger.warning(
                            "fixed income analytics failed",
                            symbol=event.instrument.symbol,
                            error=str(exc),
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
