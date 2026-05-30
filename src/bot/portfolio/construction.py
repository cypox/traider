"""Portfolio construction: combines signals and sizes positions via vol targeting."""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import numpy as np
import structlog

from bot.core.instruments import Instrument
from bot.core.money import Money
from bot.core.positions import TargetPosition
from bot.core.signals import Direction, Forecast
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.events.portfolio import TargetPositionEvent
from bot.events.signals import ForecastEvent, SignalEvent
from bot.portfolio.state import PortfolioState

_logger = structlog.get_logger(__name__)

_MAX_SIGNAL_BUFFER = 100
_MIN_RETURNS_FOR_VOL = 20


class SignalCombinationMethod(Enum):
    """How signals from multiple strategies are combined into a single forecast."""

    SIMPLE_AVERAGE = "SIMPLE_AVERAGE"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    HIGHEST_CONVICTION = "HIGHEST_CONVICTION"


@dataclass(frozen=True, slots=True)
class PortfolioConstructionConfig:
    """Immutable configuration for the portfolio construction process."""

    target_annual_vol: float
    vol_lookback_days: int
    min_forecast_for_trade: float
    combination_method: SignalCombinationMethod
    strategy_weights: dict[str, float] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.strategy_weights is not None:
            total = sum(self.strategy_weights.values())
            if not math.isclose(total, 1.0, abs_tol=1e-9):
                raise ValueError(f"strategy_weights must sum to 1.0; got {total}")
        if (
            self.combination_method == SignalCombinationMethod.WEIGHTED_AVERAGE
            and self.strategy_weights is None
        ):
            raise ValueError(
                "strategy_weights must be provided when combination_method is WEIGHTED_AVERAGE"
            )


class PortfolioConstruction:
    """Combines signals from strategies and computes volatility-targeted positions.

    Subscribes to ``SignalEvent``, ``QuoteEvent``, and ``BarEvent`` on the
    provided ``EventBus``.  Each incoming signal triggers signal combination
    and, when a price is available, publishes a ``TargetPositionEvent``.
    """

    def __init__(
        self,
        config: PortfolioConstructionConfig,
        event_bus: EventBus,
        portfolio_state: PortfolioState,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._portfolio_state = portfolio_state
        self._signal_buffer: dict[Instrument, list[SignalEvent]] = {}
        self._returns_buffer: dict[Instrument, list[float]] = {}
        self._bar_last_close: dict[Instrument, Decimal] = {}
        self._price_cache: dict[Instrument, Decimal] = {}
        event_bus.subscribe(SignalEvent, self._on_signal)
        event_bus.subscribe(QuoteEvent, self._on_quote)
        event_bus.subscribe(BarEvent, self._on_bar)

    def _on_signal(self, event: SignalEvent) -> None:
        """Append incoming signal to buffer and process the instrument."""
        instrument = event.signal.instrument
        if instrument not in self._signal_buffer:
            self._signal_buffer[instrument] = []
        self._signal_buffer[instrument].append(event)
        if len(self._signal_buffer[instrument]) > _MAX_SIGNAL_BUFFER:
            self._signal_buffer[instrument] = self._signal_buffer[instrument][-_MAX_SIGNAL_BUFFER:]
        self._process_instrument(instrument)

    def _on_quote(self, event: QuoteEvent) -> None:
        """Cache the mid price for volatility-target quantity computation."""
        self._price_cache[event.instrument] = event.mid

    def _on_bar(self, event: BarEvent) -> None:
        """Compute and store close-to-close returns for vol estimation."""
        instrument = event.instrument
        if instrument in self._bar_last_close:
            prev_close = float(self._bar_last_close[instrument])
            curr_close = float(event.close)
            if prev_close > 0.0:
                ret = (curr_close - prev_close) / prev_close
                if instrument not in self._returns_buffer:
                    self._returns_buffer[instrument] = []
                self._returns_buffer[instrument].append(ret)
                max_len = self._config.vol_lookback_days * 2
                if len(self._returns_buffer[instrument]) > max_len:
                    self._returns_buffer[instrument] = self._returns_buffer[instrument][-max_len:]
        self._bar_last_close[instrument] = event.close

    def _combine_signals(self, events: list[SignalEvent]) -> float:
        """Combine signals using the configured method. Returns a value in [-1, 1]."""
        method = self._config.combination_method

        def dir_sign(direction: Direction) -> float:
            return 1.0 if direction == Direction.LONG else -1.0

        if method == SignalCombinationMethod.SIMPLE_AVERAGE:
            values = [dir_sign(e.signal.direction) * e.signal.confidence for e in events]
            result = sum(values) / len(values)
            return max(-1.0, min(1.0, result))

        if method == SignalCombinationMethod.WEIGHTED_AVERAGE:
            weights = self._config.strategy_weights or {}
            weighted_sum = sum(
                weights.get(e.source, 0.0) * dir_sign(e.signal.direction) * e.signal.confidence
                for e in events
            )
            total_weight = sum(weights.get(e.source, 0.0) for e in events)
            if total_weight == 0.0:
                return 0.0
            result = weighted_sum / total_weight
            return max(-1.0, min(1.0, result))

        # HIGHEST_CONVICTION
        best = max(events, key=lambda e: e.signal.confidence)
        return dir_sign(best.signal.direction) * best.signal.confidence

    def _process_instrument(self, instrument: Instrument) -> None:
        """Combine signals, then publish forecast and—if priced—target position."""
        events = self._signal_buffer.get(instrument, [])
        if not events:
            return

        # 1. Combine signals into a single forecast value
        forecast_value = self._combine_signals(events)

        # 2. Build Forecast and publish ForecastEvent
        now = datetime.now(UTC)
        forecast = Forecast(
            instrument=instrument,
            timestamp=now,
            value=forecast_value,
            source="portfolio_construction",
        )
        self._event_bus.publish(
            ForecastEvent(
                timestamp=now,
                source="portfolio_construction",
                forecast=forecast,
            )
        )

        # 3. Ignore weak signals
        if abs(forecast_value) < self._config.min_forecast_for_trade:
            return

        # 4. Estimate annualized instrument volatility
        returns = self._returns_buffer.get(instrument, [])
        if len(returns) < _MIN_RETURNS_FOR_VOL:
            annualized_vol = self._config.target_annual_vol
        else:
            lookback: list[float] = returns[-self._config.vol_lookback_days :]
            daily_vol = float(np.std(lookback, ddof=1))
            annualized_vol = daily_vol * math.sqrt(252)

        if annualized_vol <= 0.0:
            annualized_vol = self._config.target_annual_vol

        # 5. Compute target notional (volatility parity sizing)
        portfolio_equity = self._portfolio_state.get_cash()
        equity_amount = float(portfolio_equity.amount)
        currency = portfolio_equity.currency
        raw_notional = equity_amount * self._config.target_annual_vol / annualized_vol
        raw_notional *= abs(forecast_value)
        raw_notional = min(raw_notional, equity_amount)  # cap at 100% in one instrument

        # 6. Require a current price to compute quantity
        current_price = self._price_cache.get(instrument)
        if current_price is None:
            _logger.warning(
                "no price available for instrument, skipping target",
                symbol=instrument.symbol,
            )
            return

        price_float = float(current_price)
        if price_float <= 0.0:
            return

        raw_qty = raw_notional / price_float
        qty_int = round(raw_qty)
        if qty_int == 0:
            return

        # 7. Apply direction sign and publish TargetPositionEvent
        sign = 1 if forecast_value > 0.0 else -1
        target_qty = Decimal(sign * qty_int)
        target_notional = Money(Decimal(str(round(raw_notional, 10))), currency)
        target = TargetPosition(
            instrument=instrument,
            target_quantity=target_qty,
            target_notional=target_notional,
            source_forecast=forecast,
        )
        self._event_bus.publish(
            TargetPositionEvent(
                timestamp=datetime.now(UTC),
                source="portfolio_construction",
                target=target,
            )
        )
        _logger.debug(
            "target position computed",
            symbol=instrument.symbol,
            forecast=forecast_value,
            target_qty=str(target_qty),
            target_notional=str(target_notional.amount),
            annualized_vol=annualized_vol,
        )
