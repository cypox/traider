"""Unit tests for PortfolioConstruction and PortfolioConstructionConfig."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.signals import Direction, Signal
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.events.portfolio import TargetPositionEvent
from bot.events.signals import ForecastEvent, SignalEvent
from bot.portfolio.construction import (
    PortfolioConstruction,
    PortfolioConstructionConfig,
    SignalCombinationMethod,
)
from bot.portfolio.state import PortfolioState

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_INITIAL_CASH = Decimal("100000")


def _spy() -> Instrument:
    return Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _qqq() -> Instrument:
    return Instrument(symbol="QQQ", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def make_config(
    combination_method: SignalCombinationMethod = SignalCombinationMethod.SIMPLE_AVERAGE,
    min_forecast_for_trade: float = 0.1,
    strategy_weights: dict[str, float] | None = None,
    target_annual_vol: float = 0.15,
    vol_lookback_days: int = 60,
) -> PortfolioConstructionConfig:
    return PortfolioConstructionConfig(
        target_annual_vol=target_annual_vol,
        vol_lookback_days=vol_lookback_days,
        min_forecast_for_trade=min_forecast_for_trade,
        combination_method=combination_method,
        strategy_weights=strategy_weights,
    )


def make_signal_event(
    instrument: Instrument,
    direction: Direction,
    confidence: float,
    source: str = "strategy",
) -> SignalEvent:
    signal = Signal(
        instrument=instrument,
        timestamp=_TS,
        direction=direction,
        confidence=confidence,
        reason="test",
    )
    return SignalEvent(timestamp=_TS, source=source, signal=signal)


def make_bar(instrument: Instrument, close: float, ts_offset: int = 0) -> BarEvent:
    ts = _TS + timedelta(days=ts_offset)
    c = Decimal(str(close))
    return BarEvent(
        timestamp=ts,
        source="provider",
        instrument=instrument,
        open=c,
        high=c,
        low=c,
        close=c,
        volume=Decimal("1000000"),
        interval_seconds=86400,
    )


def make_quote(instrument: Instrument, price: float) -> QuoteEvent:
    p = Decimal(str(price))
    half_spread = Decimal("0.01")
    return QuoteEvent(
        timestamp=_TS,
        source="provider",
        instrument=instrument,
        bid=p - half_spread,
        ask=p + half_spread,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spy() -> Instrument:
    return _spy()


@pytest.fixture
def qqq() -> Instrument:
    return _qqq()


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def portfolio_state(bus: EventBus) -> PortfolioState:
    return PortfolioState(Money(_INITIAL_CASH, "USD"), bus)


# ---------------------------------------------------------------------------
# PortfolioConstructionConfig validation
# ---------------------------------------------------------------------------


class TestPortfolioConstructionConfig:
    def test_weighted_average_weights_not_sum_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            PortfolioConstructionConfig(
                target_annual_vol=0.15,
                vol_lookback_days=60,
                min_forecast_for_trade=0.1,
                combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE,
                strategy_weights={"a": 0.6, "b": 0.3},  # sums to 0.9
            )

    def test_weighted_average_no_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy_weights must be provided"):
            PortfolioConstructionConfig(
                target_annual_vol=0.15,
                vol_lookback_days=60,
                min_forecast_for_trade=0.1,
                combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE,
                strategy_weights=None,
            )

    def test_valid_config_with_weights(self) -> None:
        config = PortfolioConstructionConfig(
            target_annual_vol=0.15,
            vol_lookback_days=60,
            min_forecast_for_trade=0.1,
            combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE,
            strategy_weights={"a": 0.6, "b": 0.4},
        )
        assert config.strategy_weights == {"a": 0.6, "b": 0.4}

    def test_simple_average_no_weights_ok(self) -> None:
        config = make_config()
        assert config.strategy_weights is None


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------


class TestSignalCombination:
    def test_simple_average_single_long(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(min_forecast_for_trade=0.0)
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        bus.publish(make_signal_event(spy, Direction.LONG, 0.8))

        assert len(forecasts) == 1
        assert abs(forecasts[0].forecast.value - 0.8) < 1e-9

    def test_simple_average_mixed_signals(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(min_forecast_for_trade=0.0)
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        bus.publish(make_signal_event(spy, Direction.LONG, 0.6, source="strat_a"))
        bus.publish(make_signal_event(spy, Direction.SHORT, 0.4, source="strat_b"))

        # After 2nd signal: (0.6 + (-0.4)) / 2 = 0.1
        assert len(forecasts) == 2
        assert abs(forecasts[-1].forecast.value - 0.1) < 1e-9

    def test_weighted_average_weights_applied(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(
            combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE,
            strategy_weights={"strat_a": 0.7, "strat_b": 0.3},
            min_forecast_for_trade=0.0,
        )
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        bus.publish(make_signal_event(spy, Direction.LONG, 1.0, source="strat_a"))
        bus.publish(make_signal_event(spy, Direction.SHORT, 1.0, source="strat_b"))

        # signals=[strat_a LONG 1.0, strat_b SHORT 1.0]
        # weighted_sum = 0.7*(+1) + 0.3*(-1) = 0.4; total_weight = 1.0 → 0.4
        assert abs(forecasts[-1].forecast.value - 0.4) < 1e-9

    def test_weighted_average_unknown_source_treated_as_zero_weight(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(
            combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE,
            strategy_weights={"known": 1.0},
            min_forecast_for_trade=0.0,
        )
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        # Two signals: one from known (LONG 0.5) and one from unknown (SHORT 1.0)
        # unknown weight = 0, so only known contributes; total_weight = 1.0
        # forecast = (1.0 * 0.5) / 1.0 = 0.5
        bus.publish(make_signal_event(spy, Direction.LONG, 0.5, source="known"))
        bus.publish(make_signal_event(spy, Direction.SHORT, 1.0, source="unknown"))

        assert abs(forecasts[-1].forecast.value - 0.5) < 1e-9

    def test_highest_conviction_picks_max_confidence(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(
            combination_method=SignalCombinationMethod.HIGHEST_CONVICTION,
            min_forecast_for_trade=0.0,
        )
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        bus.publish(make_signal_event(spy, Direction.LONG, 0.3, source="strat_a"))
        bus.publish(make_signal_event(spy, Direction.SHORT, 0.9, source="strat_b"))

        # Best is SHORT 0.9 → forecast = -0.9
        assert abs(forecasts[-1].forecast.value - (-0.9)) < 1e-9

    def test_no_signals_no_forecast(self, bus: EventBus, portfolio_state: PortfolioState) -> None:
        config = make_config()
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        assert len(forecasts) == 0

    def test_signal_clamp_above_one(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        # SIMPLE_AVERAGE: two LONG signals with max confidence → exactly 1.0
        config = make_config(min_forecast_for_trade=0.0)
        PortfolioConstruction(config, bus, portfolio_state)

        forecasts: list[ForecastEvent] = []
        bus.subscribe(ForecastEvent, lambda e: forecasts.append(e))

        bus.publish(make_signal_event(spy, Direction.LONG, 1.0))
        bus.publish(make_signal_event(spy, Direction.LONG, 1.0))

        # (1.0 + 1.0) / 2 = 1.0, clamped to 1.0
        assert forecasts[-1].forecast.value == 1.0


# ---------------------------------------------------------------------------
# Min-forecast threshold
# ---------------------------------------------------------------------------


class TestMinForecastThreshold:
    def test_forecast_below_threshold_no_target(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(min_forecast_for_trade=0.5)
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        bus.publish(make_signal_event(spy, Direction.LONG, 0.3))  # forecast = 0.3 < 0.5

        assert len(targets) == 0

    def test_forecast_at_threshold_no_target(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        # Strict less-than: exactly at threshold is excluded (abs(v) < min)
        config = make_config(min_forecast_for_trade=0.3)
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        bus.publish(make_signal_event(spy, Direction.LONG, 0.3))  # forecast = 0.3, not < 0.3

        # abs(0.3) is NOT < 0.3, so we proceed
        assert len(targets) == 1

    def test_forecast_above_threshold_publishes_target(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(min_forecast_for_trade=0.1)
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        bus.publish(make_signal_event(spy, Direction.LONG, 0.8))

        assert len(targets) == 1


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------


class TestVolTargeting:
    def test_fewer_than_20_returns_uses_target_vol(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(
            target_annual_vol=0.15,
            min_forecast_for_trade=0.0,
        )
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        # Publish 5 bars → 4 returns (< 20), so fallback to target_annual_vol
        for i in range(5):
            bus.publish(make_bar(spy, 100.0 + i * 0.01, ts_offset=i))

        bus.publish(make_signal_event(spy, Direction.LONG, 1.0))

        assert len(targets) == 1
        # target_notional = equity * 0.15 / 0.15 * 1.0 = 100000; qty = 100000/100 = 1000
        assert targets[0].target.target_quantity == Decimal("1000")

    def test_high_vol_instrument_smaller_quantity(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument, qqq: Instrument
    ) -> None:
        config = make_config(
            target_annual_vol=0.15,
            vol_lookback_days=60,
            min_forecast_for_trade=0.0,
        )
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        bus.publish(make_quote(qqq, 100.0))

        # Low-vol bars for SPY: ±0.1% daily moves → small std
        price = 100.0
        for i in range(22):
            bus.publish(make_bar(spy, price, ts_offset=i))
            price = price * (1.001 if i % 2 == 0 else 0.999)

        # High-vol bars for QQQ: ±2% daily moves → large std
        price = 100.0
        for i in range(22):
            bus.publish(make_bar(qqq, price, ts_offset=i))
            price = price * (1.02 if i % 2 == 0 else 0.98)

        bus.publish(make_signal_event(spy, Direction.LONG, 1.0))
        bus.publish(make_signal_event(qqq, Direction.LONG, 1.0))

        spy_qty = abs(int(targets[-2].target.target_quantity))
        qqq_qty = abs(int(targets[-1].target.target_quantity))

        assert spy_qty > qqq_qty

    def test_long_forecast_positive_quantity(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(min_forecast_for_trade=0.0)
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        bus.publish(make_signal_event(spy, Direction.LONG, 0.8))

        assert targets[-1].target.target_quantity > Decimal("0")

    def test_short_forecast_negative_quantity(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config(min_forecast_for_trade=0.0)
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        bus.publish(make_signal_event(spy, Direction.SHORT, 0.8))

        assert targets[-1].target.target_quantity < Decimal("0")

    def test_zero_vol_instrument_uses_fallback_vol(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        """When all returns are zero (flat price), fall back to target_annual_vol."""
        config = make_config(
            target_annual_vol=0.15,
            min_forecast_for_trade=0.0,
        )
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        bus.publish(make_quote(spy, 100.0))
        # 22 bars with identical price → 21 zero returns → daily_vol = 0.0
        for i in range(22):
            bus.publish(make_bar(spy, 100.0, ts_offset=i))

        bus.publish(make_signal_event(spy, Direction.LONG, 1.0))

        assert len(targets) == 1
        # Falls back to target_annual_vol; qty = equity / price = 1000
        assert targets[0].target.target_quantity == Decimal("1000")

    def test_returns_buffer_bounded(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        """Returns buffer is capped at vol_lookback_days * 2 entries."""
        config = make_config(vol_lookback_days=5, min_forecast_for_trade=0.0)
        pc = PortfolioConstruction(config, bus, portfolio_state)

        bus.publish(make_quote(spy, 100.0))
        # Publish more bars than the buffer limit (5*2=10)
        price = 100.0
        for i in range(20):
            bus.publish(make_bar(spy, price + i * 0.1, ts_offset=i))

        assert len(pc._returns_buffer.get(spy, [])) <= 10  # noqa: SLF001


# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------


class TestPriceCache:
    def test_no_price_no_target_and_warning_logged(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config()
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        with capture_logs() as logs:
            bus.publish(make_signal_event(spy, Direction.LONG, 0.8))

        assert len(targets) == 0
        warning_logs = [log for log in logs if log.get("log_level") == "warning"]
        assert any(log.get("symbol") == spy.symbol for log in warning_logs)

    def test_quote_event_updates_price_cache(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config()
        PortfolioConstruction(config, bus, portfolio_state)

        targets: list[TargetPositionEvent] = []
        bus.subscribe(TargetPositionEvent, lambda e: targets.append(e))

        # Signal with no price → no target
        bus.publish(make_signal_event(spy, Direction.LONG, 0.8))
        assert len(targets) == 0

        # Provide price
        bus.publish(make_quote(spy, 100.0))

        # Second signal → target published now
        bus.publish(make_signal_event(spy, Direction.LONG, 0.8))
        assert len(targets) == 1

    def test_quote_mid_price_is_used(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        """QuoteEvent mid = (bid + ask) / 2 is stored in price cache."""
        config = make_config(min_forecast_for_trade=0.0)
        pc = PortfolioConstruction(config, bus, portfolio_state)

        # bid=99.99, ask=100.01 → mid=100.00
        bus.publish(
            QuoteEvent(
                timestamp=_TS,
                source="provider",
                instrument=spy,
                bid=Decimal("99.99"),
                ask=Decimal("100.01"),
            )
        )
        assert pc._price_cache[spy] == Decimal("100.00")  # noqa: SLF001


# ---------------------------------------------------------------------------
# Bar event returns computation
# ---------------------------------------------------------------------------


class TestBarEventReturns:
    def test_first_bar_does_not_compute_return(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config()
        pc = PortfolioConstruction(config, bus, portfolio_state)

        bus.publish(make_bar(spy, 100.0, ts_offset=0))

        assert len(pc._returns_buffer.get(spy, [])) == 0  # noqa: SLF001

    def test_two_bars_produce_one_return(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        config = make_config()
        pc = PortfolioConstruction(config, bus, portfolio_state)

        bus.publish(make_bar(spy, 100.0, ts_offset=0))
        bus.publish(make_bar(spy, 101.0, ts_offset=1))

        returns = pc._returns_buffer.get(spy, [])  # noqa: SLF001
        assert len(returns) == 1
        assert abs(returns[0] - 0.01) < 1e-9

    def test_signal_buffer_bounded(
        self, bus: EventBus, portfolio_state: PortfolioState, spy: Instrument
    ) -> None:
        """Signal buffer is capped at 100 entries per instrument."""
        config = make_config(min_forecast_for_trade=0.0)
        pc = PortfolioConstruction(config, bus, portfolio_state)

        bus.publish(make_quote(spy, 100.0))
        for _ in range(110):
            bus.publish(make_signal_event(spy, Direction.LONG, 0.5))

        assert len(pc._signal_buffer.get(spy, [])) == 100  # noqa: SLF001
