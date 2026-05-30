"""Unit tests for the backtesting engine — 95% coverage target."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot.backtest.data import BarData, HistoricalDataset
from bot.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from bot.backtest.walkforward import WalkForwardConfig, WalkForwardResult, run_walkforward
from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.signals import Direction, Signal
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.events.signals import SignalEvent
from bot.portfolio.construction import (
    PortfolioConstruction,
    PortfolioConstructionConfig,
    SignalCombinationMethod,
)
from bot.portfolio.state import PortfolioState
from bot.risk.config import RiskConfig
from bot.risk.engine import RiskEngine
from bot.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_BASE_DATE = datetime(2024, 1, 1, tzinfo=UTC)
_N_DAYS = 30
_INITIAL_CASH = Money(Decimal("100000"), "USD")


def _spy() -> Instrument:
    return Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _qqq() -> Instrument:
    return Instrument(symbol="QQQ", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def make_bars(n_days: int = _N_DAYS) -> list[BarData]:
    """Generate synthetic daily OHLCV bars for SPY and QQQ."""
    instruments = [_spy(), _qqq()]
    bars: list[BarData] = []
    for day in range(n_days):
        ts = _BASE_DATE + timedelta(days=day)
        for inst in instruments:
            base = Decimal("100") + Decimal(day)
            bars.append(
                BarData(
                    instrument=inst,
                    timestamp=ts,
                    open=base,
                    high=base + Decimal("5"),
                    low=base - Decimal("1"),
                    close=base + Decimal("2"),
                    volume=Decimal("1000000"),
                )
            )
    return bars


def make_backtest_config(
    fill_on: str = "close",
    slippage_bps: Decimal = Decimal("10"),
    commission_per_share: Decimal = Decimal("0.01"),
) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=_INITIAL_CASH,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        fill_on=fill_on,
    )


def make_risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_usd=Decimal("200000"),
        max_gross_exposure_usd=Decimal("500000"),
        max_drawdown_pct=Decimal("0.50"),
        max_concentration_pct=Decimal("1"),
        daily_loss_limit_usd=Decimal("50000"),
    )


def make_pc_config() -> PortfolioConstructionConfig:
    return PortfolioConstructionConfig(
        target_annual_vol=0.20,
        vol_lookback_days=20,
        min_forecast_for_trade=0.01,
        combination_method=SignalCombinationMethod.SIMPLE_AVERAGE,
    )


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class DoNothingStrategy(Strategy):
    """Never emits signals."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus, name="do_nothing")

    def on_quote(self, event: QuoteEvent) -> None:
        pass

    def on_bar(self, event: BarEvent) -> None:
        pass


class BuyEverythingStrategy(Strategy):
    """Emits a LONG signal for every bar's instrument."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus, name="buy_everything")

    def on_quote(self, event: QuoteEvent) -> None:
        pass

    def on_bar(self, event: BarEvent) -> None:
        signal = Signal(
            instrument=event.instrument,
            timestamp=event.timestamp,
            direction=Direction.LONG,
            confidence=1.0,
            reason="buy everything",
        )
        self._event_bus.publish(
            SignalEvent(
                timestamp=event.timestamp,
                source=self._name,
                signal=signal,
            )
        )


class ShortEverythingStrategy(Strategy):
    """Emits a SHORT signal for every bar's instrument."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus, name="short_everything")

    def on_quote(self, event: QuoteEvent) -> None:
        pass

    def on_bar(self, event: BarEvent) -> None:
        signal = Signal(
            instrument=event.instrument,
            timestamp=event.timestamp,
            direction=Direction.SHORT,
            confidence=1.0,
            reason="short everything",
        )
        self._event_bus.publish(
            SignalEvent(
                timestamp=event.timestamp,
                source=self._name,
                signal=signal,
            )
        )


class TimestampRecorderStrategy(Strategy):
    """Records all bar timestamps in order of receipt."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus, name="recorder")
        self.received_timestamps: list[datetime] = []

    def on_quote(self, event: QuoteEvent) -> None:
        pass

    def on_bar(self, event: BarEvent) -> None:
        self.received_timestamps.append(event.timestamp)


# ---------------------------------------------------------------------------
# Engine assembly helper
# ---------------------------------------------------------------------------


def build_engine(
    strategy_factory: Callable[[EventBus], Strategy],
    bars: list[BarData] | None = None,
    fill_on: str = "close",
    slippage_bps: Decimal = Decimal("10"),
    commission_per_share: Decimal = Decimal("0.01"),
) -> tuple[BacktestEngine, PortfolioState]:
    """Wire a :class:`BacktestEngine` with a single strategy and return it.

    *strategy_factory* is called with the shared :class:`EventBus` so the
    strategy is subscribed to the same bus the engine publishes on.
    """
    if bars is None:
        bars = make_bars()
    dataset = HistoricalDataset(bars)
    config = BacktestConfig(
        initial_cash=_INITIAL_CASH,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        fill_on=fill_on,
    )
    bus = EventBus()
    portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
    strategy = strategy_factory(bus)
    pc = PortfolioConstruction(
        config=make_pc_config(),
        event_bus=bus,
        portfolio_state=portfolio_state,
    )
    risk = RiskEngine(
        config=make_risk_config(),
        event_bus=bus,
        portfolio_state=portfolio_state,
    )
    engine = BacktestEngine(
        config=config,
        dataset=dataset,
        event_bus=bus,
        strategies=[strategy],
        portfolio_construction=pc,
        risk_engine=risk,
        portfolio_state=portfolio_state,
    )
    return engine, portfolio_state


# ---------------------------------------------------------------------------
# HistoricalDataset tests
# ---------------------------------------------------------------------------


class TestHistoricalDataset:
    def test_all_bars_sorted_chronological_order(self) -> None:
        dataset = HistoricalDataset(make_bars())
        sorted_bars = dataset.get_all_bars_sorted()
        timestamps = [b.timestamp for b in sorted_bars]
        assert timestamps == sorted(timestamps)

    def test_all_bars_sorted_symbol_tiebreak(self) -> None:
        dataset = HistoricalDataset(make_bars())
        sorted_bars = dataset.get_all_bars_sorted()
        for i in range(len(sorted_bars) - 1):
            a, b = sorted_bars[i], sorted_bars[i + 1]
            if a.timestamp == b.timestamp:
                assert a.instrument.symbol <= b.instrument.symbol

    def test_get_bars_range_inclusive(self) -> None:
        dataset = HistoricalDataset(make_bars(5))
        start = _BASE_DATE + timedelta(days=1)
        end = _BASE_DATE + timedelta(days=3)
        result = dataset.get_bars(_spy(), start, end)
        assert len(result) == 3
        assert all(start <= b.timestamp <= end for b in result)

    def test_get_bars_chronological(self) -> None:
        dataset = HistoricalDataset(make_bars(5))
        result = dataset.get_bars(_spy(), _BASE_DATE, _BASE_DATE + timedelta(days=4))
        timestamps = [b.timestamp for b in result]
        assert timestamps == sorted(timestamps)

    def test_get_bars_empty_range(self) -> None:
        dataset = HistoricalDataset(make_bars(5))
        future = _BASE_DATE + timedelta(days=100)
        result = dataset.get_bars(_spy(), future, future + timedelta(days=1))
        assert result == []

    def test_split_no_overlap(self) -> None:
        dataset = HistoricalDataset(make_bars())
        split_point = _BASE_DATE + timedelta(days=14)
        in_sample, out_sample = dataset.split(split_point)
        in_keys = {(b.instrument, b.timestamp) for b in in_sample.get_all_bars_sorted()}
        out_keys = {(b.instrument, b.timestamp) for b in out_sample.get_all_bars_sorted()}
        assert in_keys & out_keys == set()

    def test_split_all_bars_accounted_for(self) -> None:
        dataset = HistoricalDataset(make_bars())
        split_point = _BASE_DATE + timedelta(days=14)
        in_sample, out_sample = dataset.split(split_point)
        assert in_sample.bar_count + out_sample.bar_count == dataset.bar_count

    def test_split_boundary_bar_in_sample(self) -> None:
        dataset = HistoricalDataset(make_bars(5))
        split_point = _BASE_DATE + timedelta(days=2)
        in_sample, out_sample = dataset.split(split_point)
        # Bar at split_point belongs to in_sample (inclusive)
        in_ts = {b.timestamp for b in in_sample.get_all_bars_sorted()}
        assert split_point in in_ts

    def test_date_range_returns_min_max(self) -> None:
        dataset = HistoricalDataset(make_bars(5))
        earliest, latest = dataset.date_range
        assert earliest == _BASE_DATE
        assert latest == _BASE_DATE + timedelta(days=4)

    def test_bar_count(self) -> None:
        dataset = HistoricalDataset(make_bars(10))
        assert dataset.bar_count == 20  # 2 instruments × 10 days

    def test_instruments_property(self) -> None:
        dataset = HistoricalDataset(make_bars(5))
        assert dataset.instruments == {_spy(), _qqq()}

    def test_all_bars_sorted_total_count(self) -> None:
        dataset = HistoricalDataset(make_bars())
        assert len(dataset.get_all_bars_sorted()) == _N_DAYS * 2


# ---------------------------------------------------------------------------
# BacktestEngine — do-nothing strategy
# ---------------------------------------------------------------------------


class TestBacktestEngineDoNothing:
    def test_runs_without_error(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_equity_curve_one_entry_per_bar(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        expected_bars = _N_DAYS * 2  # 2 instruments
        assert len(result.equity_curve) == expected_bars

    def test_final_equity_equals_initial_cash(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert all(equity == _INITIAL_CASH.amount for _, equity in result.equity_curve)

    def test_no_trades_generated(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert result.trades == ()

    def test_total_commission_is_zero(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert result.total_commission.amount == Decimal("0")

    def test_sharpe_zero_with_flat_equity(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert result.sharpe_ratio == Decimal("0")

    def test_max_drawdown_zero_with_flat_equity(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert result.max_drawdown_pct == Decimal("0")

    def test_total_return_zero_with_flat_equity(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert result.total_return_pct == Decimal("0")


# ---------------------------------------------------------------------------
# BacktestEngine — buy-everything strategy
# ---------------------------------------------------------------------------


class TestBacktestEngineBuyEverything:
    def test_fills_are_generated(self) -> None:
        engine, _ = build_engine(BuyEverythingStrategy)
        result = engine.run()
        assert len(result.trades) > 0

    def test_equity_curve_changes_after_fills(self) -> None:
        engine, _ = build_engine(BuyEverythingStrategy)
        result = engine.run()
        equities = [eq for _, eq in result.equity_curve]
        # After first fill, equity deviates from initial cash
        assert not all(eq == _INITIAL_CASH.amount for eq in equities)

    def test_commissions_are_nonzero(self) -> None:
        engine, _ = build_engine(BuyEverythingStrategy, commission_per_share=Decimal("0.01"))
        result = engine.run()
        assert result.total_commission.amount > Decimal("0")

    def test_equity_curve_length_matches_bar_count(self) -> None:
        engine, _ = build_engine(BuyEverythingStrategy)
        result = engine.run()
        assert len(result.equity_curve) == _N_DAYS * 2

    def test_all_fills_have_correct_side(self) -> None:
        engine, _ = build_engine(BuyEverythingStrategy, commission_per_share=Decimal("0"))
        result = engine.run()
        # With a buy-everything strategy, all initial fills should be LONG
        long_fills = [f for f in result.trades if f.side == Direction.LONG]
        assert len(long_fills) > 0


# ---------------------------------------------------------------------------
# Fill price slippage tests
# ---------------------------------------------------------------------------


def _make_single_bar_engine(
    strategy: Strategy,
    close: Decimal,
    slippage_bps: Decimal,
    bus: EventBus,
) -> BacktestEngine:
    """Build an engine with a single SPY bar at *close* price."""
    inst = _spy()
    bar = BarData(
        instrument=inst,
        timestamp=_BASE_DATE,
        open=close,
        high=close + Decimal("5"),
        low=close - Decimal("1"),
        close=close,
        volume=Decimal("1000000"),
    )
    dataset = HistoricalDataset([bar])
    config = BacktestConfig(
        initial_cash=_INITIAL_CASH,
        slippage_bps=slippage_bps,
        commission_per_share=Decimal("0"),
    )
    portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
    pc = PortfolioConstruction(
        config=make_pc_config(), event_bus=bus, portfolio_state=portfolio_state
    )
    risk = RiskEngine(config=make_risk_config(), event_bus=bus, portfolio_state=portfolio_state)
    return BacktestEngine(
        config=config,
        dataset=dataset,
        event_bus=bus,
        strategies=[strategy],
        portfolio_construction=pc,
        risk_engine=risk,
        portfolio_state=portfolio_state,
    )


class TestSlippage:
    def test_buy_fill_price_includes_positive_slippage(self) -> None:
        """Buy fill = close * (1 + slippage_bps / 10000)."""
        bus = EventBus()
        strategy = BuyEverythingStrategy(bus)
        close = Decimal("100")
        engine = _make_single_bar_engine(strategy, close, Decimal("10"), bus)
        result = engine.run()

        assert len(result.trades) > 0
        buy_fill = result.trades[0]
        assert buy_fill.side == Direction.LONG
        expected = close * (Decimal("1") + Decimal("10") / Decimal("10000"))
        assert buy_fill.fill_price.amount == expected

    def test_sell_fill_price_includes_negative_slippage(self) -> None:
        """Sell fill = close * (1 - slippage_bps / 10000)."""
        bus = EventBus()
        strategy = ShortEverythingStrategy(bus)
        close = Decimal("100")
        engine = _make_single_bar_engine(strategy, close, Decimal("10"), bus)
        result = engine.run()

        assert len(result.trades) > 0
        sell_fill = result.trades[0]
        assert sell_fill.side == Direction.SHORT
        expected = close * (Decimal("1") - Decimal("10") / Decimal("10000"))
        assert sell_fill.fill_price.amount == expected

    def test_zero_slippage_fill_at_close(self) -> None:
        bus = EventBus()
        strategy = BuyEverythingStrategy(bus)
        close = Decimal("150")
        engine = _make_single_bar_engine(strategy, close, Decimal("0"), bus)
        result = engine.run()

        assert len(result.trades) > 0
        assert result.trades[0].fill_price.amount == close


# ---------------------------------------------------------------------------
# fill_on="open" tests
# ---------------------------------------------------------------------------


class TestFillOnOpen:
    def test_fills_at_next_bar_open(self) -> None:
        """Approvals from bar T are filled at bar T+1's open."""
        inst = _spy()
        close_day0 = Decimal("100")
        open_day1 = Decimal("105")
        bars = [
            BarData(
                instrument=inst,
                timestamp=_BASE_DATE,
                open=close_day0,
                high=close_day0 + Decimal("5"),
                low=close_day0 - Decimal("1"),
                close=close_day0,
                volume=Decimal("1000000"),
            ),
            BarData(
                instrument=inst,
                timestamp=_BASE_DATE + timedelta(days=1),
                open=open_day1,
                high=open_day1 + Decimal("5"),
                low=open_day1 - Decimal("1"),
                close=open_day1 + Decimal("3"),
                volume=Decimal("1000000"),
            ),
        ]
        engine, _ = build_engine(
            BuyEverythingStrategy, bars=bars, fill_on="open", slippage_bps=Decimal("0")
        )
        result = engine.run()

        # First fill should use day-1's open (approval from day-0 deferred)
        assert len(result.trades) > 0
        first_fill = result.trades[0]
        assert first_fill.side == Direction.LONG
        assert first_fill.fill_price.amount == open_day1

    def test_fill_on_open_vs_close_differ(self) -> None:
        """fill_on='open' and fill_on='close' produce different fill prices."""
        bars = make_bars(5)

        engine_close, _ = build_engine(
            BuyEverythingStrategy, bars=bars, fill_on="close", slippage_bps=Decimal("0")
        )
        result_close = engine_close.run()

        engine_open, _ = build_engine(
            BuyEverythingStrategy, bars=bars, fill_on="open", slippage_bps=Decimal("0")
        )
        result_open = engine_open.run()

        # Both should produce fills but at different prices
        assert len(result_close.trades) > 0
        assert len(result_open.trades) > 0
        close_prices = {f.fill_price.amount for f in result_close.trades}
        open_prices = {f.fill_price.amount for f in result_open.trades}
        assert close_prices != open_prices


# ---------------------------------------------------------------------------
# No-lookahead test
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_strategy_sees_bars_in_order(self) -> None:
        """Bar timestamps received by strategy are monotonically non-decreasing."""
        bars = make_bars(5)
        dataset = HistoricalDataset(bars)
        bus = EventBus()
        strategy = TimestampRecorderStrategy(bus)
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        pc = PortfolioConstruction(
            config=make_pc_config(), event_bus=bus, portfolio_state=portfolio_state
        )
        risk = RiskEngine(config=make_risk_config(), event_bus=bus, portfolio_state=portfolio_state)
        engine = BacktestEngine(
            config=make_backtest_config(),
            dataset=dataset,
            event_bus=bus,
            strategies=[strategy],
            portfolio_construction=pc,
            risk_engine=risk,
            portfolio_state=portfolio_state,
        )
        engine.run()

        assert len(strategy.received_timestamps) == dataset.bar_count
        for i in range(1, len(strategy.received_timestamps)):
            assert strategy.received_timestamps[i] >= strategy.received_timestamps[i - 1]

    def test_no_future_timestamps_seen(self) -> None:
        """Strategy never sees a bar from the future relative to the current bar."""
        bars = make_bars(5)
        sorted_bars = HistoricalDataset(bars).get_all_bars_sorted()
        bus = EventBus()
        strategy = TimestampRecorderStrategy(bus)
        portfolio_state = PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)
        pc = PortfolioConstruction(
            config=make_pc_config(), event_bus=bus, portfolio_state=portfolio_state
        )
        risk = RiskEngine(config=make_risk_config(), event_bus=bus, portfolio_state=portfolio_state)
        engine = BacktestEngine(
            config=make_backtest_config(),
            dataset=HistoricalDataset(bars),
            event_bus=bus,
            strategies=[strategy],
            portfolio_construction=pc,
            risk_engine=risk,
            portfolio_state=portfolio_state,
        )
        engine.run()

        # The i-th timestamp received should equal the i-th bar's timestamp
        for i, received_ts in enumerate(strategy.received_timestamps):
            assert received_ts == sorted_bars[i].timestamp


# ---------------------------------------------------------------------------
# BacktestResult.summary()
# ---------------------------------------------------------------------------


class TestBacktestResultSummary:
    def test_summary_contains_key_metrics(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        summary = result.summary()
        assert "BacktestResult:" in summary
        assert "Sharpe" in summary
        assert "Drawdown" in summary
        assert "Commission" in summary

    def test_summary_is_string(self) -> None:
        engine, _ = build_engine(DoNothingStrategy)
        result = engine.run()
        assert isinstance(result.summary(), str)


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------


def _make_wf_factories(
    strategy_class: type[Strategy],
) -> tuple[
    Callable[[EventBus], list[Strategy]],
    Callable[[EventBus, PortfolioState], PortfolioConstruction],
    Callable[[EventBus, PortfolioState], RiskEngine],
]:
    """Return factory callables for walk-forward testing."""

    def strategy_factory(bus: EventBus) -> list[Strategy]:
        return [strategy_class(bus)]

    def pc_factory(bus: EventBus, ps: PortfolioState) -> PortfolioConstruction:
        return PortfolioConstruction(config=make_pc_config(), event_bus=bus, portfolio_state=ps)

    def risk_factory(bus: EventBus, ps: PortfolioState) -> RiskEngine:
        return RiskEngine(config=make_risk_config(), event_bus=bus, portfolio_state=ps)

    return strategy_factory, pc_factory, risk_factory


class TestWalkForward:
    def test_correct_number_of_windows(self) -> None:
        """With 30 days, train=10, test=5, expect 4 windows."""
        dataset = HistoricalDataset(make_bars(_N_DAYS))
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=9)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        assert len(result.windows) == 4

    def test_each_window_independent_equity(self) -> None:
        """Each window's equity starts at initial_cash (fresh PortfolioState)."""
        dataset = HistoricalDataset(make_bars(_N_DAYS))
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=9)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        for window in result.windows:
            # Do-nothing strategy: all equity values equal initial cash
            assert all(eq == _INITIAL_CASH.amount for _, eq in window.equity_curve)

    def test_combined_equity_is_concatenation_of_windows(self) -> None:
        """Combined equity curve is the concatenation of all window curves."""
        dataset = HistoricalDataset(make_bars(_N_DAYS))
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=9)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        expected_total = sum(len(w.equity_curve) for w in result.windows)
        assert len(result.combined_equity_curve) == expected_total

    def test_combined_equity_matches_per_window_curves(self) -> None:
        """Combined equity equals concatenated per-window equity curves."""
        dataset = HistoricalDataset(make_bars(_N_DAYS))
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=9)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        concatenated = tuple(entry for w in result.windows for entry in w.equity_curve)
        assert result.combined_equity_curve == concatenated

    def test_empty_dataset_returns_empty_result(self) -> None:
        dataset = HistoricalDataset([])
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=5)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        assert isinstance(result, WalkForwardResult)
        assert result.windows == ()
        assert result.combined_equity_curve == ()

    def test_min_train_days_filter_skips_windows(self) -> None:
        """Windows without enough training data are excluded."""
        dataset = HistoricalDataset(make_bars(15))
        # Require 20 training days but window only has 10 → all windows skipped
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=20)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        assert result.windows == ()

    def test_combined_sharpe_and_drawdown_are_decimal(self) -> None:
        dataset = HistoricalDataset(make_bars(_N_DAYS))
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=9)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(DoNothingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        assert isinstance(result.combined_sharpe, Decimal)
        assert isinstance(result.combined_max_drawdown, Decimal)

    def test_walkforward_with_buy_strategy_generates_fills(self) -> None:
        dataset = HistoricalDataset(make_bars(_N_DAYS))
        wf_config = WalkForwardConfig(train_days=10, test_days=5, min_train_days=9)
        backtest_config = make_backtest_config()
        sf, pc_f, rf = _make_wf_factories(BuyEverythingStrategy)

        result = run_walkforward(wf_config, backtest_config, dataset, sf, pc_f, rf)

        total_trades = sum(len(w.trades) for w in result.windows)
        assert total_trades > 0
