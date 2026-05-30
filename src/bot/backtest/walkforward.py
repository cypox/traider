"""Walk-forward testing: slides a rolling window across a historical dataset."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from bot.backtest.data import HistoricalDataset
from bot.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    _compute_max_drawdown,
    _compute_sharpe,
)
from bot.events.bus import EventBus
from bot.portfolio.construction import PortfolioConstruction
from bot.portfolio.state import PortfolioState
from bot.risk.engine import RiskEngine
from bot.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Immutable configuration for a walk-forward test."""

    train_days: int
    test_days: int
    min_train_days: int  # skip window if training data has fewer unique dates


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Results of a completed walk-forward test."""

    windows: tuple[BacktestResult, ...]
    combined_equity_curve: tuple[tuple[datetime, Decimal], ...]
    combined_sharpe: Decimal
    combined_max_drawdown: Decimal


def run_walkforward(
    config: WalkForwardConfig,
    backtest_config: BacktestConfig,
    dataset: HistoricalDataset,
    strategy_factory: Callable[[EventBus], list[Strategy]],
    portfolio_construction_factory: Callable[[EventBus, PortfolioState], PortfolioConstruction],
    risk_engine_factory: Callable[[EventBus, PortfolioState], RiskEngine],
) -> WalkForwardResult:
    """Slide a walk-forward window across *dataset* and return combined results.

    For each window:
      - Training period: ``[window_start, window_start + train_days)``
      - Test period:     ``[window_start + train_days, window_start + train_days + test_days)``
      - Advance by ``test_days`` each iteration.

    Windows whose training portion has fewer than ``min_train_days`` unique
    dates are skipped.  Fresh components are created for each test window via
    the provided factory callables, ensuring independent equity curves.
    """
    if dataset.bar_count == 0:
        return WalkForwardResult(
            windows=(),
            combined_equity_curve=(),
            combined_sharpe=Decimal("0"),
            combined_max_drawdown=Decimal("0"),
        )

    all_bars = dataset.get_all_bars_sorted()
    min_dt, max_dt = dataset.date_range
    window_start = min_dt.date()
    max_date = max_dt.date()

    windows: list[BacktestResult] = []
    combined_equity: list[tuple[datetime, Decimal]] = []

    while True:
        train_end_date = window_start + timedelta(days=config.train_days)
        test_end_date = train_end_date + timedelta(days=config.test_days)

        if train_end_date > max_date:
            break

        # Check minimum training data
        train_bars = [b for b in all_bars if window_start <= b.timestamp.date() < train_end_date]
        train_unique_dates = {b.timestamp.date() for b in train_bars}
        if len(train_unique_dates) < config.min_train_days:
            window_start += timedelta(days=config.test_days)
            continue

        # Gather test bars
        test_bars = [b for b in all_bars if train_end_date <= b.timestamp.date() < test_end_date]
        if not test_bars:
            window_start += timedelta(days=config.test_days)
            continue

        test_dataset = HistoricalDataset(test_bars)

        # Create fresh, independent components for this window
        bus = EventBus()
        portfolio_state = PortfolioState(
            initial_cash=backtest_config.initial_cash,
            event_bus=bus,
        )
        strategies = strategy_factory(bus)
        portfolio_construction = portfolio_construction_factory(bus, portfolio_state)
        risk_engine = risk_engine_factory(bus, portfolio_state)

        engine = BacktestEngine(
            config=backtest_config,
            dataset=test_dataset,
            event_bus=bus,
            strategies=strategies,
            portfolio_construction=portfolio_construction,
            risk_engine=risk_engine,
            portfolio_state=portfolio_state,
        )
        result = engine.run()
        windows.append(result)
        combined_equity.extend(result.equity_curve)

        window_start += timedelta(days=config.test_days)

    combined_equity_tuple = tuple(combined_equity)
    combined_sharpe = _compute_sharpe(list(combined_equity))
    combined_max_dd = _compute_max_drawdown(list(combined_equity))

    return WalkForwardResult(
        windows=tuple(windows),
        combined_equity_curve=combined_equity_tuple,
        combined_sharpe=combined_sharpe,
        combined_max_drawdown=combined_max_dd,
    )
