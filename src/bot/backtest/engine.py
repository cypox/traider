"""Backtesting engine: replays historical bars through the live trading pipeline."""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import structlog

from bot.backtest.data import HistoricalDataset
from bot.core.instruments import Instrument
from bot.core.money import Money
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent
from bot.events.portfolio import ApprovedPositionEvent, FillEvent
from bot.portfolio.construction import PortfolioConstruction
from bot.portfolio.state import PortfolioSnapshot, PortfolioState
from bot.risk.engine import RiskEngine
from bot.strategies.base import Strategy

_logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable configuration for the backtesting engine."""

    initial_cash: Money
    slippage_bps: Decimal
    commission_per_share: Decimal
    fill_on: str = field(default="close")  # "close" or "open" of next bar


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Results of a completed backtest run."""

    equity_curve: tuple[tuple[datetime, Decimal], ...]
    trades: tuple[FillEvent, ...]
    final_portfolio: PortfolioSnapshot
    sharpe_ratio: Decimal
    max_drawdown_pct: Decimal
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    total_commission: Money

    def summary(self) -> str:
        """Return a human-readable multi-line performance summary."""
        lines = [
            "BacktestResult:",
            f"  Bars:               {len(self.equity_curve)}",
            f"  Trades:             {len(self.trades)}",
            f"  Total Return:       {float(self.total_return_pct) * 100:.2f}%",
            f"  Annualized Return:  {float(self.annualized_return_pct) * 100:.2f}%",
            f"  Sharpe Ratio:       {float(self.sharpe_ratio):.4f}",
            f"  Max Drawdown:       {float(self.max_drawdown_pct) * 100:.2f}%",
            f"  Total Commission:   {self.total_commission.amount} {self.total_commission.currency}",  # noqa: E501
        ]
        return "\n".join(lines)


def _compute_sharpe(equity_curve: list[tuple[datetime, Decimal]]) -> Decimal:
    """Return the annualized Sharpe ratio for *equity_curve*.

    Uses sample standard deviation (ddof=1).  Returns ``Decimal("0")`` when
    there are fewer than two returns or the standard deviation is zero.
    """
    if len(equity_curve) < 2:
        return Decimal("0")
    values = [float(eq) for _, eq in equity_curve]
    daily_returns = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] != 0.0
    ]
    if len(daily_returns) < 2:
        return Decimal("0")
    mean_r = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std_r = math.sqrt(variance)
    if std_r == 0.0:
        return Decimal("0")
    sharpe = mean_r / std_r * math.sqrt(252)
    return Decimal(str(round(sharpe, 10)))


def _compute_max_drawdown(equity_curve: list[tuple[datetime, Decimal]]) -> Decimal:
    """Return the maximum peak-to-trough drawdown as a decimal fraction."""
    if not equity_curve:
        return Decimal("0")
    peak = equity_curve[0][1]
    max_dd = Decimal("0")
    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > Decimal("0"):
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


class BacktestEngine:
    """Replays a :class:`HistoricalDataset` through the full live pipeline.

    All components—strategies, portfolio construction, risk engine—are
    pre-wired to a shared :class:`EventBus` by the caller.
    :class:`BacktestEngine` publishes :class:`QuoteEvent` and
    :class:`BarEvent` objects in chronological order and simulates fills for
    every :class:`ApprovedPositionEvent` that the pipeline produces.

    This design ensures zero divergence between back-test and live execution.
    """

    def __init__(
        self,
        config: BacktestConfig,
        dataset: HistoricalDataset,
        event_bus: EventBus,
        strategies: list[Strategy],
        portfolio_construction: PortfolioConstruction,
        risk_engine: RiskEngine,
        portfolio_state: PortfolioState,
    ) -> None:
        self._config = config
        self._dataset = dataset
        self._event_bus = event_bus
        self._strategies = strategies
        self._portfolio_construction = portfolio_construction
        self._risk_engine = risk_engine
        self._portfolio_state = portfolio_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        instrument: Instrument,
        delta_qty: Decimal,
        fill_price_base: Decimal,
        fill_timestamp: datetime,
    ) -> FillEvent:
        """Build a :class:`FillEvent` for *delta_qty* shares with slippage."""
        side = Direction.LONG if delta_qty > Decimal("0") else Direction.SHORT
        sign = Decimal("1") if side == Direction.LONG else Decimal("-1")
        slippage_factor = Decimal("1") + self._config.slippage_bps / Decimal("10000") * sign
        fill_price = Money(fill_price_base * slippage_factor, instrument.currency)
        commission = Money(
            self._config.commission_per_share * abs(delta_qty),
            instrument.currency,
        )
        return FillEvent(
            timestamp=fill_timestamp,
            source="backtest",
            order_id=str(uuid.uuid4()),
            instrument=instrument,
            filled_quantity=abs(delta_qty),
            fill_price=fill_price,
            commission=commission,
            side=side,
        )

    def _get_equity(self, latest_prices: dict[Instrument, Decimal]) -> Decimal:
        """Return current equity using the most recently seen bar prices."""
        snapshot = self._portfolio_state.snapshot()
        equity = snapshot.cash.amount
        for pos in snapshot.positions:
            price = latest_prices.get(pos.instrument)
            if price is not None:
                equity += pos.quantity * price
        return equity

    def _process_approvals(
        self,
        approvals: list[ApprovedPositionEvent],
        fill_price_base: dict[Instrument, Decimal],
        fill_timestamp: datetime,
    ) -> tuple[list[FillEvent], Money]:
        """Convert *approvals* to fills; return (fills, total_commission)."""
        fills: list[FillEvent] = []
        total_commission = Money.zero(self._config.initial_cash.currency)
        for approval_event in approvals:
            instrument = approval_event.approved.instrument
            target_qty = approval_event.approved.approved_quantity
            existing = self._portfolio_state.get_position(instrument)
            current_qty = existing.quantity if existing is not None else Decimal("0")
            delta = target_qty - current_qty
            if delta == Decimal("0"):
                continue
            price_base = fill_price_base.get(instrument)
            if price_base is None:
                _logger.warning(
                    "no fill price for instrument, skipping fill",
                    symbol=instrument.symbol,
                )
                continue
            fill = self._simulate_fill(instrument, delta, price_base, fill_timestamp)
            self._event_bus.publish(fill)
            fills.append(fill)
            total_commission = total_commission + fill.commission
        return fills, total_commission

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Replay the dataset and return a :class:`BacktestResult`.

        Steps per bar:
          1. Publish :class:`QuoteEvent` (bid=ask=close) to seed price caches.
          2. Publish :class:`BarEvent` — triggers strategies → portfolio
             construction → risk engine → :class:`ApprovedPositionEvent`.
          3. Simulate fills for each approved position (delta vs. current).
          4. Record *(timestamp, equity)* to the equity curve.
        """
        bars = self._dataset.get_all_bars_sorted()
        pending_approvals: list[ApprovedPositionEvent] = []

        def _collect(event: ApprovedPositionEvent) -> None:
            pending_approvals.append(event)

        self._event_bus.subscribe(ApprovedPositionEvent, _collect)

        equity_curve: list[tuple[datetime, Decimal]] = []
        all_fills: list[FillEvent] = []
        currency = self._config.initial_cash.currency
        total_commission = Money.zero(currency)
        latest_prices: dict[Instrument, Decimal] = {}

        # Maps instrument → approvals pending fill at next bar's open
        deferred: dict[Instrument, list[ApprovedPositionEvent]] = {}

        for bar in bars:
            # ── fill_on="open": fill deferred approvals at this bar's open ──
            if self._config.fill_on == "open":
                deferred_for_inst = deferred.pop(bar.instrument, [])
                if deferred_for_inst:
                    fills, comm = self._process_approvals(
                        deferred_for_inst,
                        {bar.instrument: bar.open},
                        bar.timestamp,
                    )
                    all_fills.extend(fills)
                    total_commission = total_commission + comm

            # ── Publish quote so all price caches are current ──
            self._event_bus.publish(
                QuoteEvent(
                    timestamp=bar.timestamp,
                    source="backtest",
                    instrument=bar.instrument,
                    bid=bar.close,
                    ask=bar.close,
                )
            )
            latest_prices[bar.instrument] = bar.close

            # ── Clear collector, then publish bar (full pipeline fires) ──
            pending_approvals.clear()
            self._event_bus.publish(
                BarEvent(
                    timestamp=bar.timestamp,
                    source="backtest",
                    instrument=bar.instrument,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    interval_seconds=86400,
                )
            )

            # ── Simulate fills for new approvals ──
            if self._config.fill_on == "close":
                fills, comm = self._process_approvals(
                    list(pending_approvals),
                    {bar.instrument: bar.close},
                    bar.timestamp,
                )
                all_fills.extend(fills)
                total_commission = total_commission + comm
            else:  # "open" of next bar
                for approval_event in pending_approvals:
                    inst = approval_event.approved.instrument
                    deferred.setdefault(inst, []).append(approval_event)

            # ── Record equity ──
            equity_curve.append((bar.timestamp, self._get_equity(latest_prices)))

        self._event_bus.unsubscribe(ApprovedPositionEvent, _collect)

        # ── Performance metrics ──
        sharpe = _compute_sharpe(equity_curve)
        max_dd = _compute_max_drawdown(equity_curve)

        initial_equity = self._config.initial_cash.amount
        final_equity = equity_curve[-1][1] if equity_curve else initial_equity

        if initial_equity != Decimal("0"):
            total_return = (final_equity - initial_equity) / initial_equity
        else:
            total_return = Decimal("0")

        if len(equity_curve) >= 2:
            holding_days = (equity_curve[-1][0] - equity_curve[0][0]).days
        else:
            holding_days = 0

        if holding_days > 0:
            ann_return = Decimal(str((1.0 + float(total_return)) ** (365.0 / holding_days) - 1.0))
        else:
            ann_return = Decimal("0")

        return BacktestResult(
            equity_curve=tuple(equity_curve),
            trades=tuple(all_fills),
            final_portfolio=self._portfolio_state.snapshot(),
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_return_pct=total_return,
            annualized_return_pct=ann_return,
            total_commission=total_commission,
        )
