"""Transaction cost model for backtesting.

Models three cost components:
  - Bid-ask spread (half-spread paid on each fill)
  - Commission (per-share with a per-trade minimum)
  - Market impact (square-root model based on participation rate)

All functions are pure — no I/O or side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bot.core.money import Money
from bot.core.signals import Direction


@dataclass(frozen=True, slots=True)
class CostModel:
    """Parameters that fully describe the transaction cost model."""

    spread_bps: Decimal
    commission_per_share: Decimal
    market_impact_bps: Decimal
    min_commission: Decimal


def compute_spread_cost(
    price: Decimal,
    quantity: Decimal,
    spread_bps: Decimal,
) -> Money:
    """Half-spread cost for a single fill.

    The trader pays half the bid-ask spread on entry and half on exit.
    This function models one side: cost = price * |qty| * (spread_bps / 2) / 10000.
    """
    cost = price * abs(quantity) * (spread_bps / Decimal("2")) / Decimal("10000")
    return Money(amount=cost, currency="USD")


def compute_commission(
    quantity: Decimal,
    commission_per_share: Decimal,
    min_commission: Decimal,
) -> Money:
    """Broker commission with a per-trade minimum."""
    raw = abs(quantity) * commission_per_share
    commission = max(raw, min_commission)
    return Money(amount=commission, currency="USD")


def compute_market_impact(
    price: Decimal,
    quantity: Decimal,
    average_daily_volume: Decimal,
    market_impact_bps: Decimal,
) -> Money:
    """Square-root market impact model.

    participation_rate = |qty| / average_daily_volume
    impact_bps         = market_impact_bps * sqrt(participation_rate)
    impact_cost        = price * |qty| * impact_bps / 10000
    """
    if average_daily_volume == Decimal("0"):
        return Money(amount=Decimal("0"), currency="USD")

    participation_rate = abs(quantity) / average_daily_volume
    sqrt_participation = Decimal(float(participation_rate) ** 0.5)
    impact_bps = market_impact_bps * sqrt_participation
    impact_cost = price * abs(quantity) * impact_bps / Decimal("10000")
    return Money(amount=impact_cost, currency="USD")


def compute_total_cost(
    price: Decimal,
    quantity: Decimal,
    average_daily_volume: Decimal,
    cost_model: CostModel,
) -> Money:
    """Total transaction cost: spread + commission + market impact."""
    spread = compute_spread_cost(price, quantity, cost_model.spread_bps)
    commission = compute_commission(
        quantity, cost_model.commission_per_share, cost_model.min_commission
    )
    impact = compute_market_impact(
        price, quantity, average_daily_volume, cost_model.market_impact_bps
    )
    total = spread.amount + commission.amount + impact.amount
    return Money(amount=total, currency="USD")


def adjust_fill_price(
    mid_price: Decimal,
    side: Direction,
    cost_model: CostModel,
    quantity: Decimal,
    average_daily_volume: Decimal,
) -> Decimal:
    """Effective fill price after half-spread and market impact.

    Buys:  fill = mid * (1 + (spread_bps/2 + impact_bps) / 10000)
    Sells: fill = mid * (1 - (spread_bps/2 + impact_bps) / 10000)
    """
    if average_daily_volume == Decimal("0"):
        impact_bps = Decimal("0")
    else:
        participation_rate = abs(quantity) / average_daily_volume
        impact_bps = cost_model.market_impact_bps * Decimal(float(participation_rate) ** 0.5)

    half_spread_bps = cost_model.spread_bps / Decimal("2")
    total_bps = half_spread_bps + impact_bps

    if side == Direction.LONG:
        return mid_price * (Decimal("1") + total_bps / Decimal("10000"))
    else:
        return mid_price * (Decimal("1") - total_bps / Decimal("10000"))
