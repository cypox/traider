"""Unit tests for the transaction cost model — 100% coverage target."""

from decimal import Decimal

import hypothesis.strategies as st
from hypothesis import given, settings

from bot.backtest.costs import (
    CostModel,
    adjust_fill_price,
    compute_commission,
    compute_market_impact,
    compute_spread_cost,
    compute_total_cost,
)
from bot.core.money import Money
from bot.core.signals import Direction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRICE = Decimal("100")
_QTY = Decimal("100")
_ADV = Decimal("1000000")  # 1 million shares average daily volume


def _model(
    spread_bps: Decimal = Decimal("10"),
    commission_per_share: Decimal = Decimal("0.005"),
    market_impact_bps: Decimal = Decimal("5"),
    min_commission: Decimal = Decimal("1"),
) -> CostModel:
    return CostModel(
        spread_bps=spread_bps,
        commission_per_share=commission_per_share,
        market_impact_bps=market_impact_bps,
        min_commission=min_commission,
    )


# ---------------------------------------------------------------------------
# compute_spread_cost
# ---------------------------------------------------------------------------


class TestComputeSpreadCost:
    def test_zero_spread_returns_zero(self) -> None:
        result = compute_spread_cost(_PRICE, _QTY, Decimal("0"))
        assert result.amount == Decimal("0")

    def test_doubling_quantity_doubles_cost(self) -> None:
        c1 = compute_spread_cost(_PRICE, _QTY, Decimal("10"))
        c2 = compute_spread_cost(_PRICE, _QTY * 2, Decimal("10"))
        assert c2.amount == c1.amount * 2

    def test_doubling_price_doubles_cost(self) -> None:
        c1 = compute_spread_cost(_PRICE, _QTY, Decimal("10"))
        c2 = compute_spread_cost(_PRICE * 2, _QTY, Decimal("10"))
        assert c2.amount == c1.amount * 2

    def test_result_is_money(self) -> None:
        result = compute_spread_cost(_PRICE, _QTY, Decimal("10"))
        assert isinstance(result, Money)

    def test_known_value(self) -> None:
        # price=100, qty=100, spread_bps=10 → 100*100*(10/2)/10000 = 5
        result = compute_spread_cost(Decimal("100"), Decimal("100"), Decimal("10"))
        assert result.amount == Decimal("5")

    def test_negative_quantity_treated_as_absolute(self) -> None:
        pos = compute_spread_cost(_PRICE, _QTY, Decimal("10"))
        neg = compute_spread_cost(_PRICE, -_QTY, Decimal("10"))
        assert pos.amount == neg.amount


# ---------------------------------------------------------------------------
# compute_commission
# ---------------------------------------------------------------------------


class TestComputeCommission:
    def test_small_trade_returns_min_commission(self) -> None:
        # qty=1, per_share=0.001, min=1.0 → raw=0.001 < 1.0 → returns 1.0
        result = compute_commission(Decimal("1"), Decimal("0.001"), Decimal("1"))
        assert result.amount == Decimal("1")

    def test_large_trade_returns_per_share_amount(self) -> None:
        # qty=1000, per_share=0.005, min=1.0 → raw=5.0 > 1.0 → returns 5.0
        result = compute_commission(Decimal("1000"), Decimal("0.005"), Decimal("1"))
        assert result.amount == Decimal("5.0")

    def test_zero_quantity_returns_min_commission(self) -> None:
        result = compute_commission(Decimal("0"), Decimal("0.005"), Decimal("1"))
        assert result.amount == Decimal("1")

    def test_result_is_money(self) -> None:
        result = compute_commission(_QTY, Decimal("0.005"), Decimal("1"))
        assert isinstance(result, Money)

    def test_negative_quantity_treated_as_absolute(self) -> None:
        pos = compute_commission(_QTY, Decimal("0.005"), Decimal("1"))
        neg = compute_commission(-_QTY, Decimal("0.005"), Decimal("1"))
        assert pos.amount == neg.amount

    def test_exact_min_boundary(self) -> None:
        # qty=200, per_share=0.005, min=1.0 → raw=1.0 == min → returns 1.0
        result = compute_commission(Decimal("200"), Decimal("0.005"), Decimal("1"))
        assert result.amount == Decimal("1.0")


# ---------------------------------------------------------------------------
# compute_market_impact
# ---------------------------------------------------------------------------


class TestComputeMarketImpact:
    def test_zero_volume_returns_zero(self) -> None:
        result = compute_market_impact(_PRICE, _QTY, Decimal("0"), Decimal("5"))
        assert result.amount == Decimal("0")

    def test_larger_quantity_higher_impact(self) -> None:
        small = compute_market_impact(_PRICE, Decimal("100"), _ADV, Decimal("5"))
        large = compute_market_impact(_PRICE, Decimal("10000"), _ADV, Decimal("5"))
        assert large.amount > small.amount

    def test_full_participation_rate_impact_equals_model_bps(self) -> None:
        # qty == adv → participation_rate=1.0 → sqrt=1.0 → impact_bps = market_impact_bps
        # impact_cost = price * qty * market_impact_bps / 10000
        adv = Decimal("1000")
        qty = adv  # 100% participation
        price = Decimal("100")
        impact_bps = Decimal("5")
        result = compute_market_impact(price, qty, adv, impact_bps)
        expected = price * qty * impact_bps / Decimal("10000")
        # Allow small float rounding tolerance
        assert abs(result.amount - expected) < Decimal("0.001")

    def test_result_is_money(self) -> None:
        result = compute_market_impact(_PRICE, _QTY, _ADV, Decimal("5"))
        assert isinstance(result, Money)

    def test_negative_quantity_treated_as_absolute(self) -> None:
        pos = compute_market_impact(_PRICE, _QTY, _ADV, Decimal("5"))
        neg = compute_market_impact(_PRICE, -_QTY, _ADV, Decimal("5"))
        assert pos.amount == neg.amount

    def test_sqrt_relationship(self) -> None:
        # impact scales with sqrt(participation_rate)
        # q1=100, q2=400 with adv=10000 → sqrt(0.01)=0.1, sqrt(0.04)=0.2
        # So cost2/cost1 ≈ sqrt(400/100) = 2.0
        q1 = compute_market_impact(_PRICE, Decimal("100"), Decimal("10000"), Decimal("10"))
        q2 = compute_market_impact(_PRICE, Decimal("400"), Decimal("10000"), Decimal("10"))
        ratio = q2.amount / q1.amount
        # ratio ≈ sqrt(4) * (400/100) = 2 * 4 = 8 (both qty and sqrt scale)
        # cost = price * qty * impact_bps * sqrt(qty/adv) / 10000
        # q2_cost / q1_cost = (400 * sqrt(400/10000)) / (100 * sqrt(100/10000))
        #                   = (400 * 0.2) / (100 * 0.1) = 80/10 = 8
        assert abs(ratio - Decimal("8")) < Decimal("0.01")


# ---------------------------------------------------------------------------
# compute_total_cost
# ---------------------------------------------------------------------------


class TestComputeTotalCost:
    def test_equals_sum_of_components(self) -> None:
        model = _model()
        spread = compute_spread_cost(_PRICE, _QTY, model.spread_bps)
        commission = compute_commission(_QTY, model.commission_per_share, model.min_commission)
        impact = compute_market_impact(_PRICE, _QTY, _ADV, model.market_impact_bps)
        total = compute_total_cost(_PRICE, _QTY, _ADV, model)
        expected = spread.amount + commission.amount + impact.amount
        assert total.amount == expected

    def test_increases_with_quantity(self) -> None:
        model = _model()
        c1 = compute_total_cost(_PRICE, Decimal("100"), _ADV, model)
        c2 = compute_total_cost(_PRICE, Decimal("1000"), _ADV, model)
        assert c2.amount > c1.amount

    def test_increases_with_price(self) -> None:
        model = _model()
        c1 = compute_total_cost(Decimal("50"), _QTY, _ADV, model)
        c2 = compute_total_cost(Decimal("100"), _QTY, _ADV, model)
        assert c2.amount > c1.amount

    def test_result_is_money(self) -> None:
        result = compute_total_cost(_PRICE, _QTY, _ADV, _model())
        assert isinstance(result, Money)

    def test_zero_adv_does_not_raise(self) -> None:
        model = _model()
        result = compute_total_cost(_PRICE, _QTY, Decimal("0"), model)
        assert isinstance(result, Money)
        assert result.amount >= Decimal("0")


# ---------------------------------------------------------------------------
# adjust_fill_price
# ---------------------------------------------------------------------------


class TestAdjustFillPrice:
    def test_buy_fill_price_above_mid(self) -> None:
        fill = adjust_fill_price(_PRICE, Direction.LONG, _model(), _QTY, _ADV)
        assert fill > _PRICE

    def test_sell_fill_price_below_mid(self) -> None:
        fill = adjust_fill_price(_PRICE, Direction.SHORT, _model(), _QTY, _ADV)
        assert fill < _PRICE

    def test_zero_spread_and_zero_impact_equals_mid(self) -> None:
        model = _model(spread_bps=Decimal("0"), market_impact_bps=Decimal("0"))
        buy = adjust_fill_price(_PRICE, Direction.LONG, model, _QTY, _ADV)
        sell = adjust_fill_price(_PRICE, Direction.SHORT, model, _QTY, _ADV)
        assert buy == _PRICE
        assert sell == _PRICE

    def test_larger_quantity_worse_fill_for_buy(self) -> None:
        model = _model()
        fill_small = adjust_fill_price(_PRICE, Direction.LONG, model, Decimal("100"), _ADV)
        fill_large = adjust_fill_price(_PRICE, Direction.LONG, model, Decimal("10000"), _ADV)
        assert fill_large > fill_small

    def test_larger_quantity_worse_fill_for_sell(self) -> None:
        model = _model()
        fill_small = adjust_fill_price(_PRICE, Direction.SHORT, model, Decimal("100"), _ADV)
        fill_large = adjust_fill_price(_PRICE, Direction.SHORT, model, Decimal("10000"), _ADV)
        assert fill_large < fill_small

    def test_returns_decimal(self) -> None:
        result = adjust_fill_price(_PRICE, Direction.LONG, _model(), _QTY, _ADV)
        assert isinstance(result, Decimal)

    def test_zero_adv_uses_only_spread(self) -> None:
        # With zero adv, impact_bps=0, so only half-spread applies
        model = _model(spread_bps=Decimal("20"), market_impact_bps=Decimal("100"))
        buy = adjust_fill_price(_PRICE, Direction.LONG, model, _QTY, Decimal("0"))
        # Expected: 100 * (1 + 10/10000) = 100 * 1.001 = 100.1
        assert buy == Decimal("100") * (Decimal("1") + Decimal("10") / Decimal("10000"))


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

_pos_decimal = st.decimals(
    min_value="0.01", max_value="10000", allow_nan=False, allow_infinity=False
).map(Decimal)

_nonneg_bps = st.decimals(
    min_value="0", max_value="500", allow_nan=False, allow_infinity=False
).map(Decimal)


@settings(max_examples=200)
@given(
    price=_pos_decimal,
    quantity=_pos_decimal,
    adv=_pos_decimal,
    spread_bps=_nonneg_bps,
    commission_per_share=_pos_decimal,
    market_impact_bps=_nonneg_bps,
    min_commission=_pos_decimal,
)
def test_total_cost_always_nonneg(
    price: Decimal,
    quantity: Decimal,
    adv: Decimal,
    spread_bps: Decimal,
    commission_per_share: Decimal,
    market_impact_bps: Decimal,
    min_commission: Decimal,
) -> None:
    model = CostModel(
        spread_bps=spread_bps,
        commission_per_share=commission_per_share,
        market_impact_bps=market_impact_bps,
        min_commission=min_commission,
    )
    result = compute_total_cost(price, quantity, adv, model)
    assert result.amount >= Decimal("0")


@settings(max_examples=200)
@given(
    price=_pos_decimal,
    quantity=_pos_decimal,
    adv=_pos_decimal,
    spread_bps=_nonneg_bps,
    commission_per_share=_pos_decimal,
    market_impact_bps=_nonneg_bps,
    min_commission=_pos_decimal,
)
def test_buy_price_above_sell_price(
    price: Decimal,
    quantity: Decimal,
    adv: Decimal,
    spread_bps: Decimal,
    commission_per_share: Decimal,
    market_impact_bps: Decimal,
    min_commission: Decimal,
) -> None:
    model = CostModel(
        spread_bps=spread_bps,
        commission_per_share=commission_per_share,
        market_impact_bps=market_impact_bps,
        min_commission=min_commission,
    )
    buy = adjust_fill_price(price, Direction.LONG, model, quantity, adv)
    sell = adjust_fill_price(price, Direction.SHORT, model, quantity, adv)
    assert buy >= sell


@settings(max_examples=200)
@given(
    price=_pos_decimal,
    quantity=_pos_decimal,
    adv=_pos_decimal,
    spread_bps=_nonneg_bps,
    commission_per_share=_pos_decimal,
    market_impact_bps=_nonneg_bps,
    min_commission=_pos_decimal,
)
def test_all_functions_return_decimal(
    price: Decimal,
    quantity: Decimal,
    adv: Decimal,
    spread_bps: Decimal,
    commission_per_share: Decimal,
    market_impact_bps: Decimal,
    min_commission: Decimal,
) -> None:
    model = CostModel(
        spread_bps=spread_bps,
        commission_per_share=commission_per_share,
        market_impact_bps=market_impact_bps,
        min_commission=min_commission,
    )
    assert isinstance(compute_spread_cost(price, quantity, spread_bps).amount, Decimal)
    assert isinstance(
        compute_commission(quantity, commission_per_share, min_commission).amount, Decimal
    )
    assert isinstance(
        compute_market_impact(price, quantity, adv, market_impact_bps).amount, Decimal
    )
    assert isinstance(compute_total_cost(price, quantity, adv, model).amount, Decimal)
    assert isinstance(adjust_fill_price(price, Direction.LONG, model, quantity, adv), Decimal)
