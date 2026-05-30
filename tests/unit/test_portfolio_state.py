"""Unit tests for PortfolioState and PortfolioSnapshot."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.signals import Direction
from bot.events.bus import EventBus
from bot.events.portfolio import FillEvent
from bot.portfolio.state import PortfolioState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_INITIAL_CASH = Money(Decimal("100000"), "USD")


def _spy() -> Instrument:
    return Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _qqq() -> Instrument:
    return Instrument(symbol="QQQ", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def _fill(
    instrument: Instrument,
    qty: str,
    price: str,
    side: Direction,
    commission: str = "0",
    order_id: str = "ORD-001",
) -> FillEvent:
    return FillEvent(
        timestamp=_TS,
        source="broker",
        order_id=order_id,
        instrument=instrument,
        filled_quantity=Decimal(qty),
        fill_price=Money(Decimal(price), "USD"),
        commission=Money(Decimal(commission), "USD"),
        side=side,
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
def state(bus: EventBus) -> PortfolioState:
    return PortfolioState(initial_cash=_INITIAL_CASH, event_bus=bus)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_no_positions(self, state: PortfolioState, spy: Instrument) -> None:
        assert state.get_position(spy) is None

    def test_cash_equals_initial(self, state: PortfolioState) -> None:
        assert state.get_cash() == _INITIAL_CASH

    def test_fill_count_zero(self, state: PortfolioState) -> None:
        assert state.fill_count() == 0


# ---------------------------------------------------------------------------
# Single buy fill
# ---------------------------------------------------------------------------


class TestSingleBuyFill:
    def test_position_quantity(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("10")

    def test_average_price(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.average_price == Money(Decimal("100"), "USD")

    def test_cash_reduced_by_cost_plus_commission(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG, commission="1.50"))
        expected_cash = _INITIAL_CASH - Money(Decimal("1001.50"), "USD")
        assert state.get_cash() == expected_cash

    def test_fill_count_increments(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        assert state.fill_count() == 1


# ---------------------------------------------------------------------------
# Adding to an existing long position
# ---------------------------------------------------------------------------


class TestAddToLong:
    def test_quantity_accumulates(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "10", "110", Direction.LONG, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("20")

    def test_weighted_average_price(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "10", "110", Direction.LONG, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.average_price == Money(Decimal("105"), "USD")

    def test_cash_debited_for_both_fills(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG, commission="1"))
        bus.publish(_fill(spy, "10", "110", Direction.LONG, commission="1", order_id="ORD-002"))
        expected = _INITIAL_CASH - Money(Decimal("1001"), "USD") - Money(Decimal("1101"), "USD")
        assert state.get_cash() == expected


# ---------------------------------------------------------------------------
# Partial sell
# ---------------------------------------------------------------------------


class TestPartialSell:
    def test_quantity_reduced(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "4", "110", Direction.SHORT, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("6")

    def test_average_price_unchanged(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "4", "110", Direction.SHORT, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.average_price == Money(Decimal("100"), "USD")

    def test_cash_credited_for_sell(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG, commission="1"))
        bus.publish(_fill(spy, "4", "110", Direction.SHORT, commission="1", order_id="ORD-002"))
        # Cash: 100000 - 1001 + 440 - 1 = 99438
        expected = _INITIAL_CASH - Money(Decimal("1001"), "USD") + Money(Decimal("439"), "USD")
        assert state.get_cash() == expected


# ---------------------------------------------------------------------------
# Full close
# ---------------------------------------------------------------------------


class TestFullClose:
    def test_position_removed(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "10", "110", Direction.SHORT, order_id="ORD-002"))
        assert state.get_position(spy) is None

    def test_fill_count_two(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "10", "110", Direction.SHORT, order_id="ORD-002"))
        assert state.fill_count() == 2


# ---------------------------------------------------------------------------
# Position flip (long → short)
# ---------------------------------------------------------------------------


class TestPositionFlip:
    def test_quantity_is_negative(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "15", "110", Direction.SHORT, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("-5")

    def test_average_price_is_fill_price(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(spy, "15", "110", Direction.SHORT, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.average_price == Money(Decimal("110"), "USD")


# ---------------------------------------------------------------------------
# Short position (open short, add to short, reduce short, flip short → long)
# ---------------------------------------------------------------------------


class TestShortPosition:
    def test_open_short(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "5", "100", Direction.SHORT))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("-5")
        assert pos.average_price == Money(Decimal("100"), "USD")

    def test_add_to_short_weighted_avg(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.SHORT))
        bus.publish(_fill(spy, "10", "90", Direction.SHORT, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("-20")
        assert pos.average_price == Money(Decimal("95"), "USD")

    def test_reduce_short_avg_unchanged(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.SHORT))
        bus.publish(_fill(spy, "4", "90", Direction.LONG, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("-6")
        assert pos.average_price == Money(Decimal("100"), "USD")

    def test_flip_short_to_long(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.SHORT))
        bus.publish(_fill(spy, "15", "90", Direction.LONG, order_id="ORD-002"))
        pos = state.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("5")
        assert pos.average_price == Money(Decimal("90"), "USD")

    def test_close_short(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.SHORT))
        bus.publish(_fill(spy, "10", "90", Direction.LONG, order_id="ORD-002"))
        assert state.get_position(spy) is None


# ---------------------------------------------------------------------------
# Commission effects on cash
# ---------------------------------------------------------------------------


class TestCommission:
    def test_commission_deducted_on_buy(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "1", "100", Direction.LONG, commission="1"))
        assert state.get_cash() == _INITIAL_CASH - Money(Decimal("101"), "USD")

    def test_commission_deducted_on_sell(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "1", "100", Direction.LONG))
        bus.publish(_fill(spy, "1", "100", Direction.SHORT, commission="1", order_id="ORD-002"))
        # Buy: -100, Sell: +100 - 1 = +99 net from sell → cash = 100000 - 100 + 99 = 99999
        assert state.get_cash() == _INITIAL_CASH - Money(Decimal("1"), "USD")


# ---------------------------------------------------------------------------
# Multiple instruments
# ---------------------------------------------------------------------------


class TestMultipleInstruments:
    def test_independent_positions(
        self,
        state: PortfolioState,
        bus: EventBus,
        spy: Instrument,
        qqq: Instrument,
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(qqq, "5", "200", Direction.LONG, order_id="ORD-002"))
        assert state.get_position(spy) is not None
        assert state.get_position(qqq) is not None
        assert state.get_position(spy).quantity == Decimal("10")  # type: ignore[union-attr]
        assert state.get_position(qqq).quantity == Decimal("5")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# PortfolioSnapshot
# ---------------------------------------------------------------------------


class TestPortfolioSnapshot:
    def test_total_position_count(
        self, state: PortfolioState, bus: EventBus, spy: Instrument, qqq: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(qqq, "5", "200", Direction.LONG, order_id="ORD-002"))
        snap = state.snapshot()
        assert snap.total_position_count == 2

    def test_instruments_set(
        self, state: PortfolioState, bus: EventBus, spy: Instrument, qqq: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(qqq, "5", "200", Direction.LONG, order_id="ORD-002"))
        snap = state.snapshot()
        assert spy in snap.instruments
        assert qqq in snap.instruments

    def test_get_position_on_snapshot(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        snap = state.snapshot()
        pos = snap.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("10")

    def test_get_position_unknown_returns_none(
        self, state: PortfolioState, bus: EventBus, spy: Instrument, qqq: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        snap = state.snapshot()
        assert snap.get_position(qqq) is None

    def test_gross_notional(self, state: PortfolioState, bus: EventBus, spy: Instrument) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        snap = state.snapshot()
        prices = {spy: Money(Decimal("105"), "USD")}
        # 10 * 105 = 1050
        assert snap.gross_notional(prices) == Money(Decimal("1050"), "USD")

    def test_gross_notional_counts_short_as_positive(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.SHORT))
        snap = state.snapshot()
        prices = {spy: Money(Decimal("105"), "USD")}
        # abs(-10) * 105 = 1050
        assert snap.gross_notional(prices) == Money(Decimal("1050"), "USD")

    def test_gross_notional_empty_is_zero(self, state: PortfolioState) -> None:
        snap = state.snapshot()
        assert snap.gross_notional({}) == Money.zero("USD")

    def test_net_notional_signed(
        self, state: PortfolioState, bus: EventBus, spy: Instrument, qqq: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        bus.publish(_fill(qqq, "5", "200", Direction.SHORT, order_id="ORD-002"))
        snap = state.snapshot()
        prices = {
            spy: Money(Decimal("110"), "USD"),
            qqq: Money(Decimal("210"), "USD"),
        }
        # SPY: 10 * 110 = 1100; QQQ: -5 * 210 = -1050; net = 50
        assert snap.net_notional(prices) == Money(Decimal("50"), "USD")

    def test_net_notional_empty_is_zero(self, state: PortfolioState) -> None:
        snap = state.snapshot()
        assert snap.net_notional({}) == Money.zero("USD")

    def test_total_market_value(
        self, state: PortfolioState, bus: EventBus, spy: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        snap = state.snapshot()
        prices = {spy: Money(Decimal("110"), "USD")}
        # net_notional = 10 * 110 = 1100
        # cash = 100000 - 1000 = 99000
        # total = 1100 + 99000 = 100100
        expected = Money(Decimal("100100"), "USD")
        assert snap.total_market_value(prices) == expected

    def test_snapshot_immutable_after_state_change(
        self, state: PortfolioState, bus: EventBus, spy: Instrument, qqq: Instrument
    ) -> None:
        bus.publish(_fill(spy, "10", "100", Direction.LONG))
        snap = state.snapshot()
        # Now modify state
        bus.publish(_fill(spy, "5", "110", Direction.LONG, order_id="ORD-002"))
        bus.publish(_fill(qqq, "3", "200", Direction.LONG, order_id="ORD-003"))
        # Snapshot should still show original state
        assert snap.total_position_count == 1
        pos = snap.get_position(spy)
        assert pos is not None
        assert pos.quantity == Decimal("10")

    def test_snapshot_has_utc_timestamp(self, state: PortfolioState) -> None:
        snap = state.snapshot()
        assert snap.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Hypothesis: cash conservation across arbitrary fill sequences
# ---------------------------------------------------------------------------

_PRICES = st.decimals(
    min_value=Decimal("1"),
    max_value=Decimal("1000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
_QUANTITIES = st.decimals(
    min_value=Decimal("1"),
    max_value=Decimal("1000"),
    allow_nan=False,
    allow_infinity=False,
    places=0,
)
_COMMISSIONS = st.decimals(
    min_value=Decimal("0"),
    max_value=Decimal("10"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
_SIDES = st.sampled_from([Direction.LONG, Direction.SHORT])


class TestHypothesisCashConservation:
    @given(
        fills=st.lists(
            st.tuples(_PRICES, _QUANTITIES, _COMMISSIONS, _SIDES),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_cash_conservation(
        self,
        fills: list[tuple[Decimal, Decimal, Decimal, Direction]],
    ) -> None:
        """Final cash = initial_cash − net_cost − total_commission."""
        initial = Money(Decimal("1000000"), "USD")
        bus = EventBus()
        ps = PortfolioState(initial_cash=initial, event_bus=bus)
        spy = _spy()

        net_cost = Decimal("0")
        total_commission = Decimal("0")

        for i, (price, qty, commission, side) in enumerate(fills):
            bus.publish(
                FillEvent(
                    timestamp=_TS,
                    source="broker",
                    order_id=f"ORD-{i:04d}",
                    instrument=spy,
                    filled_quantity=qty,
                    fill_price=Money(price, "USD"),
                    commission=Money(commission, "USD"),
                    side=side,
                )
            )
            signed = Decimal("1") if side == Direction.LONG else Decimal("-1")
            net_cost += signed * price * qty
            total_commission += commission

        expected = initial.amount - net_cost - total_commission
        assert ps.get_cash().amount == expected
