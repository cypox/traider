"""Unit tests for fixed income analytics."""

from decimal import Decimal
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bot.analytics.errors import ConvergenceError, InvalidInputError
from bot.analytics.fixed_income import (
    compute_convexity,
    compute_discount_factor,
    compute_dv01,
    compute_macaulay_duration,
    compute_modified_duration,
    compute_ytm,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_VALID_YTM = st.decimals(
    min_value=Decimal("0.001"),
    max_value=Decimal("0.4"),
    allow_nan=False,
    allow_infinity=False,
    places=6,
)
_VALID_DAYS = st.integers(min_value=1, max_value=7300)
_VALID_FACE = st.decimals(
    min_value=Decimal("1"),
    max_value=Decimal("10000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
_VALID_DURATION = st.decimals(
    min_value=Decimal("0.1"),
    max_value=Decimal("30"),
    allow_nan=False,
    allow_infinity=False,
    places=6,
)
_VALID_PRICE_PCT = st.decimals(
    min_value=Decimal("50"),
    max_value=Decimal("150"),
    allow_nan=False,
    allow_infinity=False,
    places=4,
)


# ---------------------------------------------------------------------------
# compute_ytm
# ---------------------------------------------------------------------------


class TestComputeYtm:
    def test_par_bond_ytm_approx_5_pct(self) -> None:
        # days=730 is exactly 4 semi-annual periods → clean par-bond case
        ytm = compute_ytm(
            face_value=Decimal("100"),
            coupon_rate=Decimal("0.05"),
            price_pct=Decimal("100"),
            days_to_maturity=730,
        )
        assert abs(float(ytm) - 0.05) < 1e-6

    def test_premium_bond_ytm_below_coupon(self) -> None:
        ytm = compute_ytm(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0.05"),
            price_pct=Decimal("105"),
            days_to_maturity=730,
        )
        assert float(ytm) < 0.05

    def test_discount_bond_ytm_above_coupon(self) -> None:
        ytm = compute_ytm(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0.05"),
            price_pct=Decimal("95"),
            days_to_maturity=730,
        )
        assert float(ytm) > 0.05

    def test_zero_coupon_returns_positive_yield(self) -> None:
        ytm = compute_ytm(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0"),
            price_pct=Decimal("99"),
            days_to_maturity=30,
        )
        assert ytm > Decimal(0)

    def test_long_bond_converges(self) -> None:
        # 30-year bond at a deep discount
        ytm = compute_ytm(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0.025"),
            price_pct=Decimal("85"),
            days_to_maturity=10950,
        )
        assert ytm > Decimal(0)

    def test_invalid_days_to_maturity_zero(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_ytm(Decimal("1000"), Decimal("0.05"), Decimal("100"), 0)

    def test_invalid_days_to_maturity_negative(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_ytm(Decimal("1000"), Decimal("0.05"), Decimal("100"), -1)

    def test_invalid_price_pct_zero(self) -> None:
        with pytest.raises(InvalidInputError, match="price_pct"):
            compute_ytm(Decimal("1000"), Decimal("0.05"), Decimal("0"), 365)

    def test_invalid_price_pct_negative(self) -> None:
        with pytest.raises(InvalidInputError, match="price_pct"):
            compute_ytm(Decimal("1000"), Decimal("0.05"), Decimal("-1"), 365)

    def test_invalid_face_value_zero(self) -> None:
        with pytest.raises(InvalidInputError, match="face_value"):
            compute_ytm(Decimal("0"), Decimal("0.05"), Decimal("100"), 365)

    def test_invalid_face_value_negative(self) -> None:
        with pytest.raises(InvalidInputError, match="face_value"):
            compute_ytm(Decimal("-100"), Decimal("0.05"), Decimal("100"), 365)

    def test_convergence_error_when_max_iterations_zero(self) -> None:
        with patch("bot.analytics.fixed_income._MAX_ITERATIONS", 1):
            with pytest.raises(ConvergenceError, match="did not converge"):
                compute_ytm(
                    face_value=Decimal("1000"),
                    coupon_rate=Decimal("0.05"),
                    price_pct=Decimal("99"),
                    days_to_maturity=3650,
                )

    def test_returns_decimal_type(self) -> None:
        result = compute_ytm(Decimal("100"), Decimal("0.05"), Decimal("100"), 730)
        assert isinstance(result, Decimal)

    def test_fractional_period_bond(self) -> None:
        # days=1000 with semi-annual coupon → non-round first_period_fraction
        ytm = compute_ytm(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0.05"),
            price_pct=Decimal("98"),
            days_to_maturity=1000,
        )
        assert ytm > Decimal(0)

    def test_annual_coupon_frequency(self) -> None:
        ytm = compute_ytm(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0.04"),
            price_pct=Decimal("100"),
            days_to_maturity=365,
            coupon_frequency=1,
        )
        assert abs(float(ytm) - 0.04) < 1e-6


# ---------------------------------------------------------------------------
# compute_discount_factor
# ---------------------------------------------------------------------------


class TestComputeDiscountFactor:
    def test_ytm_zero_returns_one(self) -> None:
        df = compute_discount_factor(Decimal("0"), 365)
        assert df == Decimal("1")

    def test_positive_ytm_gives_discount(self) -> None:
        df = compute_discount_factor(Decimal("0.05"), 365)
        assert df < Decimal("1")

    def test_higher_ytm_gives_lower_df(self) -> None:
        df_low = compute_discount_factor(Decimal("0.05"), 365)
        df_high = compute_discount_factor(Decimal("0.10"), 365)
        assert df_low > df_high

    def test_days_zero_returns_one(self) -> None:
        df = compute_discount_factor(Decimal("0.05"), 0)
        assert df == Decimal("1")

    def test_known_value(self) -> None:
        # 1 / (1 + 0.05)^1 ≈ 0.952381
        df = compute_discount_factor(Decimal("0.05"), 365)
        assert abs(float(df) - 1 / 1.05) < 1e-6

    def test_invalid_negative_ytm(self) -> None:
        with pytest.raises(InvalidInputError, match="ytm"):
            compute_discount_factor(Decimal("-0.01"), 365)

    def test_invalid_negative_days(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_discount_factor(Decimal("0.05"), -1)

    def test_returns_decimal_type(self) -> None:
        result = compute_discount_factor(Decimal("0.05"), 365)
        assert isinstance(result, Decimal)


# ---------------------------------------------------------------------------
# compute_macaulay_duration
# ---------------------------------------------------------------------------


class TestComputeMacaulayDuration:
    def test_zero_coupon_duration_approx_years_to_maturity(self) -> None:
        # For a zero-coupon bond, Macaulay duration ≈ years to maturity
        days = 730
        dur = compute_macaulay_duration(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0"),
            ytm=Decimal("0.05"),
            days_to_maturity=days,
        )
        assert abs(float(dur) - days / 365) < 0.05  # within 18 days

    def test_coupon_bond_has_shorter_duration_than_zero_coupon(self) -> None:
        kwargs = {
            "face_value": Decimal("1000"),
            "ytm": Decimal("0.05"),
            "days_to_maturity": 1825,
        }
        dur_zero = compute_macaulay_duration(coupon_rate=Decimal("0"), **kwargs)  # type: ignore[arg-type]
        dur_coupon = compute_macaulay_duration(coupon_rate=Decimal("0.05"), **kwargs)  # type: ignore[arg-type]
        assert dur_coupon < dur_zero

    def test_longer_bond_has_longer_duration(self) -> None:
        common = {
            "face_value": Decimal("1000"),
            "coupon_rate": Decimal("0.05"),
            "ytm": Decimal("0.05"),
        }
        dur_short = compute_macaulay_duration(days_to_maturity=730, **common)  # type: ignore[arg-type]
        dur_long = compute_macaulay_duration(days_to_maturity=1825, **common)  # type: ignore[arg-type]
        assert dur_long > dur_short

    def test_invalid_days_to_maturity_zero(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_macaulay_duration(Decimal("1000"), Decimal("0.05"), Decimal("0.05"), 0)

    def test_invalid_days_to_maturity_negative(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_macaulay_duration(Decimal("1000"), Decimal("0.05"), Decimal("0.05"), -1)

    def test_returns_decimal_type(self) -> None:
        result = compute_macaulay_duration(Decimal("1000"), Decimal("0.05"), Decimal("0.05"), 730)
        assert isinstance(result, Decimal)


# ---------------------------------------------------------------------------
# compute_modified_duration
# ---------------------------------------------------------------------------


class TestComputeModifiedDuration:
    def test_modified_less_than_macaulay_for_positive_ytm(self) -> None:
        macaulay = Decimal("5")
        ytm = Decimal("0.05")
        modified = compute_modified_duration(macaulay, ytm)
        assert modified < macaulay

    def test_modified_equals_macaulay_for_zero_ytm(self) -> None:
        macaulay = Decimal("5")
        modified = compute_modified_duration(macaulay, Decimal("0"))
        assert modified == macaulay

    def test_known_value(self) -> None:
        # 5 / (1 + 0.05/2) = 5 / 1.025 ≈ 4.878
        modified = compute_modified_duration(Decimal("5"), Decimal("0.05"))
        assert abs(float(modified) - 5 / 1.025) < 1e-6

    def test_returns_decimal_type(self) -> None:
        result = compute_modified_duration(Decimal("5"), Decimal("0.05"))
        assert isinstance(result, Decimal)

    def test_annual_coupon_frequency(self) -> None:
        # 5 / (1 + 0.05/1) = 5 / 1.05 ≈ 4.762
        modified = compute_modified_duration(Decimal("5"), Decimal("0.05"), coupon_frequency=1)
        assert abs(float(modified) - 5 / 1.05) < 1e-6


# ---------------------------------------------------------------------------
# compute_dv01
# ---------------------------------------------------------------------------


class TestComputeDv01:
    def test_linear_in_face_value(self) -> None:
        dv01_1 = compute_dv01(Decimal("1000"), Decimal("5"), Decimal("100"))
        dv01_2 = compute_dv01(Decimal("2000"), Decimal("5"), Decimal("100"))
        assert abs(float(dv01_2) - 2 * float(dv01_1)) < 1e-8

    def test_linear_in_modified_duration(self) -> None:
        dv01_1 = compute_dv01(Decimal("1000"), Decimal("5"), Decimal("100"))
        dv01_2 = compute_dv01(Decimal("1000"), Decimal("10"), Decimal("100"))
        assert abs(float(dv01_2) - 2 * float(dv01_1)) < 1e-8

    def test_known_value(self) -> None:
        # DV01 = 5 * (100/100) * 1000 * 0.0001 = 0.5
        dv01 = compute_dv01(Decimal("1000"), Decimal("5"), Decimal("100"))
        assert abs(float(dv01) - 0.5) < 1e-8

    def test_returns_decimal_type(self) -> None:
        result = compute_dv01(Decimal("1000"), Decimal("5"), Decimal("100"))
        assert isinstance(result, Decimal)

    def test_dv01_scales_with_price_pct(self) -> None:
        dv01_par = compute_dv01(Decimal("1000"), Decimal("5"), Decimal("100"))
        dv01_disc = compute_dv01(Decimal("1000"), Decimal("5"), Decimal("50"))
        assert abs(float(dv01_par) - 2 * float(dv01_disc)) < 1e-8


# ---------------------------------------------------------------------------
# compute_convexity
# ---------------------------------------------------------------------------


class TestComputeConvexity:
    def test_zero_coupon_positive_convexity(self) -> None:
        conv = compute_convexity(
            face_value=Decimal("1000"),
            coupon_rate=Decimal("0"),
            ytm=Decimal("0.05"),
            days_to_maturity=1825,
        )
        assert conv > Decimal(0)

    def test_coupon_reduces_convexity_vs_zero_coupon(self) -> None:
        kwargs = {
            "face_value": Decimal("1000"),
            "ytm": Decimal("0.05"),
            "days_to_maturity": 1825,
        }
        conv_zero = compute_convexity(coupon_rate=Decimal("0"), **kwargs)  # type: ignore[arg-type]
        conv_coupon = compute_convexity(coupon_rate=Decimal("0.08"), **kwargs)  # type: ignore[arg-type]
        assert conv_coupon < conv_zero

    def test_longer_bond_has_higher_convexity(self) -> None:
        common = {
            "face_value": Decimal("1000"),
            "coupon_rate": Decimal("0.05"),
            "ytm": Decimal("0.05"),
        }
        conv_short = compute_convexity(days_to_maturity=730, **common)  # type: ignore[arg-type]
        conv_long = compute_convexity(days_to_maturity=3650, **common)  # type: ignore[arg-type]
        assert conv_long > conv_short

    def test_invalid_days_to_maturity_zero(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_convexity(Decimal("1000"), Decimal("0.05"), Decimal("0.05"), 0)

    def test_invalid_days_to_maturity_negative(self) -> None:
        with pytest.raises(InvalidInputError, match="days_to_maturity"):
            compute_convexity(Decimal("1000"), Decimal("0.05"), Decimal("0.05"), -1)

    def test_returns_decimal_type(self) -> None:
        result = compute_convexity(Decimal("1000"), Decimal("0.05"), Decimal("0.05"), 730)
        assert isinstance(result, Decimal)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


class TestHypothesisProperties:
    @given(ytm=_VALID_YTM, days=_VALID_DAYS)
    @settings(max_examples=50)
    def test_discount_factor_returns_decimal(self, ytm: Decimal, days: int) -> None:
        result = compute_discount_factor(ytm, days)
        assert isinstance(result, Decimal)

    @given(
        ytm1=st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("0.3"),
            allow_nan=False,
            allow_infinity=False,
            places=6,
        ),
        ytm2=st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("0.3"),
            allow_nan=False,
            allow_infinity=False,
            places=6,
        ),
        days=_VALID_DAYS,
    )
    @settings(max_examples=50)
    def test_discount_factor_monotone_decreasing_in_ytm(
        self, ytm1: Decimal, ytm2: Decimal, days: int
    ) -> None:
        from hypothesis import assume

        assume(ytm1 < ytm2)
        df1 = compute_discount_factor(ytm1, days)
        df2 = compute_discount_factor(ytm2, days)
        assert df1 > df2

    @given(
        fv1=st.decimals(
            min_value=Decimal("1"),
            max_value=Decimal("999"),
            allow_nan=False,
            allow_infinity=False,
            places=2,
        ),
        fv2=st.decimals(
            min_value=Decimal("1001"),
            max_value=Decimal("100000"),
            allow_nan=False,
            allow_infinity=False,
            places=2,
        ),
        duration=_VALID_DURATION,
        price_pct=_VALID_PRICE_PCT,
    )
    @settings(max_examples=50)
    def test_dv01_monotone_increasing_in_face_value(
        self,
        fv1: Decimal,
        fv2: Decimal,
        duration: Decimal,
        price_pct: Decimal,
    ) -> None:
        dv01_1 = compute_dv01(fv1, duration, price_pct)
        dv01_2 = compute_dv01(fv2, duration, price_pct)
        assert dv01_1 < dv01_2

    @given(ytm=_VALID_YTM, days=_VALID_DAYS)
    @settings(max_examples=50)
    def test_compute_ytm_returns_decimal_for_valid_par_like_input(
        self, ytm: Decimal, days: int
    ) -> None:
        # Construct a par bond for the given ytm to guarantee convergence
        face = Decimal("1000")
        # Use annual coupon = ytm * face as the coupon rate → price ≈ par
        result = compute_ytm(
            face_value=face,
            coupon_rate=ytm,
            price_pct=Decimal("100"),
            days_to_maturity=days,
        )
        assert isinstance(result, Decimal)
