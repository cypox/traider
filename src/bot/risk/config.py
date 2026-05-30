"""Risk configuration dataclass."""

from dataclasses import dataclass
from decimal import Decimal

from bot.risk.errors import InvalidRiskConfigError


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Immutable risk limits applied by the RiskEngine.

    All monetary values are in USD.  All percentage values are expressed as
    fractions (e.g. ``0.05`` for 5 %).
    """

    max_position_usd: Decimal
    max_gross_exposure_usd: Decimal
    max_drawdown_pct: Decimal
    max_concentration_pct: Decimal
    daily_loss_limit_usd: Decimal

    def __post_init__(self) -> None:
        checks: list[tuple[str, Decimal]] = [
            ("max_position_usd", self.max_position_usd),
            ("max_gross_exposure_usd", self.max_gross_exposure_usd),
            ("max_drawdown_pct", self.max_drawdown_pct),
            ("max_concentration_pct", self.max_concentration_pct),
            ("daily_loss_limit_usd", self.daily_loss_limit_usd),
        ]
        for field_name, value in checks:
            if value <= Decimal(0):
                raise InvalidRiskConfigError(f"{field_name} must be positive; got {value}")
