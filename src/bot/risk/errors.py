"""Risk engine exception types."""

from bot.core.errors import TradingError


class InvalidRiskConfigError(TradingError):
    """Raised when a RiskConfig is constructed with invalid field values."""
