"""Analytics exception hierarchy."""

from bot.core.errors import TradingError


class AnalyticsError(TradingError):
    """Base class for analytics errors."""


class ConvergenceError(AnalyticsError):
    """Raised when a numerical solver fails to converge."""


class InvalidInputError(AnalyticsError):
    """Raised on bad input parameters."""
