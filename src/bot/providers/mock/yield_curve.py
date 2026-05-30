"""Mock yield-curve provider for testing."""

from datetime import datetime

from bot.providers.base import YieldCurveProvider
from bot.providers.models import YieldCurve


class MockYieldCurveProvider(YieldCurveProvider):
    """In-process yield-curve provider that always returns a fixed curve."""

    def __init__(self, fixed_curve: YieldCurve) -> None:
        self._fixed_curve = fixed_curve

    async def get_curve(self, as_of: datetime) -> YieldCurve:
        """Return the fixed curve regardless of *as_of*."""
        return self._fixed_curve
