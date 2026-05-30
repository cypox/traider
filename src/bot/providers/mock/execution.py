"""Mock execution provider for testing."""

from bot.core.execution import ExecutionIntent
from bot.providers.base import ExecutionProvider


class MockExecutionProvider(ExecutionProvider):
    """In-process execution provider that records all orders.

    Use :attr:`placed_orders` and :attr:`cancelled_orders` to inspect what was
    submitted.  Order IDs are sequential: ``MOCK-ORDER-001``, ``MOCK-ORDER-002``,
    etc.
    """

    def __init__(self) -> None:
        self._is_connected = False
        self._order_counter = 0
        self.placed_orders: list[ExecutionIntent] = []
        self.cancelled_orders: list[str] = []

    async def connect(self) -> None:
        self._is_connected = True

    async def disconnect(self) -> None:
        self._is_connected = False

    async def place_order(self, intent: ExecutionIntent) -> str:
        """Record *intent* and return a sequential mock order ID."""
        self.placed_orders.append(intent)
        self._order_counter += 1
        return f"MOCK-ORDER-{self._order_counter:03d}"

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled_orders.append(order_id)

    @property
    def is_connected(self) -> bool:
        return self._is_connected
