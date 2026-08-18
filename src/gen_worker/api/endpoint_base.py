"""``gen_worker.Endpoint`` -- the required base for ship-code-as-is endpoints.

Paul's main_v2.py review ruling (pgw#1367 program): every lanes-surface
endpoint class inherits this minimal base. It carries typed no-op lifecycle
hooks and NOTHING else -- capabilities reach author code through ``ctx``
only, and the ``@endpoint`` decorator stays the metadata carrier. The
serving lane (pgw#1372) owns this class's evolution; keep it minimal here.
"""

from __future__ import annotations

from typing import Any


class Endpoint:
    """Base class for ``@endpoint(lanes=...)`` classes.

    ``setup`` runs once per resident lane before any handler; ``teardown``
    runs when the instance is retired. Both are no-op hooks an endpoint
    overrides as needed -- handlers are ordinary public methods
    ``(self, ctx, payload)``.
    """

    def setup(self, ctx: Any) -> None:  # noqa: B027 - deliberate no-op hook
        """Load and mark models; the default does nothing."""

    def teardown(self, ctx: Any) -> None:  # noqa: B027 - deliberate no-op hook
        """Release what setup acquired; the default does nothing."""


__all__ = ["Endpoint"]
