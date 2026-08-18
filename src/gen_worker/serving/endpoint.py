"""The REQUIRED minimal endpoint base class (Paul ruling, 2026-08-18).

The split: the ``@endpoint`` decorator carries DATA (lanes, compile targets —
extracted at publish); this base is the BEHAVIOR CONTRACT — typed overridable
hooks only. ``setup(ctx)`` and ``teardown(ctx)`` default to no-ops, so the
smallest endpoint is a subclass with one handler method.

HARD GUARDRAIL: framework capabilities arrive via ``ctx`` ONLY — never via
inherited methods or self-state. This base is a typed skeleton, not a
toolbox; nothing else goes on it without a Paul ruling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep the base import-weightless
    from .context import ServeContext


class Endpoint:
    """Base every endpoint class inherits: ``class MyEndpoint(Endpoint)``.

    Handlers are ordinary public methods ``(self, ctx, payload)`` with a
    msgspec payload annotation; the loader routes to them by function name.
    """

    def setup(self, ctx: "ServeContext") -> None:
        """One-time boot hook: load the pipeline from ``ctx.checkpoint_dir``
        under ``ctx.lane.dtype``, keep it on ``self``. No-op by default."""

    def teardown(self, ctx: "ServeContext") -> None:
        """Shutdown hook: release what ``setup`` acquired. No-op by default."""


__all__ = ["Endpoint"]
