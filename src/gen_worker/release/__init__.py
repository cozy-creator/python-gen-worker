"""Publish-time release machinery (pgw#1370): the instrumented derive.

``gen-worker release derive`` runs INSIDE the release env, drives the
module's ``@entrypoint`` functions with payloads auto-enumerated from their
schemas under torchcg's instrumented discovery, and emits the static
release metadata document.
Serving pods adopt that document; they never derive (pgw#1372).
"""

from .derive import DeriveError, ReleaseDeriveResult, derive_release
from .trace_context import TraceLoadContext, TraceRequestContext

__all__ = [
    "DeriveError",
    "ReleaseDeriveResult",
    "TraceLoadContext",
    "TraceRequestContext",
    "derive_release",
]
