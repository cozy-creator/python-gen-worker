"""Stable host-integration contract shared across the gen-worker CLI."""

from __future__ import annotations

from typing import List

# Wire-format contract version. Bump when the request/response/cancel frame shapes, the describe document, or the serve sidecar change incompatibly.
PROTOCOL_VERSION = 1

CAPABILITIES: List[str] = [
    "endpoint_handle",
    "dispatch_counts",
    "hub_resolve",
]


def gen_worker_version() -> str:
    """Best-effort installed ``gen-worker`` version string (never raises)."""
    try:
        import importlib.metadata as _md

        return _md.version("gen-worker")
    except Exception:
        return "unknown"
