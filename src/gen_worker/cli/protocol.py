"""Stable host-integration contract shared across the gen-worker CLI.

A host orchestrator (cozy-local) drives gen-worker over the CLI + the serve
socket. To integrate without scraping ``--help`` or guessing wire shapes, every
machine-readable surface (``describe --json``, the serve ready-sidecar) carries
``protocol_version`` + ``capabilities``. Bump ``PROTOCOL_VERSION`` on any
wire-format change; advertise an optional feature by adding its token to
``CAPABILITIES`` once it actually ships.

The full design lives in ``progress.json`` issue #349.
"""

from __future__ import annotations

from typing import List

# Wire-format contract version. Bump when the request/response/cancel frame
# shapes, the describe document, or the serve sidecar change incompatibly.
PROTOCOL_VERSION = 1

# Optional features a host can rely on without scraping ``--help``. A token
# goes in ONLY when the feature actually ships. pgw#1491 removed six that named
# deleted verbs (`describe`, `list_functions`, `prefetch`, `cancel`,
# `streaming`, `serve_sidecar`, `tcp_listen`) — a capability list that lies is
# worse than no list, because an integrator branches on it:
#   - "endpoint_handle" : the ~/.cache/cozy/resident/<key>/endpoint.json handle
#                         `up` publishes and `run`/`down` read
#   - "dispatch_counts" : every response carries compiled-vs-eager call counts
#   - "hub_resolve"     : standalone Hub-ref resolve via TENSORHUB_URL
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
