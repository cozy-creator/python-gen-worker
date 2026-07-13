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

# Optional features a host can rely on without scraping ``--help``. Add a token
# only when the feature is actually implemented (see the issues in parens):
#   - "describe"       : `gen-worker describe --json` (#349)
#   - "list_functions" : `gen-worker serve --list-functions [--json]` (#340)
#   - "prefetch"       : `gen-worker prefetch` (#21)
#   - "cancel"         : per-request `{"cancel":{"request_id"}}` control (#352)
#   - "streaming"      : multi-frame streamed responses, `{"stream":true}` (#344)
#   - "tcp_listen"     : `serve --listen tcp://host:port` (#347)
#   - "serve_sidecar"  : machine-readable `.gen-worker.serve.json` handle (#349)
#   - "hub_resolve"    : standalone Hub-ref resolve via TENSORHUB_URL (#379)
CAPABILITIES: List[str] = [
    "describe",
    "list_functions",
    "prefetch",
    "cancel",
    "streaming",
    "tcp_listen",
    "serve_sidecar",
    "hub_resolve",
]


def gen_worker_version() -> str:
    """Best-effort installed ``gen-worker`` version string (never raises)."""
    try:
        import importlib.metadata as _md

        return _md.version("gen-worker")
    except Exception:
        return "unknown"
