"""Producing compiled-graph programs by tracing this worker's endpoints.

**This is pgw's half of the tcg#90 seam.** torchcg is `program -> keyed
artifact`: identity, mint, store, adopt, refuse, and it knows nothing of lanes,
endpoints or forward hooks. Everything that DRIVES author code to find out which
programs exist -- lane vocabulary, the hollow (weights-free) trace session, the
forward-hook observation pass, the dynamic-axis plan, and the document types the
release produces at derive and consumes at adopt-first boot -- is release-process
orchestration and lives here.

Moved from torchcg's vendored tree as pgw#1603 left it: a move and a rewire, not
a rewrite. The primitives these modules stand on (`graph_hash`,
`build_call_ingress`, `respecialize`, `strip_diagnostics`, `compile_stack`) are
IMPORTED from torchcg rather than duplicated.
"""

from __future__ import annotations

from .document import (
    DocumentError,
    GraphRecord,
    GraphSetDocument,
    LaneGraphs,
)
from .lane import (
    LaneError,
    LaneRef,
    PassOrderError,
    parse_lane_id,
    require_lane_id,
    require_passes,
    require_targets,
    resolve_target,
)

#: Names that live in torch-shaped modules. Resolved on ATTRIBUTE ACCESS so that
#: importing `gen_worker.graphs` — which `document` and `lane` alone satisfy —
#: never drags torch in. The derive imports torch anyway; the adopt-first boot
#: and the lock reader do not, and they read documents.
_LAZY = {
    "DiscoveryError": "discovery",
    "discover_lane": "discovery",
    "discover_modules": "discovery",
    "HollowError": "hollow",
    "HollowSession": "hollow",
    "hollow_session": "hollow",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted([*__all__, *_LAZY])


__all__ = [
    "DiscoveryError",
    "DocumentError",
    "GraphRecord",
    "GraphSetDocument",
    "HollowError",
    "HollowSession",
    "LaneError",
    "LaneGraphs",
    "LaneRef",
    "PassOrderError",
    "parse_lane_id",
    "require_lane_id",
    "require_passes",
    "require_targets",
    "discover_lane",
    "discover_modules",
    "hollow_session",
    "resolve_target",
]
