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

__all__ = [
    "DocumentError",
    "GraphRecord",
    "GraphSetDocument",
    "LaneError",
    "LaneGraphs",
    "LaneRef",
    "PassOrderError",
    "parse_lane_id",
    "require_lane_id",
    "require_passes",
    "require_targets",
    "resolve_target",
]
