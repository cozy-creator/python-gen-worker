from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from gen_worker import graph_facts

import tcg_artifacts

SPECIALIZATION_HASH = str(tcg_artifacts.metadata()["graph_specialization"]["specialization_hash"])


def exported_compiled_graph_meta(
    *,
    sm: str = "sm_89",
    graph_specialization: str = tcg_artifacts.GRAPH_CLASS,
    witness: str = "fedcba9876543210",
    toolchain: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """One exported (``aot-inductor``) compiled graph's metadata, as TCG builds it."""
    return tcg_artifacts.metadata(
        graph_specialization=graph_specialization, witness=witness, sm=sm, toolchain=toolchain)
