"""The publish path's fixture, built by TCG — never by hand (pgw#1046/pgw#1341).

The publish path recomputes a compiled graph's identity from its own recorded blocks and
refuses anything that cannot state one, because the axis map it declares is what
tensorhub's RunAttempt producer builds ``Arm.artifact`` /
``Arm.graph_contract_digest`` out of, and pgw#904's landed consumer refuses an
``ArtifactIdentity`` missing any of them.

**THE FIXTURE FENCE, and why this module was rewritten.** Until pgw#1341 this
built the envelope BY HAND — a dict with ``family``, ``sku``, ``gen_worker``,
``weight_lane``, ``manifest_digest`` and ``env_seal`` in it. Since pgw#1270 TCG
mints every artifact and ``torchcg.artifact.validate_metadata`` refuses metadata
whose field set is not exactly ``artifact_meta.compiled_graph_metadata_fields()`` — which
holds none of those six names. So the fixture described a compiled graph that cannot
exist, and the whole publish path was tested against it: ``_identity_axes``
raised ``CompiledGraphPublishRefused("compiled graph records no env_seal block")`` for every real
artifact on the fleet while this file kept CI green. That is pgw#1277's finding
verbatim (*"CI stayed green because every fixture built the obsolete shape"*),
one seam further along.

The metadata therefore comes from ``tcg_artifacts.metadata()``, i.e. from
``torchcg.artifact.build_metadata`` — the same builder production uses — and
there is deliberately no keyword here for a field TCG does not accept.

pgw#1573 deleted the v1 local compiled-graph store, and with it the
``MintProvenance`` sidecar this harness used to build: a v2 artifact's
envelope carries its own metadata, so there is nothing left to state beside
the bytes.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from gen_worker import graph_facts

import tcg_artifacts

#: The fixture's graph-specialization hash, derived by TCG from the declaration
#: ``tcg_artifacts`` builds — read, never asserted, so a TCG fold change moves
#: this with the key instead of leaving a stale literal behind.
SPECIALIZATION_HASH = str(tcg_artifacts.metadata()["graph_specialization"]["specialization_hash"])


def exported_compiled_graph_meta(
    *,
    sm: str = "sm_89",
    graph_specialization: str = tcg_artifacts.GRAPH_CLASS,
    witness: str = "fedcba9876543210",
    toolchain: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """One exported (``aot-inductor``) compiled graph's metadata, as TCG builds it.

    Every keyword is an axis a REAL artifact carries. There is no keyword for
    ``family``/``sku``/``gen_worker``/``weight_lane``: a compiled graph states none of
    them, and a fixture that let a test pretend otherwise is what hid pgw#1341
    for two wheels.
    """
    return tcg_artifacts.metadata(
        graph_specialization=graph_specialization, witness=witness, sm=sm, toolchain=toolchain)
