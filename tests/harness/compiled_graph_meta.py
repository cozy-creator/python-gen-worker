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

Everything the publish needs and the artifact cannot say now lives in
:func:`exported_compiled_graph_provenance`, which is the object production writes beside
the bytes (``local_compiled_graph_store.MintProvenance``).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from gen_worker import graph_facts
from gen_worker.local_compiled_graph_store import MintProvenance

import tcg_artifacts

#: The fixture's graph-class hash, derived by TCG from the declaration
#: ``tcg_artifacts`` builds — read, never asserted, so a TCG fold change moves
#: this with the key instead of leaving a stale literal behind.
CLASS_HASH = str(tcg_artifacts.metadata()["graph_class"]["class_hash"])


def exported_compiled_graph_meta(
    *,
    sm: str = "sm_89",
    graph_class: str = tcg_artifacts.GRAPH_CLASS,
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
        graph_class=graph_class, witness=witness, sm=sm, toolchain=toolchain)


def exported_compiled_graph_provenance(
    *,
    lane: str = "bf16-w16a16",
    sku: str = "l4",
    gen_worker: str = "0.87.0",
    env_seal: str = "seal-" + "1" * 16,
    graph_contract: str = "",
) -> MintProvenance:
    """The mint facts that ride BESIDE the artifact (pgw#1341).

    Production writes this into the local store's sidecar at the moment the
    bytes become durable, and both the immediate publish and a later boot's
    ``resume_owed_publishes`` read it back. A test that publishes must supply
    one for the same reason a pod must: the artifact cannot.
    """
    return MintProvenance(
        env_seal=env_seal, lane=lane,
        graph_contract=(graph_contract
                        or graph_facts.manifest_digest([CLASS_HASH])),
        sku=sku, gen_worker=gen_worker)
