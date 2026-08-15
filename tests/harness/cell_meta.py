"""pgw#1046: build a REAL exported-cell envelope for the publish-path tests.

The publish path recomputes a cell's identity from its own recorded blocks and
refuses anything that cannot state one — because the axis map it declares is
what tensorhub's RunAttempt producer builds ``Arm.artifact`` /
``Arm.graph_contract_digest`` out of, and pgw#904's landed consumer refuses an
``ArtifactIdentity`` missing any of them. A stub ``{"compiled_graph_key": …, "sku": …}``
is therefore no longer a publishable cell in any tree, and a test that ships one
proves only that an unarmable row still uploads.

This is the minimal envelope that IS one: the same blocks ``aot_mint`` records
(``aot_serve.entry_metadata`` + ``shared_identity_blocks``), at fixture
scale. The ``compiled_graph_key`` is COMPUTED, never invented, so the stamp and the axes
agree — which is itself one of the publish path's refusals.

pgw#1176: one artifact is ONE graph class, so this builds an ``entry`` block
and no ``entries`` map. There is deliberately no way to ask this harness for a
multi-entry artifact: production cannot pack one any more, and a fixture that
could would be testing a shape nothing can produce.
"""

from __future__ import annotations

from typing import Any, Dict

from gen_worker import aot_serve, compiled_graph_key

CLASS_HASH = "a" * 16


def exported_cell_meta(
    *,
    family: str = "fam",
    sku: str = "l4",
    sm: str = "89",
    gen_worker: str = "0.87.0",
    weight_lane: str = "",
    lora_bucket: int = 0,
    **extra: Any,
) -> Dict[str, Any]:
    """One exported (``aot-inductor``) cell's metadata, key stamped from it."""
    meta: Dict[str, Any] = {
        "family": family, "sku": sku, "sm": sm, "gen_worker": gen_worker,
        "kind": aot_serve.ARTIFACT_KIND, "format": "pt2",
        "weight_lane": weight_lane, "lora_bucket": int(lora_bucket),
        "strict_export": True,
        compiled_graph_key.ENTRY_BLOCK_KEY: {
            "name": "unet/main",
            "target": "unet", "fork": [], "class_dims": [],
            "range_digest": "r1", "class_hash": CLASS_HASH, "graph": {"v": 2},
        },
        # The declaration-wide coverage LABEL. Telemetry, never identity —
        # `_identity_axes` publishes it as `graph_contract`, and it may repeat
        # across the entries of one declaration by construction.
        "manifest_digest": compiled_graph_key.manifest_digest([CLASS_HASH]),
        "env_seal": {"v": 1, "torch": "2.9.0"},
        "toolchain": {"torch": "2.9.0", "cuda": "12.8"},
    }
    meta.update(extra)
    meta["compiled_graph_key"] = compiled_graph_key.from_entry_metadata(meta).digest
    return meta
