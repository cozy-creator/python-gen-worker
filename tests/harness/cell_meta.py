"""Build one minimal TCG metadata object for worker publish-path tests.

TCG is the sole compiled-graph identity authority.  This fixture deliberately
has no worker ``cell_key``/``entry``/manifest compatibility shape: it records a
closed ``graph_class`` object and stamps the key that TCG derives from it.
"""

from __future__ import annotations

from typing import Any, Dict

from torch_compiled_graphs import COMPILED_GRAPH_FORMAT
from torch_compiled_graphs.identity import from_artifact_metadata

from gen_worker import aot_serve

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
    """One exported compiled graph's metadata, key stamped by TCG."""
    meta: Dict[str, Any] = {
        "family": family, "sku": sku, "sm": sm, "gen_worker": gen_worker,
        "kind": aot_serve.ARTIFACT_KIND,
        "compiled_graph_format": COMPILED_GRAPH_FORMAT,
        "weight_lane": weight_lane, "lora_bucket": int(lora_bucket),
        "graph_class": {
            "name": "unet/main",
            "target": "unet",
            "class_hash": CLASS_HASH,
            "graph": {
                "v": 3,
                "constant_fqns": [],
                "lifted_inputs": [],
                "pytree": {},
                "specialization": {},
            },
        },
        "env_seal": {"v": 1, "torch": "2.9.0"},
        "toolchain": {"torch": "2.9.0", "cuda": "12.8"},
    }
    meta.update(extra)
    meta["compiled_graph_key"] = from_artifact_metadata(meta).value
    return meta
