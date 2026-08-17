"""Precision classes — the producer's classification of a flavor token.

A quant token's *precision class* names its quantization lane (``fp8``,
``svdq-int4``, ...). Produced flavors carry it in
``checkpoints.metadata["placement"]["precision_class"]`` (written at publish by
:func:`gen_worker.convert.publish.publish_flavors`), which is the hub's
strongest evidence for a stored class where no tensor-layout contract is proven
(tensorhub ``precision.StoredPrecisionOf``). The ladder WALK, the family-root
table and the family lane policy live hub-side and reach the worker in the
resolved ref/HelloAck; this module is the PRODUCER half only — classification.

pgw#1300 / th#2055: the PLACEMENT half of this module is deleted. ``Placement``,
``default_placement`` and ``placement_to_metadata`` stamped an SM allow-list, an
SM floor and an engine list into that same block, and the hub read them to admit
or refuse a card. **Pod purchase now depends only on the endpoint owner's
(GPU, lane) ladder**, so the hub deleted every reader (`PlacementFromMetadata`,
`defaultPlacement`, `admitted()`, `AdmitLiteral`'s sm/engine arms) and
`StoredPrecisionOf` reads `precision_class` and nothing else. The stamp was not
merely unread — it was WRONG and vetoing: `sm_allowed=(120, 121)` plus
`engines=("nunchaku",)` for svdq-fp4 described a kernel window and a wheel the
native engine has not needed since pgw#685, so a B200 (sm_100) could not bind a
flavor it serves. Correcting it was rejected: a correct allow-list still vetoes
an owner's own rung. There is no local walk and no local AUTO fp8 fold; locally,
fit is the loading layer's job and selection within a tag group is §1.33
contract compatibility.
"""

from __future__ import annotations

CLASS_BASE = "base"  # bare bf16/fp16/fp32 row — runs anywhere a card fits it
CLASS_FP8 = "fp8"  # fp8-E4M3 storage; universal (bf16-upcast path needs no fp8 silicon)
CLASS_SVDQ_FP4 = "svdq-fp4"  # SVDQuant fp4 — served natively (svdq_native)
CLASS_SVDQ_INT4 = "svdq-int4"  # SVDQuant int4 — no native engine; a typed refusal
CLASS_NVFP4 = "nvfp4"  # plain nvfp4 artifact — no serving lane (not a diffusers rung)
CLASS_NVFP4_W4A4 = "nvfp4-w4a4"  # calibrated nvfp4, two-level scales — fp4 scaled_mm lane

_BASE_TOKENS = ("", "bf16", "fp16", "fp32")


def classify_flavor_token(flavor: str) -> str:
    """Flavor token -> precision class; "" when unrecognized (gguf/etc.
    stay opaque — never ladder rungs)."""
    token = str(flavor or "").strip().lower()
    if token in _BASE_TOKENS:
        return CLASS_BASE
    if token.startswith("svdq-fp4"):
        return CLASS_SVDQ_FP4
    if token.startswith("svdq-int4"):
        return CLASS_SVDQ_INT4
    if token == "fp8" or token.startswith("fp8-"):
        return CLASS_FP8
    if token == "nvfp4-w4a4" or token.startswith("nvfp4-w4a4-"):
        return CLASS_NVFP4_W4A4
    if token == "nvfp4" or token.startswith("nvfp4-"):
        return CLASS_NVFP4
    return ""


# `classify_flavor_token` is a package-internal choke point, not public API. It
# dies when typed descriptors are backfilled; nothing new may grow on it.
__all__ = [
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_NVFP4",
    "CLASS_NVFP4_W4A4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "classify_flavor_token",
]
