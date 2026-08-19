"""Precision classes — the VOCABULARY a producer declares its class from.

A *precision class* names an artifact's quantization lane (``fp8``,
``svdq-int4``, ...). Produced flavors carry it in
``checkpoints.metadata["placement"]["precision_class"]`` (written at publish by
:func:`gen_worker.convert.publish.publish_flavors`), which is the hub's
strongest evidence for a stored class where no tensor-layout contract is proven
(tensorhub ``precision.StoredPrecisionOf``). The ladder WALK, the family-root
table and the family lane policy live hub-side and reach the worker in the
resolved ref/HelloAck; this module is the PRODUCER half only.

A18 / §1.32(d), pgw#1319: ``classify_flavor_token`` is DELETED with
``ProducedFlavor.flavor``. The token was never an address, and inferring a
precision class from a producer-local LABEL was the last thing that made it
behave like one — it also got the answer wrong in both directions, matching
``fp8``/``fp8-*`` but never ``*-fp8``, so ``coherent-fp8`` and
``encoder-trunc-fp8`` published as base for as long as they existed (te#225).
**The class is DECLARED now**, in the producer's own attribute bag, derived
from a structural fact (the artifact's precision field, the quantization
scheme, the requested dtype, the safetensors header). What remains here is the
vocabulary those declarations are checked against: one home, and the publish
gate refuses a value outside it rather than forwarding prose the hub cannot
read.

pgw#1300 / th#2055: the PLACEMENT half of this module is deleted — no SM
allow-list, no SM floor, no engine list. **Pod purchase depends only on the
endpoint owner's (GPU, lane) ladder**, so the hub reads `precision_class` and
nothing else. Correcting the stamp was REJECTED rather than attempted: a
correct allow-list still vetoes an owner's own rung. There is no local walk and
no local AUTO fp8 fold; locally,
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
#: pgw#1498. GGML block-quantized storage (Q4_K, Q6_K, Q8_0, ...), served on the
#: ordinary torch path by decoding each weight inside its own forward
#: (`models/gguf_torch`). ONE class, not one per qtype: the qtype is a property
#: of the artifact's bytes — it travels in the tensor-layout contract and the
#: checkpoint's own quant metadata — while the CLASS is what the hub's ladder
#: ranks, and every ggml qtype ranks the same way against fp8 and base. Naming
#: it here is what lets a producer declare it at all; the ladder still SELECTS
#: the artifact and never manufactures one (`rung.py:8-22`).
#: **OWED, hub-side and named:** `precision.StoredPrecisionOf` does not rank
#: this class yet, so a stamped row is not yet selectable by the hub ladder.
#: That is the tensorhub half of the same storage ruling as the quant
#: layout-contract registration (`loading.py`'s `@unregistered_decode_path`).
CLASS_GGUF = "gguf"

#: Every class a producer may declare. A publish naming anything else is
#: refused at the call site (`convert.publish`): the hub maps these and only
#: these, so an unrecognized value would land in checkpoint metadata as prose
#: and the row would serve as base — the silent outcome A18 exists to end.
PRECISION_CLASSES = frozenset({
    CLASS_BASE,
    CLASS_FP8,
    CLASS_GGUF,
    CLASS_NVFP4,
    CLASS_NVFP4_W4A4,
    CLASS_SVDQ_FP4,
    CLASS_SVDQ_INT4,
})


__all__ = [
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_GGUF",
    "CLASS_NVFP4",
    "CLASS_NVFP4_W4A4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "PRECISION_CLASSES",
]
