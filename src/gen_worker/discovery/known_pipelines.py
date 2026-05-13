"""Non-restricting pipeline / model base classes (legacy compat table).

These deliberately-broad base classes (``DiffusionPipeline``,
``AutoModelForCausalLM``, etc.) are pipeline auto-dispatchers — declaring
one of them on an injected parameter means "any subclass of this family".

In 0.7.0 the SDK no longer auto-derives `pipeline_classes` from the
parameter annotation (the explicit tenant-declared
``.allow_override(*classes)`` allowlist is the only authoritative source).
The orchestrator still references this list when normalizing caller-supplied
override classes; the helpers below stay as a reference for that side.

Kept in lockstep with the parallel Go list in
``gen-orchestrator/internal/release/compat_classes.go``.
"""

from __future__ import annotations

# Revision token — incremented whenever the set below changes. Keep in sync
# with compat_classes.KnownPipelinesRevision on the Go side.
KNOWN_PIPELINES_REVISION = "2026-04-24.1"

# Diffusers auto-dispatch / permissive base classes.
# `DiffusionPipeline.from_pretrained(...)` returns whichever concrete subclass
# the `model_index.json._class_name` field points at. A function declaring
# `DiffusionPipeline` is intentionally saying "any diffusers pipeline" — we
# don't gate it.
DIFFUSERS_NON_RESTRICTING: frozenset[str] = frozenset({
    "DiffusionPipeline",
    "AutoPipelineForText2Image",
    "AutoPipelineForImage2Image",
    "AutoPipelineForInpainting",
})

# Transformers auto-dispatch / permissive base classes.
# `AutoModelForCausalLM.from_pretrained(...)` returns whichever concrete
# architecture the `config.json.architectures[0]` points at.
TRANSFORMERS_NON_RESTRICTING: frozenset[str] = frozenset({
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForVision2Seq",
    "AutoModelForImageClassification",
    "AutoModelForImageTextToText",
    "AutoModelForAudioClassification",
    "AutoModelForSpeechSeq2Seq",
    "PreTrainedModel",
})

NON_RESTRICTING_CLASSES: frozenset[str] = DIFFUSERS_NON_RESTRICTING | TRANSFORMERS_NON_RESTRICTING


def is_non_restricting(class_name: str) -> bool:
    """True when `class_name` is a permissive base class that should NOT
    produce a derived pipeline-class gate."""
    return class_name in NON_RESTRICTING_CLASSES


__all__ = [
    "KNOWN_PIPELINES_REVISION",
    "DIFFUSERS_NON_RESTRICTING",
    "TRANSFORMERS_NON_RESTRICTING",
    "NON_RESTRICTING_CLASSES",
    "is_non_restricting",
]
