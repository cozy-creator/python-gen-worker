"""Non-restricting pipeline / model base classes (e2e progress.json #46).

When a ``@inference_function`` uses ``Src.PAYLOAD_REF`` on a parameter whose
type annotation is one of these deliberately-broad base classes, discovery
emits NO derived pipeline-class gate — callers can pass any subclass-family
ref and the orchestrator's compat validator falls through to the other
axes (file_layout / lineage / attributes) for that gate. Concrete leaf
classes (``StableDiffusionXLPipeline``, ``Flux2KleinPipeline``, ...) DO
produce a restricting gate.

Kept in lockstep with the parallel Go list in
``gen-orchestrator/internal/release/compat_classes.go``. A drift between
these two lists is a correctness bug: signatures judged non-restricting on
the publish side but restricting on the orchestrator side (or vice versa)
would silently change gate behavior.

When adding a new class: add the entry in BOTH files + bump the shared
``KNOWN_PIPELINES_REVISION`` string on both sides. A lint / CI check asserts
revisions match before a build succeeds.
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
