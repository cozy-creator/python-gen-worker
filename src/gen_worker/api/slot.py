"""``Slot`` — a hub-resolved model slot (SDK v2, pgw#647).

The model SET is catalog, not code: tensorhub owns the mapping from a
``models={}`` slot to the checkpoint(s) it may resolve to. Under SDK v2 the
declaration shrinks to the ROOT of the component tree::

    from gen_worker import HF, RequestContext, Slot, endpoint
    from gen_worker.families import SdxlDefaults

    @endpoint(models={
        "pipeline": Slot(StableDiffusionXLPipeline, selected_by="model"),
    })
    class Generate:
        def setup(self, pipeline: StableDiffusionXLPipeline) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext[SdxlDefaults], p: In) -> Out:
            steps = p.steps if p.steps is not None else ctx.defaults.steps

Component parts (``pipeline.unet`` / ``pipeline.vae`` /
``pipeline.text_encoder`` / ``pipeline.scheduler``) are DERIVED from the
pipeline class and addressable by path — never declared as sibling slots
(``gen_worker.api.tree``). Component overrides (the SDXL VAE fix) are
CATALOG DATA (th#1116), not endpoint code. Explicit multi-slot declaration
survives only as the escape hatch for runtimes the SDK cannot introspect
(llama/gguf, custom engines).

The per-model CONFIG SCHEMA is derived from the handler's context
annotation (``ctx: RequestContext[SdxlDefaults]``), never declared on the
Slot: code owns the schema, the catalog owns the values (th#1116 stamps one
resolved recipe per slot). ``Slot(default_config=...)`` and
``Slot(share_components=...)`` are DELETED (v2 hard cut): code-side recipe
values are gone, and component sharing is automatic by content address.
"""

from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar

import msgspec

from .binding import ModelRef
from ..families.base import KIND_LORA, GenerationDefaults, family_for

D = TypeVar("D", bound=GenerationDefaults)

# th#1017 inference regimes: a checkpoint-level fact about what inference
# configuration the WEIGHTS demand. "distilled" routes (CFG-off contract
# only); "v_prediction" configures (scheduler prediction_type) — both are
# hub-classified at ingest; the SDK only consumes.
REGIMES = ("standard", "v_prediction", "distilled")
DEFAULT_REGIMES = ("standard",)


class RegimeMismatchError(ValueError):
    """A resolved checkpoint's inference_regime is outside the invoked
    function's declared ``regimes=`` (th#1017). Hub routing enforces this
    upstream; reaching here means version skew or a hub bug."""


class Slot(Generic[D]):
    """One ``models={}`` slot as a hub-resolved value.

    ``pipeline_cls`` names the slot's load-time compat — what ``setup()``/
    handler injection constructs. When it is an introspectable pipeline
    class (diffusers-style: exposes ``_get_signature_keys`` or a components
    signature) the SDK derives the slot's COMPONENT TREE from it at
    discovery time and publishes the tree into the release manifest; parts
    are then addressable by path (``pipeline.vae``) for catalog policy and
    component-level overrides. A ``str``/``Path`` class is the escape hatch
    for self-loading runtimes — no tree is derived.

    ``selected_by`` names the ``str``-typed payload field that branches this
    slot at request time. Validated at registration (registry.py) against
    the handler's payload type — the field must exist and be typed plain
    ``str`` (the schema enum of legal values is overlaid live by the hub,
    never baked into the SDK).

    ``default_checkpoint`` seeds the hub mapping at first publish and is the
    ONLY resolution source in hub-less mode (``cozy run``, hermetic tests) —
    a live hub mapping always wins when present. ``None`` means this slot
    has no code-side bootstrap ref: it only resolves against a hub mapping.
    """

    __slots__ = ("pipeline_cls", "selected_by", "default_checkpoint")

    def __init__(
        self,
        pipeline_cls: type,
        *,
        selected_by: str = "",
        default_checkpoint: Optional[ModelRef] = None,
    ) -> None:
        if not isinstance(pipeline_cls, type):
            raise TypeError(
                f"Slot(pipeline_cls=...) must be a class, got "
                f"{type(pipeline_cls).__name__}"
            )
        if default_checkpoint is not None and not isinstance(default_checkpoint, ModelRef):
            raise TypeError(
                f"Slot(default_checkpoint=...) must be a ModelRef (Hub/HF/"
                f"Civitai/ModelScope), got {type(default_checkpoint).__name__}"
            )
        self.pipeline_cls = pipeline_cls
        self.selected_by = str(selected_by or "").strip()
        self.default_checkpoint = default_checkpoint

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Slot({self.pipeline_cls.__name__}, selected_by={self.selected_by!r}, "
            f"default_checkpoint={self.default_checkpoint!r})"
        )


class ResolvedSlot(Generic[D]):
    """What ``ctx.slots[name]`` hands the handler: the resolved
    :class:`ModelRef` plus ONE typed defaults object — the catalog-resolved
    recipe decoded against the handler's declared config schema
    (``ctx: RequestContext[D]``).

    Explicit PAYLOAD values still win over ``.defaults`` — that precedence
    is handler logic; this object only carries the resolved catalog recipe.

    ``regime`` (th#1017) is the resolved checkpoint's inference regime —
    ``"standard"`` unless the hub classified the weights otherwise
    (``"v_prediction"`` | ``"distilled"``). ``ctx.for_request`` applies it
    to the per-request scheduler view automatically; handlers may also
    branch on it (e.g. a dual-mode turbo lane skips its distillation LoRA
    for an already-distilled checkpoint).
    """

    __slots__ = ("ref", "defaults", "regime")

    def __init__(self, ref: ModelRef, defaults: D, regime: str = "standard") -> None:
        self.ref = ref
        self.defaults = defaults
        self.regime = regime

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ResolvedSlot(ref={self.ref!r}, defaults={self.defaults!r}, "
            f"regime={self.regime!r})"
        )


def _apply_lora_overrides(
    name: str, base: D, fam: str, lora_metadata_json: Sequence[str],
) -> D:
    """pgw#516 composition rule: apply each lora's non-``None`` fields onto
    ``base`` (the checkpoint's already-resolved recipe) FIELD BY FIELD, in
    ``lora_metadata_json`` order (a later lora's non-``None`` field wins over
    an earlier one's on the same field).

    Only fields ``base``'s own struct declares participate — a lora's
    LoRA-only fields (``trigger_words``, ``recommended_weight``: no
    checkpoint-recipe analog) and ``schema_version`` are never merged in.
    Missing/empty entries are skipped. A lora family with no registered
    ``kind="lora"`` vocabulary is skipped silently (best-effort enhancement:
    an unmerged lora override never blocks the checkpoint's own resolved
    recipe). A present-but-MALFORMED lora metadata document (tensorhub
    already schema-validated it at PUT time — a decode failure here means
    real version skew) raises, matching the checkpoint metadata's own
    fail-loud posture.
    """
    if not lora_metadata_json:
        return base
    lora_cls = family_for(fam, kind=KIND_LORA) if fam else None
    if lora_cls is None:
        return base
    base_fields = set(type(base).__struct_fields__)
    result = base
    for i, raw in enumerate(lora_metadata_json):
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            lora_defaults: Any = msgspec.json.decode(raw.encode("utf-8"), type=lora_cls)
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            raise ValueError(
                f"slot {name!r}: loras[{i}] inference-defaults metadata failed "
                f"{lora_cls.__name__} validation: {exc}"
            ) from exc
        overrides: Dict[str, Any] = {}
        for f in lora_defaults.__struct_fields__:
            if f == "schema_version" or f not in base_fields:
                continue
            v = getattr(lora_defaults, f)
            if v is not None:
                overrides[f] = v
        if overrides:
            result = msgspec.structs.replace(result, **overrides)
    return result


def _finish_resolved(
    name: str,
    ref: ModelRef,
    defaults: Any,
    *,
    inference_regime: str,
    allowed_regimes: Optional[Sequence[str]],
) -> "ResolvedSlot[Any]":
    """Build the ``ResolvedSlot`` and, when the caller knows the invoked
    function's declared regimes, enforce the th#1017 backstop: the hub
    enforces checkpoint-regime/function-regime compatibility at deploy and
    request time upstream — reaching a mismatch here means version skew or
    a hub bug, never a normal-path outcome."""
    resolved = ResolvedSlot(ref=ref, defaults=defaults, regime=inference_regime)
    if allowed_regimes is not None and resolved.regime not in allowed_regimes:
        raise RegimeMismatchError(
            f"slot {name!r}: resolved checkpoint regime {resolved.regime!r} is "
            f"not in the invoked function's declared regimes {tuple(allowed_regimes)!r}"
        )
    return resolved


def resolve_slot(
    name: str,
    slot: "Slot[D]",
    *,
    ref: Optional[ModelRef],
    defaults_cls: Optional[Type[D]] = None,
    family: str = "",
    raw_metadata_json: str = "",
    lora_metadata_json: Sequence[str] = (),
    inference_regime: str = "standard",
    allowed_regimes: Optional[Sequence[str]] = None,
) -> "ResolvedSlot[Any]":
    """SDK v2 resolution chain: decode the catalog-resolved recipe
    (``raw_metadata_json``) against the handler's DERIVED config schema
    (``defaults_cls``, from ``ctx: RequestContext[D]``), then apply per-lora
    FIELD-LEVEL overrides — shared by the production executor and the
    hub-less CLI path.

    th#1116 moved recipe VALUES to the catalog: the hub stamps ONE resolved
    recipe per slot, so in production the metadata branch always runs. With
    no metadata (hub-less ``cozy run``, hermetic tests, a family the hub has
    not stamped) the NEUTRAL SCHEMA DEFAULTS (``defaults_cls()``) apply —
    exactly the hub's neutral stamp, so both paths agree. There is no
    code-side recipe fallback (v2 deleted ``Slot(default_config=...)``):
    code owns the schema only.

    ``lora_metadata_json`` (pgw#516, in lora-ride order) applies LAST, field
    by field — see :func:`_apply_lora_overrides`.

    ``inference_regime`` (th#1017) is the resolved checkpoint's hub-
    classified regime ("standard" on hubs/paths that don't send one).
    ``allowed_regimes``, when given, is the invoked function's declared
    ``regimes=`` — see :func:`_finish_resolved`.
    """
    if ref is None:
        raise ValueError(
            f"slot {name!r}: no resolved model ref for this request (no "
            "Slot(default_checkpoint=...) and no hub resolution)"
        )
    fam = str(family or "").strip()
    if defaults_cls is None and fam:
        defaults_cls = family_for(fam)  # type: ignore[assignment]
    raw = (raw_metadata_json or "").strip()
    if raw:
        if defaults_cls is None:
            raise ValueError(
                f"slot {name!r}: catalog recipe metadata present but no "
                "config schema is derivable — annotate the handler's context "
                "parameter as RequestContext[YourDefaults] (or declare "
                "Compile(family=...)) so the SDK knows which vocabulary to "
                "decode it against"
            )
        try:
            defaults: Any = msgspec.json.decode(raw.encode("utf-8"), type=defaults_cls)
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            raise ValueError(
                f"slot {name!r}: catalog inference-defaults metadata failed "
                f"{defaults_cls.__name__} validation: {exc}"
            ) from exc
        defaults = _apply_lora_overrides(name, defaults, fam, lora_metadata_json)
        return _finish_resolved(
            name, ref, defaults,
            inference_regime=inference_regime, allowed_regimes=allowed_regimes,
        )
    if defaults_cls is not None:
        neutral = _apply_lora_overrides(name, defaults_cls(), fam, lora_metadata_json)
        return _finish_resolved(
            name, ref, neutral,
            inference_regime=inference_regime, allowed_regimes=allowed_regimes,
        )
    # No schema declared and no metadata: the ref itself still resolves
    # (handlers that never read .defaults — self-loading runtimes).
    return _finish_resolved(
        name, ref, None,
        inference_regime=inference_regime, allowed_regimes=allowed_regimes,
    )


__all__ = [
    "DEFAULT_REGIMES", "REGIMES", "RegimeMismatchError", "ResolvedSlot",
    "Slot", "resolve_slot",
]
