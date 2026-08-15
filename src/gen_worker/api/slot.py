"""``Slot`` — a hub-resolved model slot (SDK v2).

The model SET is catalog, not code: tensorhub owns the mapping from a
``models={}`` slot to the checkpoint(s) it may resolve to. Under SDK v2 the
declaration shrinks to the ROOT of the component tree::

    from gen_worker import HF, RequestContext, Slot, endpoint
    from myendpoint.defaults import SdxlDefaults  # your endpoint's own vocabulary

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
CATALOG DATA, not endpoint code. Explicit multi-slot declaration
survives only as the escape hatch for runtimes the SDK cannot introspect
(llama/gguf, custom engines).

The per-model CONFIG SCHEMA is derived from the handler's context
annotation (``ctx: RequestContext[SdxlDefaults]``), never declared on the
Slot: code owns the schema, the catalog owns the values (the hub stamps one
resolved recipe per slot). ``Slot(default_config=...)`` and
``Slot(share_components=...)`` are DELETED (v2 hard cut): code-side recipe
values are gone, and component sharing is automatic by content address.
"""

from __future__ import annotations

from typing import Any, Dict, Generic, Mapping, Optional, Sequence, Type, TypeVar

import msgspec

from .binding import ModelRef
from ..families.base import KIND_LORA, GenerationDefaults, family_for
from ..models.tensor_layout_contract import (
    LayoutDeclarationError,
    LayoutRequirements,
    normalize_layout_demand,
    normalize_layout_requirements,
    normalize_layout_undeclarable,
)

D = TypeVar("D", bound=GenerationDefaults)

# The objective split: two ORTHOGONAL checkpoint-level facts, both
# hub-classified at ingest; the SDK only consumes.
#
# ``objective`` — what the network predicts (the training objective).
# Drives scheduler math at VIEW construction: prediction_type (+ zero-SNR
# rescale) for v_prediction; a flow-match scheduler class for flow. ""
# means the hub did not stamp one (hub-less runs, unclassified rows) — the
# view then applies nothing and the declaration backstop does not fire.
#
# ``distilled`` — a post-training property, orthogonal to the objective
# (z-image-turbo is distilled FLOW; a distilled v-pred fine-tune keeps its
# v-pred fact). Drives the recipe (few steps, CFG off), the graph choice
# (cfg_off/batch-1 class) and adapter policy (never stack a distillation
# LoRA onto already-distilled weights).
OBJECTIVES = ("epsilon", "v_prediction", "flow")

# SERVING TASKS — what a handler asks the model to DO, as opposed to
# what the model IS. Orthogonal to both facts above: qwen-image-edit and
# qwen-image share one objective (flow) and one architecture, but an edit path
# must PRESERVE an input image while a text-to-image path has nothing to
# preserve. Quantization damage differs accordingly, so the hub's human
# quality sign-off is scoped per task and a handler must say which it serves.
# Mirrors tensorhub's internal/modeltask vocabulary exactly.
TASKS = (
    "text_to_image",
    "image_edit",
    "image_upscale",
    "text_to_video",
    "image_to_video",
    "video_edit",
    "text_to_audio",
    "text_generation",
)


class ObjectiveMismatchError(ValueError):
    """A resolved checkpoint's stamped objective/distilled facts are outside
    the invoked function's declared ``@worker_function(objectives=...,
    distilled=...)`` contract. Hub routing enforces this upstream; reaching
    here means version skew or a hub bug."""


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

    ``family`` is the slot's architecture-family intent. ``None`` keeps the
    compatibility inference: a non-root, defaultless slot is treated as
    family-agnostic, while a root/ref-bearing slot inherits the handler's
    family. A non-empty value explicitly declares this slot's family and wins
    over that inference; ``""`` explicitly declares a family-agnostic
    auxiliary slot. This distinction matters for non-root model lanes that are
    deploy-bound rather than hardcoding a checkpoint ref.

    ``default_checkpoint`` seeds the hub mapping at first publish and is the
    ONLY resolution source in hub-less mode (``cozy run``, hermetic tests) —
    a live hub mapping always wins when present. ``None`` means this slot
    has no code-side bootstrap ref: it only resolves against a hub mapping.

    ``root=True`` marks THE root slot of a multi-slot
    class: the slot ``ctx.defaults`` / ``ctx.for_request`` resolve when no
    explicit ``slot=`` is passed. A single-slot class needs no marker, and
    a slot literally named ``"pipeline"`` is the implicit root; any OTHER
    multi-slot shape must mark exactly one root — ambiguity is a
    decoration-time error, never a silent fallback.

    ``layouts`` is the slot's per-component DEMAND (§1.33): a mapping from component path to the SET of tensor-layout contract handles
    this slot's code can execute. ``"*"`` is the whole-tree default; a
    component key overrides it for that component only::

        Slot(StableDiffusionXLPipeline, selected_by="model", layouts={
            "*":            (CONTRACT_PLAIN_BF16,),
            "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16),
        })

    **The set is a compatibility FILTER; its order carries NO preference**
    (§1.33 point 2). Preference has exactly one
    authority — the author-configured ordered ladder of (GPU, lane) pairs —
    so the handles are stored and published in CANONICAL order, not as
    written. Two authors who spell the same set differently state the same
    demand, and no downstream reader can mistake a position for a ranking.

    The handles are validated here against the SDK's
    transcribed ``KNOWN_CONTRACTS``; the KEYS are validated at decoration
    against the slot's DERIVED component tree, so a key that would silently
    match nothing is a build error naming the tree. An empty mapping or an
    empty tuple is a decoration-time error, because collapsing the tri-state
    is a fail-open defect.

    **A19 (Paul, 2026-08-15): the declaration is MANDATORY.** Absence used to
    be the UNDECLARED tri-state; measured fleet-wide it was only ever the
    default, so "no opinion" and "nobody wrote the line" were one state.
    Discovery now REFUSES a model slot carrying neither ``layouts=`` nor
    ``layouts_undeclarable=`` — see :func:`gen_worker.models.
    tensor_layout_contract.undeclared_slot_refusal`. The refusal is at
    discovery/publish, not here: a bare ``Slot(...)`` remains constructible
    in-process, and it is the IMAGE that may not ship undeclared.

    ``layouts_undeclarable`` is the explicit third rung, and it is a
    declaration rather than a default: a REASON string, refused when blank,
    mutually exclusive with ``layouts=``. It exists for slots whose bytes no
    registered handle names — a tokenizer tree with no tensors, a GGUF quant
    axis (th#1809 T3), a compressed-tensors checkpoint with no descriptor.
    Inventing a handle for those is the failure mode, not the fix.

    This lives on ``Slot`` and NOT on ``Compile``: ``Compile``'s fields feed
    ``contract_axes()``, a cell-key input, and §1.33 point 5 is that
    conversion is upstream of compute and invisible to cell identity. A
    layout declaration there would either re-key every cell in the fleet or
    sit inside the key struct while deliberately not participating.

    ``layout_requirements`` is the REQUIREMENTS axis (Paul, 2026-08-15): what
    EXECUTING a declared contract needs of the machine, keyed by the handle it
    guards. Dual form — the compact ``"sm100+"`` or the structured
    ``LayoutRequirements(min_sm=100)``::

        Slot(QwenImagePipeline, selected_by="model",
             layouts={"*": (CONTRACT_PLAIN_BF16, CONTRACT_COZY_SVDQ_NVFP4_LR8)},
             layout_requirements={CONTRACT_COZY_SVDQ_NVFP4_LR8: "sm100+"})

    A handle names bytes at rest; the floor its kernels need is an EXECUTION
    fact, which is why it lives here and not on the artifact contract — the
    same placement, and the same name, as tensorhub's
    ``contractspec.DecodeEntry.MinSM``, which is the field it feeds. It is per
    (slot, handle) because one contract has different floors in different
    code: ``cozy.fp8-rowwise@1`` is sm89 per-tensor and sm90 rowwise.

    NO DEFAULTS, as everywhere else on this surface: a declared requirement is
    checked, an undeclared one is not evaluated. A key naming a handle
    ``layouts=`` does not accept is a refusal — a requirement guarding nothing
    is never checked. Kernel and torch-version requirements are named in the
    ruling and deliberately NOT built; the compact grammar is a comma-separated
    term list so they can be added without re-spelling a declaration, and an
    unknown term is refused rather than ignored.

    ``optional`` is DERIVED, never passed: a slot is optional exactly when
    its ``setup()`` parameter carries a default (``edit: Pipe | None =
    None``). The signature is the single source of truth, so the two can
    never disagree. An optional slot may be left unbound at deploy time —
    the deploy chooses which lanes a release serves — and
    ``setup()`` then runs with the parameter's own default.
    """

    __slots__ = (
        "pipeline_cls", "selected_by", "family", "default_checkpoint", "root",
        "optional", "layouts", "layouts_undeclarable", "layout_requirements",
    )

    def __init__(
        self,
        pipeline_cls: type,
        *,
        selected_by: str = "",
        family: Optional[str] = None,
        default_checkpoint: Optional[ModelRef] = None,
        root: bool = False,
        layouts: Optional[Mapping[str, Sequence[str]]] = None,
        layouts_undeclarable: str = "",
        layout_requirements: Optional[Mapping[str, Any]] = None,
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
        self.family = None if family is None else str(family or "").strip()
        self.default_checkpoint = default_checkpoint
        self.root = bool(root)
        self.optional = False  # derived at decoration from setup()'s default
        where = f"Slot({pipeline_cls.__name__})"
        if layouts is not None and str(layouts_undeclarable or "").strip():
            raise LayoutDeclarationError(
                f"{where}: layouts= and layouts_undeclarable= are mutually "
                "exclusive — a slot either names the contracts it consumes or "
                "says why none is nameable, never both."
            )
        # None is UNDECLARED at construction and stays None; discovery is what
        # refuses it (A19). Anything else normalizes now, at the declaration
        # site, so the traceback names the Slot the author wrote rather than a
        # manifest key.
        self.layouts: Optional[Dict[str, tuple[str, ...]]] = (
            None if layouts is None
            else normalize_layout_demand(layouts, where=where)
        )
        # `""` is the parameter's own default — the author passed nothing, and
        # discovery is what refuses that. Anything else was WRITTEN, so a
        # whitespace-only reason is a refusal here rather than a silent escape.
        self.layouts_undeclarable: str = (
            "" if layouts_undeclarable == ""
            else normalize_layout_undeclarable(
                layouts_undeclarable, where=where)
        )
        # The REQUIREMENTS axis, keyed by the handle it guards. Declared
        # against the accepted SET, so a requirement can never guard a
        # contract this slot does not consume.
        if layout_requirements is None:
            self.layout_requirements: Dict[str, LayoutRequirements] = {}
        elif self.layouts is None:
            raise LayoutDeclarationError(
                f"{where}: layout_requirements= without layouts=. A "
                "requirement guards a declared contract; there is none here "
                "to guard."
            )
        else:
            self.layout_requirements = normalize_layout_requirements(
                layout_requirements, where=where,
                accepted={h for hs in self.layouts.values() for h in hs},
            )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Slot({self.pipeline_cls.__name__}, selected_by={self.selected_by!r}, "
            f"family={self.family!r}, default_checkpoint={self.default_checkpoint!r}, "
            f"root={self.root!r}, optional={self.optional!r}, "
            f"layouts={self.layouts!r}, "
            f"layouts_undeclarable={self.layouts_undeclarable!r}, "
            f"layout_requirements={self.layout_requirements!r})"
        )


class ResolvedSlot(Generic[D]):
    """What ``ctx.slots[name]`` hands the handler: the resolved
    :class:`ModelRef` plus ONE typed defaults object — the catalog-resolved
    recipe decoded against the handler's declared config schema
    (``ctx: RequestContext[D]``).

    Explicit PAYLOAD values still win over ``.defaults`` — that precedence
    is handler logic; this object only carries the resolved catalog recipe.

    ``objective``/``distilled`` are the resolved checkpoint's
    hub-stamped values. ``distilled_status`` distinguishes an evidenced false
    value from an unclassified or inconclusive one. An empty status means an
    older hub omitted the additive field. ``ctx.for_request`` applies the
    objective to the per-request scheduler view automatically; handlers may
    also branch on the distillation value and its evidence status.
    """

    __slots__ = (
        "ref", "defaults", "objective", "distilled", "distilled_status",
    )

    def __init__(
        self, ref: ModelRef, defaults: D, objective: str = "",
        distilled: bool = False, distilled_status: str = "",
    ) -> None:
        self.ref = ref
        self.defaults = defaults
        self.objective = objective
        self.distilled = distilled
        self.distilled_status = distilled_status

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ResolvedSlot(ref={self.ref!r}, defaults={self.defaults!r}, "
            f"objective={self.objective!r}, distilled={self.distilled!r}, "
            f"distilled_status={self.distilled_status!r})"
        )


def _apply_lora_overrides(
    name: str, base: D, fam: str, lora_metadata_json: Sequence[str],
) -> D:
    """Composition rule: apply each lora's non-``None`` fields onto
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
    # NOT `D`, and not `Optional[D]` either: `ResolvedSlot.__init__` declares
    # `defaults: D` but `resolve_slot` passes None on the no-schema path below,
    # so this helper cannot honestly promise `ResolvedSlot[D]`. The PUBLIC
    # `resolve_slot` below still is `-> ResolvedSlot[D]`.
    defaults: Any,
    *,
    objective: str,
    distilled: bool,
    distilled_status: str,
    allowed_objectives: Optional[Sequence[str]],
    allowed_distilled: Optional[bool],
) -> "ResolvedSlot[Any]":
    """Build the ``ResolvedSlot`` and, when the caller knows the invoked
    function's declared objective/distilled contract, enforce the backstop:
    the hub gates checkpoint<->function compatibility at deploy and request
    time upstream — reaching a mismatch here means version skew or a hub
    bug, never a normal-path outcome. The backstop only fires on STAMPED
    facts. Explicitly unknown distillation evidence fails closed when the
    function declares that axis. Empty status preserves the old-sender
    behavior: objective-stamped bindings treat the bool as known, while fully
    unstamped bindings pass as they did before the additive field existed."""
    status = str(distilled_status or "").strip()
    if status not in ("", "classified", "unclassified", "inconclusive"):
        raise ObjectiveMismatchError(
            f"slot {name!r}: resolved checkpoint carries unknown "
            f"distilled_status={status!r}"
        )
    resolved = ResolvedSlot(
        ref=ref, defaults=defaults, objective=objective, distilled=distilled,
        distilled_status=status,
    )
    if resolved.objective:
        if allowed_objectives is not None and resolved.objective not in allowed_objectives:
            raise ObjectiveMismatchError(
                f"slot {name!r}: resolved checkpoint objective "
                f"{resolved.objective!r} is not in the invoked function's "
                f"declared objectives {tuple(allowed_objectives)!r}"
            )
    if allowed_distilled is not None:
        if status in ("unclassified", "inconclusive"):
            raise ObjectiveMismatchError(
                f"slot {name!r}: resolved checkpoint distillation evidence is "
                f"{status}; the invoked function declares "
                f"distilled={allowed_distilled!r}"
            )
        # Empty is an old sender. Preserve its pre-field behavior: only an
        # objective-stamped binding treated the adjacent bool as evidenced.
        distilled_known = (
            status == "classified" or (not status and bool(resolved.objective)))
        if distilled_known and resolved.distilled != allowed_distilled:
            raise ObjectiveMismatchError(
                f"slot {name!r}: resolved checkpoint distilled="
                f"{resolved.distilled!r} but the invoked function declares "
                f"distilled={allowed_distilled!r}"
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
    objective: str = "",
    distilled: bool = False,
    distilled_status: str = "",
    allowed_objectives: Optional[Sequence[str]] = None,
    allowed_distilled: Optional[bool] = None,
) -> "ResolvedSlot[D]":
    """SDK v2 resolution chain: decode the catalog-resolved recipe
    (``raw_metadata_json``) against the handler's DERIVED config schema
    (``defaults_cls``, from ``ctx: RequestContext[D]``), then apply per-lora
    FIELD-LEVEL overrides — shared by the production executor and the
    hub-less CLI path.

    Recipe VALUES live in the catalog: the hub stamps ONE resolved recipe per
    slot, so in production the metadata branch always runs. With
    no metadata (hub-less ``cozy run``, hermetic tests, a family the hub has
    not stamped) the NEUTRAL SCHEMA DEFAULTS (``defaults_cls()``) apply —
    exactly the hub's neutral stamp, so both paths agree. There is no
    code-side recipe fallback (v2 deleted ``Slot(default_config=...)``):
    code owns the schema only.

    ``lora_metadata_json`` (in lora-ride order) applies LAST, field
    by field — see :func:`_apply_lora_overrides`.

    ``objective``/``distilled`` are the resolved checkpoint's
    hub-stamped values; ``distilled_status`` distinguishes an evidenced false
    from unknown evidence (empty means an older sender).
    ``allowed_objectives``/``allowed_distilled``, when given, are the
    invoked function's declared ``@worker_function`` contract — see
    :func:`_finish_resolved`.
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
            objective=objective, distilled=distilled,
            distilled_status=distilled_status,
            allowed_objectives=allowed_objectives,
            allowed_distilled=allowed_distilled,
        )
    if defaults_cls is not None:
        neutral = _apply_lora_overrides(name, defaults_cls(), fam, lora_metadata_json)
        return _finish_resolved(
            name, ref, neutral,
            objective=objective, distilled=distilled,
            distilled_status=distilled_status,
            allowed_objectives=allowed_objectives,
            allowed_distilled=allowed_distilled,
        )
    # No schema declared and no metadata: the ref itself still resolves
    # (handlers that never read .defaults — self-loading runtimes).
    return _finish_resolved(
        name, ref, None,
        objective=objective, distilled=distilled,
        distilled_status=distilled_status,
        allowed_objectives=allowed_objectives,
        allowed_distilled=allowed_distilled,
    )


__all__ = [
    "OBJECTIVES", "ObjectiveMismatchError", "ResolvedSlot",
    "Slot", "resolve_slot",
]
