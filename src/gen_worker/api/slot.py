"""``Slot`` — a hub-resolved model slot (pgw#520 / th#767).

The model SET is catalog, not code: tensorhub owns the mapping from a
``models={}` slot to the checkpoint(s) it may resolve to. The endpoint code
declares only what the HUB needs to enforce that mapping and what the
WORKER needs when no hub is configured — everything else (the curated list,
per-repo pricing, hot hints) moved to platform config.

    from gen_worker import HF, Slot, endpoint
    from gen_worker.families import SdxlDefaults

    @endpoint(models={
        "pipeline": Slot(
            StableDiffusionXLPipeline,
            selected_by="model",
            default_checkpoint=HF("stabilityai/stable-diffusion-xl-base-1.0"),
            default_config=SdxlDefaults(steps=28, guidance=6.0),
        ),
        "vae": Slot(AutoencoderKL, default_checkpoint=HF("madebyollin/sdxl-vae-fp16-fix")),
    })
    class Generate:
        def setup(self, pipeline: StableDiffusionXLPipeline, vae: AutoencoderKL) -> None: ...

        def generate(self, ctx: RequestContext, p: In) -> Out:
            resolved = ctx.slots["pipeline"]   # ResolvedSlot[SdxlDefaults]
            steps = p.steps if p.steps is not None else resolved.defaults.steps

A bare :class:`~gen_worker.api.binding.ModelRef` value in ``models={}``/
``model=`` is sugar for ``Slot(<inferred pipeline class>, default_checkpoint=ref)``
— the ``@endpoint`` decorator performs that inference from the
``setup()``/handler parameter annotation the same way it always resolved a
bare ref's slot NAME.
"""

from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Sequence, TypeVar

import msgspec

from .binding import ModelRef
from ..families.base import KIND_LORA, FamilyDefaults, family_for

D = TypeVar("D", bound=FamilyDefaults)

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
    handler injection constructs (the role a bare ``ModelRef``'s consuming
    annotation played before; here it's explicit because a Slot's actual
    resolved ref is no longer necessarily the ``default``).

    ``selected_by`` names the ``str``-typed payload field that branches this
    slot at request time. Validated at registration (registry.py) against
    the handler's payload type — the field must exist and be typed plain
    ``str`` (the schema enum of legal values is overlaid live by the hub,
    never baked into the SDK).

    ``default_checkpoint`` seeds the hub mapping at first publish and is the
    ONLY resolution source in hub-less mode (``cozy run``, hermetic tests) —
    a live hub mapping always wins when present. ``None`` means this slot
    has no code-side bootstrap ref: it only resolves against a hub mapping.

    ``default_config`` is this slot's code-side :class:`FamilyDefaults`
    preset, used when the resolved repo carries no inference-defaults
    metadata. It LOSES to repo metadata (th#767 precedence: payload > repo
    metadata > this default_config — a recipe of last resort).

    ``share_components`` (pgw#636) names pipeline components (snapshot
    top-level subfolders, e.g. ``("text_encoder", "text_encoder_2")``) that
    the WORKER may factor into independent content-keyed residency entries
    shared ACROSS checkpoint picks of this slot (gw#479 machinery): equal
    bytes load once; per-pick exclusive weights (the denoiser) become
    independently LRU-swappable entries so one card packs several
    checkpoints hot. Sharing stays content-honest — components whose bytes
    differ between picks simply never alias. Purely a worker load policy;
    not part of the hub manifest contract.
    """

    __slots__ = (
        "pipeline_cls", "selected_by", "default_checkpoint", "default_config",
        "share_components",
    )

    def __init__(
        self,
        pipeline_cls: type,
        *,
        selected_by: str = "",
        default_checkpoint: Optional[ModelRef] = None,
        default_config: Optional[D] = None,
        share_components: Sequence[str] = (),
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
        if default_config is not None and not isinstance(default_config, FamilyDefaults):
            raise TypeError(
                f"Slot(default_config=...) must be a FamilyDefaults subclass "
                f"instance, got {type(default_config).__name__}"
            )
        cleaned = tuple(
            s for s in (str(c or "").strip() for c in share_components) if s
        )
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(
                f"Slot(share_components=...) has duplicate names: {cleaned!r}"
            )
        self.pipeline_cls = pipeline_cls
        self.selected_by = str(selected_by or "").strip()
        self.default_checkpoint = default_checkpoint
        self.default_config = default_config
        self.share_components: tuple[str, ...] = cleaned

    @property
    def family(self) -> str:
        """Family name from ``default_config``'s registration, or ``""``
        when this slot has no default_config (the endpoint's
        ``Compile(family=...)`` is the other source the decorator
        reconciles against — see ``gen_worker.api.decorators``)."""
        if self.default_config is None:
            return ""
        return self.default_config.family

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Slot({self.pipeline_cls.__name__}, selected_by={self.selected_by!r}, "
            f"default_checkpoint={self.default_checkpoint!r}, "
            f"default_config={self.default_config!r}, "
            f"share_components={self.share_components!r})"
        )


class ResolvedSlot(Generic[D]):
    """What ``ctx.slots[name]`` hands the handler: the resolved
    :class:`ModelRef` plus ONE typed defaults object — repo metadata merged
    over the endpoint's code fallback (pgw#520 resolution chain).

    Explicit PAYLOAD values still win over ``.defaults`` — that precedence
    is handler logic; this object only carries the merged HUB-vs-CODE
    result.

    ``regime`` (th#1017) is the resolved checkpoint's inference regime —
    ``"standard"`` unless the hub classified the weights otherwise
    (``"v_prediction"`` | ``"distilled"``). Handlers branch on it (e.g. a
    dual-mode turbo lane skips its distillation LoRA for an
    already-distilled checkpoint).
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
    an earlier one's on the same field) — NOT the whole-object precedence
    :func:`resolve_slot` uses for repo-metadata-over-fallback above.

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
    defaults: D,
    *,
    inference_regime: str,
    allowed_regimes: Optional[Sequence[str]],
) -> "ResolvedSlot[D]":
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
    family: str = "",
    raw_metadata_json: str = "",
    lora_metadata_json: Sequence[str] = (),
    inference_regime: str = "standard",
    allowed_regimes: Optional[Sequence[str]] = None,
) -> "ResolvedSlot[D]":
    """Merge repo-metadata inference defaults over ``slot.default_config``,
    then apply per-lora FIELD-LEVEL overrides — the pgw#520/pgw#516
    resolution chain shared by the production executor and the hub-less CLI
    path.

    Precedence: repo metadata (``raw_metadata_json``, when non-empty) wins
    over ``slot.default_config`` entirely (a repo either fully specifies its
    family vocabulary or it doesn't — tensorhub validates the whole object
    at metadata-PUT time, so a partial merge would silently hide invalid
    metadata behind the code default). ``default_config`` LOSES to repo
    metadata — it is a recipe of last resort. Missing metadata AND no
    default_config is a clear error, not a silent empty object.

    ``lora_metadata_json`` (pgw#516, in lora-ride order — riding
    ``ModelBinding.loras[i].inference_defaults`` on the wire) applies LAST,
    field by field, on top of whichever recipe precedence above picked — see
    :func:`_apply_lora_overrides`.

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
    fam = str(family or slot.family or "").strip()
    defaults_cls = type(slot.default_config) if slot.default_config is not None else (
        family_for(fam) if fam else None
    )
    raw = (raw_metadata_json or "").strip()
    if raw:
        if defaults_cls is None:
            raise ValueError(
                f"slot {name!r}: repo metadata present but no family is "
                "resolvable (no Slot(default_config=...) and no "
                "Compile(family=...) on the endpoint) — cannot determine "
                "which vocabulary to decode it against"
            )
        try:
            defaults: Any = msgspec.json.decode(raw.encode("utf-8"), type=defaults_cls)
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            raise ValueError(
                f"slot {name!r}: repo inference-defaults metadata failed "
                f"{defaults_cls.__name__} validation: {exc}"
            ) from exc
        defaults = _apply_lora_overrides(name, defaults, fam, lora_metadata_json)
        return _finish_resolved(
            name, ref, defaults,
            inference_regime=inference_regime, allowed_regimes=allowed_regimes,
        )
    if slot.default_config is not None:
        merged = _apply_lora_overrides(name, slot.default_config, fam, lora_metadata_json)
        return _finish_resolved(
            name, ref, merged,
            inference_regime=inference_regime, allowed_regimes=allowed_regimes,
        )
    raise ValueError(
        f"slot {name!r}: no repo inference-defaults metadata for the "
        "resolved model and no Slot(default_config=...) on the endpoint — "
        "nothing to resolve this slot's defaults from"
    )


__all__ = [
    "DEFAULT_REGIMES", "REGIMES", "RegimeMismatchError", "ResolvedSlot",
    "Slot", "resolve_slot",
]
