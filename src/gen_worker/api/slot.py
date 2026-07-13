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

from typing import Any, Dict, Generic, Mapping, Optional, TypeVar

import msgspec

from .binding import ModelRef
from ..families.base import FamilyDefaults, family_for

D = TypeVar("D", bound=FamilyDefaults)


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
    """

    __slots__ = ("pipeline_cls", "selected_by", "default_checkpoint", "default_config")

    def __init__(
        self,
        pipeline_cls: type,
        *,
        selected_by: str = "",
        default_checkpoint: Optional[ModelRef] = None,
        default_config: Optional[D] = None,
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
        self.pipeline_cls = pipeline_cls
        self.selected_by = str(selected_by or "").strip()
        self.default_checkpoint = default_checkpoint
        self.default_config = default_config

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
            f"default_config={self.default_config!r})"
        )


class ResolvedSlot(Generic[D]):
    """What ``ctx.slots[name]`` hands the handler: the resolved
    :class:`ModelRef` plus ONE typed defaults object — repo metadata merged
    over the endpoint's code fallback (pgw#520 resolution chain).

    Explicit PAYLOAD values still win over ``.defaults`` — that precedence
    is handler logic; this object only carries the merged HUB-vs-CODE
    result.
    """

    __slots__ = ("ref", "defaults")

    def __init__(self, ref: ModelRef, defaults: D) -> None:
        self.ref = ref
        self.defaults = defaults

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ResolvedSlot(ref={self.ref!r}, defaults={self.defaults!r})"


def resolve_slot(
    name: str,
    slot: "Slot[D]",
    *,
    ref: Optional[ModelRef],
    family: str = "",
    raw_metadata_json: str = "",
) -> "ResolvedSlot[D]":
    """Merge repo-metadata inference defaults over ``slot.default_config``
    — the pgw#520 resolution chain shared by the production executor and
    the hub-less CLI path.

    Precedence: repo metadata (``raw_metadata_json``, when non-empty) wins
    over ``slot.default_config`` entirely (a repo either fully specifies its
    family vocabulary or it doesn't — tensorhub validates the whole object
    at metadata-PUT time, so a partial merge would silently hide invalid
    metadata behind the code default). ``default_config`` LOSES to repo
    metadata — it is a recipe of last resort. Missing metadata AND no
    default_config is a clear error, not a silent empty object.
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
        return ResolvedSlot(ref=ref, defaults=defaults)
    if slot.default_config is not None:
        return ResolvedSlot(ref=ref, defaults=slot.default_config)
    raise ValueError(
        f"slot {name!r}: no repo inference-defaults metadata for the "
        "resolved model and no Slot(default_config=...) on the endpoint — "
        "nothing to resolve this slot's defaults from"
    )


def resolve_slots(
    slots: Mapping[str, "Slot[Any]"],
    *,
    refs: Mapping[str, Optional[ModelRef]],
    families: Mapping[str, str] = {},
    raw_metadata: Mapping[str, str] = {},
) -> Dict[str, "ResolvedSlot[Any]" | Exception]:
    """Resolve every declared slot, collecting per-slot failures instead of
    raising — callers (RequestContext) surface each failure lazily, only
    when the handler actually reads that slot ("clear error at request
    time", not a blanket dispatch-time failure for slots the handler never
    touches)."""
    out: Dict[str, "ResolvedSlot[Any]" | Exception] = {}
    for name, slot in slots.items():
        try:
            out[name] = resolve_slot(
                name, slot,
                ref=refs.get(name),
                family=families.get(name, ""),
                raw_metadata_json=raw_metadata.get(name, ""),
            )
        except ValueError as exc:
            out[name] = exc
    return out


__all__ = ["ResolvedSlot", "Slot", "resolve_slot", "resolve_slots"]
