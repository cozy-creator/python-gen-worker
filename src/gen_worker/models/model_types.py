"""Model types — the pgw#1377 seam, minimal until that lane lands.

A MODEL TYPE (``SDXL``) classifies checkpoints and carries the typed
serving-defaults vocabulary; a MODEL CLASS (``SdxlModel(Model[SDXL])``) is
the runnable the author writes (pgw#1382). This module holds the seam shape
pgw#1377 builds out: ``ModelType`` (generic over its ``Defaults`` struct, so
``LoadContext[SDXL].defaults()`` types statically), ``Knob`` (the one
clamp/default surface), and the ``SDXL`` launch type. The hub-row decode
matrix, the schema/name export emitter and the ingest assist are pgw#1377's;
nothing here may grow compile-related fields (pgw#1367 — the catalog stays
dead).

Every ``Defaults`` zero-arg construction is SERVABLE values — platform
fallbacks double as publish-time trace fixtures (pgw#1376 rule), never
sentinel garbage.
"""

from __future__ import annotations

from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar

import msgspec

#: The declared schedule vocabulary for step-distilled recipes; schedule ->
#: scheduler-class construction stays endpoint code (the main_v2 pattern).
Schedule = Literal["euler_trailing", "lcm"]

#: The PLATFORM-WIDE sampler vocabulary — types checkpoint METADATA (the
#: hub-row `sampler` field, written platform-wide). An endpoint serves its
#: own subset (its request field is its own Literal); a metadata value an
#: endpoint doesn't serve warns and falls through to the tree's scheduler.
SamplerName = Literal[
    "dpmpp_2m_karras", "dpmpp_2m", "dpmpp_sde", "dpmpp_sde_karras",
    "euler", "euler_trailing", "euler_a", "heun", "unipc", "ddim",
    "ddpm", "lcm", "pndm", "deis",
]

N = TypeVar("N", int, float)
DT_co = TypeVar("DT_co", bound=msgspec.Struct, covariant=True)


class SupportsAdjust(Protocol):
    """The caller-visibility half of ``Knob.resolve`` — any context whose
    ``clamp`` records an adjustment row the caller sees."""

    def clamp(
        self,
        field: str,
        requested: float,
        *,
        lo: float | None = None,
        hi: float | None = None,
        reason: str = "",
    ) -> float: ...


class Knob(msgspec.Struct, Generic[N], frozen=True):
    """A reusable (default, lo, hi) triple — the one clamp/default surface.

    ``resolve(value, ctx)``: ``None`` -> the default; a value -> clamped into
    [lo, hi] with a caller-visible adjustment via ``ctx``. It ASSUMES the
    value already passed the endpoint's own ``msgspec.Meta`` API envelope
    (which REJECTS with a typed 400) — resolve itself never rejects
    (pgw#1377 point 4, Paul's two-layer contract).
    """

    default: N
    lo: N | None = None
    hi: N | None = None
    field: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        # Struct-default instances learn their field name for the
        # caller-visible adjustment row; decode-created Knobs carry the name
        # from the wire (or none — pgw#1377 polishes the decode side).
        if not self.field:
            msgspec.structs.force_setattr(self, "field", name)

    def resolve(self, value: N | None, ctx: SupportsAdjust) -> N:
        if value is None:
            return self.default
        applied = ctx.clamp(
            self.field or "value",
            float(value),
            lo=None if self.lo is None else float(self.lo),
            hi=None if self.hi is None else float(self.hi),
            reason="outside this checkpoint's supported range",
        )
        out: N = type(self.default)(applied)
        return out


class ModelType(Generic[DT_co]):
    """Base for model types. Subclasses set ``name`` (the wire string), nest
    a ``Defaults`` struct (field defaults ARE the platform fallbacks), and —
    once tensorfs#111 ships contract objects — ``canonical_contract`` (the
    lane a model class gets when ``lanes=`` is omitted; an import, never a
    pointer-string)."""

    name: ClassVar[str] = ""
    #: The imported tensorfs Contract OBJECT, or None until tensorfs#111.
    canonical_contract: ClassVar[Any] = None
    #: tensorfs contract-name patterns — the ingest fingerprint (pgw#1377).
    contracts: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:  # pragma: no cover - defensive
        raise TypeError(
            f"{type(self).__name__} is a model TYPE (a classification), "
            "never instantiated; write a model CLASS instead: "
            f"class MyModel(Model[{type(self).__name__}])"
        )


class SDXL(ModelType["SDXL.Defaults"]):
    name = "sdxl"

    class Recipe(msgspec.Struct, frozen=True):
        """The SERVING RECIPE — the typed fields that drive an entrypoint's
        behavior, whichever source they came from (the checkpoint's own row
        or a distillation adapter's overlay): both Defaults types INHERIT
        this, so an entrypoint holds ONE type, never a union (Paul's
        recipe-not-mode-flag ruling). Fields are independent axes: ``cfg``
        (batch-2 guidance vs none), ``schedule`` (``None`` = keep the
        checkpoint's own scheduler; construction stays endpoint code),
        ``steps``/``guidance`` knobs, pinned ``timesteps``."""

        steps: Knob[int] = Knob(30, lo=15, hi=60, field="num_inference_steps")
        guidance: Knob[float] = Knob(6.0, lo=1.5, hi=12.0, field="guidance_scale")
        cfg: bool = True
        schedule: Schedule | None = None
        timesteps: tuple[int, ...] = ()
        #: The checkpoint-metadata sampler preference (platform vocabulary);
        #: None = the checkpoint tree's shipped scheduler stands.
        sampler: SamplerName | None = None

    class Defaults(Recipe, frozen=True):
        """SDXL per-checkpoint serving defaults (pgw#1377 point 2): the
        Recipe axes plus checkpoint-only facts and the prompt-enhancement
        vocabulary. Zero-arg construction is the platform fallback AND the
        trace fixture."""

        #: A fused step-distilled merge; stacking a step-distillation
        #: adapter on one is warned-and-ignored (cfg is a SEPARATE axis:
        #: guidance-distilled full-step checkpoints have cfg off but
        #: step_distilled False and may take a turbo LoRA).
        step_distilled: bool = False
        positive_preamble: str = ""
        negative_preamble: str = ""

    class Lora:
        """Adapter-of-base scoping: ``SDXL.Lora.Defaults`` (wire ``sdxl.lora``).

        The Defaults class is bound below the SDXL body (a nested class body
        cannot see the enclosing class scope to inherit ``Recipe`` directly).
        """

        name = "sdxl.lora"
        Defaults: ClassVar[type["SdxlLoraDefaults"]]


class SdxlLoraDefaults(SDXL.Recipe, frozen=True):
    """``SDXL.Lora.Defaults`` — a :class:`SDXL.Recipe` (the adapter's
    distillation recipe; zero-arg = Lightning-4-step-ish servable
    trace-fixture values) plus adapter-only fields (pgw#1377 point 7)."""

    steps: Knob[int] = Knob(4, lo=1, hi=12, field="num_inference_steps")
    cfg: bool = False
    schedule: Schedule | None = "euler_trailing"
    trigger_words: tuple[str, ...] = ()
    strength: Knob[float] = Knob(1.0, lo=0.0, hi=2.0, field="strength")


SDXL.Lora.Defaults = SdxlLoraDefaults


__all__ = [
    "Knob",
    "ModelType",
    "SDXL",
    "SamplerName",
    "Schedule",
    "SupportsAdjust",
]
