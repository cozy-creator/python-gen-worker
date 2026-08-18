"""Model-typed serving defaults — the pgw#1376 vocabulary (pgw#1377).

A ModelType is **name + ``Defaults`` struct + ingest fingerprint. NOTHING
ELSE** — no compile knowledge, no runners, no layouts, no per-family serve
logic. ``Defaults`` field defaults ARE the platform fallback values:
``SDXL.Defaults()`` zero-arg is the platform opinion, and every zero-arg
construction must be SERVABLE values (they double as trace fixtures).

The types are GENERIC carriers: ``class SDXL(ModelType[SdxlDefaults])`` binds
the model type to its ``Defaults`` struct so a context generic over the model
type (``Model[SDXL]`` / ``LoadContext[SDXL]``, pgw#1382) can project the
return of ``ctx.defaults()`` statically via a
``def defaults(self: "LoadContext[ModelType[D]]") -> D`` self-type — mypy
resolves ``LoadContext[SDXL].defaults() -> SDXL.Defaults`` with no cast on
the caller's side.

Import-light on purpose: this module may cost only msgspec + stdlib, so the
lazy ``gen_worker.models`` package index can serve it without pulling any
weight-loading machinery.

Platform values are sourced, never invented — each family docstring cites
where its numbers were taken from (the live per-family schema registry this
system replaces, ``tensorhub internal/modelfamily/inferencedefaults/families/
*.schema.json``, dying in th#2140).
"""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import (
    ClassVar,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
)

import msgspec

Number = TypeVar("Number", int, float)


class SupportsClamp(Protocol):
    """The caller-visibility seam ``Knob.resolve`` clamps through.

    ``gen_worker.request_context.RequestContext.clamp`` satisfies it: a clamp
    that changes the value records a caller-visible adjustment naming the
    field and reason (rides ``JobResult.adjustments``).
    """

    def clamp(
        self,
        field: str,
        requested: float,
        *,
        lo: float | None = None,
        hi: float | None = None,
        reason: str = "",
    ) -> float: ...


class Knob(msgspec.Struct, Generic[Number], frozen=True):
    """The reusable min/max/default triple (pgw#1376 point 5).

    ``resolve(value, ctx)`` is the ONE clamp/default surface. It ASSUMES the
    value already passed the endpoint's API ``msgspec.Meta`` bounds (the hard
    envelope, which REJECTS with a typed 400 upstream), so it NEVER rejects:
    None resolves to the checkpoint default, anything else clamps into
    [lo, hi] caller-visibly via ``ctx.clamp``. The Knob a decoded ``Defaults``
    carries already holds the NARROWEST range across the platform and
    checkpoint layers (``defaults_decode`` merges them), so ``self.lo/hi``
    IS the effective range. Typed per instantiation:
    ``Knob[int].resolve(...) -> int``, ``Knob[float].resolve(...) -> float``.
    """

    default: Number
    lo: Number | None = None
    hi: Number | None = None
    # The field this knob lives under, so a clamp can name it. Stamped by the
    # struct definitions and re-stamped by the row decode; never wire input.
    name: str = ""

    def resolve(self, value: Number | None, ctx: SupportsClamp) -> Number:
        if value is None:
            return self.default
        applied = ctx.clamp(
            self.name or "value",
            value,
            lo=None if self.lo is None else float(self.lo),
            hi=None if self.hi is None else float(self.hi),
            reason="outside this checkpoint's supported range",
        )
        # ctx.clamp speaks float; an int knob's bounds are ints, so the
        # clamped value is integral by construction and the constructor of
        # the default's own type restores the instantiation's return type.
        return type(self.default)(applied)


# ── tensorfs contract seam ───────────────────────────────────────────────────
#
# tensorfs#111 seam: a canonical contract is an IMPORTED OBJECT, never a
# pointer-string (Paul's contract-objects ruling 2026-08-18). Until the
# vendored tensorfs ships ``contracts`` (tensorfs#111/#113), these are
# placeholder objects carrying the eventual identity shape (``name`` +
# ``version``; stamp ``<name>@<version>``, the identity
# ``ContractRegistry.stamps()`` speaks). The swap is one edit: replace the
# ``PendingContract`` assignments below with
# ``from gen_worker._vendor.tensorfs.contracts import ...`` and delete
# ``PendingContract``.


class TensorLayoutContract(Protocol):
    """What ``canonical_contract`` is typed against until tensorfs#111."""

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> int: ...


class PendingContract(msgspec.Struct, frozen=True):
    """Placeholder tensorfs Contract object (tensorfs#111 seam, see above)."""

    name: str
    version: int

    @property
    def stamp(self) -> str:
        return f"{self.name}@{self.version}"


#: stabilityai/stable-diffusion-xl-base-1.0 scheduler_config.json — SDXL's
#: training-time noise schedule (scaled_linear 0.00085/0.012, epsilon).
SDXL_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "EulerDiscreteScheduler",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "interpolation_type": "linear",
    "trained_betas": None,
    "use_karras_sigmas": False,
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "clip_sample": False,
}

#: runwayml/stable-diffusion-v1-5 scheduler_config.json — SD1.x's
#: training-time noise schedule (same betas, PNDM-class shipping config).
SD15_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "PNDMScheduler",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "steps_offset": 1,
    "trained_betas": None,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "clip_sample": False,
}

SDXL_DIFFUSERS_BF16: Final = PendingContract("sdxl.diffusers-bf16", 1)
SD15_DIFFUSERS_BF16: Final = PendingContract("sd15.diffusers-bf16", 1)
SD2_DIFFUSERS_BF16: Final = PendingContract("sd2.diffusers-bf16", 1)
HIDREAM_O1_DIFFUSERS_BF16: Final = PendingContract("hidream-o1.diffusers-bf16", 1)
WAN22_DIFFUSERS_BF16: Final = PendingContract("wan22.diffusers-bf16", 1)


# ── bases ────────────────────────────────────────────────────────────────────

D_co = TypeVar("D_co", bound=msgspec.Struct, covariant=True)


class ModelType(Generic[D_co]):
    """A serving-defaults vocabulary: name + ``Defaults`` + ingest fingerprint.

    Generic over its ``Defaults`` struct (see module docstring). Subclasses
    assign ``Defaults`` as a class attribute so the contract-file spelling
    ``SDXL.Defaults`` holds. Never instantiated.

    ``contracts`` is the ingest fingerprint — ``fnmatch`` patterns over
    recorded tensorfs contract stamps (``<name>@<version>``). The actual
    checkpoint sniffing rides tensorfs's matcher at ingest
    (``ContractRegistry.detect_file``); this side only maps a recorded stamp
    to a model name (``model_type_for_contract``). Sniffed structure is
    authoritative; declared metadata (sai-model-spec
    ``modelspec.architecture``, Civitai ``baseModel``) is a seed/hint, never
    trusted alone.
    """

    name: ClassVar[str] = ""
    contracts: ClassVar[tuple[str, ...]] = ()
    canonical_contract: ClassVar[TensorLayoutContract]
    #: The architecture's TRAINING-TIME scheduler config (the standard
    #: scheduler_config.json content) — an architecture fact beside the
    #: fingerprint (Paul's ruling, 2026-08-18: the endpoint layer-3 scheduler
    #: backstop is DELETED; a bare scheduler class carries library-default
    #: betas, not the family's training schedule). CONSUMER: INGEST — a
    #: classified tree missing scheduler_config.json gets this synthesized
    #: into its config snapshot (single-file imports), so the pipeline
    #: constructor's scheduler invariant always holds and no endpoint ever
    #: guesses a noise schedule (executes hub-side with classification
    #: pre-fill, th#2140). Empty = no canonical recorded; ingest synthesizes
    #: nothing.
    canonical_scheduler_config: ClassVar[Mapping[str, object]] = {}

    def __init__(self) -> None:
        raise TypeError(f"{type(self).__name__} is a vocabulary, not a value")


class LoraOverlay:
    """Base of the nested adapter-of-base overlay types (``SDXL.Lora``).

    Never standalone: the base linkage is structural (nesting), and the wire
    name is ``<base>.lora``. Never instantiated — it's a vocabulary.
    """

    name: ClassVar[str] = ""
    Defaults: ClassVar[type[msgspec.Struct]]

    def __init__(self) -> None:
        raise TypeError(f"{type(self).__name__} is a vocabulary, not a value")


SchedulerName = Literal[
    "dpmpp_2m_karras",
    "dpmpp_2m",
    "euler",
    "euler_trailing",
    "euler_a",
    "unipc",
    "ddim",
    "lcm",
]
"""The PLATFORM-WIDE scheduler-name vocabulary (Paul's scheduler rulings,
pgw#1376 point 6; naming chain schedule→sampler→scheduler): ADAPTER metadata
rows write these names — checkpoints carry NO scheduler metadata at all (the
tree IS their choice; ingest's canonical synthesis covers trees shipping
none). The additive-only evolution rule applies (a new community scheduler
is a new member, never a changed one). Endpoints own their SUPPORTED SUBSET
and the name→scheduler-constructor tables (author code, like lanes — never
here); a demanded name an endpoint doesn't serve warns and falls through to
the tree's shipped scheduler."""


# ── the launch defaults structs (field defaults = PLATFORM VALUES) ───────────


class SdxlConfig(msgspec.Struct, frozen=True):
    """The SERVING CONFIG — the axes both SDXL defaults types share, as ONE
    nominal type ("Defaults are a config plus extras": the endpoint annotates
    ``config: SDXL.Config``, never a union). The axes are INDEPENDENT:
    ``cfg`` (CFG on/off) and few-step are separate facts — guidance-distilled
    models are cfg-off at FULL steps, Hyper-SD CFG-preserving LoRAs are
    few-step with CFG ON at 5-8. ``scheduler`` is deliberately NOT here
    (Paul's tree-only ruling): CHECKPOINTS CARRY NO SCHEDULER METADATA — the
    tree IS the checkpoint's choice (ingest's canonical-synthesis guarantee
    covers trees shipping none); only the adapter overlay declares a
    scheduler demand. ``timesteps`` empty = derive from steps (a pinned
    ladder like DMD2's 999/749/499/249 goes here — a fused merge keeps its
    ladder HERE while its scheduler ships in its tree).

    Platform values: steps 28 / guidance 6.0 from the live sdxl.schema.json
    registry entry; [lo, hi] soft ranges mirror the contract file's endpoint
    envelope (``main_v2.py``: steps 1..80, guidance 1.5..15.0)."""

    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(6.0, lo=1.5, hi=15.0, name="guidance")
    cfg: bool = True
    timesteps: tuple[int, ...] = ()


class SdxlDefaults(SdxlConfig, frozen=True):
    """The checkpoint row: the config plus the prompt vocabulary.
    ``positive_preamble`` carries the quality vocabulary (Paul's ruling,
    pgw#1376 point 2 — the old endpoint ``_QUALITY_MARKERS``
    "masterpiece, best quality" vocabulary lives HERE, banned from endpoint
    code); ``negative_preamble`` is its symmetric counterpart. A fused
    step-distilled merge (DMD2/Lightning full checkpoint) carries its config
    in the inherited fields (cfg=False, pinned timesteps) — its
    scheduler ships in its TREE (a proper DMD2 export carries LCMScheduler)."""

    positive_preamble: str = "masterpiece, best quality"
    negative_preamble: str = "worst quality, low quality"
    #: CHECKPOINT-level fact, deliberately NOT on Config: is this checkpoint
    #: itself a step-distillation product (fused DMD2/Lightning/LCM merge)?
    #: Decoupled from ``cfg`` (purely the guidance axis): a guidance-distilled
    #: full-step checkpoint is cfg=False + step_distilled=False and MAY take a
    #: turbo LoRA; step_distilled=True is what makes stacking a distillation
    #: adapter harmful — the endpoint warns and ignores the adapter (never an
    #: error). Ingest may infer True from the merge classification.
    step_distilled: bool = False


class SdxlLoraDefaults(SdxlConfig, frozen=True):
    """The adapter row: the config (platform overrides = Lightning-4-step-ish
    servable trace fixture: cfg off, euler_trailing, 4 steps) plus the
    adapter's own facts. ``guidance`` inherits the base platform knob — inert
    while cfg=False, sane if a CFG-preserving adapter row (Hyper-SD) flips
    cfg on and narrows the range to its recommended 5-8. ``strength`` range
    from the one shipped precedent, Civitai min/maxStrength
    (sdxl.lora.schema.json recommended_weight −4..4); steps bound from the
    same file (1..150)."""

    steps: Knob[int] = Knob(4, lo=1, hi=150, name="steps")
    cfg: bool = False
    #: The riding distillation's scheduler DEMAND — the base tree cannot know
    #: a LoRA needs trailing-Euler/LCM. None = no demand, the tree stands.
    scheduler: SchedulerName | None = "euler_trailing"
    strength: Knob[float] = Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    trigger_words: tuple[str, ...] = ()
    #: Classification marker: is this adapter a step/guidance DISTILLATION
    #: (Lightning/DMD2/LCM/Hyper-SD rows set True; style/character LoRAs stay
    #: False)? The hub refuses a distillation-slot pick whose row lacks it
    #: (th#2140 envelope validation); ingest pre-fills from known distill
    #: fingerprints; curators may set it.
    distillation: bool = False


class Sd15Config(msgspec.Struct, frozen=True):
    """SD1.x serving config — same shared axes and semantics as
    :class:`SdxlConfig` (scheduler is adapter-only, tree-only for
    checkpoints). Platform values from the live sd15.schema.json registry
    entry: steps 30 (bounds 1..80, the family payload envelope), guidance
    7.0 (schema minimum 0, no declared max)."""

    steps: Knob[int] = Knob(30, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(7.0, lo=0.0, name="guidance")
    cfg: bool = True
    timesteps: tuple[int, ...] = ()


class Sd15Defaults(Sd15Config, frozen=True):
    """The checkpoint row — the config alone (no ruled SD1.x prompt
    vocabulary; fields stay additive if one is ruled)."""


class Sd15LoraDefaults(Sd15Config, frozen=True):
    """The adapter row. Zero-arg = LCM-LoRA-SD1.5-ish 4-step platform values
    (sd15.lora.schema.json records the 4-step distilled recipe shape;
    Hyper-SD15's ``ddim_trailing`` is outside the launch SchedulerName vocabulary —
    flagged in the pgw#1377 tracker section, not invented here). Bounds from
    the same file (recommended_weight −4..4, num_inference_steps 1..80)."""

    steps: Knob[int] = Knob(4, lo=1, hi=80, name="steps")
    cfg: bool = False
    scheduler: SchedulerName | None = "lcm"
    strength: Knob[float] = Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    trigger_words: tuple[str, ...] = ()
    distillation: bool = False


class Sd2Defaults(msgspec.Struct, frozen=True):
    """Values from the live sd2.schema.json registry entry: steps 1 (bounds
    1..4 — SD-Turbo is a 1-4 step ADD distillation), guidance 0.0 (the
    distilled lane pins CFG off)."""

    steps: Knob[int] = Knob(1, lo=1, hi=4, name="steps")
    guidance: Knob[float] = Knob(0.0, lo=0.0, name="guidance")


class HiDreamO1Defaults(msgspec.Struct, frozen=True):
    """Values from the live hidream-o1.schema.json registry entry: steps 28
    (bounds 1..150), guidance 1.0 (its ``cfg_scale`` default — the default
    ``dev`` variant is guidance-distilled; schema minimum 0)."""

    steps: Knob[int] = Knob(28, lo=1, hi=150, name="steps")
    guidance: Knob[float] = Knob(1.0, lo=0.0, name="guidance")


class Wan22Defaults(msgspec.Struct, frozen=True):
    """Values from the live wan22.schema.json registry entry: steps 40
    (bounds 1..80), guidance 4.0 and ``guidance_2`` 3.0 (the A14B expert
    pair's high/low-noise guidance; schema minimum 0)."""

    steps: Knob[int] = Knob(40, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=0.0, name="guidance")
    guidance_2: Knob[float] = Knob(3.0, lo=0.0, name="guidance_2")


# ── the model types (launch set, pgw#1376 point 1) ───────────────────────────


class SDXL(ModelType[SdxlDefaults]):
    """Stable Diffusion XL."""

    name = "sdxl"
    contracts = ("sdxl.*",)
    canonical_contract = SDXL_DIFFUSERS_BF16
    canonical_scheduler_config = SDXL_SCHEDULER_CONFIG
    # TypeAlias so `config: SDXL.Config` is a valid annotation (main_v2.py).
    Config: TypeAlias = SdxlConfig
    Defaults = SdxlDefaults

    class Lora(LoraOverlay):
        name = "sdxl.lora"
        Defaults = SdxlLoraDefaults


class SD15(ModelType[Sd15Defaults]):
    """Stable Diffusion 1.x — one type covers every SD1.x fine-tune."""

    name = "sd15"
    contracts = ("sd15.*",)
    canonical_contract = SD15_DIFFUSERS_BF16
    canonical_scheduler_config = SD15_SCHEDULER_CONFIG
    Config: TypeAlias = Sd15Config
    Defaults = Sd15Defaults

    class Lora(LoraOverlay):
        name = "sd15.lora"
        Defaults = Sd15LoraDefaults


class SD2(ModelType[Sd2Defaults]):
    """Stable Diffusion 2.x / SD-Turbo — its own root, so a 1-step Turbo
    recipe can never validate against the SD1.x vocabulary."""

    name = "sd2"
    contracts = ("sd2.*",)
    canonical_contract = SD2_DIFFUSERS_BF16
    Defaults = Sd2Defaults


class HiDreamO1(ModelType[HiDreamO1Defaults]):
    """HiDream-O1."""

    name = "hidream-o1"
    contracts = ("hidream-o1.*",)
    canonical_contract = HIDREAM_O1_DIFFUSERS_BF16
    Defaults = HiDreamO1Defaults


class Wan22(ModelType[Wan22Defaults]):
    """Wan 2.2 — T2V/I2V A14B + TI2V-5B serve under one vocabulary root, as
    today's registry does."""

    name = "wan22"
    contracts = ("wan22.*",)
    canonical_contract = WAN22_DIFFUSERS_BF16
    Defaults = Wan22Defaults


MODEL_TYPES: Final[tuple[type[ModelType[msgspec.Struct]], ...]] = (
    SDXL,
    SD15,
    SD2,
    HiDreamO1,
    Wan22,
)

LORA_OVERLAYS: Final[tuple[type[LoraOverlay], ...]] = (SDXL.Lora, SD15.Lora)


def model_type_by_name(name: str) -> type[ModelType[msgspec.Struct]] | None:
    """The recognized-name lookup (hub ``model`` column values)."""
    for mt in MODEL_TYPES:
        if mt.name == name:
            return mt
    return None


def model_type_for_contract(stamp: str) -> type[ModelType[msgspec.Struct]] | None:
    """Ingest classification assist: recorded tensorfs contract stamp
    (``<name>@<version>``) → model type, via the fingerprint patterns.
    Returns None for an unrecognized stamp — unclassified is LEGAL and
    VISIBLE (NULL ``model``, serves on fallbacks with the named warning)."""
    for mt in MODEL_TYPES:
        if any(fnmatchcase(stamp, pattern) for pattern in mt.contracts):
            return mt
    return None


def defaults_vocabularies() -> dict[str, type[msgspec.Struct]]:
    """Every recognized name → its ``Defaults`` struct (base types first,
    then LoRA overlays) — the export emitter's source."""
    return {
        SDXL.name: SDXL.Defaults,
        SD15.name: SD15.Defaults,
        SD2.name: SD2.Defaults,
        HiDreamO1.name: HiDreamO1.Defaults,
        Wan22.name: Wan22.Defaults,
        SDXL.Lora.name: SDXL.Lora.Defaults,
        SD15.Lora.name: SD15.Lora.Defaults,
    }



# ── interim lane-dtype seam (pgw#1370's derive consumes this) ────────────────
#
# INTERIM dtype resolution for lanes spelled as bare contract HANDLES. A lane
# is a tensorfs layout contract; when the author imports the contract OBJECT
# (tensorfs#111), dtype rides on it and this table is not consulted. Bare
# handles resolve here until the canonical per-model-type entries land in
# tensorfs ``spec/v1/contracts``.

CONTRACT_DTYPES: dict[str, object] = {}


def register_contract_dtype(handle: str, dtype: object) -> None:
    known = CONTRACT_DTYPES.get(handle)
    if known is not None and known != dtype:
        raise ValueError(
            f"contract {handle!r} already resolves to {known!r}; refusing to "
            f"re-register it as {dtype!r}"
        )
    CONTRACT_DTYPES[handle] = dtype


def _seed_sdxl_contracts() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-less installs never derive
        return
    register_contract_dtype("sdxl.diffusers-bf16@1", torch.bfloat16)
    # The fp8-rowwise lane LOADS bf16 (the quantized artifact path is the fp8
    # pipeline's; the serve host's from_pretrained dtype stays bf16).
    register_contract_dtype("cozy.sdxl-fp8-rowwise@1", torch.bfloat16)


_seed_sdxl_contracts()


__all__ = [
    "CONTRACT_DTYPES",
    "HIDREAM_O1_DIFFUSERS_BF16",
    "HiDreamO1",
    "HiDreamO1Defaults",
    "Knob",
    "LORA_OVERLAYS",
    "LoraOverlay",
    "MODEL_TYPES",
    "ModelType",
    "PendingContract",
    "SD15",
    "SchedulerName",
    "SD15_DIFFUSERS_BF16",
    "SD2",
    "SD2_DIFFUSERS_BF16",
    "SDXL",
    "SDXL_DIFFUSERS_BF16",
    "Sd15Defaults",
    "Sd15LoraDefaults",
    "Sd15Config",
    "Sd2Defaults",
    "SdxlDefaults",
    "SdxlLoraDefaults",
    "SdxlConfig",
    "SupportsClamp",
    "TensorLayoutContract",
    "WAN22_DIFFUSERS_BF16",
    "Wan22",
    "Wan22Defaults",
    "defaults_vocabularies",
    "model_type_by_name",
    "model_type_for_contract",
    "register_contract_dtype",
]
