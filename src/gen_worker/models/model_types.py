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
    """Placeholder tensorfs Contract object (tensorfs#111 seam, see above).

    Satisfies the SDK's ``LaneContract`` protocol (pgw#1382): ``contract`` is
    the string handle and ``dtype`` the load dtype, so a canonical contract
    object may be written straight into a ``Model[...]`` ``lanes=`` header.
    """

    name: str
    version: int

    @property
    def stamp(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def contract(self) -> str:
        return self.stamp

    @property
    def dtype(self) -> object:
        return CONTRACT_DTYPES.get(self.stamp)


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

#: black-forest-labs/FLUX.2-klein-4B scheduler/scheduler_config.json, fetched
#: VERBATIM from the family owner's own repo (pgw#1393). FLUX.2 Klein is
#: FLOW-MATCHING, so there is no beta schedule here at all — the shape is a
#: shift ladder, and ``use_dynamic_shifting`` is exactly why pgw#1393 refused
#: to invent one: the effective shift is a function of image sequence length
#: between ``base_image_seq_len`` and ``max_image_seq_len``, so no frozen
#: triple could have stood in for it. 4B is the ONE BFL flux repo that is not
#: HF-gated; 9B is the same architecture (its endpoint module differs by a
#: single VRAM floor) and rides the same schedule. FLUX.1's own repos are
#: gated, so ``Flux1.canonical_scheduler_config`` stays empty — do not invent
#: it by analogy with these numbers.
FLUX2_KLEIN_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "use_dynamic_shifting": True,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "time_shift_type": "exponential",
    "shift_terminal": None,
    "invert_sigmas": False,
    "stochastic_sampling": False,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

SDXL_DIFFUSERS_BF16: Final = PendingContract("sdxl.diffusers-bf16", 1)
SD15_DIFFUSERS_BF16: Final = PendingContract("sd15.diffusers-bf16", 1)
SD2_DIFFUSERS_BF16: Final = PendingContract("sd2.diffusers-bf16", 1)
HIDREAM_O1_DIFFUSERS_BF16: Final = PendingContract("hidream-o1.diffusers-bf16", 1)
WAN22_DIFFUSERS_BF16: Final = PendingContract("wan22.diffusers-bf16", 1)
#: The one name here that is ALREADY a real tensorfs library contract
#: (``contracts.MINIMAX_H3_DIT_DIFFUSERS``, beside ``minimax.h3-dit-native``) —
#: the H3 contract file names it in ``lanes=``. Still a PendingContract only
#: because the vendored tensorfs snapshot predates the ``contracts`` module.
MINIMAX_H3_DIT_DIFFUSERS: Final = PendingContract("minimax.h3-dit-diffusers", 1)
#: Also already a real tensorfs library contract — and the ONLY shipped one
#: whose own ``description`` names the Flux family (pgw#1393). It is a
#: family-PLURAL block-spelling FRAGMENT, not a lane document: no top-level
#: ``dtype`` (so ``ctx.lane.dtype`` reads None on it, pgw#970) and its
#: patterns are the timm/native ``blocks.{i}.attn.qkv`` spelling. Flux's own
#: ``flux1.*``/``flux2-klein.*`` lane documents are OWED (tracked pgw#1393);
#: until then this is the honest placeholder, and every migrating flux
#: endpoint spells ``lanes=`` explicitly anyway.
DIT_BLOCKS_FUSED_QKV: Final = PendingContract("dit.blocks-fused-qkv", 1)


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
    #: The lane an endpoint gets when ``lanes=`` is omitted. None for an
    #: AUXILIARY type whose bytes have no canonical tensorfs layout contract
    #: yet (Rife) — the same "do not invent" rule that leaves
    #: ``canonical_scheduler_config`` empty for SD2/HiDreamO1/Wan22.
    canonical_contract: ClassVar[TensorLayoutContract | None] = None
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


class MiniMaxH3Defaults(msgspec.Struct, frozen=True):
    """MiniMax-H3 (joint video+audio DiT). Values from the OLD endpoint's own
    per-checkpoint schema (`minimax_h3/main.py` ``MiniMaxH3Defaults``, the
    ``register_family("minimax-h3", ...)`` row this vocabulary replaces).

    H3-Base is GUIDANCE-DISTILLED: guidance is baked into the weights — no
    guider, no negative prompt, no guidance_scale, one forward pass per step.
    Declaring a ``guidance`` knob would declare a knob the model does not
    have, so this type deliberately carries neither ``guidance`` nor a
    ``steps`` knob (the endpoint's duration/canvas presets fix the step
    count). The two shift values ARE the per-checkpoint facts."""

    video_shift: float = 12.0
    audio_shift: float = 3.0


class RifeDefaults(msgspec.Struct, frozen=True):
    """RIFE has no serving knobs — the interpolator takes source/target fps
    from the delivery preset, nothing per-checkpoint. Empty on purpose: the
    type exists for its NAME and its ingest fingerprint, so an auxiliary slot
    can be classified and typed like any other. Fields stay additive if a
    knob is ever ruled."""


class Flux1Defaults(msgspec.Struct, frozen=True):
    """FLUX.1 — BFL's rectified-flow DiT (dev, schnell, and the Flex.2-preview
    redistill). Values are SOURCED from the shipped flux endpoints, which are
    the family owner's own code; the citations live in the pgw#1393 tracker
    section and are repeated here field by field.

    ``steps`` 28 / ``guidance`` 3.5 are the BFL FLUX.1 base card numbers both
    endpoints already register (``flux.1-dev/main.py:69-71`` and
    ``flux.1-schnell/main.py:61-63`` call ``register_family("flux1", ...)``
    with a byte-identical schema — the family owner saying dev and schnell are
    ONE vocabulary). ``steps`` bounds 1..100 are the WIDEST envelope any
    ``flux1`` lane declares (``flux.1-schnell/main.py:267``, the Flex.2 lane);
    the platform knob must be the widest because ``defaults_decode`` only ever
    NARROWS — dev's ``le=50`` (``flux.1-dev/main.py:276``) and schnell's
    ``le=4`` (``flux.1-schnell/main.py:254``) are CHECKPOINT rows narrowing
    into it. ``guidance`` hi 10.0 from ``flux.1-dev/main.py:281``; its lo is
    0.0, NOT dev's wire ``ge=1.0``, and that floor is load-bearing — schnell
    PINS ``guidance_scale=0.0`` (``flux.1-schnell/main.py:388-389``), and
    since the knob merge only narrows, a platform floor of 1.0 makes the
    schnell checkpoint's own row unreachable (MEASURED: it decoded to
    ``lo=1.0, hi=0.0``, an empty range). The endpoint keeps its narrower wire
    bound; the platform envelope has to admit the family's real checkpoints.
    Same shape as ``Sd15Config.guidance`` (``Knob(7.0, lo=0.0)``).

    ``cfg`` is False because FLUX.1's ``guidance`` is NOT CFG: it is the
    guidance-distillation EMBEDDING, a DiT input tensor, so batch stays 1 and
    true-CFG is unreachable on the served path (``flux.1-dev/main.py:168-176``
    and ``:277-280``); schnell pins ``guidance_scale=0.0``
    (``flux.1-schnell/main.py:388-389``). A Flex.2 row flips it on.

    DELIBERATELY ABSENT: ``canonical_scheduler_config`` (Flux is flow-matching
    — no beta schedule to record, and ``FlowMatchEulerDiscreteScheduler``'s
    shift parameters are RESOLUTION-dependent; BFL's own
    ``scheduler/scheduler_config.json`` is HF-gated and unfetchable from the
    workspace, so it stays ``{}`` like SD2/HiDreamO1/Wan22 rather than being
    invented); no ``.Lora`` overlay (no flux endpoint registers a lora
    vocabulary, so there is no strength range or scheduler demand to source);
    no ``timesteps`` ladder (no flux endpoint passes one).
    """

    steps: Knob[int] = Knob(28, lo=1, hi=100, name="steps")
    guidance: Knob[float] = Knob(3.5, lo=0.0, hi=10.0, name="guidance")
    #: FLUX.1's guidance is the DISTILLATION EMBEDDING, not CFG — both BFL
    #: checkpoints serve cfg-off; the Flex.2 redistill runs a real CFG walk
    #: (``flux.1-schnell/main.py:109-113``) and its row sets True.
    cfg: bool = False
    #: schnell IS a 1-4 step timestep distillation
    #: (``flux.1-schnell/main.py:106-107``) and its row sets True; dev is not.
    #: Same checkpoint-level fact as ``SdxlDefaults.step_distilled``.
    step_distilled: bool = False
    #: The T5 text-sequence pin. A plain int, NOT a Knob (the MiniMaxH3 idiom):
    #: no endpoint exposes it on the wire, so there is nothing to clamp
    #: caller-visibly. 512 for dev/Flex.2 (``flux.1-dev/main.py:117``,
    #: ``flux.1-schnell/main.py:104``); schnell's row sets 256, BFL's own
    #: reference-snippet value (``flux.1-schnell/main.py:93``, ``:103``).
    max_sequence_length: int = 512


class Flux2KleinDefaults(msgspec.Struct, frozen=True):
    """FLUX.2 Klein — 4B and 9B under ONE vocabulary root (their endpoint
    modules diff to nothing but the type name and one VRAM floor, ``vram30g``
    vs ``vram44g``; a per-lane VRAM floor is a ``requires=`` fact on the model
    CLASS, never ModelType state). Same one-root precedent as :class:`Wan22`.

    A SEPARATE type from :class:`Flux1`, on the endpoints' own evidence: rope
    coordinates are ``(B, T, 4)`` int64 here
    (``flux.2-klein-4b/main.py:235-236``) against FLUX.1's batchless
    ``(T, 3)`` (``flux.1-dev/main.py:244-245``, which states the contrast
    outright at ``:211-212``), the pipeline class is ``Flux2KleinPipeline``,
    and — the one that matters to a defaults vocabulary — Klein Base is a
    CFG-MECHANISM model: it passes ``guidance=None`` to the transformer always
    (``:239-241``) and runs a real second uncond forward, batch-2 on every
    legal request (``:123-129``, ``:307``). One ``cfg`` bool cannot be the
    platform default for both families.

    Values: ``steps`` 28 / ``guidance`` 4.0 are BFL's Base card numbers
    (``flux.2-klein-4b/main.py:84-86``). The bounds deliberately do NOT copy
    the Base HANDLER's wire envelope (``ge=12``/``ge=1.5``, ``:306``/``:310``):
    the platform knob must admit the Turbo checkpoint's published four-step
    guidance-1.0 recipe (``:94-95``), and ``_merge_int_knob`` clamps a row's
    default INTO the platform range — a floor of 12 would silently serve a
    Turbo row at 12 steps. So 1..50 and 1.0..10.0, with the endpoint keeping
    its own narrower wire bounds. Upper bounds are the endpoint's (``:306``,
    ``:310``).

    The noise schedule IS recorded for this family —
    :data:`FLUX2_KLEIN_SCHEDULER_CONFIG`, fetched verbatim from the one BFL
    flux repo that is not HF-gated. DELIBERATELY ABSENT, same reasons as
    :class:`Flux1Defaults`: a ``.Lora`` overlay, a ``timesteps`` ladder (the
    flow-match sigma ladder is derived from the shift parameters, never
    pinned by an endpoint). Also absent: the
    preset grids / megapixel tiers and the 1..4 ordered-reference bound
    (``flux.2-klein-4b/main.py:136-137``, ``:357-360``) — those are endpoint
    PAYLOAD vocabulary, and a ModelType is name + Defaults + fingerprint,
    nothing else.
    """

    steps: Knob[int] = Knob(28, lo=1, hi=50, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=1.0, hi=10.0, name="guidance")
    #: Klein Base is a CFG-mechanism model (second uncond forward); the Turbo
    #: distillation's row sets False.
    cfg: bool = True
    #: Turbo rows set True — ``flux.2-klein-4b/main.py:93-95``, ``:418-431``.
    step_distilled: bool = False
    #: ``flux.2-klein-4b/main.py:121``. Plain int for the same reason as
    #: :class:`Flux1Defaults`.
    max_sequence_length: int = 512


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


class MiniMaxH3(ModelType[MiniMaxH3Defaults]):
    """MiniMax-H3 — COMPILABLE like any other type (Paul, 2026-08-20,
    reversing the old F3 misfiling: the "737k static classes" refusal was an
    artifact of the pre-enumerated static-bucket design, not a property of
    H3; F3's eager-permanent tier now means EXTERNAL-BINARY runtimes only).
    Nothing in a ModelType was ever compile-related, so that ruling costs
    this vocabulary nothing — it is why H3 is an ORDINARY entry here."""

    name = "minimax-h3"
    contracts = ("minimax.h3-*",)
    canonical_contract = MINIMAX_H3_DIT_DIFFUSERS
    Defaults = MiniMaxH3Defaults


class Rife(ModelType[RifeDefaults]):
    """RIFE v4.25 frame interpolation — a small AUXILIARY model with its own
    checkpoint, bound beside a video model to serve the fps>24 delivery
    presets. Name + fingerprint seam only: no canonical lane (its artifact is
    a plain diffusers-layout repo, and inventing a tensorfs contract name for
    it would be a guess) and no Defaults vocabulary beyond the base."""

    name = "rife"
    contracts = ("rife.*",)
    Defaults = RifeDefaults


class Flux1(ModelType[Flux1Defaults]):
    """FLUX.1 — dev, schnell and the Flex.2-preview redistill under ONE root.

    Not a judgement call: ``flux.1-dev`` and ``flux.1-schnell`` both call
    ``register_family("flux1", Flux1Defaults)`` with a byte-identical schema,
    and schnell's docstring says so ("the SAME family vocabulary flux.1-dev
    declares"). schnell's 1-4 step distillation is handled as a caller-visible
    CLAMP against that shared schema (``flux.1-schnell/main.py:356-371``) —
    a ``Defaults`` fact, not a second vocabulary. Flex.2-preview registers no
    family of its own and reads the same ``ctx.defaults`` (``:436-441``).

    The fingerprint is ``flux1.*`` — the future flux-specific contract names,
    mirroring how ``sdxl.*`` matches ``sdxl.diffusers-bf16@1``.
    ``dit.blocks-fused-qkv*`` is deliberately NOT a fingerprint pattern: that
    document is family-PLURAL ("shared by Flux-family and timm-derived
    transformers"), so matching on it would classify every timm ViT as Flux.
    An unmatched stamp is legal and visible (see
    :func:`model_type_for_contract`).
    """

    name = "flux1"
    contracts = ("flux1.*",)
    canonical_contract = DIT_BLOCKS_FUSED_QKV
    Defaults = Flux1Defaults


class Flux2Klein(ModelType[Flux2KleinDefaults]):
    """FLUX.2 Klein — 4B and 9B, Base and Turbo, under one vocabulary root
    (see :class:`Flux2KleinDefaults` for the architecture evidence separating
    it from :class:`Flux1`, and for why the 4B/9B split is a ``requires=``
    fact rather than a type)."""

    name = "flux2-klein"
    contracts = ("flux2-klein.*",)
    canonical_contract = DIT_BLOCKS_FUSED_QKV
    canonical_scheduler_config = FLUX2_KLEIN_SCHEDULER_CONFIG
    Defaults = Flux2KleinDefaults


MODEL_TYPES: Final[tuple[type[ModelType[msgspec.Struct]], ...]] = (
    SDXL,
    SD15,
    SD2,
    HiDreamO1,
    Wan22,
    MiniMaxH3,
    Rife,
    Flux1,
    Flux2Klein,
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
        MiniMaxH3.name: MiniMaxH3.Defaults,
        Rife.name: Rife.Defaults,
        Flux1.name: Flux1.Defaults,
        Flux2Klein.name: Flux2Klein.Defaults,
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
    "DIT_BLOCKS_FUSED_QKV",
    "FLUX2_KLEIN_SCHEDULER_CONFIG",
    "Flux1",
    "Flux1Defaults",
    "Flux2Klein",
    "Flux2KleinDefaults",
    "HIDREAM_O1_DIFFUSERS_BF16",
    "HiDreamO1",
    "HiDreamO1Defaults",
    "Knob",
    "LORA_OVERLAYS",
    "LoraOverlay",
    "MINIMAX_H3_DIT_DIFFUSERS",
    "MiniMaxH3",
    "MiniMaxH3Defaults",
    "Rife",
    "RifeDefaults",
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
