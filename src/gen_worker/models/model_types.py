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
# A canonical contract is an IMPORTED OBJECT, never a pointer-string (Paul's
# contract-objects ruling 2026-08-18), and pgw#1391 makes the object REAL: the
# constants below resolve against the vendored tensorfs contract library, so a
# lane stamp is now falsifiable rather than free text.
#
# What that replaced (pgw#1391, measured): ``PendingContract(name, version)``
# was a placeholder that answered ``stamp``/``contract`` for ANY name. Four of
# the six constants named documents tensorfs does not ship, so a ``Model``
# subclass that omitted ``lanes=`` fell through ``model_lanes()`` to one of
# them and claimed, all the way into the published manifest, a lane the hub
# cannot intern (se#757 blocker A). The live sdxl lane was no better off: a
# placeholder has no ``document`` and no ``digest``, so every release shipped
# ``"document": null`` for a contract tensorfs really does publish.
#
# A name the library does not carry becomes a ``MissingContract`` sentinel.
# Import stays clean — the module must remain importable or every consumer
# breaks — and CLAIMING one as a lane is what refuses. All ten current names
# resolve, so no sentinel is live today; the MACHINERY STAYS, because se#757
# blocker B is 21 more endpoints that will each want a ``ModelType`` before
# their document exists, and this is the shape that lets them declare one
# without shipping a lie. Clearing the original four took zero code change,
# which is the property worth keeping.

from .._vendor.tensorfs import contracts as _tensorfs_contracts


class TensorLayoutContract(Protocol):
    """What ``canonical_contract`` is typed against: the tensorfs ``Contract``
    surface a lane header, discovery and derive all read."""

    @property
    def stamp(self) -> str: ...
    @property
    def document(self) -> str: ...
    @property
    def digest(self) -> str: ...
    @property
    def dtype(self) -> str: ...


class MissingContractError(LookupError):
    """A lane names a layout contract tensorfs ships no document for.

    Raised on ANY lane-ish read of a :class:`MissingContract`, which is what
    turns the se#757 silent lie loud: the stamp cannot be spelled, so it cannot
    reach a manifest, a release document or a load.
    """


class MissingContract:
    """A named contract the vendored tensorfs library does not carry.

    INERT AT IMPORT, LOUD ON USE. A hard refusal at module import would make
    ``gen_worker.models`` unimportable and break everything downstream, so the
    constant exists — but every attribute a lane header, discovery, derive or a
    load would read (``stamp``, ``contract``, ``dtype``, ``document``,
    ``digest``, ``label``, ``tensors``, ``sets``) refuses instead of answering.
    It is deliberately NOT a ``LaneContract``-shaped liar: there is no path by
    which it produces a string somebody could publish.
    """

    _name: str
    _version: int

    __slots__ = ("_name", "_version")

    def __init__(self, name: str, version: int) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_version", version)

    def __setattr__(self, attribute: str, value: object) -> None:
        raise AttributeError("a MissingContract is frozen")

    def _refuse(self) -> MissingContractError:
        carried = ", ".join(sorted(contract.stamp for contract in _tensorfs_contracts.all()))
        return MissingContractError(
            f"lane {self._name}@{self._version} names a tensorfs layout "
            f"contract that DOES NOT EXIST. The vendored library carries: "
            f"{carried}. A lane stamp is a promise the hub interns and the "
            f"loader reads — it cannot be claimed by naming it. Author "
            f"'{self._name}.v{self._version}.json' in tensorfs "
            f"spec/v1/contracts (see its README), then re-vendor per "
            f"gen_worker/_vendor/VENDORED.toml; this constant becomes a real "
            f"contract with no code change. Until then declare a lane the "
            f"library carries, or lanes=() for eager-permanent."
        )

    # Every read a lane header, discovery, derive or a load performs.
    @property
    def stamp(self) -> str:
        raise self._refuse()

    @property
    def contract(self) -> str:
        raise self._refuse()

    @property
    def dtype(self) -> str:
        raise self._refuse()

    @property
    def document(self) -> str:
        raise self._refuse()

    @property
    def digest(self) -> str:
        raise self._refuse()

    @property
    def label(self) -> str:
        raise self._refuse()

    @property
    def tensors(self) -> tuple[object, ...]:
        raise self._refuse()

    @property
    def sets(self) -> dict[str, tuple[str, ...]]:
        raise self._refuse()

    def __repr__(self) -> str:
        return f"MissingContract({self._name!r}, {self._version!r}, NO DOCUMENT)"


def _library(name: str, version: int) -> TensorLayoutContract:
    """The library contract ``name@version``, or the sentinel that refuses.

    The lookup happens at import so a name the library DOES carry is a real
    ``Contract`` from that moment on — document, digest and dtype included —
    while a name it does not carry costs nothing until somebody claims it.
    """

    try:
        return _tensorfs_contracts.get(f"{name}@{version}")
    except KeyError:
        return MissingContract(name, version)


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
#: `tensorhub/qwen-image` scheduler/scheduler_config.json, read VERBATIM from
#: the hub tree the fleet serves (pgw#1426). BYTE-IDENTICAL on
#: `tensorhub/qwen-image-edit-2511`, which is one more piece of the evidence
#: that the two arms are one family. Flow-matching, so there is no beta
#: schedule; `use_dynamic_shifting` makes the effective shift a function of
#: image sequence length between `base_image_seq_len` and `max_image_seq_len`,
#: which is exactly why no frozen triple could stand in for it.
QWEN_IMAGE_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "use_dynamic_shifting": True,
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "time_shift_type": "exponential",
    "shift_terminal": 0.02,
    "invert_sigmas": False,
    "stochastic_sampling": False,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

#: `tensorhub/z-image` scheduler/scheduler_config.json, read VERBATIM from the
#: hub tree (pgw#1426). THE BASE CHECKPOINT'S, DELIBERATELY: the served
#: `tensorhub/z-image-turbo` tree ships the same class at `shift: 3.0` against
#: this `6.0`, and a canonical scheduler config is the FAMILY ROOT's training
#: schedule, not an average. That difference costs nothing at serve time —
#: ingest only synthesizes this into a tree that ships NO scheduler_config.json,
#: and both served z-image trees ship their own.
Z_IMAGE_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 6.0,
}

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

#: baidu/ERNIE-Image ``scheduler/scheduler_config.json``, fetched VERBATIM from
#: the HUB's revision-pinned clone (``tensorhub/ernie-image@prod``,
#: ``source_repo baidu/ERNIE-Image``, ``source_revision
#: 5346b31d68c9c23758ba56ef8be5e9dc174c7f99``). ``ernie-image-turbo`` ships a
#: BYTE-IDENTICAL file (both ``sha256:96031c39fcd4651ae…``, 482 B), so one
#: config covers the family.
#:
#: NOT filled by analogy with :data:`FLUX2_KLEIN_SCHEDULER_CONFIG`, and the
#: contrast is the reason to say so: ERNIE is flow-matching too, but ships
#: ``shift 4.0`` with ``use_dynamic_shifting`` FALSE against Klein's 3.0/TRUE.
#: The temptation is live rather than hypothetical — ingest currently
#: MISCLASSIFIES both ernie checkpoints as flux (``metadata.model_family
#: "flux"``, ``model_family_variant "flux2"``, because ``model_index.json``
#: declares ``vae: AutoencoderKLFlux2`` and ``_class_name
#: ErnieImagePipeline`` is not in the family map), so the wrong analogy is
#: exactly the one a reader is handed. Copying Klein's numbers would have
#: silently changed this family's sigma ladder.
ERNIE_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 4.0,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "use_dynamic_shifting": False,
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

#: REAL — a tensorfs library document. Live on deploy lane
#: ``a55e45a571cdb6188``.
SDXL_DIFFUSERS_BF16: Final = _library("sdxl.diffusers-bf16", 1)
#: REAL — beside ``minimax.h3-dit-native``; the H3 contract file names it in
#: ``lanes=``. Live on deploy lane ``ab56185761f1597f1``.
MINIMAX_H3_DIT_DIFFUSERS: Final = _library("minimax.h3-dit-diffusers", 1)

#: REAL as of tensorfs#121, which authored these four for se#757 blocker A.
#: They were ``MissingContract`` sentinels between pgw#1391's first commit and
#: its re-vendor, and clearing them took NO code change here — the property the
#: sentinel shape exists to have. ``sd15`` and ``sd2`` are SEPARATE documents
#: with separate digests, not one shared one.
SD15_DIFFUSERS_BF16: Final = _library("sd15.diffusers-bf16", 1)
SD2_DIFFUSERS_BF16: Final = _library("sd2.diffusers-bf16", 1)
HIDREAM_O1_DIFFUSERS_BF16: Final = _library("hidream-o1.diffusers-bf16", 1)
WAN22_DIFFUSERS_BF16: Final = _library("wan22.diffusers-bf16", 1)

#: REAL as of tensorfs#131 — the qwen-image and z-image transformer-only lane
#: documents (45 and 56 declarations), both derived from the trees the fleet
#: SERVES rather than from HuggingFace. `qwen-image.diffusers-bf16@1` covers
#: BOTH shipped repos (`tensorhub/qwen-image` and
#: `tensorhub/qwen-image-edit-2511`), whose transformers are byte-layout
#: identical — 1933 tensors, zero name/shape/dtype diffs;
#: `z-image.diffusers-bf16@1` likewise covers `tensorhub/z-image` and
#: `tensorhub/z-image-turbo` (521 tensors, same measurement).
#:
#: THE Z-IMAGE DOCUMENT IS DELIBERATELY NOT THE HUGGINGFACE PACKAGING.
#: `Tongyi-MAI/Z-Image-Turbo` ships its transformer in F32 (521/521, measured);
#: the hub's served bf16 republish of the same revision is BF16 (521/521), with
#: identical names and shapes. The document follows the served tree, so the F32
#: packaging misses it entirely instead of half-matching.
QWEN_IMAGE_DIFFUSERS_BF16: Final = _library("qwen-image.diffusers-bf16", 1)
Z_IMAGE_DIFFUSERS_BF16: Final = _library("z-image.diffusers-bf16", 1)

#: REAL as of tensorfs#124 — FLUX.2 Klein's own transformer-only lane document
#: (29 declarations). This is what makes ``flux.2-klein-4b``/``-9b``
#: migratable, and it is what ``Flux2Klein.canonical_contract`` points at.
FLUX2_KLEIN_DIFFUSERS_BF16: Final = _library("flux2-klein.diffusers-bf16", 1)

#: REAL as of tensorfs#124's second half (tensorfs#136). It was a
#: ``MissingContract`` sentinel until this document was vendored, and clearing
#: it took NO code change — the property the sentinel shape exists to have.
#:
#: THE SENTINEL WAS RIGHT TWICE OVER, and the second reason was only measured
#: when the document was authored. ``Flux1`` once pointed at
#: ``dit.blocks-fused-qkv@1``, a family-PLURAL block-spelling FRAGMENT in the
#: timm/native ``blocks.{i}.attn.qkv`` spelling: it declares that pattern
#: ``required: true`` and matches ZERO tensors in a real Flux transformer
#: header, whose tree is
#: ``transformer_blocks.*``/``single_transformer_blocks.*``. That was a
#: guaranteed refusal dressed as a resolved lane. But the alternative on offer
#: was worse and quieter — ``flux2-klein.diffusers-bf16@1`` explains 308 of a
#: FLUX.1 transformer's 1160 tensors with NO dtype or rank refusal, so it WON
#: every FLUX.1 file until this document existed (measured upstream on dev,
#: schnell and Flex.2-preview alike). Pointing at nothing beat both.
#:
#: The document is transformer-only and covers all three checkpoints the fleet
#: serves: FLUX.1-dev 1160/1160, FLUX.1-schnell 1156/1156 (the four-tensor
#: ``guidance_embedder`` delta is declared optional) and ostris/Flex.2-preview
#: 808/808, whose ``x_embedder`` is 196 channels wide rather than 64 — its
#: declarations constrain rank and dtype and never shape.
FLUX1_DIFFUSERS_BF16: Final = _library("flux1.diffusers-bf16", 1)

#: REAL, and the only shipped document whose own ``description`` names the Flux
#: family — but it is a FRAGMENT, not a lane document, and nothing here points
#: a ``canonical_contract`` at it any more (see ``FLUX1_DIFFUSERS_BF16``). It
#: declares no top-level ``dtype`` on purpose, as the two ``sdxl.clip-g-*``
#: component fragments do, so claiming it as a serve lane refuses at
#: declaration instead of reading ``None`` at load.
DIT_BLOCKS_FUSED_QKV: Final = _library("dit.blocks-fused-qkv", 1)


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


class QwenTextGenDefaults(msgspec.Struct, frozen=True):
    """The AUTOREGRESSIVE sampling vocabulary — the first non-diffusion
    ``Defaults`` in this module, and the axes both qwen3.6 roots share as ONE
    nominal type (the ``SdxlConfig`` idiom).

    None of the diffusion vocabulary means anything here: there are no steps,
    no guidance, no cfg, no timesteps and no scheduler. A decode is one
    autoregressive pass per token, so the knobs are the sampler's —
    ``max_tokens`` / ``temperature`` / ``top_p``.

    🔴 ``max_tokens`` DELIBERATELY CARRIES NO ``hi``, and that is the one
    judgement in this vocabulary. Both endpoints clamp it — 32768 for the 27b
    (``qwen3.6-27b-mtp-gguf/main.py:83``) and 16384 for the 35b
    (``qwen3.6-35b-a3b/main.py:98``) — but each of those numbers is a CARD
    BUDGET, not a model fact, and both comments say so in the endpoint's own
    words ("32k with q8_0 KV fits the 48GB class next to the 17.9GB q4
    weights", ``:31-32``; "KV pre-allocation cap for the 48GB class",
    ``:26``). The checkpoints themselves declare
    ``max_position_embeddings: 262144``.

    Promoting a 48GB budget to the platform layer would repeat pgw#1393's
    MEASURED ``Flux1.guidance`` defect one field over: ``defaults_decode``
    only ever NARROWS, so a platform ``hi`` of 16384 makes a legitimate
    262k-context checkpoint row UNREACHABLE and no larger card could widen
    back out. The endpoints keep their own narrower wire clamps, which is
    where a card budget belongs. ``lo=1`` IS sourced (both clamps).

    ``temperature``/``top_p`` carry a default and NO bounds: neither endpoint
    declares a range, and inventing a sampling envelope is the same defect.
    They stay ``Knob`` rather than plain values because the wire exposes both
    (``CompletionInput.temperature``/``top_p``), so a per-checkpoint row can
    narrow them later without a schema change.
    """

    max_tokens: Knob[int] = Knob(256, lo=1, name="max_tokens")
    temperature: Knob[float] = Knob(0.7, name="temperature")
    top_p: Knob[float] = Knob(0.95, name="top_p")


class QwenMtpDefaults(QwenTextGenDefaults, frozen=True):
    """Qwen3.6-27B-MTP, served as GGUF through llama.cpp with MTP speculative
    decoding. Values from the family owner's own
    ``register_family("qwen3.6-27b-mtp", QwenMtpDefaults)`` row
    (``qwen3.6-27b-mtp-gguf/src/qwen36_27b_mtp_gguf/main.py:35-43``):
    ``max_tokens`` 256, ``temperature`` 0.6, ``top_p`` 0.95.

    ``temperature`` 0.6 is the ONLY axis on which this root differs from
    :class:`QwenA3bDefaults` (0.7) — which is precisely why the two are
    separate roots rather than one shared vocabulary, unlike pgw#1393's
    ``Flux1`` where dev and schnell register byte-identical schemas.

    The speculative-decoding knobs (``--spec-draft-n-max 6``) are NOT here:
    they are engine flags on the endpoint's own launch line, invisible to the
    wire, and a caller cannot set them.
    """

    temperature: Knob[float] = Knob(0.6, name="temperature")


class QwenA3bDefaults(QwenTextGenDefaults, frozen=True):
    """Qwen3.6-35B-A3B, served fp8 through vLLM. Values from the family
    owner's own ``register_family("qwen3.6-35b-a3b", QwenA3bDefaults)`` row
    (``qwen3.6-35b-a3b/src/qwen36_35b_a3b/main.py:30-37``): ``max_tokens``
    256, ``temperature`` 0.7, ``top_p`` 0.95 — i.e. the shared base
    unchanged, so this type adds no field of its own.

    Upstream ``generation_config.json`` independently corroborates
    ``top_p: 0.95``. It also carries ``temperature: 1.0`` and ``top_k: 20``;
    0.7 is the family owner's SERVING choice and is what this vocabulary
    ports, per the "source from the family's own endpoint code" rule.
    ``top_k`` is deliberately absent — no endpoint in this family exposes it,
    so there is nothing to clamp and nothing sourced to default it to.
    """


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

    DELIBERATELY ABSENT: ``canonical_scheduler_config``, and the REASON was
    corrected at tensorfs#136 — the old one ("BFL's
    ``scheduler/scheduler_config.json`` is HF-gated and unfetchable from the
    workspace") has DISSOLVED and must not be left standing, because a ``{}``
    with a stale reason gets cleared by the next reader who assumes it was
    only ever an access problem. The file is fetchable: it is a whole-file
    entry in the hub's own resolve manifest for ``tensorhub/flux1-dev`` and
    ``tensorhub/flux1-schnell``, and both were read.

    It stays ``{}`` on the MEASURED fact instead, which is a stronger reason
    than the access one ever was: the two checkpoints under this one family
    root DISAGREE. Both are ``FlowMatchEulerDiscreteScheduler`` with
    ``base_shift`` 0.5, ``max_shift`` 1.15 and image_seq_len 256..4096, but
    dev is ``shift`` 3.0 with ``use_dynamic_shifting`` TRUE while schnell is
    ``shift`` 1.0 with it FALSE. One canonical schedule for the root would
    therefore be wrong for one of the two, and copying FLUX.2 Klein's
    (3.0/dynamic — see :data:`FLUX2_KLEIN_SCHEDULER_CONFIG`) by family
    resemblance would be right for dev and wrong for schnell. The per-arm
    values are CHECKPOINT facts and belong in the catalog row, not here.
    No ``.Lora`` overlay (no flux endpoint registers a lora
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


class Krea2Defaults(msgspec.Struct, frozen=True):
    """Krea 2 — a 12.9B single-stream MMDiT (rectified flow) with a Qwen3-VL-4B
    text encoder and the Qwen-Image f8 VAE, served as TWO checkpoints under one
    vocabulary: the undistilled Raw mirror and the TDM-distilled Turbo mirror.

    Ported from the family owner's own ``Krea2Defaults``
    (``krea-2/src/krea_2/main.py:72-85``, the ``register_family("krea-2", ...)``
    row this vocabulary replaces). Values: ``steps`` 28 (``:83``) and
    ``guidance`` 3.5 (``:84``) — diffusers' own ``Krea2Pipeline`` signature
    defaults, which the endpoint states are deliberately the UNCONFIGURED stamp
    rather than either lane's card recipe (``:76-81``).

    BOUNDS ARE NOT THE RAW HANDLER'S WIRE ENVELOPE, and that is load-bearing
    twice over. ``_merge_int_knob`` clamps a checkpoint row's default INTO the
    platform range, so a floor copied from the Raw lane silently rewrites the
    Turbo checkpoint. Both cases were RED/GREEN VERIFIED against the real
    ``defaults_decode`` merge functions, not reasoned about:

    * ``steps`` — Raw's wire floor is ``ge=20`` (``:313``) but Turbo's published
      TDM recipe is 8 (``:94``). MEASURED: platform ``lo=20`` decodes the Turbo
      row's 8 to **20**. So ``lo=1``; ``hi=80`` is Raw's own ceiling (``:313``),
      which already admits Turbo's ``le=16`` (``:340``).
    * ``guidance`` — Raw's wire floor is ``ge=1.0`` (``:316``) but the Turbo lane
      PINS guidance to 0.0 (``_TURBO_GUIDANCE``, ``:95``) and the distill is only
      valid guidance-free (``:517-518``). MEASURED: platform ``lo=1.0`` against
      the Turbo row decodes to ``lo=1.0, hi=0.0`` — an EMPTY RANGE. That is the
      defect :class:`Flux1Defaults` records for schnell (see its docstring),
      reproduced in a SECOND family, which is what makes it a class rather than
      an anecdote. Floor is 0.0; ``hi`` 10.0 is Raw's ceiling (``:316``).

    The endpoints keep their narrower wire bounds; the platform envelope has to
    admit the family's real checkpoints.

    DELIBERATELY ABSENT — endpoint PAYLOAD vocabulary, not ModelType state (the
    :class:`Flux2KleinDefaults` exclusion): the ``AspectRatio`` enum and the
    1mp/2mp preset grids (``:117-149``), the ``megapixels`` tier (``:311``),
    ``output_format`` (``:321``), and ``_KREA2_TOKEN_DIV`` (``:177``, a
    compile-geometry constant). Also absent: ``canonical_scheduler_config`` —
    krea-2's checkpoints are not in the hub at all (see :class:`Krea2`), so the
    file is UNREACHABLE rather than nonexistent, and inventing one is forbidden.
    """

    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(3.5, lo=0.0, hi=10.0, name="guidance")
    #: Krea 2 Raw runs a REAL uncond forward — but as two SEQUENTIAL batch-1
    #: calls rather than a batch-2 pass (``krea-2/src/krea_2/main.py:160-163``,
    #: ``:200-205``). Still the CFG mechanism, so True; the Turbo row sets False.
    #: Note the contrast with ``Flux1Defaults.cfg``, False because FLUX.1's
    #: guidance is a distillation EMBEDDING and no uncond forward exists at all
    #: — a different reason for a different value.
    cfg: bool = True
    #: The Turbo mirror IS a TDM timestep distillation and its row sets True
    #: (``:499`` declares ``distilled=True``); Raw is not (``:464``).
    step_distilled: bool = False
    #: ``krea-2/src/krea_2/main.py:152``, passed as a call argument at ``:446``.
    #: A plain int, NOT a Knob (the :class:`MiniMaxH3Defaults` idiom): no
    #: endpoint exposes it on the wire, so there is nothing to clamp
    #: caller-visibly.
    max_sequence_length: int = 512


class AnimaDefaults(msgspec.Struct, frozen=True):
    """Anima (``circlestone-labs/Anima``) — an anime-focused ~2B text-to-image
    DiT on NVIDIA's Cosmos-Predict2-2B backbone. ComfyUI-native (a split
    checkpoint, not a diffusers repo), run through DiffSynth's
    ``AnimaImagePipeline``.

    Ported from ``anima/src/anima/main.py:110-122``. Values: ``steps`` 35
    (``:119``) and ``guidance`` 4.5 (``:120``) — the model card recipe the
    docstring quotes at ``:111`` ("CFG ~4-5, 30-50 steps"); ``negative`` ""
    (``:121``).

    BOUNDS — the same clamp-direction reasoning as :class:`Krea2Defaults`, and
    this family has the sharpest instance of it. Both RED/GREEN VERIFIED:

    * ``steps`` — the base handler's wire floor is ``ge=30`` (``:313``) while the
      turbo distill regime is 10 (``_TURBO_STEPS``, ``:133``). MEASURED: platform
      ``lo=30`` decodes an already-distilled row's 10 to **30** — three times the
      work at the wrong regime, silently. ``hi=50`` (``:313``). ``lo=1`` rather
      than the widest DECLARED floor of 10: a step count's physical floor is 1,
      every other family in :data:`MODEL_TYPES` spells it that way, and a floor
      of 1 cannot clamp any row UP — the only direction that corrupts. Widening
      below what a lane declares is safe here precisely because the hazard is
      one-directional.
    * ``guidance`` — base wire floor ``ge=1.5`` (``:314``); the turbo lane pins
      CFG to 1.0 (``_TURBO_CFG``, ``:134``). MEASURED: ``lo=1.5`` decodes the
      turbo row's 1.0 to **1.5**, i.e. CFG ON for a distill that is only valid
      with it off. ``lo=1.0``, ``hi=10.0`` (``:314``).

    That path is REACHABLE IN PRODUCTION, which is why the floors matter here
    and are not defensive: ``generate_turbo`` reads ``resolved.defaults`` for an
    already-distilled bind (``:549-560``), and that is exactly the row the
    platform knob merges.

    ``negative`` is a plain ``str``, not a ``Knob`` — ``Knob`` is numeric
    (``SupportsClamp``) and a default negative prompt has no range to clamp.
    Read at ``:525``.

    DELIBERATELY ABSENT:

    * ``max_sequence_length`` — anima declares NO ``compile=`` block at all and
      therefore no ``Compile(text_len=)`` pin (``:469-478``, where the omission
      is stated as load-bearing: torch.compile measured no win, and a compile
      declaration would classify the function as hub-delivered). There is no
      text-sequence pin to source, so the field stays out. This is the one field
      where anima honestly differs from krea-2 and ernie.
    * ``canonical_scheduler_config`` — the prod checkpoint holds THREE files
      (DiT, text encoder, VAE) and none is a scheduler config; DiffSynth builds
      the scheduler in code. This is the HONEST empty of the "do not invent"
      rule (the value does not exist), NOT the unreachable empty.
    * a ``.Lora`` overlay — the endpoint declares ``lora_bucket=32`` (``:468``)
      and the fleet binds ``tensorhub/anima-turbo-lora@prod`` at weight 1.0
      (``e2e/manifests/bindings.yaml:141``), but ``lora_bucket`` is an
      ``@endpoint`` declaration about adapter RANK, not a defaults vocabulary,
      and no anima code registers a lora Defaults struct. Same call and same
      reason as :class:`Flux1Defaults`.
    * the ``AspectRatio`` enum and preset grid (``:137-158``) and
      ``output_format`` — endpoint payload vocabulary.
    """

    steps: Knob[int] = Knob(35, lo=1, hi=50, name="steps")
    guidance: Knob[float] = Knob(4.5, lo=1.0, hi=10.0, name="guidance")
    negative: str = ""
    #: The base lane runs a real CFG walk (``cfg_scale`` through DiffSynth with
    #: the wire floor at 1.5, ``anima/src/anima/main.py:314``, ``:520-524``); the
    #: turbo overlay/distilled row sets False (``:590`` serves at ``_TURBO_CFG``).
    cfg: bool = True
    #: The curated turbo distill's row sets True; the base is undistilled
    #: (``:506`` declares ``distilled=False``).
    step_distilled: bool = False


class ErnieDefaults(msgspec.Struct, frozen=True):
    """Baidu ERNIE-Image — an 8B single-stream DiT (Apache 2.0) served through
    diffusers' ``ErnieImagePipeline`` as a full checkpoint plus a step-distilled
    Turbo.

    Ported from ``ernie/src/ernie/main.py:84-98``. Values: ``steps`` 28 (``:95``,
    which the docstring records as ie#533's CORRECTED base value — the card's 50
    is "more of a max than a recommendation"), ``guidance`` 4.0 (``:96``),
    ``negative`` "" (``:97``).

    BOUNDS:

    * ``steps`` — base wire is ``ge=1, le=100`` (``:235``) and the distilled
      recipe is 8 (``_DISTILLED_STEPS``, ``:81``), which a floor of 1 already
      admits. NO hazard here, and it is worth stating that it was checked rather
      than assumed: ``lo=1, hi=100`` also admits Turbo's ``le=16`` (``:254``).
    * ``guidance`` — base wire floor is ``ge=1.5`` (``:238``, chosen so the
      batch-2 CFG graph shape stays invariant) but the Turbo class PINS guidance
      to 1.0 (``:458``). MEASURED: platform ``lo=1.5`` decodes the Turbo row's
      1.0 to **1.5**. So ``lo=1.0`` — the value both real checkpoints actually
      reach — and ``hi=15.0`` (``:238``). The floor is 1.0 rather than 0.0
      because no ernie lane pins guidance off; sourcing stops at the family's
      real checkpoints rather than widening for a checkpoint that does not exist.

    ``canonical_scheduler_config`` IS recorded for this family
    (:data:`ERNIE_SCHEDULER_CONFIG`) — fetched verbatim from the hub's
    revision-pinned clone, byte-identical across base and Turbo.

    DELIBERATELY ABSENT: the ``AspectRatio`` preset grid (``:66-74``), ``use_pe``
    (``:242`` — a payload switch for the prompt-enhancer LLM), ``output_format``,
    and ``_ERNIE_LATENT_SCALE`` (``:135``, compile geometry).
    """

    steps: Knob[int] = Knob(28, lo=1, hi=100, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=1.0, hi=15.0, name="guidance")
    negative: str = ""
    #: The base pipeline cats the latent batch x2 when guidance > 1 and chunks
    #: the prediction (``ernie/src/ernie/main.py:186-190``) — a genuine batch-2
    #: CFG graph. The Turbo class is the distilled release at CFG 1.0, batch-1,
    #: and its row sets False.
    cfg: bool = True
    #: Turbo rows set True (``:437`` declares ``distilled=True``); the base class
    #: is undistilled (``:398``).
    step_distilled: bool = False
    #: ``ernie/src/ernie/main.py:77``, the ``Compile(text_len=)`` declaration,
    #: made true on the pipeline by ``_pin_text_sequence`` (``:269-304``, which
    #: sets ``tokenizer.model_max_length``). Plain int, MiniMaxH3 idiom — no
    #: endpoint exposes it on the wire.
class QwenImageDefaults(msgspec.Struct, frozen=True):
    """Qwen-Image — text-to-image AND native editing under ONE vocabulary root.

    THE ONE-ROOT CALL IS A MEASUREMENT, NOT A JUDGEMENT (pgw#1426). The two
    served checkpoints' transformers are byte-layout IDENTICAL: 1933 tensors
    each, name sets equal, ZERO shape diffs, ZERO dtype diffs, both
    ``_class_name: QwenImageTransformer2DModel`` with the same
    ``axes_dims_rope [16, 56, 56]`` / ``num_layers 60`` /
    ``attention_head_dim 128`` / ``joint_attention_dim 3584``, and byte-identical
    ``scheduler_config.json``. The only config delta is ``zero_cond_t`` on the
    edit arm — a forward-pass flag, not a tensor and not a knob.

    Run against the standard :class:`Flux2KleinDefaults` states for SPLITTING a
    root: the rope convention is IDENTICAL (same module, same axes), the
    pipeline class DIFFERS (``QwenImagePipeline`` vs
    ``QwenImageEditPlusPipeline``) — and the one that decides a DEFAULTS
    vocabulary, the CFG MECHANISM, is identical. Both arms run true-CFG
    (``do_true_cfg = true_cfg_scale > 1`` with a negative prompt), share one
    guidance axis and one recipe resolver in the shipped endpoint
    (``qwen-image/src/qwen_image/main.py:147-150``, ``:672-695``). ONE ``cfg``
    bool IS the right platform default for both, so the condition Flux2Klein
    names for a split is not met.

    THE ENDPOINT'S TWO ``@endpoint`` CLASSES ARE A COMPILE FACT, NOT A
    VOCABULARY FACT, and pgw#1112 says so in its own words: a compile target is
    an attribute PATH on ONE pipeline object, both slots owned ``.transformer``,
    so declaration order silently handed every mint to t2i. That is a reason for
    two ``Model`` classes and two cells. It is not a reason for two roots — the
    same shape :class:`Wan22` already serves with three model classes on one.
    The shipped code agrees outright: ONE
    ``register_family("qwen-image", QwenImageDefaults)`` (``:142``), both
    classes' handlers typed ``RequestContext[QwenImageDefaults]``, and the edit
    slot's ``family="qwen-image"`` declared EXPLICIT and called load-bearing
    (``:823``).

    Values, all sourced from the family's own v1 endpoint code:

    ``steps`` 30 is ``QwenImageDefaults.steps`` (``:138``) — the ie#488 gate-4
    H100 same-seed sweep, s30-cfg4 at parity with the 50-step card copy.
    ``guidance`` 4.0 is ``:139`` (the true-CFG scale). The BOUNDS deliberately
    do NOT copy the handler's wire envelope (``ge=10`` steps at ``:317``,
    ``ge=1.5`` guidance at ``:324``): the platform knob must admit the Lightning
    lane's published 8-step CFG-off recipe (``_TURBO_REGIME``, ``:430``), and
    ``_merge_int_knob`` clamps a row's default INTO the platform range — a floor
    of 10 would silently serve an 8-step Lightning row at 10 steps, and a
    guidance floor of 1.5 would make its 1.0 unreachable. That is the same
    measured trap :class:`Flux1Defaults` records for schnell's pinned 0.0. So
    1..80 and 1.0..12.0, with the endpoint keeping its narrower wire bounds;
    the upper bounds ARE the endpoint's (``:317``, ``:324``).

    ``negative`` is the true-CFG unconditioned prompt and Qwen's convention is a
    single SPACE, never the empty string (``:140``, enforced at ``:491-497``).

    DELIBERATELY ABSENT. ``max_guidance`` (v1's ``:141``) does not come across:
    it was a per-checkpoint CLAMP expressed as a second field because v1 had no
    range type, and ``Knob.hi`` IS that clamp now — a checkpoint row narrows
    ``guidance`` directly and ``ctx.clamp`` reports it caller-visibly. Carrying
    both would be two spellings of one fact, with only one of them enforced. No
    ``.Lora`` overlay: the Lightning adapters ride as deploy data and the family
    owner's code fixes their regime in ENDPOINT constants (8 steps, CFG off,
    plus a ln(3) exponential-shift scheduler override, ``:428-440``) — the
    scheduler demand is a config-override MAPPING, which the platform's
    ``SchedulerName`` vocabulary cannot spell, and no shipped code states a
    strength range to source. Absent beats invented (the
    :class:`Flux1Defaults` posture). No preset grids, megapixel tiers or the
    1..3 ordered-reference bound — those are endpoint PAYLOAD vocabulary.
    """

    steps: Knob[int] = Knob(30, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=1.0, hi=12.0, name="guidance")
    #: Both arms are true-CFG mechanism models (a second uncond forward). A
    #: Lightning-fused row sets False.
    cfg: bool = True
    #: Both shipped handlers declare ``distilled=False`` (``:761``, ``:776``,
    #: ``:842``, ``:864``) — Lightning arrives as an OVERLAY, never as distilled
    #: base weights. A fused Lightning checkpoint's row sets True.
    step_distilled: bool = False
    #: The true-CFG unconditioned prompt. A single space, not "" (``:140``).
    negative: str = " "
    #: The text-sequence pin. A plain int, NOT a Knob (the :class:`MiniMaxH3`
    #: idiom): no endpoint exposes it on the wire, so there is nothing to clamp
    #: caller-visibly. 512 is the t2i lane's ``_TEXT_LEN`` (``:173``) and the
    #: family root's value; the EDIT checkpoint's row sets 1024 (``:303``),
    #: because its condition images' vision tokens ride BEFORE the user text
    #: (~188 per image at 384^2, up to 3) and 512 would truncate the images
    #: away. Same checkpoint-level shape as :class:`Flux1Defaults`' schnell 256.
    max_sequence_length: int = 512


class ZImageDefaults(msgspec.Struct, frozen=True):
    """Z-Image — the undistilled base and the Decoupled-DMD Turbo distillation
    under ONE vocabulary root, the same one-root shape as :class:`Wan22`.

    Measured (pgw#1426): ``tensorhub/z-image`` and ``tensorhub/z-image-turbo``
    ship the SAME transformer layout — 521 tensors, identical names, shapes and
    dtypes, identical ``transformer/config.json`` (``dim 3840``, ``n_layers
    30``, ``n_heads 30``). Turbo is a different training of one architecture,
    which is a checkpoint row, not a second vocabulary.

    Values, all sourced from ``z-image/src/z_image/main.py``:

    ``steps`` 28 and ``guidance`` 4.0 are the base model card's own numbers, the
    v1 ``ZImageDefaults`` schema stamp (``:90-91``). ``steps`` bounds are the
    WIDEST envelope any z-image lane declares — the base handler's ``ge=1,
    le=80`` (``:275``) rather than the turbo handler's ``le=16`` (``:296``),
    because ``defaults_decode`` only ever NARROWS and a checkpoint row must be
    able to state the tighter bound itself.

    ``guidance``'s FLOOR IS 0.0 AND THAT IS LOAD-BEARING, not the base
    handler's wire ``ge=1.0`` (``:278``). Both turbo lanes PIN
    ``guidance=0.0`` (``:669``, ``:702``); since the knob merge only narrows, a
    platform floor of 1.0 makes the distilled checkpoints' own rows
    unreachable. This is exactly the empty-range defect
    :class:`Flux1Defaults` records as MEASURED for schnell. The endpoint keeps
    its narrower wire bound; the platform envelope has to admit the family's
    real checkpoints.

    DELIBERATELY ABSENT, for the reasons :class:`Flux1Defaults` gives: no
    ``.Lora`` overlay (the PAI 2603 8-step distillation rides the deploy
    binding as an adapter, and while its 8 steps and weight 0.8 ARE sourced
    (``:244``, README), no shipped code states a STRENGTH RANGE or a
    ``SchedulerName`` demand to source — a Knob needs a range, and inventing
    one is the thing this vocabulary refuses to do); no ``timesteps`` ladder
    (flow-matching derives its sigma ladder from the shift parameters, and no
    z-image endpoint pins one); no aspect/megapixel grid, which is endpoint
    PAYLOAD vocabulary.
    """

    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=0.0, hi=15.0, name="guidance")
    #: The base is a CFG-mechanism model and z-image batches CFG into ONE
    #: forward — ``latents.repeat(2,1,1,1)`` plus a list concatenation of the
    #: prompt embeds, so the pytree arity doubles (``:179-184``). Both turbo
    #: rows set False.
    cfg: bool = True
    #: The base handler declares ``distilled=False`` (``:580``). The official
    #: Decoupled-DMD Turbo checkpoint's row sets True — its card recipe is 9
    #: scheduler steps (``_TURBO_DMD_STEPS``, ``:245``), and the PAI 8-step
    #: adapter (``:244``) reaches the same state as an overlay.
    step_distilled: bool = False
    #: ``_MAX_SEQUENCE_LENGTH`` (``:131``). A plain int for the same reason as
    #: :class:`QwenImageDefaults` — no endpoint exposes it on the wire.
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


class Qwen36Mtp(ModelType[QwenMtpDefaults]):
    """Qwen3.6-27B-MTP — an autoregressive LLM served as GGUF by llama.cpp.

    NAME + FINGERPRINT SEAM ONLY, no canonical lane, and unlike :class:`Rife`
    this one can never acquire one. A serve-lane document's top-level
    ``dtype`` is MANDATORY (pgw#1391) and is defined as the serve-side load
    dtype in TORCH SPELLING; GGUF block-quant containers (``UD-Q4_K_XL``,
    ``IQ4_NL``, …) have no torch spelling at all, which
    ``serving/streaming/engine.py`` states as a live refusal: "Sub-byte and
    block-quantized containers are the external-binary class (the #1303
    ladder's tier 3), not the serving pytorch path". The upstream repo also
    ships no ``config.json`` and no ``model_index.json`` — there is no
    safetensors header to derive a document FROM.

    Do not read that as "tensorfs cannot handle GGUF": it ships a real
    ``gguf-v1`` PLANNER profile and chunks these files properly. Storage and
    serve-lane description are different layers, and only the second one is
    closed here.

    ``canonical_scheduler_config`` is ``{}`` by DEFINITION, not by
    unreachability: the field exists so ingest can synthesize a missing
    ``scheduler/scheduler_config.json`` for a diffusers pipeline
    constructor's scheduler invariant. An autoregressive decoder has no noise
    schedule and no such invariant.
    """

    name = "qwen3.6-27b-mtp"
    contracts = ("qwen3.6-27b-mtp.*",)
    Defaults = QwenMtpDefaults


class Qwen36A3b(ModelType[QwenA3bDefaults]):
    """Qwen3.6-35B-A3B — a hybrid linear-attention/full-attention MoE
    (``Qwen3_5MoeForConditionalGeneration``, ``full_attention_interval: 4``),
    served fp8 by vLLM.

    NAME + FINGERPRINT SEAM ONLY — but for a NARROWER reason than
    :class:`Qwen36Mtp`, and the distinction matters because a later lane
    could legitimately author the document this one does not.

    The packaging is mixed, MEASURED by ranged header reads rather than
    assumed from the repo's advertised dtype: ``outside.safetensors`` is
    336/336 BF16, and every layer shard is ~784 BF16 beside ~774 F8_E4M3
    (``config.json``'s ``quantization_config.modules_to_not_convert``
    corroborates it — the whole vision tower is excluded from conversion).
    **That is expressible and is NOT a blocker**: the top-level ``dtype`` is
    the LANE's load dtype while per-tensor ``dtypes`` carries the matcher's
    constraint, and ``sdxl.diffusers-fp8-rowwise@1`` already ships exactly
    this shape — a flat ``float8_e4m3fn`` lane over 257 per-tensor
    declarations of which many are ``["BF16"]`` (both text encoders are
    entirely BF16 inside an "fp8" document).

    What actually leaves this type lane-less is the SERVE PATH: vLLM
    self-loads a directory, so ``ctx.load`` is never called, the endpoint
    declares ``lanes=()`` (eager-permanent — the tier :class:`MiniMaxH3`'s
    docstring names, "EXTERNAL-BINARY runtimes only"), and ``ctx.lane`` would
    raise. A lane document would therefore have no consumer on the path this
    family is actually served by, and publishing one invites a future author
    to point ``lanes={document: …}`` at it — asserting a pytorch streaming
    load that this endpoint does not perform. Absent is the honest state, not
    a placeholder for missing work.

    A3B names the ACTIVE parameter count, not the resident one: 8 of 256
    experts fire per token, but all 256 stay resident — 37.5 GB, not ~3 GB.
    That is a placement fact for the endpoint's own floor, recorded here only
    so the name cannot mislead a reader of this file.
    """

    name = "qwen3.6-35b-a3b"
    contracts = ("qwen3.6-35b-a3b.*",)
    Defaults = QwenA3bDefaults


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
    canonical_contract = FLUX1_DIFFUSERS_BF16
    Defaults = Flux1Defaults


class Flux2Klein(ModelType[Flux2KleinDefaults]):
    """FLUX.2 Klein — 4B and 9B, Base and Turbo, under one vocabulary root
    (see :class:`Flux2KleinDefaults` for the architecture evidence separating
    it from :class:`Flux1`, and for why the 4B/9B split is a ``requires=``
    fact rather than a type)."""

    name = "flux2-klein"
    contracts = ("flux2-klein.*",)
    canonical_contract = FLUX2_KLEIN_DIFFUSERS_BF16
    canonical_scheduler_config = FLUX2_KLEIN_SCHEDULER_CONFIG
    Defaults = Flux2KleinDefaults


class Krea2(ModelType[Krea2Defaults]):
    """Krea 2 (Raw + TDM Turbo).

    NO ``canonical_contract``, and uniquely in se#769 the reason is not merely
    that tensorfs ships no ``krea-2.*`` document yet: **the checkpoints are not
    in the hub at all.** krea-2 is ``wave: blocked`` on ie#632
    (``e2e/manifests/fleet.yaml:497-505``) for a MIRROR defect —
    ``tensorhub/krea-2-raw``'s mirrored ``model_index`` declares
    ``text_encoder_select_layers`` that upstream ``krea/Krea-2-Raw`` never
    carried (th#1675 class). The recorded verdict is *"fix the mirror, not the
    endpoint"*, and the deploy bindings are deliberately kept live for a
    one-command redeploy (``e2e/manifests/bindings.yaml:295-301``).

    So there is no header to derive a document FROM. Both available shortcuts
    are wrong: deriving one from the defective mirror bakes in the defect, and
    deriving one from upstream bytes documents a packaging the fleet does not
    serve. The document waits on the mirror fix. Declaring a contract now would
    be the standing lie the ``MissingContract`` sentinel was retired for.

    The endpoint is NOT a corpse — it is a deploy-blocked live product with a
    quality probe (ie#531) and a production incident (a 24 GB card broke 0.3.3
    with ``model_load_failure_streak``, 2026-08-01) behind it.
    """

    name = "krea-2"
    contracts = ("krea-2.*",)
    Defaults = Krea2Defaults


class Anima(ModelType[AnimaDefaults]):
    """Anima — Cosmos-Predict2-2B backbone, DiffSynth-native split checkpoint.

    NO ``canonical_contract`` yet: the ``anima.*`` lane document is owed and its
    header evidence is already derived (685 flat-BF16 tensors in
    ``split_files/diffusion_models/anima-base-v1.0.safetensors``). It lands in
    the same commit as the document.

    ⚠️ FOR WHOEVER AUTHORS THAT DOCUMENT: every anima DiT tensor is ``net.*``
    **on disk**. The endpoint STRIPS that prefix at construction
    (``anima/src/anima/main.py:281-285``) and bridges it for artifact swaps
    (``_cozy_w8a8_key_map``, ``:223-224``), so every reference in handler code
    shows the STRIPPED name. tensorfs matches the safetensors HEADER, so the
    document must spell ``net.blocks.*``; one written from the pipeline code
    matches ZERO tensors and passes review. Author tensor names from a header
    dump only — handler code is authoritative for ``Defaults`` VALUES, never for
    tensor NAMES.

    MEASURED, as a proof rather than a claim (the flux1 lane's form: "I authored
    from headers" is a claim; "my intersection check would zero if I hadn't" is
    a proof) — against the real header of
    ``tensorhub/anima@prod`` checkpoint ``sha256:9474851b0309…``::

        header names   n real header = 685/685
        stripped names n real header =   0/685

    Exclusions, all measured on that same header:

    * the VAE — anima ships the Qwen-Image VAE
      (``split_files/vae/qwen_image_vae.safetensors``, 194 tensors), and a shared
      VAE makes family detection TIE (the tensorfs#122 regression). ``DiT n VAE
      = 0``, so the transformer-only document is cleanly separable.
    * the text encoder — Qwen3-0.6B (310 tensors); ``DiT n TE = 0``.
    * NO T5 TOWER IS EXPOSED, checked because ``net.llm_adapter.embed.weight`` is
      ``[32128, 1024]`` and 32128 IS the T5 vocab size, which makes this look
      like a T5 declaration and is exactly the shape the never-declare rule
      targets. It is not one: the 118 ``llm_adapter`` tensors are anima's OWN
      in-model adapter grammar (``net.llm_adapter.blocks.N.self_attn.q_proj``),
      and a T5-grammar probe (``SelfAttention``/``EncDecAttention``/
      ``DenseReluDense``/``relative_attention_bias``/``encoder.block.``) matches
      **0 of 685**. The adapter merely CONSUMES T5 token ids — the endpoint uses
      T5-XXL as a tokenizer only (``anima/src/anima/main.py:5-8``). Safe to
      declare; the vocab-size coincidence is not a shared tower.
    """

    name = "anima"
    contracts = ("anima.*",)
    Defaults = AnimaDefaults


class Ernie(ModelType[ErnieDefaults]):
    """Baidu ERNIE-Image (base + step-distilled Turbo).

    NO ``canonical_contract`` yet: the ``ernie.*`` lane document is owed and its
    header evidence is derived (409 flat-BF16 tensors over 8 transformer
    shards, ``layers.{0..35}``). It lands in the same commit as the document.

    ⚠️ FOR WHOEVER AUTHORS THAT DOCUMENT: ERNIE is **split-QKV**
    (``layers.0.self_attention.to_q/to_k/to_v.weight``, each ``[4096, 4096]``,
    confirmed against ``transformer/config.json``: ``hidden_size 4096``,
    ``num_layers 36``, ``num_attention_heads 32``). It therefore matches ZERO of
    :data:`DIT_BLOCKS_FUSED_QKV` — the fragment that silently captured
    :class:`Flux1`. Do not reach for it because it is "close".

    Cover the TRANSFORMER ONLY. Measured: ``tensorhub/ernie-image`` and
    ``ernie-image-turbo`` are digest-identical on the VAE, all four
    ``text_encoder`` shards and all four ``pe`` shards, and differ on all eight
    transformer shards — so a document covering the VAE or TE would tie the two
    checkpoints to each other. The VAE also carries an ``I64`` tensor among 250
    BF16, so a flat-bf16 document over it would refuse the fleet's own
    checkpoint.
    """

    name = "ernie"
    contracts = ("ernie.*",)
    canonical_scheduler_config = ERNIE_SCHEDULER_CONFIG
    Defaults = ErnieDefaults
class QwenImage(ModelType[QwenImageDefaults]):
    """Qwen-Image — text-to-image AND native editing (Qwen-Image-Edit-2511)
    under one vocabulary root.

    Not a judgement call: the two served checkpoints' transformers are
    byte-layout identical, and the endpoint registers ONE family for both arms.
    See :class:`QwenImageDefaults` for the measurement and for why the
    endpoint's two ``@endpoint`` classes are a COMPILE fact rather than a
    vocabulary one.
    """

    name = "qwen-image"
    contracts = ("qwen-image.*",)
    canonical_contract = QWEN_IMAGE_DIFFUSERS_BF16
    canonical_scheduler_config = QWEN_IMAGE_SCHEDULER_CONFIG
    Defaults = QwenImageDefaults


class ZImage(ModelType[ZImageDefaults]):
    """Z-Image — the undistilled base and the Decoupled-DMD Turbo under one
    vocabulary root (see :class:`ZImageDefaults`)."""

    name = "z-image"
    contracts = ("z-image.*",)
    canonical_contract = Z_IMAGE_DIFFUSERS_BF16
    canonical_scheduler_config = Z_IMAGE_SCHEDULER_CONFIG
    Defaults = ZImageDefaults


MODEL_TYPES: Final[tuple[type[ModelType[msgspec.Struct]], ...]] = (
    SDXL,
    SD15,
    SD2,
    HiDreamO1,
    Wan22,
    MiniMaxH3,
    Rife,
    Qwen36Mtp,
    Qwen36A3b,
    Flux1,
    Flux2Klein,
    Krea2,
    Anima,
    Ernie,
    QwenImage,
    ZImage,
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
        Qwen36Mtp.name: Qwen36Mtp.Defaults,
        Qwen36A3b.name: Qwen36A3b.Defaults,
        Flux1.name: Flux1.Defaults,
        Flux2Klein.name: Flux2Klein.Defaults,
        QwenImage.name: QwenImage.Defaults,
        ZImage.name: ZImage.Defaults,
        Krea2.name: Krea2.Defaults,
        Anima.name: Anima.Defaults,
        Ernie.name: Ernie.Defaults,
        SDXL.Lora.name: SDXL.Lora.Defaults,
        SD15.Lora.name: SD15.Lora.Defaults,
    }



# ── the lane-dtype side-channel is DELETED (pgw#1391) ───────────────────────
#
# `CONTRACT_DTYPES` was a module-level MUTABLE dict and `register_contract_dtype()`
# wrote to it, so the serve dtype was whatever code happened to register —
# structurally disconnected from what the tensorfs document declares. Measured
# at master `a9bec13e`: `sdxl.diffusers-bf16@1` answered `torch.bfloat16` only
# because a seed function seeded it, while `minimax.h3-dit-diffusers@1` answered
# `None` even though its document had declared `bfloat16` since tensorfs#121.
# Had the registration and the document ever disagreed, the registration would
# have won and no gate would have noticed.
#
# That is the same defect as the `PendingContract` lie one layer down: a fact
# asserted BESIDE the real source instead of read FROM it. A lane's dtype now
# comes from `Contract.dtype`, which reads the document, and from nowhere else.
# There is deliberately no replacement hook: a caller wanting to register a
# dtype for a document that declares none is treating a FRAGMENT as a serve
# lane, which is the bug rather than a reason to keep the dict.


__all__ = [
    "DIT_BLOCKS_FUSED_QKV",
    "FLUX1_DIFFUSERS_BF16",
    "FLUX2_KLEIN_DIFFUSERS_BF16",
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
    "Qwen36A3b",
    "Qwen36Mtp",
    "QwenA3bDefaults",
    "QwenMtpDefaults",
    "QwenTextGenDefaults",
    "Rife",
    "RifeDefaults",
    "MODEL_TYPES",
    "QWEN_IMAGE_DIFFUSERS_BF16",
    "QWEN_IMAGE_SCHEDULER_CONFIG",
    "QwenImage",
    "QwenImageDefaults",
    "ModelType",
    "MissingContract",
    "MissingContractError",
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
    "Z_IMAGE_DIFFUSERS_BF16",
    "Z_IMAGE_SCHEDULER_CONFIG",
    "ZImage",
    "ZImageDefaults",
    "defaults_vocabularies",
    "model_type_by_name",
    "model_type_for_contract",
]
