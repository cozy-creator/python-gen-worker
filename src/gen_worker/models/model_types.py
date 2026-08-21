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
    """The caller-visibility seam ``Knob.resolve`` clamps through."""

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

    default: Number
    lo: Number | None = None
    hi: Number | None = None
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
        return type(self.default)(applied)


# ── The tensor-layout seam, after v2 (pgw#1621) ──────────────────────────────
#
# THE CONTRACT-OBJECT CONSTANTS ARE GONE, and so is the machinery around them:
# `TensorLayoutContract`, `MissingContract`, `MissingContractError`, `_library`
# and the fifteen `NAME: Final = _library(...)` rows.
#
# They existed because a v1 lane was an IMPORTED OBJECT (Paul's
# contract-objects ruling, 2026-08-18) resolved against a per-lane document
# library, and because that library did not carry a document for every lane the
# fleet wanted to declare — so `MissingContract` was the shape that let a
# `ModelType` exist before its document did, refusing loudly on any read
# instead of answering a `stamp` for a name nobody had authored.
#
# v2 removes the problem the sentinel solved rather than the sentinel alone.
# There is no per-lane document any more: a lane is `(topology, quant)`, the
# eight quant rules cover the whole fleet, and a topology is EXTRACTED
# MECHANICALLY from a reference checkpoint's headers. So "this lane's document
# has not been authored yet" is no longer a state a model type can be in — the
# topology either was extracted from real bytes or it does not exist, and
# `parse_lane_stamp` refuses the second case at class definition, naming the
# corpus. A sentinel that stood in for an unauthored document has nothing left
# to stand in for.
#
# What remains here is the INGEST fingerprint below (`ModelType.contracts`),
# which maps a recorded stamp to a model name and never gates anything.
#
# SEVERAL `ModelType` DOCSTRINGS BELOW STILL NAME THE SENTINEL, AND THAT IS
# DELIBERATE. They are the ratified reasoning for why a family declares no lane
# at all — MusicGen, Hunyuan3d, Rife, Ltx2Upsampler, Krea2 — and the reasoning
# outlives the class: *when a family genuinely has no lane, declare NOTHING
# rather than a placeholder, because a placeholder's own refusal text instructs
# the reader to author a document that will never exist, which is a standing lie
# with a to-do attached.* Under v2 that argument is stronger, not weaker: there
# is no per-lane document left to be late, so "absent" is the only honest state
# and the sentinel has nothing left to stand in for. Read those mentions as
# history, not as a live type.

SDXL_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "EulerDiscreteScheduler",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "steps_offset": 1,
    "timestep_spacing": "trailing",
    "interpolation_type": "linear",
    "trained_betas": None,
    "use_karras_sigmas": False,
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "clip_sample": False,
}

SDXL_AYS_10: Final[tuple[int, ...]] = (999, 845, 730, 587, 443, 310, 193, 116, 53, 13)

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

LTX2_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "base_shift": 0.95,
    "max_shift": 2.05,
    "use_dynamic_shifting": False,
    "base_image_seq_len": 1024,
    "max_image_seq_len": 4096,
    "time_shift_type": "exponential",
    "shift_terminal": None,
    "invert_sigmas": False,
    "stochastic_sampling": False,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

INTERNVL_U_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "DPMSolverMultistepScheduler",
    "algorithm_type": "dpmsolver++",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "dynamic_thresholding_ratio": 0.995,
    "euler_at_final": False,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lambda_min_clipped": float("-inf"),
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_type": "midpoint",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
    "use_lu_lambdas": False,
    "variance_type": None,
}


STABLE_AUDIO_SCHEDULER_CONFIG: Final[Mapping[str, object]] = {
    "_class_name": "CosineDPMSolverMultistepScheduler",
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "sigma_data": 1.0,
    "sigma_max": 500,
    "sigma_min": 0.3,
    "sigma_schedule": "exponential",
    "solver_order": 2,
    "solver_type": "midpoint",
    "final_sigmas_type": "zero",
    "euler_at_final": False,
    "lower_order_final": True,
    "rho": 7.0,
}


D_co = TypeVar("D_co", bound=msgspec.Struct, covariant=True)


class ModelType(Generic[D_co]):
    """A serving-defaults vocabulary: name + ``Defaults`` + ingest fingerprint."""

    name: ClassVar[str] = ""
    contracts: ClassVar[tuple[str, ...]] = ()
    canonical_scheduler_config: ClassVar[Mapping[str, object]] = {}

    def __init__(self) -> None:
        raise TypeError(f"{type(self).__name__} is a vocabulary, not a value")


class LoraOverlay:
    """Base of the nested adapter-of-base overlay types (``SDXL.Lora``)."""

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


class SdxlConfig(msgspec.Struct, frozen=True):
    """The SERVING CONFIG — the axes both SDXL defaults types share, as ONE nominal type ("Defaults are a config plus extras": the endpoint annotates ``config: SDXL.Config``, never a union)."""

    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(6.0, lo=1.5, hi=15.0, name="guidance")
    cfg: bool = True
    timesteps: tuple[int, ...] = ()
    guidance_rescale: Knob[float] = Knob(
        0.6, lo=0.0, hi=1.0, name="guidance_rescale")


class SdxlDefaults(SdxlConfig, frozen=True):
    """The checkpoint row: the config plus the prompt vocabulary."""

    positive_preamble: str = "masterpiece, best quality"
    negative_preamble: str = "worst quality, low quality"
    step_distilled: bool = False
    enable_freeu: bool = False
    freeu_b1: float = 1.1
    freeu_b2: float = 1.2
    freeu_s1: float = 0.6
    freeu_s2: float = 0.4


class SdxlLoraDefaults(SdxlConfig, frozen=True):
    """The adapter row: the config (platform overrides = Lightning-4-step-ish servable trace fixture: cfg off, euler_trailing, 4 steps) plus the adapter's own facts."""

    steps: Knob[int] = Knob(4, lo=1, hi=150, name="steps")
    cfg: bool = False
    scheduler: SchedulerName | None = "euler_trailing"
    strength: Knob[float] = Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    trigger_words: tuple[str, ...] = ()
    distillation: bool = False


class Sd15Config(msgspec.Struct, frozen=True):
    """SD1.x serving config — same shared axes and semantics as :class:`SdxlConfig` (scheduler is adapter-only, tree-only for checkpoints)."""

    steps: Knob[int] = Knob(30, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(7.0, lo=0.0, name="guidance")
    cfg: bool = True
    timesteps: tuple[int, ...] = ()


class Sd15Defaults(Sd15Config, frozen=True):
    """The checkpoint row — the config alone (no ruled SD1.x prompt vocabulary; fields stay additive if one is ruled)."""


class Sd15LoraDefaults(Sd15Config, frozen=True):
    """The adapter row."""

    steps: Knob[int] = Knob(4, lo=1, hi=80, name="steps")
    cfg: bool = False
    scheduler: SchedulerName | None = "lcm"
    strength: Knob[float] = Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    trigger_words: tuple[str, ...] = ()
    distillation: bool = False


class Sd2Defaults(msgspec.Struct, frozen=True):
    """Values from the live sd2.schema.json registry entry: steps 1 (bounds 1..4 — SD-Turbo is a 1-4 step ADD distillation), guidance 0.0 (the distilled lane pins CFG off)."""

    steps: Knob[int] = Knob(1, lo=1, hi=4, name="steps")
    guidance: Knob[float] = Knob(0.0, lo=0.0, name="guidance")


class HiDreamO1Defaults(msgspec.Struct, frozen=True):
    """Values from the live hidream-o1.schema.json registry entry: steps 28 (bounds 1..150), guidance 1.0 (its ``cfg_scale`` default — the default ``dev`` variant is guidance-distilled; schema minimum 0)."""

    steps: Knob[int] = Knob(28, lo=1, hi=150, name="steps")
    guidance: Knob[float] = Knob(1.0, lo=0.0, name="guidance")


class Wan22Defaults(msgspec.Struct, frozen=True):
    """Values from the live wan22.schema.json registry entry: steps 40 (bounds 1..80), guidance 4.0 and ``guidance_2`` 3.0 (the A14B expert pair's high/low-noise guidance; schema minimum 0)."""

    steps: Knob[int] = Knob(40, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=0.0, name="guidance")
    guidance_2: Knob[float] = Knob(3.0, lo=0.0, name="guidance_2")


class MiniMaxH3Defaults(msgspec.Struct, frozen=True):
    """MiniMax-H3 (joint video+audio DiT)."""

    video_shift: float = 12.0
    audio_shift: float = 3.0


class RifeDefaults(msgspec.Struct, frozen=True):
    """RIFE has no serving knobs — the interpolator takes source/target fps from the delivery preset, nothing per-checkpoint."""


class QwenTextGenDefaults(msgspec.Struct, frozen=True):
    """The AUTOREGRESSIVE sampling vocabulary — the first non-diffusion ``Defaults`` in this module, and the axes both qwen3.6 roots share as ONE nominal type (the ``SdxlConfig`` idiom)."""

    max_tokens: Knob[int] = Knob(256, lo=1, name="max_tokens")
    temperature: Knob[float] = Knob(0.7, name="temperature")
    top_p: Knob[float] = Knob(0.95, name="top_p")


class QwenMtpDefaults(QwenTextGenDefaults, frozen=True):
    """Qwen3.6-27B-MTP, served as GGUF through llama.cpp with MTP speculative decoding."""

    temperature: Knob[float] = Knob(0.6, name="temperature")


class QwenA3bDefaults(QwenTextGenDefaults, frozen=True):
    """Qwen3.6-35B-A3B, served fp8 through vLLM."""


class InternVLUDefaults(msgspec.Struct, frozen=True):
    """InternVL-U — OpenGVLab's unified VLM, serving TEXT-TO-IMAGE here."""

    steps: Knob[int] = Knob(20, lo=1, hi=50, name="steps")


class Flux1Defaults(msgspec.Struct, frozen=True):
    """FLUX.1 — BFL's rectified-flow DiT (dev, schnell, and the Flex.2-preview redistill)."""

    steps: Knob[int] = Knob(28, lo=1, hi=100, name="steps")
    guidance: Knob[float] = Knob(3.5, lo=0.0, hi=10.0, name="guidance")
    cfg: bool = False
    step_distilled: bool = False
    max_sequence_length: int = 512


class Flux2KleinDefaults(msgspec.Struct, frozen=True):
    """FLUX.2 Klein — 4B and 9B under ONE vocabulary root (their endpoint modules diff to nothing but the type name and one VRAM floor, ``vram30g`` vs ``vram44g``; a per-lane VRAM floor is a ``requires=``..."""

    steps: Knob[int] = Knob(28, lo=1, hi=50, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=1.0, hi=10.0, name="guidance")
    cfg: bool = True
    step_distilled: bool = False
    max_sequence_length: int = 512


class Krea2Defaults(msgspec.Struct, frozen=True):
    """Krea 2 — a 12.9B single-stream MMDiT (rectified flow) with a Qwen3-VL-4B text encoder and the Qwen-Image f8 VAE, served as TWO checkpoints under one vocabulary: the undistilled Raw mirror and the T..."""

    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(3.5, lo=0.0, hi=10.0, name="guidance")
    cfg: bool = True
    step_distilled: bool = False
    max_sequence_length: int = 512


class AnimaDefaults(msgspec.Struct, frozen=True):
    """Anima (``circlestone-labs/Anima``) — an anime-focused ~2B text-to-image DiT on NVIDIA's Cosmos-Predict2-2B backbone."""

    steps: Knob[int] = Knob(35, lo=1, hi=50, name="steps")
    guidance: Knob[float] = Knob(4.5, lo=1.0, hi=10.0, name="guidance")
    negative: str = ""
    cfg: bool = True
    step_distilled: bool = False


class ErnieDefaults(msgspec.Struct, frozen=True):
    """Baidu ERNIE-Image — an 8B single-stream DiT (Apache 2.0) served through diffusers' ``ErnieImagePipeline`` as a full checkpoint plus a step-distilled Turbo."""

    steps: Knob[int] = Knob(28, lo=1, hi=100, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=1.0, hi=15.0, name="guidance")
    negative: str = ""
    cfg: bool = True
    step_distilled: bool = False
    max_sequence_length: int = 512


class QwenImageDefaults(msgspec.Struct, frozen=True):
    """Qwen-Image — text-to-image AND native editing under ONE vocabulary root."""

    steps: Knob[int] = Knob(30, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=1.0, hi=12.0, name="guidance")
    cfg: bool = True
    step_distilled: bool = False
    negative: str = " "
    max_sequence_length: int = 512


class ZImageDefaults(msgspec.Struct, frozen=True):
    """Z-Image — the undistilled base and the Decoupled-DMD Turbo distillation under ONE vocabulary root, the same one-root shape as :class:`Wan22`."""

    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    guidance: Knob[float] = Knob(4.0, lo=0.0, hi=15.0, name="guidance")
    cfg: bool = True
    step_distilled: bool = False
    max_sequence_length: int = 512
class StableAudioDefaults(msgspec.Struct, frozen=True):
    """Stable Audio Open 1.0 — Stability's latent-diffusion text-to-audio DiT, and every fine-tune of it."""

    steps: Knob[int] = Knob(100, lo=1, hi=250, name="steps")
    guidance: Knob[float] = Knob(7.0, lo=1.5, hi=15.0, name="guidance")
    cfg: bool = True
    negative: str = ""


class MusicGenDefaults(msgspec.Struct, frozen=True):
    """Meta MusicGen — a T5 encoder + AUTOREGRESSIVE transformer decoder emitting EnCodec tokens (``MusicgenForConditionalGeneration``)."""

    guidance: Knob[float] = Knob(3.0, lo=1.0, hi=10.0, name="guidance")
    temperature: Knob[float] = Knob(1.0, lo=0.0, hi=2.0, name="temperature")
    cfg: bool = True
class Trellis2Defaults(msgspec.Struct, frozen=True):
    """TRELLIS.2 — Microsoft's image-to-3D sparse-structure + SLAT flow stack."""

    steps: Knob[int] = Knob(12, lo=1, hi=50, name="steps")
    cfg: bool = True


class Hunyuan3dDefaults(msgspec.Struct, frozen=True):
    """Hunyuan3D-2.1 — Tencent's image-to-3D shape DiT plus PBR texture paint."""

    num_shape_steps: Knob[int] = Knob(50, lo=1, hi=100, name="num_shape_steps")
    guidance_scale: Knob[float] = Knob(5.0, lo=1.0, hi=15.0, name="guidance_scale")


class Ltx2Defaults(msgspec.Struct, frozen=True):
    """LTX-2 (Lightricks LTX-2.3) — a 22B JOINT audio+video DiT that denoises both modalities in one pass."""

    guidance: float = 1.0
    audio_guidance: float = 1.0
    sigmas: tuple[float, ...] = (
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
    )
    stage2_sigmas: tuple[float, ...] = (0.909375, 0.725, 0.421875)
    cfg: bool = False
    max_sequence_length: int = 1024


class Ltx2UpsamplerDefaults(msgspec.Struct, frozen=True):
    """The LTX-2 2x spatial LATENT upsampler has no serving knobs — it takes its scale from the checkpoint's own ``rational_spatial_scale`` and everything else from the two-stage recipe on the DiT beside it."""


class JoyCaptionDefaults(msgspec.Struct, frozen=True):
    """JoyCaption — a LLaVA image captioner (``LlavaForConditionalGeneration`` + ``LlavaProcessor`` out of ONE mirror snapshot)."""

    max_new_tokens: Knob[int] = Knob(512, lo=1, hi=2048, name="max_new_tokens")
    temperature: Knob[float] = Knob(0.6, name="temperature")
    top_p: Knob[float] = Knob(0.9, name="top_p")


class SDXL(ModelType[SdxlDefaults]):
    """Stable Diffusion XL."""

    name = "sdxl"
    contracts = ("sdxl.*", "sdxl-inpainting.*")
    canonical_scheduler_config = SDXL_SCHEDULER_CONFIG
    Config: TypeAlias = SdxlConfig
    Defaults = SdxlDefaults

    class Lora(LoraOverlay):
        name = "sdxl.lora"
        Defaults = SdxlLoraDefaults


class SD15(ModelType[Sd15Defaults]):
    """Stable Diffusion 1.x — one type covers every SD1.x fine-tune."""

    name = "sd15"
    contracts = ("sd15.*",)
    canonical_scheduler_config = SD15_SCHEDULER_CONFIG
    Config: TypeAlias = Sd15Config
    Defaults = Sd15Defaults

    class Lora(LoraOverlay):
        name = "sd15.lora"
        Defaults = Sd15LoraDefaults


class SD2(ModelType[Sd2Defaults]):
    """Stable Diffusion 2.x / SD-Turbo — its own root, so a 1-step Turbo recipe can never validate against the SD1.x vocabulary."""

    name = "sd2"
    contracts = ("sd2.*",)
    Defaults = Sd2Defaults


class HiDreamO1(ModelType[HiDreamO1Defaults]):
    """HiDream-O1."""

    name = "hidream-o1"
    contracts = ("hidream-o1.*",)
    Defaults = HiDreamO1Defaults


class Wan22(ModelType[Wan22Defaults]):
    """Wan 2.2 — T2V/I2V A14B + TI2V-5B serve under one vocabulary root, as today's registry does."""

    name = "wan22"
    contracts = ("wan22.*",)
    Defaults = Wan22Defaults


class MiniMaxH3(ModelType[MiniMaxH3Defaults]):

    name = "minimax-h3"
    contracts = ("minimax-h3.*",)
    Defaults = MiniMaxH3Defaults


class InternVLU(ModelType[InternVLUDefaults]):
    """InternVL-U — the unified VLM, serving text-to-image."""

    name = "internvl-u"
    contracts = ("internvl-u.*",)
    canonical_scheduler_config = INTERNVL_U_SCHEDULER_CONFIG
    Defaults = InternVLUDefaults


class Rife(ModelType[RifeDefaults]):
    """RIFE v4.25 frame interpolation — a small AUXILIARY model with its own checkpoint, bound beside a video model to serve the fps>24 delivery presets."""

    name = "rife"
    contracts = ("rife.*",)
    Defaults = RifeDefaults


class Qwen36Mtp(ModelType[QwenMtpDefaults]):
    """Qwen3.6-27B-MTP — an autoregressive LLM served as GGUF by llama.cpp."""

    name = "qwen3.6-27b-mtp"
    contracts = ("qwen3.6-27b-mtp.*",)
    Defaults = QwenMtpDefaults


class Qwen36A3b(ModelType[QwenA3bDefaults]):
    """Qwen3.6-35B-A3B — a hybrid linear-attention/full-attention MoE (``Qwen3_5MoeForConditionalGeneration``, ``full_attention_interval: 4``), served fp8 by vLLM."""

    name = "qwen3.6-35b-a3b"
    contracts = ("qwen3-6-35b-a3b.*",)
    Defaults = QwenA3bDefaults


class Flux1(ModelType[Flux1Defaults]):
    """FLUX.1 — dev, schnell and the Flex.2-preview redistill under ONE root."""

    name = "flux1"
    contracts = ("flux1.*",)
    Defaults = Flux1Defaults


class Flux2Klein(ModelType[Flux2KleinDefaults]):
    """FLUX.2 Klein — 4B and 9B, Base and Turbo, under one vocabulary root (see :class:`Flux2KleinDefaults` for the architecture evidence separating it from :class:`Flux1`, and for why the 4B/9B split is ..."""

    name = "flux2-klein"
    contracts = ("flux2-klein.*",)
    canonical_scheduler_config = FLUX2_KLEIN_SCHEDULER_CONFIG
    Defaults = Flux2KleinDefaults


class Krea2(ModelType[Krea2Defaults]):
    """Krea 2 (Raw + TDM Turbo)."""

    name = "krea-2"
    contracts = ("krea-2.*",)
    Defaults = Krea2Defaults


class Anima(ModelType[AnimaDefaults]):

    name = "anima"
    contracts = ("anima.*",)
    Defaults = AnimaDefaults


class Ernie(ModelType[ErnieDefaults]):

    name = "ernie"
    contracts = ("ernie.*",)
    canonical_scheduler_config = ERNIE_SCHEDULER_CONFIG
    Defaults = ErnieDefaults
class QwenImage(ModelType[QwenImageDefaults]):
    """Qwen-Image — text-to-image AND native editing (Qwen-Image-Edit-2511) under one vocabulary root."""

    name = "qwen-image"
    contracts = ("qwen-image.*",)
    canonical_scheduler_config = QWEN_IMAGE_SCHEDULER_CONFIG
    Defaults = QwenImageDefaults


class ZImage(ModelType[ZImageDefaults]):
    """Z-Image — the undistilled base and the Decoupled-DMD Turbo under one vocabulary root (see :class:`ZImageDefaults`)."""

    name = "z-image"
    contracts = ("z-image.*",)
    canonical_scheduler_config = Z_IMAGE_SCHEDULER_CONFIG
    Defaults = ZImageDefaults
class StableAudio(ModelType[StableAudioDefaults]):
    """Stable Audio Open 1.0 and its fine-tunes — stable-audio-open and foundation-1 under ONE root (identical transformer header sha256, differing weight digests). The fingerprint is stable-audio.* and must NEVER be flux*: a bare flux.*/flux2.* pattern can match this audio checkpoint's tree."""

    name = "stable-audio"
    contracts = ("stable-audio.*",)
    canonical_scheduler_config = STABLE_AUDIO_SCHEDULER_CONFIG
    Defaults = StableAudioDefaults


class MusicGen(ModelType[MusicGenDefaults]):
    """Meta MusicGen text-to-music — name + fingerprint + ``Defaults`` only."""

    name = "musicgen"
    contracts = ("musicgen.*",)
    Defaults = MusicGenDefaults
class Trellis2(ModelType[Trellis2Defaults]):
    """TRELLIS.2 — the sparse-structure and both SLAT flow DiTs under one vocabulary root (see :class:`Trellis2Defaults` for the header evidence that all five checkpoints are one architecture, and for why..."""

    name = "trellis2"
    contracts = ("trellis2.*",)
    Defaults = Trellis2Defaults


class Hunyuan3d(ModelType[Hunyuan3dDefaults]):
    """Hunyuan3D-2.1 — image-to-3D shape DiT plus PBR texture paint."""

    name = "hunyuan3d"
    contracts = ("hunyuan3d.*",)
    Defaults = Hunyuan3dDefaults


class Ltx2(ModelType[Ltx2Defaults]):
    """LTX-2 — the joint audio+video DiT ``ltx-video-2.3`` serves."""

    name = "ltx-2"
    contracts = ("ltx2.*",)
    canonical_scheduler_config = LTX2_SCHEDULER_CONFIG
    Defaults = Ltx2Defaults


class Ltx2Upsampler(ModelType[Ltx2UpsamplerDefaults]):
    """The LTX-2 2x spatial latent upsampler — a small AUXILIARY model with its own checkpoint (``tensorhub/ltx-2.3-upsampler``), bound beside the DiT to serve the two-stage 1080p/4K recipe."""

    name = "ltx-2-upsampler"
    contracts = ("ltx2-upsampler.*",)
    Defaults = Ltx2UpsamplerDefaults


class JoyCaption(ModelType[JoyCaptionDefaults]):
    """JoyCaption — the LLaVA image captioner ``joycaption`` serves."""

    name = "joycaption"
    contracts = ("joycaption.*",)
    Defaults = JoyCaptionDefaults


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
    StableAudio,
    MusicGen,
    Ltx2,
    Ltx2Upsampler,
    InternVLU,
    Trellis2,
    Hunyuan3d,
    JoyCaption,
)

LORA_OVERLAYS: Final[tuple[type[LoraOverlay], ...]] = (SDXL.Lora, SD15.Lora)


def model_type_by_name(name: str) -> type[ModelType[msgspec.Struct]] | None:
    """The recognized-name lookup (hub ``model`` column values)."""
    for mt in MODEL_TYPES:
        if mt.name == name:
            return mt
    return None


def model_type_for_contract(stamp: str) -> type[ModelType[msgspec.Struct]] | None:
    """Ingest classification assist: a recorded layout stamp → model type.

    Accepts the v2 pair rendering (``"sdxl.diffusers@1+plain.bf16@1"``) or a
    bare topology handle, and **matches on the TOPOLOGY HALF ONLY**. Which
    architecture a checkpoint is, is a fact about WHICH TENSORS it has — the
    topology — and never about how they are quantized: `sdxl.diffusers@1` is
    SDXL whether the weights are bf16, fp8 or nvfp4, and a pattern that could
    see the quant half would be a fingerprint that changes when nothing about
    the architecture did.

    Splitting first also keeps the patterns from matching by accident. They are
    prefix globs, and the topology is the prefix of the render, so
    ``fnmatchcase`` over the whole string would agree today and start
    disagreeing the first time a quant rule is named after a family.

    Returns None for an unrecognized stamp — unclassified is LEGAL and VISIBLE
    (NULL ``model``, serves on fallbacks with the named warning).
    """
    topology = stamp.split("+", 1)[0].strip()
    if not topology:
        return None
    for mt in MODEL_TYPES:
        if any(fnmatchcase(topology, pattern) for pattern in mt.contracts):
            return mt
    return None


def defaults_vocabularies() -> dict[str, type[msgspec.Struct]]:
    """Every recognized name → its ``Defaults`` struct (base types first, then LoRA overlays) — the export emitter's source."""
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
        StableAudio.name: StableAudio.Defaults,
        MusicGen.name: MusicGen.Defaults,
        Ltx2.name: Ltx2.Defaults,
        Ltx2Upsampler.name: Ltx2Upsampler.Defaults,
        InternVLU.name: InternVLU.Defaults,
        Trellis2.name: Trellis2.Defaults,
        Hunyuan3d.name: Hunyuan3d.Defaults,
        JoyCaption.name: JoyCaption.Defaults,
        SDXL.Lora.name: SDXL.Lora.Defaults,
        SD15.Lora.name: SD15.Lora.Defaults,
    }



__all__ = [
    "FLUX2_KLEIN_SCHEDULER_CONFIG",
    "LTX2_SCHEDULER_CONFIG",
    "Ltx2",
    "Ltx2Defaults",
    "Ltx2Upsampler",
    "Ltx2UpsamplerDefaults",
    "Flux1",
    "Flux1Defaults",
    "Flux2Klein",
    "Flux2KleinDefaults",
    "Hunyuan3d",
    "Hunyuan3dDefaults",
    "HiDreamO1",
    "HiDreamO1Defaults",
    "JoyCaption",
    "JoyCaptionDefaults",
    "Knob",
    "LORA_OVERLAYS",
    "LoraOverlay",
    "MiniMaxH3",
    "MiniMaxH3Defaults",
    "MusicGen",
    "MusicGenDefaults",
    "Qwen36A3b",
    "Qwen36Mtp",
    "QwenA3bDefaults",
    "QwenMtpDefaults",
    "QwenTextGenDefaults",
    "Rife",
    "RifeDefaults",
    "STABLE_AUDIO_SCHEDULER_CONFIG",
    "StableAudio",
    "StableAudioDefaults",
    "MODEL_TYPES",
    "QWEN_IMAGE_SCHEDULER_CONFIG",
    "QwenImage",
    "QwenImageDefaults",
    "ModelType",
    "SD15",
    "SchedulerName",
    "SD2",
    "SDXL",
    "Sd15Defaults",
    "Sd15LoraDefaults",
    "Sd15Config",
    "Sd2Defaults",
    "SdxlDefaults",
    "SdxlLoraDefaults",
    "SdxlConfig",
    "SupportsClamp",
    "Trellis2",
    "Trellis2Defaults",
    "Wan22",
    "Wan22Defaults",
    "Z_IMAGE_SCHEDULER_CONFIG",
    "ZImage",
    "ZImageDefaults",
    "defaults_vocabularies",
    "model_type_by_name",
    "model_type_for_contract",
]
