"""A fixture endpoint shaped exactly like serverless-endpoints sdxl main_v2.py.

Ship-code-as-is under the pgw#1382 split: ``SdxlModel(Model[SDXL], lanes=…)``
is the stateful half (load/compile/defaults + the self-restoring mutation
scopes); ONE stateless module-level ``@entrypoint`` whose behavior is driven
by the ACTIVE CONFIG's typed fields (Paul's config-not-mode-flag ruling),
with the endpoint-served scheduler table, the DistillationAdapter slot-kind
guard, and structured output evidence. ``TinyPipeline`` stands in for
``StableDiffusionXLPipeline`` (CPU, fake weights from a config-only
checkpoint) with REAL diffusers schedulers, so the scheduler-scope semantics
under test are the real ones; ``contracts`` stands in for
``tensorfs.contracts`` (tensorfs#111 pending).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict, Unpack, cast

import msgspec
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    UniPCMultistepScheduler,
)

from gen_worker import (
    Adapter,
    DistillationAdapter,
    ImageAsset,
    ImageFormat,
    LoadContext,
    Model,
    PromptRole,
    RequestContext,
    entrypoint,
)
from gen_worker.models import SDXL

from . import contracts


class AspectRatio(StrEnum):
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"


# Tiny buckets — the contract file's ~1MP SDXL buckets, CPU-sized.
_BUCKETS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.RATIO_1_1: (16, 16),
    AspectRatio.RATIO_3_4: (12, 16),
}


# The schedulers THIS endpoint serves. SdxlScheduler types the REQUEST field,
# so an unsupported name is a typed 400 at the API boundary. SchedulerName
# (the platform-wide vocabulary in gen_worker.models) appears only on ADAPTER
# metadata (an applied distillation demands its scheduler); checkpoints carry
# no scheduler metadata at all — their tree ships it (ingest-guaranteed).
SdxlScheduler = Literal[
    "dpmpp_2m_karras", "dpmpp_2m", "euler", "euler_trailing",
    "euler_a", "unipc", "ddim", "lcm",
]
# diffusers' lazy top-level exports defeat mypy's subclass view of the
# scheduler classes; the rows are real SchedulerMixin subclasses at runtime.
_SchedulerRow = tuple["type[Any]", Mapping[str, str | bool]]
_SCHEDULERS: dict[SdxlScheduler, _SchedulerRow] = {
    "dpmpp_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {}),
    "euler": (EulerDiscreteScheduler, {}),
    "euler_trailing": (EulerDiscreteScheduler, {"timestep_spacing": "trailing"}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
    "ddim": (DDIMScheduler, {}),
    "lcm": (LCMScheduler, {}),
}


class TextToImageInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, PromptRole("positive")]
    negative_prompt: Annotated[str, PromptRole("negative")] = ""
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    # None -> the applied distillation adapter's demand when one rides, else
    # the checkpoint's own shipped scheduler; a name -> caller override,
    # typed by THIS endpoint's served set.
    scheduler: SdxlScheduler | None = None
    num_inference_steps: Annotated[int, msgspec.Meta(ge=1, le=80)] | None = None
    guidance_scale: Annotated[float, msgspec.Meta(ge=1.5, le=15.0)] | None = None
    enhance_prompt: bool = True
    seed: int | None = None
    output_format: ImageFormat = "webp"


class LoraUsed(msgspec.Struct):
    ref: str      # org/repo@release — the hub identity it was resolved from
    scale: float


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    #: The checkpoint served: its fully-pinned hub ref (org/repo@release).
    model: str
    #: Riding adapters in application order (the distillation adapter first
    #: when one rode, then the request's style LoRAs).
    loras: list[LoraUsed] = []


class _PipeCall(TypedDict, total=False):
    """The exact kwargs this endpoint passes to the pipeline __call__."""

    prompt: str
    prompt_2: str
    negative_prompt: str
    negative_prompt_2: str
    guidance_scale: float
    width: int
    height: int
    timesteps: list[int] | None
    guidance_rescale: float


# --- the author-owned pipeline (the fixture's "diffusers") -----------------


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.scale + self.bias


class PipelineResult:
    def __init__(self, images: list) -> None:
        self.images = images


class TinyPipeline:
    """Author code as-is: its loop, a REAL diffusers scheduler, LoRA hooks."""

    def __init__(self, unet: TinyUnet, scheduler: Any, dtype: torch.dtype) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.dtype = dtype
        self.loaded_loras: list[str] = []
        self.active_adapters: list[tuple[str, float]] = []
        self.adapter_history: list[list[tuple[str, float]]] = []

    @property
    def components(self) -> dict:
        return {"unet": self.unet}

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: Path | str, torch_dtype: torch.dtype = torch.float32
    ) -> "TinyPipeline":
        config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        torch.manual_seed(int(config.get("seed", 0)))
        unet = TinyUnet().to(torch_dtype)
        # Ingest guarantees a scheduler config in every projected tree
        # (synthesizing the model type's canonical one when absent) — the
        # fixture mirrors that guarantee.
        scheduler = EulerDiscreteScheduler.from_config(
            dict(EulerDiscreteScheduler().config,  # type: ignore[attr-defined]
                 **config.get("scheduler", {}))
        )
        return cls(unet, scheduler, torch_dtype)

    def load_lora_weights(self, path: Path | str, adapter_name: str = "") -> None:
        self.loaded_loras.append(adapter_name or str(path))

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        self.active_adapters = list(zip(names, adapter_weights))
        self.adapter_history.append(list(self.active_adapters))

    def unload_lora_weights(self) -> None:
        self.loaded_loras.clear()
        self.active_adapters = []

    def __call__(
        self,
        *,
        prompt: str,
        prompt_2: str = "",
        negative_prompt: str = "",
        negative_prompt_2: str = "",
        guidance_scale: float,
        width: int,
        height: int,
        num_inference_steps: int,
        timesteps: Optional[list[int]] = None,
        guidance_rescale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        callback_on_step_end: Optional[Callable[..., object]] = None,
    ) -> PipelineResult:
        from PIL import Image

        if prompt == "explode":  # a mid-request author bug, on demand
            raise RuntimeError("pipeline exploded mid-request")
        steps = len(timesteps) if timesteps else int(num_inference_steps)
        sample = torch.zeros((height, width), dtype=self.dtype)
        for step in range(steps):
            sample = self.unet(sample)
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 0, {})
        # Real per-pixel signal (the worker's output-integrity floor refuses
        # a blank render), deterministically seeded.
        rng = generator or torch.Generator("cpu").manual_seed(0)
        noise = torch.randint(
            0, 256, (height, width, 3), generator=rng, dtype=torch.uint8
        )
        image = Image.fromarray(noise.numpy(), "RGB")
        return PipelineResult([image])


# --- the endpoint, shaped exactly like main_v2.py --------------------------


class SdxlModel(
    Model[SDXL],
    # A lane IS a tensorfs layout contract — an imported object carrying the
    # actual layout. Omitting lanes= means one lane: SDXL's canonical contract.
    lanes=(contracts.SDXL_DIFFUSERS_BF16, contracts.COZY_SDXL_FP8_ROWWISE),
):
    """The stateful half: weights, compile-marked modules, defaults. One
    instance per (checkpoint x lane), LRU-resident, single-flight."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(TinyPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)
        self.defaults = ctx.defaults()

    # unload(ctx): inherited no-op — eviction is framework-generic.

    # This class persists in memory. Request-time changes go through these
    # self-restoring scopes, so one request cannot mutate pipeline state in
    # a way that affects the next request.
    @contextmanager
    def scheduler(
        self,
        scheduler_cls: type[Any],
        overrides: Mapping[str, str | bool] | None = None,
    ) -> Iterator[None]:
        prev = self.pipe.scheduler
        self.pipe.scheduler = scheduler_cls.from_config(
            prev.config, **(overrides or {})
        )
        try:
            yield
        finally:
            self.pipe.scheduler = prev

    @contextmanager
    def adapters(self, applied: Sequence[Adapter]) -> Iterator[None]:
        if applied:
            for a in applied:
                self.pipe.load_lora_weights(a.path, adapter_name=a.name)
            self.pipe.set_adapters(
                [a.name for a in applied],
                adapter_weights=[a.scale for a in applied],
            )
        try:
            yield
        finally:
            if applied:
                self.pipe.unload_lora_weights()


def _pick_scheduler(ctx: RequestContext, payload: TextToImageInput,
                    turbo: DistillationAdapter | None) -> SdxlScheduler | None:
    """The scheduler precedence, one layer per return:
    1. the request's pick (typed to this endpoint's served set);
    2. the applied distillation adapter's demand (its metadata — the base
       checkpoint's tree cannot know a LoRA needs trailing-Euler/LCM);
    3. None = the checkpoint's own shipped scheduler stands, untouched.
       (Checkpoints carry NO scheduler metadata: the tree IS their choice,
       and ingest guarantees one exists.)"""
    if payload.scheduler is not None:
        return payload.scheduler
    if turbo is not None and turbo.defaults.scheduler is not None:
        if turbo.defaults.scheduler in _SCHEDULERS:
            return cast(SdxlScheduler, turbo.defaults.scheduler)
        ctx.warn(f"adapter demands scheduler {turbo.defaults.scheduler!r}, "
                 "which this endpoint does not serve; using the checkpoint's")
    return None


def _run(model: SdxlModel, ctx: RequestContext, *, steps: int,
         fmt: ImageFormat, seed: int | None,
         **call_kwargs: Unpack[_PipeCall]) -> ImageAsset:
    # Pure composition: reads model state, mutates nothing. The generator
    # lives on the pipe's device — placement is the worker's decision, and
    # this code never names one.
    generator = (
        torch.Generator(device=model.pipe.device).manual_seed(seed)
        if seed is not None else None
    )
    with torch.inference_mode():
        result = model.pipe(
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=ctx.step_callback(steps),
            **call_kwargs,
        )
    return ctx.save_image(result.images[0], format=fmt)


@entrypoint
def generate(ctx: RequestContext, payload: TextToImageInput, model: SdxlModel,
             turbo: DistillationAdapter | None, loras: list[Adapter]) -> ImageOutput:
    """One entrypoint; behavior is driven by the ACTIVE CONFIG's typed fields,
    not a mode flag. The config is the distillation adapter's defaults when one
    rides, else the checkpoint's own."""

    ctx.raise_if_cancelled()
    d = model.defaults

    # -- the active config and applied adapters -----------------------------
    if turbo is not None and d.step_distilled:
        # Stacking a step-distillation on a step-distilled checkpoint fries
        # the output — ignore the adapter, serve the checkpoint as deployed.
        # (cfg is a separate axis: a guidance-distilled full-step checkpoint
        # has cfg off but step_distilled False, and MAY take a turbo LoRA.)
        ctx.warn(f"distillation adapter {turbo.ref!r} ignored: this "
                 "checkpoint is already step-distilled")
        turbo = None
    config: SDXL.Config = turbo.defaults if turbo is not None else d
    adapters: list[Adapter] = ([turbo] if turbo is not None else []) + loras

    # -- prompts: positive preamble applies in every mode; negatives exist
    # -- only under CFG ------------------------------------------------------
    prompt = payload.prompt.strip()
    if payload.enhance_prompt and d.positive_preamble \
            and d.positive_preamble not in prompt:
        prompt = f"{d.positive_preamble}, {prompt}"
    negative = payload.negative_prompt.strip()

    # -- guidance ------------------------------------------------------------
    if config.cfg:
        guidance = config.guidance.resolve(payload.guidance_scale, ctx)
        if payload.enhance_prompt and d.negative_preamble \
                and d.negative_preamble not in negative:
            negative = f"{d.negative_preamble}, {negative}" if negative else d.negative_preamble
    else:
        guidance = 0.0
        if payload.guidance_scale is not None:
            ctx.warn("guidance_scale ignored: this serving runs without "
                     "classifier-free guidance")
        if payload.negative_prompt:
            ctx.warn("negative_prompt ignored: no unconditional branch "
                     "exists without classifier-free guidance")

    # -- scheduler and step ladder ------------------------------------------
    picked = _pick_scheduler(ctx, payload, turbo)
    steps = config.steps.resolve(payload.num_inference_steps, ctx)
    timesteps: list[int] | None = None
    if config.timesteps:
        # A pinned ladder belongs to the config's own scheduler; a request
        # that overrode the scheduler makes it meaningless. When active, the
        # ladder owns the step count.
        if payload.scheduler is not None:
            ctx.warn("pinned denoising timesteps dropped: they belong to the "
                     f"config's scheduler, not {payload.scheduler!r}")
        else:
            if payload.num_inference_steps is not None:
                ctx.warn("num_inference_steps ignored: this config pins its "
                         "denoising timesteps")
            steps, timesteps = len(config.timesteps), list(config.timesteps)

    # -- assemble the pipeline call -----------------------------------------
    width, height = _BUCKETS[payload.aspect_ratio]
    call_kwargs: _PipeCall = dict(
        prompt=prompt, prompt_2=prompt,
        width=width, height=height, guidance_scale=guidance,
    )
    if config.cfg:
        call_kwargs.update({
            "negative_prompt": negative, "negative_prompt_2": negative,
        })
        # The checkpoint's own scheduler config drives objective handling;
        # v-prediction needs the zero-terminal-SNR rescale at call time.
        if model.pipe.scheduler.config.get("prediction_type") == "v_prediction":
            call_kwargs["guidance_rescale"] = 0.7
    if timesteps is not None:
        call_kwargs["timesteps"] = timesteps

    sched_scope = (
        nullcontext() if picked is None
        else model.scheduler(*_SCHEDULERS[picked])
    )
    with model.adapters(adapters), sched_scope:
        image = _run(model, ctx, steps=steps, fmt=payload.output_format,
                     seed=payload.seed, **call_kwargs)

    return ImageOutput(
        image=image,
        model=ctx.checkpoint_ref,
        loras=[LoraUsed(ref=a.ref, scale=a.scale) for a in adapters],
    )
