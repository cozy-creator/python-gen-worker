"""A fixture endpoint shaped exactly like serverless-endpoints sdxl main_v2.py."""

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
)

from gen_worker import (
    STATIC,
    Adapter,
    DistillationAdapter,
    ImageAsset,
    ImageFormat,
    LoadContext,
    Model,
    PromptRole,
    RequestContext,
    entrypoint,
    lane,
)
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL

from . import contracts


class AspectRatio(StrEnum):
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"


_BUCKETS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.RATIO_1_1: (16, 16),
    AspectRatio.RATIO_3_4: (12, 16),
}


SdxlScheduler = Literal[
    "dpmpp_2m_karras", "dpmpp_2m", "euler", "euler_trailing",
    "euler_a", "ddim", "lcm",
]
_SchedulerRow = tuple["type[Any]", Mapping[str, str | bool]]
_SCHEDULERS: dict[SdxlScheduler, _SchedulerRow] = {
    "dpmpp_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {}),
    "euler": (EulerDiscreteScheduler, {}),
    "euler_trailing": (EulerDiscreteScheduler, {"timestep_spacing": "trailing"}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "ddim": (DDIMScheduler, {}),
    "lcm": (LCMScheduler, {}),
}


class TextToImageInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, PromptRole("positive")]
    negative_prompt: Annotated[str, PromptRole("negative")] = ""
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    scheduler: SdxlScheduler | None = None
    num_inference_steps: Annotated[int, msgspec.Meta(ge=1, le=80)] | None = None
    guidance_scale: Annotated[float, msgspec.Meta(ge=1.5, le=15.0)] | None = None
    enhance_prompt: bool = True
    seed: int | None = None
    output_format: ImageFormat = "webp"


class LoraUsed(msgspec.Struct):
    ref: str
    scale: float


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model: str
    loras: list[LoraUsed] = []


class _PipeCall(TypedDict, total=False):

    prompt: str
    prompt_2: str
    negative_prompt: str
    negative_prompt_2: str
    guidance_scale: float
    width: int
    height: int
    timesteps: list[int] | None
    guidance_rescale: float


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

        if "explode" in prompt:
            raise RuntimeError("pipeline exploded mid-request")
        steps = len(timesteps) if timesteps else int(num_inference_steps)
        sample = torch.zeros((height, width), dtype=self.dtype)
        for step in range(steps):
            sample = self.unet(sample)
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 0, {})
        rng = generator or torch.Generator("cpu").manual_seed(0)
        noise = torch.randint(
            0, 256, (height, width, 3), generator=rng, dtype=torch.uint8
        )
        image = Image.fromarray(noise.numpy(), "RGB")
        return PipelineResult([image])


class SdxlModel(
    Model[SDXL],
    # A lane IS a tensor-layout v2 STAMP PAIR — `(topology, quant)`, both
    # halves ratified documents — and its VALUE is that lane's own DEMAND
    # FORMULA, never a VRAM string (pgw#1599: there is no single number,
    # because a 4 MP image is not a 1 MP image). Per-lane and not per-model for
    # the reason visible here: fp8 halves the weight bytes AND shrinks the
    # activation coefficient, so one formula would be wrong for one of these
    # two. The compute-capability floor is DERIVED from each lane's QUANT RULE
    # (`capability_floor_sm`, 80 and 89 here) and is never written on a header.
    lanes={
        contracts.SDXL_DIFFUSERS_BF16: lane(
            request=const(MiB(96)) + per_mp_batch(MiB(24)),
        ),
        contracts.COZY_SDXL_FP8_ROWWISE: lane(
            request=const(MiB(48)) + per_mp_batch(MiB(12)),
        ),
    },
    shapes={"aspect": STATIC},
):
    """The stateful half: weights, compile-marked modules, defaults."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(TinyPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)
        self.defaults = ctx.defaults()

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
    """One entrypoint; behavior is driven by the ACTIVE CONFIG's typed fields, not a mode flag."""

    ctx.raise_if_cancelled()
    d = model.defaults

    if turbo is not None and d.step_distilled:
        ctx.warn(f"distillation adapter {turbo.ref!r} ignored: this "
                 "checkpoint is already step-distilled")
        turbo = None
    config: SDXL.Config = turbo.defaults if turbo is not None else d
    adapters: list[Adapter] = ([turbo] if turbo is not None else []) + loras

    prompt = payload.prompt.strip()
    if payload.enhance_prompt and d.positive_preamble \
            and d.positive_preamble not in prompt:
        prompt = f"{d.positive_preamble}, {prompt}"
    negative = payload.negative_prompt.strip()

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

    picked = _pick_scheduler(ctx, payload, turbo)
    steps = config.steps.resolve(payload.num_inference_steps, ctx)
    timesteps: list[int] | None = None
    if config.timesteps:
        if payload.scheduler is not None:
            ctx.warn("pinned denoising timesteps dropped: they belong to the "
                     f"config's scheduler, not {payload.scheduler!r}")
        else:
            if payload.num_inference_steps is not None:
                ctx.warn("num_inference_steps ignored: this config pins its "
                         "denoising timesteps")
            steps, timesteps = len(config.timesteps), list(config.timesteps)

    width, height = _BUCKETS[payload.aspect_ratio]
    call_kwargs: _PipeCall = dict(
        prompt=prompt, prompt_2=prompt,
        width=width, height=height, guidance_scale=guidance,
    )
    if config.cfg:
        call_kwargs.update({
            "negative_prompt": negative, "negative_prompt_2": negative,
        })
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
