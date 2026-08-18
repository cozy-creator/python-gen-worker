"""A fixture endpoint shaped exactly like serverless-endpoints sdxl main_v2.py.

Ship-code-as-is under the pgw#1382 split: ``SdxlModel(Model[SDXL], lanes=…)``
is the stateful half (load/compile/defaults + the self-restoring mutation
scopes); ONE stateless module-level ``@entrypoint`` serves both modes — the
DEPLOYMENT decides turbo (bound distillation adapter or cfg-off checkpoint)
vs regular CFG (Paul's merge ruling). ``TinyPipeline`` stands in for
``StableDiffusionXLPipeline`` (CPU, fake weights from a config-only
checkpoint) with REAL diffusers schedulers, so the scheduler-scope semantics
under test are the real ones; ``contracts`` stands in for
``tensorfs.contracts`` (tensorfs#111 pending).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional, TypedDict, Unpack

import msgspec
import torch
from diffusers import EulerDiscreteScheduler, LCMScheduler, SchedulerMixin

from gen_worker import (
    Adapter,
    ImageAsset,
    ImageFormat,
    LoadContext,
    Model,
    PromptRole,
    RequestContext,
    ValidationError,
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


class TextToImageInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, PromptRole("positive")]
    negative_prompt: Annotated[str, PromptRole("negative")] = ""
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    # API bounds REJECT (typed 400); within them, the checkpoint's own range
    # CLAMPS. None -> the active recipe's default.
    num_inference_steps: Annotated[int, msgspec.Meta(ge=1, le=80)] | None = None
    guidance_scale: Annotated[float, msgspec.Meta(ge=1.5, le=15.0)] | None = None
    enhance_prompt: bool = True
    seed: int | None = None
    output_format: ImageFormat = "png"


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model_used: str


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

    def __init__(self, unet: TinyUnet, scheduler: SchedulerMixin, dtype: torch.dtype) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.dtype = dtype
        self.loaded_loras: list[str] = []
        self.active_adapters: list[tuple[str, float]] = []
        self.adapter_history: list[list[tuple[str, float]]] = []

    @property
    def components(self) -> dict:
        return {"unet": self.unet}

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

    # THE MUTATION CONTRACT: request-time changes go through these
    # self-restoring scopes; at entrypoint return, serving configuration
    # equals the post-load baseline. Caches may grow; configuration may not
    # drift. Single-flight per instance makes the scopes race-free.
    @contextmanager
    def scheduler(
        self,
        scheduler_cls: type[SchedulerMixin],
        *,
        timestep_spacing: Literal["leading", "trailing", "linspace"] | None = None,
    ) -> Iterator[None]:
        prev = self.pipe.scheduler
        overrides = (
            {} if timestep_spacing is None
            else {"timestep_spacing": timestep_spacing}
        )
        self.pipe.scheduler = scheduler_cls.from_config(
            prev.config, **overrides)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            self.pipe.scheduler = prev

    @contextmanager
    def adapters(self, riding: Sequence[Adapter]) -> Iterator[None]:
        if riding:
            for a in riding:
                self.pipe.load_lora_weights(a.path, adapter_name=a.name)
            self.pipe.set_adapters(
                [a.name for a in riding],
                adapter_weights=[a.scale for a in riding],
            )
        try:
            yield
        finally:
            if riding:
                self.pipe.unload_lora_weights()


def _run(model: SdxlModel, ctx: RequestContext, *, steps: int,
         fmt: ImageFormat, seed: int | None,
         **call_kwargs: Unpack[_PipeCall]) -> ImageAsset:
    # Pure composition: reads model state, mutates nothing.
    generator = (
        torch.Generator("cpu").manual_seed(seed) if seed is not None else None
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
             turbo: Adapter | None, loras: list[Adapter]) -> ImageOutput:
    """One entrypoint; the DEPLOYMENT and the request's picks decide the mode.
    A bound distillation adapter or a step-distilled checkpoint (cfg off)
    serves turbo — fewer steps, distilled scheduler, guidance pinned 0.
    `loras` are the request's style/character picks (envelope-resolved,
    classification-gated to SDXL, scales clamped through each LoRA's strength
    knob) and compose with either mode."""
    ctx.raise_if_cancelled()
    width, height = _BUCKETS[payload.aspect_ratio]
    d = model.defaults

    if turbo is not None and not d.cfg:
        raise ValidationError(
            "this checkpoint is already step-distilled; a distillation "
            "adapter cannot be stacked on it"
        )
    distilled = turbo is not None or not d.cfg
    riding: list[Adapter] = ([turbo] if turbo is not None else []) + loras

    if distilled:
        # Guidance and negative prompts are meaningless at guidance 0 —
        # refuse explicit values rather than silently ignoring them.
        if payload.guidance_scale is not None:
            raise ValidationError(
                "guidance_scale is not accepted on a distilled serving "
                "(guidance is pinned to 0)"
            )
        if payload.negative_prompt:
            raise ValidationError(
                "negative_prompt is not accepted on a distilled serving "
                "(no unconditional branch exists at guidance 0)"
            )
        recipe: SDXL.Lora.Defaults | SDXL.Defaults = (
            turbo.defaults if turbo is not None else d
        )
        steps = recipe.steps.resolve(payload.num_inference_steps, ctx)
        sched_scope = (
            model.scheduler(LCMScheduler)  # type: ignore[arg-type]
            if recipe.schedule == "lcm"
            else model.scheduler(EulerDiscreteScheduler,  # type: ignore[arg-type]
                                 timestep_spacing="trailing")
        )
        prompt = payload.prompt.strip()
        with model.adapters(riding), sched_scope:
            image = _run(
                model, ctx, steps=steps, fmt=payload.output_format,
                seed=payload.seed, prompt=prompt, prompt_2=prompt,
                guidance_scale=0.0, width=width, height=height,
                timesteps=list(recipe.timesteps) or None,
            )
    else:
        steps = d.steps.resolve(payload.num_inference_steps, ctx)
        guidance = d.guidance.resolve(payload.guidance_scale, ctx)

        prompt = payload.prompt.strip()
        negative = payload.negative_prompt.strip()
        if payload.enhance_prompt:
            if d.positive_preamble and d.positive_preamble not in prompt:
                prompt = f"{d.positive_preamble}, {prompt}"
            if d.negative_preamble and d.negative_preamble not in negative:
                negative = f"{d.negative_preamble}, {negative}" if negative else d.negative_preamble

        call_kwargs: _PipeCall = dict(
            prompt=prompt, prompt_2=prompt,
            negative_prompt=negative, negative_prompt_2=negative,
            guidance_scale=guidance, width=width, height=height,
        )
        # The checkpoint's own scheduler config drives objective handling;
        # v-prediction needs the zero-terminal-SNR rescale at call time.
        if model.pipe.scheduler.config.get(  # type: ignore[attr-defined]
                "prediction_type") == "v_prediction":
            call_kwargs["guidance_rescale"] = 0.7

        with model.adapters(riding):
            image = _run(model, ctx, steps=steps, fmt=payload.output_format,
                         seed=payload.seed, **call_kwargs)

    return ImageOutput(
        image=image,
        model_used="+".join([ctx.checkpoint_ref] + [a.name for a in riding]),
    )
