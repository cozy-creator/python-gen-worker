"""A fixture endpoint shaped exactly like serverless-endpoints sdxl main_v2.py.

Ship-code-as-is: this module is the WHOLE author surface — the pipeline is
author-owned code (a tiny stand-in for diffusers, CPU-only, fake weights
from a config-only checkpoint), the endpoint class declares lanes + sample
payloads, handlers read the pgw#1372 serving context surface
(``checkpoint_dir``, ``lane``, ``checkpoint_defaults``, ``adapter``,
``is_trace``, ``step_callback``, ``save_image``, ``clamp``…). The body
annotates ``ctx: RequestContext`` exactly as ``main_v2.py`` does; the import
binds that name to the serving surface until the pgw#1373 hardcut re-points
``gen_worker.RequestContext`` itself.

The shipped author surface imports ``torchcg`` and decorates with
``@endpoint(lanes=..., samples=...)``; the decorator is the author-surface
lane's — this fixture stamps the marker the decorator will stamp, through
the loader's declared seam, and imports Lane from the worker's vendored
torchcg because the packaging decision (standalone vs vendored) is Paul's
open question on pgw#1367.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional

import msgspec
import torch

from gen_worker import Endpoint, ImageAsset, PromptRole, StringEnum, ValidationError
from gen_worker._vendor.torchcg import Lane
from gen_worker.serving import ServeContext as RequestContext
from gen_worker.serving.loader import EndpointDeclaration


class AspectRatio(StringEnum):
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"


_BUCKETS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.RATIO_1_1: (16, 16),
    AspectRatio.RATIO_3_4: (12, 16),
}


class TinyDefaults(msgspec.Struct, frozen=True):
    """Per-checkpoint tunables; the hub may override any field."""

    steps: int = 3
    guidance: float = 6.0
    max_guidance: Optional[float] = None
    negative: str = ""


class TextToImageInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, PromptRole("positive")]
    negative_prompt: Annotated[str, PromptRole("negative")] = ""
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    num_inference_steps: Optional[Annotated[int, msgspec.Meta(ge=1, le=16)]] = None
    guidance_scale: Optional[Annotated[float, msgspec.Meta(ge=1.0, le=15.0)]] = None
    seed: Optional[int] = None
    output_format: Literal["webp", "png"] = "png"


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, PromptRole("positive")]
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    output_format: Literal["webp", "png"] = "png"


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model_used: str


# --- the author-owned pipeline (the fixture's "diffusers") -----------------


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.scale + self.bias


class TinyScheduler:
    def __init__(self, config: dict) -> None:
        self.config = dict(config)


class PipelineResult:
    def __init__(self, images: list) -> None:
        self.images = images


class TinyPipeline:
    """Author code as-is: its loop, its scheduler, its components mapping."""

    def __init__(self, unet: TinyUnet, scheduler: TinyScheduler, dtype: torch.dtype) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.dtype = dtype

    @property
    def components(self) -> dict:
        return {"unet": self.unet}

    @classmethod
    def from_pretrained(cls, checkpoint_dir: Path | str, torch_dtype: torch.dtype) -> "TinyPipeline":
        config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        torch.manual_seed(int(config.get("seed", 0)))
        unet = TinyUnet().to(torch_dtype)
        return cls(unet, TinyScheduler(config.get("scheduler", {})), torch_dtype)

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float,
        width: int,
        height: int,
        num_inference_steps: int,
        generator: Optional[torch.Generator] = None,
        callback_on_step_end: Optional[Callable[..., object]] = None,
    ) -> PipelineResult:
        from PIL import Image

        sample = torch.zeros((height, width), dtype=self.dtype)
        if generator is not None:
            sample = torch.rand(
                (height, width), generator=generator, dtype=self.dtype)
        for step in range(int(num_inference_steps)):
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


def _sample_payloads() -> tuple[msgspec.Struct, ...]:
    """Trace coverage: every bucket through the handler."""
    return tuple(
        TextToImageInput(prompt="trace", aspect_ratio=ar) for ar in AspectRatio
    )


class TinyImageEndpoint(Endpoint):
    def setup(self, ctx: RequestContext) -> None:
        self.pipe = TinyPipeline.from_pretrained(
            ctx.checkpoint_dir, torch_dtype=ctx.lane.dtype
        )
        # torch.compile-style marking: at derive this records+hooks the
        # module; at serve it swaps in the (lane, sm) compiled graphs or
        # returns the module unchanged (eager) and registers the holes.
        self.pipe.unet = ctx.compile(self.pipe.unet)
        self.defaults = ctx.checkpoint_defaults(TinyDefaults)

    def teardown(self, ctx: RequestContext) -> None:
        self.torn_down = True  # the base-class hook, observable in tests

    def generate(self, ctx: RequestContext, payload: TextToImageInput) -> ImageOutput:
        ctx.raise_if_cancelled()
        d = self.defaults
        width, height = _BUCKETS[payload.aspect_ratio]
        steps = payload.num_inference_steps or d.steps
        guidance = payload.guidance_scale if payload.guidance_scale is not None else d.guidance
        if d.max_guidance is not None and guidance > d.max_guidance:
            guidance = ctx.clamp(
                "guidance_scale", guidance, hi=d.max_guidance, reason="model maximum"
            )
        generator = (
            torch.Generator("cpu").manual_seed(payload.seed)
            if payload.seed is not None
            else None
        )
        with torch.inference_mode():
            result = self.pipe(
                prompt=payload.prompt.strip(),
                negative_prompt=payload.negative_prompt.strip() or d.negative,
                guidance_scale=guidance,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=ctx.step_callback(steps),
            )
        image = ctx.save_image(result.images[0], format=payload.output_format)
        return ImageOutput(image=image, model_used=ctx.checkpoint_ref)

    def generate_turbo(self, ctx: RequestContext, payload: TurboInput) -> ImageOutput:
        ctx.raise_if_cancelled()
        adapter = ctx.adapter  # hub-resolved distillation LoRA, or None
        if adapter is None and not ctx.is_trace:
            raise ValidationError(
                "no distillation adapter is bound to this deployment; "
                "refusing to serve undistilled output as turbo"
            )
        width, height = _BUCKETS[payload.aspect_ratio]
        with torch.inference_mode():
            result = self.pipe(
                prompt=payload.prompt.strip(),
                guidance_scale=0.0,
                width=width,
                height=height,
                num_inference_steps=1,
                callback_on_step_end=ctx.step_callback(1),
            )
        image = ctx.save_image(result.images[0], format=payload.output_format)
        suffix = f"+{adapter.name}" if adapter else ""
        return ImageOutput(image=image, model_used=f"{ctx.checkpoint_ref}{suffix}")


# What the @endpoint decorator (DATA: lanes) will stamp — the loader's seam.
TinyImageEndpoint.__cozy_endpoint__ = EndpointDeclaration(  # type: ignore[attr-defined]
    lanes=(
        Lane("bf16", compile=("unet",), contract="plain.fp32@1", dtype=torch.float32),
    ),
    samples=_sample_payloads,
)
