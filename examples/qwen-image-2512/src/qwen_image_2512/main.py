from __future__ import annotations

import logging
import os
import threading
from io import BytesIO
from typing import Annotated, Optional

import msgspec
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

qwen_resources = ResourceRequirements()
_compile_lock = threading.Lock()
_compiled_transformer = False


class GenerateInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = " "
    # Qwen Image quality default from official examples is 50 steps.
    num_inference_steps: int = 50
    # Qwen Image uses `true_cfg_scale` (not guidance_scale).
    true_cfg_scale: float = 4.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    max_sequence_length: int = 256
    # quality | balanced | fast | custom
    preset: str = "quality"


class GenerateOutput(msgspec.Struct):
    image: Asset


def _round_size(x: int) -> int:
    x = max(512, min(2048, int(x)))
    return max(512, (x // 16) * 16)


def _apply_preset(payload: GenerateInput) -> tuple[int, float]:
    preset = (payload.preset or "").strip().lower()
    if preset == "fast":
        return 28, 3.0
    if preset == "balanced":
        return 40, 3.5
    if preset == "quality":
        return 50, 4.0
    steps = max(20, min(70, int(payload.num_inference_steps)))
    cfg = max(1.0, min(12.0, float(payload.true_cfg_scale)))
    return steps, cfg


def _try_compile_transformer(pipeline: DiffusionPipeline) -> None:
    global _compiled_transformer
    if _compiled_transformer:
        return
    if os.getenv("QWEN_IMAGE_COMPILE", "").strip().lower() not in ("1", "true", "yes"):
        return
    with _compile_lock:
        if _compiled_transformer:
            return
        try:
            mode = os.getenv("QWEN_IMAGE_COMPILE_MODE", "reduce-overhead")
            if hasattr(pipeline, "transformer"):
                pipeline.transformer = torch.compile(pipeline.transformer, mode=mode)  # type: ignore[attr-defined]
                _compiled_transformer = True
                logger.info("enabled torch.compile for Qwen transformer (mode=%s)", mode)
        except Exception as e:
            logger.warning("torch.compile unavailable for Qwen transformer: %s", e)


@worker_function(qwen_resources)
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        DiffusionPipeline,
        ModelRef(Src.FIXED, "qwen_image"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    steps, cfg = _apply_preset(payload)
    width = _round_size(payload.width)
    height = _round_size(payload.height)
    neg = payload.negative_prompt if payload.negative_prompt is not None else " "
    if cfg > 1.0 and not str(neg).strip():
        # Qwen Image expects a non-empty negative prompt when CFG > 1.
        neg = " "

    logger.info(
        "[run_id=%s] qwen-image prompt=%r preset=%s steps=%s cfg=%.2f size=%sx%s",
        ctx.run_id,
        payload.prompt[:80],
        payload.preset,
        steps,
        cfg,
        width,
        height,
    )

    generator = None
    if payload.seed is not None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=dev).manual_seed(int(payload.seed))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            vram = int(torch.cuda.get_device_properties(0).total_memory)
            if vram <= 10 * 1024**3 and hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if vram <= 8 * 1024**3 and hasattr(pipeline, "enable_model_cpu_offload"):
                try:
                    pipeline.enable_model_cpu_offload()
                except TypeError:
                    pipeline.enable_model_cpu_offload(gpu_id=0)
            torch.cuda.empty_cache()
        except Exception:
            pass

    _try_compile_transformer(pipeline)

    call_kwargs = dict(
        prompt=payload.prompt,
        negative_prompt=neg,
        num_inference_steps=int(steps),
        true_cfg_scale=float(cfg),
        width=int(width),
        height=int(height),
        max_sequence_length=int(payload.max_sequence_length),
    )
    if generator is not None:
        call_kwargs["generator"] = generator

    try:
        with torch.inference_mode():
            result = pipeline(**call_kwargs)
    except TypeError:
        # Older diffusers builds may not expose every kwarg.
        call_kwargs.pop("max_sequence_length", None)
        with torch.inference_mode():
            result = pipeline(**call_kwargs)

    image: Image.Image = result.images[0]
    buf = BytesIO()
    image.save(buf, format="PNG")
    out = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())
    return GenerateOutput(image=out)
