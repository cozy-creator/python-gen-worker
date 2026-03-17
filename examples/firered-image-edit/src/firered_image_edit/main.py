from __future__ import annotations

import logging
import threading
from io import BytesIO
from typing import Annotated, Optional

import msgspec
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

from gen_worker import RequestContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_firered_resources = ResourceRequirements()
_pipeline_locks_guard = threading.Lock()
_pipeline_locks: dict[int, threading.Lock] = {}


def _lock_for_pipeline(pipeline: object) -> threading.Lock:
    key = id(pipeline)
    with _pipeline_locks_guard:
        lock = _pipeline_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _pipeline_locks[key] = lock
    return lock


class EditInput(msgspec.Struct):
    image: Asset
    prompt: str
    negative_prompt: str = " "
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: Optional[int] = None
    num_images_per_prompt: int = 1


class EditOutput(msgspec.Struct):
    image: Asset


@worker_function(_firered_resources)
def edit(
    ctx: RequestContext,
    pipeline: Annotated[
        QwenImageEditPlusPipeline,
        ModelRef(Src.FIXED, "firered_image_edit"),
    ],
    payload: EditInput,
) -> EditOutput:
    """
    FireRed image edit function.

    Mirrors the upstream `FireRedTeam/FireRed-Image-Edit` inference.py behavior:
      - QwenImageEditPlusPipeline
      - image passed as a 1-element list
      - true_cfg_scale (not guidance_scale)
      - default negative_prompt = " "
    """
    if payload.image.local_path is None:
        raise RuntimeError("image.local_path missing (input image was not materialized)")

    if ctx.is_canceled():
        raise InterruptedError("canceled")

    steps = max(1, min(100, int(payload.num_inference_steps)))
    cfg = float(payload.true_cfg_scale)
    n = max(1, min(4, int(payload.num_images_per_prompt)))
    neg = payload.negative_prompt if payload.negative_prompt is not None else " "
    if cfg > 1.0 and not str(neg).strip():
        neg = " "

    logger.info(
        "[request_id=%s] firered-image-edit prompt=%r steps=%s cfg=%.2f n=%d",
        ctx.request_id,
        payload.prompt[:120],
        steps,
        cfg,
        n,
    )

    image = Image.open(payload.image.local_path).convert("RGB")

    generator = None
    if payload.seed is not None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=dev).manual_seed(int(payload.seed))

    with _lock_for_pipeline(pipeline):
        with torch.inference_mode():
            out = pipeline(
                image=[image],
                prompt=payload.prompt,
                generator=generator,
                true_cfg_scale=cfg,
                negative_prompt=neg,
                num_inference_steps=steps,
                num_images_per_prompt=n,
            )

    result: Image.Image = out.images[0]
    buf = BytesIO()
    result.save(buf, format="PNG")
    asset = ctx.save_bytes(f"runs/{ctx.request_id}/outputs/edited.png", buf.getvalue())
    return EditOutput(image=asset)
