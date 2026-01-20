from __future__ import annotations

import asyncio
import os
from typing import Any, List

import msgspec
import torch
from safetensors.torch import load_file

from gen_worker import ActionContext, CozyHubDownloader, ResourceRequirements, worker_function

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cozy-model-cache")
DEFAULT_MODEL_REF = "download/model-files/2"

class TinyLinearInput(msgspec.Struct):
    x: List[float]
    model_ref: str = DEFAULT_MODEL_REF


class TinyLinearOutput(msgspec.Struct):
    y: float


def _as_float_list(value: Any) -> List[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("x must be a list of two numbers")
    out: List[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError("x must be a list of numbers")
        out.append(float(item))
    return out


def _download_model(model_ref: str) -> str:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    safe_name = model_ref.replace("/", "_").replace(":", "_")
    target_path = os.path.join(MODEL_CACHE_DIR, safe_name)
    if os.path.exists(target_path):
        return target_path

    base_url = os.getenv("COZY_HUB_URL", "").strip()
    token = os.getenv("COZY_HUB_TOKEN", "").strip() or None
    downloader = CozyHubDownloader(base_url, token=token)
    return asyncio.run(downloader.download(model_ref, MODEL_CACHE_DIR, filename=safe_name))


@worker_function(resources=ResourceRequirements())
def tiny_linear(ctx: ActionContext, payload: TinyLinearInput) -> TinyLinearOutput:
    if ctx.is_canceled():
        raise InterruptedError("request canceled")

    if not payload.model_ref:
        raise ValueError("model_ref must be a non-empty string")

    x = _as_float_list(payload.x)
    path = _download_model(payload.model_ref)

    tensors = load_file(path)
    if "weight" not in tensors or "bias" not in tensors:
        raise ValueError("model file missing weight/bias tensors")

    weight = tensors["weight"]
    bias = tensors["bias"]

    x_tensor = torch.tensor(x, dtype=weight.dtype)
    with torch.no_grad():
        y = torch.matmul(weight, x_tensor) + bias

    return TinyLinearOutput(y=float(y.squeeze().item()))
