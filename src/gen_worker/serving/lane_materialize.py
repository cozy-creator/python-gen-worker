from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models import execution_lanes as el
from .lane_ladder import ResolvedLane, is_baseline

logger = logging.getLogger(__name__)


class LaneMaterializationError(RuntimeError):
    """The pipeline's modules do not execute the lane it resolved to."""


def _denoisers(pipeline: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    for name in ("transformer", "unet", "denoiser", "dit"):
        mod = getattr(pipeline, name, None)
        if mod is not None and hasattr(mod, "named_modules"):
            found.append((name, mod))
    if not found and hasattr(pipeline, "named_modules"):
        found.append(("<root>", pipeline))
    return found


def lane_census(pipeline: Any) -> dict[str, int]:
    """Count the quantized leaves actually present, by marker."""
    counts = {"w8a8": 0, "w4a4": 0, "plain": 0}
    for _, root in _denoisers(pipeline):
        for _, mod in root.named_modules():
            if getattr(mod, "_cozy_w8a8_linear", False):
                counts["w8a8"] += 1
            elif getattr(mod, "_cozy_w4a4_linear", False):
                counts["w4a4"] += 1
            elif type(mod).__name__ == "Linear":
                counts["plain"] += 1
    return counts


def _swap_fp8(pipeline: Any, tree: Path, compute_dtype: Any) -> int:
    from ..models.w8a8 import detect_w8a8_artifacts, swap_w8a8_linears

    arts = detect_w8a8_artifacts(tree)
    if not arts:
        raise LaneMaterializationError(
            f"lane fp8-w8a8: no fp8 artifact was detected in {tree}. The "
            "ladder resolved this lane because the deploy said its bytes were "
            "staged; detection reads the safetensors HEADERS (an F8_E4M3 "
            "`.weight` paired with a `.weight_scale`), so an empty result "
            "means the staged tree is not the one the contract names. "
            "Refusing rather than serving bf16 under an fp8 lane"
        )
    swapped = 0
    for _, root in _denoisers(pipeline):
        for art in arts:
            swapped += int(swap_w8a8_linears(
                root, art, compute_dtype=compute_dtype,
                key_map=getattr(pipeline, "_cozy_w8a8_key_map", None),
            ) or 0)
    return swapped


def _swap_nvfp4(pipeline: Any, tree: Path, compute_dtype: Any) -> int:
    from ..models.w4a4 import detect_w4a4_artifact, swap_w4a4_linears

    art = detect_w4a4_artifact(tree)
    if art is None:
        raise LaneMaterializationError(
            f"lane nvfp4-w4a4: no fp4 artifact was detected in {tree}. Same "
            "reasoning as the fp8 arm: detection is by bytes (a uint8 packed "
            "`.weight` with an e4m3 `.weight_scale` and an fp32 "
            "`.weight_scale_2`), so nothing detected means the staged tree "
            "does not carry this contract's layout"
        )
    swapped = 0
    for _, root in _denoisers(pipeline):
        swapped += int(swap_w4a4_linears(
            root, art, compute_dtype=compute_dtype,
            key_map=getattr(pipeline, "_cozy_w4a4_key_map", None),
        ) or 0)
    return swapped


def materialize(
    pipeline: Any,
    resolved: ResolvedLane,
    *,
    tree: Path,
    compute_dtype: Any = None,
) -> Any:
    """Put `pipeline`'s modules on `resolved`'s lane, then prove they are."""
    body = resolved.body
    if is_baseline(body):
        swapped = 0
    elif body.startswith(el.WEIGHTS_FP8 + "-" + el.ACT_W8A8):
        swapped = _swap_fp8(pipeline, tree, compute_dtype)
    elif body.startswith(el.WEIGHTS_NVFP4):
        swapped = _swap_nvfp4(pipeline, tree, compute_dtype)
    else:
        raise LaneMaterializationError(
            f"lane {body!r} resolved, and this loader has no materializer for "
            f"it. The ladder must not be able to pick a body nothing can "
            f"build — if this fires, the lane table and the materializer "
            f"disagree and one of them is wrong"
        )
    _assert_lane(pipeline, resolved, swapped=swapped)
    return pipeline


def _assert_lane(pipeline: Any, resolved: ResolvedLane, *, swapped: int) -> None:
    counts = lane_census(pipeline)
    body = resolved.body
    if is_baseline(body):
        stray = counts["w8a8"] + counts["w4a4"]
        if stray:
            raise LaneMaterializationError(
                f"lane {body}: the pipeline carries {stray} quantized leaf/"
                f"leaves ({counts}) on a BASELINE lane. Either the tree is not "
                f"the one the resolved contract names, or a previous load left "
                f"converted modules behind. Refusing: a baseline lane's "
                f"pricing, compiled-graph key and executed-lane claim all say "
                f"bf16, and these modules do not"
            )
        return
    got = counts["w8a8"] if body.startswith(el.WEIGHTS_FP8) else counts["w4a4"]
    if got == 0:
        raise LaneMaterializationError(
            f"lane {body}: the swap reported {swapped} module(s) and the "
            f"census finds ZERO quantized leaves ({counts}). Refusing to serve "
            f"a silently-bf16 model under a quantized lane name — this is the "
            f"defect torchcg's matched-nothing refusal exists for, and the "
            f"reason this loader is never its own only witness"
        )
    if counts["plain"] and got and swapped and got < swapped:
        raise LaneMaterializationError(
            f"lane {body}: {swapped} module(s) were planned and only {got} "
            f"quantized leaves exist ({counts}) — a HALF-quantized model, "
            f"which serves mixed numerics under one lane name"
        )
    logger.info("lane %s materialized: %d quantized leaf/leaves, %d plain "
                "(pgw#1606)", body, got, counts["plain"])


def lane_of(pipeline: Any) -> str:
    """The lane body a built pipeline's modules ACTUALLY execute."""
    counts = lane_census(pipeline)
    if counts["w8a8"]:
        return el.WEIGHTS_FP8 + "-" + el.ACT_W8A8 + "-" + el.SCALE_DYNAMIC
    if counts["w4a4"]:
        return el.WEIGHTS_NVFP4 + "-" + el.ACT_W4A4 + "-" + el.SCALE_STATIC
    return el.WEIGHTS_BF16 + "-" + el.ACT_W16A16


__all__ = [
    "LaneMaterializationError",
    "lane_census",
    "lane_of",
    "materialize",
]
