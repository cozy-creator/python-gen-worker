"""Per-step progress helper for diffusers pipelines."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from ..request_context import RequestContext

DEFAULT_STEP_MIN_INTERVAL_S = 0.25

Window = Tuple[float, float]

_FULL_WINDOW: Window = (0.0, 1.0)


def diffusers_step_callback(
    ctx: "RequestContext[Any]",
    num_inference_steps: int,
    *,
    stage: Optional[str] = "denoise",
    min_interval_s: float = DEFAULT_STEP_MIN_INTERVAL_S,
    window: Window = _FULL_WINDOW,
) -> Callable[..., Dict[str, Any]]:
    """Wire a diffusers pipeline's per-step callback to ``ctx.progress``."""
    total = int(num_inference_steps)
    start, end = window
    if not (0.0 <= start <= end <= 1.0):
        raise ValueError(f"window must satisfy 0.0 <= start <= end <= 1.0, got {window!r}")
    span = end - start
    stage_name = str(stage or "denoise").strip() or "denoise"
    last_emit: Optional[float] = None

    def _on_step_end(
        _pipe: Any,
        step_index: int,
        _timestep: Any = None,
        callback_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal last_emit
        ctx.raise_if_cancelled()
        step = int(step_index) + 1
        timer = getattr(ctx, "_stages", None)
        if timer is not None:
            timer.mark_step(stage_name, step)
        now = time.monotonic()
        is_last = total > 0 and step >= total
        if last_emit is None or is_last or (now - last_emit) >= min_interval_s:
            last_emit = now
            step_fraction = min(step / total, 1.0) if total > 0 else 0.0
            ctx.progress(start + span * step_fraction, stage, step=step, total=total)
        return callback_kwargs if callback_kwargs is not None else {}

    return _on_step_end
