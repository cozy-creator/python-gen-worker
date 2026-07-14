"""Per-step progress helper for diffusers pipelines (pgw#482).

Free function over a context method so :class:`RequestContext` keeps its
capped surface. Does not import diffusers — it just returns a callable that
matches the ``callback_on_step_end`` contract.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from ..request_context import RequestContext

#: Min seconds between emitted step events; first and last steps always emit.
#: Mirrors the ``training_metric`` throttle so 20-step turbo runs don't spam.
DEFAULT_STEP_MIN_INTERVAL_S = 0.25


def diffusers_step_callback(
    ctx: "RequestContext",
    num_inference_steps: int,
    *,
    stage: Optional[str] = "denoise",
    min_interval_s: float = DEFAULT_STEP_MIN_INTERVAL_S,
) -> Callable[..., Dict[str, Any]]:
    """Wire a diffusers pipeline's per-step callback to ``ctx.progress``.

    One line in an endpoint function::

        pipe(..., callback_on_step_end=diffusers_step_callback(ctx, steps))

    After each denoise step it emits ``ctx.progress(step/total, stage,
    step=step, total=total)``, throttled to one event per ``min_interval_s``
    (first and last steps always emit). It also calls
    ``ctx.raise_if_cancelled()`` every step, so a cancelled request aborts
    the pipeline mid-run instead of denoising to completion.
    """
    total = int(num_inference_steps)
    last_emit: Optional[float] = None

    def _on_step_end(
        _pipe: Any,
        step_index: int,
        _timestep: Any = None,
        callback_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal last_emit
        ctx.raise_if_cancelled()
        step = int(step_index) + 1  # fires after the step ends -> 1-based count
        now = time.monotonic()
        is_first = last_emit is None
        is_last = total > 0 and step >= total
        if is_first or is_last or (now - last_emit) >= min_interval_s:
            last_emit = now
            fraction = min(step / total, 1.0) if total > 0 else 0.0
            ctx.progress(fraction, stage, step=step, total=total)
        return callback_kwargs if callback_kwargs is not None else {}

    return _on_step_end
