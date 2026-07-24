"""Per-step progress helper for diffusers pipelines (pgw#482).

Free function over a context method so :class:`RequestContext` keeps its
capped surface. Does not import diffusers — it just returns a callable that
matches the ``callback_on_step_end`` contract.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from ..request_context import RequestContext

#: Min seconds between emitted step events; first and last steps always emit.
#: Mirrors the ``training_metric`` throttle so 20-step turbo runs don't spam.
DEFAULT_STEP_MIN_INTERVAL_S = 0.25

#: A ``(start, end)`` sub-range of the request's overall 0..1 progress bar,
#: both bounds in ``[0.0, 1.0]`` with ``start <= end``.
Window = Tuple[float, float]

#: Default window: the callback owns the whole progress bar (single-stage).
_FULL_WINDOW: Window = (0.0, 1.0)


def diffusers_step_callback(
    ctx: "RequestContext",
    num_inference_steps: int,
    *,
    stage: Optional[str] = "denoise",
    min_interval_s: float = DEFAULT_STEP_MIN_INTERVAL_S,
    window: Window = _FULL_WINDOW,
) -> Callable[..., Dict[str, Any]]:
    """Wire a diffusers pipeline's per-step callback to ``ctx.progress``.

    One line in an endpoint function::

        pipe(..., callback_on_step_end=diffusers_step_callback(ctx, steps))

    After each denoise step it emits ``ctx.progress(fraction, stage,
    step=step, total=total)``, throttled to one event per ``min_interval_s``
    (first and last steps always emit). It also calls
    ``ctx.raise_if_cancelled()`` every step, so a cancelled request aborts
    the pipeline mid-run instead of denoising to completion.

    Multi-stage pipelines (e.g. a base denoise pass followed by a latent
    upsample/refine pass) compose two calls, each reporting into its own
    ``window`` of the request's overall progress bar instead of every stage
    resetting the bar to 0::

        pipe(..., callback_on_step_end=diffusers_step_callback(
            ctx, base_steps, stage="denoise", window=(0.0, 0.6),
        ))
        ...
        pipe(..., callback_on_step_end=diffusers_step_callback(
            ctx, refine_steps, stage="refine", window=(0.65, 0.9),
        ))

    ``step``/``total`` on the wire always describe progress within the
    current stage (e.g. "3/8"); ``fraction`` is what maps into ``window``.
    """
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
        step = int(step_index) + 1  # fires after the step ends -> 1-based count
        # th#1111: every step end is a timing mark, un-throttled (the emit
        # below is throttled; the measurement must not be). This is what gives
        # denoise total + per-step timing on all 14 endpoints wired to this
        # callback with no endpoint change.
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
