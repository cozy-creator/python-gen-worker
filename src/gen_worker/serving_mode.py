"""What actually served this request (pgw#764 / th#1293).

``JobMetrics.lane`` already says something like ``fp8-w8a8-dynamic+compiled``,
and for a long time that looked like enough. It is not, for two reasons:

1. **``+compiled`` is a BINARY axis platform-wide** — ``precision.ExecCompiled``,
   ``lanes.EXEC_COMPILED``, ``serving_tier`` — so it cannot tell an AOT ``.pt2``
   replay from a JIT dynamo cell, and it names no artifact. The worker knows
   the difference on every single request; the fact dies on the pod. So
   "AOT vs JIT per-request latency on 4090s for sdxl w8a8" is unanswerable
   over our own production traffic, which is the only traffic that matters.
2. **A compiled lane that fell back to eager for ONE request still reports
   ``+compiled``.** A pgw#680 guard miss, a router heal/volatile verdict, or an
   ``aot_serve`` ingress refusal all serve that request eager while the tier
   stays compiled by design. Those samples then contaminate every
   compiled-vs-eager comparison with eager numbers — the measurement silently
   argues against the optimization that is actually working.

This module derives the missing axes from state the worker already holds. It is
deliberately duck-typed and free of executor internals so it can be unit-tested
without a pipeline, a GPU, or a hub.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from . import aot_serve
from . import compile_cache

logger = logging.getLogger(__name__)

# --- serving modes (wire-shared with tensorhub) ------------------------------
MODE_EAGER = "eager"
MODE_JIT_CELL = "jit_cell"
MODE_AOT_CELL = "aot_cell"

# --- eager-fallback reason classes (wire-shared) -----------------------------
FALLBACK_GUARD_MISS = "guard_miss"
FALLBACK_INGRESS_REFUSED = "ingress_refused"
FALLBACK_HEALING = "healing"
FALLBACK_VOLATILE = "volatile"

#: Step-count field names, in precedence order. Matches warmup._STEP_FIELDS so
#: the boot warmup and the served request agree on what "steps" means.
_STEP_FIELDS: Tuple[str, ...] = ("num_inference_steps", "steps")
_WIDTH_FIELDS: Tuple[str, ...] = ("width",)
_HEIGHT_FIELDS: Tuple[str, ...] = ("height",)


@dataclass(frozen=True)
class ServedIdentity:
    """The dimension set for one completed request."""

    serving_mode: str = MODE_EAGER
    served_cell_ref: str = ""
    served_eager_fallback: bool = False
    fallback_reason: str = ""
    sm: str = ""


def classify_mode(active_compile_ref: str, pipeline: Any = None) -> str:
    """``eager`` | ``jit_cell`` | ``aot_cell``.

    The discriminator is the ARMED artifact, never the lane string: both cell
    kinds set ``active_compile_ref``, and stamped ck5 keys are string-shape
    identical, so the ref alone cannot be pattern-matched. ``aot_serve`` owns
    the answer (``is_aot_ref`` for the recorded kind, the ``_cozy_aot`` marker
    for what is live on the pipeline right now).
    """
    ref = str(active_compile_ref or "").strip()
    if not ref:
        return MODE_EAGER
    try:
        if aot_serve.is_aot_ref(ref):
            return MODE_AOT_CELL
        if pipeline is not None and aot_serve.is_armed(pipeline):
            return MODE_AOT_CELL
    except Exception:
        # An unclassifiable ref must not be reported as AOT on a guess: a
        # wrong dimension is worse than a coarse one.
        logger.debug("serving-mode classification failed", exc_info=True)
    return MODE_JIT_CELL


def normalize_sm(raw: str) -> str:
    """``sm_89`` / ``sm89`` / ``89`` -> ``89``, matching
    ``WorkerResources.gpu_sm`` so the request row and the worker row join."""
    token = str(raw or "").strip().lower()
    if token.startswith("sm_"):
        token = token[3:]
    elif token.startswith("sm"):
        token = token[2:]
    return token


def detect_sm() -> str:
    """This device's compute capability, or "" when there is no CUDA device.

    Reads ``compile_cache.runtime_key()`` — the same source the cell key is
    built from, so a request's ``sm`` and the cell it ran is keyed by can never
    disagree.
    """
    try:
        return normalize_sm(str(compile_cache.runtime_key().get("sm") or ""))
    except Exception:
        logger.debug("sm detection failed", exc_info=True)
        return ""


def fallback_of(router: Any, target: Any) -> str:
    """The eager-fallback reason class for ``target`` on ``router``, or "".

    ``hot_swap.Router`` records a heal/volatile verdict per target after a
    pgw#680 guard miss; either verdict routes the next calls EAGER while the
    serving tier stays compiled.
    """
    if router is None or target is None:
        return ""
    try:
        if target in (getattr(router, "volatile", None) or ()):
            return FALLBACK_VOLATILE
        if target in (getattr(router, "healing", None) or ()):
            return FALLBACK_HEALING
    except Exception:
        logger.debug("router fallback probe failed", exc_info=True)
    return ""


def resolve(
    *,
    active_compile_ref: str = "",
    pipeline: Any = None,
    router: Any = None,
    target: Any = None,
    guard_missed: bool = False,
    ingress_refused: bool = False,
    sm: Optional[str] = None,
) -> ServedIdentity:
    """The full dimension set for one request.

    ``guard_missed`` / ``ingress_refused`` are per-REQUEST facts the caller
    observes (pgw#680's ``_compile_guard_missed`` / ``aot_serve``'s ingress
    refusal); the router verdicts are per-target state. Any of them means this
    request was served eager by a compiled lane, which is exactly the sample
    that must not be counted as compiled.
    """
    mode = classify_mode(active_compile_ref, pipeline)
    if guard_missed:
        reason = FALLBACK_GUARD_MISS
    elif ingress_refused:
        reason = FALLBACK_INGRESS_REFUSED
    else:
        reason = fallback_of(router, target)
    return ServedIdentity(
        serving_mode=mode,
        served_cell_ref=str(active_compile_ref or "").strip(),
        served_eager_fallback=bool(reason),
        fallback_reason=reason,
        sm=normalize_sm(sm) if sm is not None else detect_sm(),
    )


def _first_int(payload: Any, defaults: Any, names: Tuple[str, ...]) -> int:
    """The first present, positive, int-valued field among ``names``, taking
    the EXECUTED value (payload) and falling back to the endpoint's defaults —
    the same precedence ``RuntimeFormula.term_values_from_struct`` uses."""
    for source in (payload, defaults):
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                n = int(value)
                if n > 0:
                    return n
    return 0


def shape_of(payload: Any, defaults: Any = None) -> Tuple[int, int, int]:
    """``(steps, width, height)`` for the EXECUTED request, defaults applied.

    0 means "not applicable / not reported" rather than "zero": a non-spatial
    function has no width, and reporting it as 0 keeps that honestly distinct
    from a 0-step request, which cannot happen.
    """
    return (
        _first_int(payload, defaults, _STEP_FIELDS),
        _first_int(payload, defaults, _WIDTH_FIELDS),
        _first_int(payload, defaults, _HEIGHT_FIELDS),
    )


__all__ = [
    "MODE_EAGER",
    "MODE_JIT_CELL",
    "MODE_AOT_CELL",
    "FALLBACK_GUARD_MISS",
    "FALLBACK_INGRESS_REFUSED",
    "FALLBACK_HEALING",
    "FALLBACK_VOLATILE",
    "ServedIdentity",
    "classify_mode",
    "detect_sm",
    "fallback_of",
    "normalize_sm",
    "resolve",
    "shape_of",
]
