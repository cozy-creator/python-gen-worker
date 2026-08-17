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
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

from . import aot_serve
# pgw#1331: the two FACTS this module reads, not the 3,100-line arming brain
# that also writes them. `compile_cache` imports `models.loading`/`memory`/
# `w8a8_lora`, so importing it here put diffusers and transformers in the
# adopt-only serve role's static closure — through a per-request REPORTING
# module whose own docstring promises it needs no pipeline, GPU or hub.
from . import compile_facts
from . import serve_posture
from .cell_adopt import EagerPhase

logger = logging.getLogger(__name__)

# --- serving modes (wire-shared with tensorhub) ------------------------------
MODE_EAGER = "eager"
MODE_JIT_CELL = "jit_cell"
MODE_AOT_CELL = "aot_cell"

# --- eager-fallback reason classes (wire-shared) -----------------------------
# A COMPILED lane that fell back to eager for THIS request.
FALLBACK_GUARD_MISS = "guard_miss"
FALLBACK_INGRESS_REFUSED = "ingress_refused"
FALLBACK_HEALING = "healing"
FALLBACK_VOLATILE = "volatile"
#: pgw#888: the hub PINNED an exact compile cell and this target no longer
#: serves it (de-armed for cause, revoked, superseded). The other four classes
#: all describe a cell that is armed and was skipped for one request; this one
#: describes a cell that is not here at all — and until the pgw#888 ruling it
#: was not a fallback reason but a refusal, so the request had no serving mode
#: to report. It rides here so an eager sample charged to a dispatch the hub
#: believed was compiled stays subtractable from the compiled measurement.
FALLBACK_PINNED_CELL_UNAVAILABLE = "pinned_cell_unavailable"
_PER_REQUEST_FALLBACKS = frozenset({
    FALLBACK_GUARD_MISS, FALLBACK_INGRESS_REFUSED,
    FALLBACK_HEALING, FALLBACK_VOLATILE,
    FALLBACK_PINNED_CELL_UNAVAILABLE,
})

# --- eager POSTURE reason classes (pgw#824, wire-shared) ---------------------
# The four classes above answer "a cell was armed and this request did not use
# it". They cannot answer the much commoner case: NOTHING is armed at all. That
# request reported `serving_mode=eager, fallback_reason=""` — indistinguishable
# from a release that never declared a compile target, from a pod whose mint is
# still running, and from a pod that declined the mint for cause. So "why is
# this fleet eager right now" had no query.
#
# The posture token is the SAME token the decline's `self_mint_skipped` /
# `self_mint_started` activity event carries in `phase`, so a request row and
# the worker's own event stream join on one string instead of on a sentence.
#
# pgw#1035: these are ALIASES of :class:`cell_adopt.EagerPhase`, not a second
# spelling of it. They used to be bare literals here while the arming lane's own
# tokens lived in the enum — two lists of the same wire vocabulary, which is the
# drift channel `EagerPhase` exists to close, and only `mint_in_progress` was
# ever pinned across. Values are unchanged; the hub's grouped history is
# untouched.
#: The arming brain has not answered yet (boot in flight, setup not finished).
POSTURE_ARM_PENDING = EagerPhase.ARM_PENDING.value
#: A mint is being built right now (delegated child, background driver); this
#: worker serves eager until it adopts. Transient BY CONSTRUCTION.
POSTURE_MINT_IN_PROGRESS = EagerPhase.MINT_IN_PROGRESS.value
#: The release declared no compile target at all — eager is the contract, not
#: a degradation. Kept distinct so it never pollutes the defect classes.
POSTURE_NO_COMPILE_DECLARED = EagerPhase.NO_COMPILE_DECLARED.value
#: Terminal fallback when a decline reached the request path unclassified.
POSTURE_UNCOMPILED = EagerPhase.UNCOMPILED.value
#: pgw#1082: the declared region did not trace whole under fullgraph.
POSTURE_GRAPH_BREAK = EagerPhase.GRAPH_BREAK.value
#: pgw#1082: the declaration named a dynamic range its own inputs leave.
POSTURE_DECLARED_RANGE_EXCEEDED = EagerPhase.DECLARED_RANGE_EXCEEDED.value
#: pgw#1142 / §4.32 item 4: an operator ordered this worker eager-only. The one
#: posture in this list that is a DECISION rather than a condition, and the one
#: that can be taken back — a request row carrying it says the platform was
#: asked for eager, not that anything failed to arm.
POSTURE_OPERATOR_EAGER_ONLY = EagerPhase.OPERATOR_EAGER_ONLY.value

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
    kinds set ``active_compile_ref``, and stamped cell keys are string-shape
    identical, so the ref alone cannot be pattern-matched. ``aot_serve`` owns
    the answer (``is_aot_ref`` for the recorded kind, the ``_cozy_aot`` marker
    for what is live on the pipeline right now).

    pgw#1010: a ref is no longer NECESSARY for ``jit_cell``. JIT intake arms a
    pipeline that compiles its own graphs and names no artifact — a cell ref is
    exactly what it does not have — and reporting those requests as ``eager``
    would delete the whole JIT arm of the AOT-vs-JIT latency comparison this
    module exists to make answerable. The armed pipeline itself is the
    evidence.
    """
    ref = str(active_compile_ref or "").strip()
    if not ref:
        if pipeline is not None and compile_facts.is_compile_armed(pipeline):
            return MODE_JIT_CELL
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


@lru_cache(maxsize=1)
def detect_sm() -> str:
    """This device's compute capability, or "" when there is no CUDA device.

    Reads ``compile_facts.runtime_key()`` — the same source the cell key is
    built from, so a request's ``sm`` and the cell it ran is keyed by can never
    disagree.

    Memoized (pgw#789): this is called on every request terminal now, and
    ``runtime_key()`` probes torch, the device capability AND the libcuda driver
    version on each call — and logs a warning on every failure. A device's
    compute capability cannot change within a process, so probing it once is not
    a cache with a staleness question. ``detect_sm.cache_clear()`` for tests.
    """
    try:
        return normalize_sm(str(compile_facts.runtime_key().get("sm") or ""))
    except Exception:
        logger.debug("sm detection failed", exc_info=True)
        return ""


def fallback_of(router: Any, sig: Any) -> str:
    """The eager-fallback reason class for input signature ``sig``, or "".

    ``hot_swap.Router.healing`` / ``.volatile`` are sets of input SIGNATURES
    (not target names — pgw#789 corrected this: a caller that passed a target
    name got "" for every request and the axis read clean while requests were
    silently falling back). Either verdict routes the next calls for that
    signature EAGER while the serving tier stays compiled.

    A caller that does not know the request's sig must NOT probe here — use
    ``resolve(guard_missed=...)`` / ``resolve(verdict=...)``, which carry the
    per-request fact the guard-miss callback observed directly.
    """
    if router is None or sig is None:
        return ""
    try:
        if sig in (getattr(router, "volatile", None) or ()):
            return FALLBACK_VOLATILE
        if sig in (getattr(router, "healing", None) or ()):
            return FALLBACK_HEALING
    except Exception:
        logger.debug("router fallback probe failed", exc_info=True)
    return ""


def resolve(
    *,
    active_compile_ref: str = "",
    pipeline: Any = None,
    router: Any = None,
    sig: Any = None,
    guard_missed: bool = False,
    ingress_refused: bool = False,
    verdict: str = "",
    sm: Optional[str] = None,
    eager_posture: str = "",
) -> ServedIdentity:
    """The full dimension set for one request.

    ``guard_missed`` / ``ingress_refused`` are per-REQUEST facts the caller
    observes (pgw#680's ``_compile_guard_missed`` / ``aot_serve``'s ingress
    refusal); ``verdict`` is the router's own heal verdict for THIS request
    (``healing`` | ``volatile``, as reported on ``GuardMiss.heal``) and outranks
    the generic guard-miss class because it says whether the fallback is
    transient or permanent. ``router``+``sig`` is the state-probe fallback for
    callers that know the signature. Any of them means this request was served
    eager by a compiled lane, which is exactly the sample that must not be
    counted as compiled.

    ``eager_posture`` (pgw#824) answers the OTHER eager case — no cell armed at
    all — with the arming brain's own classified token. It applies only when
    the mode is already ``eager``: a per-request fallback on a compiled lane
    always outranks it, and it never sets ``served_eager_fallback`` (nothing
    fell back; there was nothing to fall back FROM), so every existing
    compiled-vs-eager comparison keeps exactly its old meaning.
    """
    mode = classify_mode(active_compile_ref, pipeline)
    named = str(verdict or "").strip()
    if named in _PER_REQUEST_FALLBACKS:
        reason = named
    elif guard_missed:
        reason = FALLBACK_GUARD_MISS
    elif ingress_refused:
        reason = FALLBACK_INGRESS_REFUSED
    else:
        reason = fallback_of(router, sig)
    fell_back = bool(reason)
    if not reason and mode == MODE_EAGER:
        reason = str(eager_posture or "").strip()
    if serve_posture.eager_only():
        # pgw#1142 / §4.32 item 4: an operator ordered eager. The cell may
        # still be ARMED — deliberately, so the order can be taken back
        # without a re-arm — and `classify_mode` reads exactly that armed
        # artifact, so without this the request would report `aot_cell` for a
        # forward the artifact never ran. That is the wire lie pgw#1082/#1093
        # spent two pods closing, arriving from the opposite direction.
        #
        # It outranks the per-request fallback classes because it PRECEDES
        # them: nothing was dispatched to a compiled callable, so no guard
        # could miss and no ingress could refuse. And it is not a fallback —
        # nothing fell back, there was nothing to fall back FROM — so the
        # compiled-vs-eager comparison keeps its meaning.
        mode = MODE_EAGER
        reason = POSTURE_OPERATOR_EAGER_ONLY
        fell_back = False
    return ServedIdentity(
        serving_mode=mode,
        served_cell_ref=str(active_compile_ref or "").strip(),
        served_eager_fallback=fell_back,
        fallback_reason=reason,
        sm=normalize_sm(sm) if sm is not None else detect_sm(),
    )


def _first_int(payload: Any, defaults: Any, names: Tuple[str, ...]) -> int:
    """The first present, positive, int-valued field among ``names``, taking
    the EXECUTED value (payload) and falling back to the endpoint's defaults —
    the same precedence ``RuntimeFormula.term_values_from_struct`` uses.

    Either source may be an ATTRIBUTE holder (the decoded msgspec payload,
    which already has struct defaults applied) or a MAPPING (the executor's
    ``_effective_config`` values). Accepting both is what lets one call site
    pass the pair it actually has instead of adapting one into the other.
    """
    for source in (payload, defaults):
        if source is None:
            continue
        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
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
    "POSTURE_ARM_PENDING",
    "POSTURE_MINT_IN_PROGRESS",
    "POSTURE_NO_COMPILE_DECLARED",
    "POSTURE_UNCOMPILED",
    "ServedIdentity",
    "classify_mode",
    "detect_sm",
    "fallback_of",
    "normalize_sm",
    "resolve",
    "shape_of",
]
