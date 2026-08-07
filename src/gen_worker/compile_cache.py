"""Per-SKU torch.compile cache artifacts (#384).

torch.compile wins 15-34% warm latency on flux-class models but costs 20-46s
of compile per (model, resolution) and needs a C toolchain the prod worker
images don't ship. The split: a platform compile job (training-endpoints
``produce-inductor-cache``) compiles once per GPU SKU and publishes the
inductor+triton cache dirs as a repo flavor; workers that opt in via
``@endpoint(compile=Compile(...))`` seed those dirs before load and hit the
cache with no compiler and no stall.

Policy: cache miss / key mismatch / no artifact leaves ordinary lanes eager,
never causing a boot stall or a runtime compile attempt in prod. A declared
W8A8 lane instead fails retryably: eager/dequantized execution cannot claim
W8A8. The compile job itself opts into cold compilation through the explicit
``allow_cold=True`` library argument (requires a toolchain).

Artifacts are FAMILY-keyed (settled 2026-07-06): torch.compile caches key on
the traced graph + shapes, not the weights, so one artifact serves every
fine-tune of a model family. They live in a system-owned repo per family
(``root/family-<family>``), one flavor per (SKU, torch) cell — and they
are CODE: only the platform's first-party compile job publishes shared ones.

Artifact = deterministic ``.tar.gz``::

    metadata.json      key: family, sku/torch/triton/cuda, shapes, targets,
                       image guidance classes, diffusers/transformers
                       versions (+ source_ref info)
    inductor/**        TORCHINDUCTOR_CACHE_DIR contents
    triton/**          TRITON_CACHE_DIR contents

Key sensitivity (all exact-match): family (graph identity), GPU SKU
(autotune choices + cubin arch), torch (fx-graph cache key), triton
(cubin/launcher cache key), diffusers (the traced graph is its code), and
gen-worker itself plus the producer's low-VRAM prep mode (gw#391: the
worker's load/wrap/placement code shapes the traced graph — a cell produced
by a different gen-worker, or traced under different low-VRAM flags, can
pass every other key yet miss inductor's FX-graph cache at trace time,
serving eager while reporting adopted). ``source_ref`` records which family member the producer
compiled from — informational, never part of the match.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import ctypes
import filecmp
import functools
import importlib.util
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple,
)

import inspect
import pickle
import sys

from . import cell_key, env_seal, guard_closure, hot_swap
from .api.errors import RetryableError
from .models import w8a8_lora
from .models.loading import load_from_pretrained, pipeline_weight_lane
from .models.loading import pipeline_weight_lane as _traced_execution_lane
from .models.memory import low_vram_mode, place_pipeline, reconcile_resident_mode
from .models.refs import parse_model_ref
from .models.w8a8_lora import RANK_BUCKETS
from .registry import CompileCell
from .models import execution_lanes as lanespec
from .models import loading as _loading

logger = logging.getLogger(__name__)

METADATA_NAME = "metadata.json"
# 2 (gw#391): key gained the producer gen-worker version. ie#496 extends its
# metadata with the canonical module graph, shape/target table and weight-lane
# schema without gratuitously invalidating proven non-W8A8 cells. New W8A8
# consumers require those fields; checkpoint bytes remain deliberately absent.
ARTIFACT_FORMAT = 2
_MARKER_ATTR = "_cozy_compile"
_JUNK_SUFFIXES = (".lock", ".tmp", ".pid")
# Cache directories and torch's in-process cache latches are process-global.
# Serialize the complete seed+arm transaction so another setup can never arm
# against a half-merged artifact. RLock keeps prepare -> seed_artifact and
# seed_artifact -> capture_env composable without another configuration layer.
_SEED_ARM_LOCK = threading.RLock()
_LOCK_TYPE = type(threading.Lock())

# ---------------------------------------------------------------------------
# pgw#637: dynamo's in-memory code cache is a THIRD serving surface.
#
# Cell keys carry no checkpoint digest, so arming a SECOND checkpoint of an
# already-proven family creates a new pipeline whose target forward shares
# the class ``__code__`` dynamo already compiled — on torch 2.13 (inlined
# nn-modules) the cached entry's guards match the new instance and the
# warmup call runs COMPILED with zero FX/AOT counter movement
# (calls>0, hits=0, misses=0). That signature against a cell this process
# already proved is service, not silence: disproving it bricked the
# compiled lane fleet-wide on every multi-checkpoint session (2026-07-24
# incident, 6/6 workers). The registry below records every cell proven in
# this process (a real FX/AOT hit, or a finalized self-mint); crediting the
# in-memory surface additionally requires DIRECT evidence from dynamo that
# compiled code for this object's targets is live
# (:func:`has_inmemory_compiled_code`) — the registry alone would let one
# object's hit certify another's silence, which gw#603/gw#611 forbid.
_PROVEN_CELLS_LOCK = threading.Lock()
_PROVEN_CELLS: set[str] = set()
# pgw#672: cells whose serve/finalize proof FAILED in this process. Consulted
# at selection and self-mint arm time so one boot never loops adopt-fail /
# mint-fail on the identical identity (the L4 churn loop's worker half).
_QUARANTINED_CELLS: set[str] = set()
# Live armed pipelines (weakly held): the disproof path must never
# ``torch._dynamo.reset()`` globally while a HEALTHY sibling's compiled
# code is live — the global reset killed the first checkpoint's proven
# lane whenever a second checkpoint's proof failed.
_ARMED_PIPELINES: Optional["weakref.WeakSet[Any]"] = None


def _cell_ref_identity(ref: str) -> str:
    """Process-registry identity for one cell ref (pgw#672 / th#1166).

    The SAME cell can be named in two forms — the mint path's
    ``system_repo(family)#<key>`` vs the store's delivered ref (tag/digest
    decorated). Exact-string matching between those forms manufactured
    false negatives in the pgw#637 escape. A key-flavored ref collapses to
    its (family, key); anything else keeps its literal string."""
    ref = str(ref or "").strip()
    if not ref:
        return ""
    family, flavor = parse_cell_ref(ref)
    if family and flavor:

        if cell_key.is_key(flavor):
            return f"{family}#{flavor}"
    return ref


def record_cell_proven(ref: str) -> None:
    """Mark one cell identity as served-and-proven in this process."""
    identity = _cell_ref_identity(ref)
    if not identity:
        return
    with _PROVEN_CELLS_LOCK:
        _PROVEN_CELLS.add(identity)


def cell_proven_in_process(ref: str) -> bool:
    identity = _cell_ref_identity(ref)
    with _PROVEN_CELLS_LOCK:
        return bool(identity) and identity in _PROVEN_CELLS


def record_cell_quarantined(ref: str) -> None:
    """Mark one cell identity as proof-failed in this process (pgw#672)."""
    identity = _cell_ref_identity(ref)
    if not identity:
        return
    with _PROVEN_CELLS_LOCK:
        _QUARANTINED_CELLS.add(identity)
        _PROVEN_CELLS.discard(identity)


def cell_quarantined_in_process(ref: str) -> bool:
    identity = _cell_ref_identity(ref)
    with _PROVEN_CELLS_LOCK:
        return bool(identity) and identity in _QUARANTINED_CELLS


def _armed_pipelines() -> "weakref.WeakSet[Any]":
    global _ARMED_PIPELINES
    if _ARMED_PIPELINES is None:
        _ARMED_PIPELINES = weakref.WeakSet()
    return _ARMED_PIPELINES


def has_inmemory_compiled_code(pipeline: Any) -> bool:
    """True when dynamo holds live compiled code for this pipeline's compile
    targets (pgw#637).

    This is the DIRECT evidence that separates "the warmup ran compiled off
    dynamo's in-memory code cache" from "the wrapper ran eager and nothing
    was ever compiled". Dynamo keys its code cache on the target's
    ``__code__`` object, which every instance of the pipeline class shares —
    exactly why a 2nd checkpoint of an already-minted family serves compiled
    with zero FX/AOT counter movement. Regional compiles (ie#381) live on
    per-block forwards rather than the wrapped target, so those fall back to
    dynamo's total live-entry count.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if not marker:
        return False
    # pgw#657: every `return False` below is a NEGATIVE compile-proof verdict,
    # and a false negative here kills a healthy compiled lane (the pgw#637 /
    # gw#603 class). A probe that fails must therefore say WHY — silently it
    # is indistinguishable from "nothing was ever compiled".
    try:
        from torch._dynamo import eval_frame
    except Exception:
        logger.warning(
            "compile-cache: torch._dynamo.eval_frame unavailable — in-memory "
            "compile evidence reads as ABSENT (false-negative risk)",
            exc_info=True)
        return False

    def _has_entries(fn: Any) -> bool:
        code = getattr(getattr(fn, "__func__", fn), "__code__", None)
        if code is None:
            return False
        try:
            return bool(eval_frame._debug_get_cache_entry_list(code))
        except Exception:
            logger.warning(
                "compile-cache: dynamo cache-entry probe failed for %s — "
                "counted as NOT compiled", getattr(fn, "__qualname__", fn),
                exc_info=True)
            return False

    for _owner, _attr, fn in marker.get("originals") or ():
        if _has_entries(fn):
            return True
    for mod in marker.get("regional_mods") or ():
        # ie#381 regional compile puts the graphs on the repeated BLOCK
        # forwards, not on the wrapped target — probe the block classes.
        try:
            children = list(mod.modules())
        except Exception:
            logger.warning(
                "compile-cache: could not enumerate regional submodules of %s "
                "— its compiled blocks read as ABSENT",
                type(mod).__name__, exc_info=True)
            continue
        for child in children:
            if _has_entries(getattr(type(child), "forward", None)):
                return True
    return False


def reset_target_code(pipeline: Any) -> int:
    """Drop dynamo's in-memory compiled code for ``pipeline``'s armed compile
    targets (pgw#672 root-cause fix, honesty half).

    Dynamo keys its code cache on the target's class-shared ``__code__``, so
    in a WARM process a later arm's proof warmup is served straight from a
    sibling's resident compiled code with ZERO FX/AOT counter movement —
    a pending self-mint then captures nothing (``finish_fleet_mint:
    captured nothing``) and a seeded adoption proves nothing (hits=0,
    misses=0), the exact ``warmups=N, calls=N, cache_hits=0,
    cache_misses=0`` live loop. Calling this immediately before a proof
    window forces the warmup through the real lookup path: a mint truly
    compiles into its capture dir, an adoption truly hits its seeded FX
    entries.

    Sibling safety: the live cache root is an ADDITIVE union
    (:func:`_merge_staged_cache`), so a healthy sibling whose in-memory
    code this reset also drops (shared ``__code__``) re-traces into an FX
    cache HIT on its next call — seconds, never a recompile. Callers run
    inside the exclusive-GPU proof window or an idle adopt, so no compiled
    frame is mid-flight during the reset.

    Returns the number of code objects reset (0 when dynamo/torch is
    unavailable or nothing is armed — a fresh process is a no-op).
    """
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if not marker:
        return 0
    try:
        import torch._dynamo
    except Exception:
        return 0
    codes: list[Any] = []
    for _owner, _attr, fn in marker.get("originals") or ():
        code = getattr(getattr(fn, "__func__", fn), "__code__", None)
        if code is not None:
            codes.append(code)
    for mod in marker.get("regional_mods") or ():
        try:
            children = list(mod.modules())
        except Exception:
            logger.warning(
                "compile-cache: could not enumerate regional submodules of %s "
                "for the proof-window code reset", type(mod).__name__,
                exc_info=True)
            continue
        for child in children:
            fwd = getattr(type(child), "forward", None)
            code = getattr(getattr(fwd, "__func__", fwd), "__code__", None)
            if code is not None:
                codes.append(code)
    reset = 0
    for code in dict.fromkeys(codes):
        try:
            torch._dynamo.reset_code(code)
            reset += 1
        except Exception:
            # pgw#657: a silent skip here is indistinguishable from "no stale
            # in-memory code existed" — say it, the proof may be dishonest.
            logger.warning(
                "compile-cache: dynamo reset_code failed for %r — the proof "
                "warmup may be served from stale in-memory compiled code",
                code, exc_info=True)
    if reset:
        logger.info(
            "compile-cache: dropped in-memory compiled code for %d target "
            "code object(s) ahead of the proof window (pgw#672)", reset)
    return reset


# ---------------------------------------------------------------------------
# pgw#680: guard-miss doctrine — fail-on-recompile at serve time.
#
# The 187s incident class: a tenant request whose inputs miss every cached
# guard set used to pay dynamo's INLINE recompile inside the request (and,
# single-flight, stall every request queued behind it). Doctrine: tenant
# requests on compiled lanes run under a fail-on-recompile stance; the raise
# is caught in the guard wrappers, THIS request serves eager immediately, the
# guard-failure reason is recorded verbatim (Activity event + hub-countable),
# and the exact input class is healed by the existing background warm driver
# so the SECOND request of that shape is compiled.
#
# Stance choice (deliberate, torch 2.13): ``torch._dynamo.config
# .error_on_recompile`` scoped via ``config.patch`` around the guarded
# compiled call — NOT ``torch.compiler.set_stance("fail_on_recompile")``.
# Two reasons, both verified against torch 2.13.0:
#   1. Scope. ConfigModule user overrides are ContextVars, i.e. THREAD-LOCAL
#      (the same mechanics gw#608 measured for ``enable_autograd_cache``).
#      The stance therefore arms exactly the serving thread's guarded call;
#      the hot-swap shape-warm thread and the background mint driver — which
#      run CONCURRENTLY with tenant requests by design (pgw#671) — keep
#      compiling freely. ``set_stance`` mutates a module-global and swaps the
#      process-wide eval-frame callback: it would fail every concurrent
#      warm/mint compile for the duration of a tenant call.
#   2. Semantics. ``error_on_recompile`` raises only on a genuine RECOMPILE
#      (existing cache entries, none matching — the guard-miss), composing
#      with the multi-graph cache: warm entries keep serving under it, and a
#      first-ever compile of a new code object never trips it.
#      ``fail_on_recompile``'s callback raises for ANY tensor frame reaching
#      dynamo while set, first compiles included.
#
# Windows: only the tenant execute window (executor ``tenant_serve_window``)
# arms the stance. Warm / mint / adopt / boot-proof windows never enter it —
# they exist to compile — and the marking is positive (default off), so any
# path not explicitly marked keeps today's semantics.
# ---------------------------------------------------------------------------

_SERVE_WINDOW: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gw_tenant_serve_window", default=False)


@contextlib.contextmanager
def tenant_serve_window() -> Iterator[None]:
    """Mark the current context as tenant-request execution (pgw#680).

    Entered by the executor around the tenant handler call ONLY. ContextVars
    propagate through ``asyncio.to_thread``/``create_task``, so the handler
    thread that ultimately invokes the guarded compiled targets sees it."""
    token = _SERVE_WINDOW.set(True)
    try:
        yield
    finally:
        _SERVE_WINDOW.reset(token)


def in_tenant_serve_window() -> bool:
    return _SERVE_WINDOW.get()


@contextlib.contextmanager
def _fail_on_recompile() -> Iterator[bool]:
    """Arm the serve-window recompile stance for the calling thread.

    Yields True when armed. No-op (False) outside the tenant serve window
    or when torch/dynamo is unavailable — proof/warm/mint windows and
    non-torch environments keep exact current semantics."""
    if not _SERVE_WINDOW.get():
        yield False
        return
    try:
        import torch._dynamo

        patch = torch._dynamo.config.patch(error_on_recompile=True)
    except Exception:
        logger.debug(
            "compile-cache: recompile stance unavailable", exc_info=True)
        yield False
        return
    with patch:
        yield True


def _is_recompile_error(exc: BaseException) -> bool:
    """True for dynamo's fail-on-recompile raise (the guard-miss signal)."""
    try:
        from torch._dynamo import exc as dexc

        return isinstance(exc, dexc.RecompileError)
    except Exception:
        return False


@dataclass(frozen=True)
class GuardMiss:
    """One tenant request that hit fail-on-recompile on a compiled target.

    ``reason`` is torch's verbatim message — per cached entry, the exact
    guard that failed (size/stride/scalar specialization, …): the confession
    is the data for the cell-reusability investigation (Paul's GPU-A/B
    hypothesis). ``sig`` is the request's shape/axis identity as the hot-swap
    router sees it; ``heal`` is the background-heal verdict."""

    target: str
    reason: str
    sig: str
    heal: str
    misses: int


_GUARD_REASON_LINE_RE = re.compile(r"^\s*-\s*(?:\d+/\d+:\s*)?(.+)$")


def guard_miss_reason_class(reason: str) -> str:
    """A short, stable class token for one verbatim guard-miss reason.

    The first per-entry failure line, entry index stripped, whitespace
    collapsed, clipped — so top-N reasons are one ``sort | uniq -c`` away on
    the hub's activity log (per pgw#680 the reasons ARE the instrument)."""
    for line in str(reason or "").splitlines():
        m = _GUARD_REASON_LINE_RE.match(line)
        if m:
            return _clip(m.group(1), 120)
    first = str(reason or "").strip().splitlines()
    return _clip(first[0], 120) if first else "unclassified"


def set_guard_miss_callback(
    pipeline: Any, callback: Callable[[GuardMiss], None],
) -> bool:
    """Bind serve-time guard-miss telemetry to an armed consumer guard.

    Observability only: a failing callback is logged and swallowed — it must
    never break the eager serve of the request that confessed."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return False
    signal["on_guard_miss"] = callback
    return True


def guard_miss_count(pipeline: Any) -> int:
    """Serve-time guard misses observed on this exact pipeline's guards."""
    return _proof_count(pipeline, "guard_misses")


def _record_guard_miss(
    label: str,
    exc: BaseException,
    args: tuple,
    kwargs: dict,
    failure_signal: Optional[Dict[str, Any]],
    heal_target: Callable[..., Any],
) -> GuardMiss:
    """The catch half of pgw#680: count, heal, confess. Never raises."""
    signal = failure_signal or {}
    router = signal.get("router")
    sig: Tuple[Any, ...] = (label, hot_swap.signature(args, kwargs))
    heal = "no_router"
    if isinstance(router, hot_swap.Router):
        heal = router.record_guard_miss(sig, label, heal_target, args, kwargs)
    misses = 1
    lock = signal.get("lock")
    if isinstance(lock, _LOCK_TYPE):
        with lock:
            misses = int(signal.get("guard_misses", 0)) + 1
            signal["guard_misses"] = misses
    miss = GuardMiss(
        target=label,
        reason=str(exc),
        sig=repr(sig[1]),
        heal=heal,
        misses=misses,
    )
    logger.warning(
        "compile-cache: guard-miss on compiled %s — serving THIS request "
        "eager, background heal=%s (pgw#680; miss #%d for this object). "
        "Torch's verbatim reason:\n%s",
        label, heal, misses, miss.reason,
    )
    callback = signal.get("on_guard_miss")
    if callable(callback):
        try:
            callback(miss)
        except Exception:
            logger.exception(
                "compile-cache: guard-miss telemetry callback failed "
                "(request still served eager)")
    return miss


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------

#: The axes :func:`verify` refuses on — the runtime facts a seeded FX cache
#: is genuinely pinned to. ``sku`` left in the pgw#691/ck3 collapse (see
#: :func:`verify`) and must never return; ``sm`` is the GPU identity. The
#: exported lane declares the same shape as ``aot_serve.IDENTITY_AXES``.
IDENTITY_AXES: Tuple[str, ...] = ("torch", "triton", "sm", "cuda",
                                  "image_digest")


def sku_slug(gpu_name: str) -> str:
    """Deterministic SKU slug: ``NVIDIA GeForce RTX 4090`` -> ``rtx-4090``,
    ``NVIDIA H100 80GB HBM3`` -> ``h100-80gb-hbm3``."""
    s = str(gpu_name or "").lower()
    for noise in ("nvidia", "geforce"):
        s = s.replace(noise, " ")
    out = "".join(c if c.isalnum() else "-" for c in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out


def _cuda_driver_version() -> str:
    """CUDA driver API version without shelling out to provider tooling."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if lib.cuInit(0) != 0 or lib.cuDriverGetVersion(ctypes.byref(value)) != 0:
            return ""
        return str(int(value.value))
    except Exception:
        return ""


def runtime_key() -> Dict[str, str]:
    """The consumer-side half of the cache key, probed from this process."""
    key = {
        "sku": "", "sm": "", "torch": "", "triton": "", "cuda": "",
        "cuda_driver": "", "image_digest": os.environ.get(
            "WORKER_IMAGE_DIGEST", "").strip(),
    }
    try:
        import torch

        key["torch"] = str(torch.__version__)
        key["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
            major, minor = torch.cuda.get_device_capability(0)
            key["sm"] = f"sm_{major}{minor}"
            # CUDA's integer encoding (e.g. 13000), obtained from libcuda
            # rather than provider-specific nvidia-smi output.
            key["cuda_driver"] = _cuda_driver_version()
    except Exception:
        # pgw#657: silently leaving these EMPTY manufactures a different cell
        # key than every healthy pod computes — i.e. a guaranteed cache miss
        # (and a mint) whose cause is invisible. Say it.
        logger.warning(
            "compile-cache: torch/CUDA runtime-key probe failed — cell identity "
            "falls back to empty sku/sm/torch fields; expect a cache MISS",
            exc_info=True)
    try:
        import triton

        key["triton"] = str(triton.__version__)
    except Exception:
        logger.debug("compile-cache: triton version unavailable", exc_info=True)
    return key


def _lib_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lib in ("diffusers", "transformers"):
        try:
            out[lib] = str(__import__(lib).__version__)
        except Exception:
            pass
    return out


def gen_worker_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("gen-worker"))
    except Exception:
        return ""


def execution_lane_bucket(execution_lane: str) -> Tuple[str, int]:
    """(base lane, rank bucket) for a weight lane in stamp OR label-token
    form: ``"w8a8-lora128"`` -> ``("w8a8", 128)``, ``"lora32"`` -> ``("", 32)``,
    ``"w8a8"`` -> ``("w8a8", 0)``. Sparse stamps (eager-only, never cells)
    do not parse as bucketed — they pass through as their whole string."""
    m = re.search(r"(?:^|-)lora(\d+)$", str(execution_lane or ""))
    if m is None:
        return str(execution_lane or ""), 0
    base = execution_lane[: m.start()]
    return base, int(m.group(1))


def execution_lane_token(weight_lane: str) -> str:
    """Label token for a traced weight lane (gw#534): cells of different
    lanes are DIFFERENT graphs and must not collide on one flavor label.
    "" (plain resident, incl. bf16-resident) stays unsuffixed. LoRA-branch
    lanes (gw#547/gw#561) keep their base lane's token + the bucket suffix:
    ``w8a8-lora128`` -> ``w8a8-lora128``, ``fp8-hooks-lora32`` ->
    ``w8a16-lora32``, ``lora32`` -> ``lora32`` — one graph family per
    (base lane, rank bucket)."""
    base, bucket = execution_lane_bucket(str(weight_lane or ""))
    tok = {"": "", "fp8-hooks": "w8a16", "w8a8": "w8a8",
           "w4a4": "w4a4"}.get(base, base)
    if bucket:
        return f"{tok}-lora{bucket}" if tok else f"lora{bucket}"
    return tok


def cell_base_execution_lane(pipeline: Any) -> str:
    """Base weight lane for CELL-IDENTITY computation (advertised requested
    keys, pull-by-key lookups, local-store lookups): the pipeline probe
    first, then the denoiser's own lane markers — the identical resolution
    the mint's ``stamp_lane`` memoizes, so requested == published by
    construction (pgw#686). Dispatch/policy surfaces keep the raw
    :func:`loading.pipeline_weight_lane` probe; this is cell identity only."""
    return w8a8_lora.effective_base_execution_lane(pipeline)


def compile_target_execution_lane_error(weight_lane: str, lora_bucket: int) -> str:
    """Return why a worker compile-target lane is not wire-canonical.

    This is the Python half of Tensorhub's compile-target descriptor contract:
    the worker reports the *raw pipeline lane* (Tensorhub maps ``fp8-hooks`` to
    the ``w8a16`` cell token), with an optional exact canonical LoRA suffix.
    Keeping this vocabulary explicit prevents a test or future loader from
    advertising a target the scheduler must reject.
    """
    execution_lane = str(weight_lane or "")
    declared = int(lora_bucket or 0)
    base, observed = execution_lane_bucket(execution_lane)
    if base not in ("", "fp8-hooks", "w8a16", "w8a8", "w4a4"):
        return f"unsupported pipeline_weight_lane {execution_lane!r}"

    if observed not in (0, *RANK_BUCKETS):
        return f"unsupported LoRA bucket {observed} in lane {execution_lane!r}"
    canonical = f"{base}-lora{observed}" if base and observed else (
        f"lora{observed}" if observed else base
    )
    if execution_lane != canonical:
        return f"non-canonical pipeline_weight_lane {execution_lane!r}; expected {canonical!r}"
    if observed != declared:
        return (
            f"pipeline lane LoRA bucket {observed} != declared "
            f"Compile.lora_bucket {declared}"
        )
    return ""


def flavor_label(sku: str, torch_version: str, weight_lane: str = "") -> str:
    """Repo-flavor label for an artifact: ``inductor-rtx-4090-torch2.9`` (+
    ``-w8a8``/``-w8a16`` for non-plain weight lanes, gw#534). The full
    versions live in metadata; the label is for humans + selection. MUST stay
    byte-compatible with tensorhub's compilecache.FlavorLabel."""
    short = ".".join(str(torch_version).split("+")[0].split(".")[:2])
    label = f"inductor-{sku}-torch{short}"
    tok = execution_lane_token(weight_lane)
    return f"{label}-{tok}" if tok else label


def system_repo(family: str) -> str:
    """The system-owned repo holding one family's compiled-artifact cells."""
    fam = str(family or "").strip()
    if not fam:
        raise ValueError("compile-cache family must be non-empty")
    return f"root/family-{fam}"


def parse_cell_ref(ref: str) -> Tuple[str, str]:
    """(family, flavor) from a system cell ref
    (``root/family-<f>[:tag][@digest][#<flavor>]``) via the ONE ref
    grammar (gw#492); ('', '') when the ref is not a system-family ref."""

    try:
        parsed = parse_model_ref(str(ref or ""))
    except ValueError:
        return "", ""
    th = parsed.tensorhub
    if th is None or th.owner != "root" or not th.repo.startswith("family-"):
        return "", ""
    return th.repo[len("family-"):], th.flavor or ""


def cell_execution_lane(ref: str) -> str:
    """The compiled weight-lane token encoded in a system-cell ref.

    The flavor is human/routing metadata; artifact metadata remains the
    authority. This narrow parser exists so a worker presented several cells
    for one family tries the exact lane instead of whichever mapping entry
    happened to arrive first (ie#496).
    """
    _family, flavor = parse_cell_ref(ref)
    _prefix, sep, suffix = flavor.partition("-torch")
    if not sep:
        return ""
    _version, sep, execution_lane = suffix.partition("-")
    return execution_lane if sep else ""


def family_from_ref(ref: str) -> str:
    """Family encoded in a compile-cache ref; '' when the ref is not a
    system-family cell ref."""
    return parse_cell_ref(ref)[0]


def is_cache_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names an inductor compile-cache cell (optionally of
    one specific family). Cells are flavored either with the legacy human
    label (``inductor-<sku>-torch<mm>[-lane]``) or, post-th#883, with the
    worker-computed cell key itself (``ck1-<sha256>`` — pull-by-key)."""

    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
    return flavor.startswith("inductor-") or cell_key.is_key(flavor)


def declared_contract_facts(cfg: Any, *, lora_bucket_override: Optional[int] = None) -> Dict[str, Any]:
    """Canonical declared-shape-contract facts for ``cfg`` (a
    ``registry.CompileCell`` or any duck with the same fields) — the ck2
    ``contract`` cell-key axis digests exactly this dict (pgw#647)."""
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if lora_bucket_override is not None:
        bucket = int(lora_bucket_override)
    text_lens = tuple(getattr(cfg, "text_lens", ()) or ())
    if not text_lens and getattr(cfg, "text_len", None) is not None:
        text_lens = (int(cfg.text_len),)
    return {
        "v": 3,
        "shapes": sorted(
            [int(v) for v in row] for row in getattr(cfg, "shapes", ())),
        "targets": [str(t) for t in getattr(cfg, "targets", ())],
        # pgw#654 gap #6: the CLASS's per-lane text-pin UNION — sibling
        # functions with different pins share one cell contract.
        "text_lens": sorted({int(v) for v in text_lens}),
        "dynamic": [
            {"dim": d.dim, "min": d.min, "max": d.max}
            for d in getattr(cfg, "dynamic", ())
        ],
        "regional": bool(getattr(cfg, "regional", False)),
        "lora_bucket": bucket,
        "guidance": sorted(
            float(v) for v in getattr(cfg, "guidance_scales", ())),
    }


# --- static code closure (recipe identity, Paul's exact-identity
# ruling) -------------------------------------------------------------------
#
# "Look at our code and say 'this is the graph we need', ideally with pure
# static analysis, and that is our unique identifier." The closure is the
# import graph reachable from the compile/composition entrypoints, resolved
# by AST inspection only (no execution): every reached source file is
# content-digested, and the sorted (module-path, digest) list digests into
# the ``code_closure`` axis. Paul's root-imports convention (top-of-file
# imports, no runtime imports) is exactly what makes this static graph
# SOUND — and the mint-time completeness gate below turns that convention
# into a hard check where the key's honesty depends on it.

_CLOSURE_ENTRYPOINTS = (
    "gen_worker.compile_cache",
    "gen_worker.guard_closure",
    "gen_worker.cell_key",
    "gen_worker.env_seal",
    "gen_worker.models.loading",
    "gen_worker.models.provision",
    "gen_worker.models.memory",
)


@functools.lru_cache(maxsize=8192)
def _closure_file_digest(path: str, mtime_ns: int, size: int) -> str:
    """Content digest of one closure file; keyed on (path, mtime, size) so
    repeated key computations never re-read unchanged files."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _module_source(name: str) -> Optional[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _static_imports(path: Path, module_name: str) -> set[str]:
    """Absolute module names imported by ``path``, by AST inspection."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    package = module_name if path.name == "__init__.py" \
        else module_name.rsplit(".", 1)[0] if "." in module_name else ""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                if node.level > 1:
                    parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts)
            else:
                base = node.module or ""
            if node.level and node.module:
                base = f"{base}.{node.module}" if base else node.module
            if base:
                out.add(base)
                # `from pkg import mod` — mod may itself be a module.
                out.update(f"{base}.{alias.name}" for alias in node.names)
    return out


@functools.lru_cache(maxsize=32)
def static_code_closure(roots: Tuple[str, ...] = ()) -> Tuple[Tuple[str, str], ...]:
    """The recipe's code identity: sorted (module path, content digest) of
    every source file statically reachable from the compile entrypoints
    (plus ``roots`` — the ENDPOINT modules, whose source shapes the traced
    graphs too). Restricted to the gen_worker package and the root
    packages; torch/diffusers/transformers content rides the ``toolchain``
    axis at package granularity instead. Deterministic: module-derived
    relative paths, sorted, content digests — never absolute paths, never
    bytecode."""
    packages = {"gen_worker"} | {r.split(".", 1)[0] for r in roots if r}
    queue: List[str] = list(_CLOSURE_ENTRYPOINTS) + [r for r in roots if r]
    seen: set[str] = set()
    out: Dict[str, str] = {}
    while queue:
        name = queue.pop()
        if name in seen or name.split(".", 1)[0] not in packages:
            continue
        seen.add(name)
        # Parent packages execute on import: they are part of the closure.
        if "." in name:
            queue.append(name.rsplit(".", 1)[0])
        path = _module_source(name)
        if path is None:
            continue
        rel = name.replace(".", "/") + (
            "/__init__.py" if path.name == "__init__.py" else ".py")
        try:
            st = path.stat()
        except OSError:
            continue
        out[rel] = _closure_file_digest(str(path), st.st_mtime_ns, st.st_size)
        queue.extend(_static_imports(path, name))
    return tuple(sorted(out.items()))


def closure_completeness_gap(roots: Tuple[str, ...] = ()) -> List[str]:
    """Loaded modules inside the composition namespaces that the static
    import walk cannot see. NOT a mint gate (Paul's pgw#990 ruling demoted the
    closure to a possible future memo — this check's only job was memo
    honesty, so it rides the deferred memo issue): kept as diagnostics for
    that issue. Note the live finding it produced: executor-side models/*
    modules (disk_gc, lane_gate, ...) load outside the composition walk,
    so any future memo scope must be the walk's namespaces, not the
    package prefix."""
    static = {rel for rel, _digest in static_code_closure(tuple(roots))}
    scope_prefixes = ("gen_worker.models",) + _CLOSURE_ENTRYPOINTS + tuple(
        r for r in roots if r)
    gaps: List[str] = []
    for name, module in sorted(sys.modules.items()):
        if module is None:
            continue
        if not any(name == p or name.startswith(p + ".")
                   for p in scope_prefixes):
            continue
        file = getattr(module, "__file__", None)
        if not file or not str(file).endswith(".py"):
            continue
        rel = name.replace(".", "/") + (
            "/__init__.py" if Path(str(file)).name == "__init__.py" else ".py")
        if rel not in static:
            gaps.append(name)
    return gaps


def assert_closure_complete(roots: Tuple[str, ...] = ()) -> None:
    gaps = closure_completeness_gap(roots)
    if gaps:
        raise RuntimeError(
            f"code-closure completeness gate: {len(gaps)} loaded "
            f"module(s) outside the static import closure — a dynamic "
            f"import is hiding trace-relevant code from the recipe key: "
            f"{gaps[:10]!r}")


@functools.lru_cache(None)
def toolchain_digest() -> Tuple[Tuple[str, str], ...]:
    """pgw#710: CONTENT identity of the compile toolchain, per component —
    the equivalence precondition that lets ``image_digest`` be relaxed
    (pgw#700) without degrading the compile stack's identity to version
    strings (the ccache ``compiler_check=mtime`` failure class; sccache's
    answer — hash the compiler binary and its runtime libs — is the
    precedent).

    Components: the dist-info ``RECORD`` of torch/triton and every
    ``nvidia-*`` runtime wheel (RECORD already carries per-file sha256s, so
    hashing it is whole-package content identity with no multi-GB re-walk)
    plus the bundled CUDA tool BINARIES (ptxas/nvdisasm ride triton's
    wheel; a swapped ptxas silently changes emitted cubins). Recorded in
    metadata, never a key axis."""
    out: Dict[str, str] = {}
    try:
        import importlib.metadata

        # diffusers/transformers/peft ride here at package granularity
        # (their VERSION axes left the key; content replaces them).
        wanted = ("torch", "triton", "diffusers", "transformers", "peft")
        for dist in importlib.metadata.distributions():
            name = str(dist.metadata.get("Name") or "").lower()
            if name in wanted or name.startswith("nvidia-"):
                record = dist.read_text("RECORD") or ""
                out[name] = hashlib.sha256(record.encode()).hexdigest()[:16]
    except Exception:
        logger.debug("toolchain_digest: dist-info walk failed", exc_info=True)
    try:
        import triton

        bin_dir = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
        if bin_dir.is_dir():
            for tool in sorted(bin_dir.iterdir()):
                if tool.is_file():
                    out[f"bin:{tool.name}"] = hashlib.sha256(
                        tool.read_bytes()).hexdigest()[:16]
    except Exception:
        logger.debug("toolchain_digest: cuda tool hash failed", exc_info=True)
    return tuple(sorted(out.items()))


@functools.lru_cache(None)
def content_keys() -> Tuple[Tuple[str, str], ...]:
    """torch/triton CONTENT identity as upstream computes it (cache-design
    review §6.5: ``torch_key`` hashes the whole torch package's bytes;
    ``triton_key`` per-file shas + the libtriton binary). Recorded in
    metadata for observability/forensics — the key's content identity
    for the same stack rides the ``toolchain`` axis (dist-info RECORDs +
    tool binaries), which is cheaper and covers the cuda runtime too."""
    out: Dict[str, str] = {"torch": "", "triton": ""}
    try:
        from torch._inductor.codecache import torch_key

        out["torch"] = hashlib.sha256(torch_key()).hexdigest()[:16]
    except Exception:
        logger.debug("content_keys: torch_key unavailable", exc_info=True)
    try:
        from triton.runtime.cache import triton_key

        out["triton"] = hashlib.sha256(
            str(triton_key()).encode()).hexdigest()[:16]
    except Exception:
        logger.debug("content_keys: triton_key unavailable", exc_info=True)
    return tuple(sorted(out.items()))


def artifact_metadata(
    *,
    family: str,
    source_ref: str = "",
    source_digest: str = "",
    shapes: Iterable[Tuple[int, ...]] = (),
    targets: Iterable[str] = (),
    guidance_scales: Iterable[float] = (),
    low_vram_mode: str = "",
    storage_dtype: str = "",
    compile_mode: str = "whole",
    weight_lane: str = "",
    lora_bucket: int = 0,
    graph_signature: str = "",
    weight_contract: Optional[Dict[str, Any]] = None,
    shape_contract: Optional[Dict[str, Any]] = None,
    composition: Iterable[Tuple[str, str]] = (),
) -> Dict[str, Any]:
    """Producer-side metadata for :func:`pack` (no timestamps: artifacts of
    identical content must be byte-identical). ``source_ref``/``source_digest``
    record the family member compiled from — informational only.
    ``low_vram_mode`` is the prep mode the producer pipeline was traced under
    (gw#391): its flags are traced into the graphs, so a consumer prepped in a
    different mode must reject the cell. ``storage_dtype`` records the weight
    storage the binding REQUESTED — informational only. ``weight_lane`` is the
    lane the built pipeline ACTUALLY traced under (gw#534:
    ``loading.pipeline_weight_lane`` — "" plain-resident, "fp8-hooks"
    fp8-resident weights with a per-layer upcast, traced INTO the graphs
    (ie#381; pgw#727 made that structure instead of hooks) and is
    parity-checked at :func:`enable` like ``low_vram_mode``. Shape rows are
    (w, h) or (w, h, frames); ``guidance_scales`` records the image CFG /
    no-CFG graph classes captured for every 2-D row — see ``Compile``."""
    meta: Dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "kind": "torch-inductor-cache",
        **runtime_key(),
        "gen_worker": gen_worker_version(),
        "family": str(family or ""),
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
        "shapes": [[int(v) for v in s] for s in shapes],
        "targets": list(targets),
        "guidance_scales": [float(v) for v in guidance_scales],
        "low_vram_mode": str(low_vram_mode or ""),
        "storage_dtype": str(storage_dtype or ""),
        "compile_mode": str(compile_mode or "whole"),
        "weight_lane": str(weight_lane or ""),
        "lora_bucket": int(lora_bucket or 0),
        "graph_signature": str(graph_signature or ""),
        "weight_contract": dict(weight_contract or {}),
        "shape_contract": dict(shape_contract or {}),
        # pgw#697: per-module fingerprint rows so an adoption refusal can
        # name the exact drifted module, not just a digest mismatch.
        "composition": [[str(p), str(d)] for p, d in composition],
        # pgw#696: the execution-environment seal rides verbatim — the
        # env_seal axis is recomputed FROM it, never trusted as a stamp.
        env_seal.SEAL_KEY: env_seal.effective_seal(),
        # recipe facts: toolchain + static code closure feed the key
        # axes (recomputed from these blocks, never trusted as stamps);
        # content_keys stay observability (review §6.5). Endpoint closure
        # roots ride in when the executor passes them (train-lane wiring).
        "content_keys": dict(content_keys()),
        "toolchain": dict(toolchain_digest()),
        "code_closure": dict(static_code_closure()),
        # pgw#719: the per-library list behind the seal's combined
        # loaded_libs digest — a mismatch names the library.
        "loaded_libs": dict(env_seal.frozen_library_digests()),
        "libs": _lib_versions(),
    }
    # gw#581/th#883: stamp the worker-owned cell key the recorded axes
    # describe. Derived FROM the metadata (never probed separately), so the
    # stamp can never disagree with the axes it summarizes. Callers that
    # later override a key axis (build()'s serving image digest) re-stamp.

    return cell_key.stamp(meta)


def verify(meta: Dict[str, Any], *, family: str = "") -> str:
    """'' when the artifact matches this runtime, else the mismatch reason.

    STRICT on every axis (cache-design review §6.9): a cell that is SILENT
    on an axis is refused, named — never accepted. The old
    ``if want and want != have`` conditionals were the exact shape of JAX
    PR #27814's one documented wrong-cache-hit (a version axis "only
    sometimes incorporated"). Pre-launch, no external consumers, so the
    legacy silent-axis compatibility path is retired outright.

    Family is the graph-identity half of the key: fine-tunes of one family
    share caches by design."""
    if int(meta.get("format") or 0) != ARTIFACT_FORMAT:
        return f"format {meta.get('format')!r} != {ARTIFACT_FORMAT}"
    here = runtime_key()
    # sku is NOT here (pgw#691/ck3): it left the identity axes — sm + cuda +
    # torch + triton pin every hardware fact the compiled artifacts carry,
    # and a same-sm cell minted on a different SKU must arm, not refuse.
    # It stays recorded in metadata for observability and selection only.
    # cuda_driver is deliberately NOT here either (gw#577): triton's disk
    # cache keys on the wheel's ptxas + SM arch; the host libcuda build
    # never enters any compiled-artifact key. Recorded for observability.
    for field in IDENTITY_AXES:
        want, have = str(meta.get(field) or ""), here[field]
        if want != have:
            return f"{field} {want!r} != runtime {have!r}"
    want_gw, have_gw = str(meta.get("gen_worker") or ""), gen_worker_version()
    if want_gw != have_gw:
        # gw#391: the producer's gen-worker shapes the traced graph; a version
        # drift means the FX-graph cache keys may no longer match.
        return f"gen_worker {want_gw!r} != runtime {have_gw!r}"
    libs = meta.get("libs") or {}
    here_libs = _lib_versions()
    for lib in sorted(set(libs) | set(here_libs)):
        want, have = str(libs.get(lib) or ""), str(here_libs.get(lib) or "")
        if want != have:
            return f"{lib} {want!r} != runtime {have!r}"
    want_fam = str(meta.get("family") or "")
    if family and want_fam != family:
        return f"family {want_fam!r} != {family!r}"
    return ""


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def _clean_tarinfo(ti: tarfile.TarInfo, executable: bool = False) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0
    ti.mode = 0o755 if executable else 0o644
    return ti


_ELF_MAGIC = b"\x7fELF"


def _cubin_arch(path: Path) -> int:
    """The SM arch a cubin was compiled for (nvidia ELF ``e_flags`` low
    byte), 0 when the file is unreadable or not ELF."""
    try:
        with open(path, "rb") as f:
            header = f.read(0x34)
    except OSError:
        return 0
    if len(header) < 0x34 or not header.startswith(_ELF_MAGIC):
        return 0
    # EI_CLASS: 2 = ELF64 (e_flags at 0x30), 1 = ELF32 (e_flags at 0x24).
    offset = 0x30 if header[4] == 2 else 0x24
    flags = int.from_bytes(header[offset:offset + 4], "little")
    return flags & 0xFF


def _ptx_jit_gaps(
    files: Iterable[Path], cache_root: Path, sm: str,
) -> list[str]:
    """PTX-JIT exposure per kernel (pgw#698). A kernel whose only compiled
    form is PTX makes the HOST DRIVER's JIT compile it at load time — the
    one path where the deliberately-unkeyed driver version (gw#577) can
    re-enter compiled-kernel behavior. Every ``.ptx`` must ship a sibling
    ``.cubin``, and when the artifact declares its sm the cubin arch must
    match it exactly."""
    want_arch = 0
    if sm.startswith("sm_"):
        try:
            want_arch = int(sm[3:])
        except ValueError:
            want_arch = 0
    kernels: Dict[Tuple[Path, str], Dict[str, Path]] = {}
    for p in files:
        if p.suffix in (".ptx", ".cubin"):
            kernels.setdefault((p.parent, p.stem), {})[p.suffix] = p
    gaps: list[str] = []
    for (_parent, _stem), forms in sorted(
        kernels.items(), key=lambda kv: str(kv[0][0] / kv[0][1]),
    ):
        ptx, cubin = forms.get(".ptx"), forms.get(".cubin")
        if ptx is not None and cubin is None:
            gaps.append(
                f"{ptx.relative_to(cache_root)}: PTX only — no cubin, the "
                "driver JIT would compile it")
            continue
        if cubin is not None and want_arch:
            arch = _cubin_arch(cubin)
            if arch and arch != want_arch:
                gaps.append(
                    f"{cubin.relative_to(cache_root)}: cubin arch sm_{arch} "
                    f"!= artifact sm_{want_arch}")
    return gaps


def _inductor_cache_config() -> Dict[str, Any]:
    """The inductor cache flags that decide whether a compile could have
    written a reusable entry AT ALL. Read live, never assumed: on 2.13
    ``bundle_triton_into_fx_graph_cache`` defaults True, which moves triton
    artifacts INSIDE the fx entry and therefore changes what lands on disk."""
    facts: Dict[str, Any] = {}
    # Never trigger a FRESH heavy import from a diagnostic: importing
    # torch._inductor runs cache_dir(), which mkdirs TORCHINDUCTOR_CACHE_DIR
    # and raises on an unwritable one — leaving a half-initialized module that
    # poisons every later import (measured: the mega-cache artifact factory
    # double-registers). By pack time inductor is always imported anyway; if it
    # is not, its config was never consulted and has nothing to explain.
    if "torch._inductor.config" not in sys.modules:
        return facts
    try:
        from torch._inductor import config as inductor_config

        for name in ("fx_graph_cache", "force_disable_caches",
                     "bundle_triton_into_fx_graph_cache", "freezing"):
            facts[name] = getattr(inductor_config, name, "?")
    except Exception:  # noqa: BLE001 — a diagnostic must never raise
        logger.debug("compile-cache: inductor config unreadable", exc_info=True)
    return facts


def _capture_forensics(capture: Path, pipe: Any = None) -> str:
    """Why a pack refusal happened, IN the refusal itself.

    First-real-pod audit: the first real-GPU mint ran its whole plan and was
    refused at pack for both functions, and the verbatim reason was LOST (the
    hub persists no typed worker events) — one pod run per refusal. Every pack
    refusal now carries the facts that discriminate its candidate causes, so
    ONE run settles which gate fired and why:

    * ``latched`` — did the compile write HERE, or is inductor pointed
      somewhere else (gw#608's class);
    * ``tree`` — what DID land, per subdir, with counts;
    * ``inductor`` — was caching disabled, bypassed or bundled;
    * ``proof``/``process`` — did this process compile at all, or reuse
      in-process compiled code and write nothing (pgw#604's class).

    Never raises: a diagnostic that can fail is a second lost refusal.
    """
    parts: List[str] = [f"capture={capture}"]
    try:
        want = {str(Path(capture) / sub) for sub in ("inductor", "triton")}
        live = {
            env: os.environ.get(env, "")
            for env in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        }
        parts.append(
            "latched=" + ("yes" if set(live.values()) == want else "NO")
            + " " + " ".join(f"{k}={v or '-'}" for k, v in sorted(live.items()))
        )
        tree: List[str] = []
        for sub in ("inductor", "triton"):
            base = Path(capture) / sub
            if not base.is_dir():
                tree.append(f"{sub}:absent")
                continue
            children = sorted(p for p in base.iterdir())
            if not children:
                tree.append(f"{sub}:empty")
            for child in children:
                count = sum(1 for p in child.rglob("*") if p.is_file()) \
                    if child.is_dir() else 1
                tree.append(f"{sub}/{child.name}:{count}")
        parts.append("tree=[" + ", ".join(tree) + "]")
        config_facts = _inductor_cache_config()
        if config_facts:
            parts.append("inductor=" + " ".join(
                f"{k}={v}" for k, v in sorted(config_facts.items())))
        if pipe is not None:
            parts.append(
                f"proof=hits:{cache_hit_count(pipe)} "
                f"misses:{cache_miss_count(pipe)}")
        counters = inductor_counters()
        if counters:
            parts.append("process=" + " ".join(
                f"{k}:{v}" for k, v in sorted(counters.items())))
    except Exception:  # noqa: BLE001 — see the docstring
        logger.debug("compile-cache: capture forensics failed", exc_info=True)
    return " | ".join(parts)


def pack(cache_root: Path, out_path: Path, metadata: Dict[str, Any]) -> Path:
    """Deterministic artifact from a capture root holding ``inductor/`` and
    ``triton/``: sorted entries, zeroed times/owners, gzip mtime 0 — identical
    content always packs to identical bytes. Refuses (pgw#698) when any
    kernel would rely on driver PTX JIT, naming the kernels."""
    cache_root = Path(cache_root)
    out_path = Path(out_path)
    files: list[Path] = []
    for sub in ("inductor", "triton"):
        base = cache_root / sub
        if base.is_dir():
            files.extend(
                p for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(_JUNK_SUFFIXES)
            )
    files.sort(key=lambda p: str(p.relative_to(cache_root)))
    gaps = _ptx_jit_gaps(files, cache_root, str(metadata.get("sm") or ""))
    if gaps:
        shown = "; ".join(gaps[:10])
        more = f" (+{len(gaps) - 10} more)" if len(gaps) > 10 else ""
        # The gate's own census rides the refusal: a real PTX exposure and a
        # false gap (e.g. cubins bundled into the fx entry rather than written
        # beside the ptx) look identical without these counts.
        ptx = sum(1 for p in files if p.suffix == ".ptx")
        cubin = sum(1 for p in files if p.suffix == ".cubin")
        config_facts = _inductor_cache_config()
        raise RuntimeError(
            f"cubin-completeness gate (pgw#698): {len(gaps)} kernel(s) "
            f"would rely on driver PTX JIT — {shown}{more} "
            f"[census: {ptx} ptx, {cubin} cubin, {len(files)} files, "
            f"sm={metadata.get('sm') or '-'}, bundle_triton="
            f"{config_facts.get('bundle_triton_into_fx_graph_cache', '?')}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                meta_bytes = json.dumps(metadata, sort_keys=True, indent=1).encode()
                ti = _clean_tarinfo(tarfile.TarInfo(METADATA_NAME))
                ti.size = len(meta_bytes)
                tar.addfile(ti, io.BytesIO(meta_bytes))
                for p in files:
                    rel = str(p.relative_to(cache_root))
                    ti = _clean_tarinfo(
                        tarfile.TarInfo(rel), executable=os.access(p, os.X_OK)
                    )
                    ti.size = p.stat().st_size
                    with open(p, "rb") as f:
                        tar.addfile(ti, f)
    return out_path


def unpack(artifact: Path, dest_root: Path) -> Dict[str, Any]:
    """Extract an artifact's ``inductor/``+``triton/`` trees into ``dest_root``
    (merging with whatever is already seeded) and return its metadata."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {}
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            name = member.name
            if name == METADATA_NAME:
                if not member.isfile():
                    raise ValueError(
                        f"unsafe {METADATA_NAME} member in compile-cache artifact"
                    )
                f = tar.extractfile(member)
                meta = json.loads(f.read().decode()) if f else {}
                continue
            posix = PurePosixPath(name)
            parts = posix.parts
            if (
                not member.isfile()
                or not parts
                or parts[0] not in ("inductor", "triton")
                or any(part in ("", ".", "..") for part in parts)
                or posix.is_absolute()
            ):
                raise ValueError(f"unsafe or unknown member in compile-cache artifact: {member.name!r}")
            target = dest_root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            assert src is not None
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if member.mode & 0o100:
                target.chmod(0o755)
    if not meta:
        raise ValueError(f"compile-cache artifact {artifact} has no {METADATA_NAME}")
    return meta


# ---------------------------------------------------------------------------
# Capture (producer) / seed (consumer)
# ---------------------------------------------------------------------------


def _normalize_system_info(info: Dict[str, Any], sm: str) -> Dict[str, Any]:
    """The pure half of the fx system-key shim: replace the GPU marketing
    name with the sm token and recompute the embedded hash exactly the way
    ``CacheBase.get_system`` does."""
    device = info.get("device")
    if not sm or not isinstance(device, dict) or not device.get("name"):
        return info
    from torch._inductor.codecache import SYSTEM_CACHE_KEY_STRATEGY

    normalized = json.loads(json.dumps(info))
    normalized["device"]["name"] = sm
    normalized["hash"] = SYSTEM_CACHE_KEY_STRATEGY.key_from_json(
        {"device": normalized["device"], "version": normalized.get("version")})
    return normalized


def _install_fx_system_shim() -> None:
    """P0 (cache-design review §6.1, VERIFIED on a real B200 cell): inductor
    hashes ``torch.cuda.get_device_properties().name`` — the GPU MARKETING
    string — into every FX graph key via ``CacheBase.get_system()``
    (torch 2.13 codecache.py:287-311, consumed :1503). A cell minted on an
    a40 therefore never HITS on an rtx-3090 despite the identical sm_86 ck
    key: the pgw#691 sku collapse delivers zero cross-SKU hits until the
    inner key is normalized. This shim rewrites the device name to our
    ``sm_XX`` token (hash recomputed with torch's own strategy), installed
    identically on mint and consumer (both arm through ``apply``).

    Upstream precedent in the SAME file: AOTInductor already keys on
    ``AOTI_COMPUTE_CAPABILITY`` (``get_compute_capability()``,
    codecache.py:260) — capability, not name. Upstream ask (capability-based
    ``get_system``) is tracked on the pgw#708 upstream-watch issue.

    Version-pinned: test_determinism_pgw694 asserts the upstream source
    shape and fails loudly on a torch bump that changes it (pgw#705 gate).
    """
    try:
        from torch._inductor.codecache import CacheBase
    except Exception:
        logger.debug("fx system shim: inductor unavailable", exc_info=True)
        return
    if getattr(CacheBase.get_system, "_cozy_sm_normalized", False):
        return
    original = CacheBase.get_system

    @functools.lru_cache(None)
    def _normalized_get_system() -> Dict[str, Any]:
        return _normalize_system_info(dict(original()), runtime_key()["sm"])

    _normalized_get_system._cozy_sm_normalized = True  # type: ignore[attr-defined]
    CacheBase.get_system = staticmethod(_normalized_get_system)  # type: ignore[assignment,method-assign]


def _semantic_cache_tag(pipeline: Any, cfg: Any) -> str:
    """Digest of the SEMANTIC identity (format|kind|family|lane|mode|
    contract) — bound into every inner torch.compile key via
    ``cache_key_tag`` (review §6.3), so a delivered cell's entries are
    mechanically unconsumable by a process whose declared semantic identity
    differs. Environment facts are deliberately excluded: the inner FX key
    already hashes them natively (system info, config, dtypes) and the
    outer key pins them via env_seal/toolchain/code_closure — the tag's job
    is semantics only."""
    execution_lane = cell_key._canonical_execution_lane(
        pipeline_weight_lane(pipeline),
        int(getattr(cfg, "lora_bucket", 0) or 0))
    payload = "|".join((
        str(ARTIFACT_FORMAT), "inductor",
        str(getattr(cfg, "family", "") or ""), execution_lane,
        "regional" if bool(getattr(cfg, "regional", False)) else "whole",
        cell_key.contract_digest(declared_contract_facts(cfg)),
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _set_semantic_cache_tag(pipeline: Any, cfg: Any) -> None:
    """Install the semantic tag for THIS arm's compiles. Process-global
    (torch.compiler.config), set at every arm before its warm compiles; a
    later cross-family arm in the same process retags before its own
    compiles — a mid-serve heal recompile under the newer tag can only
    MISS, never cross-consume."""
    try:
        import torch.compiler.config as compiler_config

        compiler_config.cache_key_tag = _semantic_cache_tag(pipeline, cfg)
    except Exception:
        logger.debug("semantic cache tag: unavailable", exc_info=True)


def capture_env(root: Path) -> Path:
    """Point inductor+triton at the dirs under ``root`` (producer capture and
    consumer seeding share this contract). Safe mid-process: latched inductor
    path caches are cleared so a hot adoption's re-seed actually takes effect
    (gw#391 — the worker has been serving eager long before seeding)."""
    root = Path(root)
    for sub, env in (("inductor", "TORCHINDUCTOR_CACHE_DIR"), ("triton", "TRITON_CACHE_DIR")):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(d)
    _disable_aot_autograd_cache()
    _reset_inductor_latch()
    return root


def _disable_aot_autograd_cache() -> None:
    """gw#608: the AOTAutogradCache key hashes ``fx_kwargs[get_decomp_fn]``
    via the function's REPR — which embeds the process memory address
    (ASLR), so AOT keys can NEVER match across processes/pods. On the
    consumer, the AOT-layer miss recompiles without consulting the on-disk
    FX entries, so a byte-portable cell reports cache_hits=0 (live: two
    hosts, 8/8 misses on graphs whose FxGraphCache keys were bit-identical
    across three independent mints). Compiled-cell portability therefore
    requires the FX cache to be the lookup surface: disable the AOT layer
    symmetrically for producer capture and consumer seeding. Costs a cheap
    AOT re-analysis per fresh process; the expensive inductor compile still
    serves from the (portable) FX entries.

    LIVE DISPROOF of the 0.40.4/0.40.5 shape (2026-07-21, B200 pods,
    gen-worker 0.40.5): the mint capture still packed 8 ASLR-keyed
    ``aotautograd/`` entries and the store-served sibling still failed 8/8
    — because in torch 2.13 ``ConfigModule`` user overrides are a
    ContextVar, i.e. THREAD-LOCAL: the assignment below ran on the arming
    thread while the warmup compile ran on another thread that still saw
    the default True. The env var is no rescue post-import
    (``env_name_force`` is read once at config install). Process-global
    disable therefore needs BOTH: the pre-torch-import env in the
    entrypoint (fresh processes, incl. compile-worker subprocesses) and
    the installed config entry's ``env_value_force`` mutated here (torch
    already imported — tools, tests, embedders)."""
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"

    if "torch" not in sys.modules:
        return
    try:
        import torch._functorch.config as fconf

        fconf.enable_autograd_cache = False  # this thread (fast path, public API)
        # Process-global: user overrides are thread-local ContextVars in
        # torch>=2.13; the entry-level env force is consulted by every
        # thread with top precedence.
        fconf._config["enable_autograd_cache"].env_value_force = False  # type: ignore[attr-defined]
    except Exception:
        logger.debug("compile-cache: AOT autograd cache disable unavailable", exc_info=True)


def _reset_inductor_latch() -> None:
    """Clear inductor's in-memory caches that may have latched the previous
    cache-dir paths (torch's own ``temporary_cache_dir`` does the same)."""

    if "torch" not in sys.modules:
        return
    try:
        from torch._inductor.utils import clear_caches

        clear_caches()
    except Exception:
        logger.debug("compile-cache: inductor latch reset unavailable", exc_info=True)


# seeding reuses the same env contract
seed_env = capture_env

# gw#608: True once this process seeded a verified DELIVERED cell into the
# live cache root. The inductor/triton cache dirs are process-global, so a
# self-mint capture in the same process would re-point them away from the
# seeded entries; fleet arming declines the mint instead.
_DELIVERED_SEEDED = False


def delivered_cell_seeded() -> bool:
    return _DELIVERED_SEEDED


def inductor_counters() -> Dict[str, int]:
    """This process's compiled-artifact cache counters (monotonic). The delta
    across a warmup is the honest adopted-vs-silently-eager signal (gw#391):
    zero hits means the seeded cell never served the trace.

    gw#611: the AOT-autograd layer is a SECOND serving surface. In bundled
    mode an AOT hit loads the compiled artifact without ever consulting
    FxGraphCache (measured on torch 2.13: fxgraph counters fully silent on a
    served call), so a proof reading only fxgraph_* sees hits=0/misses=0 on
    a healthy serving cell and fail-closes it — the th#954 SDXL second-boot
    release-bricking shape. AOT hits are therefore reported alongside
    (``aot_cache_hit``/``aot_cache_miss``) and count as serving evidence.
    Production pins the AOT layer OFF (gw#608 portability), so these stay 0
    unless a config regression re-enables it — in which case a served
    warmup must still prove, never brick."""
    try:
        from torch._dynamo.utils import counters

        c = counters["inductor"]
        out = {
            k: int(c.get(k, 0))
            for k in ("fxgraph_cache_hit", "fxgraph_cache_miss", "fxgraph_cache_bypass")
        }
        a = counters["aot_autograd"]
        out["aot_cache_hit"] = int(a.get("autograd_cache_hit", 0))
        out["aot_cache_miss"] = int(a.get("autograd_cache_miss", 0))
        return out
    except Exception:
        return {}


def counters_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


def compile_wall_seconds() -> float:
    """This process's cumulative torch.compile wall time (monotonic, seconds).

    gw#587: the store-served-boot runtime assertion samples this before/after
    a boot warmup window to measure the actual inductor compile wall (not
    just a hit/miss count) — a delivered cell should cost ~0 here. Mirrors
    ``inductor_counters()``: process-global, so callers must scope the
    before/after sample to a window where no OTHER boot is compiling
    concurrently (the executor already holds the exclusive GPU permit for
    exactly this reason)."""
    try:
        from torch._dynamo.utils import calculate_time_spent

        return float(calculate_time_spent().get("total_wall_time", 0.0))
    except Exception:
        return 0.0


def toolchain_present() -> bool:
    """Any C or C++ compiler — the DYNAMO lane's requirement.

    Deliberately NOT tightened to C++ (pgw#823): on CUDA, dynamo emits Triton
    kernels behind a PYTHON wrapper and compiles fine with no C++ compiler at
    all — leg 2's 24-47 minute mints are the proof. Tightening this predicate
    would refuse the only mint lane the fleet currently has working. The
    AOT lane's stricter question is :func:`cxx_toolchain_present`.
    """
    return any(shutil.which(c) for c in ("cc", "gcc", "g++", "clang"))


#: What ``torch._inductor`` reaches for when it needs to build C++, in its own
#: preference order. ``CXX`` wins because inductor honours it too.
_CXX_CANDIDATES = ("g++", "clang++", "c++")


def cxx_compiler() -> str:
    """The C++ compiler AOTInductor would actually invoke, or ``""``.

    Asks in the same order inductor does — ``config.cpp.cxx`` (which already
    folds in ``CXX``) when torch is importable, else the plain PATH search —
    so this predicate cannot drift from the thing it is predicting.
    """
    try:
        from torch._inductor import config as _icfg

        declared = getattr(getattr(_icfg, "cpp", None), "cxx", None)
        # torch states this as a tuple of candidates, e.g. (None, 'g++').
        names = [str(c) for c in (declared or ()) if c] \
            if isinstance(declared, (tuple, list)) else \
            ([str(declared)] if declared else [])
        for name in names:
            found = shutil.which(name)
            if found:
                return found
        if names:
            # torch named its candidates and NONE resolved — that is exactly
            # the InvalidCxxCompiler the linker would raise. Do not fall
            # through to a broader guess and contradict it.
            return ""
    except Exception:  # noqa: BLE001 — no torch (discovery/build time) or an
        pass          # unexpected config shape: fall back to the PATH search
    for name in (os.environ.get("CXX", ""),) + _CXX_CANDIDATES:
        if name:
            found = shutil.which(name)
            if found:
                return found
    return ""


def cxx_toolchain_present() -> bool:
    """Whether AOTInductor can link a kernel on this pod (pgw#823).

    A C compiler is NOT enough: AOTI forces inductor's C++ wrapper and links
    a real ``.so``. Measured on a real L4 (release `39ac3726`, 0.84.0) — the
    endpoint image carries a C compiler, so ``toolchain_present()`` passed,
    and the mint spent **336 s** loading, exporting both graph classes and
    reaching the linker before torch said
    ``InvalidCxxCompiler: No working C++ compiler found ... (None, 'g++')``.
    """
    return bool(cxx_compiler())


# ---------------------------------------------------------------------------
# FX-key forensics (gw#608)
#
# torch 2.13 CompiledFxGraph entries embed ``_fx_graph_cache_debug_lines`` —
# the complete FxGraphHashDetails per-component dump behind their own key.
# A store-served boot that recompiles (the gw#608 signature) SAVES its fresh
# entries into the live cache dir before the warmup proof fails, so at
# failure time both sides of the divergence are on disk: the seeded cell's
# key inputs (inside the artifact tar) and this boot's (freshly written).
# Diffing them names the exact diverging key component in the wire-visible
# error — no pod-log access needed. Pure observability: every helper
# degrades to empty on any error.
# ---------------------------------------------------------------------------


def _fx_entry_lines(data: bytes) -> Tuple[str, list]:
    """(key, hash-details lines) from one pickled FxGraphCache entry."""

    obj = pickle.loads(data)
    key = str(getattr(obj, "_fx_graph_cache_key", "") or "")
    lines = list(getattr(obj, "_fx_graph_cache_debug_lines", None) or [])
    return key, lines


def artifact_fx_lines(artifact: Path) -> Dict[str, list]:
    """key -> FxGraphHashDetails lines for every fxgraph entry in a cell."""
    out: Dict[str, list] = {}
    try:
        with tarfile.open(Path(artifact), mode="r:*") as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if (
                    len(parts) < 4
                    or parts[:2] != ("inductor", "fxgraph")
                    or not member.isfile()
                ):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    key, lines = _fx_entry_lines(f.read())
                except Exception:
                    continue
                if key and lines:
                    out.setdefault(key, lines)
    except Exception:
        logger.debug("fx forensics: artifact unreadable", exc_info=True)
    return out


def live_fx_lines(inductor_dir: Optional[Path] = None) -> Dict[str, list]:
    """key -> FxGraphHashDetails lines from the live inductor cache dir
    (defaults to the seeded ``TORCHINDUCTOR_CACHE_DIR``)."""
    out: Dict[str, list] = {}
    base = Path(inductor_dir) if inductor_dir else Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""))
    fx_root = base / "fxgraph"
    if not str(base) or not fx_root.is_dir():
        return out
    for entry in sorted(fx_root.glob("*/*/*")):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        try:
            key, lines = _fx_entry_lines(entry.read_bytes())
        except Exception:
            continue
        if key and lines:
            out.setdefault(key, lines)
    return out


_FX_COMPONENT_RE = re.compile(r"\A\[(\S+)\]\s+([^:]+):\s?(.*)\Z", re.DOTALL)


def _fx_components(lines: list) -> Dict[str, Tuple[str, str]]:
    """component name -> (hash, value text) from one entry's debug lines."""
    out: Dict[str, Tuple[str, str]] = {}
    for line in lines:
        m = _FX_COMPONENT_RE.match(str(line))
        if m:
            out[m.group(2).strip()] = (m.group(1), m.group(3).strip())
    return out


def _clip(value: str, limit: int = 120) -> str:
    flat = " ".join(str(value).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def fx_cache_failure_report(artifact: Optional[Path] = None) -> str:
    """Exhaustive FX-cache state for a failed store-served warmup proof
    (gw#608). ALWAYS returns a non-empty report — counts alone discriminate
    the failure classes with zero pod-log access:

    - fresh_keys>0            => the boot computed DIFFERENT keys (B1); the
                                 per-component divergence is appended.
    - fresh_keys=0 with
      samekey_resave rows     => the boot computed the SAME keys and torch
                                 re-saved next to the seeded entries — the
                                 miss is in the candidate LOAD path (B2:
                                 unpickle / extern-libs guard), which only
                                 ever executes on consumers (a mint has
                                 nothing to iterate); the sibling diff and
                                 probes below name the failing step.
    - cell_keys=0             => the artifact itself was unreadable here.

    Every sub-probe degrades to an err token; this never raises."""

    out: list = []
    seeded_lines: Dict[str, list] = {}
    seeded_names: Dict[str, set] = {}
    cell_extern = None
    cell_guards = "<none>"
    if artifact is not None:
        try:
            with tarfile.open(Path(artifact), mode="r:*") as tar:
                for member in tar:
                    parts = PurePosixPath(member.name).parts
                    if (
                        len(parts) < 5
                        or parts[:2] != ("inductor", "fxgraph")
                        or not member.isfile()
                    ):
                        continue
                    key, entry_name = parts[3], parts[4]
                    seeded_names.setdefault(key, set()).add(entry_name)
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    try:
                        obj = pickle.loads(f.read())
                    except Exception as exc:  # noqa: BLE001
                        out.append(
                            f"cell_unpickle=EXC:{type(exc).__name__}:"
                            f"{_clip(str(exc), 80)}")
                        continue
                    if cell_extern is None:
                        cell_extern = str(
                            getattr(obj, "extern_libs_key", None))
                        cell_guards = repr(
                            getattr(obj, "guards_expr", None))
                    lines = list(getattr(
                        obj, "_fx_graph_cache_debug_lines", None) or [])
                    if lines:
                        seeded_lines.setdefault(key, lines)
        except Exception as exc:  # noqa: BLE001
            out.append(f"cell_read=EXC:{type(exc).__name__}")
    out.insert(0, f"cell_keys={len(seeded_names)}")
    out.append(f"cell_guards={cell_guards}")
    out.append(f"cell_extern={_clip(str(cell_extern), 90)}")

    base = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "") or "")
    fx_root = base / "fxgraph"
    live_files: Dict[str, list] = {}
    if str(base) and fx_root.is_dir():
        for keydir in sorted(fx_root.glob("*/*")):
            if keydir.is_dir():
                files = sorted(
                    p for p in keydir.iterdir()
                    if p.is_file() and not p.name.startswith("."))
                if files:
                    live_files[keydir.name] = files
    else:
        out.append(f"live_dir_missing={str(fx_root) or '<unset>'}")
    fresh = sorted(k for k in live_files if k not in seeded_names)
    out.append(f"live_keys={len(live_files)}")
    out.append(f"fresh_keys={len(fresh)}")

    # Same-key re-save: seeded entry FILENAMES are their content sha, so any
    # other file inside a seeded key dir is THIS boot's save of the same key.
    resaves = 0
    for key, files in sorted(live_files.items()):
        names = seeded_names.get(key)
        if not names:
            continue
        fresh_sibs = [p for p in files if p.name not in names]
        seed_sibs = [p for p in files if p.name in names]
        if not fresh_sibs or not seed_sibs:
            continue
        resaves += 1
        if resaves == 1:
            try:
                a = pickle.loads(seed_sibs[0].read_bytes())
                b = pickle.loads(fresh_sibs[0].read_bytes())
                out.append(
                    f"samekey_resave[{key[:12]}]: guards "
                    f"cell={getattr(a, 'guards_expr', None)!r} "
                    f"boot={getattr(b, 'guards_expr', None)!r}; extern "
                    f"cell={_clip(str(getattr(a, 'extern_libs_key', None)), 90)} "
                    f"boot={_clip(str(getattr(b, 'extern_libs_key', None)), 90)}")
            except Exception as exc:  # noqa: BLE001
                out.append(
                    f"samekey_probe=EXC:{type(exc).__name__}:"
                    f"{_clip(str(exc), 80)}")
    out.append(f"samekey_resaves={resaves}")

    # Emulate torch's candidate-load preconditions on one seeded live entry.
    probe = next(
        ((k, v) for k, v in live_files.items() if k in seeded_names), None)
    if probe is not None:
        try:
            pickle.loads(probe[1][0].read_bytes())
            out.append("live_cell_entry_unpickle=ok")
        except Exception as exc:  # noqa: BLE001
            out.append(
                f"live_cell_entry_unpickle=EXC:{type(exc).__name__}:"
                f"{_clip(str(exc), 80)}")
    try:
        import torch.utils._triton as _tu

        out.append("extern_current=" + _clip(
            _tu._extern_libs_key(_tu.triton_backend()) or "<empty>", 90))
    except Exception as exc:  # noqa: BLE001
        out.append(
            f"extern_current=EXC:{type(exc).__name__}:{_clip(str(exc), 80)}")

    if fresh:
        try:
            observed = {
                k: _fx_entry_lines(live_files[k][0].read_bytes())[1]
                for k in fresh[:2]
            }
            divergence = fx_key_forensics(seeded_lines, observed)
            if divergence:
                out.append("divergence: " + divergence)
        except Exception as exc:  # noqa: BLE001
            out.append(f"divergence=EXC:{type(exc).__name__}")
    return "; ".join(str(v) for v in out)


def fx_key_forensics(
    seeded: Dict[str, list],
    observed: Dict[str, list],
    *,
    max_fresh: int = 2,
    max_components: int = 4,
) -> str:
    """Name the FxGraphHashDetails components on which this boot's freshly
    compiled FX entries diverge from the seeded cell's (gw#608). Each fresh
    key (present live, absent from the cell) is matched to the seeded key
    with the fewest differing component hashes — the graphs are counterparts,
    so the minimal diff IS the key defect. '' when there is nothing to say."""
    fresh = {k: v for k, v in observed.items() if k not in seeded}
    if not seeded or not fresh:
        return ""
    seeded_components = {k: _fx_components(v) for k, v in seeded.items()}
    reports = []
    for key, lines in sorted(fresh.items())[:max_fresh]:
        fresh_c = _fx_components(lines)
        best_key = ""
        best_diff: Optional[list] = None
        for skey, sc in seeded_components.items():
            diffs = [
                name for name in sorted(set(fresh_c) | set(sc))
                if fresh_c.get(name, ("", ""))[0] != sc.get(name, ("", ""))[0]
            ]
            if best_diff is None or len(diffs) < len(best_diff):
                best_key, best_diff = skey, diffs
        if best_diff is None:
            continue
        sc = seeded_components[best_key]
        named = "; ".join(
            f"{name}: cell={_clip(sc.get(name, ('', '<absent>'))[1])} != "
            f"boot={_clip(fresh_c.get(name, ('', '<absent>'))[1])}"
            for name in best_diff[:max_components]
        )
        reports.append(
            f"fresh key {key} vs nearest cell key {best_key}: "
            f"{len(best_diff)} differing component(s): {named or 'none'}"
        )
    return " | ".join(reports)


class AdoptError(RuntimeError):
    """Classified adoption failure (ModelEvent ``adopt_failed:<reason>``)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


class CellSelectionBugError(RuntimeError):
    """A SELF-REQUESTED, identity-verified cell failed to arm (th#883).

    Under worker-owned selection the worker never refuses a cell it asked
    for: the artifact's axes describe exactly the key this runtime computed
    for itself, so any arm failure is by construction a bug in the one
    shared selection/parity brain — never a compatibility outcome. Callers
    must surface it as the ``cell_selection_bug`` event class (loud, wire-
    visible), never as a silent eager fallback."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class CompiledExecutionLaneUnavailableError(RetryableError):
    """A precision lane whose production contract requires a cell is unsafe."""


def find_artifact(root: Path) -> Optional[Path]:
    """The compile-cache tarball inside a downloaded snapshot dir (or the
    file itself)."""
    root = Path(root)
    if root.is_file():
        return root
    return next(iter(sorted(root.rglob("*.tar.gz"))), None)


def _merge_staged_cache(staged: Path, live: Path) -> None:
    """Safely add one already-verified staging tree to ``live``.

    Inductor/Triton paths are cache-KEY-addressed, not content-addressed
    across machines (pgw#699/#711 respec, pgw#751): same-key members are
    byte-divergent between producers (embedded paths, codegen
    nondeterminism), so an existing path with different bytes is the SAME
    cache entry, not a conflict — the LOCAL copy wins (it may already be
    mmapped/served by this process, and torch's own consumption is keyed,
    so serving semantics are identical) and the merge stays additive. The
    live 7-of-13 ``adopt_failed:cache_collision`` epidemic was exactly
    this: any pod that had compiled anything before delivery could never
    install a cell. A structural conflict (a directory where a file is
    expected) still refuses typed.

    New files become visible one at a time via ``os.replace`` (there is no
    portable whole-directory union swap), but the process lock prevents
    normal arming consumers from observing that interval. An in-process
    failure removes every newly added file. A process crash can leave only
    complete, verified new files; replay treats those as identical and
    finishes the additive merge.
    """
    files = sorted(
        path
        for sub in ("inductor", "triton")
        for path in (staged / sub).rglob("*")
        if path.is_file()
    )
    additions: list[tuple[Path, Path]] = []
    local_wins: list[str] = []
    for source in files:
        target = live / source.relative_to(staged)
        if target.exists():
            if not target.is_file():
                raise AdoptError(
                    "cache_collision",
                    f"verified cache path {source.relative_to(staged)!s} "
                    "exists locally as a non-file — structural conflict",
                )
            if not filecmp.cmp(source, target, shallow=False):
                local_wins.append(str(source.relative_to(staged)))
            continue
        additions.append((source, target))
    if local_wins:
        logger.info(
            "cell merge: %d same-key byte-divergent member(s) kept LOCAL "
            "(pgw#751 — bytes are not the identity; first: %s)",
            len(local_wins), ", ".join(local_wins[:3]))

    live.mkdir(parents=True, exist_ok=True)
    added: list[Path] = []
    try:
        for source, target in additions:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            added.append(target)
    except BaseException:
        for target in reversed(added):
            target.unlink(missing_ok=True)
        raise


@dataclass
class _StagedArtifact:
    metadata: Dict[str, Any]
    staged_root: Path
    live_root: Path
    temporary: tempfile.TemporaryDirectory[str]
    activated: bool = False

    def close(self) -> None:
        self.temporary.cleanup()


def stage_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None,
) -> _StagedArtifact:
    """Extract and validate an artifact without touching process-global state."""
    root = (Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker")
    root = root / "compile-cache"
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix="compile-cache-stage-", dir=root.parent,
    )
    staged = Path(temporary.name) / "cache"
    try:
        meta = unpack(Path(artifact), staged)
        reason = verify(meta, family=family)
        if reason:
            raise AdoptError("key_mismatch", reason)
        return _StagedArtifact(meta, staged, root, temporary)
    except AdoptError:
        temporary.cleanup()
        raise
    except Exception as exc:
        temporary.cleanup()
        raise AdoptError("artifact_invalid", str(exc)) from exc


def _activate_staged(staged: _StagedArtifact) -> Dict[str, Any]:
    """Publish a verified staging tree while holding ``_SEED_ARM_LOCK``."""
    global _DELIVERED_SEEDED
    if not staged.activated:
        _merge_staged_cache(staged.staged_root, staged.live_root)
        staged.activated = True
    seed_env(staged.live_root)
    # gw#608: the process now serves from a seeded delivered cell; any later
    # self-mint capture would re-point the ONE process-global cache dir away
    # from it and every seeded lookup would miss (the LTX consumer shape).
    _DELIVERED_SEEDED = True
    return staged.metadata


def seed_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Verify in isolation, then seed one artifact under the process lock.

    A malformed, unsafe, corrupt, or runtime-mismatched tar never writes into
    the live Inductor/Triton cache. Returns metadata or raises
    :class:`AdoptError` without changing the live tree.
    """
    staged = stage_artifact(artifact, family, cache_dir=cache_dir)
    try:
        try:
            with _SEED_ARM_LOCK:
                return _activate_staged(staged)
        except AdoptError:
            raise
        except Exception as exc:
            raise AdoptError("activation_failed", str(exc)) from exc
    finally:
        staged.close()


def mode_drift(meta: Dict[str, Any], pipeline: Any) -> str:
    """'' when the producer's low-VRAM prep mode matches this pipeline's, else
    the mismatch (gw#391). The prep flags (VAE tiling/slicing, attention
    slicing, offload hooks) are traced into the FX graphs, so a mode drift is
    a guaranteed cache miss. Enforced only when the producer recorded one —
    the check is per-pipeline, so it lives outside :func:`verify`."""
    want = str(meta.get("low_vram_mode") or "")
    if not want:
        return ""

    have = low_vram_mode(pipeline)
    if want != have:
        return f"low_vram_mode {want!r} != pipeline {have!r}"
    return ""


def apply_lora_execution_lane(pipeline: Any, bucket: int) -> bool:
    """Put the pipeline on the branch-bearing graph family for ``bucket``
    (gw#561): canonical zeroed rank-``bucket`` branches on every
    branch-capable denoiser Linear (the gw#547 compiled-lane contract) + the
    ``<base>-lora<bucket>`` lane stamp, so :func:`lane_drift` admits exactly
    the matching lora cells. Raises when the pipeline has no branch-capable
    denoiser — a declared bucket that cannot trace must fail loud, not
    publish/adopt the wrong graph.

    gw#679: the container is allocated on EVERY denoiser the pipeline
    carries, so a dual-expert MoE traces both experts branch-bearing and a
    per-expert adapter set can land at request time without a recompile."""
    if not bucket:
        return False

    targets = w8a8_lora.enable_branch_execution_lanes(pipeline, int(bucket))
    if not targets:
        raise RuntimeError(
            "Compile.lora_bucket declared but the pipeline has no "
            "branch-capable denoiser (transformer/transformer_2/unet)"
        )
    w8a8_lora.stamp_execution_lane(pipeline, targets)
    return True


def drop_lora_execution_lane(pipeline: Any) -> None:
    """Undo :func:`apply_lora_lane`: drop the branch buffers on every
    denoiser and restore the branchless lane stamp (the eager rollback —
    canonical zeroed branches cost +21-32% eager, gw#547)."""

    targets = w8a8_lora.branch_targets(pipeline)
    if not targets:
        return
    w8a8_lora.disable_branch_execution_lanes(pipeline)
    w8a8_lora.stamp_execution_lane(pipeline, targets)


def execution_lane_drift(meta: Dict[str, Any], pipeline: Any) -> str:
    """'' when the cell's traced weight lane matches this pipeline's, else the
    mismatch (gw#534). Enforced SYMMETRICALLY (unlike ``mode_drift``): a
    bf16-resident pipeline must never adopt hook-cast-traced graphs and vice
    versa — both directions are guaranteed FX-graph misses that would serve
    eager while reporting adopted (the gw#391 bug class)."""
    want = str(meta.get("weight_lane") or "")

    have = pipeline_weight_lane(pipeline)
    if want != have:
        return f"weight_lane {want!r} != pipeline {have!r}"
    return ""


def prepare(
    family: str,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Verify and seed one explicitly delivered artifact for this runtime.

    Production obtains ``artifact`` from Tensorhub's immutable RunJob/
    DesiredInstance snapshot attachment. Local tooling passes an explicit path
    or uses the explicit local-cell store; environment fallbacks deliberately
    do not participate in serving placement.
    """
    try:
        if artifact is None:
            logger.info("compile-cache: no delivered artifact; staying eager")
            return None
        artifact = Path(artifact)
        if not artifact.exists():
            logger.warning("compile-cache: attached artifact %s does not exist", artifact)
            return None
        meta = seed_artifact(artifact, family, cache_dir=cache_dir)
        logger.info(
            "compile-cache: seeded verified artifact (sku=%s torch=%s shapes=%s)",
            meta.get("sku"), meta.get("torch"), meta.get("shapes"),
        )
        return meta
    except Exception as exc:
        logger.warning("compile-cache: artifact unusable (%s); staying eager", exc)
        return None


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _resolve_target(pipeline: Any, target: str) -> Optional[Tuple[Any, str, Callable[..., Any]]]:
    """``"transformer"`` -> (module, 'forward', fn); ``"vae.decode"`` ->
    (vae, 'decode', fn). None when the pipeline has no such attribute."""
    obj = pipeline
    parts = target.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    leaf = getattr(obj, parts[-1], None)
    if leaf is None:
        return None
    if callable(getattr(leaf, "forward", None)) and parts[-1] != "forward":
        # a Module: compile its bound forward
        return leaf, "forward", leaf.forward
    if callable(leaf):
        return obj, parts[-1], leaf
    return None


class CompileArmRefused(RuntimeError):
    """A NAMED, deterministic reason this process cannot arm this pipeline.

    pgw#985: what decides whether a second pod gets bought is the
    CLASSIFICATION, not the message. ``begin_fleet_mint`` used to raise a bare
    ``RuntimeError`` for every decline — which the mint child let out as exit
    1 (``CRASHED``, retryable) while the AOT recipe typed the identical
    condition as a refusal (``EXIT_REFUSED``, terminal). Same fact, two
    vocabularies, and the retryable one billed a second mint that could not
    possibly succeed.
    """


def resolve_targets(
    pipeline: Any, cfg: Any,
) -> List[Tuple[str, Any, str, Callable[..., Any]]]:
    """The declared targets ``pipeline`` actually OWNS — the ONE authority.

    ``(declared name, owner, attribute, eager callable)`` per resolvable
    target, in declaration order. :func:`has_compile_target`, :func:`apply`
    and :func:`begin_fleet_mint` all read THIS list (§1.29, one relation) —
    it used to be scanned independently by the first two, which is how a
    reader of the third could not tell which scan had spoken.

    Whether the pipeline OWNS a declared target is the only question answered
    here. Whether this process can ARM the targets it owns is a different
    question with a different answer, and :func:`arming_block` owns it;
    conflating the two is what let a cardless mint pod report "no compile
    targets resolved on TinyDiffusionPipeline" about a pipeline whose
    ``.unet`` had resolved a frame earlier (pgw#985).
    """
    out: List[Tuple[str, Any, str, Callable[..., Any]]] = []
    for target in tuple(getattr(cfg, "targets", ()) or ()):
        resolved = _resolve_target(pipeline, str(target))
        if resolved is None:
            continue
        owner, attr, fn = resolved
        out.append((str(target), owner, attr, fn))
    return out


def has_compile_target(pipeline: Any, cfg: Any) -> bool:
    """Whether ``pipeline`` owns at least one callable declared by ``cfg``.

    A setup may inject support objects (for example SDXL's standalone VAE)
    alongside the actual pipeline. Only the object whose graph targets resolve
    is a compile-adoption target; family-wide scans must not try to wrap every
    resident model object.
    """
    return bool(resolve_targets(pipeline, cfg))


def arming_block(
    pipeline: Any, cfg: Any, *, cache_ready: bool, allow_cold: bool,
) -> str:
    """Why :func:`apply` would decline to arm — ``""`` when nothing blocks it.

    The ONE precondition authority, for the same reason
    :func:`resolve_targets` is the one target authority. Every reason here is
    deterministic for the life of this process: none of them can differ on a
    retry, so a caller that must classify (the mint child) can refuse on any
    of them without losing a mint a second attempt would have made.

    Deliberately side-effect free — :func:`apply` still owns the arming
    mutations; this only names.
    """
    if _PROCESS_COMPILES_DISABLED:
        return f"process compiles are disabled: {_PROCESS_COMPILES_DISABLED}"
    if operator_eager_pin(pipeline):
        return ("the hub-resolved execution lane is operator-pinned to +eager "
                "(pgw#714 kill switch)")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — a torchless process is eager
        return f"torch is not importable ({type(exc).__name__}: {exc})"
    if not torch.cuda.is_available():
        return "torch reports no CUDA device in this process"
    if cache_ready:
        return ""
    if not allow_cold:
        return "no verified cache artifact was seeded and cold compile was not requested"
    if not toolchain_present():
        return "cold compile was requested but no C compiler is on PATH"
    return ""


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _direct_tensor_schema(module: Any) -> list[list[Any]]:
    """Names/shapes/dtypes only; tensor values and checkpoint IDs stay out."""
    rows: list[list[Any]] = []
    for kind, method in (
        ("parameter", getattr(module, "named_parameters", None)),
        ("buffer", getattr(module, "named_buffers", None)),
    ):
        if not callable(method):
            continue
        try:
            tensors = method(recurse=False)
        except TypeError:
            tensors = method()
        for name, tensor in tensors:
            rows.append([
                kind, str(name), [int(v) for v in getattr(tensor, "shape", ())],
                str(getattr(tensor, "dtype", "")),
            ])
    return sorted(rows)


def _module_hooks(module: Any) -> Dict[str, int]:
    """Hook PRESENCE per module (pgw#697): installed hooks are traced into
    the compiled graphs (the offload rung's windows, and — until pgw#727
    restructured it into module types — the fp8 cast, ie#381), so a
    hook-count drift is a composition drift. Counts only — never hook
    identities or closures."""
    out: Dict[str, int] = {}
    for fact, attr in (
        ("forward_pre", "_forward_pre_hooks"),
        ("forward", "_forward_hooks"),
        ("backward", "_backward_hooks"),
        ("full_backward", "_full_backward_hooks"),
    ):
        hooks = getattr(module, attr, None)
        count = len(hooks) if hooks is not None else 0
        if count:
            out[fact] = count
    return out


def _module_entry(path: str, module: Any) -> Dict[str, Any]:
    """One module's composition facts: class, tensor schema, hook presence.
    Exactly what the graph signature hashes and what the pgw#697 per-module
    fingerprint digests — one builder so the two can never disagree."""
    entry: Dict[str, Any] = {
        "path": path,
        "type": _type_name(module),
        "tensors": _direct_tensor_schema(module),
    }
    hooks = _module_hooks(module)
    if hooks:
        entry["hooks"] = hooks
    return entry


def composition_fingerprint(pipeline: Any, cfg: Any) -> list[Tuple[str, str]]:
    """``(path, digest)`` per resolved-target module — the pgw#697 adoption
    fence. Digests the same per-module facts the graph signature hashes, so
    a signature mismatch resolves to the exact drifted module (the pgw#683
    class: one submodule left in Half inside a bf16 tree died as a raw
    matmul RuntimeError — this names ``path: cell x != consumer y``
    instead). Fine-tunes stay shared: no tensor VALUES enter any row."""
    rows: list[Tuple[str, str]] = []
    for target in tuple(getattr(cfg, "targets", ()) or ()):
        resolved = _resolve_target(pipeline, str(target))
        if resolved is None:
            continue
        owner, _attr, _fn = resolved
        named = getattr(owner, "named_modules", None)
        module_rows = list(named()) if callable(named) else [("", owner)]
        for name, module in module_rows:
            path = f"{target}:{name}" if name else str(target)
            encoded = json.dumps(
                _module_entry(path, module), sort_keys=True,
                separators=(",", ":"),
            ).encode()
            rows.append((path, hashlib.sha256(encoded).hexdigest()[:16]))
    return sorted(rows)


def _first_composition_difference(
    cell_rows: Iterable[Iterable[str]], here_rows: Iterable[Tuple[str, str]],
) -> str:
    """Name the first module whose composition digest differs (or that only
    one side has)."""
    cell = {str(p): str(d) for p, d in cell_rows}
    here = {str(p): str(d) for p, d in here_rows}
    for path in sorted(set(cell) | set(here)):
        want, have = cell.get(path), here.get(path)
        if want == have:
            continue
        if want is None:
            return f"module {path!r} exists only in the consumer pipeline"
        if have is None:
            return f"module {path!r} exists only in the cell"
        return f"{path}: cell composition {want} != consumer {have}"
    return ""


def execution_contract(pipeline: Any, cfg: Any) -> Tuple[str, Dict[str, Any]]:
    """Canonical family-graph and weight-lane contract for one loaded model.

    Fine-tunes with the same module graph produce the same result: no ref,
    tag, source/checkpoint digest or tensor value is read. A structural
    SDXL/Pony/Illustrious incompatibility (different module class/shape or
    different scaled-mm exclusion surface) produces a different signature
    and is rejected before adoption.

    The signature hashes ONLY the traced module structure — the resolved
    targets' module types, paths, tensor schemas (param shapes/dtypes,
    buffer dtypes) and hook presence (pgw#697) — never the wrapping
    pipeline class (gw#577): torch.compile wraps target callables such as
    ``transformer.forward``; the pipeline class never enters any traced
    graph. Conversion producers load via generic ``DiffusionPipeline``
    (model_index -> e.g. LTX2Pipeline) while serving loads the endpoint's
    declared wrapper (e.g. LTX2ConditionPipeline) over a byte-identical
    module tree — proven identical-graph, and must share one cell.
    """
    graph_targets: list[Dict[str, Any]] = []
    quantized: list[Dict[str, Any]] = []
    excluded: list[Dict[str, Any]] = []
    seen_modules: set[int] = set()

    for target in tuple(getattr(cfg, "targets", ()) or ()):
        resolved = _resolve_target(pipeline, str(target))
        if resolved is None:
            graph_targets.append({"target": str(target), "missing": True})
            continue
        owner, attr, _fn = resolved
        modules: list[Dict[str, Any]] = []
        named = getattr(owner, "named_modules", None)
        module_rows = list(named()) if callable(named) else [("", owner)]
        for name, module in module_rows:
            path = f"{target}:{name}" if name else str(target)
            modules.append(_module_entry(path, module))
            # A target such as vae.decode can overlap another declaration;
            # record each module once in the W8A8 manifest.
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)
            if not isinstance(in_features, int) or not isinstance(out_features, int):
                continue
            row = {
                "path": path,
                "in_features": int(in_features),
                "out_features": int(out_features),
            }
            if bool(getattr(module, "_cozy_w8a8_linear", False)):
                # gw#564: the activation-scale granularity is a graph property
                # (per-row rowwise sm_90+, per-tensor epilogue sm_89).
                if getattr(module, "input_scale", None) is not None:
                    row["activation"] = "static"
                elif getattr(module, "gemm_mode", "") == "pertensor":
                    row["activation"] = "dynamic-per-tensor"
                else:
                    row["activation"] = "dynamic-per-row"
                quantized.append(row)
            elif bool(getattr(module, "_cozy_w4a4_linear", False)):
                # gw#540: block scales are always dynamic per-16-block; the
                # graph property is the second-level activation scale mode.
                row["activation"] = (
                    "static" if getattr(module, "input_scale", None)
                    is not None else "dynamic-per-tensor")
                if getattr(module, "pre_quant_scale", None) is not None:
                    row["pre_quant_scale"] = True
                quantized.append(row)
            else:
                row["type"] = _type_name(module)
                excluded.append(row)
        graph_targets.append({
            "target": str(target), "attr": str(attr), "modules": modules,
        })

    graph = {
        "targets": graph_targets,
    }
    encoded = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()

    execution_lane = pipeline_weight_lane(pipeline)
    weight_contract: Dict[str, Any] = {"lane": execution_lane}
    if execution_lane.startswith(("w8a8", "w4a4")):
        activations = sorted({str(r["activation"]) for r in quantized})
        weight_contract.update({
            "artifact_schema": (
                "nvfp4-w4a4-v1" if execution_lane.startswith("w4a4") else "fp8-w8a8-v1"),
            "operator": "torch._scaled_mm",
            "weight_scaling": (
                "per-16-block+per-tensor" if execution_lane.startswith("w4a4")
                else "per-output-channel"),
            "activation_scaling": activations,
            "quantized": sorted(quantized, key=lambda r: str(r["path"])),
            "excluded": sorted(excluded, key=lambda r: str(r["path"])),
        })
    return hashlib.sha256(encoded).hexdigest(), weight_contract


def _regional_dynamic_decline(cfg: Any, target: str) -> str:
    """"" when the DYNAMO regional branch may arm ``target``, else the reason.

    pgw#817/D4 moved the `regional + dynamic` refusal out of the declaration
    (where it forbade a combination the EXPORT lane measured as free) and into
    the one lane that genuinely cannot honour it. `compile_repeated_blocks(
    dynamic=None)` never applies the declared marks, so a dynamo regional arm
    over a declaration carrying `dynamic=(...)` would serve a graph that does
    not implement the contract its cell key asserts — the exact failure class
    pgw#716 exists to prevent. Declining here sends the target to the
    whole-forward branch, which DOES mark, so the declaration is still served;
    it is only served by the other lane.
    """
    dyn = tuple(getattr(cfg, "dynamic", ()) or ())
    if not dyn:
        return ""
    names = ", ".join(str(getattr(d, "dim", "") or "?") for d in dyn)
    return (
        f"target {target!r} declares regional=True AND dynamic=({names}) — "
        f"the dynamo regional branch calls compile_repeated_blocks("
        f"dynamic=None) and never applies the declared marks, so it declines "
        f"and this target takes the whole-forward branch (which does). The "
        f"AOT export lane implements regional+dynamic directly (pgw#812 "
        f"RESULT 3: free on a conv-free region)")


def _apply_declared_shape_config(cfg: Any) -> None:
    """The v2 dynamo posture: nothing becomes dynamic by accident.

    ``automatic_dynamic_shapes=False`` — never promote a dim on change (a
    novel signature is a guard miss routed by the consumer guards, never a
    silent recompile-to-dynamic); ``assume_static_by_default=True`` —
    unmarked dims are static. Declared dynamism arrives ONLY through
    explicit ``mark_dynamic`` marks (``_with_declared_marks``)."""
    try:
        import torch._dynamo

        torch._dynamo.config.automatic_dynamic_shapes = False
        torch._dynamo.config.assume_static_by_default = True
    except Exception:
        logger.debug("compile-cache: could not set dynamo shape config",
                     exc_info=True)


def _with_declared_marks(fn: Callable[..., Any], dynamic_dims: tuple) -> Callable[..., Any]:
    """Wrap a compiled callable so every call marks the DECLARED dynamic
    dims on its tensor inputs before dynamo sees them.

    Mapping of logical axes to tensor dims: ``batch`` marks dim 0 of every
    floating tensor argument; ``sequence`` marks dim 1 of rank-3 floating
    tensors (the ``[B, seq, hidden]`` conditioning shape). A dim smaller
    than the declared ``min`` is left unmarked — torch's 0/1 specialization
    is not overridable (ie#543) and gets its own free specialized graph. A
    mark torch cannot honor raises ``ConstraintViolationError`` at
    compile/warm time — a LOUD build failure, never a silent fallback to
    recompilation (the mint's warm calls do not guard)."""

    import torch

    def _mark(t: Any) -> None:
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return
        for d in dynamic_dims:
            # Only the two logical dynamo axes are markable here. A named
            # declared Dim (pgw#739) carries (input, axis) bindings and is
            # the EXPORT lane's business — marking it at axis 1 by the old
            # sequence heuristic would mark the wrong axis silently.
            if d.dim == "batch":
                dim = 0
            elif d.dim == "sequence" and t.dim() >= 3:
                dim = 1
            else:
                continue
            if t.dim() <= dim:
                continue
            if int(t.shape[dim]) < int(d.min):
                continue  # 0/1 (and sub-min) sizes keep their free static graph
            torch._dynamo.mark_dynamic(t, dim, min=int(d.min), max=int(d.max))

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for a in args:
            _mark(a)
        for v in kwargs.values():
            _mark(v)
        return fn(*args, **kwargs)

    return wrapper


def execution_contract_digest(pipeline: Any, cfg: Any) -> str:
    """Digest every graph-compatibility axis enforced by the consumer.

    ``execution_contract()[0]`` is intentionally only the module-graph
    signature. Scheduler fencing needs the complete contract: declared graph
    shapes/targets/CFG classes, whole-vs-regional mode, actual weight lane and
    activation-scaling schema, LoRA bucket, and observed low-VRAM preparation.
    Tensor values and checkpoint identities remain excluded so compatible
    fine-tunes share one family cell.
    """
    graph_signature, weight_contract = execution_contract(pipeline, cfg)

    text_lens = tuple(getattr(cfg, "text_lens", ()) or ())
    if not text_lens and getattr(cfg, "text_len", None) is not None:
        text_lens = (int(cfg.text_len),)
    payload = {
        "version": 3,
        "family": str(getattr(cfg, "family", "") or ""),
        "shapes": sorted(
            [int(v) for v in row] for row in getattr(cfg, "shapes", ())
        ),
        "targets": [str(v) for v in getattr(cfg, "targets", ())],
        "guidance_scales": [
            float(v) for v in getattr(cfg, "guidance_scales", ())
        ],
        # pgw#654 gap #6: sibling lanes with different text pins share one
        # cell — the digest carries the class UNION, never one lane's pin.
        "text_lens": sorted({int(v) for v in text_lens}),
        "dynamic": [
            {"dim": d.dim, "min": d.min, "max": d.max}
            for d in getattr(cfg, "dynamic", ())
        ],
        "compile_mode": (
            "regional" if bool(getattr(cfg, "regional", False)) else "whole"
        ),
        "lora_bucket": int(getattr(cfg, "lora_bucket", 0) or 0),
        "low_vram_mode": low_vram_mode(pipeline),
        "graph_signature": graph_signature,
        "weight_contract": weight_contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _first_contract_difference(cell: Dict[str, Any], here: Dict[str, Any]) -> str:
    """Name the first differing weight-contract key with compact values, so a
    refusal is diagnosable from the reason alone (gw#577)."""
    for key in sorted(set(cell) | set(here)):
        want, have = cell.get(key), here.get(key)
        if want == have:
            continue
        if isinstance(want, list) and isinstance(have, list):
            return (
                f"{key}: cell has {len(want)} row(s), consumer {len(have)}; "
                f"first cell-only rows {[r for r in want if r not in have][:2]!r} "
                f"vs consumer-only {[r for r in have if r not in want][:2]!r}"
            )
        return f"{key}: cell {want!r} != consumer {have!r}"
    return "identical keys, differing encoding"


def contract_drift(meta: Dict[str, Any], pipeline: Any, cfg: Any) -> str:
    """Mismatch between the cell's declared graph and the loaded consumer."""
    shapes = sorted(
        [int(v) for v in row] for row in getattr(cfg, "shapes", ()))
    cell_shapes = sorted(
        [int(v) for v in row] for row in (meta.get("shapes") or ()))
    if cell_shapes != shapes:
        return f"shapes {cell_shapes!r} != declared {shapes!r}"
    targets = [str(v) for v in getattr(cfg, "targets", ())]
    if meta.get("targets") != targets:
        return f"targets {meta.get('targets')!r} != declared {targets!r}"
    guidance_scales = [float(v) for v in getattr(cfg, "guidance_scales", ())]
    cell_guidance_scales = [float(v) for v in (meta.get("guidance_scales") or ())]
    if cell_guidance_scales != guidance_scales:
        return (
            f"guidance_scales {cell_guidance_scales!r} != declared "
            f"{guidance_scales!r}"
        )
    # SDK v2: the recorded shape contract must be the declared one — a
    # worker on a newer contract must never serve an older cell (pgw#647).
    #
    # NOT CUT by pgw#950, deliberately: a cell recording NO shape_contract is
    # skipped here, which is the same silent-axis shape as the arm below, but
    # ~9 test fixtures build metadata without one (every PRODUCTION mint passes
    # ``shape_contract=declared_contract_facts(cfg)``, so the gap is fixtures,
    # not producers). Tightening it is a fixture sweep, not a deletion, so it
    # is its own change.
    cell_contract = meta.get("shape_contract") or {}
    if cell_contract:
        here_contract = declared_contract_facts(cfg)
        if cell_contract != here_contract:
            return (
                "shape contract mismatch: "
                + _first_contract_difference(cell_contract, here_contract)
            )
    signature, weight_contract = execution_contract(pipeline, cfg)
    meta_signature = str(meta.get("graph_signature") or "")
    meta_weights = meta.get("weight_contract") or {}
    if not meta_signature:
        # pgw#950: this used to ``return ""`` — COMPATIBLE — for a cell silent
        # on both graph_signature and weight_contract on a non-quantized lane
        # ("legacy format-2"). Every production mint passes a signature (
        # ``execution_contract`` always digests a structure, never ""), so the
        # arm only ever admitted pre-format-3 cells, and admitting one is a
        # wrong cache hit on the module graph itself.
        return "cell records no graph_signature (pre-format-3 cell)"
    if meta_signature != signature:
        # pgw#697: when the cell carries per-module fingerprint rows, name
        # the exact drifted module instead of two digest prefixes.
        cell_rows = meta.get("composition") or []
        if cell_rows:
            named = _first_composition_difference(
                cell_rows, composition_fingerprint(pipeline, cfg))
            if named:
                return f"module composition: {named}"
        return (
            f"module graph signature: cell {meta_signature[:12]!r} != "
            f"consumer {signature[:12]!r}"
        )
    if meta_weights != weight_contract:
        return (
            "weight-lane artifact schema/exclusion manifest mismatch: "
            + _first_contract_difference(meta_weights, weight_contract)
        )
    cell_execution_lane_base = str(weight_contract.get("lane") or "")
    if cell_execution_lane_base.startswith(("w8a8", "w4a4")):
        activations = weight_contract.get("activation_scaling") or []
        if cell_execution_lane_base.startswith("w8a8"):
            # DYNAMIC only, one homogeneous granularity per graph (gw#564:
            # per-row = rowwise sm_90+, per-tensor = the sm_89 epilogue lane).
            if activations not in (["dynamic-per-row"], ["dynamic-per-tensor"]):
                return (f"W8A8 activation scaling must be dynamic "
                        f"(per-row or per-tensor), got {activations!r}")
        else:
            # gw#540: one homogeneous second-level activation scale mode per
            # graph (static = calibrated input_scale, the production mode).
            if activations not in (["static"], ["dynamic-per-tensor"]):
                return (f"W4A4 activation scaling must be homogeneous "
                        f"static or dynamic-per-tensor, got {activations!r}")
        if not weight_contract.get("quantized"):
            return (f"{cell_execution_lane_base[:4].upper()} graph contains no "
                    "torch._scaled_mm modules")
        here_digest = runtime_key()["image_digest"]
        # cuda_driver excluded (gw#577): host-lottery axis, see verify().
        for field in ("sm", "cuda", "image_digest"):
            if not str(meta.get(field) or ""):
                if field == "image_digest" and not here_digest:
                    # Bare-metal local runtime (gw#555 self-mint): no image
                    # identity axis exists on either side. Production images
                    # always carry WORKER_IMAGE_DIGEST, so fleet cells stay
                    # fully pinned.
                    continue
                return f"quantized-lane cell missing {field} identity"
    return ""


def _guarded(
    original: Callable[..., Any], compiled: Callable[..., Any], label: str,
    *, fail_closed: bool = False,
    failure_signal: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Guard one exact compiled callable and record its own warm-call proof.

    The process-wide Dynamo counters are sampled *inside this wrapper* around
    this object's compiled call.  Executor adoption therefore cannot use a
    cache hit produced by a different resident pipeline as proof for this one.
    """
    state: Dict[str, Any] = {
        "failed": False,
        "detail": "",
        "revocation_error": "",
    }

    def revoke(detail: str) -> None:
        callback = (failure_signal or {}).get("callback")
        if callable(callback):
            try:
                callback(detail)
            except Exception as exc:
                state["revocation_error"] = (
                    "compiled-state revocation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception("compile-cache: %s", state["revocation_error"])
                raise CompiledExecutionLaneUnavailableError(
                    state["revocation_error"]
                ) from exc

    def proof_before() -> Optional[Dict[str, int]]:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return None
        # Inductor counts graph lookup/compile hits, not every execution of an
        # already-loaded graph. Capture activation once for this exact wrapper;
        # successful_calls below remains a per-invocation alias proof. Executor
        # proof warmups exclude concurrent GPU work, so the process-wide delta
        # cannot come from another resident object.
        with lock:
            if int(signal.get("cache_hits", 0)) > 0:
                return None
        return inductor_counters()

    def record_success(before: Optional[Dict[str, int]]) -> None:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return
        stats = counters_delta(before, inductor_counters()) if before is not None else {}
        with lock:
            signal["successful_calls"] = int(signal.get("successful_calls", 0)) + 1
            # gw#611: an AOT-layer hit serves the artifact without an
            # FxGraphCache lookup (bundled mode: fxgraph counters silent);
            # it is serving evidence, never a disproof.
            signal["cache_hits"] = int(signal.get("cache_hits", 0)) + max(
                0, int(stats.get("fxgraph_cache_hit", 0))) + max(
                0, int(stats.get("aot_cache_hit", 0)))
            signal["cache_misses"] = int(signal.get("cache_misses", 0)) + max(
                0, int(stats.get("fxgraph_cache_miss", 0)))

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledExecutionLaneUnavailableError(state["revocation_error"])
        if state["failed"]:
            return original(*args, **kwargs)
        # pgw#622: a novel input signature serves EAGER immediately while a
        # background thread warms the compiled path, then hot-swaps.
        router = (failure_signal or {}).get("router")
        sig = None
        if router is not None:
            verdict, sig = router.route(label, compiled, args, kwargs)
            if verdict == "eager":
                return original(*args, **kwargs)
        before = proof_before()
        try:
            # pgw#680: tenant serve windows run fail-on-recompile — a guard
            # miss raises instead of paying dynamo's inline recompile in the
            # request. Warm cache entries serve normally under the stance;
            # proof/warm/mint windows never arm it (see _fail_on_recompile).
            with _fail_on_recompile():
                result = compiled(*args, **kwargs)
            record_success(before)
            if router is not None:
                router.mark_warm(sig)
            return result
        except Exception as exc:  # noqa: BLE001 — every lane degrades to eager
            if _is_recompile_error(exc):
                # pgw#680 catch: the compiled lane is HEALTHY for its known
                # input classes — this request's class missed. Serve it eager
                # now, confess loudly, heal the exact class in background.
                # Never the permanent-degrade path below.
                _record_guard_miss(
                    label, exc, args, kwargs, failure_signal, compiled)
                return original(*args, **kwargs)
            state["failed"] = True
            state["detail"] = (
                f"compiled {'W8A8 ' if fail_closed else ''}target {label} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            # Revoke scheduler-visible compiled proof synchronously before the
            # eager fallback: the tier flips to explicit eager on the wire.
            revoke(state["detail"])
            # pgw#672/pgw#673 posture: a broken optimization must never kill a
            # serving worker. Mandatory lanes used to raise here (and the
            # setup/dispatch paths then disabled every declared function —
            # sm120 CantSplit retired the pod for $0.25 of nothing). They now
            # degrade like every other lane, LOUDLY: the revocation above is
            # the wire-visible tier flip, never silent eager (gw#586).
            log = logger.error if fail_closed else logger.warning
            log(
                "compile-cache: compiled %s failed (%s: %s); serving eager for "
                "the rest of this process%s", label, type(exc).__name__, exc,
                " (mandatory lane DEGRADED, pgw#672)" if fail_closed else "",
            )
            return original(*args, **kwargs)

    return wrapper


def _clear_regional(mod: Any) -> None:
    """Undo nn.Module.compile() on every submodule (regional rollback)."""
    for m in mod.modules():
        if getattr(m, "_compiled_call_impl", None) is not None:
            m._compiled_call_impl = None


def _guarded_regional(
    mod: Any,
    original: Callable[..., Any],
    label: str,
    *,
    fail_closed: bool = False,
    failure_signal: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Regional analogue of :func:`_guarded`: blocks are compiled in place,
    so eager fallback must first CLEAR the block compilations, then retry."""
    state: Dict[str, Any] = {
        "failed": False,
        "detail": "",
        "revocation_error": "",
    }

    def proof_before() -> Optional[Dict[str, int]]:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return None
        # See _guarded: one exact-object activation is sufficient; subsequent
        # aliases still need their own successful wrapper invocation.
        with lock:
            if int(signal.get("cache_hits", 0)) > 0:
                return None
        return inductor_counters()

    def record_success(before: Optional[Dict[str, int]]) -> None:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return
        stats = counters_delta(before, inductor_counters()) if before is not None else {}
        with lock:
            signal["successful_calls"] = int(signal.get("successful_calls", 0)) + 1
            # gw#611: an AOT-layer hit serves the artifact without an
            # FxGraphCache lookup (bundled mode: fxgraph counters silent);
            # it is serving evidence, never a disproof.
            signal["cache_hits"] = int(signal.get("cache_hits", 0)) + max(
                0, int(stats.get("fxgraph_cache_hit", 0))) + max(
                0, int(stats.get("aot_cache_hit", 0)))
            signal["cache_misses"] = int(signal.get("cache_misses", 0)) + max(
                0, int(stats.get("fxgraph_cache_miss", 0)))

    def _eager_once(args: tuple, kwargs: dict) -> Any:
        """One fully-eager call of the in-place-compiled module (pgw#680).

        Regional blocks have no separable eager callable — the compiled
        impls live ON the blocks — so the guard-miss eager serve runs the
        original with dynamo disabled for this thread/call: existing
        compiled entries are bypassed, nothing recompiles, block state is
        untouched (verified on torch 2.13: ``config.disable`` is the same
        thread-local ContextVar surface as the stance)."""
        try:
            import torch._dynamo

            patch = torch._dynamo.config.patch(disable=True)
        except Exception:
            return original(*args, **kwargs)
        with patch:
            return original(*args, **kwargs)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledExecutionLaneUnavailableError(state["revocation_error"])
        if not state["failed"]:
            before = proof_before()
            try:
                # pgw#680: the compiled block impls execute inside this
                # call — the serve-window stance covers them here.
                with _fail_on_recompile():
                    result = original(*args, **kwargs)
                record_success(before)
                return result
            except Exception as exc:  # noqa: BLE001 — every lane degrades to eager
                if _is_recompile_error(exc):
                    _record_guard_miss(
                        label, exc, args, kwargs, failure_signal, original)
                    return _eager_once(args, kwargs)
                state["failed"] = True
                state["detail"] = (
                    f"regional compiled {'W8A8 ' if fail_closed else ''}"
                    f"target {label} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Regional eager state is real only after the in-place block
                # compilations are gone. Revoke proof after that mutation and
                # before a state delta can be scheduled.
                _clear_regional(mod)
                callback = (failure_signal or {}).get("callback")
                if callable(callback):
                    try:
                        callback(state["detail"])
                    except Exception as callback_exc:
                        state["revocation_error"] = (
                            "compiled-state revocation failed: "
                            f"{type(callback_exc).__name__}: {callback_exc}"
                        )
                        logger.exception(
                            "compile-cache: %s", state["revocation_error"])
                        raise CompiledExecutionLaneUnavailableError(
                            state["revocation_error"]
                        ) from callback_exc
                # pgw#672/pgw#673 posture: mandatory lanes degrade to explicit
                # eager (revocation above flips the wire tier) instead of
                # raising — a broken optimization never kills serving.
                log = logger.error if fail_closed else logger.warning
                log(
                    "compile-cache: regional-compiled %s failed (%s: %s); "
                    "eager for the rest of this process%s",
                    label, type(exc).__name__, exc,
                    " (mandatory lane DEGRADED, pgw#672)" if fail_closed
                    else "",
                )
        return original(*args, **kwargs)

    return wrapper


def _vae_supports_channels_last(vae: Any) -> bool:
    """True only when every VAE weight is rank<=4 (2D convs). channels_last
    is a rank-4 memory format; rank-5 Conv3d weights (causal/video VAEs)
    raise on it (gw#574)."""
    try:
        return all(p.dim() <= 4 for p in vae.parameters())
    except Exception:
        return False


#: The hub-resolved execution-lane descriptor for the checkpoint this
#: pipeline serves (th#913 ``lane`` string), stamped by the executor at
#: injection time. Consumed by :func:`mandatory_serving` only — cell keys
#: keep the weight-lane brain (pgw#686).
EXECUTION_LANE_ATTR = "_cozy_execution_lane"

#: Setup-scoped fallback (pgw#677 reopen): the executor opens this window
#: around one record's whole setup, so pipelines armed through ANY path —
#: slot injection or a self-loaded ``arm_compile`` inside ``setup()``
#: (ArmingScope) — get the same lane stamped by :func:`apply`.
_SETUP_EXEC_EXECUTION_LANE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "gw_setup_execution_lane", default="")

#: pgw#714: pin provenance of the stamped execution lane. True only when the
#: hub resolved this lane from an OPERATOR pin (`ModelResolution.lane_pinned`),
#: which makes an `+eager` execution axis a real kill switch: :func:`apply`
#: refuses to arm at all (no router, no background mint, no foreground
#: compile) instead of treating the pin as merely "serve eager while minting".
EXECUTION_LANE_PINNED_ATTR = "_cozy_execution_lane_pinned"
_SETUP_EXEC_EXECUTION_LANE_PINNED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gw_setup_execution_lane_pinned", default=False)

#: pgw#714: process-wide compile kill switch. Set once (with a reason) when
#: the crash registry shows a previous PROCESS DEATH attributed to a
#: background compile on this pod: the honest degrade is to stop compiling
#: and serve eager, not to re-run the native crash into a pod recycle loop.
_PROCESS_COMPILES_DISABLED = ""


def disable_process_compiles(reason: str) -> None:
    global _PROCESS_COMPILES_DISABLED
    if not _PROCESS_COMPILES_DISABLED:
        _PROCESS_COMPILES_DISABLED = str(reason or "disabled")
        logger.error(
            "compile-cache: COMPILES DISABLED for this process — %s "
            "(pgw#714 degrade-never-die: serving stays eager)",
            _PROCESS_COMPILES_DISABLED)


def process_compiles_disabled() -> str:
    return _PROCESS_COMPILES_DISABLED


def operator_eager_pin(pipeline: Any) -> bool:
    """True when the hub-resolved execution lane stamped on ``pipeline`` was
    OPERATOR-PINNED to the eager execution axis (pgw#714 kill switch)."""
    pinned = bool(getattr(pipeline, EXECUTION_LANE_PINNED_ATTR, False))
    execution_lane_str = str(getattr(pipeline, EXECUTION_LANE_ATTR, "") or "").strip()
    if not execution_lane_str:
        execution_lane_str = _SETUP_EXEC_EXECUTION_LANE.get().strip()
        pinned = _SETUP_EXEC_EXECUTION_LANE_PINNED.get()
        if execution_lane_str:
            try:
                setattr(pipeline, EXECUTION_LANE_ATTR, execution_lane_str)
                setattr(pipeline, EXECUTION_LANE_PINNED_ATTR, pinned)
            except Exception:
                pass
    if not (execution_lane_str and pinned):
        return False

    try:
        execution_lane = lanespec.parse_execution_lane(execution_lane_str)
    except ValueError:
        return False
    return execution_lane.execution == lanespec.EXEC_EAGER


def eager_tier_available(pipeline: Any) -> bool:
    """Can this pipeline answer a forward with NOTHING armed? (pgw#813)

    This is the question a background/out-of-process mint actually asks, and
    it is NOT :func:`mandatory_serving`. Using the latter as a serveability
    proxy is a category error, and it is the one that left AOT unmintable on
    every lane: the plain lane declines by #730's measured hold, and the w8a8
    lane — the lane the AOT program exists to serve — declined because
    "executes quantized activations" was read as "cannot serve eager".

    A quantized lane serves eager fine. ``_Fp8ScaledLinear.forward`` and
    ``_W4A4Linear.forward`` are complete eager forwards (``torch._scaled_mm``
    inline, scales computed per call), the fleet's own cold-boot ladder
    measures w8a8 eager serving, and pgw#672/#673 already retired the
    "mandatory lanes raise instead of degrade" posture inside :func:`_guard`
    — a mandatory lane whose compiled callable fails now serves
    ``original(...)`` LOUDLY. What ``mandatory_serving`` still answers, and
    should keep answering, is whether the COMPILED tier is the intended
    production tier (router fail-closed: novel shapes stay sequential rather
    than being routed eager behind the tenant's back).

    False only when an armed non-eager backend has REPLACED the callable —
    an AOTI export or a TRT engine — because there the eager forward is gone
    until the artifact is unwrapped.
    """
    # CYCLE: aot_serve and trt_engine both import AdoptError from this module;
    # hoisting makes compile_cache import itself through them at boot.
    from . import aot_serve, trt_engine

    try:
        if aot_serve.is_armed(pipeline):
            return False
    except Exception:  # noqa: BLE001 — an unanswerable arm is not a swap
        pass
    try:
        if trt_engine.is_armed(pipeline):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def mandatory_serving(pipeline: Any) -> bool:
    """ONE brain for "may this pipeline serve eager?" (pgw#677 reopen).

    Mandatory-ness follows the hub-resolved EXECUTION lane whenever the
    executor stamped one (``_cozy_execution_lane``, th#913/th#1059): only
    real w8a8/w4a4 ACTIVATION execution forbids the eager tier. The weight
    -lane stamp stays the CELL IDENTITY brain (pgw#686) — but it names the
    storage/branch family, not serveability: sdxl's mixed ``#fp8-w8a8``
    storage stamps ``w8a8-lora64`` while the hub serves it as
    ``fp8-w8a16+eager``, and classifying that stamp as mandatory silently
    routed the whole boot into the FOREGROUND compile-then-serve mint (the
    reopen's measured 26-minute tenant starvation). Without lane evidence
    the stamp remains the fail-closed fallback."""
    execution_lane_str = str(getattr(pipeline, EXECUTION_LANE_ATTR, "") or "").strip()
    if not execution_lane_str:
        execution_lane_str = _SETUP_EXEC_EXECUTION_LANE.get().strip()
        if execution_lane_str:
            try:
                setattr(pipeline, EXECUTION_LANE_ATTR, execution_lane_str)
            except Exception:
                pass
    if execution_lane_str:

        try:
            execution_lane = lanespec.parse_execution_lane(execution_lane_str)
        except ValueError:
            pass
        else:
            return execution_lane.activation in (lanespec.ACT_W8A8, lanespec.ACT_W4A4)
    # Module-attr call (not the top-level import): tests monkeypatch
    # models.loading.pipeline_weight_lane; stay late-bound.

    return _loading.pipeline_weight_lane(pipeline).startswith(
        ("w8a8", "w4a4"))


def apply(
    pipeline: Any,
    cfg: Any,
    *,
    cache_ready: bool,
    guard: bool = True,
    allow_cold: bool = False,
) -> bool:
    """Wrap ``cfg.targets`` on ``pipeline`` with compiled callables.

    Only compiles when a verified cache artifact was seeded (``cache_ready``)
    or explicit producer/local tooling passes ``allow_cold=True`` and has a C
    toolchain. Production serving never consults an environment fallback.
    Anything else is a logged no-op — eager, never a stall.

    ``guard=True`` (consumer): a failing ordinary compiled call permanently
    unwraps to eager; W8A8 fails closed. ``guard=False`` (compile job): all
    failures raise, because a silently eager warm-up would publish an empty
    artifact as success.
    """
    if getattr(pipeline, _MARKER_ATTR, None) is not None:
        return True
    # pgw#985: ONE reading of the preconditions, named. `begin_fleet_mint`
    # reads the same function to say WHY it refused, so the two can never
    # again describe the same decline in two different sentences.
    block = arming_block(
        pipeline, cfg, cache_ready=cache_ready, allow_cold=allow_cold)
    if block:
        logger.log(
            logging.ERROR if _PROCESS_COMPILES_DISABLED else logging.INFO,
            "compile-cache: not arming (%s); staying eager", block)
        return False
    import torch

    # gw#608: cross-pod cell portability requires the (portable) FX graph
    # cache to be the lookup surface — see _disable_aot_autograd_cache.
    _disable_aot_autograd_cache()
    # The two inner-key alignments (both symmetric mint/consumer by
    # construction: every compile path arms through apply()):
    _install_fx_system_shim()          # SKU name -> sm token (P0, review §6.1)
    _set_semantic_cache_tag(pipeline, cfg)  # semantic identity tag (§6.3)

    # Dynamo's per-code-object recompile limit defaults to 8; a preset table
    # bigger than that (LTX: 12 video graphs, ie#381) would silently fall
    # back to eager for every shape past the limit. Size it to the declared
    # shape set — never lower an operator-raised value.
    try:
        import torch._dynamo

        want = len(tuple(cfg.shapes)) + 8
        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit), want)
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = max(
                int(torch._dynamo.config.recompile_limit), want)
    except Exception:
        logger.debug("compile-cache: could not raise recompile limit", exc_info=True)

    regional = bool(getattr(cfg, "regional", False))

    # pgw#677 reopen: fail-closed follows the ONE serveability brain — the
    # hub-resolved execution lane when stamped, the weight-lane prefix
    # otherwise. A fail-closed router can never enable eager-while-compiling
    # routing, so misclassifying an eager-serveable lane here re-creates the
    # sequential inline-compile boot for the whole record.
    fail_closed = mandatory_serving(pipeline)

    failure_signal: Dict[str, Any] = {
        "callback": None,
        "lock": threading.Lock(),
        "successful_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        # pgw#680: serve-window guard misses (count) + telemetry callback.
        "guard_misses": 0,
        "on_guard_miss": None,
        # pgw#622: whole-graph consumer guards route novel signatures
        # through this; sequential until hot_swap.enable() post-proof.
        "router": hot_swap.Router(fail_closed=fail_closed) if guard else None,
    }
    applied: list[str] = []
    originals: list[Tuple[Any, str, Callable[..., Any]]] = []
    regional_mods: list[Any] = []
    for target, owner, attr, fn in resolve_targets(pipeline, cfg):
        # pgw#817/D4: computed BEFORE the branch so a declined regional target
        # falls through to the whole-forward branch (which does apply the
        # declared marks) instead of being skipped entirely.
        regional_decline = _regional_dynamic_decline(cfg, target) \
            if regional else ""
        if regional_decline:
            logger.info("compile-cache: %s", regional_decline)
        if (
            regional
            and not regional_decline
            and attr == "forward"
            and callable(getattr(owner, "compile_repeated_blocks", None))
        ):
            # Per-block graphs (ie#381): bounded memory under fp8 layerwise
            # casting + much cheaper cold compile. Blocks are compiled in
            # place; the guard wrapper clears them on the first failure.
            #
            _apply_declared_shape_config(cfg)
            owner.compile_repeated_blocks(dynamic=None)
            # pgw#681: regional entry crosses the same canonical boundary as
            # whole-graph entry — block guards mint over canonical inputs.
            ingress = guard_closure.canonical_ingress(fn, target)
            if guard:
                setattr(owner, attr, _guarded_regional(
                    owner,
                    ingress,
                    target,
                    fail_closed=fail_closed,
                    failure_signal=failure_signal,
                ))
            else:
                setattr(owner, attr, ingress)
            originals.append((owner, attr, fn))
            regional_mods.append(owner)
            applied.append(target)
            continue
        if regional:
            logger.info(
                "compile-cache: %r has no compile_repeated_blocks; "
                "whole-forward compile for it", target)
        if target.startswith("vae"):
            # channels_last + compiled decode is the measured win combo (#382);
            # memory format changes strides, so it is part of the cache key —
            # producer and consumer both come through here. channels_last is a
            # RANK-4 format: causal/video VAEs (Conv3d, rank-5 weights — qwen,
            # LTX) crash on it (gw#574), so gate on the actual weight ranks.
            # The gate is deterministic per model class, so producer and
            # consumer always agree on the resulting strides.
            vae = getattr(pipeline, "vae", None)
            if vae is not None and _vae_supports_channels_last(vae):
                vae.to(memory_format=torch.channels_last)
        # SDK v2 shape contract (ie#543 measured encoding): unmarked dims
        # are STATIC, promotion-on-change is OFF, and explicit marks are the
        # ONLY dynamism. `dynamic=False` + mark_dynamic is NOT expressible
        # (torch raises ConstraintViolationError), so the global guard is
        # the config pair + dynamic=None, never dynamic=False.
        _apply_declared_shape_config(cfg)
        compiled = torch.compile(fn, dynamic=None)
        declared_dynamic = tuple(getattr(cfg, "dynamic", ()) or ())
        if declared_dynamic:
            compiled = _with_declared_marks(compiled, declared_dynamic)
        # pgw#681: the single compiled-graph ingress — canonical strides +
        # dtype asserts OUTSIDE the declared marks, so the marks (and the
        # traced guards) always see the canonical form serving presents.
        compiled = guard_closure.canonical_ingress(compiled, target)
        setattr(owner, attr, _guarded(
            fn,
            compiled,
            target,
            fail_closed=fail_closed,
            failure_signal=failure_signal,
        ) if guard else compiled)
        applied.append(target)
        originals.append((owner, attr, fn))
    if not applied:
        return False
    setattr(pipeline, _MARKER_ATTR, {
        "targets": applied,
        "shapes": [tuple(s) for s in cfg.shapes],
        "cache": bool(cache_ready),
        "originals": originals,
        "regional_mods": regional_mods,
        "failure_signal": failure_signal,
    })
    try:
        _armed_pipelines().add(pipeline)
    except TypeError:
        logger.debug("compile-cache: pipeline not weakref-able; sibling-aware "
                     "dynamo reset scoping unavailable for it (pgw#637)")
    logger.info(
        "compile-cache: torch.compile armed for %s (cache=%s regional=%s)",
        applied, cache_ready, regional)
    return True


def set_guard_failure_callback(
    pipeline: Any, callback: Callable[[str], None],
) -> bool:
    """Bind scheduler-state revocation to an armed consumer guard."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return False
    signal["callback"] = callback
    return True


def _proof_count(pipeline: Any, key: str) -> int:
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return 0
    lock = signal.get("lock")
    if isinstance(lock, _LOCK_TYPE):
        with lock:
            return int(signal.get(key, 0))
    return int(signal.get(key, 0))


def execution_count(pipeline: Any) -> int:
    """Successful compiled calls observed on this exact pipeline object."""
    return _proof_count(pipeline, "successful_calls")


def cache_hit_count(pipeline: Any) -> int:
    """FX-graph cache hits observed inside this exact pipeline's guard."""
    return _proof_count(pipeline, "cache_hits")


def cache_miss_count(pipeline: Any) -> int:
    """FX-graph cache misses observed inside this exact pipeline's guard."""
    return _proof_count(pipeline, "cache_misses")


def unwrap(pipeline: Any) -> bool:
    """Restore the eager callables :func:`apply` wrapped and drop dynamo's
    in-memory compiled code so a later :func:`apply` re-traces against the
    then-seeded caches. Used on adoption rollback (zero cache hits => back to
    true eager, gw#391) and before re-adoption of a re-published cell."""
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if marker is None:
        return False
    signal = marker.get("failure_signal")
    if isinstance(signal, dict):
        router = signal.get("router")
        if router is not None:
            router.close()  # pgw#622: in-flight background warms discard
    for owner, attr, fn in marker.get("originals") or ():
        try:
            setattr(owner, attr, fn)
        except Exception:
            logger.warning("compile-cache: could not restore eager %s.%s", type(owner).__name__, attr)
    for mod in marker.get("regional_mods") or ():
        try:
            _clear_regional(mod)
        except Exception:
            logger.warning("compile-cache: could not clear regional compile on %s", type(mod).__name__)
    try:
        delattr(pipeline, _MARKER_ATTR)
    except AttributeError:
        setattr(pipeline, _MARKER_ATTR, None)
    try:
        _armed_pipelines().discard(pipeline)
    except TypeError:
        pass
    # pgw#637: torch._dynamo.reset() is PROCESS-GLOBAL — it drops every
    # armed pipeline's in-memory compiled code, not just this one's. With a
    # healthy sibling still armed (multi-checkpoint packing), a failing 2nd
    # arm's cleanup must never kill the 1st checkpoint's proven lane; the
    # global reset only runs when this was the last armed pipeline.
    siblings = 0
    try:
        siblings = len(_armed_pipelines())
    except Exception:
        siblings = 0
    if siblings > 0:
        logger.info(
            "compile-cache: skipping global dynamo reset on unwrap "
            "(%d sibling armed pipeline(s) live, pgw#637)", siblings)
        return True
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass
    return True


def _reconcile_resident_mode(meta: Optional[Dict[str, Any]], pipeline: Any) -> None:
    """gw#588: 'off' and 'vae_only' are both fully-resident preps differing
    only in flag groups — converge the pipeline to the cell's traced mode so
    an honest :func:`mode_drift` passes. Offload drift keeps refusing."""
    if not meta:
        return
    want = str(meta.get("low_vram_mode") or "")

    resident = ("off", "vae_only")
    have = low_vram_mode(pipeline)
    if want != have and want in resident and have in resident:
        reconcile_resident_mode(pipeline, want)


def artifact_drift(meta: Dict[str, Any], pipeline: Any, cfg: Any) -> str:
    """The complete pipeline/config compatibility verdict for one cell."""
    drift = (
        mode_drift(meta, pipeline)
        or execution_lane_drift(meta, pipeline)
        or contract_drift(meta, pipeline, cfg)
    )
    if drift:
        return drift
    want = "regional" if getattr(cfg, "regional", False) else "whole"
    have = str(meta.get("compile_mode") or "whole")
    if have != want:
        return f"cell compile_mode {have!r} != declared {want!r}"
    # pgw#695: the cell's sealed mint posture must be the posture THIS
    # process presents at arm time — a drift here would otherwise surface
    # later as an undiagnosable ambient guard miss.
    manifest = meta.get(guard_closure.MANIFEST_KEY)
    if isinstance(manifest, dict) and manifest:
        sealed = manifest.get(guard_closure.POSTURE_KEY)
        if not isinstance(sealed, dict) or not sealed:
            return "cell guard manifest carries no posture seal (pre-pgw#695 mint)"
        try:
            guard_closure.assert_posture(sealed, label="arm")
        except guard_closure.PostureError as exc:
            return str(exc)
    # pgw#719 (config half): the cell's recorded env seal must be the LIVE
    # effective environment at arm time — posture is named above; every
    # other seal fact (config flags, inductor digest, epoch, loaded libs)
    # is named here. In practice the ck key already pins this (env_seal is
    # an axis); the named drift makes a hand-delivered/foreign cell
    # diagnosable instead of a silent inner-key miss.
    sealed_env = meta.get(env_seal.SEAL_KEY)
    if isinstance(sealed_env, dict) and sealed_env:
        live_env = env_seal.effective_seal()
        for fact in sorted(set(sealed_env) | set(live_env)):
            if fact == "posture":
                continue  # named by the manifest posture check above
            cell_v, live_v = sealed_env.get(fact), live_env.get(fact)
            if isinstance(cell_v, dict) and isinstance(live_v, dict):
                for sub in sorted(set(cell_v) | set(live_v)):
                    if cell_v.get(sub) != live_v.get(sub):
                        return (
                            f"env seal drift at arm: {fact}/{sub}: cell "
                            f"{cell_v.get(sub)!r} != process {live_v.get(sub)!r}")
            elif cell_v != live_v:
                return (
                    f"env seal drift at arm: {fact}: cell {cell_v!r} != "
                    f"process {live_v!r}")
    return ""


def arm_staged_artifact(
    pipeline: Any,
    cfg: Any,
    staged: _StagedArtifact,
) -> Dict[str, Any]:
    """Activate and arm an already-verified artifact under the process lock.

    This strict entry point is used by hot adoption: unlike :func:`enable`, a
    mismatch is returned as a classified :class:`AdoptError` instead of an
    eager fallback. Expensive tar extraction happened before the executor's
    model/GPU locks; the process lock covers only atomic cache activation and
    wrapper installation.
    """
    try:
        with _SEED_ARM_LOCK:
            meta = staged.metadata
            _reconcile_resident_mode(meta, pipeline)
            drift = artifact_drift(meta, pipeline, cfg)
            if drift:
                raise AdoptError("key_mismatch", drift)
            _activate_staged(staged)
            unwrap(pipeline)
            try:
                if not apply(pipeline, cfg, cache_ready=True):
                    raise AdoptError("no_target")
            except Exception:
                unwrap(pipeline)
                raise
            # pgw#672: hot adoption runs idle-only; drop stale in-memory
            # compiled code so the adoption's proof warmup consults the
            # just-seeded FX entries (a real hit) instead of being served
            # counter-silently by a prior arm's resident code.
            reset_target_code(pipeline)
            return meta
    finally:
        staged.close()


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """The one consumer entry point (executor + local CLI): seed an explicitly
    attached verified artifact, then arm compile under the safety policy.

    A W8A8 refusal names its exact cause — the mismatched key axis with the
    cell-vs-runtime values, the drift verdict, or the missing delivery
    (gw#577): the raise IS the wire-visible job error, and serve pods expose
    no logs, so a generic message makes a refused cell undiagnosable."""
    staged: Optional[_StagedArtifact] = None
    refusal = "no cell artifact delivered"
    if artifact is not None:
        try:
            staged = stage_artifact(
                Path(artifact), getattr(cfg, "family", "") or "",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            refusal = f"cell rejected: {exc}"
            logger.warning("compile-cache: artifact unusable (%s); staying eager", exc)
    try:
        with _SEED_ARM_LOCK:
            meta: Optional[Dict[str, Any]] = None
            self_key = ""
            if staged is not None:
                meta = staged.metadata
                # th#883/gw#581: is this MY cell — the artifact whose axes
                # describe exactly the key this runtime computes for itself
                # with the one shared brain? If so, a refusal below is by
                # construction a selection/parity bug, never compatibility.
                try:
                    from .models.loading import (
                        pipeline_weight_lane as _pwl,
                    )

                    # gw#632: the EFFECTIVE bucket — a slot object with no
                    # resolvable compile target (sdxl's bare vae) never rides
                    # the branch lane (provision downgrades apply_lora_execution_lane
                    # the same way, 0.52.1), so its self-key must not claim
                    # the family's lora<bucket> cell and then explode on
                    # lane drift (live: `weight_lane 'lora64' != pipeline ''`
                    # -> CellSelectionBugError -> gw#608 seeded-cell refusal
                    # -> all_declared_functions_disabled pod retire).
                    eff_bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
                    if eff_bucket and not has_compile_target(pipeline, cfg):
                        eff_bucket = 0
                    want = cell_key.compute(
                        str(getattr(cfg, "family", "") or ""),
                        _pwl(pipeline),
                        eff_bucket,
                        contract=cell_key.contract_digest(
                            declared_contract_facts(
                                cfg, lora_bucket_override=eff_bucket)),
                        regional=bool(getattr(cfg, "regional", False)),
                    )
                    if not cell_key.mismatch(meta, want):
                        self_key = want.digest
                except Exception:
                    self_key = ""
                _reconcile_resident_mode(meta, pipeline)
                drift = artifact_drift(meta, pipeline, cfg)
                if drift:
                    # low_vram prep mode is DYNAMIC (free-VRAM placement at
                    # load) and outside the key: its drift is a legitimate
                    # miss even on a self-requested cell, never the bug class.
                    if self_key and not drift.startswith("low_vram_mode"):
                        raise CellSelectionBugError(
                            f"self-requested cell {self_key} refused to "
                            f"arm: {drift}"
                        )
                    refusal = f"cell rejected: {drift}"
                    logger.warning("compile-cache: %s; staying eager", drift)
                    meta = None
                else:
                    try:
                        _activate_staged(staged)
                    except Exception as exc:
                        refusal = f"cell activation failed: {exc}"
                        logger.warning(
                            "compile-cache: cache activation failed (%s); "
                            "staying eager", exc)
                        meta = None
            armed = apply(pipeline, cfg, cache_ready=meta is not None)
            if meta is not None and not armed and self_key:
                raise CellSelectionBugError(
                    f"self-requested cell {self_key} activated but armed "
                    "no compile target"
                )

            quant_execution_lane = pipeline_weight_lane(pipeline)
            if quant_execution_lane.startswith(("w8a8", "w4a4")) and not armed:
                if meta is not None:
                    refusal = "verified cell armed no compile target"
                execution_lane_name = quant_execution_lane[:4].upper()
                raise CompiledExecutionLaneUnavailableError(
                    f"{execution_lane_name} requires an exact compatible Forge cell "
                    f"({refusal}); eager/dequantized execution is not a "
                    f"{execution_lane_name} production lane"
                )
            return armed
    finally:
        if staged is not None:
            staged.close()


# ---------------------------------------------------------------------------
# Build (the compile job / conversion producer)
# ---------------------------------------------------------------------------


def resolve_pipeline_class(name: str) -> Any:
    """Resolve a serving pipeline class name for a mint (gw#586).

    The traced FX graphs depend on the pipeline's CALL path, not just the
    module tree — an unknown name must refuse loudly, because a silent
    generic-load fallback would trace the wrong call and publish a cell no
    serving lookup can ever hit.
    """
    import diffusers

    cleaned = str(name or "").strip()
    if not cleaned:
        raise RuntimeError("pipeline_class must be a non-empty class name")
    cls = getattr(diffusers, cleaned, None)
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise RuntimeError(
            f"pipeline_class {cleaned!r} is not a loadable diffusers "
            "pipeline class in this producer image; a generic-load fallback "
            "would trace the wrong call path (gw#586), so the mint refuses"
        )
    return cls


def _warm_text_lens(cfg: Any) -> tuple:
    """The text pins a warm loop must trace (pgw#654 gap #6): the class
    UNION when present (dual-lane classes trace one graph per pin), else
    the single declared pin, else (None,) — one unpinned pass."""
    pins = tuple(getattr(cfg, "text_lens", ()) or ())
    if not pins:
        tl = getattr(cfg, "text_len", None)
        pins = (int(tl),) if tl is not None else ()
    pins = tuple(sorted({int(v) for v in pins if int(v) > 0}))
    return pins if pins else (None,)


def _warm_call(
    pipe: Any,
    shape: Tuple[int, ...],
    *,
    steps: int,
    prompt: str,
    decode: bool,
    guidance_scales: Iterable[float] = (),
    text_len: Optional[int] = None,
) -> None:
    """One warm-up call for ``shape``. (w, h) is the classic image call;
    (w, h, frames) is a video call (ie#381): the DiT graph keys on the token
    count only, so a plain single-pipeline call traces the same graph the
    serving path (including a two-stage refine, whose latents arrive from an
    upsampler of identical shape) will look up. Video calls force the
    batch-1 no-CFG serving class (CFG is a graph shape — ``Compile``) and
    skip decode unless a vae target is declared. Image calls run once per
    explicitly declared guidance scale, capturing CFG and no-CFG graphs in
    one family cell.

    Guidance-kwarg convention (gw#595, the gw#586 class one axis over): on
    classes exposing ``true_cfg_scale`` (qwen-style), ``guidance_scale`` is
    the distilled-guidance embed no-op and classic CFG rides
    ``true_cfg_scale`` + a non-None ``negative_prompt`` — warming through
    ``guidance_scale`` there traces the SAME unconditioned graph for every
    declared scale and the serving CFG lookup can never hit."""

    import torch

    kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        width=int(shape[0]),
        height=int(shape[1]),
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    # SDK v2 text pin (ie#544): warm through the SAME pinned token length
    # the serving path uses, when the pipeline exposes the knob — the
    # traced sequence dim must match serving or every request misses.
    if text_len and text_len > 0:
        if "max_sequence_length" in inspect.signature(type(pipe).__call__).parameters:
            kwargs["max_sequence_length"] = int(text_len)
    if len(shape) == 3:
        params = inspect.signature(type(pipe).__call__).parameters
        kwargs["num_frames"] = int(shape[2])
        kwargs["output_type"] = "np" if decode else "latent"
        if "frame_rate" in params:
            kwargs["frame_rate"] = 24.0
        if "guidance_scale" in params:
            kwargs["guidance_scale"] = 1.0
        if "audio_guidance_scale" in params:
            kwargs["audio_guidance_scale"] = 1.0
        pipe(**kwargs)
        return

    scales = tuple(float(v) for v in guidance_scales)
    if not scales:
        pipe(**kwargs)
        return
    params = inspect.signature(type(pipe).__call__).parameters
    if "true_cfg_scale" in params:
        # Serving parity with the endpoint call: true_cfg_scale always
        # passed; negative_prompt only when CFG is on (scale > 1), matching
        # the CFG-off batch-1 graph exactly (no uncond pass).
        for scale in scales:
            call = dict(kwargs, true_cfg_scale=scale)
            if scale > 1.0:
                call["negative_prompt"] = " "
            pipe(**call)
        return
    accepts_guidance = "guidance_scale" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if not accepts_guidance:
        raise RuntimeError(
            f"{type(pipe).__name__} cannot warm declared guidance_scales; "
            "its call signature has no guidance_scale"
        )
    for scale in scales:
        pipe(**kwargs, guidance_scale=scale)


def build(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    shapes: Iterable[Tuple[int, ...]],
    targets: Iterable[str] = ("transformer", "vae.decode"),
    guidance_scales: Iterable[float] = (),
    text_len: Optional[int] = None,
    text_lens: Iterable[int] = (),
    dynamic: Iterable[Any] = (),
    family: str = "",
    source_ref: str = "",
    source_digest: str = "",
    dtype: str = "bf16",
    storage_dtype: str = "",
    regional: bool = False,
    steps: int = 2,
    prompt: str = "cache warm-up: a lighthouse on a cliff at dawn, detailed",
    declared_vram_gb: float = 0.0,
    serving_image_digest: str = "",
    lora_bucket: int = 0,
    requested_cell_key: str = "",
    pipeline_class: str = "",
) -> Tuple[Path, Dict[str, Any], Dict[str, float]]:
    """Compile a diffusers pipeline over ``shapes`` and package the resulting
    inductor+triton caches as a per-SKU artifact.

    ``storage_dtype`` mirrors the serving binding's weight-storage lane
    (gw#389 fp8 storage): the per-layer upcast is traced INTO the FX graphs
    (as module types since pgw#727), so a cell for an fp8-served model must be
    built from an fp8-loaded pipeline or every request misses the cache
    (ie#381).

    ``pipeline_class`` (gw#586) names the diffusers pipeline class the
    SERVING endpoint declares (e.g. ``"LTX2ConditionPipeline"``). The traced
    FX graphs depend on the pipeline's CALL path, not just the module tree:
    LTX2ConditionPipeline drives the DiT with PER-TOKEN timestep/modulation
    tensors even for a plain unconditioned call, while the generic
    model_index class broadcasts them — structurally different graphs, so a
    cell minted through the generic load can never serve the serving path's
    lookups (found live: warmups=1, cache_hits=0). The gw#577
    ``graph_signature`` remains class-agnostic (same module tree) — this is
    call-path parity, not module identity. Unknown class names refuse
    loudly: a silent generic fallback would re-open the exact parity gap.

    Runs on the TARGET GPU SKU with a C toolchain present (cold compile).
    Returns ``(artifact_path, metadata, per-shape warm-up seconds)`` — the
    first call per shape is the compile cost. Raises on any compile failure
    or an empty capture; a silently-eager build must never publish.
    """

    _W8A8_MINT_NEEDS_DIGEST = (
        "W8A8 cell mint requires serving_image_digest (the endpoint "
        "serving image's immutable OCI digest); a cell stamped with the "
        "producer image identity can never be adopted by the fleet"
    )
    if (("w8a8" in str(storage_dtype) or "w4a4" in str(storage_dtype))
            and not str(serving_image_digest).strip()):
        # gw#577 finding (b): contract_drift requires image_digest identity on
        # W8A8 cells and verify() pins it exactly. Without the SERVING digest
        # the artifact stamps the PRODUCER pod's WORKER_IMAGE_DIGEST — every
        # serving worker then rejects it and W8A8 serves NOTHING
        # (fail-closed). Refuse loudly before any load or compile.
        raise RuntimeError(_W8A8_MINT_NEEDS_DIGEST)
    if not toolchain_present():
        raise RuntimeError(
            "compile-cache build needs a C toolchain (cc/gcc); run in the "
            "compile-job image, not a prod worker image"
        )
    out_dir = Path(out_dir)
    capture_root = out_dir / "capture"
    capture_env(capture_root)

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("compile-cache build requires CUDA")

    cfg = CompileCell(
        shapes=tuple(tuple(int(v) for v in row) for row in shapes),
        targets=tuple(targets), family=str(family or ""),
        regional=bool(regional),
        text_len=(int(text_len) if text_len is not None else None),
        dynamic=tuple(dynamic),
        lora_bucket=int(lora_bucket or 0),
        guidance_scales=tuple(float(v) for v in guidance_scales),
        text_lens=tuple(int(v) for v in text_lens),
    )
    load_cls: Any = DiffusionPipeline
    if str(pipeline_class or "").strip():
        # gw#586 call-path parity: trace through the SERVING pipeline class.
        load_cls = resolve_pipeline_class(str(pipeline_class))
    pipe = load_from_pretrained(
        load_cls, str(model_path), dtype=dtype,
        storage_dtype=storage_dtype,
        # Producer/consumer LANE parity is now structural (pgw#772): the
        # voluntary free-VRAM bf16-resident upgrade is removed, so both
        # sides land the lane the declared config names. declared_vram_gb
        # is dead plumbing pending the coordinated wire sweep (pgw#772
        # follow-up in the tracker).
        declared_vram_gb=declared_vram_gb)
    # Producer/consumer graph parity (gw#391): the worker prepares pipelines
    # with place_pipeline (placement + vae/attention low-VRAM flags), and
    # those flags are traced INTO the graphs — the FX-graph cache key. A cell
    # built from a differently-prepared pipeline misses at request time, so
    # the producer must come through the exact same prep, and the mode it
    # traced under travels in the metadata for adopt-time parity checks. Run
    # on a pod with the same free-VRAM class as the target workers.
    placed = place_pipeline(pipe)

    if _traced_execution_lane(pipe).startswith(("w8a8", "w4a4")) and not str(
        serving_image_digest
    ).strip():
        # The lane can materialize as w8a8/w4a4 from the SOURCE flavor alone
        # (e.g. storage_dtype="fp8+te" over an fp8-w8a8 checkpoint), so the
        # authoritative check is on the traced lane, before the compile.
        raise RuntimeError(_W8A8_MINT_NEEDS_DIGEST)
    # gw#561: branch-bearing cells trace WITH canonical zeroed rank-bucket
    # branches installed — zeroed slots are bit-exact with branchless output
    # (gw#547), so the warm calls render normally while the traced graphs
    # carry the branch GEMMs.
    apply_lora_execution_lane(pipe, int(lora_bucket))
    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    # Cold compilation is an explicit producer-library operation; serving
    # workers have no environment switch that can enter this path.
    if not apply(pipe, cfg, cache_ready=False, guard=False, allow_cold=True):
        # pgw#985: name the fact that actually declined — "no compile targets"
        # was this line's answer to every one of them, including "no CUDA".
        raise CompileArmRefused(
            f"cannot arm {type(pipe).__name__} for a cold compile: "
            + (arming_block(pipe, cfg, cache_ready=False, allow_cold=True)
               or f"no compile target resolves for targets="
                  f"{[str(t) for t in (getattr(cfg, 'targets', ()) or ())]}"))

    timings: Dict[str, float] = {}
    decode = any(t.startswith("vae") for t in cfg.targets)
    for shape in cfg.shapes:
        torch.cuda.synchronize()
        t = time.monotonic()
        for pin in _warm_text_lens(cfg):
            _warm_call(
                pipe, shape, steps=int(steps), prompt=prompt, decode=decode,
                guidance_scales=cfg.guidance_scales,
                text_len=pin,
            )
        torch.cuda.synchronize()
        key = "x".join(str(v) for v in shape)
        timings[key] = round(time.monotonic() - t, 2)
        logger.info("compile-cache build: warmed %s in %.1fs", key, timings[key])

    # th#1322: the same numbers, on the wire. The log line above is for whoever
    # is watching the producer run; the event is for whoever asks the hub next
    # week how long a JIT mint takes.
    emit_jit_compile_event(
        timings, family=family, execution_lane=pipeline_weight_lane(pipe), route="build")

    captured = [p for p in (capture_root / "inductor").rglob("*") if p.is_file()]
    if not captured:
        raise RuntimeError(
            "compile warm-up captured nothing under TORCHINDUCTOR_CACHE_DIR — "
            "was inductor already latched to another dir in this process?"
        )

    # pgw#681/#756: the guard-closure audit is ADVISORY — suspected
    # out-of-contract guards are named in the log, emitted as a countable
    # `guard_leak` event, and recorded in the manifest that rides the cell;
    # the mint continues (the consumer re-evaluates these guards on every
    # call, so a real leak degrades to explicit eager there).
    # pgw#719: the environment this capture traced under must still be the
    # BOOT environment — drift (endpoint code mutating config/env behind
    # our back) fails the mint red, naming the fact.
    env_seal.assert_seal_unchanged("mint")
    guard_manifest = guard_closure.closure_manifest(pipe, cfg, label=family)

    graph_signature, weight_contract = execution_contract(pipe, cfg)
    meta = artifact_metadata(
        family=family, source_ref=source_ref, source_digest=source_digest,
        shapes=cfg.shapes, targets=cfg.targets,
        guidance_scales=cfg.guidance_scales,
        low_vram_mode=str(placed.get("mode") or ""),
        storage_dtype=storage_dtype,
        compile_mode="regional" if regional else "whole",
        # gw#534: the lane the pipeline ACTUALLY traced under — the loader may
        # have upgraded a requested fp8 cast to bf16-resident on this pod.
        weight_lane=pipeline_weight_lane(pipe),
        lora_bucket=int(lora_bucket or 0),
        graph_signature=graph_signature,
        weight_contract=weight_contract,
        shape_contract=declared_contract_facts(cfg),
        composition=composition_fingerprint(pipe, cfg),
    )
    meta[guard_closure.MANIFEST_KEY] = guard_manifest
    if serving_image_digest:
        # The producer image contains a compiler; the graph is consumed by
        # the endpoint's serving image. Tensorhub supplies that immutable OCI
        # digest from the release, so it—not the producer container—is the
        # identity the worker must match.
        meta["image_digest"] = str(serving_image_digest).strip()
    if str(pipeline_class or "").strip():
        # gw#586 observability only — NOT a key axis (graph_signature and the
        # ck1 key stay class-agnostic; the class shapes the traced CALL, and
        # a wrong class shows up as serving cache misses, which the warmup
        # proof refuses loudly).
        meta["pipeline_class"] = str(pipeline_class).strip()
    # gw#581/th#883: re-stamp the key over the final axes, then honor the
    # forge's echo — a demand-driven mint names the exact worker-computed
    # key it must satisfy, and publishing a cell under a key its own axes
    # do not describe would be a permanently un-armable store entry.

    cell_key.stamp(meta)
    if str(requested_cell_key or "").strip():
        reason = cell_key.mismatch(meta, str(requested_cell_key).strip())
        if reason:
            try:
                axes = cell_key.from_artifact_metadata(meta).canonical()
            except cell_key.CellKeyError:
                axes = "<not key-complete>"
            raise RuntimeError(
                "cell mint does not satisfy the requested cell key "
                f"({reason}); producer axes: {axes}"
            )
    label = flavor_label(meta["sku"], meta["torch"], meta.get("weight_lane", ""))
    artifact = pack(capture_root, out_dir / f"{label}.tar.gz", meta)
    return artifact, meta, timings


def emit_jit_compile_event(
    timings: Mapping[str, float],
    *,
    family: str,
    execution_lane: str = "",
    route: str = "",
    n_graphs: int = 0,
) -> None:
    """th#1322: report a JIT (dynamo/inductor) compile as typed NUMERIC events.

    ``timings`` maps one warm-shape key ("1024x1024") to its measured seconds.
    Emits one ``phase=shape:<key>`` event per shape plus a ``phase=minted``
    roll-up carrying the sum — the same shape ``aot_mint_phases`` uses, so
    "AOT mint vs JIT mint duration" is one grouped query over
    ``worker_activity_events`` instead of a regex over one side's free text and
    a grep of the other side's pod log (which a serve pod does not even
    expose, pgw#760).

    Telemetry must never fail the compile it reports on.
    """
    try:
        from . import activity as activity_mod

        total_s = sum(float(v or 0.0) for v in timings.values())
        head = f"family={family or '(unset)'}"
        if execution_lane:
            head += f" lane={execution_lane}"
        if route:
            head += f" route={route}"
        for key, seconds in timings.items():
            value = float(seconds or 0.0)
            if value <= 0:
                continue
            activity_mod.emit_event(
                activity_mod.KIND_JIT_COMPILE,
                f"{head} shape={key} compile_s={round(value, 2)}",
                phase=f"shape:{key}",
                duration_ms=int(round(value * 1000)),
            )
        if total_s <= 0:
            return
        activity_mod.emit_event(
            activity_mod.KIND_JIT_COMPILE,
            f"{head} n_shapes={len(timings)} n_graphs={n_graphs} "
            f"total_s={round(total_s, 2)} shapes={dict(timings)}",
            phase=activity_mod.PHASE_MINTED,
            duration_ms=int(round(total_s * 1000)),
        )
    except Exception:  # pragma: no cover — telemetry never fails the work
        logger.debug("compile-cache: jit_compile event emission failed",
                     exc_info=True)


def _compile_and_warm(pipe: Any, cfg: Any, *, steps: int = 2, say: Any = None) -> None:
    """Cold-compile ``pipe`` over the declared shape table (the only part of
    a mint that needs CUDA + a toolchain). ``guard=False``: a failing warm
    call must fail the mint — a silently-eager capture must never be saved."""
    _say = say if callable(say) else (lambda msg: logger.info("%s", msg))
    if not apply(pipe, cfg, cache_ready=False, guard=False, allow_cold=True):
        # pgw#985: name the fact that actually declined — "no compile targets"
        # was this line's answer to every one of them, including "no CUDA".
        raise CompileArmRefused(
            f"cannot arm {type(pipe).__name__} for a cold compile: "
            + (arming_block(pipe, cfg, cache_ready=False, allow_cold=True)
               or f"no compile target resolves for targets="
                  f"{[str(t) for t in (getattr(cfg, 'targets', ()) or ())]}"))
    import torch

    decode = any(t.startswith("vae") for t in cfg.targets)
    timings: Dict[str, float] = {}
    for shape in cfg.shapes:
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for pin in _warm_text_lens(cfg):
            _warm_call(
                pipe, shape, steps=steps,
                prompt="cache warm-up: a lighthouse on a cliff at dawn, detailed",
                decode=decode,
                guidance_scales=getattr(cfg, "guidance_scales", ()),
                text_len=pin,
            )
        torch.cuda.synchronize()
        shape_key = "x".join(str(v) for v in shape)
        timings[shape_key] = round(time.monotonic() - t0, 2)
        _say(f"  compiled {shape_key} in {timings[shape_key]:.0f}s")
    # th#1322: this line WAS the only record of JIT compile duration anywhere
    # (compile_cache.py:3803, "compiled %s in %.0fs") — a log-only important
    # metric, which is a defect class, not a style choice. Now it is a number in
    # a column too.
    emit_jit_compile_event(
        timings, family=getattr(cfg, "family", "") or "",
        execution_lane=pipeline_weight_lane(pipe), route="compile_and_warm")


def mint_artifact(
    pipe: Any,
    cfg: Any,
    family: str,
    target: Path,
    capture: Path,
    *,
    steps: int = 2,
    say: Any = None,
) -> Dict[str, Any]:
    """Self-mint (gw#555/gw#587): compile THIS pipeline over its declared
    shape table, capture the inductor/triton output, and pack the production
    artifact atomically at ``target``. Returns the stamped metadata (incl.
    the cell key its axes describe).

    The capture uses the production artifact recipe end to end
    (``capture_env`` -> warm the shape table -> ``artifact_metadata`` ->
    deterministic ``pack``), so the saved cell is byte-compatible with a
    delivered one and adopts through the identical code path. Shared by the
    cozy-local store mint and the fleet self-mint
    (fleet_cells) — ONE mint brain, different publish sinks.

    ``guard=False`` on the warm calls: a failing warm call must fail the
    mint — a silently-eager capture must never be saved.
    """
    _say = say if callable(say) else (lambda msg: logger.info("%s", msg))
    capture_env(capture)
    _compile_and_warm(pipe, cfg, steps=steps, say=_say)

    captured = [p for p in (capture / "inductor").rglob("*") if p.is_file()]
    if not captured:
        raise RuntimeError(
            "compile warm-up captured nothing under TORCHINDUCTOR_CACHE_DIR"
        )

    # pgw#681/#756: the guard-closure audit is ADVISORY — a suspected leak
    # is named and emitted as a `guard_leak` event, and the manifest rides
    # the cell as its dependency dump; the mint is not refused.
    # pgw#719: the environment this capture traced under must still be the
    # BOOT environment — drift (endpoint code mutating config/env behind
    # our back) fails the mint red, naming the fact.
    env_seal.assert_seal_unchanged("mint")
    guard_manifest = guard_closure.closure_manifest(pipe, cfg, label=family)

    # gw#564: record the execution contract exactly like the production
    # build — w8a8 cells are contract_drift-gated on the graph signature and
    # weight-lane manifest, so a mint without them can never re-adopt.
    graph_signature, weight_contract = execution_contract(pipe, cfg)
    meta = artifact_metadata(
        family=family,
        source_ref="self-mint",
        shapes=cfg.shapes,
        targets=cfg.targets,
        guidance_scales=getattr(cfg, "guidance_scales", ()),
        low_vram_mode=low_vram_mode(pipe),
        compile_mode="regional" if getattr(cfg, "regional", False) else "whole",
        weight_lane=pipeline_weight_lane(pipe),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        graph_signature=graph_signature,
        weight_contract=weight_contract,
        shape_contract=declared_contract_facts(cfg),
        composition=composition_fingerprint(pipe, cfg),
    )
    meta[guard_closure.MANIFEST_KEY] = guard_manifest
    tmp = target.with_suffix(".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    pack(capture, tmp, meta)
    os.replace(tmp, target)
    return meta


def begin_fleet_mint(pipe: Any, cfg: Any, capture: Path) -> None:
    """Arm ``pipe`` for a fleet self-mint capture (gw#587 CORRECT FIX).

    Points inductor/triton at a fresh ``capture`` dir and enables the
    declared compile targets in cold-allowed, GUARDED mode — WITHOUT
    running any synthetic warm call (the old ``mint_artifact``/
    ``_warm_call`` producer-style recipe, gw#586's defect class
    resurfacing inside self-mint, gw#587's root cause).

    The caller's real serving warmup — the executor's own warmup-proof
    window, running the endpoint's own code (the two-stage/conditioned
    call LTX and its siblings actually make) — performs the ONLY compile
    this mint will ever see. Capturing that exact execution instead of a
    second, separately-shaped call is what makes the published artifact
    byte-derived from the same execution the proof observed: there is no
    other code path that could re-create serving's call shape, so the
    mint can never diverge from what it actually serves.

    Raises :class:`CompileArmRefused` — typed and deterministic — when there
    is nothing to prove or publish; the caller's miss policy applies exactly
    as it did for a failed :func:`mint_artifact` call. pgw#985: the two facts
    that can refuse here are DIFFERENT and are now named as such. A pipeline
    that owns no declared target is a WIRING fact; a process that cannot arm
    the targets it does own (no CUDA, no toolchain, an operator eager pin) is
    an ENVIRONMENT fact. Both used to raise the wiring sentence, so a cardless
    mint pod reported "no compile targets resolved on TinyDiffusionPipeline"
    about a pipeline whose ``.unet`` had resolved a frame earlier.

    The DECISION still has one evaluator — :func:`apply` — and this only names
    what it declined on, through the same :func:`arming_block` ``apply``
    itself consulted.

    gw#608 ROOT CAUSE lived here: the cache-dir env is PROCESS-GLOBAL, and
    this function re-pointed it BEFORE knowing the arm could succeed. On a
    store-served LTX boot the no-compile-target upsampler sibling reached
    this path after the distilled lane had seeded its delivered cell: the
    env moved to a throwaway capture dir, the arm failed, the dir was
    deleted — and the real warmup then looked up FX graphs in an empty
    resurrected tmp dir (8/8 misses, live-proven; mint boots were immune
    because their own capture registers first and the sibling is declined
    BEFORE this call). The arm is therefore transactional now: nothing to
    arm never touches the env, and an arm failure restores it exactly.
    """
    family = str(getattr(cfg, "family", "") or "(unset)")
    owned = [name for name, *_ in resolve_targets(pipe, cfg)]
    if not owned:
        raise CompileArmRefused(
            f"no compile target resolves on {type(pipe).__name__} for family "
            f"{family!r}: declared "
            f"targets={[str(t) for t in (getattr(cfg, 'targets', ()) or ())]}")
    prior = {
        env: os.environ.get(env)
        for env in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    }
    capture_env(capture)
    try:
        # `apply` DECIDES — one call, and it reads `arming_block` for the same
        # preconditions this refusal then names. Asking `arming_block` here
        # first would be a second gate `apply` does not answer to, and the
        # decision must have exactly one evaluator.
        if not apply(pipe, cfg, cache_ready=False, guard=True, allow_cold=True):
            raise CompileArmRefused(
                f"{type(pipe).__name__} owns the declared compile target(s) "
                f"{owned} for family {family!r}, but this process cannot arm "
                f"them: "
                + (arming_block(pipe, cfg, cache_ready=False, allow_cold=True)
                   or "apply() declined without naming a precondition"))
    except BaseException:
        for env, value in prior.items():
            if value is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = value
        _reset_inductor_latch()
        raise


def finish_fleet_mint(
    pipe: Any, cfg: Any, family: str, target: Path, capture: Path,
    *, expected_graphs: int = 0,
) -> Dict[str, Any]:
    """Pack the capture dir a PASSED warmup proof just populated.

    Callers must invoke this ONLY after the executor's warmup proof has
    confirmed the real serving call exercised ``pipe``'s attached compile
    targets (a successful compiled call recorded — never before: packing
    ahead of the proof would reopen the publish-before-proof window
    gw#587 closes, and packing an unexercised/failed capture would
    publish bytes nothing ever proved served).

    Unlike :func:`mint_artifact`, this never compiles anything itself —
    the compile already happened, inside the proof window, driven by the
    endpoint's own warmup. This function only packages what that warmup
    produced.
    """
    captured = [p for p in (capture / "inductor").rglob("*") if p.is_file()]
    if not captured:
        raise RuntimeError(
            "self-mint proof passed but captured nothing under "
            "TORCHINDUCTOR_CACHE_DIR — the compile did not write here. "
            + _capture_forensics(capture, pipe)
        )
    # gw#608 hardening: a minting boot proves its EXECUTION; this asserts
    # its ARTIFACT. The pack must contain the FX-graph entries the proof's
    # compiled set implies — an empty/partial fxgraph store (e.g. inductor
    # latched elsewhere mid-process, or a cache-layer redirect miss) would
    # publish a cell that can never serve any consumer, and the fleet would
    # only find out one fail-closed boot at a time.
    fx_entries = 0
    fx_root = capture / "inductor" / "fxgraph"
    if fx_root.is_dir():
        fx_entries = sum(1 for p in fx_root.rglob("*") if p.is_file())
    if fx_entries <= 0:
        raise RuntimeError(
            "self-mint capture contains NO FX-graph cache entries — the "
            f"compile wrote {len(captured)} other file(s) here but no fx "
            "entry, so either the fx cache was bypassed/disabled or nothing "
            "re-compiled in this process; refusing to publish an unservable "
            "cell. " + _capture_forensics(capture, pipe)
        )
    if expected_graphs > 0 and fx_entries < expected_graphs:
        raise RuntimeError(
            f"self-mint capture holds {fx_entries} FX-graph entrie(s) but "
            f"the warmup proof compiled {expected_graphs} graph(s) — "
            "partial capture, refusing to publish. "
            + _capture_forensics(capture, pipe)
        )

    # pgw#681/#756: the guard-closure audit. The proof confirmed the graphs
    # SERVE; this DOCUMENTS what they depend on. Guards the contract does
    # not pin are named and emitted as a `guard_leak` event but do NOT
    # refuse the mint — dynamo re-checks them at the consumer, where a real
    # leak fails its guards and degrades to explicit eager (pgw#680). The
    # returned manifest rides the cell as its dependency dump.
    # pgw#719: the environment this capture traced under must still be the
    # BOOT environment — drift (endpoint code mutating config/env behind
    # our back) fails the mint red, naming the fact.
    env_seal.assert_seal_unchanged("mint")
    guard_manifest = guard_closure.closure_manifest(pipe, cfg, label=family)

    # gw#564: record the execution contract exactly like the production
    # build — w8a8 cells are contract_drift-gated on the graph signature and
    # weight-lane manifest, so a mint without them can never re-adopt. Both
    # are STATIC (module structure + declared shapes/targets), computed the
    # same way whether sampled before or after the real compile.
    graph_signature, weight_contract = execution_contract(pipe, cfg)
    meta = artifact_metadata(
        family=family,
        source_ref="self-mint",
        shapes=cfg.shapes,
        targets=cfg.targets,
        guidance_scales=getattr(cfg, "guidance_scales", ()),
        low_vram_mode=low_vram_mode(pipe),
        compile_mode="regional" if getattr(cfg, "regional", False) else "whole",
        weight_lane=pipeline_weight_lane(pipe),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        graph_signature=graph_signature,
        weight_contract=weight_contract,
        shape_contract=declared_contract_facts(cfg),
        composition=composition_fingerprint(pipe, cfg),
    )
    meta[guard_closure.MANIFEST_KEY] = guard_manifest
    tmp = target.with_suffix(".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    pack(capture, tmp, meta)
    os.replace(tmp, target)
    return meta


__all__ = [
    "ARTIFACT_FORMAT",
    "AdoptError",
    "CellSelectionBugError",
    "CompileArmRefused",
    "CompiledExecutionLaneUnavailableError",
    "build",
    "apply",
    "apply_lora_execution_lane",
    "arming_block",
    "resolve_targets",
    "artifact_fx_lines",
    "artifact_metadata",
    "begin_fleet_mint",
    "capture_env",
    "cell_base_execution_lane",
    "drop_lora_execution_lane",
    "cell_execution_lane",
    "contract_drift",
    "counters_delta",
    "delivered_cell_seeded",
    "cache_hit_count",
    "cache_miss_count",
    "enable",
    "execution_count",
    "composition_fingerprint",
    "execution_contract",
    "execution_contract_digest",
    "family_from_ref",
    "finish_fleet_mint",
    "parse_cell_ref",
    "find_artifact",
    "flavor_label",
    "fx_cache_failure_report",
    "fx_key_forensics",
    "gen_worker_version",
    "GuardMiss",
    "guard_miss_count",
    "guard_miss_reason_class",
    "has_compile_target",
    "in_tenant_serve_window",
    "set_guard_miss_callback",
    "tenant_serve_window",
    "inductor_counters",
    "is_cache_ref",
    "execution_lane_bucket",
    "execution_lane_token",
    "execution_lane_drift",
    "live_fx_lines",
    "mint_artifact",
    "mode_drift",
    "pack",
    "prepare",
    "resolve_pipeline_class",
    "runtime_key",
    "record_cell_proven",
    "record_cell_quarantined",
    "cell_proven_in_process",
    "cell_quarantined_in_process",
    "reset_target_code",
    "seed_artifact",
    "seed_env",
    "set_guard_failure_callback",
    "sku_slug",
    "system_repo",
    "cxx_compiler",
    "cxx_toolchain_present",
    "toolchain_present",
    "unpack",
    "unwrap",
    "verify",
]
