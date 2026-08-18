"""pgw#1331: the compile facts a SERVING process reads, with nothing behind them.

``serving_mode`` answers "what actually served this request" and its own
docstring promises it is *"deliberately duck-typed and free of executor
internals so it can be unit-tested without a pipeline, a GPU, or a hub"*. It
was not: it imported :mod:`gen_worker.compile_cache` — 3,100 lines of the
EAGER-CAPABLE arming brain — to read two things, a marker attribute and a
version probe. ``compile_cache`` imports ``models.loading`` / ``models.memory``
/ ``models.w8a8_lora``, which construct diffusers and transformers objects
inside their functions and are entitled to, so the whole adopt-only serve role
inherited a model library through a per-request REPORTING module.

So the facts live here and the machinery stays there. This module reads a
marker somebody else wrote and probes the toolchain it is running on. It holds
no state, arms nothing, and imports nothing above ``hostfacts`` — which is what
lets ``gen_worker.serve.role.MODEL_FREE_MODULES`` name the whole role rather
than a surface inside it.

**There is one definition of each name, not two.** ``compile_cache`` imports
``runtime_key`` / ``sku_slug`` / ``is_compile_armed`` from here and re-exports
them, so ``compile_cache.runtime_key`` IS this function and every existing
caller — and every test that monkeypatches it on ``compile_cache`` — is
unaffected. The other three readers have no re-export at all: their one caller
(``executor``) asks this module directly. Writing the marker is still
``compile_cache``'s alone; this side only ever reads.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from .hostfacts import cuda_ready

logger = logging.getLogger(__name__)

#: The attribute ``compile_cache.apply()`` stamps onto an armed pipeline. The
#: readers below are the whole of what a serving process needs from it.
MARKER_ATTR = "_cozy_compile"


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


def runtime_key() -> Dict[str, str]:
    """The consumer-side half of the cache key, probed from this process.

    pgw#1034: no ``cuda_driver``. gw#577 ruled it a host-lottery axis and took
    it out of every key and every gate; what was left was a fact nothing read,
    bought with a ``libcuda.so.1`` dlopen and a ``cuInit(0)`` on each call —
    and this function is called per ``verify()``, not once.
    """
    key = {
        "sku": "", "sm": "", "torch": "", "triton": "", "cuda": "",
        "image_digest": os.environ.get("WORKER_IMAGE_DIGEST", "").strip(),
    }
    try:
        import torch

        key["torch"] = str(torch.__version__)
        key["cuda"] = str(torch.version.cuda or "")
        if cuda_ready():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
            major, minor = torch.cuda.get_device_capability(0)
            key["sm"] = f"sm_{major}{minor}"
    except Exception:
        # pgw#657: silently leaving these EMPTY manufactures a different compiled graph
        # key than every healthy pod computes — i.e. a guaranteed cache miss
        # (and a mint) whose cause is invisible. Say it.
        logger.warning(
            "compile-cache: torch/CUDA runtime-key probe failed — compiled graph identity "
            "falls back to empty sku/sm/torch fields; expect a cache MISS",
            exc_info=True)
    try:
        import triton

        key["triton"] = str(triton.__version__)
    except Exception:
        logger.debug("compile-cache: triton version unavailable", exc_info=True)
    return key


def _failure_signal(pipeline: Any) -> Dict[str, Any] | None:
    marker = getattr(pipeline, MARKER_ATTR, None)
    signal = marker.get("failure_signal") if isinstance(marker, dict) else None
    return signal if isinstance(signal, dict) else None


def is_compile_armed(pipeline: Any) -> bool:
    """True when this pipeline is serving COMPILED code right now.

    pgw#1010: the JIT INTAKE arm names no artifact, so ``active_compile_ref``
    is empty for a pipeline that is nonetheless serving compiled code. This is
    the fact that separates it from true eager, and ``serving_mode`` reads it
    per request — hence the cheap attribute probe rather than a target walk.

    A guard that permanently degraded this target to eager (``_guarded``'s
    fallback) clears the answer even though the wrapper is still installed:
    reporting a degraded pipeline as compiled is the same lie as reporting an
    unproven compiled graph as adopted.
    """
    if getattr(pipeline, MARKER_ATTR, None) is None:
        return False
    signal = _failure_signal(pipeline)
    return not (signal is not None and signal.get("degraded"))


def graph_break_reason(pipeline: Any) -> str:
    """Torch's verbatim fullgraph refusal for this pipeline, or "".

    Non-empty means the declared region did not trace whole and this process
    permanently degraded it to eager. The executor turns it into the
    ``graph_break`` eager posture, so every request the pod serves afterwards
    names the real cause instead of an empty ``fallback_reason``."""
    signal = _failure_signal(pipeline)
    return str(signal.get("graph_break") or "") if signal else ""


def degrade_reason(pipeline: Any) -> str:
    """pgw#1093: why this ARMED pipeline is permanently eager, or "".

    Non-empty means `apply()` DID install the compiled callables and a served
    call then failed permanently. That is a different fact from "no target
    was ever installed", and before this the two were the same reading:
    `is_compile_armed` False, `metrics.lane=…+eager`,
    `fallback_reason=uncompiled`. The executor turns this into a
    `compiled_degraded` eager posture so the distinction survives to the wire.
    """
    signal = _failure_signal(pipeline)
    return str(signal.get("degrade_reason") or "") if signal else ""


def declared_range_refusal(pipeline: Any) -> str:
    """The typed declared-range refusal for this pipeline, or ""."""
    signal = _failure_signal(pipeline)
    return str(signal.get("declared_range_exceeded") or "") if signal else ""


__all__ = [
    "MARKER_ATTR",
    "declared_range_refusal",
    "degrade_reason",
    "graph_break_reason",
    "is_compile_armed",
    "runtime_key",
    "sku_slug",
]
