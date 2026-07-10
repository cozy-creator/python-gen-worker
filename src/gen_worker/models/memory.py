"""VRAM/memory decisions for the models layer (#358/#366).

One low-VRAM decider for the whole worker, driven by FREE VRAM only (never
total capacity), plus size/measurement probes used by residency accounting.

Ladder (auto mode, least-aggressive first):

  off           : no optimizations (pipeline on CUDA as-is)
  vae_only      : VAE slicing + tiling (+ attention slicing when available)
  model_offload : vae_only + ``enable_model_cpu_offload()``  (~10% slower)
  group_offload : leaf-level group offload with CUDA streams   (~25% slower)
  sequential    : ``enable_sequential_cpu_offload()``          (~50%+ slower)

Upstream foot-gun: ``enable_sequential_cpu_offload`` must NOT be called on a
pipeline already moved to CUDA; ``apply_low_vram_config`` moves it back first.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

_LOG = logging.getLogger(__name__)

Mode = str  # "auto" | "off" | "vae_only" | "model_offload" | "group_offload" | "sequential"

_VALID_MODES: tuple[str, ...] = (
    "auto", "off", "vae_only", "model_offload", "group_offload", "sequential",
)

_DEFAULT_VAE_SLICE_THRESHOLD_GB = 10.0
_DEFAULT_MODEL_OFFLOAD_THRESHOLD_GB = 8.0
_DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB = 6.0
# Safety margin below free VRAM reserved for activations.
_DEFAULT_SAFETY_MARGIN_GB = 2.0
# Free headroom beyond the requirement above which "off" beats "vae_only".
_DEFAULT_OFF_HEADROOM_GB = 8.0

# Sentinel attribute set on pipelines to make apply_low_vram_config idempotent.
_COZY_MODE_ATTR = "_cozy_low_vram_mode"

# Authors declare ``Resources(vram_gb=X)`` as the TOTAL VRAM of the smallest
# card they target ("runs on a 24 GB card") — a placement recommendation, not
# measurable free bytes. The platform reserves this much for the fixed
# driver/framebuffer/CUDA-context overhead when comparing the recommendation
# against probed VRAM, so vram_gb=24 serves on a 24 GB card (~23.6 GB free).
GPU_VRAM_OVERHEAD_GB = 1.0


def effective_vram_requirement_gb(recommended_gb: float) -> float:
    """The probed-VRAM floor implied by a ``vram_gb`` recommendation.

    Single definition of "usable VRAM" for every gate/fit comparison:
    compare probed total or free GB against THIS, never against the raw
    recommendation.
    """
    return max(0.0, float(recommended_gb) - GPU_VRAM_OVERHEAD_GB)

# Modes that spill model weights to system RAM and run parts of inference on CPU.
_CPU_OFFLOAD_MODES = ("model_offload", "group_offload", "sequential")


def _forbid_cpu_inference(detail: str) -> None:
    """Raise when GEN_WORKER_FORBID_CPU_OFFLOAD=1 vetoes a CPU-touching placement.

    Set on dev machines so agents/tests can't silently melt the box with
    CPU-offloaded inference; real-model runs belong on the GPU CI lane.
    """
    if os.environ.get("GEN_WORKER_FORBID_CPU_OFFLOAD") == "1":
        raise RuntimeError(
            f"GEN_WORKER_FORBID_CPU_OFFLOAD=1: refusing {detail}. "
            "Real-model inference that does not fit in free VRAM must run on "
            "the GPU CI lane, not this machine."
        )


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def get_available_vram_gb(device_index: int = 0) -> float:
    """Currently-free VRAM on the selected CUDA device. 0.0 if no CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info(device_index)
        return float(free) / float(1024**3)
    except Exception:
        return 0.0


def get_available_ram_gb() -> float:
    """Available system RAM (for disk-offload decisions). 0.0 if psutil missing."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / float(1024**3)
    except Exception:
        return 0.0


def get_total_ram_gb() -> float:
    """Total system RAM (adaptive RAM-floor input). 0.0 if psutil missing."""
    try:
        import psutil

        return float(psutil.virtual_memory().total) / float(1024**3)
    except Exception:
        return 0.0


def cuda_allocated_bytes(device_index: Optional[int] = None) -> int:
    """``torch.cuda.memory_allocated`` (0 without CUDA). Deltas across a load
    are the measured VRAM footprint reported in ModelEvent.vram_bytes."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(device_index))
    except Exception:
        pass
    return 0


def _named_components(pipeline: Any) -> List[tuple[str, Any]]:
    out: List[tuple[str, Any]] = []
    raw = getattr(pipeline, "components", None)
    if isinstance(raw, dict):
        out.extend(raw.items())
    else:
        for attr in ("unet", "transformer", "vae", "text_encoder",
                     "text_encoder_2", "text_encoder_3"):
            v = getattr(pipeline, attr, None)
            if v is not None:
                out.append((attr, v))
    if not out and hasattr(pipeline, "parameters"):
        out.append(("", pipeline))  # bare nn.Module
    return out


def _iter_components(pipeline: Any) -> List[Any]:
    return [c for _, c in _named_components(pipeline)]


def device_mismatches(obj: Any, device: str) -> List[tuple[str, str, str]]:
    """Every parameter/buffer of ``obj``'s module components that is NOT on
    ``device``'s device type, as ``(component, tensor, actual_device)``.

    The paranoid post-move walk (gw#409): a pipeline ``.to()`` that raises or
    skips mid-way leaves a mixed-device pipeline that fatals mid-denoise
    ("Expected all tensors to be on the same device"); this surfaces the miss
    at move time instead. [] without torch / for tensor-less objects."""
    try:
        import torch

        target = torch.device(device).type
    except Exception:
        return []
    out: List[tuple[str, str, str]] = []
    for cname, comp in _named_components(obj):
        if comp is None or not hasattr(comp, "named_parameters"):
            continue
        try:
            named = list(comp.named_parameters())
            if hasattr(comp, "named_buffers"):
                named.extend(comp.named_buffers())
        except Exception:
            continue
        for tname, t in named:
            if isinstance(t, torch.Tensor) and t.device.type != target:
                out.append((cname, tname, str(t.device)))
    return out


def repair_device_placement(obj: Any, device: str) -> List[tuple[str, str, str]]:
    """Targeted ``.to(device)`` on each component holding off-device tensors,
    then re-walk. Returns the remaining mismatches ([] = fully repaired)."""
    missed = device_mismatches(obj, device)
    if not missed:
        return []
    bad = {c for c, _, _ in missed}
    for cname, comp in _named_components(obj):
        if cname not in bad:
            continue
        try:
            comp.to(device)
        except Exception as exc:
            _LOG.warning("device repair: %s.to(%s) failed: %s", cname or "obj", device, exc)
    return device_mismatches(obj, device)


def _sum_tensor_bytes(objs: Iterable[Any], *, cuda_only: bool) -> int:
    import torch

    total = 0
    seen: set[int] = set()  # data_ptr dedupe: shared storages counted ONCE
    for obj in objs:
        for c in _iter_components(obj):
            if c is None or not hasattr(c, "parameters"):
                continue
            tensors = list(c.parameters())
            if hasattr(c, "buffers"):
                tensors.extend(c.buffers())
            for t in tensors:
                if not isinstance(t, torch.Tensor):
                    continue
                if cuda_only and t.device.type != "cuda":
                    continue
                try:
                    key = t.data_ptr()
                except Exception:
                    key = id(t)
                if key in seen:
                    continue
                seen.add(key)
                total += t.numel() * t.element_size()
    return total


def estimate_pipeline_size_gb(pipeline: Any) -> float:
    """Total weight bytes of a pipeline regardless of device — the *requirement*
    estimate the offload ladder compares against free VRAM. Tensors that share
    storage (shared components) are counted once. 0.0 without torch."""
    try:
        return float(_sum_tensor_bytes([pipeline], cuda_only=False)) / float(1024**3)
    except Exception:
        return 0.0


def estimate_cuda_resident_gb(*objects: Any) -> float:
    """CUDA-resident bytes across the given pipelines/modules, shared storages
    counted once — the *residency accounting* estimate (#358: CPU-offloaded
    pipelines must not be booked as full VRAM; shared components once)."""
    try:
        return float(_sum_tensor_bytes(objects, cuda_only=True)) / float(1024**3)
    except Exception:
        return 0.0


def flush_memory() -> None:
    """gc + empty_cache + reset_peak_memory_stats. Always safe to call."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mode selection (auto) — the ONE low-VRAM decider, free-VRAM inputs only
# ---------------------------------------------------------------------------


def select_auto_mode(
    *,
    pipeline: Any,
    available_vram_gb: Optional[float] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
) -> str:
    """Pick the least-aggressive ladder step that keeps the pipeline in memory.

    Decisions are made against FREE VRAM (what is actually available right
    now), never the card's TOTAL capacity: a second model on an occupied card
    must see the reduced free space.

    ``peak_vram_gb`` is the endpoint's DECLARED per-request peak
    (``Resources.peak_vram_per_request_gb``, #339); when provided the fit
    requirement becomes ``max(model_gb, peak_vram_gb)``.
    """
    avail = available_vram_gb if available_vram_gb is not None else get_available_vram_gb()
    if avail <= 0.0:
        return "off"

    model_gb = model_size_gb if model_size_gb is not None else estimate_pipeline_size_gb(pipeline)
    requirement = model_gb
    if peak_vram_gb is not None and peak_vram_gb > 0.0:
        requirement = max(model_gb, float(peak_vram_gb))
    margin = _DEFAULT_SAFETY_MARGIN_GB

    # Very low free VRAM: even a fitting model needs aggressive help for activations.
    if avail <= _DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB:
        return "group_offload"

    if requirement > 0.0:
        usable = max(0.0, avail - margin)
        if requirement > usable:
            return "model_offload"
        # Fits. With generous FREE headroom run fully unoptimized; on a tighter
        # card keep the cheap vae_only guard for VAE-decode spikes.
        if (usable - requirement) >= _DEFAULT_OFF_HEADROOM_GB:
            return "off"
        return "vae_only"

    # Unknown model size: conservative free-VRAM thresholds.
    if avail <= _DEFAULT_MODEL_OFFLOAD_THRESHOLD_GB:
        return "model_offload"
    return "vae_only"


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _call_if_present(obj: Any, method: str, **kwargs: Any) -> bool:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return False
    try:
        fn(**kwargs) if kwargs else fn()
        return True
    except TypeError:
        try:
            fn()
            return True
        except Exception:
            return False
    except Exception as exc:
        _LOG.debug("low_vram: %s() raised %s", method, exc)
        return False


def _move_pipeline_to_cpu(pipeline: Any) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if callable(getattr(pipeline, "to", None)):
            pipeline.to("cpu")
    except Exception as exc:
        _LOG.debug("low_vram: move-to-cpu failed: %s", exc)


def _apply_vae_and_attention(pipeline: Any, applied: Dict[str, bool]) -> None:
    if not _call_if_present(pipeline, "enable_vae_slicing"):
        vae = getattr(pipeline, "vae", None)
        if vae is not None and _call_if_present(vae, "enable_slicing"):
            applied["vae_slicing"] = True
    else:
        applied["vae_slicing"] = True

    if not _call_if_present(pipeline, "enable_vae_tiling"):
        vae = getattr(pipeline, "vae", None)
        if vae is not None and _call_if_present(vae, "enable_tiling"):
            applied["vae_tiling"] = True
    else:
        applied["vae_tiling"] = True

    if _call_if_present(pipeline, "enable_attention_slicing"):
        applied["attention_slicing"] = True


def _apply_group_offload(
    pipeline: Any,
    applied: Dict[str, bool],
    *,
    offload_to_disk_path: Optional[str],
) -> bool:
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False

    kwargs: Dict[str, Any] = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": True,
    }
    if offload_to_disk_path:
        kwargs["offload_to_disk_path"] = offload_to_disk_path

    fn = getattr(pipeline, "enable_group_offload", None)
    if callable(fn):
        try:
            fn(**kwargs)
            applied["group_offload"] = True
            if offload_to_disk_path:
                applied["disk_offload_path"] = True
            return True
        except Exception as exc:
            _LOG.debug("low_vram: pipeline.enable_group_offload failed: %s", exc)

    any_applied = False
    try:
        from diffusers.hooks import apply_group_offloading
    except Exception:
        apply_group_offloading = None

    for attr in ("transformer", "unet", "vae", "text_encoder", "text_encoder_2"):
        mod = getattr(pipeline, attr, None)
        if mod is None:
            continue
        mod_fn = getattr(mod, "enable_group_offload", None)
        if callable(mod_fn):
            try:
                mod_fn(**kwargs)
                any_applied = True
                continue
            except Exception as exc:
                _LOG.debug("low_vram: %s.enable_group_offload failed: %s", attr, exc)
        if apply_group_offloading is not None:
            try:
                apply_group_offloading(
                    mod,
                    onload_device=kwargs["onload_device"],
                    offload_type="block_level",
                    num_blocks_per_group=2,
                    **({"offload_to_disk_path": offload_to_disk_path} if offload_to_disk_path else {}),
                )
                any_applied = True
            except Exception as exc:
                _LOG.debug("low_vram: apply_group_offloading(%s) failed: %s", attr, exc)

    if any_applied:
        applied["group_offload"] = True
        if offload_to_disk_path:
            applied["disk_offload_path"] = True
    return any_applied


def place_pipeline(pipeline: Any, *, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Worker-owned placement + offload policy for a freshly-loaded pipeline.

    Runs the one low-VRAM decider against free VRAM: plenty of headroom puts
    the whole pipeline on CUDA; tighter budgets step down the offload ladder.
    Endpoints never write device/offload code — the worker calls this around
    ``setup()`` injection. No-op without CUDA.
    """
    log = logger or _LOG
    try:
        import torch

        if not torch.cuda.is_available():
            _forbid_cpu_inference("CPU-only inference (no CUDA available)")
            return {"mode": "cpu"}
    except Exception:
        _forbid_cpu_inference("CPU-only inference (torch/CUDA unavailable)")
        return {"mode": "cpu"}
    mode = select_auto_mode(pipeline=pipeline)
    if mode in ("off", "vae_only") and callable(getattr(pipeline, "to", None)):
        try:
            pipeline.to("cuda")
        except Exception as exc:
            log.warning("place_pipeline: .to('cuda') failed: %s", exc)
    return apply_low_vram_config(pipeline, mode=mode, logger=log)


def apply_low_vram_config(
    pipeline: Any,
    *,
    mode: Mode = "auto",
    logger: Optional[logging.Logger] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
    offload_to_disk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a low-VRAM configuration to a diffusers pipeline.

    ``mode="auto"`` runs :func:`select_auto_mode` against free VRAM. Returns a
    dict describing what was applied. Idempotent per pipeline object.
    """
    log = logger or _LOG
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid low-VRAM mode: {mode!r}; expected one of {_VALID_MODES}")

    prior = getattr(pipeline, _COZY_MODE_ATTR, None)
    if prior is not None:
        return {"mode": prior, "already_applied": True}

    effective_mode = mode
    if effective_mode == "auto":
        effective_mode = select_auto_mode(
            pipeline=pipeline, model_size_gb=model_size_gb, peak_vram_gb=peak_vram_gb,
        )
        log.info("low_vram: auto-selected mode=%s", effective_mode)
    if effective_mode in _CPU_OFFLOAD_MODES:
        _forbid_cpu_inference(f"CPU-offloaded inference (mode={effective_mode})")

    applied: Dict[str, Any] = {
        "mode": effective_mode,
        "vae_slicing": False,
        "vae_tiling": False,
        "attention_slicing": False,
        "model_offload": False,
        "group_offload": False,
        "sequential_offload": False,
        "disk_offload_path": False,
        "already_applied": False,
    }

    if effective_mode == "off":
        setattr(pipeline, _COZY_MODE_ATTR, "off")
        return applied

    _apply_vae_and_attention(pipeline, applied)

    if effective_mode == "vae_only":
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: vae_only applied (%s)", _applied_summary(applied))
        return applied

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: CUDA unavailable, stopping at vae_only")
        return applied

    if offload_to_disk_path is None and _should_auto_disk_offload():
        offload_to_disk_path = _default_disk_offload_path()
        if offload_to_disk_path:
            log.warning(
                "low_vram: CPU RAM tight (%.1f GB free); enabling disk offload at %s",
                get_available_ram_gb(), offload_to_disk_path,
            )

    if effective_mode == "model_offload":
        ok = _call_if_present(pipeline, "enable_model_cpu_offload")
        if not ok:
            try:
                pipeline.enable_model_cpu_offload(gpu_id=0)
                ok = True
            except Exception as exc:
                log.warning("low_vram: enable_model_cpu_offload failed: %s", exc)
        applied["model_offload"] = ok
        setattr(pipeline, _COZY_MODE_ATTR, "model_offload")
        log.info("low_vram: model_offload applied (%s)", _applied_summary(applied))
        return applied

    if effective_mode == "group_offload":
        ok = _apply_group_offload(pipeline, applied, offload_to_disk_path=offload_to_disk_path)
        if not ok:
            log.warning("low_vram: group_offload unavailable; falling back to sequential")
            effective_mode = "sequential"

    if effective_mode == "sequential":
        _move_pipeline_to_cpu(pipeline)
        flush_memory()
        ok = _call_if_present(pipeline, "enable_sequential_cpu_offload")
        if not ok:
            try:
                pipeline.enable_sequential_cpu_offload(gpu_id=0)
                ok = True
            except Exception as exc:
                log.error("low_vram: enable_sequential_cpu_offload failed: %s", exc)
        applied["sequential_offload"] = ok
        applied["mode"] = "sequential"
        setattr(pipeline, _COZY_MODE_ATTR, "sequential")
        log.info("low_vram: sequential_offload applied (%s)", _applied_summary(applied))
        return applied

    setattr(pipeline, _COZY_MODE_ATTR, effective_mode)
    log.info("low_vram: %s applied (%s)", effective_mode, _applied_summary(applied))
    return applied


def _applied_summary(applied: Dict[str, Any]) -> str:
    keys = [k for k, v in applied.items() if v and k not in ("mode", "already_applied")]
    return ",".join(keys) or "none"


def _should_auto_disk_offload() -> bool:
    ram = get_available_ram_gb()
    return 0.0 < ram < 16.0


def _default_disk_offload_path() -> Optional[str]:
    try:
        p = "/tmp/cozy-offload"
        os.makedirs(p, exist_ok=True)
        return p
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OOM retry
# ---------------------------------------------------------------------------


_DEFAULT_ESCALATION: tuple[str, ...] = (
    "vae_only", "model_offload", "group_offload", "sequential",
)

T = TypeVar("T")


def _escalate_pipeline_mode(
    pipeline: Any,
    *,
    logger: logging.Logger,
    escalation: tuple[str, ...],
) -> bool:
    """Move a pipeline one step further up the offload ladder. False if maxed."""
    cur = getattr(pipeline, _COZY_MODE_ATTR, None)
    idx = escalation.index(cur) if cur in escalation else -1
    next_idx = idx + 1
    if next_idx >= len(escalation):
        return False
    next_mode = escalation[next_idx]
    try:
        delattr(pipeline, _COZY_MODE_ATTR)
    except Exception:
        try:
            setattr(pipeline, _COZY_MODE_ATTR, None)
        except Exception:
            pass
    logger.warning("low_vram: escalating %s -> %s on OOM", cur, next_mode)
    apply_low_vram_config(pipeline, mode=next_mode, logger=logger)
    return True


def with_oom_retry(
    fn: Callable[..., T],
    *args: Any,
    pipelines: Optional[List[Any]] = None,
    max_retries: int = 2,
    escalation: tuple[str, ...] = _DEFAULT_ESCALATION,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> T:
    """Call ``fn``; on ``torch.cuda.OutOfMemoryError`` flush, escalate offload
    one ladder step on each pipeline, retry (up to ``max_retries`` extra)."""
    log = logger or _LOG
    try:
        import torch

        OOMType = torch.cuda.OutOfMemoryError
    except Exception:
        return fn(*args, **kwargs)

    last_exc: Optional[BaseException] = None
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except OOMType as exc:
            last_exc = exc
            flush_memory()
            if attempt == attempts - 1:
                break
            escalated_any = False
            for pipe in (pipelines or []):
                if _escalate_pipeline_mode(pipe, logger=log, escalation=escalation):
                    escalated_any = True
            if not escalated_any and not pipelines:
                log.warning("low_vram: OOM (attempt %d/%d), retrying after flush", attempt + 1, attempts)
                continue
            if not escalated_any:
                log.warning("low_vram: OOM (attempt %d/%d); all pipelines already at max offload", attempt + 1, attempts)
                break
            log.warning("low_vram: OOM (attempt %d/%d), retrying with escalated offload", attempt + 1, attempts)
    assert last_exc is not None
    raise last_exc


def low_vram_mode(pipeline: Any) -> str:
    """The low-VRAM mode :func:`apply_low_vram_config` prepped this pipeline
    with ('' when never prepped). Part of the compile-cache graph key (gw#391):
    the flags are traced into the FX graphs."""
    return str(getattr(pipeline, _COZY_MODE_ATTR, "") or "")


__all__ = [
    "apply_low_vram_config",
    "low_vram_mode",
    "place_pipeline",
    "with_oom_retry",
    "select_auto_mode",
    "device_mismatches",
    "repair_device_placement",
    "estimate_pipeline_size_gb",
    "estimate_cuda_resident_gb",
    "cuda_allocated_bytes",
    "get_available_vram_gb",
    "GPU_VRAM_OVERHEAD_GB",
    "effective_vram_requirement_gb",
    "get_available_ram_gb",
    "get_total_ram_gb",
    "flush_memory",
]
