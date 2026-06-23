"""
Robust low-VRAM inference helpers for diffusers pipelines.

This module provides a progressive offload ladder and OOM-retry machinery so
that inference on hosts with less VRAM than a model nominally requires degrades
gracefully (slower) instead of crashing with ``torch.cuda.OutOfMemoryError``.

Ladder (auto mode, least-aggressive first):

  off           : no optimizations (pipeline on CUDA as-is)
  vae_only      : VAE slicing + VAE tiling (+ attention slicing when available)
  model_offload : vae_only + ``enable_model_cpu_offload()``  (~10% slower)
  group_offload : leaf-level group offload with CUDA streams   (~25% slower)
  sequential    : ``enable_sequential_cpu_offload()``          (~50%+ slower)

Defaults are chosen from upstream diffusers docs:
  https://huggingface.co/docs/diffusers/main/en/optimization/memory

Upstream foot-gun: ``enable_sequential_cpu_offload`` must NOT be called on a
pipeline that was already moved to CUDA, or the memory savings are minimal.
``apply_low_vram_config`` moves the pipeline back to CPU first when escalating
to sequential mode.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar

_LOG = logging.getLogger(__name__)

Mode = str  # Literal["auto", "off", "vae_only", "model_offload", "group_offload", "sequential"]

_VALID_MODES: tuple[str, ...] = (
    "auto",
    "off",
    "vae_only",
    "model_offload",
    "group_offload",
    "sequential",
)

# Default VRAM thresholds in GB for auto mode.
# These match the de-facto standards used by the qwen-image and
# multi-sdxl-checkpoints endpoints before centralization.
_DEFAULT_VAE_SLICE_THRESHOLD_GB = 10.0
_DEFAULT_MODEL_OFFLOAD_THRESHOLD_GB = 8.0
_DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB = 6.0
# Safety margin below total VRAM we reserve for activations.
_DEFAULT_SAFETY_MARGIN_GB = 2.0
# When a model fits with at least this much VRAM still free beyond it, run
# fully unoptimized (mode "off") — VAE slicing/tiling only trades speed for
# memory we demonstrably don't need. Below this slack we keep "vae_only" as a
# cheap guard against high-resolution VAE-decode spikes.
_DEFAULT_OFF_HEADROOM_GB = 8.0

# Sentinel attribute set on pipelines to make apply_low_vram_config idempotent.
_COZY_MODE_ATTR = "_cozy_low_vram_mode"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def get_total_vram_gb(device_index: int = 0) -> float:
    """Total VRAM on the selected CUDA device. 0.0 if no CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(device_index)
        return float(props.total_memory) / float(1024**3)
    except Exception:
        return 0.0


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


def estimate_pipeline_size_gb(pipeline: Any) -> float:
    """
    Best-effort size estimate for a loaded pipeline via parameter bytes.
    Returns 0.0 when torch is unavailable or the pipeline has no parameters.
    """
    try:
        import torch

        total = 0
        seen: set[int] = set()
        # Diffusers pipelines expose `components` dict; fall back to attrs.
        comps: List[Any] = []
        raw_components = getattr(pipeline, "components", None)
        if isinstance(raw_components, dict):
            comps.extend(raw_components.values())
        else:
            for attr in ("unet", "transformer", "vae", "text_encoder",
                         "text_encoder_2", "text_encoder_3"):
                v = getattr(pipeline, attr, None)
                if v is not None:
                    comps.append(v)

        for c in comps:
            if c is None or not hasattr(c, "parameters"):
                continue
            for p in c.parameters():
                if not isinstance(p, torch.Tensor):
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                total += p.numel() * p.element_size()
            # Also count buffers (norm stats etc.)
            if hasattr(c, "buffers"):
                for b in c.buffers():
                    if not isinstance(b, torch.Tensor):
                        continue
                    bid = id(b)
                    if bid in seen:
                        continue
                    seen.add(bid)
                    total += b.numel() * b.element_size()
        return float(total) / float(1024**3)
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
# Mode selection (auto)
# ---------------------------------------------------------------------------


def select_auto_mode(
    *,
    pipeline: Any,
    total_vram_gb: Optional[float] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
) -> str:
    """
    Pick the least-aggressive ladder step that should keep the pipeline in memory.

    Decision logic:
      - no CUDA                                -> off
      - model fits with generous headroom      -> off
      - model fits (incl. safety margin)       -> vae_only
      - total VRAM <= group_offload_threshold  -> group_offload
      - total VRAM <= model_offload_threshold  -> model_offload
      - total VRAM <= vae_slice_threshold      -> vae_only
      - requirement > (total - margin)         -> model_offload
      - otherwise                              -> vae_only

    ``peak_vram_gb`` is the endpoint's DECLARED peak VRAM during one request
    (``Resources.peak_vram_per_request_gb``, issue #339). When provided it
    drives the fit check instead of the framework guessing from weights alone:
    the requirement becomes ``max(model_gb, peak_vram_gb)`` so a tenant who
    declares a large per-request peak (big activations / many images) offloads
    sooner, while one who declares a small peak still never drops below the
    measured weight footprint. Absent a declaration the behavior is unchanged.
    """
    total = total_vram_gb if total_vram_gb is not None else get_total_vram_gb()
    if total <= 0.0:
        return "off"

    model_gb = model_size_gb if model_size_gb is not None else estimate_pipeline_size_gb(pipeline)
    # Declared per-request peak (#339) raises — never lowers — the requirement.
    requirement = model_gb
    if peak_vram_gb is not None and peak_vram_gb > 0.0:
        requirement = max(model_gb, float(peak_vram_gb))
    margin = _DEFAULT_SAFETY_MARGIN_GB
    t_group = _DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB
    t_model = _DEFAULT_MODEL_OFFLOAD_THRESHOLD_GB
    t_vae = _DEFAULT_VAE_SLICE_THRESHOLD_GB

    # Very low VRAM: even a model that "fits" needs aggressive help for activations.
    if total <= t_group:
        return "group_offload"

    # Size/peak-aware (the key efficiency rule): when we know the requirement
    # and it fits within (total VRAM - activation margin), keep it FULLY RESIDENT
    # (no offload) even on a modest card — offload ONLY when it genuinely won't
    # fit. This avoids the ~10-50% offload penalty for models that fit (e.g. sd1.5
    # on an 8GB card) while big models (e.g. SDXL @1024 on an 8GB card) still
    # offload. Falls through to total-VRAM thresholds only when size is unknown.
    if requirement > 0.0:
        usable = max(0.0, total - margin)
        if requirement > usable:
            return "model_offload"
        # Fits. With generous headroom, run fully unoptimized — slicing/tiling
        # would only trade speed for memory we don't need. Otherwise keep the
        # cheap vae_only guard for high-res VAE-decode spikes.
        if (usable - requirement) >= _DEFAULT_OFF_HEADROOM_GB:
            return "off"
        return "vae_only"

    # Unknown model size: conservative total-VRAM thresholds.
    if total <= t_model:
        return "model_offload"
    if total <= t_vae:
        return "vae_only"
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
        # Some signatures accept fewer kwargs on older diffusers.
        try:
            fn()
            return True
        except Exception:
            return False
    except Exception as exc:
        _LOG.debug("low_vram: %s() raised %s", method, exc)
        return False


def _move_pipeline_to_cpu(pipeline: Any) -> None:
    """Move the pipeline back to CPU (required before sequential offload)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if hasattr(pipeline, "to") and callable(getattr(pipeline, "to", None)):
            pipeline.to("cpu")
    except Exception as exc:
        _LOG.debug("low_vram: move-to-cpu failed: %s", exc)


def _apply_vae_and_attention(pipeline: Any, applied: Dict[str, bool]) -> None:
    # VAE slicing / tiling: prefer the pipeline-level methods when present,
    # otherwise reach into the VAE module directly.
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

    # Attention slicing is UNet-flavored; safe no-op on pipelines that don't expose it.
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

    onload = torch.device("cuda")
    offload = torch.device("cpu")
    kwargs: Dict[str, Any] = {
        "onload_device": onload,
        "offload_device": offload,
        "offload_type": "leaf_level",
        "use_stream": True,
    }
    if offload_to_disk_path:
        kwargs["offload_to_disk_path"] = offload_to_disk_path

    # Prefer the pipeline-level entry point when diffusers exposes it.
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

    # Fallback: apply to individual components that support it.
    any_applied = False
    try:
        from diffusers.hooks import apply_group_offloading  # type: ignore
    except Exception:
        apply_group_offloading = None  # type: ignore

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
                    onload_device=onload,
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


def apply_low_vram_config(
    pipeline: Any,
    *,
    mode: Mode = "auto",
    logger: Optional[logging.Logger] = None,
    model_size_gb: Optional[float] = None,
    peak_vram_gb: Optional[float] = None,
    offload_to_disk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply a low-VRAM configuration to a diffusers pipeline.

    Args:
      pipeline: diffusers pipeline or a component exposing ``enable_*`` methods.
      mode: "auto" | "off" | "vae_only" | "model_offload" | "group_offload" | "sequential".
            An env override ``COZY_INFERENCE_MEMORY_MODE`` takes precedence when set.
      logger: optional logger for user-facing INFO lines.
      model_size_gb: precomputed model size in GB (skips the probe).
      peak_vram_gb: endpoint-declared peak VRAM per request
            (``Resources.peak_vram_per_request_gb``, #339). In ``auto`` mode it
            raises the fit requirement so a declared-heavy endpoint offloads
            sooner; ignored for explicit modes.
      offload_to_disk_path: if set, group-offload stores offloaded weights on disk
            instead of CPU RAM. Required when CPU RAM is insufficient.

    Returns a dict describing what was applied, with keys:
      mode, vae_slicing, vae_tiling, attention_slicing, model_offload,
      group_offload, sequential_offload, disk_offload_path, already_applied.
    """
    log = logger or _LOG

    if mode not in _VALID_MODES:
        raise ValueError(f"invalid low-VRAM mode: {mode!r}; expected one of {_VALID_MODES}")

    effective_mode = mode

    # Idempotency
    prior = getattr(pipeline, _COZY_MODE_ATTR, None)
    if prior is not None:
        return {"mode": prior, "already_applied": True}

    if effective_mode == "auto":
        effective_mode = select_auto_mode(
            pipeline=pipeline, model_size_gb=model_size_gb, peak_vram_gb=peak_vram_gb,
        )
        log.info("low_vram: auto-selected mode=%s", effective_mode)

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

    # vae_only (and every escalation above it) always turns on VAE tiling/slicing.
    _apply_vae_and_attention(pipeline, applied)

    if effective_mode == "vae_only":
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: vae_only applied (%s)", _applied_summary(applied))
        return applied

    # For model/group/sequential offload we need CUDA.
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        # No GPU: nothing more to do.
        setattr(pipeline, _COZY_MODE_ATTR, "vae_only")
        log.info("low_vram: CUDA unavailable, stopping at vae_only")
        return applied

    # Auto disk-offload when CPU RAM is tight.
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
                # Older diffusers builds require gpu_id.
                pipeline.enable_model_cpu_offload(gpu_id=0)  # type: ignore[attr-defined]
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
            # Fall through to sequential if group offload is not available.
            log.warning("low_vram: group_offload unavailable; falling back to sequential")
            effective_mode = "sequential"

    if effective_mode == "sequential":
        # Upstream requires pipeline NOT be on CUDA before enabling sequential offload.
        _move_pipeline_to_cpu(pipeline)
        flush_memory()
        ok = _call_if_present(pipeline, "enable_sequential_cpu_offload")
        if not ok:
            try:
                pipeline.enable_sequential_cpu_offload(gpu_id=0)  # type: ignore[attr-defined]
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
    if ram <= 0.0:
        return False
    threshold = 16.0
    return ram < threshold


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
    "vae_only",
    "model_offload",
    "group_offload",
    "sequential",
)


T = TypeVar("T")


def _escalate_pipeline_mode(
    pipeline: Any,
    *,
    logger: logging.Logger,
    escalation: tuple[str, ...],
) -> bool:
    """Move a pipeline one step further up the offload ladder. Returns False if already maxed."""
    cur = getattr(pipeline, _COZY_MODE_ATTR, None)
    try:
        idx = escalation.index(cur) if cur in escalation else -1
    except ValueError:
        idx = -1
    next_idx = idx + 1
    if next_idx >= len(escalation):
        return False
    next_mode = escalation[next_idx]
    # Reset the sentinel so apply_low_vram_config re-runs.
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
    """
    Call ``fn(*args, **kwargs)``. On ``torch.cuda.OutOfMemoryError``, flush memory,
    escalate offload on each pipeline in ``pipelines`` by one ladder step, and retry.

    Stops after ``max_retries`` additional attempts (default: 2).
    Raises the last OOM if escalation cannot recover.
    """
    log = logger or _LOG
    try:
        import torch

        OOMType = torch.cuda.OutOfMemoryError
    except Exception:
        # No torch: degenerate case - just call.
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
                # No pipelines to escalate; just retry after flushing.
                log.warning("low_vram: OOM (attempt %d/%d), retrying after flush", attempt + 1, attempts)
                continue
            if not escalated_any:
                log.warning("low_vram: OOM (attempt %d/%d); all pipelines already at max offload", attempt + 1, attempts)
                break
            log.warning("low_vram: OOM (attempt %d/%d), retrying with escalated offload", attempt + 1, attempts)
    assert last_exc is not None
    raise last_exc


__all__ = [
    "apply_low_vram_config",
    "with_oom_retry",
    "select_auto_mode",
    "estimate_pipeline_size_gb",
    "get_total_vram_gb",
    "get_available_vram_gb",
    "get_available_ram_gb",
    "flush_memory",
]
