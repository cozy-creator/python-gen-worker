"""Quantization helpers for diffusion / DiT models (#324).

.. note::
   For the common case, prefer :func:`gen_worker.accel.apply_nvfp4` —
   the canonical five-call surface in :mod:`gen_worker.accel` covers most
   SerialWorker endpoints. This module remains available for advanced
   cases (FP8 / INT8 fallbacks, custom calibration, multi-precision
   quant pipelines).

Wraps NVIDIA Model-Optimizer (Apache 2.0, open source) for NVFP4 / FP8
weight quantization, and bitsandbytes for the INT8 fallback. Calibration
artifacts are content-addressed-cached via a sibling of #322's compile
cache (``$TORCHINDUCTOR_CACHE_DIR/../gen_worker_quant_cache``, or
``$GEN_WORKER_QUANT_CACHE_DIR`` if set explicitly).

Tenant usage:

    @inference(models={"pipe": flux_klein})
    class FluxKleinGenerate:
        def setup(self, pipe):
            pipe.transformer = gen_worker.quant.nvfp4(pipe.transformer)
            ...

NVFP4 gives ~3-6x on Blackwell (B200) for Flux.2 per NVIDIA's measurements.
FP8 gives ~40% VRAM reduction on H100 SXM. INT8 (bitsandbytes) is the
fallback for older / consumer hardware.

All third-party imports are lazy: importing this module without calling
the helpers does not require nvidia-modelopt or bitsandbytes to be
installed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


class QuantUnavailableError(RuntimeError):
    """Raised when a quant helper's third-party dependency is missing."""


# ---------------------------------------------------------------------------
# Cache-dir resolution (sibling of #322's compile cache)
# ---------------------------------------------------------------------------


def _quant_cache_root() -> Path:
    """Resolve the calibration-artifact cache root.

    Resolution order:
      1. ``$GEN_WORKER_QUANT_CACHE_DIR`` — explicit override.
      2. ``$TORCHINDUCTOR_CACHE_DIR/../gen_worker_quant_cache`` — sibling of
         the compile cache set by #322 so both live on the same persistent
         volume.
      3. ``~/.cache/gen_worker/quant`` — last-resort local fallback.

    The directory is created on demand.
    """
    explicit = os.environ.get("GEN_WORKER_QUANT_CACHE_DIR")
    if explicit:
        root = Path(explicit)
    else:
        inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if inductor:
            root = Path(inductor).parent / "gen_worker_quant_cache"
        else:
            root = Path.home() / ".cache" / "gen_worker" / "quant"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(
    method: str,
    module: Any,
    extra: dict[str, Any] | None = None,
) -> str:
    """Compute a content-addressed key for a (method, module, env) tuple.

    Hashed inputs:
      - quant method name ("nvfp4" / "fp8" / "int8" / variant)
      - module class fully-qualified name
      - sorted state_dict keys + shapes (cheap structural fingerprint)
      - GPU compute capability (major.minor)
      - torch version
      - extra kwargs (e.g. quant scheme, group_size)

    NOTE: we hash structure (keys + shapes) rather than weight bytes —
    that keeps the key cheap while still catching shape mismatches.
    Weight content is identified upstream via #322's compile-cache key
    which folds in `model_digest` from the manifest.
    """
    h = hashlib.sha256()
    h.update(method.encode("utf-8"))
    h.update(b"|cls=")
    h.update(f"{type(module).__module__}.{type(module).__qualname__}".encode("utf-8"))

    try:
        state = module.state_dict()
    except Exception:  # noqa: BLE001 — non-nn.Module modules are still valid input
        state = {}

    for k in sorted(state.keys()):
        v = state[k]
        h.update(b"|k=")
        h.update(k.encode("utf-8"))
        shape = getattr(v, "shape", None)
        if shape is not None:
            h.update(f"|s={tuple(shape)}".encode("utf-8"))
        dtype = getattr(v, "dtype", None)
        if dtype is not None:
            h.update(f"|d={dtype}".encode("utf-8"))

    try:
        import torch

        h.update(f"|tv={torch.__version__}".encode("utf-8"))
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            h.update(f"|cc={cc}".encode("utf-8"))
    except Exception:  # noqa: BLE001
        pass

    if extra:
        for k in sorted(extra.keys()):
            h.update(f"|x={k}={extra[k]}".encode("utf-8"))

    return h.hexdigest()


def _cache_path(method: str, key: str) -> Path:
    root = _quant_cache_root()
    bucket = root / method / key[:2]
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket / f"{key}.safetensors"


# ---------------------------------------------------------------------------
# Hardware capability detection
# ---------------------------------------------------------------------------


def _detect_sm_major() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# modelopt shared path
# ---------------------------------------------------------------------------


def _modelopt_quantize(
    module: Any,
    *,
    method: str,
    config_name: str,
    calib_loader: Iterable[Any] | None,
    forward_loop: Callable[[Any], None] | None,
    extra: dict[str, Any] | None,
) -> Any:
    """Apply nvidia-modelopt PTQ with caching.

    method: "nvfp4" | "fp8" — selects the cache namespace.
    config_name: modelopt cfg key (e.g. "NVFP4_DEFAULT_CFG").
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise QuantUnavailableError("quantization requires PyTorch") from e

    try:
        import modelopt.torch.quantization as mtq
    except ImportError as e:
        raise QuantUnavailableError(
            "gen_worker.quant requires `nvidia-modelopt`. Install with "
            "`pip install nvidia-modelopt` (Apache 2.0) and rebuild the "
            "endpoint image."
        ) from e

    config = getattr(mtq, config_name, None)
    if config is None:
        # Newer versions expose configs via a dict; older versions via attrs.
        config = getattr(mtq, "CONFIG_CHOICES", {}).get(config_name)
    if config is None:
        raise QuantUnavailableError(
            f"modelopt config '{config_name}' not found in installed "
            f"modelopt.torch.quantization. Upgrade nvidia-modelopt."
        )

    key = _cache_key(method, module, extra=extra)
    artifact = _cache_path(method, key)

    # Cache hit: restore quantizer state without re-running calibration.
    if artifact.exists():
        try:
            import safetensors.torch as st
            state = st.load_file(str(artifact))
            # modelopt patches the module in-place; we still need to enable
            # quantizers + load the scale/zero-point state.
            mtq.quantize(module, config, forward_loop=lambda m: None)
            module.load_state_dict(state, strict=False)
            logger.info(
                "%s cache hit: loaded calibration artifact from %s",
                method,
                artifact,
            )
            return module
        except Exception as e:  # noqa: BLE001 — bad cache = re-calibrate
            warnings.warn(
                f"{method} cache hit at {artifact} failed to load "
                f"({type(e).__name__}: {e}); re-calibrating.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Cache miss: calibrate.
    def _loop(m: Any) -> None:
        if forward_loop is not None:
            forward_loop(m)
            return
        if calib_loader is None:
            return
        for batch in calib_loader:
            if isinstance(batch, dict):
                m(**batch)
            elif isinstance(batch, (list, tuple)):
                m(*batch)
            else:
                m(batch)

    mtq.quantize(module, config, forward_loop=_loop)

    # Persist calibration state.
    try:
        import safetensors.torch as st
        st.save_file(module.state_dict(), str(artifact))
        logger.info(
            "%s calibrated and cached to %s",
            method,
            artifact,
        )
    except Exception as e:  # noqa: BLE001 — caching failure is non-fatal
        warnings.warn(
            f"{method} succeeded but failed to persist artifact to "
            f"{artifact} ({type(e).__name__}: {e}); next run will recalibrate.",
            RuntimeWarning,
            stacklevel=2,
        )

    return module


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nvfp4(
    module: Any,
    *,
    calib_loader: Iterable[Any] | None = None,
    forward_loop: Callable[[Any], None] | None = None,
    min_sm: int = 10,
    fallback: str = "fp8",
    **extra: Any,
) -> Any:
    """Apply NVFP4 weight quantization (Blackwell hardware).

    NVFP4 is NVIDIA's 4-bit floating-point quantization scheme. Requires
    Blackwell-class GPUs (SM 10.x, B100/B200/B300). On pre-Blackwell
    hardware we fall back per ``fallback`` (default: FP8).

    Args:
        module: the module to quantize (typically ``pipe.transformer``).
        calib_loader: optional iterable yielding calibration batches
            (dict / tuple / single tensor). One of ``calib_loader`` or
            ``forward_loop`` must be provided to drive modelopt's PTQ.
        forward_loop: optional callable ``forward_loop(module)`` that
            performs one or more representative forward passes.
        min_sm: minimum SM major required. Default 10 (Blackwell).
        fallback: ``"fp8"`` | ``"int8"`` | ``"passthrough"`` | ``"raise"``
            when the device is too old.
        **extra: additional fields folded into the cache key (e.g.
            ``group_size=16``, ``scheme="weight_only"``).
    """
    sm = _detect_sm_major()
    if sm is not None and sm < min_sm:
        msg = (
            f"nvfp4() detected SM {sm}.x, requires SM {min_sm}+ "
            f"(Blackwell). Falling back to '{fallback}'."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        if fallback == "passthrough":
            return module
        if fallback == "raise":
            raise QuantUnavailableError(msg)
        if fallback == "fp8":
            return fp8(
                module,
                calib_loader=calib_loader,
                forward_loop=forward_loop,
                **extra,
            )
        if fallback == "int8":
            return int8(module, **extra)
        raise QuantUnavailableError(f"unknown nvfp4 fallback '{fallback}'")

    return _modelopt_quantize(
        module,
        method="nvfp4",
        config_name="NVFP4_DEFAULT_CFG",
        calib_loader=calib_loader,
        forward_loop=forward_loop,
        extra=extra or None,
    )


def fp8(
    module: Any,
    *,
    calib_loader: Iterable[Any] | None = None,
    forward_loop: Callable[[Any], None] | None = None,
    min_sm: int = 9,
    fallback: str = "int8",
    **extra: Any,
) -> Any:
    """Apply FP8 weight quantization (Hopper / Blackwell hardware).

    FP8 with per-tensor scaling. Requires Hopper-class GPUs (SM 9.x) or
    newer. On pre-Hopper hardware we fall back per ``fallback``.

    Args:
        module: the module to quantize.
        calib_loader / forward_loop: same as :func:`nvfp4`.
        min_sm: minimum SM major. Default 9 (Hopper).
        fallback: ``"int8"`` | ``"passthrough"`` | ``"raise"``.
        **extra: cache-key extras.
    """
    sm = _detect_sm_major()
    if sm is not None and sm < min_sm:
        msg = (
            f"fp8() detected SM {sm}.x, requires SM {min_sm}+ (Hopper). "
            f"Falling back to '{fallback}'."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        if fallback == "passthrough":
            return module
        if fallback == "raise":
            raise QuantUnavailableError(msg)
        if fallback == "int8":
            return int8(module, **extra)
        raise QuantUnavailableError(f"unknown fp8 fallback '{fallback}'")

    return _modelopt_quantize(
        module,
        method="fp8",
        config_name="FP8_DEFAULT_CFG",
        calib_loader=calib_loader,
        forward_loop=forward_loop,
        extra=extra or None,
    )


def int8(
    module: Any,
    *,
    threshold: float = 6.0,
    skip_modules: list[str] | None = None,
    **extra: Any,
) -> Any:
    """Apply INT8 weight quantization (broadest hardware support).

    Uses bitsandbytes' LLM.int8() linear replacement. Works on all CUDA
    GPUs supported by bitsandbytes (Turing+). No calibration step needed —
    bitsandbytes uses online activation-aware quantization.

    Args:
        module: the module to quantize. nn.Linear submodules are replaced
            in-place with bnb.nn.Linear8bitLt.
        threshold: LLM.int8() outlier threshold. 6.0 is the upstream default.
        skip_modules: list of submodule attribute names to leave in fp16/bf16
            (e.g. ``["proj_out", "norm_out"]`` to keep precision on output
            heads).
        **extra: forwarded to bnb.nn.Linear8bitLt.
    """
    try:
        import torch
    except ImportError as e:
        raise QuantUnavailableError("int8() requires PyTorch") from e

    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise QuantUnavailableError(
            "int8() requires `bitsandbytes`. Install with "
            "`pip install bitsandbytes` and rebuild the endpoint image."
        ) from e

    skip = set(skip_modules or [])

    def _replace(parent: Any, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if name in skip or full in skip:
                continue
            if isinstance(child, torch.nn.Linear):
                new = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                    **extra,
                )
                new.weight = bnb.nn.Int8Params(
                    child.weight.data, requires_grad=False, has_fp16_weights=False
                )
                if child.bias is not None:
                    new.bias = child.bias
                setattr(parent, name, new)
            else:
                _replace(child, full)

    _replace(module)
    logger.info(
        "int8() replaced nn.Linear with bnb.Linear8bitLt (threshold=%s)",
        threshold,
    )
    return module


__all__ = [
    "nvfp4",
    "fp8",
    "int8",
    "QuantUnavailableError",
]
