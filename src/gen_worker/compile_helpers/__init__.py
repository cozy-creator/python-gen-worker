"""Compilation backends for diffusion / DiT modules (#324).

Wrappers around torch.compile, OneDiff-Nexfort, and TensorRT to compile
the heavy module of a Diffusers pipeline (usually `pipe.transformer`).

Naming: this module is `gen_worker.compile_helpers` (not `gen_worker.compile`)
because `compile` shadows the Python builtin and confuses type-checkers.
Re-exported in the top-level package as ``gen_worker.compile``.

Tenant usage:

    @inference(models={"pipe": flux_klein})
    class FluxKleinGenerate:
        def setup(self, pipe):
            pipe.transformer = gen_worker.compile.torch_compile(
                pipe.transformer, mode='reduce-overhead'
            )
            ...

The SDK sets TORCHINDUCTOR_CACHE_DIR before setup() runs so compile
artifacts persist across worker restarts (#322 cross-cutting hook).

All third-party dependencies (onediff, nexfort, nvidia-modelopt) are
imported lazily inside the helper so importing this module without
calling the helper does not require the deps to be installed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

logger = logging.getLogger(__name__)


class CompileUnavailableError(RuntimeError):
    """Raised when a compile helper's third-party dependency is missing."""


def torch_compile(
    module: Any,
    *,
    mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> Any:
    """Compile a torch.nn.Module via torch.compile.

    Args:
        module: the module to compile (typically ``pipe.transformer`` or
            ``pipe.unet``).
        mode: torch.compile mode. ``reduce-overhead`` is the recommended
            default — adds CUDA graph capture on top of Inductor codegen.
        fullgraph: if True, require the trace to capture the entire graph
            (no Python fallbacks). Use only when the module is known to
            trace cleanly.
        dynamic: if True, allow dynamic shapes (more recompiles but
            handles variable input shapes). Off by default — endpoints
            should declare `allowed_shapes=[(H, W)]` on the class instead.

    Returns the compiled module. Compilation is LAZY — the actual compile
    fires on the first forward pass that hits a new shape. Tenants must
    call ``warmup()`` to amortize the compile out of the request path.
    """
    try:
        import torch
    except ImportError as e:
        raise CompileUnavailableError("torch_compile requires PyTorch") from e
    return torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


def nexfort_compile(
    module: Any,
    *,
    mode: str = "max-autotune:cudagraphs",
    fullgraph: bool = True,
    dynamic: bool = False,
    options: dict[str, Any] | None = None,
) -> Any:
    """Compile via SiliconFlow's OneDiff / Nexfort backend.

    Nexfort is OneDiff's torch.compile backend (https://github.com/siliconflow/onediff).
    It plugs into torch.compile via ``backend="nexfort"``. On supported DiTs
    (Flux, SD3, etc.) it typically beats stock Inductor + reduce-overhead by
    another 10-30%.

    Args:
        module: the module to compile (typically ``pipe.transformer``).
        mode: nexfort mode string. ``max-autotune:cudagraphs`` is the upstream
            recommended default.
        fullgraph: require the trace to capture the entire graph. Default
            True — nexfort generally requires fullgraph.
        dynamic: allow dynamic shapes.
        options: extra options forwarded to torch.compile.

    Requires (lazy-imported here):
        pip install nexfort onediff  # Apache 2.0
    """
    try:
        import torch
    except ImportError as e:
        raise CompileUnavailableError("nexfort_compile requires PyTorch") from e

    # Probe nexfort backend availability before calling torch.compile so we
    # can give a clear install hint instead of a deep torch.compile traceback.
    try:
        import nexfort  # noqa: F401
    except ImportError as e:
        raise CompileUnavailableError(
            "nexfort_compile requires `nexfort` + `onediff`. Install with "
            "`pip install nexfort onediff` (Apache 2.0) and rebuild the "
            "endpoint image."
        ) from e

    compile_options = dict(options or {})
    compiled = torch.compile(
        module,
        backend="nexfort",
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        options=compile_options,
    )
    logger.info(
        "nexfort_compile applied (mode=%s, fullgraph=%s, dynamic=%s)",
        mode,
        fullgraph,
        dynamic,
    )
    return compiled


# Backwards-compat alias used by earlier drafts.
def onediff(module: Any, **kwargs: Any) -> Any:
    """Deprecated alias for :func:`nexfort_compile`.

    Older endpoint code referenced ``gen_worker.compile.onediff``; the
    public name is now ``nexfort_compile`` to make the backend explicit.
    Kept around so already-pinned endpoint code keeps importing.
    """
    return nexfort_compile(module, **kwargs)


def _detect_sm_major() -> int | None:
    """Best-effort GPU SM major-version detector.

    Returns the major capability of CUDA device 0 (e.g. 9 for Hopper,
    10 for Blackwell), or None when torch/CUDA isn't available. Used by
    ``tensorrt()`` to no-op on pre-Hopper hardware.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major)
    except Exception:  # noqa: BLE001 — device probing is best-effort
        return None


def tensorrt(
    module: Any,
    *,
    precision: Literal["fp16", "bf16", "fp8", "nvfp4"] = "bf16",
    min_sm: int = 9,
    fallback: Literal["torch_compile", "raise", "passthrough"] = "torch_compile",
    **kwargs: Any,
) -> Any:
    """Compile via TensorRT (NVIDIA Model-Optimizer torch->TRT bridge).

    Uses modelopt.torch._deploy or the ``torch_tensorrt`` package depending
    on what's installed. On pre-Hopper hardware (SM < ``min_sm``, default 9)
    this is a no-op with a warning — TRT compile on Ampere typically loses
    to a well-tuned torch.compile run, and TRT 10 + Model-Optimizer's NVFP4
    path needs Hopper or newer regardless.

    Args:
        module: the module to compile.
        precision: target TRT precision. ``nvfp4`` requires Blackwell (SM 10+).
        min_sm: minimum CUDA SM major to attempt TRT compile. Below this,
            falls back per ``fallback``.
        fallback: what to do on unsupported hardware or missing deps:
            ``torch_compile`` (default) reuses :func:`torch_compile`;
            ``raise`` surfaces ``CompileUnavailableError``;
            ``passthrough`` returns the module unchanged.
        **kwargs: extra options forwarded to the underlying TRT compiler.
    """
    sm = _detect_sm_major()
    if sm is not None and sm < min_sm:
        msg = (
            f"tensorrt() skipping TRT compile: detected SM {sm}.x, "
            f"requires SM {min_sm}+ (Hopper or newer). "
            f"Falling back to {fallback}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        if fallback == "raise":
            raise CompileUnavailableError(msg)
        if fallback == "passthrough":
            return module
        return torch_compile(module)

    # Try Model-Optimizer's torch->TRT deployment path first.
    modelopt_compile: Any
    try:
        from modelopt.torch._deploy import compile as _modelopt_compile

        modelopt_compile = _modelopt_compile
    except ImportError:
        modelopt_compile = None

    if modelopt_compile is not None:
        try:
            compiled = modelopt_compile(module, precision=precision, **kwargs)
            logger.info(
                "tensorrt() compiled via modelopt.torch._deploy (precision=%s, sm=%s)",
                precision,
                sm,
            )
            return compiled
        except Exception as e:  # noqa: BLE001 — modelopt path is optional
            msg = (
                "tensorrt() modelopt deploy path failed: "
                f"{type(e).__name__}: {e}. Falling back."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning(msg)

    # Try torch_tensorrt as a secondary path.
    torch_tensorrt: Any
    try:
        import torch_tensorrt as _torch_tensorrt

        torch_tensorrt = _torch_tensorrt
    except ImportError:
        torch_tensorrt = None

    if torch_tensorrt is not None:
        try:
            import torch

            compiled = torch.compile(module, backend="torch_tensorrt", **kwargs)
            logger.info("tensorrt() compiled via torch_tensorrt (precision=%s)", precision)
            return compiled
        except Exception as e:  # noqa: BLE001
            msg = (
                "tensorrt() torch_tensorrt path failed: "
                f"{type(e).__name__}: {e}. Falling back."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning(msg)

    # All TRT paths unavailable.
    miss = (
        "tensorrt() found neither `modelopt` nor `torch_tensorrt`. "
        "Install with `pip install nvidia-modelopt` (Apache 2.0). "
        f"Falling back to {fallback}."
    )
    warnings.warn(miss, RuntimeWarning, stacklevel=2)
    logger.warning(miss)
    if fallback == "raise":
        raise CompileUnavailableError(miss)
    if fallback == "passthrough":
        return module
    return torch_compile(module)


__all__ = [
    "torch_compile",
    "nexfort_compile",
    "onediff",
    "tensorrt",
    "CompileUnavailableError",
]
