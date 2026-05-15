"""SerialWorker diffusion acceleration helpers (#324).

A small, well-documented surface around the canonical Apache-2.0
acceleration stack. The entry points are designed to be called from a
:func:`@inference.setup` so that the per-request fast path stays clean.

Public surface — the only entry points tenants should call:

    from gen_worker import accel

    # 1. Probe hardware. Detection runs once and is cached.
    caps = accel.gpu_capability()
    if caps.has_nvfp4:
        ...

    # 2. Compile the heavy DiT module (torch.compile wrapper).
    pipe.transformer = accel.compile_diffusion(
        pipe.transformer, mode="reduce-overhead",
    )

    # 3. ParaAttention First-Block Cache (~1.5-2x on Flux/SD3/Qwen-Image).
    accel.apply_fbcache(pipe, residual_diff_threshold=0.12)

    # 4. ParaAttention general adapter (sequence-parallel inference).
    accel.apply_para_attn(pipe)

    # 5. NVFP4 weight quantization (Blackwell-only; no-op + warn elsewhere).
    accel.apply_nvfp4(model)

Design notes
------------

* **No required dependencies.** ``para_attn`` and ``nvidia-modelopt`` are
  tenant-installed via the endpoint image's ``pyproject.toml``. Every
  third-party integration imports lazily inside the helper so simply
  importing :mod:`gen_worker.accel` never forces a heavy install.
* **Go-style Python.** Free functions plus a frozen
  :class:`msgspec.Struct` capability report. No classes-with-methods.
* **Pure additive.** This module does not modify ``decorators.py``,
  ``worker.py``, or ``discovery/``. It is safe to land independently.
* **Complements (does not replace)** :mod:`gen_worker.cache`,
  :mod:`gen_worker.compile_helpers`, :mod:`gen_worker.quant`, and
  :mod:`gen_worker.parallelism`. The older modules carry richer surfaces
  (multiple cache backends, multi-precision quant fallbacks, sequence
  parallelism, etc.); :mod:`gen_worker.accel` is the tight five-call
  surface tenants reach for first.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Final, Literal

import msgspec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability report
# ---------------------------------------------------------------------------


Arch = Literal[
    "blackwell",
    "hopper",
    "ampere",
    "lovelace",
    "turing",
    "unknown",
    "none",
]


class GpuCapabilityReport(msgspec.Struct, frozen=True, kw_only=True):
    """Frozen snapshot of the host GPU's relevant capabilities.

    All fields are populated by :func:`gpu_capability`. The result is
    cached for the lifetime of the process so repeated calls are free.

    Attributes:
        arch: high-level architecture label. ``"none"`` means no CUDA
            device is visible to torch; ``"unknown"`` means a CUDA device
            is present but its compute capability didn't match any known
            family.
        compute_capability: ``"major.minor"`` string (e.g. ``"9.0"`` for
            Hopper, ``"10.0"`` for Blackwell datacenter). Empty string
            when no CUDA device is visible.
        device_name: GPU marketing name as reported by torch
            (e.g. ``"NVIDIA H100 80GB HBM3"``).
        vram_gb_total: total device memory in GiB, rounded to one decimal.
        gpu_count: number of visible CUDA devices.
        has_fp8: True when the host can execute FP8 tensor-core math
            natively (Hopper or newer).
        has_nvfp4: True when the host can execute NVFP4 tensor-core math
            natively (Blackwell or newer).
        torch_version: torch version string, or empty if torch isn't
            importable.
    """

    arch: Arch
    compute_capability: str
    device_name: str
    vram_gb_total: float
    gpu_count: int
    has_fp8: bool
    has_nvfp4: bool
    torch_version: str


# Module-level cache so repeated probes are O(1).
_CAPABILITY_CACHE: GpuCapabilityReport | None = None


def _classify_arch(major: int, minor: int) -> Arch:
    """Map (sm_major, sm_minor) to an architecture label.

    Reference: https://developer.nvidia.com/cuda-gpus

        - SM 10.x / 12.x — Blackwell (B100, B200, B300, RTX 50-series)
        - SM 9.x — Hopper (H100, H200, H800)
        - SM 8.9 — Lovelace (RTX 40-series, L40, L4)
        - SM 8.0 / 8.6 — Ampere (A100, A40, RTX 30-series)
        - SM 7.5 — Turing (T4, RTX 20-series)
    """
    if major >= 10:
        # SM 10.0 (Blackwell datacenter) and SM 12.0 (Blackwell consumer)
        # share the architecture name even though their feature sets
        # diverge slightly.
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "lovelace"
    if major == 8:
        return "ampere"
    if major == 7 and minor == 5:
        return "turing"
    return "unknown"


def _empty_report() -> GpuCapabilityReport:
    """The 'no CUDA device' report. Used on CI and on CPU-only hosts."""
    return GpuCapabilityReport(
        arch="none",
        compute_capability="",
        device_name="",
        vram_gb_total=0.0,
        gpu_count=0,
        has_fp8=False,
        has_nvfp4=False,
        torch_version="",
    )


def gpu_capability(*, refresh: bool = False) -> GpuCapabilityReport:
    """Detect the host GPU's acceleration-relevant capabilities.

    Reads from :mod:`torch.cuda` if available; otherwise returns the
    ``arch="none"`` report. Never raises — every failure path collapses to
    a sensible report so tenant ``setup()`` code can branch on
    ``caps.arch`` without try/except.

    The result is cached at module level. Pass ``refresh=True`` to force a
    re-probe (useful for tests that monkey-patch ``torch.cuda``).

    Args:
        refresh: when True, discard the cached report and re-probe.

    Returns:
        :class:`GpuCapabilityReport` describing the visible CUDA device 0.
        Multi-GPU hosts: the report reflects device 0; ``gpu_count`` is
        the total visible-device count.
    """
    global _CAPABILITY_CACHE
    if _CAPABILITY_CACHE is not None and not refresh:
        return _CAPABILITY_CACHE

    try:
        import torch
    except ImportError:
        _CAPABILITY_CACHE = _empty_report()
        return _CAPABILITY_CACHE

    torch_version = getattr(torch, "__version__", "") or ""

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — device probing is best-effort
        cuda_available = False

    if not cuda_available:
        _CAPABILITY_CACHE = GpuCapabilityReport(
            arch="none",
            compute_capability="",
            device_name="",
            vram_gb_total=0.0,
            gpu_count=0,
            has_fp8=False,
            has_nvfp4=False,
            torch_version=torch_version,
        )
        return _CAPABILITY_CACHE

    try:
        gpu_count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        gpu_count = 0

    try:
        major, minor = torch.cuda.get_device_capability(0)
        major, minor = int(major), int(minor)
        cc_str = f"{major}.{minor}"
        arch = _classify_arch(major, minor)
    except Exception:  # noqa: BLE001
        major, minor = 0, 0
        cc_str = ""
        arch = "unknown"

    try:
        device_name = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        device_name = ""

    try:
        props = torch.cuda.get_device_properties(0)
        total_bytes = int(getattr(props, "total_memory", 0))
        vram_gb_total = round(total_bytes / (1024**3), 1)
    except Exception:  # noqa: BLE001
        vram_gb_total = 0.0

    # FP8 tensor-core support starts on Hopper (SM 9.0+). NVFP4 tensor-
    # core support starts on Blackwell (SM 10.0+).
    has_fp8 = major >= 9
    has_nvfp4 = major >= 10

    _CAPABILITY_CACHE = GpuCapabilityReport(
        arch=arch,
        compute_capability=cc_str,
        device_name=device_name,
        vram_gb_total=vram_gb_total,
        gpu_count=gpu_count,
        has_fp8=has_fp8,
        has_nvfp4=has_nvfp4,
        torch_version=torch_version,
    )
    return _CAPABILITY_CACHE


# ---------------------------------------------------------------------------
# compile_diffusion: torch.compile wrapper
# ---------------------------------------------------------------------------


CompileMode = Literal["default", "reduce-overhead", "max-autotune"]

# torch.compile became production-ready for diffusion DiTs in 2.5. On older
# torch we silently no-op so tenant code is portable.
_MIN_TORCH_FOR_COMPILE: Final[tuple[int, int]] = (2, 5)


def _torch_version_tuple() -> tuple[int, int] | None:
    """Parse ``torch.__version__`` into (major, minor). Returns None if
    torch isn't importable.
    """
    try:
        import torch
    except ImportError:
        return None
    version = getattr(torch, "__version__", "") or ""
    # torch versions look like "2.5.1+cu121" or "2.6.0.dev20250101+cu126".
    head = version.split("+", 1)[0]
    parts = head.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def compile_diffusion(
    model: Any,
    *,
    mode: CompileMode = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
    **kwargs: Any,
) -> Any:
    """Compile a diffusion DiT module via :func:`torch.compile`.

    Thin, well-behaved wrapper. The intended call site is::

        pipe.transformer = accel.compile_diffusion(
            pipe.transformer, mode="reduce-overhead",
        )

    Args:
        model: the module to compile. Typically ``pipe.transformer`` or
            ``pipe.unet``.
        mode: torch.compile mode. ``"reduce-overhead"`` is the recommended
            default for inference — adds CUDA graph capture on top of
            Inductor codegen.
        fullgraph: require the trace to capture the entire graph (no
            Python fallbacks). Off by default; turn on only for modules
            known to trace cleanly.
        dynamic: allow dynamic shapes (more recompiles, supports variable
            input sizes). Off by default — endpoints should declare a
            fixed shape set instead and pre-warm.
        **kwargs: forwarded to :func:`torch.compile` unchanged (e.g.
            ``backend="inductor"``, ``options={...}``).

    Returns:
        The compiled module, or the original module unchanged when torch
        is missing, torch is older than 2.5, or no CUDA device is visible.
        Compilation is lazy — the first forward pass at a new shape pays
        the compile cost. Tenants should pre-warm in ``setup()``.

    The fallback path emits a single line to ``stderr`` so the warning is
    visible in worker logs even when logging isn't configured.
    """
    try:
        import torch
    except ImportError:
        print(
            "gen_worker.accel.compile_diffusion: torch not installed — "
            "returning model unchanged.",
            file=sys.stderr,
        )
        return model

    tv = _torch_version_tuple()
    if tv is not None and tv < _MIN_TORCH_FOR_COMPILE:
        print(
            "gen_worker.accel.compile_diffusion: torch "
            f"{'.'.join(str(x) for x in tv)} < 2.5 — returning model unchanged.",
            file=sys.stderr,
        )
        return model

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_available = False

    if not cuda_available:
        print(
            "gen_worker.accel.compile_diffusion: no CUDA device visible — "
            "returning model unchanged.",
            file=sys.stderr,
        )
        return model

    compiled = torch.compile(
        model,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        **kwargs,
    )
    logger.info(
        "compile_diffusion: torch.compile(mode=%s, fullgraph=%s, dynamic=%s)",
        mode,
        fullgraph,
        dynamic,
    )
    return compiled


# ---------------------------------------------------------------------------
# apply_fbcache: ParaAttention First-Block Cache
# ---------------------------------------------------------------------------


_PARA_ATTN_INSTALL_HINT: Final[str] = (
    "gen_worker.accel needs `para-attention` for this call. Install with "
    "`pip install para-attention` (Apache 2.0, "
    "https://github.com/chengzeyi/ParaAttention) and rebuild the endpoint "
    "image."
)


def apply_fbcache(
    pipe: Any,
    *,
    residual_diff_threshold: float = 0.12,
) -> Any:
    """Apply ParaAttention's First-Block Cache to a Diffusers pipeline.

    FBCache skips redundant DiT block evaluations across denoising steps
    when the first block's residual is unchanged within
    ``residual_diff_threshold``. ParaAttention's published benchmarks
    show ~1.5-2x on Flux and similar gains on SD3 / Qwen-Image.

    The wrapper patches the pipeline in-place and also returns it for
    chaining::

        accel.apply_fbcache(pipe, residual_diff_threshold=0.12)
        # equivalent to:
        pipe = accel.apply_fbcache(pipe, residual_diff_threshold=0.12)

    Args:
        pipe: the Diffusers pipeline (must expose a transformer/DiT module
            ParaAttention's adapter recognises).
        residual_diff_threshold: cache-invalidation threshold. 0.12 is the
            ParaAttention-recommended default. Lower values invalidate
            more aggressively (higher fidelity, smaller speedup); higher
            values invalidate less often (more speedup, more quality
            drift).

    Raises:
        ImportError: when ``para_attn`` is not installed. The message
            points at the canonical install command.
    """
    try:
        from para_attn.first_block_cache.diffusers_adapters import (
            apply_cache_on_pipe,
        )
    except ImportError as e:
        raise ImportError(_PARA_ATTN_INSTALL_HINT) from e

    apply_cache_on_pipe(pipe, residual_diff_threshold=residual_diff_threshold)
    logger.info(
        "apply_fbcache: patched %s (residual_diff_threshold=%s)",
        type(pipe).__name__,
        residual_diff_threshold,
    )
    return pipe


# ---------------------------------------------------------------------------
# apply_para_attn: ParaAttention general adapter
# ---------------------------------------------------------------------------


def apply_para_attn(pipe: Any) -> Any:
    """Apply ParaAttention's general (non-FBCache) adapter to a pipeline.

    ParaAttention's general adapter rewrites the pipeline's attention
    layers for **sequence-parallel inference** — splitting the sequence
    dimension across multiple GPUs to accelerate a single request. This
    is distinct from :func:`apply_fbcache`, which is a per-request
    cross-timestep cache and runs on a single GPU.

    Use this on multi-GPU workers when you want one request to fan out
    across all visible CUDA devices. On single-GPU workers the adapter
    still installs cleanly but the parallelism degree is 1, so the
    speedup is zero — it's safe to call unconditionally.

    The wrapper patches the pipeline in-place and also returns it for
    chaining.

    Args:
        pipe: the Diffusers pipeline.

    Raises:
        ImportError: when ``para_attn`` is not installed. The message
            points at the canonical install command.

    Notes:
        ParaAttention's general adapter entry point has shifted between
        releases. We try the modern path first
        (``para_attn.diffusers_adapters.apply_adapter_on_pipe``); if that
        attribute is absent we fall back to the older
        ``apply_para_attn_on_pipe`` shape. If neither is found we surface
        an ImportError with the install hint so the tenant can upgrade.
    """
    try:
        import para_attn  # noqa: F401
    except ImportError as e:
        raise ImportError(_PARA_ATTN_INSTALL_HINT) from e

    # Try modern adapter shape first.
    try:
        from para_attn.diffusers_adapters import apply_adapter_on_pipe  # type: ignore[attr-defined]
    except ImportError:
        apply_adapter_on_pipe = None  # type: ignore[assignment]

    if apply_adapter_on_pipe is not None:
        apply_adapter_on_pipe(pipe)
        logger.info(
            "apply_para_attn: patched %s via diffusers_adapters.apply_adapter_on_pipe",
            type(pipe).__name__,
        )
        return pipe

    # Older API surface (kept around so already-pinned endpoint images
    # don't break when they upgrade gen-worker).
    try:
        from para_attn import apply_para_attn_on_pipe  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            _PARA_ATTN_INSTALL_HINT
            + " (could not find either `para_attn.diffusers_adapters."
            "apply_adapter_on_pipe` or `para_attn.apply_para_attn_on_pipe` "
            "in the installed version)."
        ) from e

    apply_para_attn_on_pipe(pipe)
    logger.info(
        "apply_para_attn: patched %s via legacy apply_para_attn_on_pipe",
        type(pipe).__name__,
    )
    return pipe


# ---------------------------------------------------------------------------
# apply_nvfp4: NVIDIA Model-Optimizer NVFP4 weight quantization
# ---------------------------------------------------------------------------


_MODELOPT_INSTALL_HINT: Final[str] = (
    "gen_worker.accel.apply_nvfp4 requires `nvidia-modelopt` "
    "(NVIDIA TensorRT Model Optimizer, Apache 2.0). Install with "
    "`pip install nvidia-modelopt` and rebuild the endpoint image."
)


def apply_nvfp4(model: Any) -> Any:
    """Apply NVFP4 weight quantization to a transformer / DiT module.

    NVFP4 is NVIDIA's 4-bit floating-point quantization scheme. The
    tensor-core implementation requires Blackwell-class GPUs
    (SM 10.0+, e.g. B100 / B200 / B300 / RTX 50-series). On any other
    architecture this helper logs a warning and returns the model
    unchanged — it is **safe to call unconditionally** in ``setup()``.

    Args:
        model: the module to quantize. Typically ``pipe.transformer``.

    Returns:
        The (possibly) quantized module. On non-Blackwell hardware the
        original module is returned unchanged.

    Raises:
        ImportError: when ``modelopt`` is not installed AND the host is
            Blackwell-class. We do not raise on non-Blackwell hosts so
            tenants can leave the call in their ``setup()`` without
            branching.
    """
    caps = gpu_capability()
    if caps.arch != "blackwell":
        msg = (
            "apply_nvfp4: detected arch=%s (compute_capability=%s); "
            "NVFP4 requires Blackwell (SM 10.0+). Returning model unchanged."
        )
        logger.warning(msg, caps.arch, caps.compute_capability or "unknown")
        return model

    try:
        import modelopt.torch.quantization as mtq
    except ImportError as e:
        raise ImportError(_MODELOPT_INSTALL_HINT) from e

    # modelopt's PTQ entry point: `mtq.quantize(module, config, forward_loop=...)`.
    # NVFP4 config naming has shifted between modelopt releases; resolve
    # whichever attribute the installed version exposes.
    config = (
        getattr(mtq, "NVFP4_DEFAULT_CFG", None)
        or getattr(mtq, "NVFP4_KV_CFG", None)
        or getattr(mtq, "CONFIG_CHOICES", {}).get("NVFP4_DEFAULT_CFG")
    )
    if config is None:
        raise ImportError(
            "apply_nvfp4: installed `modelopt` does not expose an NVFP4 "
            "config (looked for NVFP4_DEFAULT_CFG / NVFP4_KV_CFG / "
            "CONFIG_CHOICES['NVFP4_DEFAULT_CFG']). Upgrade nvidia-modelopt."
        )

    # Weight-only quantization: no calibration data needed for the
    # tenant's setup() path. Pass a no-op forward_loop.
    mtq.quantize(model, config, forward_loop=lambda _m: None)
    logger.info(
        "apply_nvfp4: applied NVFP4 weight quantization (arch=%s, cc=%s)",
        caps.arch,
        caps.compute_capability,
    )
    return model


__all__ = [
    "Arch",
    "GpuCapabilityReport",
    "gpu_capability",
    "compile_diffusion",
    "apply_fbcache",
    "apply_para_attn",
    "apply_nvfp4",
]
