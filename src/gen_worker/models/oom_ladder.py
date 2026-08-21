from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .memory import (
    _confess_serve_degrade,
    flush_memory,
    get_available_vram_gb,
    is_cuda_oom,
)
from .rung import transition_line

_LOG = logging.getLogger(__name__)

VAE_TILED_RETRY_PHASE = "vae_tiled_retry"
ATTENTION_SLICED_RETRY_PHASE = "attention_sliced_retry"

_INSTALLED = "_cozy_oom_ladder"

ACTIVATION_BUFFERS = 17

BUDGET_FRACTION = 0.8

ATTENTION_LADDER: Tuple[str, ...] = ("auto", "max")


@dataclass(frozen=True)
class TilePlan:
    """One rung of the tiling ladder, in LATENT units."""

    edge: int
    frames: int = 0

    @property
    def overlap(self) -> int:
        return max(1, self.edge // 4)

    def __str__(self) -> str:
        t = f"x{self.frames}f" if self.frames else ""
        return f"{self.edge}²{t}"


def tile_bytes(plan: TilePlan, *, latent_h: int, latent_w: int,
               latent_frames: int, bytes_per_latent: float) -> float:
    """Peak bytes of ONE tile's decode."""
    h = min(plan.edge, latent_h)
    w = min(plan.edge, latent_w)
    t = min(plan.frames, latent_frames) if plan.frames else max(1, latent_frames)
    return float(h * w * t) * float(bytes_per_latent)


def solve_tile_ladder(
    *,
    latent_h: int,
    latent_w: int,
    latent_frames: int = 0,
    bytes_per_latent: float,
    budget_bytes: float,
    base_edge: int = 32,
    min_edge: int = 8,
    min_frames: int = 1,
    max_rungs: int = 5,
) -> Tuple[TilePlan, ...]:
    """The tile ladder for one decode, largest tile first."""
    latent_h = max(1, int(latent_h))
    latent_w = max(1, int(latent_w))
    latent_frames = max(0, int(latent_frames))
    span = max(latent_h, latent_w)

    def fits(edge: int, frames: int) -> bool:
        return tile_bytes(
            TilePlan(edge, frames), latent_h=latent_h, latent_w=latent_w,
            latent_frames=latent_frames, bytes_per_latent=bytes_per_latent,
        ) <= budget_bytes

    edge = max(min_edge, min(base_edge, span))
    frames = latent_frames
    if frames:
        while frames > min_frames and not fits(edge, frames):
            frames = -(-frames // 2)
    while edge > min_edge and not fits(edge, frames):
        edge = max(min_edge, edge // 2)
    while edge * 2 <= span and fits(edge * 2, frames):
        edge *= 2

    covers_all = edge >= span and (not frames or frames >= latent_frames)
    if covers_all:
        if frames > min_frames:
            frames = max(min_frames, frames // 2)
        else:
            edge = max(min_edge, min(edge, span) // 2)

    ladder = [TilePlan(edge, frames)]
    while len(ladder) < max_rungs:
        last = ladder[-1]
        if last.frames > min_frames:
            nxt = TilePlan(last.edge, max(min_frames, last.frames // 2))
        elif last.edge > min_edge:
            nxt = TilePlan(max(min_edge, last.edge // 2), last.frames)
        else:
            break
        ladder.append(nxt)
    return tuple(ladder)


def decode_bytes_per_latent(vae: Any, *, dtype_bytes: int = 2) -> float:
    """Estimated peak decode bytes per LATENT element for this VAE."""
    cfg = getattr(vae, "config", None)
    channels = 128
    blocks = getattr(cfg, "block_out_channels", None) if cfg is not None else None
    if blocks:
        channels = int(blocks[0])
    spatial = int(getattr(vae, "spatial_compression_ratio", 0) or 0)
    if spatial <= 0 and blocks:
        spatial = 2 ** (len(blocks) - 1)
    spatial = spatial or 8
    temporal = int(getattr(vae, "temporal_compression_ratio", 0) or 0) or 1
    return float(ACTIVATION_BUFFERS * channels * spatial * spatial * temporal
                 * max(1, int(dtype_bytes)))


def apply_tile_plan(vae: Any, plan: TilePlan) -> Dict[str, Any]:
    """Turn a ``TilePlan`` into whatever tiling knobs THIS VAE exposes."""
    enable = getattr(vae, "enable_tiling", None)
    if not callable(enable):
        return {}
    spatial = int(getattr(vae, "spatial_compression_ratio", 0) or 0) or 8
    temporal = int(getattr(vae, "temporal_compression_ratio", 0) or 0) or 1
    sample_edge = plan.edge * spatial
    sample_stride = max(spatial, (plan.edge - plan.overlap) * spatial)
    sample_frames = plan.frames * temporal if plan.frames else 0

    try:
        params = set(inspect.signature(enable).parameters)
    except (TypeError, ValueError):
        params = set()
    kwargs: Dict[str, Any] = {}
    for name, value in (
        ("tile_sample_min_height", sample_edge),
        ("tile_sample_min_width", sample_edge),
        ("tile_sample_stride_height", sample_stride),
        ("tile_sample_stride_width", sample_stride),
        ("tile_sample_min_num_frames", sample_frames),
        ("tile_sample_stride_num_frames", max(1, sample_frames // 2)),
    ):
        if name in params and value:
            kwargs[name] = value

    applied: Dict[str, Any] = {}
    if not kwargs:
        attrs: Tuple[Tuple[str, Any], ...] = (
            ("tile_sample_min_size", sample_edge),
            ("tile_latent_min_size", plan.edge),
            ("tile_overlap_factor", 0.25),
        )
        for attr, attr_value in attrs:
            if hasattr(vae, attr):
                setattr(vae, attr, attr_value)
                applied[attr] = attr_value
    enable(**kwargs)
    applied.update(kwargs)
    applied["plan"] = str(plan)
    return applied


def _grad_warning() -> str:
    try:
        import torch

        if torch.is_grad_enabled():
            return (" AUTOGRAD IS ENABLED for this decode, so every tile's "
                    "activations are retained for backward and tiling cannot "
                    "bound the peak — decode under `torch.no_grad()`.")
    except Exception:  # noqa: BLE001 — a diagnostic may never be the failure
        pass
    return ""


def _latent_geometry(args: Sequence[Any], kwargs: Dict[str, Any]) -> Tuple[int, int, int, int]:
    tensor = None
    for candidate in list(args) + list(kwargs.values()):
        if hasattr(candidate, "shape") and hasattr(candidate, "ndim"):
            tensor = candidate
            break
    if tensor is None or tensor.ndim < 4:
        return (0, 0, 0, 2)
    shape = tuple(int(d) for d in tensor.shape)
    frames = shape[2] if len(shape) >= 5 else 0
    dtype_bytes = int(getattr(getattr(tensor, "dtype", None), "itemsize", 2) or 2)
    return (shape[-2], shape[-1], frames, dtype_bytes)


def _wrap_vae_decode(vae: Any, log: logging.Logger) -> bool:
    original = getattr(vae, "decode", None)
    if not callable(original):
        return False
    state: Dict[str, Any] = {"rung": -1, "ladder": ()}

    def decode(*args: Any, **kwargs: Any) -> Any:
        while True:
            failed = ""
            try:
                return original(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — re-raised below unless OOM
                if not is_cuda_oom(exc):
                    raise
                failed = f"{type(exc).__name__}: {exc}"
                exc.__traceback__ = None

            flush_memory()
            ladder: Tuple[TilePlan, ...] = state["ladder"]
            if not ladder:
                h, w, frames, dtype_bytes = _latent_geometry(args, kwargs)
                if not h:
                    raise RuntimeError(
                        f"VAE decode ran out of memory and the latent shape is "
                        f"unreadable, so no tiling can be planned ({failed})")
                ladder = solve_tile_ladder(
                    latent_h=h, latent_w=w, latent_frames=frames,
                    bytes_per_latent=decode_bytes_per_latent(
                        vae, dtype_bytes=dtype_bytes),
                    budget_bytes=get_available_vram_gb() * BUDGET_FRACTION * 2**30,
                )
                state["ladder"] = ladder
            state["rung"] += 1
            if state["rung"] >= len(ladder):
                raise RuntimeError(
                    f"VAE decode ran out of memory at every tile the ladder has "
                    f"({', '.join(str(p) for p in ladder)}); last failure: "
                    f"{failed}.{_grad_warning()}")
            plan = ladder[state["rung"]]
            applied = apply_tile_plan(vae, plan)
            if not applied:
                raise RuntimeError(
                    f"VAE decode ran out of memory and {type(vae).__name__} "
                    f"exposes no tiling to retry with ({failed})")
            _confess_serve_degrade(
                phase=VAE_TILED_RETRY_PHASE,
                line=transition_line(
                    event="engaged", phase="decode",
                    from_rung="full_frame" if state["rung"] == 0 else "tiled",
                    to_rung=f"tiled:{plan}",
                    free_gb=get_available_vram_gb(),
                    detail=f"VAE decode OOM ({failed}); retrying TILED {applied}",
                ),
                detail=(
                    f"vae={type(vae).__name__}: decode exhausted device memory "
                    f"({failed}) and is retrying tiled at {plan} "
                    f"(rung {state['rung'] + 1}/{len(ladder)}). The request still "
                    f"serves; a tiled decode re-runs the decoder per tile and "
                    f"blends overlaps, so this pod is serving DEGRADED."
                    + _grad_warning()
                ),
                log=log,
            )

    _carry_signature(decode, original)
    setattr(vae, "decode", decode)
    return True


def _carry_signature(wrapper: Any, wrapped: Any) -> None:
    """Make `wrapper` present the signature of what it wraps: torchcg adoption reads inspect.signature(module.forward) to decide whether a module claims a graph record, so a bare *args/**kwargs wrapper silently and permanently disables the compiled path for that module. Anything installed onto a module's forward must present the wrapped signature (fence-tested). functools.wraps is deliberately not used (it copies __dict__/__wrapped__); an unsignaturable wrapped object (builtin/C forward) is left as-is — an unsigned ladder must never fail a load."""
    try:
        wrapper.__signature__ = inspect.signature(wrapped)
    except Exception:  # noqa: BLE001 — an unsigned ladder must never fail a load
        return
    for attribute in ("__name__", "__qualname__", "__doc__"):
        try:
            setattr(wrapper, attribute, getattr(wrapped, attribute))
        except AttributeError:
            pass


def _denoiser(pipeline: Any) -> Optional[Any]:
    for name in ("transformer", "unet"):
        module = getattr(pipeline, name, None)
        if module is not None and callable(getattr(module, "forward", None)):
            return module
    return None


def _wrap_denoiser_forward(pipeline: Any, module: Any, log: logging.Logger) -> bool:
    slicer = getattr(pipeline, "enable_attention_slicing", None)
    original = getattr(module, "forward", None)
    if not callable(slicer) or not callable(original):
        return False
    state = {"rung": -1}

    def forward(*args: Any, **kwargs: Any) -> Any:
        while True:
            failed = ""
            try:
                return original(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — re-raised below unless OOM
                if not is_cuda_oom(exc):
                    raise
                failed = f"{type(exc).__name__}: {exc}"
                exc.__traceback__ = None

            flush_memory()
            state["rung"] += 1
            if state["rung"] >= len(ATTENTION_LADDER):
                raise RuntimeError(
                    f"denoise step ran out of memory at every attention slice "
                    f"the ladder has ({', '.join(ATTENTION_LADDER)}); last "
                    f"failure: {failed}")
            slice_size = ATTENTION_LADDER[state["rung"]]
            slicer(slice_size)
            _confess_serve_degrade(
                phase=ATTENTION_SLICED_RETRY_PHASE,
                line=transition_line(
                    event="engaged", phase="denoise",
                    from_rung="fused" if state["rung"] == 0 else "sliced",
                    to_rung=f"sliced:{slice_size}",
                    free_gb=get_available_vram_gb(),
                    detail=f"denoise step OOM ({failed}); retrying with "
                           f"attention_slicing={slice_size}",
                ),
                detail=(
                    f"module={type(module).__name__}: a denoise step exhausted "
                    f"device memory ({failed}) and is retrying with "
                    f"attention_slicing={slice_size} (rung {state['rung'] + 1}/"
                    f"{len(ATTENTION_LADDER)}). The request still serves; sliced "
                    f"attention replaces the fused kernel with a chunked loop, "
                    f"so this pod is serving DEGRADED."
                ),
                log=log,
            )

    _carry_signature(forward, original)
    setattr(module, "forward", forward)
    return True


def install(pipeline: Any, *, logger: Optional[logging.Logger] = None) -> Dict[str, bool]:
    """Arm both ladders on ``pipeline``."""
    if pipeline is None or getattr(pipeline, _INSTALLED, None) is not None:
        return {}
    log = logger or _LOG
    armed: Dict[str, bool] = {}
    try:
        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            armed["vae_tiled_retry"] = _wrap_vae_decode(vae, log)
        module = _denoiser(pipeline)
        if module is not None:
            armed["attention_sliced_retry"] = _wrap_denoiser_forward(
                pipeline, module, log)
        setattr(pipeline, _INSTALLED, armed)
    except Exception:  # noqa: BLE001 — an unarmed ladder must never fail a load
        log.debug("oom_ladder: install failed on %s",
                  type(pipeline).__name__, exc_info=True)
    return armed


__all__ = [
    "ATTENTION_LADDER",
    "ATTENTION_SLICED_RETRY_PHASE",
    "TilePlan",
    "VAE_TILED_RETRY_PHASE",
    "apply_tile_plan",
    "decode_bytes_per_latent",
    "install",
    "solve_tile_ladder",
    "tile_bytes",
]
