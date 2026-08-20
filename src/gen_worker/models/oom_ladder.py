"""Reactive OOM ladders (pgw#1499) — IN-RUNG retries for the catchable cases.

The placement ladder (``rung.py``) answers *"where do the weights live"* and
descends a rung when a LOAD OOMs. It has nothing to say about a request that
loads fine and then exhausts the card inside one op — a full-frame VAE decode,
or an attention matmul at an unusually long sequence. Those are eager, they are
catchable, and the cheap answer is to retry the SAME op smaller, on the same
rung, rather than to spill the whole pipeline to host RAM.

Two wrappers, installed once per pipeline by ``memory.apply_low_vram_config``:

* ``vae.decode`` — full frame first; on OOM, tile and retry, shrinking the tile
  each further OOM. Tiling is applied REACTIVELY here, unlike
  ``_apply_vae_and_attention``'s per-rung proactive tiling: a pipeline that
  fits natively pays nothing until the day a request is too big for the card.
* the denoiser's ``forward`` (``unet``/``transformer``) — on OOM, empty the
  cache and retry ONE step with a finer attention slice, to a cap.

Both are degradations and both confess through ``memory._confess_serve_degrade``
— a tiled retry is a confession, not a silent save.

SCOPE, stated so it is not widened by accident. This is the EAGER half only.
A mid-graph OOM inside a compiled artifact is not catchable (pgw#1255 leg 2);
``serving/context.py`` decides that case by admission before the weights land
and ``aot_serve`` serves the request eager. Nothing here changes either.
"""

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

#: Phase tokens for the two confessions. Countable hub-side in
#: `worker_activity_events` beside `cpu_offload_engaged`.
VAE_TILED_RETRY_PHASE = "vae_tiled_retry"
ATTENTION_SLICED_RETRY_PHASE = "attention_sliced_retry"

#: Marker attribute — install is idempotent per object, and
#: `apply_low_vram_config` runs again on every rung of a placement descent.
_INSTALLED = "_cozy_oom_ladder"

#: Peak live decoder activations, counted in units of
#: "one full-resolution buffer at the decoder's WIDEST channel count".
#: Calibrated against ComfyUI's hand-measured per-VAE table: their sd15/SDXL
#: coefficient is 2178*64 bytes per latent element, and
#: 17 * 128ch * 8*8 spatial * 2 B/elem = 278528 vs their 278784 — 0.1%.
#: Wan-2.1 lands within 1.5x of their number on the same formula. It is a
#: FIRST GUESS only: every rung below it is chosen by a real OOM, not by this.
ACTIVATION_BUFFERS = 17

#: Fraction of free VRAM one tile may plan to occupy.
BUDGET_FRACTION = 0.8

#: diffusers' attention-slicing knob has exactly these two useful settings
#: ("auto" halves the slice dim, "max" slices to 1) — that IS the chunk-count
#: doubling for the sliced-attention implementation we run.
ATTENTION_LADDER: Tuple[str, ...] = ("auto", "max")


# ---------------------------------------------------------------------------
# The tile solver — pure arithmetic, no torch, unit-testable on a cardless box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilePlan:
    """One rung of the tiling ladder, in LATENT units.

    ``frames`` is 0 for an image VAE (no temporal axis to tile).
    """

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
    """Peak bytes of ONE tile's decode. A tiled decode's peak is per-tile, so a
    tile clamped below the real extent is what makes the estimate drop."""
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
    """The tile ladder for one decode, largest tile first.

    Rung 0 is the ComfyUI solve: start at ``base_edge``, halve the TEMPORAL
    tile until one tile fits the budget, then double the SPATIAL tile while it
    still fits. Temporal first because a video decoder's activations are linear
    in frames and quadratic in edge — shrinking time costs the least seam.

    One addition ComfyUI does not need. Their base tile is smaller than any
    latent they tile, so their rung 0 is always a real shrink; this ladder is
    solved only ONCE THE FULL FRAME HAS ALREADY OOMED, and it can be handed a
    latent no bigger than the base tile — or a budget that says the frame fits,
    which the allocator has just falsified. A rung that reproduces the shape
    that failed is not a rung, so the spatial axis halves while the budget says
    it does not fit, and rung 0 is shrunk once more if it still covers the
    whole latent.

    Every rung after that halves again (time to ``min_frames``, then space to
    ``min_edge``), because the budget is an ESTIMATE and the allocator is the
    only authority: rung N+1 is only ever reached by rung N actually OOMing.
    """
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
    """Estimated peak decode bytes per LATENT element for this VAE.

    ``ACTIVATION_BUFFERS`` buffers at FULL output resolution, which is where a
    decoder's activations dominate: ``block_out_channels[0]`` is the width
    there (the 512-channel blocks live at 1/8 the resolution and cost less
    despite the wider tensor). A VAE that answers none of the config gets the
    sd-family shape, which is the wrong answer for exactly the models whose
    next rung is one OOM away anyway.
    """
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


# ---------------------------------------------------------------------------
# Applying a plan to a diffusers VAE
# ---------------------------------------------------------------------------


def apply_tile_plan(vae: Any, plan: TilePlan) -> Dict[str, Any]:
    """Turn a ``TilePlan`` into whatever tiling knobs THIS VAE exposes.

    diffusers is not consistent here: ``AutoencoderKL.enable_tiling()`` takes
    no arguments and reads ``tile_sample_min_size``/``tile_overlap_factor``
    attributes, while every video VAE takes keyword tile sizes and each takes a
    different subset of them. So the knobs are discovered by signature, and the
    attribute route is the fallback rather than a second implementation.

    Returns what was actually set — the confession quotes it, so an operator
    reads the tile the decode RAN at, not the one we asked for.
    """
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
        # The attribute route (AutoencoderKL and friends).
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


# ---------------------------------------------------------------------------
# The wrappers
# ---------------------------------------------------------------------------


def _grad_warning() -> str:
    """Empty unless autograd is on — in which case say so, because it is the
    one condition under which tiling CANNOT help.

    Measured on a real card (sd1.5 VAE, 320² latent, RTX 4070): with grad
    enabled every tile's activations are retained for backward, so the tiled
    retry accumulates instead of bounding, and the ladder walks all four rungs
    down to an 8² tile and still fails. Under ``no_grad`` — which is what
    diffusers' pipelines decode under — the SAME decode succeeds on the first
    tiled rung at 2.7 GiB peak. Serving never wants the graph, so this is a
    caller defect; the confession names it rather than leaving an operator to
    infer it from four identical OOMs.
    """
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
    """``(h, w, frames, dtype_bytes)`` of the latent this decode was handed."""
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
    """Full-frame decode, tiled retry on OOM, smaller tile on every OOM after.

    THE ORDER MATTERS AND IT IS NOT COSMETIC. Inside the ``except`` block the
    traceback holds every frame of the failed decode, and those frames hold
    every live activation — so a retry started there runs against a card that
    is still full and OOMs again on a tile that would have fitted. The except
    block therefore does the minimum: record that it happened, drop the
    traceback, and let the retry run after the scope closes.
    """
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
                # Drop the frames (and with them the activations) NOW; keep the
                # text, never the exception object, so nothing outlives this.
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

    # Same rule as the denoiser's (pgw#1534). `decode` is not a `forward` and
    # adoption does not read it today, but a wrapper that erases the signature
    # of what it wraps is wrong for the same reason wherever it is installed,
    # and the two wrappers must not diverge on it.
    _carry_signature(decode, original)
    setattr(vae, "decode", decode)
    return True


def _carry_signature(wrapper: Any, wrapped: Any) -> None:
    """Make ``wrapper`` present the signature of what it wraps (pgw#1534).

    A ``*args, **kwargs`` wrapper is invisible to anything that INTROSPECTS the
    forward it replaced, and adoption is exactly that: torchcg claims a graph
    record for a marked module when the module's forward accepts every
    parameter the record's ingress names. An erased signature therefore does not
    degrade adoption — it silently switches it off for that module, everywhere,
    permanently, and no mint can turn it back on because nothing is missing.

    ``functools.wraps`` is not used: it copies ``__dict__`` and ``__wrapped__``
    too, and this wrapper's identity should stay its own. Only the one fact a
    caller introspects is carried, and a wrapped object that has no readable
    signature (a builtin, a C-implemented forward) leaves the wrapper as it is
    rather than failing a load — an unarmed ladder must never fail a load, and
    neither must an unsigned one.
    """
    try:
        wrapper.__signature__ = inspect.signature(wrapped)
    except Exception:  # noqa: BLE001 — an unsigned ladder must never fail a load
        # Deliberately broad, and it is this module's own rule rather than a
        # shrug: `inspect.signature` runs arbitrary code through `__signature__`
        # descriptors and `__get__`, so the exception set is not enumerable.
        # `install` catching it one frame up would abandon the WHOLE ladder over
        # a cosmetic step, which is a worse trade than an unsigned wrapper.
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
    """One denoise step, retried with a finer attention slice on OOM.

    The step is the right granularity: it is a pure function of its inputs, so
    a retry is free of side effects, and it is where a long-sequence attention
    matmul actually allocates. Retrying the whole request instead would pay for
    every step already taken.

    ``nn.Module.__call__`` resolves ``self.forward`` as an INSTANCE attribute,
    which is why this can be installed without touching the class — and it is
    the same seam ``aot_serve`` swaps, so an armed artifact simply captures
    this wrapper as its eager fallback.

    🔴 **AND IT MUST CARRY THE WRAPPED SIGNATURE (pgw#1534).** That seam is not
    only where a graph is armed; it is where adoption decides WHETHER to arm
    one. ``torchcg`` claims a graph record for a marked module when the
    module's forward accepts every parameter the record's ingress names, and it
    reads those names off ``inspect.signature(module.forward)``. A bare
    ``*args, **kwargs`` wrapper has no named parameters at all, so after this
    installed, a 13-parameter unet forward presented an EMPTY set and **every
    record in the lane went unclaimed** — measured on a real boot with all 14
    artifacts present in the store: ``adopted: 0, holes: 0, armed: 0``.

    pgw#1499 armed this ladder on EVERY rung, which is correct and is also what
    made a per-OOM-path wrapper into a permanent, universal disabling of the
    compiled path. ``models/lora_lifted.py`` already restores ``__signature__``
    on its own forward wrapper; this one did not, and the two wrappers sit four
    files apart. The rule is general and belongs to the seam, not to either
    wrapper: **anything installed onto a module's ``forward`` must present the
    signature it wrapped** — a fence test holds it.
    """
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
    """Arm both ladders on ``pipeline``. Idempotent, cheap, never raises.

    Nothing here costs anything until an OOM: both wrappers are one Python
    frame around a call that was going to happen anyway.
    """
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
