"""Native svdq-fp4 serving — SVDQuant checkpoints without nunchaku (pgw#685).

The ``"native"`` svdq ENGINE. `SvdqLinear` is `W4A4Linear` plus the three things
an SVDQuant checkpoint needs (pgw#682 A3/G-C, the "contract v2" gw#540 named):

1. a per-OUTPUT-CHANNEL second-level weight scale (``wcscales``) as well as the
   scalar form (``wtscale``) — free, the epilogue already multiplies;
2. the low-rank branch ``y += (x @ proj_down) @ proj_up.T``, which is what makes
   4-bit survive qwen-class outliers (plain nvfp4 PTQ measured lpips 0.63-0.69
   vs the official svdq artifact's 0.105, th#1055/th#1094);
3. ``smooth_factor``, which DIVIDES the activation feeding the 4-bit branch
   ONLY — the low-rank branch consumes RAW x, because deepcompressor
   pre-divides ``proj_down`` by the smooth vector at export. Getting this
   backwards silently corrupts every output.

And one thing it must NOT do: an svdq checkpoint has NO ``input_scale``.
nunchaku's activation quant is fully dynamic single-level per-16-block, so
SvdqLinear always runs the dynamic path — a real contract difference from
``#nvfp4-w4a4``, not an omission in the artifact.

Why this exists: nunchaku's fp4 kernels are ``sm_120a``-only, and its wheels
couple to one (torch minor, CUDA) build AND one diffusers transformer signature
window per release (gw#405, th#1211). The native engine is stock
``torch._scaled_mm`` + one triton quantizer, so it needs no nunchaku wheel, no
diffusers window, no pin-matrix row and no torch downgrade — and it adds
sm_100/103 coverage nunchaku will never have.

Degrade, never refuse: a host without fp4 tensor cores serves the SAME artifact
through :func:`fold_to_dense`, which collapses the 4-bit weight, the smoothing
vector and the low-rank branch into ONE plain bf16 Linear. That is exact in the
dequant limit (``W_eff = W_q / smooth + proj_up @ proj_down.T``), so the
artifact stays servable everywhere at bf16 cost.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional, Sequence

from .nvfp4_quant import E2M1_MAX, FP8_MAX, quantize_activation
from .svdq_layout import (
    DecodedLinear,
    SvdqBuffers,
    SvdqLayoutError,
    dequantize_decoded,
    split_decoded,
    to_buffers,
)

logger = logging.getLogger(__name__)

SVDQ_ENGINE_NATIVE = "native"
SVDQ_ENGINE_NUNCHAKU = "nunchaku"

# Blackwell fp4 tensor cores — the SAME silicon window as the #nvfp4-w4a4 lane
# (both are block-scaled nvfp4 through torch._scaled_mm / cuBLASLt). torch's own
# gate is only `major >= 9 || (8,9)`, which ADMITS sm_89/sm_90, but neither Ada
# nor Hopper has fp4 tensor cores; below Blackwell the honest degrade is fp8
# rowwise, which we already ship. Never emulate fp4.
SVDQ_NATIVE_FP4_SMS = (100, 103, 120, 121)

# nunchaku fuses these projections; diffusers keeps them separate. The split is
# exact in the logical domain (svdq_layout.split_decoded) and is verified
# against the target model's actual out_features, never assumed.
_FUSED_SPLITS: dict[str, tuple[str, ...]] = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_qkv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
}

_K_ALIGN = 32  # fp4 scaled_mm: in_features % 32
_N_ALIGN = 16  # fp4 scaled_mm: out_features % 16


class SvdqNativeError(RuntimeError):
    """Typed native-svdq failure."""


def svdq_native_sm_supported(gpu_sm: int) -> bool:
    return int(gpu_sm) in SVDQ_NATIVE_FP4_SMS


def svdq_native_reason() -> Optional[str]:
    """Why the native engine cannot serve fp4 HERE, or None when it can. No
    nunchaku, no diffusers window, no pin matrix — only silicon + torch."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "native svdq-fp4 requires a CUDA GPU"
    if getattr(torch, "float4_e2m1fn_x2", None) is None:
        return f"torch {torch.__version__} has no float4_e2m1fn_x2 dtype"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if not svdq_native_sm_supported(sm):
        return (f"native svdq-fp4 needs Blackwell fp4 tensor cores "
                f"(sm_{'/'.join(str(s) for s in SVDQ_NATIVE_FP4_SMS)}); "
                f"this GPU is sm_{sm}")
    return None


def svdq_native_available() -> bool:
    """Silicon + the real w4a4 arming path (kernel probe, numerics self-check,
    profitability gate, fused-quantizer bit-identity gate) — the native engine
    shares all of it with the ``#nvfp4-w4a4`` lane."""
    if svdq_native_reason() is not None:
        return False
    from .w4a4 import w4a4_gemm_mode

    return w4a4_gemm_mode() == "blockwise"


# ---------------------------------------------------------------------------
# The module.
# ---------------------------------------------------------------------------


def _build_svdq_linear_class() -> type:
    import torch
    import torch.nn as nn

    class _SvdqLinear(nn.Module):
        """SVDQuant W4A4 linear: 4-bit branch + low-rank bf16 branch.

        ``y = Q4(x / smooth) @ Wq.T * (s2_act * wscale2) + (x @ down) @ up.T
        + bias``

        The activation second-level scale is always DYNAMIC (svdq checkpoints
        carry no ``input_scale``). The low-rank branch is a separate skinny
        bf16 GEMM pair costing ~10-15% of the 4-bit win; fusing it the way
        nunchaku does (down into the quantize kernel, up into the GEMM
        epilogue) is named headroom, not a blocker. NOTE: never
        ``.to(dtype=...)`` this module (device moves are fine)."""

        # Structural marker, twin of _cozy_w4a4_linear.
        _cozy_svdq_linear = True

        def __init__(self, in_features: int, out_features: int, *,
                     rank: int, bias: bool, compute_dtype: Any,
                     per_channel_scale: bool, smooth: bool) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.rank = int(rank)
            if in_features % _K_ALIGN or out_features % _N_ALIGN:
                raise SvdqNativeError(
                    f"SvdqLinear dims [{out_features}, {in_features}] break "
                    f"fp4 scaled_mm alignment (in%{_K_ALIGN}, out%{_N_ALIGN})")
            meta = torch.device("meta")
            self.register_buffer("weight", torch.empty(
                out_features, in_features // 2, dtype=torch.uint8, device=meta))
            nrb = (out_features + 127) // 128
            ncb = (in_features // 16 + 3) // 4
            self.register_buffer("weight_scale", torch.empty(
                nrb * 128 * ncb * 4, dtype=torch.float8_e4m3fn, device=meta))
            self.register_buffer("weight_scale_2", torch.empty(
                1, out_features if per_channel_scale else 1,
                dtype=torch.float32, device=meta))
            if smooth:
                self.register_buffer("smooth_factor", torch.empty(
                    in_features, dtype=compute_dtype, device=meta))
            else:
                self.smooth_factor = None
            if self.rank:
                self.register_buffer("proj_down", torch.empty(
                    in_features, self.rank, dtype=compute_dtype, device=meta))
                self.register_buffer("proj_up", torch.empty(
                    out_features, self.rank, dtype=compute_dtype, device=meta))
            else:
                self.proj_down = None
                self.proj_up = None
            if bias:
                self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(
                    out_features, dtype=compute_dtype, device=meta))
            else:
                self.bias = None

        def forward(self, x: Any) -> Any:
            from .w4a4 import _gemm_w4a4

            shape = x.shape
            x2 = x.reshape(-1, self.in_features)
            # smooth_factor DIVIDES the 4-bit branch's activation only; the
            # low-rank branch below consumes RAW x2.
            xs = x2 if self.smooth_factor is None else x2 / self.smooth_factor
            # Dynamic per-tensor second level — an svdq checkpoint has none.
            s2 = (xs.abs().amax().float()
                  / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
            xq, sa_blocked = quantize_activation(xs, s2)
            y = _gemm_w4a4(xq, self.weight, sa_blocked, self.weight_scale,
                           x.dtype)
            y = y * (s2 * self.weight_scale_2).to(y.dtype)
            if self.rank:
                y = y + (x2 @ self.proj_down) @ self.proj_up.t()
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, rank={self.rank}, "
                    f"bias={self.bias is not None}, "
                    f"per_channel_scale={self.weight_scale_2.shape[-1] > 1}, "
                    f"smooth={self.smooth_factor is not None}")

    return _SvdqLinear


@functools.lru_cache(maxsize=1)
def svdq_linear_class() -> type:
    return _build_svdq_linear_class()


def build_svdq_linear(buf: SvdqBuffers, *, compute_dtype: Any = None,
                      device: Any = None) -> Any:
    """A device-resident :class:`SvdqLinear` from converted buffers."""
    import torch
    import torch.nn as nn

    compute = compute_dtype or torch.bfloat16
    cls = svdq_linear_class()
    mod = cls(buf.in_features, buf.out_features, rank=buf.rank,
              bias=buf.bias is not None, compute_dtype=compute,
              per_channel_scale=buf.second_kind == "per_channel",
              smooth=buf.smooth_factor is not None)
    dev = device or "cpu"
    mod.weight = buf.weight.contiguous().to(dev)
    mod.weight_scale = buf.weight_scale.to(dev)
    mod.weight_scale_2 = buf.weight_scale_2.float().to(dev)
    if buf.smooth_factor is not None:
        mod.smooth_factor = buf.smooth_factor.to(compute).reshape(-1).to(dev)
    if buf.rank:
        mod.proj_down = buf.proj_down.to(compute).to(dev)
        mod.proj_up = buf.proj_up.to(compute).to(dev)
    if buf.bias is not None:
        mod.bias = nn.Parameter(buf.bias.detach().to(compute).to(dev),
                                requires_grad=False)
    return mod


def fold_to_dense(dec: DecodedLinear, *, compute_dtype: Any = None) -> Any:
    """The any-hardware fallback: ONE plain bf16 ``nn.Linear`` equivalent to the
    whole svdq linear.

    Exact in the dequant limit. The 4-bit branch sees ``x / smooth``, so its
    dequantized weight absorbs the smoothing as ``W_q / smooth`` (broadcast over
    in-channels); the low-rank branch adds ``(x @ down) @ up.T``, i.e. a weight
    of ``up @ down.T``. Hence ``W_eff = W_q / smooth + up @ down.T``."""
    import torch
    import torch.nn as nn

    compute = compute_dtype or torch.bfloat16
    w = dequantize_decoded(dec)
    if dec.smooth_factor is not None:
        w = w / dec.smooth_factor.float().reshape(1, -1)
    if dec.rank:
        w = w + (dec.proj_up.float() @ dec.proj_down.float().t())
    lin = nn.Linear(dec.in_features, dec.out_features,
                    bias=dec.bias is not None, dtype=compute)
    with torch.no_grad():
        lin.weight.copy_(w.to(compute))
        if dec.bias is not None:
            lin.bias.copy_(dec.bias.to(compute))
    lin.weight.requires_grad_(False)
    if lin.bias is not None:
        lin.bias.requires_grad_(False)
    return lin


# ---------------------------------------------------------------------------
# Swapping a decoded checkpoint into a real diffusers denoiser.
# ---------------------------------------------------------------------------


def _module_at(model: Any, path: str) -> Any:
    try:
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        return getattr(parent, leaf)
    except AttributeError:
        return None


def _set_module(model: Any, path: str, new: Any) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, leaf, new)


def plan_targets(model: Any, prefix: str) -> tuple[tuple[str, int], ...]:
    """Where one nunchaku linear prefix lands in ``model``.

    ``((path, out_features), ...)`` — one entry for a 1:1 name, three for a
    fused ``to_qkv``-style prefix. The split is validated against the target
    modules' ACTUAL ``out_features``; an unknown or mismatched layout raises
    rather than guessing."""
    import torch.nn as nn

    direct = _module_at(model, prefix)
    if isinstance(direct, nn.Linear):
        return ((prefix, int(direct.out_features)),)

    parent_path, _, leaf = prefix.rpartition(".")
    parts = _FUSED_SPLITS.get(leaf)
    if parts is None:
        raise SvdqLayoutError(
            f"svdq checkpoint names {prefix!r}, which is neither a Linear in "
            f"{type(model).__name__} nor a known fused projection "
            f"({', '.join(sorted(_FUSED_SPLITS))})")
    out: list[tuple[str, int]] = []
    for part in parts:
        path = f"{parent_path}.{part}" if parent_path else part
        mod = _module_at(model, path)
        if not isinstance(mod, nn.Linear):
            raise SvdqLayoutError(
                f"fused svdq prefix {prefix!r} needs {path!r} to be a Linear "
                f"in {type(model).__name__}")
        out.append((path, int(mod.out_features)))
    return tuple(out)


def swap_svdq_linears(
    model: Any,
    decoded: Dict[str, DecodedLinear],
    *,
    compute_dtype: Any = None,
    mode: str = "",
    device: Any = None,
) -> dict:
    """Replace ``model``'s Linears with the checkpoint's svdq linears.

    ``mode`` ``"blockwise"`` (fp4 tensor cores) | ``"dense"`` (folded bf16
    fallback); default probes the host. Fused ``to_qkv``-style prefixes are
    split across the diffusers projections. Layers whose dims break fp4
    alignment fold to dense individually rather than refusing."""
    import torch

    compute = compute_dtype or torch.bfloat16
    if mode not in ("blockwise", "dense"):
        mode = "blockwise" if svdq_native_available() else "dense"
    counts = {"blockwise": 0, "dense": 0, "prefixes": 0, "linears": 0}
    for prefix in sorted(decoded):
        dec = decoded[prefix]
        targets = plan_targets(model, prefix)
        counts["prefixes"] += 1
        if len(targets) == 1:
            parts: Sequence[DecodedLinear] = (dec,)
        else:
            sections = tuple(out_f for _, out_f in targets)
            parts = split_decoded(dec, sections)
        for (path, out_f), part in zip(targets, parts):
            if int(part.out_features) != int(out_f):
                raise SvdqLayoutError(
                    f"svdq {prefix!r} -> {path!r}: decoded out_features "
                    f"{part.out_features} != module {out_f}")
            counts["linears"] += 1
            fp4_ok = (mode == "blockwise"
                      and part.in_features % _K_ALIGN == 0
                      and part.out_features % _N_ALIGN == 0)
            if fp4_ok:
                new = build_svdq_linear(to_buffers(part),
                                        compute_dtype=compute, device=device)
                counts["blockwise"] += 1
            else:
                new = fold_to_dense(part, compute_dtype=compute)
                if device is not None:
                    new = new.to(device)
                counts["dense"] += 1
            _set_module(model, path, new)
    logger.info(
        "svdq native swap: %d prefixes -> %d linears (%d fp4, %d folded bf16)",
        counts["prefixes"], counts["linears"], counts["blockwise"],
        counts["dense"])
    return counts


__all__ = [
    "SVDQ_ENGINE_NATIVE",
    "SVDQ_ENGINE_NUNCHAKU",
    "SVDQ_NATIVE_FP4_SMS",
    "SvdqNativeError",
    "build_svdq_linear",
    "fold_to_dense",
    "plan_targets",
    "svdq_linear_class",
    "svdq_native_available",
    "svdq_native_reason",
    "svdq_native_sm_supported",
    "swap_svdq_linears",
]
