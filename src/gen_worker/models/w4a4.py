"""W4A4 nvfp4 loader mode (gw#540) — the gw#534 W8A8 pattern one tier down.

A ``#nvfp4-w4a4`` flavor is a normal diffusers tree whose denoiser holds
calibrated nvfp4 weights WITH two-level scales — the artifact contract
(frozen in gw#540, consumed verbatim by the conversion side):

- per quantized Linear ``L`` (modelopt export_hf_checkpoint shapes):
  ``L.weight`` (uint8 [out, in/2]: packed e2m1 nibble pairs, element 2j in
  the LOW nibble — torch.float4_e2m1fn_x2 convention), ``L.weight_scale``
  (float8_e4m3fn [out, in/16]: per-16-block scales, FLAT row-major),
  ``L.weight_scale_2`` (float32 scalar per-tensor second-level scale),
  optional ``L.input_scale`` (float32 scalar static activation second-level
  scale — nvfp4 calibrates unconditionally, te#80), optional
  ``L.pre_quant_scale`` (float [in], AWQ-lite smoothing: x *= it before
  activation quant), ``L.bias`` unquantized;
- excluded layers carry NO scale tensors — detection is per-layer by the
  (uint8 weight + e4m3 weight_scale + weight_scale_2) triple, never by name
  lists. The triple also disambiguates from w8a8 (whose weight IS e4m3).

Dequant: W ≈ e2m1(weight) * weight_scale.float() * weight_scale_2.

Execution: quantized Linears become :class:`W4A4Linear` — blockwise
``torch._scaled_mm`` over RESIDENT packed fp4 weights (Blackwell fp4 tensor
cores, ~2x the fp8 rate: 4378 TFLOPS / 3.07x bf16 measured on B200, gw#540
DP1). Activations are quantized per call: static per-tensor second-level
scale from calibration (dynamic amax fallback), per-16-block e4m3 scales
computed dynamically. torch's blockwise scaled_mm consumes scales in the
cuBLAS 2-D tiled ("blocked") layout — weight scales are swizzled ONCE at
load, activation scales per call (a reshape/permute that fuses under
inductor). SM >= 100 only (sm_100/103 datacenter + sm_120/121 consumer
Blackwell); the hub never schedules the flavor below that. Hosts of a
qualifying arch whose kernel probe, micro-benchmark, or numerics self-check
fails DEQUANT once at load into plain bf16-resident weights — same
numerics, no speed win, never a refusal.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

W4A4_FLAVOR = "nvfp4-w4a4"
# Blackwell fp4 tensor cores: sm_100/103 (B200/B300) + sm_120/121 (RTX 50xx).
W4A4_MIN_SM = 100
_E2M1_MAX = 6.0
_FP8_MAX = 448.0
_SCALE_MIN = 2.0 ** -9  # e4m3 underflow guard (modelopt clamp)
_BLOCK = 16
# torch fp4 scaled_mm operand checks: packed K/2 % 16 == 0 and both mat2
# dims % 16 == 0 => in_features % 32 == 0, out_features % 16 == 0.
_K_ALIGN = 32
_N_ALIGN = 16
_MAX_HEADER_BYTES = 100 << 20
_COMPONENT_DIRS = ("transformer", "unet")
_AUX_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale",
                 ".pre_quant_scale")


class W4a4Error(RuntimeError):
    """Typed w4a4 loader-mode failure."""


class W4a4SnapshotError(W4a4Error):
    """The flavor snapshot violates the artifact contract."""


@dataclass(frozen=True)
class W4a4Artifact:
    # Denoiser dir name ("transformer"/"unet") for diffusers trees; "" for a
    # root-layout snapshot whose weight set IS the denoiser (gw#562).
    component: str
    files: tuple[Path, ...]
    quantized: tuple[str, ...]  # module names with the contract triple
    static_input_scales: bool


def _read_header(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or n > _MAX_HEADER_BYTES:
                return {}
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return {}
    return header if isinstance(header, dict) else {}


def _quantized_layers(files: tuple[Path, ...]) -> tuple[tuple[str, ...], bool]:
    dtypes: Dict[str, str] = {}
    for f in files:
        for name, info in _read_header(f).items():
            if isinstance(info, dict) and "dtype" in info:
                dtypes[name] = str(info["dtype"])
    quantized = tuple(sorted(
        key[: -len(".weight_scale_2")]
        for key in dtypes
        if key.endswith(".weight_scale_2")
        and dtypes.get(key[: -len(".weight_scale_2")] + ".weight") == "U8"
        and dtypes.get(
            key[: -len(".weight_scale_2")] + ".weight_scale") == "F8_E4M3"
    ))
    static = any(f"{n}.input_scale" in dtypes for n in quantized)
    return quantized, static


def _root_weight_files(d: Path) -> tuple[Path, ...]:
    sharded: set[str] = set()
    for idx in sorted(d.glob("*.safetensors.index.json")):
        try:
            weight_map = json.loads(idx.read_text("utf-8")).get("weight_map") or {}
            sharded.update(str(v) for v in weight_map.values())
        except (OSError, ValueError):
            continue
    files = [d / s for s in sorted(sharded) if (d / s).is_file()]
    files += [p for p in sorted(d.glob("*.safetensors"))
              if p.is_file() and p.name not in sharded]
    return tuple(dict.fromkeys(files))


def detect_w4a4_artifact(model_path: Path) -> Optional[W4a4Artifact]:
    """Header-sniff a snapshot for the w4a4 contract triple. A w8a8 tree
    (e4m3 weights) or a scale-free tree never matches. Cheap: header reads
    only."""
    root = Path(model_path)
    if not root.is_dir():
        return None
    if (root / "model_index.json").exists():
        for comp in _COMPONENT_DIRS:
            comp_dir = root / comp
            if not comp_dir.is_dir():
                continue
            files = tuple(sorted(
                p for p in comp_dir.glob("*.safetensors") if p.is_file()))
            if not files:
                continue
            quantized, static = _quantized_layers(files)
            if quantized:
                return W4a4Artifact(
                    component=comp, files=files, quantized=quantized,
                    static_input_scales=static,
                )
        return None
    files = _root_weight_files(root)
    if files:
        quantized, static = _quantized_layers(files)
        if quantized:
            return W4a4Artifact(
                component="", files=files, quantized=quantized,
                static_input_scales=static,
            )
    return None


# ---------------------------------------------------------------------------
# e2m1 quantize/dequantize + the cuBLAS blocked-scale swizzle (pure torch —
# used by the producer, the dequant lane, and W4A4Linear's activation quant).
# ---------------------------------------------------------------------------


def _e2m1_lut(device: Any) -> Any:
    import torch

    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device, dtype=torch.float32)


def _e2m1_bounds(device: Any) -> Any:
    import torch

    return torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=device, dtype=torch.float32)


def cast_e2m1(t: Any) -> Any:
    """Float tensor -> e2m1 nibble codes (uint8, values 0-15). Round-to-
    nearest with ties at the odd bounds (0.75/1.75/2.5) rounding UP —
    byte-identical to modelopt's NVFP4QTensor._cast_fp4."""
    import torch

    v = t.float()
    sign = (v < 0).to(torch.uint8)
    a = v.abs()
    ord_ = torch.searchsorted(
        _e2m1_bounds(v.device), a.contiguous(), out_int32=True).to(torch.uint8)
    tie = ((a == 0.75) | (a == 1.75) | (a == 2.5)).to(torch.uint8)
    return (sign << 3) + ord_ + tie


def pack_e2m1(codes: Any) -> Any:
    """[..., K] nibble codes -> [..., K/2] packed uint8 (element 2j in the
    LOW nibble — torch.float4_e2m1fn_x2 / modelopt convention). Explicitly
    contiguous: cuBLASLt refuses strided fp4 operands
    (CUBLAS_STATUS_NOT_SUPPORTED, found live on B200), and TensorIterator
    may propagate the slice strides into the packed output."""
    return ((codes[..., 1::2] << 4) | codes[..., 0::2]).contiguous()


def unpack_e2m1(packed: Any) -> Any:
    """[..., K/2] packed uint8 -> [..., K] float32 e2m1 values."""
    import torch

    lut = _e2m1_lut(packed.device)
    shape = list(packed.shape)
    shape[-1] = shape[-1] * 2
    out = torch.empty(shape, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = lut[(packed & 0x0F).long()]
    out[..., 1::2] = lut[(packed >> 4).long()]
    return out


def to_blocked_scales(scales: Any) -> Any:
    """Flat per-block scales [rows, k_blocks] (e4m3) -> the cuBLAS 2-D tiled
    layout torch's blockwise scaled_mm consumes: rows padded to 128, block
    columns padded to 4, 128x4 tiles rearranged (32, 4, 4) — returned as a
    contiguous 1-D e4m3 tensor (the kernel checks numel + contiguity only).
    Same rearrangement as modelopt's swizzle_nvfp4_scales / torchao's
    to_blocked."""
    import torch

    rows, cols = scales.shape
    nrb = (rows + 127) // 128
    ncb = (cols + 3) // 4
    # fp8 has no pad/zeros kernels everywhere — swizzle a uint8 view.
    raw = scales.view(torch.uint8)
    padded = raw.new_zeros(nrb * 128, ncb * 4)
    padded[:rows, :cols] = raw
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    out = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
    return out.contiguous().view(torch.float8_e4m3fn)


def quantize_nvfp4_tensor(w: Any) -> tuple[Any, Any, Any]:
    """Quantize a 2-D float tensor to the contract triple:
    (packed uint8 [out, in/2], weight_scale e4m3 [out, in/16],
    weight_scale_2 fp32 scalar). modelopt-identical math."""
    import torch

    wf = w.float()
    out_f, in_f = wf.shape
    ws2 = (wf.abs().amax() / (_E2M1_MAX * _FP8_MAX)).clamp(min=1e-12)
    blocks = wf.reshape(out_f, in_f // _BLOCK, _BLOCK)
    bmax = blocks.abs().amax(dim=-1)
    bs = bmax / (_E2M1_MAX * ws2)
    bs[bs == 0] = 1.0
    bs_fp8 = bs.clamp(min=_SCALE_MIN, max=_FP8_MAX).to(torch.float8_e4m3fn)
    q = blocks / (bs_fp8.float().unsqueeze(-1) * ws2)
    codes = cast_e2m1(q.reshape(out_f, in_f))
    return pack_e2m1(codes), bs_fp8, ws2


def dequantize_nvfp4_tensor(packed: Any, weight_scale: Any,
                            weight_scale_2: Any) -> Any:
    """Contract triple -> float32 [out, in]. The dequant-lane inverse of
    :func:`quantize_nvfp4_tensor` (and of the modelopt export)."""
    vals = unpack_e2m1(packed)
    out_f, in_f = vals.shape
    scales = weight_scale.float().reshape(out_f, in_f // _BLOCK, 1)
    return (vals.reshape(out_f, in_f // _BLOCK, _BLOCK)
            * scales * weight_scale_2.float()).reshape(out_f, in_f)


# ---------------------------------------------------------------------------
# Device qualification: kernel probe + numerics self-check + micro-benchmark.
# ---------------------------------------------------------------------------


def _fp4_dtype() -> Any:
    import torch

    return getattr(torch, "float4_e2m1fn_x2", None)


def _gemm_w4a4(xq: Any, wq: Any, sa_blocked: Any, sb_blocked: Any,
               out_dtype: Any) -> Any:
    import torch

    fp4 = _fp4_dtype()
    return torch._scaled_mm(
        xq.view(fp4), wq.view(fp4).t(), scale_a=sa_blocked,
        scale_b=sb_blocked, out_dtype=out_dtype)


def _numerics_ok() -> bool:
    """Quantize a real probe pair, run the blocked-layout scaled_mm, and
    compare against the dequant fp32 reference. torch 2.13's operator
    validates only scale NUMEL — a wrong layout returns silent garbage, so
    the lane arms only when the kernel's numerics match our swizzle."""
    import torch

    m, k, n = 128, 256, 128
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda")
    w = torch.randn(n, k, device="cuda")
    xq, xs, xs2 = quantize_nvfp4_tensor(x)
    wq, ws, ws2 = quantize_nvfp4_tensor(w)
    try:
        y = _gemm_w4a4(xq, wq, to_blocked_scales(xs), to_blocked_scales(ws),
                       torch.bfloat16)
    except Exception as exc:  # noqa: BLE001 — any kernel gap => dequant lane
        logger.warning("w4a4: blockwise scaled_mm probe failed (%s)", exc)
        return False
    y = y.float() * (xs2 * ws2)
    ref = dequantize_nvfp4_tensor(xq, xs, xs2) @ dequantize_nvfp4_tensor(
        wq, ws, ws2).t()
    rel = ((y - ref).norm() / ref.norm().clamp(min=1e-9)).item()
    ok = rel < 2e-2  # identical quantized operands; bf16 accumulation only
    if not ok:
        logger.warning(
            "w4a4: scaled_mm numerics self-check failed (rel err %.4f) — "
            "scale-layout mismatch on this stack; dequant lane", rel)
    return ok


_BENCH_DIM = 4096
_BENCH_WARMUP = 3
_BENCH_ITERS = 10
# fp4 must beat the bf16 GEMM by a real margin to arm (B200 measured 3.07x;
# a fallback/emulated path measures well under 1).
_BENCH_MIN_SPEEDUP = 1.10


def _median_ms(fn: Any) -> float:
    import torch

    for _ in range(_BENCH_WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(_BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _gemm_profitable() -> bool:
    import torch

    m = n = k = _BENCH_DIM
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    xq = torch.randint(0, 255, (m, k // 2), device="cuda", dtype=torch.uint8)
    wq = torch.randint(0, 255, (n, k // 2), device="cuda", dtype=torch.uint8)
    ones = torch.ones(m, k // _BLOCK, device="cuda")
    sa = to_blocked_scales(ones.to(torch.float8_e4m3fn))
    sb = to_blocked_scales(torch.ones(
        n, k // _BLOCK, device="cuda").to(torch.float8_e4m3fn))

    def fp4_op() -> Any:
        return _gemm_w4a4(xq, wq, sa, sb, torch.bfloat16) * 1.0

    def bf16_op() -> Any:
        return x @ w.t()

    fp4_ms, bf16_ms = _median_ms(fp4_op), _median_ms(bf16_op)
    speedup = bf16_ms / max(fp4_ms, 1e-9)
    logger.info(
        "w4a4 gemm gate: fp4=%.3fms bf16=%.3fms speedup=%.2fx (min %.2fx)",
        fp4_ms, bf16_ms, speedup, _BENCH_MIN_SPEEDUP)
    return speedup >= _BENCH_MIN_SPEEDUP


@functools.lru_cache(maxsize=1)
def w4a4_gemm_mode() -> str:
    """The fp4 GEMM dispatch for THIS device, chosen once per process:
    ``"blockwise"`` or ``""`` (dequant lane). Arms only when the arch
    qualifies (sm >= 100), the fp4 dtype + kernel exist, the numerics
    self-check passes (scale-layout truth), and the micro-benchmark beats
    bf16 — probe-pass != profitable (the gw#564 lesson)."""
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available() or _fp4_dtype() is None:
        return ""
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < W4A4_MIN_SM:
        return ""
    try:
        if not _numerics_ok():
            return ""
        if not _gemm_profitable():
            logger.warning(
                "w4a4: fp4 GEMM shows no real win on this device; dequant lane")
            return ""
    except Exception as exc:  # noqa: BLE001 — bench failure => dequant lane
        logger.warning("w4a4: qualification failed (%s); dequant lane", exc)
        return ""
    return "blockwise"


# ---------------------------------------------------------------------------
# The module.
# ---------------------------------------------------------------------------


def _build_module_class() -> type:
    import torch
    import torch.nn as nn

    class _W4A4Linear(nn.Module):
        """Packed nvfp4 weights RESIDENT; y = blockwise scaled_mm + epilogue.

        Per call: optional AWQ-lite smoothing (``x *= pre_quant_scale``),
        per-tensor activation second-level scale (static ``input_scale``
        from calibration, else dynamic amax/(6*448)), dynamic per-16-block
        e4m3 scales, e2m1 cast + nibble pack, blockwise ``torch._scaled_mm``
        (weight scales pre-swizzled at load, activation scales swizzled per
        call), then the fp32 second-level epilogue ``y *= s2_act * s2_w``
        and bias. The quantize/pack ops fuse under inductor — the win lives
        in the compiled lane (gw#534's lesson holds one tier down). NOTE:
        never ``.to(dtype=...)`` this module (device moves are fine)."""

        weight: Any        # uint8 [out, in/2] packed e2m1 pairs
        weight_scale: Any  # e4m3 1-D, cuBLAS blocked layout (swizzled at load)
        weight_scale_2: Any  # fp32 [1, 1]
        # Structural marker consumed by compile_cache.execution_contract.
        _cozy_w4a4_linear = True

        def __init__(self, in_features: int, out_features: int, *,
                     bias: bool, compute_dtype: Any,
                     static_input_scale: bool,
                     pre_quant_scale: bool = False) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            if in_features % _K_ALIGN or out_features % _N_ALIGN:
                raise W4a4SnapshotError(
                    f"W4A4Linear dims [{out_features}, {in_features}] break "
                    f"fp4 scaled_mm alignment (in%{_K_ALIGN}, out%{_N_ALIGN})")
            meta = torch.device("meta")
            self.register_buffer("weight", torch.empty(
                out_features, in_features // 2, dtype=torch.uint8, device=meta))
            nrb = (out_features + 127) // 128
            ncb = (in_features // _BLOCK + 3) // 4
            self.register_buffer("weight_scale", torch.empty(
                nrb * 128 * ncb * 4, dtype=torch.float8_e4m3fn, device=meta))
            self.register_buffer("weight_scale_2", torch.empty(
                1, 1, dtype=torch.float32, device=meta))
            if static_input_scale:
                self.register_buffer("input_scale", torch.empty(
                    1, 1, dtype=torch.float32, device=meta))
            else:
                self.input_scale = None
            if pre_quant_scale:
                self.register_buffer("pre_quant_scale", torch.empty(
                    in_features, dtype=compute_dtype, device=meta))
            else:
                self.pre_quant_scale = None
            if bias:
                self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(
                    out_features, dtype=compute_dtype, device=meta))
            else:
                self.bias = None

        def forward(self, x: Any) -> Any:
            shape = x.shape
            x2 = x.reshape(-1, self.in_features)
            if self.pre_quant_scale is not None:
                x2 = x2 * self.pre_quant_scale
            if self.input_scale is not None:
                s2 = self.input_scale.reshape(())
            else:
                s2 = (x2.abs().amax().float()
                      / (_E2M1_MAX * _FP8_MAX)).clamp(min=1e-12)
            xb = x2.reshape(-1, self.in_features // _BLOCK, _BLOCK).float()
            bmax = xb.abs().amax(dim=-1)
            sa = (bmax / (_E2M1_MAX * s2)).clamp(
                min=_SCALE_MIN, max=_FP8_MAX).to(torch.float8_e4m3fn)
            q = xb / (sa.float().unsqueeze(-1) * s2)
            codes = cast_e2m1(q.reshape(-1, self.in_features))
            xq = pack_e2m1(codes)
            y = _gemm_w4a4(
                xq, self.weight, to_blocked_scales(sa), self.weight_scale,
                x.dtype)
            y = y * (s2 * self.weight_scale_2.reshape(())).to(y.dtype)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, "
                    f"bias={self.bias is not None}, "
                    f"static_input_scale={self.input_scale is not None}, "
                    f"pre_quant_scale={self.pre_quant_scale is not None}")

    return _W4A4Linear


@functools.lru_cache(maxsize=1)
def w4a4_linear_class() -> type:
    return _build_module_class()


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------


def _denoiser_class(root: Path, component: str) -> Any:
    index = json.loads((root / "model_index.json").read_text("utf-8"))
    entry = index.get(component)
    if not (isinstance(entry, list) and len(entry) == 2):
        raise W4a4SnapshotError(
            f"model_index.json has no [library, class] entry for {component!r}")
    lib, name = str(entry[0]), str(entry[1])
    try:
        mod = importlib.import_module(lib)
    except ImportError:
        mod = importlib.import_module("diffusers")
    cls = getattr(mod, name, None)
    if cls is None:
        raise W4a4SnapshotError(f"{lib} has no model class {name!r}")
    return cls


def _dequant_into(sd: Dict[str, Any], name: str, compute: Any) -> None:
    """Replace one quantized layer's tensors with a dequantized weight.
    AWQ-lite smoothing folds BACK into the weight: serve-time applies
    ``x *= pre_quant_scale``, so the stored weight carries its inverse —
    an unfolded dequant would silently mis-scale every in-channel."""
    w = dequantize_nvfp4_tensor(
        sd[f"{name}.weight"], sd[f"{name}.weight_scale"],
        sd[f"{name}.weight_scale_2"])
    pqs = sd.get(f"{name}.pre_quant_scale")
    if pqs is not None:
        w = w * pqs.float().reshape(1, -1)
    sd[f"{name}.weight"] = w.to(compute)
    for suffix in _AUX_SUFFIXES:
        sd.pop(f"{name}{suffix}", None)


def load_w4a4_denoiser(root: Path, art: W4a4Artifact, *,
                       compute_dtype: Any = None, mode: str = "") -> Any:
    """Materialize the quantized denoiser: skeleton on meta, quantized
    Linears swapped for W4A4Linear, tensors assigned from the shards.
    ``mode`` "blockwise" | "dequant" (default: probe). Layers whose dims
    break fp4 scaled_mm alignment are dequantized individually (AWQ-lite
    ``pre_quant_scale`` folds back into the dequantized weight)."""
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    compute = compute_dtype or torch.bfloat16
    if mode not in ("blockwise", "dequant"):
        mode = w4a4_gemm_mode() or "dequant"

    cls = _denoiser_class(root, art.component)
    cfg = dict(cls.load_config(str(root / art.component)))
    cfg.pop("quantization_config", None)
    with init_empty_weights():
        model = cls.from_config(cfg)

    sd: Dict[str, Any] = {}
    for f in art.files:
        sd.update(load_file(str(f)))

    lin_cls = w4a4_linear_class()
    swapped = 0
    for name in art.quantized:
        w = sd[f"{name}.weight"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1]) * 2
        try:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            old = getattr(parent, leaf)
        except AttributeError as exc:
            raise W4a4SnapshotError(
                f"quantized tensor {name!r} has no module in "
                f"{type(model).__name__}") from exc
        has_pqs = f"{name}.pre_quant_scale" in sd
        eligible = (mode != "dequant" and isinstance(old, nn.Linear)
                    and in_f % _K_ALIGN == 0 and out_f % _N_ALIGN == 0)
        if eligible:
            has_static = f"{name}.input_scale" in sd
            new = lin_cls(in_f, out_f, bias=old.bias is not None,
                          compute_dtype=compute, static_input_scale=has_static,
                          pre_quant_scale=has_pqs)
            setattr(parent, leaf, new)
            sd[f"{name}.weight_scale"] = to_blocked_scales(
                sd[f"{name}.weight_scale"])
            sd[f"{name}.weight_scale_2"] = (
                sd[f"{name}.weight_scale_2"].float().reshape(1, 1))
            if has_static:
                sd[f"{name}.input_scale"] = (
                    sd[f"{name}.input_scale"].float().reshape(1, 1))
            if has_pqs:
                sd[f"{name}.pre_quant_scale"] = (
                    sd[f"{name}.pre_quant_scale"].to(compute).reshape(-1))
            swapped += 1
        else:
            _dequant_into(sd, name, compute)

    for key, value in list(sd.items()):
        if (value.is_floating_point()
                and value.dtype != torch.float8_e4m3fn
                and not key.endswith(_AUX_SUFFIXES)):
            sd[key] = value.to(compute)

    result = model.load_state_dict(sd, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise W4a4SnapshotError(
            f"w4a4 state dict mismatch: missing={list(result.missing_keys)[:5]} "
            f"unexpected={list(result.unexpected_keys)[:5]}")
    model.eval()
    model._cozy_w4a4_mode = mode
    logger.info(
        "w4a4 loader mode: %s — %d/%d quantized Linears on fp4 scaled_mm "
        "(component %s, static input scales: %s)",
        mode, swapped, len(art.quantized), art.component,
        art.static_input_scales,
    )
    return model


def load_w4a4_pipeline(cls: Any, path: Path, art: W4a4Artifact, *,
                       compute_dtype: Any = None,
                       components: Optional[Dict[str, Any]] = None) -> Any:
    """Build the pipeline with the w4a4 denoiser wired in (svdq-style
    component injection). Stamps ``_cozy_weight_lane`` ("w4a4" on the fp4
    scaled_mm lane, "bf16-resident" on the dequant lane) — the compile-cache
    graph key (lane_drift, gw#534/gw#540)."""
    import torch

    compute = compute_dtype or torch.bfloat16
    mode = w4a4_gemm_mode() or "dequant"
    denoiser = load_w4a4_denoiser(path, art, compute_dtype=compute, mode=mode)
    kwargs: Dict[str, Any] = dict(components or {})
    kwargs[art.component] = denoiser
    pipe = cls.from_pretrained(str(path), torch_dtype=compute, **kwargs)
    try:
        pipe._cozy_weight_lane = (
            "w4a4" if mode != "dequant" else "bf16-resident")
    except Exception:
        pass
    return pipe


def sanitize_w4a4_state_dict(
    state_dict: Dict[str, Any], compute_dtype: Any = None,
) -> Dict[str, Any]:
    """Dequantize w4a4 tensors in a raw state dict: contract triples become
    compute-dtype weights, aux tensors drop. A non-matching dict passes
    through unchanged, so manual snapshot loaders can call it
    unconditionally."""
    import torch

    compute = compute_dtype or torch.bfloat16
    out: Dict[str, Any] = {}
    quantized = {
        key[: -len(".weight_scale_2")]
        for key in state_dict
        if key.endswith(".weight_scale_2")
        and isinstance(state_dict.get(
            key[: -len(".weight_scale_2")] + ".weight"), torch.Tensor)
        and state_dict[
            key[: -len(".weight_scale_2")] + ".weight"].dtype == torch.uint8
        and f"{key[: -len('.weight_scale_2')]}.weight_scale" in state_dict
    }
    for key, t in state_dict.items():
        layer = key[: -len(".weight")] if key.endswith(".weight") else ""
        if layer in quantized:
            w = dequantize_nvfp4_tensor(
                t, state_dict[f"{layer}.weight_scale"],
                state_dict[f"{layer}.weight_scale_2"])
            pqs = state_dict.get(f"{layer}.pre_quant_scale")
            if pqs is not None:
                w = w * pqs.float().reshape(1, -1)
            out[key] = w.to(compute)
            continue
        if any(key.endswith(s) and key[: -len(s)] in quantized
               for s in _AUX_SUFFIXES):
            continue
        out[key] = t
    return out


def swap_w4a4_linears(
    model: Any,
    art: W4a4Artifact,
    *,
    compute_dtype: Any = None,
    key_map: Optional[Any] = None,
) -> int:
    """Swap the artifact's quantized Linears in an ALREADY-CONSTRUCTED model
    onto :class:`W4A4Linear`, assigning packed weights + scales from the
    shards (the root-layout lane, gw#562 pattern). Misaligned or non-Linear
    layers keep their dequantized weights. Returns swapped count."""
    import torch
    import torch.nn as nn
    from safetensors import safe_open

    compute = compute_dtype or torch.bfloat16
    lin_cls = w4a4_linear_class()
    where: Dict[str, Path] = {}
    for f in art.files:
        for name in _read_header(f):
            if name != "__metadata__":
                where[name] = f
    handles: Dict[Path, Any] = {}

    def _tensor(name: str) -> Any:
        src = where.get(name)
        if src is None:
            raise W4a4SnapshotError(f"artifact tensor {name!r} missing from shards")
        fh = handles.get(src)
        if fh is None:
            fh = handles[src] = safe_open(str(src), framework="pt", device="cpu")
        return fh.get_tensor(name)

    swapped = 0
    try:
        for layer in art.quantized:
            target = str(key_map(layer)) if key_map is not None else layer
            parent_path, _, leaf = target.rpartition(".")
            try:
                parent = (model.get_submodule(parent_path)
                          if parent_path else model)
                old = getattr(parent, leaf)
            except AttributeError as exc:
                raise W4a4SnapshotError(
                    f"quantized layer {layer!r} has no module {target!r} in "
                    f"{type(model).__name__} — wrong key_map?") from exc
            if not isinstance(old, nn.Linear) or type(old) is not nn.Linear:
                logger.warning(
                    "w4a4 swap: %s is %s, not a plain Linear; layer stays "
                    "dequantized", target, type(old).__name__)
                continue
            w = _tensor(f"{layer}.weight")
            out_f, in_f = int(w.shape[0]), int(w.shape[1]) * 2
            if (out_f, in_f) != (int(old.out_features), int(old.in_features)):
                raise W4a4SnapshotError(
                    f"quantized layer {layer!r} shape [{out_f}, {in_f}] != "
                    f"module {target!r} [{old.out_features}, {old.in_features}]")
            if in_f % _K_ALIGN or out_f % _N_ALIGN:
                continue
            has_static = f"{layer}.input_scale" in where
            has_pqs = f"{layer}.pre_quant_scale" in where
            dev = old.weight.device
            new = lin_cls(in_f, out_f, bias=old.bias is not None,
                          compute_dtype=compute, static_input_scale=has_static,
                          pre_quant_scale=has_pqs)
            new.weight = w.contiguous().to(dev)
            new.weight_scale = to_blocked_scales(
                _tensor(f"{layer}.weight_scale")).to(dev)
            new.weight_scale_2 = _tensor(
                f"{layer}.weight_scale_2").float().reshape(1, 1).to(dev)
            if has_static:
                new.input_scale = _tensor(
                    f"{layer}.input_scale").float().reshape(1, 1).to(dev)
            if has_pqs:
                new.pre_quant_scale = _tensor(
                    f"{layer}.pre_quant_scale").to(compute).reshape(-1).to(dev)
            if old.bias is not None:
                new.bias = nn.Parameter(
                    old.bias.detach().to(compute), requires_grad=False)
            setattr(parent, leaf, new)
            swapped += 1
    finally:
        for fh in handles.values():
            try:
                fh.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
    logger.info("w4a4 swap: %d/%d quantized Linears on fp4 scaled_mm",
                swapped, len(art.quantized))
    return swapped


def _root_denoiser(pipe: Any) -> Any:
    import torch.nn as nn

    for name in ("transformer", "unet", "dit"):
        mod = getattr(pipe, name, None)
        if isinstance(mod, nn.Module):
            return mod
    if isinstance(pipe, nn.Module):
        return pipe
    raise W4a4SnapshotError(
        f"{type(pipe).__name__} exposes no denoiser module "
        "(transformer/unet/dit) for the root w4a4 lane")


def load_w4a4_root_pipeline(
    cls: Any, path: Path, art: W4a4Artifact, *, compute_dtype: Any = None,
) -> Any:
    """Serve a root-layout w4a4 snapshot through the pipeline class's own
    ``from_pretrained`` (whose loader must run sanitize_w4a4_state_dict),
    then swap the denoiser's quantized Linears when the host qualifies."""
    import torch

    compute = compute_dtype or torch.bfloat16
    mode = w4a4_gemm_mode() or "dequant"
    pipe = cls.from_pretrained(str(path), torch_dtype=compute)
    denoiser = _root_denoiser(pipe)
    if mode != "dequant":
        if not swap_w4a4_linears(denoiser, art, compute_dtype=compute):
            raise W4a4SnapshotError(
                "fp4 scaled_mm host but no quantized Linear swapped — "
                f"artifact keys do not match {type(denoiser).__name__} modules")
    try:
        denoiser._cozy_w4a4_mode = mode
        pipe._cozy_weight_lane = (
            "w4a4" if mode != "dequant" else "bf16-resident")
    except Exception:
        pass
    logger.info("w4a4 root lane: %s (%d quantized layers, component root)",
                mode, len(art.quantized))
    return pipe


# ---------------------------------------------------------------------------
# Data-free producer — contract round-trips in tests + the parity harness.
# Production calibrated artifacts come from the conversion side (modelopt,
# te#79/te#80: nvfp4 calibrates unconditionally); this writes the identical
# on-disk contract with dynamic per-tensor activation scales.
# ---------------------------------------------------------------------------


def quantize_tree_w4a4(
    src_tree: Path,
    out_tree: Path,
    *,
    exclude: tuple[str, ...] = ("embed", "norm"),
) -> Path:
    """Copy a diffusers tree, rewriting the denoiser's eligible 2D weights
    to the gw#540 contract triple. Eligible: ``*.weight``, 2D, float,
    in %% 32 == 0, out %% 16 == 0, name not matching ``exclude``."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    src_tree, out_tree = Path(src_tree), Path(out_tree)
    comp = next((c for c in _COMPONENT_DIRS if (src_tree / c).is_dir()), None)
    if comp is None or not (src_tree / "model_index.json").exists():
        raise W4a4SnapshotError(f"{src_tree} is not a diffusers tree with a denoiser")
    if out_tree.exists():
        shutil.rmtree(out_tree)
    shutil.copytree(src_tree, out_tree,
                    ignore=shutil.ignore_patterns("*.safetensors"))
    for f in sorted(src_tree.rglob("*.safetensors")):
        rel = f.relative_to(src_tree)
        dst = out_tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts[0] != comp:
            shutil.copy2(f, dst)
            continue
        tensors = load_file(str(f))
        out: Dict[str, Any] = {}
        quantized = 0
        for name, t in tensors.items():
            layer = name[: -len(".weight")] if name.endswith(".weight") else ""
            if (layer and t.ndim == 2 and t.is_floating_point()
                    and t.dtype != torch.float8_e4m3fn
                    and t.shape[0] % _N_ALIGN == 0
                    and t.shape[1] % _K_ALIGN == 0
                    and not any(x in layer for x in exclude)):
                packed, bs, ws2 = quantize_nvfp4_tensor(t)
                out[name] = packed
                out[f"{layer}.weight_scale"] = bs
                out[f"{layer}.weight_scale_2"] = ws2.reshape(())
                quantized += 1
            else:
                out[name] = t
        save_file(out, str(dst), metadata={
            "quant_scheme": W4A4_FLAVOR, "calibration_corpus": "",
            "modelopt_version": "",
        })
        logger.info("w4a4 producer: %s — %d layers quantized", rel, quantized)
    cfg_path = out_tree / comp / "config.json"
    cfg = json.loads(cfg_path.read_text("utf-8")) if cfg_path.exists() else {}
    cfg["quantization_config"] = {
        "quant_method": "modelopt", "quant_algo": "NVFP4"}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return out_tree


__all__ = [
    "W4A4_FLAVOR",
    "W4A4_MIN_SM",
    "W4a4Artifact",
    "W4a4Error",
    "W4a4SnapshotError",
    "cast_e2m1",
    "dequantize_nvfp4_tensor",
    "detect_w4a4_artifact",
    "load_w4a4_denoiser",
    "load_w4a4_pipeline",
    "load_w4a4_root_pipeline",
    "pack_e2m1",
    "quantize_nvfp4_tensor",
    "quantize_tree_w4a4",
    "sanitize_w4a4_state_dict",
    "swap_w4a4_linears",
    "to_blocked_scales",
    "unpack_e2m1",
    "w4a4_gemm_mode",
    "w4a4_linear_class",
]
