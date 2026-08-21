"""Native svdq-fp4 serving — SVDQuant checkpoints without nunchaku."""

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
import json
import time
from .w4a4 import w4a4_gemm_mode
from .w4a4 import _gemm_w4a4
from .native_kernels import svdq_linear_execution_lane
from .svdq_fused import build_svdq_fused_linear, fused_shape_supported
from .svdq import _read_safetensors_metadata
from .materialized_view import third_party_dir
from .tensor_source import open_tensor_source
from .svdq_awq import decode_awq_linear, is_awq_linear
from .svdq_layout import LOWRANK_QUANT_KEY, LOWRANK_QUANT_SCHEMES, decode_linear
from .native_kernels import svdq_modulation_execution_lane
from .svdq_awq_packed import awq_packed_supported, build_awq_packed_linear
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

SVDQ_ENGINE_NATIVE = "native"

_SVDQ_WHY = (
    "the svdq denoiser's every W4A4 linear, AWQ modulation layer and plain "
    "tensor comes from this one read; without it the model cannot be built"
)

# Blackwell fp4 tensor cores — the same silicon window as the #nvfp4-w4a4 lane. torch's own gate is only `major >= 9 || (8,9)`, which ADMITS sm_89/sm_90, but neither Ada nor Hopper has fp4 tensor cores; below Blackwell the honest degrade is fp8 rowwise, which we already ship. Never emulate fp4.
SVDQ_NATIVE_FP4_SMS = (100, 103, 120, 121)

_FUSED_SPLITS: dict[str, tuple[str, ...]] = {
    "to_qkv": ("to_q", "to_k", "to_v"),
    "add_qkv_proj": ("add_q_proj", "add_k_proj", "add_v_proj"),
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
}

_K_ALIGN = 32
_N_ALIGN = 16


class SvdqNativeError(RuntimeError):
    """Typed native-svdq failure."""


def svdq_native_sm_supported(gpu_sm: int) -> bool:
    return int(gpu_sm) in SVDQ_NATIVE_FP4_SMS


def svdq_native_reason() -> Optional[str]:
    """Why the native engine cannot serve fp4 HERE, or None when it can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not cuda_ready():
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
    """Silicon + the real w4a4 arming path (kernel probe, numerics self-check, profitability gate, fused-quantizer bit-identity gate) — the native engine shares all of it with the ``#nvfp4-w4a4`` lane."""
    if svdq_native_reason() is not None:
        return False

    return w4a4_gemm_mode() == "blockwise"


def _build_svdq_linear_class() -> type:
    import torch
    import torch.nn as nn

    class _SvdqLinear(nn.Module):

        _cozy_svdq_linear = True

        def __init__(self, in_features: int, out_features: int, *,
                     rank: int, bias: bool, compute_dtype: Any,
                     per_channel_scale: bool, smooth: bool) -> None:
            super().__init__()
            self.compute_dtype = compute_dtype
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.rank = int(rank)
            self.per_channel_scale = bool(per_channel_scale)
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

            shape = x.shape
            x2 = x.reshape(-1, self.in_features)
            xs = x2 if self.smooth_factor is None else x2 / self.smooth_factor
            s2 = (xs.abs().amax().float()
                  / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
            xq, sa_blocked = quantize_activation(xs, s2)
            y = _gemm_w4a4(xq, self.weight, sa_blocked, self.weight_scale,
                           x.dtype)
            y = y * (s2 * self.weight_scale_2).to(y.dtype)
            down, up = self.proj_down, self.proj_up
            if down is not None and up is not None:
                y = y + (x2 @ down) @ up.t()
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, rank={self.rank}, "
                    f"bias={self.bias is not None}, "
                    f"per_channel_scale={self.per_channel_scale}, "
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
    if buf.proj_down is not None and buf.proj_up is not None:
        mod.proj_down = buf.proj_down.to(compute).to(dev)
        mod.proj_up = buf.proj_up.to(compute).to(dev)
    if buf.bias is not None:
        mod.bias = nn.Parameter(buf.bias.detach().to(compute).to(dev),
                                requires_grad=False)
    return mod


def fold_to_dense(dec: DecodedLinear, *, compute_dtype: Any = None) -> Any:
    """The any-hardware fallback: ONE plain bf16 ``nn.Linear`` equivalent to the whole svdq linear."""
    import torch
    import torch.nn as nn

    compute = compute_dtype or torch.bfloat16
    w = dequantize_decoded(dec)
    if dec.smooth_factor is not None:
        w = w / dec.smooth_factor.float().reshape(1, -1)
    up, down = dec.proj_up, dec.proj_down
    if up is not None and down is not None:
        w = w + (up.float() @ down.float().t())
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
    """Where one nunchaku linear prefix lands in ``model``."""
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
    """Replace ``model``'s Linears with the checkpoint's svdq linears."""
    import torch

    compute = compute_dtype or torch.bfloat16
    if mode not in ("blockwise", "dense"):
        mode = "blockwise" if svdq_native_available() else "dense"
    execution_lane = svdq_linear_execution_lane() if mode == "blockwise" else "baseline"
    counts = {"blockwise": 0, "dense": 0, "fused": 0, "prefixes": 0,
              "linears": 0}
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
            fused_ok = (fp4_ok and execution_lane == "fused" and fused_shape_supported(
                part.out_features, part.in_features, part.rank))
            if fused_ok:
                new = build_svdq_fused_linear(part, compute_dtype=compute,
                                              device=device)
                counts["fused"] += 1
            elif fp4_ok:
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
        "svdq native swap: %d prefixes -> %d linears (%d fused, %d fp4, "
        "%d folded bf16)", counts["prefixes"], counts["linears"],
        counts["fused"], counts["blockwise"], counts["dense"])
    return counts


_ADANORM_SPLITS: dict[tuple[str, str], int] = {
    ("QwenImageTransformer2DModel", "img_mod.1"): 6,
    ("QwenImageTransformer2DModel", "txt_mod.1"): 6,
}


def adanorm_splits_for(model_class: str, prefix: str) -> int:
    """adaLN split count for one AWQ modulation layer, or a typed refusal."""
    for (cls, suffix), n in _ADANORM_SPLITS.items():
        if cls == model_class and prefix.endswith(suffix):
            return n
    raise SvdqNativeError(
        f"unknown adaLN split count for AWQ layer {prefix!r} in "
        f"{model_class} — the exporter's adanorm transform cannot be inferred "
        f"from the tensors, and a wrong count corrupts output silently; add "
        f"the (class, suffix) entry to _ADANORM_SPLITS after verifying it")


def native_denoiser_class(model_class: str) -> Any:
    """The diffusers class behind a nunchaku ``model_class`` name (the native engine subclasses nothing — it loads the STOCK diffusers module)."""
    import diffusers

    name = (model_class[len("Nunchaku"):]
            if model_class.startswith("Nunchaku") else model_class)
    cls = getattr(diffusers, name, None)
    if cls is None:
        raise SvdqNativeError(
            f"diffusers has no {name!r} (from checkpoint model_class "
            f"{model_class!r})")
    return cls


def _group_by_prefix(names: Any) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key in names:
        if key == "__metadata__":
            continue
        prefix, _, leaf = key.rpartition(".")
        groups.setdefault(prefix, []).append(leaf)
    return groups


def load_svdq_native_denoiser(art: Any, *, compute_dtype: Any = None,
                             mode: str = "", device: Any = None) -> Any:
    """Materialize a nunchaku-format svdq checkpoint as a STOCK diffusers denoiser: skeleton on meta, W4A4 linears swapped for :class:`SvdqLinear` (fused ``to_qkv`` split across the diffusers projections)..."""

    import torch
    from accelerate import init_empty_weights

    compute = compute_dtype or torch.bfloat16
    meta = _read_safetensors_metadata(art.file)
    model_class = str(meta.get("model_class") or "")
    lowrank_quant = str(meta.get(LOWRANK_QUANT_KEY) or "bf16")
    if lowrank_quant not in ("bf16",) + LOWRANK_QUANT_SCHEMES:
        raise SvdqNativeError(
            f"svdq checkpoint {art.file.name} declares "
            f"{LOWRANK_QUANT_KEY}={lowrank_quant!r} — not a known low-rank "
            f"branch scheme (bf16, {', '.join(LOWRANK_QUANT_SCHEMES)})")
    cfg_raw = meta.get("config")
    try:
        cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else dict(cfg_raw or {})
    except ValueError as exc:
        raise SvdqNativeError(
            f"svdq checkpoint {art.file.name} has an unparseable config") from exc
    if not cfg:
        raise SvdqNativeError(
            f"svdq checkpoint {art.file.name} carries no config in its "
            f"__metadata__ — cannot build the diffusers module")
    cfg.pop("quantization_config", None)

    cls = native_denoiser_class(model_class)
    with init_empty_weights():
        model = cls.from_config(cfg)

    if mode not in ("blockwise", "dense"):
        mode = "blockwise" if svdq_native_available() else "dense"
    if device is None:
        device = ("cuda" if mode == "blockwise" and cuda_ready()
                  else "cpu")
    dev = torch.device(device)

    mod_execution_lane = svdq_modulation_execution_lane() if mode == "blockwise" else "dense"
    t0 = time.perf_counter()
    plain: Dict[str, Any] = {}
    swapped = awq = awq_packed = 0
    with open_tensor_source(art.file, device=str(dev), why=_SVDQ_WHY) as fh:
        groups = _group_by_prefix(fh.keys())
        for prefix, leaves in sorted(groups.items()):
            tensors = {leaf: fh.get_tensor(f"{prefix}.{leaf}") for leaf in leaves}
            if is_awq_linear(tensors):
                target = _module_at(model, prefix)
                if target is None or not hasattr(target, "out_features"):
                    raise SvdqNativeError(
                        f"AWQ layer {prefix!r} has no Linear in "
                        f"{type(model).__name__}")
                splits = adanorm_splits_for(type(model).__name__, prefix)
                out_f = int(target.out_features)
                in_f = int(target.in_features)
                if mod_execution_lane == "packed" and awq_packed_supported(out_f, in_f):
                    _set_module(model, prefix, build_awq_packed_linear(
                        tensors, out_f, in_f, adanorm_splits=splits,
                        compute_dtype=compute, device=dev))
                    awq_packed += 1
                else:
                    _set_module(model, prefix, decode_awq_linear(
                        tensors, out_f, in_f, adanorm_splits=splits,
                        compute_dtype=compute, device=dev))
                    awq += 1
                continue
            if "qweight" in tensors and "wscales" in tensors:
                targets = plan_targets(model, prefix)
                out_f = sum(o for _, o in targets)
                in_f = int(_module_at(model, targets[0][0]).in_features)
                dec = decode_linear(tensors, out_f, in_f,
                                    lowrank_quant=lowrank_quant)
                swapped += swap_svdq_linears(
                    model, {prefix: dec}, compute_dtype=compute,
                    mode=mode, device=dev)["linears"]
                continue
            for leaf, t in tensors.items():
                key = f"{prefix}.{leaf}" if prefix else leaf
                plain[key] = t.to(compute) if t.is_floating_point() else t

    result = model.load_state_dict(plain, strict=False, assign=True)
    still_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    still_meta += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if still_meta:
        raise SvdqNativeError(
            f"svdq native load left {len(still_meta)} tensor(s) on meta "
            f"(e.g. {still_meta[:5]}) — checkpoint keys do not cover "
            f"{type(model).__name__}")
    if result.unexpected_keys:
        logger.warning("svdq native load: %d unexpected checkpoint keys (e.g. %s)",
                       len(result.unexpected_keys), list(result.unexpected_keys)[:5])
    model.eval()
    model._cozy_svdq_engine = SVDQ_ENGINE_NATIVE
    model._cozy_svdq_mode = mode
    logger.info(
        "svdq native loader: %s mode=%s lowrank=%s device=%s — %d W4A4 "
        "linears, %d AWQ modulation layers (%d packed-resident), %d plain "
        "tensors in %.1fs",
        type(model).__name__, mode, lowrank_quant, dev, swapped,
        awq + awq_packed, awq_packed, len(plain), time.perf_counter() - t0)
    return model


def load_svdq_native_pipeline(cls: Any, path: Any, art: Any, *,
                             compute_dtype: Any = None) -> Any:
    """Build the pipeline with a natively-loaded svdq denoiser wired in."""
    import torch

    compute = compute_dtype or torch.bfloat16
    if not art.component:
        raise SvdqNativeError(
            f"svdq snapshot {path} is a bare single-file transformer; a "
            f"servable flavor must be a full diffusers tree with the "
            f"checkpoint under its denoiser directory")
    denoiser = load_svdq_native_denoiser(art, compute_dtype=compute)
    pipe = cls.from_pretrained(
        str(third_party_dir(path, why="svdq non-denoiser parts from_pretrained")),
        torch_dtype=compute,
                              **{art.component: denoiser})
    try:
        pipe._cozy_weight_lane = (
            "svdq-native" if getattr(denoiser, "_cozy_svdq_mode", "") == "blockwise"
            else "bf16-resident")
    except Exception:  # noqa: BLE001
        pass
    return pipe


__all__ = [
    "SVDQ_ENGINE_NATIVE",
    "SVDQ_NATIVE_FP4_SMS",
    "SvdqNativeError",
    "adanorm_splits_for",
    "build_svdq_linear",
    "fold_to_dense",
    "load_svdq_native_denoiser",
    "load_svdq_native_pipeline",
    "native_denoiser_class",
    "plan_targets",
    "svdq_linear_class",
    "svdq_native_available",
    "svdq_native_reason",
    "svdq_native_sm_supported",
    "swap_svdq_linears",
]
