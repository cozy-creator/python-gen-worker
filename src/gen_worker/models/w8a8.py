"""W8A8 fp8-GEMM loader mode (gw#534).

A ``#fp8-w8a8`` flavor is a normal diffusers tree whose denoiser holds
calibrated fp8-E4M3 weights WITH scales — the artifact contract (frozen in
gw#534, consumed verbatim by the conversion side):

- per quantized Linear ``L``: ``L.weight`` (F8_E4M3, [out, in]),
  ``L.weight_scale`` (F32 DEQUANT multiplier — scalar or [out]/[out, 1]
  per-out-channel), optional ``L.input_scale`` (F32 scalar static activation
  scale), ``L.bias`` unquantized;
- excluded layers are stored at full precision with NO scale tensor —
  detection is per-layer by (fp8 dtype + ``weight_scale`` present), never by
  name lists;
- the denoiser's config.json carries
  ``quantization_config {"quant_method": "modelopt", "quant_algo": "FP8"}``
  (corroborating; the header evidence above is authoritative).

Execution: quantized Linears become :class:`Fp8ScaledLinear` —
``torch._scaled_mm`` over RESIDENT fp8 weights (no per-layer upcast: the
whole point, gw#534 measured the cast tax at +44% H100 / +73% B200), dynamic
per-row activation quant by default, the static scale when present. Hosts
without usable scaled_mm (pre-sm89 or missing kernels) DEQUANT once at load
into plain bf16-resident weights — same numerics, no speed win, never a
refusal.
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

W8A8_FLAVOR = "fp8-w8a8"
# torch._scaled_mm needs fp8 tensor cores (sm_89 Ada +) and 16-aligned dims.
W8A8_MIN_SM = 89
_FP8_MAX = 448.0
_DIM_ALIGN = 16
_MAX_HEADER_BYTES = 100 << 20
_COMPONENT_DIRS = ("transformer", "unet")


class W8a8Error(RuntimeError):
    """Typed w8a8 loader-mode failure."""


class W8a8SnapshotError(W8a8Error):
    """The flavor snapshot violates the artifact contract."""


@dataclass(frozen=True)
class W8a8Artifact:
    component: str            # denoiser dir name ("transformer"/"unet")
    files: tuple[Path, ...]   # the component's safetensors shards
    quantized: tuple[str, ...]  # module names with weight+weight_scale pairs
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


def detect_w8a8_artifact(model_path: Path) -> Optional[W8a8Artifact]:
    """Header-sniff a snapshot for the w8a8 contract: any denoiser-dir layer
    with an F8_E4M3 ``weight`` AND a ``weight_scale`` twin. A scale-FREE fp8
    tree (the storage-cast ``#fp8`` flavor) never matches — the scales are
    the distinguisher. Cheap: header reads only."""
    root = Path(model_path)
    if not root.is_dir() or not (root / "model_index.json").exists():
        return None
    for comp in _COMPONENT_DIRS:
        comp_dir = root / comp
        if not comp_dir.is_dir():
            continue
        files = tuple(sorted(p for p in comp_dir.glob("*.safetensors") if p.is_file()))
        if not files:
            continue
        dtypes: Dict[str, str] = {}
        for f in files:
            for name, info in _read_header(f).items():
                if isinstance(info, dict) and "dtype" in info:
                    dtypes[name] = str(info["dtype"])
        quantized = tuple(sorted(
            key[: -len(".weight_scale")]
            for key in dtypes
            if key.endswith(".weight_scale")
            and dtypes.get(key[: -len(".weight_scale")] + ".weight") == "F8_E4M3"
        ))
        if quantized:
            static = any(f"{n}.input_scale" in dtypes for n in quantized)
            return W8a8Artifact(
                component=comp, files=files, quantized=quantized,
                static_input_scales=static,
            )
    return None


@functools.lru_cache(maxsize=1)
def scaled_mm_supported() -> bool:
    """Live probe: does THIS device run a rowwise-scaled fp8 GEMM? sm gate
    first (fail-fast on old silicon), then one 16x16 kernel call — torch's
    scaled-mm backends vary by (torch, cuda, arch), so trusting a version
    table instead of the device is how mixed fleets break."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        return False
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < W8A8_MIN_SM:
        return False
    try:
        n = _DIM_ALIGN
        a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        sa = torch.ones(n, 1, device="cuda")
        sb = torch.ones(1, n, device="cuda")
        torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                         out_dtype=torch.bfloat16)
        return True
    except Exception as exc:  # noqa: BLE001 — any kernel gap => dequant lane
        logger.warning("w8a8: scaled_mm probe failed (%s); dequant lane", exc)
        return False


def _build_module_class() -> type:
    """Define the nn.Module lazily so importing this module never needs
    torch (discovery/CPU tools)."""
    import torch
    import torch.nn as nn

    class _Fp8ScaledLinear(nn.Module):
        """fp8 weights RESIDENT; y = scaled_mm(quant(x), W^T) + bias.

        ``weight_scale`` is the [out, 1] dequant multiplier (per-tensor
        scalars are expanded at load). Activation quant is per-row dynamic
        (amax/448) unless a static ``input_scale`` was calibrated. NOTE:
        never ``.to(dtype=...)`` this module — a dtype cast would upcast the
        fp8 buffer (device moves are fine).

        Optional LoRA side-branch (gw#547): ``lora_a`` [bucket, in] /
        ``lora_b`` [out, bucket] compute-dtype buffers, rank-padded to a
        fixed bucket so every adapter in the bucket shares one traced graph.
        The branch reads the ORIGINAL bf16 activation and adds onto the bf16
        output — quantized weights are never touched; hot-swap is a buffer
        copy (see models.w8a8_lora)."""

        weight: Any  # fp8 buffer (annotated: Module.__getattr__ unions confuse mypy)
        weight_scale: Any
        lora_a: Any  # None, or [bucket, in] bf16 branch buffer (gw#547)
        lora_b: Any  # None, or [out, bucket] with scale folded in
        # Structural marker consumed by compile_cache.execution_contract.
        # Unlike a class-name check it survives refactors and records no
        # checkpoint-specific data.
        _cozy_w8a8_linear = True

        def __init__(self, in_features: int, out_features: int, *,
                     bias: bool, compute_dtype: Any,
                     static_input_scale: bool) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            meta = torch.device("meta")
            self.register_buffer("weight", torch.empty(
                out_features, in_features, dtype=torch.float8_e4m3fn, device=meta))
            self.register_buffer("weight_scale", torch.empty(
                out_features, 1, dtype=torch.float32, device=meta))
            if static_input_scale:
                self.register_buffer("input_scale", torch.empty(
                    1, 1, dtype=torch.float32, device=meta))
            else:
                self.input_scale = None
            if bias:
                self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(
                    out_features, dtype=compute_dtype, device=meta))
            else:
                self.bias = None
            self.lora_a = None
            self.lora_b = None

        def _lora_addend(self, x2: Any) -> Any:
            # Per-adapter scale is folded into lora_b at copy time.
            return (x2 @ self.lora_a.t()) @ self.lora_b.t()

        def forward(self, x: Any) -> Any:
            shape = x.shape
            x2 = x.reshape(-1, self.in_features).contiguous()
            if self.input_scale is not None:
                sa = self.input_scale.expand(x2.shape[0], 1).contiguous()
            else:
                sa = (x2.abs().amax(dim=-1, keepdim=True).float()
                      / _FP8_MAX).clamp(min=1e-12)
            # Quantize in the COMPUTE dtype (reciprocal multiply) — fp32
            # intermediates here doubled the eager activation traffic and made
            # eager w8a8 as slow as the cast hooks it replaces (H100 measured;
            # bf16 mantissa loss is irrelevant next to the fp8 target).
            xq = (x2 * (1.0 / sa).to(x2.dtype)).clamp(
                -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            scaled_mm: Any = torch._scaled_mm
            y = scaled_mm(
                xq, self.weight.t(), scale_a=sa, scale_b=self.weight_scale.t(),
                bias=self.bias, out_dtype=x.dtype,
            )
            if self.lora_a is not None:
                y = y + self._lora_addend(x2)
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, "
                    f"bias={self.bias is not None}, "
                    f"static_input_scale={self.input_scale is not None}")

    return _Fp8ScaledLinear


@functools.lru_cache(maxsize=1)
def fp8_scaled_linear_class() -> type:
    return _build_module_class()


def _scale_2d(scale: Any, out_features: int) -> Any:
    """Normalize an on-disk weight_scale (scalar / [out] / [out,1]) to the
    module's [out, 1] float32 buffer shape."""
    s = scale.float()
    if s.numel() == 1:
        return s.reshape(1, 1).expand(out_features, 1).contiguous()
    if s.numel() != out_features:
        raise W8a8SnapshotError(
            f"weight_scale has {s.numel()} values for {out_features} out-channels")
    return s.reshape(out_features, 1).contiguous()


def _denoiser_class(root: Path, component: str) -> Any:
    index = json.loads((root / "model_index.json").read_text("utf-8"))
    entry = index.get(component)
    if not (isinstance(entry, list) and len(entry) == 2):
        raise W8a8SnapshotError(
            f"model_index.json has no [library, class] entry for {component!r}")
    lib, name = str(entry[0]), str(entry[1])
    try:
        mod = importlib.import_module(lib)
    except ImportError:
        mod = importlib.import_module("diffusers")
    cls = getattr(mod, name, None)
    if cls is None:
        raise W8a8SnapshotError(f"{lib} has no model class {name!r}")
    return cls


def load_w8a8_denoiser(root: Path, art: W8a8Artifact, *,
                       compute_dtype: Any = None, mode: str = "") -> Any:
    """Materialize the quantized denoiser: skeleton on meta, quantized
    Linears swapped for Fp8ScaledLinear, tensors assigned from the shards.
    ``mode`` "scaled_mm" | "dequant" (default: probe). Layers whose dims
    break scaled_mm's 16-alignment are dequantized individually."""
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    compute = compute_dtype or torch.bfloat16
    if mode not in ("scaled_mm", "dequant"):
        mode = "scaled_mm" if scaled_mm_supported() else "dequant"

    cls = _denoiser_class(root, art.component)
    cfg = dict(cls.load_config(str(root / art.component)))
    cfg.pop("quantization_config", None)
    with init_empty_weights():
        model = cls.from_config(cfg)

    sd: Dict[str, Any] = {}
    for f in art.files:
        sd.update(load_file(str(f)))

    lin_cls = fp8_scaled_linear_class()
    swapped = 0
    for name in art.quantized:
        w = sd[f"{name}.weight"]
        scale = sd[f"{name}.weight_scale"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        eligible = (mode == "scaled_mm"
                    and in_f % _DIM_ALIGN == 0 and out_f % _DIM_ALIGN == 0)
        try:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            old = getattr(parent, leaf)
        except AttributeError as exc:
            raise W8a8SnapshotError(
                f"quantized tensor {name!r} has no module in "
                f"{type(model).__name__}") from exc
        if eligible and isinstance(old, nn.Linear):
            has_static = f"{name}.input_scale" in sd
            new = lin_cls(in_f, out_f, bias=old.bias is not None,
                          compute_dtype=compute, static_input_scale=has_static)
            setattr(parent, leaf, new)
            sd[f"{name}.weight_scale"] = _scale_2d(scale, out_f)
            if has_static:
                sd[f"{name}.input_scale"] = (
                    sd[f"{name}.input_scale"].float().reshape(1, 1))
            swapped += 1
        else:
            # Per-layer dequant: misaligned dims, non-Linear owner, or the
            # host-wide dequant lane. Same numerics as the artifact.
            sd[f"{name}.weight"] = (
                w.float() * _scale_2d(scale, out_f)).to(compute)
            del sd[f"{name}.weight_scale"]
            sd.pop(f"{name}.input_scale", None)

    for key, value in list(sd.items()):
        if value.is_floating_point() and value.dtype not in (
                torch.float8_e4m3fn,) and not key.endswith(
                (".weight_scale", ".input_scale")):
            sd[key] = value.to(compute)

    result = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in result.missing_keys]
    if missing or result.unexpected_keys:
        raise W8a8SnapshotError(
            f"w8a8 state dict mismatch: missing={missing[:5]} "
            f"unexpected={list(result.unexpected_keys)[:5]}")
    model.eval()
    model._cozy_w8a8_mode = mode
    logger.info(
        "w8a8 loader mode: %s — %d/%d quantized Linears on scaled_mm "
        "(component %s, static input scales: %s)",
        mode, swapped, len(art.quantized), art.component,
        art.static_input_scales,
    )
    return model


def load_w8a8_pipeline(cls: Any, path: Path, art: W8a8Artifact, *,
                       compute_dtype: Any = None,
                       components: Optional[Dict[str, Any]] = None,
                       fp8_text_encoders: bool = False) -> Any:
    """Build the pipeline with the w8a8 denoiser wired in (svdq-style
    component injection). Stamps ``_cozy_weight_lane`` ("w8a8" on the
    scaled_mm lane, "bf16-resident" on the dequant lane) — the compile-cache
    graph key (lane_drift, gw#534). ``fp8_text_encoders`` (the
    ``storage_dtype="fp8+te"`` binding, gw#557) arms the gw#460 block-window
    fp8 storage on the TEXT ENCODERS ONLY — the denoiser already holds fp8
    scaled-mm modules that cast hooks must never touch."""
    import torch

    compute = compute_dtype or torch.bfloat16
    mode = "scaled_mm" if scaled_mm_supported() else "dequant"
    denoiser = load_w8a8_denoiser(
        path, art, compute_dtype=compute, mode=mode)
    kwargs: Dict[str, Any] = dict(components or {})
    kwargs[art.component] = denoiser
    pipe = cls.from_pretrained(str(path), torch_dtype=compute, **kwargs)
    try:
        pipe._cozy_weight_lane = "w8a8" if mode == "scaled_mm" else "bf16-resident"
    except Exception:
        pass
    if fp8_text_encoders:
        from .loading import _FP8_TEXT_ENCODER_COMPONENTS, apply_fp8_storage

        targets = tuple(
            n for n in _FP8_TEXT_ENCODER_COMPONENTS
            if hasattr(getattr(pipe, n, None), "parameters"))
        if targets:
            apply_fp8_storage(pipe, compute_dtype=compute, components=targets)
        else:
            logger.warning(
                "w8a8: storage_dtype=fp8+te requested but %s has no text "
                "encoders; serving without TE windows", type(pipe).__name__)
    return pipe


# ---------------------------------------------------------------------------
# Data-free producer — contract round-trips in tests + the parity harness.
# Production calibrated artifacts come from the conversion side (modelopt,
# te#79); this writes the identical on-disk contract with per-out-channel
# amax scales and no calibration.
# ---------------------------------------------------------------------------


def quantize_tree_w8a8(
    src_tree: Path,
    out_tree: Path,
    *,
    exclude: tuple[str, ...] = ("embed", "norm"),
) -> Path:
    """Copy a diffusers tree, rewriting the denoiser's eligible 2D weights to
    fp8 + per-out-channel ``weight_scale`` per the gw#534 contract. Eligible:
    ``*.weight``, 2D, bf16/fp16/fp32, both dims 16-aligned, name not matching
    ``exclude`` substrings."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    src_tree, out_tree = Path(src_tree), Path(out_tree)
    comp = next((c for c in _COMPONENT_DIRS if (src_tree / c).is_dir()), None)
    if comp is None or not (src_tree / "model_index.json").exists():
        raise W8a8SnapshotError(f"{src_tree} is not a diffusers tree with a denoiser")
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
                    and t.shape[0] % _DIM_ALIGN == 0
                    and t.shape[1] % _DIM_ALIGN == 0
                    and not any(x in layer for x in exclude)):
                w = t.float()
                scale = (w.abs().amax(dim=1, keepdim=True)
                         / _FP8_MAX).clamp(min=1e-12)
                out[name] = (w / scale).clamp(
                    -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
                out[f"{layer}.weight_scale"] = scale.reshape(-1)
                quantized += 1
            else:
                out[name] = t
        save_file(out, str(dst), metadata={
            "quant_scheme": W8A8_FLAVOR, "calibration_corpus": "",
            "modelopt_version": "",
        })
        logger.info("w8a8 producer: %s — %d layers quantized", rel, quantized)
    cfg_path = out_tree / comp / "config.json"
    cfg = json.loads(cfg_path.read_text("utf-8")) if cfg_path.exists() else {}
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "FP8"}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return out_tree


__all__ = [
    "W8A8_FLAVOR",
    "W8A8_MIN_SM",
    "W8a8Artifact",
    "W8a8Error",
    "W8a8SnapshotError",
    "detect_w8a8_artifact",
    "fp8_scaled_linear_class",
    "load_w8a8_denoiser",
    "load_w8a8_pipeline",
    "quantize_tree_w8a8",
    "scaled_mm_supported",
]
