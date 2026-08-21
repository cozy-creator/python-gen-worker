"""W8A8 fp8-GEMM loader mode — the tensor-layout contract, consumed verbatim by the conversion side. Per quantized Linear L: L.weight F8_E4M3 [out, in]; L.weight_scale F32 DEQUANT multiplier (scalar, or [out]/[out, 1] per-out-channel); optional L.input_scale F32 scalar static activation scale; bias unquantized. Excluded layers are stored at full precision with NO scale tensor — detection is per-layer by (fp8 dtype + weight_scale present), never by name lists; config.json's quantization_config block is corroborating, the header evidence is authoritative. Execution: Fp8ScaledLinear over RESIDENT fp8 weights via torch._scaled_mm, with TWO dispatch branches chosen once at load by w8a8_gemm_mode — "rowwise" (scale vectors inside the GEMM, sm_90+) and "pertensor" (scalar-scaled cuBLASLt + per-channel epilogue rescale, the Ada/sm_89 fast path). Hosts where no branch wins the load-time micro-benchmark DEQUANT once into bf16-resident weights — same numerics, never a refusal."""

from __future__ import annotations

import functools
import importlib
import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from .. import activity as activity_mod
from ..component_vocab import denoiser_components
from .materialized_view import third_party_dir
from .safetensors_header import read_header
from .tensor_source import load_state_dict, open_tensor_source
from .tensor_layout_contract import implements_quant_rule
from typing import Any, Dict, List, Optional
import shutil
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

W8A8_FLAVOR = "fp8-w8a8"
W8A8_MIN_SM = 89
W8A8_ROWWISE_MIN_SM = 90
_FP8_MAX = 448.0
_DIM_ALIGN = 16


class W8a8Error(RuntimeError):
    """Typed w8a8 loader-mode failure."""


class W8a8SnapshotError(W8a8Error):
    """The flavor snapshot violates the tensor-layout contract."""


@dataclass(frozen=True)
class W8a8Artifact:
    component: str
    files: tuple[Path, ...]
    quantized: tuple[str, ...]
    static_input_scales: bool


_TENSOR_WHY = (
    "the w8a8 denoiser's fp8 weights and scales come from this read"
)

_HEADER_WHY = (
    "an fp8 w8a8 artifact whose scales go unseen is routed to the "
    "plain bf16 lane and loads as the wrong model"
)


def _read_header(path: Path) -> dict:

    return read_header(path, why=_HEADER_WHY)


def _quantized_layers(files: tuple[Path, ...]) -> tuple[tuple[str, ...], bool]:
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
    static = any(f"{n}.input_scale" in dtypes for n in quantized)
    return quantized, static


def _root_weight_files(d: Path) -> tuple[Path, ...]:
    dirs = [d] + sorted(
        p for p in d.rglob("*")
        if p.is_dir()
        and not any(part.startswith(".") for part in p.relative_to(d).parts))
    files: list[Path] = []
    for sub in dirs:
        sharded: set[str] = set()
        for idx in sorted(sub.glob("*.safetensors.index.json")):
            try:
                weight_map = json.loads(idx.read_text("utf-8")).get("weight_map") or {}
                sharded.update(str(v) for v in weight_map.values())
            except (OSError, ValueError):
                continue
        files += [sub / s for s in sorted(sharded) if (sub / s).is_file()]
        files += [p for p in sorted(sub.glob("*.safetensors"))
                  if p.is_file() and p.name not in sharded]
    return tuple(dict.fromkeys(files))


def detect_w8a8_artifacts(model_path: Path) -> tuple[W8a8Artifact, ...]:
    """EVERY quantized denoiser in a snapshot, in vocabulary order."""
    root = Path(model_path)
    if not root.is_dir():
        return ()
    found: list[W8a8Artifact] = []
    if (root / "model_index.json").exists():
        for comp in denoiser_components():
            comp_dir = root / comp
            if not comp_dir.is_dir():
                continue
            files = tuple(sorted(
                p for p in comp_dir.glob("*.safetensors") if p.is_file()))
            if not files:
                continue
            quantized, static = _quantized_layers(files)
            if quantized:
                found.append(W8a8Artifact(
                    component=comp, files=files, quantized=quantized,
                    static_input_scales=static,
                ))
        return tuple(found)
    files = _root_weight_files(root)
    if files:
        quantized, static = _quantized_layers(files)
        if quantized:
            found.append(W8a8Artifact(
                component="", files=files, quantized=quantized,
                static_input_scales=static,
            ))
    return tuple(found)


def detect_w8a8_artifact(model_path: Path) -> Optional[W8a8Artifact]:
    """The FIRST quantized denoiser, for callers that ask a yes/no question or address one named component."""
    arts = detect_w8a8_artifacts(model_path)
    return arts[0] if arts else None


def _probe_scales(mode: str, m: int, n: int, device: str = "cuda") -> tuple:
    import torch

    if mode == "rowwise":
        return (torch.ones(m, 1, device=device),
                torch.ones(1, n, device=device))
    return (torch.ones(1, 1, device=device),
            torch.ones(1, 1, device=device))


def _gemm_call_ok(mode: str) -> bool:
    import torch

    try:
        n = _DIM_ALIGN
        a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        sa, sb = _probe_scales(mode, n, n)
        torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                         out_dtype=torch.bfloat16)
        return True
    except Exception as exc:  # noqa: BLE001 — any kernel gap => next candidate
        logger.warning("w8a8: %s scaled_mm probe failed (%s)", mode, exc)
        return False


_BENCH_DIM = 4096
_BENCH_WARMUP = 3
_BENCH_ITERS = 10
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


def _bench_gemm_pair(mode: str) -> tuple[float, float]:
    import torch

    m = n = k = _BENCH_DIM
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    xq = x.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    wq = w.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    sa, sb = _probe_scales(mode, m, n)
    ws_t = torch.full((1, n), 0.01, device="cuda", dtype=torch.bfloat16)

    def fp8_op() -> Any:
        y = torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=sb,
                             out_dtype=torch.bfloat16)
        return y * ws_t if mode == "pertensor" else y

    def bf16_op() -> Any:
        return x @ w.t()

    return _median_ms(fp8_op), _median_ms(bf16_op)


def _gemm_profitable(mode: str) -> bool:
    fp8_ms, bf16_ms = _bench_gemm_pair(mode)
    speedup = bf16_ms / max(fp8_ms, 1e-9)
    logger.info(
        "w8a8 gemm gate: mode=%s fp8=%.3fms bf16=%.3fms speedup=%.2fx "
        "(min %.2fx)", mode, fp8_ms, bf16_ms, speedup, _BENCH_MIN_SPEEDUP)
    return speedup >= _BENCH_MIN_SPEEDUP


def _choose_gemm_mode(sm: int) -> str:
    if sm < W8A8_MIN_SM:
        return ""
    candidates = (("rowwise", "pertensor") if sm >= W8A8_ROWWISE_MIN_SM
                  else ("pertensor", "rowwise"))
    for mode in candidates:
        if not _gemm_call_ok(mode):
            continue
        try:
            if _gemm_profitable(mode):
                return mode
        except Exception as exc:  # noqa: BLE001 — bench failure => next lane
            logger.warning("w8a8: %s micro-benchmark failed (%s)", mode, exc)
    logger.warning(
        "w8a8: no fp8 GEMM branch engages a real win on this device (sm_%d); "
        "dequant lane", sm)
    activity_mod.emit_event(
        activity_mod.KIND_SERVE_DEGRADE,
        f"sm_{sm}: no fp8 GEMM branch qualified (probe or micro-benchmark "
        f"declined every candidate), so this pod serves the w8a8 artifact on "
        f"the DEQUANT lane — the memory saving is kept, the fused-GEMM speed "
        f"is not",
        phase="w8a8_gemm_unqualified",
    )
    return ""


@functools.lru_cache(maxsize=1)
def w8a8_gemm_mode() -> str:
    """The fp8 GEMM dispatch for THIS device, chosen once per process: ``"rowwise"`` (scale vectors consumed inside ``_scaled_mm`` — CUTLASS fast kernels, sm_90+), ``"pertensor"`` (scalar-scaled cuBLASLt ..."""
    try:
        import torch
    except ImportError:
        return ""
    if not cuda_ready() or not hasattr(torch, "float8_e4m3fn"):
        return ""
    major, minor = torch.cuda.get_device_capability()
    return _choose_gemm_mode(major * 10 + minor)


def _build_quantizer(torch: Any) -> Any:

    def _quant_src(x2: Any, pertensor: bool, static: Any) -> tuple:
        if static is not None:
            sa = (static if pertensor
                  else static.expand(x2.shape[0], 1).contiguous())
        elif pertensor:
            sa = (x2.abs().amax().float()
                  / _FP8_MAX).clamp(min=1e-12).reshape(1, 1)
        else:
            sa = (x2.abs().amax(dim=-1, keepdim=True).float()
                  / _FP8_MAX).clamp(min=1e-12)
        xq = (x2 * (1.0 / sa).to(x2.dtype)).clamp(
            -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        return xq, sa

    state: dict[str, Any] = {"fused": None, "off": False}

    def quantize(x2: Any, pertensor: bool, static: Any) -> tuple:
        if state["off"] or torch.compiler.is_compiling():
            return _quant_src(x2, pertensor, static)
        fused = state["fused"]
        if fused is None:
            try:
                fused = torch.compile(_quant_src, dynamic=False, fullgraph=True)
            except Exception as exc:  # noqa: BLE001 — no inductor => eager, not a refusal
                _quant_degrade(exc)
                state["off"] = True
                return _quant_src(x2, pertensor, static)
            state["fused"] = fused
        try:
            return fused(x2, pertensor, static)
        except Exception as exc:  # noqa: BLE001
            _quant_degrade(exc)
            state["off"] = True
            return _quant_src(x2, pertensor, static)

    quantize.eager_source = _quant_src  # type: ignore[attr-defined]
    return quantize


def _quant_degrade(exc: BaseException) -> None:
    logger.warning(
        "w8a8: the fused activation-quantize did not build (%s); this process "
        "quantizes op-by-op", exc)
    activity_mod.emit_event(
        activity_mod.KIND_SERVE_DEGRADE,
        f"the fp8 activation quantize could not be fused ({type(exc).__name__}: "
        f"{exc}), so it runs as six separate passes over every activation — on "
        f"narrow-output projections that is SLOWER than bf16 (pgw#1156)",
        phase="w8a8_quant_unfused",
    )


def _build_module_class() -> type:
    import torch
    import torch.nn as nn

    quantize = _build_quantizer(torch)

    class _Fp8ScaledLinear(nn.Module):

        weight: Any
        weight_scale: Any
        lora_a: Any
        lora_b: Any
        _cozy_w8a8_linear = True

        def __init__(self, in_features: int, out_features: int, *,
                     bias: bool, compute_dtype: Any,
                     static_input_scale: bool,
                     gemm_mode: str = "rowwise") -> None:
            super().__init__()
            if gemm_mode not in ("rowwise", "pertensor"):
                raise ValueError(f"invalid gemm_mode {gemm_mode!r}")
            self.gemm_mode = gemm_mode
            self.compute_dtype = compute_dtype
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
            self.register_buffer("lora_a", None, persistent=False)
            self.register_buffer("lora_b", None, persistent=False)

        def _lora_addend(self, x2: Any) -> Any:
            return (x2 @ self.lora_a.t()) @ self.lora_b.t()

        def forward(self, x: Any) -> Any:
            shape = x.shape
            x2 = x.reshape(-1, self.in_features).contiguous()
            pertensor = self.gemm_mode == "pertensor"
            xq, sa = quantize(x2, pertensor, self.input_scale)
            scaled_mm: Any = torch._scaled_mm
            if pertensor:
                y = scaled_mm(
                    xq, self.weight.t(), scale_a=sa,
                    scale_b=torch.ones_like(sa), out_dtype=x.dtype,
                )
                y = y * self.weight_scale.t().to(y.dtype)
                if self.bias is not None:
                    y = y + self.bias
            else:
                y = scaled_mm(
                    xq, self.weight.t(), scale_a=sa,
                    scale_b=self.weight_scale.t(),
                    bias=self.bias, out_dtype=x.dtype,
                )
            if self.lora_a is not None:
                y = y + self._lora_addend(x2)
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, "
                    f"bias={self.bias is not None}, "
                    f"gemm_mode={self.gemm_mode}, "
                    f"static_input_scale={self.input_scale is not None}")

    return _Fp8ScaledLinear


@functools.lru_cache(maxsize=1)
def fp8_scaled_linear_class() -> type:
    return _build_module_class()


def _scale_2d(scale: Any, out_features: int) -> Any:
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


# `cozy.fp8-rowwise@1` carries the per-output-row F32 `weight_scale` and the
# BF16 passthrough as its own conventions, so the element/scale axes that used
# to be declared here are the HANDLE now.
#
# ONE convention it states that this decoder does NOT hold to: the rule's
# `activation_scale` is "dynamic (no stored input_scale)", and its description
# calls a tree carrying `input_scale` "a different, statically calibrated
# layout". This loader DECODES that leaf — `Fp8ScaledLinear` registers the
# buffer and the GEMM reads it — so the statically-calibrated variant is
# readable here and NAMEABLE NOWHERE: no ratified rule describes it, and the
# side axis that used to say so (`SCALE_STATIC_ACTIVATION`) is gone with the
# v1 vocabulary. Closing that needs its own `spec/v2/rules/` document, not a
# widening of this one.
@implements_quant_rule(
    rule="cozy.fp8-rowwise@1",
    serves=("fp8-w8a8-dynamic",),
    composes_lora=True,
    why="gw#547: Fp8ScaledLinear reads lora_a/lora_b non-persistent buffers "
        "in its own forward, so the w8a8 lane composes runtime adapters "
        "natively.",
)
def load_w8a8_denoiser(root: Path, art: W8a8Artifact, *,
                       compute_dtype: Any = None, mode: str = "",
                       cls: Any = None) -> Any:
    """Materialize the quantized denoiser: skeleton on meta, quantized Linears swapped for Fp8ScaledLinear, tensors assigned from the shards."""
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights

    compute = compute_dtype or torch.bfloat16
    if mode not in ("rowwise", "pertensor", "dequant"):
        mode = w8a8_gemm_mode() or "dequant"

    if cls is None:
        cls = _denoiser_class(root, art.component)
    cfg = dict(cls.load_config(str(root / art.component)))
    cfg.pop("quantization_config", None)
    with init_empty_weights():
        model = cls.from_config(cfg)

    sd: Dict[str, Any] = {}
    for f in art.files:
        sd.update(load_state_dict(f, why=_TENSOR_WHY))

    lin_cls = fp8_scaled_linear_class()
    swapped = 0
    for name in art.quantized:
        w = sd[f"{name}.weight"]
        scale = sd[f"{name}.weight_scale"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        eligible = (mode != "dequant"
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
                          compute_dtype=compute, static_input_scale=has_static,
                          gemm_mode=mode)
            setattr(parent, leaf, new)
            sd[f"{name}.weight_scale"] = _scale_2d(scale, out_f)
            if has_static:
                sd[f"{name}.input_scale"] = (
                    sd[f"{name}.input_scale"].float().reshape(1, 1))
            swapped += 1
        else:
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
    """Build the pipeline with EVERY quantized denoiser wired in (svdq-style component injection)."""
    import torch

    compute = compute_dtype or torch.bfloat16
    mode = w8a8_gemm_mode() or "dequant"
    kwargs: Dict[str, Any] = dict(components or {})
    arts = detect_w8a8_artifacts(Path(path)) or (art,)
    if not any(a.component == art.component for a in arts):
        arts = arts + (art,)
    for a in arts:
        kwargs[a.component] = load_w8a8_denoiser(
            path, a, compute_dtype=compute, mode=mode)
    if len(arts) > 1:
        logger.info("w8a8: %d quantized denoisers wired (%s)", len(arts),
                    ", ".join(a.component for a in arts))
    pipe = cls.from_pretrained(
        str(third_party_dir(path, why="w8a8 quantizer from_pretrained")),
        torch_dtype=compute, **kwargs)
    try:
        pipe._cozy_weight_lane = "w8a8" if mode != "dequant" else "bf16-resident"
    except Exception:
        pass
    if fp8_text_encoders:
        from ..component_vocab import text_encoder_components
        from .loading import apply_fp8_storage

        targets = tuple(
            n for n in text_encoder_components()
            if hasattr(getattr(pipe, n, None), "parameters"))
        if targets:
            apply_fp8_storage(pipe, compute_dtype=compute, components=targets)
        else:
            logger.warning(
                "w8a8: storage_dtype=fp8+te requested but %s has no text "
                "encoders; serving without TE windows", type(pipe).__name__)
    return pipe


def sanitize_w8a8_state_dict(
    state_dict: Dict[str, Any], compute_dtype: Any = None,
) -> Dict[str, Any]:
    """Dequantize w8a8 tensors in a raw state dict: fp8 weights with a ``weight_scale`` twin become compute-dtype weights, scale tensors drop."""
    import torch

    compute = compute_dtype or torch.bfloat16
    out: Dict[str, Any] = {}
    for key, t in state_dict.items():
        if key.endswith((".weight_scale", ".input_scale")):
            continue
        if (key.endswith(".weight") and isinstance(t, torch.Tensor)
                and t.dtype == torch.float8_e4m3fn
                and f"{key[: -len('.weight')]}.weight_scale" in state_dict):
            layer = key[: -len(".weight")]
            scale = _scale_2d(state_dict[f"{layer}.weight_scale"], int(t.shape[0]))
            out[key] = (t.float() * scale).to(compute)
        else:
            out[key] = t
    return out


def swap_w8a8_linears(
    model: Any,
    art: W8a8Artifact,
    *,
    compute_dtype: Any = None,
    key_map: Optional[Any] = None,
    gemm_mode: str = "rowwise",
) -> int:
    """Swap the artifact's quantized Linears in an ALREADY-CONSTRUCTED model onto :class:`Fp8ScaledLinear` in ``gemm_mode``, assigning fp8 weight + scale from the artifact shards (whatever the constructin..."""
    import torch
    import torch.nn as nn

    compute = compute_dtype or torch.bfloat16
    lin_cls = fp8_scaled_linear_class()
    where: Dict[str, Path] = {}
    for f in art.files:
        for name in _read_header(f):
            if name != "__metadata__":
                where[name] = f
    stack = ExitStack()
    handles: Dict[Path, Any] = {}

    def _tensor(name: str) -> Any:
        src = where.get(name)
        if src is None:
            raise W8a8SnapshotError(f"artifact tensor {name!r} missing from shards")
        fh = handles.get(src)
        if fh is None:
            fh = handles[src] = stack.enter_context(
                open_tensor_source(src, why=_TENSOR_WHY)
            )
        return fh.get_tensor(name)

    swapped = 0
    skipped: Dict[str, int] = {}
    samples: List[str] = []

    def _skip(cls: str, target: str) -> None:
        skipped[cls] = skipped.get(cls, 0) + 1
        if len(samples) < 3:
            samples.append(f"{target}({cls})")

    try:
        for layer in art.quantized:
            target = str(key_map(layer)) if key_map is not None else layer
            parent_path, _, leaf = target.rpartition(".")
            try:
                parent = (model.get_submodule(parent_path)
                          if parent_path else model)
                old = getattr(parent, leaf)
            except AttributeError as exc:
                raise W8a8SnapshotError(
                    f"quantized layer {layer!r} has no module {target!r} in "
                    f"{type(model).__name__} — wrong key_map?") from exc
            if not isinstance(old, nn.Linear) or type(old) is not nn.Linear:
                logger.warning(
                    "w8a8 swap: %s is %s, not a plain Linear; layer stays "
                    "dequantized", target, type(old).__name__)
                _skip("not_plain_linear", target)
                continue
            w = _tensor(f"{layer}.weight")
            out_f, in_f = int(w.shape[0]), int(w.shape[1])
            if (out_f, in_f) != (int(old.out_features), int(old.in_features)):
                raise W8a8SnapshotError(
                    f"quantized layer {layer!r} shape [{out_f}, {in_f}] != "
                    f"module {target!r} [{old.out_features}, {old.in_features}]")
            if in_f % _DIM_ALIGN or out_f % _DIM_ALIGN:
                _skip("dim_unaligned", target)
                continue
            has_static = f"{layer}.input_scale" in where
            dev = old.weight.device
            new = lin_cls(in_f, out_f, bias=old.bias is not None,
                          compute_dtype=compute, static_input_scale=has_static,
                          gemm_mode=gemm_mode)
            new.weight = w.contiguous().to(dev)
            new.weight_scale = _scale_2d(_tensor(f"{layer}.weight_scale"),
                                         out_f).to(dev)
            if has_static:
                new.input_scale = (
                    _tensor(f"{layer}.input_scale").float().reshape(1, 1).to(dev))
            if old.bias is not None:
                new.bias = nn.Parameter(
                    old.bias.detach().to(compute), requires_grad=False)
            setattr(parent, leaf, new)
            swapped += 1
    finally:
        stack.close()
    logger.info("w8a8 swap: %d/%d quantized Linears on scaled_mm",
                swapped, len(art.quantized))
    if skipped:
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"model={type(model).__name__}: {swapped}/{len(art.quantized)} "
            f"quantized Linears landed on scaled_mm; "
            f"{sum(skipped.values())} stayed DEQUANTIZED at full precision "
            f"({', '.join(f'{k}={v}' for k, v in sorted(skipped.items()))}; "
            f"e.g. {', '.join(samples)}). This pipeline reports the w8a8 lane "
            f"while part of it serves bf16 — over its budgeted VRAM, and "
            f"slower than the lane promises",
            phase="w8a8_partial_swap",
        )
    return swapped


def _root_denoiser(pipe: Any) -> Any:
    import torch.nn as nn

    for name in denoiser_components():
        mod = getattr(pipe, name, None)
        if isinstance(mod, nn.Module):
            return mod
    if isinstance(pipe, nn.Module):
        return pipe
    raise W8a8SnapshotError(
        f"{type(pipe).__name__} exposes no denoiser module "
        "(transformer/unet/dit) for the root w8a8 lane")


def load_w8a8_root_pipeline(
    cls: Any, path: Path, art: W8a8Artifact, *, compute_dtype: Any = None,
) -> Any:
    """Serve a root-layout w8a8 snapshot through the pipeline class's own ``from_pretrained`` (which must sanitize — see module docstring), then swap the denoiser's quantized Linears onto scaled_mm when t..."""
    import torch

    compute = compute_dtype or torch.bfloat16
    mode = w8a8_gemm_mode() or "dequant"
    pipe = cls.from_pretrained(
        str(third_party_dir(path, why="w8a8 quantizer from_pretrained")),
        torch_dtype=compute)
    denoiser = _root_denoiser(pipe)
    key_map = (getattr(pipe, "_cozy_w8a8_key_map", None)
               or getattr(cls, "_cozy_w8a8_key_map", None))
    if mode != "dequant":
        if not swap_w8a8_linears(denoiser, art, compute_dtype=compute,
                                 key_map=key_map, gemm_mode=mode):
            raise W8a8SnapshotError(
                "scaled_mm host but no quantized Linear swapped — artifact "
                f"keys do not match {type(denoiser).__name__} modules")
    try:
        denoiser._cozy_w8a8_mode = mode
        pipe._cozy_weight_lane = (
            "w8a8" if mode != "dequant" else "bf16-resident")
    except Exception:
        pass
    logger.info("w8a8 root lane: %s (%d quantized layers, component root)",
                mode, len(art.quantized))
    return pipe


def quantize_tree_w8a8(
    src_tree: Path,
    out_tree: Path,
    *,
    exclude: tuple[str, ...] = ("embed", "norm"),
) -> Path:
    """Copy a diffusers tree, rewriting the denoiser's eligible 2D weights to fp8 + per-out-channel ``weight_scale`` per the module contract."""

    import torch
    from safetensors.torch import save_file

    src_tree, out_tree = Path(src_tree), Path(out_tree)
    comp = next((c for c in denoiser_components() if (src_tree / c).is_dir()), None)
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
        tensors = load_state_dict(f, why="the w8a8 data-free producer reads a source shard")
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
    "detect_w8a8_artifacts",
    "fp8_scaled_linear_class",
    "load_w8a8_denoiser",
    "load_w8a8_pipeline",
    "load_w8a8_root_pipeline",
    "quantize_tree_w8a8",
    "sanitize_w8a8_state_dict",
    "w8a8_gemm_mode",
    "swap_w8a8_linears",
]
