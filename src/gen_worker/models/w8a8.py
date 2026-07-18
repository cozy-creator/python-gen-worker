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
activation quant by default, the static scale when present. TWO dispatch
branches over the ONE artifact, chosen once at load by :func:`w8a8_gemm_mode`
(gw#564): "rowwise" (scale vectors inside the GEMM — CUTLASS fast, sm_90+)
and "pertensor" (scalar-scaled cuBLASLt GEMM + per-channel epilogue rescale —
the Ada/sm_89 fast path; torch's rowwise kernels fall back to ~half rate
there, ie#498). Hosts where no branch wins the load-time micro-benchmark
DEQUANT once at load into plain bf16-resident weights — same numerics, no
speed win, never a refusal.
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
# torch's fast ROWWISE-scaled kernels are CUTLASS sm_90+; sm_89 runs rowwise
# on a ~half-rate fallback (ie#498) and takes the pertensor branch instead.
W8A8_ROWWISE_MIN_SM = 90
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
    # Denoiser dir name ("transformer"/"unet") for diffusers trees; "" for a
    # root-layout snapshot (singlefile/sharded-transformers, gw#562) whose
    # weight set IS the denoiser.
    component: str
    files: tuple[Path, ...]   # the weight set's safetensors shards
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
    """Root weight set: index-mapped shards plus loose safetensors, at the
    root and under nested dirs (split-checkpoint layouts keep component
    files under subdirs). Hidden dirs (``.cache`` download metadata) are
    skipped."""
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


def detect_w8a8_artifact(model_path: Path) -> Optional[W8a8Artifact]:
    """Header-sniff a snapshot for the w8a8 contract: any denoiser layer with
    an F8_E4M3 ``weight`` AND a ``weight_scale`` twin. A scale-FREE fp8 tree
    (the storage-cast ``#fp8`` flavor) never matches — the scales are the
    distinguisher. Diffusers trees scan the denoiser component dirs;
    everything else scans the root weight set (singlefile/sharded-
    transformers layouts, gw#562). Cheap: header reads only."""
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
                return W8a8Artifact(
                    component=comp, files=files, quantized=quantized,
                    static_input_scales=static,
                )
        return None
    files = _root_weight_files(root)
    if files:
        quantized, static = _quantized_layers(files)
        if quantized:
            return W8a8Artifact(
                component="", files=files, quantized=quantized,
                static_input_scales=static,
            )
    return None


def _probe_scales(mode: str, m: int, n: int, device: str = "cuda") -> tuple:
    """(scale_a, scale_b) for one ``torch._scaled_mm`` call in ``mode``:
    rowwise = [m,1]x[1,n] vectors consumed inside the GEMM; pertensor =
    scalar [1,1]x[1,1] (the cuBLASLt tensorwise path — per-channel weight
    scales are applied OUTSIDE as the epilogue rescale, gw#564)."""
    import torch

    if mode == "rowwise":
        return (torch.ones(m, 1, device=device),
                torch.ones(1, n, device=device))
    return (torch.ones(1, 1, device=device),
            torch.ones(1, 1, device=device))


def _gemm_call_ok(mode: str) -> bool:
    """One tiny kernel call in ``mode``'s scale shape — torch's scaled-mm
    backends vary by (torch, cuda, arch), so trusting a version table
    instead of the device is how mixed fleets break."""
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


_BENCH_DIM = 4096      # square GEMM; big enough that tensor-core rate dominates
_BENCH_WARMUP = 3
_BENCH_ITERS = 10
# A candidate arms only when the fp8 GEMM beats the bf16 kernel by a real
# margin. Genuinely fast paths measure ~1.5-2x here; the sm_89 rowwise
# fallback measures ~0.5x (ie#498's +79% compiled loss) — 1.10 separates
# them with headroom against timer noise.
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
    """(fp8_ms, bf16_ms) medians for one representative square GEMM. The
    activations are PRE-quantized — dynamic-quant overhead fuses away under
    inductor (verdicts are on compiled arms), so the gate times the KERNEL,
    which is exactly what ie#498 showed can silently be a half-rate
    fallback. The pertensor arm includes its per-channel epilogue multiply."""
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
    """Load-time micro-benchmark gate (gw#564): the probe must verify the
    fast path ENGAGES, not that the call succeeds — ie#498 measured sm_89
    rowwise probe-pass at ~half the bf16 rate (+79% compiled)."""
    fp8_ms, bf16_ms = _bench_gemm_pair(mode)
    speedup = bf16_ms / max(fp8_ms, 1e-9)
    logger.info(
        "w8a8 gemm gate: mode=%s fp8=%.3fms bf16=%.3fms speedup=%.2fx "
        "(min %.2fx)", mode, fp8_ms, bf16_ms, speedup, _BENCH_MIN_SPEEDUP)
    return speedup >= _BENCH_MIN_SPEEDUP


def _choose_gemm_mode(sm: int) -> str:
    """Candidate branches in preference order for this SKU class; each must
    both CALL successfully and WIN the micro-benchmark to arm. sm_90+ keeps
    the shipped rowwise-in-GEMM lane first; sm_89 (Ada: 4090/L40S) prefers
    the per-tensor cuBLASLt path + epilogue rescale. The non-preferred
    candidate stays listed so a future stack that flips the kernel story
    arms the right branch without a code change — the bench decides."""
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
    return ""


@functools.lru_cache(maxsize=1)
def w8a8_gemm_mode() -> str:
    """The fp8 GEMM dispatch for THIS device, chosen once per process:
    ``"rowwise"`` (scale vectors consumed inside ``_scaled_mm`` — CUTLASS
    fast kernels, sm_90+), ``"pertensor"`` (scalar-scaled cuBLASLt GEMM +
    per-channel epilogue rescale — the Ada/sm_89 fast path, gw#564), or
    ``""`` (dequant lane). Live-probed AND micro-benchmarked: a branch arms
    only when its kernel call succeeds and beats the bf16 reference —
    probe-pass ≠ profitable (ie#498)."""
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        return ""
    major, minor = torch.cuda.get_device_capability()
    return _choose_gemm_mode(major * 10 + minor)


def _build_module_class() -> type:
    """Define the nn.Module lazily so importing this module never needs
    torch (discovery/CPU tools)."""
    import torch
    import torch.nn as nn

    class _Fp8ScaledLinear(nn.Module):
        """fp8 weights RESIDENT; y = scaled_mm(quant(x), W^T) + bias.

        ``weight_scale`` is the [out, 1] dequant multiplier (per-tensor
        scalars are expanded at load). Two GEMM dispatch branches, chosen
        ONCE at load by SKU (gw#564; ONE weight artifact serves both):

        - ``gemm_mode="rowwise"`` (sm_90+): scale vectors consumed inside
          ``_scaled_mm`` (per-row dynamic activation scale x per-channel
          weight scale), bias fused into the GEMM.
        - ``gemm_mode="pertensor"`` (sm_89 Ada): scalar-scaled GEMM (the
          cuBLASLt fast path) with the SAME per-channel ``weight_scale``
          applied as a post-GEMM column-multiply epilogue (one broadcast
          multiply — fuses under inductor), bias added after the rescale.
          Activation scale is per-TENSOR dynamic. Mathematically identical
          to rowwise up to activation-quant granularity + one bf16 rounding.

        Activation quant is DYNAMIC (amax/448) unless a static
        ``input_scale`` was calibrated. NOTE: never ``.to(dtype=...)`` this
        module — a dtype cast would upcast the fp8 buffer (device moves are
        fine).

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
                     static_input_scale: bool,
                     gemm_mode: str = "rowwise") -> None:
            super().__init__()
            if gemm_mode not in ("rowwise", "pertensor"):
                raise ValueError(f"invalid gemm_mode {gemm_mode!r}")
            self.gemm_mode = gemm_mode
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
            pertensor = self.gemm_mode == "pertensor"
            if self.input_scale is not None:
                # Static scales are per-tensor scalars already — the
                # pertensor GEMM consumes [1,1] directly.
                sa = (self.input_scale if pertensor else
                      self.input_scale.expand(x2.shape[0], 1).contiguous())
            elif pertensor:
                sa = (x2.abs().amax().float()
                      / _FP8_MAX).clamp(min=1e-12).reshape(1, 1)
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
            if pertensor:
                # Scalar-scaled GEMM (cuBLASLt's Ada fast path, gw#564); the
                # per-channel weight_scale rides as a post-GEMM column
                # multiply — bias joins AFTER the rescale, never inside.
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
    ``mode`` "rowwise" | "pertensor" | "dequant" (default: probe). Layers
    whose dims break scaled_mm's 16-alignment are dequantized individually."""
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    compute = compute_dtype or torch.bfloat16
    if mode not in ("rowwise", "pertensor", "dequant"):
        mode = w8a8_gemm_mode() or "dequant"

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
    mode = w8a8_gemm_mode() or "dequant"
    denoiser = load_w8a8_denoiser(
        path, art, compute_dtype=compute, mode=mode)
    kwargs: Dict[str, Any] = dict(components or {})
    kwargs[art.component] = denoiser
    pipe = cls.from_pretrained(str(path), torch_dtype=compute, **kwargs)
    try:
        pipe._cozy_weight_lane = "w8a8" if mode != "dequant" else "bf16-resident"
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
# Root-layout serve path (gw#562): pipelines the worker does not assemble
# (DiffSynth families — hidream-o1's sharded-transformers root, anima's
# split checkpoint). The pipeline class's OWN from_pretrained constructs the
# model; its loader runs sanitize_w8a8_state_dict so a quantized snapshot
# lands correct dequantized weights on ANY host, and scaled_mm hosts then
# swap the quantized Linears onto fp8 GEMM in place.
# ---------------------------------------------------------------------------


def sanitize_w8a8_state_dict(
    state_dict: Dict[str, Any], compute_dtype: Any = None,
) -> Dict[str, Any]:
    """Dequantize w8a8 tensors in a raw state dict: fp8 weights with a
    ``weight_scale`` twin become compute-dtype weights, scale tensors drop.
    A scale-free dict passes through unchanged, so loaders that read
    snapshots manually can call it unconditionally. Never ``.to(dtype)`` an
    fp8 tensor before this — an unscaled upcast is silent garbage."""
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
    """Swap the artifact's quantized Linears in an ALREADY-CONSTRUCTED model
    onto :class:`Fp8ScaledLinear` in ``gemm_mode``, assigning fp8 weight +
    scale from the artifact shards (whatever the constructing loader
    materialized there is replaced). ``key_map`` maps an artifact layer name
    to its module path (identity default; converter-renamed families pass
    their own — e.g. anima's DiffSynth converter strips ``net.``). Layers
    whose dims break scaled_mm's 16-alignment, or whose owner is not a plain
    Linear, keep their dequantized weights. Returns the number of swapped
    modules."""
    import torch
    import torch.nn as nn
    from safetensors import safe_open

    compute = compute_dtype or torch.bfloat16
    lin_cls = fp8_scaled_linear_class()
    where: Dict[str, Path] = {}
    for f in art.files:
        for name in _read_header(f):
            if name != "__metadata__":
                where[name] = f
    handles: Dict[Path, Any] = {}

    def _tensor(name: str) -> Any:
        src = where.get(name)
        if src is None:
            raise W8a8SnapshotError(f"artifact tensor {name!r} missing from shards")
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
                raise W8a8SnapshotError(
                    f"quantized layer {layer!r} has no module {target!r} in "
                    f"{type(model).__name__} — wrong key_map?") from exc
            if not isinstance(old, nn.Linear) or type(old) is not nn.Linear:
                logger.warning(
                    "w8a8 swap: %s is %s, not a plain Linear; layer stays "
                    "dequantized", target, type(old).__name__)
                continue
            w = _tensor(f"{layer}.weight")
            out_f, in_f = int(w.shape[0]), int(w.shape[1])
            if (out_f, in_f) != (int(old.out_features), int(old.in_features)):
                raise W8a8SnapshotError(
                    f"quantized layer {layer!r} shape [{out_f}, {in_f}] != "
                    f"module {target!r} [{old.out_features}, {old.in_features}]")
            if in_f % _DIM_ALIGN or out_f % _DIM_ALIGN:
                continue  # scaled_mm alignment; dequant weights stay
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
        for fh in handles.values():
            try:
                fh.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
    logger.info("w8a8 swap: %d/%d quantized Linears on scaled_mm",
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
    raise W8a8SnapshotError(
        f"{type(pipe).__name__} exposes no denoiser module "
        "(transformer/unet/dit) for the root w8a8 lane")


def load_w8a8_root_pipeline(
    cls: Any, path: Path, art: W8a8Artifact, *, compute_dtype: Any = None,
) -> Any:
    """Serve a root-layout w8a8 snapshot through the pipeline class's own
    ``from_pretrained`` (which must sanitize — see module docstring), then
    swap the denoiser's quantized Linears onto scaled_mm when the host
    qualifies. Stamps ``_cozy_weight_lane`` like the diffusers lane.

    Converter-renamed families declare ``_cozy_w8a8_key_map`` on the
    pipeline class (a ``staticmethod`` mapping an artifact layer name to
    its module path) — forwarded to :func:`swap_w8a8_linears`."""
    import torch

    compute = compute_dtype or torch.bfloat16
    mode = w8a8_gemm_mode() or "dequant"
    pipe = cls.from_pretrained(str(path), torch_dtype=compute)
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
    "load_w8a8_root_pipeline",
    "quantize_tree_w8a8",
    "sanitize_w8a8_state_dict",
    "w8a8_gemm_mode",
    "swap_w8a8_linears",
]
