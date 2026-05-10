"""Inline conversion dispatch for clone_huggingface / clone_civitai (issue #73).

When the user requests an output dtype the source repo doesn't ship, the
clone path runs the conversion in-process — the same library code paths
other worker functions can call. The clone reuses its existing upload
session so all flavors land atomically under the same destination tag group.

Three buckets:

1. **Direct ingest** — caller decides this *before* calling here, by
   matching ``target_dtype`` against what the classifier saw in the source
   files. No conversion needed.

2. **Inline-supported** — weight-only schemes that don't need calibration:
   - ``bf16`` / ``fp16`` / ``fp32`` (streaming dtype cast)
   - ``fp8:e4m3`` / ``fp8:e5m2`` / ``int8`` / ``int4`` (torchao weight-only)
   - GGUF quants (``q4_k_m``, ``q8_0``, …) via convert_hf_to_gguf + llama-quantize

3. **Calibration-required** — raise ``InlineConversionNotPossible`` with a
   clear structured refusal:
   - ``int4:awq`` / ``int4:gptq`` (modelopt + dataset)
   - ``nvfp4`` calibrated (modelopt + dataset)
   - ``int8:awq`` / ``int8:gptq``

The exception carries structured requirements so callers can render their own
guidance without this worker package knowing about any particular published
endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeferredConversionRequirement:
    """Structured follow-up requirement for work the clone path will not run.

    This deliberately avoids endpoint names, operator commands, and tenant
    function names. Those are deployment choices owned outside the gen-worker
    library.
    """

    kind: str
    target_dtype: str
    scheme: str = ""
    requires_calibration: bool = False
    requires_gpu: bool = False
    runtime: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "target_dtype": self.target_dtype,
        }
        if self.scheme:
            out["scheme"] = self.scheme
        if self.requires_calibration:
            out["requires_calibration"] = True
        if self.requires_gpu:
            out["requires_gpu"] = True
        if self.runtime:
            out["runtime"] = self.runtime
        return out


class InlineConversionNotPossible(Exception):
    """Requested target dtype can't be produced inline by the clone path.

    Carries an optional structured ``deferred_requirement`` the caller can
    render using its own product surface.
    """

    def __init__(
        self,
        reason: str,
        *,
        target_dtype: str = "",
        deferred_requirement: DeferredConversionRequirement | None = None,
    ) -> None:
        self.reason = str(reason or "").strip()
        self.target_dtype = str(target_dtype or "").strip().lower()
        self.deferred_requirement = deferred_requirement
        super().__init__(self.reason)


# ---------------------------------------------------------------------------
# Dispatch policy
# ---------------------------------------------------------------------------

# Weight-only quants we can run inline today. These don't need a calibration
# dataset — torchao fp8_wo / int8_wo derive scales per-tensor from the
# weight itself. (int4 routes through bitsandbytes nf4 on CPU because
# torchao's Int4Tensor requires fbgemm_gpu + CUDA.)
_INLINE_TORCHAO_SCHEMES: dict[str, str] = {
    "fp8:e4m3":      "fp8_wo",
    "fp8:e5m2":      "fp8_wo",        # torchao uses one fp8 wo path; e5m2 falls back to e4m3 for now
    "fp8":           "fp8_wo",
    "int8":          "int8_wo",
}

# bitsandbytes 4-bit weight-only: nf4 / fp4. Works on CPU (the quantization
# pass + save_pretrained run end-to-end without CUDA; LLM.int8() is the only
# bnb path that genuinely requires CUDA). Loads via transformers'
# BitsAndBytesConfig integration which is mature and stable.
_INLINE_BNB_SCHEMES: dict[str, str] = {
    "nf4":           "nf4",
    "fp4":           "fp4",
    "int4":          "nf4",   # plain "int4" → nf4 (bnb's default 4-bit)
    "int4:nf4":      "nf4",
    "int4:fp4":      "fp4",
}

# Pure dtype casts that go through the streaming primitive — no HF model load.
_INLINE_CAST_DTYPES: frozenset[str] = frozenset({
    "bf16", "fp16", "fp32",
    "f16", "f32",  # GGUF spelling; treat as fp16/fp32 for safetensors output
})

# GGUF quants — run via convert_hf_to_gguf.py (+ optional llama-quantize).
_INLINE_GGUF_ENCODINGS: frozenset[str] = frozenset({
    "f32", "f16", "bf16", "q8_0",
    "q6_k", "q6_k_l",
    "q5_k_m", "q5_k_s", "q5_k_l", "q5_0", "q5_1",
    "q4_k_m", "q4_k_s", "q4_k_l", "q4_0", "q4_1",
    "q3_k_m", "q3_k_s", "q3_k_l", "q3_k_xl",
    "q2_k",
})

# Calibration-required quants. We refuse these inline — they need a calibration
# dataset and (for some) a GPU; running them silently as part of clone would
# either hang or produce garbage. The caller renders any product-specific
# follow-up guidance from the structured requirement.
_CALIBRATED_DTYPES: dict[str, DeferredConversionRequirement] = {
    "int4:awq": DeferredConversionRequirement(
        kind="calibrated_quantization",
        target_dtype="int4:awq",
        scheme="int4_awq",
        requires_calibration=True,
        requires_gpu=True,
        runtime="modelopt",
    ),
    "int4:gptq": DeferredConversionRequirement(
        kind="calibrated_quantization",
        target_dtype="int4:gptq",
        scheme="int4_gptq",
        requires_calibration=True,
        requires_gpu=True,
        runtime="modelopt",
    ),
    "int8:awq": DeferredConversionRequirement(
        kind="calibrated_quantization",
        target_dtype="int8:awq",
        scheme="int8_awq",
        requires_calibration=True,
        requires_gpu=True,
        runtime="modelopt",
    ),
    "int8:gptq": DeferredConversionRequirement(
        kind="calibrated_quantization",
        target_dtype="int8:gptq",
        scheme="int8_gptq",
        requires_calibration=True,
        requires_gpu=True,
        runtime="modelopt",
    ),
    "nvfp4": DeferredConversionRequirement(
        kind="calibrated_quantization",
        target_dtype="nvfp4",
        scheme="nvfp4",
        requires_calibration=True,
        requires_gpu=True,
        runtime="modelopt",
    ),
}


def is_calibration_required(target_dtype: str) -> bool:
    """True if the target dtype needs a calibration dataset.

    These dtypes require a calibration corpus and (often) GPU; the clone
    path refuses them and returns structured follow-up requirements.
    """
    return _normalize(target_dtype) in _CALIBRATED_DTYPES


def is_inline_supported(target_dtype: str, *, target_file_type: str = "safetensors") -> bool:
    """True if ``target_dtype`` can be produced inline given the desired file type."""
    dtype = _normalize(target_dtype)
    ftype = _normalize(target_file_type) or "safetensors"
    if dtype in _CALIBRATED_DTYPES:
        return False
    if ftype == "gguf":
        return dtype in _INLINE_GGUF_ENCODINGS
    if dtype in _INLINE_CAST_DTYPES:
        return True
    if dtype in _INLINE_TORCHAO_SCHEMES:
        return True
    if dtype in _INLINE_BNB_SCHEMES:
        return True
    return False


def deferred_conversion_requirement(target_dtype: str) -> DeferredConversionRequirement | None:
    """Return structured follow-up requirements for dtypes refused inline."""
    return _CALIBRATED_DTYPES.get(_normalize(target_dtype))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InlineConversionResult:
    """Result of an inline conversion pass.

    Mirrors the shape of streaming_dtype_cast / streaming_nvfp4_quantize so
    callers can promote ``output_paths`` + ``index_path`` to CAS the same
    way they handle the existing per-component conversion jobs.
    """

    output_paths: list[Path] = field(default_factory=list)
    index_path: Path | None = None
    target_dtype: str = ""
    target_file_type: str = "safetensors"
    attributes: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_inline_conversion(
    *,
    source_path: Path,
    out_dir: Path,
    target_dtype: str,
    target_file_type: str = "safetensors",
    component_name: str = "",
    shard_prefix: str = "model",
    source_repo_dir: Path | None = None,
) -> InlineConversionResult:
    """Run the appropriate inline conversion for the requested target_dtype.

    ``source_path`` is the materialized input — for cast / torchao paths
    this is the safetensors file (or .index.json) we'll read; for GGUF
    this is also the safetensors file (we look up sidecar configs via
    ``source_repo_dir``).

    ``source_repo_dir`` is the parent component directory containing
    ``config.json`` and tokenizer assets — used by the GGUF path and by
    the torchao path (which loads the component as an HF model). When
    omitted we fall back to ``source_path.parent``.

    Raises ``InlineConversionNotPossible`` for calibrated dtypes; the
    exception carries structured requirements the caller can render to the
    user.
    """
    dtype = _normalize(target_dtype)
    ftype = _normalize(target_file_type) or "safetensors"
    if dtype == "":
        raise ValueError("inline_convert: target_dtype is required")

    # Calibration-required quants — refuse cleanly with a suggested job.
    if dtype in _CALIBRATED_DTYPES:
        raise InlineConversionNotPossible(
            reason=(
                f"requested {dtype} needs a calibration dataset; the clone "
                f"path doesn't run dataset-dependent quantization inline"
            ),
            target_dtype=dtype,
            deferred_requirement=deferred_conversion_requirement(dtype),
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GGUF target — convert_hf_to_gguf.py (+ optional llama-quantize) regardless of source dtype.
    if ftype == "gguf":
        return _run_gguf_inline(
            source_path=source_path,
            source_repo_dir=source_repo_dir or source_path.parent,
            out_dir=out_dir,
            encoding=dtype,
        )

    # Pure streaming dtype cast — bf16/fp16/fp32 (and the GGUF spelling f16/f32).
    if dtype in _INLINE_CAST_DTYPES:
        return _run_cast_inline(
            source_path=source_path,
            out_dir=out_dir,
            target_dtype=dtype,
            shard_prefix=shard_prefix,
        )

    # Torchao weight-only quantization — fp8 / int8. Loads the component as
    # an HF model and saves via flatten_tensor_state_dict + safetensors.
    if dtype in _INLINE_TORCHAO_SCHEMES:
        return _run_torchao_inline(
            source_path=source_path,
            source_repo_dir=source_repo_dir or source_path.parent,
            out_dir=out_dir,
            target_dtype=dtype,
            component_name=component_name,
        )

    # bitsandbytes nf4 / fp4 — runs on CPU for the quant pass; save_pretrained
    # handles Params4bit transparently (no flatten helper needed). The fast
    # path for "I want a 4-bit model on a laptop" use case.
    if dtype in _INLINE_BNB_SCHEMES:
        return _run_bnb_inline(
            source_path=source_path,
            source_repo_dir=source_repo_dir or source_path.parent,
            out_dir=out_dir,
            target_dtype=dtype,
        )

    # Fallthrough: dtype is not in any known bucket.
    raise InlineConversionNotPossible(
        reason=f"target dtype {dtype!r} is not recognized as a runnable inline conversion",
        target_dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Bucket implementations
# ---------------------------------------------------------------------------

def _run_cast_inline(
    *,
    source_path: Path,
    out_dir: Path,
    target_dtype: str,
    shard_prefix: str,
) -> InlineConversionResult:
    """bf16/fp16/fp32 streaming dtype cast."""
    import torch

    from .streaming_primitives import streaming_dtype_cast

    dtype = _normalize(target_dtype)
    if dtype in {"bf16"}:
        torch_dtype = torch.bfloat16
    elif dtype in {"fp16", "f16"}:
        torch_dtype = torch.float16
    elif dtype in {"fp32", "f32"}:
        torch_dtype = torch.float32
    else:
        raise InlineConversionNotPossible(
            reason=f"_run_cast_inline got unexpected dtype {dtype!r}",
            target_dtype=dtype,
        )

    result = streaming_dtype_cast(
        Path(source_path),
        Path(out_dir),
        target_dtype=torch_dtype,
        shard_prefix=shard_prefix,
    )

    output_paths = [Path(p) for p in result.get("output_paths") or []]
    index_path = result.get("index_path")
    if index_path is not None:
        index_path = Path(index_path)

    attrs = {
        "dtype": dtype,
        "file_type": "safetensors",
        "conversion_strategy": "inline_cast",
        "tensor_count": str(int(result.get("tensor_count") or 0)),
        "converted_count": str(int(result.get("converted_count") or 0)),
    }
    return InlineConversionResult(
        output_paths=output_paths,
        index_path=index_path,
        target_dtype=dtype,
        target_file_type="safetensors",
        attributes=attrs,
        summary=dict(result),
    )


def _run_torchao_inline(
    *,
    source_path: Path,
    source_repo_dir: Path,
    out_dir: Path,
    target_dtype: str,
    component_name: str,
) -> InlineConversionResult:
    """fp8/int8/int4 torchao weight-only quantization.

    Bypasses transformers' ``TorchAoConfig`` integration entirely (broken
    in transformers 4.57 + torchao 0.17 — both the ``autoquant`` import
    and the safe-serialization allowlist refuse non-fp8 schemes). Instead:

      1. Load the model in bf16 via plain ``from_pretrained`` (no quant
         config, no integration path that would trip the autoquant import).
      2. ``torchao.quantization.quantize_(model, <Config>(version=2))``
         — modern torchao API; produces tensor subclass weights.
      3. Flatten the state_dict via ``torchao.prototype.safetensors.
         safetensors_support.flatten_tensor_state_dict`` — splits each
         subclass tensor (Float8Tensor, Int8Tensor, Int4Tensor) into
         (qdata, scale, [zero_point]) plus a JSON metadata blob describing
         how to reconstruct.
      4. ``safetensors.torch.save_file(flat_data, out, metadata=meta)`` —
         standard safetensors with the per-tensor metadata stashed in the
         file header. Inference workers that call torchao's
         ``unflatten_tensor_state_dict`` reconstruct the subclass weights.
      5. Copy config.json / tokenizer files from source. Stamp
         quantization_config in config.json so transformers' integration
         picks it up on a future ``from_pretrained`` (the integration's
         load path is intact even when the save path is broken).

    For transformers (singlefile) sources this is one model load. Diffusers
    sources are deferred to phase 2 (per-component quantize).
    """
    import json as _json
    import shutil as _shutil
    import torch

    dtype = _normalize(target_dtype)
    scheme = _INLINE_TORCHAO_SCHEMES.get(dtype)
    if scheme is None:
        raise InlineConversionNotPossible(
            reason=f"_run_torchao_inline got unexpected dtype {dtype!r}",
            target_dtype=dtype,
        )

    repo_dir = Path(source_repo_dir)
    if not repo_dir.exists() or not repo_dir.is_dir():
        repo_dir = Path(source_path).parent

    is_diffusers = (repo_dir / "model_index.json").is_file()
    is_transformers = (repo_dir / "config.json").is_file() and not is_diffusers
    if not (is_diffusers or is_transformers):
        raise InlineConversionNotPossible(
            reason=(
                f"source dir at {repo_dir} has neither model_index.json "
                f"(diffusers) nor config.json (transformers) — can't pick "
                f"an inline quantization loader"
            ),
            target_dtype=dtype,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import safetensors.torch as st
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        from torchao.prototype.safetensors.safetensors_support import (
            flatten_tensor_state_dict,
        )
        from ._hf_load import load_component_module
    except ImportError as exc:
        raise InlineConversionNotPossible(
            reason=(
                f"torchao or transformers not installed — can't run inline "
                f"{dtype} quantization: {exc}"
            ),
            target_dtype=dtype,
        ) from exc

    # version=2 is required for ALL configs to produce the modern tensor
    # subclasses (Float8Tensor / Int8Tensor / Int4Tensor) that
    # flatten_tensor_state_dict knows how to serialize. v1 produces the
    # deprecated AffineQuantizedTensor which has no flatten support.
    cfg_for_scheme: dict[str, Any] = {
        "fp8_wo":      lambda: Float8WeightOnlyConfig(version=2),
        "fp8_dynamic": lambda: Float8DynamicActivationFloat8WeightConfig(),
        "int8_wo":     lambda: Int8WeightOnlyConfig(version=2),
        "int4_wo":     lambda: Int4WeightOnlyConfig(group_size=128),
    }
    cfg_factory = cfg_for_scheme.get(scheme)
    if cfg_factory is None:
        raise InlineConversionNotPossible(
            reason=f"unsupported torchao scheme: {scheme}",
            target_dtype=dtype,
        )

    # Diffusers-layout fan-out: per-component load + quantize + flatten + save.
    # vae / scheduler / tokenizer* / feature_extractor / safety_checker pass
    # through unchanged (these are config-only or precision-sensitive).
    # transformer / unet / text_encoder* / image_encoder / prior / controlnet
    # are quantized.
    if is_diffusers:
        return _run_torchao_diffusers_inline(
            repo_dir=repo_dir,
            out_dir=out_dir,
            dtype=dtype,
            scheme=scheme,
            cfg_factory=cfg_factory,
        )

    # Use load_component_module so diffusers components (FluxTransformer2DModel,
    # CLIPTextModel, etc.) and transformers causal LMs both load via their
    # canonical class. AutoModelForCausalLM was wrong here — it only handles
    # decoder-only LM checkpoints, not diffusers transformers or CLIP encoders.
    import json as _json
    cfg_path = repo_dir / "config.json"
    try:
        cfg_blob = _json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    except Exception:
        cfg_blob = {}
    try:
        model = load_component_module(
            repo_dir, cfg_blob, torch_dtype=torch.bfloat16,
        )
    except Exception as exc:
        raise InlineConversionNotPossible(
            reason=f"failed to load source model for inline {dtype}: {exc}",
            target_dtype=dtype,
        ) from exc

    cfg = cfg_factory()
    try:
        quantize_(model, cfg)
    except ImportError as exc:
        # Common case: int4 requires fbgemm_gpu + CUDA which isn't available
        # on the CPU-only build. The error message ("Requires mslk >= 1.0.0")
        # is misleading; the real cause is libc10_cuda.so missing because the
        # cpu profile doesn't bundle CUDA runtime. Surface a clear hint.
        msg = str(exc)
        if "mslk" in msg or "fbgemm" in msg.lower() or "c10_cuda" in msg.lower():
            raise InlineConversionNotPossible(
                reason=(
                    f"{dtype} quantization requires fbgemm_gpu + CUDA runtime "
                    f"(not available on this CPU-only worker). Run on a GPU "
                    f"worker, or use fp8:e4m3 / int8 which work on CPU."
                ),
                target_dtype=dtype,
            ) from exc
        raise InlineConversionNotPossible(
            reason=f"torchao quantize_ ImportError for {dtype}: {exc}",
            target_dtype=dtype,
        ) from exc
    except Exception as exc:
        raise InlineConversionNotPossible(
            reason=f"torchao quantize_ failed for {dtype}: {exc}",
            target_dtype=dtype,
        ) from exc

    state_dict = model.state_dict()
    try:
        tensors_data, sd_metadata = flatten_tensor_state_dict(state_dict)
    except Exception as exc:
        raise InlineConversionNotPossible(
            reason=(
                f"torchao flatten_tensor_state_dict failed for {dtype}: {exc} "
                f"(this is usually a torchao version-bump issue; the "
                f"same flatten helper is used by packaged conversion "
                f"entrypoints)"
            ),
            target_dtype=dtype,
        ) from exc

    # Copy non-weight source files (config.json, tokenizer, generation_config)
    # so the destination is a complete loadable repo.
    for src_file in repo_dir.iterdir():
        if not src_file.is_file():
            continue
        # Skip the source weights — we're writing replacement quantized ones.
        if src_file.suffix in {".safetensors", ".bin", ".pt", ".ckpt", ".pth"}:
            continue
        if src_file.name.endswith(".safetensors.index.json"):
            continue
        _shutil.copy2(src_file, out_dir / src_file.name)

    # Write the flattened quantized weights to a single safetensors file. The
    # subclass metadata is stashed in the safetensors header — torchao's
    # unflatten_tensor_state_dict reads it back via `f.metadata()`.
    weights_path = out_dir / "model.safetensors"
    st.save_file(tensors_data, str(weights_path), metadata=sd_metadata)

    # Stamp quantization_config in config.json so transformers' integration
    # picks up the right TorchAoConfig on a future from_pretrained. This is
    # the side-channel that lets unflatten happen automatically at load time.
    config_json_path = out_dir / "config.json"
    if config_json_path.is_file():
        try:
            cfg_data = _json.loads(config_json_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}
        cfg_data["quantization_config"] = {
            "quant_method": "torchao",
            "quant_type": _torchao_quant_type_name(scheme),
            "modules_to_not_convert": None,
            "include_input_output_embeddings": False,
            "untie_embedding_weights": False,
        }
        config_json_path.write_text(
            _json.dumps(cfg_data, indent=2), encoding="utf-8",
        )

    saved_files: list[Path] = sorted(
        f for f in out_dir.rglob("*") if f.is_file()
    )
    if not saved_files:
        raise RuntimeError(
            f"_run_torchao_inline: no files produced for {dtype} from {repo_dir}"
        )

    try:
        import torchao
        torchao_version = str(torchao.__version__)
    except Exception:
        torchao_version = ""

    attrs = {
        "dtype": dtype,
        "file_type": "safetensors",
        "conversion_strategy": "inline_torchao",
        "quant_scheme": f"torchao:{scheme}",
        "quant_library": "torchao",
    }
    if torchao_version:
        attrs["quant_library_version"] = torchao_version
    return InlineConversionResult(
        output_paths=saved_files,
        index_path=None,
        target_dtype=dtype,
        target_file_type="safetensors",
        attributes=attrs,
        summary={
            "scheme": scheme,
            "file_count": len(saved_files),
        },
    )


def _run_bnb_inline(
    *,
    source_path: Path,
    source_repo_dir: Path,
    out_dir: Path,
    target_dtype: str,
) -> InlineConversionResult:
    """bitsandbytes nf4 / fp4 weight-only quantization (CPU-friendly).

    Routes through transformers' ``BitsAndBytesConfig`` integration which is
    mature and stable (unlike the torchao path which is broken in transformers
    4.57). bitsandbytes' ``Params4bit`` is a parameter subclass that handles
    its own save/load via ``save_pretrained`` directly — no flatten helper
    needed.

    LLM.int8() (8-bit) is the only bnb path that genuinely requires CUDA;
    nf4/fp4 quantization runs end-to-end on CPU.
    """
    import torch

    dtype = _normalize(target_dtype)
    bnb_quant_type = _INLINE_BNB_SCHEMES.get(dtype)
    if bnb_quant_type is None:
        raise InlineConversionNotPossible(
            reason=f"_run_bnb_inline got unexpected dtype {dtype!r}",
            target_dtype=dtype,
        )

    repo_dir = Path(source_repo_dir)
    if not repo_dir.exists() or not repo_dir.is_dir():
        repo_dir = Path(source_path).parent

    is_diffusers = (repo_dir / "model_index.json").is_file()
    is_transformers = (repo_dir / "config.json").is_file() and not is_diffusers
    if not (is_diffusers or is_transformers):
        raise InlineConversionNotPossible(
            reason=(
                f"source dir at {repo_dir} has neither model_index.json "
                f"(diffusers) nor config.json (transformers) — can't load "
                f"as a model for inline {dtype}"
            ),
            target_dtype=dtype,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import BitsAndBytesConfig
        from ._hf_load import load_component_module
    except ImportError as exc:
        raise InlineConversionNotPossible(
            reason=f"transformers/bitsandbytes not installed for {dtype}: {exc}",
            target_dtype=dtype,
        ) from exc

    # Diffusers-layout fan-out: per-component bnb quantization.
    if is_diffusers:
        return _run_bnb_diffusers_inline(
            repo_dir=repo_dir,
            out_dir=out_dir,
            dtype=dtype,
            bnb_quant_type=bnb_quant_type,
        )

    # bnb_4bit_compute_dtype=bf16 keeps activations in bf16 during dequant
    # at inference time. Standard bitsandbytes 4-bit recipe.
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    import json as _json
    cfg_path = repo_dir / "config.json"
    try:
        cfg_blob = _json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    except Exception:
        cfg_blob = {}
    try:
        model = load_component_module(
            repo_dir, cfg_blob, quantization_config=cfg,
        )
    except Exception as exc:
        raise InlineConversionNotPossible(
            reason=f"bitsandbytes load failed for {dtype}: {exc}",
            target_dtype=dtype,
        ) from exc

    try:
        model.save_pretrained(str(out_dir))
    except Exception as exc:
        raise InlineConversionNotPossible(
            reason=f"bitsandbytes save_pretrained failed for {dtype}: {exc}",
            target_dtype=dtype,
        ) from exc

    saved_files: list[Path] = sorted(
        f for f in out_dir.rglob("*") if f.is_file()
    )
    if not saved_files:
        raise RuntimeError(
            f"_run_bnb_inline: no files produced for {dtype} from {repo_dir}"
        )

    try:
        import bitsandbytes as bnb
        bnb_version = str(bnb.__version__)
    except Exception:
        bnb_version = ""

    attrs = {
        "dtype": dtype,
        "file_type": "safetensors",
        "conversion_strategy": "inline_bitsandbytes",
        "quant_scheme": f"bitsandbytes:{bnb_quant_type}",
        "quant_library": "bitsandbytes",
    }
    if bnb_version:
        attrs["quant_library_version"] = bnb_version
    return InlineConversionResult(
        output_paths=saved_files,
        index_path=None,
        target_dtype=dtype,
        target_file_type="safetensors",
        attributes=attrs,
        summary={
            "scheme": bnb_quant_type,
            "file_count": len(saved_files),
        },
    )


# Components that hold quantizable weights (DiT, UNet, text encoders, etc.).
# Everything not in this set passes through unchanged.
_DIFFUSERS_QUANT_COMPONENTS: frozenset[str] = frozenset({
    "transformer", "unet",
    "text_encoder", "text_encoder_2", "text_encoder_3",
    "image_encoder", "prior", "controlnet",
})

# Top-level files copied verbatim into the destination snapshot.
_DIFFUSERS_ROOT_FILES: tuple[str, ...] = (
    "model_index.json", "README.md", "LICENSE", "LICENSE.md", "USAGE_POLICY.md",
)


def _run_torchao_diffusers_inline(
    *,
    repo_dir: Path,
    out_dir: Path,
    dtype: str,
    scheme: str,
    cfg_factory: Any,
) -> InlineConversionResult:
    """Per-component torchao quantization for a diffusers-layout source.

    Walks each subdir under ``repo_dir``. Quantizes weight-bearing components
    (transformer / unet / text_encoder*); copies the rest verbatim. Each
    quantized component lands at ``<out_dir>/<comp_name>/model.safetensors``
    with a flatten metadata blob in the safetensors header. The destination
    is a complete diffusers snapshot — model_index.json + every component
    subdir + (quantized weights | passthrough copy).
    """
    import json as _json
    import shutil as _shutil
    import torch
    import safetensors.torch as st

    from torchao.quantization import quantize_
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )
    from ._hf_load import load_component_module

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quantized_components: list[str] = []
    passthrough_components: list[str] = []

    for entry in sorted(repo_dir.iterdir()):
        if not entry.is_dir():
            continue
        comp_name = entry.name
        comp_out = out_dir / comp_name

        if comp_name in _DIFFUSERS_QUANT_COMPONENTS:
            cfg_path = entry / "config.json"
            if not cfg_path.is_file():
                # No config → can't load → passthrough as a safety fallback.
                _shutil.copytree(entry, comp_out, dirs_exist_ok=True)
                passthrough_components.append(comp_name)
                continue
            try:
                cfg_data = _json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg_data = {}
            try:
                module = load_component_module(
                    entry, cfg_data, torch_dtype=torch.bfloat16,
                )
            except Exception as exc:
                raise InlineConversionNotPossible(
                    reason=(
                        f"failed to load diffusers component {comp_name!r} for "
                        f"inline {dtype}: {exc}"
                    ),
                    target_dtype=dtype,
                ) from exc

            try:
                quantize_(module, cfg_factory())
            except ImportError as exc:
                msg = str(exc)
                if "mslk" in msg or "fbgemm" in msg.lower() or "c10_cuda" in msg.lower():
                    raise InlineConversionNotPossible(
                        reason=(
                            f"{dtype} quantization requires fbgemm_gpu + CUDA "
                            f"(component {comp_name!r}). Run on a GPU worker."
                        ),
                        target_dtype=dtype,
                    ) from exc
                raise

            # Per-component flatten + safetensors save.
            comp_out.mkdir(parents=True, exist_ok=True)
            tensors_data, sd_metadata = flatten_tensor_state_dict(module.state_dict())
            st.save_file(
                tensors_data,
                str(comp_out / "model.safetensors"),
                metadata=sd_metadata,
            )
            # Carry the component's config files (not weights) so reload works.
            for src_file in entry.iterdir():
                if not src_file.is_file():
                    continue
                if src_file.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                    continue
                if src_file.name.endswith(".safetensors.index.json"):
                    continue
                _shutil.copy2(src_file, comp_out / src_file.name)
            # Stamp quantization_config in the component's config.json so a
            # downstream from_pretrained picks the right TorchAoConfig.
            comp_cfg = comp_out / "config.json"
            if comp_cfg.is_file():
                try:
                    cfg_blob = _json.loads(comp_cfg.read_text(encoding="utf-8"))
                except Exception:
                    cfg_blob = {}
                cfg_blob["quantization_config"] = {
                    "quant_method": "torchao",
                    "quant_type": _torchao_quant_type_name(scheme),
                    "modules_to_not_convert": None,
                }
                comp_cfg.write_text(
                    _json.dumps(cfg_blob, indent=2), encoding="utf-8",
                )
            quantized_components.append(comp_name)
            del module
        else:
            # Passthrough: vae / scheduler / tokenizer / feature_extractor /
            # safety_checker — copy verbatim.
            _shutil.copytree(entry, comp_out, dirs_exist_ok=True)
            passthrough_components.append(comp_name)

    # Top-level files (model_index.json, README, LICENSE).
    for fname in _DIFFUSERS_ROOT_FILES:
        src = repo_dir / fname
        if src.is_file():
            _shutil.copy2(src, out_dir / fname)

    if not quantized_components:
        raise InlineConversionNotPossible(
            reason=(
                f"diffusers source at {repo_dir} has no quantizable "
                f"components (looked for: {sorted(_DIFFUSERS_QUANT_COMPONENTS)})"
            ),
            target_dtype=dtype,
        )

    saved_files: list[Path] = sorted(
        f for f in out_dir.rglob("*") if f.is_file()
    )

    try:
        import torchao
        torchao_version = str(torchao.__version__)
    except Exception:
        torchao_version = ""

    attrs = {
        "dtype": dtype,
        "file_type": "safetensors",
        "conversion_strategy": "inline_torchao_diffusers",
        "quant_scheme": f"torchao:{scheme}",
        "quant_library": "torchao",
        "quant_components": ",".join(sorted(quantized_components)),
        "passthrough_components": ",".join(sorted(passthrough_components)),
    }
    if torchao_version:
        attrs["quant_library_version"] = torchao_version
    return InlineConversionResult(
        output_paths=saved_files,
        index_path=None,
        target_dtype=dtype,
        target_file_type="safetensors",
        attributes=attrs,
        summary={
            "scheme": scheme,
            "quantized_components": quantized_components,
            "passthrough_components": passthrough_components,
            "file_count": len(saved_files),
        },
    )


def _torchao_quant_type_name(scheme: str) -> str:
    """Map our internal scheme name → transformers' TorchAoConfig.quant_type string.

    Stamped into config.json's quantization_config so a future
    AutoModelForCausalLM.from_pretrained(...) on the destination repo
    rebuilds the right config. The string itself is what transformers'
    TorchAoConfig._get_torchao_quant_type_to_method consumes.
    """
    return {
        "int8_wo":      "int8_weight_only",
        "int4_wo":      "int4_weight_only",
        "fp8_wo":       "float8_weight_only",
        "fp8_dynamic":  "float8_dynamic_activation_float8_weight",
    }.get(scheme, scheme)


def _run_bnb_diffusers_inline(
    *,
    repo_dir: Path,
    out_dir: Path,
    dtype: str,
    bnb_quant_type: str,
) -> InlineConversionResult:
    """Per-component bitsandbytes nf4/fp4 for a diffusers-layout source.

    Same shape as `_run_torchao_diffusers_inline`: walk components, quantize
    weight-bearing ones (transformer / unet / text_encoder*), copy the rest
    verbatim. bitsandbytes' `Params4bit` is a parameter subclass that
    handles its own save/load via ``module.save_pretrained`` directly — no
    flatten helper needed.
    """
    import json as _json
    import shutil as _shutil
    import torch
    from transformers import BitsAndBytesConfig

    from ._hf_load import load_component_module

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    quantized_components: list[str] = []
    passthrough_components: list[str] = []

    for entry in sorted(repo_dir.iterdir()):
        if not entry.is_dir():
            continue
        comp_name = entry.name
        comp_out = out_dir / comp_name

        if comp_name in _DIFFUSERS_QUANT_COMPONENTS:
            cfg_path = entry / "config.json"
            if not cfg_path.is_file():
                _shutil.copytree(entry, comp_out, dirs_exist_ok=True)
                passthrough_components.append(comp_name)
                continue
            try:
                cfg_data = _json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg_data = {}
            try:
                module = load_component_module(
                    entry, cfg_data, quantization_config=cfg,
                )
            except Exception as exc:
                raise InlineConversionNotPossible(
                    reason=(
                        f"failed to load diffusers component {comp_name!r} for "
                        f"inline {dtype}: {exc}"
                    ),
                    target_dtype=dtype,
                ) from exc

            comp_out.mkdir(parents=True, exist_ok=True)
            try:
                module.save_pretrained(str(comp_out))
            except Exception as exc:
                raise InlineConversionNotPossible(
                    reason=(
                        f"bitsandbytes save_pretrained failed for component "
                        f"{comp_name!r} ({dtype}): {exc}"
                    ),
                    target_dtype=dtype,
                ) from exc
            quantized_components.append(comp_name)
            del module
        else:
            _shutil.copytree(entry, comp_out, dirs_exist_ok=True)
            passthrough_components.append(comp_name)

    for fname in _DIFFUSERS_ROOT_FILES:
        src = repo_dir / fname
        if src.is_file():
            _shutil.copy2(src, out_dir / fname)

    if not quantized_components:
        raise InlineConversionNotPossible(
            reason=(
                f"diffusers source at {repo_dir} has no quantizable "
                f"components for {dtype} (looked for: "
                f"{sorted(_DIFFUSERS_QUANT_COMPONENTS)})"
            ),
            target_dtype=dtype,
        )

    saved_files: list[Path] = sorted(
        f for f in out_dir.rglob("*") if f.is_file()
    )

    try:
        import bitsandbytes as bnb
        bnb_version = str(bnb.__version__)
    except Exception:
        bnb_version = ""

    attrs = {
        "dtype": dtype,
        "file_type": "safetensors",
        "conversion_strategy": "inline_bitsandbytes_diffusers",
        "quant_scheme": f"bitsandbytes:{bnb_quant_type}",
        "quant_library": "bitsandbytes",
        "quant_components": ",".join(sorted(quantized_components)),
        "passthrough_components": ",".join(sorted(passthrough_components)),
    }
    if bnb_version:
        attrs["quant_library_version"] = bnb_version
    return InlineConversionResult(
        output_paths=saved_files,
        index_path=None,
        target_dtype=dtype,
        target_file_type="safetensors",
        attributes=attrs,
        summary={
            "scheme": bnb_quant_type,
            "quantized_components": quantized_components,
            "passthrough_components": passthrough_components,
            "file_count": len(saved_files),
        },
    )


def _run_gguf_inline(
    *,
    source_path: Path,
    source_repo_dir: Path,
    out_dir: Path,
    encoding: str,
) -> InlineConversionResult:
    """GGUF quantization — convert_hf_to_gguf.py (+ optional llama-quantize)."""
    import os
    import subprocess

    from .gguf_utils import (
        prepare_hf_source_tree_for_gguf,
        resolve_gguf_convert_script,
        run_hf_to_gguf_conversion,
    )

    dtype = _normalize(encoding)
    if dtype not in _INLINE_GGUF_ENCODINGS:
        raise InlineConversionNotPossible(
            reason=f"unsupported gguf encoding: {dtype!r}",
            target_dtype=dtype,
        )

    # convert_hf_to_gguf.py only emits f16/bf16/q8_0/f32 directly; everything
    # else goes through a two-step F16 → llama-quantize pass.
    direct_encodings = {"f32", "f16", "bf16", "q8_0"}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = out_dir / "_gguf_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    script = resolve_gguf_convert_script()
    hf_model_dir, _arch = prepare_hf_source_tree_for_gguf(
        work_dir=work_dir,
        input_weights=Path(source_path),
        source_repo_dir=str(source_repo_dir),
    )

    final_path = out_dir / f"model-{dtype}.gguf"
    if dtype in direct_encodings:
        run_hf_to_gguf_conversion(
            script_path=script,
            hf_model_dir=hf_model_dir,
            output_path=final_path,
            encoding=dtype,
        )
    else:
        # Two-step: HF → F16 GGUF, then llama-quantize → target encoding.
        intermediate = work_dir / "model.f16.gguf"
        run_hf_to_gguf_conversion(
            script_path=script,
            hf_model_dir=hf_model_dir,
            output_path=intermediate,
            encoding="f16",
        )
        llama_quantize = os.environ.get("LLAMA_QUANTIZE_BIN", "").strip() or "llama-quantize"
        proc = subprocess.run(
            [llama_quantize, str(intermediate), str(final_path), dtype.upper()],
            capture_output=True,
            text=True,
            check=False,
            timeout=float(os.environ.get("CONVERSION_GGUF_QUANTIZE_TIMEOUT", "7200")),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-quantize failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[-2000:]}"
            )
        if not final_path.exists() or final_path.stat().st_size <= 0:
            raise RuntimeError(f"llama-quantize produced no output for {dtype}")

    attrs = {
        "dtype": f"gguf:{dtype}",
        "file_type": "gguf",
        "file_layout": "single-file",
        "conversion_strategy": "inline_gguf",
        "quant_library": "gguf",
        "quant_recipe": f"gguf:{dtype}",
        "gguf_encoding": dtype,
    }
    return InlineConversionResult(
        output_paths=[final_path],
        index_path=None,
        target_dtype=dtype,
        target_file_type="gguf",
        attributes=attrs,
        summary={"encoding": dtype, "file_size_bytes": final_path.stat().st_size},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(value: str) -> str:
    return str(value or "").strip().lower()


__all__ = [
    "InlineConversionNotPossible",
    "InlineConversionResult",
    "DeferredConversionRequirement",
    "deferred_conversion_requirement",
    "is_calibration_required",
    "is_inline_supported",
    "run_inline_conversion",
]
