"""Inline conversion dispatch for clone_huggingface / clone_civitai (issue #73).

When the user requests an output dtype the source repo doesn't ship, the
clone path runs the conversion in-process — the same library code paths
other worker functions can call. The clone reuses its existing upload
session so all flavors land atomically under the same destination tag group.

Three buckets:

1. **Direct ingest** — caller decides this *before* calling here, by
   matching ``target_dtype`` against what the classifier saw in the source
   files. No conversion needed.

2. **Inline-supported** — weight-only schemes that don't need calibration,
   streaming per-tensor (peak anon RAM ≈ largest single tensor) except bnb:
   - ``bf16`` / ``fp16`` / ``fp32`` (streaming dtype cast)
   - ``fp8`` / ``fp8:e4m3`` (streaming fp8-E4M3 storage cast — the ``#fp8``
     flavor; scale-free, consumed via diffusers layerwise casting)
   - ``nf4`` / ``fp4`` / ``int4`` (bitsandbytes; full component load)
   - GGUF quants (``q4_k_m``, ``q8_0``, …) via convert_hf_to_gguf + llama-quantize

3. **Calibration-required** — raise ``InlineConversionNotPossible`` with a
   clear structured refusal: ``nvfp4`` (modelopt + calibration dataset + GPU).

The exception carries structured requirements so callers can render their own
guidance without this worker package knowing about any particular published
endpoint.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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

# fp8-E4M3 storage flavor (`#fp8`) — a streaming per-tensor cast with the
# layerwise-casting skip patterns honored. No model load, no scales.
_INLINE_FP8_STORAGE_DTYPES: frozenset[str] = frozenset({"fp8", "fp8:e4m3"})

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
# dataset and a GPU; running them silently as part of clone would either hang
# or produce garbage. nvfp4 in particular is NOT producible weight-only: the
# te#44 quality verdict (sd15, real forward passes through a generic prompt
# pool) was a hard FAIL, so there is no honest calibration-free nvfp4 path.
# The caller renders any product-specific follow-up guidance from the
# structured requirement.
_CALIBRATED_DTYPES: dict[str, DeferredConversionRequirement] = {
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
    if dtype in _INLINE_FP8_STORAGE_DTYPES:
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

    Mirrors the shape of streaming_dtype_cast / streaming_fp8_storage_cast so
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
    shard_prefix: str = "model",
    source_repo_dir: Path | None = None,
    fp8_block_scope: bool = False,
) -> InlineConversionResult:
    """Run the appropriate inline conversion for the requested target_dtype.

    ``source_path`` is the materialized input — for cast / fp8 paths this is
    the safetensors file (or .index.json) we'll read; for GGUF this is also
    the safetensors file (we look up sidecar configs via ``source_repo_dir``).

    ``source_repo_dir`` is the parent component directory containing
    ``config.json`` and tokenizer assets — used by the GGUF path and by
    the bnb path (which loads the component as an HF model). When
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

    # fp8-E4M3 storage flavor — streaming per-tensor cast, no model load.
    if dtype in _INLINE_FP8_STORAGE_DTYPES:
        return _run_fp8_storage_inline(
            source_path=source_path,
            out_dir=out_dir,
            shard_prefix=shard_prefix,
            block_scope=fp8_block_scope,
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

    from .writer import streaming_dtype_cast

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


def _run_fp8_storage_inline(
    *,
    source_path: Path,
    out_dir: Path,
    shard_prefix: str,
    block_scope: bool = False,
) -> InlineConversionResult:
    """fp8-E4M3 storage cast of one weight set (the ``#fp8`` flavor),
    streaming per-tensor. Layerwise-casting skip patterns are honored so a
    consumer's ``apply_fp8_storage`` reproduces the runtime fp8-storage lane
    exactly. Stamps dtype ``fp8`` — the detector keys on F8_E4M3 headers.
    ``block_scope=True`` = the transformers-backbone lane (repeated-block
    params only)."""
    from .writer import streaming_fp8_storage_cast

    result = streaming_fp8_storage_cast(
        Path(source_path), Path(out_dir), shard_prefix=shard_prefix,
        block_scope=block_scope,
    )
    index_path = result.get("index_path")
    attrs = {
        "dtype": "fp8",
        "file_type": "safetensors",
        "conversion_strategy": "inline_fp8_storage_cast",
        "quant_recipe": "fp8:e4m3-storage",
        "tensor_count": str(int(result.get("tensor_count") or 0)),
        "converted_count": str(int(result.get("converted_count") or 0)),
    }
    return InlineConversionResult(
        output_paths=[Path(p) for p in result.get("output_paths") or []],
        index_path=Path(index_path) if index_path is not None else None,
        target_dtype="fp8",
        target_file_type="safetensors",
        attributes=attrs,
        summary=dict(result),
    )


def _run_bnb_inline(
    *,
    source_path: Path,
    source_repo_dir: Path,
    out_dir: Path,
    target_dtype: str,
) -> InlineConversionResult:
    """bitsandbytes nf4 / fp4 weight-only quantization (CPU-friendly).

    Routes through transformers' ``BitsAndBytesConfig`` integration.
    bitsandbytes' ``Params4bit`` is a parameter subclass that handles
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
    # Stamp the PRODUCED dtype: an "int4" request routes through bnb nf4, so
    # the checkpoint must be labeled nf4/fp4 — inference dispatches on it (#358).
    produced_dtype = bnb_quant_type  # "nf4" | "fp4"
    if produced_dtype != dtype:
        logger.warning(
            "inline bnb: requested dtype %s produces %s; stamping %s",
            dtype, produced_dtype, produced_dtype,
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
        "dtype": produced_dtype,
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
        target_dtype=produced_dtype,
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


def _run_bnb_diffusers_inline(
    *,
    repo_dir: Path,
    out_dir: Path,
    dtype: str,
    bnb_quant_type: str,
) -> InlineConversionResult:
    """Per-component bitsandbytes nf4/fp4 for a diffusers-layout source.

    Walk components, quantize
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
    import subprocess

    from .gguf_tools import (
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
        llama_quantize = "llama-quantize"
        proc = subprocess.run(
            [llama_quantize, str(intermediate), str(final_path), dtype.upper()],
            capture_output=True,
            text=True,
            check=False,
            timeout=7200.0,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-quantize failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[-2000:]}"
            )
        if not final_path.exists() or final_path.stat().st_size <= 0:
            raise RuntimeError(f"llama-quantize produced no output for {dtype}")

    # Scratch is part of the peak working set, never part of the published
    # flavor tree. A cleanup failure must fail the flavor rather than leak an
    # F16 intermediate and tool sidecars into Tensorhub.
    shutil.rmtree(work_dir)

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
