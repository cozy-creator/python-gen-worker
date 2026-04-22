"""Streaming tensor conversion: read one tensor at a time, convert, accumulate output.

Uses safetensors `safe_open` for lazy per-tensor access so only one source tensor
is in memory at a time. Supports sharded input (multiple shard files) and
auto-sharding output based on a size threshold.

For a 7.75 GB fp16 model converted to fp8:
- Without streaming: ~16 GB (source + output in memory)
- With streaming: ~8 GB (output) + ~200 MB (largest single source tensor) ≈ ~8.2 GB
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Max bytes per output shard. Files smaller than this are written as a single file.
# Matches HuggingFace's modern default (transformers v4.34+, diffusers) which
# dropped from 10 GB to 5 GB so CDNs / low-RAM loaders can stream one shard at a time.
DEFAULT_SHARD_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def _iter_tensors_from_shards(
    shard_paths: list[Path],
) -> list[tuple[str, Path]]:
    """List all (tensor_name, shard_path) pairs across shards, preserving order."""
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in seen:
                    seen.add(name)
                    result.append((name, shard_path))
    return result


def _resolve_input_shards(input_path: Path) -> list[Path]:
    """Resolve a single safetensors file, or an index.json to its shard files."""
    name = input_path.name.lower()
    if name.endswith(".safetensors.index.json"):
        from .streaming_primitives import list_shard_files_from_index
        return list_shard_files_from_index(input_path)
    elif name.endswith(".safetensors"):
        return [input_path]
    else:
        return [input_path]


def _estimate_tensor_output_size(tensor: torch.Tensor, target_dtype: torch.dtype) -> int:
    """Estimate output size in bytes for a tensor after dtype conversion."""
    numel = tensor.numel()
    itemsize = torch.tensor([], dtype=target_dtype).element_size()
    return numel * itemsize


def _save_sharded(
    tensors: dict[str, torch.Tensor],
    output_path: Path,
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> list[Path]:
    """Save tensors to one or more shard files based on size threshold.

    Returns a list of output file paths (single file, or N shards + index.json).
    Delegates planning to the canonical ``_sharding.plan_safetensors_shards``
    so shard-naming and index format match the rest of gen_worker.conversion.
    """
    from ._sharding import build_safetensors_index, plan_safetensors_shards

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem.replace(".safetensors", "")
    tensor_sizes = {name: t.numel() * t.element_size() for name, t in tensors.items()}

    plan = plan_safetensors_shards(
        tensor_sizes,
        max_shard_bytes=shard_threshold,
        shard_prefix=stem,
    )
    if len(plan.shard_names) == 1:
        save_file(tensors, str(output_path))
        return [output_path]

    buckets: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        buckets.setdefault(plan.weight_map[name], {})[name] = tensor

    written: list[Path] = []
    for shard_name, bucket in buckets.items():
        shard_path = output_path.parent / shard_name
        save_file(bucket, str(shard_path))
        written.append(shard_path)

    index_path = output_path.parent / f"{stem}.safetensors.index.json"
    index_path.write_text(
        json.dumps(build_safetensors_index(plan), separators=(",", ":")),
        encoding="utf-8",
    )
    written.append(index_path)
    return written


def streaming_convert_safetensors(
    input_path: Path,
    output_path: Path,
    *,
    convert_fn: Callable[[str, torch.Tensor], torch.Tensor],
    skip_non_float: bool = True,
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Convert a safetensors file (or sharded set) tensor-by-tensor with minimal memory.

    Args:
        input_path: Source safetensors file or .safetensors.index.json.
        output_path: Destination safetensors file (will auto-shard if needed).
        convert_fn: Called with (tensor_name, tensor) → converted tensor.
        skip_non_float: If True, non-floating-point tensors bypass convert_fn.
        shard_threshold: Max bytes per output shard (default 5 GB).

    Returns:
        Dict with conversion stats.
    """
    shard_paths = _resolve_input_shards(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, torch.Tensor] = {}
    tensor_count = 0
    converted_count = 0
    skipped_count = 0

    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                tensor_count += 1

                if skip_non_float and not tensor.is_floating_point():
                    converted[name] = tensor
                    skipped_count += 1
                    continue

                result = convert_fn(name, tensor)
                converted[name] = result
                converted_count += 1

    output_files = _save_sharded(converted, output_path, shard_threshold)

    return {
        "tensor_count": tensor_count,
        "converted_count": converted_count,
        "skipped_count": skipped_count,
        "output_files": [str(p) for p in output_files],
        "output_shards": len([p for p in output_files if p.suffix == ".safetensors"]),
    }


def streaming_dtype_cast(
    input_path: Path,
    output_path: Path,
    *,
    target_dtype: torch.dtype,
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Dtype cast via incremental writer — minimal memory.

    Reads one tensor at a time, converts, writes directly to output file.
    Peak memory: one source tensor + one output tensor (~400 MB for largest layer).
    Does NOT accumulate the full output in memory.
    """
    from ._streaming_incremental import IncrementalSafetensorsWriter, torch_dtype_to_st

    shard_paths = _resolve_input_shards(input_path)
    target_st_dtype = torch_dtype_to_st(target_dtype)

    # First pass: collect all tensor metadata to build the output header.
    tensor_metas: list[tuple[str, str, list[int], Path]] = []  # (name, st_dtype, shape, shard_path)
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.is_floating_point():
                    out_dtype = target_st_dtype
                else:
                    out_dtype = torch_dtype_to_st(tensor.dtype)
                tensor_metas.append((name, out_dtype, list(tensor.shape), shard_path))
                del tensor

    # TODO: for output > shard_threshold, we'd need to split across files.
    # For now, write a single file (handles de-sharding).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_count = 0
    converted_count = 0

    with IncrementalSafetensorsWriter(output_path) as writer:
        # Register all tensors.
        for name, out_dtype, shape, _ in tensor_metas:
            writer.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
        writer.write_header()

        # Second pass: read, convert, write one tensor at a time.
        for name, out_dtype, shape, shard_path in tensor_metas:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)
            tensor_count += 1
            if tensor.is_floating_point():
                result = tensor.to(dtype=target_dtype)
                converted_count += 1
            else:
                result = tensor
            # Write raw bytes directly from contiguous tensor storage.
            # `bytes(untyped_storage())` falls into per-element Python iteration
            # (O(N) getitem calls) — for multi-GB tensors that takes hours.
            # Flatten + reinterpret as uint8 and use numpy.tobytes() — a single
            # C-level memcpy. `.flatten()` ensures 0-dim scalars (int64, etc.)
            # become 1-D so `.view(torch.uint8)` has a last-dim to expand.
            result = result.contiguous().flatten()
            raw_bytes = result.view(torch.uint8).numpy().tobytes()
            writer.write_tensor(name, raw_bytes)
            del tensor, result, raw_bytes

    return {
        "tensor_count": tensor_count,
        "converted_count": converted_count,
        "incremental": True,
    }


def streaming_nvfp4_quantize(
    input_path: Path,
    output_path: Path,
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Per-tensor absmax FP4 quantization via streaming."""

    shard_paths = _resolve_input_shards(input_path)
    converted: dict[str, torch.Tensor] = {}
    tensor_count = 0

    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                tensor_count += 1

                if tensor.is_floating_point() and tensor.ndim >= 2:
                    fp32 = tensor.to(dtype=torch.float32)
                    amax = fp32.abs().amax()
                    scale = amax / 7.0 if amax > 0 else torch.ones(1, dtype=torch.float32)
                    quantized = torch.clamp(torch.round(fp32 / scale), -8, 7).to(torch.int8)
                    converted[name] = quantized
                    converted[name + ".__nvfp4_scale__"] = scale.reshape(1).to(torch.float32)
                else:
                    converted[name] = tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_files = _save_sharded(converted, output_path, shard_threshold)
    return {"tensor_count": tensor_count, "output_shards": len(output_files)}


def streaming_gpu_quantize(
    input_path: Path,
    output_path: Path,
    *,
    quantize_fn: Callable[[str, torch.Tensor], torch.Tensor],
    device: str = "cuda",
    skip_non_float: bool = True,
    min_ndim: int = 2,
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """GPU-accelerated quantization via tensor-by-tensor streaming through VRAM.

    For each tensor:
      1. Load from disk (CPU)
      2. Move to GPU (.to(device))
      3. Quantize on GPU (quantize_fn)
      4. Move result back to CPU (.to('cpu'))
      5. Accumulate for output, free GPU memory

    Peak VRAM: ~2x largest single tensor. Works on 8 GB GPUs for 100 GB+ models.

    Args:
        input_path: Source safetensors file or index.json.
        output_path: Destination safetensors file.
        quantize_fn: Called with (name, gpu_tensor) → quantized gpu_tensor.
        device: CUDA device (default 'cuda').
        skip_non_float: Skip non-float tensors.
        min_ndim: Skip tensors with fewer dimensions (e.g., skip scalars/biases).
        shard_threshold: Max output shard size.
    """
    shard_paths = _resolve_input_shards(input_path)
    converted: dict[str, torch.Tensor] = {}
    tensor_count = 0
    quantized_count = 0

    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                tensor_count += 1

                should_quantize = (
                    tensor.is_floating_point()
                    and tensor.ndim >= min_ndim
                    and (not skip_non_float or tensor.is_floating_point())
                )

                if should_quantize:
                    # Stream through GPU: cpu → gpu → quantize → cpu
                    gpu_tensor = tensor.to(device=device)
                    del tensor
                    result_gpu = quantize_fn(name, gpu_tensor)
                    del gpu_tensor
                    result_cpu = result_gpu.to(device="cpu")
                    del result_gpu
                    torch.cuda.empty_cache()
                    converted[name] = result_cpu
                    quantized_count += 1
                else:
                    converted[name] = tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_files = _save_sharded(converted, output_path, shard_threshold)

    return {
        "tensor_count": tensor_count,
        "quantized_count": quantized_count,
        "output_shards": len(output_files),
        "device": device,
    }
