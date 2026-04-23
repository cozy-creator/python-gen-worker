"""Streaming tensor conversion: read one tensor at a time, convert, write directly into N output shards.

Uses safetensors ``safe_open`` for lazy per-tensor reads and
``IncrementalSafetensorsWriter`` to stream converted tensor bytes directly
into the shard file they belong to — no intermediate combined file, no
``load_file``/``save_file`` round-trip. Peak memory is one source tensor +
one converted tensor (~400 MB for the largest layer in a multi-GB model).

Output is always a *directory* containing one or more shard files plus an
optional ``<prefix>.safetensors.index.json`` when sharding applied. Callers
don't branch on single-vs-sharded — the shape of the return value is the
same in both cases (``output_paths`` is always a list, ``index_path`` is
``None`` when a single shard fits under the threshold).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from safetensors import safe_open

# Max bytes per output shard. Files smaller than this are written as a single file.
# Matches HuggingFace's modern default (transformers v4.34+, diffusers) which
# dropped from 10 GB to 5 GB so CDNs / low-RAM loaders can stream one shard at a time.
DEFAULT_SHARD_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def _resolve_input_shards(input_path: Path) -> list[Path]:
    """Resolve a single safetensors file, or an index.json to its shard files."""
    name = input_path.name.lower()
    if name.endswith(".safetensors.index.json"):
        from .streaming_primitives import list_shard_files_from_index
        return list_shard_files_from_index(input_path)
    return [input_path]


def _estimate_tensor_output_size(tensor: torch.Tensor, target_dtype: torch.dtype) -> int:
    """Estimate output size in bytes for a tensor after dtype conversion."""
    numel = tensor.numel()
    itemsize = torch.tensor([], dtype=target_dtype).element_size()
    return numel * itemsize


def _write_index_if_sharded(
    out_dir: Path, plan: Any, shard_prefix: str
) -> Optional[Path]:
    from ._sharding import build_safetensors_index

    if len(plan.shard_names) <= 1:
        return None
    index_path = out_dir / f"{shard_prefix}.safetensors.index.json"
    index_path.write_text(
        json.dumps(build_safetensors_index(plan), separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return index_path


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a torch tensor to safetensors raw bytes (contiguous little-endian).

    ``bytes(tensor.untyped_storage())`` falls into per-element Python iteration
    (O(N) ``__getitem__`` calls) — for multi-GB tensors that's hours of CPU time.
    ``.flatten()`` + ``.view(torch.uint8)`` + ``.numpy().tobytes()`` is a single
    C-level memcpy. ``.flatten()`` ensures 0-dim scalars become 1-D so
    ``.view(torch.uint8)`` has a last-dim to expand.
    """
    return t.contiguous().flatten().view(torch.uint8).numpy().tobytes()


def streaming_dtype_cast(
    input_path: Path,
    out_dir: Path,
    *,
    target_dtype: torch.dtype,
    shard_prefix: str = "model",
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Dtype cast that writes directly into N shards per the shard planner.

    Each output tensor is decoded once and written into the shard it belongs
    to (per ``plan_safetensors_shards`` over the *post-cast* tensor sizes).
    No intermediate combined file is produced — peak disk usage is the
    shards themselves.

    Non-float tensors keep their source dtype (int indices, masks, etc.).

    Returns a dict with:
      - ``tensor_count``  - total tensors processed
      - ``converted_count`` - tensors that had their dtype changed
      - ``output_paths``  - list[Path] of shard files (N >= 1)
      - ``index_path``    - Path | None; present only when N > 1
      - ``shard_sizes``   - dict[shard_filename, bytes]
    """
    from ._sharding import plan_safetensors_shards
    from ._streaming_incremental import IncrementalSafetensorsWriter, torch_dtype_to_st

    shard_paths_in = _resolve_input_shards(input_path)
    target_st_dtype = torch_dtype_to_st(target_dtype)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: enumerate source tensors, compute output dtype + byte size.
    # We don't hold tensor data across the pass — `get_tensor` materializes
    # but we immediately drop the reference; safetensors caches only the
    # header, so this pass is header-walk speed, not IO-bound.
    tensor_metas: list[tuple[str, str, list[int], Path]] = []  # name, out_dtype, shape, src_shard
    size_map: dict[str, int] = {}
    for shard_path in shard_paths_in:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.is_floating_point():
                    out_dtype = target_st_dtype
                    nbytes = _estimate_tensor_output_size(tensor, target_dtype)
                else:
                    out_dtype = torch_dtype_to_st(tensor.dtype)
                    nbytes = int(tensor.numel() * tensor.element_size())
                tensor_metas.append((name, out_dtype, list(tensor.shape), shard_path))
                size_map[name] = nbytes
                del tensor

    plan = plan_safetensors_shards(
        size_map,
        max_shard_bytes=shard_threshold,
        shard_prefix=shard_prefix,
    )

    # Group tensors by destination shard, preserving source-encounter order
    # within each shard so the on-disk layout is deterministic.
    per_shard_order: dict[str, list[tuple[str, str, list[int], Path]]] = {
        name: [] for name in plan.shard_names
    }
    for name, out_dtype, shape, src in tensor_metas:
        dest = plan.weight_map.get(name)
        if dest is None:
            # Planner drops zero-byte tensors. Safetensors rejects zero-byte
            # tensors anyway, so this branch is unreachable in practice.
            continue
        per_shard_order[dest].append((name, out_dtype, shape, src))

    # Pass 2: write one shard at a time. Each shard gets one writer; we
    # register all its tensors, emit the header, then stream bytes.
    tensor_count = 0
    converted_count = 0
    for shard_name in plan.shard_names:
        entries = per_shard_order[shard_name]
        with IncrementalSafetensorsWriter(out_dir / shard_name) as w:
            for name, out_dtype, shape, _src in entries:
                w.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
            w.write_header()
            for name, _out_dtype, _shape, src_shard in entries:
                with safe_open(str(src_shard), framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(name)
                tensor_count += 1
                if tensor.is_floating_point():
                    result = tensor.to(dtype=target_dtype)
                    converted_count += 1
                else:
                    result = tensor
                raw = _tensor_to_bytes(result)
                w.write_tensor(name, raw)
                del tensor, result, raw

    index_path = _write_index_if_sharded(out_dir, plan, shard_prefix)

    return {
        "tensor_count": tensor_count,
        "converted_count": converted_count,
        "incremental": True,
        "output_paths": [out_dir / name for name in plan.shard_names],
        "index_path": index_path,
        "shard_sizes": dict(plan.shard_sizes),
    }


def streaming_nvfp4_quantize(
    input_path: Path,
    out_dir: Path,
    *,
    shard_prefix: str = "model",
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """Per-tensor absmax FP4 quantization via streaming writer.

    Quantizable tensors (floating point, ndim >= 2) emit two entries:
    ``<name>`` as int8 and ``<name>.__nvfp4_scale__`` as float32[1]. Non-
    quantizable tensors are passed through at their source dtype.

    Same output shape as ``streaming_dtype_cast`` — directory with N shards
    and an optional ``.index.json``.
    """
    from ._sharding import plan_safetensors_shards
    from ._streaming_incremental import IncrementalSafetensorsWriter, torch_dtype_to_st

    shard_paths_in = _resolve_input_shards(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_dtype_st = torch_dtype_to_st(torch.float32)
    int8_dtype_st = torch_dtype_to_st(torch.int8)

    # Pass 1: enumerate outputs. Each quantizable tensor produces TWO outputs
    # (the int8 payload + a scale tensor). Both need to be in the plan so the
    # sharder assigns them consistently — we co-locate by registering them in
    # encounter order; a sibling scale ends up next to its tensor.
    #
    # Output tuples: (out_name, out_dtype_st, out_shape, src_shard, src_name, kind)
    # where kind ∈ {"passthrough", "quantized", "scale"}.
    output_meta: list[tuple[str, str, list[int], Path, str, str]] = []
    size_map: dict[str, int] = {}
    for shard_path in shard_paths_in:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                shape = list(tensor.shape)
                is_quantizable = tensor.is_floating_point() and tensor.ndim >= 2
                if is_quantizable:
                    q_bytes = int(tensor.numel())  # int8 = 1 byte per element
                    output_meta.append((name, int8_dtype_st, shape, shard_path, name, "quantized"))
                    size_map[name] = q_bytes
                    scale_name = name + ".__nvfp4_scale__"
                    output_meta.append((scale_name, scale_dtype_st, [1], shard_path, name, "scale"))
                    size_map[scale_name] = 4  # float32[1]
                else:
                    src_dtype_st = torch_dtype_to_st(tensor.dtype)
                    nbytes = int(tensor.numel() * tensor.element_size())
                    output_meta.append((name, src_dtype_st, shape, shard_path, name, "passthrough"))
                    size_map[name] = nbytes
                del tensor

    plan = plan_safetensors_shards(
        size_map,
        max_shard_bytes=shard_threshold,
        shard_prefix=shard_prefix,
    )

    per_shard_order: dict[str, list[tuple[str, str, list[int], Path, str, str]]] = {
        name: [] for name in plan.shard_names
    }
    for row in output_meta:
        out_name = row[0]
        dest = plan.weight_map.get(out_name)
        if dest is None:
            continue
        per_shard_order[dest].append(row)

    tensor_count = 0
    quantized_count = 0
    for shard_name in plan.shard_names:
        entries = per_shard_order[shard_name]
        with IncrementalSafetensorsWriter(out_dir / shard_name) as w:
            for out_name, out_dtype, shape, _src, _sname, _kind in entries:
                w.add_tensor_metadata(out_name, dtype=out_dtype, shape=shape)
            w.write_header()
            # Cache the latest-processed scale so a scale entry sharded into
            # the same file as its base tensor reuses the computation. Entries
            # are encountered in registration order → quantized then scale.
            last_scale: tuple[str, torch.Tensor] | None = None
            for out_name, _out_dtype, _shape, src_shard, src_name, kind in entries:
                if kind == "scale":
                    if last_scale is None or last_scale[0] != src_name:
                        # Scale was sharded away from its base tensor — recompute.
                        with safe_open(str(src_shard), framework="pt", device="cpu") as f:
                            source = f.get_tensor(src_name)
                        fp32 = source.to(dtype=torch.float32)
                        amax = fp32.abs().amax()
                        scale = (amax / 7.0 if amax > 0 else torch.ones(1, dtype=torch.float32))
                        scale = scale.reshape(1).to(torch.float32)
                        del source, fp32
                    else:
                        scale = last_scale[1]
                    w.write_tensor(out_name, _tensor_to_bytes(scale))
                    continue

                with safe_open(str(src_shard), framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(src_name)
                tensor_count += 1

                if kind == "quantized":
                    fp32 = tensor.to(dtype=torch.float32)
                    amax = fp32.abs().amax()
                    scale = (amax / 7.0 if amax > 0 else torch.ones(1, dtype=torch.float32))
                    scale = scale.reshape(1).to(torch.float32)
                    quantized = torch.clamp(torch.round(fp32 / scale), -8, 7).to(torch.int8)
                    w.write_tensor(out_name, _tensor_to_bytes(quantized))
                    last_scale = (src_name, scale)
                    quantized_count += 1
                    del tensor, fp32, quantized
                else:
                    # passthrough
                    w.write_tensor(out_name, _tensor_to_bytes(tensor))
                    del tensor

    index_path = _write_index_if_sharded(out_dir, plan, shard_prefix)

    return {
        "tensor_count": tensor_count,
        "quantized_count": quantized_count,
        "incremental": True,
        "output_paths": [out_dir / name for name in plan.shard_names],
        "index_path": index_path,
        "shard_sizes": dict(plan.shard_sizes),
    }


def streaming_gpu_quantize(
    input_path: Path,
    out_dir: Path,
    *,
    quantize_fn: Callable[[str, torch.Tensor], torch.Tensor],
    device: str = "cuda",
    min_ndim: int = 2,
    shard_prefix: str = "model",
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD_BYTES,
) -> dict[str, Any]:
    """GPU-accelerated quantization via tensor-by-tensor streaming through VRAM.

    For each quantizable tensor: CPU -> GPU -> ``quantize_fn`` -> CPU ->
    write bytes -> free VRAM. Non-float tensors and low-dimensional tensors
    pass through untouched.

    Assumes the quantized output preserves the source tensor's shape and
    dtype. If a caller needs dtype-changing quantization (e.g. fp16 -> int8),
    they must bake that into ``quantize_fn`` and the size planner will be
    off by a constant factor — usually acceptable since plans use a safety
    margin.
    """
    from ._sharding import plan_safetensors_shards
    from ._streaming_incremental import IncrementalSafetensorsWriter, torch_dtype_to_st

    shard_paths_in = _resolve_input_shards(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan on the source byte sizes — quantize_fn is assumed to preserve shape
    # and dtype of the output (e.g. replace weights with quantized equivalents
    # that still round-trip through safetensors at the same byte width).
    tensor_metas: list[tuple[str, str, list[int], Path, bool]] = []  # name, dtype, shape, src, quantize
    size_map: dict[str, int] = {}
    for shard_path in shard_paths_in:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                should_q = tensor.is_floating_point() and tensor.ndim >= min_ndim
                dtype_st = torch_dtype_to_st(tensor.dtype)
                shape = list(tensor.shape)
                nbytes = int(tensor.numel() * tensor.element_size())
                tensor_metas.append((name, dtype_st, shape, shard_path, should_q))
                size_map[name] = nbytes
                del tensor

    plan = plan_safetensors_shards(
        size_map,
        max_shard_bytes=shard_threshold,
        shard_prefix=shard_prefix,
    )
    per_shard: dict[str, list[tuple[str, str, list[int], Path, bool]]] = {
        name: [] for name in plan.shard_names
    }
    for row in tensor_metas:
        dest = plan.weight_map.get(row[0])
        if dest is None:
            continue
        per_shard[dest].append(row)

    tensor_count = 0
    quantized_count = 0
    for shard_name in plan.shard_names:
        entries = per_shard[shard_name]
        with IncrementalSafetensorsWriter(out_dir / shard_name) as w:
            for name, dtype_st, shape, _src, _q in entries:
                w.add_tensor_metadata(name, dtype=dtype_st, shape=shape)
            w.write_header()
            for name, _dtype_st, _shape, src_shard, should_q in entries:
                with safe_open(str(src_shard), framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(name)
                tensor_count += 1
                if should_q:
                    gpu = tensor.to(device=device)
                    del tensor
                    result_gpu = quantize_fn(name, gpu)
                    del gpu
                    result = result_gpu.to(device="cpu")
                    del result_gpu
                    torch.cuda.empty_cache()
                    quantized_count += 1
                else:
                    result = tensor
                w.write_tensor(name, _tensor_to_bytes(result))
                del result

    index_path = _write_index_if_sharded(out_dir, plan, shard_prefix)

    return {
        "tensor_count": tensor_count,
        "quantized_count": quantized_count,
        "incremental": True,
        "output_paths": [out_dir / name for name in plan.shard_names],
        "index_path": index_path,
        "shard_sizes": dict(plan.shard_sizes),
        "device": device,
    }
