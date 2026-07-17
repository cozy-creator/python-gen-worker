"""The ONE streaming safetensors shard writer.

Collapses gen_worker.conversion's seven IO modules (writer, streaming_primitives,
_sharding, _tensor_iter, _streaming_incremental, _streaming_convert,
safetensors_io) into one:

  - shard planning (HF 5 GB convention) + index.json emit
  - IncrementalSafetensorsWriter: header first, then tensor bytes — O(1) memory
  - tensor iteration over single/sharded/pickle sources
  - streaming_dtype_cast / streaming_fp8_storage_cast: read one tensor,
    transform, write into its planned shard (peak RAM ~ largest tensor)
  - streaming_cast_snapshot / streaming_fp8_snapshot: whole-tree variants
  - shard_safetensors_by_offset: raw byte-range re-shard, zero decode

torch/safetensors imports are deferred so importing gen_worker.convert stays cheap.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional

if TYPE_CHECKING:
    import torch



class ConversionImplementationError(RuntimeError):
    """A conversion primitive can't proceed (bad input, missing dep)."""


MAX_SAFETENSORS_SHARD_BYTES: int = 2 * 1024 * 1024 * 1024
# Was 5GB (HF's own default shard size) until e2e tracker #110: tensorhub's
# per-upload /complete verifies the whole shard synchronously (streams it
# back from R2 and hashes it) in one HTTP request, and something in front of
# it (observed: a consistent, exact ~300s ceiling, almost certainly the
# ngrok tunnel this whole cloud stack rides on rather than tensorhub itself)
# kills that request outright regardless of tensorhub's own generous
# timeout. Live: a 5.36GB shard sometimes finished in ~200-260s (racy, close
# to the wall); a 9.8GB shard failed the SAME way on every one of 5 retries
# (it deterministically needs longer than the wall allows, so retrying
# doesn't help). 2GB keeps every shard's verify time comfortably clear of
# that ceiling regardless of R2 throughput variance. gen_worker.convert.hub's
# retry/poll resilience (#62/#63) still covers the remaining transient case;
# this fixes the deterministic one.

_PICKLE_EXTS = (".ckpt", ".pt", ".pth", ".bin")

# safetensors dtype name -> bytes per element
_ST_DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
_TORCH_TO_ST = {
    "torch.float16": "F16", "torch.bfloat16": "BF16", "torch.float32": "F32",
    "torch.float64": "F64", "torch.int8": "I8", "torch.int16": "I16",
    "torch.int32": "I32", "torch.int64": "I64", "torch.uint8": "U8",
    "torch.bool": "BOOL", "torch.float8_e4m3fn": "F8_E4M3",
    "torch.float8_e5m2": "F8_E5M2",
}


def torch_dtype_to_st(dtype: Any) -> str:
    key = str(dtype)
    if key not in _TORCH_TO_ST:
        raise ValueError(f"unsupported torch dtype for safetensors: {dtype}")
    return _TORCH_TO_ST[key]


# ---------------------------------------------------------------------------
# Shard planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardPlan:
    shard_names: list[str]
    weight_map: dict[str, str]
    shard_sizes: dict[str, int]
    total_size: int


def plan_shards(
    tensor_bytes: Mapping[str, int],
    *,
    max_shard_bytes: int = MAX_SAFETENSORS_SHARD_BYTES,
    shard_prefix: str = "model",
) -> ShardPlan:
    """Greedy first-fit shard plan in sorted tensor-name order (matches HF)."""
    if int(max_shard_bytes) <= 0:
        raise ValueError("max_shard_bytes must be > 0")
    entries: list[tuple[str, int]] = []
    for key in sorted(str(k).strip() for k in tensor_bytes.keys()):
        if key == "":
            continue
        size = int(tensor_bytes.get(key, 0) or 0)
        if size < 0:
            raise ValueError(f"tensor_size_invalid:{key}")
        if size > int(max_shard_bytes):
            raise ValueError(f"tensor_exceeds_max_shard_bytes:{key}")
        entries.append((key, size))

    single = f"{shard_prefix}.safetensors"
    total = sum(s for _, s in entries)
    if not entries or total <= int(max_shard_bytes):
        return ShardPlan(
            shard_names=[single],
            weight_map={k: single for k, _ in entries},
            shard_sizes={single: total},
            total_size=total,
        )

    shards: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_size = 0
    for key, size in entries:
        if current and current_size + size > int(max_shard_bytes):
            shards.append(current)
            current, current_size = [], 0
        current.append((key, size))
        current_size += size
    if current:
        shards.append(current)

    n = len(shards)
    names = [f"{shard_prefix}-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]
    weight_map: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name, group in zip(names, shards):
        sizes[name] = sum(s for _, s in group)
        for key, _ in group:
            weight_map[key] = name
    return ShardPlan(shard_names=names, weight_map=weight_map, shard_sizes=sizes, total_size=total)


def build_index(plan: ShardPlan) -> dict[str, object]:
    """HF-compatible ``model.safetensors.index.json`` payload."""
    return {"metadata": {"total_size": int(plan.total_size)}, "weight_map": dict(plan.weight_map)}


def list_shard_files_from_index(index_path: Path) -> list[Path]:
    """Shard file paths in weight-map order (deduped, first appearance)."""
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConversionImplementationError("sharded_index_unreadable") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConversionImplementationError("sharded_index_missing_weight_map")
    ordered: list[Path] = []
    seen: set[str] = set()
    for shard_name in weight_map.values():
        shard = str(shard_name).strip()
        if shard == "":
            raise ConversionImplementationError("sharded_index_invalid_shard_name")
        if shard not in seen:
            seen.add(shard)
            ordered.append(index_path.parent / shard)
    return ordered


# ---------------------------------------------------------------------------
# Incremental writer — header first, tensors streamed in order
# ---------------------------------------------------------------------------

class IncrementalSafetensorsWriter:
    """Write a safetensors file one tensor at a time (no full dict in memory).

    ``metadata`` (string-valued) is emitted as the header's ``__metadata__``.
    """

    def __init__(self, output_path: Path, *, metadata: Mapping[str, str] | None = None) -> None:
        self._output_path = Path(output_path)
        self._meta: list[tuple[str, str, list[int]]] = []  # (name, st_dtype, shape)
        self._metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        self._header_written = False
        self._fh: Any = None
        self._written: set[str] = set()
        self._sizes: list[int] = []

    def __enter__(self) -> "IncrementalSafetensorsWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def add_tensor_metadata(self, name: str, *, dtype: str, shape: list[int]) -> None:
        if self._header_written:
            raise RuntimeError("cannot add metadata after header is written")
        self._meta.append((name, dtype, list(shape)))

    def write_header(self) -> None:
        if self._header_written:
            raise RuntimeError("header already written")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._output_path, "wb")
        header: dict[str, Any] = {}
        if self._metadata:
            # Sorted for byte-determinism: safe_open().metadata() iterates in
            # randomized (Rust HashMap) order.
            header["__metadata__"] = dict(sorted(self._metadata.items()))
        offset = 0
        for name, dtype, shape in self._meta:
            elem = _ST_DTYPE_SIZES.get(dtype)
            if elem is None:
                raise ValueError(f"unknown safetensors dtype: {dtype}")
            numel = 1
            for dim in shape:
                numel *= dim
            size = numel * elem
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
            self._sizes.append(size)
            offset += size
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((8 - (len(blob) % 8)) % 8)
        self._fh.write(struct.pack("<Q", len(blob)))
        self._fh.write(blob)
        self._header_written = True

    def write_tensor(self, name: str, data: Any) -> None:
        """``data`` is any bytes-like buffer (bytes / memoryview / ndarray)."""
        if not self._header_written or self._fh is None:
            raise RuntimeError("write_header() must run before write_tensor()")
        idx = len(self._written)
        expected = self._meta[idx][0]
        if name != expected:
            raise RuntimeError(f"expected tensor {expected!r}, got {name!r} (write in order)")
        if name in self._written:
            raise RuntimeError(f"tensor {name!r} already written")
        nbytes = memoryview(data).nbytes
        if nbytes != self._sizes[idx]:
            raise RuntimeError(
                f"tensor {name!r}: got {nbytes} bytes, header declared {self._sizes[idx]}")
        self._fh.write(data)
        self._written.add(name)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _tensor_to_bytes(t: "torch.Tensor") -> Any:
    """Contiguous little-endian raw byte buffer (zero-copy numpy view)."""
    import torch

    return t.contiguous().flatten().view(torch.uint8).numpy()


# ---------------------------------------------------------------------------
# Tensor iteration (single / sharded / pickle inputs)
# ---------------------------------------------------------------------------

_WEIGHT_COMPONENT_DIRS: frozenset[str] = frozenset({
    "unet", "transformer", "vae", "text_encoder", "text_encoder_2",
    "text_encoder_3", "image_encoder", "prior", "controlnet",
})


def materialize_pickle_to_safetensors(pickle_path: Path, work_dir: Path) -> Path:
    """Convert a pickle weight file to safetensors (``weights_only=True``)."""
    import torch
    from safetensors.torch import save_file

    try:
        state = torch.load(str(pickle_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConversionImplementationError(
            f"pickle_load_failed: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    tensors = {
        k: v.contiguous() for k, v in state.items()
        if hasattr(v, "dtype") and hasattr(v, "shape")
    }
    if not tensors:
        raise ConversionImplementationError(f"pickle_no_tensors_found:{pickle_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / (pickle_path.stem + ".safetensors")
    save_file(tensors, str(out))
    return out


def materialize_safetensors_input(input_path: Path, work_dir: Path) -> Path:
    """Coerce a weight file into a path the streaming readers can open.

    ``.safetensors`` and ``.safetensors.index.json`` pass through (index is
    validated); pickle is converted via ``materialize_pickle_to_safetensors``.
    """
    path = Path(input_path)
    lower = path.name.lower()
    if lower.endswith(".safetensors"):
        return path
    if any(lower.endswith(ext) for ext in _PICKLE_EXTS):
        return materialize_pickle_to_safetensors(path, work_dir)
    if not lower.endswith(".safetensors.index.json"):
        raise ValueError("requires_safetensors_or_index_input")
    shards = list_shard_files_from_index(path)
    for shard in shards:
        if not shard.exists():
            raise ConversionImplementationError(f"sharded_index_missing_shard:{shard.name}")
    return path


def _resolve_input_shards(input_path: Path) -> list[Path]:
    if input_path.name.lower().endswith(".safetensors.index.json"):
        return list_shard_files_from_index(input_path)
    return [input_path]


def iter_component_tensors(component_dir: Path) -> Iterator[tuple[str, "torch.Tensor"]]:
    """Yield (name, tensor) for every weight in one component directory.

    Preference: sharded index > single safetensors > pickle (converted).
    """
    from safetensors import safe_open

    entry: Optional[Path] = None
    for p in sorted(component_dir.iterdir()):
        if p.name.endswith(".safetensors.index.json"):
            entry = p
            break
    if entry is None:
        st = sorted(component_dir.glob("*.safetensors"))
        if st:
            entry = st[0]
    if entry is None:
        for ext in _PICKLE_EXTS:
            found = sorted(component_dir.glob(f"*{ext}"))
            if found:
                entry = materialize_pickle_to_safetensors(
                    found[0], component_dir / ".__pickle_cache__")
                break
    if entry is None:
        return
    for shard in _resolve_input_shards(entry):
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def iter_source_tensors(
    root: Path,
    *,
    file_layout: str,
    components_filter: list[str] | None = None,
) -> Iterator[tuple[str, str, "torch.Tensor"]]:
    """Yield (component, name, tensor) across a whole source snapshot."""
    if file_layout == "singlefile":
        for name, tensor in iter_component_tensors(root):
            yield "", name, tensor
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name not in _WEIGHT_COMPONENT_DIRS:
            continue
        if components_filter is not None and entry.name not in components_filter:
            continue
        for name, tensor in iter_component_tensors(entry):
            yield entry.name, name, tensor


# ---------------------------------------------------------------------------
# Streaming per-tensor re-encode — one tensor in memory at a time
# ---------------------------------------------------------------------------

_ST_FLOAT_DTYPES: frozenset[str] = frozenset(
    {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})


def _stream_reencode(
    input_path: Path,
    out_dir: Path,
    *,
    out_st_dtype_for: Any,   # (name, src_st_dtype, shape) -> output st dtype
    transform: Any,          # (name, tensor, out_st_dtype) -> tensor
    shard_prefix: str,
    shard_threshold: int,
) -> dict[str, Any]:
    """Two-pass streaming re-encode over safetensors input(s).

    Pass 1 reads only the shard headers (``get_slice`` — no tensor data) to
    plan output shards; pass 2 reads one tensor at a time, applies
    ``transform``, and appends it to its output shard. Peak anonymous memory
    ≈ the largest single tensor. Source ``__metadata__`` is preserved.
    """
    from safetensors import safe_open

    shards_in = _resolve_input_shards(Path(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: header-only walk — output dtype + byte size per tensor.
    metas: list[tuple[str, str, list[int], Path]] = []  # name, out_dtype, shape, src_shard
    size_map: dict[str, int] = {}
    source_metadata: dict[str, str] = {}
    for shard_path in shards_in:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            md = f.metadata()
            if md:
                source_metadata.update({str(k): str(v) for k, v in md.items()})
            for name in f.keys():
                sl = f.get_slice(name)
                shape = list(sl.get_shape())
                out_dtype = str(out_st_dtype_for(name, str(sl.get_dtype()), shape))
                numel = 1
                for dim in shape:
                    numel *= dim
                metas.append((name, out_dtype, shape, shard_path))
                size_map[name] = numel * _ST_DTYPE_SIZES[out_dtype]

    plan = plan_shards(size_map, max_shard_bytes=shard_threshold, shard_prefix=shard_prefix)
    per_shard: dict[str, list[tuple[str, str, list[int], Path]]] = {n: [] for n in plan.shard_names}
    for row in metas:
        dest = plan.weight_map.get(row[0])
        if dest is not None:
            per_shard[dest].append(row)

    tensor_count = 0
    converted = 0
    handles: dict[Path, Any] = {}
    try:
        for shard_name in plan.shard_names:
            entries = per_shard[shard_name]
            with IncrementalSafetensorsWriter(
                out_dir / shard_name, metadata=source_metadata,
            ) as w:
                for name, out_dtype, shape, _src in entries:
                    w.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
                w.write_header()
                for name, out_dtype, _shape, src in entries:
                    f = handles.get(src)
                    if f is None:
                        f = handles[src] = safe_open(str(src), framework="pt", device="cpu")
                    t = f.get_tensor(name)
                    tensor_count += 1
                    result = transform(name, t, out_dtype)
                    if result is not t:
                        converted += 1
                    if torch_dtype_to_st(result.dtype) != out_dtype:
                        raise ConversionImplementationError(
                            f"transform produced {result.dtype} for {name!r}; "
                            f"planned {out_dtype}")
                    w.write_tensor(name, _tensor_to_bytes(result))
                    del t, result
    finally:
        for f in handles.values():
            try:
                f.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    index_path: Optional[Path] = None
    if len(plan.shard_names) > 1:
        index_path = out_dir / f"{shard_prefix}.safetensors.index.json"
        index_path.write_text(
            json.dumps(build_index(plan), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
    return {
        "tensor_count": tensor_count,
        "converted_count": converted,
        "output_paths": [out_dir / n for n in plan.shard_names],
        "index_path": index_path,
        "shard_sizes": dict(plan.shard_sizes),
        "metadata": dict(source_metadata),
    }


def streaming_dtype_cast(
    input_path: Path,
    out_dir: Path,
    *,
    target_dtype: "torch.dtype",
    shard_prefix: str = "model",
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
) -> dict[str, Any]:
    """Cast float tensors to ``target_dtype``, streaming directly into N shards.

    Non-float tensors keep their source dtype. Returns ``output_paths``
    (list of shard files), ``index_path`` (None for a single shard),
    ``tensor_count`` / ``converted_count`` / ``shard_sizes`` / ``metadata``.
    """
    target_st = torch_dtype_to_st(target_dtype)

    def out_st_dtype_for(_name: str, src_st: str, _shape: list[int]) -> str:
        return target_st if src_st in _ST_FLOAT_DTYPES else src_st

    def transform(_name: str, t: "torch.Tensor", _out_st: str) -> "torch.Tensor":
        if t.is_floating_point() and t.dtype != target_dtype:
            return t.to(dtype=target_dtype)
        return t

    return _stream_reencode(
        Path(input_path), Path(out_dir),
        out_st_dtype_for=out_st_dtype_for, transform=transform,
        shard_prefix=shard_prefix, shard_threshold=shard_threshold,
    )


# fp8-E4M3 storage cast (the `#fp8` flavor): matches the consumption side —
# diffusers layerwise casting (gen_worker.models.loading.apply_fp8_storage)
# casts only Linear/Conv modules whose qualified name misses the skip
# patterns, and upcasts them per-layer at compute time. The producer casts
# strictly LESS than any consumer would: only >=2-D `.weight` tensors whose
# module path misses the union of diffusers' skip patterns (defaults + every
# per-class `_skip_layerwise_casting_patterns`). Anything skipped here that a
# consumer does cast merely stores bigger — never a baked-in quality loss.
FP8_SKIP_TENSOR_PATTERNS: tuple[str, ...] = (
    "embed",            # pos_embed / patch_embed(ding|der) / *_embedder / embed_tokens / time_embedding
    "norm",
    "pooler",
    "adaln_single",
    "final_layer",
    "quantize",
    "decoder",
    "preprocess_conv", "postprocess_conv",
    r"^proj_in$", r"^proj_out$", r"^proj$",
)

_FP8_E4M3_MAX = 448.0  # torch float8_e4m3fn cast does NOT saturate; clamp first


def fp8_cast_eligible(
    name: str, src_st_dtype: str, shape: list[int],
    *, skip_patterns: tuple[str, ...] = FP8_SKIP_TENSOR_PATTERNS,
) -> bool:
    """True when a tensor is safe to store as fp8-E4M3 for the ``#fp8`` flavor."""
    import re

    if src_st_dtype not in {"F64", "F32", "F16", "BF16"}:
        return False
    if len(shape) < 2 or not name.endswith(".weight"):
        return False
    module_path = name[: -len(".weight")]
    return not any(re.search(p, module_path) for p in skip_patterns)


# ``.<block-list>.<idx>.`` segment — a param living under a repeated-block
# container (nn.ModuleList child). The transformers-backbone fp8 lane casts
# ONLY these, mirroring the runtime block-window walk
# (gen_worker.models.loading._fp8_block_windows): params outside repeated
# blocks (embeddings, final norms, heads) stay at source precision, so the
# stored flavor is a strict subset of what any consumer re-arms.
_FP8_BLOCK_SCOPE_RE = r"\.\d+\."


def _in_repeated_block(name: str) -> bool:
    import re

    return re.search(_FP8_BLOCK_SCOPE_RE, name) is not None


def streaming_fp8_storage_cast(
    input_path: Path,
    out_dir: Path,
    *,
    shard_prefix: str = "model",
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
    skip_patterns: tuple[str, ...] = FP8_SKIP_TENSOR_PATTERNS,
    block_scope: bool = False,
) -> dict[str, Any]:
    """Produce the fp8-E4M3 storage flavor of one weight set, streaming.

    Eligible weights are clamped to ±448 and stored as F8_E4M3; everything
    else keeps its source dtype. Scale-free by design: consumption is
    diffusers layerwise casting (fp8 bytes resident, bf16/fp16 compute).

    ``block_scope=True`` (the transformers-backbone lane) additionally
    requires an eligible weight to live under a repeated-block container
    (``.<idx>.`` path segment) — the stored set stays a strict subset of the
    runtime block-window walk.
    """
    import torch

    def out_st_dtype_for(name: str, src_st: str, shape: list[int]) -> str:
        if block_scope and not _in_repeated_block(name):
            return src_st
        if fp8_cast_eligible(name, src_st, shape, skip_patterns=skip_patterns):
            return "F8_E4M3"
        return src_st

    def transform(_name: str, t: "torch.Tensor", out_st: str) -> "torch.Tensor":
        if out_st == "F8_E4M3" and t.dtype != torch.float8_e4m3fn:
            return t.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return t

    return _stream_reencode(
        Path(input_path), Path(out_dir),
        out_st_dtype_for=out_st_dtype_for, transform=transform,
        shard_prefix=shard_prefix, shard_threshold=shard_threshold,
    )


# ---------------------------------------------------------------------------
# W8A8 per-channel-scaled fp8 producer (gw#557 / ie#494) — data-free requant
# from the bf16 source, streaming. The artifact contract is gw#534's
# ``#fp8-w8a8`` (consumed by gen_worker.models.w8a8): per quantized Linear a
# F8_E4M3 ``weight`` plus a F32 [out] ``weight_scale`` DEQUANT twin; excluded
# layers stay at source precision with NO scale tensor. Activation scales are
# DYNAMIC at serve time (Paul's settled design — no static calibration), so
# the producer emits no input_scale.
# ---------------------------------------------------------------------------

W8A8_QUANT_SCHEME = "fp8-w8a8"  # == gen_worker.models.w8a8.W8A8_FLAVOR (test-guarded)

# The ie#494 probe spec: quantize ONLY repeated-block Linears; MoE-style
# gate-logit projections stay full precision even when 16-aligned (the probe
# skipped LTX's 288 ``to_gate_logits`` explicitly).
W8A8_SKIP_TENSOR_PATTERNS: tuple[str, ...] = FP8_SKIP_TENSOR_PATTERNS + (
    "gate_logits",
)

_W8A8_DIM_ALIGN = 16  # torch._scaled_mm alignment (models.w8a8._DIM_ALIGN)
_SCALE_SUFFIX = ".weight_scale"


def w8a8_cast_eligible(
    name: str, src_st_dtype: str, shape: list[int],
    *, skip_patterns: tuple[str, ...] = W8A8_SKIP_TENSOR_PATTERNS,
) -> bool:
    """True when a stored tensor becomes a quantized w8a8 Linear weight:
    a 2-D float ``.weight`` under a repeated-block container, both dims
    16-aligned, missing every skip pattern. Everything else (embeddings,
    norms, projections outside blocks, misaligned or conv weights) stays at
    source precision — mirroring the ie#494 probe's flip/skip list."""
    if src_st_dtype not in {"F64", "F32", "F16", "BF16"}:
        return False
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    if shape[0] % _W8A8_DIM_ALIGN or shape[1] % _W8A8_DIM_ALIGN:
        return False
    if not _in_repeated_block(name):
        return False
    module_path = name[: -len(".weight")]
    return not any(re.search(p, module_path) for p in skip_patterns)


def streaming_w8a8_cast(
    input_path: Path,
    out_dir: Path,
    *,
    shard_prefix: str = "model",
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
    skip_patterns: tuple[str, ...] = W8A8_SKIP_TENSOR_PATTERNS,
) -> dict[str, Any]:
    """Per-channel-scaled fp8 requant of one weight set, streaming.

    Two passes like :func:`_stream_reencode`, but eligible weights emit TWO
    output tensors — the fp8 ``weight`` and its F32 [out] ``weight_scale``
    (``scale = amax(row)/448``, ``q = round(w/scale)`` in fp32; the probe's
    exact recipe). Peak anonymous memory ~ the largest single tensor.
    ``.weight`` sorts immediately before ``.weight_scale``, so a weight is
    always quantized before its scale is due — the scale is cached (tiny)
    across a shard boundary if the plan splits the pair.
    """
    import torch
    from safetensors import safe_open

    shards_in = _resolve_input_shards(Path(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: header-only walk.
    metas: list[tuple[str, str, list[int], Optional[Path]]] = []
    size_map: dict[str, int] = {}
    source_metadata: dict[str, str] = {}
    quantized_names: set[str] = set()
    for shard_path in shards_in:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            md = f.metadata()
            if md:
                source_metadata.update({str(k): str(v) for k, v in md.items()})
            for name in f.keys():
                if name.endswith(_SCALE_SUFFIX):
                    raise ConversionImplementationError(
                        f"source already carries {name!r} — refusing to "
                        "re-quantize a w8a8 artifact (requant from the "
                        "bf16 source instead)")
                sl = f.get_slice(name)
                shape = list(sl.get_shape())
                numel = 1
                for dim in shape:
                    numel *= dim
                if w8a8_cast_eligible(
                        name, str(sl.get_dtype()), shape,
                        skip_patterns=skip_patterns):
                    quantized_names.add(name)
                    metas.append((name, "F8_E4M3", shape, shard_path))
                    size_map[name] = numel * _ST_DTYPE_SIZES["F8_E4M3"]
                    scale_name = name[: -len(".weight")] + _SCALE_SUFFIX
                    metas.append((scale_name, "F32", [shape[0]], None))
                    size_map[scale_name] = shape[0] * _ST_DTYPE_SIZES["F32"]
                else:
                    metas.append((name, str(sl.get_dtype()), shape, shard_path))
                    size_map[name] = numel * _ST_DTYPE_SIZES[str(sl.get_dtype())]

    out_metadata = dict(source_metadata)
    out_metadata.update({
        "quant_scheme": W8A8_QUANT_SCHEME,
        "quant_recipe": "w8a8-pcs-dynamic",
        "calibration_corpus": "",
        "modelopt_version": "",
    })
    plan = plan_shards(size_map, max_shard_bytes=shard_threshold,
                       shard_prefix=shard_prefix)
    per_shard: dict[str, list[tuple[str, str, list[int], Optional[Path]]]] = {
        n: [] for n in plan.shard_names}
    for row in metas:
        dest = plan.weight_map.get(row[0])
        if dest is not None:
            per_shard[dest].append(row)

    tensor_count = 0
    pending_scales: dict[str, "torch.Tensor"] = {}
    handles: dict[Path, Any] = {}
    try:
        for shard_name in plan.shard_names:
            entries = per_shard[shard_name]
            with IncrementalSafetensorsWriter(
                out_dir / shard_name, metadata=out_metadata,
            ) as w:
                for name, out_dtype, shape, _src in entries:
                    w.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
                w.write_header()
                for name, _out_dtype, _shape, src in entries:
                    if src is None:  # synthesized weight_scale
                        w.write_tensor(
                            name, _tensor_to_bytes(pending_scales.pop(name)))
                        continue
                    f = handles.get(src)
                    if f is None:
                        f = handles[src] = safe_open(
                            str(src), framework="pt", device="cpu")
                    t = f.get_tensor(name)
                    tensor_count += 1
                    if name in quantized_names:
                        wf = t.float()
                        scale = (wf.abs().amax(dim=1, keepdim=True)
                                 / _FP8_E4M3_MAX).clamp(min=1e-12)
                        q = (wf / scale).clamp(
                            -_FP8_E4M3_MAX, _FP8_E4M3_MAX,
                        ).to(torch.float8_e4m3fn)
                        scale_name = name[: -len(".weight")] + _SCALE_SUFFIX
                        pending_scales[scale_name] = scale.reshape(-1).contiguous()
                        w.write_tensor(name, _tensor_to_bytes(q))
                        del wf, q
                    else:
                        w.write_tensor(name, _tensor_to_bytes(t))
                    del t
    finally:
        for f in handles.values():
            try:
                f.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
    if pending_scales:
        raise ConversionImplementationError(
            f"w8a8 cast left {len(pending_scales)} orphan scale tensor(s) "
            f"(e.g. {sorted(pending_scales)[:3]})")

    index_path: Optional[Path] = None
    if len(plan.shard_names) > 1:
        index_path = out_dir / f"{shard_prefix}.safetensors.index.json"
        index_path.write_text(
            json.dumps(build_index(plan), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
    return {
        "tensor_count": tensor_count,
        "converted_count": len(quantized_names),
        "output_paths": [out_dir / n for n in plan.shard_names],
        "index_path": index_path,
        "shard_sizes": dict(plan.shard_sizes),
        "metadata": out_metadata,
    }


# ---------------------------------------------------------------------------
# Snapshot-level streaming conversion (whole source tree, per weight group)
# ---------------------------------------------------------------------------

# Components fp8 storage targets by default: the denoiser dominates VRAM and
# is what apply_fp8_storage consumes (QUANTIZATION-POLICY.md order of
# sacrifice; TEs join via gw#392's component-wise ladder, explicitly).
FP8_DEFAULT_COMPONENTS: tuple[str, ...] = ("transformer", "unet")


def snapshot_weight_groups(source_dir: Path, layout: str) -> list[tuple[str, Path]]:
    """(component, entry_path) per weight set. entry is the index.json for
    sharded sets, else the safetensors file. A singlefile-layout source can
    still carry several root weight files (civitai bundles ship the
    diffusion model + text encoder + VAE side by side) — every one is its
    own group, or a dtype pass would convert only the first and the tree
    copy would silently drop the rest."""
    groups: list[tuple[str, Path]] = []

    def _entries_for(d: Path) -> list[Path]:
        idx = sorted(d.glob("*.safetensors.index.json"))
        sharded_members: set[str] = set()
        for i in idx:
            try:
                weight_map = json.loads(i.read_text("utf-8")).get("weight_map") or {}
                sharded_members.update(str(v) for v in weight_map.values())
            except Exception:
                pass
        loose = [p for p in sorted(d.glob("*.safetensors"))
                 if p.is_file() and p.name not in sharded_members]
        return idx + loose

    if layout == "diffusers":
        for entry in sorted(source_dir.iterdir()):
            if entry.is_dir():
                found = _entries_for(entry)
                if found:
                    groups.append((entry.name, found[0]))
    else:
        for entry_path in _entries_for(source_dir):
            groups.append(("", entry_path))
    return groups


def _link_or_copy(src: Path, dst: Path) -> None:
    import os

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        # Resolve first: HF-cache snapshots are relative-symlink farms and a
        # hardlink to the SYMLINK breaks the moment the tree is moved
        # (gw#415 live: model_index.json -> ../../blobs/... dangling).
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_non_weight_files(source_dir: Path, out_dir: Path, *, skip_components: set[str]) -> None:
    """Hardlink every non-weight file into the output tree; weight files and
    their sharded indexes for converted components are skipped."""
    for f in sorted(source_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(source_dir)
        if rel.parts[:2] == (".cache", "huggingface"):
            continue  # hf local-dir download metadata, not repo content
        comp = rel.parts[0] if len(rel.parts) > 1 else ""
        name = f.name
        is_weightish = f.suffix == ".safetensors" or name.endswith(".safetensors.index.json")
        if is_weightish and (comp in skip_components or ("" in skip_components and comp == "")):
            continue
        if name == ".civitai.json":
            continue
        _link_or_copy(f, out_dir / rel)


def _group_shard_prefix(entry: Path) -> str:
    stem = entry.name
    for suffix in (".safetensors.index.json", ".safetensors"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] or "model"
    return stem or "model"


def streaming_cast_snapshot(
    source_dir: Path,
    out_dir: Path,
    *,
    file_layout: str,
    target_dtype: "torch.dtype",
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
) -> dict[str, Any]:
    """Streaming dtype cast of a whole snapshot: every weight group is cast
    per-tensor (peak anon RAM ≈ largest tensor); configs/tokenizers hardlink
    through. Output is a complete loadable tree."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    groups = snapshot_weight_groups(source_dir, file_layout)
    if not groups:
        raise ConversionImplementationError("no safetensors weights found to cast")
    tensor_count = converted = 0
    done: set[str] = set()
    for comp, entry in groups:
        result = streaming_dtype_cast(
            entry, (out_dir / comp) if comp else out_dir,
            target_dtype=target_dtype,
            shard_prefix=_group_shard_prefix(entry),
            shard_threshold=shard_threshold,
        )
        tensor_count += int(result["tensor_count"])
        converted += int(result["converted_count"])
        done.add(comp)
    copy_non_weight_files(source_dir, out_dir, skip_components=done)
    return {"tensor_count": tensor_count, "converted_count": converted,
            "components": sorted(done), "output_dir": out_dir}


# Text-encoder components the ``fp8+te`` rung casts (must mirror
# gen_worker.models.loading._FP8_TEXT_ENCODER_COMPONENTS — drift guard in
# tests/test_fp8_te_writer.py; not imported here to keep this module's
# import cheap).
FP8_TE_COMPONENTS: tuple[str, ...] = (
    "text_encoder", "text_encoder_2", "text_encoder_3",
)


def _component_stored_tensor_names(component_dir: Path) -> frozenset[str]:
    """Tensor names as stored in the component's safetensors file(s)."""
    names: set[str] = set()
    idx = sorted(component_dir.glob("*.safetensors.index.json"))
    if idx:
        payload = json.loads(idx[0].read_text(encoding="utf-8"))
        names.update(str(k) for k in (payload.get("weight_map") or {}))
        return frozenset(names)
    for f in sorted(component_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        names.update(k for k in header if k != "__metadata__")
    return frozenset(names)


def _loader_key_translator(model: Any) -> Any:
    """Checkpoint-key -> module-graph-key translation using transformers' OWN
    load-path machinery (WeightRenaming rules + base-model-prefix
    reconciliation) — the same code ``from_pretrained`` runs, so old-layout
    checkpoints (e.g. Gemma3 ``language_model.model.*`` vs the 4.52+
    ``model.language_model.*`` graph) resolve exactly like the loader."""
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightRenaming,
            rename_source_key,
        )
    except ImportError:
        # transformers 4.x: the mapping is a class attr of regex->replacement
        # pairs, applied by from_pretrained via re.sub in order.
        import re as _re

        mapping = getattr(type(model), "_checkpoint_conversion_mapping", None)
        if not mapping:
            return lambda k: k

        def translate_4x(key: str) -> str:
            for pat, repl in mapping.items():
                key = _re.sub(pat, repl, key)
            return key

        return translate_4x
    try:
        transforms = get_model_conversion_mapping(model)
    except Exception:  # noqa: BLE001
        transforms = []
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters: "list[Any]" = [t for t in transforms if not isinstance(t, WeightRenaming)]
    meta_sd = model.state_dict()
    prefix = getattr(model, "base_model_prefix", None)

    def translate(key: str) -> str:
        try:
            renamed, _ = rename_source_key(key, renamings, converters, prefix, meta_sd)
            return renamed
        except Exception:  # noqa: BLE001
            return key

    return translate


def te_fp8_castable_keys(component_dir: Path) -> frozenset[str]:
    """STORED tensor names the ``fp8+te`` LOADER casts for a transformers
    text encoder — derived by meta-instantiating the checkpoint's
    architecture and running the SAME block-window selection the runtime
    applies (:func:`gen_worker.models.loading._fp8_block_windows`, gw#460),
    then mapping the component's stored key names onto the module graph with
    transformers' own checkpoint-conversion machinery. Using the loader's
    graph walk + the loader's key translation is the zero-drift contract:
    the stored artifact is byte-identical to what cast-at-load produces.

    Block-window rules (gw#460): castable = Linear/conv WEIGHTS inside the
    children of top-level ``nn.ModuleList`` containers, excluding params
    shared with modules outside a block (tied lm_head / embeddings).
    Embeddings, norms, biases, poolers stay at source precision."""
    import torch
    import transformers

    from gen_worker.models.loading import (
        _fp8_block_windows,
        _fp8_block_windows_whole,
    )

    component_dir = Path(component_dir)
    cfg = transformers.AutoConfig.from_pretrained(str(component_dir))
    archs = list(getattr(cfg, "architectures", None) or [])
    cls = getattr(transformers, archs[0], None) if archs else None
    if cls is None:
        raise ConversionImplementationError(
            f"cannot resolve transformers architecture for {component_dir} "
            f"(architectures={archs})")
    with torch.device("meta"):
        model = cls(cfg)
    windows = _fp8_block_windows(model) or _fp8_block_windows_whole(model)
    if not windows:
        raise ConversionImplementationError(
            f"no fp8-castable weights in {component_dir} ({archs[0]})")
    castable = {id(p) for _, _, params in windows for p in params}
    graph_keys = {
        n for n, p in model.named_parameters() if id(p) in castable}

    stored = _component_stored_tensor_names(component_dir)
    if not stored:
        raise ConversionImplementationError(
            f"no safetensors tensor names found in {component_dir}")
    translate = _loader_key_translator(model)
    matched = frozenset(k for k in stored if translate(k) in graph_keys)
    if not matched:
        raise ConversionImplementationError(
            f"fp8+te key translation matched nothing in {component_dir} "
            f"({archs[0]}: {len(graph_keys)} castable graph keys, "
            f"{len(stored)} stored keys) — layout drift vs the loader")
    return matched


def streaming_fp8_te_cast(
    input_path: Path,
    out_dir: Path,
    *,
    castable_keys: frozenset[str],
    shard_prefix: str = "model",
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
) -> dict[str, Any]:
    """fp8-E4M3 storage cast of one transformers weight set: exactly the
    ``castable_keys`` (the loader's block-window weight set) become
    F8_E4M3 (clamp ±448 first — torch's cast does not saturate); every
    other tensor keeps its source dtype byte-identically."""
    import torch

    def out_st_dtype_for(name: str, src_st: str, shape: list[int]) -> str:
        if name in castable_keys and src_st in {"F64", "F32", "F16", "BF16"}:
            return "F8_E4M3"
        return src_st

    def transform(_name: str, t: "torch.Tensor", out_st: str) -> "torch.Tensor":
        if out_st == "F8_E4M3" and t.dtype != torch.float8_e4m3fn:
            return t.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return t

    return _stream_reencode(
        Path(input_path), Path(out_dir),
        out_st_dtype_for=out_st_dtype_for, transform=transform,
        shard_prefix=shard_prefix, shard_threshold=shard_threshold,
    )


def streaming_fp8_snapshot(
    source_dir: Path,
    out_dir: Path,
    *,
    file_layout: str,
    components: tuple[str, ...] = FP8_DEFAULT_COMPONENTS,
    te_components: tuple[str, ...] = (),
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
) -> dict[str, Any]:
    """Produce the ``#fp8`` flavor of a diffusers snapshot, streaming.

    ``components`` (default: the denoiser) get the name-pattern fp8 cast;
    ``te_components`` (the ``fp8+te`` rung, gw#460 — pass
    :data:`FP8_TE_COMPONENTS`) get the transformers block-window cast whose
    eligible set is derived from the loader itself. Every other component
    passes through untouched.

    Non-diffusers layouts are supported for exactly ONE shape: a single
    root weight set (sharded-transformers backbone — the whole checkpoint
    IS the denoiser, e.g. a UiT like HiDream-O1). That set gets the
    block-scoped fp8 cast; multi-set singlefile bundles still refuse
    (component identity is ambiguous there)."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if file_layout != "diffusers":
        root_groups = snapshot_weight_groups(source_dir, file_layout)
        if len(root_groups) != 1 or root_groups[0][0] != "":
            raise ConversionImplementationError(
                "fp8 storage flavors need component identity: a diffusers "
                "layout, or a single root weight set (transformers "
                f"backbone) — found {len(root_groups)} weight set(s) in "
                f"{file_layout!r} layout")
        entry = root_groups[0][1]
        result = streaming_fp8_storage_cast(
            entry, out_dir,
            shard_prefix=_group_shard_prefix(entry),
            shard_threshold=shard_threshold,
            block_scope=True,
        )
        if not int(result["converted_count"]):
            raise ConversionImplementationError(
                "no fp8-castable weights in the root weight set (nothing "
                "under a repeated-block container missed the skip patterns)")
        copy_non_weight_files(source_dir, out_dir, skip_components={""})
        return {"tensor_count": int(result["tensor_count"]),
                "converted_count": int(result["converted_count"]),
                "components": [""], "output_dir": out_dir}
    denoiser_set, te_set = set(components), set(te_components)
    groups = [(c, e) for c, e in snapshot_weight_groups(source_dir, "diffusers")
              if c in denoiser_set | te_set]
    if not groups:
        raise ConversionImplementationError(
            f"no fp8-castable components found "
            f"(looked for {sorted(denoiser_set | te_set)})")
    tensor_count = converted = 0
    done: set[str] = set()
    for comp, entry in groups:
        if comp in te_set:
            result = streaming_fp8_te_cast(
                entry, out_dir / comp,
                castable_keys=te_fp8_castable_keys(source_dir / comp),
                shard_prefix=_group_shard_prefix(entry),
                shard_threshold=shard_threshold,
            )
        else:
            result = streaming_fp8_storage_cast(
                entry, out_dir / comp,
                shard_prefix=_group_shard_prefix(entry),
                shard_threshold=shard_threshold,
            )
        tensor_count += int(result["tensor_count"])
        converted += int(result["converted_count"])
        done.add(comp)
    copy_non_weight_files(source_dir, out_dir, skip_components=done)
    return {"tensor_count": tensor_count, "converted_count": converted,
            "components": sorted(done), "output_dir": out_dir}


def streaming_w8a8_snapshot(
    source_dir: Path,
    out_dir: Path,
    *,
    file_layout: str,
    components: tuple[str, ...] = FP8_DEFAULT_COMPONENTS,
    te_components: tuple[str, ...] = (),
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
) -> dict[str, Any]:
    """Produce the ``#fp8-w8a8`` flavor of a diffusers snapshot, streaming.

    The denoiser (``components``) gets the per-channel-scaled w8a8 requant
    (:func:`streaming_w8a8_cast`); ``te_components`` ride the SAME fp8+te
    block-window storage cast the ``#fp8`` flavor uses (scale-free — text
    encoders serve W8-storage, only the denoiser runs fp8 GEMMs). Every
    other component passes through untouched. The denoiser's config gains
    the gw#534 corroborating ``quantization_config`` (the safetensors
    headers stay authoritative for detection).

    Non-diffusers layouts (gw#562) require a SINGLE root weight set — the
    sharded-transformers/singlefile trees the DiffSynth families mirror,
    where that weight set IS the denoiser. It gets the same requant; no
    component config exists, so the headers alone carry detection."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if file_layout != "diffusers":
        if te_components:
            raise ConversionImplementationError(
                "te_components need a diffusers layout (no component "
                f"identity in {file_layout!r})")
        root_groups = snapshot_weight_groups(source_dir, file_layout)
        if len(root_groups) != 1 or root_groups[0][0] != "":
            raise ConversionImplementationError(
                "w8a8 flavors need component identity: a diffusers layout, "
                "or a single root weight set — found "
                f"{len(root_groups)} weight set(s) in {file_layout!r} layout")
        entry = root_groups[0][1]
        result = streaming_w8a8_cast(
            entry, out_dir,
            shard_prefix=_group_shard_prefix(entry),
            shard_threshold=shard_threshold,
        )
        if not int(result["converted_count"]):
            raise ConversionImplementationError(
                "no w8a8-eligible weights in the root weight set (nothing "
                "2-D/16-aligned under a repeated-block container missed "
                "the skip patterns)")
        copy_non_weight_files(source_dir, out_dir, skip_components={""})
        return {"tensor_count": int(result["tensor_count"]),
                "converted_count": int(result["converted_count"]),
                "components": [""], "output_dir": out_dir}
    denoiser_set, te_set = set(components), set(te_components)
    groups = [(c, e) for c, e in snapshot_weight_groups(source_dir, "diffusers")
              if c in denoiser_set | te_set]
    if not any(c in denoiser_set for c, _ in groups):
        raise ConversionImplementationError(
            f"no w8a8-quantizable denoiser found (looked for {sorted(denoiser_set)})")
    tensor_count = converted = 0
    done: set[str] = set()
    quantized_components: list[str] = []
    for comp, entry in groups:
        if comp in denoiser_set:
            result = streaming_w8a8_cast(
                entry, out_dir / comp,
                shard_prefix=_group_shard_prefix(entry),
                shard_threshold=shard_threshold,
            )
            if not int(result["converted_count"]):
                raise ConversionImplementationError(
                    f"no w8a8-eligible weights in component {comp!r} "
                    "(nothing 2-D/16-aligned under a repeated-block "
                    "container missed the skip patterns)")
            quantized_components.append(comp)
        else:
            result = streaming_fp8_te_cast(
                entry, out_dir / comp,
                castable_keys=te_fp8_castable_keys(source_dir / comp),
                shard_prefix=_group_shard_prefix(entry),
                shard_threshold=shard_threshold,
            )
        tensor_count += int(result["tensor_count"])
        converted += int(result["converted_count"])
        done.add(comp)
    copy_non_weight_files(source_dir, out_dir, skip_components=done)
    for comp in quantized_components:
        src_cfg = source_dir / comp / "config.json"
        cfg = json.loads(src_cfg.read_text("utf-8")) if src_cfg.exists() else {}
        cfg["quantization_config"] = {
            "quant_method": "modelopt", "quant_algo": "FP8"}
        dst_cfg = out_dir / comp / "config.json"
        if dst_cfg.exists():
            dst_cfg.unlink()  # hardlinked to the source — never write through
        dst_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return {"tensor_count": tensor_count, "converted_count": converted,
            "components": sorted(done), "output_dir": out_dir}


def verify_w8a8_snapshot(
    source_dir: Path,
    out_dir: Path,
    *,
    sample: int = 16,
    seed: int = 0,
    source_compute_dtype: str = "storage",
) -> dict[str, Any]:
    """Byte-gate a produced w8a8 tree against its source (gw#557 / ie#494).

    Detection runs through the CONSUMER's own sniffer
    (:func:`gen_worker.models.w8a8.detect_w8a8_artifact`), then for a
    deterministic sample of quantized layers: (a) the recomputed
    quant+scale bytes must be EXACTLY the artifact's (the recipe is
    deterministic — any drift is corruption or a recipe change), and
    (b) ``dequant(w_fp8 x scale)`` must sit within fp8-e4m3 format error of
    the source (max row-relative error <= 2**-4 + subnormal floor).
    Raises :class:`ConversionImplementationError` on any failure; returns a
    report dict for produce logs. ``source_compute_dtype`` names the exact
    producer-side quantization input view. For example, an immutable FP16
    checkpoint loaded as production BF16 is cast to BF16 before exact
    recomputation; storage and compute dtypes are both reported."""
    import random

    import torch
    from safetensors import safe_open

    from gen_worker.models.w8a8 import detect_w8a8_artifact

    aliases = {
        "storage": ("storage", None),
        "": ("storage", None),
        "bf16": ("bfloat16", torch.bfloat16),
        "bfloat16": ("bfloat16", torch.bfloat16),
        "fp16": ("float16", torch.float16),
        "float16": ("float16", torch.float16),
        "fp32": ("float32", torch.float32),
        "float32": ("float32", torch.float32),
    }
    dtype_key = str(source_compute_dtype or "").strip().lower()
    if dtype_key not in aliases:
        raise ConversionImplementationError(
            "byte-gate: source_compute_dtype must be one of storage, bf16, "
            f"fp16, or fp32; got {source_compute_dtype!r}"
        )
    canonical_compute_dtype, cast_dtype = aliases[dtype_key]

    source_dir, out_dir = Path(source_dir), Path(out_dir)
    art = detect_w8a8_artifact(out_dir)
    if art is None:
        raise ConversionImplementationError(
            f"byte-gate: {out_dir} does not detect as a w8a8 artifact")

    def _tensor_map(files: list[Path]) -> dict[str, Path]:
        where: dict[str, Path] = {}
        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    where[k] = f
        return where

    out_where = _tensor_map(list(art.files))
    src_where = _tensor_map(sorted(
        p for p in (source_dir / art.component).glob("*.safetensors")
        if p.is_file()))

    names = list(art.quantized)
    rng = random.Random(seed)
    picked = names if len(names) <= sample else sorted(rng.sample(names, sample))
    max_rel = 0.0
    source_storage_dtypes: set[str] = set()
    for layer in picked:
        wname, sname = f"{layer}.weight", f"{layer}{_SCALE_SUFFIX}"
        src_file = src_where.get(wname)
        if src_file is None:
            raise ConversionImplementationError(
                f"byte-gate: quantized layer {wname!r} missing from source")
        with safe_open(str(src_file), framework="pt", device="cpu") as fh:
            stored = fh.get_tensor(wname)
        source_storage_dtypes.add(str(stored.dtype).removeprefix("torch."))
        if cast_dtype is not None:
            stored = stored.to(dtype=cast_dtype)
        src = stored.float()
        with safe_open(str(out_where[wname]), framework="pt", device="cpu") as fh:
            got_q = fh.get_tensor(wname)
        with safe_open(str(out_where[sname]), framework="pt", device="cpu") as fh:
            got_s = fh.get_tensor(sname).float()
        scale = (src.abs().amax(dim=1, keepdim=True)
                 / _FP8_E4M3_MAX).clamp(min=1e-12)
        want_q = (src / scale).clamp(
            -_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        if not torch.equal(want_q.view(torch.uint8), got_q.view(torch.uint8)):
            raise ConversionImplementationError(
                f"byte-gate: {wname} recomputed fp8 bytes differ from artifact")
        if not torch.equal(scale.reshape(-1), got_s.reshape(-1)):
            raise ConversionImplementationError(
                f"byte-gate: {sname} recomputed scales differ from artifact")
        deq = got_q.float() * got_s.reshape(-1, 1)
        row_amax = src.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        rel = ((deq - src).abs() / row_amax).max().item()
        # e4m3 half-ulp relative error at worst 2**-4 for normals, plus the
        # subnormal quantization floor for near-zero elements.
        if rel > 2 ** -4 + 2 ** -9:
            raise ConversionImplementationError(
                f"byte-gate: {wname} dequant error {rel:.5f} exceeds the "
                "fp8-e4m3 format bound")
        max_rel = max(max_rel, rel)
    return {
        "component": art.component,
        "quantized_total": len(names),
        "sampled": len(picked),
        "byte_exact": True,
        "max_rel_err": max_rel,
        "source_storage_dtypes": sorted(source_storage_dtypes),
        "source_compute_dtype": canonical_compute_dtype,
    }


# ---------------------------------------------------------------------------
# Raw byte-offset re-shard (no tensor decode)
# ---------------------------------------------------------------------------

_HEADER_LEN_PREFIX = 8
_MAX_HEADER_BYTES = 512 * 1024 * 1024
_RAW_COPY_CHUNK = 8 * 1024 * 1024


def _read_safetensors_header(fd: int) -> tuple[dict, int]:
    import os

    os.lseek(fd, 0, os.SEEK_SET)
    prefix = os.read(fd, _HEADER_LEN_PREFIX)
    if len(prefix) != _HEADER_LEN_PREFIX:
        raise ValueError("safetensors: short read on header length prefix")
    header_len = int.from_bytes(prefix, "little")
    if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"safetensors: implausible header_length={header_len}")
    body = os.read(fd, header_len)
    if len(body) != header_len:
        raise ValueError("safetensors: short read on header body")
    header = json.loads(body.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors: header root must be an object")
    return header, _HEADER_LEN_PREFIX + header_len


def shard_safetensors_by_offset(
    src_path: Path,
    out_dir: Path,
    *,
    max_shard_bytes: int = MAX_SAFETENSORS_SHARD_BYTES,
    shard_prefix: str = "model",
) -> tuple[list[Path], Path, dict[str, int]]:
    """Split an oversized safetensors file into HF-convention shards by raw
    byte-range copy — the input never enters Python as a tensor.

    Returns (shard_paths, index_path, shard_sizes). The index is always
    written; callers skip uploading it for single-shard plans.
    """
    import os

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(src_path), os.O_RDONLY)
    try:
        header, data_base = _read_safetensors_header(fd)
        sizes: dict[str, int] = {}
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            offs = meta.get("data_offsets")
            if not isinstance(offs, list) or len(offs) != 2 or int(offs[1]) < int(offs[0]):
                raise ValueError(f"safetensors: tensor {name!r} has invalid data_offsets")
            sizes[name] = int(offs[1]) - int(offs[0])

        plan = plan_shards(sizes, max_shard_bytes=max_shard_bytes, shard_prefix=shard_prefix)
        reserved_md = header.get("__metadata__")

        groups: dict[str, list[tuple[str, int, int]]] = {n: [] for n in plan.shard_names}
        for tname, shard_name in plan.weight_map.items():
            offs = header[tname]["data_offsets"]
            groups[shard_name].append((tname, int(offs[0]), int(offs[1])))
        for g in groups.values():
            g.sort(key=lambda r: r[1])  # sequential source reads

        shard_paths: list[Path] = []
        for shard_name in plan.shard_names:
            entries = groups[shard_name]
            new_header: dict[str, Any] = {}
            if isinstance(reserved_md, dict):
                new_header["__metadata__"] = dict(reserved_md)
            cursor = 0
            for tname, s, e in entries:
                src_meta = header[tname]
                new_header[tname] = {
                    "dtype": src_meta["dtype"],
                    "shape": list(src_meta["shape"]),
                    "data_offsets": [cursor, cursor + (e - s)],
                }
                cursor += e - s
            blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
            shard_path = out_dir / shard_name
            dst = os.open(str(shard_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(dst, len(blob).to_bytes(_HEADER_LEN_PREFIX, "little"))
                os.write(dst, blob)
                for tname, s, e in entries:
                    remaining = e - s
                    src_abs = data_base + s
                    while remaining > 0:
                        buf = os.pread(fd, min(remaining, _RAW_COPY_CHUNK), src_abs)
                        if not buf:
                            raise IOError(f"safetensors: short read on {tname!r} at {src_abs}")
                        os.write(dst, buf)
                        remaining -= len(buf)
                        src_abs += len(buf)
            finally:
                os.close(dst)
            shard_paths.append(shard_path)

        index_path = out_dir / f"{shard_prefix}.safetensors.index.json"
        index_path.write_text(
            json.dumps(build_index(plan), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        return shard_paths, index_path, dict(plan.shard_sizes)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Canonical published filenames (gw#466 / gw#522)
# ---------------------------------------------------------------------------

CAST_NORMALIZE_DTYPES = {"fp16", "bf16", "fp32", "f16", "f32"}

VARIANT_WEIGHT_NAME_RE = re.compile(
    r"^(?P<base>.+)\.(?P<v>fp16|bf16|fp32)"
    r"(?P<shard>-\d{5}-of-\d{5})?\.safetensors(?P<idx>\.index\.json)?$"
)


def normalize_variant_filenames(tree: Path) -> None:
    """Strip dtype-variant tokens from published weight filenames — the ONE
    canonical-naming pass every publish path runs (gw#466, unified by gw#522).

    dtype is a checkpoint axis (flavor) in repo-cas — one dtype per tree — so
    HF variant suffixes are redundant, and the resharder composes an index
    name diffusers cannot find (live twice: J23 juggernaut-xl, gw#522
    sdxl-base — "diffusion_pytorch_model.fp16.safetensors.index.json" where
    diffusers' _add_variant expects
    "diffusion_pytorch_model.safetensors.index.fp16.json"; loads died).
    Canonical names sidestep the class. A directory whose canonical twin
    already exists is left untouched (dual-dtype upstream trees must not
    collide). Quant flavors (fp8/int4/…) are never normalized — their names
    carry loader semantics. Idempotent."""
    dirs = sorted({p.parent for p in Path(tree).rglob("*.safetensors*") if p.is_file()})
    for d in dirs:
        renames: dict[str, str] = {}
        for p in sorted(d.iterdir()):
            m = VARIANT_WEIGHT_NAME_RE.match(p.name)
            if not m:
                continue
            new_name = f"{m['base']}{m['shard'] or ''}.safetensors{m['idx'] or ''}"
            if (d / new_name).exists():
                renames.clear()
                break
            renames[p.name] = new_name
        for old_name, new_name in renames.items():
            (d / old_name).rename(d / new_name)
        if not renames:
            continue
        for idx in d.glob("*.safetensors.index.json"):
            try:
                payload = json.loads(idx.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                continue
            changed = False
            for key, shard in weight_map.items():
                if shard in renames:
                    weight_map[key] = renames[shard]
                    changed = True
            if changed:
                idx.write_text(
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )


__all__ = [
    "ConversionImplementationError",
    "CAST_NORMALIZE_DTYPES",
    "MAX_SAFETENSORS_SHARD_BYTES",
    "VARIANT_WEIGHT_NAME_RE",
    "normalize_variant_filenames",
    "ShardPlan",
    "plan_shards",
    "build_index",
    "list_shard_files_from_index",
    "IncrementalSafetensorsWriter",
    "torch_dtype_to_st",
    "materialize_pickle_to_safetensors",
    "materialize_safetensors_input",
    "iter_component_tensors",
    "iter_source_tensors",
    "streaming_dtype_cast",
    "streaming_fp8_storage_cast",
    "streaming_cast_snapshot",
    "streaming_fp8_snapshot",
    "W8A8_QUANT_SCHEME",
    "W8A8_SKIP_TENSOR_PATTERNS",
    "w8a8_cast_eligible",
    "streaming_w8a8_cast",
    "streaming_w8a8_snapshot",
    "verify_w8a8_snapshot",
    "snapshot_weight_groups",
    "copy_non_weight_files",
    "fp8_cast_eligible",
    "FP8_SKIP_TENSOR_PATTERNS",
    "FP8_DEFAULT_COMPONENTS",
    "FP8_TE_COMPONENTS",
    "te_fp8_castable_keys",
    "streaming_fp8_te_cast",
    "shard_safetensors_by_offset",
]
