"""The ONE streaming safetensors shard writer.

Collapses gen_worker.conversion's seven IO modules (writer, streaming_primitives,
_sharding, _tensor_iter, _streaming_incremental, _streaming_convert,
safetensors_io) into one:

  - shard planning (HF 5 GB convention) + index.json emit
  - IncrementalSafetensorsWriter: header first, then tensor bytes — O(1) memory
  - tensor iteration over single/sharded/pickle sources
  - streaming_dtype_cast: read one tensor, cast, write into its shard
  - shard_safetensors_by_offset: raw byte-range re-shard, zero decode
  - StreamingWriter: the tenant-facing per-variant output writer

torch/safetensors imports are deferred so importing cozy_convert stays cheap.
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional

if TYPE_CHECKING:
    import torch

    from .component import Component
    from .source import Source


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
# that ceiling regardless of R2 throughput variance. cozy_convert.hub's
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
    """Write a safetensors file one tensor at a time (no full dict in memory)."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = Path(output_path)
        self._meta: list[tuple[str, str, list[int]]] = []  # (name, st_dtype, shape)
        self._header_written = False
        self._fh: Any = None
        self._written: set[str] = set()

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
            offset += size
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((8 - (len(blob) % 8)) % 8)
        self._fh.write(struct.pack("<Q", len(blob)))
        self._fh.write(blob)
        self._header_written = True

    def write_tensor(self, name: str, data: bytes) -> None:
        if not self._header_written or self._fh is None:
            raise RuntimeError("write_header() must run before write_tensor()")
        expected = self._meta[len(self._written)][0]
        if name != expected:
            raise RuntimeError(f"expected tensor {expected!r}, got {name!r} (write in order)")
        if name in self._written:
            raise RuntimeError(f"tensor {name!r} already written")
        self._fh.write(data)
        self._written.add(name)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _tensor_to_bytes(t: "torch.Tensor") -> bytes:
    """Contiguous little-endian raw bytes via one C-level memcpy."""
    import torch

    return t.contiguous().flatten().view(torch.uint8).numpy().tobytes()


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
# Streaming dtype cast — one tensor in memory at a time, direct-to-shard
# ---------------------------------------------------------------------------

def streaming_dtype_cast(
    input_path: Path,
    out_dir: Path,
    *,
    target_dtype: "torch.dtype",
    shard_prefix: str = "model",
    shard_threshold: int = MAX_SAFETENSORS_SHARD_BYTES,
) -> dict[str, Any]:
    """Cast float tensors to ``target_dtype``, writing directly into N shards.

    Non-float tensors keep their source dtype. Returns ``output_paths``
    (list of shard files), ``index_path`` (None for a single shard),
    ``tensor_count`` / ``converted_count`` / ``shard_sizes``.
    """
    from safetensors import safe_open

    shards_in = _resolve_input_shards(Path(input_path))
    target_st = torch_dtype_to_st(target_dtype)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: metadata walk — output dtype + byte size per tensor.
    metas: list[tuple[str, str, list[int], Path]] = []  # name, out_dtype, shape, src_shard
    size_map: dict[str, int] = {}
    elem_size = _ST_DTYPE_SIZES[target_st]
    for shard_path in shards_in:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if t.is_floating_point():
                    out_dtype = target_st
                    nbytes = int(t.numel()) * elem_size
                else:
                    out_dtype = torch_dtype_to_st(t.dtype)
                    nbytes = int(t.numel() * t.element_size())
                metas.append((name, out_dtype, list(t.shape), shard_path))
                size_map[name] = nbytes
                del t

    plan = plan_shards(size_map, max_shard_bytes=shard_threshold, shard_prefix=shard_prefix)
    per_shard: dict[str, list[tuple[str, str, list[int], Path]]] = {n: [] for n in plan.shard_names}
    for row in metas:
        dest = plan.weight_map.get(row[0])
        if dest is not None:
            per_shard[dest].append(row)

    tensor_count = 0
    converted = 0
    for shard_name in plan.shard_names:
        entries = per_shard[shard_name]
        with IncrementalSafetensorsWriter(out_dir / shard_name) as w:
            for name, out_dtype, shape, _src in entries:
                w.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
            w.write_header()
            for name, _out_dtype, _shape, src in entries:
                with safe_open(str(src), framework="pt", device="cpu") as f:
                    t = f.get_tensor(name)
                tensor_count += 1
                if t.is_floating_point():
                    result = t.to(dtype=target_dtype)
                    converted += 1
                else:
                    result = t
                w.write_tensor(name, _tensor_to_bytes(result))
                del t, result

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
# StreamingWriter — the tenant-facing per-variant output writer
# ---------------------------------------------------------------------------

_PASSTHROUGH_FILES = frozenset({
    "model_index.json", "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "scheduler_config.json", "preprocessor_config.json",
    "generation_config.json", "tokenizer.model",
})


class StreamingWriter:
    """Per-variant output writer for conversion tenants.

    Tenants iterate ``source.iter_tensors()`` and ``write(component, name,
    tensor)`` each output tensor. ``finalize()`` flushes per-component
    safetensors (auto-sharded past 5 GB), auto-passes-through untouched
    components + known config/tokenizer files, and returns the output path.
    """

    def __init__(self, *, source: "Source", out_dir: Path) -> None:
        self._source = source
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._by_component: dict[str, dict[str, Any]] = {}
        self._written_components: set[str] = set()
        self._explicit_passthrough: set[str] = set()
        self._finalized = False

    def write(self, component: str, name: str, tensor: "torch.Tensor") -> None:
        if self._finalized:
            raise RuntimeError("StreamingWriter.write called after finalize()")
        self._by_component.setdefault(component, {})[name] = tensor
        self._written_components.add(component)

    def passthrough(self, component: "Component") -> None:
        if self._finalized:
            raise RuntimeError("StreamingWriter.passthrough called after finalize()")
        self._explicit_passthrough.add(component.name)

    def finalize(self) -> Path:
        if self._finalized:
            raise RuntimeError("StreamingWriter.finalize called twice")
        self._finalized = True

        if self._source.file_layout == "singlefile":
            written = self._write_safetensors(
                self._out_dir / "model.safetensors", self._by_component.get("", {}))
            self._copy_passthrough_files(self._source.path, self._out_dir)
            if written is None:
                return self._out_dir
            return written if written.suffix == ".safetensors" else self._out_dir

        for comp_name, tensors in self._by_component.items():
            if not comp_name:
                raise RuntimeError(
                    "StreamingWriter.write called with empty component on a "
                    "diffusers source — name the component (unet, vae, transformer, ...)")
            self._write_component(comp_name, tensors)

        source_components = set(self._source.components.keys())
        touched = self._written_components | self._explicit_passthrough
        for comp_name in source_components - touched:
            src = self._source.components[comp_name].path
            dst = self._out_dir / comp_name
            if not dst.exists():
                shutil.copytree(str(src), str(dst))
        for entry in self._source.path.iterdir():
            if entry.is_dir() and entry.name not in source_components:
                dst = self._out_dir / entry.name
                if not dst.exists():
                    shutil.copytree(str(entry), str(dst))
        self._copy_passthrough_files(self._source.path, self._out_dir)
        return self._out_dir

    # ---- internals ----

    def _write_component(self, comp_name: str, tensors: dict[str, Any]) -> None:
        subdir = self._out_dir / comp_name
        subdir.mkdir(parents=True, exist_ok=True)
        if comp_name.startswith("text_encoder") or comp_name == "image_encoder":
            prefix = "model"
        else:
            prefix = "diffusion_pytorch_model"
        self._write_safetensors(subdir, tensors, shard_prefix=prefix)
        self._copy_passthrough_files(self._source.components[comp_name].path, subdir)

    def _write_safetensors(
        self, target: Path, tensors: dict[str, Any], *, shard_prefix: str = "model",
    ) -> Optional[Path]:
        if not tensors:
            return None
        from safetensors.torch import save_file

        def _size(t: Any) -> int:
            try:
                return int(t.numel()) * int(t.element_size())
            except AttributeError:
                return int(getattr(t, "nbytes"))

        sizes = {name: _size(t) for name, t in tensors.items()}
        if sum(sizes.values()) <= MAX_SAFETENSORS_SHARD_BYTES:
            if target.suffix == ".safetensors":
                out = target
                out.parent.mkdir(parents=True, exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                out = target / f"{shard_prefix}.safetensors"
            save_file(tensors, str(out))
            return out

        shard_dir = target.parent if target.suffix == ".safetensors" else target
        shard_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_shards(sizes, shard_prefix=shard_prefix)
        buckets: dict[str, dict[str, Any]] = {}
        for name, tensor in tensors.items():
            buckets.setdefault(plan.weight_map[name], {})[name] = tensor
        for shard_name, bucket in buckets.items():
            save_file(bucket, str(shard_dir / shard_name))
        index_path = shard_dir / f"{shard_prefix}.safetensors.index.json"
        index_path.write_text(json.dumps(build_index(plan), separators=(",", ":")))
        return index_path

    def _copy_passthrough_files(self, src_dir: Path, dst_dir: Path) -> None:
        if not src_dir.is_dir():
            return
        for entry in src_dir.iterdir():
            if entry.is_file() and entry.name in _PASSTHROUGH_FILES:
                dst = dst_dir / entry.name
                if not dst.exists():
                    shutil.copy2(str(entry), str(dst))


__all__ = [
    "ConversionImplementationError",
    "MAX_SAFETENSORS_SHARD_BYTES",
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
    "shard_safetensors_by_offset",
    "StreamingWriter",
]
