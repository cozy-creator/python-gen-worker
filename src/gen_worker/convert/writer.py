"""The ONE streaming safetensors writer."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import struct
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Sequence

from gen_worker.models.loading import _fp8_block_windows, _fp8_block_windows_whole
from gen_worker.models.safetensors_header import header_len_ok
from gen_worker.models.w8a8 import detect_w8a8_artifact

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ConversionImplementationError(RuntimeError):
    """A conversion primitive can't proceed (bad input, missing dep)."""


_PICKLE_EXTS = (".ckpt", ".pt", ".pth", ".bin")

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


class IncrementalSafetensorsWriter:
    """Write a safetensors file one tensor at a time (no full dict in memory)."""

    def __init__(self, output_path: Path, *, metadata: Mapping[str, str] | None = None) -> None:
        self._output_path = Path(output_path)
        self._temp_path = self._output_path.with_name(f".{self._output_path.name}.partial")
        self._meta: list[tuple[str, str, list[int]]] = []
        self._metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        self._header_written = False
        self._fh: Any = None
        self._written: set[str] = set()
        self._sizes: list[int] = []

    def __enter__(self) -> "IncrementalSafetensorsWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(commit=exc_type is None)

    def add_tensor_metadata(self, name: str, *, dtype: str, shape: list[int]) -> None:
        if self._header_written:
            raise RuntimeError("cannot add metadata after header is written")
        self._meta.append((name, dtype, list(shape)))

    def write_header(self) -> None:
        if self._header_written:
            raise RuntimeError("header already written")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._temp_path, "wb")
        header: dict[str, Any] = {}
        if self._metadata:
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

    def close(self, *, commit: bool = True) -> None:
        """Durably finalize (or discard) the output."""
        if self._fh is None:
            self._discard()
            return
        try:
            self._fh.flush()
        finally:
            self._fh.close()
            self._fh = None
        complete = len(self._written) == len(self._meta)
        if not commit or not complete:
            if not complete and commit:
                logger.warning(
                    "safetensors writer: %d of %d tensors written — discarding %s",
                    len(self._written), len(self._meta), self._output_path)
            self._discard()
            return
        _fsync_file(self._temp_path)
        os.replace(self._temp_path, self._output_path)
        _fsync_dir(self._output_path.parent)

    def _discard(self) -> None:
        try:
            self._temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _tensor_to_bytes(t: "torch.Tensor") -> Any:
    import torch

    return t.contiguous().flatten().view(torch.uint8).numpy()


from ..models.file_layout import MULTI_FILE, SINGLE_FILE
from ..component_vocab import (
    denoiser_components,
    text_encoder_components,
    weight_components,
)

def _weight_component_dirs() -> frozenset[str]:
    return frozenset(weight_components())


def _resolve_input_shards(input_path: Path) -> list[Path]:
    if input_path.name.lower().endswith(".safetensors.index.json"):
        return list_shard_files_from_index(input_path)
    return [input_path]


def iter_component_tensors(component_dir: Path) -> Iterator[tuple[str, "torch.Tensor"]]:
    """Yield (name, tensor) for every weight in one component directory."""
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
                raise ConversionImplementationError(
                    f"pickle_only:{found[0].name}: this component offers only "
                    f"a pickle, and pickles are refused. Mirror the source "
                    f"repo without the pickle (safetensors) and convert that."
                )
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
    if file_layout == SINGLE_FILE:
        for name, tensor in iter_component_tensors(root):
            yield "", name, tensor
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name not in _weight_component_dirs():
            continue
        if components_filter is not None and entry.name not in components_filter:
            continue
        for name, tensor in iter_component_tensors(entry):
            yield entry.name, name, tensor


_ST_FLOAT_DTYPES: frozenset[str] = frozenset(
    {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})


def stream_reencode(
    input_path: Path,
    out_dir: Path,
    *,
    out_st_dtype_for: Any,
    transform: Any,
    output_stem: str,
) -> dict[str, Any]:
    """Two-pass streaming re-encode over safetensors input(s)."""
    from safetensors import safe_open

    shards_in = _resolve_input_shards(Path(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas: list[tuple[str, str, list[int], Path]] = []
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

    metas.sort(key=lambda row: row[0])
    out_path = out_dir / f"{output_stem}.safetensors"

    tensor_count = 0
    converted = 0
    handles: dict[Path, Any] = {}
    try:
        with IncrementalSafetensorsWriter(out_path, metadata=source_metadata) as w:
            for name, out_dtype, shape, _src in metas:
                w.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
            w.write_header()
            for name, out_dtype, _shape, src in metas:
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

    assert_one_file_per_component(out_dir, producer="streaming re-encode")
    return {
        "tensor_count": tensor_count,
        "converted_count": converted,
        "output_paths": [out_path],
        "metadata": dict(source_metadata),
    }


def streaming_dtype_cast(
    input_path: Path,
    out_dir: Path,
    *,
    target_dtype: "torch.dtype",
    output_stem: str = "model",
) -> dict[str, Any]:
    """Cast float tensors to ``target_dtype``, streaming directly into N shards."""
    target_st = torch_dtype_to_st(target_dtype)

    def out_st_dtype_for(_name: str, src_st: str, _shape: list[int]) -> str:
        return target_st if src_st in _ST_FLOAT_DTYPES else src_st

    def transform(_name: str, t: "torch.Tensor", _out_st: str) -> "torch.Tensor":
        if t.is_floating_point() and t.dtype != target_dtype:
            return t.to(dtype=target_dtype)
        return t

    return stream_reencode(
        Path(input_path), Path(out_dir),
        out_st_dtype_for=out_st_dtype_for, transform=transform,
        output_stem=output_stem,
    )


FP8_SKIP_TENSOR_PATTERNS: tuple[str, ...] = (
    "embed",
    "norm",
    "pooler",
    "adaln_single",
    "final_layer",
    "quantize",
    "decoder",
    "preprocess_conv", "postprocess_conv",
    r"^proj_in$", r"^proj_out$", r"^proj$",
)

_FP8_E4M3_MAX = 448.0


def fp8_cast_eligible(
    name: str, src_st_dtype: str, shape: list[int],
    *, skip_patterns: tuple[str, ...] = FP8_SKIP_TENSOR_PATTERNS,
) -> bool:
    """True when a tensor is safe to store as fp8-E4M3 for the ``#fp8`` flavor."""

    if src_st_dtype not in {"F64", "F32", "F16", "BF16"}:
        return False
    if len(shape) < 2 or not name.endswith(".weight"):
        return False
    module_path = name[: -len(".weight")]
    return not any(re.search(p, module_path) for p in skip_patterns)


_FP8_BLOCK_SCOPE_RE = r"\.\d+\."


def _in_repeated_block(name: str) -> bool:

    return re.search(_FP8_BLOCK_SCOPE_RE, name) is not None


def streaming_fp8_storage_cast(
    input_path: Path,
    out_dir: Path,
    *,
    output_stem: str = "model",
    skip_patterns: tuple[str, ...] = FP8_SKIP_TENSOR_PATTERNS,
    block_scope: bool = False,
) -> dict[str, Any]:
    """Produce the fp8-E4M3 storage flavor of one weight set, streaming."""
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

    return stream_reencode(
        Path(input_path), Path(out_dir),
        out_st_dtype_for=out_st_dtype_for, transform=transform,
        output_stem=output_stem,
    )


W8A8_QUANT_SCHEME = "fp8-w8a8"

W8A8_SKIP_TENSOR_PATTERNS: tuple[str, ...] = FP8_SKIP_TENSOR_PATTERNS + (
    "gate_logits",
)

_W8A8_DIM_ALIGN = 16
_SCALE_SUFFIX = ".weight_scale"


def w8a8_cast_eligible(
    name: str, src_st_dtype: str, shape: list[int],
    *, skip_patterns: tuple[str, ...] = W8A8_SKIP_TENSOR_PATTERNS,
) -> bool:
    """True when a stored tensor becomes a quantized w8a8 Linear weight: a 2-D float ``.weight`` under a repeated-block container, both dims 16-aligned, missing every skip pattern."""
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
    output_stem: str = "model",
    skip_patterns: tuple[str, ...] = W8A8_SKIP_TENSOR_PATTERNS,
) -> dict[str, Any]:
    """Per-channel-scaled fp8 requant of one weight set, streaming."""
    import torch
    from safetensors import safe_open

    shards_in = _resolve_input_shards(Path(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    metas.sort(key=lambda row: row[0])
    out_path = out_dir / f"{output_stem}.safetensors"

    tensor_count = 0
    pending_scales: dict[str, "torch.Tensor"] = {}
    handles: dict[Path, Any] = {}
    try:
        with IncrementalSafetensorsWriter(out_path, metadata=out_metadata) as w:
            for name, out_dtype, shape, _src in metas:
                w.add_tensor_metadata(name, dtype=out_dtype, shape=shape)
            w.write_header()
            for name, _out_dtype, _shape, src in metas:
                if src is None:
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

    assert_one_file_per_component(out_dir, producer="w8a8 cast")
    return {
        "tensor_count": tensor_count,
        "converted_count": len(quantized_names),
        "output_paths": [out_path],
        "metadata": out_metadata,
    }


def fp8_default_components() -> tuple[str, ...]:
    """Components fp8 storage targets by default: the denoisers."""
    return denoiser_components()


def snapshot_weight_groups(source_dir: Path, layout: str) -> list[tuple[str, Path]]:
    """(component, entry_path) per weight set."""
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

    if layout == MULTI_FILE:
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

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_non_weight_files(source_dir: Path, out_dir: Path, *, skip_components: set[str]) -> None:
    """Materialize the PASSTHROUGH half of a produced tree: hardlink every file the caller is not writing itself."""
    copied_indexes: list[Path] = []
    for f in sorted(source_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(source_dir)
        if rel.parts[:2] == (".cache", "huggingface"):
            continue
        comp = rel.parts[0] if len(rel.parts) > 1 else ""
        name = f.name
        is_weightish = f.suffix == ".safetensors" or name.endswith(".safetensors.index.json")
        if is_weightish and (comp in skip_components or ("" in skip_components and comp == "")):
            continue
        if name == ".civitai.json":
            continue
        _link_or_copy(f, out_dir / rel)
        if name.endswith(".safetensors.index.json"):
            copied_indexes.append(out_dir / rel)
    for index_path in copied_indexes:
        if index_path.is_file():
            deshard_indexed_safetensors(index_path)


_OBJECTIVE_SCHEDULER_OVERRIDES: dict[str, dict[str, Any]] = {
    "v_prediction": {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True},
}
_DISTILLED_SCHEDULER_OVERRIDES: dict[str, Any] = {"timestep_spacing": "trailing"}


def apply_objective_scheduler_config(
    out_dir: Path, objective: str, distilled: bool = False,
) -> None:
    """Stamp the checkpoint's objective/distilled scheduler overrides into a produced diffusers snapshot's ``scheduler/config.json``."""
    overrides: dict[str, Any] = dict(
        _OBJECTIVE_SCHEDULER_OVERRIDES.get(str(objective or ""), {}))
    if distilled:
        overrides.update(_DISTILLED_SCHEDULER_OVERRIDES)
    if not overrides:
        return
    cfg_path = Path(out_dir) / "scheduler" / "config.json"
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update(overrides)
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def component_output_stem(entry: Path) -> str:
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
) -> dict[str, Any]:
    """Streaming dtype cast of a whole snapshot: every weight group is cast per-tensor (peak anon RAM ≈ largest tensor); configs/tokenizers hardlink through."""
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
            output_stem=component_output_stem(entry),
        )
        tensor_count += int(result["tensor_count"])
        converted += int(result["converted_count"])
        done.add(comp)
    copy_non_weight_files(source_dir, out_dir, skip_components=done)
    return {"tensor_count": tensor_count, "converted_count": converted,
            "components": sorted(done), "output_dir": out_dir}


def fp8_te_components() -> tuple[str, ...]:
    return text_encoder_components()


def component_stored_tensor_names(component_dir: Path) -> frozenset[str]:
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
            if not header_len_ok(header_len):
                raise ValueError(
                    f"safetensors: implausible header_length={header_len} in {f.name}")
            header = json.loads(fh.read(header_len))
        names.update(k for k in header if k != "__metadata__")
    return frozenset(names)


def _loader_key_translator(model: Any) -> Any:
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightRenaming,
            rename_source_key,
        )
    except ImportError:
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
    """STORED tensor names the ``fp8+te`` LOADER casts for a transformers text encoder — derived by meta-instantiating the checkpoint's architecture and running the SAME block-window selection the runtime..."""
    import torch
    import transformers

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

    stored = component_stored_tensor_names(component_dir)
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
    output_stem: str = "model",
) -> dict[str, Any]:
    """fp8-E4M3 storage cast of one transformers weight set: exactly the ``castable_keys`` (the loader's block-window weight set) become F8_E4M3 (clamp ±448 first — torch's cast does not saturate); every ..."""
    import torch

    def out_st_dtype_for(name: str, src_st: str, shape: list[int]) -> str:
        if name in castable_keys and src_st in {"F64", "F32", "F16", "BF16"}:
            return "F8_E4M3"
        return src_st

    def transform(_name: str, t: "torch.Tensor", out_st: str) -> "torch.Tensor":
        if out_st == "F8_E4M3" and t.dtype != torch.float8_e4m3fn:
            return t.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return t

    return stream_reencode(
        Path(input_path), Path(out_dir),
        out_st_dtype_for=out_st_dtype_for, transform=transform,
        output_stem=output_stem,
    )


def streaming_fp8_snapshot(
    source_dir: Path,
    out_dir: Path,
    *,
    file_layout: str,
    components: tuple[str, ...] | None = None,
    te_components: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Produce the ``#fp8`` flavor of a diffusers snapshot, streaming."""
    if components is None:
        components = fp8_default_components()
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if file_layout != MULTI_FILE:
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
            output_stem=component_output_stem(entry),
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
    groups = [(c, e) for c, e in snapshot_weight_groups(source_dir, MULTI_FILE)
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
                output_stem=component_output_stem(entry),
            )
        else:
            result = streaming_fp8_storage_cast(
                entry, out_dir / comp,
                output_stem=component_output_stem(entry),
            )
        tensor_count += int(result["tensor_count"])
        converted += int(result["converted_count"])
        done.add(comp)
    copy_non_weight_files(source_dir, out_dir, skip_components=done)
    return {"tensor_count": tensor_count, "converted_count": converted,
            "components": sorted(done), "output_dir": out_dir}


def _nested_weight_sets(d: Path) -> list[tuple[str, Path, tuple[Path, ...]]]:
    sets: list[tuple[str, Path, tuple[Path, ...]]] = []
    dirs = [d] + sorted(
        p for p in d.rglob("*")
        if p.is_dir()
        and not any(part.startswith(".") for part in p.relative_to(d).parts))
    for sub in dirs:
        sharded: set[str] = set()
        for idx in sorted(sub.glob("*.safetensors.index.json")):
            members: list[Path] = [idx]
            try:
                weight_map = json.loads(idx.read_text("utf-8")).get("weight_map") or {}
            except (OSError, ValueError):
                continue
            sharded.update(str(v) for v in weight_map.values())
            members += [sub / s for s in sorted(set(weight_map.values()))
                        if (sub / s).is_file()]
            sets.append((str(idx.relative_to(d)), idx, tuple(members)))
        for f in sorted(sub.glob("*.safetensors")):
            if f.is_file() and f.name not in sharded:
                sets.append((str(f.relative_to(d)), f, (f,)))
    return sets


def streaming_w8a8_snapshot(
    source_dir: Path,
    out_dir: Path,
    *,
    file_layout: str,
    components: tuple[str, ...] | None = None,
    te_components: tuple[str, ...] = (),
    weight_set_patterns: tuple[str, ...] = (),
    skip_patterns: tuple[str, ...] = W8A8_SKIP_TENSOR_PATTERNS,
) -> dict[str, Any]:
    """Produce the ``#fp8-w8a8`` flavor of a diffusers snapshot, streaming."""
    if components is None:
        components = fp8_default_components()

    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if file_layout != MULTI_FILE:
        if te_components:
            raise ConversionImplementationError(
                "te_components need a diffusers layout (no component "
                f"identity in {file_layout!r})")
        sets = _nested_weight_sets(source_dir)
        if not sets:
            raise ConversionImplementationError(
                f"no safetensors weight sets found in {file_layout!r} layout")
        rels = [rel for rel, _e, _m in sets]
        if weight_set_patterns:
            selected = [s for s in sets
                        if any(fnmatch(s[0], p) for p in weight_set_patterns)]
            if not selected:
                raise ConversionImplementationError(
                    f"weight_set_patterns {list(weight_set_patterns)} match "
                    f"none of the discovered weight sets {rels}")
        elif len(sets) == 1:
            selected = sets
        else:
            raise ConversionImplementationError(
                f"{len(sets)} weight sets in {file_layout!r} layout "
                f"({rels}) — pass weight_set_patterns to select the "
                "denoiser set(s); the rest pass through byte-identical")
        tensor_count = converted = 0
        skip_rel: set[str] = set()
        for rel, entry, members in selected:
            result = streaming_w8a8_cast(
                entry, out_dir / Path(rel).parent,
                output_stem=component_output_stem(entry),
                skip_patterns=skip_patterns,
            )
            tensor_count += int(result["tensor_count"])
            converted += int(result["converted_count"])
            skip_rel |= {str(m.relative_to(source_dir)) for m in members}
        if not converted:
            raise ConversionImplementationError(
                "no w8a8-eligible weights in the selected weight set(s) "
                "(nothing 2-D/16-aligned under a repeated-block container "
                "missed the skip patterns)")
        for f in sorted(source_dir.rglob("*")):
            if not f.is_file():
                continue
            rel_path = f.relative_to(source_dir)
            if rel_path.parts[:2] == (".cache", "huggingface"):
                continue
            if f.name == ".civitai.json" or str(rel_path) in skip_rel:
                continue
            _link_or_copy(f, out_dir / rel_path)
        return {"tensor_count": tensor_count,
                "converted_count": converted,
                "components": sorted(rel for rel, _e, _m in selected),
                "output_dir": out_dir}
    if weight_set_patterns:
        raise ConversionImplementationError(
            "weight_set_patterns applies to non-diffusers layouts only "
            "(diffusers selection is by component name)")
    denoiser_set, te_set = set(components), set(te_components)
    groups = [(c, e) for c, e in snapshot_weight_groups(source_dir, MULTI_FILE)
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
                output_stem=component_output_stem(entry),
                skip_patterns=skip_patterns,
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
                output_stem=component_output_stem(entry),
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
            dst_cfg.unlink()
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
    """Byte-gate a produced w8a8 tree against its source."""

    import torch
    from safetensors import safe_open

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
    if art.component:
        src_files = sorted(
            p for p in (source_dir / art.component).glob("*.safetensors")
            if p.is_file())
    else:
        from gen_worker.models.w8a8 import _root_weight_files

        src_files = [p for p in _root_weight_files(source_dir) if p.is_file()]
    src_where = _tensor_map(src_files)

    names = list(art.quantized)
    rng = random.Random(seed)
    picked = names if len(names) <= sample else sorted(rng.sample(names, sample))
    max_rel = 0.0
    max_scale_ulp = 0
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
        if not bool(torch.isfinite(src).all()):
            raise ConversionImplementationError(
                f"byte-gate: source tensor {wname} contains non-finite values")
        with safe_open(str(out_where[wname]), framework="pt", device="cpu") as fh:
            got_q = fh.get_tensor(wname)
        with safe_open(str(out_where[sname]), framework="pt", device="cpu") as fh:
            got_s = fh.get_tensor(sname).float()
        expected_scale = (src.abs().amax(dim=1)
                          / _FP8_E4M3_MAX).clamp(min=1e-12)
        if not bool(torch.isfinite(expected_scale).all()):
            raise ConversionImplementationError(
                f"byte-gate: source-derived scale for {sname} is non-finite")
        artifact_scale = got_s.reshape(-1)
        if artifact_scale.numel() != expected_scale.numel():
            raise ConversionImplementationError(
                f"byte-gate: {sname} has {artifact_scale.numel()} values for "
                f"{expected_scale.numel()} source rows")
        if not bool(torch.isfinite(artifact_scale).all()) or not bool(
            (artifact_scale > 0).all()
        ):
            raise ConversionImplementationError(
                f"byte-gate: {sname} contains non-finite/non-positive values")

        # ModelOpt derives the scale on CUDA; this verifier recomputes on CPU, and FP32 division by 448 legitimately differs by one ULP across devices. Compare positive FP32 bit patterns directly — admits only the two valid adjacent results, not an arbitrary epsilon.
        expected_bits = expected_scale.contiguous().view(torch.int32).to(torch.int64)
        artifact_bits = artifact_scale.contiguous().view(torch.int32).to(torch.int64)
        scale_ulp = (expected_bits - artifact_bits).abs()
        layer_max_scale_ulp = int(scale_ulp.max().item())
        if layer_max_scale_ulp > 1:
            raise ConversionImplementationError(
                f"byte-gate: {sname} differs from source-derived amax/448 by "
                f"{layer_max_scale_ulp} FP32 ULPs (maximum 1)")

        want_q = (src / artifact_scale.reshape(-1, 1)).clamp(
            -_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        if not torch.equal(want_q.view(torch.uint8), got_q.view(torch.uint8)):
            raise ConversionImplementationError(
                f"byte-gate: {wname} recomputed fp8 bytes differ from artifact")
        deq = got_q.float() * artifact_scale.reshape(-1, 1)
        row_amax = src.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        rel = ((deq - src).abs() / row_amax).max().item()
        if not math.isfinite(rel):
            raise ConversionImplementationError(
                f"byte-gate: {wname} dequant error is non-finite")
        if rel > 2 ** -4 + 2 ** -9:
            raise ConversionImplementationError(
                f"byte-gate: {wname} dequant error {rel:.5f} exceeds the "
                "fp8-e4m3 format bound")
        max_rel = max(max_rel, rel)
        max_scale_ulp = max(max_scale_ulp, layer_max_scale_ulp)
    return {
        "component": art.component,
        "quantized_total": len(names),
        "sampled": len(picked),
        "byte_exact": True,
        "max_scale_ulp": max_scale_ulp,
        "max_rel_err": max_rel,
        "source_storage_dtypes": sorted(source_storage_dtypes),
        "source_compute_dtype": canonical_compute_dtype,
    }


_HEADER_LEN_PREFIX = 8
_RAW_COPY_CHUNK = 8 * 1024 * 1024


def read_safetensors_header(fd: int) -> tuple[dict, int]:

    os.lseek(fd, 0, os.SEEK_SET)
    prefix = os.read(fd, _HEADER_LEN_PREFIX)
    if len(prefix) != _HEADER_LEN_PREFIX:
        raise ValueError("safetensors: short read on header length prefix")
    header_len = int.from_bytes(prefix, "little")
    if not header_len_ok(header_len):
        raise ValueError(f"safetensors: implausible header_length={header_len}")
    body = os.read(fd, header_len)
    if len(body) != header_len:
        raise ValueError("safetensors: short read on header body")
    header = json.loads(body.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors: header root must be an object")
    return header, _HEADER_LEN_PREFIX + header_len


def shard_tensor_entries(path: Path) -> list[tuple[str, dict]]:
    """Every ``(name, header entry)`` of one shard, in DATA order."""
    with open(path, "rb") as fh:
        header, _ = read_safetensors_header(fh.fileno())
    rows: list[tuple[str, dict]] = []
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offs = meta.get("data_offsets")
        if not isinstance(offs, list) or len(offs) != 2 or int(offs[1]) < int(offs[0]):
            raise ValueError(f"safetensors: tensor {name!r} has invalid data_offsets")
        rows.append((str(name), meta))
    rows.sort(key=lambda r: int(r[1]["data_offsets"][0]))
    return rows


def shard_metadata(path: Path) -> dict[str, str]:
    """One shard's ``__metadata__``, as strings."""
    with open(path, "rb") as fh:
        header, _ = read_safetensors_header(fh.fileno())
    md = header.get("__metadata__")
    return {str(k): str(v) for k, v in md.items()} if isinstance(md, dict) else {}


def shard_payload_digests(path: Path) -> dict[str, str]:
    """Per-tensor payload sha256, keyed by tensor name."""
    out: dict[str, str] = {}
    with open(path, "rb") as fh:
        header, base = read_safetensors_header(fh.fileno())
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            fh.seek(base + start)
            out[str(name)] = hashlib.sha256(fh.read(end - start)).hexdigest()
    return out


def shard_content_digest(path: Path) -> str:
    """CONTENT identity of one shard: its tensors' (name, dtype, shape, payload) and its ``__metadata__``, digested in a canonical order."""
    rows = []
    with open(path, "rb") as fh:
        header, base = read_safetensors_header(fh.fileno())
        for name, meta in sorted(
            (n, m) for n, m in header.items()
            if n != "__metadata__" and isinstance(m, dict)
        ):
            start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            fh.seek(base + start)
            rows.append([
                str(name), str(meta["dtype"]),
                [int(d) for d in meta["shape"]],
                hashlib.sha256(fh.read(end - start)).hexdigest(),
            ])
    payload = json.dumps(
        {"v": 1, "tensors": rows, "metadata": shard_metadata(path)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_safetensors_shard(
    path: Path,
    tensors: Mapping[str, tuple[str, Sequence[int], bytes]],
    *,
    metadata: Optional[Mapping[str, str]] = None,
) -> Path:
    """Write ONE safetensors file from raw ``(dtype, shape, payload)`` triples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    cursor = 0
    ordered: list[tuple[str, bytes]] = []
    for name, (dtype, shape, payload) in tensors.items():
        if not name or name == "__metadata__":
            raise ValueError(f"safetensors: cannot write a tensor named {name!r}")
        header[str(name)] = {
            "dtype": str(dtype), "shape": [int(d) for d in shape],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        ordered.append((str(name), payload))
        cursor += len(payload)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    tmp = path.parent / f".{path.name}.writing"
    with open(tmp, "wb") as out:
        out.write(len(blob).to_bytes(_HEADER_LEN_PREFIX, "little"))
        out.write(blob)
        for _name, payload in ordered:
            out.write(payload)
    tmp.replace(path)
    return path


def rewrite_safetensors_keys(
    source: Path,
    target: Path,
    key_map: Mapping[str, str],
    *,
    extra_metadata: Optional[Mapping[str, str]] = None,
) -> Path:
    """Rename tensors by RAW BYTE-RANGE COPY — no tensor ever enters Python."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = shard_tensor_entries(source)
    missing = [name for name, _ in entries if name not in key_map]
    if missing and key_map:
        raise ValueError(
            f"rewrite_safetensors_keys: {len(missing)} key(s) have no mapping "
            f"({missing[:5]}). An unmapped key is a REFUSAL, never a silent "
            "skip — a partial rename produces a file that loads as neither "
            "layout.")
    renamed = [(key_map.get(name, name), meta) for name, meta in entries]
    seen: dict[str, str] = {}
    for (new, _meta), (old, _m) in zip(renamed, entries):
        if new in seen:
            raise ValueError(
                f"rewrite_safetensors_keys: {old!r} and {seen[new]!r} both map "
                f"to {new!r}; the key map is not injective")
        seen[new] = old

    metadata = shard_metadata(source)
    if extra_metadata:
        metadata.update({str(k): str(v) for k, v in extra_metadata.items()})

    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    cursor = 0
    for new, meta in renamed:
        start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        header[new] = {
            "dtype": meta["dtype"], "shape": list(meta["shape"]),
            "data_offsets": [cursor, cursor + (end - start)],
        }
        cursor += end - start
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")

    tmp = target.parent / f".{target.name}.rewriting"
    src_fd = os.open(str(source), os.O_RDONLY)
    try:
        _, base = read_safetensors_header(src_fd)
        dst = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(dst, len(blob).to_bytes(_HEADER_LEN_PREFIX, "little"))
            os.write(dst, blob)
            for (_new, meta) in renamed:
                start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
                remaining = end - start
                src_abs = base + start
                while remaining > 0:
                    buf = os.pread(src_fd, min(remaining, _RAW_COPY_CHUNK), src_abs)
                    if not buf:
                        raise IOError(f"safetensors: short read at {src_abs}")
                    os.write(dst, buf)
                    remaining -= len(buf)
                    src_abs += len(buf)
        finally:
            os.close(dst)
    finally:
        os.close(src_fd)
    tmp.replace(target)
    return target


NEVER_SHARD_MAX_SIZE = "1024GB"

_SHARD_MEMBER_NAME_RE = re.compile(r"^.+-\d{5}-of-\d{5}\.(safetensors|bin|pt|ckpt)$")


def find_producer_shards(tree: Path) -> list[str]:
    """Tree-relative paths that make this output a SHARD SET."""
    root = Path(tree)
    if root.is_file():
        return ([root.name] if _SHARD_MEMBER_NAME_RE.match(root.name)
                or root.name.endswith(".safetensors.index.json") else [])
    found: list[str] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.name.endswith(".safetensors.index.json") \
                or _SHARD_MEMBER_NAME_RE.match(f.name):
            found.append(str(f.relative_to(root)))
    return found


def assert_one_file_per_component(tree: Path, *, producer: str) -> None:
    shards = find_producer_shards(Path(tree))
    if shards:
        raise ConversionImplementationError(
            f"sharded_producer_output: {producer} emitted a shard set "
            f"({len(shards)} file(s), e.g. {shards[:3]}) under {tree}; "
            f"our producers write one file per component (th#1362)")


def merge_safetensors_by_offset(
    shard_paths: Sequence[Path],
    out_path: Path,
) -> Path:
    """Concatenate an HF shard set into ONE safetensors file by raw byte-range copy — the input never enters Python as a tensor."""
    if not shard_paths:
        raise ValueError("merge_safetensors_by_offset: no shards")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fds: list[int] = []
    try:
        entries: list[tuple[str, dict, int, int, int, int]] = []
        merged_md: dict[str, str] = {}
        seen: dict[str, Path] = {}
        for shard in shard_paths:
            fd = os.open(str(shard), os.O_RDONLY)
            fds.append(fd)
            header, data_base = read_safetensors_header(fd)
            md = header.get("__metadata__")
            if isinstance(md, dict):
                for k, v in md.items():
                    if k in merged_md and merged_md[k] != v:
                        raise ValueError(
                            f"safetensors shards disagree on __metadata__[{k!r}]: "
                            f"{merged_md[k]!r} vs {v!r}")
                    merged_md[str(k)] = v
            rows = []
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                offs = meta.get("data_offsets")
                if (not isinstance(offs, list) or len(offs) != 2
                        or int(offs[1]) < int(offs[0])):
                    raise ValueError(
                        f"safetensors: tensor {name!r} has invalid data_offsets")
                if name in seen:
                    raise ValueError(
                        f"safetensors shards both define {name!r} "
                        f"({seen[name].name} and {shard.name})")
                seen[name] = shard
                rows.append((name, meta, fd, data_base, int(offs[0]), int(offs[1])))
            rows.sort(key=lambda r: r[4])
            entries.extend(rows)

        new_header: dict[str, Any] = {}
        if merged_md:
            new_header["__metadata__"] = merged_md
        cursor = 0
        for name, meta, _fd, _base, s, e in entries:
            new_header[name] = {
                "dtype": meta["dtype"],
                "shape": list(meta["shape"]),
                "data_offsets": [cursor, cursor + (e - s)],
            }
            cursor += e - s
        blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")

        tmp = out_path.parent / f".{out_path.name}.merging"
        dst = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(dst, len(blob).to_bytes(_HEADER_LEN_PREFIX, "little"))
            os.write(dst, blob)
            for name, _meta, fd, base, s, e in entries:
                remaining = e - s
                src_abs = base + s
                while remaining > 0:
                    buf = os.pread(fd, min(remaining, _RAW_COPY_CHUNK), src_abs)
                    if not buf:
                        raise IOError(
                            f"safetensors: short read on {name!r} at {src_abs}")
                    os.write(dst, buf)
                    remaining -= len(buf)
                    src_abs += len(buf)
        finally:
            os.close(dst)
        tmp.replace(out_path)
        return out_path
    finally:
        for fd in fds:
            os.close(fd)


def deshard_indexed_safetensors(index_path: Path) -> Path:
    """Collapse ONE HF shard set into a single ``<prefix>.safetensors``."""
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    member_names: list[str] = []
    for name in weight_map.values():
        name = str(name)
        if Path(name).name != name:
            raise ValueError(
                f"safetensors index member must be a basename: {index_path}")
        if name not in member_names:
            member_names.append(name)
    members = [index_path.parent / name for name in member_names]
    missing = [m for m in members if not m.is_file()]
    if missing:
        raise ValueError(f"safetensors index references a missing shard: {index_path}")

    prefix = index_path.name.removesuffix(".safetensors.index.json")
    merged = index_path.parent / f"{prefix}.safetensors"
    if merged.exists() and merged not in members:
        raise ValueError(f"deshard destination already exists: {merged}")
    merge_safetensors_by_offset(members, merged)

    with open(merged, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        # bound-justified: `merged` was written by merge_safetensors_by_offset in
        # this same call, from shard headers each already refused by
        # header_len_ok. This length is our own output, not external input, so a
        # second check here would bound us rather than an attacker (§4.24: a
        # limit must name a runaway nothing else prevents).
        header = json.loads(f.read(header_len).decode("utf-8"))
    got = {k for k in header if k != "__metadata__"}
    want = {str(k) for k in weight_map}
    if got != want:
        merged.unlink(missing_ok=True)
        raise ValueError(
            f"deshard of {index_path} produced {len(got)} tensors, index names "
            f"{len(want)} (missing={sorted(want - got)[:5]}, "
            f"extra={sorted(got - want)[:5]})")

    for member in members:
        if member != merged:
            member.unlink()
    index_path.unlink()
    logger.info("deshard index=%s shards=%d -> %s",
                index_path.name, len(members), merged.name)
    return merged


def tree_has_sharded_safetensors(tree: Path) -> bool:
    """Does this tree carry an HF shard set that mirror ingest must collapse?"""
    return any(Path(tree).rglob("*.safetensors.index.json"))


def deshard_mirror_tree(tree: Path) -> int:
    """De-shard every HF shard set in a tree, in place."""
    n = 0
    for index_path in sorted(Path(tree).rglob("*.safetensors.index.json")):
        deshard_indexed_safetensors(index_path)
        n += 1
    return n


CAST_NORMALIZE_DTYPES = {"fp16", "bf16", "fp32", "f16", "f32"}

VARIANT_WEIGHT_NAME_RE = re.compile(
    r"^(?P<base>.+)\.(?P<v>fp16|bf16|fp32)"
    r"(?P<shard>-\d{5}-of-\d{5})?\.safetensors(?P<idx>\.index\.json)?$"
)
VARIANT_INDEX_NAME_RE = re.compile(
    r"^(?P<base>.+)\.safetensors\.index\."
    r"(?P<v>fp16|bf16|fp32)\.json$"
)


def normalize_variant_filenames(tree: Path) -> None:
    """Strip dtype-variant tokens from published weight filenames — the ONE canonical-naming pass every publish path runs (gw#466, unified by gw#522)."""
    dirs = sorted({p.parent for p in Path(tree).rglob("*.safetensors*") if p.is_file()})
    for d in dirs:
        renames: dict[str, str] = {}
        for p in sorted(d.iterdir()):
            m = VARIANT_WEIGHT_NAME_RE.match(p.name)
            if m:
                new_name = f"{m['base']}{m['shard'] or ''}.safetensors{m['idx'] or ''}"
            else:
                index_match = VARIANT_INDEX_NAME_RE.match(p.name)
                if index_match is None:
                    continue
                new_name = f"{index_match['base']}.safetensors.index.json"
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
    "VARIANT_WEIGHT_NAME_RE",
    "normalize_variant_filenames",
    "list_shard_files_from_index",
    "merge_safetensors_by_offset",
    "deshard_indexed_safetensors",
    "deshard_mirror_tree",
    "tree_has_sharded_safetensors",
    "NEVER_SHARD_MAX_SIZE",
    "find_producer_shards",
    "assert_one_file_per_component",
    "IncrementalSafetensorsWriter",
    "torch_dtype_to_st",
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
    "component_output_stem",
    "component_stored_tensor_names",
    "stream_reencode",
    "fp8_cast_eligible",
    "FP8_SKIP_TENSOR_PATTERNS",
    "fp8_default_components",
    "fp8_te_components",
    "te_fp8_castable_keys",
    "streaming_fp8_te_cast",
]
