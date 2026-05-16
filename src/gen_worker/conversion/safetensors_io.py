"""Path-in-path-out safetensors helpers for the clone pipeline.

The transform-endpoint tenants use StreamingWriter (see writer.py) — library
drives the loop, writer handles single-vs-sharded on finalize. Tenants that
operate on explicit local file paths (e.g. clone_pipeline's external-URL
ingest path) use these primitives:

  - ``materialize_safetensors_input(path, workdir)`` — coerce a source weight
    file (``.safetensors``, ``.safetensors.index.json``, or pickle
    ``.ckpt``/``.pt``/``.pth``/``.bin``) into a path the streaming readers
    open. Pickle is converted via ``torch.load(weights_only=True)``.

  - ``shard_safetensors_by_offset(src, outdir, ...)`` — split an oversized
    safetensors file into HF-convention shards without decoding any tensor.

  - ``persist_safetensors_output(ctx, shard_paths, index_path, output_ref)``
    — upload shards produced by a streaming conversion primitive. No
    ``load_file``/``save_file`` round-trip; bytes go straight from the
    writer's filesystem output into the repo-CAS upload session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..api.types import Tensors
from ..request_context import RequestContext
from ._sharding import (
    MAX_SAFETENSORS_SHARD_BYTES,
    build_safetensors_index,
    plan_safetensors_shards,
)
from .core_types import ConversionArtifact
from .streaming_primitives import (
    ConversionImplementationError,
    list_shard_files_from_index,
)

_PICKLE_EXTENSIONS = (".bin", ".pt", ".pth", ".ckpt")


def materialize_safetensors_input(
    input_path: Path, work_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """Return a path the streaming readers can open + metadata dict.

    - ``.safetensors`` file → returned as-is.
    - ``.safetensors.index.json`` → validated; returned as-is (downstream
      streaming readers resolve shards via the index's weight_map).
    - Pickle (.ckpt/.pt/.pth/.bin) → safely converted to safetensors in
      ``work_dir`` using ``torch.load(weights_only=True)``.
    """
    path = Path(input_path)
    lower = path.name.lower()
    if lower.endswith(".safetensors"):
        return path, {"input_sharded": "0", "input_shard_count": "1"}

    if any(lower.endswith(ext) for ext in _PICKLE_EXTENSIONS):
        return _convert_pickle_to_safetensors(path, work_dir)

    if not lower.endswith(".safetensors.index.json"):
        raise ValueError("requires_safetensors_or_index_input")

    shard_paths = list_shard_files_from_index(path)
    if not shard_paths:
        raise ConversionImplementationError("sharded_index_missing_weight_map")
    for shard in shard_paths:
        if not shard.exists():
            raise ConversionImplementationError(f"sharded_index_missing_shard:{shard.name}")
    return path, {"input_sharded": "1", "input_shard_count": str(len(shard_paths))}


def _convert_pickle_to_safetensors(
    input_path: Path, work_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """Safely convert a pickle weight file to safetensors via weights_only=True."""
    try:
        import torch
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - dependency controlled
        raise ConversionImplementationError("safetensors_dependencies_missing") from exc

    try:
        state = torch.load(str(input_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ConversionImplementationError(
            f"pickle_load_failed: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc

    # torch.load may return a nested dict wrapping the real state_dict.
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    tensors = {k: v for k, v in state.items() if hasattr(v, "dtype") and hasattr(v, "shape")}
    if not tensors:
        raise ConversionImplementationError("pickle_no_tensors_found")

    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "materialized-input.safetensors"
    try:
        save_file(tensors, str(out))
    except Exception as exc:  # pragma: no cover - backend specific
        raise ConversionImplementationError("pickle_to_safetensors_save_failed") from exc
    return out, {"input_sharded": "0", "input_shard_count": "1", "source_format": "pickle"}


def persist_safetensors_output(
    ctx: RequestContext,
    *,
    shard_paths: list[Path],
    index_path: Path | None,
    output_ref: str,
) -> tuple[Tensors, list[ConversionArtifact], dict[str, str]]:
    """Upload per-shard safetensors output (+ optional index) produced by a
    streaming conversion primitive.

    Input is always "what the writer produced": ``shard_paths`` is the
    writer's ordered list of output files (1 or more), ``index_path`` is its
    HF-compatible ``.safetensors.index.json`` (or ``None`` when only one
    shard was emitted). Nothing is re-decoded or re-serialized here — every
    shard goes directly into the repo-CAS upload session.

    ``output_ref`` is the single-file ref the caller would have chosen for
    an un-sharded result (e.g. ``.../transformer/diffusion_pytorch_model.safetensors``).
    We preserve the parent directory as the upload root; multi-shard uploads
    place shards at ``<parent>/<shard_filename>`` and the index at
    ``<output_ref>.index.json``.

    Returns ``(primary, additional_artifacts, metadata)``:
      * primary — single-shard: the shard's Tensors; multi-shard: the index's
        Tensors (entry point for sharded loaders).
      * additional_artifacts — per-shard Tensors and (multi-shard only) the
        index, in the order the manifest builder should emit them.
      * metadata — shard descriptors for ``ConversionOutput.metadata``.
    """
    out_ref = str(output_ref or "").strip()
    if not out_ref.endswith(".safetensors"):
        raise ValueError("output_ref must end with .safetensors")
    if not shard_paths:
        raise ValueError("shard_paths must be non-empty")

    base_dir = str(Path(out_ref).parent).strip()
    if base_dir == ".":
        base_dir = ""

    # Single-shard fast path: upload the one file at the caller's chosen
    # output_ref. This keeps the canonical diffusers filename
    # (``transformer/diffusion_pytorch_model.safetensors``) instead of the
    # planner's "-00001-of-00001" suffix which only makes sense on disk.
    if len(shard_paths) == 1 and index_path is None:
        saved = ctx.save_checkpoint(out_ref, str(shard_paths[0]), format="safetensors")
        return saved, [], {
            "output_sharded": "0",
            "output_shard_count": "1",
            "output_max_shard_bytes": str(MAX_SAFETENSORS_SHARD_BYTES),
            "source_artifact_refs": str(saved.ref or out_ref),
        }

    # Issue #269/#13: parallelize shard uploads across the adaptive file
    # pool. Each shard is up to MAX_SAFETENSORS_SHARD_BYTES of independent
    # bytes; fanning out lets the multipart PUT pipelines stack across files.
    from ..request_context._concurrent_upload import parallel_map_uploads

    def _upload_shard(shard_path_local: Path) -> tuple[Path, str, Tensors]:
        shard_name_local = shard_path_local.name
        shard_ref_local = f"{base_dir}/{shard_name_local}" if base_dir else shard_name_local
        saved_local = ctx.save_checkpoint(shard_ref_local, str(shard_path_local), format="safetensors")
        return shard_path_local, shard_ref_local, saved_local

    shard_refs: list[str] = []
    shard_artifacts: list[ConversionArtifact] = []
    for shard_path, shard_ref, saved in parallel_map_uploads(
        list(shard_paths), _upload_shard, label="shard-upload"
    ):
        shard_refs.append(str(saved.ref or shard_ref))
        shard_artifacts.append(ConversionArtifact(rel_name=shard_path.name, tensors=saved))

    if index_path is None:
        # Multi-shard without an index — shouldn't happen for a well-formed
        # writer output, but don't silently drop data: promote the first shard.
        primary = shard_artifacts[0].tensors
        return primary, shard_artifacts[1:], {
            "output_sharded": "1",
            "output_shard_count": str(len(shard_paths)),
            "output_shard_refs": ";".join(shard_refs),
            "output_max_shard_bytes": str(MAX_SAFETENSORS_SHARD_BYTES),
            "source_artifact_refs": ";".join(shard_refs),
        }

    index_ref = f"{out_ref}.index.json"
    index_saved = ctx.save_checkpoint(index_ref, str(index_path), format="json")
    index_name = Path(out_ref).name + ".index.json"
    additional = [ConversionArtifact(rel_name=index_name, tensors=index_saved), *shard_artifacts]
    return index_saved, additional, {
        "output_sharded": "1",
        "output_shard_count": str(len(shard_paths)),
        "output_index_ref": str(index_saved.ref or index_ref),
        "output_shard_refs": ";".join(shard_refs),
        "output_max_shard_bytes": str(MAX_SAFETENSORS_SHARD_BYTES),
        "source_artifact_refs": ";".join([str(index_saved.ref or index_ref), *shard_refs]),
    }


# ---------------------------------------------------------------------------
# Raw-byte sharding — no tensor decode.
# ---------------------------------------------------------------------------

# Safetensors file format (from huggingface/safetensors spec):
#
#   [8 bytes]  header_length_le: little-endian uint64 length of the JSON header
#   [header_length_le] JSON:
#     {
#       "tensor_name_1": {"dtype": "BF16", "shape": [...], "data_offsets": [start, end]},
#       "tensor_name_2": {...},
#       "__metadata__":  {"arbitrary": "tenant KV"}   # optional
#     }
#   [tensor data, contiguous]
#
# data_offsets are relative to the start of the DATA section, i.e. they
# run from 0 (first tensor start) to total_data_bytes (last tensor end).
# Absolute offset = 8 + header_length + data_offsets[0].

_SAFETENSORS_HEADER_LEN_PREFIX = 8
_MAX_HEADER_BYTES = 512 * 1024 * 1024  # 512 MB — protects against absurd headers
_RAW_COPY_CHUNK = 8 * 1024 * 1024  # 8 MB pread/pwrite chunks


def _read_safetensors_header(src_fd: int) -> tuple[dict, int]:
    """Read the JSON header from an open safetensors file descriptor.

    Returns (header, data_section_offset). data_section_offset is the absolute
    byte offset in the file where tensor bytes begin (== 8 + header_length).
    Caller owns the fd; this function seeks but doesn't close it.
    """
    import os

    os.lseek(src_fd, 0, os.SEEK_SET)
    prefix = os.read(src_fd, _SAFETENSORS_HEADER_LEN_PREFIX)
    if len(prefix) != _SAFETENSORS_HEADER_LEN_PREFIX:
        raise ValueError("safetensors: short read on header length prefix")
    header_len = int.from_bytes(prefix, byteorder="little", signed=False)
    if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"safetensors: implausible header_length={header_len}")

    header_bytes = os.read(src_fd, header_len)
    if len(header_bytes) != header_len:
        raise ValueError("safetensors: short read on header body")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors: header root must be an object")
    return header, _SAFETENSORS_HEADER_LEN_PREFIX + header_len


def _tensor_byte_sizes_from_header(header: dict) -> dict[str, int]:
    """Extract {tensor_name: byte_length} from a parsed safetensors header.

    Skips the reserved ``__metadata__`` entry. Byte length is derived from
    data_offsets (end - start) so it's exact — no dtype-based calculation.
    """
    sizes: dict[str, int] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            continue
        offs = meta.get("data_offsets")
        if not isinstance(offs, list) or len(offs) != 2:
            raise ValueError(f"safetensors: tensor {name!r} missing data_offsets")
        start, end = int(offs[0]), int(offs[1])
        if end < start:
            raise ValueError(f"safetensors: tensor {name!r} has negative length")
        sizes[name] = end - start
    return sizes


def shard_safetensors_by_offset(
    src_path: Path,
    out_dir: Path,
    *,
    max_shard_bytes: int = MAX_SAFETENSORS_SHARD_BYTES,
    shard_prefix: str = "model",
) -> tuple[list[Path], Path, dict[str, int]]:
    """Split a single safetensors file into shards without decoding tensors.

    Reads the source header, runs the standard shard planner on the
    {tensor_name: byte_length} map, and copies the raw tensor byte ranges
    into new per-shard files via ``os.pread``/``os.pwrite``. The input
    never enters Python as a tensor — zero decode cost, memory use caps at
    a small per-tensor scratch buffer.

    Returns
    -------
    shards:    list[Path]     - ordered shard output paths
    index:     Path           - path to ``{shard_prefix}.safetensors.index.json``
                                (HF-compatible; empty for single-shard case,
                                caller can choose not to upload it when
                                ``len(shards) == 1``)
    shard_sizes: dict[str,int] - {shard_filename: total_bytes} for caller telemetry

    Caller is responsible for creating ``out_dir`` (parents exist). The
    source file must be a valid safetensors (8-byte LE header-length
    prefix + JSON header + contiguous tensor data).
    """
    import os

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_fd = os.open(str(src_path), os.O_RDONLY)
    try:
        header, data_base = _read_safetensors_header(src_fd)
        tensor_bytes = _tensor_byte_sizes_from_header(header)
        plan = plan_safetensors_shards(
            tensor_bytes,
            max_shard_bytes=max_shard_bytes,
            shard_prefix=shard_prefix,
        )

        # Preserve the original order of tensors in the source file — a
        # sorted walk would break contiguity of the `os.pread` chunks
        # for sources whose layout doesn't match alphabetical order.
        # Shard assignments are alphabetical (that's what the planner
        # emits), so within each shard we re-order by source offset to
        # keep the raw-copy pass sequential on the source side.
        reserved_metadata = header.get("__metadata__")

        # Group tensors per shard, with source-side data_offsets so we
        # can sort by source start offset inside each shard.
        shard_groups: dict[str, list[tuple[str, int, int]]] = {
            name: [] for name in plan.shard_names
        }
        for tname, shard_name in plan.weight_map.items():
            meta = header[tname]
            offs = meta["data_offsets"]
            shard_groups[shard_name].append((tname, int(offs[0]), int(offs[1])))
        for g in shard_groups.values():
            g.sort(key=lambda r: r[1])

        shard_paths: list[Path] = []
        for shard_name in plan.shard_names:
            entries = shard_groups[shard_name]
            # Build the new header: re-based offsets for each tensor,
            # preserving dtype + shape from the source header verbatim.
            new_header: dict[str, Any] = {}
            if isinstance(reserved_metadata, dict):
                new_header["__metadata__"] = dict(reserved_metadata)
            cursor = 0
            for tname, src_start, src_end in entries:
                size = src_end - src_start
                src_meta = header[tname]
                new_header[tname] = {
                    "dtype": src_meta["dtype"],
                    "shape": list(src_meta["shape"]),
                    "data_offsets": [cursor, cursor + size],
                }
                cursor += size

            header_bytes = json.dumps(new_header, separators=(",", ":"), sort_keys=False).encode("utf-8")
            shard_path = out_dir / shard_name
            dst_fd = os.open(str(shard_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(dst_fd, len(header_bytes).to_bytes(_SAFETENSORS_HEADER_LEN_PREFIX, "little"))
                os.write(dst_fd, header_bytes)
                # Stream tensor bytes — read from source at absolute offset,
                # write sequentially into shard data section.
                for tname, src_start, src_end in entries:
                    remaining = src_end - src_start
                    src_abs = data_base + src_start
                    while remaining > 0:
                        n = min(remaining, _RAW_COPY_CHUNK)
                        buf = os.pread(src_fd, n, src_abs)
                        if not buf:
                            raise IOError(f"safetensors: short read on tensor {tname!r} at offset {src_abs}")
                        os.write(dst_fd, buf)
                        remaining -= len(buf)
                        src_abs += len(buf)
            finally:
                os.close(dst_fd)
            shard_paths.append(shard_path)

        # Emit the HF-compatible index.json. For single-shard plans we
        # still emit it (consumers can ignore); caller decides whether to
        # upload based on len(shard_paths).
        index_path = out_dir / f"{shard_prefix}.safetensors.index.json"
        index_path.write_text(
            json.dumps(build_safetensors_index(plan), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

        return shard_paths, index_path, dict(plan.shard_sizes)
    finally:
        os.close(src_fd)


__all__ = [
    "materialize_safetensors_input",
    "persist_safetensors_output",
    "shard_safetensors_by_offset",
]
