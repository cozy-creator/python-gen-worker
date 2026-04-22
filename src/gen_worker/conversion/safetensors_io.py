"""Path-in-path-out safetensors helpers for tenants outside the transform contract.

The transform-endpoint tenants use StreamingWriter (see writer.py) — library
drives the loop, writer handles single-vs-sharded on finalize. Tenants that
operate on explicit local file paths (e.g. clone_pipeline's external-URL
ingest path) use the two primitives here:

  - materialize_safetensors_input(): take a raw weight file path (.safetensors,
    sharded .index.json, or pickle .ckpt/.pt/.pth/.bin) and return a path the
    streaming readers can open. Handles pickle → safetensors via
    torch.load(weights_only=True) for security.

  - persist_safetensors_output(): take a converted safetensors file on local
    disk, upload to the destination ref. Auto-shards at 5 GB using the
    canonical _sharding primitives. Returns (primary_tensors, additional_artifacts,
    metadata) matching ConversionOutput's shape.
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
    converted_path: Path,
    output_ref: str,
) -> tuple[Tensors, list[ConversionArtifact], dict[str, str]]:
    """Upload a converted safetensors file; auto-shard at MAX_SAFETENSORS_SHARD_BYTES.

    Returns:
      primary                 -- entry-point Tensors (single .safetensors OR .index.json)
      additional_artifacts    -- per-shard Tensors + the index (empty for unsharded)
      metadata                -- shard descriptor strings for ConversionOutput.metadata
    """
    out_ref = str(output_ref or "").strip()
    if not out_ref.endswith(".safetensors"):
        raise ValueError("output_ref must end with .safetensors")
    if not converted_path.exists():
        raise RuntimeError("converted_output_missing")

    size = int(converted_path.stat().st_size)
    if size <= MAX_SAFETENSORS_SHARD_BYTES:
        saved = ctx.save_checkpoint(out_ref, str(converted_path), format="safetensors")
        return saved, [], {
            "output_sharded": "0",
            "output_shard_count": "1",
            "output_max_shard_bytes": str(MAX_SAFETENSORS_SHARD_BYTES),
            "source_artifact_refs": str(saved.ref or out_ref),
        }

    # Sharded path: load tensors, plan shards, write + upload each, plus index.
    from safetensors.torch import load_file, save_file

    tensors: dict[str, Any] = dict(load_file(str(converted_path)))
    sizes = {k: int(v.numel() * v.element_size()) for k, v in tensors.items()}
    plan = plan_safetensors_shards(
        sizes,
        max_shard_bytes=MAX_SAFETENSORS_SHARD_BYTES,
        shard_prefix=Path(out_ref).name[: -len(".safetensors")],
    )
    if len(plan.shard_names) <= 1:
        # Planner concluded it fits — upload single.
        saved = ctx.save_checkpoint(out_ref, str(converted_path), format="safetensors")
        return saved, [], {
            "output_sharded": "0",
            "output_shard_count": "1",
            "output_max_shard_bytes": str(MAX_SAFETENSORS_SHARD_BYTES),
            "source_artifact_refs": str(saved.ref or out_ref),
        }

    work = converted_path.parent / "sharded-output"
    work.mkdir(parents=True, exist_ok=True)
    base_dir = str(Path(out_ref).parent).strip()
    if base_dir == ".":
        base_dir = ""

    shard_refs: list[str] = []
    shard_artifacts: list[ConversionArtifact] = []
    for shard_name in plan.shard_names:
        shard_path = work / shard_name
        shard_tensors = {k: tensors[k] for k, target in plan.weight_map.items() if target == shard_name}
        save_file(shard_tensors, str(shard_path))
        shard_ref = f"{base_dir}/{shard_name}" if base_dir else shard_name
        saved = ctx.save_checkpoint(shard_ref, str(shard_path), format="safetensors")
        shard_refs.append(str(saved.ref or shard_ref))
        shard_artifacts.append(ConversionArtifact(rel_name=shard_name, tensors=saved))

    index_path = work / f"{Path(out_ref).name}.index.json"
    index_path.write_text(
        json.dumps(build_safetensors_index(plan), separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    index_ref = f"{out_ref}.index.json"
    index_saved = ctx.save_checkpoint(index_ref, str(index_path), format="json")
    index_name = Path(out_ref).name + ".index.json"
    additional = [ConversionArtifact(rel_name=index_name, tensors=index_saved), *shard_artifacts]

    return index_saved, additional, {
        "output_sharded": "1",
        "output_shard_count": str(len(plan.shard_names)),
        "output_index_ref": str(index_saved.ref or index_ref),
        "output_shard_refs": ";".join(shard_refs),
        "output_max_shard_bytes": str(MAX_SAFETENSORS_SHARD_BYTES),
        "source_artifact_refs": ";".join([str(index_saved.ref or index_ref), *shard_refs]),
    }


__all__ = [
    "materialize_safetensors_input",
    "persist_safetensors_output",
]
