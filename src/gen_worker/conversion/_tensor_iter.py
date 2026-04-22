"""Tensor iteration primitives for Source / Component.

Handles the three input shapes that tenant functions shouldn't have to think
about:

  1. Single .safetensors file — just open and iterate keys.
  2. Sharded .safetensors with .index.json — read weight_map, iterate each
     shard in the order shards appear in the map.
  3. Pickle .ckpt/.pt/.bin — materialize to safetensors first via a private
     on-disk conversion (torch.load(weights_only=True) + safetensors.save_file),
     then iterate the converted file.

Diffusers component layout: walk each component subdir, apply the same rules
per-component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

if TYPE_CHECKING:
    import torch

_PICKLE_EXTS = (".ckpt", ".pt", ".pth", ".bin")

_WEIGHT_COMPONENT_DIRS: frozenset[str] = frozenset({
    "unet", "transformer", "vae", "text_encoder", "text_encoder_2",
    "text_encoder_3", "image_encoder", "prior", "controlnet",
})


def _open_safetensors(path: Path) -> "safetensors.safe_open":  # type: ignore[name-defined]
    from safetensors import safe_open
    return safe_open(str(path), framework="pt")


def _iter_single_safetensors(path: Path) -> Iterator[tuple[str, "torch.Tensor"]]:
    with _open_safetensors(path) as f:
        for name in f.keys():
            yield name, f.get_tensor(name)


def _iter_sharded_safetensors(
    index_path: Path,
) -> Iterator[tuple[str, "torch.Tensor"]]:
    """Iterate tensors across shards listed in a safetensors index.json."""
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"sharded_index_missing_weight_map:{index_path}")
    # Group keys by shard file to avoid re-opening the same shard repeatedly.
    shard_to_keys: dict[str, list[str]] = {}
    for name, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(name)
    shard_dir = index_path.parent
    for shard_file, names in shard_to_keys.items():
        shard_path = shard_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"sharded_index_missing_shard:{shard_path}")
        with _open_safetensors(shard_path) as f:
            for name in names:
                yield name, f.get_tensor(name)


def _find_primary_weight_in_dir(
    dir_path: Path,
) -> tuple[Literal["safetensors", "sharded", "pickle"], Path] | None:
    """Discover the primary weight file in a directory.

    Preference: sharded index > single safetensors > pickle. Returns None if
    no weight file is present.
    """
    # Sharded wins — the .index.json names all shards
    for entry in sorted(dir_path.iterdir()):
        if entry.suffix == ".json" and entry.name.endswith(".safetensors.index.json"):
            return ("sharded", entry)
    # Then single safetensors
    safetensors_files = sorted(dir_path.glob("*.safetensors"))
    if safetensors_files:
        return ("safetensors", safetensors_files[0])
    # Fallback: pickle
    for ext in _PICKLE_EXTS:
        for entry in sorted(dir_path.glob(f"*{ext}")):
            return ("pickle", entry)
    return None


def _materialize_pickle_to_safetensors(
    pickle_path: Path, work_dir: Path,
) -> Path:
    """Convert a pickle weight file to safetensors in work_dir, return the path.

    Uses ``torch.load(weights_only=True)`` to prevent arbitrary code execution
    from malicious pickle payloads.
    """
    import torch
    from safetensors.torch import save_file
    try:
        state = torch.load(str(pickle_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"pickle_load_failed: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc
    # Some checkpoints wrap state_dict under a top-level key
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    tensors = {
        k: v for k, v in state.items()
        if hasattr(v, "dtype") and hasattr(v, "shape")
    }
    if not tensors:
        raise RuntimeError(f"pickle_no_tensors_found:{pickle_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / (pickle_path.stem + ".safetensors")
    save_file(tensors, str(out))
    return out


def iter_component_tensors(
    component_dir: Path,
) -> Iterator[tuple[str, "torch.Tensor"]]:
    """Yield (name, tensor) for every weight in a single component directory."""
    found = _find_primary_weight_in_dir(component_dir)
    if found is None:
        return
    kind, path = found
    if kind == "safetensors":
        yield from _iter_single_safetensors(path)
    elif kind == "sharded":
        yield from _iter_sharded_safetensors(path)
    elif kind == "pickle":
        # Materialize in-place next to the pickle; subsequent iterations
        # on the same directory will prefer the new safetensors file.
        out = _materialize_pickle_to_safetensors(path, component_dir / ".__pickle_cache__")
        yield from _iter_single_safetensors(out)


def iter_source_tensors(
    root: Path,
    *,
    file_layout: str,
    components_filter: list[str] | None = None,
) -> Iterator[tuple[str, str, "torch.Tensor"]]:
    """Yield (component, name, tensor) across the whole source snapshot."""
    if file_layout == "singlefile":
        for name, tensor in iter_component_tensors(root):
            yield "", name, tensor
        return
    # diffusers
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name not in _WEIGHT_COMPONENT_DIRS:
            continue
        if components_filter is not None and entry.name not in components_filter:
            continue
        for name, tensor in iter_component_tensors(entry):
            yield entry.name, name, tensor


__all__ = ["iter_component_tensors", "iter_source_tensors"]
