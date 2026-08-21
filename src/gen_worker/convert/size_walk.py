"""Cheap intrinsic-size walker for a materialized snapshot directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..component_vocab import weight_components

_WEIGHT_EXTS: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")

def _diffusers_weight_component_dirs() -> frozenset[str]:
    return frozenset(weight_components())


def compute_size_facts(snapshot_path: Path | str) -> dict[str, Any]:
    """Return ``{full_model_bytes, largest_component_bytes, components, schema_version}``."""
    path = Path(snapshot_path)
    if not path.is_dir():
        return {
            "full_model_bytes": 0,
            "largest_component_bytes": 0,
            "components": {},
            "schema_version": 1,
        }

    components: dict[str, dict[str, int]] = {}

    diffusers_entries = [
        entry for entry in path.iterdir()
        if entry.is_dir() and entry.name in _diffusers_weight_component_dirs()
    ]
    if diffusers_entries:
        for entry in sorted(diffusers_entries):
            total = 0
            count = 0
            for f in entry.rglob("*"):
                if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                    try:
                        total += f.stat().st_size
                        count += 1
                    except OSError:
                        continue
            if total > 0:
                components[entry.name] = {"total_bytes": total, "file_count": count}
    elif (path / "config.json").is_file():
        total = 0
        count = 0
        for f in path.rglob("*"):
            if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                try:
                    total += f.stat().st_size
                    count += 1
                except OSError:
                    continue
        if total > 0:
            components["model"] = {"total_bytes": total, "file_count": count}
    else:
        total = 0
        count = 0
        for f in path.rglob("*"):
            if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
                try:
                    total += f.stat().st_size
                    count += 1
                except OSError:
                    continue
        if total > 0:
            components["model"] = {"total_bytes": total, "file_count": count}

    full = sum(c["total_bytes"] for c in components.values())
    largest = max((c["total_bytes"] for c in components.values()), default=0)
    return {
        "full_model_bytes": int(full),
        "largest_component_bytes": int(largest),
        "components": components,
        "schema_version": 1,
    }


__all__ = ["compute_size_facts"]
