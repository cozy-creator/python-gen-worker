"""Cheap intrinsic-size walker for a materialized snapshot directory.

Computes ``full_model_bytes`` and ``largest_component_bytes`` by stat-ing the
weight files on disk. No model instantiation, no torch import, no GPU. The
output is intended to be attached to a checkpoint as ``attributes.size_facts``
at ingest time so the orchestrator can gate VRAM placement at submit
(gen-orchestrator #320):

  required_vram = base + size_mult × size_facts[vram_must_fit] + Σ vram_coef[f] × payload[f]

For a typical diffusers-layout snapshot, the walker iterates each component
subdir (``transformer/``, ``text_encoder/``, ``vae/``, ...) and sums the
bytes of weight files inside. For transformers-layout singlefile snapshots
(``config.json`` + weights at the top level) the whole snapshot counts as
one component named ``"model"``.

Output shape::

    {
      "full_model_bytes": int,
      "largest_component_bytes": int,
      "components": {
        "<component_name>": {"total_bytes": int, "file_count": int}
      },
      "schema_version": 1
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Weight file extensions counted toward a component's size. Restricted to
# real weight containers — config.json, tokenizer.json, README, etc. are
# excluded since they don't load into VRAM.
_WEIGHT_EXTS: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")

# Component subdirs that are weight-bearing. Matches gen-worker's
# conversion._DIFFUSERS_COMPONENT_DIRS roughly; kept local so we don't
# import a heavy module.
_DIFFUSERS_WEIGHT_COMPONENT_DIRS: frozenset[str] = frozenset({
    "transformer",
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "image_encoder",
    "prior",
    "controlnet",
})


def compute_size_facts(snapshot_path: Path | str) -> dict[str, Any]:
    """Return ``{full_model_bytes, largest_component_bytes, components, schema_version}``.

    Cheap stat-based walk. Safe to call on any snapshot directory; returns
    zeros + an empty components map if the directory is empty or contains
    no weight files.
    """
    path = Path(snapshot_path)
    if not path.is_dir():
        return {
            "full_model_bytes": 0,
            "largest_component_bytes": 0,
            "components": {},
            "schema_version": 1,
        }

    components: dict[str, dict[str, int]] = {}

    # Diffusers layout: per-component subdirs.
    diffusers_entries = [
        entry for entry in path.iterdir()
        if entry.is_dir() and entry.name in _DIFFUSERS_WEIGHT_COMPONENT_DIRS
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
        # Transformers-style singlefile snapshot — whole snapshot is one component.
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
        # Bare singlefile or unknown layout — sum any weight files at the top.
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
