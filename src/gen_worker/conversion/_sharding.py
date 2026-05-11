"""Internal sharding helpers for StreamingWriter.

Tenants don't touch this — the library decides at `finalize()` whether a
component's output is small enough to fit in one safetensors file or needs
to be split across multiple shards + an index.json. Rule of thumb: if the
component's total size crosses ``MAX_SAFETENSORS_SHARD_BYTES`` we plan a
shard layout that keeps each file under the cap; otherwise we emit one
file.

Algorithm is greedy first-fit in sorted tensor-name order — matches
HuggingFace transformers' own state_dict_sharded emit, so consumers that
load via ``from_pretrained(..., index_filename=...)`` get the same shape
they'd get from a native HF save.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


MAX_SAFETENSORS_SHARD_BYTES: int = 5 * 1024 * 1024 * 1024
# 5 GB per shard. Matches HuggingFace transformers/diffusers' default shard
# size for save_pretrained — any consumer that loads via from_pretrained
# already handles 5 GB shards natively. Components > 5 GB (flux-transformer
# dev at ~24 GB; klein-4b text_encoder at ~7.5 GB) automatically shard into
# multiple files named ``{prefix}-NNNNN-of-NNNNN.safetensors`` + an index.


@dataclass(frozen=True)
class SafetensorsShardPlan:
    """Output of the shard planner — names, weight→shard map, sizes."""

    shard_names: list[str]
    weight_map: dict[str, str]
    shard_sizes: dict[str, int]
    total_size: int


def plan_safetensors_shards(
    tensor_bytes: Mapping[str, int],
    *,
    max_shard_bytes: int = MAX_SAFETENSORS_SHARD_BYTES,
    shard_prefix: str = "model",
) -> SafetensorsShardPlan:
    """Plan a safetensors shard layout for a dict of tensor sizes.

    Returns a single-shard plan when the total fits under max_shard_bytes.
    Names follow the HF convention: ``model.safetensors`` for the single
    case, ``model-00001-of-00008.safetensors`` for sharded.
    """
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

    if not entries:
        single_name = f"{shard_prefix}.safetensors"
        return SafetensorsShardPlan(
            shard_names=[single_name],
            weight_map={},
            shard_sizes={single_name: 0},
            total_size=0,
        )

    total_size = sum(size for _, size in entries)
    if total_size <= int(max_shard_bytes):
        # Fits in one file — common case.
        single_name = f"{shard_prefix}.safetensors"
        return SafetensorsShardPlan(
            shard_names=[single_name],
            weight_map={k: single_name for k, _ in entries},
            shard_sizes={single_name: total_size},
            total_size=total_size,
        )

    # Multi-shard: greedy first-fit
    target_bytes = int(max_shard_bytes)
    shards: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_size = 0
    for key, size in entries:
        if current and current_size + size > target_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append((key, size))
        current_size += size
    if current:
        shards.append(current)

    shard_count = len(shards)
    shard_names = [
        f"{shard_prefix}-{idx + 1:05d}-of-{shard_count:05d}.safetensors"
        for idx in range(shard_count)
    ]
    weight_map: dict[str, str] = {}
    shard_sizes: dict[str, int] = {}
    for shard_name, shard_entries in zip(shard_names, shards):
        shard_total = 0
        for key, size in shard_entries:
            weight_map[key] = shard_name
            shard_total += size
        shard_sizes[shard_name] = shard_total

    return SafetensorsShardPlan(
        shard_names=shard_names,
        weight_map=weight_map,
        shard_sizes=shard_sizes,
        total_size=total_size,
    )


def build_safetensors_index(plan: SafetensorsShardPlan) -> dict[str, object]:
    """Build the HF-compatible ``model.safetensors.index.json`` payload."""
    return {
        "metadata": {"total_size": int(plan.total_size)},
        "weight_map": dict(plan.weight_map),
    }


def _tensor_size_bytes(tensor) -> int:
    """Best-effort element-size-times-numel calculation for torch tensors."""
    try:
        numel = int(tensor.numel())
        elem = int(tensor.element_size())
        return numel * elem
    except AttributeError:
        # Fallback: use nbytes attribute if present
        nb = getattr(tensor, "nbytes", None)
        if nb is not None:
            return int(nb)
        raise


__all__ = [
    "MAX_SAFETENSORS_SHARD_BYTES",
    "SafetensorsShardPlan",
    "build_safetensors_index",
    "plan_safetensors_shards",
    "_tensor_size_bytes",
]
