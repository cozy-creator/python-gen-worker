"""Lower-level streaming-conversion primitives.

Moved from conversion-endpoints/conversions/ as part of e2e progress.json #9.
These are lower-level than StreamingWriter — they operate on raw safetensors
files and let callers (clone_pipeline) drive the conversion loop themselves.
New transform-endpoint tenants use StreamingWriter instead; these primitives
are exposed for the legacy clone_pipeline path and any tenant that needs
finer-grained control.

Combines three formerly-separate conversion-endpoints modules:
  - streaming_convert.py  → streaming_dtype_cast, streaming_gpu_quantize,
                            streaming_nvfp4_quantize
  - incremental_safetensors.py → IncrementalSafetensorsWriter
  - sharded_index.py → list_shard_files_from_index

Separate module rather than inlined into writer.py because these bypass
Source/StreamingWriter and operate on explicit file paths.
"""

from __future__ import annotations

import json
from pathlib import Path


class ConversionImplementationError(RuntimeError):
    """Raised when a conversion primitive can't proceed.

    Moved from conversion-endpoints/conversions/common.py. Same semantics —
    a runtime error flavor that callers can distinguish from unrelated
    exceptions via isinstance.
    """


def list_shard_files_from_index(index_path: Path) -> list[Path]:
    """Return shard file paths in weight-map order from a safetensors index.json.

    Deduplicates while preserving first-appearance order. Raises
    ``ConversionImplementationError`` on malformed input.
    """
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
        if shard in seen:
            continue
        seen.add(shard)
        ordered.append(index_path.parent / shard)
    return ordered


# Re-export the higher-level streaming + incremental helpers. Heavy imports
# (torch, safetensors) are lazy so importing gen_worker.conversion doesn't
# pay torch-import cost unless a caller actually touches these primitives.
__all__ = [
    "ConversionImplementationError",
    "list_shard_files_from_index",
    "IncrementalSafetensorsWriter",
    "streaming_dtype_cast",
    "streaming_gpu_quantize",
    "streaming_nvfp4_quantize",
]


def __getattr__(name: str):  # lazy-load the torch-dependent primitives
    if name == "IncrementalSafetensorsWriter":
        from ._streaming_incremental import IncrementalSafetensorsWriter
        return IncrementalSafetensorsWriter
    if name in {"streaming_dtype_cast", "streaming_gpu_quantize", "streaming_nvfp4_quantize"}:
        from . import _streaming_convert
        return getattr(_streaming_convert, name)
    raise AttributeError(name)
