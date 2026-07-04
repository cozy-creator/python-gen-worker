from __future__ import annotations

from pathlib import Path
from typing import Any

from .writer import ConversionImplementationError


def convert_safetensors_to_flashpack(
    input_path: Path,
    output_path: Path,
    *,
    target_dtype: str = "preserve",
) -> dict[str, Any]:
    """Adapted endpoint module for safetensors->flashpack conversion.

    This module provides deterministic error semantics when flashpack deps are not
    installed in the runtime environment.
    """

    if target_dtype != "preserve":
        raise ConversionImplementationError("flashpack_dtype_mode_not_wired")

    try:
        from flashpack import pack_to_file  # type: ignore
        from safetensors.torch import load_file  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ConversionImplementationError("flashpack_dependencies_missing") from exc

    if not input_path.exists():
        raise ConversionImplementationError("input_missing")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(input_path))
    pack_to_file(tensors, str(output_path), target_dtype=None)

    return {
        "output_path": str(output_path),
        "target_dtype": target_dtype,
    }
