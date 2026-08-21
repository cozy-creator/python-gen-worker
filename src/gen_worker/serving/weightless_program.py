"""Load a WEIGHTS-FREE exported program — the one the derive actually writes."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping

WEIGHTS_DIR = "weights"

SHARED_PAYLOAD = "weight_0"


def is_weightless(archive_reader: Any, model_name: str) -> bool:
    """True when this archive declares parameters but carries no payload bytes."""
    from torch.export.pt2_archive._package import (  # type: ignore[attr-defined]
        WEIGHTS_CONFIG_FILENAME_FORMAT,
    )

    config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
    if config_file not in archive_reader.get_file_names():
        return False
    try:
        payload = archive_reader.read_bytes(os.path.join(WEIGHTS_DIR, SHARED_PAYLOAD))
    except Exception:  # noqa: BLE001 - an unreadable payload is an empty one
        return True
    return not payload


def _tensor_from_meta(tensor_meta: Mapping[str, Any]) -> Any:
    import torch
    from torch._export.serde.serialize import _SERIALIZE_TO_TORCH_DTYPE

    dtype = _SERIALIZE_TO_TORCH_DTYPE[int(tensor_meta["dtype"])]
    sizes = [int(s["as_int"]) for s in tensor_meta["sizes"]]
    device_meta = tensor_meta.get("device") or {}
    device_type = str(device_meta.get("type") or "cpu")
    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(device_type, int(device_meta.get("index") or 0))
    return torch.zeros(sizes, dtype=dtype, device=device)


def state_dict_from_config(archive_reader: Any, model_name: str) -> dict[str, Any]:
    """The state dict this archive DESCRIBES, built from its own metadata."""
    import torch
    from torch.export.pt2_archive._package import (  # type: ignore[attr-defined]
        WEIGHTS_CONFIG_FILENAME_FORMAT,
    )

    config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
    config = json.loads(archive_reader.read_bytes(config_file))["config"]

    rebuilt: dict[str, Any] = {}
    for fqn, entry in config.items():
        tensor = _tensor_from_meta(entry["tensor_meta"])
        if entry.get("is_param"):
            tensor = torch.nn.Parameter(tensor, requires_grad=False)
        rebuilt[fqn] = tensor
    return rebuilt


def install() -> None:
    """Teach ``torch.export.load`` to read a weights-free archive."""
    from torch.export.pt2_archive import _package

    if getattr(_package._load_state_dict, "_pgw1468", False):
        return

    original: Callable[..., Any] = _package._load_state_dict

    def _load_state_dict(archive_reader: Any, model_name: str) -> Any:
        if not is_weightless(archive_reader, model_name):
            return original(archive_reader, model_name)
        return state_dict_from_config(archive_reader, model_name)

    _load_state_dict._pgw1468 = True  # type: ignore[attr-defined]
    _package._load_state_dict = _load_state_dict


def uninstall() -> None:
    """Restore torch's own loader."""
    from torch.export.pt2_archive import _package

    patched = _package._load_state_dict
    if not getattr(patched, "_pgw1468", False):
        return
    for slot in getattr(patched, "__closure__", None) or ():
        candidate = slot.cell_contents
        if callable(candidate) and not getattr(candidate, "_pgw1468", False):
            _package._load_state_dict = candidate
            return


__all__ = [
    "SHARED_PAYLOAD",
    "WEIGHTS_DIR",
    "install",
    "is_weightless",
    "state_dict_from_config",
    "uninstall",
]
