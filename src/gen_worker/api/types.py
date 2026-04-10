from __future__ import annotations

import os
from typing import IO, Optional

import msgspec


class Asset(msgspec.Struct):
    """Reference to a file in the invoking owner's file store.

    The worker runtime should populate `local_path` before invoking tenant code
    so the function can open/read the file efficiently.
    """

    ref: str
    owner: Optional[str] = None
    local_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    download_token: Optional[str] = None

    def __fspath__(self) -> str:
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        return self.local_path

    def open(self, mode: str = "rb") -> IO[bytes]:
        if "b" not in mode:
            raise ValueError("Asset.open only supports binary modes")
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        return open(self.local_path, mode)

    def exists(self) -> bool:
        if self.local_path is None:
            return False
        return os.path.exists(self.local_path)

    def read_bytes(self, max_bytes: Optional[int] = None) -> bytes:
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        with open(self.local_path, "rb") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("asset too large to read into memory")
        return data


class Tensors(msgspec.Struct):
    """Reference to checkpoint/model-weight artifacts.

    This mirrors `Asset` behavior but gives tensor/checkpoint payloads a
    first-class type for training/conversion code paths.
    """

    ref: str
    owner: Optional[str] = None
    local_path: Optional[str] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    download_token: Optional[str] = None

    def __fspath__(self) -> str:
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        return self.local_path

    def open(self, mode: str = "rb") -> IO[bytes]:
        if "b" not in mode:
            raise ValueError("Tensors.open only supports binary modes")
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        return open(self.local_path, mode)

    def exists(self) -> bool:
        if self.local_path is None:
            return False
        return os.path.exists(self.local_path)

    def read_bytes(self, max_bytes: Optional[int] = None) -> bytes:
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        with open(self.local_path, "rb") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("tensors file too large to read into memory")
        return data


class LoraSpec(msgspec.Struct):
    """A LoRA adapter to load for a single inference request.

    ``file`` is materialized by the worker before the function runs, so
    ``file.local_path`` is guaranteed to be set when your function executes.
    ``weight`` controls the adapter scale (fuse strength).
    ``adapter_name`` is optional; if omitted the worker assigns ``lora_0``,
    ``lora_1``, ... based on list position.
    """

    file: Asset
    weight: float = 1.0
    adapter_name: Optional[str] = None
