"""Component — one diffusers subfolder (unet/transformer/vae/text_encoder/...)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from .writer import iter_component_tensors

if TYPE_CHECKING:
    import torch


class Component:
    """One diffusers subfolder."""

    __slots__ = ("_name", "_path", "_config")

    def __init__(self, name: str, path: Path) -> None:
        self._name = name
        self._path = path
        self._config: dict | None = None

    @property
    def name(self) -> str:
        """Subfolder name: 'unet', 'transformer', 'vae', 'text_encoder', ..."""
        return self._name

    @property
    def path(self) -> Path:
        """Absolute path to the component subdir under the source snapshot."""
        return self._path

    @property
    def config(self) -> dict:
        """Parsed component ``config.json``."""
        if self._config is None:
            cfg_path = self._path / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    def iter_tensors(self) -> Iterator[tuple[str, "torch.Tensor"]]:
        """Yield ``(name, tensor)`` pairs for every weight in this component."""

        yield from iter_component_tensors(self._path)

__all__ = ["Component"]
