"""Component — one diffusers subfolder (unet/transformer/vae/text_encoder/...).

A diffusers-layout ``Source`` exposes ``components: dict[str, Component]``.
Each Component is a self-contained HF module directory with its own
config.json and weight file(s). Tenants can iterate tensors of one specific
component (``source.components['transformer'].iter_tensors()``) or let the
library's ``StreamingWriter`` auto-passthrough components the tenant didn't
touch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:
    import torch


class Component:
    """One diffusers subfolder. Constructed by the library; tenants don't."""

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
        """Parsed component ``config.json``. Empty dict if the file is absent."""
        if self._config is None:
            cfg_path = self._path / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    def iter_tensors(self) -> Iterator[tuple[str, "torch.Tensor"]]:
        """Yield ``(name, tensor)`` pairs for every weight in this component.

        Handles sharded safetensors via .index.json transparently. Names are
        local to the component (e.g. ``conv_in.weight``), NOT prefixed with
        the component name.
        """
        from ._tensor_iter import iter_component_tensors

        yield from iter_component_tensors(self._path)

    def as_hf_module(self, **kwargs: Any) -> Any:
        """Load this component as its HF class.

        Dispatches on ``config['_class_name']`` or ``config['class_name']`` —
        e.g. ``UNet2DConditionModel``, ``AutoencoderKL``, ``CLIPTextModel``.
        Falls back to generic loading via transformers.AutoModel when no
        diffusers class is matched.
        """
        from ._hf_load import load_component_module

        return load_component_module(self._path, self.config, **kwargs)


__all__ = ["Component"]
