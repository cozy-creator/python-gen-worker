"""Source — library-constructed handle to the materialized source snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from .component import Component
from .writer import iter_source_tensors

if TYPE_CHECKING:
    import torch

from ..models.file_layout import MULTI_FILE, SINGLE_FILE, FileLayout  # noqa: F401  (re-exported)
from ..models.materialized_view import third_party_dir

from ..component_vocab import (
    pipeline_component_dirs,
    weight_components,
)


def _diffusers_component_dirs() -> frozenset[str]:
    return frozenset(pipeline_component_dirs())
def _weight_component_dirs() -> frozenset[str]:
    return frozenset(weight_components())


def _detect_file_layout(path: Path) -> FileLayout:
    if (path / "model_index.json").exists():
        return MULTI_FILE
    return SINGLE_FILE


def _enumerate_components(path: Path) -> dict[str, Component]:
    result: dict[str, Component] = {}
    if not path.is_dir():
        return result
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in _diffusers_component_dirs():
            result[entry.name] = Component(entry.name, entry)
    return result


class Source:
    """Handle to a materialized source snapshot."""

    def __init__(
        self,
        path: Path,
        *,
        attributes: dict | None = None,
        ref: str = "",
    ) -> None:
        self._path = Path(path)
        self._attributes = dict(attributes or {})
        self._ref = ref
        self._file_layout: FileLayout = _detect_file_layout(self._path)
        self._components: dict[str, Component] | None = None
        self._config: dict | None = None
        self._tokenizer: Any = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def file_layout(self) -> FileLayout:
        return self._file_layout

    @property
    def attributes(self) -> dict:
        return self._attributes

    @property
    def ref(self) -> str:
        return self._ref

    @property
    def components(self) -> dict[str, Component]:
        """Diffusers component map."""
        if self._components is None:
            if self._file_layout == MULTI_FILE:
                self._components = _enumerate_components(self._path)
            else:
                self._components = {}
        return self._components

    def config(self) -> dict:
        """Parsed top-level config."""
        if self._config is None:
            if self._file_layout == MULTI_FILE:
                candidate = self._path / "model_index.json"
            else:
                candidate = self._path / "config.json"
            if candidate.exists():
                with open(candidate) as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    def tokenizer(self) -> Any:
        """Load via ``transformers.AutoTokenizer.from_pretrained(source.path)``."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(third_party_dir(self._path, why="AutoTokenizer.from_pretrained")))
        return self._tokenizer

    def diffusers_variant(self) -> str | None:
        """Detect a diffusers ``variant=`` value from files on disk (e.g."""
        if self._file_layout != MULTI_FILE:
            return None
        candidates = ("bf16", "fp8", "fp16", "int8", "int4")
        try:
            for p in self._path.rglob("*.safetensors"):
                name = p.name.lower()
                for v in candidates:
                    if f".{v}." in name or name.endswith(f".{v}.safetensors"):
                        return v
        except OSError:
            return None
        return None

    def as_hf_model(self, **kwargs: Any) -> Any:
        """Auto-dispatch model load."""
        model_cls = kwargs.pop("model_cls", None)
        if self._file_layout == MULTI_FILE and "variant" not in kwargs:
            if v := self.diffusers_variant():
                kwargs["variant"] = v
        if model_cls is not None:
            return model_cls.from_pretrained(
                str(third_party_dir(self._path, why="conversion source model_cls")),
                **kwargs)
        if self._file_layout == MULTI_FILE:
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(
                str(third_party_dir(self._path, why="conversion source pipeline")),
                **kwargs)
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            str(third_party_dir(self._path, why="conversion source causal LM")),
            **kwargs)

    def iter_tensors(
        self, components: list[str] | None = None,
    ) -> Iterator[tuple[str, str, "torch.Tensor"]]:
        """Stream every weight tensor."""

        yield from iter_source_tensors(
            self._path,
            file_layout=self._file_layout,
            components_filter=components,
        )

    def state_dict(
        self, components: list[str] | None = None,
    ) -> dict[str, "torch.Tensor"]:
        """Eager variant of iter_tensors."""
        result: dict[str, Any] = {}
        for component, name, tensor in self.iter_tensors(components=components):
            key = f"{component}.{name}" if component else name
            result[key] = tensor
        return result

    def hf_dir(self) -> Path:
        """Return a directory path suitable for path-in-path-out subprocess tools."""
        return self._path

    def weights_size_bytes(self) -> int:
        """Approximate on-disk size of all weight files in this snapshot."""
        weight_exts = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
        total = 0
        if self._file_layout == MULTI_FILE:
            for comp_name in _weight_component_dirs():
                comp_path = self._path / comp_name
                if not comp_path.is_dir():
                    continue
                for f in comp_path.rglob("*"):
                    if f.is_file() and f.suffix in weight_exts:
                        try:
                            total += f.stat().st_size
                        except OSError:
                            continue
        else:
            for f in self._path.rglob("*"):
                if f.is_file() and f.suffix in weight_exts:
                    try:
                        total += f.stat().st_size
                    except OSError:
                        continue
        return total



__all__ = ["Source", "FileLayout"]
