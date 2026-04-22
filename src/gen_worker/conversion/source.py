"""Source — library-constructed handle to the materialized source snapshot.

Tenants receive a ``Source`` as the reserved ``source`` parameter (and as
additional ``Annotated[Source, ModelRef(Src.PAYLOAD, ...)]`` parameters).
Source abstracts over singlefile vs diffusers layouts, handles pickle →
safetensors conversion, resolves sharded-safetensors via .index.json, and
provides convenience methods for loading into HF / diffusers / tokenizer APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal

from .component import Component

if TYPE_CHECKING:
    import torch

FileLayout = Literal["singlefile", "diffusers"]


_DIFFUSERS_COMPONENT_DIRS: frozenset[str] = frozenset({
    "unet", "transformer", "vae", "text_encoder", "text_encoder_2",
    "text_encoder_3", "image_encoder", "prior", "controlnet", "scheduler",
    "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor",
    "safety_checker",
})
# Component dirs that carry model weights (as opposed to scheduler/tokenizer
# configuration). iter_tensors skips the rest unless explicitly named.
_WEIGHT_COMPONENT_DIRS: frozenset[str] = frozenset({
    "unet", "transformer", "vae", "text_encoder", "text_encoder_2",
    "text_encoder_3", "image_encoder", "prior", "controlnet",
})


def _detect_file_layout(path: Path) -> FileLayout:
    """Return 'diffusers' if the snapshot has a model_index.json, else 'singlefile'."""
    if (path / "model_index.json").exists():
        return "diffusers"
    return "singlefile"


def _enumerate_components(path: Path) -> dict[str, Component]:
    """Build the ``components`` map for a diffusers-layout snapshot."""
    result: dict[str, Component] = {}
    if not path.is_dir():
        return result
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in _DIFFUSERS_COMPONENT_DIRS:
            result[entry.name] = Component(entry.name, entry)
    return result


class Source:
    """Handle to a materialized source snapshot.

    Constructed by the library from ``ctx.source_path`` + the resolved variant's
    attributes. Tenants never construct directly.

    Public surface:
      path              -- root of materialized snapshot (filesystem escape hatch)
      file_layout       -- "singlefile" | "diffusers"
      attributes        -- full resolved variant attribute map (provenance)
      ref               -- the wire ref string (e.g. "owner/repo") for logging
      components        -- dict[str, Component] for diffusers; {} for singlefile
      config()          -- parsed config.json / model_index.json
      tokenizer()       -- AutoTokenizer.from_pretrained(path)
      as_hf_model()     -- auto-dispatch to CausalLM / DiffusionPipeline / ...
      iter_tensors()    -- yield (component, name, tensor) across all weights
      state_dict()      -- eager variant of iter_tensors
      hf_dir()          -- directory suitable for path-in-path-out tools
    """

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

    # ----- simple attrs ------------------------------------------------

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
        """Diffusers component map. Empty for singlefile sources."""
        if self._components is None:
            if self._file_layout == "diffusers":
                self._components = _enumerate_components(self._path)
            else:
                self._components = {}
        return self._components

    # ----- cached loaders ---------------------------------------------

    def config(self) -> dict:
        """Parsed top-level config. model_index.json for diffusers, config.json for singlefile.

        Returns ``{}`` if no config file is present (rare — a snapshot should
        always have one but we don't want to crash the tenant on odd sources).
        """
        if self._config is None:
            if self._file_layout == "diffusers":
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
        """Load via ``transformers.AutoTokenizer.from_pretrained(source.path)``.

        Cached across calls within the same tenant invocation. Raises if the
        snapshot doesn't contain tokenizer files.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
        return self._tokenizer

    def as_hf_model(self, **kwargs: Any) -> Any:
        """Auto-dispatch model load.

        Diffusers layout → ``diffusers.DiffusionPipeline.from_pretrained``.
        Singlefile layout → ``transformers.AutoModelForCausalLM.from_pretrained``.
        Override by passing an explicit ``model_cls=SomeClass`` kwarg.
        """
        model_cls = kwargs.pop("model_cls", None)
        if model_cls is not None:
            return model_cls.from_pretrained(str(self._path), **kwargs)
        if self._file_layout == "diffusers":
            from diffusers import DiffusionPipeline
            return DiffusionPipeline.from_pretrained(str(self._path), **kwargs)
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(str(self._path), **kwargs)

    # ----- tensor access ----------------------------------------------

    def iter_tensors(
        self, components: list[str] | None = None,
    ) -> Iterator[tuple[str, str, "torch.Tensor"]]:
        """Stream every weight tensor. Yields ``(component, name, tensor)``.

        - For singlefile sources: component='' for all yields.
        - For diffusers sources: component is the subdir name (unet/vae/...).
          Only components with weight files are iterated; scheduler/tokenizer
          subdirs are skipped.
        - If ``components`` is passed, only those components are iterated.
          The library's StreamingWriter auto-passes untouched components on
          finalize() — the tenant can filter iteration without losing output
          coverage.

        Handles pickle → safetensors conversion and sharded-safetensors via
        .index.json internally. Tenant sees a flat iteration.
        """
        from ._tensor_iter import iter_source_tensors

        yield from iter_source_tensors(
            self._path,
            file_layout=self._file_layout,
            components_filter=components,
        )

    def state_dict(
        self, components: list[str] | None = None,
    ) -> dict[str, "torch.Tensor"]:
        """Eager variant of iter_tensors.

        Returns ``{dotted_name: tensor}``. For diffusers, dotted names include
        the component prefix (e.g. 'unet.conv_in.weight'). For singlefile,
        dotted names are the raw safetensors keys.
        """
        result: dict[str, Any] = {}
        for component, name, tensor in self.iter_tensors(components=components):
            key = f"{component}.{name}" if component else name
            result[key] = tensor
        return result

    def hf_dir(self) -> Path:
        """Return a directory path suitable for path-in-path-out subprocess tools.

        For most cases this is ``self.path`` directly. Subclasses / future
        helpers may return a prepared subtree (e.g. for llama.cpp's
        prepare_hf_source_tree_for_gguf fixup).
        """
        return self._path


__all__ = ["Source", "FileLayout"]
