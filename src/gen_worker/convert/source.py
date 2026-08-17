"""Source — library-constructed handle to the materialized source snapshot.

Tenants receive a ``Source`` as the reserved ``source`` parameter (and on
additional ``Source``-typed parameters tagged with the training-private
``_PayloadRef`` Annotated marker).
Source abstracts over singlefile vs diffusers layouts, resolves
sharded-safetensors via .index.json, and provides convenience methods for
loading into HF / diffusers / tokenizer APIs.

There is NO pickle -> safetensors conversion. pgw#1227 deleted the converter
and the pickle ban is absolute (E1/E5): reading a pickle IS the banned act, so
a pickle-only source is a typed ``pickle_only`` RepoRefusal before a byte
moves, never a format this class quietly normalizes.
"""

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
# Component dirs that carry model weights (as opposed to scheduler/tokenizer
# configuration). iter_tensors skips the rest unless explicitly named.
def _weight_component_dirs() -> frozenset[str]:
    return frozenset(weight_components())


def _detect_file_layout(path: Path) -> FileLayout:
    """MULTI_FILE when the snapshot has a model_index.json, else SINGLE_FILE."""
    if (path / "model_index.json").exists():
        return MULTI_FILE
    return SINGLE_FILE


def _enumerate_components(path: Path) -> dict[str, Component]:
    """Build the ``components`` map for a diffusers-layout snapshot."""
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
    """Handle to a materialized source snapshot.

    Constructed by the library from ``ctx.source_path`` + the resolved variant's
    attributes. Tenants never construct directly.

    Public surface:
      path              -- root of materialized snapshot (filesystem escape hatch)
      file_layout       -- "single-file" | "multi-file" (th#1937 vocabulary)
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
            if self._file_layout == MULTI_FILE:
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
        """Load via ``transformers.AutoTokenizer.from_pretrained(source.path)``.

        Cached across calls within the same tenant invocation. Raises if the
        snapshot doesn't contain tokenizer files.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(third_party_dir(self._path, why="AutoTokenizer.from_pretrained")))
        return self._tokenizer

    def diffusers_variant(self) -> str | None:
        """Detect a diffusers ``variant=`` value from files on disk
        (e.g. ``unet/diffusion_pytorch_model.fp16.safetensors`` → ``"fp16"``).
        Mirrors gen_worker.models.loading.detect_diffusers_variant — repo-cas
        mirrors cloned with a dtype preference keep HF's variant suffix."""
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
        """Auto-dispatch model load.

        Diffusers layout → ``diffusers.DiffusionPipeline.from_pretrained``.
        Singlefile layout → ``transformers.AutoModelForCausalLM.from_pretrained``.
        Override by passing an explicit ``model_cls=SomeClass`` kwarg.
        """
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

        Resolves sharded-safetensors via .index.json internally; the tenant
        sees a flat iteration. A pickle weight file is REFUSED, not converted.
        """

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

    def weights_size_bytes(self) -> int:
        """Approximate on-disk size of all weight files in this snapshot.

        Walks the weight-bearing component dirs (transformer / unet / vae /
        text_encoder* / image_encoder / prior / controlnet) for
        diffusers-layout sources, or the entire snapshot for singlefile
        sources, and sums the file sizes of any ``.safetensors``,
        ``.bin``, ``.pt``, ``.pth``, ``.ckpt`` files found.

        Used by quant tenants to size their own working set from real bytes
        without depending on the snapshot manifest plumbing — the loader only
        needs a number of bytes to reason about. It feeds no placement gate:
        th#1867 deleted the per-function VRAM declarations entirely. For bf16 sources
        this number ≈ ``num_params * 2``, which is the right multiplicand
        for the heuristic ``required_vram = scheme_factor * source_size +
        working_overhead``.
        """
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
