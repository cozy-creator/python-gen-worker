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
from .loaded_component import LoadedComponent

if TYPE_CHECKING:
    import torch

FileLayout = Literal["singlefile", "diffusers"]


# Default set of component subdirs that the iter_hf_components passthrough
# path covers when the tenant doesn't explicitly opt them out. These are
# config / tokenizer / scheduler / non-quantizable bits that should travel
# verbatim into the output snapshot. vae is here because most quant tenants
# (bnb / torchao) leave the vae bf16; modelopt may override per spec.
_DEFAULT_PASSTHROUGH_COMPONENTS: frozenset[str] = frozenset({
    "vae", "scheduler",
    "tokenizer", "tokenizer_2", "tokenizer_3",
    "feature_extractor", "safety_checker",
})

# Components that are candidates for quantization. text_encoder_3 is in here
# because flux.2-klein-9b / SD3 use three text encoders; older tenants that
# hardcoded ('transformer', 'unet', 'text_encoder', 'text_encoder_2', 'vae')
# silently skipped text_encoder_3.
_DEFAULT_QUANT_CANDIDATE_COMPONENTS: frozenset[str] = frozenset({
    "transformer", "unet",
    "text_encoder", "text_encoder_2", "text_encoder_3",
    "image_encoder", "prior", "controlnet",
})

# Top-level file basenames yielded by the synthetic '_root' LoadedComponent.
# model_index.json is the diffusers pipeline manifest; everything else is
# documentation / license that callers expect to travel with the snapshot.
_ROOT_PASSTHROUGH_FILES: tuple[str, ...] = (
    "model_index.json",
    "README.md",
    "LICENSE.md",
    "LICENSE",
    "USAGE_POLICY.md",
)


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

    def weights_size_bytes(self) -> int:
        """Approximate on-disk size of all weight files in this snapshot.

        Walks the weight-bearing component dirs (transformer / unet / vae /
        text_encoder* / image_encoder / prior / controlnet) for
        diffusers-layout sources, or the entire snapshot for singlefile
        sources, and sums the file sizes of any ``.safetensors``,
        ``.bin``, ``.pt``, ``.pth``, ``.ckpt`` files found.

        Used by quant tenants to compute per-scheme `require_vram(...)` gates
        without depending on the snapshot manifest plumbing — the loader
        only needs a number of bytes to reason about. For bf16 sources
        this number ≈ ``num_params * 2``, which is the right multiplicand
        for the heuristic ``required_vram = scheme_factor * source_size +
        working_overhead``.
        """
        weight_exts = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
        total = 0
        if self._file_layout == "diffusers":
            for comp_name in _WEIGHT_COMPONENT_DIRS:
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

    def largest_quant_component_size_bytes(self) -> int:
        """Bytes of the largest single quantizable component in this snapshot.

        ``iter_hf_components`` loads ONE component at a time via its own
        ``from_pretrained(quantization_config=...)`` call and the tenant
        drops the reference (``component.save_to(...)`` then loop again)
        before the next component loads. Peak VRAM during quantization is
        therefore bounded by the bf16 footprint of the **largest single
        component** plus a fixed working overhead — NOT the sum of all
        components.

        Quant tenants should use this method (not ``weights_size_bytes()``)
        as the multiplicand for their ``require_vram(...)`` heuristic.
        Using the sum over-gates by ~2x on multi-component diffusion
        pipelines (FLUX-class: transformer ≈ text_encoder ≈ 7.5 GB each,
        sum = 15 GB; largest = 7.5 GB; the gate difference rules
        consumer-GPU hosts in or out of int8 quant).

        For singlefile sources (one component anyway) this returns the
        same value as ``weights_size_bytes()``.
        """
        weight_exts = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
        if self._file_layout != "diffusers":
            return self.weights_size_bytes()
        largest = 0
        # Iterate only the quant-candidate set (matches iter_hf_components'
        # default). VAE is a passthrough component in current quant tenants
        # and isn't a candidate for VRAM-peak; including it would harmlessly
        # inflate the floor by a tiny amount but the strict reading is
        # quant-candidates only.
        for comp_name in _DEFAULT_QUANT_CANDIDATE_COMPONENTS:
            comp_path = self._path / comp_name
            if not comp_path.is_dir():
                continue
            comp_size = 0
            for f in comp_path.rglob("*"):
                if f.is_file() and f.suffix in weight_exts:
                    try:
                        comp_size += f.stat().st_size
                    except OSError:
                        continue
            if comp_size > largest:
                largest = comp_size
        return largest

    # ----- streaming, offload-aware quantization loop ------------------

    def iter_hf_components(
        self,
        *,
        quant_only: Iterable[str] | None = None,
        quantization_config: Any = None,
        compute_dtype: "torch.dtype" | None = None,
        offload_folder: Path | str | None = None,
        passthrough_only: Iterable[str] | None = None,
    ) -> Iterator[LoadedComponent]:
        """Yield ``LoadedComponent`` payloads in arrival / discovery order.

        This is the recommended path for quant tenants (bnb / torchao /
        modelopt). It pushes three concerns into the library:

          1. **Per-component HF loading.** Each heavy component is loaded
             via ``<Class>.from_pretrained(component_path,
             quantization_config=..., torch_dtype=compute_dtype,
             device_map='auto', offload_folder=offload_folder)``. The
             quantization_config is what triggers bnb / torchao / modelopt
             to swap weights in-place during load. ``device_map='auto'``
             plus ``offload_folder`` lets accelerate spill to disk when
             VRAM is tight — that's how a 4 GB transformer fits on a 7 GB
             card without OOM-walks in tenant code.

          2. **Layout dispatch.** The same iteration shape works for
             diffusers-layout (multiple component subdirs), diffusers
             singlefile (one .safetensors reconstructed via
             ``from_single_file``), and transformers singlefile (one
             ``AutoModelForCausalLM.from_pretrained``). The tenant doesn't
             care which it is.

          3. **Passthrough vs quant decision.** Components matching
             ``quant_only`` are loaded + quantized; the rest are yielded
             with ``kind='passthrough'`` and ``save_to`` does a verbatim
             ``shutil.copytree``. A final synthetic ``LoadedComponent
             (name='_root')`` carries top-level files (``model_index.json``,
             README, LICENSE).

        Args:
          quant_only: Iterable of component names to quantize. Default is
            the library's "all weight-bearing components" set
            (``transformer``, ``unet``, ``text_encoder``, ``text_encoder_2``,
            ``text_encoder_3``, ``image_encoder``, ``prior``,
            ``controlnet``). Passing an explicit iterable narrows or
            widens.
          quantization_config: HuggingFace quant config object —
            ``BitsAndBytesConfig``, ``TorchAoConfig``,
            ``ModeloptQuantizationConfig``, etc. ``None`` skips quant
            (then ``quant_only`` becomes a passthrough-on-load filter,
            useful for dtype-cast tenants).
          compute_dtype: torch dtype used as ``torch_dtype`` on the
            ``from_pretrained`` call. For bnb 4-bit this is the bf16/fp16
            compute dtype the dequant uses at inference.
          offload_folder: Path accelerate uses when ``device_map='auto'``
            decides to spill layers off-GPU during load. Pass
            ``ctx.mktemp() / 'offload'``; the library does NOT clean it
            up — caller controls lifetime via ctx.
          passthrough_only: Iterable of component names that bypass the
            loader entirely and are copied verbatim. Defaults to
            ``vae / scheduler / tokenizer* / feature_extractor /
            safety_checker``. Anything in ``passthrough_only`` AND
            ``quant_only`` is treated as quant_only (explicit wins).

        Yields LoadedComponent objects. Order is loosely the diffusers
        component-discovery order; expect ``passthrough`` components to
        arrive before heavy ``quantized`` ones because the loader work
        dominates wall-clock.
        """
        quant_set = (
            frozenset(quant_only)
            if quant_only is not None
            else _DEFAULT_QUANT_CANDIDATE_COMPONENTS
        )
        passthrough_set = (
            frozenset(passthrough_only)
            if passthrough_only is not None
            else _DEFAULT_PASSTHROUGH_COMPONENTS
        )
        offload_path = Path(offload_folder) if offload_folder is not None else None
        if offload_path is not None:
            offload_path.mkdir(parents=True, exist_ok=True)

        if self._file_layout == "diffusers":
            yield from _iter_diffusers_components(
                self._path,
                quant_set=quant_set,
                passthrough_set=passthrough_set,
                quantization_config=quantization_config,
                compute_dtype=compute_dtype,
                offload_folder=offload_path,
            )
        else:
            yield from _iter_singlefile_components(
                self._path,
                quantization_config=quantization_config,
                compute_dtype=compute_dtype,
                offload_folder=offload_path,
            )

    def iter_hf_components_streaming(
        self,
        *,
        quant_only: Iterable[str] | None = None,
        compute_dtype: "torch.dtype" | None = None,
        passthrough_only: Iterable[str] | None = None,
    ) -> Iterator[LoadedComponent]:
        """Yield ``LoadedComponent`` payloads with CPU-resident bf16 modules.

        Same iteration shape as :meth:`iter_hf_components`, but quantization
        is the tenant's responsibility. Quant-candidate components are loaded
        with ``device_map='cpu'`` + ``torch_dtype=compute_dtype`` and yielded
        as ``kind='bf16_cpu'``. The tenant mutates ``component.module``
        in-place (typically via a library-specific streaming-quant call) and
        then invokes ``component.save_to(out_dir)``.

        Use this for libraries that have a documented streaming-quant API,
        like torchao's ``quantize_(module, config, device='cuda')``, which
        sends each ``Linear`` to GPU individually rather than staging the
        full bf16 component there. Peak VRAM ≈ size of the largest single
        ``Linear`` plus the accumulating quantized result — not the full
        component. Lets FLUX-class int8/fp8 conversions complete on 8 GB
        consumer GPUs.

        For non-streaming libraries (bnb, modelopt), keep using
        :meth:`iter_hf_components` with a ``quantization_config``.

        Args:
          quant_only: Iterable of component names to load on CPU. Default
            is the library's "all weight-bearing components" set.
          compute_dtype: torch dtype used as ``torch_dtype`` on the load.
            Typically ``torch.bfloat16``.
          passthrough_only: Iterable of component names that bypass the
            loader entirely and are copied verbatim (vae / scheduler /
            tokenizer*, etc.).

        Yields LoadedComponent objects. quant-candidate components arrive
        with ``kind='bf16_cpu'``; passthrough with ``kind='passthrough'``;
        the synthetic ``_root`` component arrives last with ``kind='root'``.
        """
        quant_set = (
            frozenset(quant_only)
            if quant_only is not None
            else _DEFAULT_QUANT_CANDIDATE_COMPONENTS
        )
        passthrough_set = (
            frozenset(passthrough_only)
            if passthrough_only is not None
            else _DEFAULT_PASSTHROUGH_COMPONENTS
        )

        if self._file_layout == "diffusers":
            yield from _iter_diffusers_components_streaming(
                self._path,
                quant_set=quant_set,
                passthrough_set=passthrough_set,
                compute_dtype=compute_dtype,
            )
        else:
            yield from _iter_singlefile_components_streaming(
                self._path,
                compute_dtype=compute_dtype,
            )


def _iter_diffusers_components(
    snapshot_path: Path,
    *,
    quant_set: frozenset[str],
    passthrough_set: frozenset[str],
    quantization_config: Any,
    compute_dtype: Any,
    offload_folder: Path | None,
) -> Iterator[LoadedComponent]:
    """Walk a diffusers-layout snapshot directory and yield LoadedComponents.

    Each subdir under ``snapshot_path`` whose name is in the diffusers
    component allowlist is yielded once. ``quant_set`` decides quantize-vs-
    passthrough. A final ``_root`` LoadedComponent carries the top-level
    files (model_index.json, README, etc).
    """
    from ._hf_load import load_component_module

    if not snapshot_path.is_dir():
        return

    seen: set[str] = set()
    for entry in sorted(snapshot_path.iterdir()):
        if not entry.is_dir() or entry.name not in _DIFFUSERS_COMPONENT_DIRS:
            continue
        seen.add(entry.name)
        if entry.name in quant_set and quantization_config is not None:
            # device_map shape — when the caller passed an offload_folder the
            # intent is "spill to CPU/disk if needed", so use accelerate's
            # auto-placement. When no offload_folder is set, the caller has
            # already gated VRAM availability (typically via require_vram on
            # the tenant) and asks for hard GPU-only placement: failing loud
            # on OOM is cleaner than silent spillage that bnb can't honor for
            # nf4/fp4 unless `llm_int8_enable_fp32_cpu_offload=True`.
            kwargs: dict[str, Any] = {"quantization_config": quantization_config}
            if compute_dtype is not None:
                kwargs["torch_dtype"] = compute_dtype
            if offload_folder is not None:
                kwargs["device_map"] = "auto"
                # Per-component subdir under the shared offload root, so
                # parallel components don't clobber each other if accelerate
                # uses identical layer-key names across submodules.
                comp_offload = offload_folder / entry.name
                comp_offload.mkdir(parents=True, exist_ok=True)
                kwargs["offload_folder"] = str(comp_offload)
            else:
                kwargs["device_map"] = {"": 0}
            module = load_component_module(entry, _read_component_config(entry), **kwargs)
            yield LoadedComponent(
                name=entry.name,
                kind="quantized",
                _module=module,
                metadata={
                    "loader": "from_pretrained",
                    "quantization_config_class": type(quantization_config).__name__,
                },
            )
            continue
        # Default: copy verbatim. We don't load passthrough components into
        # memory. Anything not in quant_set lands here even if it's also
        # not in passthrough_set — being explicit about passthrough_set is
        # documentation-only; the absence of quant intent is what triggers
        # the copy.
        yield LoadedComponent(
            name=entry.name,
            kind="passthrough",
            _source_path=entry,
            metadata={"loader": "passthrough_copy"},
        )

    # Heads-up if the snapshot contains a quant_set member that doesn't
    # exist on disk — usually means a hardcoded enumeration drifted vs the
    # actual model. Don't raise; just don't yield it.
    missing_from_quant_set = quant_set - seen
    if missing_from_quant_set:
        # The Source.iter_hf_components docstring warns tenants to expect
        # only-yielded-when-present semantics; this comment locks in that
        # behavior.
        pass

    # Final synthetic _root: the top-level files that don't live under any
    # component subdir. model_index.json is the diffusers pipeline manifest;
    # without it the saved snapshot won't reload as a pipeline.
    root_files = [
        snapshot_path / fname
        for fname in _ROOT_PASSTHROUGH_FILES
        if (snapshot_path / fname).is_file()
    ]
    if root_files:
        yield LoadedComponent(name="_root", kind="root", _root_files=root_files)


def _iter_singlefile_components(
    snapshot_path: Path,
    *,
    quantization_config: Any,
    compute_dtype: Any,
    offload_folder: Path | None,
) -> Iterator[LoadedComponent]:
    """Yield a single LoadedComponent for a singlefile-layout snapshot.

    Two flavors handled here:
      - **transformers singlefile** — directory with ``config.json`` +
        weights at the top level (e.g. a llama / qwen model dump). Loaded
        via ``AutoModelForCausalLM.from_pretrained(path, quantization_config=
        ..., torch_dtype=..., device_map='auto', offload_folder=...)``.
      - **diffusers singlefile** — one ``.safetensors`` file reconstructible
        via ``from_single_file``. Less common; the library defers to the
        diffusers-layout caller (``StableDiffusionPipeline.from_single_file``)
        and yields the resulting pipeline as a single ``LoadedComponent
        (name='model')``.

    For now the diffusers singlefile path raises NotImplementedError if
    invoked — the calling tenant can fall back to the legacy ``as_hf_model``
    + manual quant path. Wiring it up is a small follow-up.
    """
    if (snapshot_path / "config.json").is_file():
        # See _iter_diffusers_components for the device_map rationale —
        # offload_folder=None means GPU-only (VRAM-gated tenant).
        kwargs: dict[str, Any] = {}
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if compute_dtype is not None:
            kwargs["torch_dtype"] = compute_dtype
        if offload_folder is not None:
            kwargs["device_map"] = "auto"
            kwargs["offload_folder"] = str(offload_folder)
        else:
            kwargs["device_map"] = {"": 0}
        from transformers import AutoModelForCausalLM
        module = AutoModelForCausalLM.from_pretrained(str(snapshot_path), **kwargs)
        yield LoadedComponent(
            name="model",
            kind="quantized" if quantization_config is not None else "passthrough",
            _module=module if quantization_config is not None else None,
            _source_path=snapshot_path if quantization_config is None else None,
            metadata={
                "loader": "from_pretrained",
                "layout": "transformers_singlefile",
                "quantization_config_class": (
                    type(quantization_config).__name__
                    if quantization_config is not None else None
                ),
            },
        )
        return

    raise NotImplementedError(
        "iter_hf_components: diffusers-singlefile sources are not yet "
        "supported. Use the legacy Source.as_hf_model() path and quantize "
        "the resulting DiffusionPipeline manually for now."
    )


def _iter_diffusers_components_streaming(
    snapshot_path: Path,
    *,
    quant_set: frozenset[str],
    passthrough_set: frozenset[str],
    compute_dtype: Any,
) -> Iterator[LoadedComponent]:
    """Streaming-quant variant of ``_iter_diffusers_components``.

    Yields ``kind='bf16_cpu'`` for quant-candidate components (loaded on CPU
    in ``compute_dtype``, no ``quantization_config``). Tenant is expected
    to invoke a library-specific in-place quant call (e.g.
    ``torchao.quantize_(component.module, config, device='cuda')``) before
    ``component.save_to(out_dir)``.

    Passthrough and root yields are identical to the non-streaming variant.
    """
    from ._hf_load import load_component_module

    if not snapshot_path.is_dir():
        return

    seen: set[str] = set()
    for entry in sorted(snapshot_path.iterdir()):
        if not entry.is_dir() or entry.name not in _DIFFUSERS_COMPONENT_DIRS:
            continue
        seen.add(entry.name)
        if entry.name in quant_set:
            # CPU load; no quantization_config; no device_map placement on
            # accelerator. The tenant moves layers to GPU one at a time via
            # the library's streaming-quant API.
            kwargs: dict[str, Any] = {"device_map": "cpu"}
            if compute_dtype is not None:
                kwargs["torch_dtype"] = compute_dtype
            module = load_component_module(entry, _read_component_config(entry), **kwargs)
            yield LoadedComponent(
                name=entry.name,
                kind="bf16_cpu",
                _module=module,
                metadata={
                    "loader": "from_pretrained",
                    "streaming_quant": True,
                },
            )
            continue
        # Same passthrough behavior as the non-streaming path.
        yield LoadedComponent(
            name=entry.name,
            kind="passthrough",
            _source_path=entry,
            metadata={"loader": "passthrough_copy"},
        )

    # Same _root behavior — top-level files unaffected by streaming mode.
    root_files = [
        snapshot_path / fname
        for fname in _ROOT_PASSTHROUGH_FILES
        if (snapshot_path / fname).is_file()
    ]
    if root_files:
        yield LoadedComponent(name="_root", kind="root", _root_files=root_files)


def _iter_singlefile_components_streaming(
    snapshot_path: Path,
    *,
    compute_dtype: Any,
) -> Iterator[LoadedComponent]:
    """Streaming-quant variant of ``_iter_singlefile_components``.

    Loads a transformers-style singlefile snapshot onto CPU in
    ``compute_dtype`` and yields it as ``kind='bf16_cpu'``. The diffusers-
    singlefile path remains NotImplementedError, matching the non-streaming
    variant.
    """
    if (snapshot_path / "config.json").is_file():
        kwargs: dict[str, Any] = {"device_map": "cpu"}
        if compute_dtype is not None:
            kwargs["torch_dtype"] = compute_dtype
        from transformers import AutoModelForCausalLM
        module = AutoModelForCausalLM.from_pretrained(str(snapshot_path), **kwargs)
        yield LoadedComponent(
            name="model",
            kind="bf16_cpu",
            _module=module,
            metadata={
                "loader": "from_pretrained",
                "layout": "transformers_singlefile",
                "streaming_quant": True,
            },
        )
        return

    raise NotImplementedError(
        "iter_hf_components_streaming: diffusers-singlefile sources are not "
        "yet supported. Use the legacy Source.as_hf_model() path and "
        "quantize the resulting DiffusionPipeline manually for now."
    )


def _read_component_config(component_path: Path) -> dict:
    cfg_path = component_path / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = ["Source", "FileLayout"]
