"""Step 1 of ``ctx.load``: the pipeline, built from CONFIGS, holding no bytes.

``model_index.json`` names every component and the class that builds it. Each
nn.Module component is constructed ``from_config`` under
:func:`gen_worker.models.meta_init.init_empty_weights`, so its parameters land
on ``meta`` — no storage, no read, no allocation. Everything else (scheduler,
tokenizer, feature extractor, image processor) keeps its stock
``from_pretrained`` against the projected tree: those are small REAL files that
stay symlinks, and reading a 2 KB tokenizer through a streaming engine would
be ceremony without a payoff.

The result is a genuine instance of the pipeline class the author named. It is
not a proxy, not a subclass, not a patched object — handlers, ``ctx.compile``
marking and ``load_lora_weights`` all meet exactly what they would have met
after ``from_pretrained``, minus the weights, which arrive next.

Symmetry with the publish moment (pgw#1370): the SAME construction on meta,
with nothing streamed into it, is what the derive traces. One surface, two
moments — the ``ctx.compile`` duality, for weights.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ...models.meta_init import init_empty_weights

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

MODEL_INDEX = "model_index.json"


class SkeletonError(RuntimeError):
    """The meta skeleton could not be built from configs alone."""


@dataclass(slots=True)
class Skeleton:
    """A meta-built pipeline and the components weights must reach."""

    pipeline: Any
    #: component name -> the nn.Module built on meta. Component ``""`` is a
    #: single-module checkpoint with no ``model_index.json``.
    modules: Dict[str, Any]
    #: Components that came from a stock ``from_pretrained`` (small real
    #: files); no tensor container of theirs is streamed.
    passthrough: Tuple[str, ...] = ()


def _resolve(library: str, class_name: str) -> type:
    """The class a ``model_index.json`` entry names.

    A library entry is EITHER an importable module (``transformers``,
    ``diffusers``) OR a diffusers PIPELINE SUBMODULE name — ``stable_diffusion``
    for ``StableDiffusionSafetyChecker``, which is not importable as a
    top-level module and never was. diffusers' own loader tests exactly this,
    in this order (`hasattr(diffusers.pipelines, library_name)` →
    ``getattr(pipeline_module, class_name)``), so this mirrors it rather than
    inventing a second rule for the same file format.

    Found by RUNNING it (pgw#1518, via pgw#1491's acceptance): every sd15 checkpoint on the
    hub names ``stable_diffusion.StableDiffusionSafetyChecker``, and
    `gen-worker up` died on it. It had gone unseen because the eager bridge
    (``from_pretrained``) resolves it correctly and this skeleton is reached
    ONLY when the tree has a chunk store behind it — i.e. exactly the
    production-shaped path, and never the bare-tree local one the campaign
    used.
    """
    module: Any
    try:
        module = importlib.import_module(library)
    except ImportError:
        module = None
    if module is None or not hasattr(module, class_name):
        pipelines = _diffusers_pipelines()
        submodule = getattr(pipelines, library, None) if pipelines else None
        if submodule is not None and hasattr(submodule, class_name):
            module = submodule
    if module is None:
        raise SkeletonError(
            f"{MODEL_INDEX} names component class {library}.{class_name}, "
            f"but {library!r} is neither importable in this image nor a "
            f"diffusers pipeline submodule"
        )
    found = getattr(module, class_name, None)
    if found is None or not isinstance(found, type):
        raise SkeletonError(
            f"{library}.{class_name} is not a class in this image's "
            f"{library} version"
        )
    return found


def _diffusers_pipelines() -> Any:
    """``diffusers.pipelines``, or ``None`` when diffusers is absent.

    Absent is legal: a non-diffusers endpoint's model_index names no pipeline
    submodule, and importing diffusers to prove that would be a hard dependency
    this module does not otherwise have.
    """
    try:
        import diffusers.pipelines as pipelines
    except ImportError:
        return None
    return pipelines


def _is_module(cls: type) -> bool:
    import torch

    return issubclass(cls, torch.nn.Module)


def _build_on_meta(cls: type, directory: Path, component: str) -> Any:
    """Construct one nn.Module component from its config, on meta."""
    load_config = getattr(cls, "load_config", None)
    from_config = getattr(cls, "from_config", None)
    if callable(load_config) and callable(from_config):
        # diffusers ConfigMixin: the config is a plain dict on disk.
        config = load_config(str(directory))
        if isinstance(config, tuple):
            config = config[0]
        with init_empty_weights():
            return from_config(config)

    config_class = getattr(cls, "config_class", None)
    if config_class is not None and hasattr(config_class, "from_pretrained"):
        # transformers PreTrainedModel: the config is a typed object.
        config = config_class.from_pretrained(str(directory))
        with init_empty_weights():
            return cls(config)

    raise SkeletonError(
        f"component {component!r} ({cls.__module__}.{cls.__name__}) exposes "
        f"neither a diffusers from_config nor a transformers config_class, so "
        f"it cannot be built weight-free; ctx.load never reads a checkpoint "
        f"file to find out what a class wants"
    )


def build(
    pipeline_cls: type,
    checkpoint_dir: Path,
    *,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Skeleton:
    """Build ``pipeline_cls`` from configs only. Reads no tensor bytes."""
    index_path = Path(checkpoint_dir) / MODEL_INDEX
    if not index_path.is_file():
        raise SkeletonError(
            f"{checkpoint_dir} carries no {MODEL_INDEX}; ctx.load builds a "
            f"pipeline from its component index, and a tree without one is "
            f"not a pipeline checkpoint"
        )
    try:
        index = json.loads(index_path.read_text())
    except ValueError as exc:
        raise SkeletonError(f"{index_path} is not readable JSON: {exc}") from exc

    components: Dict[str, Any] = {}
    modules: Dict[str, Any] = {}
    passthrough: List[str] = []

    for name, spec in sorted(index.items()):
        if name.startswith("_"):
            continue
        # pgw#1618: A SCALAR ENTRY IS PIPELINE CONFIG, NOT A COMPONENT, and a
        # CANONICAL diffusers index carries several. SDXL's ships
        # `"force_zeros_for_empty_prompt": true` and `"add_watermarker": null`
        # right beside its eight component pairs; neither starts with `_`, so
        # the `_`-prefix skip above does not cover them. Diffusers itself
        # treats exactly this shape as config — anything that is not a
        # (library, class_name) pair is handed to the pipeline's `__init__`,
        # never resolved as a module.
        #
        # MEASURED ON A RENTED POD, not reasoned about: sdxl 0.4.0 on an
        # RTX 4090 (pod cwnu3dyz72c1wn, se#819) fetched its 6.7 GB checkpoint,
        # then died `SkeletonError: … entry 'force_zeros_for_empty_prompt' is
        # True, which is not [library, class_name]` — first as
        # `boot_warmup_failed`, then as a FATAL request result that the blame
        # ladder burned the worker on. Every diffusers-family endpoint on the
        # streaming loader is exposed to this.
        if not isinstance(spec, (list, tuple)):
            continue
        # pgw#1481 is UNCHANGED for the case it was actually about: an entry
        # that IS list-shaped but that this walker cannot read is REFUSED BY
        # NAME, never skipped. A bare `continue` over those collapsed nine
        # specific failures into one "declares no nn.Module component" that
        # named none of them — and that sentence is false when the index
        # declares nine. A scalar is not that case; it is not a component at
        # all, and refusing it made a correct index unloadable.
        if len(spec) < 2:
            raise SkeletonError(
                f"{index_path} entry {name!r} is {spec!r}, which is not "
                f"[library, class_name]"
            )
        library, class_name = spec[0], spec[1]
        if len(spec) > 2:
            # A MODULAR index entry: [library, class_name, {type_hint,
            # pretrained_model_name_or_path, subfolder, variant, revision}].
            # Its first two elements mean exactly what the classic ones mean,
            # so it is loadable — but only while the metadata does not send us
            # somewhere else. Anything that relocates the component is refused
            # rather than silently read from `checkpoint_dir/<name>`.
            meta = spec[2] if len(spec) == 3 and isinstance(spec[2], dict) else None
            if meta is None:
                raise SkeletonError(
                    f"{index_path} entry {name!r} has {len(spec)} elements and no "
                    f"trailing metadata object; expected [library, class_name] or a "
                    f"modular [library, class_name, {{...}}]"
                )
            subfolder = meta.get("subfolder")
            if subfolder is not None and str(subfolder) != name:
                raise SkeletonError(
                    f"{index_path} entry {name!r} is a modular entry whose "
                    f"subfolder is {subfolder!r}, not {name!r}. The projected tree "
                    f"is addressed by component name, so this component's weights "
                    f"are not where streaming would look for them."
                )
            for field in ("variant", "revision"):
                if meta.get(field) is not None:
                    raise SkeletonError(
                        f"{index_path} entry {name!r} is a modular entry pinning "
                        f"{field}={meta[field]!r}. The projected tree carries one "
                        f"cut of this component and cannot honour that pin."
                    )
        if library is None or class_name is None:
            # A declared-but-absent optional component (safety checker and
            # friends). The pipeline takes None and says so itself.
            components[name] = None
            continue
        directory = Path(checkpoint_dir) / name
        if not directory.is_dir():
            raise SkeletonError(
                f"{MODEL_INDEX} declares component {name!r} but "
                f"{directory} is not in the projected tree"
            )
        cls = _resolve(str(library), str(class_name))
        if _is_module(cls):
            components[name] = _build_on_meta(cls, directory, name)
            modules[name] = components[name]
        else:
            # PASSTHROUGH: a non-`nn.Module` component (tokenizer, scheduler,
            # feature extractor). Its files are small REAL files, so the stock
            # `from_pretrained` is correct — but ONLY while that stays true.
            #
            # pgw#1549: prove it instead of assuming it. A projected tree
            # chunks TENSOR CONTAINERS into the CAS and leaves a ~128 B
            # TFSSTUB1 stub at the path; if one ever lands under a passthrough
            # component, `from_pretrained` reads the stub's first eight bytes
            # as a header length and raises `SafetensorError: header too
            # large` — a LIE ABOUT THE CHECKPOINT that cost two days once
            # already (pgw#1513). One `stub_at_any` call converts that into a
            # named refusal that says which component and what to do.
            from ...models import projection

            if projection.stub_at_any(directory):
                raise SkeletonError(
                    f"component {name!r} is a PASSTHROUGH component "
                    f"({library}.{class_name} is not an nn.Module, so it is "
                    f"built by the stock from_pretrained) but {directory} "
                    f"holds a TFSSTUB1 pointer stub: its bytes are in the CAS "
                    f"and no file reader can reach them. The stock loader "
                    f"would read the stub as a truncated header and blame the "
                    f"checkpoint. This component needs a tensorfs-aware "
                    f"loader (gen_worker.models.tensor_source), or the "
                    f"publisher must not chunk its containers."
                )
            components[name] = cls.from_pretrained(str(directory))  # type: ignore[attr-defined]
            passthrough.append(name)

    if not modules:
        raise SkeletonError(
            f"{index_path} declares no nn.Module component; there would be "
            f"no weights to stream"
        )

    kwargs = dict(components)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    try:
        pipeline = pipeline_cls(**kwargs)
    except TypeError as exc:
        raise SkeletonError(
            f"{pipeline_cls.__name__} refused the components "
            f"{sorted(kwargs)} named by {index_path}: {exc}"
        ) from exc

    # pgw#1410: THE PIPELINE MUST ACTUALLY CARRY WHAT WE ARE ABOUT TO STREAM.
    #
    # `pipeline_cls(**kwargs)` is the classic `DiffusionPipeline.__init__`
    # contract. `ModularPipeline.__init__` routes `**kwargs` to `load_config`
    # and DROPS every component, then `register_components` sets each one to
    # None. Nothing raised: the skeleton returned a pipeline whose components
    # are all None while `modules` held the real objects, `StreamingLoader`
    # streamed the entire checkpoint into those ORPHANS, `meta_survivors`
    # passed (it is per-module, and the modules were fine — it was the PIPELINE
    # that was empty), and the failure surfaced as `None` where a component
    # belongs, on a rented pod, after a full weight load had been paid for.
    #
    # Identity, not truthiness: a pipeline that rebuilt or copied the component
    # would stream into the copy we hold and serve the one it holds.
    orphans = [
        name for name, module in modules.items()
        if name and getattr(pipeline, name, None) is not module
    ]
    if orphans:
        raise SkeletonError(
            f"{pipeline_cls.__name__} did not keep the component(s) "
            f"{sorted(orphans)} it was constructed with — the pipeline carries "
            f"something else (or None) under those names, so streaming would "
            f"fill objects this pipeline never reads and serving would find "
            f"them empty. A `ModularPipeline` does exactly this: its "
            f"`__init__` sends **kwargs to `load_config` and registers every "
            f"component as None. Give the class an `__init__` that calls "
            f"`update_components(...)` after `super().__init__()`, or hand "
            f"`ctx.load` a pipeline class that keeps its constructor arguments."
        )

    logger.info(
        "ctx.load: meta skeleton %s built from configs — %d module component(s), "
        "%d passthrough, 0 tensor bytes read",
        pipeline_cls.__name__,
        len(modules),
        len(passthrough),
    )
    return Skeleton(pipeline=pipeline, modules=modules, passthrough=tuple(passthrough))


def meta_survivors(module: "torch.nn.Module") -> Tuple[str, ...]:
    """Every parameter or buffer still on ``meta``.

    A survivor is never acceptable: it is a name the checkpoint did not carry,
    silently serving garbage on the first request. ``remove_duplicate=False``
    so a tied weight is reported under every name it answers to rather than
    hiding behind whichever alias happened to be assigned.
    """
    left: List[str] = []
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if parameter.device.type == "meta":
            left.append(name)
    for name, buffer in module.named_buffers(remove_duplicate=False):
        if buffer.device.type == "meta":
            left.append(name)
    return tuple(sorted(left))


__all__ = ["MODEL_INDEX", "Skeleton", "SkeletonError", "build", "meta_survivors"]
