"""Step 1 of ``ctx.load``: the pipeline, built from CONFIGS, holding no bytes."""

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
    modules: Dict[str, Any]
    passthrough: Tuple[str, ...] = ()


def _resolve(library: str, class_name: str) -> type:
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
    try:
        import diffusers.pipelines as pipelines
    except ImportError:
        return None
    return pipelines


def _is_module(cls: type) -> bool:
    import torch

    return issubclass(cls, torch.nn.Module)


def _build_on_meta(
    cls: type, directory: Path, component: str, compute_dtype: Any = None,
) -> Any:
    load_config = getattr(cls, "load_config", None)
    from_config = getattr(cls, "from_config", None)
    built: Any = None
    if callable(load_config) and callable(from_config):
        config = load_config(str(directory))
        if isinstance(config, tuple):
            config = config[0]
        with init_empty_weights():
            built = from_config(config)
    else:
        config_class = getattr(cls, "config_class", None)
        if config_class is not None and hasattr(config_class, "from_pretrained"):
            config = config_class.from_pretrained(str(directory))
            with init_empty_weights():
                built = cls(config)
    if built is not None:
        return built if compute_dtype is None else built.to(compute_dtype)

    raise SkeletonError(
        f"component {component!r} ({cls.__module__}.{cls.__name__}) exposes "
        f"neither a diffusers from_config nor a transformers config_class, so "
        f"it cannot be built weight-free; ctx.load never reads a checkpoint "
        f"file to find out what a class wants"
    )


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One ``model_index.json`` component declaration, already validated.

    ``library`` and ``class_name`` are ``None`` for a declared-but-absent
    optional component — the ``[null, null]`` spelling, which is the ONE way
    an index says "this pipeline accepts None here".
    """

    name: str
    library: Optional[str]
    class_name: Optional[str]

    @property
    def absent(self) -> bool:
        return self.library is None or self.class_name is None


def read_index(checkpoint_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """The tree's ``model_index.json``, parsed. Reads no tensor bytes."""
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
    return index_path, index


def component_specs(index_path: Path, index: Mapping[str, Any]) -> List[ComponentSpec]:
    """Every component a parsed index declares, in name order.

    This walk — and every refusal in it — is shared by :func:`build` and
    :func:`build_modules`, so the two cannot disagree about what an index
    says. That is the whole reason it is a function: the conformance suite
    and the release-build fence read the index through the SAME reader the
    production loader does, or they are proving something about a second
    parser nobody serves with.
    """
    specs: List[ComponentSpec] = []
    for name, spec in sorted(index.items()):
        if name.startswith("_"):
            continue
        if not isinstance(spec, (list, tuple)):
            continue
        if len(spec) < 2:
            raise SkeletonError(
                f"{index_path} entry {name!r} is {spec!r}, which is not "
                f"[library, class_name]"
            )
        library, class_name = spec[0], spec[1]
        if len(spec) > 2:
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
        specs.append(
            ComponentSpec(
                name=name,
                library=None if library is None else str(library),
                class_name=None if class_name is None else str(class_name),
            )
        )
    return specs


def build_modules(
    checkpoint_dir: Path, *, compute_dtype: Any = None,
) -> Dict[str, Any]:
    """Every WEIGHT-BEARING component of a tree, built on meta. No pipeline.

    :func:`build` is the production path and stays the production path: it
    also constructs the passthrough components and the pipeline object, which
    is what serving needs. This is the half that answers "do the modules this
    tree declares come up on meta, and does anything stay there" — and it is
    separate because the passthrough half needs files a CONFIG-ONLY tree does
    not carry (a tokenizer's `vocab.json` is megabytes of real bytes that
    every checkpoint-config fixture on the fleet deliberately omits) and
    optional third-party backends a pipeline's scheduler may import.

    Neither of those is the question. A tokenizer has no parameters, so it
    cannot leave one on meta; making the meta/tie check depend on a
    sentencepiece model would mean the check simply does not run for half the
    fleet, which is worse than any answer it could give.
    """
    index_path, index = read_index(checkpoint_dir)
    modules: Dict[str, Any] = {}
    for spec in component_specs(index_path, index):
        if spec.absent:
            continue
        # The class is resolved BEFORE the directory is required — the reverse
        # of `build`'s order, and deliberately. A component this function will
        # not construct needs no files, so a tree missing a tokenizer's bytes
        # must not stop the modules beside it from being answered for.
        cls = _resolve(str(spec.library), str(spec.class_name))
        if not _is_module(cls):
            continue
        directory = Path(checkpoint_dir) / spec.name
        if not directory.is_dir():
            raise SkeletonError(
                f"{MODEL_INDEX} declares component {spec.name!r} but "
                f"{directory} is not in the projected tree"
            )
        modules[spec.name] = _build_on_meta(cls, directory, spec.name, compute_dtype)
    if not modules:
        raise SkeletonError(
            f"{index_path} declares no nn.Module component; there would be "
            f"no weights to stream"
        )
    return modules


def build(
    pipeline_cls: type,
    checkpoint_dir: Path,
    *,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    compute_dtype: Any = None,
) -> Skeleton:
    """Build ``pipeline_cls`` from configs only."""
    index_path, index = read_index(checkpoint_dir)

    components: Dict[str, Any] = {}
    modules: Dict[str, Any] = {}
    passthrough: List[str] = []

    for spec in component_specs(index_path, index):
        name, library, class_name = spec.name, spec.library, spec.class_name
        if spec.absent:
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
            components[name] = _build_on_meta(cls, directory, name, compute_dtype)
            modules[name] = components[name]
        else:
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


def retie(module: "torch.nn.Module") -> bool:
    """Re-establish this module's tied weights."""
    tie = getattr(module, "tie_weights", None)
    if not callable(tie):
        return False
    tie()
    return True


def tied_names(module: "torch.nn.Module") -> Tuple[str, ...]:
    """The parameter names this class declares to be ALIASES of another name."""
    declared = getattr(module, "_tied_weights_keys", None)
    if not declared:
        return ()
    return tuple(sorted(str(name) for name in declared))


def meta_survivors(module: "torch.nn.Module") -> Tuple[str, ...]:
    """Every parameter or buffer still on ``meta``."""
    left: List[str] = []
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if parameter.device.type == "meta":
            left.append(name)
    for name, buffer in module.named_buffers(remove_duplicate=False):
        if buffer.device.type == "meta":
            left.append(name)
    return tuple(sorted(left))


__all__ = [
    "MODEL_INDEX",
    "Skeleton",
    "SkeletonError",
    "build",
    "meta_survivors",
    "retie",
    "tied_names",
]
