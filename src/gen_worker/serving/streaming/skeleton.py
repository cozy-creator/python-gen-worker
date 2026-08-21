"""Step 1 of ``ctx.load``: the pipeline, built from CONFIGS, holding no bytes.

The skeleton is what ``from_pretrained`` would have handed back MINUS THE
WEIGHTS — and that is a claim about the PREPARATION, not only about bytes.
``cls(config)`` runs a constructor; ``from_pretrained`` runs a constructor
inside a preparation, and every step of it this module skips is a STRUCTURAL
difference the engine then blames the checkpoint for. Two members have been
paid for on hardware: ``post_init()``/``tie_weights()`` (pgw#1626, one orphan
alias) and the quantizer's module swap (pgw#1638, 357 orphan
``weight_scale_inv`` — ``cls(config)`` leaves plain ``nn.Linear`` where
``from_pretrained`` leaves ``FP8Linear``). A third was found by AUDITING the
family instead of renting a pod for it: ``model.eval()``, which both
``from_pretrained`` implementations end with and this one never did, leaving
every weight-bearing component on the fleet serving with dropout armed.

So the preparation is written out here in the order ``from_pretrained`` runs
it — construct, cast to the lane, swap the modules the config's quantizer
declares, put the module in eval mode — and its trailing half, which cannot
run before bytes land, is :func:`finish_quantized`.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple,
)

from ...models.meta_init import init_empty_weights

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

MODEL_INDEX = "model_index.json"


class SkeletonError(RuntimeError):
    """The meta skeleton could not be built from configs alone."""


@dataclass(frozen=True, slots=True)
class Quantization:
    """What one component's DECLARED quantizer did to its meta skeleton.

    ``rule`` is the cozy quant rule those bytes are, so a refusal and the
    decode-set check name the contract rather than transformers' internal
    method string. ``tensors`` is every parameter/buffer name the swap now
    owns: their dtypes are the RULE's (fp8 weights, an F32 128x128 scale
    grid), never the lane's, which is why the engine's lane cast and lane
    assertion both skip them. ``quantizer`` is kept for
    :func:`finish_quantized`.
    """

    rule: str
    quantizer: Any
    tensors: FrozenSet[str]


@dataclass(slots=True)
class Skeleton:
    """A meta-built pipeline and the components weights must reach."""

    pipeline: Any
    modules: Dict[str, Any]
    passthrough: Tuple[str, ...] = ()
    #: component -> its :class:`Quantization`, for the components whose
    #: config declared one. Absent means "no quantizer ran", never "unknown".
    quantized: Dict[str, Quantization] = field(default_factory=dict)


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


def _declared_quantization(config: Any) -> Any:
    """The ``quantization_config`` a component's config declares, or None."""
    if isinstance(config, Mapping):
        return config.get("quantization_config")
    return getattr(config, "quantization_config", None)


def _quant_rule(declared: Any, component: str, directory: Path) -> str:
    """The cozy quant rule this ``quantization_config`` names.

    A method with no rule here is REFUSED rather than swapped on
    transformers' word alone: the swap is only as honest as the image's
    declaration behind it, and a rule this image cannot decode must fail by
    name at the skeleton rather than as N orphan scale tensors two steps
    later.
    """
    from ...models import hf_fp8_blockwise

    if hf_fp8_blockwise.declares_quant_rule(declared):
        return hf_fp8_blockwise.QUANT_RULE
    method = getattr(declared, "quant_method", None)
    if method is None and isinstance(declared, Mapping):
        method = declared.get("quant_method")
    raise SkeletonError(
        f"component {component!r} ({directory}) declares "
        f"quantization_config quant_method={getattr(method, 'value', method)!r}, "
        f"which names no tensor-layout quant rule this loader knows. "
        f"ctx.load builds the module set the config asks for and cannot "
        f"guess which linears a method replaces; a checkpoint quantized by a "
        f"method with no rule is refused here rather than loaded as plain "
        f"linears whose scale tensors then name nothing (pgw#1638)"
    )


def _prepare_quantized(
    built: Any, config: Any, *, component: str, directory: Path,
    transformers_class: bool,
) -> Optional[Quantization]:
    """Run the config's own quantizer over the meta skeleton (pgw#1638).

    ``HfQuantizer.preprocess_model`` is the step ``from_pretrained`` runs
    between construction and weight loading, and its own docstring says the
    model "should be initialized on the meta device" at that point — so this
    is the API used as designed, not a second implementation of the swap.
    ``pre_quantized=True``: this loader reads pre-quantized artifacts and
    never quantizes.

    ``validate_environment`` is deliberately NOT called. Whether this card
    can run the rule's kernels is the lane contract's ``capability_floor_sm``
    and the hub's pod pick, which is the one place that fact is allowed to
    live; asking transformers again here would fork it and would make the
    skeleton unbuildable off a GPU, which is where the conformance suite runs.
    """
    declared = _declared_quantization(config)
    if declared is None:
        return None
    rule = _quant_rule(declared, component, directory)

    from ...discovery.decode_set import require_decodable

    require_decodable(rule, where=f"{directory} (component {component!r})")

    if not transformers_class:
        raise SkeletonError(
            f"component {component!r} ({type(built).__name__}) declares "
            f"quantization_config {rule!r} but is built through the diffusers "
            f"from_config path, which runs no quantizer. Streaming it would "
            f"fill plain modules and leave every scale tensor naming nothing "
            f"(pgw#1638's shape). This component needs a diffusers-side "
            f"quantizer preparation before it can be served on this loader"
        )

    from transformers.quantizers.auto import AutoHfQuantizer

    quantizer = AutoHfQuantizer.from_config(declared, pre_quantized=True)
    before = {name: type(sub) for name, sub in built.named_modules()}
    with init_empty_weights():
        quantizer.preprocess_model(model=built, config=config)
    swapped = [
        name for name, sub in built.named_modules()
        if before.get(name) is not type(sub)
    ]
    owned: Set[str] = set()
    for prefix in swapped:
        sub = built.get_submodule(prefix) if prefix else built
        head = f"{prefix}." if prefix else ""
        for leaf, _ in sub.named_parameters(remove_duplicate=False):
            owned.add(head + leaf)
        for leaf, _ in sub.named_buffers(remove_duplicate=False):
            owned.add(head + leaf)
    if not swapped:
        raise SkeletonError(
            f"component {component!r} declares {rule!r} but "
            f"{type(quantizer).__name__} replaced no module in the meta "
            f"skeleton. Every scale tensor the container carries would name "
            f"nothing; a quantized config that swaps nothing is a mismatch "
            f"between this image's transformers and the tree, not a load"
        )
    logger.info(
        "ctx.load: component %r prepared for %s — %d module(s) swapped, "
        "%d tensor(s) now belong to the rule",
        component, rule, len(swapped), len(owned),
    )
    return Quantization(rule=rule, quantizer=quantizer, tensors=frozenset(owned))


def _build_on_meta(
    cls: type, directory: Path, component: str, compute_dtype: Any = None,
) -> Tuple[Any, Optional[Quantization]]:
    load_config = getattr(cls, "load_config", None)
    from_config = getattr(cls, "from_config", None)
    built: Any = None
    config: Any = None
    transformers_class = False
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
            transformers_class = True
            with init_empty_weights():
                built = cls(config)
    if built is not None:
        # The lane cast runs BEFORE the swap: it is for the residue the
        # container does not name, and every tensor a swapped module holds
        # comes out of the container at the RULE's dtype. Casting after would
        # round a 128x128 F32 scale grid to bf16 and call it a repair.
        if compute_dtype is not None:
            built = built.to(compute_dtype)
        quantization = _prepare_quantized(
            built, config, component=component, directory=directory,
            transformers_class=transformers_class,
        )
        # THE THIRD MEMBER OF THE FAMILY (pgw#1638's audit). Both
        # `from_pretrained` implementations end with `model.eval()` —
        # transformers says why in its own source: "Set model in evaluation
        # mode to deactivate Dropout modules by default". A config-built
        # module is in TRAIN mode, and nothing on this path ever changed it,
        # so every one of the 44 weight-bearing components on the fleet has
        # been serving with dropout ARMED. Five of them carry a live
        # `Dropout(p=0.1)`: every T5/UMT5 conditioner on the fleet
        # (flux.1-dev, flux.1-schnell, foundation-1, stable-audio-open,
        # wan-2.2), i.e. randomized conditioning, nondeterministic output, no
        # error anywhere. AFTER the swap, because a quantizer's replacement
        # modules are constructed in train mode like any other.
        built.eval()
        return built, quantization

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
        modules[spec.name], _ = _build_on_meta(
            cls, directory, spec.name, compute_dtype)
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
    quantized: Dict[str, Quantization] = {}
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
            components[name], quantization = _build_on_meta(
                cls, directory, name, compute_dtype)
            modules[name] = components[name]
            if quantization is not None:
                quantized[name] = quantization
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
        "%d passthrough, %d quantized, 0 tensor bytes read",
        pipeline_cls.__name__,
        len(modules),
        len(passthrough),
        len(quantized),
    )
    return Skeleton(
        pipeline=pipeline, modules=modules, passthrough=tuple(passthrough),
        quantized=quantized,
    )


def finish_quantized(module: "torch.nn.Module", quantization: Quantization) -> None:
    """The trailing half of the quantizer mirror. Call AFTER streaming.

    ``postprocess_model`` is what ``from_pretrained`` runs once bytes have
    landed, and it is not cosmetic: for a ``scale_fmt="ue8m0"`` tree it
    rewrites every F32 scale container into the exponent dtype the kernels
    read. Running :func:`_prepare_quantized` without it would be half a
    mirror — the same shape as the defect this whole seam exists to end.

    AFTER the stream and after the lane cast, for the same reason
    :func:`retie` is: it reads the tensors that are actually there.
    """
    quantization.quantizer.postprocess_model(module)


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
    "Quantization",
    "Skeleton",
    "SkeletonError",
    "build",
    "finish_quantized",
    "meta_survivors",
    "retie",
    "tied_names",
]
