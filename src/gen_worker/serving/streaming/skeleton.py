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

pgw#1638: "MINUS THE WEIGHTS" IS A CLAIM ABOUT THE PREPARATION, NOT ONLY
ABOUT BYTES, and it is only true if the preparation runs. ``cls(config)``
runs a constructor; ``from_pretrained`` runs a constructor inside a
preparation, and every step of it this module skips is a STRUCTURAL
difference the engine then blames the checkpoint for. Two members of that
family have been paid for on hardware: ``post_init()``/``tie_weights()``
(pgw#1626, one orphan alias) and the quantizer's module swap (pgw#1638, 357
orphan ``weight_scale_inv`` — ``cls(config)`` leaves plain ``nn.Linear``
where ``from_pretrained`` leaves ``FP8Linear``). A third was found by AUDITING
the family instead of renting a pod for it: ``model.eval()``, which both
``from_pretrained`` implementations end with and this one never did, leaving
every weight-bearing component on the fleet serving with dropout armed. So the
preparation is written out here in the order ``from_pretrained`` runs it, and
its trailing half, which cannot run before bytes land, is
:func:`finish_quantized`.

Symmetry with the publish moment (pgw#1370): the SAME construction on meta,
with nothing streamed into it, is what the derive traces. One surface, two
moments — the ``ctx.compile`` duality, for weights.
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

    ``contract`` is the tensor-layout contract those bytes are, so a refusal
    and the decode-set check name the contract rather than transformers'
    internal method string. ``tensors`` is every parameter/buffer name the
    swap now owns — their dtypes are the CONTRACT's (fp8 weights, an F32
    128x128 scale grid), never anything else's. ``quantizer`` is kept for
    :func:`finish_quantized`.
    """

    contract: str
    quantizer: Any
    tensors: FrozenSet[str]


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
    #: component -> its :class:`Quantization`, for the components whose
    #: config declared one. Absent means "no quantizer ran", never "unknown".
    quantized: Dict[str, Quantization] = field(default_factory=dict)


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


def _declared_quantization(config: Any) -> Any:
    """The ``quantization_config`` a component's config declares, or None."""
    if isinstance(config, Mapping):
        return config.get("quantization_config")
    return getattr(config, "quantization_config", None)


def _quant_contract(declared: Any, component: str, directory: Path) -> str:
    """The tensor-layout contract this ``quantization_config`` names.

    A method with no contract here is REFUSED rather than swapped on
    transformers' word alone: the swap is only as honest as the image's
    declaration behind it, and a contract this image cannot decode must fail
    by name at the skeleton rather than as N orphan scale tensors two steps
    later.
    """
    from ...models import hf_fp8_blockwise

    if hf_fp8_blockwise.declares_contract(declared):
        return hf_fp8_blockwise.CONTRACT_HF_FP8_BLOCKWISE
    method = getattr(declared, "quant_method", None)
    if method is None and isinstance(declared, Mapping):
        method = declared.get("quant_method")
    raise SkeletonError(
        f"component {component!r} ({directory}) declares "
        f"quantization_config quant_method={getattr(method, 'value', method)!r}, "
        f"which names no tensor-layout contract this loader knows. "
        f"ctx.load builds the module set the config asks for and cannot "
        f"guess which linears a method replaces; a checkpoint quantized by a "
        f"method with no contract is refused here rather than loaded as plain "
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

    ``validate_environment`` is deliberately NOT called. Whether this card can
    run the contract's kernels is the lane contract's capability floor and the
    hub's pod pick, which is the one place that fact is allowed to live;
    asking transformers again here would fork it and would make the skeleton
    unbuildable off a GPU.
    """
    declared = _declared_quantization(config)
    if declared is None:
        return None
    contract = _quant_contract(declared, component, directory)

    from ...discovery.decode_set import require_decodable

    require_decodable(contract, where=f"{directory} (component {component!r})")

    if not transformers_class:
        raise SkeletonError(
            f"component {component!r} ({type(built).__name__}) declares "
            f"quantization_config {contract!r} but is built through the "
            f"diffusers from_config path, which runs no quantizer. Streaming "
            f"it would fill plain modules and leave every scale tensor naming "
            f"nothing (pgw#1638's shape). This component needs a "
            f"diffusers-side quantizer preparation before it can be served"
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
            f"component {component!r} declares {contract!r} but "
            f"{type(quantizer).__name__} replaced no module in the meta "
            f"skeleton. Every scale tensor the container carries would name "
            f"nothing; a quantized config that swaps nothing is a mismatch "
            f"between this image's transformers and the tree, not a load"
        )
    logger.info(
        "ctx.load: component %r prepared for %s — %d module(s) swapped, "
        "%d tensor(s) now belong to the contract",
        component, contract, len(swapped), len(owned),
    )
    return Quantization(
        contract=contract, quantizer=quantizer, tensors=frozenset(owned))


def _build_on_meta(
    cls: type, directory: Path, component: str,
) -> Tuple[Any, Optional[Quantization]]:
    """Construct one nn.Module component from its config, on meta."""
    load_config = getattr(cls, "load_config", None)
    from_config = getattr(cls, "from_config", None)
    built: Any = None
    config: Any = None
    transformers_class = False
    if callable(load_config) and callable(from_config):
        # diffusers ConfigMixin: the config is a plain dict on disk.
        config = load_config(str(directory))
        if isinstance(config, tuple):
            config = config[0]
        with init_empty_weights():
            built = from_config(config)
    else:
        config_class = getattr(cls, "config_class", None)
        if config_class is not None and hasattr(config_class, "from_pretrained"):
            # transformers PreTrainedModel: the config is a typed object.
            config = config_class.from_pretrained(str(directory))
            transformers_class = True
            with init_empty_weights():
                built = cls(config)
    if built is not None:
        quantization = _prepare_quantized(
            built, config, component=component, directory=directory,
            transformers_class=transformers_class,
        )
        # THE THIRD MEMBER OF THE FAMILY (pgw#1638's audit). Both
        # `from_pretrained` implementations end with `model.eval()` —
        # transformers says why in its own source: "Set model in evaluation
        # mode to deactivate Dropout modules by default". A config-built
        # module is in TRAIN mode, and nothing on this path ever changed it,
        # so every weight-bearing component on the fleet has been serving with
        # dropout ARMED. Measured on master's conformance corpus: 44 of 44
        # components in train mode, five of them carrying a live
        # `Dropout(p=0.1)` — every T5/UMT5 conditioner on the fleet
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
    quantized: Dict[str, Quantization] = {}
    passthrough: List[str] = []

    for name, spec in sorted(index.items()):
        if name.startswith("_"):
            continue
        # pgw#1481: an entry this walker cannot read is REFUSED BY NAME, never
        # skipped. `continue` here collapsed nine specific failures into one
        # "declares no nn.Module component" that named none of them — and that
        # sentence is false when the index declares nine.
        if not isinstance(spec, (list, tuple)) or len(spec) < 2:
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
            components[name], quantization = _build_on_meta(
                cls, directory, name)
            if quantization is not None:
                quantized[name] = quantization
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

    AFTER the stream, for the same reason :func:`retie` is: it reads the
    tensors that are actually there.
    """
    quantization.quantizer.postprocess_model(module)


def retie(module: "torch.nn.Module") -> bool:
    """Re-establish this module's tied weights. Call AFTER streaming.

    pgw#1626. A meta skeleton is built from a config alone, and construction
    is the ONLY thing that runs: `from_config` / `cls(config)` under
    :func:`init_empty_weights` never reaches `post_init()`, which is what
    normally calls `tie_weights()`. So on the skeleton a tied pair is not one
    tensor under two names — it is TWO INDEPENDENT META TENSORS, and the alias
    has no container to fill it, because a correctly packaged checkpoint stores
    the source alone. A `T5EncoderModel` stores `shared.weight` and not
    `encoder.embed_tokens.weight`; that is what every T5 on the hub looks like.
    Every T5-bearing pipeline on this loader — StableAudio, every FLUX.1
    text_encoder_2 — therefore failed 100% of its invokes at `NameMismatch`,
    with a message that blamed the checkpoint for a defect in the loader.

    AFTER, not before: :func:`gen_worker.serving.streaming.engine._install`
    REBINDS ``_parameters[leaf]`` to a freshly allocated Parameter, so a tie
    established at build time would be broken by the very stream that fills
    it — the alias would keep pointing at the old meta tensor.

    Returns whether a tie ran. Not every component is a
    ``transformers.PreTrainedModel``; a diffusers `ModelMixin` exposes no
    `tie_weights` and needs none.
    """
    tie = getattr(module, "tie_weights", None)
    if not callable(tie):
        return False
    tie()
    return True


def tied_names(module: "torch.nn.Module") -> Tuple[str, ...]:
    """The parameter names this class declares to be ALIASES of another name.

    Advisory, and for the refusal message only — never an exemption. A name
    listed here that is STILL on meta after :func:`retie` means the tensor it
    aliases was not filled either, which is a genuinely absent tensor and must
    still be refused.
    """
    # A dict ({alias: source}, transformers 5) iterates its ALIASES; the older
    # list form is already the alias names. One expression reads both.
    declared = getattr(module, "_tied_weights_keys", None)
    if not declared:
        return ()
    return tuple(sorted(str(name) for name in declared))


def meta_survivors(module: "torch.nn.Module") -> Tuple[str, ...]:
    """Every parameter or buffer still on ``meta``.

    A survivor is never acceptable: it is a name the checkpoint did not carry,
    silently serving garbage on the first request. ``remove_duplicate=False``
    so a tied weight is reported under every name it answers to rather than
    hiding behind whichever alias happened to be assigned — which is sound
    only once :func:`retie` has run and the tie actually EXISTS. Called on an
    untied skeleton, the same flag manufactures the failure it was written to
    detect.
    """
    left: List[str] = []
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if parameter.device.type == "meta":
            left.append(name)
    for name, buffer in module.named_buffers(remove_duplicate=False):
        if buffer.device.type == "meta":
            left.append(name)
    return tuple(sorted(left))


def off_target(module: "torch.nn.Module", target: Any) -> Tuple[Tuple[str, str], ...]:
    """Every parameter or buffer that is neither on ``target`` nor on ``meta``.

    pgw#1644. :func:`meta_survivors` answers "was it filled"; this answers "did
    it LAND", and the two are not the same question. A non-persistent buffer is
    in neither the container nor ``state_dict``, so the stream never names it
    and the survivor check never sees it — it is simply built by ``__init__``
    on the default device and left there. Qwen3-VL's RoPE
    ``inv_freq``/``original_inv_freq`` are exactly that: 146 floats on the CPU
    under a model whose every weight is on CUDA, which surfaces at the first
    forward as ``mat1 is on cpu`` from inside a CUDA ``addmm`` — a whole rental
    away from the load that caused it.

    ``meta`` is excluded deliberately: a tensor still on meta is
    :func:`meta_survivors`' refusal to make, and naming it here too would
    report one defect as two.
    """
    import torch

    want = torch.device(target)

    def lands_on_target(where: "torch.device") -> bool:
        # INDEX-TOLERANT ON PURPOSE. `torch.device("cuda") != torch.device(
        # "cuda", 0)` is True, and the stream lands tensors on `cuda:0` while
        # callers routinely pass a bare "cuda". Comparing with `!=` would make
        # this fence refuse every healthy load on the most common spelling of
        # the device it is checking against — a fence that fires on correct
        # input is worse than the defect it guards.
        if where.type != want.type:
            return False
        if want.index is None or where.index is None:
            return True
        return where.index == want.index

    stray: List[Tuple[str, str]] = []
    for name, tensor in list(module.named_parameters(remove_duplicate=False)) + list(
        module.named_buffers(remove_duplicate=False)
    ):
        if tensor is None or tensor.device.type == "meta":
            continue
        if not lands_on_target(tensor.device):
            stray.append((name, str(tensor.device)))
    return tuple(sorted(stray))


__all__ = [
    "MODEL_INDEX",
    "Quantization",
    "Skeleton",
    "SkeletonError",
    "build",
    "finish_quantized",
    "meta_survivors",
    "off_target",
    "retie",
    "tied_names",
]
