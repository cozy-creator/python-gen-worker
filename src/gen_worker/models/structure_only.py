"""pgw#1080 increment 2: the COMPONENT-LEVEL structure-only builder.

The mint child exports and compiles a family's declared compile targets. Until
now it reached them by running the endpoint's own ``setup()``, which loads the
checkpoint — so the process that traces held every byte the process that serves
holds, and on a two-pipeline family (z-image, 42.53 GiB) the two copies do not
fit on one card at all (ie#638). Paul's ruling: export and compile must not
require real weights resident AT ALL.

WHY THIS IS PER-COMPONENT AND NOT A ``structure_only=True`` LOADER FLAG
-----------------------------------------------------------------------
A device context cannot make ``from_pretrained`` structure-only: ``with
torch.device("meta")`` is a CONSTRUCTION DEFAULT and the weight READ is
unaffected (safetensors mmaps to CPU; the subsequent ``load_state_dict`` either
refuses or copies real data in). The pattern that actually skips the state dict
is ``init_empty_weights()`` + ``from_config`` — which is a per-CLASS capability
(diffusers' ``ConfigMixin``), and which every quantized loader in this tree
already uses to build its denoiser (:mod:`.w8a8` , :mod:`.w4a4`,
:mod:`.svdq_native`). The context manager itself is :mod:`.meta_init`, owned
here rather than imported from ``accelerate``: see that module for the pods
that measured what an undeclared import on this path costs (pgw#1123). A
PIPELINE's config is its component CLASS MAP
(``model_index.json``), not a weights layout, so ``from_config`` on a pipeline
builds nothing the export traces. Hence: build the COMPONENT, inject it through
the ``components=`` seam the loader already has, and let the pipeline class
compose the rest exactly as it composes a preloaded shared component.

WHY THE PARAMETERS END UP **FAKE** AND THE BUFFERS STAY **REAL**
---------------------------------------------------------------
Measured on torch 2.13.0, not assumed (four variants, cardless rig):

* meta parameters + real inputs → ``torch.export`` refuses: *Tensor device
  mismatch … cpu and meta*. Meta is not a device the compile can target either
  — AOTInductor would codegen for ``meta``, not for the card.
* parameters as FAKE tensors on the TARGET device → export succeeds, and
  ``aot_compile`` succeeds when it runs inside
  ``torch._guards.tracing(TracingContext(<that fake mode>))``. Fake tensors
  carry a faithful device, dtype and stride and allocate NOTHING, which is
  exactly the property this slice exists for.
* buffers stay REAL, on the device. They are pure functions of config (rope
  tables, sinusoidal features) and they are KB-to-MB scale, and they are what
  a literal-bearing family ships INSIDE the compiled graph: a fake buffer would make
  the compiled graph's literal constants unpackable. Keeping them real also arms the
  folding fence — a literal derived from a PARAMETER is fake and fails loudly,
  and a value-dependent fold over a rebindable weight is precisely what
  pgw#1056 forbids.

WHAT IS REFUSED RATHER THAN GUESSED
-----------------------------------
A compile-target class with no ``load_config``/``from_config`` surface, a tree
with no ``model_index.json`` entry naming the component's class, and the
quantized artifact lanes whose denoiser is built from an artifact's own weight
table. Each refuses typed, naming the class and what it lacks — honest partial
coverage beats total coverage that lies.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..api.errors import WorkerError
from .. import meta_instantiation as mi
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

#: Stamped on a structure-only module and on the pipeline composed around it.
STAMP = "_cozy_structure_only"
#: The fake mode every virtual parameter of a structure-only tree belongs to.
MODE_STAMP = "_cozy_structure_fake_mode"
FACTS_STAMP = "_cozy_structure_facts"

class StructureOnlyUnsupported(WorkerError):
    """This component cannot be built from code + config alone.

    Named-axis refusal: WHICH component, WHICH class, and WHAT it lacks — the
    fix is always an authoring or packaging change (expose the ConfigMixin
    surface; declare the component in ``model_index.json``), never a knob.
    """

    def __init__(self, *, component: str, cls_name: str, lacks: str,
                 tree: str = "") -> None:
        self.component = str(component or "")
        self.cls_name = str(cls_name or "")
        self.lacks = str(lacks or "")
        self.tree = str(tree or "")
        super().__init__(self._sentence())

    def _sentence(self) -> str:
        return (
            f"structure-only build of component {self.component!r} "
            f"({self.cls_name or 'unknown class'}) is not possible: "
            f"{self.lacks}. The zero-download forge instantiates a compile "
            f"target from CODE + CONFIG (`load_config` + `from_config` under "
            f"`meta_init.init_empty_weights()` — the shape "
            f"`models/w8a8.py` already uses), so a class without that surface "
            f"has no structure-only path and this family is stranded on the "
            f"real-weight mint until it grows one"
            + (f" (tree: {self.tree})" if self.tree else "")
        )


class StructureCapabilityMissing(StructureOnlyUnsupported):
    """The PROCESS cannot meta-instantiate — nothing about this family is wrong.

    A subclass so every existing ``except StructureOnlyUnsupported`` keeps its
    never-fatal degradation, and a distinct type because the two mean opposite
    things to whoever reads the boot-adopt event. Its parent is a permanent,
    correct property of some trees (a quantized artifact lane has no
    config-only structure and never will); this one is a broken install, is
    never normal, and strands EVERY family in the image rather than one — which
    is exactly how it went unnoticed on two paid pods (pgw#1123).
    """

    def __init__(self, *, component: str, cls_name: str, lacks: str,
                 capability: str, tree: str = "") -> None:
        self.capability = str(capability or "")
        super().__init__(component=component, cls_name=cls_name, lacks=lacks,
                         tree=tree)

    def _sentence(self) -> str:
        return (
            f"structure-only build of component {self.component!r} is "
            f"impossible in THIS PROCESS, for every family it serves: "
            f"{self.lacks}. Missing capability: {self.capability}. This is an "
            f"IMAGE defect, not an authoring one — no boot key can be derived "
            f"here, so this pod can never ask the hub for a compiled graph and will "
            f"self-mint on every boot"
            + (f" (tree: {self.tree})" if self.tree else "")
        )


#: The ``boot_adopt`` phase token for each kind of structure-only refusal.
#: Read out of this source by
#: ``tests/test_boot_adopt_observability_pgw1116.py``'s vocabulary fence.
TOKEN_UNSUPPORTED = "structure_unsupported"
TOKEN_CAPABILITY_MISSING = "structure_capability_missing"


def refusal_token(exc: StructureOnlyUnsupported) -> str:
    """Which boot-adopt token a structure-only refusal reports under.

    The distinction is the whole of pgw#1123: ``structure_unsupported`` is a
    family that is stranded (correct, expected forever on the quantized
    artifact lanes, and a reason to look at the FAMILY);
    ``structure_capability_missing`` is an image that cannot do the thing at
    all (a reason to look at the IMAGE). Reported under one token they are
    indistinguishable, and the second one looks exactly like a pod that chose
    to self-mint.
    """
    return (TOKEN_CAPABILITY_MISSING
            if isinstance(exc, StructureCapabilityMissing)
            else TOKEN_UNSUPPORTED)


class StructureNotHonored(StructureOnlyUnsupported):
    """A component that WAS built weight-free was not carried by the pipeline.

    Distinct from its parent on purpose. ``StructureOnlyUnsupported`` means the
    component could not be built from code+config at all (a genuinely stranded
    family: no config surface, a quantized artifact lane, an unnamed class) —
    for which loading real weights is the CORRECT outcome and the mint child
    legitimately falls back. This subclass means the opposite: ``build_component``
    SUCCEEDED (the module's parameters are fake, zero storage) and the composed
    pipeline then IGNORED the injected module and rebuilt the target from the
    checkpoint. Falling back there would export ~weight-scale REAL tensors while
    still reporting a weightless child — the exact silent failure this slice
    exists to prevent (measured on z-image: a ~40 GiB `torch.export` OOM
    misclassified `retryable`, ie#638). So it must FAIL CLOSED, not fall back.
    """


@dataclass(frozen=True)
class StructureFacts:
    """What one structure-only component costs, and what it did NOT cost."""

    component: str
    cls_name: str
    parameters: int = 0
    #: Checkpoint bytes this process would have held and does not.
    virtual_param_bytes: int = 0
    #: Config-derived tables that legitimately stay real (literals live here).
    real_buffer_bytes: int = 0
    #: Real bytes allocated during ``__init__`` and NOT retained as a buffer —
    #: transient by construction, reported because an unexplained one is the
    #: first sign a class is doing weight work at construction.
    transient_real_bytes: int = 0
    #: Endpoint sites that allocated real tensors while building the structure.
    sites: Tuple[str, ...] = ()

    def as_event(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "cls": self.cls_name,
            "parameters": self.parameters,
            "virtual_param_bytes": self.virtual_param_bytes,
            "real_buffer_bytes": self.real_buffer_bytes,
            "transient_real_bytes": self.transient_real_bytes,
        }


@dataclass
class _Collected:
    facts: List[StructureFacts] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Class resolution — the tree's own component map, never a guess
# ---------------------------------------------------------------------------


def _component_class(tree: Path, component: str) -> Any:
    from .loading import model_index_components, model_index_entry

    entry = model_index_entry(tree, component)
    if entry is None:
        known = sorted(model_index_components(tree))
        raise StructureOnlyUnsupported(
            component=component, cls_name="", tree=str(tree),
            lacks=(
                f"the tree names no class for it — model_index.json declares "
                f"{known or 'nothing (no readable model_index.json)'}"),
        )
    library, class_name = entry
    try:
        module = importlib.import_module(library)
    except Exception as exc:  # noqa: BLE001 — a missing library is a refusal
        raise StructureOnlyUnsupported(
            component=component, cls_name=f"{library}.{class_name}",
            tree=str(tree),
            lacks=f"its library {library!r} does not import ({exc})") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise StructureOnlyUnsupported(
            component=component, cls_name=f"{library}.{class_name}",
            tree=str(tree), lacks=f"{library} has no class {class_name!r}")
    return cls


def _require_config_surface(cls: Any, component: str, tree: Path) -> None:
    missing = [name for name in ("load_config", "from_config")
               if not callable(getattr(cls, name, None))]
    if missing:
        raise StructureOnlyUnsupported(
            component=component, cls_name=getattr(cls, "__name__", str(cls)),
            tree=str(tree),
            lacks=f"it exposes no {' and no '.join(missing)} classmethod")


def _refuse_artifact_lanes(root: Path, component: str, cls: Any) -> None:
    """The quantized lanes build their denoiser from an ARTIFACT's own weight
    table (which linears are quantized, at what scales). That table is a
    property of the bytes, not of the config, so a structure-only build would
    trace ``nn.Linear`` where serving runs ``Fp8ScaledLinear`` — a compiled graph for a
    graph the pod never executes. Refuse by name instead."""
    from .svdq import detect_svdq_artifact
    from .w4a4 import detect_w4a4_artifact
    from .w8a8 import detect_w8a8_artifact
    from .loading import detect_gguf_snapshot

    for label, art in (
        ("w8a8", detect_w8a8_artifact(root)),
        ("w4a4", detect_w4a4_artifact(root)),
        ("svdq", detect_svdq_artifact(root)),
    ):
        if art is not None and getattr(art, "component", component) in (
                component, ""):
            raise StructureOnlyUnsupported(
                component=component,
                cls_name=getattr(cls, "__name__", str(cls)), tree=str(root),
                lacks=(
                    f"this tree is a {label} artifact, whose module graph is "
                    f"decided by the artifact's WEIGHT TABLE (which linears "
                    f"are quantized, at which scales) and not by its config — "
                    f"a structure-only build would trace a graph the pod does "
                    f"not serve"))
    if detect_gguf_snapshot(root) is not None:
        raise StructureOnlyUnsupported(
            component=component, cls_name=getattr(cls, "__name__", str(cls)),
            tree=str(root),
            lacks=("this tree is a GGUF snapshot: its denoiser is dequantized "
                   "by the pipeline's own gguf loader, so there is no "
                   "config-only structure for it"))


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def _init_empty_weights(component: str = "") -> Any:
    """The meta-init seam, PROVEN on this process before it is used.

    pgw#1123: the mechanism is OWNED (:mod:`.meta_init`) rather than an
    undeclared ``accelerate`` import absent from the fleet's own probe image, and
    if it is ever unavailable the refusal names the missing CAPABILITY under its
    own token rather than the one a stranded family refuses under.
    """
    from . import meta_init

    try:
        meta_init.require_meta_init()
    except meta_init.MetaInitUnavailable as exc:
        logger.error(
            "structure-only build is impossible in this process: %s", exc)
        raise StructureCapabilityMissing(
            component=component, cls_name="", capability=exc.capability,
            lacks=exc.lacks) from exc
    return meta_init.init_empty_weights


def build_component(
    tree: str | Path, component: str, *, device: str = "", dtype: str = "",
) -> Tuple[Any, StructureFacts]:
    """Build ONE pipeline component from code + config, holding no weights.

    Returns the module and its measured facts. The module's PARAMETERS are
    fake tensors on ``device`` (faithful shape/dtype/device, zero storage); its
    BUFFERS are real, because they are config-derived and a literal-bearing
    family ships them.
    """
    # FIRST, before anything about this family is inspected: a process that
    # cannot meta-instantiate refuses about ITSELF, not about the tree it was
    # handed (pgw#1123 — the pod message named component '' and 'unknown
    # class', which read as a family problem and was an image problem).
    init_empty_weights = _init_empty_weights(component)

    root = Path(tree)
    cls = _component_class(root, component)
    _refuse_artifact_lanes(root, component, cls)
    _require_config_surface(cls, component, root)
    src = root / component if (root / component).is_dir() else root

    config = dict(cls.load_config(str(src)))
    # A quantization block describes bytes this build will never read, and
    # diffusers reconstructs some of them into configs whose constructors
    # refuse (pgw#689 defect 1).
    config.pop("quantization_config", None)

    with mi.observe(f"structure:{component}") as census:
        with init_empty_weights():
            module = cls.from_config(config)
    if hasattr(module, "eval"):
        module.eval()

    torch_dtype = _torch_dtype(root, component, dtype)
    virtualize(module, device=device, dtype=torch_dtype)

    facts = _facts(module, component=component,
                   cls_name=getattr(cls, "__name__", str(cls)), census=census)
    setattr(module, FACTS_STAMP, facts)
    if not cuda_ready() and device.startswith("cuda"):
        raise StructureOnlyUnsupported(
            component=component, cls_name=facts.cls_name, tree=str(root),
            lacks=f"device {device!r} is not available in this process")
    return module, facts


def _torch_dtype(root: Path, component: str, dtype: str) -> Any:
    """The compute dtype this component would have loaded at.

    Same resolution ``loading.load_component`` uses, because a structure whose
    precision differs from serving's is a different graph and a different compiled graph.
    """
    from ..families.facts import component_dtype_for_class
    from .loading import (
        composition_compute_dtype, detect_on_disk_dtype,
        get_torch_dtype, model_index_component_classes,
    )

    class_name = model_index_component_classes(root).get(component, "")
    fact = component_dtype_for_class(class_name) if class_name else None
    src = root / component if (root / component).is_dir() else root
    wanted = (
        (fact.dtype if fact is not None else "")
        or composition_compute_dtype(root, dtype)
        or detect_on_disk_dtype(src)
    )
    if wanted in ("bf16", "fp16", "bfloat16", "float16", "fp32", "float32"):
        try:
            return get_torch_dtype(wanted)
        except ImportError:
            return None
    return None


def _wrapper_parts(tensor: Any) -> Optional[Tuple[List[str], Any]]:
    """``(inner attribute names, context)`` when ``tensor`` is a traceable
    wrapper subclass — torch's own contract for seeing inside one.

    A ``setup()``-time quantizer leaves these behind on a virtual structure
    (torchao's ``Float8Tensor``: fake ``qdata`` + fake ``scale``, outer dtype
    bf16), and both directions of the mint's fake↔real swap have to rebuild
    the SUBCLASS rather than a plain tensor of the outer dtype. Flattening one
    to bf16 traces bf16 Linears for a pod that serves fp8 — a compiled graph for a graph
    the pod never executes, which is the defect ``_refuse_artifact_lanes``
    exists to prevent, arriving by the other door (pgw#1198).
    """
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if not is_traceable_wrapper_subclass(tensor):
            return None
        names, ctx = tensor.__tensor_flatten__()
    except Exception:  # noqa: BLE001 — an unwalkable object is not a subclass
        return None
    return ([str(n) for n in names], ctx) if names else None


def _rebuild_wrapper(tensor: Any, inner: Dict[str, Any]) -> Any:
    parts = _wrapper_parts(tensor)
    ctx = parts[1] if parts is not None else None
    return type(tensor).__tensor_unflatten__(
        inner, ctx, tuple(tensor.shape), tuple(tensor.stride()))


def _fake_like(tensor: Any, *, dtype: Any = None) -> Any:
    """A zero-storage twin of ``tensor``. Call inside the fake mode + device.

    ``dtype`` casts a plain floating-point tensor to the composition's compute
    precision; a wrapper subclass is rebuilt from fake inner tensors and is
    NEVER cast — its outer dtype already IS the compute precision and its
    payload's dtype is the quantization.
    """
    import torch

    parts = _wrapper_parts(tensor)
    if parts is not None:
        names, _ctx = parts
        return _rebuild_wrapper(tensor, {
            name: _fake_like(getattr(tensor, name)) for name in names})
    want = dtype if (dtype is not None
                     and tensor.is_floating_point()) else tensor.dtype
    return torch.empty(tuple(tensor.shape), dtype=want)


def virtualize(module: Any, *, device: str = "", dtype: Any = None) -> Any:
    """Parameters → fake tensors on ``device``; buffers → real, on ``device``.

    Returns the fake mode, which the export and the compile must both run
    inside: ``aot_compile`` asserts that every input belongs to ONE mode.
    """
    import torch
    import torch.nn as nn
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    dev = device or "cpu"
    mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    with mode, torch.device(dev):
        for sub in module.modules():
            for name, param in list(sub._parameters.items()):
                if param is None:
                    continue
                sub._parameters[name] = nn.Parameter(
                    _fake_like(param, dtype=dtype), requires_grad=False)
    for sub in module.modules():
        for name, buf in list(sub._buffers.items()):
            if buf is None:
                continue
            want = dtype if (dtype is not None
                             and buf.is_floating_point()) else buf.dtype
            sub._buffers[name] = buf.to(device=dev, dtype=want)
    _freeze_placement(module)
    setattr(module, STAMP, True)
    setattr(module, MODE_STAMP, mode)
    return mode


def _freeze_placement(module: Any) -> None:
    """Make ``.to()`` / ``.cuda()`` / ``.float()`` a no-op on this structure.

    A virtual structure is ALREADY on the device and dtype it claims — that
    is the whole point of building it there — and torch cannot move a fake
    parameter anyway: the worker's placement pass dies with
    ``RuntimeError: _apply(): Couldn't swap Linear.weight`` (measured on the
    RTX 4070 rig, first GPU cycle). Silently declining the move is the honest
    behaviour here, because a move that DID happen would re-key the graph the
    compiled graph is being minted for.
    """
    import types

    def _apply(self: Any, *_args: Any, **_kwargs: Any) -> Any:
        return self

    module._apply = types.MethodType(_apply, module)


def _facts(module: Any, *, component: str, cls_name: str,
           census: mi.Census) -> StructureFacts:
    virtual_bytes = 0
    params = 0
    for param in module.parameters():
        params += 1
        virtual_bytes += int(param.numel()) * int(param.element_size())
    real_buffers = 0
    for buf in module.buffers():
        if not mi.is_virtual(buf):
            real_buffers += int(buf.numel()) * int(buf.element_size())
    observed = sum(_event_bytes(e) for e in census.events)
    return StructureFacts(
        component=component, cls_name=cls_name, parameters=params,
        virtual_param_bytes=virtual_bytes, real_buffer_bytes=real_buffers,
        transient_real_bytes=max(0, observed - real_buffers),
        sites=tuple(sorted({e.site for e in census.events if e.site})[:8]),
    )


def _event_bytes(event: mi.Materialization) -> int:
    """Bytes one observed real allocation holds, from its recorded shape."""
    import torch

    try:
        shape = tuple(int(x) for x in
                      event.shape.strip("()").replace(" ", "").split(",") if x)
    except ValueError:
        return 0
    numel = 1
    for dim in shape:
        numel *= dim
    dtype = getattr(torch, event.dtype.replace("torch.", ""), None)
    itemsize = getattr(dtype, "itemsize", 0) if dtype is not None else 0
    return int(numel) * int(itemsize or 0)


# ---------------------------------------------------------------------------
# Reading a composed pipeline back
# ---------------------------------------------------------------------------


def structure_only_components(pipe: Any) -> Tuple[str, ...]:
    """Names of the pipeline attributes that are structure-only modules."""
    out: List[str] = []
    for name, value in list(vars(pipe).items() if hasattr(pipe, "__dict__")
                            else []):
        if getattr(value, STAMP, False):
            out.append(str(name).lstrip("_"))
    return tuple(sorted(set(out)))


def target_module(pipe: Any, target: str) -> Any:
    """The module holding the WEIGHTS of declared compile target ``target``.

    ``None`` when the pipeline does not carry it, which is the legitimate case
    a multi-slot family produces (a refiner slot has no denoiser of the
    primary's name).

    Resolution is delegated to ``compile_cache.resolve_targets`` — the ONE
    target authority (§1.29) — rather than re-walked here, because a target is
    an ATTRIBUTE PATH and not a component name: ``transformer.denoise`` and
    ``vae.decode`` are both declared on the fleet, and a second walk that only
    understood ``getattr(pipe, name)`` would read every dotted target as "not
    carried" and skip the fence for exactly the families whose targets are
    nested. A guard with its own weaker resolver is a guard that cannot fire.
    """
    import types

    from .. import compile_cache as cc

    name = str(target or "").strip()
    if not name:
        return None
    for _declared, owner, _attr, _fn in cc.resolve_targets(
            pipe, types.SimpleNamespace(targets=(name,))):
        if owner is not None and hasattr(owner, "named_parameters"):
            return owner
    return None


@dataclass(frozen=True)
class WeightFreeBreach:
    """One declared compile target that is NOT weight-free, and why."""

    component: str
    cls_name: str
    #: ``not_structure_only`` — carried, but never built from code+config;
    #: ``real_parameters`` — stamped structure-only and holding real storage.
    reason: str
    real_param_bytes: int = 0
    devices: Tuple[str, ...] = ()

    def sentence(self) -> str:
        where = f" on {', '.join(self.devices)}" if self.devices else ""
        if self.reason == "not_structure_only":
            return (
                f"{self.component} ({self.cls_name}) carries "
                f"{self.real_param_bytes} byte(s) of REAL parameters{where} "
                f"— it was never built from code + config")
        return (
            f"{self.component} ({self.cls_name}) is stamped structure-only "
            f"and still holds {self.real_param_bytes} byte(s) of REAL "
            f"parameters{where}")


def weight_free_breaches(
    pipe: Any, targets: Any,
) -> Tuple[WeightFreeBreach, ...]:
    """Every declared compile target ``pipe`` carries that holds real weights.

    Empty is the premise holding. **This is an ALL-of check over the declared
    targets, deliberately.** ``structure_only_components`` answers "is anything
    here virtual", which a two-target family satisfies with ONE target virtual
    while the other traces ~weight-scale real tensors — and on a
    ``place=False`` load that second target sits on the HOST, so the
    off-host walk (:func:`gen_worker.boot_trace_child.off_host_tensors`) reads
    clean too. Both guards can be green while the weight-free premise every
    VRAM conclusion downstream rests on is false (pgw#1173).

    BUFFERS ARE NOT COUNTED. A structure-only component's buffers stay real by
    construction — they are config-derived tables and a literal-bearing family
    ships them inside the compiled graph (see this module's header). Parameters are the
    checkpoint, and the checkpoint is the thing that must not be here.
    """
    import torch

    out: List[WeightFreeBreach] = []
    for name in sorted({str(t).strip() for t in (targets or ()) if str(t).strip()}):
        module = target_module(pipe, name)
        if module is None:
            continue  # not carried by THIS slot — legitimately absent
        real_bytes = 0
        devices: List[str] = []
        try:
            params = list(module.named_parameters())
        except Exception:  # noqa: BLE001 — an unwalkable target is reported below
            params = []
        for _pname, tensor in params:
            if not isinstance(tensor, torch.Tensor) or mi.is_virtual(tensor):
                continue
            real_bytes += int(tensor.numel()) * int(tensor.element_size())
            devices.append(str(tensor.device))
        stamped = bool(getattr(module, STAMP, False))
        if stamped and not real_bytes:
            continue
        if not stamped and not real_bytes and params:
            # Not stamped, and every parameter is already virtual: some other
            # mechanism (an author-side meta build) delivered the property.
            # The premise is what is fenced, not the stamp.
            continue
        out.append(WeightFreeBreach(
            component=name,
            cls_name=type(module).__name__,
            reason="real_parameters" if stamped else "not_structure_only",
            real_param_bytes=real_bytes,
            devices=tuple(sorted(set(devices))[:4]),
        ))
    return tuple(out)


def assert_weight_free(pipe: Any, targets: Any, *, what: str = "") -> None:
    """Fail closed unless every compile target ``pipe`` carries is weight-free.

    The typed fence pgw#1173 asked for. It raises :class:`StructureNotHonored`
    — the type that ALREADY means "this composition is holding weights it must
    not" and that every caller of the structure-only path is required to treat
    as fatal rather than as a reason to fall back.

    Raises when the pipeline carries NONE of its declared targets too: a trace
    with no target is not a weight-free trace, it is a trace of nothing, and
    reporting it as success is how a derivation can look clean and mean
    nothing.
    """
    names = sorted({str(t).strip() for t in (targets or ()) if str(t).strip()})
    carried = [n for n in names if target_module(pipe, n) is not None]
    cls_name = type(pipe).__name__
    if names and not carried:
        raise StructureNotHonored(
            component=",".join(names), cls_name=cls_name,
            lacks=(
                f"{cls_name} carries none of the declared compile target(s) "
                f"{names!r}{' for ' + what if what else ''}, so there is "
                f"nothing here whose weight-freedom could be proven"))
    breaches = weight_free_breaches(pipe, names)
    if not breaches:
        return
    total = sum(b.real_param_bytes for b in breaches)
    raise StructureNotHonored(
        component=",".join(b.component for b in breaches), cls_name=cls_name,
        lacks=(
            f"{len(breaches)} of {len(carried)} declared compile target(s) "
            f"on {cls_name}{' for ' + what if what else ''} hold REAL "
            f"parameters totalling {total} bytes — "
            + "; ".join(b.sentence() for b in breaches)))


def modules_of(pipe: Any) -> Tuple[Tuple[str, Any], ...]:
    """``(attribute, module)`` for every structure-only component of ``pipe``."""
    out: List[Tuple[str, Any]] = []
    if getattr(pipe, STAMP, False):
        return ((getattr(pipe, "__class__").__name__, pipe),)
    for name, value in list(vars(pipe).items() if hasattr(pipe, "__dict__")
                            else []):
        if getattr(value, STAMP, False):
            out.append((str(name).lstrip("_"), value))
    return tuple(sorted(out, key=lambda row: row[0]))


def facts_of(pipe: Any) -> Tuple[StructureFacts, ...]:
    out: List[StructureFacts] = []
    for value in list(vars(pipe).values() if hasattr(pipe, "__dict__") else []):
        facts = getattr(value, FACTS_STAMP, None)
        if isinstance(facts, StructureFacts):
            out.append(facts)
    if not out:
        facts = getattr(pipe, FACTS_STAMP, None)
        if isinstance(facts, StructureFacts):
            out.append(facts)
    return tuple(out)


def fake_mode_of(obj: Any) -> Optional[Any]:
    """The fake mode this object's virtual tensors belong to, if any.

    Derived from the tensors themselves rather than threaded through the
    call stack: the export and the compile must run inside the SAME mode, and
    the program itself is the only thing that knows which one that is.
    """
    mode = getattr(obj, MODE_STAMP, None)
    if mode is not None:
        return mode
    seen: List[Any] = []
    for getter in ("parameters", "buffers"):
        fn = getattr(obj, getter, None)
        if callable(fn):
            try:
                seen.extend(list(fn())[:8])
            except Exception:  # noqa: BLE001 — probing, never fatal
                continue
    if not seen and hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            mode = getattr(value, MODE_STAMP, None)
            if mode is not None:
                return mode
    for tensor in seen:
        mode = getattr(tensor, "fake_mode", None)
        if mode is not None:
            return mode
    return None


def fake_mode_of_program(program: Any) -> Optional[Any]:
    """The fake mode an EXPORTED PROGRAM's tensors belong to, if any.

    ``aot_compile`` asserts every one of its inputs shares one fake mode, so a
    program exported from a structure-only module must be compiled inside that
    same mode's tracing context. Read off the program rather than threaded
    through three call frames: the program is the only thing that knows.
    """
    for holder in ("state_dict", "constants"):
        table = getattr(program, holder, None)
        if not isinstance(table, dict):
            continue
        for tensor in table.values():
            mode = getattr(tensor, "fake_mode", None)
            if mode is not None:
                return mode
    return None


def program_shape_env(program: Any) -> Optional[Any]:
    """The ShapeEnv the EXPORT built — the one that knows this program's
    symbols and their value ranges."""
    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    nodes = getattr(graph, "nodes", ()) if graph is not None else ()
    for node in nodes:
        value = getattr(node, "meta", {}).get("val")
        for dim in getattr(value, "shape", ()) or ():
            env = getattr(getattr(dim, "node", None), "shape_env", None)
            if env is not None:
                return env
    return None


@contextlib.contextmanager
def compiling_under(program: Any) -> Iterator[None]:
    """Run AOTInductor inside the program's own fake mode when it has one.

    And inside the EXPORT's ShapeEnv, which is a separate fact and a
    load-bearing one. ``torch.export`` builds its own ShapeEnv while tracing,
    so the mode the structure was built in still carries the empty one it was
    constructed with — and inductor then looks up a symbol it has never seen:
    ``AssertionError: vr must not be None for symbol s21`` (measured; the
    same graph compiles with the program's env installed). Restored after,
    because the mode outlives this call.
    """
    mode = fake_mode_of_program(program)
    if mode is None:
        yield
        return
    import torch._guards as guards

    env = program_shape_env(program)
    previous = getattr(mode, "shape_env", None)
    if env is not None:
        mode.shape_env = env
    try:
        with guards.tracing(guards.TracingContext(mode)):
            yield
    finally:
        if env is not None:
            mode.shape_env = previous


@contextlib.contextmanager
def under(mode: Optional[Any]) -> Iterator[None]:
    """Run inside ``mode`` when there is one; unchanged when there is not."""
    if mode is None:
        yield
        return
    with mode:
        yield


# ---------------------------------------------------------------------------
# TWO THINGS THAT MUST NOT COME BACK HERE
# ---------------------------------------------------------------------------
#
# 1. A META round-trip (pgw#1111, deleted by pgw#1215). It existed because an
#    ExportedProgram crossed a process boundary by `torch.export.save`/`.load`
#    and a structure-only program's PARAMETERS are FAKE tensors with no storage
#    to serialize. th#1834 Phase 3 deleted the crossing — the process that
#    traces a graph class is the process that compiles it — so machinery that
#    repairs a round trip nobody takes is a second, silent way to hold a
#    program.
#
# 2. Real random values (pgw#1199): `materialize_random` / `restore_virtual` /
#    `stray_real_tensors` allocated one full checkpoint at compute dtype IN
#    THE PROCESS THIS MODULE EXISTS TO KEEP EMPTY — 56.2 GB on wan-2.2 against
#    15.5 GiB free, and the number §4.33's "a mint costs ~8 GiB" was really
#    measuring. §4.33 steps 4-5 put verification on the LIVE pipeline instead
#    (`gen_worker.handler_proof`: one warm forward through the endpoint's own
#    handler with REAL checkpoint values), which is strictly stronger. Without
#    real values here a lazily-built device-pinned table (ie#628's call-time
#    class) produces a FAKE constant the packer cannot pack — still loud, just
#    at pack time rather than at a stray walk.


__all__ = [
    "FACTS_STAMP",
    "MODE_STAMP",
    "STAMP",
    "TOKEN_CAPABILITY_MISSING",
    "TOKEN_UNSUPPORTED",
    "StructureCapabilityMissing",
    "StructureFacts",
    "StructureNotHonored",
    "StructureOnlyUnsupported",
    "WeightFreeBreach",
    "assert_weight_free",
    "build_component",
    "compiling_under",
    "facts_of",
    "fake_mode_of",
    "fake_mode_of_program",
    "modules_of",
    "program_shape_env",
    "refusal_token",
    "structure_only_components",
    "target_module",
    "virtualize",
    "weight_free_breaches",
    "under",
]
