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
:mod:`.svdq_native`). A PIPELINE's config is its component CLASS MAP
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
  a literal-bearing family ships INSIDE the cell: a fake buffer would make
  ``aot_package.literal_constants`` unpackable. Keeping them real also arms the
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..api.errors import WorkerError
from .. import meta_instantiation as mi

#: Stamped on a structure-only module and on the pipeline composed around it.
STAMP = "_cozy_structure_only"
#: The fake mode every virtual parameter of a structure-only tree belongs to.
MODE_STAMP = "_cozy_structure_fake_mode"
FACTS_STAMP = "_cozy_structure_facts"

#: Standard deviation of the random values the pgw#984 warm proof runs on.
#: Small enough that a deep stack of matmuls cannot overflow fp16, non-zero so
#: a degenerate all-zero forward cannot pass a proof a real one would fail.
RANDOM_STD = 0.02
#: Fixed, because a proof that behaves differently on two runs of the same cell
#: is not a proof. Not configurable: the value is not a tuning knob.
RANDOM_SEED = 1080


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
        super().__init__(
            f"structure-only build of component {self.component!r} "
            f"({self.cls_name or 'unknown class'}) is not possible: "
            f"{self.lacks}. The zero-download forge instantiates a compile "
            f"target from CODE + CONFIG (`load_config` + `from_config` under "
            f"`accelerate.init_empty_weights()` — the shape "
            f"`models/w8a8.py` already uses), so a class without that surface "
            f"has no structure-only path and this family is stranded on the "
            f"real-weight mint until it grows one"
            + (f" (tree: {self.tree})" if self.tree else "")
        )


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
    trace ``nn.Linear`` where serving runs ``Fp8ScaledLinear`` — a cell for a
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


def _init_empty_weights() -> Any:
    try:
        from accelerate import init_empty_weights
    except Exception as exc:  # noqa: BLE001
        raise StructureOnlyUnsupported(
            component="", cls_name="", lacks=(
                f"`accelerate` is not importable ({exc}), and "
                f"`init_empty_weights` is the mechanism that skips the state "
                f"dict")) from exc
    return init_empty_weights


def build_component(
    tree: str | Path, component: str, *, device: str = "", dtype: str = "",
) -> Tuple[Any, StructureFacts]:
    """Build ONE pipeline component from code + config, holding no weights.

    Returns the module and its measured facts. The module's PARAMETERS are
    fake tensors on ``device`` (faithful shape/dtype/device, zero storage); its
    BUFFERS are real, because they are config-derived and a literal-bearing
    family ships them.
    """
    import torch

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

    init_empty_weights = _init_empty_weights()
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
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise StructureOnlyUnsupported(
            component=component, cls_name=facts.cls_name, tree=str(root),
            lacks=f"device {device!r} is not available in this process")
    return module, facts


def _torch_dtype(root: Path, component: str, dtype: str) -> Any:
    """The compute dtype this component would have loaded at.

    Same resolution ``loading.load_component`` uses, because a structure whose
    precision differs from serving's is a different graph and a different cell.
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
                want = dtype if (dtype is not None
                                 and param.is_floating_point()) else param.dtype
                sub._parameters[name] = nn.Parameter(
                    torch.empty(tuple(param.shape), dtype=want),
                    requires_grad=False)
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
    cell is being minted for.
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


def is_structure_only(obj: Any) -> bool:
    """Whether this module — or any component of this pipeline — is virtual."""
    if getattr(obj, STAMP, False):
        return True
    return bool(structure_only_components(obj))


def structure_only_components(pipe: Any) -> Tuple[str, ...]:
    """Names of the pipeline attributes that are structure-only modules."""
    out: List[str] = []
    for name, value in list(vars(pipe).items() if hasattr(pipe, "__dict__")
                            else []):
        if getattr(value, STAMP, False):
            out.append(str(name).lstrip("_"))
    return tuple(sorted(set(out)))


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
# The pgw#984 warm proof runs on RANDOM values (coordinator ruling, 2026-08-10)
# ---------------------------------------------------------------------------


def materialize_random(module: Any, *, device: str = "") -> int:
    """Give every virtual parameter REAL random values; return the bytes.

    The endpoint's own handler is proved to RUN before a byte is exported
    (pgw#984), and a handler cannot run against zero-storage tensors. The
    ruling is random values — the proof is structural ("does it run"), and the
    ratified variability rule already forbids value-dependent program
    structure, so a handler whose control flow breaks under random weights is
    violating that rule and surfacing it is the point.

    Random values are NOT checkpoint values: nothing this process holds came
    from the model's bytes. The cost is reported (``values=random``) rather
    than hidden, because it IS weight-scale for the length of the proof.
    """
    import torch
    import torch.nn as nn

    dev = device or "cpu"
    generator = torch.Generator(device=dev).manual_seed(RANDOM_SEED)
    total = 0
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is None:
                continue
            real = torch.empty(tuple(param.shape), dtype=param.dtype,
                               device=dev)
            if param.is_floating_point() and real.dtype in (
                    torch.float16, torch.bfloat16, torch.float32,
                    torch.float64):
                real.normal_(0.0, RANDOM_STD, generator=generator)
            else:
                # fp8 and integer parameters have no `normal_`; a defined
                # value beats an undefined one, and NaN/overflow would make
                # every downstream cosine undefined (pgw#1056's guard).
                real.zero_()
            sub._parameters[name] = nn.Parameter(real, requires_grad=False)
            total += int(real.numel()) * int(real.element_size())
    return total


def stray_real_tensors(module: Any) -> Tuple[Tuple[str, Any], ...]:
    """Real tensors hanging off the tree that are NEITHER parameter nor buffer.

    This is how the z-image rope class actually shows up on a weightless mint,
    and it is the one place it CAN be seen. The gate's call-time half
    (ie#628) watches for a real allocation — but inside a fake-mode export
    every allocation is fake, so a lazily-built pinned table is invisible
    there. What it cannot hide is the RESIDUE: the pgw#984 warm proof runs the
    handler for real, the lazy build fires, and the table is cached on a plain
    attribute where ``restore_virtual`` (which only re-fakes parameters) does
    not reach. A real tensor still on the tree when the export begins is
    exactly the property failing.

    Searched: each module's own ``__dict__``, plus one level into plain
    objects it holds — the shape upstream uses (``RopeEmbedder`` is not an
    ``nn.Module``), and deep enough to name the attribute an author edits.
    """
    import torch

    out: List[Tuple[str, Any]] = []
    for prefix, sub in module.named_modules():
        registered = set(sub._parameters) | set(sub._buffers)
        for name, value in list(vars(sub).items()):
            if name.startswith("_") or name in registered:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, torch.Tensor):
                if not mi.is_virtual(value):
                    out.append((path, value))
                continue
            if isinstance(value, torch.nn.Module) or not hasattr(
                    value, "__dict__"):
                continue
            for inner, held in list(vars(value).items()):
                if isinstance(held, torch.Tensor) and not mi.is_virtual(held):
                    out.append((f"{path}.{inner}", held))
    return tuple(out)


def restore_virtual(module: Any, *, device: str = "") -> Any:
    """Return every parameter to a fake tensor, freeing the random values.

    A NEW fake mode, deliberately: the values that ran the proof are gone and
    the export must not carry a mode that ever held them.
    """
    return virtualize(module, device=device)


__all__ = [
    "FACTS_STAMP",
    "MODE_STAMP",
    "RANDOM_SEED",
    "RANDOM_STD",
    "STAMP",
    "StructureFacts",
    "StructureOnlyUnsupported",
    "build_component",
    "compiling_under",
    "facts_of",
    "fake_mode_of",
    "fake_mode_of_program",
    "is_structure_only",
    "materialize_random",
    "modules_of",
    "program_shape_env",
    "restore_virtual",
    "stray_real_tensors",
    "structure_only_components",
    "virtualize",
    "under",
]
