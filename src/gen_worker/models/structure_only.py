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

STAMP = "_cozy_structure_only"
MODE_STAMP = "_cozy_structure_fake_mode"
FACTS_STAMP = "_cozy_structure_facts"

class StructureOnlyUnsupported(WorkerError):
    """This component cannot be built from code + config alone."""

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
    """The PROCESS cannot meta-instantiate — nothing about this family is wrong."""

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


TOKEN_UNSUPPORTED = "structure_unsupported"
TOKEN_CAPABILITY_MISSING = "structure_capability_missing"


def refusal_token(exc: StructureOnlyUnsupported) -> str:
    """Which boot-adopt token a structure-only refusal reports under."""
    return (TOKEN_CAPABILITY_MISSING
            if isinstance(exc, StructureCapabilityMissing)
            else TOKEN_UNSUPPORTED)


class StructureNotHonored(StructureOnlyUnsupported):
    """A component that WAS built weight-free was not carried by the pipeline."""


@dataclass(frozen=True)
class StructureFacts:
    """What one structure-only component costs, and what it did NOT cost."""

    component: str
    cls_name: str
    parameters: int = 0
    virtual_param_bytes: int = 0
    real_buffer_bytes: int = 0
    transient_real_bytes: int = 0
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


def _init_empty_weights(component: str = "") -> Any:
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
    """Build ONE pipeline component from code + config, holding no weights."""
    init_empty_weights = _init_empty_weights(component)

    root = Path(tree)
    cls = _component_class(root, component)
    _refuse_artifact_lanes(root, component, cls)
    _require_config_surface(cls, component, root)
    src = root / component if (root / component).is_dir() else root

    config = dict(cls.load_config(str(src)))
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
    """Parameters → fake tensors on ``device``; buffers → real, on ``device``."""
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


def structure_only_components(pipe: Any) -> Tuple[str, ...]:
    """Names of the pipeline attributes that are structure-only modules."""
    out: List[str] = []
    for name, value in list(vars(pipe).items() if hasattr(pipe, "__dict__")
                            else []):
        if getattr(value, STAMP, False):
            out.append(str(name).lstrip("_"))
    return tuple(sorted(set(out)))


def target_module(pipe: Any, target: str) -> Any:
    """The module holding the WEIGHTS of declared compile target ``target``."""
    name = str(target or "").strip()
    if not name:
        return None
    owner: Any = pipe
    for part in name.split("."):
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    return owner if hasattr(owner, "named_parameters") else None


@dataclass(frozen=True)
class WeightFreeBreach:
    """One declared compile target that is NOT weight-free, and why."""

    component: str
    cls_name: str
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
    """Every declared compile target ``pipe`` carries that holds real weights."""
    import torch

    out: List[WeightFreeBreach] = []
    for name in sorted({str(t).strip() for t in (targets or ()) if str(t).strip()}):
        module = target_module(pipe, name)
        if module is None:
            continue
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
    """Fail closed unless every compile target ``pipe`` carries is weight-free."""
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
    """The fake mode this object's virtual tensors belong to, if any."""
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
    """The fake mode an EXPORTED PROGRAM's tensors belong to, if any."""
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
    """The ShapeEnv the EXPORT built — the one that knows this program's symbols and their value ranges."""
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
    """Run AOTInductor inside the program's own fake mode when it has one."""
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
