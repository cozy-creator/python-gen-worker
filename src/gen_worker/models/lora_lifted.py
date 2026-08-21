"""LoRA input-lifting: the rank-bucket adapter travels as call INPUTS — denoiser(*base_args, lora_a=<flat A>, lora_b=<flat B>, **base_kwargs) — so there is nothing an exported artifact can bake. BOTH kwargs are MANDATORY: tracing with them absent traces the branchless graph and silently constant-folds the adapter away, so a missing half refuses typed. Swaps write IN PLACE into the owned flat pair (pointer-stable call arguments, what cudagraph static inputs want); bucket 0 stays its own branchless specialization with no lifted signature. The deferred `import torch` is deliberate — discovery runs with no torch."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from ..api.errors import ValidationError
from . import w8a8_lora
from .w8a8_lora import (
    apply_branch_adapter_set,
    apply_branch_adapters,
    branch_bucket,
    branch_modules,
    branch_targets,
    clear_branch_adapters,
)
import inspect

logger = logging.getLogger(__name__)

LIFTED_INPUT_NAMES = ("lora_a", "lora_b")

_SLOT_NAMES = ("lora_a", "lora_b")
_BINDING_ATTR = "_cozy_lora_lifted"
_ORIG_FORWARD_ATTR = "_cozy_lora_lifted_orig_forward"


class _Slot(NamedTuple):

    path: str
    a_off: int
    a_shape: Tuple[int, ...]
    b_off: int
    b_shape: Tuple[int, ...]


class LiftedLoraPlan:
    """The flat layout of one denoiser's rank-``bucket`` branch container."""

    __slots__ = ("bucket", "dtype", "slots", "a_numel", "b_numel")

    def __init__(self, bucket: int, dtype: Any, slots: Sequence[_Slot],
                 a_numel: int, b_numel: int) -> None:
        self.bucket = int(bucket)
        self.dtype = dtype
        self.slots: Tuple[_Slot, ...] = tuple(slots)
        self.a_numel = int(a_numel)
        self.b_numel = int(b_numel)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (f"LiftedLoraPlan(bucket={self.bucket}, layers={len(self.slots)}, "
                f"a_numel={self.a_numel}, b_numel={self.b_numel}, "
                f"dtype={self.dtype})")

    def alloc(self, device: Any) -> Tuple[Any, Any]:
        """A zeroed flat pair on ``device``."""
        import torch

        return (torch.zeros(self.a_numel, dtype=self.dtype, device=device),
                torch.zeros(self.b_numel, dtype=self.dtype, device=device))

    def views(self, lora_a: Any, lora_b: Any) -> List[Tuple[str, Any, Any]]:
        """(module path, A view, B view) for every slot."""
        self.check_operands(lora_a, lora_b)
        return [
            (slot.path,
             lora_a.narrow(0, slot.a_off, _numel(slot.a_shape)).view(slot.a_shape),
             lora_b.narrow(0, slot.b_off, _numel(slot.b_shape)).view(slot.b_shape))
            for slot in self.slots
        ]

    def check_operands(self, lora_a: Any, lora_b: Any) -> None:
        """Refuse a flat pair that is not this plan's layout, by name."""
        for name, tensor, numel in (("lora_a", lora_a, self.a_numel),
                                    ("lora_b", lora_b, self.b_numel)):
            if tensor is None:
                raise ValidationError(
                    f"the lifted LoRA argument {name!r} is missing — a compiled "
                    "unit with a lifted adapter must be called with both "
                    f"{' and '.join(LIFTED_INPUT_NAMES)}; tracing without them "
                    "would trace the branchless graph and bake the absence"
                )
            if tensor.dim() != 1 or int(tensor.numel()) != numel:
                raise ValidationError(
                    f"lifted LoRA argument {name!r} has shape "
                    f"{tuple(tensor.shape)}, want a flat [{numel}] tensor for "
                    f"rank bucket {self.bucket} over {len(self.slots)} layers"
                )
            if tensor.dtype is not self.dtype:
                raise ValidationError(
                    f"lifted LoRA argument {name!r} is {tensor.dtype}, want "
                    f"{self.dtype} (the branch compute dtype this plan was "
                    "built at)"
                )


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def build_plan(model: Any, bucket: int = 0) -> LiftedLoraPlan:
    """The flat layout for one ARMED denoiser (``enable_lora_branches`` first)."""
    bucket = int(bucket or branch_bucket(model))
    if not bucket:
        raise ValidationError(
            "cannot lift a LoRA adapter at rank bucket 0 — the branchless "
            "pipeline is its own graph specialization and carries no adapter "
            "arguments; arm a bucket (Compile(lora_bucket=...)) first"
        )
    mods = branch_modules(model)
    if not mods:
        raise ValidationError(
            "this denoiser carries no branch-capable module (Fp8ScaledLinear, "
            "nn.Linear, nn.Conv2d) — there is no adapter to lift"
        )
    slots: List[_Slot] = []
    dtype: Any = None
    a_off = b_off = 0
    for path in sorted(mods):
        mod = mods[path]
        a = getattr(mod, "lora_a", None)
        b = getattr(mod, "lora_b", None)
        if a is None or b is None:
            raise ValidationError(
                f"branch-capable module {path!r} carries no branch container "
                f"at bucket {bucket} — input-lifting needs CANONICAL "
                "placement (a branch on every layer, zeroed where the adapter "
                "does not cover); sparse placement is eager-only"
            )
        if dtype is None:
            dtype = a.dtype
        if a.dtype is not dtype or b.dtype is not dtype:
            raise ValidationError(
                f"branch compute dtype is not uniform: {path!r} carries "
                f"{a.dtype}/{b.dtype} but the set started at {dtype} — one "
                "flat adapter tensor cannot hold two dtypes"
            )
        if int(a.shape[0]) != bucket or int(b.shape[1]) != bucket:
            raise ValidationError(
                f"branch container on {path!r} is rank "
                f"{int(a.shape[0])}/{int(b.shape[1])}, not the armed bucket "
                f"{bucket}"
            )
        a_shape = tuple(int(d) for d in a.shape)
        b_shape = tuple(int(d) for d in b.shape)
        slots.append(_Slot(path, a_off, a_shape, b_off, b_shape))
        a_off += _numel(a_shape)
        b_off += _numel(b_shape)
    return LiftedLoraPlan(bucket, dtype, slots, a_off, b_off)


ResolvedSlots = Tuple[Tuple[Any, _Slot], ...]


def resolve_slots(model: Any, plan: LiftedLoraPlan) -> ResolvedSlots:
    """Pair every slot with its module ONCE, ahead of any traced call."""
    mods = branch_modules(model)
    out: List[Tuple[Any, _Slot]] = []
    for slot in plan.slots:
        mod = mods.get(slot.path)
        if mod is None:
            raise ValidationError(
                f"this denoiser no longer carries the branch-capable module "
                f"{slot.path!r} the lifted plan was built over — the module "
                "set changed under the compiled unit"
            )
        out.append((mod, slot))
    return tuple(out)


def bind_views(resolved: ResolvedSlots, plan: LiftedLoraPlan, lora_a: Any,
               lora_b: Any) -> Tuple[Tuple[Any, Any], ...]:
    """Point every branch slot at its window into the flat pair; return the prior slot values so :func:`unbind_views` can restore them exactly."""
    plan.check_operands(lora_a, lora_b)
    prior: List[Tuple[Any, Any]] = []
    for mod, slot in resolved:
        prior.append((getattr(mod, "lora_a", None), getattr(mod, "lora_b", None)))
        mod.lora_a = lora_a.narrow(
            0, slot.a_off, _numel(slot.a_shape)).view(slot.a_shape)
        mod.lora_b = lora_b.narrow(
            0, slot.b_off, _numel(slot.b_shape)).view(slot.b_shape)
    return tuple(prior)


def unbind_views(resolved: ResolvedSlots, prior: Sequence[Tuple[Any, Any]]) -> None:
    """Restore the slot values :func:`bind_views` displaced."""
    for (mod, _slot), (a, b) in zip(resolved, prior):
        mod.lora_a = a
        mod.lora_b = b


class LiftedLoraBinding:
    """One denoiser's lifted adapter: the owned flat pair plus the swap."""

    __slots__ = ("model", "plan", "resolved", "_a", "_b")

    def __init__(self, model: Any, plan: LiftedLoraPlan, device: Any) -> None:
        self.model = model
        self.plan = plan
        self.resolved = resolve_slots(model, plan)
        self._a, self._b = plan.alloc(device)

    @property
    def tensors(self) -> Tuple[Any, Any]:
        """The (A, B) flat pair to pass in the call."""
        return (self._a, self._b)

    def call_kwargs(self) -> Dict[str, Any]:
        """``{"lora_a": ..., "lora_b": ...}`` for the call site to splat."""
        return {"lora_a": self._a, "lora_b": self._b}

    def bind(self, lora_a: Any = None, lora_b: Any = None) -> Tuple[Tuple[Any, Any], ...]:
        """Bind the owned pair (or an explicit one) onto the branch slots."""
        return bind_views(
            self.resolved, self.plan,
            self._a if lora_a is None else lora_a,
            self._b if lora_b is None else lora_b)

    def unbind(self, prior: Sequence[Tuple[Any, Any]]) -> None:
        unbind_views(self.resolved, prior)

    def swap(self, adapters: Sequence[Tuple[Dict[str, Any], float, str]], *,
             request_id: str = "") -> Dict[str, Any]:
        """Make exactly ``adapters`` the active set."""
        prior = self.bind()
        try:
            return apply_branch_adapters(
                self.model, adapters, allow_resize=False, uniform=True,
                request_id=request_id, rank_floor=self.plan.bucket)
        finally:
            self.unbind(prior)

    def clear(self) -> None:
        """Deactivate: zero B in place (the addend is exactly 0; the graph and the argument pointers are unchanged)."""
        prior = self.bind()
        try:
            clear_branch_adapters(self.model)
        finally:
            self.unbind(prior)


def _positional_arity(forward: Any) -> int:

    try:
        params = inspect.signature(forward).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover — exotic callables
        return -1
    return sum(
        1 for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.name != "self")


def _lifted_signature(forward: Any) -> Any:

    try:
        sig = inspect.signature(forward)
    except (TypeError, ValueError):  # pragma: no cover — exotic callables
        return None
    params = [p for p in sig.parameters.values()
              if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                inspect.Parameter.VAR_KEYWORD)]
    keyword_only = [p for p in params
                    if p.kind == inspect.Parameter.KEYWORD_ONLY]
    positional = [p for p in params
                  if p.kind != inspect.Parameter.KEYWORD_ONLY]
    lifted = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=None)
              for name in LIFTED_INPUT_NAMES]
    return sig.replace(parameters=positional + lifted + keyword_only)


def install_lifted_lora_forward(model: Any, bucket: int = 0) -> LiftedLoraBinding:
    """Give one ARMED denoiser the lifted call signature."""
    existing = getattr(model, _BINDING_ATTR, None)
    if isinstance(existing, LiftedLoraBinding):
        if existing.plan.bucket == int(bucket or existing.plan.bucket):
            return existing
        remove_lifted_lora_forward(model)
    plan = build_plan(model, bucket)
    mods = branch_modules(model)
    device = mods[plan.slots[0].path].lora_a.device
    binding = LiftedLoraBinding(model, plan, device)
    orig = model.forward
    resolved = binding.resolved
    arity = _positional_arity(orig)

    def _lifted_forward(*args: Any, **kwargs: Any) -> Any:
        lora_a = kwargs.pop("lora_a", None)
        lora_b = kwargs.pop("lora_b", None)
        if lora_a is None and lora_b is None and len(args) == arity + 2:
            args, lora_a, lora_b = args[:-2], args[-2], args[-1]
        if lora_a is None and lora_b is None:
            return orig(*args, **kwargs)
        prior = bind_views(resolved, plan, lora_a, lora_b)
        try:
            return orig(*args, **kwargs)
        finally:
            unbind_views(resolved, prior)

    _lifted_forward.__signature__ = _lifted_signature(orig)  # type: ignore[attr-defined]
    setattr(model, _ORIG_FORWARD_ATTR, orig)
    model.forward = _lifted_forward
    setattr(model, _BINDING_ATTR, binding)
    logger.info(
        "lora input-lifting installed: bucket=%d layers=%d dtype=%s "
        "flat_bytes=%d", plan.bucket, len(plan.slots), plan.dtype,
        (plan.a_numel + plan.b_numel) * binding.tensors[0].element_size(),
    )
    return binding


def remove_lifted_lora_forward(model: Any) -> None:
    """Restore the denoiser's own forward and drop the flat pair."""
    orig = getattr(model, _ORIG_FORWARD_ATTR, None)
    if orig is not None:
        model.forward = orig
    for attr in (_ORIG_FORWARD_ATTR, _BINDING_ATTR):
        try:
            delattr(model, attr)
        except AttributeError:
            pass


def lifted_binding(model: Any) -> Optional[LiftedLoraBinding]:
    """The installed binding, or ``None``."""
    found = getattr(model, _BINDING_ATTR, None)
    return found if isinstance(found, LiftedLoraBinding) else None


def adapter_active(model: Any) -> bool:
    """Whether an adapter is CURRENTLY placed on this denoiser's branch."""
    return bool(w8a8_lora.branches_active(model))


def install_lifted_lora_execution_lanes(pipe: Any, bucket: int = 0) -> Dict[str, LiftedLoraBinding]:
    """Install the lifted signature on every branch-capable denoiser."""
    return {comp: install_lifted_lora_forward(model, bucket)
            for comp, model in branch_targets(pipe).items()}


def arm_lifted_lora_execution_lanes(
    pipe: Any, bucket: int,
) -> Dict[str, LiftedLoraBinding]:
    """Put a pipeline on the LIFTED branch-bearing graph family: canonical branch containers first, lifted call signature second."""
    if not int(bucket or 0):
        return {}
    w8a8_lora.enable_branch_execution_lanes(pipe, int(bucket))
    return install_lifted_lora_execution_lanes(pipe, int(bucket))


def remove_lifted_lora_execution_lanes(pipe: Any) -> None:
    for model in branch_targets(pipe).values():
        remove_lifted_lora_forward(model)


def swap_lifted_execution_lane_set(
    pipe: Any,
    routed: Mapping[str, Sequence[Tuple[Dict[str, Any], float, str]]],
    *,
    request_id: str = "",
) -> Dict[str, Any]:
    """Set-level swap (the MoE contract): bind every component's views, then delegate to :func:`~.w8a8_lora.apply_branch_adapter_set` so the whole set settles in one pass with its fail-closed ordering int..."""
    bound: List[Tuple[LiftedLoraBinding, Tuple[Tuple[Any, Any], ...]]] = []
    try:
        for model in branch_targets(pipe).values():
            binding = lifted_binding(model)
            if binding is None:
                raise ValidationError(
                    "every denoiser of a lifted pipeline must carry the lifted "
                    "signature before a set swap — install_lifted_lora_lanes()"
                )
            bound.append((binding, binding.bind()))
        return apply_branch_adapter_set(
            pipe, routed, allow_resize=False, uniform=True,
            request_id=request_id)
    finally:
        for binding, saved in bound:
            binding.unbind(saved)


# Gate G3: under input-lifting the no-baked-adapter gate is a SIGNATURE check — no LoRA name in the constant table, and the signature carries the pair. Assert it on the ExportedProgram, NOT the loaded package: packing keeps a registered BUFFER's FQN (lora_a) but renames a plain-__dict__ tensor to _tensor_constant0 — and the __dict__ home is deliberate on the plain-bf16 and fp8-hooks lanes — so a name scan of the loaded package returns a FALSE PASS on two of the three branch-capable lanes; ep.constants still carries lin.lora_a.

_ANON_CONSTANT_PREFIX = "_tensor_constant"


def lora_constant_fqns(compiled: Any) -> Tuple[str, ...]:
    """Every constant/buffer/parameter FQN of a compiled artifact that NAMES a LoRA slot."""
    names: List[str] = []
    sig = getattr(compiled, "graph_signature", None)
    if sig is not None:
        for mapping in ("inputs_to_buffers", "inputs_to_parameters"):
            names.extend(str(v) for v in getattr(sig, mapping, {}).values())
        names.extend(str(k) for k in getattr(compiled, "constants", {}) or {})
    getter = getattr(compiled, "get_constant_fqns", None)
    if callable(getter):
        names.extend(str(f) for f in getter())
    return tuple(sorted({n for n in names if _names_lora(n)}))


def _names_lora(fqn: str) -> bool:
    return any(part in _SLOT_NAMES for part in fqn.split("."))


def is_exported_program(compiled: Any) -> bool:
    """True when the object still carries the export signature the gate needs."""
    return getattr(compiled, "graph_signature", None) is not None


def package_constant_audit(runner: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(LoRA-named constants, ANONYMOUS constants) of a loaded package."""
    getter = getattr(runner, "get_constant_fqns", None)
    if not callable(getter):
        return ((), ())
    fqns = [str(f) for f in getter()]
    return (tuple(sorted(f for f in fqns if _names_lora(f))),
            tuple(sorted(f for f in fqns if f.startswith(_ANON_CONSTANT_PREFIX))))


def lifted_input_names(exported: Any) -> Tuple[str, ...]:
    """The exported program's USER_INPUT names, in signature order."""
    sig = getattr(exported, "graph_signature", None)
    if sig is None:
        return ()
    out: List[str] = []
    for spec in getattr(sig, "input_specs", ()):
        if getattr(getattr(spec, "kind", None), "name", "") == "USER_INPUT":
            name = getattr(getattr(spec, "arg", None), "name", "")
            if name:
                out.append(str(name))
    return tuple(out)


def assert_no_baked_adapter(compiled: Any, *, label: str = "") -> None:
    """Gate G3 — fail red at PACK time, naming the tensor."""
    what = label or type(compiled).__name__
    if not is_exported_program(compiled):
        named, anon = package_constant_audit(compiled)
        if named:
            raise ValidationError(
                f"{what}: LoRA tensor(s) are BAKED into the package — "
                f"{', '.join(named)}; the adapter must arrive as a call input"
            )
        raise ValidationError(
            f"{what}: the no-baked-adapter gate cannot certify a LOADED "
            "package — packing renames a plain-attribute adapter to "
            f"{_ANON_CONSTANT_PREFIX}N ({len(anon)} anonymous constant(s) "
            "present here), so an empty name scan proves nothing. Gate the "
            "ExportedProgram at pack time instead"
        )
    baked = lora_constant_fqns(compiled)
    if baked:
        raise ValidationError(
            f"{what}: LoRA tensor(s) reached the compiled constant table and "
            f"would be BAKED into every request — {', '.join(baked)}; the "
            "adapter must arrive as a call input (install_lifted_lora_forward)"
        )
    names = lifted_input_names(compiled)
    missing = [n for n in LIFTED_INPUT_NAMES if n not in names]
    if missing:
        raise ValidationError(
            f"{what}: the lifted adapter argument(s) "
            f"{', '.join(missing)} are absent from the exported signature "
            f"(user inputs: {', '.join(names) or 'none'}) — the LoRA branch "
            "was traced away, which serves the base model for every request"
        )


__all__ = [
    "LIFTED_INPUT_NAMES",
    "adapter_active",
    "arm_lifted_lora_execution_lanes",
    "LiftedLoraBinding",
    "LiftedLoraPlan",
    "ResolvedSlots",
    "resolve_slots",
    "assert_no_baked_adapter",
    "bind_views",
    "build_plan",
    "install_lifted_lora_forward",
    "install_lifted_lora_execution_lanes",
    "lifted_binding",
    "lifted_input_names",
    "is_exported_program",
    "lora_constant_fqns",
    "package_constant_audit",
    "remove_lifted_lora_forward",
    "remove_lifted_lora_execution_lanes",
    "swap_lifted_execution_lane_set",
    "unbind_views",
]
