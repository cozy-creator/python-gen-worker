"""LoRA input-lifting: the rank-bucket adapter as call INPUTS.

Under AOT (``torch.export`` -> ``aoti_compile_and_package``) a compiled
artifact's tensors are either graph INPUTS or baked constants. Today's hot swap
copies into ``lora_a``/``lora_b`` module slots, which an exported artifact would
capture — a baked adapter is silent-wrong-output for every subsequent request.
Lifting removes the failure mode by construction: the whole bucket arrives as
two flat tensors in the call, so there is nothing to bake and the
no-baked-adapter gate degenerates to a signature check (G3).

**The shipped semantics are reused, not reimplemented.** The flat pair is a
re-layout of exactly the buffers canonical placement already allocates
(:func:`~.w8a8_lora.alloc_branch_buffers`), so:

- placement, rank-concat, and the ``alpha/rank * weight`` scale fold into B are
  performed by :func:`~.w8a8_lora.apply_branch_adapters` writing through VIEWS
  of the flat pair — the same code, the same numerics, the same refusals;
- deactivation is :func:`~.w8a8_lora.clear_branch_adapters`, i.e. zero-B in
  place: the addend is exactly 0 and the traced graph is unchanged;
- the addend keeps its canonical placement — ``_Fp8ScaledLinear.forward`` is
  untouched, so the branch still reads the PRE-QUANT activation and adds onto
  the bf16 output;
- ``bucket=0`` remains its own branchless graph class: it has no lifted
  signature at all, and installing one is refused.

VRAM is at parity with the shipped path (one flat pair replaces N per-module
buffers) and swaps write IN PLACE into the owned pair, so the call arguments are
pointer-stable — which is what cudagraph static inputs want.

Call convention (ONE convention for both the dynamo and export paths, because
both take it as call data):

    denoiser(*base_args, lora_a=<flat A>, lora_b=<flat B>, **base_kwargs)

Both kwargs are MANDATORY: tracing with them absent would trace the branchless
graph and silently constant-fold the adapter away, so a missing half refuses
typed.

The single deferred ``import torch`` (in :meth:`LiftedLoraPlan.alloc`) is
deliberate and matches ``w8a8``/``w8a8_lora``: ``compile_cache`` imports those at
module level and the discovery path (``python -m gen_worker.discover``, run at
image build time) has no torch. Keep it that way when wiring this in.
"""

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

#: The two lifted call arguments, in signature order.
LIFTED_INPUT_NAMES = ("lora_a", "lora_b")

_SLOT_NAMES = ("lora_a", "lora_b")
_BINDING_ATTR = "_cozy_lora_lifted"
_ORIG_FORWARD_ATTR = "_cozy_lora_lifted_orig_forward"


class _Slot(NamedTuple):
    """One branch-capable module's window into the flat pair."""

    path: str
    a_off: int
    a_shape: Tuple[int, ...]
    b_off: int
    b_shape: Tuple[int, ...]


class LiftedLoraPlan:
    """The flat layout of one denoiser's rank-``bucket`` branch container.

    Derived from the ALLOCATED branch buffers, so the layout mirrors canonical
    placement exactly and is a deterministic function of (module set, bucket) —
    which is precisely the compile cell's graph class. Offsets and shapes are
    plain ints, so the per-layer views are trace-time constants and the graph
    carries no indexing arithmetic.
    """

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
        """A zeroed flat pair on ``device``. Zeroed B == deactivated."""
        import torch

        return (torch.zeros(self.a_numel, dtype=self.dtype, device=device),
                torch.zeros(self.b_numel, dtype=self.dtype, device=device))

    def views(self, lora_a: Any, lora_b: Any) -> List[Tuple[str, Any, Any]]:
        """(module path, A view, B view) for every slot. Views ALIAS the flat
        pair, so in-place writes through them land in the call arguments."""
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
    """The flat layout for one ARMED denoiser (``enable_lora_branches`` first).

    Canonical placement is required: every branch-capable module must carry a
    branch, because the flat layout IS the graph class and a per-coverage layout
    would be a graph per adapter set. Sparse (eager-only) placement is
    therefore refused, as is a non-uniform branch compute dtype — one flat
    tensor has one dtype, and the alternative (a pair per dtype) would make the
    call signature depend on the checkpoint.
    """
    bucket = int(bucket or branch_bucket(model))
    if not bucket:
        raise ValidationError(
            "cannot lift a LoRA adapter at rank bucket 0 — the branchless "
            "pipeline is its own graph class and carries no adapter "
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


#: One plan slot paired with the module object it addresses.
ResolvedSlots = Tuple[Tuple[Any, _Slot], ...]


def resolve_slots(model: Any, plan: LiftedLoraPlan) -> ResolvedSlots:
    """Pair every slot with its module ONCE, ahead of any traced call.

    The per-call bind must not walk ``named_modules()``: that scan is O(module
    count) per denoiser step, and dynamo traces straight through it (it warns
    about the ``lru_cache`` inside ``branch_modules``) so the module walk would
    land in the compiled region.
    """
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
    """Point every branch slot at its window into the flat pair; return the
    prior slot values so :func:`unbind_views` can restore them exactly.

    Assignment goes through ``nn.Module.__setattr__``, which lands in
    ``_buffers`` for the w8a8 lane (non-persistent registered buffers) and in
    ``__dict__`` for the cast-hook/plain lanes — the same two homes
    ``alloc_branch_buffers`` uses, so both lanes' forwards read the views
    natively and no forward changes.
    """
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
    """One denoiser's lifted adapter: the owned flat pair plus the swap.

    A swap is ``apply_branch_adapters`` writing through views of the owned
    pair — identical placement/scale-fold/refusal semantics to the shipped
    buffer-copy path, because it IS that path. ``allow_resize=False`` and a
    ``rank_floor`` of the armed bucket keep it on the traced graph: a set that
    needs a wider bucket refuses instead of recompiling at swap time.
    """

    __slots__ = ("model", "plan", "resolved", "_a", "_b")

    def __init__(self, model: Any, plan: LiftedLoraPlan, device: Any) -> None:
        self.model = model
        self.plan = plan
        self.resolved = resolve_slots(model, plan)
        self._a, self._b = plan.alloc(device)

    @property
    def tensors(self) -> Tuple[Any, Any]:
        """The (A, B) flat pair to pass in the call. Pointer-stable."""
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
        """Make exactly ``adapters`` the active set. Empty = deactivate."""
        prior = self.bind()
        try:
            return apply_branch_adapters(
                self.model, adapters, allow_resize=False, uniform=True,
                request_id=request_id, rank_floor=self.plan.bucket)
        finally:
            self.unbind(prior)

    def clear(self) -> None:
        """Deactivate: zero B in place (the addend is exactly 0; the graph and
        the argument pointers are unchanged)."""
        prior = self.bind()
        try:
            clear_branch_adapters(self.model)
        finally:
            self.unbind(prior)


def _positional_arity(forward: Any) -> int:
    """How many positional parameters the denoiser's OWN forward takes."""

    try:
        params = inspect.signature(forward).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover — exotic callables
        return -1
    return sum(
        1 for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.name != "self")


def _lifted_signature(forward: Any) -> Any:
    """The denoiser's signature with ``lora_a``/``lora_b`` APPENDED as
    ordinary positional parameters.

    The mint reads this (``aot_declaration._positionalize``,
    ``aot_mint._input_names``) to bind example inputs to positional slots. They
    must NOT be keyword-only: `_positionalize` refuses a declared keyword-only
    input by name, and feeding the pair positionally would land it in ``*args``
    while the branch bound ``None`` and refused. The call convention is stated
    in the signature instead of being implied.
    """

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
    """Give one ARMED denoiser the lifted call signature. Idempotent.

    An INSTANCE forward wrap, not a wrapper module: the traced object stays the
    denoiser itself, so every parameter/buffer FQN is unchanged and FQN-keyed
    constant binding on the serve path is unaffected. Same idempotent
    install/remove shape as ``w8a8_lora._install_branch_forward``.
    """
    existing = getattr(model, _BINDING_ATTR, None)
    if isinstance(existing, LiftedLoraBinding):
        if existing.plan.bucket == int(bucket or existing.plan.bucket):
            return existing
        remove_lifted_lora_forward(model)
    plan = build_plan(model, bucket)
    # build_plan proved every slot is allocated, so the first one names the
    # device the branch container already lives on.
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
            # The all-positional MINT feed: an AOTI package's call convention
            # mirrors the traced args/kwargs split, so the pair must be
            # traceable positionally. It fills exactly the two slots this
            # wrapper appends to the denoiser's own signature, and
            # `_positionalize` fills every slot before them — so the arity is
            # exact, never a guess about a trailing tensor.
            args, lora_a, lora_b = args[:-2], args[-2], args[-1]
        if lora_a is None and lora_b is None:
            # Arming a bucket must not alter the semantics of calls that do not
            # use it. This wrapper replaces `forward` wholesale, so without this
            # branch a plain call RAISES on an armed pod. After the
            # positional-feed extraction, so it sees the pair however it
            # arrived; a HALF-supplied pair is a caller error and refuses below.
            #
            # Tracing the branchless arm with no operands is CORRECT (that arm
            # is what an `adapter=false` entry is), so this is not gated on a
            # compiling-check — and one would not work anyway:
            # `is_compiling()` reads False under strict export. The adapter
            # arm's feed is asserted at the mint instead.
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
    """Whether an adapter is CURRENTLY placed on this denoiser's branch.

    Reads ``w8a8_lora``'s own active flag — the one the per-request
    attach/clear path already maintains (``apply_branch_adapters`` sets it,
    ``clear_branch_adapters`` unsets it, and the lifted lane goes through both
    by construction). Adds no state, and deliberately does NOT inspect the
    flat pair's values: a ``lora_b.any()`` read would be a device
    synchronisation on every request.

    Serve-side routing: with a zeroed branch the branch-bearing graph and the
    branchless graph return the same tensor, so "no adapter is active" is the
    licence to serve the cheaper one.
    """
    return bool(w8a8_lora.branches_active(model))


# ---------------------------------------------------------------------------
# Set level: every branch-capable denoiser the pipeline carries gets
# its OWN flat pair — the module sets differ, so the layouts do — and they
# always move together.
# ---------------------------------------------------------------------------


def install_lifted_lora_execution_lanes(pipe: Any, bucket: int = 0) -> Dict[str, LiftedLoraBinding]:
    """Install the lifted signature on every branch-capable denoiser.

    Requires the canonical branch CONTAINERS to be allocated already — the
    flat layout is derived from them. Callers arming a pipeline from scratch
    want :func:`arm_lifted_lora_lanes`, which does both halves in order.
    """
    return {comp: install_lifted_lora_forward(model, bucket)
            for comp, model in branch_targets(pipe).items()}


def arm_lifted_lora_execution_lanes(
    pipe: Any, bucket: int,
) -> Dict[str, LiftedLoraBinding]:
    """Put a pipeline on the LIFTED branch-bearing graph family: canonical
    branch containers first, lifted call signature second.

    THE arm for every AOT path — the mint's adapter-bearing classes, the
    operator compose, and (via the same two calls) the serving arm in
    ``models.provision.arm_aot``. Order is load-bearing in one direction:
    the flat layout is derived from the allocated containers, so a lone
    :func:`install_lifted_lora_lanes` on an unarmed pipeline has nothing to lay
    out, and a lone ``enable_branch_lanes`` leaves the denoiser's own forward —
    the declaration would lift ``lora_a``/``lora_b`` while the module handed to
    ``torch.export`` never took them.

    Idempotent; a no-op at bucket 0 (the branchless pipeline is its own graph
    class and carries no adapter arguments at all).
    """
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
    """Set-level swap (the MoE contract): bind every component's views, then
    delegate to :func:`~.w8a8_lora.apply_branch_adapter_set` so the whole set
    settles in one pass with its fail-closed ordering intact."""
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


# ---------------------------------------------------------------------------
# Gate G3: under input-lifting the no-baked-adapter gate is a SIGNATURE check —
# there is no constant to bake, so the assertion is "no LoRA name reached the
# constant table, and the signature carries the pair".
#
# The gate must be asserted on the ExportedProgram, NOT on the loaded package.
# Packing keeps the FQN of a registered BUFFER (``lora_a``) but renames a
# plain-``__dict__`` tensor to ``_tensor_constant0`` — and the ``__dict__`` home
# is deliberate on the plain-bf16 and fp8-hooks lanes, so cast hooks cannot
# round-trip the branch through fp8. A name scan of ``get_constant_fqns()``
# therefore returns a FALSE PASS for two of the three branch-capable lanes: the
# adapter is baked in and no name matches. ``ep.constants`` still carries
# ``lin.lora_a``, which is why the pack-time gate reads the export.
# ---------------------------------------------------------------------------

#: Anonymous constants a pack emits when the source tensor had no buffer/param
#: name. Their contents are un-auditable by name.
_ANON_CONSTANT_PREFIX = "_tensor_constant"


def lora_constant_fqns(compiled: Any) -> Tuple[str, ...]:
    """Every constant/buffer/parameter FQN of a compiled artifact that NAMES a
    LoRA slot. Empty is the lifted invariant — but see
    :func:`package_constant_audit`: empty is NOT proof on a loaded package.

    Accepts an ``ExportedProgram`` (reads ``graph_signature`` + ``constants``,
    where FQNs survive) or a loaded ``AOTICompiledModel``.
    """
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
    """(LoRA-named constants, ANONYMOUS constants) of a loaded package.

    Defense in depth only. The anonymous half is the reason this cannot be the
    load-bearing gate: a baked ``__dict__``-home adapter lands there with its
    name erased, so an empty first half proves nothing on its own.
    """
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
    """Gate G3 — fail red at PACK time, naming the tensor.

    Assert on the ``ExportedProgram``: no LoRA FQN in its constant table
    (nothing CAN be baked) and both adapter arguments present among the user
    inputs. A MISSING pair is the same defect as a baked one — the branch was
    traced away, so every request gets the base model.

    A loaded package is REFUSED as the gate's subject: packing erases the FQN of
    a ``__dict__``-home adapter, so "no lora constant" there is a false pass.
    Use :func:`package_constant_audit` for the (advisory) package-side read.
    """
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
