"""Instrumented graph discovery over author code as-is (pgw#1370's core).

DISCOVERY, not declaration-driven enumeration: torchcg hooks the lane's
compile-target modules, the author's own code runs with the author's sample
inputs, and every distinct call the marked modules actually receive is one
graph specialization. The OBSERVED ingress set IS the lane's graph set.

Two passes, deliberately separated:

1. **Observation** -- the author's code runs untouched (its loop, its
   schedulers); forward pre-hooks record each target call's exact argument
   structure. Nothing is traced here, so anything the author's code does --
   ``.item()``, progress bars, control flow on tensor values -- is fine.
2. **Derivation** -- per distinct observed call, the target module alone is
   ``torch.export``-ed with synthesized inputs restating the observed
   structure (zeros of the observed shape/dtype; non-tensor leaves verbatim).
   Export is itself a fake-tensor trace, so this pass is CPU-cheap regardless
   of model size, and it is where the graph content hash comes from.

Sample inputs define the OBSERVED set, never a completeness claim: an
unobserved shape serves eager and background-mints on first live encounter --
a latency cost, not a correctness question.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .document import GraphRecord, LaneGraphs
from .dynamic import (
    DimPolicy,
    alias_ingress,
    contradicted_axes,
    decomposition_narrowing,
    format_narrowing,
    plan_exports,
    range_narrowing,
    resolve_policy,
    without_axes,
)
from .graph_identity import GraphIdentityError, graph_hash
from .ingress import IngressError, build_call_ingress
from .lane import LaneError, LaneRef, require_targets, resolve_target

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # torch-free import closure: transform.py is torch-shaped
    from .hollow import HollowSession
    from .transform import TransformSet


class DiscoveryError(RuntimeError):
    """A lane's targets cannot be observed or a derived graph cannot be stated."""


@dataclass(frozen=True, slots=True)
class _ObservedCall:
    """One distinct call shape a target module received."""

    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]


def _signature_of(value: Any) -> str:
    """Canonical dedup rendering: tensors by dtype/shape, leaves by repr."""

    import torch

    if isinstance(value, torch.Tensor):
        return f"t({value.dtype}|{tuple(int(d) for d in value.shape)})"
    if isinstance(value, (list, tuple)):
        body = ",".join(_signature_of(item) for item in value)
        return f"[{body}]" if isinstance(value, list) else f"({body})"
    if isinstance(value, Mapping):
        body = ",".join(
            f"{key!r}:{_signature_of(item)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    return repr(value)


def _synthesize(value: Any, device: Any = None) -> Any:
    """Restate one observed value as an exportable input.

    Tensors become zeros of the observed shape and dtype -- the observed
    tensor may be fake, real, or long gone; identity only needs its
    structure. Containers recurse; every other leaf passes through verbatim
    because its VALUE is part of the trace (a ``return_dict=False`` is a
    different graph than ``True``).

    ``device`` is not optional dressing (pgw#1458): an input synthesized on a
    device the module's parameters are not on either refuses outright
    (``FakeTensorDeviceMismatchError``) or -- worse -- exports cleanly with a
    MIXED placement that only AOTInductor rejects, minutes into a compile. It
    defaults to the device the module itself is on.
    """

    import torch

    if isinstance(value, torch.Tensor):
        return torch.zeros(
            tuple(int(d) for d in value.shape), dtype=value.dtype, device=device
        )
    if isinstance(value, (list, tuple)):
        return type(value)(_synthesize(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _synthesize(item, device) for key, item in value.items()}
    return value


def _trace_device_of(module: Any) -> Any:
    """The ONE device this module's tensors live on, or ``None`` if it has none.

    A module whose tensors straddle two devices cannot be exported to one
    graph; saying so here, by name, is the refusal that pgw#1458 spent four
    attempts not getting.
    """

    devices = {
        parameter.device
        for parameter in module.parameters()
    } | {
        buffer.device for buffer in module.buffers()
    }
    if len(devices) > 1:
        raise DiscoveryError(
            f"{type(module).__name__} straddles devices "
            f"{sorted(str(device) for device in devices)!r}; a graph is traced "
            f"onto ONE device and the device cannot be re-homed afterwards "
            f"(pgw#1458). Virtualize the whole module onto one trace device."
        )
    return devices.pop() if devices else None


def _fake_mode_of(module: Any) -> Any:
    """The one fake mode a hollow module's parameters belong to, or ``None``.

    A hollow (weights-free) module carries FAKE parameters; synthesized
    export inputs must then be created in the SAME mode -- ``torch.export``
    refuses a mix of real inputs and fake parameters, and two fake modes in
    one export refuse each other. A real-weight module answers ``None`` and
    keeps the plain real-zeros path.
    """

    from torch._subclasses.fake_tensor import FakeTensor

    modes = set()
    for parameter in module.parameters():
        if isinstance(parameter, FakeTensor):
            modes.add(parameter.fake_mode)
    if len(modes) > 1:
        raise DiscoveryError(
            f"{type(module).__name__} holds fake parameters from "
            f"{len(modes)} different fake modes; a hollow module must be "
            f"virtualized under ONE session"
        )
    return modes.pop() if modes else None


def _param_names(module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[str]:
    parameters = inspect.signature(module.forward).parameters.values()
    positional = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(args) > len(positional):
        raise DiscoveryError(
            f"{type(module).__name__}.forward received {len(args)} positional "
            f"values but declares only {len(positional)} positional parameters"
        )
    return positional[: len(args)] + list(kwargs)


def discover_lane(
    lane: LaneRef | str,
    targets: tuple[str, ...],
    roots: Mapping[str, object],
    drive: Callable[[], object],
    *,
    strict: bool = False,
    program_sink: Callable[[str, Any], object] | None = None,
    transforms: TransformSet | None = None,
    session: HollowSession | None = None,
    dynamic_dims: DimPolicy | bool | None = None,
    static_bind: bool = False,
    notes: list[str] | None = None,
) -> LaneGraphs:
    """Run the author's code once, derive every observed graph, state the rest.

    ``lane`` is a resolved ``LaneRef`` (or the bare contract handle);
    ``targets`` are the ENDPOINT-level compile paths -- they do not vary by
    lane. ``roots`` is the author's namespace (``{"pipe": pipe}``); ``drive``
    is the author's own invocation with trace inputs. The return states, per
    declared target, either its observed graphs or its membership in
    ``unobserved_targets`` -- silence is not an outcome.
    """

    import torch

    contract = lane.contract if isinstance(lane, LaneRef) else LaneRef(lane).contract
    targets = require_targets(tuple(targets))
    modules: dict[str, Any] = {}
    for path in targets:
        try:
            module = resolve_target(roots, path)
        except LaneError as exc:
            raise DiscoveryError(str(exc)) from exc
        if not isinstance(module, torch.nn.Module):
            raise DiscoveryError(
                f"lane {contract!r} target {path!r} resolves to "
                f"{type(module).__name__}, which is not a torch.nn.Module"
            )
        modules[path] = module
    return discover_modules(
        lane,
        modules,
        drive,
        strict=strict,
        program_sink=program_sink,
        transforms=transforms,
        session=session,
        dynamic_dims=dynamic_dims,
        static_bind=static_bind,
        notes=notes,
    )


def _demote_example_inputs(program: Any) -> None:
    """Drop the trace's example inputs before the program is serialized.

    A hollow derive's example inputs are FAKE tensors, and torch serializes
    them by pickling ``_reconstruct_fake_tensor`` -- which its own loader then
    refuses under ``weights_only=True``, on CPU, with a
    ``_pickle.UnpicklingError`` that surfaces as a torch-internal logging
    ``TypeError``. Measured: the identical program with example inputs
    demoted round-trips cleanly. Nothing downstream reads them -- the call
    contract is the ``CallIngress``, which is derived, recorded and verified
    -- so they are cost with no consumer.
    """

    for attribute in ("_example_inputs", "example_inputs"):
        try:
            setattr(program, attribute, None)
            return
        except AttributeError:
            continue


def _assert_literal_values_are_real(contract: str, path: str, program: Any) -> None:
    """A lifted constant must carry its VALUE, never a fake stand-in.

    ``graph_hash`` folds the literal digest in, so a graph whose constants
    were faked keys off a table of ZEROS and collides with every model that
    differs only in those constants (pgw#1458). The session used to fake them
    on a GPU-less cuda derive and swap the real values back here; since tcg#64
    it never fakes a value-bearing tensor at all, so this is the standing
    assertion that the property holds rather than a repair step.
    """

    from torch._subclasses.fake_tensor import FakeTensor

    constants = getattr(program, "constants", None) or {}
    faked = sorted(name for name, value in constants.items() if isinstance(value, FakeTensor))
    if not faked:
        return
    raise DiscoveryError(
        f"lane {contract!r} target {path!r}: lifted constant(s) {faked[:6]!r} are "
        f"FAKE, so this graph would key off a table of ZEROS and collide with "
        f"every other model that differs only in these constants (pgw#1458). A "
        f"hollow session keeps buffers and constants REAL; something else "
        f"replaced these."
    )


def discover_modules(
    lane: LaneRef | str,
    modules: Mapping[str, Any],
    drive: Callable[[], object],
    *,
    strict: bool = False,
    program_sink: Callable[[str, Any], object] | None = None,
    transforms: TransformSet | None = None,
    session: HollowSession | None = None,
    dynamic_dims: DimPolicy | bool | None = None,
    static_bind: bool = False,
    notes: list[str] | None = None,
) -> LaneGraphs:
    """Discovery over MARKED modules (the imperative ``ctx.compile`` surface).

    Paul's review ruling: the compile hint is torch.compile-style -- author
    code marks real module objects in ``setup`` (``self.pipe.unet =
    ctx.compile(self.pipe.unet)``); nothing is spelled as a string. The
    caller (the derive) names each marked module for the document's
    provenance and hands them here keyed by that name.

    ``program_sink`` receives ``(graph_hash, ExportedProgram)`` for each NEW
    graph specialization so the caller can bank this machine's serialized
    program under that identity. **Its return value is discarded** (Paul,
    2026-08-19, address-free): the bytes are local and their digest is
    machine-scoped, so nothing about them enters the document -- the graph
    hash is the identity, and a mint on any box resolves its own bytes by it.
    torchcg holds no opinion on WHERE they go: bytes-at-rest is tensorfs's
    charter, so the caller owns the sink.

    ``session`` (pgw#1458, tcg#64) is the :class:`~torchcg.hollow.
    HollowSession` the modules were virtualized under, when they were. It
    carries the two devices a GPU-less cuda derive needs -- the drive ran on
    one that exists, the graph is stamped with the one the session states --
    and ``session.restate`` is applied to each exported program HERE, before
    the hash, which is what makes the key independent of the box that derived
    it. A program whose lifted constants are FAKE is still refused by name --
    a silently zeroed literal digest is exactly the class of defect this pass
    exists to delete -- but under tcg#64 no session can produce one.

    ``dynamic_dims`` (tcg#77) decides how many EXPORTS the observed set costs.
    ``None`` (the default) is one export per observed call — the shape fan.
    ``True``, or a ``(target, input name, axis) -> bool`` predicate, marks the
    axes that VARY across the observed calls as ``torch.export.Dim``s, so the
    calls they distinguish collapse into one graph whose ingress states the
    axis as a RANGE. A refused dynamic export falls back to that group's own
    static exports, loudly. See :mod:`torchcg.dynamic`.

    ``static_bind`` (tcg#88, pgw#1603) makes the symbolic export TRACE-ONLY:
    the collapsed group still costs one ``torch.export`` of the module, but
    what is DECLARED is one STATIC record per covered call, produced by
    binding the call's concrete shapes to the symbolic parent (re-exporting
    ``parent.module()``, which re-traces a flat aten graph instead of the
    author's Python — measured far cheaper). The bound records are
    byte-identical to direct static exports — graph hash AND ingress —
    (pgw#1603's spike falsifier, torch 2.13, real and fake mode), so the
    author's STATIC serving declaration keeps its per-bucket artifacts while
    the trace pays structural-variant count, never bucket count. A cover
    whose bind is refused falls back to its own direct static export, loudly.

    ``notes``, when a caller passes a list, receives every demotion sentence
    that is otherwise only a ``logger.warning`` — a refused symbolic export,
    a guard-contradicted range, a refused bind. The derive puts them in the
    lock's own warnings so the lock SAYS an axis dropped to per-bucket
    tracing rather than silently paying the old wall (pgw#1603 acceptance c).

    ``transforms`` (tcg#52) is the SEALED :class:`~torchcg.transform.
    TransformSet` of the lane's PASS phase. Passing it is what makes PASS ->
    DISCOVERY -> EXPORT structural: the pass names enter every graph hash and
    the lane row, every transformed root must be on one of the marked
    modules' trees, and each marked module is stamped export-sealed so a pass
    can never run after this point. A lane that DECLARES passes and is handed
    no sealed set is refused -- silence is not an outcome here either.
    """

    import torch

    from .transform import require_transform_set, seal_for_export

    resolved = lane if isinstance(lane, LaneRef) else LaneRef(lane)
    contract = resolved.contract
    passes = require_transform_set(contract, transforms, resolved.passes)
    if not modules:
        raise DiscoveryError(
            f"lane {contract!r}: nothing is marked for compilation"
        )
    targets = require_targets(tuple(modules))
    for path, module in modules.items():
        if not isinstance(module, torch.nn.Module):
            raise DiscoveryError(
                f"lane {contract!r} marked object {path!r} is a "
                f"{type(module).__name__}, not a torch.nn.Module"
            )

    if transforms is not None:
        from .transform import related

        for index, root in enumerate(transforms.roots):
            if not any(related(root, module) for module in modules.values()):
                raise DiscoveryError(
                    f"lane {contract!r}: pass "
                    f"{transforms.passes[index] if index < len(transforms.passes) else '?'}"
                    f" transformed a {type(root).__name__} that is on no marked "
                    f"module's tree; the graph identified here would not be the "
                    f"graph the pass rewrote"
                )
    # From here the marked modules are export-bound: a pass that ran after
    # this point would mint an artifact for a module that no longer exists.
    for module in modules.values():
        seal_for_export(module)

    observed: dict[str, dict[str, _ObservedCall]] = {path: {} for path in targets}
    handles = []

    def _recorder(path: str) -> Callable[..., None]:
        def record(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            call = _ObservedCall(args=tuple(args), kwargs=tuple(kwargs.items()))
            key = _signature_of(args) + "|" + _signature_of(dict(kwargs))
            observed[path].setdefault(key, call)

        return record

    try:
        for path, module in modules.items():
            handles.append(
                module.register_forward_pre_hook(_recorder(path), with_kwargs=True)
            )
        drive()
    finally:
        for handle in handles:
            handle.remove()

    records: list[GraphRecord] = []
    seen_graphs: set[str] = set()
    dims = resolve_policy(dynamic_dims)

    def told(text: str) -> None:
        """One demotion sentence, to the log AND to the caller's notes."""

        logger.warning("%s", text)
        if notes is not None:
            notes.append(text)

    def declare(
        program: Any,
        path: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        names: list[str],
        plan: Any = None,
        bank: Any = None,
    ) -> None:
        # tcg#64: the drive ran on a device that EXISTS; the graph is
        # stamped with the device the session STATES. This is the one
        # place the two meet, and it is before the hash, so a GPU-less
        # box derives the same key a GPU box derives.
        if session is not None:
            session.restate(program)
        _assert_literal_values_are_real(contract, path, program)
        _demote_example_inputs(program)
        try:
            ingress = build_call_ingress(program, names, args, kwargs)
            if plan is not None:
                ingress = alias_ingress(ingress, plan)
            graph = graph_hash(program, ingress, passes=passes)
        except (IngressError, GraphIdentityError) as exc:
            raise DiscoveryError(
                f"lane {contract!r} target {path!r}: observed call cannot "
                f"state its identity: {exc}"
            ) from exc
        if graph in seen_graphs:
            # Two observed calls that trace identically ARE one graph
            # class -- dedup is the point of content identity.
            return
        seen_graphs.add(graph)
        # THE SINK'S RETURN IS DISCARDED, and that is the address-free
        # ruling in one line (Paul, 2026-08-19). The sink still banks this
        # machine's serialized program -- keyed by `graph`, which it is
        # handed precisely so it can -- but nothing about those bytes
        # enters the document. A blob digest is machine-scoped
        # (pgw#1462p2: 14/14 identities reproduce across boxes, 0/14 blob
        # digests do), so putting one in a document every machine must
        # agree on is what made the document unportable.
        if program_sink is not None:
            try:
                # tcg#88: a bound static record banks its symbolic PARENT
                # under its own identity — the CAS dedups the parent's bytes
                # to one blob across every bucket it covers, and the compile
                # seam re-binds on load (`bind.bind_static_spec`). Storage
                # follows the trace count down, not the bucket count.
                program_sink(graph, bank if bank is not None else program)
            except Exception as exc:
                raise DiscoveryError(
                    f"lane {contract!r} target {path!r}: graph {graph} "
                    f"could not be stored: {type(exc).__name__}: {exc}"
                ) from exc
        records.append(GraphRecord(graph=graph, target=path, ingress=ingress))

    for path, calls in observed.items():
        module = modules[path]
        fake_mode = _fake_mode_of(module)
        device = _trace_device_of(module)
        synthesis = fake_mode if fake_mode is not None else contextlib.nullcontext()
        synthesized: list[tuple[tuple[Any, ...], dict[str, Any], list[str]]] = []
        for call in calls.values():
            with synthesis:
                args = tuple(_synthesize(value, device) for value in call.args)
                kwargs = {name: _synthesize(value, device) for name, value in call.kwargs}
            synthesized.append((args, kwargs, _param_names(module, args, kwargs)))
        with synthesis:
            pending = plan_exports(path, synthesized, dims)
        while pending:
            plan = pending.pop(0)
            args, kwargs = plan.args, plan.kwargs
            try:
                with synthesis:
                    program = torch.export.export(
                        module,
                        args,
                        kwargs,
                        dynamic_shapes=plan.dynamic_shapes,
                        strict=strict,
                    )
            except Exception as exc:
                if plan.dynamic:
                    # tcg#77: a refused dynamic export is a MEASUREMENT, not a
                    # failure — this model cannot carry that axis symbolically,
                    # and the static fan it replaces is still exactly right.
                    # Loud, because a silent fallback is how a lane believes it
                    # collapsed a fan it did not.
                    told(
                        f"lane {contract} target {path}: dynamic export over "
                        f"{sorted(plan.symbols)} was REFUSED "
                        f"({type(exc).__name__}: {exc}); falling back to one "
                        f"static export per observed shape for this group"
                    )
                    pending = plan.static_fallback() + pending
                    continue
                raise DiscoveryError(
                    f"lane {contract!r} target {path!r}: torch.export"
                    f"(strict={strict}) failed for the observed call: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if plan.dynamic:
                # tcg#78: export ACCEPTING a Dim is not the graph agreeing to
                # serve its range. The guards that contradict a range live in
                # the DECOMPOSED graph -- `aten.to.dtype` asks nothing,
                # `torch._prims.convert_element_type` asks whether the tensor
                # can be size-1 -- so an axis torch.export admits can still be
                # narrowed to a single value 84-129 s into the AOTI compile
                # that finally decomposes it. Asking here costs ~2 s of CPU and
                # no compiler, and the artifact it prevents is not a slow
                # failure but a SILENT one: minted past this point, an sd15
                # UNet whose batch range collapsed to [2, 2] answers a batch-1
                # call with a batch-2 tensor of garbage and raises nothing.
                try:
                    with synthesis:
                        # tcg#88: under static_bind, the decompose half of the
                        # probe protects nothing — the tcg#78 hazard is a RANGE
                        # binary admitting a call its guards narrowed away, and
                        # a static bind mints no range binary: every declared
                        # record is re-traced at concrete shapes and hashed
                        # from that trace. Export-time narrowing (the measured
                        # sd15 batch case) still shows in the free check.
                        # Measured: decompose cost ~30 s per sd15-class group,
                        # for a question the static path never asks.
                        moved = (
                            range_narrowing(program)
                            if static_bind
                            else decomposition_narrowing(program)
                        )
                except Exception as exc:  # the compile would decompose too
                    moved = {}
                    told(
                        f"lane {contract} target {path}: the range probe over "
                        f"{sorted(plan.symbols)} could not decompose "
                        f"({type(exc).__name__}: {exc}); falling back to one "
                        f"static export per observed shape for this group"
                    )
                    pending = plan.static_fallback() + pending
                    continue
                if moved:
                    try:
                        refused = contradicted_axes(
                            path,
                            build_call_ingress(
                                program, list(plan.param_names), args, kwargs
                            ),
                            moved,
                        )
                    except IngressError:
                        refused = set()
                    dims = without_axes(dims, refused)
                    told(
                        f"lane {contract} target {path}: dynamic export over "
                        f"{sorted(plan.symbols)} was REFUSED — the graph's own "
                        f"guards contradict the declared range "
                        f"({format_narrowing(moved)}); re-planning this group "
                        f"with {sorted(refused) or 'the whole group'} held "
                        f"static"
                    )
                    # Only the CONTRADICTED axes are demoted, and only this
                    # group is re-planned. A batch axis the guards refuse must
                    # not take the aspect collapse down with it -- collapsing
                    # the aspect fan is what the dynamic-dims program is for.
                    # With nothing left to offer, the re-plan yields exactly
                    # tcg#77's static fallback, so there is one path, not two.
                    #
                    # The probe SHARES this program's ShapeEnv, so the program
                    # is now polluted by the refinement it revealed. It is
                    # discarded here and never declared; the re-plan exports
                    # again from the module.
                    with synthesis:
                        pending = plan_exports(path, list(plan.covers), dims) + pending
                    continue
            if plan.dynamic and static_bind:
                # tcg#88 (pgw#1603): the symbolic parent is TRACE-ONLY. What
                # is declared is one STATIC record per covered call, bound by
                # re-exporting the parent's flat graph module with the cover's
                # concrete shapes — spike-proven byte-identical to a direct
                # static export (hash and ingress), at a fraction of its cost
                # because it re-traces aten nodes, not the author's Python.
                # The PARENT is what gets banked, under every bound record's
                # identity (see `declare`'s bank=). All covers bind from the
                # pre-restate parent — the same module state a direct static
                # export traces — and only then does the parent take the
                # restate-before-serialize pass a declared program takes.
                with synthesis:
                    parent = program.module()
                bindings: list[tuple[Any, tuple[Any, ...], dict[str, Any], Any]] = []
                for cover_args, cover_kwargs, cover_names in plan.covers:
                    cover_kwargs = dict(cover_kwargs)
                    try:
                        with synthesis:
                            bound = torch.export.export(
                                parent, cover_args, cover_kwargs, strict=strict
                            )
                    except Exception as exc:
                        told(
                            f"lane {contract} target {path}: static bind of an "
                            f"observed cover was REFUSED "
                            f"({type(exc).__name__}: {exc}); falling back to "
                            f"its own direct static export"
                        )
                        try:
                            with synthesis:
                                bound = torch.export.export(
                                    module, cover_args, cover_kwargs,
                                    strict=strict,
                                )
                        except Exception as direct_exc:
                            raise DiscoveryError(
                                f"lane {contract!r} target {path!r}: "
                                f"torch.export(strict={strict}) failed for the "
                                f"observed call: "
                                f"{type(direct_exc).__name__}: {direct_exc}"
                            ) from direct_exc
                    bindings.append((bound, cover_args, cover_kwargs, cover_names))
                if session is not None:
                    session.restate(program)
                _assert_literal_values_are_real(contract, path, program)
                _demote_example_inputs(program)
                for bound, cover_args, cover_kwargs, cover_names in bindings:
                    declare(
                        bound, path, cover_args, cover_kwargs,
                        list(cover_names), bank=program,
                    )
                continue
            declare(
                program, path, args, kwargs, list(plan.param_names), plan
            )

    unobserved = tuple(sorted(path for path, calls in observed.items() if not calls))
    return LaneGraphs(
        contract=contract,
        targets=targets,
        graphs=tuple(records),
        unobserved_targets=unobserved,
        passes=passes,
    )


__all__ = ["DiscoveryError", "discover_lane", "discover_modules"]
