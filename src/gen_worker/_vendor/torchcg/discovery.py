"""Instrumented graph discovery over author code as-is (pgw#1370's core).

DISCOVERY, not declaration-driven enumeration: torchcg hooks the lane's
compile-target modules, the author's own code runs with the author's sample
inputs, and every distinct call the marked modules actually receive is one
graph class. The OBSERVED ingress set IS the lane's graph set.

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .document import GraphRecord, LaneGraphs
from .graph_identity import GraphIdentityError, graph_hash
from .ingress import IngressError, build_call_ingress
from .lane import LaneError, LaneRef, require_targets, resolve_target


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


def _synthesize(value: Any) -> Any:
    """Restate one observed value as an exportable input.

    Tensors become zeros of the observed shape and dtype -- the observed
    tensor may be fake, real, or long gone; identity only needs its
    structure. Containers recurse; every other leaf passes through verbatim
    because its VALUE is part of the trace (a ``return_dict=False`` is a
    different graph than ``True``).
    """

    import torch

    if isinstance(value, torch.Tensor):
        return torch.zeros(tuple(int(d) for d in value.shape), dtype=value.dtype)
    if isinstance(value, (list, tuple)):
        return type(value)(_synthesize(item) for item in value)
    if isinstance(value, dict):
        return {key: _synthesize(item) for key, item in value.items()}
    return value


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
    program_sink: Callable[[str, Any], str] | None = None,
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
        lane, modules, drive, strict=strict, program_sink=program_sink
    )


def discover_modules(
    lane: LaneRef | str,
    modules: Mapping[str, Any],
    drive: Callable[[], object],
    *,
    strict: bool = False,
    program_sink: Callable[[str, Any], str] | None = None,
) -> LaneGraphs:
    """Discovery over MARKED modules (the imperative ``ctx.compile`` surface).

    Paul's review ruling: the compile hint is torch.compile-style -- author
    code marks real module objects in ``setup`` (``self.pipe.unet =
    ctx.compile(self.pipe.unet)``); nothing is spelled as a string. The
    caller (the derive) names each marked module for the document's
    provenance and hands them here keyed by that name.

    ``program_sink`` (Paul, 2026-08-20) receives ``(graph_hash,
    ExportedProgram)`` for each NEW graph class and returns the CAS digest it
    stored the serialized program under. The whole traced graph is kept, not
    just its hash: trace() runs once, ever, and the runtime miner downloads
    the graph and runs inductor rather than re-tracing author code. torchcg
    holds no opinion on WHERE the bytes go -- bytes-at-rest is tensorfs's
    charter, so the caller owns the sink.
    """

    import torch

    contract = lane.contract if isinstance(lane, LaneRef) else LaneRef(lane).contract
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
    for path, calls in observed.items():
        module = modules[path]
        fake_mode = _fake_mode_of(module)
        synthesis = fake_mode if fake_mode is not None else contextlib.nullcontext()
        for call in calls.values():
            with synthesis:
                args = tuple(_synthesize(value) for value in call.args)
                kwargs = {name: _synthesize(value) for name, value in call.kwargs}
            try:
                with synthesis:
                    program = torch.export.export(module, args, kwargs, strict=strict)
            except Exception as exc:
                raise DiscoveryError(
                    f"lane {contract!r} target {path!r}: torch.export"
                    f"(strict={strict}) failed for the observed call: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            names = _param_names(module, args, kwargs)
            try:
                ingress = build_call_ingress(program, names, args, kwargs)
                graph = graph_hash(program, ingress)
            except (IngressError, GraphIdentityError) as exc:
                raise DiscoveryError(
                    f"lane {contract!r} target {path!r}: observed call cannot "
                    f"state its identity: {exc}"
                ) from exc
            if graph in seen_graphs:
                # Two observed calls that trace identically ARE one graph
                # class -- dedup is the point of content identity.
                continue
            seen_graphs.add(graph)
            program_digest = ""
            if program_sink is not None:
                try:
                    program_digest = program_sink(graph, program)
                except Exception as exc:
                    raise DiscoveryError(
                        f"lane {contract!r} target {path!r}: graph {graph} "
                        f"could not be stored: {type(exc).__name__}: {exc}"
                    ) from exc
            records.append(
                GraphRecord(
                    graph=graph, target=path, ingress=ingress, program=program_digest
                )
            )

    unobserved = tuple(sorted(path for path, calls in observed.items() if not calls))
    return LaneGraphs(
        contract=contract,
        targets=targets,
        graphs=tuple(records),
        unobserved_targets=unobserved,
    )


__all__ = ["DiscoveryError", "discover_lane"]
