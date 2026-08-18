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

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .document import GraphRecord, LaneGraphs
from .graph_identity import GraphIdentityError, graph_hash
from .ingress import IngressError, build_call_ingress
from .lane import Lane, LaneError, resolve_target


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
    lane: Lane,
    roots: Mapping[str, object],
    drive: Callable[[], object],
    *,
    strict: bool = False,
) -> LaneGraphs:
    """Run the author's code once, derive every observed graph, state the rest.

    ``roots`` is the author's namespace (``{"pipe": pipe}``); ``drive`` is the
    author's own invocation with sample inputs. The return states, per
    declared target, either its observed graphs or its membership in
    ``unobserved_targets`` -- silence is not an outcome.
    """

    import torch

    modules: dict[str, Any] = {}
    for path in lane.compile:
        try:
            module = resolve_target(roots, path)
        except LaneError as exc:
            raise DiscoveryError(str(exc)) from exc
        if not isinstance(module, torch.nn.Module):
            raise DiscoveryError(
                f"lane {lane.name!r} target {path!r} resolves to "
                f"{type(module).__name__}, which is not a torch.nn.Module"
            )
        modules[path] = module

    observed: dict[str, dict[str, _ObservedCall]] = {path: {} for path in lane.compile}
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
        for call in calls.values():
            args = tuple(_synthesize(value) for value in call.args)
            kwargs = {name: _synthesize(value) for name, value in call.kwargs}
            try:
                program = torch.export.export(module, args, kwargs, strict=strict)
            except Exception as exc:
                raise DiscoveryError(
                    f"lane {lane.name!r} target {path!r}: torch.export"
                    f"(strict={strict}) failed for the observed call: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            names = _param_names(module, args, kwargs)
            try:
                ingress = build_call_ingress(program, names, args, kwargs)
                graph = graph_hash(program, ingress)
            except (IngressError, GraphIdentityError) as exc:
                raise DiscoveryError(
                    f"lane {lane.name!r} target {path!r}: observed call cannot "
                    f"state its identity: {exc}"
                ) from exc
            if graph in seen_graphs:
                # Two observed calls that trace identically ARE one graph
                # class -- dedup is the point of content identity.
                continue
            seen_graphs.add(graph)
            records.append(GraphRecord(graph=graph, target=path, ingress=ingress))

    unobserved = tuple(sorted(path for path, calls in observed.items() if not calls))
    return LaneGraphs(
        name=lane.name,
        contract=lane.contract,
        targets=lane.compile,
        graphs=tuple(records),
        unobserved_targets=unobserved,
    )


__all__ = ["DiscoveryError", "discover_lane"]
