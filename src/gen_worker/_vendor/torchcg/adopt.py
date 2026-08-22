"""Serving the artifact: load, verify, dispatch.

**The selector IS the dispatcher.** One function decides whether a call fits a
record, and the same function states what must be done to the call to make it
fit; the dispatcher then does exactly that and nothing else. There is no plan
that can be computed and not executed, because computing it is how the match is
decided (se#837: the old tree computed a recast plan nothing ever called, so a
ddim/dpmpp/unipc request -- which ends `set_timesteps` in `.to(int64)` -- was
refused on dtype and served eager while the contract said it was covered).

There is no selection ladder here. Which artifact a pod should hold is the serve
ladder's question, not this library's.

Nothing on the hot path raises. A call that fits nothing serves eager, which is
the correct answer -- only speed is lost.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from .identity import CallIngress, CallInput, require_morphism, symbol_terms
from .refuse import AdoptError, KeyMismatch
from .store import StoredArtifact

logger = logging.getLogger("torchcg")

#: The dtypes a rank-0 feed may be recast TO. bfloat16 and float16 are
#: deliberately absent: bf16's 8 mantissa bits round timestep 999 to 1000, which
#: is a numeric change, not a normalization.
RECAST_TARGETS = ("float32", "float64")
RECAST_SOURCES = ("int8", "int16", "int32", "int64", "uint8")


@dataclass(frozen=True, slots=True)
class Recast:
    """One rank-0 feed that fits after a dtype change, and nothing else.

    Value-preserving by construction: rank 0, integral source, float target wide
    enough to hold every integer it admits. The scalar timestep dtype is a
    per-request SAMPLER fact -- euler-family present float32, ddim/dpmpp/unipc
    present int64 -- and the sampler is deliberately not a compile axis.
    """

    input: str
    position: int
    dtype: str

    def apply(self, value: Any) -> Any:
        import torch

        return value.to(getattr(torch, self.dtype))


def _row_gap(
    row: CallInput,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    bounds: Mapping[str, tuple[int, int]],
    symbols: dict[str, tuple[int, str]],
) -> tuple[str, Recast | None]:
    """Why this ONE row refuses this call, or how to make it fit.

    THE single source for the match decision, its diagnosis, AND the
    normalization. `("", None)` is a clean pass; `("", Recast(...))` is a pass
    that costs one staged copy; a non-empty sentence is a refusal. Because the
    normalization is produced by the same pass that decides admission, a plan
    that is computed and never executed cannot exist.

    A symbolic dim is not a wildcard. `bounds` is the record's own
    `symbol_bounds` and `symbols` is the binding table carried ACROSS the
    record's rows, so a dynamic record guards exactly what it was exported for:
    the value is inside the exported range, it respects the symbol's stride, and
    one symbol takes one value everywhere it appears. Without the table each row
    would answer alone and the dispatcher would enter a graph whose shape env
    refuses the call -- a wrong answer, not a slow one.
    """

    import torch

    resolved, value = row.resolve(args, kwargs)
    if not resolved:
        return (
            f"input {row.param!r} (position {row.position}, path {row.path!r}) "
            f"never resolved from the call -- absent from args/kwargs"
        ), None
    if not isinstance(value, torch.Tensor):
        received = repr(value)
        if len(received) > 60:
            received = received[:60] + "..."
        return (
            f"input {row.param!r}: received py-type {type(value).__name__} "
            f"({received}), expected a torch.Tensor {row.dtype} {tuple(row.shape)}"
        ), None
    dtype = str(value.dtype).removeprefix("torch.")
    recast: Recast | None = None
    if dtype != row.dtype:
        if (
            not row.shape
            and value.shape == ()
            and row.dtype in RECAST_TARGETS
            and dtype in RECAST_SOURCES
        ):
            recast = Recast(row.name, row.position, row.dtype)
        else:
            return (
                f"input {row.param!r}: dtype {dtype} != expected {row.dtype} "
                f"(received shape {tuple(value.shape)})"
            ), None
    if len(value.shape) != len(row.shape):
        return (
            f"input {row.param!r}: rank {len(value.shape)} (shape "
            f"{tuple(value.shape)}) != expected rank {len(row.shape)} (shape "
            f"{tuple(row.shape)})"
        ), None
    for axis, (got, want) in enumerate(zip(value.shape, row.shape, strict=True)):
        got = int(got)
        if isinstance(want, int):
            if got != want:
                return (
                    f"input {row.param!r}: shape {tuple(value.shape)} != expected "
                    f"{tuple(row.shape)}"
                ), None
            continue
        span = bounds.get(want)
        if span is None:
            return (
                f"input {row.param!r}: dim {axis} names symbol {want!r}, which this "
                f"record declares no range for"
            ), None
        low, high = span
        if not low <= got <= high:
            return (
                f"input {row.param!r}: dim {axis} (symbol {want!r}) = {got} outside "
                f"the exported range [{low}, {high}] (received shape "
                f"{tuple(value.shape)})"
            ), None
        stride, root = symbol_terms(want)
        if got % stride:
            return (
                f"input {row.param!r}: dim {axis} (symbol {want!r}) = {got} is not a "
                f"multiple of {stride}, which this graph's shape guards require "
                f"(received shape {tuple(value.shape)})"
            ), None
        bound = got // stride
        prior = symbols.get(root)
        if prior is not None and prior[0] != bound:
            return (
                f"input {row.param!r}: dim {axis} (symbol {want!r}) = {got} puts "
                f"{root} at {bound}, but input {prior[1]!r} put it at {prior[0]}"
            ), None
        symbols.setdefault(root, (bound, row.name))
    return "", recast


def fit(
    ingress: CallIngress, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> tuple[str, tuple[Recast, ...], int]:
    """(refusal sentence, normalizations, rows that passed).

    An empty sentence means the call fits once the normalizations are applied.
    """

    bounds = ingress.symbol_bounds
    symbols: dict[str, tuple[int, str]] = {}
    recasts: list[Recast] = []
    for passed, row in enumerate(ingress.inputs):
        gap, recast = _row_gap(row, args, kwargs, bounds, symbols)
        if gap:
            return gap, (), passed
        if recast is not None:
            recasts.append(recast)
    return "", tuple(recasts), len(ingress.inputs)


def _normalize(
    row: CallInput,
    recast: Recast,
    args: list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Rewrite one feed in place along its ingress path.

    Containers on the path are copied rather than mutated: the caller's own
    dict is not ours to change, and a sampler that reuses its kwargs across
    steps would otherwise see a dtype it did not set.
    """

    if row.param in kwargs:
        root: Any = kwargs[row.param]
        holder: Any = kwargs
        slot: Any = row.param
    elif row.param_position < len(args):
        root = args[row.param_position]
        holder = args
        slot = row.param_position
    else:  # pragma: no cover - fit() proved it resolves
        return
    if not row.path:
        holder[slot] = recast.apply(root)
        return
    trail: list[tuple[Any, Any]] = []
    value = root
    for step in row.path:
        trail.append((value, step))
        value = value[step]
    value = recast.apply(value)
    for container, step in reversed(trail):
        if isinstance(container, Mapping):
            replaced: Any = dict(container)
            replaced[step] = value
        else:
            replaced = list(container)
            replaced[step] = value
            if isinstance(container, tuple):
                replaced = tuple(replaced)
        value = replaced
    holder[slot] = value


@dataclass(frozen=True, slots=True)
class Record:
    """One armed graph: its call contract and the compiled thing behind it."""

    graph: str
    ingress: CallIngress
    call: Any

    @property
    def dynamic(self) -> bool:
        return bool(self.ingress.symbols)


@dataclass
class Dispatcher:
    """Exact-match-or-eager, with the normalization applied on the way in.

    Installed as an INSTANCE attribute (`module.forward = dispatcher`), so
    `Module.__call__` still routes through it while hooks and the class forward
    stay intact; deleting the attribute restores eager.
    """

    eager: Any
    records: tuple[Record, ...] = ()
    compiled_calls: int = 0
    eager_calls: int = 0
    recast_calls: int = 0
    _reported: bool = field(default=False, repr=False)

    def arm(self, records: Sequence[Record]) -> None:
        # A concrete bucket beats a dynamic range that also spans it.
        self.records = tuple(sorted(records, key=lambda r: r.dynamic))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for record in self.records:
            gap, recasts, _ = fit(record.ingress, args, kwargs)
            if gap:
                continue
            if recasts:
                rows = {row.name: row for row in record.ingress.inputs}
                call_args, call_kwargs = list(args), dict(kwargs)
                for recast in recasts:
                    _normalize(rows[recast.input], recast, call_args, call_kwargs)
                self.compiled_calls += 1
                self.recast_calls += 1
                return record.call(*call_args, **call_kwargs)
            self.compiled_calls += 1
            return record.call(*args, **kwargs)
        self.eager_calls += 1
        if self.records and not self._reported:
            self._reported = True
            self._say_first_gap(args, kwargs)
        return self.eager(*args, **kwargs)

    def _say_first_gap(self, args: Any, kwargs: Any) -> None:
        """Name the closest record once per module, never more.

        Wrapped whole: a diagnostic must never kill a serving forward.
        """

        try:
            best = max(
                (
                    (fit(record.ingress, args, kwargs), record)
                    for record in self.records
                ),
                key=lambda item: (item[0][2], item[1].graph),
            )
            (gap, _, passed), record = best
            logger.warning(
                "torchcg dispatch: NO armed graph matched a call -- %d graph(s) "
                "armed, serving eager. Closest record %s (%d/%d input row(s) "
                "matched); first divergence: %s. Reported once per module; "
                "further unmatched calls fall through silently.",
                len(self.records),
                record.graph,
                passed,
                len(record.ingress.inputs),
                gap,
            )
        except Exception:  # pragma: no cover - a diagnostic never kills a forward
            logger.debug("torchcg dispatch: no armed graph matched", exc_info=True)

    @property
    def silently_eager(self) -> bool:
        """Armed nothing and reported nothing -- indistinguishable from "no work"
        unless something asks."""

        return not self.records


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def resident_constants(module: Any) -> dict[str, Any]:
    """The live module's tensors by fqn.

    `state_dict()` PLUS `named_buffers()`: torch.export lifts non-persistent
    buffers as inputs and AOTI declares them state-dict constants, so
    `state_dict()` alone under-declares and the bind fails on a set mismatch.
    """

    values = dict(module.state_dict())
    for name, buffer in module.named_buffers():
        values.setdefault(name, buffer)
    return values


def module_device(module: Any) -> str:
    for table in (module.state_dict().values(), (b for _, b in module.named_buffers())):
        for value in table:
            if hasattr(value, "device"):
                return str(value.device)
    raise AdoptError("module holds no tensors, so it states no device")


def load(artifact: StoredArtifact, module: Any, *, key: str | None = None) -> Any:
    """Load one artifact and bind its constants to the LIVE module's weights.

    Weightless by construction: constants are bound `user_managed=True`, so they
    are raw pointers into tensors the module already holds. There is no second
    copy of the checkpoint on the device, which is also why this needs the
    module and a bare `load_package(path)` is unservable.
    """

    if key is not None and artifact.metadata.get("key") != key:
        raise KeyMismatch(
            f"artifact states key {artifact.metadata.get('key')!r}, fetched under {key!r}"
        )
    name = artifact.metadata["name"]
    declared_placement = artifact.metadata.get("placement") or []
    device = module_device(module)
    if declared_placement and device.split(":", 1)[0] not in {
        str(p).split(":", 1)[0] for p in declared_placement
    }:
        raise AdoptError(
            f"artifact was traced onto {sorted(declared_placement)!r} and the live "
            f"module is on {device!r}; a graph cannot be re-homed after tracing"
        )
    try:
        package = vars(import_module("torch._inductor.package"))["load_package"](
            str(artifact.package), model_name=name
        )
    except Exception as exc:
        raise AdoptError(f"package load failed: {type(exc).__name__}: {exc}") from exc

    declared = set(package.get_constant_fqns())
    stated = artifact.metadata.get("constants") or {}
    literals_named = set(stated.get("literal") or ())
    state_named = set(stated.get("state") or ())
    # The independent witness: the package's own table against what the mint
    # said it packed. Two producers of one fact must agree or neither is trusted.
    if declared != literals_named | state_named:
        raise AdoptError(
            f"artifact constant table disagrees with its own stamp: package-only "
            f"{sorted(declared - literals_named - state_named)[:6]!r}, stamp-only "
            f"{sorted((literals_named | state_named) - declared)[:6]!r}"
        )

    literals: dict[str, Any] = {}
    if artifact.literals is not None:
        from safetensors.torch import load_file

        literals = load_file(str(artifact.literals), device=device)
    state = resident_constants(module)
    morphism = require_morphism(artifact.metadata["declared_input_layout"])
    values: dict[str, Any] = {}
    missing: list[str] = []
    for fqn in sorted(declared):
        if fqn in literals_named:
            value = literals.get(fqn)
        else:
            value = state.get(fqn)
            if value is not None and not morphism.matches(value):
                raise AdoptError(
                    f"constant {fqn!r} is not in the declared layout "
                    f"{morphism.handle}: {morphism.describe(value)}. Deliver through "
                    f"the declared morphism or serve an artifact minted against the "
                    f"layout these bytes are in -- never a silent repack, which "
                    f"copies channels-last back to row-major at full weight cost on "
                    f"every load."
                )
        if value is None:
            missing.append(fqn)
        else:
            values[fqn] = value
    if missing:
        raise AdoptError(
            f"artifact declares {len(missing)} constant(s) nothing resolves: "
            f"{missing[:6]!r}"
        )
    try:
        package.load_constants(values, check_full_update=True, user_managed=True)
    except Exception as exc:
        raise AdoptError(f"constant binding failed: {type(exc).__name__}: {exc}") from exc
    # The user-managed pointers must not outlive their Python owners.
    package._torchcg_retained = values
    return package


def adopt(module: Any, artifacts: Sequence[tuple[StoredArtifact, Any]]) -> Dispatcher:
    """Arm one module with every artifact whose ingress its forward can serve."""

    dispatcher = Dispatcher(eager=module.forward)
    records = []
    for artifact, package in artifacts:
        ingress = CallIngress.decode(artifact.metadata["ingress"])
        records.append(Record(artifact.metadata["graph"], ingress, package))
    dispatcher.arm(records)
    module.forward = dispatcher
    return dispatcher


def release(module: Any) -> None:
    """Restore eager by removing the instance attribute."""

    if isinstance(module.__dict__.get("forward"), Dispatcher):
        del module.__dict__["forward"]


__all__ = [
    "RECAST_SOURCES",
    "RECAST_TARGETS",
    "Dispatcher",
    "Recast",
    "Record",
    "adopt",
    "fit",
    "load",
    "module_device",
    "release",
    "resident_constants",
]
