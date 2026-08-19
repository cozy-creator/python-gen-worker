"""Canonical declarations for one exported AOTInductor graph specialization."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .host_isa import HostISAError, _impose_host_policy
from .identity import CompiledGraphKey, _facts_digest, from_axes, toolchain_axis_digest
from .ingress import CallIngress, IngressError

_GRAPH_CLASS_CANONICAL_FORMAT = 1
_GRAPH_DIGEST_HEX = 16

#: tcg#55. The graph interface is now DERIVED, never supplied: every field is
#: a fact the ``ExportedProgram`` already encodes, plus the one fact it cannot
#: (the call ingress, which comes from the real call's parameter names and
#: nesting). v3's ``lifted_inputs``, ``specialization`` and ``pytree``
#: in/out spellings were read by NOTHING in this repo or in the gen-worker --
#: a parallel metadata layer that only existed to be validated, and therefore
#: only existed to be stubbed wrong (pgw#1456's two-field stub).
GRAPH_INTERFACE_FORMAT = 4
_GRAPH_INTERFACE_FIELDS = frozenset(("v", "constant_fqns", "ingress"))
#: The retired v3 spelling, kept ONLY so bytes carrying it refuse BY NAME.
_RETIRED_V3_FIELDS = frozenset(("lifted_inputs", "specialization", "pytree"))


class DeclarationError(ValueError):
    """A graph or runtime cannot state a complete v1 declaration."""


class RetiredGraphInterface(DeclarationError):
    """Bytes carry the retired v3 graph interface (tcg#55).

    Named rather than generic on purpose: graph specializations are CONTENT
    ADDRESSED, so there is nothing to migrate -- the next derive republishes
    under the v4 key. What must never happen is a v3 document being coerced
    into a v4 shape and silently keying to something no producer ever
    derived.
    """


def _refuse_retired_interface(graph: Mapping[str, Any]) -> None:
    """Refuse a v3-shaped graph interface by name, before anything else."""

    present = _RETIRED_V3_FIELDS & set(graph)
    version = graph.get("v")
    if not present and version != 3:
        return
    raise RetiredGraphInterface(
        f"graph interface is the RETIRED v3 shape "
        f"(v={version!r}, retired field(s) {sorted(present)!r}); tcg#55 derives "
        f"constant_fqns from the exported program, deletes lifted_inputs and "
        f"specialization (nothing ever read them), and promotes "
        f"pytree.ingress to a top-level 'ingress'. Graph specializations are content "
        f"addressed: re-derive to republish under a v{GRAPH_INTERFACE_FORMAT} key."
    )


class _Symbols:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}

    def name(self, symbol: Any) -> str:
        raw = str(symbol)
        known = self._ids.get(raw)
        if known is None:
            known = f"S{len(self._ids)}"
            self._ids[raw] = known
        return known

    def known(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._ids.items(), key=lambda item: item[1]))


def _render_symbol(value: Any, symbols: _Symbols) -> str:
    expression = getattr(getattr(value, "node", None), "expr", None)
    free = getattr(expression, "free_symbols", None) or ()
    if expression is None or not free:
        return str(expression if expression is not None else value)
    rendered = str(expression)
    for symbol in sorted(free, key=lambda item: len(str(item)), reverse=True):
        rendered = rendered.replace(str(symbol), symbols.name(symbol))
    return rendered


def _render_scalar(value: Any) -> str:
    import torch

    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        rendered = str(value)
        return rendered.split(":", 1)[0] if isinstance(value, torch.device) else rendered
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (int, str, bytes)):
        return repr(value)
    if isinstance(value, slice):
        return f"slice({value.start!r},{value.stop!r},{value.step!r})"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _render_value(value: Any, symbols: _Symbols) -> str:
    import torch

    if value is None:
        return "-"
    if isinstance(value, torch.Tensor):
        return _render_tensor(value, symbols, include_device=True)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    if isinstance(value, (list, tuple)):
        body = ",".join(_render_value(item, symbols) for item in value)
        return f"[{body}]" if isinstance(value, list) else f"({body})"
    if isinstance(value, dict):
        body = ",".join(
            f"{key!r}:{_render_value(item, symbols)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    return _render_scalar(value)


def _render_argument(value: Any, names: dict[Any, str], symbols: _Symbols) -> str:
    import torch

    if isinstance(value, torch.fx.Node):
        return names.get(value, "%?")
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_render_argument(item, names, symbols) for item in value) + ")"
    if isinstance(value, dict):
        body = ",".join(
            f"{key!r}:{_render_argument(item, names, symbols)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    if isinstance(value, torch.Tensor):
        return _render_tensor(value, symbols, include_device=False)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    return _render_scalar(value)


def _render_tensor(value: Any, symbols: _Symbols, *, include_device: bool) -> str:
    shape = ",".join(_render_symbol(dimension, symbols) for dimension in value.shape)
    device = f"|{value.device.type}" if include_device else ""
    return f"t({value.dtype}|[{shape}]{device})"


def _target(node: Any) -> str:
    if node.op == "placeholder":
        return ""
    if node.op in ("call_method", "get_attr", "output"):
        return str(node.target)
    module = getattr(node.target, "__module__", "")
    name = getattr(node.target, "__qualname__", "")
    return f"{module}.{name}" if module and name else str(node.target)


def _node_value(node: Any) -> tuple[bool, Any]:
    for key in ("val", "example_value", "tensor_meta"):
        if key in node.meta:
            return True, node.meta[key]
    return False, None


def _graph_lines(graph: Any, symbols: _Symbols) -> list[str]:
    names = {node: f"%{index}" for index, node in enumerate(graph.nodes)}
    lines: list[str] = []
    for index, node in enumerate(graph.nodes):
        args = ",".join(_render_argument(item, names, symbols) for item in node.args)
        kwargs = ",".join(
            f"{key}={_render_argument(item, names, symbols)}"
            for key, item in sorted(node.kwargs.items())
        )
        present, raw_value = _node_value(node)
        value = _render_value(raw_value, symbols) if present else "-"
        placeholder = f"arg{index}" if node.op == "placeholder" else ""
        lines.append(
            f"node {index} {node.op} {_target(node)} {placeholder} "
            f"args=({args}) kwargs=({kwargs}) val={value}"
        )
    return lines


def _bound(value: Any) -> str:
    if value is None:
        return "oo"
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return str(value).replace("+", "")


def _range_lines(ranges: Any, symbols: _Symbols) -> list[str]:
    if not ranges:
        return []
    known = dict(symbols.known())
    lines = []
    for symbol, value_range in ranges.items():
        canonical = known.get(str(symbol))
        if canonical is not None:
            lines.append(
                f"sym {canonical} range=[{_bound(getattr(value_range, 'lower', None))},"
                f"{_bound(getattr(value_range, 'upper', None))}]"
            )
    return sorted(lines)


def _signature_lines(program: Any) -> list[str]:
    signature = getattr(program, "graph_signature", None)
    lines: list[str] = []
    for direction, specs in (
        ("in", getattr(signature, "input_specs", None) or ()),
        ("out", getattr(signature, "output_specs", None) or ()),
    ):
        for index, spec in enumerate(specs):
            kind_object = getattr(spec, "kind", None)
            kind = getattr(kind_object, "name", str(kind_object or ""))
            target = getattr(spec, "target", None) or "-"
            lines.append(f"sig {direction} {index} kind={kind} target={target}")
    for attribute, direction in (("_in_spec", "in"), ("_out_spec", "out")):
        spec = getattr(program, attribute, None)
        if spec is not None:
            lines.append(f"spec {direction} {spec!s}".replace("\n", " "))
    return lines


def _canonical_graph(program: object) -> tuple[str, ...]:
    """Return the sole v1 canonical form for a ``torch.export.ExportedProgram``."""

    import torch

    if not isinstance(program, torch.export.ExportedProgram):
        raise DeclarationError("v1 declarations require a torch.export.ExportedProgram")
    symbols = _Symbols()
    lines = [f"v={_GRAPH_CLASS_CANONICAL_FORMAT} ir=export"]
    lines.extend(_graph_lines(program.graph_module.graph, symbols))
    lines.extend(_range_lines(program.range_constraints, symbols))
    lines.extend(_signature_lines(program))
    return tuple(lines)


def _graph_digest(program: object) -> str:
    # 64 bits, DERIVED and deliberately kept at v1. This graph-body witness is
    # one fact folded into GraphSpecializationDeclaration.specialization_hash, which is itself a
    # 16-hex choke point. Birthday bound P ~= N^2 / 2^65: about 3e-12 at 10^4
    # graph specializations and 3e-8 at 10^6. Widening only this site rekeys the corpus
    # while leaving graph-axis collision resistance at 64 bits, so BOTH
    # choke points move together or neither does.
    payload = "\n".join(_canonical_graph(program)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_GRAPH_DIGEST_HEX]


def _literal_names(program: object) -> tuple[str, ...]:
    signature = getattr(program, "graph_signature", None)
    state = {
        str(name)
        for field in ("parameters", "buffers")
        for name in (getattr(signature, field, ()) or ())
    }
    names = {str(name) for name in getattr(signature, "lifted_tensor_constants", ()) or ()}
    names.update(str(name) for name in (getattr(program, "constants", {}) or {}))
    return tuple(sorted(names - state))


def _constant_names(program: object) -> tuple[str, ...]:
    signature = getattr(program, "graph_signature", None)
    names = {
        str(name)
        for field in ("parameters", "buffers", "lifted_tensor_constants")
        for name in (getattr(signature, field, ()) or ())
    }
    names.update(str(name) for name in (getattr(program, "constants", {}) or {}))
    return tuple(sorted(names))


def _literal_digest(program: object) -> str:
    return _literal_digest_for(program, _literal_names(program))


def _literal_digest_for(program: object, names: Iterable[str]) -> str:
    names = tuple(sorted({str(name) for name in names}))
    if not names:
        return ""
    values = getattr(program, "constants", {}) or {}
    digest = hashlib.sha256()
    for name in names:
        value = values.get(name)
        if value is None:
            raise DeclarationError(f"literal constant {name!r} carries no value")
        try:
            import torch

            tensor = value.detach().cpu().contiguous().reshape(-1)
            _update_literal_digest(
                digest,
                name=name,
                dtype=str(value.dtype),
                shape=tuple(value.shape),
                chunks=(tensor.view(torch.uint8).numpy().tobytes(),),
            )
        except Exception as exc:
            raise DeclarationError(
                f"literal constant {name!r} could not be digested: {type(exc).__name__}: {exc}"
            ) from exc
    return digest.hexdigest()[:32]


def _update_literal_digest(
    digest: Any,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    chunks: Iterable[bytes],
) -> None:
    """Update a v1 literal digest from canonical facts and bounded byte chunks."""

    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(dtype.encode("utf-8"))
    digest.update(str(tuple(shape)).encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk)


def _placement(program: object) -> tuple[str, ...]:
    import torch

    devices: set[str] = set()

    def note(value: Any) -> None:
        if isinstance(value, torch.device):
            devices.add(f"{value.type}:{value.index}" if value.index is not None else value.type)
        elif isinstance(value, torch.Tensor):
            note(value.device)
        elif isinstance(value, (list, tuple)):
            for item in value:
                note(item)
        elif isinstance(value, dict):
            for item in value.values():
                note(item)

    for node in program.graph_module.graph.nodes:  # type: ignore[attr-defined]
        present, value = _node_value(node)
        if present:
            note(value)
        note(node.args)
        note(node.kwargs)
    return tuple(sorted(devices))


@dataclass(frozen=True, slots=True)
class GraphSpecializationDeclaration:
    """Immutable graph facts shared by lookup, mint, and admission."""

    name: str
    target: str
    graph: Mapping[str, Any]
    graph_witness: str
    range_digest: str
    fork: tuple[tuple[str, Any], ...] = ()
    specialization_dims: tuple[tuple[str, int], ...] = ()
    strict: bool = True
    lora_bucket: int = 0
    literal_values: str = ""
    placement: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = self.name.strip()
        target = self.target.strip()
        if not name or not target:
            raise DeclarationError("graph specialization name and target must be non-empty")
        if "\\" in name or any(
            part in ("", ".", "..") for part in name.split("/")
        ):
            raise DeclarationError(f"unsafe graph specialization name {name!r}")
        if not isinstance(self.graph, Mapping) or not self.graph:
            raise DeclarationError(
                "graph specialization graph interface must be a non-empty object"
            )
        try:
            canonical_graph = json.loads(
                json.dumps(dict(self.graph), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise DeclarationError(
                f"graph specialization graph interface is not finite JSON: {exc}"
            ) from exc
        if not isinstance(canonical_graph, dict) or not canonical_graph:
            raise DeclarationError(
                "graph specialization graph interface must be a non-empty object"
            )
        _refuse_retired_interface(canonical_graph)
        graph_fields = set(canonical_graph)
        if graph_fields not in (
            _GRAPH_INTERFACE_FIELDS,
            _GRAPH_INTERFACE_FIELDS | {"literal_values"},
        ):
            raise DeclarationError(
                "graph specialization graph interface fields must be exactly "
                f"{sorted(_GRAPH_INTERFACE_FIELDS)!r}, with literal_values only when present"
            )
        if (
            type(canonical_graph.get("v")) is not int
            or canonical_graph.get("v") != GRAPH_INTERFACE_FORMAT
        ):
            raise DeclarationError(
                f"graph specialization graph interface v must be {GRAPH_INTERFACE_FORMAT}"
            )
        values = canonical_graph.get("constant_fqns")
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ) or values != sorted(set(values)):
            raise DeclarationError(
                "graph specialization graph interface constant_fqns must be sorted unique strings"
            )
        try:
            ingress = CallIngress.from_graph(canonical_graph)
        except IngressError as exc:
            raise DeclarationError(f"graph specialization call ingress is invalid: {exc}") from exc
        if len(self.graph_witness) != _GRAPH_DIGEST_HEX or any(
            character not in "0123456789abcdef" for character in self.graph_witness
        ):
            raise DeclarationError(
                "graph_witness must be 16 lowercase hexadecimal characters"
            )
        if len(self.range_digest) != 32 or any(
            character not in "0123456789abcdef" for character in self.range_digest
        ):
            raise DeclarationError("range_digest must be 32 lowercase hexadecimal characters")
        if self.range_digest != ingress.digest():
            raise DeclarationError("range_digest does not restate graph.ingress")
        fork = tuple((str(name), value) for name, value in self.fork)
        if any(not name or name != name.strip() for name, _ in fork):
            raise DeclarationError(
                "graph specialization fork names must be non-empty canonical strings"
            )
        try:
            json.dumps([[name, value] for name, value in fork], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DeclarationError(f"graph specialization fork is not finite JSON: {exc}") from exc
        if fork != tuple(sorted(fork, key=lambda item: item[0])):
            raise DeclarationError("graph specialization fork must be sorted by name")
        specialization_dims = tuple((str(name), value) for name, value in self.specialization_dims)
        if any(
            not name
            or name != name.strip()
            or type(value) is not int
            for name, value in specialization_dims
        ):
            raise DeclarationError(
                "graph specialization dimensions must be canonical name/integer pairs"
            )
        if specialization_dims != tuple(sorted(specialization_dims, key=lambda item: item[0])):
            raise DeclarationError("graph specialization dimensions must be sorted by name")
        if type(self.strict) is not bool:
            raise DeclarationError("graph specialization strict must be a boolean")
        if type(self.lora_bucket) is not int:
            raise DeclarationError("graph specialization lora_bucket must be an integer")
        if self.literal_values and (
            len(self.literal_values) != 32
            or any(character not in "0123456789abcdef" for character in self.literal_values)
        ):
            raise DeclarationError(
                "literal-values digest must be 32 lowercase hexadecimal characters"
            )
        graph_literals = canonical_graph.get("literal_values")
        if graph_literals != (self.literal_values or None):
            raise DeclarationError(
                "graph interface literal_values must exactly match the compiled-graph payload"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "graph", canonical_graph)
        object.__setattr__(self, "fork", fork)
        object.__setattr__(self, "specialization_dims", specialization_dims)
        # tcg#55/pgw#1458: placement is ALWAYS recorded, including the
        # single-device case. It used to be dropped ("single-device placement
        # must be omitted"), which made the artifact's own device invisible in
        # its declaration -- so a cuda mint of a cpu-traced program could only
        # surface as AOTInductor's internal `Expected: cpu, Got: cuda:0`
        # assertion, four layers down, after the compile had already started.
        # It is DERIVED from the program, so recording it adds no supplied
        # field and no producer-can-lie seam.
        canonical_placement = tuple(sorted({str(device) for device in self.placement if device}))
        object.__setattr__(self, "placement", canonical_placement)

    @property
    def device_types(self) -> tuple[str, ...]:
        """The DEVICE CLASSES this graph was traced on (``cpu``, ``cuda``).

        Device is not arch. One cuda-traced graph serves every ``sm_NN`` --
        the arch enters at ARTIFACT level through
        :class:`RuntimeCompatibility` -- but it does not serve a cpu mint,
        because the trace itself stamps the device into every node's meta
        (and therefore into ``graph_witness``). Derived from placement, so
        there is nothing here a producer could state wrongly.
        """

        return tuple(sorted({device.split(":", 1)[0] for device in self.placement}))

    def facts(self) -> dict[str, object]:
        facts: dict[str, object] = {
            "v": GRAPH_INTERFACE_FORMAT,
            "target": self.target,
            "fork": [[name, value] for name, value in self.fork],
            "specialization_dims": [[name, value] for name, value in self.specialization_dims],
            "range_digest": self.range_digest,
            "graph": dict(self.graph),
            "graph_witness": self.graph_witness,
            "strict": self.strict,
            "lora_bucket": self.lora_bucket,
        }
        if self.placement:
            facts["placement"] = list(self.placement)
        return facts

    @property
    def specialization_hash(self) -> str:
        # 64 bits, DERIVED and deliberately kept at v1. THIS is the `graph`
        # axis's graph-specialization identity and the second 16-hex choke point; the
        # other is _graph_digest, which produces one fact above. The axis has
        # the MINIMUM of the two. Birthday bound P ~= N^2 / 2^65: about 3e-12
        # at 10^4 graph specializations and 3e-8 at 10^6. Widen both or neither.
        return _facts_digest(self.facts())[:_GRAPH_DIGEST_HEX]


@dataclass(frozen=True, slots=True)
class GraphSpecialization:
    """One exported program plus the only facts torch cannot know about it.

    tcg#55 (Paul: "torchcg does too much"). The producer used to hand over a
    ``graph`` interface mapping restating facts the ``ExportedProgram``
    already encodes. That mapping is gone. What is left is an
    ``ExportedProgram``, an IDENTITY (``name``/``target``), and the
    call INGRESS -- and the ingress is the one genuinely unknowable fact,
    because the parameter names and argument nesting of the call live in the
    author's ``forward`` signature, not in the traced graph.

    Every other interface field is derived in :meth:`declare`. A producer
    cannot state ``constant_fqns`` wrongly, because there is no parameter to
    state it with. The remaining constant-table checks are against an
    INDEPENDENT witness -- the compiled AOTI package's own
    ``get_constant_fqns()``, at admission (``engine._admit_constant_table``)
    and again at bind (``runner.bind``) -- never derived-against-derived.
    """

    name: str
    target: str
    program: object
    ingress: CallIngress
    fork: tuple[tuple[str, Any], ...] = ()
    specialization_dims: tuple[tuple[str, int], ...] = ()
    strict: bool = True
    lora_bucket: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.ingress, CallIngress):
            raise DeclarationError(
                "a graph specialization is declared from a typed CallIngress; the v3 raw "
                "graph-interface mapping is retired (tcg#55). Decode it first: "
                "CallIngress.decode(...)"
            )

    def declare(self) -> GraphSpecializationDeclaration:
        graph: dict[str, Any] = {
            "v": GRAPH_INTERFACE_FORMAT,
            "constant_fqns": list(_constant_names(self.program)),
            "ingress": self.ingress.as_dict(),
        }
        literals = _literal_digest(self.program)
        if literals:
            graph["literal_values"] = literals
        return GraphSpecializationDeclaration(
            self.name,
            self.target,
            graph,
            _graph_digest(self.program),
            self.ingress.digest(),
            fork=self.fork,
            specialization_dims=self.specialization_dims,
            strict=self.strict,
            lora_bucket=self.lora_bucket,
            literal_values=literals,
            placement=_placement(self.program),
        )


@dataclass(frozen=True, slots=True, init=False)
class RuntimeCompatibility:
    """Recorded compiler facts plus the current host policy for one target."""

    sm: str
    _toolchain: tuple[tuple[str, str], ...]
    _host_isa: tuple[tuple[str, str], ...]

    def __init__(self, target: str, *, toolchain: Mapping[str, str]) -> None:
        requested = str(target).strip().lower()
        if requested == "cpu":
            import torch

            architecture = f"cpu-{torch.backends.cpu.get_cpu_capability().lower()}"
        elif re.fullmatch(r"sm_[0-9]+", requested):
            import torch

            if not torch.cuda.is_available():
                raise DeclarationError("CUDA target requires a compatible visible CUDA device")
            major, minor = torch.cuda.get_device_capability()
            architecture = f"sm_{major}{minor}"
            if requested != architecture:
                raise DeclarationError(
                    f"requested CUDA target {requested!r} does not match {architecture!r}"
                )
        else:
            raise DeclarationError("runtime target must be 'cpu' or a concrete 'sm_NN'")
        cleaned = tuple(sorted((str(name), str(value)) for name, value in toolchain.items()))
        if not cleaned or any(
            not name or name != name.strip() or not value or value != value.strip()
            for name, value in cleaned
        ):
            raise DeclarationError(
                "runtime requires a worker-recorded toolchain block of non-empty strings"
            )
        try:
            host_facts = tuple(sorted(_impose_host_policy().items()))
        except HostISAError as exc:
            raise DeclarationError(f"cannot establish host ISA policy: {exc}") from exc
        object.__setattr__(self, "sm", architecture)
        object.__setattr__(self, "_toolchain", cleaned)
        object.__setattr__(self, "_host_isa", host_facts)

    @property
    def toolchain(self) -> dict[str, str]:
        return dict(self._toolchain)

    @property
    def host_isa(self) -> dict[str, str]:
        return dict(self._host_isa)

    @property
    def device_type(self) -> str:
        """The DEVICE CLASS this runtime compiles for -- ``cpu`` or ``cuda``."""

        return "cpu" if self.sm.startswith("cpu") else "cuda"

    def key(self, declaration: GraphSpecializationDeclaration) -> CompiledGraphKey:
        # pgw#1458: a program's device is established at TRACE time and cannot
        # be re-homed downstream -- four escalating attempts to move it were
        # each defeated by a different layer, and the terminal one was
        # AOTInductor's own internal `Expected: cpu, Got: cuda:0`. That
        # assertion fires minutes into a compile, names no graph specialization, and is
        # unreachable from a log. The declaration knows the trace device
        # (derived, always recorded), and this runtime knows the compile
        # device, so the disagreement is a one-line refusal BEFORE any work.
        traced = declaration.device_types
        if traced and self.device_type not in traced:
            raise DeclarationError(
                f"graph specialization {declaration.name!r} was TRACED on "
                f"{list(traced)!r} and this runtime compiles for "
                f"{self.device_type!r} ({self.sm}). Device is established at "
                f"trace time and cannot be re-homed into an exported program "
                f"(pgw#1458); re-derive the graph on {self.device_type!r}. "
                f"Note device is not arch: one {self.device_type!r} trace "
                f"serves every sm."
            )
        return from_axes(
            {
                "graph": declaration.specialization_hash,
                "sm": self.sm,
                "toolchain": toolchain_axis_digest(self.toolchain),
            }
        )

    def canonical(self) -> bytes:
        return json.dumps(
            {"sm": self.sm, "toolchain": self.toolchain},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
