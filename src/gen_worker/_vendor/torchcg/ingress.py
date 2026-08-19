"""Canonical call-ingress identity for one exported graph class."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PathStep = int | str
_INGRESS_FIELDS = frozenset(
    ("v", "parameters", "flat_arity", "inputs", "symbols", "excluded_inputs")
)
_INPUT_FIELDS = frozenset(
    ("name", "position", "param", "param_position", "path", "exported_name", "dtype", "shape")
)


class IngressError(ValueError):
    """A call-ingress declaration is incomplete or noncanonical."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = str(reason)
        super().__init__(detail if detail is not None else reason)


def _name(param: str, path: Sequence[PathStep]) -> str:
    name = param
    for step in path:
        name = step if isinstance(step, str) else f"{name}.{step}"
    return name


def exported_input_name(param: str, path: Sequence[PathStep] = ()) -> str:
    """Return the placeholder spelling produced by ``torch.export``."""

    return "_".join((str(param), *(str(step) for step in path)))


@dataclass(frozen=True, slots=True)
class CallInput:
    """One tensor feed and its exact identity in the unflattened call."""

    name: str
    position: int
    param: str
    param_position: int
    path: tuple[PathStep, ...]
    exported_name: str
    dtype: str
    shape: tuple[int | str, ...]

    def __post_init__(self) -> None:
        for field, value in (
            ("name", self.name),
            ("param", self.param),
            ("exported_name", self.exported_name),
            ("dtype", self.dtype),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise IngressError("input_invalid", f"call input {field} must be canonical")
        if type(self.position) is not int or self.position < 0:
            raise IngressError("input_invalid", "call input position must be non-negative")
        if type(self.param_position) is not int or self.param_position < 0:
            raise IngressError("input_invalid", "call input param_position must be non-negative")
        if not isinstance(self.path, tuple) or any(
            (type(step) is not int and not isinstance(step, str))
            or (type(step) is int and step < 0)
            or (isinstance(step, str) and (not step or step != step.strip()))
            for step in self.path
        ):
            raise IngressError("input_invalid", "call input path is not canonical")
        if self.name != _name(self.param, self.path):
            raise IngressError("input_invalid", "call input name does not restate param/path")
        if self.exported_name != exported_input_name(self.param, self.path):
            raise IngressError(
                "input_invalid", "call input exported_name does not restate param/path"
            )
        if not isinstance(self.shape, tuple) or any(
            (type(dimension) is not int and not isinstance(dimension, str))
            or (type(dimension) is int and dimension < 0)
            or (isinstance(dimension, str) and (not dimension or dimension != dimension.strip()))
            for dimension in self.shape
        ):
            raise IngressError("input_invalid", "call input shape is not canonical")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "param": self.param,
            "param_position": self.param_position,
            "path": list(self.path),
            "exported_name": self.exported_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    def resolve(
        self, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> tuple[bool, object | None]:
        if self.param in kwargs:
            value: object = kwargs[self.param]
        elif self.param_position < len(args):
            value = args[self.param_position]
        else:
            return False, None
        for step in self.path:
            if isinstance(step, str):
                if not isinstance(value, Mapping) or step not in value:
                    return False, None
                value = value[step]
            else:
                if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                    return False, None
                if step >= len(value):
                    return False, None
                value = value[step]
        return True, value


@dataclass(frozen=True, slots=True)
class CallIngress:
    """The closed, identity-hashed call contract consumed by graph runners."""

    parameters: tuple[str, ...]
    flat_arity: int
    inputs: tuple[CallInput, ...]
    symbols: tuple[tuple[str, tuple[int, int]], ...] = ()
    excluded_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple) or not self.parameters or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in self.parameters
        ):
            raise IngressError("ingress_invalid", "parameters must be canonical strings")
        if len(self.parameters) != len(set(self.parameters)):
            raise IngressError("ingress_invalid", "parameters must be unique")
        if type(self.flat_arity) is not int or self.flat_arity <= 0:
            raise IngressError("ingress_invalid", "flat_arity must be a positive integer")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise IngressError("ingress_invalid", "call ingress must declare tensor inputs")
        positions = tuple(row.position for row in self.inputs)
        if positions != tuple(sorted(set(positions))):
            raise IngressError("ingress_invalid", "input positions must be strictly increasing")
        if positions[-1] >= self.flat_arity:
            raise IngressError("ingress_invalid", "input position exceeds flat_arity")
        param_positions = tuple(row.param_position for row in self.inputs)
        if param_positions != tuple(sorted(param_positions)):
            raise IngressError(
                "ingress_invalid", "input param_position values must be nondecreasing"
            )
        for row in self.inputs:
            if row.param_position >= len(self.parameters):
                raise IngressError(
                    "ingress_invalid", f"input {row.name!r} param_position is out of range"
                )
            if self.parameters[row.param_position] != row.param:
                raise IngressError(
                    "ingress_invalid",
                    f"input {row.name!r} param does not match parameters[param_position]",
                )
        for attribute, values in (
            ("name", tuple(row.name for row in self.inputs)),
            ("exported_name", tuple(row.exported_name for row in self.inputs)),
        ):
            if len(values) != len(set(values)):
                raise IngressError("ingress_invalid", f"duplicate input {attribute}")

        if not isinstance(self.symbols, tuple):
            raise IngressError("ingress_invalid", "symbols must be canonical tuples")
        names = tuple(name for name, _ in self.symbols)
        if names != tuple(sorted(set(names))):
            raise IngressError("ingress_invalid", "symbols must be sorted and unique")
        for name, bounds in self.symbols:
            if not isinstance(name, str) or not name or name != name.strip():
                raise IngressError("ingress_invalid", "symbol name must be canonical")
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(type(bound) is not int for bound in bounds)
                or bounds[1] < bounds[0]
            ):
                raise IngressError("ingress_invalid", f"symbol {name!r} bounds are invalid")
        referenced = {
            dimension
            for row in self.inputs
            for dimension in row.shape
            if isinstance(dimension, str)
        }
        known = set(names)
        missing = sorted(referenced - known)
        if missing:
            raise IngressError(
                "ingress_invalid", f"symbol {missing[0]!r} has no declared bounds"
            )
        unused = sorted(known - referenced)
        if unused:
            raise IngressError("ingress_invalid", f"symbol {unused[0]!r} is unreferenced")

        if not isinstance(self.excluded_inputs, tuple) or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in self.excluded_inputs
        ):
            raise IngressError("ingress_invalid", "excluded_inputs must be canonical strings")
        if self.excluded_inputs != tuple(sorted(set(self.excluded_inputs))):
            raise IngressError("ingress_invalid", "excluded_inputs must be sorted and unique")
        overlap = sorted(set(self.excluded_inputs) & {row.name for row in self.inputs})
        if overlap:
            raise IngressError(
                "ingress_invalid", f"input {overlap[0]!r} is both declared and excluded"
            )

    @property
    def symbol_bounds(self) -> dict[str, tuple[int, int]]:
        return dict(self.symbols)

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": 1,
            "parameters": list(self.parameters),
            "flat_arity": self.flat_arity,
            "inputs": [row.as_dict() for row in self.inputs],
            "symbols": {name: list(bounds) for name, bounds in self.symbols},
            "excluded_inputs": list(self.excluded_inputs),
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()[:32]

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> CallIngress:
        if not isinstance(raw, Mapping) or set(raw) != _INGRESS_FIELDS:
            raise IngressError(
                "ingress_invalid",
                f"call ingress fields must be exactly {sorted(_INGRESS_FIELDS)!r}",
            )
        if type(raw.get("v")) is not int or raw.get("v") != 1:
            raise IngressError("ingress_invalid", "call ingress v must be 1")
        parameters = raw.get("parameters")
        if not isinstance(parameters, list):
            raise IngressError("ingress_invalid", "call ingress parameters must be an array")
        rows = raw.get("inputs")
        if not isinstance(rows, list):
            raise IngressError("ingress_invalid", "call ingress inputs must be an array")
        inputs: list[CallInput] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != _INPUT_FIELDS:
                raise IngressError(
                    "ingress_invalid",
                    f"call input {index} fields must be exactly {sorted(_INPUT_FIELDS)!r}",
                )
            raw_path = row.get("path")
            raw_shape = row.get("shape")
            if not isinstance(raw_path, list) or not isinstance(raw_shape, list):
                raise IngressError("ingress_invalid", f"call input {index} arrays are malformed")
            inputs.append(
                CallInput(
                    name=row["name"],
                    position=row["position"],
                    param=row["param"],
                    param_position=row["param_position"],
                    path=tuple(raw_path),
                    exported_name=row["exported_name"],
                    dtype=row["dtype"],
                    shape=tuple(raw_shape),
                )
            )
        raw_symbols = raw.get("symbols")
        if not isinstance(raw_symbols, Mapping):
            raise IngressError("ingress_invalid", "call ingress symbols must be an object")
        symbols: list[tuple[str, tuple[int, int]]] = []
        for name, bounds in raw_symbols.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise IngressError("ingress_invalid", f"symbol {name!r} bounds are malformed")
            symbols.append((name, (bounds[0], bounds[1])))
        excluded = raw.get("excluded_inputs")
        if not isinstance(excluded, list):
            raise IngressError("ingress_invalid", "excluded_inputs must be an array")
        return cls(
            parameters=tuple(parameters),
            flat_arity=raw["flat_arity"],
            inputs=tuple(inputs),
            symbols=tuple(symbols),
            excluded_inputs=tuple(excluded),
        )

    @classmethod
    def from_graph(cls, graph: Mapping[str, Any]) -> CallIngress:
        if "pytree" in graph:
            # tcg#55: v3 nested the ingress under a `pytree` object whose
            # other members (`in`, `out`) were read by nothing. Refuse the old
            # shape BY NAME rather than reaching into it -- these bytes key to
            # a graph class no v4 producer can derive.
            raise IngressError(
                "ingress_retired_format",
                "graph interface nests its ingress under the RETIRED v3 "
                "'pytree' object; tcg#55 promoted it to a top-level 'ingress'. "
                "Graph classes are content addressed -- re-derive.",
            )
        ingress = graph.get("ingress")
        if not isinstance(ingress, Mapping):
            raise IngressError("ingress_invalid", "graph interface declares no call ingress")
        return cls.decode(ingress)

    def bind(
        self, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> dict[str, object]:
        bound: dict[str, object] = {}
        for row in self.inputs:
            found, value = row.resolve(args, kwargs)
            if not found:
                path = "".join(f"[{step!r}]" for step in row.path)
                raise IngressError(
                    "input_missing",
                    f"declared input {row.name!r} at argument {row.param!r}{path} is absent",
                )
            bound[row.name] = value
        return bound

    def feeds(
        self, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> tuple[object, ...]:
        bound = self.bind(args, kwargs)
        return tuple(bound[row.name] for row in self.inputs)


@dataclass(frozen=True, slots=True)
class _Leaf:
    param: str
    param_position: int
    path: tuple[PathStep, ...]
    value: object

    @property
    def name(self) -> str:
        return _name(self.param, self.path)

    @property
    def exported_name(self) -> str:
        return exported_input_name(self.param, self.path)


def _walk(
    param: str,
    param_position: int,
    path: tuple[PathStep, ...],
    value: object,
    output: list[_Leaf],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key or key != key.strip():
                raise IngressError("call_invalid", "mapping input key is not canonical")
            _walk(param, param_position, path + (key,), item, output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(param, param_position, path + (index,), item, output)
    else:
        output.append(_Leaf(param, param_position, path, value))


def _flatten_call(
    param_names: Sequence[str], args: Sequence[object], kwargs: Mapping[str, object]
) -> tuple[_Leaf, ...]:
    names = tuple(param_names)
    if any(
        not isinstance(name, str) or not name or name != name.strip() for name in names
    ) or len(names) != len(set(names)):
        raise IngressError("call_invalid", "call parameter names must be canonical and unique")
    if len(args) > len(names):
        raise IngressError("call_invalid", "call has more positional values than parameters")
    values = list(args)
    for name in names[len(args) :]:
        if name not in kwargs:
            raise IngressError("call_invalid", f"call carries no value for parameter {name!r}")
        values.append(kwargs[name])
    output: list[_Leaf] = []
    for position, (param, value) in enumerate(zip(names, values, strict=True)):
        _walk(param, position, (), value, output)
    return tuple(output)


def _bound(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IngressError("range_invalid", f"symbol range bound {value!r} is not finite") from exc


def _symbol_ranges(program: object) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for symbol, interval in (getattr(program, "range_constraints", {}) or {}).items():
        lower = _bound(getattr(interval, "lower", None))
        upper = _bound(getattr(interval, "upper", None))
        ranges[str(symbol)] = (lower, upper)
    return ranges


def _placeholder_values(program: object) -> dict[str, object]:
    values: dict[str, object] = {}
    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") == "placeholder":
            values[str(node.name)] = node.meta.get("val")
    return values


def _dtype(value: object) -> str:
    raw = str(getattr(value, "dtype", "") or "")
    return raw.removeprefix("torch.")


def build_call_ingress(
    program: object,
    param_names: Sequence[str],
    args: Sequence[object],
    kwargs: Mapping[str, object],
    *,
    excluded_inputs: Sequence[str] = (),
) -> CallIngress:
    """Build the sole v1 ingress declaration from a real exported call."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise IngressError("torch_missing", "building call ingress requires PyTorch") from exc
    if not isinstance(program, torch.export.ExportedProgram):
        raise IngressError("program_invalid", "call ingress requires an ExportedProgram")
    leaves = _flatten_call(param_names, args, kwargs)
    signature = getattr(program, "graph_signature", None)
    user_inputs = tuple(getattr(signature, "user_inputs", ()) or ())
    if len(user_inputs) != len(leaves):
        raise IngressError(
            "call_invalid",
            f"export records {len(user_inputs)} flat inputs but the call has {len(leaves)} leaves",
        )
    placeholders = _placeholder_values(program)
    ranges = _symbol_ranges(program)
    used_symbols: dict[str, tuple[int, int]] = {}
    rows: list[CallInput] = []
    for position, (leaf, exported) in enumerate(zip(leaves, user_inputs, strict=True)):
        if not isinstance(exported, str):
            continue
        value = placeholders.get(exported)
        dtype = _dtype(value)
        shape_value = getattr(value, "shape", None)
        if not dtype or shape_value is None:
            continue
        if exported != leaf.exported_name:
            raise IngressError(
                "call_invalid",
                f"torch.export input {exported!r} does not match call leaf {leaf.exported_name!r}",
            )
        shape: list[int | str] = []
        for dimension in shape_value:
            text = str(dimension)
            if text.lstrip("-").isdigit():
                shape.append(int(text))
                continue
            bounds = ranges.get(text)
            if bounds is None:
                raise IngressError(
                    "range_invalid", f"input {leaf.name!r} symbol {text!r} has no finite range"
                )
            used_symbols[text] = bounds
            shape.append(text)
        rows.append(
            CallInput(
                name=leaf.name,
                position=position,
                param=leaf.param,
                param_position=leaf.param_position,
                path=leaf.path,
                exported_name=leaf.exported_name,
                dtype=dtype,
                shape=tuple(shape),
            )
        )
    return CallIngress(
        parameters=tuple(param_names),
        flat_arity=len(leaves),
        inputs=tuple(rows),
        symbols=tuple(sorted(used_symbols.items())),
        excluded_inputs=tuple(sorted(excluded_inputs)),
    )


__all__ = [
    "CallIngress",
    "CallInput",
    "IngressError",
    "PathStep",
    "build_call_ingress",
    "exported_input_name",
]
