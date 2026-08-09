"""``CompileAxis`` — compile-graph equivalence classes on PAYLOAD FIELDS
(SDK v2, pgw#647).

What selects a compiled graph is a payload field's equivalence CLASS, not
its value: CFG==0 traces a batch-1 UNet graph, CFG!=0 traces batch-2
(cond+uncond). v2 therefore annotates the FIELD with its partition instead
of duplicating a value list onto the decorator
(``Compile(guidance_scales=...)`` is deleted)::

    class In(msgspec.Struct):
        guidance_scale: Annotated[float, CompileAxis(classes=(
            AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
            AxisClass("cfg_on",  match=lambda v: v != 0, warm=5.0),
        ))] = 5.0
        aspect_ratio: Annotated[AspectRatio, CompileAxis(classes="enum")]

Each class carries a WARM representative value, which is what makes the
warm plan DERIVABLE (cross-product of axis classes x shape buckets): the
forge and the boot warmup call the pipeline once per class with ``warm``,
and "fully warmed?" becomes computable. Catalog recipes validate against
the declared class NAMES at publish time (the manifest carries names, not
lambdas).

``classes="enum"`` derives the partition from the field's type: every
member of a ``Literal[...]``/``StringEnum`` field is its own class, warm ==
the member value (``aspect_ratio`` was always such an axis; now it is
labeled as one).
"""

from __future__ import annotations

import types as py_types
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class AxisClass:
    """One equivalence class of a :class:`CompileAxis` partition.

    ``match`` decides membership; ``warm`` is the representative value warm
    calls use to trace this class's graph. ``warm`` must itself satisfy
    ``match`` (validated at construction).
    """

    __slots__ = ("name", "match", "warm")

    def __init__(self, name: str, *, match: Callable[[Any], bool], warm: Any) -> None:
        n = str(name or "").strip()
        if not n or not n.replace("-", "_").isidentifier():
            raise ValueError(f"AxisClass name {name!r} must be an identifier-like token")
        if not callable(match):
            raise TypeError(f"AxisClass {n!r}: match must be callable")
        try:
            ok = bool(match(warm))
        except Exception as exc:
            raise ValueError(
                f"AxisClass {n!r}: match({warm!r}) raised {exc!r}"
            ) from exc
        if not ok:
            raise ValueError(
                f"AxisClass {n!r}: warm value {warm!r} does not satisfy its own match"
            )
        self.name = n
        self.match = match
        self.warm = warm

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"AxisClass({self.name!r}, warm={self.warm!r})"


class CompileAxis:
    """Payload-field annotation declaring the field as a compile-graph axis.

    ``classes`` is either a tuple/list of :class:`AxisClass` (an explicit
    partition with warm representatives) or the string ``"enum"`` (partition
    = the field type's enumerated members — ``Literal[...]`` / a str-enum).
    """

    __slots__ = ("classes",)

    def __init__(self, classes: Union[str, Sequence[AxisClass]]) -> None:
        if isinstance(classes, str):
            if classes != "enum":
                raise ValueError(
                    f'CompileAxis(classes=...) string form must be "enum", got {classes!r}'
                )
            self.classes: Union[str, Tuple[AxisClass, ...]] = "enum"
            return
        rows = tuple(classes)
        if not rows or not all(isinstance(c, AxisClass) for c in rows):
            raise TypeError(
                "CompileAxis(classes=...) must be a non-empty sequence of "
                'AxisClass rows or "enum"'
            )
        names = [c.name for c in rows]
        if len(set(names)) != len(names):
            raise ValueError(f"CompileAxis classes repeat a name: {names!r}")
        self.classes = rows

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CompileAxis(classes={self.classes!r})"


class PayloadAxis:
    """One RESOLVED compile axis on a concrete payload type: the field name,
    the class names, and the warm representative per class — extracted by
    the registry walk (:func:`extract_payload_axes`) and carried on the
    ``EndpointSpec``. Lambdas stay in code; the manifest carries the
    projection (:meth:`to_manifest`)."""

    __slots__ = ("field", "class_names", "warm_values", "_matchers")

    def __init__(
        self,
        field: str,
        class_names: Tuple[str, ...],
        warm_values: Tuple[Any, ...],
        matchers: Tuple[Optional[Callable[[Any], bool]], ...],
    ) -> None:
        self.field = field
        self.class_names = class_names
        self.warm_values = warm_values
        self._matchers = matchers

    def classify(self, value: Any) -> Optional[str]:
        """The class name ``value`` falls in, or None (outside the envelope)."""
        for name, warm, match in zip(self.class_names, self.warm_values, self._matchers):
            if match is not None:
                try:
                    if match(value):
                        return name
                except Exception:
                    continue
            elif value == warm:
                return name
        return None

    def to_manifest(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "field": self.field,
            "classes": list(self.class_names),
        }
        warm: List[Any] = []
        for v in self.warm_values:
            warm.append(v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
        out["warm"] = warm
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"PayloadAxis({self.field!r}, classes={self.class_names!r})"


def _enum_members(field: str, ann: Any) -> Tuple[Any, ...]:
    """Enumerated members of a Literal/str-enum annotation, for
    ``classes="enum"`` axes."""
    origin = typing.get_origin(ann)
    if origin is typing.Literal:
        return tuple(typing.get_args(ann))
    if origin in (typing.Union, py_types.UnionType):
        members: List[Any] = []
        for arg in typing.get_args(ann):
            if arg is type(None):
                continue
            members.extend(_enum_members(field, arg))
        return tuple(members)
    if isinstance(ann, type):
        try:
            import enum

            if issubclass(ann, enum.Enum):
                return tuple(m.value for m in ann)
        except TypeError:
            pass
        # StringEnum-style: class attributes of type str on a str subclass.
        if issubclass(ann, str):
            vals = tuple(
                v for k, v in vars(ann).items()
                if not k.startswith("_") and isinstance(v, str)
            )
            if vals:
                return vals
    raise ValueError(
        f'CompileAxis(classes="enum") on field {field!r} requires a '
        f"Literal[...] / enum-typed annotation, got {ann!r}"
    )


def extract_payload_axes(owner: str, payload_type: type) -> Tuple[PayloadAxis, ...]:
    """All :class:`CompileAxis` annotations on ``payload_type``'s fields,
    resolved into :class:`PayloadAxis` rows (declaration order)."""
    try:
        hints = typing.get_type_hints(payload_type, include_extras=True)
    except Exception:
        hints = getattr(payload_type, "__annotations__", {}) or {}
    out: List[PayloadAxis] = []
    for field, ann in hints.items():
        if typing.get_origin(ann) is not typing.Annotated:
            continue
        args = typing.get_args(ann)
        base_ann, extras = args[0], args[1:]
        axes = [e for e in extras if isinstance(e, CompileAxis)]
        if not axes:
            continue
        if len(axes) > 1:
            raise ValueError(
                f"{owner}: payload field {field!r} declares more than one CompileAxis"
            )
        axis = axes[0]
        if axis.classes == "enum":
            members = _enum_members(field, base_ann)
            out.append(PayloadAxis(
                field=field,
                class_names=tuple(str(m) for m in members),
                warm_values=tuple(members),
                matchers=tuple(None for _ in members),
            ))
        else:
            rows = axis.classes
            assert not isinstance(rows, str)
            out.append(PayloadAxis(
                field=field,
                class_names=tuple(c.name for c in rows),
                warm_values=tuple(c.warm for c in rows),
                matchers=tuple(c.match for c in rows),
            ))
    return tuple(out)


def warm_guidance_values(axes: Sequence[PayloadAxis]) -> Tuple[float, ...]:
    """The warm representatives of the guidance axis, when one is declared.

    Compile warm calls key CFG graph classes off the ``guidance_scale`` /
    ``guidance`` / ``cfg`` payload field's declared classes — the v2
    replacement for the deleted ``Compile(guidance_scales=...)`` tuple.
    """
    for axis in axes:
        if axis.field in ("guidance_scale", "guidance", "cfg", "true_cfg_scale"):
            vals: List[float] = []
            for v in axis.warm_values:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return tuple(vals)
    return ()


__all__ = [
    "AxisClass",
    "CompileAxis",
    "PayloadAxis",
    "extract_payload_axes",
    "warm_guidance_values",
]
