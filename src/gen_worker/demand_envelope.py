"""Request → :class:`~gen_worker.demand.RequestShape`, and the ADVERTISED
envelope a lane's worst case is evaluated at.

pgw#1600. Two questions, one vocabulary:

* **at SERVE time** — what shape is THIS request? :func:`request_shape`.
* **at LOCK time** — what is the largest shape the API can be asked for?
  :func:`advertised_envelope`, evaluated at derive so the release document
  carries a worst case a Go reader can add manifest weight bytes to.

The platform reads a payload field as a shape axis when the field NAMES one
(`width`, `height`, `num_frames`, `num_images`, …) — no author edit needed for
the common case. :class:`Shape` exists for what only the author knows: a field
that dimensions the request without saying so numerically, most importantly an
ASPECT-RATIO ENUM whose members map to a bucket table the platform cannot see.
The annotation carries the author's OWN table by reference, so the pixels are
declared exactly once, in the dict the handler itself indexes.

**Every axis in a derived envelope states its SOURCE**, and an axis nothing
advertises is reported as `default` with the reason. That is deliberate: an
envelope is a claim about the API's ceiling, a defaulted axis is a claim the
API does not support, and a lane whose real launches exceed its envelope is
exactly what pgw#1600's `demand_miss` falsifier is built to catch. The
envelope tells the truth about what it knows; the falsifier catches the rest.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import msgspec

from .demand import SHAPE_BOUNDS, Demand, RequestShape

__all__ = [
    "AXIS_FIELD_NAMES",
    "EnvelopeAxis",
    "Envelope",
    "Shape",
    "ShapeDeclarationError",
    "advertised_envelope",
    "demand_document",
    "request_shape",
]


class ShapeDeclarationError(TypeError):
    """A `Shape(...)` annotation is not one the platform can read."""


#: The CLOSED map from a payload FIELD NAME to a shape axis. Spellings are
#: the ones the fleet's payloads already use; a new spelling is a change
#: here, never a per-endpoint annotation, because two names for one axis is
#: how an endpoint ends up measured against an envelope that does not
#: describe it.
AXIS_FIELD_NAMES: dict[str, str] = {
    "width": "width",
    "height": "height",
    "num_frames": "frames",
    "frames": "frames",
    "num_images": "batch",
    "batch_size": "batch",
    "batch": "batch",
    "latent_tokens": "latent_tokens",
}


class Shape(msgspec.Struct, frozen=True):
    """Payload-field annotation: this field DIMENSIONS the request.

    Two arms, exactly one of which is used:

    * ``Shape("width")`` — this numeric field IS that axis, under a name the
      platform vocabulary does not carry.
    * ``Shape(pixels=_BUCKETS)`` — this field's VALUES map to ``(width,
      height)``. Pass the author's own table by reference; do not retype the
      numbers, or the endpoint gains a second spelling of its own geometry.

    Read via ``Annotated[...]``, the same way ``PromptRole`` and
    ``ExpectedOutput`` are.
    """

    axis: str = ""
    pixels: Optional[Mapping[Any, tuple[int, int]]] = None

    def __post_init__(self) -> None:
        named = bool(self.axis)
        tabled = self.pixels is not None
        if named == tabled:
            raise ShapeDeclarationError(
                "Shape(...) states EITHER an axis name "
                "(`Shape('width')`) OR a value->(width, height) table "
                "(`Shape(pixels=_BUCKETS)`) — never both and never neither"
            )
        if named and self.axis not in SHAPE_BOUNDS:
            raise ShapeDeclarationError(
                f"Shape({self.axis!r}): the shape axes are CLOSED and are "
                f"{sorted(SHAPE_BOUNDS)}"
            )
        if tabled:
            table = dict(self.pixels or {})
            if not table:
                raise ShapeDeclarationError(
                    "Shape(pixels=): the table is empty; a field that "
                    "dimensions nothing is not a shape field"
                )
            for key, value in table.items():
                if (
                    not isinstance(value, Sequence)
                    or isinstance(value, (str, bytes))
                    or len(value) != 2
                    or not all(isinstance(n, int) and n > 0 for n in value)
                ):
                    raise ShapeDeclarationError(
                        f"Shape(pixels=): {key!r} maps to {value!r}; each "
                        f"value is a (width, height) pair of positive ints"
                    )


class EnvelopeAxis(msgspec.Struct, frozen=True, kw_only=True):
    """One axis of a derived envelope, with WHERE its number came from."""

    value: int
    #: ``advertised`` — a bounded payload field stated it.
    #: ``declared``   — a ``Shape(...)`` table stated it.
    #: ``default``    — NOTHING states it; see ``why``.
    source: str
    why: str = ""

    def as_document(self) -> dict[str, Any]:
        row: dict[str, Any] = {"value": self.value, "source": self.source}
        if self.why:
            row["why"] = self.why
        return row


class Envelope(msgspec.Struct, frozen=True, kw_only=True):
    """The advertised ceiling, per axis, and the shape that maximises demand."""

    axes: dict[str, EnvelopeAxis]
    #: The candidate shapes the ceiling is taken over. More than one only
    #: when a ``Shape(pixels=)`` table couples width and height — the max of
    #: ``width*height`` is not ``max(width)*max(height)``.
    candidates: tuple[RequestShape, ...]

    @property
    def advertised_axes(self) -> tuple[str, ...]:
        return tuple(
            name for name, axis in self.axes.items() if axis.source != "default"
        )

    def worst_case(self, demand: Demand) -> tuple[int, RequestShape]:
        """``(bytes, shape)`` — the candidate that maximises the formula.

        Every coefficient is non-negative by construction, so the formula is
        monotone in every axis and the per-axis ceiling is the right one to
        take; the argmax over candidates exists only for the COUPLED case.
        """

        def rank(shape: RequestShape) -> tuple[int, int, int, int]:
            # The TIE-BREAK is not cosmetic. A const-only formula evaluates
            # identically at every bucket, and "whichever the set iterated
            # first" would make the recorded worst_case_shape depend on dict
            # ordering — a document field that moves without the formula
            # moving. Largest area wins, then width, then height.
            return (
                demand.evaluate(shape),
                shape.width * shape.height,
                shape.width,
                shape.height,
            )

        best = max(self.candidates, key=rank)
        return demand.evaluate(best), best

    def as_document(self) -> dict[str, Any]:
        return {name: axis.as_document() for name, axis in self.axes.items()}


def _annotations(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """``(bare type, extras)`` for a possibly-``Annotated`` annotation."""

    if typing.get_origin(annotation) is typing.Annotated:
        args = typing.get_args(annotation)
        return args[0], tuple(args[1:])
    return annotation, ()


def _shape_marker(extras: Sequence[Any]) -> Optional[Shape]:
    for extra in extras:
        if isinstance(extra, Shape):
            return extra
    return None


def _upper_bound(annotation: Any, extras: Sequence[Any]) -> Optional[int]:
    """The advertised ceiling of a numeric field, from ``msgspec.Meta``."""

    for extra in list(extras) + list(_annotations(annotation)[1]):
        if not isinstance(extra, msgspec.Meta):
            continue
        if extra.le is not None:
            return int(extra.le)
        if extra.lt is not None:
            return int(extra.lt) - 1
    return None


def _fields(payload_type: Any) -> tuple[tuple[str, Any], ...]:
    try:
        info = msgspec.inspect.type_info(payload_type)
    except Exception:  # noqa: BLE001 — a payload we cannot introspect has no envelope
        return ()
    hints = typing.get_type_hints(payload_type, include_extras=True)
    return tuple(
        (field.name, hints.get(field.name))
        for field in getattr(info, "fields", ())
    )


def request_shape(payload: Any) -> RequestShape:
    """THIS request's shape, read off a decoded payload instance.

    Unknown fields contribute nothing. A payload that advertises no shape
    axis at all yields the zero shape, on which every non-``const`` term
    evaluates to zero — which is the honest reading of "this request has no
    such axis", not a silent floor.
    """

    values: dict[str, int] = {}
    for name, annotation in _fields(type(payload)):
        held = getattr(payload, name, None)
        if held is None:
            continue
        bare, extras = _annotations(annotation)
        marker = _shape_marker(extras)
        if marker is not None and marker.pixels is not None:
            pair = dict(marker.pixels).get(held)
            if pair is not None:
                values["width"] = int(pair[0])
                values["height"] = int(pair[1])
            continue
        axis = marker.axis if marker is not None else AXIS_FIELD_NAMES.get(name, "")
        if not axis:
            continue
        if isinstance(held, bool) or not isinstance(held, int):
            continue
        values[axis] = int(held)
    return RequestShape(
        width=values.get("width", 0),
        height=values.get("height", 0),
        batch=values.get("batch", 1),
        frames=values.get("frames", 0),
        latent_tokens=values.get("latent_tokens", 0),
    )


#: What an axis nothing advertises is assumed to be, and why the assumption is
#: stated rather than buried. ``batch`` is 1 because a launch that batches — an
#: SDXL CFG pair, say — is a decision the CHECKPOINT's config makes, not the
#: request, so no payload field can advertise it; the envelope says so and
#: `demand_miss` is what catches a lane whose launches exceed it.
_DEFAULT_AXES: dict[str, tuple[int, str]] = {
    "width": (0, "no advertised or declared width; every width-scaled term "
                 "evaluates to zero at this envelope"),
    "height": (0, "no advertised or declared height; every height-scaled "
                  "term evaluates to zero at this envelope"),
    "batch": (1, "no advertised or declared batch axis. A launch that "
                 "batches (a CFG pair) exceeds this envelope; that is what "
                 "pgw#1600's demand_miss counts"),
    "frames": (0, "not a video lane, or no advertised frame count"),
    "latent_tokens": (0, "no advertised latent sequence length"),
}


def advertised_envelope(*payload_types: Any) -> Envelope:
    """The ceiling of everything these entrypoints advertise.

    One envelope per LANE, taken over every entrypoint that lane serves: the
    worst case a pod must be bought for is the worst case of the whole
    advertised surface, not of one function.
    """

    ceilings: dict[str, EnvelopeAxis] = {}
    buckets: list[tuple[int, int]] = []

    def raise_axis(axis: str, value: int, source: str, why: str = "") -> None:
        held = ceilings.get(axis)
        if held is None or value > held.value:
            ceilings[axis] = EnvelopeAxis(value=value, source=source, why=why)

    for payload_type in payload_types:
        for name, annotation in _fields(payload_type):
            bare, extras = _annotations(annotation)
            marker = _shape_marker(extras)
            if marker is not None and marker.pixels is not None:
                buckets.extend(
                    (int(w), int(h)) for w, h in dict(marker.pixels).values()
                )
                continue
            axis = marker.axis if marker is not None else AXIS_FIELD_NAMES.get(name, "")
            if not axis:
                continue
            bound = _upper_bound(bare, extras)
            if bound is None:
                raise_axis(
                    axis, 0, "default",
                    f"payload field {name!r} carries this axis but advertises "
                    f"NO upper bound (no msgspec.Meta le=/lt=), so the API's "
                    f"ceiling on it is unstated and no worst case can be "
                    f"taken over it",
                )
                continue
            raise_axis(axis, bound, "advertised", f"payload field {name!r}")

    axes: dict[str, EnvelopeAxis] = {}
    for axis in SHAPE_BOUNDS:
        held = ceilings.get(axis)
        if held is not None and held.source == "advertised":
            axes[axis] = held
            continue
        value, why = _DEFAULT_AXES[axis]
        axes[axis] = EnvelopeAxis(
            value=value, source="default",
            why=(held.why if held is not None else why),
        )

    if buckets:
        widest = max(w for w, _ in buckets)
        tallest = max(h for _, h in buckets)
        axes["width"] = EnvelopeAxis(
            value=widest, source="declared",
            why="Shape(pixels=) table on the payload's own bucket dict",
        )
        axes["height"] = EnvelopeAxis(
            value=tallest, source="declared",
            why="Shape(pixels=) table on the payload's own bucket dict",
        )
        candidates = tuple(
            RequestShape(
                width=w, height=h,
                batch=axes["batch"].value,
                frames=axes["frames"].value,
                latent_tokens=axes["latent_tokens"].value,
            )
            for w, h in sorted(set(buckets))
        )
    else:
        candidates = (
            RequestShape(
                width=axes["width"].value,
                height=axes["height"].value,
                batch=axes["batch"].value,
                frames=axes["frames"].value,
                latent_tokens=axes["latent_tokens"].value,
            ),
        )
    return Envelope(axes=axes, candidates=candidates)


def demand_document(demand: Demand, envelope: Envelope) -> dict[str, Any]:
    """The lane's whole `demand` block for the release document.

    THE NUMBER A READER FINDS IS DERIVED, never written down: the formula's
    terms, the envelope each axis of which states its own source, the shape
    inside that envelope which maximises the formula, and the bytes. A
    non-Python reader adds tensorfs manifest weight bytes to
    ``worst_case_request_bytes`` and has the pod-buy worst case (pgw#1598 §2,
    pgw#1600 acceptance (b)) — nothing here is the whole answer on its own,
    and the block says so.
    """

    row = demand.formula_document()
    worst_bytes, worst_shape = envelope.worst_case(demand)
    row["envelope"] = envelope.as_document()
    row["worst_case_shape"] = worst_shape.as_document()
    row["worst_case_request_bytes"] = worst_bytes
    row["envelope_advertised_axes"] = list(envelope.advertised_axes)
    return row
