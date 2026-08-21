"""The demand formula is DATA: `C + Σ cᵢ·termᵢ` over a CLOSED vocabulary.

pgw#1598 §2 / pgw#1600. Three properties the rest of the platform depends on,
and each one is a design constraint rather than a nicety:

* **DATA, never Python.** An author cannot invent a term because the
  constructors below are the only way to make one, and the serialized form is
  a list of `(name, coefficient)` rows. That is what lets a NON-PYTHON reader
  — tensorhub, in Go, at pod-buy time — evaluate the same formula and get the
  same bytes without importing anything of ours.
* **EXACT INTEGER ARITHMETIC.** Every term's value is an integer numerator
  over a fixed integer scale, and evaluation is floor division. `megapixels`
  is `width*height` over 1_000_000, not a float divide — a float divide is
  where two languages stop agreeing, and the whole point of the serialization
  is that they agree. :func:`evaluate` and its Go twin are asserted equal over
  a shared conformance corpus (`gen_worker/contracts/demand_vectors.json`,
  pinned into tensorhub through its `peers.lock`).
* **EVERY COEFFICIENT CARRIES ITS PROVENANCE.** A byte count with no basis is
  a magic constant. :class:`Basis` has exactly three values, the default is
  the honest one (`UNCALIBRATED` — a declared prior, never measured), and
  claiming `MEASURED` or `LEDGER` without naming the source is a refusal.

The WEIGHT arena is not in here and never will be: its bytes are tensorfs
manifest arithmetic plus varena's alignment tax, exact and $0 (pgw#1598 §1).
This module is the REQUEST arena only.

Payload → :class:`RequestShape` extraction lives in
:mod:`gen_worker.demand_envelope`; banking predicted-vs-measured lives in
:mod:`gen_worker.demand_falsifier`. Nothing in any of the three consumes the
number for an admission decision — see pgw#1600 acceptance (d) and
`tests/test_demand_no_enforcement_pgw1600.py`, which asserts that absence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Union

import msgspec

__all__ = [
    "Basis",
    "Demand",
    "DemandDeclarationError",
    "DemandEvaluationError",
    "GiB",
    "KiB",
    "MAX_RESULT_BYTES",
    "MiB",
    "RequestShape",
    "SHAPE_BOUNDS",
    "TERM_VOCABULARY",
    "Term",
    "TermSpec",
    "VOCABULARY_VERSION",
    "const",
    "evaluate",
    "per_batch",
    "per_frame",
    "per_frame_squared",
    "per_latent_token",
    "per_mp",
    "per_mp_batch",
    "term_value",
    "vocabulary_document",
]


class DemandDeclarationError(TypeError):
    """A demand formula is not one the platform can evaluate."""


class DemandEvaluationError(ValueError):
    """A request shape is outside the domain the formula is defined on."""


def KiB(count: float) -> int:
    """Bytes."""

    return int(count * 1024)


def MiB(count: float) -> int:
    return int(count * 1024 * 1024)


def GiB(count: float) -> int:
    return int(count * 1024 * 1024 * 1024)


#: Bumped when a TERM is added, removed, or its scale/inputs change. A Go
#: evaluator reading a document with a version it does not implement REFUSES
#: rather than evaluating the subset it understands — a formula evaluated
#: without one of its terms is not a conservative answer, it is a low one.
VOCABULARY_VERSION = 1


class Basis(StrEnum):
    """Where a coefficient's NUMBER came from. There is no fourth answer.

    ``MEASURED``  — a SUCCESSFUL run on real hardware produced it, and
    ``source`` names the run. Never a death trace: a death reports only the
    free memory it consumed, which is a lower bound on nothing useful
    (pgw#1601's stamp-source rule, falsified on the card 2026-08-21).

    ``LEDGER``    — fitted from banked samples (pgw#1586's calibrator).
    ``source`` names the ledger key and the sample count.

    ``UNCALIBRATED`` — a declared prior. The author stated the formula's
    SHAPE and guessed the split; nothing has measured it. This is the DEFAULT
    because it is the only claim a bare number can support, and a lane that
    reads ``uncalibrated`` at the hub is telling the truth about itself.
    """

    MEASURED = "measured"
    LEDGER = "ledger"
    UNCALIBRATED = "uncalibrated"


class TermSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One vocabulary entry: how its integer VALUE is built, and its scale.

    ``scale`` is the denominator the coefficient is quoted against, so a
    coefficient is always "bytes per WHOLE unit" while the value stays an
    exact integer: `megapixels` has value ``width*height`` and scale
    ``1_000_000``, i.e. the coefficient is bytes per megapixel and no float
    ever appears. ``inputs`` names the :class:`RequestShape` fields the value
    reads, so a document reader can check it implements the same term rather
    than a same-named one.
    """

    scale: int
    inputs: tuple[str, ...]
    doc: str


TERM_VOCABULARY: dict[str, TermSpec] = {
    "const": TermSpec(
        scale=1, inputs=(),
        doc="a fixed floor, independent of the request",
    ),
    "megapixels": TermSpec(
        scale=1_000_000, inputs=("width", "height"),
        doc="output width x height / 1e6",
    ),
    "mp_batch": TermSpec(
        scale=1_000_000, inputs=("width", "height", "batch"),
        doc="megapixels x batch (the CFG pair counts as 2)",
    ),
    "batch": TermSpec(
        scale=1, inputs=("batch",),
        doc="the launch batch size (2 with CFG, 1 without)",
    ),
    "frames": TermSpec(
        scale=1, inputs=("frames",),
        doc="the video frame count",
    ),
    "frames_squared": TermSpec(
        scale=1, inputs=("frames",),
        doc="frame count SQUARED — the quadratic attention term",
    ),
    "latent_tokens": TermSpec(
        scale=1, inputs=("latent_tokens",),
        doc="the denoiser's latent sequence length",
    ),
}

_TERM_ORDER = {name: index for index, name in enumerate(TERM_VOCABULARY)}


#: The domain each shape scalar is defined on — the largest request the API
#: could conceivably describe, an order of magnitude past anything real. A
#: shape outside it is REFUSED rather than evaluated, in both languages.
SHAPE_BOUNDS: dict[str, int] = {
    "width": 1 << 16,
    "height": 1 << 16,
    "batch": 1 << 12,
    "frames": 1 << 20,
    "latent_tokens": 1 << 31,
}

#: The result must fit a signed 64-bit integer, because the OTHER evaluator is
#: Go and that is the widest integer the wire form can carry.
#:
#: ⚠️ THIS BOUND IS ON THE RESULT, NOT ON THE INTERMEDIATES, AND THE
#: DIFFERENCE WAS MEASURED RATHER THAN REASONED. The first version bounded the
#: shape so that a split division (`c = q*scale + r` → `q*v + (r*v)//scale`)
#: could not overflow int64 — and the cross-language corpus went RED on its own
#: ceiling row the first time Go ran it: at `mp_batch` with a 1 GiB
#: coefficient, `r*v` is 1.3e19, past int64, and Python's bignums hid it
#: entirely. Python could not have found this; only the Go side could, which is
#: the entire argument for the corpus. So the arithmetic is now EXACT in both
#: languages (Python natively, Go through `math/big`) and the only shared
#: constraint is that the ANSWER is representable. A formula whose worst case
#: does not fit int64 is refused loudly rather than wrapped into a small
#: number — small is the direction that buys too little card.
MAX_RESULT_BYTES = (1 << 63) - 1


class RequestShape(msgspec.Struct, frozen=True, kw_only=True):
    """The request-derived scalars every term is a function of.

    This is the WHOLE input surface of the algebra. A term that needs
    something not in here is a vocabulary change, which is a
    :data:`VOCABULARY_VERSION` bump and a Go-side change in the same breath.
    Zero means "this request has no such axis" (an image model has no
    ``frames``), and a term that reads a zero contributes zero.
    """

    width: int = 0
    height: int = 0
    batch: int = 1
    frames: int = 0
    latent_tokens: int = 0

    def as_document(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "batch": self.batch,
            "frames": self.frames,
            "latent_tokens": self.latent_tokens,
        }

    def validated(self) -> "RequestShape":
        for field, ceiling in SHAPE_BOUNDS.items():
            value = int(getattr(self, field))
            if value < 0:
                raise DemandEvaluationError(
                    f"request shape {field}={value} is negative; the algebra "
                    f"is defined on non-negative integers only"
                )
            if value > ceiling:
                raise DemandEvaluationError(
                    f"request shape {field}={value} exceeds the declared "
                    f"domain ({ceiling}). The evaluation is specified exact "
                    f"in 64-bit integers in both Python and Go; past this "
                    f"bound it would not be, and a wrapped worst case is a "
                    f"SMALL number — the direction that buys too little card."
                )
        return self


def term_value(name: str, shape: RequestShape) -> int:
    """The exact integer NUMERATOR of one term at ``shape``.

    Kept as one switch so the Go twin has one thing to mirror. Every branch
    is integer multiplication of bounded non-negative ints.
    """

    if name == "const":
        return 1
    if name == "megapixels":
        return shape.width * shape.height
    if name == "mp_batch":
        return shape.width * shape.height * shape.batch
    if name == "batch":
        return shape.batch
    if name == "frames":
        return shape.frames
    if name == "frames_squared":
        return shape.frames * shape.frames
    if name == "latent_tokens":
        return shape.latent_tokens
    raise DemandEvaluationError(
        f"unknown demand term {name!r}; the vocabulary is CLOSED and is "
        f"{sorted(TERM_VOCABULARY)}"
    )


def _scaled(coefficient: int, value: int, scale: int) -> int:
    """``floor(coefficient * value / scale)``, EXACT.

    Python computes this natively; Go computes it in ``math/big``. Neither
    tries to be clever about the intermediate, because the clever version was
    wrong: a split division (``q*v + (r*v)//scale``) keeps ``q*v`` small and
    lets ``r*v`` reach 1.3e19 at the domain ceiling — past int64 — and Python's
    bignums hid that completely. The cross-language corpus caught it on Go's
    first run. See :data:`MAX_RESULT_BYTES`.
    """

    return (coefficient * value) // scale


def _representable(total: int) -> int:
    """Refuse a result the other evaluator could not carry."""

    if total > MAX_RESULT_BYTES:
        raise DemandEvaluationError(
            f"the formula evaluates to {total} bytes, past the largest signed "
            f"64-bit integer ({MAX_RESULT_BYTES}). The other evaluator is Go "
            f"and cannot carry this; a wrapped answer would be a SMALL number, "
            f"which is the direction that buys too little card. Either the "
            f"coefficients or the envelope are wrong by orders of magnitude."
        )
    return total


class Term(msgspec.Struct, frozen=True, kw_only=True):
    """One `cᵢ·termᵢ`: a vocabulary name, its coefficient IN BYTES, its basis."""

    name: str
    coefficient: int
    basis: Basis = Basis.UNCALIBRATED
    source: str = ""

    def evaluate(self, shape: RequestShape) -> int:
        spec = TERM_VOCABULARY.get(self.name)
        if spec is None:
            raise DemandEvaluationError(
                f"unknown demand term {self.name!r}; the vocabulary is CLOSED"
            )
        return _scaled(self.coefficient, term_value(self.name, shape), spec.scale)

    def as_document(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "term": self.name,
            "bytes": self.coefficient,
            "basis": str(self.basis),
        }
        if self.source:
            row["source"] = self.source
        return row


class Demand(msgspec.Struct, frozen=True, kw_only=True):
    """A declared demand formula: `Σ terms`, in bytes."""

    terms: tuple[Term, ...]

    def __add__(self, other: "Demand") -> "Demand":
        if not isinstance(other, Demand):
            raise DemandDeclarationError(
                f"a demand formula adds to another demand formula, not to "
                f"{type(other).__name__}. Wrap a plain byte count in "
                f"`const(...)`."
            )
        merged: dict[str, Term] = {}
        for term in self.terms + other.terms:
            held = merged.get(term.name)
            if held is None:
                merged[term.name] = term
                continue
            if held.basis is not term.basis:
                raise DemandDeclarationError(
                    f"two {term.name!r} terms with DIFFERENT provenance "
                    f"({held.basis} and {term.basis}) cannot be summed into "
                    f"one coefficient — the sum would inherit one of the two "
                    f"claims and silently launder the other. Declare the term "
                    f"once, with the basis the whole number has."
                )
            sources = [s for s in (held.source, term.source) if s]
            merged[term.name] = Term(
                name=term.name,
                coefficient=held.coefficient + term.coefficient,
                basis=term.basis,
                source="; ".join(dict.fromkeys(sources)),
            )
        return Demand(
            terms=tuple(
                term for _, term in sorted(
                    merged.items(), key=lambda item: _TERM_ORDER[item[0]]
                )
            )
        )

    def coefficients(self) -> dict[str, int]:

        return {term.name: term.coefficient for term in self.terms}

    def bases(self) -> dict[str, Basis]:
        """Per-term provenance. A formula is only as calibrated as its worst
        term, which is why this is a map and not a single verdict."""

        return {term.name: term.basis for term in self.terms}

    def weakest_basis(self) -> Basis:
        """The claim the WHOLE formula can support."""

        held = {term.basis for term in self.terms}
        if Basis.UNCALIBRATED in held or not held:
            return Basis.UNCALIBRATED
        if Basis.LEDGER in held:
            return Basis.LEDGER
        return Basis.MEASURED

    def evaluate(self, shape: RequestShape) -> int:
        """Bytes of REQUEST arena at ``shape``. $0, CPU, deterministic."""

        checked = shape.validated()
        return _representable(sum(term.evaluate(checked) for term in self.terms))

    def as_document(self) -> list[dict[str, Any]]:
        """The release-document shape of the TERMS."""

        return [term.as_document() for term in self.terms]

    def formula_document(self) -> dict[str, Any]:
        """The whole serialized formula, evaluable by a non-Python reader.

        Carries the VOCABULARY as well as the terms. A Go reader checks each
        term's scale and inputs against its own table and refuses on a
        mismatch or an unknown name — so a vocabulary that moves in one
        language and not the other is a loud refusal instead of a number that
        is wrong by exactly one term.
        """

        row: dict[str, Any] = {
            "algebra": "C + sum(coefficient_i * value_i / scale_i), floor, int64",
            "vocabulary_version": VOCABULARY_VERSION,
            "vocabulary": vocabulary_document(),
            "shape_bounds": dict(SHAPE_BOUNDS),
            "terms": self.as_document(),
            "basis": str(self.weakest_basis()),
            "arena": "request",
            "note": (
                "REQUEST ARENA ONLY. worst_case = tensorfs manifest weight "
                "bytes + varena alignment tax + this. The weight arena is "
                "never declared (pgw#1598 §2)."
            ),
        }
        return row


def vocabulary_document() -> dict[str, Any]:
    """The closed vocabulary, serialized. One producer for both languages."""

    return {
        name: {
            "scale": spec.scale,
            "inputs": list(spec.inputs),
            "doc": spec.doc,
        }
        for name, spec in TERM_VOCABULARY.items()
    }


def evaluate(terms: Any, shape: RequestShape) -> int:
    """Evaluate a formula given either a :class:`Demand` or its DOCUMENT.

    The document arm is the one the conformance corpus exercises, because it
    is the arm Go has: a list of ``{"term": …, "bytes": …}`` rows and nothing
    else. Python must not be able to get a different answer by reading the
    richer in-memory object.
    """

    if isinstance(terms, Demand):
        return terms.evaluate(shape)
    checked = shape.validated()
    total = 0
    for row in terms:
        name = str(row["term"])
        spec = TERM_VOCABULARY.get(name)
        if spec is None:
            raise DemandEvaluationError(
                f"unknown demand term {name!r}; the vocabulary is CLOSED and "
                f"is {sorted(TERM_VOCABULARY)}"
            )
        total += _scaled(int(row["bytes"]), term_value(name, checked), spec.scale)
    return _representable(total)


def _term(
    name: str,
    nbytes: Union[int, float],
    basis: Union[Basis, str],
    source: str,
) -> Demand:
    if isinstance(nbytes, bool) or not isinstance(nbytes, (int, float)):
        raise DemandDeclarationError(
            f"{name}(): a coefficient is a BYTE COUNT; got "
            f"{type(nbytes).__name__}. Use `GiB(1.2)` / `MiB(220)` / "
            f"`KiB(64)` so the formula reads in the units it was measured in."
        )
    coefficient = int(nbytes)
    if coefficient < 0:
        raise DemandDeclarationError(
            f"{name}(): a coefficient cannot be negative ({coefficient}). A "
            f"term that REDUCES demand is a degradation TECHNIQUE's transform "
            f"(pgw#1605's catalog), never a declared term."
        )
    try:
        held = Basis(basis)
    except ValueError:
        raise DemandDeclarationError(
            f"{name}(basis=): a coefficient's provenance is one of "
            f"{[str(b) for b in Basis]}, got {basis!r}"
        ) from None
    text = str(source or "").strip()
    if held is not Basis.UNCALIBRATED and not text:
        raise DemandDeclarationError(
            f"{name}(basis={held}) without a source=. A coefficient claiming "
            f"to be {held} must name the run or the ledger key that produced "
            f"it — an unsourced measurement is a magic constant wearing a "
            f"label. Either cite it (source='tcg#80 sm_89 cold-daemon n=1') "
            f"or declare it for what it is (the default, {Basis.UNCALIBRATED})."
        )
    return Demand(
        terms=(
            Term(name=name, coefficient=coefficient, basis=held, source=text),
        )
    )


def const(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """A fixed floor: the bytes this lane needs regardless of the request."""

    return _term("const", nbytes, basis, source)


def per_mp(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """Bytes per output megapixel."""

    return _term("megapixels", nbytes, basis, source)


def per_mp_batch(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """Bytes per megapixel x batch — the activation term of an image model."""

    return _term("mp_batch", nbytes, basis, source)


def per_batch(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """Bytes per launch batch entry, independent of resolution."""

    return _term("batch", nbytes, basis, source)


def per_frame(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """Bytes per video frame — the linear video term."""

    return _term("frames", nbytes, basis, source)


def per_frame_squared(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """Bytes per frame SQUARED — the quadratic attention term."""

    return _term("frames_squared", nbytes, basis, source)


def per_latent_token(
    nbytes: Union[int, float], *,
    basis: Union[Basis, str] = Basis.UNCALIBRATED, source: str = "",
) -> Demand:
    """Bytes per latent sequence token."""

    return _term("latent_tokens", nbytes, basis, source)
