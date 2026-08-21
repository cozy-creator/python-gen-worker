"""The DEMAND FORMULA — the request arena, declared as DATA (pgw#1598 §2).

Paul, 2026-08-20, the mandate this replaces a string with: *"there is no
required VRAM … Memory requirements vary per request … the larger the image
generated, the larger the latent … For MiniMax H3, the longer the video the
more memory is required … quadratically."*

So a lane declares a FORMULA, not a number::

    from gen_worker.demand import const, per_mp_batch, GiB, MiB

    request = const(GiB(1.2)) + per_mp_batch(MiB(220))

`C + Σ cᵢ·termᵢ` over a PLATFORM vocabulary of request-derived extractors.
Two commitments the shape encodes, and they are the reason it is data rather
than a Python callable:

* **Lock-time evaluable** under pgw#1597's static bar ($0, CPU-only,
  deterministic, no weights), and SERIALIZABLE into the release document —
  so tensorhub (Go) can evaluate `worst_case = weights + demand(advertised
  shape envelope)` at pod-buy time WITHOUT running Python. That derived
  number is what replaces the hand-written MachineFloor (se#810's "7").
* **Terms are AUTHOR-OWNED, the vocabulary is PLATFORM-OWNED.** The author
  knows the model scales as megapixels or as frames²; the platform's job is
  enumeration, pricing, calibration and leak detection — it never invents a
  term, and an author cannot invent one either, because the constructors
  below are the only way to make one.

WHAT LIVES HERE AND WHAT DOES NOT (the pgw#1599 / pgw#1600 seam): this module
owns the DECLARATION — the vocabulary, the constructors, the validation and
the serialized shape. pgw#1600 owns EVALUATION: binding a term to a request's
extracted quantity, the ledger-fitted coefficient posterior, the
predicted-vs-measured falsifier and `demand_miss`. Adding evaluation adds a
method here; it changes nothing an author wrote.

The declared coefficient is a PRIOR, never a final answer — on the eager path
pgw#1586's ledger fits the posterior from measurement, and on the compiled
path the per-key driver-level stamp (pgw#1601) IS the number. A formula that
is never corrected by measurement is the defect class, not the feature.
"""

from __future__ import annotations

from typing import Any, Union

import msgspec

__all__ = [
    "Demand",
    "DemandDeclarationError",
    "GiB",
    "KiB",
    "MiB",
    "TERM_VOCABULARY",
    "Term",
    "const",
    "per_batch",
    "per_frame",
    "per_frame_squared",
    "per_latent_token",
    "per_mp",
    "per_mp_batch",
]


class DemandDeclarationError(TypeError):
    """A demand formula is not one the platform can evaluate."""


def KiB(count: float) -> int:
    """Bytes. Spelled out so a formula reads in the units it was measured in."""

    return int(count * 1024)


def MiB(count: float) -> int:
    return int(count * 1024 * 1024)


def GiB(count: float) -> int:
    return int(count * 1024 * 1024 * 1024)


#: The PLATFORM's term vocabulary: term name -> what one unit of it is.
#:
#: Every term is a REQUEST-DERIVED extractor — a quantity the platform can
#: compute from the request payload alone, before the request runs, with no
#: card and no weights. That is what makes the formula quotable to the hub
#: pre-run (pgw#1598 §4: the price is known before the request runs, and no
#: breakpoint is discovered by failing).
TERM_VOCABULARY: dict[str, str] = {
    "const": "a fixed floor, independent of the request",
    "megapixels": "output width x height / 1e6",
    "mp_batch": "megapixels x batch (the CFG pair counts as 2)",
    "batch": "the launch batch size (2 with CFG, 1 without)",
    "frames": "the video frame count",
    "frames_squared": "frame count SQUARED — the quadratic attention term",
    "latent_tokens": "the denoiser's latent sequence length",
}


class Term(msgspec.Struct, frozen=True, kw_only=True):
    """One `cᵢ·termᵢ`: a vocabulary name and its coefficient IN BYTES."""

    name: str
    coefficient: int

    def as_document(self) -> dict[str, Any]:
        return {"term": self.name, "bytes": self.coefficient}


class Demand(msgspec.Struct, frozen=True, kw_only=True):
    """A declared demand formula: `Σ terms`, in bytes.

    Built only by the constructors below and composed with ``+``. There is
    deliberately no way to write an arbitrary expression: a formula that the
    platform cannot serialize is a formula tensorhub cannot evaluate at
    pod-buy time, and a Python callable would be exactly that.
    """

    terms: tuple[Term, ...]

    def __add__(self, other: "Demand") -> "Demand":
        if not isinstance(other, Demand):
            raise DemandDeclarationError(
                f"a demand formula adds to another demand formula, not to "
                f"{type(other).__name__}. Wrap a plain byte count in "
                f"`const(...)`."
            )
        merged: dict[str, int] = {}
        for term in self.terms + other.terms:
            merged[term.name] = merged.get(term.name, 0) + term.coefficient
        return Demand(
            terms=tuple(
                Term(name=name, coefficient=coefficient)
                for name, coefficient in sorted(
                    merged.items(), key=lambda item: _TERM_ORDER[item[0]]
                )
            )
        )

    def coefficients(self) -> dict[str, int]:
        """``{term name: bytes}`` — the vector pgw#1600 fits and evaluates."""

        return {term.name: term.coefficient for term in self.terms}

    def as_document(self) -> list[dict[str, Any]]:
        """The release-document shape. Ordered by the vocabulary, so two
        formulas that declare the same terms serialize identically however
        the author wrote them."""

        return [term.as_document() for term in self.terms]


_TERM_ORDER = {name: index for index, name in enumerate(TERM_VOCABULARY)}


def _term(name: str, nbytes: Union[int, float]) -> Demand:
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
    return Demand(terms=(Term(name=name, coefficient=coefficient),))


def const(nbytes: Union[int, float]) -> Demand:
    """A fixed floor: the bytes this lane needs regardless of the request."""

    return _term("const", nbytes)


def per_mp(nbytes: Union[int, float]) -> Demand:
    """Bytes per output megapixel."""

    return _term("megapixels", nbytes)


def per_mp_batch(nbytes: Union[int, float]) -> Demand:
    """Bytes per megapixel x batch — the activation term of an image model.

    The CFG pair is batch 2, which is why this is the usual term for a
    denoiser rather than :func:`per_mp`: turning CFG off halves it, and
    pgw#1605's CFG batch-split degrades `mp_batch` to `megapixels` exactly.
    """

    return _term("mp_batch", nbytes)


def per_batch(nbytes: Union[int, float]) -> Demand:
    """Bytes per launch batch entry, independent of resolution."""

    return _term("batch", nbytes)


def per_frame(nbytes: Union[int, float]) -> Demand:
    """Bytes per video frame — the linear video term."""

    return _term("frames", nbytes)


def per_frame_squared(nbytes: Union[int, float]) -> Demand:
    """Bytes per frame SQUARED — the quadratic attention term.

    Paul's H3 case verbatim: *"the longer the video the more memory is
    required … quadratically."* This is the term temporal chunking bounds
    (pgw#1605 technique 5).
    """

    return _term("frames_squared", nbytes)


def per_latent_token(nbytes: Union[int, float]) -> Demand:
    """Bytes per latent sequence token."""

    return _term("latent_tokens", nbytes)
