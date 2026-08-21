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
    """Bytes."""

    return int(count * 1024)


def MiB(count: float) -> int:
    return int(count * 1024 * 1024)


def GiB(count: float) -> int:
    return int(count * 1024 * 1024 * 1024)


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
    """A declared demand formula: `Σ terms`, in bytes."""

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

        return {term.name: term.coefficient for term in self.terms}

    def as_document(self) -> list[dict[str, Any]]:
        """The release-document shape."""

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
    """Bytes per megapixel x batch — the activation term of an image model."""

    return _term("mp_batch", nbytes)


def per_batch(nbytes: Union[int, float]) -> Demand:
    """Bytes per launch batch entry, independent of resolution."""

    return _term("batch", nbytes)


def per_frame(nbytes: Union[int, float]) -> Demand:
    """Bytes per video frame — the linear video term."""

    return _term("frames", nbytes)


def per_frame_squared(nbytes: Union[int, float]) -> Demand:
    """Bytes per frame SQUARED — the quadratic attention term."""

    return _term("frames_squared", nbytes)


def per_latent_token(nbytes: Union[int, float]) -> Demand:
    """Bytes per latent sequence token."""

    return _term("latent_tokens", nbytes)
