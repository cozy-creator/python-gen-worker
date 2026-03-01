from __future__ import annotations

from typing import Protocol, cast


class _Detachable(Protocol):
    def detach(self) -> object:
        ...


class _ItemLike(Protocol):
    def item(self) -> object:
        ...


def to_float_scalar(value: object) -> float:
    if hasattr(value, "detach"):
        value = cast(_Detachable, value).detach()
    if hasattr(value, "item"):
        return float(cast(_ItemLike, value).item())
    return float(value)


__all__ = ["to_float_scalar"]
