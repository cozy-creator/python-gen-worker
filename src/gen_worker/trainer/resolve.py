from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ResolvedRef:
    owner: str
    repo: str
    digest: str


class ResolveClient(Protocol):
    def resolve_ref(self, ref: str) -> ResolvedRef:
        ...


__all__ = ["ResolveClient", "ResolvedRef"]
