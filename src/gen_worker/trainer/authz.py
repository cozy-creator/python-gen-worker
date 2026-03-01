from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerAuthContext:
    owner: str
    token: str
    scope: str


__all__ = ["TrainerAuthContext"]
