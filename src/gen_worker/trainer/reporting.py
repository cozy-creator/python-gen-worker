from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReporterEvent:
    event_type: str
    step: int


__all__ = ["ReporterEvent"]
