from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LifecycleLease:
    run_id: str
    lease_id: str


class JobLifecycleClient(Protocol):
    """Control-plane lifecycle ownership for training runs."""

    def claim(self) -> LifecycleLease:
        ...

    def heartbeat(self, lease: LifecycleLease) -> None:
        ...

    def request_cancel(self, lease: LifecycleLease) -> bool:
        ...

    def mark_completed(self, lease: LifecycleLease) -> None:
        ...

    def mark_failed(self, lease: LifecycleLease, error: str) -> None:
        ...
