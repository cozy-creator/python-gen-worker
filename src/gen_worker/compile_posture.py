"""WHOSE machine this mint is running on — declared by the entry point."""

from __future__ import annotations

from typing import Optional

import msgspec

USER_MACHINE_NICE = 19

FLEET_MINT_NICE = 10

USER_MACHINE_CPU_SHARE = 2

USER_MACHINE_MAX_ENTRY_WORKERS = 4

USER_MACHINE_RSS_RESERVE_BYTES = 8 * 1024**3


class CompilePosture(msgspec.Struct, frozen=True, kw_only=True):
    """Whose machine a mint runs on."""

    user_machine: bool = False

    def nice_level(self) -> int:
        """Scheduling-priority increment for the mint process tree."""
        return USER_MACHINE_NICE if self.user_machine else FLEET_MINT_NICE

    def cpu_budget_cores(self, vcpus: int, *, headroom: int) -> int:
        """Cores the pool may size itself against."""
        budget = int(vcpus) - int(headroom)
        if self.user_machine:
            budget = min(budget, int(vcpus) // USER_MACHINE_CPU_SHARE)
        return budget

    def entry_ceiling(self, default: int) -> int:
        """The hard cap on concurrent entry children."""
        if self.user_machine:
            return min(int(default), USER_MACHINE_MAX_ENTRY_WORKERS)
        return int(default)

    def rss_reserve_bytes(self, default: int) -> int:
        """Host RAM the pool must leave alone."""
        if self.user_machine:
            return max(int(default), USER_MACHINE_RSS_RESERVE_BYTES)
        return int(default)

    def facts(self) -> dict:
        """The posture as it rides the width row, so a K nobody expected can be explained without re-deriving anything."""
        return {
            "posture": "user-machine" if self.user_machine else "fleet",
            "nice": self.nice_level(),
        }


FLEET = CompilePosture(user_machine=False)

USER_MACHINE = CompilePosture(user_machine=True)


_INSTALLED: Optional[CompilePosture] = None


def install(posture: CompilePosture) -> None:
    """Publish this process's posture."""
    global _INSTALLED
    _INSTALLED = posture


def current() -> CompilePosture:
    """The installed posture, or :data:`FLEET`."""
    return FLEET if _INSTALLED is None else _INSTALLED


__all__ = [
    "FLEET_MINT_NICE",
    "USER_MACHINE_MAX_ENTRY_WORKERS",
    "USER_MACHINE_NICE",
    "USER_MACHINE_RSS_RESERVE_BYTES",
    "CompilePosture",
    "FLEET",
    "USER_MACHINE",
    "current",
    "install",
]
