"""What this worker was bought to do — a goal set, not a mode."""

from __future__ import annotations

import msgspec


class WorkerGoals(msgspec.Struct, frozen=True, kw_only=True):
    """The goals this worker currently holds."""

    serve: bool = True

    def serve_admitted(self) -> bool:
        """Whether tenant dispatch may be accepted."""
        return self.serve


SERVE_ONLY = WorkerGoals(serve=True)


_INSTALLED: WorkerGoals | None = None


def install(goals: WorkerGoals) -> None:
    """Publish the goal set for this process."""
    global _INSTALLED
    _INSTALLED = goals


def current() -> WorkerGoals:
    """The installed goal set."""
    return _INSTALLED if _INSTALLED is not None else SERVE_ONLY


def reset_for_test() -> None:
    """Drop the installed goal set."""
    global _INSTALLED
    _INSTALLED = None
