"""What this worker was bought to do — a goal set, not a mode.

Ruling §1.17, on the carrier: envs are for secrets and configuration, not for
logic gates — a worker mode is sent as a command, never as an env.

There is exactly ONE goal:

* ``serve`` — accept tenant dispatch and hold a resident serving instance.

Minting is not a goal anybody can hold, because nobody orders one. Every
serving pod mints opportunistically on a compiled graph miss, and that is the only mint
path there is — so the tenant reserves it runs under are UNCONDITIONAL, not a
function of a posture.

Carrier
-------
The Directive delivers a full-replace goal set on the control channel, at which
point :func:`install` is called with each new Directive. There is no
``WORKER_MODE`` env: a pod that has to be told it may serve is a pod the hub
should not have bought.
"""

from __future__ import annotations

import msgspec


class WorkerGoals(msgspec.Struct, frozen=True, kw_only=True):
    """The goals this worker currently holds. Passed, never inferred."""

    #: Accept tenant dispatch and hold a resident serving instance.
    serve: bool = True

    def serve_admitted(self) -> bool:
        """Whether tenant dispatch may be accepted."""
        return self.serve


#: Every pod, and the default for anything that did not declare.
SERVE_ONLY = WorkerGoals(serve=True)


_INSTALLED: WorkerGoals | None = None


def install(goals: WorkerGoals) -> None:
    """Publish the goal set for this process.

    Called on every full-replace Directive once th#1488 lands. Deliberately an
    explicit publish rather than a lazy loader: there is exactly one carrier
    and exactly one moment it is set (§4.22). A test installs a value; it does
    not clear a cache.
    """
    global _INSTALLED
    _INSTALLED = goals


def current() -> WorkerGoals:
    """The installed goal set.

    Falls back to :data:`SERVE_ONLY` when nothing has been installed, which is
    the correct reading for the standalone CLI and for library use: a process
    with no hub is not excluded from serving anything it is asked to run
    in-process.
    """
    return _INSTALLED if _INSTALLED is not None else SERVE_ONLY


def reset_for_test() -> None:
    """Drop the installed goal set. Test-only."""
    global _INSTALLED
    _INSTALLED = None
