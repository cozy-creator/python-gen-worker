"""What this worker was bought to do — a goal set, not a mode.

Ruling §1.17 (Paul, 2026-08-03), on the carrier:

    *"Envs are for secrets and configuration, not for logic gates; if you want
    to have a worker-mode you should send it as a command, not as a random
    env."*

This module replaced ``gen_worker.worker_mode`` (deleted, pgw#930), whose
closed two-tuple defined "serve" everywhere in the tree as the negation of the
mint-only pod class.

DESIGN-RULINGS §4.28 (Paul, 2026-08-10) then deleted that pod class outright,
and pgw#1092 / th#1751 W4 delete its vocabulary. There is exactly ONE goal
left:

* ``serve`` — accept tenant dispatch and hold a resident serving instance.

Minting is no longer a goal anybody can hold, because nobody orders one. Every
serving pod already mints opportunistically on a cell miss, and that is now the
only mint path there is — so the tenant reserves it runs under are
UNCONDITIONAL, not a function of a posture. The reserve terms in
:mod:`gen_worker.aot_compile_pool` no longer ask this module anything.

Carrier
-------
The Directive (th#1488) will deliver a full-replace goal set on the control
channel, at which point :func:`install` is called with each new Directive. The
``WORKER_MODE`` env and ``WorkerResources.worker_mode`` that used to seed it
are both gone (pgw#1092, th#1751 W4): a pod that has to be told it may serve is
a pod the hub should not have bought.
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
