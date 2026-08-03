"""What this worker was bought to do — a COMPOSABLE goal set, not a mode.

Ruling §1.17 (Paul, 2026-08-03), verbatim:

    *"worker-mode is wrongly designed; you can spawn a GPU and tell it to mint
    some AOT-cells, while also using it to serve jobs (inference) as needed."*

and, on the carrier:

    *"Envs are for secrets and configuration, not for logic gates; if you want
    to have a worker-mode you should send it as a command, not as a random
    env."*

This module replaces ``gen_worker.worker_mode`` (deleted, pgw#930). That module
expressed the worker's posture as a closed two-tuple ``(serve, forge)`` with a
single predicate ``is_forge()``, so "serve" was defined everywhere in the tree
as *not forge* and there was no way to spell "both". Five production sites
consumed that boolean and every one of them read it as a process-global.

The goals here are INDEPENDENT
------------------------------
* ``serve`` — accept tenant dispatch and hold a resident serving instance.
* ``mint`` — a mint goal was SCHEDULED for this pod: drive it to a terminal
  disposition and report it. This is not the same fact as "this pod may
  compile": every serving pod already mints opportunistically on a cell miss,
  arbitrated by :mod:`gen_worker.mint_budget`. That behaviour is unchanged and
  is not a goal.

All four combinations are meaningful, and the two the deleted enum could spell
are only two of them:

===========  ========  ==================================================
``serve``    ``mint``  meaning
===========  ========  ==================================================
True         False     an ordinary serving pod (the old ``WORKER_MODE=serve``)
False        True      a mint-only pod (the old ``WORKER_MODE=forge``)
True         True      **Paul's case** — serves tenant traffic AND drives a
                       scheduled mint. The tenant reserve is real, and the
                       mint driver does not retire the pod out from under
                       the serving goal.
False        False     registered, holding neither goal: a pod awaiting a
                       Directive. Refuses dispatch, drives no mint.
===========  ========  ==================================================

A blocked mint goal never turns a still-valid serve goal into a refusal — that
is the property the exclusive boolean could not have, and it is why the two
goals are reported independently rather than collapsed to one string.

Why the exclusivity was never load-bearing
------------------------------------------
``worker_mode``'s own docstring gave two reasons the mode existed, and both are
resource facts the worker already measures:

1. *"No resident serving model."* The real fact is how much VRAM is free, which
   ``mint_budget._read_device()`` reads directly. A pod that released its
   serving instance has no tenant reserve to keep because ``allocated`` fell —
   not because an env said so.
2. *"Never in the serving rotation."* That is a hub placement decision the hub
   already makes. The worker's job is to honour the goal set it was given, and
   ``serve_admitted()`` below is that one reading rather than a second copy of
   the hub's placement policy.

Carrier
-------
The Directive (th#1488) will deliver a full-replace goal set on the control
channel, at which point :func:`install` is called with each new Directive and
``WORKER_MODE`` disappears entirely — §1.17's *"an env may carry a VALUE; it may
not carry a DECISION."* Until that transport exists the goal set is SEEDED once
at bootstrap from the hub's pod-launch declaration, parsed by
:func:`from_settings` and then passed. The env is read by
``gen_worker.config`` and by nothing else (§1.18); this module never touches it.
"""

from __future__ import annotations

import msgspec

from .config import Settings

#: Bootstrap spellings of the pod-launch declaration. These are the two the hub
#: can currently send; they are NOT the goal vocabulary, which is the struct.
_DECL_SERVE = "serve"
_DECL_FORGE = "forge"


class WorkerGoals(msgspec.Struct, frozen=True, kw_only=True):
    """The goals this worker currently holds. Passed, never inferred."""

    #: Accept tenant dispatch and hold a resident serving instance.
    serve: bool = True

    #: A mint goal is scheduled for this pod; drive it and report its terminal.
    mint: bool = False

    #: The RAW bootstrap declaration, carried for reporting only.
    #:
    #: pgw#846 measured what happens when this is dropped: the process-split
    #: parent echoed the protobuf default ``""`` and every forge pod the
    #: program bought was idle-reaped as ``cold_idle_never_dispatched``. The
    #: hub must be able to SEE that it sent a spelling this image does not
    #: understand rather than infer a serving pod, so the declaration ships as
    #: declared even when :func:`from_settings` could not interpret it.
    declared: str = ""

    #: True when `declared` matched no spelling this build knows. The goals
    #: then fall back to serve-only — a worker that refuses to boot because the
    #: hub used a newer vocabulary is a dead pod, and serving is the safe
    #: reading — but the fact that a fallback happened is recorded here instead
    #: of being silently indistinguishable from an explicit `serve`.
    declaration_understood: bool = True

    def serve_admitted(self) -> bool:
        """Whether tenant dispatch may be accepted."""
        return self.serve

    def drives_mint(self) -> bool:
        """Whether the scheduled-mint driver should run."""
        return self.mint

    def retires_when_mint_completes(self) -> bool:
        """Whether finishing the mint goal means this pod is done.

        Only when nothing else is being asked of it. This is the single line
        that makes ``serve=True, mint=True`` safe: the mint driver used to call
        ``start_drain`` unconditionally because a forge pod had nothing else to
        do, and on a dual-goal pod that would drain a live serving instance.
        """
        return self.mint and not self.serve

    def tenant_reserve_applies(self) -> bool:
        """Whether VRAM / CPU / host-RAM must be held back for a tenant.

        Every reserve term in :mod:`gen_worker.mint_budget` and
        :mod:`gen_worker.aot_compile_pool` exists to protect a co-resident
        serving process. The question they were asking — *is this a forge pod*
        — was a proxy for this one, and a wrong proxy the moment both goals can
        be held at once.
        """
        return self.serve

    def wire_declaration(self) -> str:
        """The value for ``WorkerResources.worker_mode`` on the scheduler wire.

        The proto field is a vendored artifact of Tensorhub's canonical copy
        (pgw#944) and the hub still keys dispatch exclusion and drain immunity
        on it, so the goal set is projected back onto the string the hub
        understands rather than the field being changed here. It disappears
        when th#1488's ``repeated Goal goals`` lands.
        """
        if self.declared:
            return self.declared
        return _DECL_FORGE if (self.mint and not self.serve) else _DECL_SERVE


#: Every serving pod, and the default for anything that did not declare.
SERVE_ONLY = WorkerGoals(serve=True, mint=False, declared=_DECL_SERVE)

#: Mint-only: registers, receives no tenant dispatch, mints, publishes, retires.
MINT_ONLY = WorkerGoals(serve=False, mint=True, declared=_DECL_FORGE)


def from_settings(settings: Settings) -> WorkerGoals:
    """Seed the goal set from the hub's pod-launch declaration.

    The ONLY place the legacy ``WORKER_MODE`` spelling is interpreted. It is
    read off the typed `Settings` — never off the environment — so when the
    Directive replaces it, this function is the single site that changes.
    """
    raw = str(settings.worker_mode or "").strip().lower()
    if raw == _DECL_FORGE:
        return WorkerGoals(serve=False, mint=True, declared=raw)
    if raw in ("", _DECL_SERVE):
        return WorkerGoals(serve=True, mint=False, declared=raw or _DECL_SERVE)
    return WorkerGoals(
        serve=True, mint=False, declared=raw, declaration_understood=False,
    )


_INSTALLED: WorkerGoals | None = None


def install(goals: WorkerGoals) -> None:
    """Publish the goal set for this process.

    Called once at bootstrap with :func:`from_settings`, and again on every
    full-replace Directive once th#1488 lands. Deliberately an explicit publish
    rather than a lazy loader: the goals are derived from config that this
    module cannot go and find, so there is exactly one carrier and exactly one
    moment it is set (§4.22). A test installs a value; it does not clear a
    cache.
    """
    global _INSTALLED
    _INSTALLED = goals


def current() -> WorkerGoals:
    """The installed goal set.

    Falls back to :data:`SERVE_ONLY` when nothing has been installed, which is
    the correct reading for the standalone CLI and for library use: a process
    with no hub has no scheduled mint goal and is not excluded from serving
    anything it is asked to run in-process.
    """
    return _INSTALLED if _INSTALLED is not None else SERVE_ONLY


def reset_for_test() -> None:
    """Drop the installed goal set. Test-only."""
    global _INSTALLED
    _INSTALLED = None
