"""pgw#1328: what an adopt-only pod does with a boot-adopt that did not adopt.

:mod:`gen_worker.boot_adopt` is already a complete answer — 48 typed reason
tokens, every terminus reported (pgw#1116). Its docstring states the eager
pod's disposition for all of them in one sentence: *"A MISS is a complete
answer, not an absence: the pod serves eager, mints in the background."*

An adopt-only pod has neither half of that sentence available, so every one of
those tokens needs a second decision — **refuse, or route** — and this module
is the ONE place it is made. A total table, no default: a boot reason with no
entry is an adopt-only pod with no answer, and the import-time check below
turns that into a build failure rather than a pod that improvises.

THE TRACER TOKENS ARE NOT MAPPED, DELIBERATELY
-----------------------------------------------
Nineteen of ``boot_adopt.REASONS`` are ``BootKeyUnavailable`` tokens a TRACE
CHILD produces (``trace_failed``, ``child_died``, ``code_drift``, …). They are
unreachable in this role by construction: pgw#1327 made the deriver an injected
parameter and recorded that *"an adopt-only role states its posture by passing
no deriver"*, so nothing can trace. Mapping them would assert the opposite. If
one arrives anyway, the role's premise is broken and this module says so
loudly — a silent mapping would let an adopt-only pod that is quietly still
tracing look exactly like one that never could.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from .. import boot_adopt
from .refusal import AdoptOnlyRefusal, MissKind

logger = logging.getLogger(__name__)

#: Reached only through a mint-lane key deriver, which this role never injects.
TRACER_ONLY: Tuple[str, ...] = tuple(
    reason for reason in boot_adopt.DERIVE_REASONS
    if reason not in (
        "keyset_absent", "keyset_invalid", "closure_unavailable",
        "no_runtime_sm")
)

#: Total over every boot-adopt reason this role can reach. ``None`` means the
#: boot ADOPTED and there is no refusal to make.
BOOT_MISS_KINDS: Dict[str, Optional[MissKind]] = {
    # Adopted — the two termini that lead to a compiled boot.
    boot_adopt.HIT: None,
    boot_adopt.LOCAL_HIT: None,

    # The release declares nothing this pod could adopt. Same image, same
    # answer, on every pod in the fleet.
    # `no_declaration` / `invalid_declaration` are NOT here: they are
    # `BootKeyUnavailable` tokens a trace child raises, so they belong to
    # TRACER_ONLY. The two below are the executor's own pre-attempt gates,
    # which run in every role.
    "no_export_declaration": MissKind.NOT_ADOPTABLE,
    "declaration_unreadable": MissKind.NOT_ADOPTABLE,
    # pgw#1372: the boot derive is deleted; until the adopt-first release
    # pull is wired for this release there is no artifact ANY pod of this
    # image could adopt — same answer fleet-wide, so it routes nowhere.
    "boot_derive_deleted": MissKind.NOT_ADOPTABLE,

    # This pod's own posture forbids arming. A pod without it can serve.
    "eager_only": MissKind.ARM_FORBIDDEN,
    "operator_eager_only": MissKind.ARM_FORBIDDEN,
    "no_runtime_sm": MissKind.ARM_FORBIDDEN,

    # pgw#1327's key set as DATA: no document holds this closure, or the one
    # that does cannot be read.
    "keyset_absent": MissKind.NO_KEYSET,
    "closure_unavailable": MissKind.NO_KEYSET,
    "keyset_invalid": MissKind.KEYSET_INVALID,

    # §4.29: asked by key, and the artifact is not here.
    "no_compiled_graph_source": MissKind.ARTIFACT_MISS,
    "local_miss_no_hub": MissKind.ARTIFACT_MISS,
    "miss": MissKind.ARTIFACT_MISS,
    "resolve_unreachable": MissKind.ARTIFACT_MISS,
    "materialize_failed": MissKind.ARTIFACT_MISS,
    "compiled_graph_resolve_transport_unavailable": MissKind.ARTIFACT_MISS,

    # The answer itself was refused fail-closed. `invalid_request` sits here
    # rather than with the misses on purpose: the worker asked a malformed
    # question, and re-placing a malformed question is a second identical
    # failure on somebody else's card.
    "invalid_request": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_ambiguous": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_incomplete": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_client_supplied_field": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_too_many_keys": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_duplicate_key": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_short_answer": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_answer_out_of_order": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_batch_signature": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_shared_receipt": MissKind.RESOLVE_REFUSED,
    "compiled_graph_resolve_unknown_status": MissKind.RESOLVE_REFUSED,

    # pgw#1122: resolved, materialized, and the arm gate said no. Another pod
    # may establish the identity this one could not.
    "arm_refused": MissKind.ARM_REFUSED,
}


class TracerReasonInAdoptOnly(RuntimeError):
    """A boot-adopt reason only a key TRACER can produce reached this role."""


_unmapped = sorted(
    reason for reason in boot_adopt.REASONS
    if reason not in BOOT_MISS_KINDS and reason not in TRACER_ONLY)
if _unmapped:  # pragma: no cover — an import-time contradiction
    raise RuntimeError(
        f"boot-adopt reasons with no adopt-only disposition: {_unmapped}. "
        f"Every reason an adopt-only pod can reach must say refuse or route — "
        f"there is no eager fallback to absorb an unmapped one.")
_stale = sorted(set(BOOT_MISS_KINDS) - set(boot_adopt.REASONS))
if _stale:  # pragma: no cover — an import-time contradiction
    raise RuntimeError(
        f"adopt-only dispositions for reasons boot_adopt no longer has: "
        f"{_stale}. A mapping for a deleted token is a claim about nothing.")
del _unmapped, _stale


def refusal_for(outcome: boot_adopt.BootAdoptOutcome) -> Optional[AdoptOnlyRefusal]:
    """The adopt-only refusal this boot-adopt outcome means, or ``None``.

    ``None`` exactly when the boot adopted. Never guesses: an unknown reason
    raises rather than defaulting, because the default an adopt-only pod would
    need is the eager fallback it does not have.
    """
    reason = outcome.reason
    if outcome.adopted:
        return None
    if reason in TRACER_ONLY:
        raise TracerReasonInAdoptOnly(
            f"boot-adopt reported {reason!r}, which only a key tracer "
            f"produces — an adopt-only pod injects no deriver (pgw#1327), so "
            f"this pod is not the role it declared")
    if reason not in BOOT_MISS_KINDS:
        raise TracerReasonInAdoptOnly(
            f"boot-adopt reported {reason!r}, which has no adopt-only "
            f"disposition. Add it to BOOT_MISS_KINDS deliberately.")
    kind = BOOT_MISS_KINDS[reason]
    if kind is None:  # a hit terminus that did not carry an adoption
        raise TracerReasonInAdoptOnly(
            f"boot-adopt reported {reason!r} with no adoption attached")
    return AdoptOnlyRefusal(
        kind=kind, function=outcome.function, family=outcome.family,
        compiled_graph_key=outcome.derived_key,
        detail=f"boot_adopt={reason}"
               + (f" — {outcome.detail}" if outcome.detail else ""))


__all__ = [
    "BOOT_MISS_KINDS",
    "TRACER_ONLY",
    "TracerReasonInAdoptOnly",
    "refusal_for",
]
