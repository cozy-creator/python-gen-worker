"""The ONE adoption vocabulary (pgw#923).

Adopting a pre-built compiled cell — hub-delivered or catalog-discovered — used
to be described twice. The typed description rode ``ModelEvent{ADOPTED}``
(``duration_ms``, ``cache_hits``, ``cache_misses``, ``warmup_s``), which the hub
persists as ``worker_activity_events.kind='compile_cache_adopt'`` with two
partial indexes and a p50/p95/max admin surface. The other description was a
free-text ``aot_adopt`` activity event that put ``family=… key=… sku=…`` in
prose and no numbers anywhere.

Only the free-text one was ever reachable from the path adoptions actually take
(arming at boot, through ``fleet_cells`` — historically called "boot attach",
which names WHEN the cell is armed, never a hub push); the typed one was
reachable only from the
hub-commanded ``ADOPT_COMPILE_CACHE`` operation, which no stack has ever
dispatched. So the measured lane had zero rows on both live stacks while every
real adoption landed at ``duration_ms=0``, and the percentile endpoint
aggregated a population with no members.

pgw#1032 finished the argument: ``ADOPT_COMPILE_CACHE`` is GONE on both sides.
Its hub-side push was keyed off the COMPUTED (``kind="inductor"``) cell key,
and since pgw#1010 nothing mints into that key space — every publishable cell
is STAMPED ``aot-inductor`` — so the push could never have selected a cell.

**Nothing is DELIVERED to a pod any more.** th#1702 also deletes the hub's
snapshot attach (HelloAck and RunJob both). pgw#904 then deleted worker-side
fetch-and-filter discovery too: the hub RESOLVES the exact cell and names it
in ``Arm.artifact``; the worker materializes only that identity and feeds it
through the same gates. That is the only adoption there is. (pgw#904
replaces the listing with a hub-RESOLVED ``Arm.artifact`` — still a pull, not
a push.)

The free-text lane is DELETED. This module is what replaced it: the arm returns
a typed outcome instead of a bare bool, the arming policy times it and names the
candidate, and the executor — the one component that owns the wire — turns it
into the ``ModelEvent`` the hub already knows how to store. One fact, one
spelling, and the spelling that carries numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["AdoptOutcome", "CellAdoption", "EagerPhase"]


class EagerPhase(StrEnum):
    """Why a compile-declaring pipeline is serving EAGER — one token, shared.

    pgw#824 gave the arming policy's declines a classified token instead of the
    single ``mint_unavailable`` constant they used to share; the token rides
    ``self_mint_skipped``/``self_mint_started`` as ``phase`` and rides out of
    the decision as ``ArmOutcome.eager_reason``, so the hub's activity rows and
    the request row's ``fallback_reason`` join on one string.

    The tokens lived as bare literals at each ``return``, and the pgw#824 audit
    re-spelled the same nine strings in a second list to check they were all
    still there. Two lists of literals is a drift channel, and it drifted:
    pgw#923 moved every exit out of ``enable_compiled`` into ``_arming_policy``
    and the audit — which read only ``enable_compiled``'s source — reported all
    nine as lost when not one had changed. Naming the members here makes the
    audit reference the vocabulary instead of re-typing it.

    **The values are a wire contract.** The hub groups
    ``worker_activity_events`` on them and joins them to request rows; renaming
    one orphans that history, so ``test_silent_failure_audit_pgw824`` pins the
    member-to-value mapping rather than reading it back out of this file.
    """

    #: The mint-impossibility exits, each a distinct `_fail_closed` cause.
    #: pgw#1010 retired three of the original nine with the in-process
    #: capture that produced them — ``delivered_cell_seeded``,
    #: ``capture_conflict`` and ``multi_group_in_process`` all named hazards
    #: of moving the process-global inductor cache dir, and nothing moves it
    #: any more; ``capture_arm_failed`` became ``jit_arm_failed``, which is
    #: what it always measured (this process cannot arm its declared
    #: targets), minus a capture that no longer exists.
    NO_FAMILY = "no_family"
    NO_CUDA = "no_cuda"
    NO_TOOLCHAIN = "no_toolchain"
    NO_COMPILE_TARGET = "no_compile_target"
    KEY_COMPUTATION_FAILED = "key_computation_failed"
    JIT_ARM_FAILED = "jit_arm_failed"
    #: pgw#1010: a mandatory (w8a8/w4a4) lane on a family that declares no
    #: export. The lane serves only from a cell, the only cell is an AOT cell,
    #: and nothing can mint one here — so the pod fails closed instead of
    #: compiling a JIT intake arm no request may be dispatched to.
    MANDATORY_LANE_NEEDS_A_COMPILED_GRAPH = "mandatory_lane_needs_a_compiled_graph"

    #: The tenth eager exit: it declines BEFORE `_fail_closed` (a quarantined
    #: identity must not be re-minted) and was the one that only logged.
    COMPILED_GRAPH_QUARANTINED = "compiled_graph_quarantined"

    #: pgw#1340 / th#2098: the handback seam compares an axis a packed cell
    #: structurally cannot state, so the arm can only ever refuse. The mint is
    #: declined at obligation-open, for $0, instead of after 20-45 minutes of
    #: paid GPU — which is what it cost on `sd15` for two wheels running:
    #: `family: child cell states '', this runtime computed 'sd15'`, every
    #: mint, nothing published, ~$1.00 of L4 per burst.
    #:
    #: A pod reporting this is reporting a CODE defect with a named owner
    #: (an axis outside `artifact_meta.cell_metadata_fields()`), never a
    #: capability, a card or a release — which is why it does not share a
    #: token with the mint-impossibility exits above.
    ARM_AXES_UNSTATEABLE = "arm_axes_unstateable"

    #: Eager with an END — a delegated mint child is building the cell.
    MINT_IN_PROGRESS = "mint_in_progress"

    #: pgw#904: the hub's ExecutionSpec named ``eager_only`` for this arm.
    #: Eager is the ORDER, not a degradation — the worker armed nothing
    #: because nothing was named, which is a complete answer.
    HUB_ORDERED_EAGER = "hub_ordered_eager"

    #: pgw#1035: the four tokens below rode the SAME wire columns as the ones
    #: above — `phase` on `self_mint_skipped`/`self_mint_started`, and the
    #: request row's `fallback_reason` — while living as bare literals in
    #: `serving_mode` and `executor`. That is exactly the two-lists-of-literals
    #: drift channel this enum was created to close, and only
    #: `MINT_IN_PROGRESS` had ever been pinned to it. `serving_mode`'s
    #: `POSTURE_*` names are now ALIASES of these members, so there is one
    #: vocabulary and the values are unchanged (the hub's grouped history is
    #: untouched).

    #: The arming brain has not answered yet (boot in flight, setup unfinished).
    ARM_PENDING = "arm_pending"
    #: The release declared no compile target at all — eager is the contract,
    #: not a degradation. Distinct so it never pollutes the defect classes.
    NO_COMPILE_DECLARED = "no_compile_declared"
    #: Terminal fallback when a decline reached the request path unclassified.
    UNCOMPILED = "uncompiled"
    #: Setup finished with a declared compile target, nothing armed and no mint
    #: in flight: this worker serves eager for the rest of its life. Terminal,
    #: and it must mean "nothing is dispatchable" (pgw#844), never "partial".
    BOOT_ENDED_UNCOMPILED = "boot_ended_uncompiled"

    #: pgw#1082: the declared region did not trace WHOLE. Dynamo's fullgraph
    #: refusal fired, the platform refused to serve eager-glued fragments as
    #: compiled, and this instance degraded to explicit eager. An AUTHORING
    #: defect in the endpoint's block, named as one — never a silent 1.0x
    #: "compiled" lane (the ie#632/pgw#1078 failure class).
    GRAPH_BREAK = "graph_break"

    #: pgw#1082: the endpoint's `dynamic=(...)` declaration names a range its
    #: own inputs leave, so the declared marks cannot be applied and the
    #: target degraded to eager. Also an AUTHORING defect, and the one that
    #: actually cost minimax-h3 its compiled wall.
    DECLARED_RANGE_EXCEEDED = "declared_range_exceeded"

    #: pgw#1093: the target WAS installed and armed, and a served call then
    #: failed permanently for a reason that is neither a graph break nor a
    #: declared-range refusal — a kernel that refuses this shape, an OOM
    #: inside the region, a module the endpoint mutated after the arm. Before
    #: this token that degrade reached the wire as NOTHING, so it was
    #: indistinguishable from "no target was ever installed" and both read
    #: `uncompiled`. The distinction is the whole point: one is an execution
    #: failure with a named exception, the other is a WIRING failure.
    COMPILED_DEGRADED = "compiled_degraded"

    #: pgw#1093: an arm SUCCEEDED inside `setup()` and `_install_compile_targets`
    #: then resolved no declared target on the same object — the boot compiled
    #: graphs it can never dispatch to. The pgw#1078 D2 class, one layer up,
    #: and previously a bare `continue` with no note, no counter, no event.
    ARMED_TARGET_UNRESOLVED = "armed_target_unresolved"

    #: pgw#1093: the record ended setup owning ZERO compile-capable candidate
    #: objects while its spec declares a compile family. The candidate loop
    #: never ran, so not one of the per-candidate omission tokens could fire.
    NO_COMPILE_CANDIDATES = "no_compile_candidates"

    #: pgw#1122: the pod resolved a cell BY ITS OWN DERIVED KEY (§4.27
    #: boot-adopt), materialized it, and the arm refused — a receipt gate, a
    #: publisher check, an identity this pod cannot establish. Nothing named
    #: this arm but the pod itself, so there is no order to obey and no reason
    #: to take the function down: it serves eager and mints its own, which is
    #: exactly the boot every pod did before boot-adopt existed. Measured cost
    #: of the alternative: `worker_function_unavailable`, three pods reaped
    #: `state_blocked_idle`, two replacements bought.
    ADOPTED_COMPILED_GRAPH_REFUSED = "adopted_compiled_graph_refused"

    #: pgw#1142 / §4.32 item 4: an OPERATOR ordered this worker to serve eager
    #: only, over the scheduler's control channel or the cozy-local CLI. It is
    #: neither a defect nor a degradation — it is the answer, and it is
    #: reversible — so it must never be counted with the failure classes above
    #: or with `hub_ordered_eager` (which is one PLAN's backend, not a standing
    #: order about this pod). A worker holding this token has cells it could be
    #: serving from, still armed, deliberately not called.
    OPERATOR_EAGER_ONLY = "operator_eager_only"

    #: `_fail_closed`'s default, for a caller that has not classified its exit.
    #: A new decline landing here rather than on its own member is the
    #: regression pgw#824 exists to catch.
    MINT_UNAVAILABLE = "mint_unavailable"


@dataclass(frozen=True)
class AdoptOutcome:
    """The result of arming ONE candidate artifact on ONE pipeline.

    Truthy exactly when the cell armed, so the many ``if enable_compiled(...)``
    call sites read unchanged while the classified refusal — previously
    reachable only as the ``phase`` of a free-text event — becomes a value the
    caller can act on and put on the wire.

    ``reason`` is the short, stable, countable token (an ``AdoptError.reason``,
    a lane-gate refusal, ``no_cell``); ``detail`` is the human sentence.
    ``identity`` carries ``family=… key=…`` when the candidate's own metadata
    could be read — a refusal must still name the cell it refused, including
    when the refusal IS a metadata problem.
    """

    armed: bool
    reason: str = ""
    detail: str = ""
    identity: str = ""

    def __bool__(self) -> bool:
        return self.armed

    @classmethod
    def hit(cls, identity: str = "") -> "AdoptOutcome":
        return cls(armed=True, identity=identity)

    @classmethod
    def miss(cls, reason: str, detail: str = "", identity: str = "") -> "AdoptOutcome":
        return cls(armed=False, reason=reason, detail=detail[:2000], identity=identity)


@dataclass(frozen=True)
class CellAdoption:
    """One adoption ATTEMPT, measured, with the identity the hub fences on.

    ``arm_ms`` is the wall time of the arm itself — load, bind, wrap, gate —
    and is the same quantity the hub stores as the adoption's ``duration_ms``.
    The warmup half is deliberately absent: a cell is armed during injection
    and warmed later, by the setup warmup, so the two numbers are known at two
    different instants and the executor joins them (pgw#1032: an arm-time
    acquisition is the only adoption there is, so this is the only shape).
    """

    ref: str
    snapshot_digest: str
    artifact_kind: str
    arm_ms: int
    armed: bool
    reason: str = ""
    detail: str = ""
    #: ``id()`` of the pipeline this attempt armed, so the executor can join the
    #: adoption to that object's own warmup proof (its cache hit/miss deltas)
    #: instead of attributing another slot's evidence to it.
    pipeline_id: int = 0
    #: pgw#1176: WHICH graph class this attempt was about, and the ``cg-key-v1`` key
    #: it was about. A boot resolves a KEY SET, so an attempt that MISSED has
    #: no artifact ``ref`` BY CONSTRUCTION — and a miss is the most common
    #: per-entry outcome there is. These two carry the identity a ref-less
    #: attempt would otherwise have no way to state, so it can still be
    #: reported instead of dropped.
    entry: str = ""
    compiled_graph_key: str = ""
