"""pgw#1089 x pgw#1090 (DESIGN-RULINGS §4.27 steps 1-3): the boot-time adopt.

ONE function, and it is the whole of §4.27's first three steps for a serving
pod:

1. **Derive** the ``cg-key-v1`` key SET from code alone — ``boot_key.derive``,
   process-parallel structure-only traces, no weight resident anywhere. One key
   per graph class (pgw#1176), not one per pod.
2. **Ask THIS MACHINE first** (pgw#1127 S2, §4.28) — ``local_cell_store.
   lookup(key)``. The derived key is the local store's own address, so an
   offline machine holding the exact cell it needs answers here and no hub is
   asked at all. ONE key, TWO lookup routes, the same CAS.
3. **Ask the hub** for everything the machine did not hold, in ONE call —
   ``cell_resolve.resolve_batch`` (pgw#1224), which answers each requested key
   with one independently signed answer, in request order. Exact keys in, rows
   for exactly those keys out: never a listing.
4. **Materialize** a hit through the EXISTING delivery path and hand the
   caller an admission expectation, so the arm runs the gates the Plan path
   runs and not one gate fewer.

Why the local step is BEFORE the hub and not beside it: §4.28 says *"local
cell, local repo-CAS… never requested"*. A local-second ordering would ask for
what the machine already holds, and on a community-cloud pod that is a request
the hub can only answer by shipping bytes back to the machine that minted them.

Why the derivation now runs with NO hub, which it refused to do before: the old
gate reasoned that *"deriving a key nobody will answer is pure boot latency"*.
That is false on exactly the machines §4.28 is about. It survives in the
honest form — :data:`GATE_REASONS`' ``no_compiled_graph_source``, which refuses only when
BOTH answerers are absent (no hub AND an empty local store).

A MISS is a complete answer, not an absence: the pod serves eager,
mints in the background (§4.28: trusted hardware publishes, untrusted keeps it
local) and the request path waits for none of it (pgw#1091).

Why this is a module and not three lines at the call site
---------------------------------------------------------
Because a refusal at any of the three steps must **degrade to the pre-existing
behaviour and say why**, and there are eight distinct ways to fail:
the family has no structure-only build, a trace child died, the hub refused
typed, the hub answered a different key, the transport was unusable, the bytes
failed their digest, the delivered cell would not state its metadata, and
(pgw#1031) the cell's recorded graph witness is not the graph this pod traced.
Every one of them means "boot as we booted yesterday", and every one of them is
a different sentence someone will need. Spreading that across an executor
branch is how the reason ends up being `logger.debug`.

**Never fatal.** A cold boot that cannot adopt is the boot every pod did before
this existed. The one thing this must never do is prevent a pod from serving.

Never SILENT either (pgw#1116)
------------------------------
"Never fatal" was implemented as "never observed". ``reason``/``detail``/
``derived_key``/``derive_ms`` had zero readers outside this module — the
executor consumed only ``adoption`` — and the executor's own pre-attempt gates
returned a bare ``None`` that carried no reason at all. So all of the refusals
above, plus three gates that never even built an outcome, presented identically
as "a pod that quietly self-minted". A merged fix (pgw#1108) and a published
wheel (0.103.0) both shipped over a boot-adopt that called the hub ZERO times
on three real pods, because off-pod there was nothing to read: hub-spawned pods
expose no stdout (pgw#760), so a `logger.info` is not observability.

Every terminus therefore emits ONE typed :data:`~gen_worker.activity.
KIND_BOOT_ADOPT` event whose ``phase`` is the gate's own token — see
:data:`REASONS`, which is exhaustive and fenced by test. "adopt failed" is not
an answer this module is able to give.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import activity, aot_identity, boot_key, cell_resolve, local_cell_store
from .child_contract import CompileSpec, MintSlot

logger = logging.getLogger(__name__)

#: Gates that refuse BEFORE :func:`attempt` is entered — the executor's own
#: preconditions. These are the ones that used to be a bare ``return None``.
GATE_REASONS: Tuple[str, ...] = (
    # `aot_mint.export_declaration(family)` answered None: this family ships no
    # export declaration, so no key can name its class set.
    "no_export_declaration",
    # The declaration exists but `aot_declaration.cell_plans` will not enumerate
    # it (colliding entry names, no targets).
    "declaration_unreadable",
    # pgw#1127: NOBODY could answer this key — no hub (no HelloAck base URL
    # yet, or an embedded single-process worker with neither a local bearer nor
    # a control seam, pgw#1108) AND this machine's own store holds no cell at
    # all. Only then is deriving pure boot latency. This REPLACES `no_hub`,
    # which asked about one of the two answerers and refused on behalf of both:
    # on exactly the machines §4.28 is about, the derived ck1 key IS the local
    # store's address, so "nobody will answer" was false there.
    "no_compiled_graph_source",
    # The pod's topology forbids arming at all (pgw#775), so a compiled family
    # boots without asking. Correct, and previously indistinguishable from a
    # boot-adopt that asked and was refused.
    "eager_only",
    # pgw#1142 / §4.32 item 4: an OPERATOR ordered this worker eager-only, so
    # the boot did not ask for a cell it is forbidden to call. Distinct from
    # `eager_only` above because that one is a property of the pod that no
    # command can lift, and this one is a decision that can be taken back —
    # the same pod, re-asked after the order is released, adopts normally.
    "operator_eager_only",
)

#: The reason token for :data:`GATE_REASONS`' operator entry. Named so the
#: executor references the vocabulary instead of re-typing the literal — the
#: drift channel `EagerPhase` was created to close, one module over.
OPERATOR_EAGER_ONLY = "operator_eager_only"

#: Step 1.5 (pgw#1127 S2) — THIS MACHINE's own store, addressed by the DERIVED
#: key, before the hub is asked at all. §4.28: *"local cell, local repo-CAS,
#: reused across its own boots — never uploaded, never requested."*
LOCAL_REASONS: Tuple[str, ...] = (
    # The machine holds this exact cell. No hub was asked, and on an offline
    # box no hub COULD be — which is the difference between fully-offline-
    # capable as a property and as an accident of the arm-token memo.
    "local_hit",
    # Derived, asked this machine's own store, and it does not hold this key —
    # and there is no hub to ask either. The honest successor to a `no_hub`
    # that used to fire BEFORE the derivation and therefore before the local
    # store had an address to be asked at.
    "local_miss_no_hub",
)

#: Step 1 — the derivation. ``boot_key.BootKeyUnavailable.reason`` tokens,
#: including the ones a trace child names in its own report
#: (``boot_trace_child._fail``), which ``boot_key.derive`` re-raises verbatim.
DERIVE_REASONS: Tuple[str, ...] = (
    "no_classes", "code_drift", "class_set_disagreement", "class_set_gap",
    "trace_failed", "bad_report", "child_died", "refused", "child_error",
    "slots_unresolvable", "structure_unsupported", "no_compile_target",
    # pgw#1123 — the OPPOSITE finding to `structure_unsupported`, and until
    # this token existed they were the same word. `structure_unsupported` says
    # ONE FAMILY is stranded (permanent and correct on the quantized artifact
    # lanes); this says THIS IMAGE cannot meta-instantiate at all, so every
    # family it serves derives no key, asks for no cell, and self-mints
    # forever — which is what two paid pods spent an evening looking like.
    "structure_capability_missing",
    "structure_not_honored", "no_declaration", "invalid_declaration",
    "trace_refused", "empty_share",
    # pgw#1124: the child's composition put REAL tensors off the host, so it
    # would be competing for VRAM with the parent that spawned it.
    "real_weights_resident",
    # The derivation raised something that is not a typed BootKeyUnavailable.
    "derive_failed",
)

#: Steps 2-3 — asking the hub, materializing, and the pgw#1031 witness floor.
#: The hub's own typed refusal codes ride through verbatim, which is the point
#: of them being typed.
ASK_REASONS: Tuple[str, ...] = (
    "invalid_request", "resolve_unreachable",
    "miss", "materialize_failed", "hit",
    # pgw#1122 — the LAST terminus, and the one the journey previously ran off
    # the end of. `hit` was reported at resolve+materialize, and everything
    # after it (the receipt gate, the publisher check, the arm itself) was
    # somebody else's exception. So three pods emitted `boot_adopt=hit` and
    # then failed their function, and the event that promised to name the gate
    # named the gate BEFORE the one that refused.
    "arm_refused",
) + tuple(cell_resolve.REFUSAL_CODES)

#: The COMPLETE boot-adopt vocabulary. A path that can produce a token missing
#: from here is a path that can be silent again, so
#: ``tests/test_boot_adopt_observability_pgw1116.py`` reads the refusal sites
#: out of the tree and fails on any token this tuple does not carry.
REASONS: Tuple[str, ...] = (
    GATE_REASONS + DERIVE_REASONS + LOCAL_REASONS + ASK_REASONS)

#: The hub answered and the pod will arm what it named. Everything except this
#: and :data:`LOCAL_HIT` means "boot as this pod booted yesterday", and each of
#: those means it for a DIFFERENT reason.
HIT = "hit"

#: THIS MACHINE answered, by the derived key, with no hub involved (pgw#1127).
#: The second outcome that leads to a compiled boot — and the only one that
#: does so on a box with no network at all.
LOCAL_HIT = "local_hit"


@dataclass(frozen=True)
class BootAdoption:
    """A cell this boot resolved by its OWN derived key, ready to arm."""

    derived: boot_key.DerivedKey
    cell: cell_resolve.ResolvedCell
    artifact: Path

    @property
    def expected(self) -> aot_identity.ExpectedIdentity:
        return self.cell.expected_identity()

    @property
    def ref(self) -> str:
        return self.cell.cell_ref

    @property
    def snapshot_digest(self) -> str:
        return self.cell.transport.snapshot_digest


@dataclass(frozen=True)
class BootAdoptOutcome:
    """What the boot-adopt attempt did — always a complete answer.

    ``adoption`` set means arm it. ``adoption`` None with a ``reason`` means
    boot exactly as this pod booted before the seam existed, and the reason is
    the countable token for why.

    ``local_key`` (pgw#1127) is the one exception to that second sentence, and
    it is deliberately NOT an ``adoption``: a cell out of THIS MACHINE's own
    store carries no hub receipt and no publisher org, so it must not ride the
    ``arm_ordered`` path a hub-resolved cell rides — it arms through
    ``fleet_cells._arm_exported_cell``, the gate every cell this machine
    produced for itself passes. What the boot hands the arming brain is the
    ADDRESS, and the arming brain does the lookup on the same terms it does a
    memo lookup. Two routes, one CAS, one gate.
    """

    adoption: Optional[BootAdoption] = None
    reason: str = ""
    detail: str = ""
    derived_key: str = ""
    derive_ms: int = 0
    #: The ``ck1`` key THIS MACHINE's own store answered on (pgw#1127). "" on
    #: every other terminus. Equal to ``derived_key`` when set — carried
    #: separately so a reader never has to infer "and the local store had it"
    #: from a key that is also present on a miss.
    local_key: str = ""
    #: Which family and which routable function this decision was about. Carried
    #: on the outcome rather than interpolated at the emit site so a reader of
    #: the event never has to join it back to anything (pgw#1116).
    family: str = ""
    function: str = ""

    @property
    def adopted(self) -> bool:
        return self.adoption is not None


def report(outcome: BootAdoptOutcome) -> BootAdoptOutcome:
    """Emit this decision as ONE typed event, and return it unchanged.

    The whole of pgw#1116: on a hub-spawned pod stdout goes nowhere, so a
    boot-adopt outcome that is only logged is a boot-adopt outcome nobody can
    read. ``phase`` is the gate token — never a summary — because the question
    the fleet asks is *which* gate, and "adopt failed" makes eight different
    bugs look like one.

    Returns the outcome so every terminus can be ``return report(...)`` and the
    "exactly one event per decision" property is structural rather than
    remembered.
    """
    detail = (
        f"family={outcome.family or '?'} function={outcome.function or '?'} "
        f"key={outcome.derived_key or '-'}"
        + (f" — {outcome.detail}" if outcome.detail else "")
    )
    activity.emit_event(
        activity.KIND_BOOT_ADOPT, detail, phase=outcome.reason or "unreported",
        duration_ms=outcome.derive_ms)
    if outcome.reason in (HIT, LOCAL_HIT):
        logger.info("boot-adopt[%s]: %s", outcome.reason, detail)
    else:
        # A refusal is a confession, and a pod's own boot log is where an
        # operator with a shell reads it. The EVENT is what everyone else reads.
        logger.warning("boot-adopt[%s]: %s", outcome.reason, detail)
    return outcome


def refused(
    reason: str, detail: str, *, family: str = "", function: str = "",
) -> BootAdoptOutcome:
    """A gate that refuses BEFORE :func:`attempt` — reported, never a bare None.

    The executor's three pre-attempt gates used to ``return None``, which is
    the one shape that cannot say anything: no reason, no detail, no event, and
    a caller that could not tell "this pod had nobody to ask" from "this pod
    asked and was told no".
    """
    return report(BootAdoptOutcome(
        reason=str(reason), detail=str(detail),
        family=str(family), function=str(function)))


def arm_refused(
    outcome: BootAdoptOutcome, *, cause: str, detail: str,
) -> BootAdoptOutcome:
    """The adopted cell would not ARM (pgw#1122) — degrade, don't die.

    ``outcome`` is the ``hit`` this journey already reported, so the event
    carries the same family/function/key and joins to it on one query; ``cause``
    is the arm's own typed token (``OrderedArmError.reason``:
    ``artifact_receipt_refused``, ``publisher_mismatch``, …) and leads the
    detail so ``?kind=boot_adopt&phase=arm_refused`` answers *which gate*
    without a second lookup.

    Why this is a boot-adopt terminus and not the arm's business: nothing but
    this pod named the cell. A Plan-ordered arm is terminal by design (pgw#904 —
    the hub named one exact artifact and a substitute would not be it), but a
    cell this pod pulled by its OWN derived key has no order behind it, so its
    refusal means "boot as this pod booted yesterday" like every other refusal
    in this module.
    """
    return report(BootAdoptOutcome(
        adoption=None,
        reason="arm_refused",
        detail=(
            f"cause={cause}: {detail} — this pod resolved the cell by its own "
            f"derived key, so nothing ordered this arm; serving EAGER and "
            f"minting its own instead of failing the function"),
        derived_key=outcome.derived_key,
        derive_ms=outcome.derive_ms,
        family=outcome.family,
        function=outcome.function,
    ))


def no_compiled_graph_source(hub_absent: str) -> bool:
    """True when NOBODY could answer a derived key, so deriving is pure latency.

    The honest form of the gate pgw#1127 §1b found. The old one asked *"is
    there a hub"* and refused on behalf of both answerers; this asks whether
    EITHER exists. ``stored_cells`` is a listdir with no digest recomputation
    (pgw#1096), so the fleet path — whose store is always empty — pays exactly
    what the old early return paid, and the machines §4.28 is about stop being
    told there is nobody to ask when they are holding the answer.
    """
    if not hub_absent:
        return False
    try:
        return not local_cell_store.stored_cells()
    except Exception:  # noqa: BLE001 — an unreadable store is an absent one
        logger.debug("boot-adopt: local store unreadable", exc_info=True)
        return True


def _local_answer(
    key: str, derived: "boot_key.DerivedKey", *, family: str, fn: str,
) -> Optional[BootAdoptOutcome]:
    """This machine's own store, asked by the derived key. ``None`` = ask on.

    A hit is NOT returned as an ``adoption``: a self-minted cell carries no hub
    receipt and no publisher org, so the ``arm_ordered`` path a hub-resolved
    cell rides would refuse it ``receipt_gate_unconfigured``. What is returned
    is the ADDRESS, on ``local_key``, for ``fleet_cells.arm_from_local_store``
    to arm through the same gate a child's fresh mint passes.

    The derived TCG key already commits the graph class and the TCG store admits
    only bytes whose metadata reproduces that key. There is no second worker
    witness to compare.
    """
    try:
        cell = local_cell_store.lookup(key)
    except Exception as exc:  # noqa: BLE001 — a cache read is never fatal
        logger.debug("boot-adopt: local store lookup failed", exc_info=True)
        logger.warning("boot-adopt: local store unreadable (%s)", exc)
        return None
    if cell is None:
        return None
    return report(BootAdoptOutcome(
        reason="local_hit",
        detail=(
            f"THIS MACHINE minted {key} on an earlier boot and kept it "
            f"({cell.bytes / 1e6:.1f} MB); it arms from disk with no mint, no "
            f"hub and no network (§4.28)"),
        derived_key=key, local_key=key, derive_ms=derived.wall_ms,
        family=family, function=fn))


def attempt(
    *,
    function: str,
    modules: Sequence[str],
    cfg: Any,
    slots: Mapping[str, MintSlot],
    declared_hint: int,
    work_root: Path,
    memo_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    base_url: str = "",
    bearer: str = "",
    device: int = -1,
    hub_absent: str = "",
) -> Tuple[BootAdoptOutcome, ...]:
    """Derive the KEY SET, then for each key ask THIS MACHINE, ask the hub and
    materialize. Never raises.

    pgw#1176: returns ONE outcome per declared graph class. A derivation
    failure is a single-element tuple, because it is the one failure that is
    genuinely about the declaration rather than about a class.

    ``declared_hint`` is ``len(aot_declaration.cell_plans(decl))`` — it sizes
    the trace pool and nothing else.

    ``hub_absent`` (pgw#1127) is the caller's own sentence for why there is
    nobody to ask over the wire, "" when there is. It is a DETAIL, not a
    decision: the decision is made here, after the local store has been asked,
    because the derived key is an address this machine can answer at.

    Exactly ONE :data:`~gen_worker.activity.KIND_BOOT_ADOPT` event is emitted
    per call, on every terminus including both hits — see :func:`report`.
    """
    family = str(getattr(cfg, "family", "") or "")
    fn = str(function)
    spec = CompileSpec(
        shapes=tuple(
            tuple(int(v) for v in row)
            for row in (getattr(cfg, "shapes", ()) or ())),
        targets=tuple(str(t) for t in (getattr(cfg, "targets", ()) or ())),
        family=family,
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )

    try:
        derived = boot_key.derive(
            function=str(function),
            modules=tuple(str(m) for m in modules),
            family=family,
            cfg=spec,
            slots=dict(slots),
            declared_hint=int(declared_hint),
            work_root=Path(work_root),
            memo_dir=Path(memo_dir) if memo_dir else None,
            device=int(device),
        )
    except boot_key.BootKeyUnavailable as exc:
        # The reason is a trace child's own token when a child refused
        # (`boot_trace_child._fail`) — `structure_unsupported` and
        # `slots_unresolvable` are the two that name a family the boot path
        # cannot key at all, and they are the whole point of not collapsing
        # this to "derive failed".
        return (report(BootAdoptOutcome(
            reason=exc.reason, detail=exc.detail,
            family=family, function=fn)),)
    except Exception as exc:  # noqa: BLE001 — never fatal, see the docstring
        logger.debug("boot-adopt: key derivation failed", exc_info=True)
        return (report(BootAdoptOutcome(
            reason="derive_failed", detail=f"{type(exc).__name__}: {exc}",
            family=family, function=fn)),)

    # ── pgw#1176: a boot derives a KEY SET, and asks per key ──────────────
    #
    # §4.27 ruled "THE cell key" (singular). It is a derived SET plus a
    # manifest now, and every property that ruling wanted survives
    # STRENGTHENED, because a PARTIAL result is useful: a pod that resolves 30
    # of 36 keys arms 30 classes and compiles 6, where the cell key made that
    # same outcome a total miss and a full re-mint.
    #
    # One outcome PER CLASS, in sorted key order. There is deliberately no
    # combined verdict — the object that could report one was the wrong atom,
    # and a "the boot adopted" boolean over 36 classes is exactly the claim
    # that could lie.
    #
    # pgw#1224 lands PHASE 2's batch wire HERE: this machine is asked per key
    # (a local stat, no wire) and the hub is asked ONCE for everything the
    # machine did not already hold. A 36-class sdxl boot went from 36 resolve
    # round trips to one; minimax-h3 from 198 to one.
    keys = derived.keys
    if not keys:
        return (report(BootAdoptOutcome(
            reason="derive_failed",
            detail="the declaration traced to no graph class at all",
            derive_ms=derived.wall_ms, family=family, function=fn)),)
    logger.info(
        "boot-adopt: %s asking for %d key(s) (%d class(es), "
        "K=%d, memo=%s, derived in %d ms)",
        family, len(keys), len(derived.entry_keys), derived.workers,
        derived.memo, derived.wall_ms)

    # ── §4.28 / pgw#1127: THIS MACHINE, before the wire ────────────────────
    # The derived key is the local store's own address. A machine that minted
    # this graph on an earlier boot answers here and the hub is never asked for
    # it — which is what "never requested" means and what makes an offline box
    # arm exactly as well as an online one. Costs one stat per key on the fleet
    # path, where the store is always empty.
    settled: Dict[str, BootAdoptOutcome] = {}
    ask: List[str] = []
    for key in keys:
        local = _local_answer(key, derived, family=family, fn=fn)
        if local is not None:
            settled[key] = local
        elif hub_absent:
            # Derived, asked this machine, and there is nobody else. A
            # DIFFERENT fact from `no_compiled_graph_source` (which never derived) and
            # from `miss` (which asked a hub): this one says the key exists and
            # nothing here holds it.
            settled[key] = report(BootAdoptOutcome(
                reason="local_miss_no_hub", detail=hub_absent,
                derived_key=key, derive_ms=derived.wall_ms,
                family=family, function=fn))
        else:
            ask.append(key)

    answers: Dict[str, cell_resolve.ResolveAnswer] = {}
    refused: Tuple[str, str] = ("", "")
    if ask:
        try:
            for answer in cell_resolve.resolve_batch(
                    family, ask, base_url=base_url, bearer=bearer):
                answers[answer.compiled_graph_key] = answer
        except cell_resolve.CellResolveRefused as exc:
            # A WHOLE-BATCH refusal is a fact about the caller or the request,
            # so it is true of every key in the batch and each one reports it
            # under the hub's own code. A typed refusal is NOT a miss — reading
            # it as one sends the pod to a full cold mint per key believing the
            # hub holds nothing, which is false and expensive. It still
            # degrades to the old boot; it is the SENTENCE that differs, and
            # that sentence is what gets the hub fixed.
            refused = (exc.code or "resolve_unreachable", exc.detail)
        except Exception as exc:  # noqa: BLE001
            logger.debug("boot-adopt: resolve call failed", exc_info=True)
            refused = ("resolve_unreachable", f"{type(exc).__name__}: {exc}")

    out: List[BootAdoptOutcome] = []
    for key in keys:
        done = settled.get(key)
        if done is not None:
            out.append(done)
            continue
        if refused[0]:
            out.append(report(BootAdoptOutcome(
                reason=refused[0], detail=refused[1], derived_key=key,
                derive_ms=derived.wall_ms, family=family, function=fn)))
            continue
        out.append(_adopt_answer(
            answers[key], derived, family=family, fn=fn, cache_dir=cache_dir))
    return tuple(out)


def _adopt_answer(
    answer: cell_resolve.ResolveAnswer,
    derived: "boot_key.DerivedKey",
    *,
    family: str,
    fn: str,
    cache_dir: Optional[Path],
) -> BootAdoptOutcome:
    """Turn ONE of the batch's answers into this graph class's outcome.

    A miss, a per-answer hub fault, or a witness mismatch here costs THIS graph
    class. Its siblings were answered by their own keys in the same batch and
    are not affected — which is what makes a partial hit useful instead of a
    total miss, and it is the property the per-answer signing buys: nothing
    about this answer's trust depends on the collection it arrived in.
    """
    key = answer.compiled_graph_key
    if answer.refusal_code:
        # ambiguous / incomplete / transport_unavailable — a HUB-SIDE fault
        # about this one key, carrying the code it carried when the same fault
        # refused a whole request. Not a miss, and the pod must not mint over
        # it.
        return report(BootAdoptOutcome(
            reason=answer.refusal_code, detail=answer.detail,
            derived_key=key, derive_ms=derived.wall_ms,
            family=family, function=fn))
    cell = answer.cell
    if cell is None:
        # A pod that DERIVED a key and was told MISS is a different fact from a
        # pod that never derived one — `derived_key` is what tells them apart,
        # and it is on the event.
        return report(BootAdoptOutcome(
            reason="miss",
            detail=f"the hub holds no entitled compiled graph for {key}",
            derived_key=key, derive_ms=derived.wall_ms,
            family=family, function=fn))

    try:
        artifact = cell_resolve.materialize(
            cell, cache_dir=Path(cache_dir) if cache_dir else None,
            what=f"boot adopt of {key} (family {family})")
    except Exception as exc:  # noqa: BLE001
        logger.debug("boot-adopt: materialize failed", exc_info=True)
        return report(BootAdoptOutcome(
            reason="materialize_failed", detail=f"{type(exc).__name__}: {exc}",
            derived_key=key, derive_ms=derived.wall_ms,
            family=family, function=fn))

    return report(BootAdoptOutcome(
        adoption=BootAdoption(derived=derived, cell=cell, artifact=artifact),
        reason=HIT,
        detail=(
            f"resolved to {cell.cell_ref} ({cell.publisher_tier} tier), "
            f"materialized at {artifact} — arming with ZERO dispatches having "
            f"occurred"),
        derived_key=key, derive_ms=derived.wall_ms,
        family=family, function=fn))


__all__ = [
    "ASK_REASONS", "BootAdoptOutcome", "BootAdoption", "DERIVE_REASONS",
    "GATE_REASONS", "HIT", "LOCAL_HIT", "LOCAL_REASONS", "REASONS",
    "arm_refused", "attempt", "no_compiled_graph_source", "refused", "report",
]
