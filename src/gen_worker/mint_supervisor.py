"""th#1834 Phase 3 (pgw#1215 step 4): the serving parent supervises its own
compile children, per graph class.

    "...if miss then begin compiling and serve eager until the local cell is
    available; then switch over to using that."

**Two tiers, not three.** What stood here was a nested stack: the serving
parent spawned ONE mint child (``mint_delegate.build_cell`` ->
``mint_process.run_mint`` -> ``mint_child``), which loaded a second
weight-free pipeline and then itself drove the K-wide compile pool. The
middle tier existed to hold a pipeline the parent already holds a real copy
of, and to buy th#1299's property — *"no inductor on the serving loop"* —
with a whole extra process. th#1299's defect was inductor's GIL-holding
orchestration starving the one asyncio task that carries the 10 s beat and
eager serving; spawn / poll / collect / digest / CAS-write / publish are I/O
and supervision and do not starve it. So the property is bought by a
structural fence instead (``scripts/lint_serving_process_compiles.py``) and
the middle tier is gone.

What this module owns, and it is all supervision:

* the parent-side gates only a process holding the LIVE pipeline can run —
  the declared-blocker refusal (:class:`DeclaredBlockerRefusal`) and the class
  ENUMERATION (the adapter fork is decided by the composed pipeline);
* driving :func:`aot_mint.mint_graph_classes` off the event loop, one
  ``asyncio.to_thread`` per attempt, with the abandon signal reaching the
  children rather than only the task;
* the ACCRETION loop: an attempt consumes what earlier attempts already
  packed, so a retry pays neither the trace nor the compile for a class this
  pod already holds;
* adopting what came back through the ordinary delivered-cell path
  (``fleet_cells.adopt_delegated_mint``: pack -> local CAS -> verify -> arm ->
  async publish, per artifact, pgw#1183);
* the parent-side telemetry the child cannot emit — it holds no orchestrator
  session, so its phase table, its per-class trace rows and its device-peak
  census all have to be re-emitted from here.

What it does NOT own: deciding a mint is owed
(``fleet_cells.enable_compiled``), the ``self_mint_compile`` Activity's
lifetime, and advertising the adopted identity on the live compile targets.
Those stay in ``executor.py``, which is where the decision and the targets
live.

There is no ``MintOutcome`` and no exit code any more. A mint is not a
project with one terminus; coverage ACCRETES per graph class, and the
per-class outcomes are the artifacts themselves.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import msgspec

from . import activity as activity_mod
from . import artifact_meta
from . import boot_phases
from . import graph_facts
from . import compile_posture
from . import mint_workers
from . import progress as progress_mod
from .api.export_contract import (
    blocker_refusal, export_declaration, open_blockers)
from .child_contract import CompileSpec, MintFrame, MintSlot

logger = logging.getLogger(__name__)

ADOPTED = "adopted"
FAILED = "failed"
ABANDONED = "abandoned"

#: How many times the supervisor re-enters the accretion loop for one
#: obligation. Each pass is cheaper than the last by construction — it skips
#: every class already packed — so this bounds the number of times a
#: RETRYABLE shortfall (a card that could not hold K children) gets a
#: narrower attempt, not the amount of work re-done.
MAX_ATTEMPTS = 3

#: The directory this record's packed graph classes accrete in, under the
#: pending's own mint root. ATTEMPT-STABLE, and that is the whole of the
#: priced-resume fix: ``mint_delegate.build_cell`` gave every attempt a fresh
#: ``child-N`` directory, so attempt 2 of a 36-class mint re-paid 35 finished
#: classes to retry one.
GRAPH_DIRNAME = "graphs"


class DeclaredBlockerRefusal(RuntimeError):
    """This family declares an OPEN mint blocker (pgw#1115).

    Deliberately NOT ``MintRefused``: that name is live in ``aot_contract``
    and the atom raises it for *"every declared class refused"*
    (``aot_mint``, "absence is a verdict"). Both are terminal, so sharing the
    name would make the two indistinguishable at the terminus with no test
    going red and tensorhub's blame ladder attributing them identically.

    A blocked family has exactly one legal outcome — serve eager, mint
    nothing — so there is no fallback to take. Enforced HERE, on the parent,
    because the parent is now the process that decides to mint: the gate used
    to sit in ``mint_child`` and a reroute that dropped it would publish a
    cell for a class set the declaration says it cannot yet claim.
    """


@dataclass(frozen=True)
class MintTask:
    """Everything the serving process knows that a mint of one obligation
    needs.

    ``pipe`` is the LIVE serving pipeline. It is read, never handed across a
    process boundary: the supervisor enumerates the declared graph classes
    off it (the adapter fork is a property of the composed pipeline) and
    ``fleet_cells.adopt_delegated_mint`` arms the finished artifacts onto it.
    Each compile child composes its own weight-free target from ``modules`` /
    ``slots`` / ``cfg``.
    """

    pending: Any                      # fleet_cells.PendingSelfMint
    pipe: Any
    function: str
    modules: Tuple[str, ...]
    #: pgw#974: the parent's resolution of each setup slot — identity, bytes
    #: and pgw#617 composition in ONE value per slot.
    slots: Dict[str, MintSlot] = field(default_factory=dict)
    weight_lane: str = ""
    execution_lane: str = ""
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    device: Optional[int] = None
    #: §4.30 / pgw#1137: whose machine this mint runs on. DECLARED by the
    #: caller rather than read off a process global, so the one entry that
    #: knows (``local_serve``) states it and every other caller gets ``FLEET``
    #: by construction.
    posture: compile_posture.CompilePosture = compile_posture.FLEET
    #: pgw#1199: how this process proved the endpoint's handler RUNS, on the
    #: resident pipeline. The children allocate nothing for it.
    handler_proof: str = ""


@dataclass(frozen=True)
class SupervisedResult:
    """The outcome of supervising one obligation's mint."""

    status: str
    detail: str = ""
    minted: Optional[Any] = None      # fleet_cells.SelfMint
    attempts: int = 0
    #: pgw#999: the CLASSIFIED reason the artifacts did not adopt, carried up
    #: so the executor's decline names the same token the abort event did.
    reason: str = ""
    #: How many graph classes this pod holds for the obligation, and how many
    #: the declaration has. Coverage ACCRETES, so a mint has no single
    #: pass/fail size — it has a fraction.
    covered: int = 0
    declared: int = 0

    @property
    def ok(self) -> bool:
        return self.minted is not None


def delegation_refusal() -> str:
    """"" when this WORKER may mint, else the typed reason.

    Always "" today: nothing worker-wide can refuse a supervised mint. The
    seam survives because the PIPELINE half
    (``fleet_cells.delegation_refusal`` — an armed non-eager backend with no
    eager tier to serve from) is a real per-pipe refusal, and both halves must
    reach ``mint_recipe`` as one typed reason rather than as an either/or
    sentence (pgw#813).
    """
    return ""


def cfg_spec(cfg: Any) -> CompileSpec:
    """Flatten the parent's ``CompileCell`` for the child job.

    The parent states the contract because the class-scoped guidance/text-len
    unions live on the spec rather than the decorator: a child re-deriving
    this from ``@endpoint`` alone would export a different declaration than
    the parent asked for. It carries what the child READS and nothing else
    (pgw#1034) — the child computes no key, so this is not a key-parity wire.
    """
    return CompileSpec(
        shapes=tuple(tuple(int(v) for v in row) for row in (cfg.shapes or ())),
        targets=tuple(str(t) for t in (cfg.targets or ())),
        family=str(getattr(cfg, "family", "") or ""),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )


def assert_family_mintable(family: str) -> None:
    """Refuse outright while the family declares an open blocker (pgw#1115).

    pgw#1080's pattern: split the refusal by TYPE and fail closed rather than
    degrading.
    """
    if not family:
        return
    try:
        decl = export_declaration(family)
    except Exception as exc:  # noqa: BLE001 — a refusing declaration refuses
        raise DeclaredBlockerRefusal(
            f"family {family!r}'s export declaration refuses to build "
            f"({type(exc).__name__}): {exc}") from exc
    if decl is None:
        return
    blocked = open_blockers(decl)
    if blocked:
        raise DeclaredBlockerRefusal(blocker_refusal(family, blocked))


def rule_on_boot_memo(
    task: MintTask, cfg: Any, result: Any, *, declared: int,
) -> str:
    """THE honesty gate (pgw#1271): what the boot memo answered, against what
    this mint TRACED. Returns the disagreement, or ``""``.

    ``boot_key``'s memo path SKIPS THE TRACES and re-folds stored blocks into
    the ``graph`` axis of every ``cg-key-v1`` this pod goes on to publish — so
    the memo is a key generator, and the only thing standing between a stale
    memo and a wrong published key is a comparison against a freshly traced
    truth. ``boot_key.assert_memo_honest`` is that comparison. It had **zero
    src/ callers**: it was written, documented as the thing that makes the memo
    safe to have at all, exported, tested — and never invoked on a fleet pod.

    THIS is the moment, and there is only one: a mint is the single point in a
    pod's life where it holds a traced class set for the same closure the boot
    memoized. So the call sits at the seal/publish seam, before the artifacts
    ship — and the offending memo entry is invalidated by the callee, so the
    next boot re-traces instead of re-reading a hash proven wrong.

    Two things are deliberately NOT ruled on, because ruling on them would
    manufacture disagreements:

    * a PARTIAL class set. Coverage accretes (pgw#1176), so a mint that packed
      fewer classes than the declaration has cannot distinguish "the memo holds
      a class we did not trace" from "we have not traced it yet", and
      ``assert_memo_honest``'s set check would fire on every partial mint.
    * a mint whose entries carry no keying block — nothing to compare.

    Returns the reason rather than raising: a dishonest memo has already cost
    what it can cost (the entry is invalidated, the next boot re-traces), and
    the artifacts in hand were traced by THIS process and are correct. Killing
    a finished mint over it would turn a self-healing fault into a lost pod.
    """
    from . import boot_key

    memo_dir = getattr(task.pending, "cache_dir", None)
    if not memo_dir:
        return ""
    entries = tuple(getattr(result, "entries", ()) or ())
    if declared <= 0 or len(entries) != int(declared):
        return ""
    blocks: Dict[str, Dict[str, Any]] = {}
    for row in entries:
        block = dict(getattr(row, "metadata", None) or {}).get(
            graph_facts.TCG_GRAPH_CLASS_BLOCK)
        if not isinstance(block, Mapping):
            return ""
        blocks[str(row.entry)] = dict(block)
    if len(blocks) != len(entries):
        return ""
    digest = boot_key.closure_digest(
        str(getattr(cfg, "family", "") or ""),
        cfg_spec(cfg),
        function=str(task.function or ""),
        slots=dict(task.slots),
    )
    return boot_key.assert_memo_honest(Path(memo_dir), digest, blocks)


def _rule_on_boot_memo(
    task: MintTask, cfg: Any, result: Any, *, declared: int, family: str,
) -> None:
    """Run :func:`rule_on_boot_memo` and report it as a TYPED event.

    Never raises: an unrulable memo is silence, and a dishonest one is a
    countable wire fact. A log line would be neither — a hub-spawned pod's
    stdout goes nowhere (pgw#760).
    """
    try:
        reason = rule_on_boot_memo(task, cfg, result, declared=declared)
    except Exception as exc:  # noqa: BLE001 — never die with the mint
        activity_mod.emit_event(
            activity_mod.KIND_BOOT_MEMO,
            f"family={family}: the boot-key memo could not be ruled on "
            f"({type(exc).__name__}: {exc}); this mint's keys are the TRACED "
            f"ones and are unaffected",
            phase="memo_unrulable")
        return
    if not reason:
        return
    logger.error("mint-supervisor: %s", reason)
    activity_mod.emit_event(
        activity_mod.KIND_BOOT_MEMO, f"family={family}: {reason}",
        phase="memo_dishonest")


#: The mint's evidence counter, in the `family:name` shape every other counter
#: uses. Plain `mint_child_evidence` split on ":" in `progress.window_for` and
#: fell back to the generic 300 s window instead of the 600 s the `compile`
#: family declares for exactly this work (pgw#1157).
EVIDENCE_COUNTER = "compile:mint_child_evidence"

#: pgw#1137: a sink for progress frames that is NOT the hub. The fleet passes
#: nothing; ``local_serve`` passes a terminal renderer, because a 20-minute
#: compile a user cannot see is a support ticket regardless of how correct it
#: is.
Watcher = Callable[[MintFrame], None]


#: pgw#1243: the mint's BOUNDED axis — whatever the current phase is counting,
#: out of the total it declares (shares landed during `inductor_compile`, and
#: every other phase that states a total).
#:
#: ``EVIDENCE_COUNTER`` counts against no total at all, which is what let the
#: wedge hide: two mints reported `counter_done=1077.99 counter_total=0
#: self_stalled=FALSE` for an hour, and with no total there is no fraction, so
#: a frozen position is indistinguishable from a slow one. The frames have
#: carried a real `step/total` all along; it just never reached a COUNTER,
#: which is the only thing the hub's liveness rule and
#: ``progress.self_diagnosis`` read. An unbounded counter is not progress
#: evidence; this one is.
PROGRESS_COUNTER = "compile:mint_progress"


def _on_frame(act: Any, watch: Optional[Watcher] = None) -> Any:
    """Land one supervision frame on the activity the hub already reads."""

    def _apply(frame: MintFrame) -> None:
        if frame.phase:
            act.phase(frame.phase, frame.step, frame.total)
            _count_progress(act, frame)
        if frame.note:
            act.note(frame.note[:200])
        if watch is not None:
            watch(frame)

    return _apply


def _count_progress(act: Any, frame: MintFrame) -> None:
    """Carry a frame's step/total onto a real counter (best effort).

    The activity's own `step`/`total_steps` already ride the wire, but the
    hub's liveness rule and ``progress.self_diagnosis`` both read COUNTERS —
    which is why a mint whose shares had stopped landing still confessed
    ``self_stalled=FALSE``.
    """
    if frame.total <= 0:
        return
    make = getattr(act, "counter", None)
    if not callable(make):
        return
    make(PROGRESS_COUNTER, progress_mod.UNIT_STEPS,
         float(frame.total)).set_done(float(frame.step))


def _on_evidence(act: Any) -> Any:
    """pgw#824/pgw#1157: MEASURED progress, as a registered progress COUNTER.

    Registering a counter means the hub's liveness rule judges an advancing
    NUMBER rather than inferring health from a re-sent phase — which is what
    the no-magic-timeouts doctrine asks for: liveness plus progress-staleness,
    never a fixed duration.

    RE-ACQUIRED per tick, never captured. ``Activity.phase()`` finishes every
    counter the new phase does not own (pgw#962) and ``finish()`` deletes it
    from the process registry, so a captured Counter goes unreadable at the
    first phase change. Measured on RunPod A40 ``bgmdxhazxsugmk`` (0.112.0):
    62 minutes in ``trace_graph`` with no counter, no rate and no
    self-diagnosis possible, over a mint that was in fact advancing.
    """
    make = getattr(act, "counter", None)
    if not callable(make):
        # Telemetry never breaks the work it reports on: an Activity double
        # with no counter registry still gets the heartbeat, which is the half
        # the hub's liveness rule actually reads.
        def _beat_only(value: float) -> None:
            act.heartbeat()
        return _beat_only

    def _apply(value: float) -> None:
        make(EVIDENCE_COUNTER, progress_mod.UNIT_EVIDENCE).set_done(float(value))
        act.heartbeat()

    return _apply


def _pool_stat(phases: Any, key: str) -> int:
    """One measurement out of a mint's phase table ``pool`` block, or 0."""
    try:
        block = (phases or {}).get("pool")
        return int((block or {}).get(key) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _bank_device_peaks(phases: Any, *, weight_lane: str) -> int:
    """Bank every per-graph-class device reading in one phase table.

    Returns how many rows were banked, so a caller can tell "no rows" from
    "banked nothing" — an absent measurement is NO EVIDENCE, not a zero
    (§4.33), and the two must not look alike here either.

    Called on EVERY outcome, from the result AND from the snapshot: the
    attempt that ran out of device memory is the one whose reading the next
    attempt most needs, and it is exactly the attempt that seals nothing and
    writes no manifest. A bank whose writer only runs on success is not a
    bank.

    ``weight_lane`` is the parent's own fact — the same axis it keys the RSS
    bank by — and the one axis the child does not state, because the parent
    knows it first-hand.
    """
    try:
        block = (phases or {}).get("pool") or {}
        rows = block.get("entry_device_peaks") or {}
        prov = block.get("device_peak_provenance") or {}
        if not isinstance(rows, dict) or not isinstance(prov, dict):
            return 0
    except (TypeError, AttributeError):
        return 0
    banked = 0
    for graph_class, reading in rows.items():
        if not isinstance(reading, dict):
            continue
        try:
            key = mint_workers.DevicePeakKey(
                graph_class=str(graph_class),
                card=str(prov.get("card") or ""),
                sm=str(prov.get("sm") or ""),
                toolchain=str(prov.get("toolchain") or ""),
                gen_worker=str(prov.get("gen_worker") or ""),
                weight_lane=str(weight_lane or ""),
                phase=str(prov.get("phase") or ""),
            )
            mint_workers.record_entry_device_peak(
                key,
                int(reading.get("allocated_bytes") or 0),
                int(reading.get("reserved_bytes") or 0))
        except (TypeError, ValueError):
            continue
        banked += 1
    return banked


def _emit_boot_trace_rows(table: Mapping[str, Any], *, family: str) -> None:
    """pgw#1087: the mint's PER-CLASS timings, as boot-phase rows.

    The compile children hold no orchestrator session, so their per-class
    trace cost is joinable by nothing until it is re-emitted here, parent-side,
    where the stream is — which puts one ``trace_for_key`` row per graph CLASS
    in the same table as the fetch and admission phases.

    Emitted whether or not the boot window is still open: a self-mint
    routinely finishes after ``first_request_servable``, and its per-class
    costs are the point. They are spans, so ``reconciliation`` charges them
    honestly — a mint that ran after the boot closed makes ``measured_ms``
    exceed ``total_ms``, which is a TRUE statement about a worker that kept
    compiling after it went servable, not a broken ladder.
    """
    entries = table.get("entries")
    if not isinstance(entries, Mapping):
        return
    for name, timings in sorted(entries.items()):
        if not isinstance(timings, Mapping):
            continue
        export_s = float(timings.get("export_s") or 0.0)
        if export_s <= 0:
            continue
        nodes = timings.get("nodes")
        boot_phases.mark(
            boot_phases.PHASE_TRACE_FOR_KEY,
            duration_ms=int(round(export_s * 1000.0)),
            ref=family,
            function=str(name),
            detail=(
                f"nodes={nodes} " if nodes is not None else ""
            ) + f"compile_s={timings.get('compile_s') or 0} "
                f"warm_s={timings.get('warm_s') or 0}",
        )
    totals = table.get("totals")
    if isinstance(totals, Mapping):
        declare_s = float(totals.get("declare_s") or 0.0)
        if declare_s > 0:
            boot_phases.mark(
                boot_phases.PHASE_KEY_FOLD,
                duration_ms=int(round(declare_s * 1000.0)),
                ref=family,
                detail=f"classes={len(entries)}",
            )


def _emit_aot_phases(
    table: Mapping[str, Any], *, family: str, execution_lane: str,
    terminus: str, elapsed_s: float,
) -> None:
    """pgw#805: one supervised mint's phase table, emitted PARENT-side.

    ``aot_mint`` builds the table, but the process that ran the compile holds
    no orchestrator session, so the event reaches nothing from there. This is
    the process that owns the connection and the 10 s beat.

    A mint that produced NO artifact still reports its total: the seconds are
    real and worth recording, and they must not enter an AOT-vs-JIT comparison
    as if a compiled graph came out. ``terminus`` distinguishes ABANDONED — a
    co-tenancy decision destroyed a working mint — from ``aborted``; a roll-up
    that calls the first the second hides the only actionable fact in the row.
    """
    from . import aot_mint

    try:
        rows = dict(table)
        if rows and terminus and elapsed_s > 0:
            rows["totals"] = {
                **dict(rows.get("totals") or {}),
                "child_elapsed_s": round(float(elapsed_s), 2),
            }
        if rows:
            rows["terminus"] = terminus or rows.get("terminus") or ""
            aot_mint.emit_phase_events(
                family=family, execution_lane=execution_lane, table=rows,
                terminus=terminus)
            _emit_boot_trace_rows(rows, family=family)
            return
        if elapsed_s <= 0:
            return
        activity_mod.emit_event(
            activity_mod.KIND_AOT_MINT,
            f"family={family} lane={execution_lane or 'plain'} "
            f"status={terminus or 'aborted'} "
            f"total_s={round(float(elapsed_s), 2)} — no compiled graph "
            f"produced",
            phase="aborted",
            duration_ms=int(round(float(elapsed_s) * 1000)),
        )
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("mint-supervisor: phase event emission failed",
                     exc_info=True)


def _emit_abort(
    *, family: str, key: str, attempt: int, status: str, detail: str,
    retryable: bool,
) -> None:
    """A failed mint is hub-relevant truth, not a pod-log line.

    A serve pod exposes no logs (pgw#760), so the classification and the last
    words all have to ride the wire or the next person debugging this has
    nothing.
    """
    activity_mod.emit_event(
        "self_mint_abort",
        f"family={family} key={key}: the supervised mint ended {status} on "
        f"attempt {attempt} "
        f"({'retryable' if retryable else 'deterministic'}) — this worker "
        f"kept serving eager throughout: {detail[:600]}",
        phase=f"supervised_{status}",
    )


def graph_dir(pending: Any) -> Path:
    """Where this obligation's packed graph classes accrete."""
    return Path(pending.mint_root) / GRAPH_DIRNAME


def held_graph_classes(out_dir: Path) -> List[Any]:
    """Every packed graph class already on disk for this obligation.

    THE consume-on-retry read, and it is deliberately filesystem-first: the
    artifact is the durable fact, and the supervisor may have died between
    attempts. Each file is named by its own ``cg-key-v1`` key and carries its
    envelope, so both the identity and the class NAME come back off the bytes
    — nothing is remembered in memory across an attempt.

    An unreadable artifact is DROPPED rather than refused: the honest reading
    of a half-written file is that this class is not held, and the next
    attempt recompiles it. Refusing would let one truncated file cost a
    36-class mint the 35 that are intact.
    """
    from . import aot_mint

    held: List[Any] = []
    if not out_dir.is_dir():
        return held
    for path in sorted(out_dir.glob("*.tar.gz")):
        try:
            meta = dict(artifact_meta.read_metadata(path))
            name = str(meta[graph_facts.TCG_GRAPH_CLASS_BLOCK]["name"])
            key = str(meta.get("compiled_graph_key") or "")
        except Exception:  # noqa: BLE001 — see the docstring
            logger.warning(
                "mint-supervisor: %s is not a readable compiled graph; this "
                "class counts as NOT held and is recompiled", path.name)
            continue
        if not name or not key:
            continue
        held.append(aot_mint.MintedArtifact(
            key=key, entry=name, artifact=path, metadata=meta))
    return held


async def supervise(
    task: MintTask,
    *,
    act: Any,
    abandon: Any = None,
    max_attempts: int = MAX_ATTEMPTS,
    watch: Optional[Watcher] = None,
) -> SupervisedResult:
    """Accrete this obligation's compiled graphs and adopt them.

    Never raises for a mint failure — the worker must never die with its
    mint. ``ABANDONED`` is returned for the co-tenancy signal so the caller
    can tell "somebody took the card" from "this mint failed".
    """
    from . import aot_compile_pool, aot_mint, fleet_cells

    pending = task.pending
    family = str(pending.family)
    cfg = pending.cfg
    out_dir = graph_dir(pending)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(pending.mint_root) / "phases.json"

    def _abandoned() -> bool:
        return abandon is not None and abandon.is_set()

    # ── PARENT-SIDE GATES ────────────────────────────────────────────────
    #
    # The two things only a process holding the LIVE pipeline can do. They ran
    # in `mint_child` while a middle tier existed; a reroute that dropped them
    # would mint a family the declaration says is blocked, and would have no
    # class count to size K against.
    try:
        assert_family_mintable(family)
        spec = fleet_cells.aot_export_spec(task.pipe, cfg)
        decl = export_declaration(str(spec.family or ""))
        if decl is None:
            raise DeclaredBlockerRefusal(
                f"family {spec.family!r} has no registered export declaration "
                f"— a multi-graph cell derives its class set from it")
        # ONE enumeration, off the composed pipeline. A child cannot be told
        # how many classes exist (the adapter fork depends on the composition);
        # it can only be told which SHARE of them is its own.
        declared = len(aot_mint.declared_class_rows(task.pipe, spec, decl))
    except DeclaredBlockerRefusal as exc:
        fleet_cells.abandon_self_mint(pending)
        _emit_abort(
            family=family, key=str(pending.arm_token), attempt=0,
            status="refused", detail=str(exc), retryable=False)
        return SupervisedResult(
            status=FAILED, detail=str(exc), reason="declared_blocker")
    except Exception as exc:  # noqa: BLE001 — never die with the mint
        fleet_cells.abandon_self_mint(pending)
        detail = f"{type(exc).__name__}: {exc}"
        _emit_abort(
            family=family, key=str(pending.arm_token), attempt=0,
            status="refused", detail=detail, retryable=False)
        return SupervisedResult(
            status=FAILED, detail=detail, reason="declaration_unreadable")

    template = aot_compile_pool.EntryJob(
        function=task.function,
        modules=tuple(task.modules),
        cfg=cfg_spec(cfg),
        slots=dict(task.slots),
        posture=task.posture,
        out_dir=str(out_dir),
    )

    loop = asyncio.get_running_loop()
    apply_frame = _on_frame(act, watch)
    apply_evidence = _on_evidence(act)

    def _progress(phase: str, step: int, total: int, note: str) -> None:
        # Marshalled onto the loop: `mint_graph_classes` runs in a worker
        # thread and the Activity belongs to the serving task.
        loop.call_soon_threadsafe(
            apply_frame,
            MintFrame(phase=phase, step=step, total=total, note=note))

    attempts = 0
    last = ""
    reason = ""
    while attempts < max(1, max_attempts):
        attempts += 1
        if _abandoned():
            return SupervisedResult(
                status=ABANDONED, attempts=attempts, declared=declared,
                detail="abandoned before attempt")
        held = held_graph_classes(out_dir)
        have = tuple(sorted(str(row.entry) for row in held))
        # §4.33 / pgw#1175: NOTHING IS BUDGETED. A `mint_budget.co_residency`
        # gate stood at the top of this loop and charged a weight-free child
        # for the PARENT's resident weights, already excluded from the free
        # figure it compared against; four families were declined at
        # 49-113 GiB on it. The attempt IS the budget — the children are
        # separate processes, crash-isolated, and a real shortfall comes back
        # classified from the card itself.
        width = aot_compile_pool.entry_workers(
            max(1, declared - len(have)),
            # The one banked measurement that survives: what a previous
            # compile child on this pod really peaked at, in HOST RSS. K's
            # divisor and nothing else.
            peak_rss_bytes=mint_workers.compiled_graph_peak_rss(
                family, task.weight_lane),
            posture=task.posture)
        act.phase(activity_mod.PHASE_LOAD)
        started = time.monotonic()
        table: Dict[str, Any] = {}
        terminus = ""
        retryable = False
        try:
            if len(have) >= declared:
                # Every declared class is already on disk. The attempt is a
                # no-op and saying so is cheaper than proving it with a pool.
                result = aot_mint.fold_held_graph_classes(held, spec=spec)
            else:
                result = await asyncio.to_thread(
                    aot_mint.mint_graph_classes,
                    msgspec.structs.replace(template, have_classes=have),
                    workdir=Path(pending.mint_root) / "compile-pool",
                    width=width,
                    spec=spec,
                    on_progress=_progress,
                    # pgw#848: rewritten on every beat, so a mint this pod is
                    # KILLED still leaves its measurements on disk.
                    phase_snapshot=snapshot,
                    held=held,
                    should_abandon=_abandoned)
            table = _mint_phase_table_of(result)
        except aot_compile_pool.EntryCompileAbandoned as exc:
            last, terminus = str(exc), ABANDONED
        except aot_mint.MintResourceExhausted as exc:
            last, terminus, retryable = str(exc), "aborted", True
        except aot_mint.MintRefused as exc:
            last, terminus = str(exc), "aborted"
            table = dict(getattr(exc, "mint_phases", None) or {})
        except Exception as exc:  # noqa: BLE001 — never die with the mint
            last, terminus = f"{type(exc).__name__}: {exc}", "aborted"
        else:
            last, terminus = "", ""

        # ── MEASUREMENTS, ON EVERY OUTCOME ───────────────────────────────
        #
        # pgw#848: an aborted mint's classes still peaked where they peaked,
        # and the attempt that follows is exactly the one that needs the fact.
        partial = _read_snapshot(snapshot)
        for rows in (table, partial):
            mint_workers.record_compiled_graph_peak_rss(
                family, task.weight_lane,
                _pool_stat(rows, "peak_child_rss_bytes"))
            _bank_device_peaks(rows, weight_lane=task.weight_lane)
        apply_evidence(float(len(have)))
        _emit_aot_phases(
            phase_table(table, partial), family=family,
            execution_lane=task.weight_lane, terminus=terminus,
            elapsed_s=time.monotonic() - started)

        if terminus == ABANDONED:
            return SupervisedResult(
                status=ABANDONED, detail=last, attempts=attempts,
                covered=len(have), declared=declared)

        if not last:
            act.phase(activity_mod.PHASE_SEAL_PUBLISH)
            # THE honesty gate. This is the one moment a pod holds a freshly
            # TRACED class set for the closure its boot may have answered from
            # a memo — and the memo produced the `graph` axis of the key these
            # artifacts are about to be published under.
            _rule_on_boot_memo(
                task, cfg, result, declared=declared, family=family)
            # pgw#1176/pgw#1183: one artifact per graph class; the adopt
            # stages each durable, verifies, arms and stores it in turn, and a
            # class that refuses costs itself.
            minted = fleet_cells.adopt_delegated_mint(
                task.pipe, pending, [row.artifact for row in result.entries])
            if minted is not None:
                return SupervisedResult(
                    status=ADOPTED, minted=minted, attempts=attempts,
                    covered=len(result.entries), declared=declared)
            # Bytes this runtime could not adopt. `adopt_delegated_mint`
            # emitted the typed abort and cleaned up; retrying cannot change a
            # verify()/drift verdict.
            reason, why = fleet_cells.adopt_refusal(pending)
            return SupervisedResult(
                status=FAILED, attempts=attempts, reason=reason,
                covered=0, declared=declared,
                detail=(
                    f"the compiled graphs did not adopt on this runtime "
                    f"({reason}{': ' + why if why else ''})"
                    if reason else
                    "the compiled graphs did not adopt on this runtime"))

        _emit_abort(
            family=family, key=str(pending.arm_token), attempt=attempts,
            status="aborted", detail=last, retryable=retryable)
        if not (retryable and attempts < max(1, max_attempts)):
            fleet_cells.abandon_self_mint(pending)
            return SupervisedResult(
                status=FAILED, detail=last, attempts=attempts,
                covered=len(have), declared=declared)
        logger.info(
            "mint-supervisor: retrying the mint for %s (attempt %d/%d); %d of "
            "%d graph class(es) are already packed and will be SKIPPED, not "
            "recompiled", family, attempts + 1, max_attempts, len(have),
            declared)

    fleet_cells.abandon_self_mint(pending)
    return SupervisedResult(
        status=FAILED, detail=last, attempts=attempts, declared=declared)


def _mint_phase_table_of(result: Any) -> Dict[str, Any]:
    """The whole-mint phase table off a :class:`aot_mint.MintResult`.

    ``mint_graph_classes`` stamps the same table onto every entry it returns
    (it is the RESULT's view, never the packed envelope's — the artifact
    deliberately carries no wall clocks), so any entry answers.
    """
    for entry in getattr(result, "entries", ()):
        rows = dict(entry.metadata or {}).get("mint_phases")
        if isinstance(rows, dict):
            return dict(rows)
    return {}


def phase_table(
    reported: Mapping[str, Any], snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """The mint's phase table: what the run REPORTED, else the last snapshot.

    A mint that reached its own terminus is better evidence than the last beat
    before it got there, so the snapshot is a fallback and never an override.
    A recovered table SAYS it was recovered — a reader must never mistake the
    last beat of a killed mint for a table written at a terminus.
    """
    if reported:
        return dict(reported)
    if not snapshot:
        return {}
    rows = dict(snapshot)
    rows["recovered_from"] = "phase_snapshot"
    return rows


def _read_snapshot(path: Path) -> Dict[str, Any]:
    """The LIVE phase table a killed or abandoned mint leaves behind.

    pgw#848: ``phase_snapshot`` is rewritten on every beat while the report
    is written once, at a terminus the mint reaches under its own power — so
    this is what survives a signal.
    """
    try:
        rows = msgspec.json.decode(path.read_bytes())
    except Exception:  # noqa: BLE001 — an absent snapshot is the normal case
        return {}
    return dict(rows) if isinstance(rows, dict) else {}


__all__ = [
    "ABANDONED",
    "ADOPTED",
    "EVIDENCE_COUNTER",
    "PROGRESS_COUNTER",
    "FAILED",
    "GRAPH_DIRNAME",
    "MAX_ATTEMPTS",
    "DeclaredBlockerRefusal",
    "MintTask",
    "SupervisedResult",
    "Watcher",
    "assert_family_mintable",
    "cfg_spec",
    "delegation_refusal",
    "graph_dir",
    "phase_table",
    "held_graph_classes",
    "supervise",
]
