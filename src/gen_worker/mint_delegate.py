"""pgw#784: the serving worker's side of an out-of-process mint.

One function, ``build_cell``, is the whole of step 2 of Paul's contract:

    "...if miss then begin compiling and serve eager until the local cell is
    available; then switch over to using that."

It spawns a child, watches it with MEASURED evidence, retries by CLASS, and
adopts the finished artifact through the ordinary delivered-cell path. It
budgets NOTHING: §4.33 / pgw#1175 deleted the pre-flight VRAM verdict, whose
central term charged a weight-free child for the parent's resident weights. It is `async` and every wait inside it is
loop-native, so the caller's 10s beat and its eager serving keep running for
the whole mint — which is the entire point of th#1299.

Kept out of ``executor.py`` deliberately: the executor's job is to decide WHEN
a mint is owed and to advertise the result; how a cell gets built in another
process is its own concern, and Go-style free functions over an explicit task
struct make that testable without standing up an executor.

What the caller still owns
--------------------------
* deciding a mint is owed (``fleet_cells.enable_compiled(..., delegate=True)``);
* the ``self_mint_compile`` Activity — this function only calls ``phase()`` on
  it, so the hub's minting classification and "serving (optimizing in
  background)" messaging keep working with no protocol change;
* phase 4, advertising the adopted identity on the live compile targets;
* the miss policy on failure — which is unchanged, because a failed mint has
  always meant "keep serving eager, leave the cell absent".
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from . import activity as activity_mod
from . import aot_resume
from . import boot_phases
from . import compile_posture
from . import mint_workers
from . import mint_process
from . import progress as progress_mod
from .mint_process import MintOutcome, MintRequest

logger = logging.getLogger(__name__)

#: pgw#1010: ``GEN_WORKER_MINT_IN_PROCESS`` is GONE, with the shape it
#: selected. In-process minting existed only to capture and pack a dynamo
#: cell; a dynamo miss now serves JIT intake and packs nothing, so "mint in
#: this process" is not a state this worker can be in. pgw#995 recorded the
#: env as the last deletable behaviour switch and named its blocker (ten test
#: sites forcing the shape through the executor) — the shape's removal
#: discharges that debt rather than deferring it again.


def delegation_refusal() -> str:
    """"" when this WORKER may mint out of process, else the typed reason.

    Always "" today: nothing worker-wide can refuse delegation any more. The
    seam survives because the PIPELINE half (``fleet_cells.delegation_refusal``
    — an armed non-eager backend with no eager tier to serve from) is a real
    per-pipe refusal, and both halves must reach ``mint_recipe`` as one typed
    reason rather than as an either/or sentence (pgw#813).
    """
    return ""


@dataclass(frozen=True)
class MintTask:
    """Everything the serving process knows that the child needs.

    Note what is NOT here: the pipeline object. ``pipe`` is present only so
    the ADOPT step can arm the live pipeline afterwards — nothing about it
    crosses the process boundary.
    """

    pending: Any                      # fleet_cells.PendingSelfMint (delegated)
    pipe: Any
    function: str
    modules: Tuple[str, ...]
    # pgw#974: the parent's resolution of each setup slot — identity, bytes and
    # pgw#617 composition in ONE value, resolved together in
    # `_setup_locked_inner` and carried together from here to the child. See
    # `mint_process.MintSlot`.
    slots: Dict[str, mint_process.MintSlot] = field(default_factory=dict)
    weight_lane: str = ""
    execution_lane: str = ""
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    device: Optional[int] = None
    #: §4.30 / pgw#1137: whose machine this mint will run on. DECLARED here by
    #: the caller rather than read off a process global, so the one entry that
    #: knows (``local_serve``) states it and every other caller gets ``FLEET``
    #: by construction — there is no ambient value to forget to set.
    posture: compile_posture.CompilePosture = compile_posture.FLEET
    #: pgw#1199: how this process proved the endpoint's handler RUNS, on the
    #: resident pipeline, before delegating. Declared by the caller because
    #: only the caller knows — the executor gets it for free from the boot warm
    #: plan it has already run; `local_serve` runs one forward for it after
    #: setup. An empty string is honest and the child refuses on it.
    handler_proof: str = ""


@dataclass(frozen=True)
class DelegatedResult:
    """The outcome of one delegated mint, from the serving side.

    There is no DECLINED outcome any more (pgw#1175). A mint is attempted or
    it is not; a card that genuinely cannot hold the child kills the child,
    which comes back FAILED with the card's own classification.
    """

    status: str
    detail: str = ""
    minted: Optional[Any] = None      # fleet_cells.SelfMint
    attempts: int = 0
    #: pgw#999: the CLASSIFIED reason the child's cell did not adopt, carried
    #: up so the executor's decline names the same token the abort event did.
    #: Empty for every outcome that is not an adopt refusal.
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.minted is not None


ADOPTED = "adopted"
FAILED = "failed"
ABANDONED = mint_process.ABANDONED


def cfg_spec(cfg: Any) -> mint_process.CompileCellSpec:
    """Flatten the parent's ``CompileCell`` for the wire.

    The parent states the contract because the class-scoped guidance/text-len
    unions live on the spec rather than the decorator: a child re-deriving this
    from ``@endpoint`` alone would export a different declaration than the
    parent asked for. It carries what the child READS and nothing else
    (pgw#1034) — the child computes no key, so this is not a key-parity wire.
    """
    return mint_process.CompileCellSpec(
        shapes=tuple(tuple(int(v) for v in row) for row in (cfg.shapes or ())),
        targets=tuple(str(t) for t in (cfg.targets or ())),
        family=str(getattr(cfg, "family", "") or ""),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )


def build_request(
    task: MintTask, *, workdir: Path, entry_peak_rss_bytes: int = 0,
    device: Optional[int] = None,
) -> MintRequest:
    pending = task.pending
    return MintRequest(
        entry_peak_rss_bytes=int(entry_peak_rss_bytes),
        function=task.function,
        modules=tuple(task.modules),
        family=str(pending.family),
        arm_token=str(pending.arm_token),
        target=str(Path(workdir) / "cell.tar.gz"),
        work_root=str(workdir),
        report=str(Path(workdir) / mint_process.REPORT_NAME),
        # pgw#848: where the child writes its LIVE table. `report` is written
        # once, at a terminus the child reaches under its own power; this is
        # written on every beat, so a mint the parent abandons still hands
        # back what it measured.
        phases_snapshot=str(
            Path(workdir) / mint_process.PHASES_SNAPSHOT_NAME),
        # pgw#848 item 5: outside the mint's own tree entirely. Every other
        # path here is per-attempt by design; this one must outlive not just
        # the attempt but the PENDING — `abandon_self_mint` rmtree's
        # `mint_root`, and abandonment is how a crashed mint ends, so a bank
        # sited there would be deleted on its way out of the one case it
        # exists for. Keyed by the pending's `cell_key` as a SCOPE (identity is
        # the per-entry re-derivation inside it), so a mint child restarted in
        # place on the same pod — or a whole new pending for the same cell on a
        # later boot — finds the same bank.
        resume=str(aot_resume.bank_root(pending.arm_token)),
        cfg=cfg_spec(pending.cfg),
        slots=dict(task.slots),
        device=_ordinal(task.device if device is None else device),
        execution_lane=task.execution_lane,
        configs={k: dict(v) for k, v in task.configs.items()},
        posture=task.posture,
        handler_proof=str(task.handler_proof or ""),
    )


def _ordinal(device: Optional[int]) -> int:
    """``-1`` = leave the child's default; any other value names a card."""
    return -1 if device is None else int(device)


def _pool_stat(phases: Any, key: str) -> int:
    """One measurement out of a mint's phase table `pool` block, or 0.

    pgw#877: this was written out by hand three times over two banks, each
    with its own try/except and its own idea of which exceptions to swallow.
    The next reader added the fourth copy or, more likely, did not add it at
    all — which is how `peak_child_device_bytes` came to be published and
    never read.
    """
    try:
        block = (phases or {}).get("pool")
        return int((block or {}).get(key) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _on_frame(act: Any, watch: Optional[Watcher] = None) -> Any:
    def _apply(frame: mint_process.MintFrame) -> None:
        # No new protocol: the child's phase lands on the SAME
        # self_mint_compile activity the hub already reads, and ships on the
        # ordinary 10s beat — which now actually fires, because nothing on
        # this loop is compiling.
        if frame.phase:
            act.phase(frame.phase, frame.step, frame.total)
        if frame.note:
            act.note(frame.note[:200])
        # pgw#1137: ...and, on a machine with a person at it, onto that
        # person's terminal. The activity above is addressed to the HUB, and
        # cozy-local has no hub — so on a desktop every frame of a 20-minute
        # compile went nowhere a user could see it.
        if watch is not None:
            watch(frame)

    return _apply


#: The mint child's evidence counter, in the `family:name` shape every other
#: counter uses. It was plain `mint_child_evidence` (pgw#1157): `window_for`
#: splits on ":" and falls back to DEFAULT_STALL_WINDOW_S, so the mint's
#: patience was the generic 300 s by ACCIDENT rather than the 600 s the
#: `compile` family declares for exactly this work. An AOTI entry that spends
#: minutes inside one inductor call is the case that window exists for.
EVIDENCE_COUNTER = "compile:mint_child_evidence"


def _on_evidence(act: Any) -> Any:
    """pgw#824: the child's MEASURED progress, onto the parent's activity.

    ``mint_process._observe`` has always sampled the child's own tree CPU plus
    the growth of its capture directory, and ``run_mint`` has always accepted
    an ``on_evidence`` callback for it — nobody ever passed one, so the number
    existed only to decide whether to KILL the child, never to prove it was
    working.

    It is the better signal by a distance. ``activity.watchdog`` measures the
    PARENT process (plus live children) and only proves the pod is alive;
    capture-directory bytes are the artifact being built. Registering it as a
    progress COUNTER means the hub's liveness rule judges an advancing number
    rather than inferring health from a re-sent phase — which is what the
    no-magic-timeouts doctrine asks for: liveness plus progress-staleness,
    never a fixed duration. A silent multi-minute ``trace_graph`` now ticks.
    """
    # Telemetry never breaks the work it reports on: an Activity double that
    # carries no counter registry still gets the heartbeat, which is the half
    # the hub's liveness rule actually reads.
    make = getattr(act, "counter", None)
    if not callable(make):
        def _beat_only(value: float) -> None:
            act.heartbeat()
        return _beat_only

    def _apply(value: float) -> None:
        # pgw#1157: RE-ACQUIRE per tick, never capture. `Activity.counter()`
        # binds a counter to the phase that registered it, and
        # `Activity.phase()` FINISHES every counter the new phase does not own
        # (pgw#962) — `finish()` deletes it from the process registry. A
        # captured Counter therefore goes unreadable at the first phase
        # change, and every `set_done` after that feeds an object no reader
        # can see. This mint crosses `load` -> `warmup_forward` ->
        # `trace_graph`, so the counter was dead for the whole compile:
        # `activity.on_beat` found no counter for the activity, returned
        # WITHOUT emitting, and the hub got nothing but counterless beats.
        # Measured on RunPod A40 `bgmdxhazxsugmk` (0.112.0): 62 minutes in
        # `trace_graph` with no counter, no rate and no self-diagnosis
        # possible, over a mint that was in fact advancing (16 of 36 entries).
        # `progress.counter` is register-or-get, so re-acquiring costs a dict
        # lookup and re-registers under whatever phase is current — which is
        # exactly the contract `Activity.counter()` documents for producers.
        make(EVIDENCE_COUNTER, progress_mod.UNIT_EVIDENCE).set_done(float(value))
        act.heartbeat()

    return _apply


#: pgw#1137: a sink for the child's progress frames that is NOT the hub. The
#: fleet passes nothing and behaves exactly as before; ``local_serve`` passes a
#: terminal renderer, because a 20-minute compile a user cannot see is a
#: support ticket regardless of how correct it is.
Watcher = Callable[[mint_process.MintFrame], None]


async def build_cell(
    task: MintTask,
    *,
    act: Any,
    abandon: Any = None,
    max_attempts: int = mint_process.MAX_ATTEMPTS,
    watch: Optional[Watcher] = None,
) -> DelegatedResult:
    """Build and adopt one cell in a child process. Never raises for a mint
    failure — the worker must never die with its mint."""
    from . import fleet_cells

    pending = task.pending
    device = (
        task.device if task.device is not None
        else mint_workers.device_of(task.pipe))
    family = str(pending.family)
    attempts = 0
    last = ""
    while attempts < max(1, max_attempts):
        attempts += 1
        # §4.33 / pgw#1175: NOTHING IS BUDGETED HERE. A `mint_budget
        # .co_residency` gate stood at the top of this loop and charged the
        # compiling child for `allocated` — the PARENT's resident weights,
        # already excluded from the `free_bytes` it compared against. Compiles
        # are weight-free (`fc77b923`), so the term described a process that
        # does not exist, and four families were declined at 49-113 GiB on it.
        # The attempt IS the budget: the child is a separate process, it is
        # crash-isolated, it costs ~2 minutes, and a real shortfall comes back
        # classified as `MintResourceExhausted` from the card itself.
        workdir = Path(pending.mint_root) / f"child-{attempts}"
        workdir.mkdir(parents=True, exist_ok=True)
        request = build_request(
            task, workdir=workdir, device=device,
            # The one banked measurement that survives: what a previous entry
            # child on this pod really peaked at, in HOST RSS. It is K's
            # divisor and nothing else. Read here rather than in the child
            # because the child is the thing that dies.
            entry_peak_rss_bytes=mint_workers.entry_peak_rss(
                family, task.weight_lane))
        act.phase(activity_mod.PHASE_LOAD)
        outcome = await mint_process.run_mint(
            request, workdir=workdir, on_frame=_on_frame(act, watch),
            on_evidence=_on_evidence(act), abandon=abandon)
        last = outcome.detail

        if outcome.report is not None:
            # pgw#848: banked on EVERY outcome, not just a minted one — an
            # aborted mint's entries still peaked where they peaked, and the
            # attempt that follows is exactly the one that needs the fact.
            # pgw#1175: the three DEVICE banks that stood beside this one are
            # gone; this is the only measurement any decision still reads.
            mint_workers.record_entry_peak_rss(
                family, task.weight_lane,
                _pool_stat(outcome.report.mint_phases,
                           "peak_child_rss_bytes"))
        # ...and from the SNAPSHOT when the child never wrote a report at all,
        # which is every killed and every abandoned mint.
        mint_workers.record_entry_peak_rss(
            family, task.weight_lane,
            _pool_stat(outcome.partial_phases, "peak_child_rss_bytes"))
        # pgw#1010: every delegated mint is an AOT mint, so there is one
        # phase emitter. The JIT twin measured a child recipe that no longer
        # exists.
        _emit_aot_phases(
            outcome, family=family, execution_lane=task.weight_lane)

        if outcome.status == mint_process.ABANDONED:
            return DelegatedResult(
                status=ABANDONED, detail=outcome.detail, attempts=attempts)

        if outcome.minted and outcome.artifact is not None:
            act.phase(activity_mod.PHASE_SEAL_PUBLISH)
            minted = fleet_cells.adopt_delegated_mint(
                task.pipe, pending, outcome.artifact)
            if minted is not None:
                # pgw#848 item 5: the ONE terminus where the bank's job is
                # finished. It survives every failure (that is the point) and
                # is dropped on success, which is what keeps a healthy pod's
                # resume area from being the only thing that grows.
                aot_resume.discard(str(pending.arm_token))
                return DelegatedResult(
                    status=ADOPTED, minted=minted, attempts=attempts)
            # The child produced bytes this runtime could not adopt.
            # `adopt_delegated_mint` emitted the typed abort and cleaned up;
            # retrying cannot change a verify()/drift verdict.
            #
            # pgw#999: it also RECORDED why, and this is where that used to
            # die. The sentence below was the whole of what the wire got.
            reason, why = fleet_cells.adopt_refusal(pending)
            return DelegatedResult(
                status=FAILED, attempts=attempts, reason=reason,
                detail=(
                    f"the child's cell did not adopt on this runtime "
                    f"({reason}{': ' + why if why else ''})"
                    if reason else
                    "the child's cell did not adopt on this runtime"))

        _emit_abort(outcome, family, pending.arm_token, attempts)
        if not (outcome.retryable and attempts < max(1, max_attempts)):
            fleet_cells.abandon_self_mint(pending)
            return DelegatedResult(
                status=FAILED, detail=outcome.detail, attempts=attempts)
        logger.info(
            "mint-delegate: retrying the mint for %s (attempt %d/%d) after a "
            "%s outcome", family, attempts + 1, max_attempts, outcome.status)

    fleet_cells.abandon_self_mint(pending)
    return DelegatedResult(
        status=FAILED, detail=last, attempts=attempts)


def _emit_boot_trace_rows(table: Mapping[str, Any], *, family: str) -> None:
    """pgw#1087: the mint's PER-CLASS timings, as boot-phase rows.

    The mint child holds no orchestrator session and its phase table has always
    been a single blob event — readable by a human, joinable by nothing. The
    per-class trace cost was the biggest thing pgw#1087 could not answer ("10-20
    s/class, guessed, never measured"), and the number already exists: the child
    measures `export_s` per entry and hands it back in the report. Re-emitting
    it here — parent-side, where the stream is — puts one `trace_for_key` row
    per graph CLASS in the same table as the fetch and admission phases, so a
    boot's compile half and its I/O half are finally comparable.

    Emitted whether or not the boot window is still open: a self-mint routinely
    finishes after `first_request_servable`, and its per-class costs are the
    point. They are spans (not cumulative milestones), so `reconciliation`
    charges them honestly — a mint that ran after the boot closed makes
    `measured_ms` exceed `total_ms`, which is a TRUE statement about a worker
    that kept compiling after it went servable, not a broken ladder.

    OWED (pgw#1087, deliberately not taken here): the node count per class. It
    is one line in `aot_mint._export_entry` — `timings["nodes"] =
    len(program.graph_module.graph.nodes)` — and that function is being edited
    by the pgw#1080 lane in the same window, so this lane does not race it. The
    row already carries the shape the count will ride (`nodes=` in `detail`).
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
        # The fold: per-class hashing plus the combine into
        # `combined_graph_hash`, which the mint measures as `declare_s`.
        declare_s = float(totals.get("declare_s") or 0.0)
        if declare_s > 0:
            boot_phases.mark(
                boot_phases.PHASE_KEY_FOLD,
                duration_ms=int(round(declare_s * 1000.0)),
                ref=family,
                detail=f"classes={len(entries)}",
            )


def _emit_aot_phases(
    outcome: MintOutcome, *, family: str, execution_lane: str,
) -> None:
    """pgw#805: one delegated AOT mint's phase table, re-emitted PARENT-side.

    ``aot_mint`` already emits `aot_mint_phases` — but it runs in the mint
    CHILD, which holds no orchestrator session, so those events reach nothing.
    Re-emitting from the parent (which owns the connection and the 10 s beat)
    is what finally puts rows in a table that has been empty on both stacks
    since th#1322 shipped the column.

    A mint that produced NO cell still reports its total, under
    `phase=aborted`: the seconds are real and worth recording, and they must
    not enter an AOT-vs-JIT comparison as if a cell came out.
    """
    from . import aot_mint

    report = outcome.report
    table = dict(getattr(report, "mint_phases", None) or {}) \
        if report is not None else {}
    if not table and outcome.partial_phases:
        # pgw#848: the ABANDONED path. `f9c1b2d` gave the aborted path its
        # measurements back; this is the same code with a different exit, and
        # it never got them. Attempt sixteen compiled for 29 minutes and
        # reported ONE row — `status=abandoned total_s=1741.33 — no cell
        # produced`, zero `entry:` rows, no `pool` row — because the child was
        # group-killed before it could write a report. The snapshot is what
        # survives a signal.
        table = dict(outcome.partial_phases)
        table["recovered_from"] = "phase_snapshot"
    try:
        total_s = float(
            report.elapsed_s if report is not None and report.elapsed_s > 0
            else outcome.elapsed_s)
        if table and not outcome.minted and total_s > 0:
            # pgw#825: the CHILD's whole elapsed (load + export + compile)
            # folded into the table rather than emitted as a second aborted
            # event — two `phase=aborted` rows for one mint would double-count
            # it in any duration roll-up, and the table's own total (the
            # mint's wall clock) is the one a compile comparison wants.
            table["totals"] = {
                **dict(table.get("totals") or {}),
                "child_elapsed_s": round(total_s, 2),
            }
        # pgw#848: an ABANDONED mint is not an aborted one. Nothing about the
        # mint failed — a co-tenancy decision (a drain of the endpoint
        # instances) destroyed it while it was working, and a roll-up that
        # calls that "aborted" hides the only actionable fact in the row.
        terminus = ""
        if not outcome.minted:
            terminus = ABANDONED if outcome.status == ABANDONED else "aborted"
        if table:
            table["terminus"] = terminus or table.get("terminus") or ""
            aot_mint.emit_phase_events(
                family=family, execution_lane=execution_lane, table=table, terminus=terminus)
            _emit_boot_trace_rows(table, family=family)
        if outcome.minted or table:
            return
        if total_s <= 0:
            return
        activity_mod.emit_event(
            activity_mod.KIND_AOT_MINT,
            f"family={family} lane={execution_lane or 'plain'} status={outcome.status} "
            f"total_s={round(total_s, 2)} — no cell produced",
            phase="aborted",
            duration_ms=int(round(total_s * 1000)),
        )
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("mint-delegate: aot phase event emission failed",
                     exc_info=True)


def _emit_abort(
    outcome: MintOutcome, family: str, key: str, attempt: int,
) -> None:
    """A failed mint is hub-relevant truth, not a pod-log line.

    A serve pod exposes no logs (pgw#760), so the classification, the phase it
    died in and the child's own last words all have to ride the wire or the
    next person debugging this has nothing.
    """
    activity_mod.emit_event(
        "self_mint_abort",
        f"family={family} key={key}: the mint PROCESS ended "
        f"{outcome.status} on attempt {attempt} "
        f"(phase={outcome.last_phase or 'unknown'}, "
        f"exit={outcome.exit_code}, "
        f"{'retryable' if outcome.retryable else 'deterministic'}) — this "
        f"worker kept serving eager throughout: {outcome.detail[:600]}",
        phase=f"delegated_{outcome.status}",
    )


def scratch_root() -> Path:
    """Where a delegated mint's child workdirs live when no mint root exists
    (tests / ad-hoc drives)."""
    return Path(tempfile.mkdtemp(prefix="mint-delegate-"))


__all__ = [
    "ABANDONED",
    "ADOPTED",
    "FAILED",
    "DelegatedResult",
    "MintTask",
    "build_cell",
    "build_request",
    "cfg_spec",
    "scratch_root",
]
