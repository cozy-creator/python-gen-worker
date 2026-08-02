"""pgw#784: the serving worker's side of an out-of-process mint.

One function, ``build_cell``, is the whole of step 2 of Paul's contract:

    "...if miss then begin compiling and serve eager until the local cell is
    available; then switch over to using that."

It budgets for a co-resident child, spawns it, watches it with MEASURED
evidence, retries by CLASS, and adopts the finished artifact through the
ordinary delivered-cell path. It is `async` and every wait inside it is
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
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import activity as activity_mod
from . import mint_budget
from . import mint_process
from . import progress as progress_mod
from .mint_process import MintOutcome, MintRequest

logger = logging.getLogger(__name__)

#: Kill switch, for red-verifying the in-process shape only. Delegation is the
#: default because the in-process shape VIOLATES the liveness contract
#: (WORKER-CONTRACTS §1/§2) — it is kept reachable to prove that, not to run.
ENV_IN_PROCESS = "GEN_WORKER_MINT_IN_PROCESS"


#: Typed refusals :func:`delegation_refusal` can return. They are the OPERATOR
#: half of the decision (env kill switches); the PIPELINE half lives in
#: ``fleet_cells.delegation_refusal``. pgw#813: the two were collapsed into one
#: either/or sentence on the wire, so a real refusal named two causes that were
#: both false and never named the one that was true.
REFUSAL_IN_PROCESS_FORCED = "mint_in_process_forced"
REFUSAL_EAGER_FIRST_DISABLED = "eager_first_disabled"


def delegation_refusal() -> str:
    """"" when this WORKER may mint out of process, else the typed reason.

    Delegation IS eager-first: the live pipeline is never armed, so a boot that
    has eager-first turned off has no route to serve while a child compiles.
    The two switches therefore move together rather than combining into a
    fourth, undefined shape.
    """
    if os.environ.get(ENV_IN_PROCESS, "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return REFUSAL_IN_PROCESS_FORCED
    if os.environ.get("GEN_WORKER_EAGER_FIRST_BOOT", "1").strip() == "0":
        return REFUSAL_EAGER_FIRST_DISABLED
    return ""


def delegated() -> bool:
    """Whether a compile-cell miss should be minted out of process."""
    return not delegation_refusal()


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
    snapshots: Dict[str, str] = field(default_factory=dict)
    # pgw#816: the parent's RESOLVED component overrides, slot -> comp ->
    # local tree. Part of the composition, not a detail: the base snapshot is
    # fetched with these components EXCLUDED (th#1330 B2), so a child handed
    # only `snapshots` is handed a tree that cannot load.
    component_paths: Dict[str, Dict[str, str]] = field(default_factory=dict)
    weight_lane: str = ""
    lane: str = ""
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    device: Optional[int] = None


@dataclass(frozen=True)
class DelegatedResult:
    """The outcome of one delegated mint, from the serving side.

    ``declined`` is not a failure and never was (pgw#737): the card had no room
    for a co-resident child, the worker serves eager, the cell stays absent,
    and a roomier pod mints it later.
    """

    status: str
    detail: str = ""
    minted: Optional[Any] = None      # fleet_cells.SelfMint
    attempts: int = 0
    budget: Optional[mint_budget.MintBudget] = None

    @property
    def declined(self) -> bool:
        return self.status == DECLINED

    @property
    def ok(self) -> bool:
        return self.minted is not None


ADOPTED = "adopted"
DECLINED = "declined"
FAILED = "failed"
ABANDONED = mint_process.ABANDONED


def cfg_spec(cfg: Any) -> mint_process.CompileCellSpec:
    """Flatten the parent's ``CompileCell`` for the wire.

    The parent states the contract because the parent owns the KEY: the ck2
    ``contract`` axis digests ``CompileCell.contract_facts()``, and the
    class-scoped guidance/text-len unions live on the spec rather than the
    decorator. A child re-deriving this from ``@endpoint`` alone would compute
    a different key, and the parent would then refuse its own artifact on an
    axis nobody changed.
    """
    return mint_process.CompileCellSpec(
        shapes=tuple(tuple(int(v) for v in row) for row in (cfg.shapes or ())),
        targets=tuple(str(t) for t in (cfg.targets or ())),
        family=str(getattr(cfg, "family", "") or ""),
        regional=bool(getattr(cfg, "regional", False)),
        text_len=getattr(cfg, "text_len", None),
        dynamic=tuple(
            mint_process.DynamicDimSpec(
                dim=str(d.dim), min=int(d.min), max=int(d.max))
            for d in (getattr(cfg, "dynamic", ()) or ())),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        guidance_scales=tuple(
            float(v) for v in (getattr(cfg, "guidance_scales", ()) or ())),
        text_lens=tuple(int(v) for v in (getattr(cfg, "text_lens", ()) or ())),
    )


def build_request(
    task: MintTask, *, workdir: Path, cap_bytes: int,
    entry_peak_rss_bytes: int = 0,
) -> MintRequest:
    pending = task.pending
    return MintRequest(
        entry_peak_rss_bytes=int(entry_peak_rss_bytes),
        function=task.function,
        modules=tuple(task.modules),
        family=str(pending.family),
        cell_key=str(pending.cell_key),
        target=str(Path(workdir) / "cell.tar.gz"),
        capture=str(pending.capture_dir),
        report=str(Path(workdir) / mint_process.REPORT_NAME),
        # pgw#848: where the child writes its LIVE table. `report` is written
        # once, at a terminus the child reaches under its own power; this is
        # written on every beat, so a mint the parent abandons still hands
        # back what it measured.
        phases_snapshot=str(
            Path(workdir) / mint_process.PHASES_SNAPSHOT_NAME),
        cfg=cfg_spec(pending.cfg),
        snapshots=dict(task.snapshots),
        component_paths={
            slot: dict(comps)
            for slot, comps in task.component_paths.items() if comps
        },
        device=-1 if task.device is None else int(task.device),
        vram_cap_bytes=int(cap_bytes),
        lane=task.lane,
        configs={k: dict(v) for k, v in task.configs.items()},
        # pgw#805: the recipe rides the pending the arming brain built. The
        # child never decides it — the recipe determines the artifact KIND,
        # and a pod that mints the kind its own discovery does not accept is
        # the permanent-miss loop this issue exists to close.
        recipe=str(getattr(pending, "recipe", "dynamo") or "dynamo"),
    )


def _on_frame(act: Any) -> Any:
    def _apply(frame: mint_process.MintFrame) -> None:
        # No new protocol: the child's phase lands on the SAME
        # self_mint_compile activity the hub already reads, and ships on the
        # ordinary 10s beat — which now actually fires, because nothing on
        # this loop is compiling.
        if frame.phase:
            act.phase(frame.phase, frame.step, frame.total)
        if frame.note:
            act.note(frame.note[:200])

    return _apply


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
    counter = (
        make("mint_child_evidence", progress_mod.UNIT_EVIDENCE)
        if callable(make) else None)

    def _apply(value: float) -> None:
        if counter is not None:
            counter.set_done(float(value))
        act.heartbeat()

    return _apply


async def build_cell(
    task: MintTask,
    *,
    act: Any,
    abandon: Any = None,
    max_attempts: int = mint_process.MAX_ATTEMPTS,
) -> DelegatedResult:
    """Build and adopt one cell in a child process. Never raises for a mint
    failure — the worker must never die with its mint."""
    from . import fleet_cells

    pending = task.pending
    device = (
        task.device if task.device is not None
        else mint_budget.device_of(task.pipe))
    family = str(pending.family)
    attempts = 0
    budget: Optional[mint_budget.MintBudget] = None
    last = ""

    while attempts < max(1, max_attempts):
        attempts += 1
        # Re-budget EVERY attempt. The first shortfall may have been the
        # tenant's peak; the second may be a card that genuinely cannot hold a
        # co-resident child, and that must decline rather than retry.
        budget = mint_budget.co_residency(
            device, family=family, weight_lane=task.weight_lane)
        if not budget.fits:
            reason = (
                "insufficient_vram" if attempts == 1
                else "insufficient_vram_after_retry")
            detail = budget.line("self_mint_skipped", reason)
            logger.info("mint-delegate: %s", detail)
            activity_mod.emit_event("self_mint_skipped", detail, phase=reason)
            return DelegatedResult(
                status=DECLINED, detail=detail, attempts=attempts,
                budget=budget)

        workdir = Path(pending.mint_root) / f"child-{attempts}"
        workdir.mkdir(parents=True, exist_ok=True)
        request = build_request(
            # pgw#848: the CEILING, not the estimate. `need_bytes` answers
            # "should this start"; handing it to the child as a hard cap is
            # what pinned two mints at 11.09 GiB on cards with 21.48 GiB free.
            task, workdir=workdir, cap_bytes=budget.cap_bytes,
            # pgw#848: whatever a previous mint on this pod measured one entry
            # child to peak at. Read here rather than in the child because the
            # child is the thing that dies.
            entry_peak_rss_bytes=mint_budget.entry_peak_rss(
                family, task.weight_lane))
        act.phase(activity_mod.PHASE_LOAD)
        outcome = await mint_process.run_mint(
            request, workdir=workdir, on_frame=_on_frame(act),
            on_evidence=_on_evidence(act), abandon=abandon)
        last = outcome.detail

        if outcome.report is not None:
            # pgw#848: banked on EVERY outcome, not only a minted one. This
            # was gated on a truthy `peak_vram_bytes`, which a child dying
            # inside `torch.export` never produced — so attempt N+1 re-asked
            # identically, forever, which is most of why the failure presented
            # as deterministic. The device side now has the shape the host
            # side got below: the attempt that FAILED is exactly the attempt
            # whose measurement the next one needs.
            mint_budget.record_child_peak(
                family, task.weight_lane,
                int(outcome.report.peak_vram_bytes or 0))
            # pgw#848: the host half of the same loop. The pool has always
            # measured `peak_child_rss_bytes` and put it in the phase table
            # that rides `mint_phases`; nothing has ever read it, so every
            # mint sized `mem_workers` off a 3 GiB constant. Banked on EVERY
            # outcome, not just a minted one: an aborted mint's entries still
            # peaked where they peaked, and the attempt that follows it is
            # exactly the one that needs the fact.
            try:
                pool_block = (outcome.report.mint_phases or {}).get("pool")
                peak_rss = int(
                    (pool_block or {}).get("peak_child_rss_bytes") or 0)
            except (TypeError, ValueError, AttributeError):
                peak_rss = 0
            mint_budget.record_entry_peak_rss(
                family, task.weight_lane, peak_rss)
        # pgw#848: ...and from the SNAPSHOT when the child never wrote a
        # report at all, which is every killed and every abandoned mint.
        try:
            snap_pool = (outcome.partial_phases or {}).get("pool")
            mint_budget.record_entry_peak_rss(
                family, task.weight_lane,
                int((snap_pool or {}).get("peak_child_rss_bytes") or 0))
        except (TypeError, ValueError, AttributeError):
            pass
        if str(getattr(pending, "recipe", "")) == fleet_cells.RECIPE_AOT:
            _emit_aot_phases(outcome, family=family, lane=task.weight_lane)
        else:
            _emit_jit_compile(
                outcome, family=family, lane=task.weight_lane,
                key=str(pending.cell_key), attempt=attempts)

        if outcome.status == mint_process.ABANDONED:
            return DelegatedResult(
                status=ABANDONED, detail=outcome.detail, attempts=attempts,
                budget=budget)

        if outcome.minted and outcome.artifact is not None:
            act.phase(activity_mod.PHASE_SEAL_PUBLISH)
            minted = fleet_cells.adopt_delegated_mint(
                task.pipe, pending, outcome.artifact)
            if minted is not None:
                return DelegatedResult(
                    status=ADOPTED, minted=minted, attempts=attempts,
                    budget=budget)
            # The child produced bytes this runtime could not adopt.
            # adopt_delegated_mint already emitted the typed abort and
            # cleaned up; retrying cannot change a verify()/drift verdict.
            return DelegatedResult(
                status=FAILED, attempts=attempts, budget=budget,
                detail="the child's cell did not adopt on this runtime")

        _emit_abort(outcome, family, pending.cell_key, attempts)
        if not (outcome.retryable and attempts < max(1, max_attempts)):
            fleet_cells.abandon_self_mint(pending)
            return DelegatedResult(
                status=FAILED, detail=outcome.detail, attempts=attempts,
                budget=budget)
        logger.info(
            "mint-delegate: retrying the mint for %s (attempt %d/%d) after a "
            "%s outcome", family, attempts + 1, max_attempts, outcome.status)

    fleet_cells.abandon_self_mint(pending)
    return DelegatedResult(
        status=FAILED, detail=last, attempts=attempts, budget=budget)


def _emit_aot_phases(
    outcome: MintOutcome, *, family: str, lane: str,
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
                family=family, lane=lane, table=table, terminus=terminus)
        if outcome.minted or table:
            return
        if total_s <= 0:
            return
        activity_mod.emit_event(
            activity_mod.KIND_AOT_MINT,
            f"family={family} lane={lane or 'plain'} status={outcome.status} "
            f"total_s={round(total_s, 2)} — no cell produced",
            phase="aborted",
            duration_ms=int(round(total_s * 1000)),
        )
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("mint-delegate: aot phase event emission failed",
                     exc_info=True)


def _emit_jit_compile(
    outcome: MintOutcome, *, family: str, lane: str, key: str, attempt: int,
) -> None:
    """th#1322: one delegated JIT mint's duration, as typed NUMERIC events.

    This is the fleet's real JIT compile path — the child arms COLD and drives
    the endpoint's own warm plan (gw#587), so its `warmup_forward` +
    `inductor_compile` spans ARE "how long does JIT take". Before this the
    number lived only in the child's stdout, and a serve pod exposes no logs
    (pgw#760), so it was unrecoverable the moment the pod went away.

    Shape matches ``aot_mint_phases`` exactly: `phase=minted` carries the total,
    `phase=child:<phase>` the spans inside it. A mint that did NOT produce a
    cell reports its total under `phase=aborted` instead — the seconds are real
    and worth recording, but they must not enter an AOT-vs-JIT comparison as if
    a cell came out.

    Telemetry never changes the mint's outcome.
    """
    report = outcome.report
    try:
        head = f"family={family} lane={lane or 'plain'} key={key} attempt={attempt}"
        phases: Dict[str, float] = dict(
            report.phases) if report is not None else {}
        for name, seconds in sorted(phases.items()):
            value = float(seconds or 0.0)
            if value <= 0:
                continue
            activity_mod.emit_event(
                activity_mod.KIND_JIT_COMPILE,
                f"{head} child_phase={name} seconds={round(value, 2)}",
                phase=f"child:{name}",
                duration_ms=int(round(value * 1000)),
            )
        # The child's own elapsed is authoritative when it wrote a report; the
        # parent's wall clock covers a child that died before writing one (spawn
        # + run, which is the honest cost of that attempt).
        total_s = float(
            report.elapsed_s if report is not None and report.elapsed_s > 0
            else outcome.elapsed_s)
        if total_s <= 0:
            return
        activity_mod.emit_event(
            activity_mod.KIND_JIT_COMPILE,
            f"{head} status={outcome.status} total_s={round(total_s, 2)} "
            f"phases={phases}",
            phase=(
                activity_mod.PHASE_MINTED if outcome.minted else "aborted"),
            duration_ms=int(round(total_s * 1000)),
        )
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("mint-delegate: jit_compile event emission failed",
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
    "DECLINED",
    "ENV_IN_PROCESS",
    "FAILED",
    "DelegatedResult",
    "MintTask",
    "build_cell",
    "build_request",
    "cfg_spec",
    "delegated",
    "scratch_root",
]
