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
from .mint_process import MintOutcome, MintRequest

logger = logging.getLogger(__name__)

#: Kill switch, for red-verifying the in-process shape only. Delegation is the
#: default because the in-process shape VIOLATES the liveness contract
#: (WORKER-CONTRACTS §1/§2) — it is kept reachable to prove that, not to run.
ENV_IN_PROCESS = "GEN_WORKER_MINT_IN_PROCESS"


def delegated() -> bool:
    """Whether a compile-cell miss should be minted out of process.

    Delegation IS eager-first: the live pipeline is never armed, so a boot that
    has eager-first turned off has no route to serve while a child compiles.
    The two switches therefore move together rather than combining into a
    fourth, undefined shape.
    """
    if os.environ.get(ENV_IN_PROCESS, "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return False
    return os.environ.get("GEN_WORKER_EAGER_FIRST_BOOT", "1").strip() != "0"


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
) -> MintRequest:
    pending = task.pending
    return MintRequest(
        function=task.function,
        modules=tuple(task.modules),
        family=str(pending.family),
        cell_key=str(pending.cell_key),
        target=str(Path(workdir) / "cell.tar.gz"),
        capture=str(pending.capture_dir),
        report=str(Path(workdir) / mint_process.REPORT_NAME),
        cfg=cfg_spec(pending.cfg),
        snapshots=dict(task.snapshots),
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
            task, workdir=workdir, cap_bytes=budget.need_bytes)
        act.phase(activity_mod.PHASE_LOAD)
        outcome = await mint_process.run_mint(
            request, workdir=workdir, on_frame=_on_frame(act),
            abandon=abandon)
        last = outcome.detail

        if outcome.report is not None and outcome.report.peak_vram_bytes:
            mint_budget.record_child_peak(
                family, task.weight_lane, outcome.report.peak_vram_bytes)
        if str(getattr(pending, "recipe", "")) == "aot":
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
    try:
        if table:
            aot_mint.emit_phase_events(family=family, lane=lane, table=table)
        if outcome.minted:
            return
        total_s = float(
            report.elapsed_s if report is not None and report.elapsed_s > 0
            else outcome.elapsed_s)
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
        f"exit={outcome.exit_code}) — this worker kept serving eager "
        f"throughout: {outcome.detail[:600]}",
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
