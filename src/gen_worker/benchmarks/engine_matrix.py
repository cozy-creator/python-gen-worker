"""Acceleration-matrix driver: one arm per technique, one lane per arm.

The question is "which acceleration technique is worth adopting on OUR
silicon" for a model whose every published number was measured on somebody
else's, on hardware we do not have. That is a MATRIX — technique x compiled graph — and
the expensive part is not the measurement, it is the pod. Three facts fix the
shape of this harness:

* **An arm is DATA.** A technique is a name plus a settings fragment plus a
  request-payload fragment. Flipping techniques is choosing an arm — never
  editing a serving path, and deliberately never an env flag (standing rule:
  envs carry secrets and config VALUES, never logic forks). Measurement arms
  have no business existing on the request path at all.
* **The lane is pluggable, because the candidates do not share a runtime.**
  Some techniques are server CLI flags (SGLang ``--attention-backend``,
  ``--quantization fp8``); the ones ranked first are
  in-process diffusers changes (the AdaLN-branch cache, ``cache-dit``, a
  resolution-matched warmup, a sage backend swap). A driver that assumed a
  server subprocess could not measure the techniques we care most about, so it
  takes an :class:`Lane` factory and knows nothing about either.
* **Evidence must survive the pod.** RunPod exposes no container-log API, so a
  row that only ever reached stdout is a row we do not have. Every row is
  appended to the output file the moment it completes, before the next arm
  opens. A killed run keeps every arm it finished.

**Budget is structural, not advisory.** Every cloud GPU run is watchdog +
wall-capped by standing policy, and an unbounded matrix is how a dead job bills
for eight hours. ``wall_cap_s`` bounds the WHOLE run; an arm that cannot fit in
the remaining budget is skipped with a recorded reason rather than started and
abandoned.

Usage (pod-side)::

    python -m gen_worker.benchmarks.engine_matrix \\
        --plan <plan.json> --tier tier0 --out /workspace/h3-matrix.jsonl \\
        --wall-cap-s 5400 --lane <module>:<factory>

A plan file is DATA (:mod:`gen_worker.benchmarks.plans`): the GPU lane adds an
arm by adding a row, not by touching this module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

__all__ = [
    "Arm",
    "Plan",
    "Row",
    "ArmResult",
    "Lane",
    "load_plan",
    "select_arms",
    "run_matrix",
    "summarize",
]


class Arm(msgspec.Struct, frozen=True, kw_only=True):
    """One measurable configuration.

    ``settings`` and ``payload`` are the ONLY things that vary between arms;
    anything else differing means two arms are not comparable and the speedup
    computed from them is not a speedup.
    """

    name: str
    #: Tier ("tier0" config-only, "tier1" single flags, "tier2" integration).
    tier: str = ""
    #: Opaque lane settings — server argv for a server lane, technique names
    #: and thresholds for an in-process one. The lane factory interprets it;
    #: this module only records it.
    settings: Dict[str, Any] = msgspec.field(default_factory=dict)
    #: Request-body fragment merged over the plan's base payload (step count,
    #: resolution, duration — the COMPILED GRAPH half of "technique x compiled graph").
    payload: Dict[str, Any] = msgspec.field(default_factory=dict)
    #: The arm this one's speedup is computed against. Empty = the plan's first
    #: arm. A speedup with no stated denominator is not a number.
    baseline: str = ""
    #: Published expectation WITH its citation, so a measured/expected gap is
    #: visible in the output rather than in someone's memory.
    expect: str = ""
    #: APPROXIMATE arms change the output and need a quality verdict before
    #: adoption; LOSSLESS ones do not. Every published H3 quality number is
    #: VIDEO-ONLY, so an approximate arm that has not been through the audio
    #: gate (cozy-eval#8) is not adoptable whatever this harness measures.
    approximate: bool = True
    #: Non-empty = do not run, and say why (a kernel needing an SM this pod
    #: lacks, a technique with a known-open upstream defect).
    skip: str = ""


class Plan(msgspec.Struct, frozen=True, kw_only=True):
    """A named matrix: base payload + arms."""

    name: str
    #: Applied to every arm, then overridden by the arm's own ``payload``.
    payload: Dict[str, Any] = msgspec.field(default_factory=dict)
    arms: Tuple[Arm, ...] = ()
    #: Timed repetitions per arm AFTER the warmup reps. MEDIANS, not means:
    #: warm variance measures 2.7x on an identical payload, and a mean over
    #: that is a number about the tail, not about the technique.
    reps: int = 3
    #: Discarded reps run first. A cold first call is a different measurement
    #: (measured 4.4x cold/warm) and mixing them hides both.
    warmup_reps: int = 1
    notes: str = ""


class Row(msgspec.Struct, frozen=True, kw_only=True):
    """One timed repetition. Written to the output stream immediately."""

    arm: str
    rep: int
    warmup: bool
    wall_s: float
    ok: bool
    error: str = ""
    #: Whatever residency number the lane can state. For fp8 arms the
    #: residency IS the prize (-38% memory at 1.056x speed), so an arm that
    #: reports no VRAM has not measured its own headline.
    vram_bytes: int = 0
    #: Lane-supplied per-rep facts (cache hits/misses, engaged attention
    #: backend, degraded rung). Free-form because lanes differ; recorded
    #: verbatim because a cache that silently no-ops looks exactly like a
    #: cache that is not helping (see the cache-dit H3 aliasing defect).
    detail: Dict[str, Any] = msgspec.field(default_factory=dict)
    unix_s: float = 0.0


class ArmResult(msgspec.Struct, frozen=True, kw_only=True):
    """Per-arm verdict rolled up from its rows."""

    arm: str
    tier: str = ""
    ok: bool = False
    status: str = ""              # measured | skipped | open_failed | ...
    reps_ok: int = 0
    median_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    open_s: float = 0.0
    peak_vram_bytes: int = 0
    baseline: str = ""
    speedup_vs_baseline: float = 0.0
    expect: str = ""
    approximate: bool = True
    detail: str = ""


class Lane(Protocol):
    """One armed configuration, ready to serve. Opened per arm, closed after.

    One lane per arm is not waste: a server flag like ``--quantization`` is
    read at engine start and an in-process technique is installed at pipeline
    build, so re-using a warm lane across arms would measure the first arm
    forever.
    """

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Drive ONE generation to completion. Return per-rep facts (may be
        empty). Raising fails that repetition, not the run."""

    def vram_bytes(self) -> int:
        """Best residency number this lane can state; 0 when it cannot."""

    def close(self) -> None:
        """Tear down. Must be safe to call after a failed ``run``."""


#: ``open_lane(arm, payload) -> Lane``. Raising here fails the ARM (recorded),
#: never the run — an unbuildable technique is a result.
LaneFactory = Callable[[Arm, Dict[str, Any]], Lane]


# --- plan loading -----------------------------------------------------------


def load_plan(path: str) -> Plan:
    """Decode a plan file. Unknown fields are a hard error — a mistyped arm key
    that silently became a no-op would produce a real-looking number for a
    configuration nobody ran."""
    with open(path, "rb") as f:
        return msgspec.json.decode(f.read(), type=Plan, strict=True)


def select_arms(
    plan: Plan, *, tier: str = "", names: Sequence[str] = ()
) -> Tuple[Arm, ...]:
    """Arms matching ``tier`` and/or ``names``, in plan order.

    Each picked arm's baseline is pulled in even when it matches no filter: an
    arm set with no denominator can produce timings but not speedups.
    """
    wanted = set(names)
    picked = [
        a for a in plan.arms
        if (not tier or a.tier == tier) and (not wanted or a.name in wanted)
    ]
    if not picked:
        return ()
    by_name = {a.name: a for a in plan.arms}
    have = {a.name for a in picked}
    default_base = plan.arms[0].name if plan.arms else ""
    for arm in list(picked):
        base = arm.baseline or default_base
        if base and base not in have and base in by_name:
            have.add(base)
    return tuple(a for a in plan.arms if a.name in have)


# --- the run ----------------------------------------------------------------


class _Budget:
    """Wall cap over the WHOLE run. Standing policy: no unwatched GPU run."""

    def __init__(self, wall_cap_s: float) -> None:
        self.cap_s = float(wall_cap_s)
        self.started = time.monotonic()

    @property
    def remaining_s(self) -> float:
        return self.cap_s - (time.monotonic() - self.started)

    def affords(self, need_s: float) -> bool:
        return self.cap_s <= 0 or self.remaining_s >= need_s


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class _Sink:
    """Append-as-you-go JSONL. A killed run keeps every finished row."""

    def __init__(self, path: str = "") -> None:
        self.path = path
        self.rows: List[Row] = []
        self.results: List[ArmResult] = []

    def row(self, row: Row) -> None:
        self.rows.append(row)
        self._write({"type": "row", **msgspec.structs.asdict(row)})

    def result(self, result: ArmResult) -> None:
        self.results.append(result)
        self._write({"type": "arm", **msgspec.structs.asdict(result)})

    def _write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, sort_keys=True, default=str)
        print(line, flush=True)
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            logger.exception("matrix row could not be persisted to %s", self.path)


def run_matrix(
    plan: Plan,
    arms: Sequence[Arm],
    *,
    open_lane: LaneFactory,
    wall_cap_s: float = 0.0,
    arm_budget_s: float = 0.0,
    out_path: str = "",
) -> List[ArmResult]:
    """Open each arm's lane, time it, tear it down, record it."""
    budget = _Budget(wall_cap_s)
    sink = _Sink(out_path)
    by_name: Dict[str, ArmResult] = {}
    default_base = arms[0].name if arms else ""

    for arm in arms:
        payload = {**plan.payload, **arm.payload}
        baseline = arm.baseline or (default_base if arm.name != default_base else "")
        stub: Dict[str, Any] = dict(
            arm=arm.name, tier=arm.tier, baseline=baseline, expect=arm.expect,
            approximate=arm.approximate)
        if arm.skip:
            sink.result(ArmResult(ok=False, status="skipped", detail=arm.skip, **stub))
            continue
        if not budget.affords(arm_budget_s):
            sink.result(ArmResult(
                ok=False, status="out_of_budget",
                detail=(f"{budget.remaining_s:.0f}s left of a {budget.cap_s:.0f}s "
                        f"cap, arm needs {arm_budget_s:.0f}s"), **stub))
            continue

        result = _run_arm(
            plan, arm, stub=stub, payload=payload, open_lane=open_lane,
            budget=budget, arm_budget_s=arm_budget_s, sink=sink)
        base = by_name.get(baseline)
        if base is not None and base.median_s > 0 and result.median_s > 0:
            result = msgspec.structs.replace(
                result,
                speedup_vs_baseline=round(base.median_s / result.median_s, 4))
        by_name[arm.name] = result
        sink.result(result)

    return list(sink.results)


def _run_arm(
    plan: Plan,
    arm: Arm,
    *,
    stub: Dict[str, Any],
    payload: Dict[str, Any],
    open_lane: LaneFactory,
    budget: _Budget,
    arm_budget_s: float,
    sink: _Sink,
) -> ArmResult:
    logger.info("arm %s: opening lane %s", arm.name, arm.settings or "{}")
    started = time.monotonic()
    lane: Optional[Lane] = None
    try:
        lane = open_lane(arm, payload)
    except Exception as exc:
        return ArmResult(
            ok=False, status="open_failed", open_s=round(time.monotonic() - started, 3),
            detail=f"{type(exc).__name__}: {exc}", **stub)

    open_s = time.monotonic() - started
    deadline = time.monotonic() + arm_budget_s if arm_budget_s > 0 else 0.0
    timed: List[float] = []
    peak_vram = 0
    note = ""
    try:
        for i in range(plan.warmup_reps + plan.reps):
            warm = i < plan.warmup_reps
            if deadline and time.monotonic() > deadline:
                note = f"arm budget {arm_budget_s:.0f}s exhausted after {i} reps"
                break
            if not budget.affords(0.0):
                note = f"run wall cap {budget.cap_s:.0f}s exhausted after {i} reps"
                break
            rep_started = time.monotonic()
            ok, err, facts = True, "", {}
            try:
                facts = dict(lane.run(payload) or {})
            except Exception as exc:
                ok, err = False, f"{type(exc).__name__}: {exc}"
            wall = time.monotonic() - rep_started
            try:
                vram = int(lane.vram_bytes())
            except Exception:
                vram = 0
            peak_vram = max(peak_vram, vram)
            sink.row(Row(
                arm=arm.name, rep=i, warmup=warm, wall_s=round(wall, 4), ok=ok,
                error=err, vram_bytes=vram, detail=facts, unix_s=time.time()))
            if ok and not warm:
                timed.append(wall)
    finally:
        try:
            lane.close()
        except Exception:
            logger.exception("arm %s: lane teardown failed", arm.name)

    if not timed:
        return ArmResult(
            ok=False, status="no_successful_reps", open_s=round(open_s, 3),
            peak_vram_bytes=peak_vram,
            detail=note or "every timed repetition failed", **stub)
    return ArmResult(
        ok=True, status="measured", reps_ok=len(timed),
        median_s=round(_median(timed), 4), min_s=round(min(timed), 4),
        max_s=round(max(timed), 4), open_s=round(open_s, 3),
        peak_vram_bytes=peak_vram, detail=note, **stub)


def summarize(results: Sequence[ArmResult]) -> str:
    """A markdown table, because the verdict lands in a tracker issue."""
    lines = [
        ("| arm | tier | status | median s | speedup | peak VRAM GiB | kind | "
         "expected (published) |"),
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        speed = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else "—"
        vram = (f"{r.peak_vram_bytes / float(1024 ** 3):.1f}"
                if r.peak_vram_bytes else "—")
        median = f"{r.median_s:.2f}" if r.median_s else "—"
        note = r.expect
        if r.detail:
            note = (note + " — " if note else "") + r.detail
        lines.append(
            f"| `{r.arm}` | {r.tier} | {r.status} | {median} | {speed} | {vram} | "
            f"{'approx' if r.approximate else 'LOSSLESS'} | {note} |")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------


def _resolve_lane_factory(spec: str) -> LaneFactory:
    """``package.module:attribute`` -> the callable.

    The lane lives OUTSIDE this wheel on purpose: it is the thing that knows
    about a specific model, pipeline and technique set, and baking one in would
    make this driver a model-specific script.
    """
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise SystemExit(f"--lane must be 'module:attribute', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--plan", required=True, help="plan JSON path")
    ap.add_argument("--lane", required=True, help="module:attribute lane factory")
    ap.add_argument("--tier", default="", help="run only this tier")
    ap.add_argument("--arm", action="append", default=[], help="run only these arms")
    ap.add_argument("--out", default="", help="JSONL evidence path (append)")
    ap.add_argument("--wall-cap-s", type=float, default=0.0,
                    help="hard cap over the WHOLE run; 0 = uncapped (never on a pod)")
    ap.add_argument("--arm-budget-s", type=float, default=0.0,
                    help="per-arm cap; an arm that cannot fit is skipped, not started")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plan = load_plan(args.plan)
    arms = select_arms(plan, tier=args.tier, names=args.arm)
    if not arms:
        raise SystemExit(f"plan {plan.name!r}: no arms matched tier/arm filters")
    if args.wall_cap_s <= 0:
        logger.warning(
            "running UNCAPPED — standing policy requires a wall cap on every "
            "cloud GPU run; pass --wall-cap-s")
    results = run_matrix(
        plan, arms, open_lane=_resolve_lane_factory(args.lane),
        wall_cap_s=args.wall_cap_s, arm_budget_s=args.arm_budget_s,
        out_path=args.out)
    print(summarize(results))
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
