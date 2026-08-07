"""pgw#1011: the acceleration-matrix driver, exercised through its real entry
points with an in-process lane standing in for a GPU one.

The lane is the ONLY thing faked, and it is faked at the seam the driver
publishes (:class:`Lane`) rather than by patching internals — so `run_matrix`,
`select_arms`, `load_plan`, the budget, the JSONL sink and the summary all run
exactly as they will on a pod.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker.benchmarks import engine_matrix as em

PLAN_PATH = (
    Path(em.__file__).resolve().parent / "plans" / "h3_speed.json"
)


class _Lane:
    """A lane whose cost is declared by the arm, so timings are predictable."""

    def __init__(self, arm: em.Arm, payload: Dict[str, Any], log: List[str]) -> None:
        self.arm = arm
        self.payload = payload
        self.log = log
        self.closed = False
        self.calls = 0
        log.append(f"open:{arm.name}")

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        if self.arm.settings.get("boom"):
            raise RuntimeError("technique refused to arm")
        time.sleep(float(self.arm.settings.get("cost_s", 0.01)))
        return {"steps": payload.get("steps"), "engaged": self.arm.name}

    def vram_bytes(self) -> int:
        return int(self.arm.settings.get("vram", 0))

    def close(self) -> None:
        self.closed = True
        self.log.append(f"close:{self.arm.name}")


def _factory(log: List[str]):
    def open_lane(arm: em.Arm, payload: Dict[str, Any]) -> em._Lane:  # type: ignore[name-defined]
        if arm.settings.get("unbuildable"):
            raise ImportError("cache-dit 1.3.0 is not installed")
        return _Lane(arm, payload, log)
    return open_lane


def _plan(*arms: em.Arm, **kw: Any) -> em.Plan:
    kw.setdefault("reps", 2)
    kw.setdefault("warmup_reps", 1)
    kw.setdefault("payload", {"steps": 50})
    return em.Plan(name="t", arms=tuple(arms), **kw)


def test_shipped_h3_plan_decodes_and_every_arm_states_a_denominator() -> None:
    """The plan the GPU lane will actually run must load under the strict
    decoder — a mistyped arm key would otherwise become a silent no-op that
    produces a real-looking number for a configuration nobody ran."""
    plan = em.load_plan(str(PLAN_PATH))
    assert plan.name == "h3_speed"
    names = [a.name for a in plan.arms]
    assert names[0] == "base_50", "the denominator must be the first arm"
    assert len(names) == len(set(names)), "arm names must be unique"
    for arm in plan.arms:
        assert arm.expect, f"{arm.name}: an arm with no published expectation is a guess"
        if arm.baseline:
            assert arm.baseline in names, f"{arm.name}: baseline {arm.baseline} not in plan"

    # The three arms that are output-exact must be declared as such: a lossless
    # technique needs no quality gate, and mislabelling one as approximate
    # would park a free win behind cozy-eval#8 forever.
    lossless = {a.name for a in plan.arms if not a.approximate}
    assert {"adaln_cache", "warmup_matched", "attn_cudnn"} <= lossless

    # Both known-defective techniques ship SKIPPED with the defect named, so a
    # pod run cannot accidentally measure them.
    skipped = {a.name: a.skip for a in plan.arms if a.skip}
    assert "sage_fp16_pv" in skipped and "15263" in skipped["sage_fp16_pv"]
    assert "turbo_lora_8" in skipped


def test_speedup_is_computed_against_the_named_baseline() -> None:
    log: List[str] = []
    plan = _plan(
        em.Arm(name="base", tier="t0", settings={"cost_s": 0.04}),
        em.Arm(name="fast", tier="t0", settings={"cost_s": 0.01}),
    )
    results = em.run_matrix(plan, plan.arms, open_lane=_factory(log))
    by = {r.arm: r for r in results}
    assert by["base"].ok and by["fast"].ok
    assert by["base"].speedup_vs_baseline == 0.0, "the denominator has no speedup"
    assert by["fast"].speedup_vs_baseline > 2.0
    # One lane per arm, and every lane closed — an arm that leaked a lane would
    # leave weights resident and poison the next arm's residency number.
    assert log == ["open:base", "close:base", "open:fast", "close:fast"]


def test_warmup_reps_are_excluded_from_the_median() -> None:
    """A cold first call is a different measurement (ie#612 measured 4.4x) and
    mixing it in hides both."""
    plan = _plan(em.Arm(name="a", settings={"cost_s": 0.01}), reps=2, warmup_reps=1)
    results = em.run_matrix(plan, plan.arms, open_lane=_factory([]))
    assert results[0].reps_ok == 2


def test_a_failed_arm_is_a_recorded_result_not_a_dead_run() -> None:
    log: List[str] = []
    plan = _plan(
        em.Arm(name="base", settings={"cost_s": 0.01}),
        em.Arm(name="unbuildable", settings={"unbuildable": True}),
        em.Arm(name="refuses", settings={"boom": True}),
        em.Arm(name="after", settings={"cost_s": 0.01}),
    )
    results = em.run_matrix(plan, plan.arms, open_lane=_factory(log))
    by = {r.arm: r for r in results}
    assert by["unbuildable"].status == "open_failed"
    assert "cache-dit" in by["unbuildable"].detail
    assert by["refuses"].status == "no_successful_reps"
    assert by["after"].ok, "a failed arm must not end the matrix"
    # The arm whose lane opened is still closed even though every rep failed.
    assert "close:refuses" in log


def test_skip_and_wall_cap_refuse_before_spending() -> None:
    plan = _plan(
        em.Arm(name="base", settings={"cost_s": 0.01}),
        em.Arm(name="known_broken", skip="Comfy-Org #15263: pure noise on H3"),
        em.Arm(name="expensive", settings={"cost_s": 0.01}),
    )
    log: List[str] = []
    results = em.run_matrix(
        plan, plan.arms, open_lane=_factory(log),
        wall_cap_s=0.001, arm_budget_s=600.0)
    by = {r.arm: r for r in results}
    assert by["known_broken"].status == "skipped"
    assert "15263" in by["known_broken"].detail
    assert by["expensive"].status == "out_of_budget"
    # Neither refused arm opened a lane: the cap has to bite BEFORE the spend,
    # not after (standing policy — no unwatched, unbounded GPU run).
    assert "open:known_broken" not in log and "open:expensive" not in log


def test_rows_are_persisted_as_they_complete(tmp_path: Path) -> None:
    """RunPod has no container-log API, so a row that only reached stdout is a
    row we do not have. Every completed row must already be on disk."""
    out = tmp_path / "matrix.jsonl"
    plan = _plan(em.Arm(name="a", settings={"cost_s": 0.01, "vram": 1024}))
    em.run_matrix(plan, plan.arms, open_lane=_factory([]), out_path=str(out))
    written = [json.loads(line) for line in out.read_text().splitlines()]
    rows = [r for r in written if r["type"] == "row"]
    arms = [r for r in written if r["type"] == "arm"]
    assert len(rows) == 3 and len(arms) == 1
    assert rows[0]["warmup"] is True and rows[1]["warmup"] is False
    assert rows[0]["detail"]["steps"] == 50, "lane facts ride the row verbatim"
    assert arms[0]["peak_vram_bytes"] == 1024


def test_select_arms_pulls_in_the_denominator_a_filter_would_drop() -> None:
    plan = _plan(
        em.Arm(name="base", tier="tier0"),
        em.Arm(name="cache", tier="tier1", baseline="base"),
        em.Arm(name="other", tier="tier2"),
    )
    picked = [a.name for a in em.select_arms(plan, tier="tier1")]
    assert picked == ["base", "cache"], "a filtered arm keeps its denominator"
    assert em.select_arms(plan, names=["nope"]) == ()


def test_summary_marks_lossless_arms_distinctly() -> None:
    plan = _plan(
        em.Arm(name="base", settings={"cost_s": 0.02}),
        em.Arm(name="exact", approximate=False, settings={"cost_s": 0.01},
               expect="bit-identical (NVlabs AdaLN precompute)"),
    )
    table = em.summarize(em.run_matrix(plan, plan.arms, open_lane=_factory([])))
    assert "LOSSLESS" in table and "approx" in table
    assert "NVlabs" in table


def test_cli_refuses_a_plan_with_no_matching_arms(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        em.main([
            "--plan", str(PLAN_PATH), "--tier", "nonexistent",
            "--lane", "gen_worker.benchmarks.engine_matrix:load_plan",
        ])
