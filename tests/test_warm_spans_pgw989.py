from __future__ import annotations

import sys
import types
from typing import Dict, List

import pytest
from gen_worker._vendor.torchcg import spans

from gen_worker import warm_spans


REAL_DELTA: Dict[str, float] = {
    "compile_file": 8.638,
    "_compile.compile_inner": 5.054,
    "compile_attempt_0": 5.022,
    "OutputGraph.call_user_compiler": 4.927,
    "create_aot_dispatcher_function": 4.355,
    "compile_fx.<locals>.fw_compiler_base": 4.326,
    "compile_fx_inner": 4.163,
    "fx_codegen_and_compile": 4.163,
    "GraphLowering.compile_to_fn": 4.091,
    "GraphLowering.compile_to_module": 4.091,
    "PyCodeCache.load_by_key_path": 3.203,
    "async_compile.wait": 2.886,
    "GraphLowering.codegen": 0.882,
    "Scheduler.codegen": 0.828,
    "CacheBase.get_system.triton_key": 0.424,
    "_recursive_joint_graph_passes": 0.163,
    "inductor_codecache_torch_key": 0.112,
    "bytecode_tracing": 0.087,
    "variable_builder_call": 0.062,
    "Scheduler.__init__": 0.045,
    "GraphLowering.run": 0.039,
    "build_guards": 0.023,
    "_recursive_post_grad_passes": 0.016,
    "_recursive_pre_grad_passes": 0.004,
}


def test_partition_sums_to_the_dynamo_total_with_a_named_residual() -> None:
    members, _overlays = warm_spans.partition(REAL_DELTA)
    total = members["dynamo_compile_s"]
    assert total == REAL_DELTA[warm_spans.TOTAL_KEY]
    named = sum(v for k, v in members.items() if k != "dynamo_compile_s")
    assert named == pytest.approx(total, abs=0.01)
    assert members["compile_other_s"] > 0


def test_the_aot_partition_would_have_missed_this_compile() -> None:
    """The reason this module exists rather than reusing the AOT key set."""
    aot_members, _o, _raw = spans.phase_delta({}, REAL_DELTA)
    aot_total = sum(aot_members.values())
    jit_members, _ = warm_spans.partition(REAL_DELTA)

    assert aot_members["host_compile_s"] == 0.0
    assert aot_total < 0.3 * jit_members["dynamo_compile_s"]
    assert jit_members["kernel_compile_s"] == pytest.approx(3.203)


def test_overlays_are_reported_and_never_summed_into_the_partition() -> None:
    members, overlays = warm_spans.partition(REAL_DELTA)
    assert overlays["async_wait_s"] == pytest.approx(2.886)
    assert overlays["parallel_kernel_cpu_s"] > members["dynamo_compile_s"]
    assert set(overlays) & set(members) == set()


class _FakeDynamoUtils:

    def __init__(self) -> None:
        self.compilation_time_metrics: Dict[str, List[float]] = {}

    def add(self, delta: Dict[str, float]) -> None:
        for key, value in delta.items():
            self.compilation_time_metrics.setdefault(key, []).append(value)


@pytest.fixture()
def fake_dynamo(monkeypatch: pytest.MonkeyPatch) -> _FakeDynamoUtils:
    utils = _FakeDynamoUtils()
    torch = types.ModuleType("torch")
    dynamo = types.ModuleType("torch._dynamo")
    torch._dynamo = dynamo  # type: ignore[attr-defined]
    dynamo.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", dynamo)
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", utils)
    return utils


def test_ledger_splits_compile_from_forward_per_job(
    fake_dynamo: _FakeDynamoUtils,
) -> None:
    ledger = warm_spans.WarmLedger()

    with ledger.job("generate/a"):
        fake_dynamo.add(REAL_DELTA)
    with ledger.job("generate/b"):
        pass

    table = ledger.table()
    totals = table["totals"]
    assert totals["warm_jobs"] == 2
    assert totals["warm_jobs_compiling"] == 1
    assert totals["warm_compile_s"] == pytest.approx(5.054)
    assert totals["warm_execute_s"] == pytest.approx(
        totals["warm_wall_s"] - totals["warm_compile_s"], abs=0.01)
    assert "compile_fraction" not in totals

    rows = {r["job"]: r for r in table["jobs"]}
    assert rows["generate/a"]["compile_s"] == pytest.approx(5.054)
    assert rows["generate/b"]["compile_s"] == 0.0


def test_an_empty_plan_reports_absence_not_zero_cost() -> None:
    table = warm_spans.WarmLedger().table()
    assert table["totals"]["warm_wall_s"] == 0.0
    assert table["totals"]["warm_jobs"] == 0
    assert table["jobs"] == []


def test_a_failing_warm_job_still_lands_in_the_ledger(
    fake_dynamo: _FakeDynamoUtils,
) -> None:
    ledger = warm_spans.WarmLedger()
    with pytest.raises(RuntimeError, match="boom"):
        with ledger.job("generate/dies"):
            fake_dynamo.add({"_compile.compile_inner": 2.0})
            raise RuntimeError("boom")
    assert ledger.table()["totals"]["warm_jobs"] == 1
    assert ledger.jobs[0]["compile_s"] == pytest.approx(2.0)

