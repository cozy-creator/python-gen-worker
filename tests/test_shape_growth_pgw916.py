"""pgw#916 — the AOT arm has a shape-gap fact, and growth is ONE module.

The gap this issue names: ``hot_swap.enable`` returns False unless the
pipeline carries a **dynamo** router, and an AOT-armed pipeline never has one
(``models.provision.enable_compiled`` returns as soon as ``arm_aot`` succeeds,
so ``compile_cache.enable`` — the only thing that installs a router — is never
reached).  Every one of the executor's three growth call sites is therefore a
silent no-op on an AOT arm, the ``_shape_warm_republisher`` closure is built
and discarded, and ``fleet_cells.republish_after_shape_warm`` has no reachable
caller.  Measured cost on the standing stack: **16 of 18 declared graph
classes serve eager**, permanently, on every pod.

These tests hold the three things that are decidable without a GPU:

1. an AOT ingress refusal NAMES the missing declared class and books a
   countable ``shape_gap`` (the AOT counterpart of pgw#680's ``guard_miss``,
   which is dynamo-only);
2. the absence of a growth backend is LOUD, never silent — the defect was
   invisible for months precisely because the only observable was a success
   log line that never printed;
3. growth lives in one arm-agnostic module that does NOT import the dynamo
   router and does NOT implement a second task/device scheduler.  Building one
   here to close the AOT arm sooner would violate this issue's own acceptance.

Real modules, real events (captured through the activity sink), no mocks of
the code under test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import activity as activity_mod
from gen_worker import aot_serve, hot_swap, shape_growth


@pytest.fixture(autouse=True)
def _clean_ledger():
    shape_growth.LEDGER.clear()
    yield
    shape_growth.LEDGER.clear()


@pytest.fixture
def events(monkeypatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []

    def capture(kind: str, detail: str, phase: str = "", duration_ms: int = 0):
        seen.append((kind, phase, detail))

    monkeypatch.setattr(activity_mod, "emit_event", capture)
    monkeypatch.setattr(shape_growth.activity_mod, "emit_event", capture)
    return seen


# ---------------------------------------------------------------------------
# 1. the missing class is NAMED, and countable
# ---------------------------------------------------------------------------


class _Tensorish:
    def __init__(self, shape: Tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


def test_the_refused_call_names_a_declared_class_not_just_a_shape():
    """A shape is not a growable unit of work: a mint is asked for a CLASS,
    which is the whole ingress coordinate of one target."""
    name = aot_serve.ingress_class_name(
        "unet",
        (),
        {"sample": _Tensorish((2, 4, 112, 144), "torch.bfloat16"),
         "encoder_hidden_states": _Tensorish((2, 77, 2048), "torch.bfloat16"),
         "return_dict": False},
    )
    assert name.startswith("unet/")
    assert "sample=bfloat16[2,4,112,144]" in name
    assert "encoder_hidden_states=bfloat16[2,77,2048]" in name
    # A scalar that dynamo would specialize on is part of the coordinate too.
    assert "return_dict=False" in name
    # Two aspect rows of one area-preserving family are DIFFERENT classes at
    # a whole-graph target — the naming must not collapse them.
    other = aot_serve.ingress_class_name(
        "unet", (), {"sample": _Tensorish((2, 4, 144, 112), "torch.bfloat16")})
    assert other != aot_serve.ingress_class_name(
        "unet", (), {"sample": _Tensorish((2, 4, 112, 144), "torch.bfloat16")})


def test_an_aot_shape_gap_is_a_countable_typed_fact(events):
    gap = shape_growth.ShapeGap(
        arm=shape_growth.ARM_AOT, family="sdxl", target="unet",
        declared_class="unet/sample=bfloat16[2,4,112,144]",
        reason=shape_growth.REASON_UNCOVERED,
        cell_key="ck1-0d945144")
    assert shape_growth.report(gap) is True
    kinds = [(k, p) for k, p, _d in events]
    assert (activity_mod.KIND_SHAPE_GAP, "no_entry_admits") in kinds
    detail = next(d for k, _p, d in events if k == activity_mod.KIND_SHAPE_GAP)
    assert "arm=aot" in detail and "class=unet/sample=" in detail


def test_one_uncovered_class_books_one_growth_job_however_many_requests(events):
    gap = shape_growth.ShapeGap(
        arm=shape_growth.ARM_AOT, family="sdxl", target="unet",
        declared_class="unet/x", reason=shape_growth.REASON_UNCOVERED)
    assert shape_growth.report(gap) is True
    for _ in range(40):
        assert shape_growth.report(gap) is False
    assert shape_growth.LEDGER.counts()[gap.key] == 41
    assert len([1 for k, _p, _d in events
                if k == activity_mod.KIND_SHAPE_GAP]) == 1


def test_an_ambiguous_dispatch_is_recorded_but_never_submitted(events):
    """pgw#917's failure is a DECLARATION defect. Compiling another entry
    cannot fix it, so growth must not try — and must say why."""
    calls: List[shape_growth.ShapeGap] = []

    class _Backend:
        def grow(self, gap: shape_growth.ShapeGap) -> bool:
            calls.append(gap)
            return True

    shape_growth.register_backend(shape_growth.ARM_AOT, _Backend())
    try:
        assert shape_growth.report_and_submit(shape_growth.ShapeGap(
            arm=shape_growth.ARM_AOT, family="sdxl",
            target="unet[BasicTransformerBlock#0]",
            declared_class="unet[BasicTransformerBlock#0]/h=bf16[2,16128,640]",
            reason=shape_growth.REASON_AMBIGUOUS)) is False
        assert not calls
        # ...and it is still COUNTED, under its own phase.
        assert (activity_mod.KIND_SHAPE_GAP, "entry_ambiguous") in [
            (k, p) for k, p, _d in events]
    finally:
        shape_growth.register_backend(shape_growth.ARM_AOT, None)


def test_a_growable_gap_reaches_the_registered_backend_once():
    calls: List[shape_growth.ShapeGap] = []

    class _Backend:
        def grow(self, gap: shape_growth.ShapeGap) -> bool:
            calls.append(gap)
            return True

    shape_growth.register_backend(shape_growth.ARM_AOT, _Backend())
    try:
        gap = shape_growth.ShapeGap(
            arm=shape_growth.ARM_AOT, family="sdxl", target="unet",
            declared_class="unet/y", reason=shape_growth.REASON_UNCOVERED)
        assert shape_growth.report_and_submit(gap) is True
        assert shape_growth.report_and_submit(gap) is False
        assert [g.declared_class for g in calls] == ["unet/y"]
    finally:
        shape_growth.register_backend(shape_growth.ARM_AOT, None)


# ---------------------------------------------------------------------------
# 2. the absence of a growth path is loud
# ---------------------------------------------------------------------------


def test_no_backend_is_a_named_refusal_not_a_silent_noop(caplog):
    gap = shape_growth.ShapeGap(
        arm=shape_growth.ARM_AOT, family="sdxl", target="unet",
        declared_class="unet/z", reason=shape_growth.REASON_UNCOVERED)
    assert shape_growth.backend_for(shape_growth.ARM_AOT) is None
    with caplog.at_level("WARNING"):
        assert shape_growth.submit(gap) is False
    assert "no growth backend registered for arm='aot'" in caplog.text
    assert "stays EAGER" in caplog.text


def test_the_executor_confesses_an_armed_target_with_no_growth_path():
    """The named observable this issue found absent. ``hot_swap.enable``
    returning False must produce a countable fact, not nothing."""
    from gen_worker import executor

    source = inspect.getsource(executor.Executor._report_no_growth_path)
    assert "KIND_SHAPE_GAP" in source
    assert "no_growth_path" in source
    # ...and it is actually reached from the post-proof growth call site.
    caller = inspect.getsource(executor.Executor)
    assert "self._report_no_growth_path(spec, target, pipeline)" in caller


# ---------------------------------------------------------------------------
# 3. one module, no second scheduler
# ---------------------------------------------------------------------------


def _imports_of(module: Any) -> List[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.extend(f"{base}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    return names


def test_growth_does_not_import_the_dynamo_router():
    """Arm-agnostic by dependency, not by intention. ``compile_cache`` owns
    the router the AOT arm does not have; importing it here would rebuild the
    coupling this issue exists to break."""
    imported = _imports_of(shape_growth)
    assert not [n for n in imported if "compile_cache" in n], imported
    assert not [n for n in imported if "hot_swap" in n], imported
    assert not [n for n in imported if "aot_serve" in n], imported


def test_the_dependency_edge_points_at_the_shared_module():
    """Both arms REACH it: the dynamo arm imports it (and uses its turn-gate
    and debounce types), the AOT serve path imports it."""
    assert any("shape_growth" in n for n in _imports_of(hot_swap))
    assert any("shape_growth" in n for n in _imports_of(aot_serve))
    # One implementation, not two copies that can drift.
    assert hot_swap.Debounce is shape_growth.Debounce
    assert hot_swap.TurnGateBusy is shape_growth.TurnGateBusy
    assert hot_swap.TurnGateClosed is shape_growth.TurnGateClosed


def test_growth_implements_no_second_task_or_device_scheduler():
    """The acceptance forbids it by name. ``Debounce``'s single serialized
    republish thread is the ONE lifted primitive; anything queue- or
    pool-shaped here would be the second scheduler."""
    source = Path(shape_growth.__file__).read_text(encoding="utf-8")
    for banned in ("queue.Queue", "ThreadPoolExecutor", "asyncio.Queue",
                   "cuda.Stream", "set_device"):
        assert banned not in source, banned
    threads = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr == "Thread"]
    assert len(threads) == 1, "only Debounce's serialized republish thread"


def test_the_dynamo_arm_books_its_permanent_holes_in_the_same_ledger():
    """One population, not two half-populations: a dynamo signature whose
    background warm failed is eager forever, which is the same fact the AOT
    arm reports for an uncovered declared class."""
    source = inspect.getsource(hot_swap._run_warm_compile)
    assert "shape_growth.report(" in source
    assert "ARM_DYNAMO" in source


def test_coverage_reports_convergence_not_only_the_initial_gap():
    declared = [f"unet/class{i}" for i in range(18)]
    assert "18/18" in shape_growth.coverage_line(declared, {})
    gaps: Dict[Tuple[str, str, str], int] = {
        ("sdxl", "unet", name): 3 for name in declared[2:]}
    line = shape_growth.coverage_line(declared, gaps)
    assert "2/18" in line and "16 still serve EAGER" in line
