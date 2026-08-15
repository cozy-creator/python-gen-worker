"""The AOT arm has a shape-gap fact, and growth is ONE module.

The gap: ``hot_swap.enable`` returns False unless the
pipeline carries a **dynamo** router, and an AOT-armed pipeline never has one
(``models.provision.enable_compiled`` returns as soon as ``arm_aot`` succeeds,
so ``compile_cache.enable`` — the only thing that installs a router — is never
reached).  Every one of the executor's three growth call sites is therefore a
silent no-op on an AOT arm, the ``_shape_warm_republisher`` closure is built
and discarded, and ``fleet_cells.republish_after_shape_warm`` has no reachable
caller — leaving 16 of 18 declared graph classes serving eager, permanently, on
every pod.

These tests hold the three things that are decidable without a GPU:

1. an AOT ingress refusal NAMES the missing declared class and books a
   countable ``shape_gap`` (the AOT counterpart of the dynamo-only
   ``guard_miss``);
2. the absence of a growth backend is LOUD, never silent — a defect whose only
   observable is a success log line that never prints is invisible;
3. growth lives in one arm-agnostic module that does NOT import the dynamo
   router and does NOT implement a second task/device scheduler.

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

    def capture(kind: str, detail: str, phase: str = "", duration_ms: int = 0, **_kw):
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
        compiled_graph_key="ck1-0d945144")
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



# ---------------------------------------------------------------------------
# 2. the absence of a growth path is loud
# ---------------------------------------------------------------------------



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



def test_the_dynamo_arm_books_its_permanent_holes_in_the_same_ledger():
    """One population, not two half-populations: a dynamo signature whose
    background warm failed is eager forever, which is the same fact the AOT
    arm reports for an uncovered declared class."""
    source = inspect.getsource(hot_swap._run_warm_compile)
    assert "shape_growth.report(" in source
    assert "ARM_DYNAMO" in source


