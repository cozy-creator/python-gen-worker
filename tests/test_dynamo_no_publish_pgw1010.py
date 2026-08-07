"""pgw#1010 — the JIT recipe serves, and produces nothing.

Paul's ratified reuse ruling ("reuse = AOT cells only; JIT = intake mode,
honest cold boots") composed with two facts already in the tree:
``aot_cells._candidates`` rejects ``kind="torch-inductor-cache"`` BY NAME, and
nothing else has ever adopted one. A dynamo cell therefore had zero possible
consumers, and every one minted was pod time and platform storage spent on an
artifact that could not be adopted — attempt 23's ``ck5-a53e02a7…`` among them.

The dynamo recipe KEEPS its serving role (intake for a family with no export
declaration — ruled, untouched). What it loses is the artifact: no seal, no
key, no publish, no ``cell_store`` row, no mint obligation.

Two guards, because either alone is weak:

* a SOURCE guard (the ``mint_recipe`` silent-decline AST guard's style): the
  intake branch cannot reach the seal/publish vocabulary, and the vocabulary
  itself is gone where it was dynamo-only. A future edit that re-wires a pack
  onto this branch fails here rather than on a rented pod;
* a BEHAVIOURAL guard: drive the real arming policy on a family with no export
  declaration and assert what the pod does — armed (it compiles and serves),
  no pending, no publisher call, and a typed decline on the wire.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import activity as activity_mod
from gen_worker import cell_adopt
from gen_worker import compile_cache as cc
from gen_worker import fleet_cells as fc
from gen_worker import serving_mode
from gen_worker.models import provision


# ---------------------------------------------------------------------------
# source guard
# ---------------------------------------------------------------------------

#: Every name that seals a cell, publishes one, or records one hub-side. A
#: call to any of these from the intake branch is the defect this file exists
#: to make impossible.
_SEAL_PUBLISH_VOCABULARY = frozenset({
    "pack",
    "publish",
    "publish_self_mint",
    "withhold_self_mint_publish",
    "adopt_delegated_mint",
    "_publish_async",
    "PendingSelfMint",
    "mint_artifact",
})


def _intake_branch() -> ast.If:
    """The ``if recipe != RECIPE_AOT:`` block inside the arming policy."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fc._arming_policy)))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "recipe"
            and any(isinstance(op, ast.NotEq) for op in test.ops)
        ):
            return node
    raise AssertionError(
        "the arming policy no longer branches on `recipe != RECIPE_AOT` — the "
        "one place the JIT intake posture is separated from the AOT mint")


def test_the_intake_branch_cannot_reach_a_seal_or_a_publish() -> None:
    """No path from RECIPE_DYNAMO to an artifact (pgw#1010's whole point)."""
    branch = _intake_branch()
    called: List[str] = []
    for node in ast.walk(branch):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else "")
        if name in _SEAL_PUBLISH_VOCABULARY:
            called.append(name)
    assert not called, (
        "the JIT intake branch calls seal/publish surface "
        f"{sorted(set(called))} — a dynamo miss must produce NO artifact "
        "(pgw#1010). Delete the call; do not gate it.")


def test_the_intake_branch_returns_before_a_cell_key_exists() -> None:
    """Intake has no identity. A key computed for it is a key the hub would
    then be asked for, and a demand row nothing can ever satisfy."""
    branch = _intake_branch()
    for node in ast.walk(branch):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "compute":
                raise AssertionError(
                    "the intake branch computes a cell key — intake names no "
                    "artifact, so it must not claim an identity")
    assert any(isinstance(node, ast.Return) for node in ast.walk(branch)), (
        "the intake branch must RETURN — falling through into the AOT mint is "
        "how a dynamo miss used to acquire a pending")


def test_the_dynamo_seal_surface_is_deleted_not_gated() -> None:
    """§4.24 hardcut: the functions that packed and republished a JIT cell are
    gone. A flag that re-enables them is exactly what this issue refuses."""
    for name in ("finish_fleet_mint", "begin_fleet_mint"):
        assert not hasattr(cc, name), (
            f"compile_cache.{name} is back — the JIT capture/pack path was "
            "deleted, not disabled")
    assert hasattr(cc, "arm_jit_intake"), (
        "the intake ARM is what survived the cut — serving is untouched")
    for name in ("finalize_self_mint", "republish_after_shape_warm"):
        assert not hasattr(fc, name), (
            f"fleet_cells.{name} is back — the in-process capture packed a "
            "dynamo cell and the shape-warm republish shipped one")


def test_every_pending_is_a_delegated_aot_mint() -> None:
    """A pending IS a mint obligation. Since pgw#1010 only AOT opens one, so
    the pending carries no recipe axis to disagree about and no capture dir."""
    fields = {f.name for f in fc.dataclasses.fields(fc.PendingSelfMint)}
    assert "recipe" not in fields
    assert "capture_dir" not in fields
    pending = fc.PendingSelfMint(
        family="fam", cell_key="ck5-x", ref="r#ck5-x", cfg=None,
        target=Path("/tmp/cell.tar.gz"), mint_root=Path("/tmp"),
        publisher=None)
    assert pending.delegated is True


def test_the_retired_capture_tokens_are_gone_from_the_vocabulary() -> None:
    """The three eager causes that only ever described the process-global
    inductor cache-dir move die with it — a token nobody can reach is a cause
    a reader hunts for and never finds (pgw#813's rule, applied to a deletion).
    """
    values = {member.value for member in cell_adopt.EagerPhase}
    assert values.isdisjoint({
        "delivered_cell_seeded", "capture_conflict", "multi_group_in_process",
        "capture_arm_failed",
    })
    assert cell_adopt.EagerPhase.JIT_ARM_FAILED.value == "jit_arm_failed"


# ---------------------------------------------------------------------------
# behavioural guard — the real arming policy, no mocks of the code under test
# ---------------------------------------------------------------------------


class _Denoiser:
    def forward(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


class _Pipe:
    _cozy_low_vram_mode = "off"

    def __init__(self) -> None:
        self.transformer = _Denoiser()


class _Cfg:
    family = "fam"
    shapes = ((64, 64),)
    targets = ("transformer",)
    regional = False
    guidance_scales = ()
    lora_bucket = 0


@pytest.fixture(autouse=True)
def _clear_pending() -> Any:
    with fc._PENDING_LOCK:
        fc._PENDING.clear()
    yield
    with fc._PENDING_LOCK:
        fc._PENDING.clear()


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str, str]]:
    seen: List[Tuple[str, str, str]] = []

    def capture(kind: str, detail: str, phase: str = "",
                duration_ms: int = 0) -> None:
        seen.append((kind, phase, detail))

    monkeypatch.setattr(activity_mod, "emit_event", capture)
    monkeypatch.setattr(fc.activity_mod, "emit_event", capture)
    return seen


def _publisher(calls: List[Any]) -> fc.CellPublisher:
    class _Pub(fc.CellPublisher):
        def publish(self, family: str, artifact: Path, meta: Dict[str, Any],
                    mint_duration_ms: int = 0) -> str:  # pragma: no cover
            calls.append((family, artifact))
            raise AssertionError(
                "a JIT intake arm published a cell (pgw#1010)")

    return _Pub(base_url="http://hub", worker_jwt=lambda: "jwt",
                image_digest="sha256:img")


def _miss_with_no_declaration(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """A compile-cell MISS on a family that declares no export: the exact
    shape that used to mint, pack and publish a dynamo cell."""
    armed: List[str] = []
    monkeypatch.setattr(
        provision, "enable_compiled",
        lambda *a, **k: provision.AdoptOutcome.miss("no_cell"))
    monkeypatch.setattr(fc, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)
    monkeypatch.setattr(fc, "export_declaration", lambda family: None)
    monkeypatch.setattr(
        cc, "arm_jit_intake",
        lambda pipe, cfg: armed.append(str(getattr(cfg, "family", ""))))
    return armed


def test_a_dynamo_miss_serves_intake_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, events: List[Tuple[str, str, str]],
    tmp_path: Path,
) -> None:
    """The whole cut, end to end through the production arming policy."""
    armed = _miss_with_no_declaration(monkeypatch)
    published: List[Any] = []

    outcome = fc.enable_compiled(
        _Pipe(), _Cfg(), tmp_path, None, publisher=_publisher(published))

    # It SERVES: the pod compiles its own graphs and runs them.
    assert outcome.armed is True
    assert armed == ["fam"]
    # It owes NOTHING: no pending, no obligation, no publish, no key.
    assert outcome.self_mint is None
    assert not published
    with fc._PENDING_LOCK:
        assert not fc._PENDING
    # And it SAYS so, once, with the cause the hub counts on.
    skipped = [
        (phase, detail) for kind, phase, detail in events
        if kind == "self_mint_skipped"]
    assert [phase for phase, _ in skipped] == ["no_export_declaration"]
    assert "MINTS NOTHING" in skipped[0][1]
    assert not [kind for kind, _, _ in events if kind == "self_mint_started"]


def test_an_intake_armed_pipeline_reports_jit_not_eager() -> None:
    """The measurement half. An intake pod serves COMPILED and names no
    artifact, so classifying it by the (absent) cell ref would report every
    JIT request as eager and delete the JIT arm of the AOT-vs-JIT comparison.

    ...and a pipeline whose guard permanently degraded it back to eager must
    say EAGER again, wrapper or no wrapper — reporting a degraded lane as
    compiled is the same lie as reporting an unproven cell as adopted.
    """
    pipe = _Pipe()
    assert serving_mode.classify_mode("", pipe) == serving_mode.MODE_EAGER
    signal: Dict[str, Any] = {}
    setattr(pipe, cc._MARKER_ATTR, {"originals": (), "failure_signal": signal})
    assert serving_mode.classify_mode("", pipe) == serving_mode.MODE_JIT_CELL
    signal["degraded"] = True
    assert serving_mode.classify_mode("", pipe) == serving_mode.MODE_EAGER
