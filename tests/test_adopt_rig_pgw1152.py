"""pgw#1152: the adopt-path rig re-finds a bug we already fixed, and the fence
makes the fixture that hid it unwriteable.

THE BUG CLASS, stated once. Four of the six reuse-circle gates — pgw#1108,
pgw#1122, pgw#1141, pgw#1141b — were ONE defect: the boot-adopt path
structurally differs from the self-mint path (it arms BEFORE setup, its compute
child holds no JWT, and it is fed by ``arm_ordered`` rather than by the
self-produced routes), and every gate was written and validated against
self-mint. The tests could not see it because their fixtures SIMULATED an
adoption using self-mint machinery: thirteen rows called
``aot_serve.note_aot_key(key)`` by hand — production's single missing line.

Two things close that class, and both are asserted here:

1. :mod:`harness.adopt_rig` — the real vehicle, promoted out of
   ``test_adopted_arm_lane_pgw1141b.py``. This file proves it by making it
   RE-FIND pgw#1141b: delete the landed fix and the rig reproduces POD PROOF
   #4's four-row chain verbatim. A rig that cannot re-find a bug we already
   fixed proves nothing.
2. ``scripts/lint_arm_state_feeders.py`` — the fence. Every production feeder of
   arming/serving process-global state is classified by enclosing function
   (pgw#1122's ``lint_credential_identity.py`` shape), unclassified is red,
   stale is red, and a TEST that hand-feeds one is red with no label to write
   down. The rows below drive the lint over synthetic trees, so proving it goes
   red never requires writing the bug into this repo.

Run: uv run pytest tests/test_adopt_rig_pgw1152.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker import activity, aot_serve, cell_adopt
from gen_worker import executor as ex_mod

from harness import exported_cell as cell868
from harness.adopt_rig import AdoptRig, production_cfgs, reintroduce
# The REAL receipt gate: a real RSA key, a real JWKS/receipt HTTP hub.
from harness.receipt_hub import HubStub, hub, pub_map, rsa_key  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
LINT = REPO / "scripts" / "lint_arm_state_feeders.py"


# ===========================================================================
# 1. THE RIG RE-FINDS pgw#1141b
# ===========================================================================


def test_the_rig_serves_a_boot_adopted_cell_COMPILED(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """The control. With the tree as it stands, a boot-adopted cell that the
    warmup never dispatched through keeps its arm, gets its target, and serves
    compiled — §4.31 + §4.32 + pgw#1141b, end to end, off-pod."""
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    assert boot.adopted(), "the boot did not adopt at all"
    assert boot.is_armed() is True
    assert boot.holds_cell() is True
    target = boot.compile_target()
    assert target is not None, "armed and still no installed compile target"
    assert "generate" in target.function_names
    assert target.active_compile_ref == boot.cell.cell_ref
    assert boot.record.eager_posture == ""
    assert boot.serves_compiled()
    assert not boot.phases(activity.KIND_SERVE_DEGRADE)


def test_the_rig_RE_FINDS_pgw1141b_when_its_fix_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """THE PROOF THE RIG IS WORTH ANYTHING.

    ``reintroduce(monkeypatch, "pgw1141b")`` deletes both halves of ``f3ab710e``
    — the registration inside ``arm_entry``, and the object-first lane
    reader — putting the tree back where the fix found it. The rig must then
    reproduce POD PROOF #4 (RTX A4500, sm_86, gen-worker 0.111.0) verbatim::

        seq 13  serve_eager_posture  target_applicability_incomplete
                  functions=() … owned_slots=['pipeline']
        seq 14  serve_eager_posture  armed_target_unresolved
        seq 15  serve_degrade        armed_target_unresolved

    If this row ever goes green, the rig has stopped entering through the
    boot-adopt path and is measuring the self-mint one again — which is the
    whole reason six gates shipped broken.
    """
    reintroduce(monkeypatch, "pgw1141b")
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    assert boot.adopted(), "the boot did not adopt at all"
    postures = boot.phases("serve_eager_posture")
    assert "target_applicability_incomplete" in postures, (
        "the rig did NOT re-find pgw#1141b — with the fix deleted, a "
        "boot-adopted cell must be scored on the dynamo ledger and disarmed")
    assert cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value in postures
    assert boot.phases(activity.KIND_SERVE_DEGRADE)
    assert boot.compile_target() is None
    assert not boot.serves_compiled()

    # …and the locus itself, named: the process armed these exact bytes and
    # still did not recognize their ref as an exported cell.
    assert not aot_serve.is_aot_ref(boot.cell.cell_ref)


def test_the_registration_is_what_the_re_find_turns_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """One variable moved. The fix's first half alone (the wrap registers the
    key) is enough to keep the arm, which is what makes the deletion above a
    bisection rather than a vibe."""
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()
    assert aot_serve.is_aot_ref(boot.cell.cell_ref)
    assert aot_serve.is_aot_ref(boot.cell.cell_ref, boot.cell.family)

    # The structural half: with the registry deliberately emptied AFTER the
    # wrap, the readers must still put this object on the exported lane.
    monkeypatch.setattr(aot_serve, "_KNOWN_AOT_KEYS", set())
    assert not aot_serve.is_aot_ref(boot.cell.cell_ref)
    assert aot_serve.holds_exported_cell(boot.pipeline) is True
    assert ex_mod._exported_arm(boot.pipeline, boot.cell.cell_ref) is True


# ===========================================================================
# 1b. THE THIRD VARIANT — the rig hands the arm the TYPE production hands it
# ===========================================================================


def test_the_arm_receives_the_type_PRODUCTION_builds_not_a_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """pgw#1150's second pass, folded in. Raw ``Compile`` never travels past
    the registry, so a suite that only ever passes one is measuring a shape no
    fleet path constructs — which is why deleting ``registry.py``'s two
    ``numerics_floor=`` lines left the old gate suite green.

    The rig does not assert this by construction: it OBSERVES what production
    handed ``provision.arm_aot`` on a real boot.
    """
    from gen_worker.api.decorators import Compile
    from gen_worker.registry import CompileCell

    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    assert isinstance(boot.armed_cfg, CompileCell), type(boot.armed_cfg)
    assert not isinstance(boot.armed_cfg, Compile)
    # …and the declared band really reaches the gate through it (pgw#1150).
    assert boot.armed_cfg.numerics_floor == cell868.FLOOR
    assert boot.armed_cfg.numerics_warn == cell868.WARN


def test_both_production_call_sites_build_the_SAME_cell_object() -> None:
    """The one-constructor claim, asserted on the two real call sites rather
    than on the constructor's own docstring. ``EndpointSpec.compile_cell()`` and
    ``cli.run``'s §4.28 desktop arm differ only in the four spec-scoped
    enrichments a raw declaration cannot know; every other field is a straight
    carry, and a field that reaches one and not the other silently judges a
    whole serving path at a default nobody chose."""
    import dataclasses

    cfgs = production_cfgs()
    registry_cfg, cli_cfg = cfgs["registry"], cfgs["cli"]
    enrichments = {"lora_bucket", "text_len", "guidance_scales", "text_lens"}
    for field in dataclasses.fields(type(registry_cfg)):
        if field.name in enrichments:
            continue
        assert getattr(registry_cfg, field.name) == getattr(cli_cfg, field.name), (
            f"{field.name} reaches one Compile->CompileCell map and not the "
            "other — that is pgw#1150's cause, not its instance")


@pytest.mark.parametrize("origin", ["registry", "cli"])
def test_the_declared_band_survives_EVERY_production_route(origin: str) -> None:
    """Parametrised over the cfg objects the fleet actually hands a gate, not
    over the convenient one. This is the pattern that would have caught
    pgw#1150 before it shipped."""
    cfg = production_cfgs()[origin]
    assert cfg.numerics_floor == cell868.FLOOR
    assert cfg.numerics_warn == cell868.WARN


# ===========================================================================
# 2. THE RIG FORCES OUTCOMES BY BREAKING REAL INPUTS
# ===========================================================================


def test_a_cell_whose_ENTRY_RAISES_revokes_itself_and_installs_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """The de-arm still has teeth, forced the honest way: the packaged entry
    really raises when the arm dispatches through it, so the artifact revokes
    ITSELF. Nothing here sets ``failed`` by hand.

    A fix that made a cell undisarmable would be worse than the bug — an object
    whose every call runs eager must never advertise ``serving_mode=aot_cell``.
    """
    boot = AdoptRig(
        tmp_path, monkeypatch, hub,
        package_raises="dlopen: undefined symbol", warm_dispatches=1,
    ).boot()

    assert boot.is_armed() is False
    assert boot.holds_cell() is True, (
        "a revoked cell stopped being recognizable as an exported one, which "
        "would route it back onto the dynamo lane's ledger")
    assert boot.compile_target() is None
    assert boot.record.eager_posture


def test_a_hub_that_serves_NO_RECEIPT_refuses_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """Forcing by REMOVING an input, not by supplying one: the hub simply has
    no receipt for these bytes, so the real gate refuses on a 404 and the pod
    serves eager rather than the cell it resolved."""
    boot = AdoptRig(tmp_path, monkeypatch, hub, serve_receipt=False).boot()

    assert boot.adopted()
    assert boot.is_armed() is False
    assert boot.holds_cell() is False, (
        "an unreceipted cell was wrapped onto the pipeline anyway")
    # pgw#1122: an ordered arm this pod ordered ITSELF degrades to eager rather
    # than failing the function, so the boot falls through to the ordinary
    # policy — a target is still registered (active-less, advertising the key
    # for peer adoption), and it must name NO artifact.
    target = boot.compile_target()
    assert target is not None
    assert target.active_compile_ref == "", (
        "a refused cell is still advertised as the served artifact")
    assert not boot.serves_compiled()


def test_a_WARM_DISPATCH_moves_the_artifacts_own_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """The other arm of the proof pass, and it is a REAL dispatch: the handler
    calls the wrapped forward at a declared shape row, so the packaged entry's
    own invocation count moves. pgw#1141's rig moved a counter by hand."""
    boot = AdoptRig(tmp_path, monkeypatch, hub, warm_dispatches=1).boot()

    assert boot.is_armed() is True
    assert aot_serve.execution_count(boot.pipeline) > 0, (
        "the handler dispatched through the cell and its counter did not move")
    served = [p.invocations for p in boot.packages.values()]
    assert sum(served) > 0, served
    assert boot.serves_compiled()


# ===========================================================================
# 3. THE FENCE — a hand-feed has no label to write down
# ===========================================================================


def _run_lint(src: Path, tests: Path, allowlist: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(LINT), "--src", str(src), "--tests", str(tests),
         "--allowlist", str(allowlist)],
        capture_output=True, text=True)


def _tree(root: Path, files: Dict[str, str]) -> None:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


SEAM = '''
def arm_entry(pipeline, *a, **k):
    meta = {}
    note_aot_key(str(meta.get("cell_key") or ""))
    return meta
'''


def test_the_fence_passes_when_the_only_feeder_is_the_seam(tmp_path: Path) -> None:
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {"aot_serve.py": SEAM})
    tests.mkdir()
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# nothing to classify\n")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 0, got.stderr


def test_the_fence_goes_RED_on_a_production_feeder_off_the_seam(
    tmp_path: Path,
) -> None:
    """pgw#1141b's actual shape: a second arm route feeds the registry by
    convention. There is no classification that makes this legal."""
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {
        "aot_serve.py": SEAM,
        "fleet_cells.py": (
            "from . import aot_serve\n\n\n"
            "def arm_ordered(pipe, order):\n"
            "    aot_serve.note_aot_key(order.cell_key)\n"),
    })
    tests.mkdir()
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 1
    assert "UNCLASSIFIED" in got.stderr
    assert "arm_ordered::aot_serve.note_aot_key" in got.stderr


def test_the_fence_goes_RED_on_a_HAND_REGISTRATION_IN_A_TEST(
    tmp_path: Path,
) -> None:
    """THE ROW THIS ISSUE EXISTS FOR. ``_fake_adopt_arm`` called
    ``note_aot_key`` by hand and thirteen green rows entered one gate east of
    the bug. A test may not feed a production registry — it goes through
    ``harness.adopt_rig``. There is deliberately no label for this."""
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {"aot_serve.py": SEAM})
    _tree(tests, {"test_adopt.py": (
        "from gen_worker import aot_serve\n\n\n"
        "def _fake_adopt_arm(key):\n"
        "    aot_serve.note_aot_key(key)\n")})
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 1
    assert "harness/adopt_rig.py" in got.stderr
    assert "_fake_adopt_arm" in got.stderr


def test_the_fence_goes_RED_on_a_STUBBED_LANE_ACCESSOR_IN_A_TEST(
    tmp_path: Path,
) -> None:
    """The second fixture sin, same class: pgw#1141's suite stubbed an accessor
    so the gate under test answered from the fixture rather than the object."""
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {"aot_serve.py": SEAM})
    _tree(tests, {"test_lanes.py": (
        "from gen_worker import aot_serve\n\n\n"
        "def test_x(monkeypatch):\n"
        '    monkeypatch.setattr(aot_serve, "is_aot_ref", lambda r: True)\n')})
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 1
    assert "STUBBED" in got.stderr


def test_the_fence_goes_RED_when_the_SEAM_STOPS_FEEDING(tmp_path: Path) -> None:
    """The registration lives at the wrap because that is the one function every
    arm route passes. Delete it and a new route is one convention away from
    pgw#1141b again — so the seam's own call is an asserted fact, not a habit.
    """
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {"aot_serve.py": "def arm_entry(pipeline, *a, **k):\n    return {}\n"})
    tests.mkdir()
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 1
    assert "SEAM" in got.stderr


def test_the_fence_goes_RED_on_a_STALE_ALLOWLIST_ROW(tmp_path: Path) -> None:
    """A row matching nothing is a boundary that lies (pgw#1122's rule)."""
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {"aot_serve.py": SEAM})
    tests.mkdir()
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text(
        "src/gone.py::vanished::aot_serve.note_aot_key  VERDICT  gone\n")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 1
    assert "stale allowlist row" in got.stderr


def test_there_is_no_CONVENTION_classification(tmp_path: Path) -> None:
    """The pgw#1122 rule, transplanted: the third instance of this bug has no
    label to write down. ``CONVENTION`` names exactly what went wrong twice, so
    it may not be spellable."""
    src, tests = tmp_path / "src", tmp_path / "tests"
    _tree(src, {
        "aot_serve.py": SEAM,
        "fleet_cells.py": (
            "from . import aot_serve\n\n\n"
            "def arm_ordered(pipe, order):\n"
            "    aot_serve.note_aot_key(order.cell_key)\n"),
    })
    tests.mkdir()
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text(
        "src/fleet_cells.py::arm_ordered::aot_serve.note_aot_key  "
        "CONVENTION  whoever reads a key off an envelope registers it\n")

    got = _run_lint(src, tests, allowlist)
    assert got.returncode == 1
    assert "unknown classification" in got.stderr


# ===========================================================================
# 4. THE FENCE RUNS ON THIS REPO
# ===========================================================================


def test_the_fence_is_green_on_the_tree() -> None:
    """The allowlist is a live inventory, not a document: every production
    feeder of arming/serving process state is classified right now."""
    got = subprocess.run(
        [sys.executable, str(LINT)], capture_output=True, text=True, cwd=REPO)
    assert got.returncode == 0, got.stdout + got.stderr


def test_no_test_module_hand_feeds_the_arm_registries() -> None:
    """Stated separately so a failure names the right thing: if this goes red,
    somebody re-simulated an adoption instead of driving ``harness.adopt_rig``.
    """
    got = subprocess.run(
        [sys.executable, str(LINT)], capture_output=True, text=True, cwd=REPO)
    offenders: List[str] = [
        line for line in (got.stderr or "").splitlines()
        if line.startswith("tests/")]
    assert not offenders, "\n".join(offenders)


def test_the_rig_itself_registers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rig is the thing every other test is sent to, so it is the one place
    a hand-feed would be invisible. It touches ``_KNOWN_AOT_KEYS`` exactly once,
    to EMPTY it — removing an input is legal, supplying one is not."""
    import ast
    import inspect

    from harness import adopt_rig

    source = inspect.getsource(adopt_rig)
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", None) or getattr(node.func, "id", None))
        == "note_aot_key"
    ]
    assert not calls, (
        "the rig hand-registers a cell key — the exact fixture sin it exists "
        f"to replace (line {[c.lineno for c in calls]})")
    # …and it touches the registry exactly once, to EMPTY it. Removing an
    # input is legal; supplying one is not.
    assert source.count('setattr(aot_serve, "_KNOWN_AOT_KEYS", set())') == 1
