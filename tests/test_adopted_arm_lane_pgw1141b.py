"""A BOOT-ADOPTED cell is on the EXPORTED lane, and the setup pass has to know
that from the OBJECT — not from a process-global set of cell keys somebody
remembered to announce.

The contract under test, in two parts:

1. ``aot_serve.arm_entry`` registers the key AT THE WRAP — the single seam
   every arm route passes, and the moment the fact becomes true;
2. the executor's three lane readers ask the OBJECT
   (``aot_serve.holds_exported_cell`` via ``executor._exported_arm``), so the
   disarm authority cannot be exercised over an object carrying a live cell
   even if no registry ever learned its key.

Without both, a boot-adopted artifact lands in ``proof_before`` — the DYNAMO
lane's cache-hit ledger, which an AOTI artifact can never move — scores
calls=0, is folded into ``unproven`` and unwrapped, and the install resolves
``functions=()`` -> ``target_applicability_incomplete`` ->
``armed_target_unresolved`` -> eager serving.

WHAT RUNS FOR REAL HERE. The whole chain from the ordered arm to the install:
``Executor.ensure_setup`` -> ``_injection_kwargs`` -> ``_enable_compiled`` with
a real ``_ArmOrder(adopt=…)`` -> ``fleet_cells.arm_ordered`` -> the real
receipt gate against a real RSA-signed receipt from a real HTTP hub ->
``provision.arm_aot`` -> ``aot_serve.arm_entry`` on a real packed artifact
-> the real boot warmup -> the real proof pass -> the real
``_install_compile_targets`` and ``_assert_armed_targets_installed``.

THREE SEAMS, all WEST of the locus and all named:

* ``Executor._boot_adopt`` returns a constructed HIT; its derive+resolve half
  is driven end to end by ``test_boot_adopt_observability_pgw1116.py``.
* ``provision.load_slot`` returns the probe pipeline instead of reading
  weights off disk (the class-annotated slot shape micro-diffusion uses —
  ``Slot(MicroPipeline, selected_by="model")`` — which is what routes the arm
  order to ``_enable_compiled`` at all).
* ``aot_serve._load_package``: an AOTI ``.so`` needs a GPU; the rig owns this
  substitution and it is the only piece deferred to a pod.

Nothing about applicability, the lane split, the proof pass or the target
install is stubbed — those are the code under test.

Run: uv run pytest tests/test_adopted_arm_lane_pgw1141b.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker import activity, aot_serve, cell_adopt
from gen_worker import executor as ex_mod

# The vehicle this file built is now `tests/harness/adopt_rig.py`,
# the DEFAULT rig for anything arming-adjacent. It is the same chain, verbatim
# — real `ensure_setup` -> real `_enable_compiled` -> a real `_ArmOrder(adopt=…)`
# -> real `arm_ordered` -> the real receipt gate against a real RSA-signed
# receipt from a real HTTP hub -> `provision.arm_aot` -> `arm_entry` on a
# real packed cell -> real warmup, proof pass and target install.
from harness.adopt_rig import FAMILY, AdoptRig  # noqa: F401
from harness.receipt_hub import HubStub, hub, pub_map, rsa_key  # noqa: F401


# ===========================================================================
# 1. THE HEADLINE RED — the pod's four-event chain, reproduced off-pod
# ===========================================================================


def test_a_boot_adopted_cell_is_NOT_scored_on_the_dynamo_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """THE RED. On the tree before ``f3ab710e`` this reproduces POD PROOF #4's
    chain exactly: ``target_applicability_incomplete functions=()`` then
    ``armed_target_unresolved`` then ``serve_degrade``, and the pod serves eager
    for life having thrown away the cell it just materialized. That deletion is
    now a first-class row of its own —
    ``test_adopt_rig_pgw1152::test_the_rig_RE_FINDS_pgw1141b_when_its_fix_is_deleted``.

    The assertions below are the four rows the pod emitted, in order.
    """
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    assert boot.adopted(), (
        "the boot did not adopt at all — this row is testing the wrong thing")

    # seq 13: the object's applicability. `functions=()` is where the pod died.
    assert "target_applicability_incomplete" not in boot.phases(
        "serve_eager_posture"), (
        "the boot-adopted cell computed EMPTY applicability — the ordered arm "
        "route never taught `is_aot_ref` its key, so the exported cell was "
        "scored on the dynamo lane's cache-hit ledger and disarmed")
    # seq 14 + 15: the orphan report and the degrade.
    assert cell_adopt.EagerPhase.ARMED_TARGET_UNRESOLVED.value not in \
        boot.phases("serve_eager_posture")
    assert not boot.phases(activity.KIND_SERVE_DEGRADE), (
        "a cell that materialized, armed and was never dispatched through "
        "degraded the whole record")

    # ...and the positive statement of the same fact: it SERVES.
    assert boot.is_armed() is True
    target = boot.compile_target()
    assert target is not None, "armed and still no installed compile target"
    assert "generate" in target.function_names
    assert target.active_compile_ref == boot.cell.cell_ref
    assert boot.record.eager_posture == ""
    assert boot.serves_compiled()


def test_the_ordered_arm_teaches_the_recognizer_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """THE LOCUS, stated as one fact. ``note_aot_key`` had two feeders and both
    were self-produced routes; the ordered arm — every hub Plan and every §4.27
    boot-adopt — fed it nothing, so an adopted cell's own ref answered "not an
    AOT cell" on the pod that was serving it.

    pgw#1152 then DELETED both of those feeders: the wrap is now the registry's
    only writer, so this fact has exactly one producer.
    """
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    assert aot_serve.is_aot_ref(boot.cell.cell_ref), (
        "the process armed this exact artifact and still does not recognize "
        "its ref as an exported cell")
    assert aot_serve.is_aot_ref(boot.cell.cell_ref, FAMILY)


def test_the_lane_is_read_off_the_OBJECT_not_a_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """The structural half. Registering the key is a convention a future arm
    route can forget again — twice now a disarm authority has survived one gate
    east of its fix. With the registry deliberately emptied AFTER the wrap, the
    readers must still put this object on the exported lane."""
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    monkeypatch.setattr(aot_serve, "_KNOWN_AOT_KEYS", set())
    assert not aot_serve.is_aot_ref(boot.cell.cell_ref)
    assert aot_serve.holds_exported_cell(boot.pipeline) is True
    assert ex_mod._exported_arm(boot.pipeline, boot.cell.cell_ref) is True


def test_the_posture_row_names_the_adoption_not_the_dynamo_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """seq 12 on the pod carried §4.31's new sentence, which is why the leg
    read as "pgw#1141 works". It did — but the reason under it was the DYNAMO
    lane's ("this boot holds no evidence either way"), because the cell had
    been sorted onto that lane. An adopted cell's row must say what it is."""
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()

    rows = boot.details(activity.KIND_CELL_NUMERICS)
    assert len(rows) == 1, rows
    assert "STAYS ARMED" in rows[0]
    assert "adoption runs no quality gate" in rows[0], (
        "the adopted cell was confessed under the dynamo lane's reason, which "
        "is the sorting error this issue is about")


# ===========================================================================
# 2. THE NEGATIVE — the honest de-arm paths keep their teeth
# ===========================================================================


def test_a_REVOKED_cell_still_de_arms_and_installs_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """§4.31's sticky de-arm, unchanged. The artifact ran and failed before any
    guard was bound, so nothing may advertise ``serving_mode=aot_cell`` for it.
    A fix that made a cell undisarmable would be worse than the bug.

    the revocation is forced by BREAKING a real input — the packaged
    entry raises when the arm dispatches through it — rather than by setting
    ``failed`` on a marker the test wrote itself.
    """
    boot = AdoptRig(
        tmp_path, monkeypatch, hub,
        package_raises="dlopen: undefined symbol", warm_dispatches=1,
    ).boot()

    assert boot.is_armed() is False
    assert boot.compile_target() is None, (
        "a revoked cell kept an installed target, so the wire would say "
        "aot_cell on a pipeline whose every call runs eager")
    assert cell_adopt.EagerPhase.COMPILED_DEGRADED.value in boot.phases(
        "serve_eager_posture")
    assert boot.record.eager_posture


def test_an_operator_eager_only_order_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """§4.32 item 4: the consumer can always opt out of compiled serving. The
    exported-lane recognition must not route around the operator's order."""
    from gen_worker import serve_posture

    serve_posture.apply_command(True, actor="operator", reason="pgw#1141b row")
    try:
        boot = AdoptRig(tmp_path, monkeypatch, hub).boot()
        # Nothing armed and nothing adopted: the order is obeyed at the
        # arming brain, before the lane question is ever asked.
        assert boot.is_armed() is False
        assert boot.holds_cell() is False
        assert boot.record.eager_posture == cell_adopt.EagerPhase.OPERATOR_EAGER_ONLY
        assert boot.executor.serving_tiers()["generate"] == "eager"
        assert not boot.serves_compiled()
    finally:
        serve_posture.reset()


# ===========================================================================
# 3. THE FENCE — every route that wraps an exported cell feeds ONE recognizer
# ===========================================================================


def test_the_registration_lives_at_the_wrap_not_at_the_call_sites() -> None:
    """The fence against a FOURTH reader. ``note_aot_key`` was a convention
    ("whoever reads a cell_key off an aot-inductor envelope registers it") and
    the ordered arm simply did not keep it. Registration now happens inside
    ``arm_entry``, the one function every arm route passes, so a new route
    inherits it instead of having to remember it.

    pgw#1152 made this a repo-wide lint (``scripts/lint_arm_state_feeders.py``)
    rather than one file's assertion; this row stays because it names the fact
    at the place a reader of THIS issue looks for it.
    """
    import inspect

    body = inspect.getsource(aot_serve.arm_entry)
    assert "note_aot_key(" in body, (
        "the wrap no longer registers the key it just armed; a new arm route "
        "is one convention away from pgw#1141b happening again")


def test_a_wrapped_object_answers_the_lane_question_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub: HubStub,  # noqa: F811
) -> None:
    """``holds_exported_cell`` is deliberately NOT ``is_armed``: the install
    path must tell a REVOKED exported cell (never advertise it) apart from an
    object carrying no cell at all (an ordinary dynamo/eager object)."""
    boot = AdoptRig(tmp_path, monkeypatch, hub).boot()
    pipe = boot.pipeline

    assert aot_serve.holds_exported_cell(pipe) is True
    # REVOKING is de-arming every entry, not setting a flag.
    # `is_armed` asks the REGISTRY what is armed — deliberately, because a
    # boolean can claim more than the pod serves — so a fixture that flipped
    # `state["failed"]` was revoking a cell-level thing that no longer exists.
    # `disarm_entry` is what production calls, and it is sticky for the boot.
    for name in list(aot_serve.armed_entries(pipe)):
        aot_serve.disarm_entry(pipe, name, "revoked by the lane-question row")
    assert aot_serve.is_armed(pipe) is False
    assert aot_serve.holds_exported_cell(pipe) is True, (
        "a revoked cell stopped being recognizable as an exported one, which "
        "would route it back onto the dynamo lane's ledger")

    aot_serve.unwrap(pipe)
    assert aot_serve.holds_exported_cell(pipe) is False
