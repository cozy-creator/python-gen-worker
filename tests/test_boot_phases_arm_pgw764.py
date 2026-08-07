"""pgw#764 end-to-end arm instrumentation — STAGED, NOT YET COMMITTABLE.

These two cases drive the real ``aot_serve.enable`` path and assert that a
refused arm becomes ``outcome=refused`` carrying the classified
``AdoptError.reason``. They depend on the boot-phase hooks in
``aot_serve.py`` / ``aot_cells.py``, and a sibling lane (the sm-vs-sku axis
work) holds uncommitted WIP in both of those files — committing them would
sweep that lane's work, which the shared-worktree policy forbids.

So the hooks and these tests stay in the worktree until that WIP lands. The
refusal CONTRACT itself is covered in the committed suite
(``test_boot_phases_pgw764.py::test_a_typed_refusal_is_recorded_as_refused``);
what is deferred is only the end-to-end wiring assertion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from gen_worker import aot_serve, boot_phases
from gen_worker.compile_cache import AdoptError
from gen_worker.pb import worker_scheduler_pb2 as pb

# pgw#797 adjudication (test-health audit): SKIP, do not FAIL, while the src
# half is absent. This file is the test half of pgw#764's arm instrumentation;
# the `cell_arm` span it asserts lives in `aot_serve.py`, which HEAD does not
# carry (`git show HEAD:src/gen_worker/aot_serve.py | grep -c boot_phases` == 0)
# — it is a sibling lane's uncommitted WIP, exactly as the docstring above says.
# Both cases PASS against the worktree that has that WIP and fail against HEAD,
# so this is a staged pair, not a broken test: deleting it would destroy live
# work and committing it would land a permanently-red test.
# REMOVE THIS GUARD when the aot_serve.py hook lands; the probe then goes True
# on its own and the file needs no other change.
if getattr(aot_serve, "boot_mod", None) is None:
    pytest.skip(
        "pgw#764 STAGED: the cell_arm boot-phase hook in aot_serve.py has not "
        "landed yet (sibling WIP). The refusal CONTRACT is covered by "
        "test_boot_phases_pgw764.py::test_a_typed_refusal_is_recorded_as_refused; "
        "what is deferred here is only the end-to-end wiring assertion.",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _reset() -> Any:
    boot_phases.reset_for_tests()
    yield
    boot_phases.reset_for_tests()


def test_a_refused_arm_rides_a_boot_row_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise AdoptError("key_mismatch", "cell was minted for sm_90, host is sm_89")

    monkeypatch.setattr(aot_serve, "load_and_wrap", refuse)
    artifact = tmp_path / "ck1_deadbeef.tar.gz"
    artifact.write_bytes(b"not-a-real-artifact")

    # Behaviour is unchanged: a refused arm stays eager and returns False.
    assert aot_serve.enable(object(), object(), artifact=artifact).armed is False

    arms = [r for r in boot_phases.recorded_rows()
            if r.terminal and r.phase == boot_phases.PHASE_CELL_ARM]
    assert len(arms) == 1
    row = arms[0]
    # REFUSED, not FAILED: the worker declined this cell and serves eager.
    assert row.outcome == boot_phases.OUTCOME_REFUSED
    assert row.reason == "key_mismatch"
    assert "sm_90" in row.detail
    assert row.artifact_kind == aot_serve.ARTIFACT_KIND
    assert row.artifact_key == "ck1_deadbeef"


def test_an_armed_cell_rides_an_ok_boot_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    meta = {"family": "sdxl", "cell_key": "ck1_feed", "entries": {"unet": {}},
            "sku": "rtx-4090", "torch": "2.13", "precision": "w8a8"}
    monkeypatch.setattr(aot_serve, "load_and_wrap", lambda *a, **k: meta)
    artifact = tmp_path / "ck1_feed.tar.gz"
    artifact.write_bytes(b"x")

    assert aot_serve.enable(object(), object(), artifact=artifact).armed is True
    arms = [r for r in boot_phases.recorded_rows()
            if r.terminal and r.phase == boot_phases.PHASE_CELL_ARM]
    assert len(arms) == 1
    assert arms[0].outcome == boot_phases.OUTCOME_OK
    assert "key=ck1_feed" in arms[0].detail


