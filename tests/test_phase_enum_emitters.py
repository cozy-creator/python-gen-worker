"""pgw#1480: a phase-vocabulary member nothing references cannot fire.

`phase` is a typed wire column, so a phase enum is an INSTRUMENT — an operator
counts by it, and se#780 wrote its headline pass condition against one:
`boot_ended_uncompiled` must be ABSENT. It was absent unconditionally.

⚠️ **The fence's own arms are what is tested here, not the census contents.**
A gate that cannot go red is the same defect one level up, and this session has
already shipped one instrument that could not fail (tcg#58's kwarg-free
fixture). So each of the three properties is driven to RED against a real tree.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FENCE = REPO / "scripts" / "lint_phase_enum_emitters.py"
CENSUS = REPO / "scripts" / "phase_enum_census.txt"
ENUM = REPO / "src" / "gen_worker" / "compiled_graph_adopt.py"
POSTURE = REPO / "src" / "gen_worker" / "serve_posture.py"

#: The exemplar this file drives the fence with: a member that is STILL
#: censused as unwired. It was `BOOT_ENDED_UNCOMPILED` until pgw#1480's fix
#: wired that one — which is the census doing its job, and the reason the
#: exemplar is named ONCE here instead of being spelled into four arms. When
#: this one gets wired too, move the name; do not delete the arm.
DEAD_EXEMPLAR = "ARM_PENDING"
DEAD_EXEMPLAR_VALUE = "arm_pending"


def _run() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FENCE)], capture_output=True, text=True, cwd=REPO
    )


@pytest.fixture()
def restore():
    """Every arm mutates the real tree, so every arm puts it back."""

    saved = {p: p.read_bytes() for p in (CENSUS, ENUM, POSTURE)}
    yield
    for path, data in saved.items():
        path.write_bytes(data)


def test_the_fence_is_green_on_the_tree_as_it_stands() -> None:
    result = _run()
    assert result.returncode == 0, result.stderr
    assert "phase-vocabulary fence: clean" in result.stdout


def test_the_census_is_not_empty_and_still_names_dead_members() -> None:
    """If the census were empty the green above would be vacuous."""

    rows = [r for r in CENSUS.read_text().splitlines()
            if r.strip() and not r.startswith("#")]
    assert len(rows) >= 17
    assert f"EagerPhase.{DEAD_EXEMPLAR}" in rows


def test_the_member_that_started_this_is_WIRED_and_OFF_the_census() -> None:
    """pgw#1480's fix, asserted from the fence's side.

    `boot_ended_uncompiled` was the member se#780 wrote its headline pass
    condition against, and it was referenced by nothing. It now has an emitter
    (`ServeAdoption._say_boot_end`), so its census row MUST be gone — property
    2 of the fence is exactly the check that would have caught a fix that wired
    the member and left the row behind, pretending it was still dead."""

    rows = CENSUS.read_text()
    assert "EagerPhase.BOOT_ENDED_UNCOMPILED" not in rows
    emitter = (REPO / "src" / "gen_worker" / "serving" / "serve_adoption.py")
    assert "EagerPhase.BOOT_ENDED_UNCOMPILED" in emitter.read_text()


def test_RED_when_a_new_member_arrives_unwired(restore: None) -> None:
    """Property 1: the class cannot GROW."""

    ENUM.write_text(
        ENUM.read_text().replace(
            '    BOOT_ENDED_UNCOMPILED = "boot_ended_uncompiled"',
            '    BOOT_ENDED_UNCOMPILED = "boot_ended_uncompiled"\n'
            '    BRAND_NEW_FOLKLORE = "brand_new_folklore"',
        )
    )
    result = _run()
    assert result.returncode == 1
    assert "BRAND_NEW_FOLKLORE" in result.stderr


def test_RED_when_a_censused_member_gains_a_reference(restore: None) -> None:
    """Property 2: the census must SHRINK, so it can never become a graveyard.

    This is the arm that makes the census a burn-down list rather than a
    permanent exemption — without it, wiring a member up would be invisible and
    the row would sit there forever.
    """

    POSTURE.write_text(
        POSTURE.read_text().replace(
            "REASON: str = EagerPhase.OPERATOR_EAGER_ONLY.value",
            "REASON: str = EagerPhase.OPERATOR_EAGER_ONLY.value\n"
            f"_WIRED_NOW = EagerPhase.{DEAD_EXEMPLAR}.value",
        )
    )
    result = _run()
    assert result.returncode == 1
    assert DEAD_EXEMPLAR in result.stderr
    assert "graveyard" in result.stderr


def test_RED_on_a_stale_census_row(restore: None) -> None:
    """Property 3: no row may outlive its member."""

    CENSUS.write_text(CENSUS.read_text() + "EagerPhase.NO_SUCH_MEMBER\n")
    result = _run()
    assert result.returncode == 1
    assert "no longer exists" in result.stderr


def test_prose_naming_a_member_does_not_count_as_a_reference(
    restore: None, tmp_path: Path
) -> None:
    """A comment EXPLAINING a dead member must not make it read as alive.

    The inverse of the docstring-stripping rule the raw-AOTI fence needed: here
    counting prose would mark a dead instrument live, which is the failure that
    matters.
    """

    scratch = REPO / "src" / "gen_worker" / "_pgw1480_prose_probe.py"
    scratch.write_text(
        '"""probe."""\n'
        f"# EagerPhase.{DEAD_EXEMPLAR} and {DEAD_EXEMPLAR_VALUE!r}\n"
    )
    try:
        result = _run()
        assert result.returncode == 0, (
            "a comment naming a censused member made it read as referenced; "
            "the census would then demand its own removal")
    finally:
        scratch.unlink(missing_ok=True)
        shutil.rmtree(REPO / "src" / "gen_worker" / "__pycache__", ignore_errors=True)
