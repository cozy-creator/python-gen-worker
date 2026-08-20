"""Every diff-local repo guard says WHICH SIDE its finding falls on.

# pgw#1521: `fast gates` is the only required context and bundles ~25 guards, so
# an `--admin` past somebody else's master-red silently bypasses a guard that is
# refusing YOUR OWN file. Four instances in one night, and pgw has no post-merge
# CI to catch the fifth.

These run the REAL guard scripts as subprocesses against a real planted
violation, with the real attribution file `fast gates` writes — the production
path end to end, not a call into `_lint_side` with hand-made inputs. A guard
whose findings stop naming a file fails here rather than in the field.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

#: The violation these tests plant. It names a retired package on purpose:
#: the guard under test is the one that refuses it.
VIOLATION = "import hashrepo\n"  # retired-name: the fixture IS the violation


@pytest.fixture()
def fixture_tree():
    """A violation inside the repo, so the guard reports a repo-relative path.

    Under `scripts/` deliberately: the guards spell findings relative to the
    repo root, and attribution reads that spelling.
    """
    root = SCRIPTS / f"_pgw1521_fixture_{uuid.uuid4().hex[:8]}"
    root.mkdir()
    try:
        yield root
    finally:
        for child in root.rglob("*"):
            child.unlink()
        root.rmdir()


def _emit(path: Path, base: str, files: tuple[str, ...]) -> Path:
    path.write_text(f"# base {base}\n" + "".join(f"{f}\n" for f in files))
    return path


def _run(script: str, target: Path,
         diff_file: Path | None) -> "subprocess.CompletedProcess[str]":
    env = dict(os.environ)
    env.pop("PGW1521_DIFF_FILES", None)
    if diff_file is not None:
        env["PGW1521_DIFF_FILES"] = str(diff_file)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), str(target)],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=300,
    )


def test_a_finding_in_this_diff_is_named_as_yours(fixture_tree, tmp_path):
    (fixture_tree / "mod.py").write_text(VIOLATION)
    rel = (fixture_tree / "mod.py").relative_to(REPO).as_posix()
    diff = _emit(tmp_path / "diff.txt", "deadbeefcafe", (rel,))

    done = _run("lint_retired_package_names.py", fixture_tree, diff)

    assert done.returncode == 1, done.stdout + done.stderr
    out = done.stdout + done.stderr
    assert "[YOUR DIFF]" in out, out
    # The verdict is the line the merge button needs, and it must refuse to
    # read as somebody else's problem.
    assert "AT LEAST ONE OF THESE IS YOURS" in out, out


def test_a_finding_outside_this_diff_is_named_as_the_base(fixture_tree, tmp_path):
    (fixture_tree / "mod.py").write_text(VIOLATION)
    diff = _emit(tmp_path / "diff.txt", "deadbeefcafe",
                 ("src/gen_worker/cli/daemon.py",))

    done = _run("lint_retired_package_names.py", fixture_tree, diff)

    assert done.returncode == 1, done.stdout + done.stderr
    out = done.stdout + done.stderr
    assert "[PRE-EXISTING]" in out, out
    assert "TARGETED" in out, out


def test_an_unresolvable_base_is_unknown_and_never_reads_as_not_yours(
    fixture_tree, tmp_path
):
    """UNKNOWN is the one verdict that must not be mistaken for a pass.

    A gate that cannot attribute must say so — reporting "not in your diff"
    when the diff is unknown is the exact bypass pgw#1521 exists to stop.
    """
    (fixture_tree / "mod.py").write_text(VIOLATION)
    diff = _emit(tmp_path / "diff.txt", "UNKNOWN", ())

    done = _run("lint_retired_package_names.py", fixture_tree, diff)

    assert done.returncode == 1, done.stdout + done.stderr
    out = done.stdout + done.stderr
    assert "[SIDE UNKNOWN]" in out, out
    assert "UNKNOWN is not 'not yours'" in out, out
    assert "PRE-EXISTING" not in out, out


def test_a_resolved_base_with_an_empty_diff_is_not_unknown(fixture_tree, tmp_path):
    """An empty diff is a fact; an unresolved base is not. They differ."""
    (fixture_tree / "mod.py").write_text(VIOLATION)
    diff = _emit(tmp_path / "diff.txt", "deadbeefcafe", ())

    done = _run("lint_retired_package_names.py", fixture_tree, diff)

    out = done.stdout + done.stderr
    assert done.returncode == 1, out
    assert "[PRE-EXISTING]" in out, out


def test_the_attribution_helper_selftest_passes():
    """The helper's own red arms — every one of them runs here too."""
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "_lint_side.py"), "--selftest"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr


#: The `scripts/lint_*.py` that are NOT `fast gates` diff-local guards, and so
#: are not expected to carry the attribution seam. Transcribed, not derived —
#: deriving it from the same `_lint_side` probe the rows below use would make
#: the exemption self-granting, which is the whole defect this literal closes.
#:
#: `lint_skip_census.py` is the pgw#966 census guard. It runs in the `tests`
#: job, not `fast gates`, and its finding is never a file in a diff — it is a
#: skip key, which has no side to fall on.
NOT_DIFF_LOCAL = frozenset({"lint_skip_census.py"})


def test_the_set_of_guards_exempt_from_attribution_is_exactly_the_named_one():
    """pgw#1528. This row replaces a `pytest.skip` and the replacement is the
    point: the parametrized rows below used to SKIP when a script had no
    `_lint_side` — so a guard that silently LOST the seam produced a skip, not
    a failure. That is the exact shape pgw#966's census exists to catch (a row
    that stops measuring and reports nothing), reached from inside the guard
    that was supposed to prevent it. A skip is also not free: it minted an
    unclassified census key that failed the `tests` job for every PR in the
    repo until somebody disposed of it.

    Stated as a set instead, an unconverted NEW guard is red HERE, naming
    itself, and a conversion that regresses is red rather than quiet.
    """
    unconverted = {p.name for p in SCRIPTS.glob("lint_*.py")
                   if "_lint_side" not in p.read_text()}
    assert unconverted == NOT_DIFF_LOCAL, (
        "a `fast gates` guard lost (or never got) the attribution seam: "
        f"{sorted(unconverted - NOT_DIFF_LOCAL)}; or an exempt one was "
        f"converted and this literal is stale: {sorted(NOT_DIFF_LOCAL - unconverted)}"
    )


@pytest.mark.parametrize("script", sorted(
    p.name for p in SCRIPTS.glob("lint_*.py") if p.name not in NOT_DIFF_LOCAL
))
def test_every_fast_gates_guard_can_attribute(script):
    """A guard that stopped importing the attribution seam is a guard that
    silently went back to being unattributable. Structural, not a promise.

    The parametrization EXCLUDES the exempt set rather than skipping over it,
    so every row here executes and the exemption is a decision recorded in one
    place (`NOT_DIFF_LOCAL`, fenced as a set above) instead of a condition
    evaluated per row.
    """
    source = (SCRIPTS / script).read_text()
    assert "_lint_side" in source, (
        f"{script} carries no attribution seam at all — add it, or name it in "
        "NOT_DIFF_LOCAL with the reason it has no side to report"
    )
    assert "_lint_side.report" in source or "_lint_side.verdict" in source, (
        f"{script} imports the attribution seam but never calls it — the import "
        f"is the only thing left of the conversion"
    )
