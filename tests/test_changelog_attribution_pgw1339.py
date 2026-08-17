"""pgw#1339: `--version X` assembles only fragments whose WORK is inside X.

The defect this covers was reproduced twice, by two cutters who had not read
each other's notes. `assemble_changelog.py --version 0.121.0`, run to date one
late fragment into an already-released section, swept **every** pending
fragment into `## 0.121.0` -- 16 of them on the second attempt, including work
that had shipped in no wheel at all. The old rule dated a fragment by the
commit that added the fragment FILE, and fell back to "the version being cut"
when no tag contained it; an uncommitted or late-added fragment therefore
always took the fallback, and the fallback cannot tell a repair from a cut.

Driven against a REAL git repository in a tmpdir -- real commits, real `v*`
tags, the real script as a subprocess -- because every claim here is a claim
about what git answers. The fixture inherits NO git config
(`GIT_CONFIG_GLOBAL`/`SYSTEM` are /dev/null), so it needs no signing bypass.

The fixture is the 0.123.0 cutter's shape at small scale: `v0.121.0` contains
pgw#1323's commit and nothing else pending does, so a `--version 0.121.0` run
must assemble exactly one fragment and hold the rest.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "assemble_changelog.py"
LEDGER = "changelog.d/consumed.tsv"
LATE = "Attributed after the cut (pgw#1226)"

# The pending fragments that landed AFTER v0.121.0 -- the sweep's victims.
AFTER_TAG = ("pgw1330", "pgw1340", "pgw1350")

CHANGELOG = """\
# Changelog

## Unreleased

Unreleased entries live in `changelog.d/`, one file per issue.

## 0.121.0 (2026-08-16) — a hub-dispatched `@job` reaches its body

- **pgw#1338: the dispatch half.**

## 0.120.0 (2026-08-15) — the boot half

- **pgw#1300: the boot half.**
"""

GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "pgw1339 fixture",
    "GIT_AUTHOR_EMAIL": "pgw1339@fixture.invalid",
    "GIT_COMMITTER_NAME": "pgw1339 fixture",
    "GIT_COMMITTER_EMAIL": "pgw1339@fixture.invalid",
}


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *args), cwd=root, env=GIT_ENV, capture_output=True, text=True
    )
    assert done.returncode == 0, f"git {args}: {done.stderr}"
    return done.stdout.strip()


def fragment(root: Path, stem: str, text: str) -> None:
    (root / "changelog.d" / f"{stem}.md").write_text(f"- **{stem}: {text}**\n")


def work(root: Path, stem: str, subject: str) -> None:
    """A commit whose SUBJECT claims the issue -- the thing being dated."""
    number = stem.removeprefix("pgw")
    src = root / f"{stem}_work.py"
    src.write_text(f"{src.read_text() if src.exists() else ''}# {subject}\n")
    git(root, "add", f"{stem}_work.py")
    git(root, "commit", "-q", "-m", f"pgw#{number}: {subject}")


def assemble(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(SCRIPT), "--root", str(root), *args),
        capture_output=True,
        text=True,
    )


def section(text: str, version: str) -> str:
    m = re.search(rf"^## {re.escape(version)}\b.*?(?=^## |\Z)", text, re.M | re.S)
    assert m, f"no '## {version}' section in:\n{text}"
    return m.group(0)


@pytest.fixture
def repair_repo(tmp_path: Path) -> Path:
    """v0.121.0 holds pgw#1323's commit; three later issues are pending.

    pgw#1323's own fragment does not exist yet -- that is the live shape: the
    change shipped as a silent release note and the fragment is being written
    now, months of commits later.
    """
    root = tmp_path / "pgw"
    (root / "changelog.d").mkdir(parents=True)
    (root / "CHANGELOG.md").write_text(CHANGELOG)
    (root / "changelog.d" / "README.md").write_text("# changelog.d\n")
    git(root, "init", "-q", "-b", "master")
    git(root, "add", "CHANGELOG.md", "changelog.d")
    git(root, "commit", "-q", "-m", "base")
    git(root, "tag", "v0.120.0")

    work(root, "pgw1323", "the cross-repo consumer fence stopped reading")
    git(root, "tag", "v0.121.0")

    for stem in AFTER_TAG:
        work(root, stem, "landed after the 0.121.0 tag")
        fragment(root, stem, "landed after the 0.121.0 tag")
        git(root, "add", f"changelog.d/{stem}.md")
        git(root, "commit", "-q", "-m", f"pgw#{stem.removeprefix('pgw')}: fragment")
    return root


# --------------------------------------------------------------------------
# the defect: `--version <older>` swept every pending fragment into it
# --------------------------------------------------------------------------


def test_a_repair_run_does_not_sweep_unshipped_fragments_into_the_old_section(
    repair_repo: Path,
) -> None:
    fragment(repair_repo, "pgw1323", "the cross-repo consumer fence")

    done = assemble(repair_repo, "--version", "0.121.0")
    assert done.returncode == 0, done.stderr

    text = (repair_repo / "CHANGELOG.md").read_text()
    assert "pgw1323" in section(text, "0.121.0")
    # the whole defect: work that is in NO tag may not be dated into a shipped
    # wheel, and 0.121.0 is shipped.
    for stem in AFTER_TAG:
        assert stem not in text, f"{stem} was swept into a released section"

    # ...and they are not silently dropped either.
    assert f"{len(AFTER_TAG)} fragment(s) pending, not in 0.121.0:" in done.stdout
    for stem in AFTER_TAG:
        assert f"{stem}.md:" in done.stdout
    assert "is in no release tag" in done.stdout


def test_the_ledger_records_only_what_was_assembled(repair_repo: Path) -> None:
    fragment(repair_repo, "pgw1323", "the cross-repo consumer fence")
    assert assemble(repair_repo, "--version", "0.121.0").returncode == 0

    rows = [
        ln
        for ln in (repair_repo / LEDGER).read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert rows == ["0.121.0\tpgw1323"]
    # held fragments stay pending, so the NEXT cut picks them up
    for stem in AFTER_TAG:
        assert (repair_repo / "changelog.d" / f"{stem}.md").exists()


def test_the_repair_merges_into_the_existing_section_and_does_not_duplicate_it(
    repair_repo: Path,
) -> None:
    """Bug 2: the repair used to PREPEND a second `## 0.121.0`, out of order."""
    fragment(repair_repo, "pgw1323", "the cross-repo consumer fence")
    assert assemble(repair_repo, "--version", "0.121.0").returncode == 0

    text = (repair_repo / "CHANGELOG.md").read_text()
    assert text.count("## 0.121.0") == 1
    body = section(text, "0.121.0")
    assert LATE in body
    assert body.index("pgw#1338") < body.index("pgw1323")  # appended, not prepended
    assert text.index("## 0.121.0") > text.index("## Unreleased")


def test_a_fragment_committed_after_the_tag_is_still_dated_by_its_work(
    repair_repo: Path,
) -> None:
    """The FILE is a document about the change; the COMMIT is the change."""
    fragment(repair_repo, "pgw1323", "the cross-repo consumer fence")
    git(repair_repo, "add", "changelog.d/pgw1323.md")
    git(repair_repo, "commit", "-q", "-m", "pgw#1339: write the missing fragment")

    done = assemble(repair_repo, "--version", "0.121.0")
    assert done.returncode == 0, done.stderr
    assert "pgw1323" in section(
        (repair_repo / "CHANGELOG.md").read_text(), "0.121.0"
    )


def test_nothing_assembled_is_a_refusal_and_writes_nothing(
    repair_repo: Path,
) -> None:
    before = (repair_repo / "CHANGELOG.md").read_text()
    done = assemble(repair_repo, "--version", "0.121.0")
    assert done.returncode != 0
    assert "nothing assembled" in done.stdout + done.stderr
    assert (repair_repo / "CHANGELOG.md").read_text() == before
    assert not (repair_repo / LEDGER).exists()


def test_partly_shipped_work_is_held_rather_than_dated_into_the_old_wheel(
    repair_repo: Path,
) -> None:
    """A note is true only once EVERYTHING it describes has shipped."""
    work(repair_repo, "pgw1323", "the follow-up that did NOT make 0.121.0")
    fragment(repair_repo, "pgw1323", "the cross-repo consumer fence")

    done = assemble(repair_repo, "--version", "0.121.0")
    assert done.returncode != 0
    assert "pgw1323.md:" in done.stdout
    assert "pgw1323" not in (repair_repo / "CHANGELOG.md").read_text()


# --------------------------------------------------------------------------
# ...and a real cut still cuts
# --------------------------------------------------------------------------


def test_a_real_cut_takes_the_unshipped_work_and_late_attributes_the_rest(
    repair_repo: Path,
) -> None:
    fragment(repair_repo, "pgw1323", "the cross-repo consumer fence")

    done = assemble(repair_repo, "--version", "0.122.0", "--headline", "**the cut**")
    assert done.returncode == 0, done.stderr

    text = (repair_repo / "CHANGELOG.md").read_text()
    assert "pgw1323" in section(text, "0.121.0")  # its work is in that tag
    assert "pgw1323" not in section(text, "0.122.0")
    for stem in AFTER_TAG:
        assert stem in section(text, "0.122.0")
    assert "**the cut**" in text
    rows = [
        ln
        for ln in (repair_repo / LEDGER).read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert rows == [
        "0.121.0\tpgw1323",
        "0.122.0\tpgw1330",
        "0.122.0\tpgw1340",
        "0.122.0\tpgw1350",
    ]


def test_work_outside_the_cut_ref_rides_the_next_cut(repair_repo: Path) -> None:
    """`--cut-ref` is the release commit: work not in it is not in the wheel."""
    cut_ref = git(repair_repo, "rev-parse", "HEAD~1")  # before pgw1350's commit

    done = assemble(repair_repo, "--version", "0.122.0", "--cut-ref", cut_ref)
    assert done.returncode == 0, done.stderr

    text = (repair_repo / "CHANGELOG.md").read_text()
    assert "pgw1330" in section(text, "0.122.0")
    assert "pgw1350" not in text
    assert "1 fragment(s) pending, not in 0.122.0:" in done.stdout
    assert f"is not in {cut_ref}" in done.stdout


# --------------------------------------------------------------------------
# per-lane fragment names: one issue, many lanes, no shared path
# --------------------------------------------------------------------------


def test_per_lane_fragments_of_one_issue_fold_into_one_section(
    repair_repo: Path,
) -> None:
    """pgw#1346's ~10 lanes shared `pgw1346.md` and re-serialised the queue.

    Disjoint files, one issue: both are dated by `pgw#1346`'s commits, both land
    in the same section, adjacent and ordered, each with its own ledger row.
    """
    work(repair_repo, "pgw1346", "the B3 math half")
    fragment(repair_repo, "pgw1346-b4-video", "the video half")
    fragment(repair_repo, "pgw1346-b3-math", "the math half")
    fragment(repair_repo, "pgw1346", "the roll-up")

    done = assemble(repair_repo, "--version", "0.122.0")
    assert done.returncode == 0, done.stderr

    body = section((repair_repo / "CHANGELOG.md").read_text(), "0.122.0")
    for stem in ("pgw1346", "pgw1346-b3-math", "pgw1346-b4-video"):
        assert stem in body
    # unsuffixed first, then suffixes in order, and all three adjacent -- the
    # section must not depend on filesystem order.
    assert (
        body.index("pgw1346:")
        < body.index("pgw1346-b3-math:")
        < body.index("pgw1346-b4-video:")
    )
    assert body.index("pgw1346-b4-video:") < body.index("pgw1350:")

    rows = [
        ln
        for ln in (repair_repo / LEDGER).read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert "0.122.0\tpgw1346" in rows
    assert "0.122.0\tpgw1346-b3-math" in rows
    assert "0.122.0\tpgw1346-b4-video" in rows


def test_a_suffixed_fragment_is_dated_by_its_ISSUE_not_its_suffix(
    repair_repo: Path,
) -> None:
    """The suffix is a filename. `pgw#1323`'s tag is what dates the lane file."""
    fragment(repair_repo, "pgw1323-fence", "the cross-repo consumer fence")

    done = assemble(repair_repo, "--version", "0.121.0")
    assert done.returncode == 0, done.stderr
    assert "pgw1323-fence" in section(
        (repair_repo / "CHANGELOG.md").read_text(), "0.121.0"
    )


@pytest.mark.parametrize("stem", ["pgw-b3-math", "b3-math", "pgw1346-B3", "pgw1346-"])
def test_a_name_with_no_issue_number_is_refused(repair_repo: Path, stem: str) -> None:
    """A suffix may not become an escape hatch: no number, nothing to date it."""
    (repair_repo / "changelog.d" / f"{stem}.md").write_text("- **a lane half**\n")

    done = assemble(repair_repo, "--check")
    assert done.returncode != 0
    assert "must be <prefix><number>[-<suffix>].md" in done.stdout + done.stderr


def test_an_issue_with_no_subject_commit_falls_back_to_the_fragments_own_commit(
    repair_repo: Path,
) -> None:
    """pgw#1226's original signal, kept as the fallback it always was.

    A 0.122.0 whose tree holds `pgw1360.md`, added under a subject that names
    no issue: nothing else can date it, so the file's own commit does — and
    dating it to the version being cut is the pgw#1226 defect all over again.
    """
    changelog = repair_repo / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            "## 0.121.0",
            "## 0.122.0 (2026-08-17) — the interim cut\n\n"
            "- **pgw#1355: the interim half.**\n\n## 0.121.0",
            1,
        )
    )
    fragment(repair_repo, "pgw1360", "committed under a subject that names nothing")
    git(repair_repo, "add", "CHANGELOG.md", "changelog.d/pgw1360.md")
    git(repair_repo, "commit", "-q", "-m", "housekeeping")
    git(repair_repo, "tag", "v0.122.0")

    done = assemble(repair_repo, "--version", "0.123.0")
    assert done.returncode == 0, done.stderr
    text = (repair_repo / "CHANGELOG.md").read_text()
    assert "pgw1360" in section(text, "0.122.0")
    assert "## 0.123.0" not in text
