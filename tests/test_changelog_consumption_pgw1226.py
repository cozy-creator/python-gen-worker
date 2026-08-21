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

CHANGELOG = """\
# Changelog

## Unreleased

Unreleased entries live in `changelog.d/`, one file per issue.

## 0.114.3 (2026-08-13) — the resume half

- **pgw#1240: the resume load half.**

## 0.114.2 (2026-08-12) — the format axis

- **pgw#1230: the format axis fix.**
"""

GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "pgw1226 fixture",
    "GIT_AUTHOR_EMAIL": "pgw1226@fixture.invalid",
    "GIT_COMMITTER_NAME": "pgw1226 fixture",
    "GIT_COMMITTER_EMAIL": "pgw1226@fixture.invalid",
}


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *args), cwd=root, env=GIT_ENV, capture_output=True, text=True
    )
    assert done.returncode == 0, f"git {args}: {done.stderr}"
    return done.stdout.strip()


def fragment(root: Path, stem: str, text: str) -> None:
    (root / "changelog.d" / f"{stem}.md").write_text(f"- **{stem}: {text}**\n")


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
def cut_repo(tmp_path: Path) -> Path:
    """Two releases tagged; three fragments none of them consumed."""
    root = tmp_path / "pgw"
    (root / "changelog.d").mkdir(parents=True)
    (root / "CHANGELOG.md").write_text(CHANGELOG)
    (root / "changelog.d" / "README.md").write_text("# changelog.d\n")
    git(root, "init", "-q", "-b", "master")

    fragment(root, "pgw1200", "in both tags")
    git(root, "add", "CHANGELOG.md", "changelog.d")
    git(root, "commit", "-q", "-m", "base")
    git(root, "tag", "v0.114.2")

    fragment(root, "pgw1244", "in the 0.114.3 tree, never assembled")
    git(root, "add", "changelog.d/pgw1244.md")
    git(root, "commit", "-q", "-m", "pgw#1244")
    git(root, "tag", "v0.114.3")

    fragment(root, "pgw1250", "landed after the tag")
    git(root, "add", "changelog.d/pgw1250.md")
    git(root, "commit", "-q", "-m", "pgw#1250")
    return root


def test_tagged_fragments_are_attributed_back_and_nothing_is_deleted(
    cut_repo: Path,
) -> None:
    done = assemble(cut_repo, "--version", "0.115.0")
    assert done.returncode == 0, done.stderr

    text = (cut_repo / "CHANGELOG.md").read_text()

    assert "pgw1200" in section(text, "0.114.2")
    assert "pgw1244" in section(text, "0.114.3")
    assert "pgw1250" in section(text, "0.115.0")

    assert "pgw1244" not in section(text, "0.115.0")
    assert "pgw1200" not in section(text, "0.115.0")

    assert LATE in section(text, "0.114.3")
    assert LATE in section(text, "0.114.2")
    assert LATE not in section(text, "0.115.0")

    for stem in ("pgw1200", "pgw1244", "pgw1250"):
        assert (cut_repo / "changelog.d" / f"{stem}.md").exists()

    ledger = (cut_repo / LEDGER).read_text()
    rows = [ln for ln in ledger.splitlines() if not ln.startswith("#")]
    assert rows == ["0.114.2\tpgw1200", "0.114.3\tpgw1244", "0.115.0\tpgw1250"]


def test_a_consumed_fragment_is_never_assembled_twice(cut_repo: Path) -> None:
    assert assemble(cut_repo, "--version", "0.115.0").returncode == 0
    before = (cut_repo / "CHANGELOG.md").read_text()

    again = assemble(cut_repo, "--version", "0.116.0")
    assert again.returncode != 0
    assert "nothing to release" in again.stdout + again.stderr
    assert (cut_repo / "CHANGELOG.md").read_text() == before


def test_the_ledger_is_what_stops_the_second_write(cut_repo: Path) -> None:
    """Severance: the tag rule fixes the VERSION, the ledger fixes the COUNT."""
    assert assemble(cut_repo, "--version", "0.115.0").returncode == 0
    (cut_repo / LEDGER).unlink()

    assert assemble(cut_repo, "--version", "0.116.0").returncode == 0
    assert section((cut_repo / "CHANGELOG.md").read_text(), "0.114.3").count(
        "pgw1244"
    ) == 2


def test_consumed_fragments_are_pruned_one_release_later(cut_repo: Path) -> None:
    assert assemble(cut_repo, "--version", "0.115.0").returncode == 0
    git(cut_repo, "add", "CHANGELOG.md", "changelog.d")
    git(cut_repo, "commit", "-q", "-m", "cut 0.115.0")
    git(cut_repo, "tag", "v0.115.0")

    fragment(cut_repo, "pgw1260", "after the 0.115.0 tag")
    git(cut_repo, "add", "changelog.d/pgw1260.md")
    git(cut_repo, "commit", "-q", "-m", "pgw#1260")

    done = assemble(cut_repo, "--version", "0.116.0")
    assert done.returncode == 0, done.stderr

    frags = cut_repo / "changelog.d"
    assert (frags / "pgw1250.md").exists()
    assert not (frags / "pgw1200.md").exists()
    assert not (frags / "pgw1244.md").exists()

    rows = [
        ln for ln in (cut_repo / LEDGER).read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert rows == [
        "0.114.2\tpgw1200",
        "0.114.3\tpgw1244",
        "0.115.0\tpgw1250",
        "0.116.0\tpgw1260",
    ]
    assert "pgw1260" in section((cut_repo / "CHANGELOG.md").read_text(), "0.116.0")


def test_attribution_refuses_without_release_tags(cut_repo: Path) -> None:
    for tag in ("v0.114.2", "v0.114.3"):
        git(cut_repo, "tag", "-d", tag)
    done = assemble(cut_repo, "--version", "0.115.0")
    assert done.returncode != 0
    assert "no vX.Y.Z tags" in done.stdout + done.stderr


def test_check_needs_no_git_and_reports_both_states(tmp_path: Path) -> None:
    """`--check` is the fast-gates guard, so it must not depend on a repository."""
    root = tmp_path / "bare"
    (root / "changelog.d").mkdir(parents=True)
    (root / "changelog.d" / "README.md").write_text("# changelog.d\n")
    fragment(root, "pgw1250", "pending")
    fragment(root, "pgw1244", "consumed")
    (root / LEDGER).write_text("0.114.3\tpgw1244\n")

    done = assemble(root, "--check")
    assert done.returncode == 0, done.stderr
    assert "pending  changelog.d/pgw1250.md" in done.stdout
    assert "consumed changelog.d/pgw1244.md -> 0.114.3" in done.stdout


@pytest.mark.parametrize(
    "row, expected",
    [
        ("0.114.3 pgw1244\n", "want '<version>"),
        ("0.114\tpgw1244\n", "is not an X.Y.Z version"),
        ("0.114.3\tpgw868_a4\n", "is not a <prefix><number>"),
        ("0.114.3\tpgw-a4\n", "is not a <prefix><number>"),
        ("0.114.3\tpgw1244\n0.115.0\tpgw1244\n", "recorded twice"),
    ],
)
def test_check_refuses_a_malformed_ledger(
    tmp_path: Path, row: str, expected: str
) -> None:
    root = tmp_path / "bad"
    (root / "changelog.d").mkdir(parents=True)
    (root / LEDGER).write_text(row)
    done = assemble(root, "--check")
    assert done.returncode != 0
    assert expected in done.stdout + done.stderr


def test_the_repos_own_pending_fragments_parse() -> None:
    done = assemble(REPO, "--check")
    assert done.returncode == 0, done.stderr
