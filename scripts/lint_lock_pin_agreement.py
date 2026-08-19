#!/usr/bin/env python3
"""pgw#1477: `pyproject.toml`'s git pins and `uv.lock` must name the same rev.

Every job in every workflow begins with `uv sync --locked`. When a re-vendor
moves the torchcg rev in `[tool.uv.sources]` and nobody re-locks, that step dies
with

    error: The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.

before ANY gate runs — so `fast gates`, `tests` and `drift` all go red at
install, on PRs that touched neither file, and nothing in the required set can
report the cause. It happened twice on 2026-08-19, twenty minutes apart.

This check is the FIRST step of those jobs, ahead of `Install uv`: stdlib only,
no venv, no network, ~30 ms. Its verdict NAMES both files and both revs.

It also fences the third spelling — `_vendor/VENDORED.toml`'s `rev`, the
snapshot the mint and serving path run — so a re-vendor cannot leave the derive
and the mint on different libraries either.

Usage:
    python3 scripts/lint_lock_pin_agreement.py
    python3 scripts/lint_lock_pin_agreement.py --selftest   # prove it goes red
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: uv writes a git source as `<repo>?rev=<rev>#<resolved>` (lock `source.git`),
#: and as `<repo>?rev=<rev>` in the `[package.metadata]` tables. One pattern
#: reads both.
_LOCK_GIT = re.compile(r"^(?P<repo>[^?#]+)\?(?:[^#]*&)?rev=(?P<rev>[0-9a-fA-F]+)")

RELOCK = "uv lock   # then commit uv.lock"


def _norm(repo: str) -> str:
    """Compare repo URLs the way a human means them: no `.git`, no trailing /."""
    return repo.removesuffix("/").removesuffix(".git")


def _agree(a: str, b: str) -> bool:
    """Revs agree when one is a prefix of the other (short vs 40-hex spelling)."""
    return a.startswith(b) or b.startswith(a)


def _pyproject_git_pins(pyproject: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """`{package: (repo, rev)}` for every `[tool.uv.sources]` entry pinned by rev."""
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    pins: dict[str, tuple[str, str]] = {}
    for name, spec in sources.items():
        for entry in spec if isinstance(spec, list) else [spec]:
            if isinstance(entry, dict) and "git" in entry and "rev" in entry:
                pins[name] = (_norm(str(entry["git"])), str(entry["rev"]))
    return pins


def _strings(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return []


def _lock_revs_for_repo(lock: dict[str, Any], repo: str) -> set[str]:
    """Every rev uv.lock spells for `repo`, anywhere in the document."""
    revs: set[str] = set()
    for url in _strings(lock):
        match = _LOCK_GIT.match(url)
        if match and _norm(match["repo"]) == repo:
            revs.add(match["rev"])
    return revs


def check(root: Path) -> list[str]:
    """Return one human-readable failure per divergent pin (empty == green)."""
    pyproject_path = root / "pyproject.toml"
    lock_path = root / "uv.lock"
    manifest_path = root / "src" / "gen_worker" / "_vendor" / "VENDORED.toml"

    pyproject = tomllib.loads(pyproject_path.read_text())
    lock = tomllib.loads(lock_path.read_text())
    pins = _pyproject_git_pins(pyproject)

    failures: list[str] = []
    for name, (repo, rev) in sorted(pins.items()):
        lock_revs = _lock_revs_for_repo(lock, repo)
        if not lock_revs:
            failures.append(
                f"{name}: pyproject.toml pins {repo} @ {rev} in [tool.uv.sources], "
                f"and uv.lock names that repository nowhere. Run `{RELOCK}`."
            )
            continue
        stale = sorted(r for r in lock_revs if not _agree(r, rev))
        if stale:
            failures.append(
                f"{name}: the pin in pyproject.toml and uv.lock DISAGREE.\n"
                f"      pyproject.toml  [tool.uv.sources].{name}  rev = {rev}\n"
                f"      uv.lock         {repo}  rev = {', '.join(stale)}\n"
                f"      Fix: `{RELOCK}`. Better: re-vendor through "
                f"`scripts/vendor_snapshot.py {name} <rev>`, which re-locks in "
                f"the same act so these two cannot diverge."
            )

    # Third spelling: the vendored snapshot. Same library, different copy.
    if manifest_path.is_file():
        manifest = tomllib.loads(manifest_path.read_text())
        for name, package in sorted(manifest.get("packages", {}).items()):
            if name not in pins:
                continue
            repo, rev = pins[name]
            vendored = str(package.get("rev", ""))
            if _norm(str(package.get("repo", repo))) != repo:
                continue
            if vendored and not _agree(vendored, rev):
                failures.append(
                    f"{name}: the VENDORED SNAPSHOT and the dev pin DISAGREE.\n"
                    f"      pyproject.toml                     rev = {rev}\n"
                    f"      src/gen_worker/_vendor/VENDORED.toml  rev = {vendored}\n"
                    f"      The mint/serving path runs the snapshot and the derive "
                    f"runs the pin; they are a matched pair (pgw#1457). Re-vendor "
                    f"through `scripts/vendor_snapshot.py {name} <rev>`."
                )
    return failures


_SELFTEST_PYPROJECT = """
[project]
name = "x"
version = "0"
[tool.uv.sources]
lib = {{ git = "https://example.invalid/org/lib", rev = "{rev}" }}
"""

_SELFTEST_LOCK = """
version = 1
[[package]]
name = "lib"
version = "0.1.0"
source = {{ git = "https://example.invalid/org/lib?rev={rev}#{rev}" }}
"""


def _selftest() -> int:
    """Prove the check can go red — a guard that cannot fail proves nothing."""
    import tempfile

    agreeing = "a" * 40
    divergent = "b" * 40
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            _SELFTEST_PYPROJECT.format(rev=agreeing)
        )

        (root / "uv.lock").write_text(_SELFTEST_LOCK.format(rev=agreeing))
        green = check(root)
        if green:
            print(f"SELFTEST FAILED: an agreeing tree went red: {green}")
            return 1

        (root / "uv.lock").write_text(_SELFTEST_LOCK.format(rev=divergent))
        red = check(root)
        if not red:
            print("SELFTEST FAILED: a DIVERGENT tree went green — the check is dead")
            return 1
        message = "\n".join(red)
        for owed in ("pyproject.toml", "uv.lock", agreeing[:8], divergent[:8]):
            if owed not in message:
                print(f"SELFTEST FAILED: the verdict never names {owed!r}:\n{message}")
                return 1

        # And a lock that names the repo nowhere is red too, not silently green.
        (root / "uv.lock").write_text("version = 1\n")
        if not check(root):
            print("SELFTEST FAILED: a lock missing the pin entirely went green")
            return 1

    print("selftest ok: agreeing tree green, divergent tree red naming both files")
    return 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()

    root = Path(argv[1]) if len(argv) > 1 else ROOT
    failures = check(root)
    if not failures:
        return 0

    print(
        "pgw#1477: uv.lock is STALE against pyproject.toml.\n"
        "\n"
        "  Every CI job starts with `uv sync --locked`, which dies on this\n"
        "  BEFORE any gate runs — three red checks, none of them the cause.\n",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
