#!/usr/bin/env python3
"""Assemble changelog.d/ fragments into CHANGELOG.md sections at cut time.

Every lane appending to one mutable CHANGELOG.md makes a conflict near-certain
for any PR that sits through a sibling merge. Lanes write
changelog.d/<issue>.md -- a path nobody else touches -- and a cut concatenates
them here.

pgw#1226: a cut MARKS the fragments it consumed in changelog.d/consumed.tsv; it
no longer deletes them, and it does not decide which version a fragment belongs
to. That is DERIVED: a fragment is attributed to the earliest release tag whose
tree contains it, and to the version being cut if no tag contains it. So

  * a fragment that lands after the tag is in no tag, and rides the NEXT cut;
  * a fragment that was in a TAGGED tree and never got assembled is attributed
    BACK to that version instead of being silently re-dated to this one.

The second case is the failure this replaces. 0.114.3 shipped pgw#1244's code
with changelog.d/pgw1244.md still unconsumed, and under the old tooling the next
cut would have written that bullet under its own version -- a release note
pointing at the wrong wheel, with nothing in the tree able to notice.

    scripts/assemble_changelog.py --check
    scripts/assemble_changelog.py --version 0.92.0 --headline "**what changed**"
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# <prefix><number>.md -- prefix names the id space (pgw, th, te, ...).
NAME = re.compile(r"^(?P<prefix>[a-z]+)(?P<number>\d+)$")
UNRELEASED = re.compile(r"^## Unreleased\b.*?(?=^## )", re.M | re.S)
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
RELEASE_TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")

LEDGER = "consumed.tsv"
LEDGER_HEADER = (
    "# pgw#1226: fragments a cut has already assembled, and the version whose TREE\n"
    "# contains them. Written by scripts/assemble_changelog.py; lanes never edit it.\n"
    "# <version>\\t<fragment stem>\n"
)
LATE_NOTE = (
    "*Attributed after the cut (pgw#1226): this fragment was in the tagged tree "
    "and was not assembled into the section at cut time.*"
)
# Consumed fragments stay on disk for this many PRIOR releases, so that a
# fragment a recently-merged branch could still be carrying is never removed
# underneath it. The cut's own are always kept.
RETAIN_PRIOR_RELEASES = 1


def fragments_dir(root: Path) -> Path:
    return root / "changelog.d"


def ledger_path(root: Path) -> Path:
    return fragments_dir(root) / LEDGER


def read_ledger(root: Path) -> list[tuple[str, str]]:
    path = ledger_path(root)
    if not path.exists():
        return []
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            sys.exit(f"{path}:{lineno}: want '<version>\\t<fragment>', got {line!r}")
        version, stem = parts[0].strip(), parts[1].strip()
        if not VERSION.match(version):
            sys.exit(f"{path}:{lineno}: {version!r} is not an X.Y.Z version")
        if not NAME.match(stem):
            sys.exit(f"{path}:{lineno}: {stem!r} is not a <prefix><number> fragment")
        if stem in seen:
            sys.exit(f"{path}:{lineno}: {stem} is recorded twice")
        seen.add(stem)
        rows.append((version, stem))
    return rows


def write_ledger(root: Path, rows: list[tuple[str, str]]) -> None:
    body = "".join(f"{version}\t{stem}\n" for version, stem in rows)
    ledger_path(root).write_text(LEDGER_HEADER + body)


def collect(root: Path) -> tuple[list[tuple[int, str, Path]], list[tuple[str, Path]]]:
    """(pending, consumed) fragments -- pending ordered by issue NUMBER."""
    consumed_at = {stem: version for version, stem in read_ledger(root)}
    pending: list[tuple[int, str, Path]] = []
    consumed: list[tuple[str, Path]] = []
    for path in fragments_dir(root).glob("*.md"):
        if path.name == "README.md":
            continue
        m = NAME.match(path.stem)
        if not m:
            sys.exit(
                f"{path}: name must be <prefix><number>.md (e.g. pgw968.md); "
                "the number is what orders the release section."
            )
        if not path.read_text().strip():
            sys.exit(f"{path}: empty fragment")
        if path.stem in consumed_at:
            consumed.append((consumed_at[path.stem], path))
            continue
        pending.append((int(m["number"]), m["prefix"], path))
    return sorted(pending), sorted(consumed)


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *args), cwd=root, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def release_versions(root: Path) -> list[str]:
    """Released versions, oldest first."""
    tags = git(root, "tag", "--list", "v*", "--sort=v:refname").splitlines()
    return [m["version"] for t in tags if (m := RELEASE_TAG.match(t.strip()))]


def attribution(root: Path, path: Path, cutting: str) -> str:
    """The earliest released version whose tree holds this fragment, else `cutting`.

    Derived rather than declared: the cutter cannot mis-remember which sweep
    they ran, and a fragment that shipped inside a tag cannot be re-dated.
    """
    rel = path.relative_to(root).as_posix()
    added = git(root, "log", "--diff-filter=A", "--format=%H", "-1", "--", rel)
    if not added:
        return cutting  # not committed yet -- it is this cut's own
    contains = git(root, "tag", "--contains", added, "--sort=v:refname").splitlines()
    for tag in contains:
        if m := RELEASE_TAG.match(tag.strip()):
            return m["version"]
    return cutting


def render(body: list[Path]) -> str:
    return "\n\n".join(p.read_text().strip() for p in body)


def new_section(version: str, date: str, headline: str, body: list[Path]) -> str:
    heading = f"## {version} ({date})"
    if headline:
        heading += f" — {headline}"
    return f"{heading}\n\n{render(body)}\n\n"


def insert_new_section(text: str, section: str) -> str:
    m = UNRELEASED.search(text)
    if not m:
        sys.exit("CHANGELOG.md: no '## Unreleased' block to insert after")
    return text[: m.end()] + section + text[m.end() :]


def append_to_section(text: str, version: str, body: list[Path]) -> str:
    m = re.search(rf"^## {re.escape(version)}\b.*?(?=^## |\Z)", text, re.M | re.S)
    if not m:
        sys.exit(
            f"CHANGELOG.md: no '## {version}' section to attribute "
            f"{', '.join(p.stem for p in body)} to. That version is released and "
            "its section should exist; do not invent one."
        )
    note = "" if LATE_NOTE in m.group(0) else f"{LATE_NOTE}\n\n"
    return text[: m.end()] + f"{note}{render(body)}\n\n" + text[m.end() :]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--version")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--headline", default="")
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    pending, consumed = collect(root)

    if args.check:
        for _, _, path in pending:
            print(f"pending  {path.relative_to(root)}")
        for version, path in consumed:
            print(f"consumed {path.relative_to(root)} -> {version}")
        return 0
    if not args.version:
        ap.error("--version is required unless --check")
    if not VERSION.match(args.version):
        ap.error(f"--version {args.version!r} is not X.Y.Z")
    if not pending:
        sys.exit("no unconsumed fragments in changelog.d/ -- nothing to release")

    # Attribution reads tags, and a graft file makes `tag --contains` answer
    # about a history the repo does not have. Refuse rather than mis-attribute.
    if git(root, "rev-parse", "--is-shallow-repository") == "true":
        sys.exit("refusing: shallow repository -- run `git fetch --unshallow` first")
    released = release_versions(root)
    if not released:
        sys.exit("refusing: no vX.Y.Z tags found -- attribution cannot be derived")

    by_version: dict[str, list[Path]] = {}
    for _, _, path in pending:
        by_version.setdefault(attribution(root, path, args.version), []).append(path)

    changelog = root / "CHANGELOG.md"
    text = changelog.read_text()
    # Late attributions first: inserting the new section shifts every offset
    # below it, and the sections being appended to are all below it.
    for version in sorted(v for v in by_version if v != args.version):
        text = append_to_section(text, version, by_version[version])
        for path in by_version[version]:
            print(f"{path.name} -> {version} (late; its code shipped in v{version})")
    section = ""
    mine = by_version.get(args.version, [])
    if mine:
        section = new_section(args.version, args.date, args.headline, mine)
        text = insert_new_section(text, section)
        for path in mine:
            print(f"{path.name} -> {args.version}")
    else:
        # Not an error: naming the NEXT version with nothing pending for it is
        # how a mis-attribution is repaired without cutting anything.
        print(f"note: nothing pending belongs to {args.version} -- no new section")

    ledger = read_ledger(root) + [
        (version, path.stem)
        for version in sorted(by_version)
        for path in by_version[version]
    ]

    if args.dry_run:
        print(section, end="")
        return 0

    changelog.write_text(text)
    write_ledger(root, ledger)

    # Consumed fragments are marked, not deleted -- but they do not accumulate
    # forever either. Prune what an older release already consumed.
    keep = {args.version, *released[-RETAIN_PRIOR_RELEASES:]}
    for version, path in consumed:
        if version not in keep:
            path.unlink()
            print(f"pruned {path.name} (consumed by {version})")

    print(f"{len(pending)} fragment(s) assembled; {len(ledger)} in {LEDGER}")
    print("now: git add CHANGELOG.md changelog.d  (with the version bump)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
