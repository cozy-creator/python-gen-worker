#!/usr/bin/env python3
"""Assemble changelog.d/ fragments into CHANGELOG.md sections at cut time.

Every lane appending to one mutable CHANGELOG.md makes a conflict near-certain
for any PR that sits through a sibling merge. Lanes write
changelog.d/<issue>.md -- a path nobody else touches -- and a cut concatenates
them here.

pgw#1226: a cut MARKS the fragments it consumed in changelog.d/consumed.tsv; it
no longer deletes them, and it does not decide which version a fragment belongs
to. That is DERIVED from git.

pgw#1339: what is derived is the version containing the fragment's SUBJECT WORK,
not the version containing the fragment FILE. A fragment is a document about a
change; the change is the thing a release note dates. So

  * the issue's authored commits are resolved from commit SUBJECTS
    (`pgw#1323: ...`), falling back to the commit that added the fragment file
    when no subject names the issue;
  * the fragment is attributed to the earliest release tag whose tree contains
    ALL of them -- a note is only true once everything it describes has shipped;
  * if no tag contains them it belongs to the version being cut, and only if
    that version is not itself already released and the work is in the cut ref.
    Otherwise the fragment stays PENDING and is listed, never swept.

A fragment name may carry a per-lane suffix -- `pgw1346-b3-math.md` -- so the
lanes of one batched issue write DISJOINT paths instead of queueing behind a
shared `pgw1346.md`. The `<prefix><number>` core is what dates and orders it, so
it stays mandatory; the suffix is only a filename.

Two failures this kills, both observed:

  * 0.114.3 shipped pgw#1244's code with changelog.d/pgw1244.md unconsumed, and
    the old tooling would have written that bullet under the NEXT version -- a
    release note pointing at a wheel that did not contain the change.
  * `--version 0.121.0`, run to date ONE late fragment into an already-released
    section, dated every other pending fragment to 0.121.0 as well: 16 of them
    on the 0.123.0 cutter's tree, including work that had shipped in no wheel at
    all. The old rule's "in no tag -> the version being cut" fallback cannot
    tell a repair from a cut. This one can: those 16 are in no tag AND 0.121.0
    is already released, so they stay pending.

    scripts/assemble_changelog.py --check
    scripts/assemble_changelog.py --version 0.92.0 --headline "**what changed**"
    scripts/assemble_changelog.py --version 0.121.0   # repair: append-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# <prefix><number>[-<suffix>].md -- prefix names the id space (pgw, th, te, ...).
#
# The optional suffix is pgw#1346's lesson: ~10 lanes of one batched issue were
# all appending to a single `pgw1346.md`, which re-serialised the merge queue on
# a shared path and cost repeated CONFLICTING ejections -- the exact failure
# `changelog.d/` exists to remove. `pgw1346-b3-math.md` and `pgw1346-b4-video.md`
# are disjoint files that still resolve to issue pgw#1346 for dating, and land
# adjacent in the section. The <prefix><number> core stays mandatory: it is what
# attribution keys on, so a suffix can never smuggle in an undatable fragment.
NAME = re.compile(
    # A 2+ letter prefix, so a suffixed name cannot be read as one: `b3-math`
    # would otherwise parse as issue b#3 and be dated by nothing.
    r"^(?P<prefix>[a-z]{2,})(?P<number>\d+)(?:-(?P<suffix>[a-z0-9][a-z0-9-]*))?$"
)
UNRELEASED = re.compile(r"^## Unreleased\b.*?(?=^## )", re.M | re.S)
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
RELEASE_TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")
# `pgw#1323: ...`, `th#2082 follow-up: ...` -- an issue reference in a commit
# SUBJECT. Bodies are excluded on purpose: a body cites related issues, a
# subject claims authorship of the change.
#
# Every ref in the subject counts, not just the leading one: 8 of the last 451
# subjects here trail it (`ci: fence the launch boundary (pgw#1239)`), and
# dropping those would date those fragments by the wrong signal. The cost is
# that a subject cross-referencing a second issue (67 of 451) lends it a commit
# it did not author -- which the all-commits rule below fails SAFE on: a stray
# commit outside a tag only ever moves a fragment LATER or into the printed
# pending list, never into a wheel that lacks the change.
SUBJECT_REF = re.compile(r"(?<![0-9a-z])(?P<prefix>[a-z]+)#0*(?P<number>\d+)(?![0-9])")

LEDGER = "consumed.tsv"
LEDGER_HEADER = (
    "# pgw#1226: fragments a cut has already assembled, and the version whose TREE\n"
    "# contains them. Written by scripts/assemble_changelog.py; lanes never edit it.\n"
    "# <version>\\t<fragment stem>\n"
)
LATE_NOTE_MARK = "*Attributed after the cut (pgw#1226)"
LATE_NOTE = (
    f"{LATE_NOTE_MARK}: the change below shipped in this version's tag and was "
    "not assembled into the section at cut time.*"
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
            sys.exit(
                f"{path}:{lineno}: {stem!r} is not a "
                "<prefix><number>[-<suffix>] fragment"
            )
        if stem in seen:
            sys.exit(f"{path}:{lineno}: {stem} is recorded twice")
        seen.add(stem)
        rows.append((version, stem))
    return rows


def write_ledger(root: Path, rows: list[tuple[str, str]]) -> None:
    body = "".join(f"{version}\t{stem}\n" for version, stem in rows)
    ledger_path(root).write_text(LEDGER_HEADER + body)


def collect(
    root: Path,
) -> tuple[list[tuple[int, str, str, Path]], list[tuple[str, Path]]]:
    """(pending, consumed) fragments.

    Pending is ordered by issue NUMBER, then by suffix -- so one issue's
    per-lane fragments land adjacent in the section, unsuffixed one first.
    """
    consumed_at = {stem: version for version, stem in read_ledger(root)}
    pending: list[tuple[int, str, str, Path]] = []
    consumed: list[tuple[str, Path]] = []
    for path in fragments_dir(root).glob("*.md"):
        if path.name == "README.md":
            continue
        m = NAME.match(path.stem)
        if not m:
            sys.exit(
                f"{path}: name must be <prefix><number>[-<suffix>].md "
                "(e.g. pgw968.md, or pgw1346-b3-math.md for one lane of a "
                "batched issue); the number is what orders the release section "
                "and what dates the fragment, so it is never optional."
            )
        if not path.read_text().strip():
            sys.exit(f"{path}: empty fragment")
        if path.stem in consumed_at:
            consumed.append((consumed_at[path.stem], path))
            continue
        pending.append((int(m["number"]), m["prefix"], m["suffix"] or "", path))
    return sorted(pending), sorted(consumed)


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", *args), cwd=root, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def release_versions(root: Path) -> list[str]:
    """Released versions, oldest first."""
    tags = git(root, "tag", "--list", "v*", "--sort=v:refname").splitlines()
    return [m["version"] for t in tags if (m := RELEASE_TAG.match(t.strip()))]


def subject_index(root: Path, refs: list[str]) -> dict[str, list[str]]:
    """`pgw1323` -> the commits whose SUBJECT claims that issue.

    One `git log` pass over the whole search space, because doing it per
    fragment is the same history walked N times. Merges are excluded: a merge
    subject names the PR number, not the issue, and its parents are here anyway.

    The space is the cut ref, HEAD and the release tags -- deliberately NOT
    `--all`: every open lane worktree carries commits whose subjects claim an
    issue, and letting unmerged work count would hold that issue's fragment out
    of a cut it genuinely belongs in.
    """
    index: dict[str, list[str]] = {}
    out = git(root, "log", "--no-merges", "--format=%H%x1f%s", *refs)
    for line in out.splitlines():
        commit, _, subject = line.partition("\x1f")
        for m in SUBJECT_REF.finditer(subject):
            key = f"{m['prefix']}{int(m['number'])}"
            index.setdefault(key, []).append(commit)
    return index


def added_commit(root: Path, path: Path) -> str:
    """The commit that added this fragment FILE, or '' if it is uncommitted."""
    rel = path.relative_to(root).as_posix()
    return git(root, "log", "--diff-filter=A", "--format=%H", "-1", "--", rel)


def containing_versions(root: Path, commit: str) -> set[str]:
    tags = git(root, "tag", "--contains", commit).splitlines()
    return {m["version"] for t in tags if (m := RELEASE_TAG.match(t.strip()))}


def is_ancestor(root: Path, commit: str, ref: str) -> bool:
    done = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, ref),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0


@dataclass(frozen=True)
class Work:
    """What a fragment's issue actually shipped in, derived from git."""

    stem: str
    issue: str  # `pgw#1346` -- the suffix does not date anything
    commits: tuple[str, ...]
    from_subject: bool
    released_in: str | None  # earliest release tag containing ALL of `commits`


def resolve(
    root: Path, path: Path, index: dict[str, list[str]], cache: dict[str, set[str]]
) -> Work:
    m = NAME.match(path.stem)
    assert m is not None, path  # collect() already refused anything else
    key = f"{m['prefix']}{int(m['number'])}"
    issue = f"{m['prefix']}#{int(m['number'])}"
    commits = list(dict.fromkeys(index.get(key, [])))
    from_subject = bool(commits)
    if not commits and (added := added_commit(root, path)):
        commits = [added]
    contained: set[str] | None = None
    for commit in commits:
        if commit not in cache:
            cache[commit] = containing_versions(root, commit)
        contained = cache[commit] if contained is None else contained & cache[commit]
    released_in = min(contained, key=version_key) if contained else None
    return Work(path.stem, issue, tuple(commits), from_subject, released_in)


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
    note = "" if LATE_NOTE_MARK in m.group(0) else f"{LATE_NOTE}\n\n"
    return text[: m.end()] + f"{note}{render(body)}\n\n" + text[m.end() :]


def held_reason(work: Work, cutting: str, cutting_released: bool, ref: str) -> str:
    """Why a fragment is not being assembled -- printed, never silent."""
    if not work.commits:
        return (
            f"no commit subject names {work.issue}, and the fragment file is not "
            f"committed: nothing proves it shipped in {cutting}"
        )
    where = "commit subjects" if work.from_subject else "the fragment's add commit"
    short = ", ".join(c[:8] for c in work.commits)
    if cutting_released:
        return (
            f"its work ({where}: {short}) is in no release tag, and {cutting} is "
            f"already released -- unshipped work cannot be dated into it"
        )
    return f"its work ({where}: {short}) is not in {ref}"


def plan_versions(
    root: Path,
    pending: list[tuple[int, str, str, Path]],
    cutting: str,
    cutting_released: bool,
    cut_ref: str,
    index: dict[str, list[str]],
) -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
    """Split the pending fragments into {version: fragments} and held-back."""
    by_version: dict[str, list[Path]] = {}
    held: list[tuple[Path, str]] = []
    cache: dict[str, set[str]] = {}
    for *_, path in pending:
        work = resolve(root, path, index, cache)
        if work.released_in is not None:
            by_version.setdefault(work.released_in, []).append(path)
            continue
        # In no release tag. It rides the version being cut only if that
        # version is not itself already released -- otherwise this is a repair
        # run, and dating unshipped work into a shipped wheel is the defect.
        in_cut = all(is_ancestor(root, c, cut_ref) for c in work.commits)
        if not cutting_released and in_cut:
            by_version.setdefault(cutting, []).append(path)
        else:
            held.append((path, held_reason(work, cutting, cutting_released, cut_ref)))
    return by_version, held


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--version")
    ap.add_argument(
        "--cut-ref",
        default="HEAD",
        help="the commit being cut; unreleased work counts only if it is here",
    )
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--headline", default="")
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    pending, consumed = collect(root)

    if args.check:
        for *_, path in pending:
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

    cutting: str = args.version
    cutting_released = cutting in released
    refs = list(dict.fromkeys([args.cut_ref, "HEAD", *(f"v{v}" for v in released)]))
    index = subject_index(root, refs)
    by_version, held = plan_versions(
        root, pending, cutting, cutting_released, args.cut_ref, index
    )

    changelog = root / "CHANGELOG.md"
    text = changelog.read_text()
    # Late attributions first: inserting the new section shifts every offset
    # below it, and the sections being appended to are all below it.
    late = sorted(
        (v for v in by_version if v != cutting or cutting_released), key=version_key
    )
    for version in late:
        text = append_to_section(text, version, by_version[version])
        for path in by_version[version]:
            print(f"{path.name} -> {version} (late; its code shipped in v{version})")
    section = ""
    mine = [] if cutting_released else by_version.get(cutting, [])
    if mine:
        section = new_section(cutting, args.date, args.headline, mine)
        text = insert_new_section(text, section)
        for path in mine:
            print(f"{path.name} -> {cutting}")
    elif not late:
        # Not an error to name a version nothing belongs to -- but nothing is
        # written for it either, and the held listing below says why.
        print(f"note: nothing pending belongs to {cutting} -- no new section")
    if held:
        print(f"{len(held)} fragment(s) pending, not in {cutting}:")
        for path, why in held:
            print(f"  {path.name}: {why}")
    if not by_version:
        sys.exit(f"nothing assembled: no pending fragment's work is in {cutting}")

    ledger = read_ledger(root) + sorted(
        ((version, path.stem) for version in by_version for path in by_version[version]),
        key=lambda row: (version_key(row[0]), row[1]),
    )

    if args.dry_run:
        print(section, end="")
        return 0

    changelog.write_text(text)
    write_ledger(root, ledger)

    # Consumed fragments are marked, not deleted -- but they do not accumulate
    # forever either. Prune what an older release already consumed. The window
    # is anchored on the NEWEST version known, so a repair run of an older
    # version prunes exactly what a cut would, and no more.
    known = sorted({*released, cutting}, key=version_key)
    keep = {cutting, *known[-(RETAIN_PRIOR_RELEASES + 1) :]}
    for version, path in consumed:
        if version not in keep:
            path.unlink()
            print(f"pruned {path.name} (consumed by {version})")

    assembled = sum(len(paths) for paths in by_version.values())
    print(f"{assembled} fragment(s) assembled; {len(ledger)} in {LEDGER}")
    print("now: git add CHANGELOG.md changelog.d  (with the version bump)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
