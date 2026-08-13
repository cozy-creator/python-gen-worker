#!/usr/bin/env python3
"""Collapse changelog.d/ fragments into a CHANGELOG.md section at cut time.

every lane appending to one mutable CHANGELOG.md made a conflict on
`dev` near-certain for any PR that sat through a sibling merge. Lanes now write
changelog.d/<issue>.md -- a path nobody else touches -- and a cut concatenates
them here.

    scripts/assemble_changelog.py --check
    scripts/assemble_changelog.py --version 0.92.0 --headline "**what changed**"
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS = ROOT / "changelog.d"
CHANGELOG = ROOT / "CHANGELOG.md"

# <prefix><number>.md -- prefix names the id space (pgw, th, te, ...).
NAME = re.compile(r"^(?P<prefix>[a-z]+)(?P<number>\d+)$")
UNRELEASED = re.compile(r"^## Unreleased\b.*?(?=^## )", re.M | re.S)


def collect() -> list[tuple[int, str, Path]]:
    """Fragments ordered by issue NUMBER, never by filesystem order."""
    out = []
    for path in FRAGMENTS.glob("*.md"):
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
        out.append((int(m["number"]), m["prefix"], path))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--headline", default="")
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    fragments = collect()
    if args.check:
        for _, _, path in fragments:
            print(path.relative_to(ROOT))
        return 0
    if not args.version:
        ap.error("--version is required unless --check")
    if not fragments:
        sys.exit("changelog.d/ is empty -- nothing to release")

    heading = f"## {args.version} ({args.date})"
    if args.headline:
        heading += f" — {args.headline}"
    body = "\n\n".join(p.read_text().strip() for _, _, p in fragments)
    section = f"{heading}\n\n{body}\n\n"

    text = CHANGELOG.read_text()
    m = UNRELEASED.search(text)
    if not m:
        sys.exit("CHANGELOG.md: no '## Unreleased' block to insert after")
    updated = text[: m.end()] + section + text[m.end() :]

    if args.dry_run:
        print(section)
        return 0

    CHANGELOG.write_text(updated)
    for _, _, path in fragments:
        path.unlink()  # `git add changelog.d` stages the removals
    print(f"{len(fragments)} fragment(s) -> {heading}")
    print("now: git add CHANGELOG.md changelog.d  (with the version bump)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
