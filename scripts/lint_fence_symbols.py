#!/usr/bin/env python3
"""THE GUARDS ARE GUARDED. Every symbol a fence NAMES must exist.

A fence names symbols in string literals: the arm-state feeder list, the
compiled graph-key derivation allowlist, the layout fence's ``AXIS_PRODUCERS``. When one
of those symbols is renamed or deleted, the fence keeps running, keeps exiting
0, and **guards nothing** — nothing is failing, so no amount of running things
finds it. This script asks every fence what symbol it names NOW, mechanically.

IT SCANS ITSELF: this script is a guard that names things, so it has exactly
the failure mode it detects. Pointed at itself rather than exempted — if it
names a symbol that no longer exists it goes red about itself, and the
self-reference terminates rather than recursing.

DELIBERATELY SMALL: string literals in lint scripts and fence-shaped tests,
checked against ``src/gen_worker``. NOT a general dead-symbol linter —
``lint_unreached_surface.py`` owns that question, and this one's whole value is
that it looks at **the guards**.

THE FALSE-POSITIVE SHAPE. Two kinds, handled differently on purpose:

1. A bare identifier in a string can be a MODULE name, a dataclass FIELD, or a
   path component — none of which are top-level defs. Filtered structurally, by
   collecting every name a module legitimately exposes rather than only its
   functions and classes.
2. A fence may name a dead symbol **ON PURPOSE** — a historical-evidence
   fixture, a collision proof, a negative case reconstructing a format the tree
   can no longer produce. These are NOT filtered automatically: they carry
   ``# fence-symbol-exempt: <reason>`` on the line. The reason is mandatory,
   because the whole point is that somebody looked; silent suppression is what
   lets the real ones through.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from typing import Dict, Set, Tuple

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src" / "gen_worker"

#: The guards. This script is IN this list on purpose — see the module
#: docstring. A file added here must be a fence: something whose job is to
#: refuse, naming the symbols it refuses about.
FENCES: Tuple[str, ...] = (
    "scripts/lint_fence_symbols.py",
    "scripts/lint_compiled_graph_key_layout_fence.py",
    "scripts/lint_serving_process_compiles.py",
    "scripts/lint_arm_state_feeders.py",
    "scripts/lint_unreached_surface.py",
    # pgw#1373: `tests/test_compiled_graph_key_pgw1059.py` was deleted with
    # the v1 keyset/declaration key space it fenced.
)

#: Only literals that LOOK like this repo's identity/arm vocabulary are
#: candidates. Without this every English word in a docstring is a symbol.
VOCABULARY = re.compile(
    r"compiled graph|entry|entries|graph|arm|mint|artifact|key|manifest|adopt|axis|axes")

#: `"name"` or `"name("` — the two shapes a fence names a symbol in.
LITERAL = re.compile(r'"([a-zA-Z_][a-zA-Z0-9_]{3,})\(?"')

#: A deliberate mention of a symbol that no longer exists. The REASON is
#: required: an exemption nobody had to justify is a suppression.
EXEMPT = re.compile(r"#\s*fence-symbol-exempt:\s*\S")

#: How far above a line the marker may sit. Small on purpose: an exemption
#: that can drift far from what it exempts stops describing it.
EXEMPT_LOOKBACK = 4


def live_names(src: pathlib.Path) -> Set[str]:
    """Every name a module in ``src`` legitimately exposes.

    Not just functions and classes: a fence may name a MODULE, a dataclass
    FIELD, an assigned constant or a method, and each of those is a real
    symbol whose disappearance matters. Under-collecting here is what
    produces the false positives this gate must not have.
    """
    names: Set[str] = set()
    for path in sorted(src.rglob("*.py")):
        names.add(path.stem)  # module names
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
    return names


def main() -> int:
    if not SRC.is_dir():
        print(f"lint_fence_symbols: no {SRC} — not a gen_worker checkout",
              file=sys.stderr)
        return 2
    live = live_names(SRC)
    dead: Dict[str, Set[str]] = {}
    scanned = 0
    for rel in FENCES:
        path = ROOT / rel
        if not path.exists():
            print(f"lint_fence_symbols: FENCE MISSING: {rel} — remove it from "
                  f"FENCES in the same commit that deletes it, or this gate is "
                  f"guarding a file that is not there", file=sys.stderr)
            return 1
        scanned += 1
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            # A deliberate mention of a dead symbol declares itself, with a
            # reason. `_old_schema_digest`'s whole job is to be different from
            # the current format — see the module docstring.
            #
            # The marker may sit on the line itself or in the comment block
            # immediately above it, because the reason is usually a sentence
            # and forcing it onto the code line would make it unwriteable —
            # and an exemption nobody can write is one nobody records.
            window = lines[max(0, index - EXEMPT_LOOKBACK):index + 1]
            if any(EXEMPT.search(row) for row in window):
                continue
            for match in LITERAL.finditer(line):
                name = match.group(1)
                if not VOCABULARY.search(name) or name in live:
                    continue
                dead.setdefault(rel, set()).add(name)

    if dead:
        print("lint_fence_symbols: A FENCE NAMES A SYMBOL THAT NO LONGER "
              "EXISTS.\n", file=sys.stderr)
        for rel in sorted(dead):
            for name in sorted(dead[rel]):
                print(f"  {rel}: {name!r}", file=sys.stderr)
        print(
            "\nThat fence still runs and still exits 0 — and guards NOTHING "
            "for this symbol.\nA green fence is not evidence after a rename. "
            "Either retarget it to the symbol\nthat replaced this one, or "
            "delete the entry in the same commit that deleted\nthe symbol. "
            "Suppressing it here is the one option that is never right.",
            file=sys.stderr)
        return 1

    print(f"lint_fence_symbols: {scanned} fence(s) scanned (this script "
          f"included); every named symbol exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
