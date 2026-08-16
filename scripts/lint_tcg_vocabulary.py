#!/usr/bin/env python3
"""pgw#1299: TCG owns the compiled-graph metadata vocabulary; nobody respells it.

`torch_compiled_graphs` exports `GRAPH_CLASS_BLOCK` and `ARTIFACT_KIND` for one
reason: a consumer that spells the string instead of importing it does not fail
at import when TCG renames the key. It fails at run time, silently, and in the
worst possible shape — every one of these sites is a `.get(...)` on an
artifact-metadata dict, so a rename returns `None`, the arming path reads an
un-armed cell as a cache miss, and the fleet re-mints. That is money, not a
warning.

This has now been filed three times (pgw#1288 deleted four sites, pgw#1299
found fourteen more). Remembering failed; this is the fence.

WHAT IT LOOKS FOR is an AST string CONSTANT equal to `graph_class` or
`aot-inductor` anywhere under `src/gen_worker/`. AST rather than grep, so prose
in a docstring or a comment is not a finding — the vocabulary is only at risk
where it is a value the interpreter compares against.

THERE IS NO PATH ALLOWLIST, deliberately, for the reason
`lint_repo_ref_pins.py` states: a filename lets a real respelling hide. The one
legitimate use is recognised by a PROOF AT THE LINE instead — a
`# tcg-vocab: <reason>` comment. Delete the reason and the line goes red.

The legitimate use is real and worth naming, because conflating it with the
metadata block would be its own bug. `GraphClassDeclaration.graph_class` is a
DATACLASS FIELD holding a class NAME; `GRAPH_CLASS_BLOCK` is the
ARTIFACT-METADATA BLOCK key. They are equal strings from different vocabularies.
`boot_key.serialize_declaration` writes the field name into pgw's own boot-memo
wire format, and a TCG field rename there is an `AttributeError` on
`declaration.graph_class` — LOUD, which is precisely the failure mode this lint
exists to prevent, already handled by Python. Substituting `GRAPH_CLASS_BLOCK`
there would invent a coupling that does not exist: a rename of the metadata
block would silently change pgw's on-disk boot memo format.

Usage:

    python scripts/lint_tcg_vocabulary.py [PATH ...]
    python scripts/lint_tcg_vocabulary.py --selftest

Defaults to `src/gen_worker/`. Generated protobuf (`pb/`) is skipped: it is
owned by the `.proto`, not by a human, and regenerating it is the only way to
change it.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "src" / "gen_worker"

#: value -> the authority a consumer must import from `torch_compiled_graphs`.
OWNED: dict[str, str] = {
    "graph_class": "GRAPH_CLASS_BLOCK",
    "aot-inductor": "ARTIFACT_KIND",
}

#: The marker that turns a finding into a classified exemption. It must carry a
#: reason after the colon — a bare marker is not a proof.
MARKER = "# tcg-vocab:"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every Constant that is a module/class/function docstring."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def scan_text(text: str, where: str) -> List[Tuple[int, str, str]]:
    """(line, value, authority) for every unexempted respelling in `text`."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise SystemExit(f"{where}: cannot parse: {exc}") from exc
    lines = text.splitlines()
    skip = _docstring_nodes(tree)
    found: List[Tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        if not isinstance(node.value, str) or node.value not in OWNED:
            continue
        line = node.lineno
        source = lines[line - 1] if 0 < line <= len(lines) else ""
        marker = source.find(MARKER)
        if marker != -1 and source[marker + len(MARKER):].strip():
            continue
        found.append((line, node.value, OWNED[node.value]))
    return sorted(set(found))


def _python_files(roots: Iterator[Path]) -> Iterator[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for path in sorted(root.rglob("*.py")):
            # Generated protobuf is owned by the .proto, not by a human.
            if "pb" in path.relative_to(root).parts:
                continue
            yield path


def selftest() -> int:
    """A sweep that can only ever print 'clean' guards nothing."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.py"
        bad.write_text('x = meta.get("graph_class")\ny = dict(kind="aot-inductor")\n')
        hits = scan_text(bad.read_text(), str(bad))
        if len(hits) != 2:
            print(f"SELFTEST FAIL: expected 2 findings, got {hits}")
            ok = False

        exempt = Path(tmp) / "exempt.py"
        exempt.write_text('x = {"graph_class": d.graph_class}  # tcg-vocab: declaration field\n')
        if scan_text(exempt.read_text(), str(exempt)):
            print("SELFTEST FAIL: a classified line must not be a finding")
            ok = False

        bare = Path(tmp) / "bare.py"
        bare.write_text('x = meta.get("graph_class")  # tcg-vocab:\n')
        if not scan_text(bare.read_text(), str(bare)):
            print("SELFTEST FAIL: a marker with no reason is not a proof")
            ok = False

        prose = Path(tmp) / "prose.py"
        prose.write_text('"""Reads the graph_class block of an aot-inductor artifact."""\n')
        if scan_text(prose.read_text(), str(prose)):
            print("SELFTEST FAIL: prose in a docstring is not a respelling")
            ok = False

    print("selftest: OK" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    roots = [Path(a).resolve() for a in argv] or [DEFAULT_ROOT]
    findings: List[str] = []
    for path in _python_files(iter(roots)):
        for line, value, authority in scan_text(path.read_text(), str(path)):
            findings.append(
                f"{path.relative_to(REPO)}:{line}: {value!r} is TCG's vocabulary — "
                f"import {authority} from torch_compiled_graphs"
            )
    if findings:
        print("pgw#1299: compiled-graph vocabulary respelled as a literal\n")
        for row in findings:
            print(f"  {row}")
        print(
            f"\n{len(findings)} finding(s). A TCG rename does not raise on these — "
            "it returns None and the arm reads as a cache miss.\n"
            f"If a site is genuinely a DIFFERENT vocabulary, say so at the line "
            f"with `{MARKER} <reason>`."
        )
        return 1
    print("pgw#1299: clean — no compiled-graph vocabulary is respelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
