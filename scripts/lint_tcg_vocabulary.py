#!/usr/bin/env python3
"""pgw#1299: TCG owns the compiled-graph metadata vocabulary; nobody respells it.

`torchcg` exports `GRAPH_SPECIALIZATION_BLOCK` and `ARTIFACT_KIND` for one
reason: a consumer that spells the string instead of importing it does not fail
at import when TCG renames the key. It fails at run time, silently, and in the
worst possible shape — every one of these sites is a `.get(...)` on an
artifact-metadata dict, so a rename returns `None`, the arming path reads an
un-armed compiled graph as a cache miss, and the fleet re-mints. That is money, not a
warning.

This has now been filed three times (pgw#1288 deleted four sites, pgw#1299
found fourteen more). Remembering failed; this is the fence.

WHAT IT LOOKS FOR is an AST string CONSTANT equal to `graph_specialization` or
`aot-inductor` anywhere under `src/gen_worker/`. AST rather than grep, so prose
in a docstring or a comment is not a finding — the vocabulary is only at risk
where it is a value the interpreter compares against.

THERE IS NO PATH ALLOWLIST, deliberately, for the reason
`lint_repo_ref_pins.py` states: a filename lets a real respelling hide. A
legitimate use is recognised by a PROOF AT THE LINE instead — a
`# tcg-vocab: <reason>` comment. Delete the reason and the line goes red.

CORRECTED at tcg#56 (2026-08-19), because the paragraph that stood here had gone
stale in BOTH of its claims and a reader would have planned against it.

It said the exemption existed because the declaration's display-name FIELD and
the artifact-metadata BLOCK key were "equal strings from different vocabularies"
— they were spelled `GraphClassDeclaration.graph_class` and `GRAPH_CLASS_BLOCK`.  # graph-class-spelling: quoting the dead spellings is the whole point of the correction
That collision is GONE: tcg#56 renamed the field to `.name` and the block to
`graph_specialization`, so they are now two different words and cannot be
confused for one another. The ambiguity this paragraph warned about was itself
an argument for the rename.

It also cited `boot_key.serialize_declaration` as the one caller that must spell
the field. **That module and that function no longer exist in this tree** — grep
before believing it. There are currently ZERO `# tcg-vocab:` markers under
`src/gen_worker/`, so the exemption has no users at all. It is kept because the
mechanism is right and a future caller may earn one, not because anything needs
it today.

What the fence is still for is unchanged and is the whole point: every site is a
`.get(...)` on an artifact-metadata dict, so a respelling returns `None`, the
arming path reads an un-armed compiled graph as a cache miss, and the fleet
re-mints. That failure is SILENT. Import the constant; never spell the string.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_scope import is_unowned  # noqa: E402
import _lint_side  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "src" / "gen_worker"

#: value -> the authority a consumer must import from `torchcg`.
OWNED: dict[str, str] = {
    "graph_specialization": "GRAPH_SPECIALIZATION_BLOCK",
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
            # Generated and vendored subtrees are not ours to respell — and the
            # vendored TCG IS the authority this guard measures against.
            if is_unowned(path, root):
                continue
            yield path


def selftest() -> int:
    """A sweep that can only ever print 'clean' guards nothing."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.py"
        bad.write_text('x = meta.get("graph_specialization")\ny = dict(kind="aot-inductor")\n')
        hits = scan_text(bad.read_text(), str(bad))
        if len(hits) != 2:
            print(f"SELFTEST FAIL: expected 2 findings, got {hits}")
            ok = False

        exempt = Path(tmp) / "exempt.py"
        exempt.write_text(
            'x = {"graph_specialization": 1}  # tcg-vocab: literal key in a test fixture\n'
        )
        if scan_text(exempt.read_text(), str(exempt)):
            print("SELFTEST FAIL: a classified line must not be a finding")
            ok = False

        bare = Path(tmp) / "bare.py"
        bare.write_text('x = meta.get("graph_specialization")  # tcg-vocab:\n')
        if not scan_text(bare.read_text(), str(bare)):
            print("SELFTEST FAIL: a marker with no reason is not a proof")
            ok = False

        prose = Path(tmp) / "prose.py"
        prose.write_text('"""Reads the graph_specialization block of an aot-inductor artifact."""\n')
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
                f"import {authority} from torchcg"
            )
    if findings:
        print("pgw#1299: compiled-graph vocabulary respelled as a literal\n")
        _lint_side.report(findings, "pgw#1299 TCG metadata vocabulary",
                          stream=sys.stdout)
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
