#!/usr/bin/env python3
"""pgw#1013 / th#1662: a length read off external bytes may not size a read
until something has bounded it.

WHY THIS GUARD EXISTS. The §4.24 bounds census (pgw#973 / th#1620) enumerated
every numeric constant and adjudicated each against "name the runaway you
prevent". That procedure can only inspect bounds that EXIST, so it is blind by
construction to a read site that needs one and has none — and it walked past two
real ones in files it had already swept:

    convert/writer.component_stored_tensor_names   json.loads(fh.read(header_len))
    models/loading  (deshard path)                 json.loads(f.read(n))

Both took the attacker-controlled 8-byte safetensors prefix and allocated on it.
RED, measured: `OverflowError: byte string is too large` attempting a 2**63-byte
allocation. Fixed in pgw#973 wave 2 — this guard is what stops the class coming
back, because the census that missed them cannot be re-run to catch a new one.

THE RULE. Inside a function, if a name is bound from a binary length prefix —

    (n,) = struct.unpack("<Q", ...)      n = struct.unpack(...)[0]
    n = int.from_bytes(prefix, "little")

— and that same name is later used to size a read or an allocation —

    f.read(n)      os.read(fd, n)      bytearray(n)      b"\\x00" * n

— then the function must also bound it, in one of two ways:

  1. call a sanctioned validator on it, e.g. ``header_len_ok(n)``; or
  2. carry an explicit justification comment, on the read line or in the
     comment block just above it (see JUSTIFICATION_LOOKBACK):
         ``# bound-justified: <why this cannot run away>``

Two ways, deliberately: §4.24 asks for a bound OR a stated reason none is
needed, and a guard that only accepts the bound would push honest exemptions
into silence.

WHAT IT DOES NOT CATCH, stated so nobody reads a green run as proof of absence:
this matches the SHAPE the specimens had — a binary length prefix feeding a read
in the SAME function. A size that arrives via a return value, an attribute, a
JSON field, or a DB column is out of reach of a rule this cheap. Widening it
means dataflow analysis, and a noisy guard gets suppressed, which is worse than
a narrow one that holds. The broader enumeration is the issue's job; this is the
ratchet under it.

Usage: scripts/lint_unbounded_reads.py   (exit 1 on any finding)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "gen_worker"

# Calls whose result is a length taken from bytes the process did not compute.
LENGTH_SOURCES = {"unpack", "unpack_from", "from_bytes"}

# Names that, called on the length, count as bounding it. A project adds to this
# set when it grows another validator — it is deliberately explicit.
SANCTIONED_VALIDATORS = {"header_len_ok"}

# Call/expression shapes that turn a length into memory or IO.
SIZING_METHODS = {"read", "readinto", "recv", "pread"}

JUSTIFICATION = "bound-justified:"

# How far above the read line a justification comment may sit.
JUSTIFICATION_LOOKBACK = 8


def _iter_targets(node: ast.AST):
    """Names bound by an assignment target, including tuple unpacking."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _iter_targets(elt)


def _call_func_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _is_length_source(value: ast.AST) -> bool:
    """struct.unpack(...)  /  int.from_bytes(...)  possibly subscripted."""
    node = value
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Call) and _call_func_name(node) in LENGTH_SOURCES


class FunctionScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.tainted: set[str] = set()
        self.validated: set[str] = set()
        self.sized: list[tuple[str, int]] = []  # (name, lineno)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_length_source(node.value):
            for t in node.targets:
                self.tainted.update(_iter_targets(t))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_func_name(node)
        # A sanctioned validator applied to a tainted name bounds it.
        if name in SANCTIONED_VALIDATORS:
            for a in node.args:
                if isinstance(a, ast.Name):
                    self.validated.add(a.id)
        # A read/alloc sized by a tainted name is the site we care about.
        if name in SIZING_METHODS:
            for a in node.args:
                if isinstance(a, ast.Name):
                    self.sized.append((a.id, node.lineno))
        self.generic_visit(node)


def scan_file(path: Path, src: str) -> list[str]:
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc})"]

    lines = src.splitlines()
    findings: list[str] = []

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scan = FunctionScan()
        for stmt in fn.body:
            scan.visit(stmt)
        for name, lineno in scan.sized:
            if name not in scan.tainted:
                continue
            if name in scan.validated:
                continue
            # The justification may sit on the read line or in the comment
            # block immediately above it. A one-line-only rule would push a
            # real reason into a cramped trailing comment, which is how
            # justifications become noise nobody reads.
            start = max(0, lineno - 1 - JUSTIFICATION_LOOKBACK)
            window = lines[start:lineno]
            if any(JUSTIFICATION in ln for ln in window):
                continue
            rel = path.relative_to(SRC_ROOT.parent.parent)
            findings.append(
                f"{rel}:{lineno}: `{name}` is a length read off external bytes and sizes a "
                f"read in `{fn.name}()` with nothing bounding it.\n"
                f"    Fix: call a sanctioned validator ({', '.join(sorted(SANCTIONED_VALIDATORS))}) "
                f"on it, or state why none is needed with a `# {JUSTIFICATION} ...` comment."
            )
    return findings


def main() -> int:
    findings: list[str] = []
    for py in sorted(SRC_ROOT.rglob("*.py")):
        findings.extend(scan_file(py, py.read_text(encoding="utf-8")))

    if findings:
        print("unbounded-read guard: an external length sizes a read with no bound\n")
        for f in findings:
            print(f)
        print(
            "\npgw#1013 / th#1662: the bounds census could not see these — it enumerated "
            "bounds that exist, not sites that need one."
        )
        return 1

    print("unbounded-read guard: OK — every external length feeding a read is bounded or justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
