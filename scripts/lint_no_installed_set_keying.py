#!/usr/bin/env python3
"""THE INSTALLED SET IS NOT AN IDENTITY (pgw#1489).

The env half of an artifact key is the COMPILE STACK — torch, triton and the
`nvidia-*` libraries, READ OFF the endpoint's own `uv.lock` (Paul, 2026-08-19,
DESIGN-RULINGS addendum 4 as corrected, and the full-artifact-axis entry).

What this fence exists to stop coming back: keying, gating or auditing on an
enumeration of what is INSTALLED in the running process. That was a second
representation of an environment the lock already pins, and pgw#1472 measured
all three ways the two could never agree (PEP 503 spelling, the `+cu129` local
segment a lock cannot express, platform-conditional rows). It also split the
artifact pool on 43-package diffs between envs that serve identically — a
docs extra invalidated every compiled graph on the box.

Scope is THE KEY PATH (`KEY_PATH` below): the modules that give an artifact
its identity, position it, or admit it. There, `importlib.metadata` is banned
outright with ONE allowlisted exception —
`env_identity.installed_stack_drift`, which returns strings for a log line and
which nothing may gate on. Outside the key path, reading installed metadata is
ordinary and untouched (version reporting, the host inventory, rigcheck): those
are records about a machine, never identities of artifacts. The two dead
spellings `installed_closure`/`closure_hash` are banned repo-wide, because no
context makes either right again.

Run::

    python scripts/lint_no_installed_set_keying.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402
SRC = REPO / "src" / "gen_worker"
VENDOR = SRC / "_vendor"

#: Dead by this issue, anywhere in first-party code: both spelled an identity
#: over a package SET. Repo-wide because there is no context in which either
#: is the right call again.
BANNED = ("installed_closure", "closure_hash")

#: THE KEY PATH: the modules that decide an artifact's identity, position it,
#: or admit it. Inside these, `importlib.metadata` is banned outright — what a
#: process happens to have installed cannot enter a key, a position or a gate.
#: Elsewhere it is fine and common (version reporting, the host inventory,
#: rigcheck): those are records about a machine, not identities of artifacts.
KEY_PATH = (
    "env_identity.py",
    "release/derive.py",
    "serving/host.py",
    "serving/serve_adoption.py",
    "serving/hub_store.py",
    "serving/mint.py",
    "serving/self_mint.py",
    "serving/__main__.py",
    "local_compiled_graph_store.py",
)

#: (file, function) that may call `importlib.metadata` inside the key path.
#: One entry, and it returns strings for a log line.
DIAGNOSTIC = ("env_identity.py", "installed_stack_drift")


def _code_only(source: str, filename: str) -> str:
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                first.value.value = ""
    return ast.unparse(tree)


def _metadata_users(source: str, filename: str) -> list[str]:
    """Functions in this file that touch `importlib.metadata`."""

    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.unparse(node)
        if "importlib.metadata" in body or "importlib import metadata" in body:
            found.append(node.name)
    return found


def main() -> int:
    violations: list[str] = []
    scanned = 0
    for file in sorted(SRC.rglob("*.py")):
        if VENDOR in file.parents:
            continue
        scanned += 1
        source = file.read_text()
        code = _code_only(source, str(file))
        where = file.relative_to(REPO)
        for line_number, line in enumerate(code.splitlines(), start=1):
            for spelling in BANNED:
                if spelling in line:
                    violations.append(
                        f"{where}: names `{spelling}` in code. The env half of "
                        f"an artifact key is the compile stack read from the "
                        f"endpoint's uv.lock (`env_identity."
                        f"compile_stack_from_lockfile`); an enumeration of the "
                        f"installed set is a second representation of it and "
                        f"cannot key, gate or audit anything (pgw#1489). "
                        f"(code line {line_number}: {line.strip()})"
                    )
        relative = where.as_posix()
        if not any(relative.endswith(f"gen_worker/{name}") for name in KEY_PATH):
            continue
        for function in _metadata_users(source, str(file)):
            if (file.name, function) == DIAGNOSTIC:
                continue
            violations.append(
                f"{where}:{function}() reads `importlib.metadata`. The one "
                f"sanctioned reader is {DIAGNOSTIC[0]}:{DIAGNOSTIC[1]}(), which "
                f"is DIAGNOSTIC — it returns strings for a log line and nothing "
                f"gates on it. In the key path, an artifact's identity comes "
                f"from the endpoint's uv.lock and never from what happens to "
                f"be installed (pgw#1489)."
            )
    if not scanned:
        print("lint_no_installed_set_keying: scanned NOTHING — the tree moved",
              file=sys.stderr)
        return 1
    if violations:
        print(f"lint_no_installed_set_keying: {len(violations)} violation(s)",
              file=sys.stderr)
        _lint_side.report(violations, "pgw#1489 installed-set keying")
        return 1
    print(f"lint_no_installed_set_keying: clean ({scanned} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
