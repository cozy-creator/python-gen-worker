#!/usr/bin/env python3
"""pgw#1327: THE SERVE-BOOT KEY PATH MAY NOT REACH A TRACER.

A serve pod used to learn its own ``cg-key-v1`` set by running ``torch.export``
in child processes at boot. pgw#1327 ships that key set as data, and the change
is only real if the code that reads it CANNOT reach the code that traces. A
runtime test proves one boot did not export; this proves no boot can.

The rule: starting from ``gen_worker.boot_adopt`` and the ``gen_worker.keyset``
package, the transitive STATIC import closure inside ``gen_worker`` must not
contain any banned module. Function-local imports count — they are the shape a
lazy tracer import would take, and an import-time-only fence would miss it.

``gen_worker.keyset.emit`` is a ROOT of the mint side, not of the serve side: it
consumes a tracer's output and is deliberately not imported by
``gen_worker.keyset.__init__``. It is excluded from the walk, and a serve module
that imports it is caught by the banned list below.

This is pgw#1327's NARROW precursor to pgw#1328's serve-role guard, which names
the whole role's module set (``mint_supervisor``, ``aot_mint``, ``aot_compile_*``,
``boot_trace_child``) and supersedes this file. Keeping it narrow is deliberate:
this one must be green the day pgw#1327 lands, on an executor that still hosts
both halves.

Run::

    python scripts/lint_serve_keyset_closure.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gen_worker"

#: Where the serve-boot key path starts.
ROOTS: Tuple[str, ...] = (
    "gen_worker.boot_adopt",
    "gen_worker.keyset",
    "gen_worker.keyset.boot",
    "gen_worker.keyset.closure",
    "gen_worker.keyset.document",
    "gen_worker.keyset.fold",
    "gen_worker.keyset.identifiers",
    "gen_worker.keyset.store",
)

#: The mint lane, as pgw#1327 leaves it. A serve-boot module that reaches any of
#: these is a pod that can still be made to trace at boot.
BANNED: Tuple[str, ...] = (
    "gen_worker.boot_key",
    "gen_worker.boot_trace_child",
    "gen_worker.keyset.emit",
)

#: Modules the walk does not enter. `executor` is BOTH hosts until pgw#1328
#: splits it, and nothing on the serve-boot key path imports it — but a stray
#: `TYPE_CHECKING` edge into it must not drag the whole tree in.
STOP: Tuple[str, ...] = ("gen_worker.executor",)


def _rel(name: str) -> str:
    return name[len("gen_worker"):].lstrip(".").replace(".", "/")


def _is_package(name: str) -> bool:
    return (SRC / _rel(name) / "__init__.py").is_file()


def _module_path(name: str) -> Path | None:
    if not name.startswith("gen_worker"):
        return None
    rel = _rel(name)
    for candidate in (SRC / rel / "__init__.py", SRC / f"{rel}.py"):
        if candidate.is_file():
            return candidate
    return None


def _package_of(module: str) -> str:
    """The package a relative import inside ``module`` is resolved against."""
    if _is_package(module):
        return module
    return module.rsplit(".", 1)[0]


def _imports(path: Path, module: str) -> Set[str]:
    """Every ``gen_worker`` module this file names, relative imports resolved."""
    package = _package_of(module)
    out: Set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gen_worker"):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                prefix = ".".join(parts)
                if node.module:
                    prefix = f"{prefix}.{node.module}"
            else:
                prefix = node.module or ""
            if not prefix.startswith("gen_worker"):
                continue
            out.add(prefix)
            for alias in node.names:
                out.add(f"{prefix}.{alias.name}")
    return out


def closure() -> Tuple[Set[str], Dict[str, str]]:
    """Every gen_worker module the roots reach, and who first reached it."""
    seen: Set[str] = set()
    via: Dict[str, str] = {}
    queue: List[str] = list(ROOTS)
    for root in ROOTS:
        via[root] = "<root>"
    while queue:
        name = queue.pop()
        if name in seen or name in STOP:
            continue
        path = _module_path(name)
        if path is None:
            continue
        seen.add(name)
        for target in sorted(_imports(path, name)):
            if target in seen or target in STOP:
                continue
            via.setdefault(target, name)
            queue.append(target)
    return seen, via


def main() -> int:
    seen, via = closure()
    if not seen:
        print("lint_serve_keyset_closure: the roots resolved to NOTHING — the "
              "module names rotted and this guard is guarding nothing")
        return 1
    bad = sorted(name for name in BANNED if name in seen)
    if bad:
        for name in bad:
            print(
                f"pgw#1327: the serve-boot key path reaches {name} "
                f"(first via {via.get(name, '?')}). The key set is DATA; the "
                f"path that reads it must not be able to trace.")
        return 1
    print(
        f"pgw#1327: serve-boot key closure is tracer-free "
        f"({len(seen)} gen_worker modules, {len(BANNED)} banned names absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
