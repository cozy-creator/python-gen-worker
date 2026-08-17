#!/usr/bin/env python3
"""pgw#1316: no test may create a process by forking this interpreter.

Under `-n 4 --dist loadfile` an xdist worker reuses one process across files.
A worker that already ran a grpc.aio file still has gRPC's event-engine
threads live, and gRPC skips its `pthread_atfork` handlers whenever another
thread is inside gRPC (`fork_posix.cc:71`). The child then inherits a poller
gRPC never got to reset and dies on
`ev_epoll1_linux.cc: Check failed: next_worker->state == KICKED` — a SIGABRT
in a test whose subject is not fork semantics, red on master with no PR
responsible.

`multiprocessing.get_context("spawn")` is the only sanctioned way for a test
to reach a second process. Bare `multiprocessing.Process`/`Pool` count: their
default start method is `fork` on Linux.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (REPO / "tests", REPO / "tests_v2")

#: Attributes of `multiprocessing` (or a context) that start a process.
_STARTERS = {"Process", "Pool"}


def _mp_attr(node: ast.AST) -> str | None:
    """Name of a `multiprocessing.<attr>` access, else None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in {"multiprocessing", "mp"}:
            return node.attr
    return None


def _literal_method(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant):
        value = call.args[0].value
        return value if isinstance(value, str) else None
    for kw in call.keywords:
        if kw.arg == "method" and isinstance(kw.value, ast.Constant):
            return kw.value.value if isinstance(kw.value.value, str) else None
    return None


def check_file(path: Path) -> List[Tuple[int, str]]:
    problems: List[Tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr = _mp_attr(node.func)
        if attr in {"get_context", "set_start_method"}:
            method = _literal_method(node)
            if method != "spawn":
                problems.append(
                    (node.lineno, f'multiprocessing.{attr}({method!r})')
                )
        elif attr in _STARTERS:
            problems.append(
                (node.lineno, f"bare multiprocessing.{attr} (defaults to fork)")
            )
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "fork":
            if _mp_attr(node.func) is None and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "os":
                    problems.append((node.lineno, "os.fork()"))
    return problems


def scan(roots: Tuple[Path, ...]) -> List[str]:
    findings: List[str] = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            for lineno, why in check_file(path):
                rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
                findings.append(f"{rel}:{lineno}: {why}")
    return findings


def _selftest() -> int:
    """RED on every fork-shaped spawn; GREEN only on an explicit spawn context."""
    red = {
        "fork_ctx": 'import multiprocessing\nc = multiprocessing.get_context("fork")\n',
        "start_method": 'import multiprocessing\nmultiprocessing.set_start_method("fork")\n',
        "default_ctx": 'import multiprocessing\nc = multiprocessing.get_context()\n',
        "bare_process": 'import multiprocessing\np = multiprocessing.Process(target=f)\n',
        "os_fork": "import os\npid = os.fork()\n",
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, body in red.items():
            path = root / f"{name}.py"
            path.write_text(body)
            if len(scan((path,))) != 1:
                print(f"SELFTEST FAILED: {name} not caught", file=sys.stderr)
                return 1
        green = root / "green.py"
        green.write_text(
            'import multiprocessing\n'
            'c = multiprocessing.get_context("spawn")\n'
            'p = c.Process(target=f)\nq = c.Queue()\n'
        )
        if scan((green,)):
            print("SELFTEST FAILED: a spawn context was flagged", file=sys.stderr)
            return 1
    print("lint_test_fork_start_method selftest: red on fork, green on spawn")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(tuple(r for r in roots if r.exists()))
    if findings:
        print(
            "pgw#1316: a test forks this interpreter. gRPC's threads outlive the "
            "file that started them under `--dist loadfile`, its fork handlers are "
            "skipped while any thread is inside gRPC, and the child aborts on an "
            "inherited poller. Use `multiprocessing.get_context(\"spawn\")` and keep "
            "everything crossing the boundary picklable.\n",
            file=sys.stderr,
        )
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print("lint_test_fork_start_method: no test forks this interpreter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
