"""pgw#981 — every module must import when it is the FIRST module imported.

An import cycle is invisible to any harness that imports the package once and
then walks it in a single process: the first entry order pre-warms the package,
and every later import is a cache hit. pgw#976's first sweep did exactly that,
passed 0/230, and hid two real cycles. Only one fresh interpreter PER MODULE
sees them, so that is what this does.

The cycle this file was written for::

    cli/serve.py:56   from . import run as run_mod
    cli/run.py:38     from .serve import DEFAULT_SOCKET_PATH

`serve` legitimately depends on `run` — it reuses its dispatch, its exit codes
and its errors — so the back-edge was the defect. `import gen_worker.cli.serve`
as a first import raised ``ImportError: cannot import name
'DEFAULT_SOCKET_PATH' from partially initialized module``, and it was masked
only because ``cli/__init__.py`` happens to reach `run` before `serve`. Fixed by
moving the constant DOWN to `cli/transport.py`, a leaf, not by deferring the
import: a deferred import keeps the cycle and hides it again.

A module that cannot import because an OPTIONAL EXTRA is absent is not a cycle
and is reported as such — 429 of this package's function-body imports exist
precisely so `import gen_worker` works without torch (pgw#976).
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "gen_worker"

# Substrings that identify a partially-initialized-module failure. CPython
# phrases it two ways depending on whether the name or the module was the
# unresolved half.
CYCLE_MARKERS = ("partially initialized module", "circular import")


class Outcome(NamedTuple):
    module: str
    returncode: int
    stderr: str

    @property
    def cycle(self) -> bool:
        return any(m in self.stderr for m in CYCLE_MARKERS)

    @property
    def missing_extra(self) -> Optional[str]:
        """The absent third-party module, when that is the whole story."""
        marker = "ModuleNotFoundError: No module named '"
        if self.returncode == 0 or marker not in self.stderr or self.cycle:
            return None
        name = self.stderr.rsplit(marker, 1)[1].split("'", 1)[0]
        return None if name.split(".")[0] == "gen_worker" else name


def _modules() -> List[str]:
    out: List[str] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(SRC)
        parts = list(rel.parts)
        if any(p.startswith((".", "_pycache")) or p == "__pycache__" for p in parts):
            continue
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        if not parts:
            continue
        out.append(".".join(parts))
    return out


def _import_alone(module: str) -> Outcome:
    """One fresh interpreter, this module as the very first import."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=str(SRC),
        capture_output=True,
        text=True,
    )
    return Outcome(module, proc.returncode, proc.stderr)


def _walk() -> List[Outcome]:
    modules = _modules()
    assert len(modules) > 100, f"module discovery found only {len(modules)} — check SRC"
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_import_alone, modules))


@pytest.fixture(scope="module")
def outcomes() -> List[Outcome]:
    return _walk()


def test_no_module_is_unimportable_as_a_first_import(outcomes: List[Outcome]) -> None:
    """The whole-package form: no cycle, from any entry order."""
    cycles = [o for o in outcomes if o.cycle]
    if cycles:
        detail = "\n\n".join(f"--- {o.module} ---\n{o.stderr.strip()}" for o in cycles)
        pytest.fail(
            f"{len(cycles)} module(s) close an import cycle when imported first:\n\n{detail}\n\n"
            "Break the cycle by moving the shared name DOWN to a leaf module. A deferred "
            "import inside a function keeps the cycle and only hides it again."
        )


def test_no_module_fails_to_import_for_any_other_reason(outcomes: List[Outcome]) -> None:
    """Anything non-zero that is not a cycle and not an absent optional extra.

    Kept separate from the cycle assertion so a missing `[dev]` extra in some
    future environment reads as what it is instead of as a cycle regression.
    """
    broken: List[Tuple[str, str]] = [
        (o.module, o.stderr.strip())
        for o in outcomes
        if o.returncode != 0 and not o.cycle and o.missing_extra is None
    ]
    if broken:
        detail = "\n\n".join(f"--- {m} ---\n{e}" for m, e in broken)
        pytest.fail(f"{len(broken)} module(s) fail to import on their own:\n\n{detail}")


@pytest.mark.parametrize(
    "module",
    ["gen_worker.cli.serve", "gen_worker.cli.run", "gen_worker.cli.invoke", "gen_worker.cli"],
)
def test_cli_entry_orders(module: str) -> None:
    """The four entry orders into the cli package, named individually.

    `cli/__init__` reaching `run` before `serve` is what masked the original
    defect, so "the CLI works" was never evidence about `cli.serve`.
    """
    outcome = _import_alone(module)
    assert outcome.returncode == 0, f"{module} as a first import:\n{outcome.stderr}"
