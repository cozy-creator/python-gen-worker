"""Every module must import when it is the FIRST module imported."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import pytest

UNOWNED_DIRS = ("pb", "_vendor")


def is_unowned(path: Path, root: Path) -> bool:
    return any(name in path.relative_to(root).parts for name in UNOWNED_DIRS)


SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "gen_worker"

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
        if is_unowned(path, PKG):
            continue
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
    """Anything non-zero that is not a cycle and not an absent optional extra."""
    broken: List[Tuple[str, str]] = [
        (o.module, o.stderr.strip())
        for o in outcomes
        if o.returncode != 0 and not o.cycle and o.missing_extra is None
    ]
    if broken:
        detail = "\n\n".join(f"--- {m} ---\n{e}" for m, e in broken)
        pytest.fail(f"{len(broken)} module(s) fail to import on their own:\n\n{detail}")


def _cli_entry_orders() -> List[str]:
    mods = sorted(
        f"gen_worker.cli.{p.stem}"
        for p in (PKG / "cli").glob("*.py")
        if p.stem not in ("__init__", "__main__")
    )
    return ["gen_worker.cli", *mods]


@pytest.mark.parametrize("module", _cli_entry_orders())
def test_cli_entry_orders(module: str) -> None:
    """Every entry order into the cli package, named individually."""
    outcome = _import_alone(module)
    assert outcome.returncode == 0, f"{module} as a first import:\n{outcome.stderr}"
