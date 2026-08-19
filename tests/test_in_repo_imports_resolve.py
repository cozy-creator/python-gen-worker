"""Every in-repo import in ``src/`` names a module that EXISTS.

pgw#1373 deleted 466 files. Three imports of deleted modules survived it, each
on a live production path, and each was invisible because nothing imports the
whole package eagerly:

* ``cli/release.py`` -> ``cli.run`` (deleted): ``gen-worker release derive``
  raised ``ModuleNotFoundError`` on every invocation — and that command is the
  only route that puts compiled-graph program blobs where a pod can reach them.
* ``procsplit/measure.py`` -> ``.models.hub_policy``: ``probe_hardware`` moved
  from ``gen_worker/lifecycle.py`` (where one dot reached ``gen_worker.models``)
  into ``gen_worker/procsplit/`` (where it reaches
  ``gen_worker.procsplit.models``, which does not exist) and the ``except
  Exception: pass`` around it ate the error. Every worker reported an empty
  ``gpu_sm``/``torch_version``/``cuda_version`` and then refused every request
  carrying a pgw#984-derived ``min_sm`` with ``gpu_capability_incompatible``
  (pgw#1417/#1436 — diagnosed as a bad CUDA runtime for a whole wave).
* ``serve/__main__.py`` -> ``worker_main`` (renamed back to ``entrypoint``):
  ``python -m gen_worker.serve``, the adopt-only worker entry, died at step 3
  of its own four-statement contract.

A grep cannot find these — two of the three are RELATIVE imports whose meaning
depends on the importing module's package, which is exactly what went wrong
both times a function moved. So this resolves them the way Python does.

``find_spec`` rather than an import: a module that needs a GPU, a checkpoint or
an optional heavy dep must not have to be importable here for its NAME to be
checked. The question is whether the module exists, not whether this host can
run it.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import List, Tuple

SRC = Path(__file__).resolve().parents[1] / "src"


def _module_and_package(path: Path) -> Tuple[str, str]:
    """``(module name, package the relative imports are relative TO)``."""
    rel = path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
    if rel.endswith(".__init__"):
        return rel[:-9], rel[:-9]
    return rel, rel.rsplit(".", 1)[0] if "." in rel else ""


def _targets(path: Path) -> List[Tuple[int, str]]:
    """Every ``gen_worker*`` module name this file imports, absolutised."""
    _module, package = _module_and_package(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:  # not ours to police here
        return []
    out: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0] if "." in base else ""
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            out.append((node.lineno, target))
        elif isinstance(node, ast.Import):
            out.extend((node.lineno, alias.name) for alias in node.names)
    return [(lineno, t) for lineno, t in out if t.startswith("gen_worker")]


def test_every_in_repo_import_names_a_module_that_exists() -> None:
    broken: List[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts:
            # Vendored trees carry their upstream's own import graph and are
            # fenced by their recorded digests (pgw#1310), not by this.
            continue
        for lineno, target in _targets(path):
            try:
                found = importlib.util.find_spec(target) is not None
            except (ImportError, AttributeError, ValueError):
                # The PARENT package failed to resolve — the same defect, one
                # level up, and `find_spec` reports it by raising.
                found = False
            if not found:
                broken.append(
                    f"{path.relative_to(SRC.parent)}:{lineno} imports "
                    f"{target!r}, which does not exist"
                )
    assert not broken, (
        "in-repo imports naming modules that do not exist — each one is a "
        "production path that raises ModuleNotFoundError the first time it is "
        "taken, or, worse, one swallowed by an `except Exception` into a "
        "silently empty answer:\n  " + "\n  ".join(broken)
    )
