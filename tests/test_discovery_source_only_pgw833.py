"""pgw#833: the bake gate must run the RUNTIME's import predicate.

``discover_functions`` injects ``root`` and ``root/src`` into ``sys.path`` so
an uninstalled source tree can bake a lock. wan-2.2 1.6.0 showed the trap:
``src/cozy_finish.py`` was missing from the wheel's hatch ``only-include``,
the bake imported it from the source tree and PASSED, and the worker's own
walk (same ``find_endpoints``) then died at boot with ModuleNotFoundError on
every pod the release staffed — untyped, pre-Hello, fleet-wide the moment a
relock reaches >=0.87.0.

The gate: when the walked project is INSTALLED, every top-level module the
walk imported must resolve without the source tree. Dev trees (project not
installed anywhere else) keep working unchanged.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from gen_worker.discovery.discover import discover_functions

ENDPOINT_SRC = """
    import msgspec
    import {rootmod}  # the wan_2_2/finish.py shape: a package-ROOT sibling
    from gen_worker import RequestContext, endpoint

    class In_(msgspec.Struct):
        text: str = ""

    class Out_(msgspec.Struct):
        y: str

    @endpoint()
    class Gen:
        def generate(self, ctx: RequestContext, data: In_) -> Out_:
            return Out_(y={rootmod}.VALUE)
"""


def _write_pkg(base: Path, pkg_name: str, rootmod: str) -> None:
    pkg = base / pkg_name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        textwrap.dedent(ENDPOINT_SRC).format(rootmod=rootmod)
    )


def _cleanup_modules(*names: str) -> None:
    for name in list(sys.modules):
        if any(name == n or name.startswith(n + ".") for n in names):
            del sys.modules[name]


@pytest.fixture()
def clean_sys_path():
    before = list(sys.path)
    yield
    sys.path[:] = before


def test_installed_project_with_a_source_only_root_module_fails_the_bake(
    tmp_path, clean_sys_path,
):
    """The wan-2.2 1.6.0 shape exactly: the package IS installed (site dir),
    but the package-root sibling module it imports exists only in the source
    tree. The bake must REFUSE — on the unfixed tree it passed and shipped an
    image whose every pod died pre-Hello (RED evidence for this fix)."""
    pkg, rootmod = "pgw833_ep_a", "pgw833_root_a"
    site = tmp_path / "site"
    site.mkdir()
    _write_pkg(site, pkg, rootmod)          # installed package, imports rootmod
    root = tmp_path / "project"
    _write_pkg(root / "src", pkg, rootmod)  # source tree
    (root / "src" / f"{rootmod}.py").write_text("VALUE = 'ok'\n")  # SRC-ONLY

    sys.path.append(str(site))
    try:
        with pytest.raises(ValueError) as ei:
            discover_functions(root, main_module=f"{pkg}.main")
        msg = str(ei.value)
        assert rootmod in msg and "source tree" in msg, msg
    finally:
        _cleanup_modules(pkg, rootmod)


def test_installed_project_whose_imports_all_resolve_installed_passes(
    tmp_path, clean_sys_path,
):
    """Same shape with the packaging CORRECT (the root module is installed
    too, wan-2.2's ``cozy_rife`` precedent): the bake passes."""
    pkg, rootmod = "pgw833_ep_b", "pgw833_root_b"
    site = tmp_path / "site"
    site.mkdir()
    _write_pkg(site, pkg, rootmod)
    (site / f"{rootmod}.py").write_text("VALUE = 'ok'\n")   # installed
    root = tmp_path / "project"
    _write_pkg(root / "src", pkg, rootmod)
    (root / "src" / f"{rootmod}.py").write_text("VALUE = 'ok'\n")

    sys.path.append(str(site))
    try:
        fns = discover_functions(root, main_module=f"{pkg}.main")
        assert len(fns) == 1
    finally:
        _cleanup_modules(pkg, rootmod)


def test_uninstalled_dev_tree_keeps_baking(tmp_path, clean_sys_path):
    """No installed copy anywhere: the source tree IS the module set (every
    local `python -m gen_worker.discovery` run in a repo) — no refusal."""
    pkg, rootmod = "pgw833_ep_c", "pgw833_root_c"
    root = tmp_path / "project"
    _write_pkg(root / "src", pkg, rootmod)
    (root / "src" / f"{rootmod}.py").write_text("VALUE = 'ok'\n")

    try:
        fns = discover_functions(root, main_module=f"{pkg}.main")
        assert len(fns) == 1
    finally:
        _cleanup_modules(pkg, rootmod)
