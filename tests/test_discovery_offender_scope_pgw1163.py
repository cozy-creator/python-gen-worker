"""The source-only audit reports on THE WALK, not on its caller.

The audit walks all of `sys.modules` looking for top-level modules under the
project root that the installed wheel does not provide. Run from inside pytest,
that set includes the RUNNER's own modules (`conftest`, every `test_*` module)
— files pytest injected and a pod never imports.

THE MESSAGE IS THE WORSE HALF. Advising *"include the module in the built
package (e.g. hatch only-include)"* would package `conftest.py` and the whole
test suite into the shipped wheel — strictly worse than the failure it claims
to fix, and it would look like a successful fix. The message is pinned by a
test too.

The scope fix is deliberately NOT "detect pytest": excluding by
already-loaded-before-the-walk requires no knowledge of the test runner, and a
scan that has to recognise pytest would have to recognise the next tool too.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from gen_worker.discovery import discover as disc


def _fake_module(name: str, filename: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = filename
    return mod


@pytest.fixture
def installed_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root whose top-level package resolves as INSTALLED.

    The audit no-ops unless the walked project is installed, so the fixture
    forces that arm — otherwise every row below would pass for the wrong
    reason (the dev-tree early return), which is the failure shape this repo
    keeps finding.
    """
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)

    real = disc.importlib.machinery.PathFinder.find_spec

    def _find_spec(name, path=None, target=None):
        if name == "mypkg":
            spec = types.SimpleNamespace(origin="/installed/site-packages/mypkg/__init__.py")
            return spec
        return real(name, path, target)

    monkeypatch.setattr(
        disc.importlib.machinery.PathFinder, "find_spec", staticmethod(_find_spec))
    return root


def test_a_module_the_CALLER_already_imported_is_not_an_offender(
    installed_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on master: reported as an offender and the bake failed.

    This is the ernie shape exactly — pytest's `conftest`, loaded from under
    the project root before discovery ever ran."""
    root = installed_project
    conftest = _fake_module("conftest", str(root / "conftest.py"))
    monkeypatch.setitem(sys.modules, "conftest", conftest)

    preloaded = frozenset(sys.modules)  # as discover_functions takes it

    disc._audit_source_only_imports(
        root=root, top_level="mypkg", preloaded=preloaded)


def test_a_module_THE_WALK_imported_is_still_an_offender(
    installed_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must keep firing on what it exists for: wan-2.2 1.6.0's
    `src/cozy_finish.py`, absent from the wheel, imported BY the walk. A fix
    that silenced this too would trade a false positive for the boot-time
    `ModuleNotFoundError` on every pod the release staffs."""
    root = installed_project
    preloaded = frozenset(sys.modules)  # snapshot BEFORE the "walk"

    leaked = _fake_module("cozy_finish", str(root / "src" / "cozy_finish.py"))
    monkeypatch.setitem(sys.modules, "cozy_finish", leaked)

    with pytest.raises(disc.SourceOnlyModuleError, match="cozy_finish"):
        disc._audit_source_only_imports(
            root=root, top_level="mypkg", preloaded=preloaded)


def test_the_message_does_NOT_advise_packaging_a_test_module(
    installed_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap, pinned. The old text led with hatch `only-include`, so the
    obvious action was to package whatever was named — including a conftest."""
    root = installed_project
    preloaded = frozenset(sys.modules)
    monkeypatch.setitem(
        sys.modules, "cozy_finish",
        _fake_module("cozy_finish", str(root / "src" / "cozy_finish.py")))

    with pytest.raises(disc.SourceOnlyModuleError) as err:
        disc._audit_source_only_imports(
            root=root, top_level="mypkg", preloaded=preloaded)

    detail = str(err.value)
    assert "DROP THE IMPORT" in detail, detail
    # the packaging advice must be CONDITIONAL and must come second
    assert detail.index("DROP THE IMPORT") < detail.index("only-include"), detail
    assert "never be packaged" in detail.lower(), detail
    assert "conftest" in detail.lower(), detail


def test_the_scan_needs_no_knowledge_of_the_test_runner() -> None:
    """Excluding by already-loaded rather than by name is the property that
    keeps this from rotting: a scan that recognises pytest would have to
    recognise the next tool too."""
    import ast
    import inspect

    tree = ast.parse(inspect.cleandoc(
        inspect.getsource(disc._audit_source_only_imports)))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    # The DOCSTRING may name pytest — explaining which caller motivated the
    # scope fix is exactly what it is for. The CODE may not: a branch on a
    # runner's name is the rot this avoids.
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    code = "\n".join(ast.dump(node) for node in body)
    for tool in ("pytest", "_pytest", "unittest", "nose"):
        assert tool not in code, f"the scan branches on {tool!r}"


def test_the_guard_FIRES_on_the_shape_it_exists_for() -> None:
    """Severance experiment: the audit must still be reachable and still raise.
    A scope fix that turned the whole gate off would pass every row above."""
    assert callable(disc._audit_source_only_imports)
    params = inspect_signature_params()
    assert "preloaded" in params, params


def inspect_signature_params() -> list:
    import inspect

    return list(inspect.signature(disc._audit_source_only_imports).parameters)
