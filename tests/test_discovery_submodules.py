"""Regression tests for worker discovery across package layouts.

The original ``Worker._discover_and_register_functions`` rejected any class
whose ``__module__`` didn't exactly match the walked module. That silently
killed every @inference class defined in a submodule and re-exported through
``__init__.py`` — the conversion-endpoint outage of 2026-05-16.

Issue #12 consolidated build-time and runtime discovery into a single
walker, ``gen_worker.discovery.walk.find_endpoint_classes``. These tests
exercise that walker through the package layouts the worker has to handle:

  1. Class defined in walked module → returned.
  2. Class defined in submodule, re-exported via ``from .foo import *`` →
     returned (declaring ``__module__`` matches the walked-package prefix).
  3. Class defined in submodule, NOT re-exported → returned via the
     submodule walk (``pkgutil.walk_packages``).
  4. Class defined in unrelated package, imported into the walked
     namespace → SKIPPED with a diagnostic log (third-party leak).

All four behaviors are intrinsic to the walker. Runtime registration sits
on top of the same walker — same behavior, same tests, no separately
maintained copy.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest

from gen_worker.discovery.walk import find_endpoint_classes


@pytest.fixture
def tmp_pkg(tmp_path, monkeypatch) -> Iterator[Path]:
    """Make tmp_path importable as a Python package root for the test."""
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    # Clear any cached modules between tests so the next test gets a fresh
    # import state — pkgutil.walk_packages + importlib.import_module
    # would otherwise re-serve a stale module object.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("ep_") or mod_name.startswith("other_pkg"):
            sys.modules.pop(mod_name, None)


def _make_endpoint_class_source(name: str, function_name: str) -> str:
    """Source for a minimal @inference class exposing one function."""
    return textwrap.dedent(f"""
        from gen_worker.api.decorators import inference

        @inference()
        class {name}:
            def setup(self) -> None:
                pass

            @inference.function(name="{function_name}")
            def {function_name}(self, ctx, payload) -> dict:
                return {{}}

            def shutdown(self) -> None:
                pass
    """)


def test_class_in_walked_module_is_registered(tmp_pkg: Path):
    """Smoke: a class defined directly in the walked module is returned."""
    pkg = tmp_pkg / "ep_flat"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        _make_endpoint_class_source("FlatEndpoint", "do_thing")
    )

    found = find_endpoint_classes(["ep_flat"])
    qualnames = {f.qualname for f in found}
    assert "FlatEndpoint" in qualnames, f"flat class missing; got: {qualnames}"


def test_class_in_submodule_reexported_through_init(tmp_pkg: Path):
    """The conversion-endpoint pattern: class lives in a submodule,
    re-exported via __init__.py's ``from .main import *``."""
    pkg = tmp_pkg / "ep_reexport"
    pkg.mkdir()
    (pkg / "clone_huggingface.py").write_text(
        _make_endpoint_class_source("CloneHuggingFace", "clone_huggingface")
    )
    (pkg / "main.py").write_text(
        "from .clone_huggingface import CloneHuggingFace\n"
    )
    (pkg / "__init__.py").write_text("from .main import *  # noqa: F403\n")

    found = find_endpoint_classes(["ep_reexport"])
    qualnames = {f.qualname for f in found}
    assert "CloneHuggingFace" in qualnames, (
        f"submodule-defined class not returned; got: {qualnames}"
    )

    # Dedup invariant: even though CloneHuggingFace appears in three
    # namespaces (defining module, ``main``, and ``__init__``), the walker
    # must return it exactly once.
    count = sum(1 for f in found if f.qualname == "CloneHuggingFace")
    assert count == 1, f"re-export dedup broken; got {count} copies"


def test_class_in_submodule_not_reexported_fallback_walk(tmp_pkg: Path):
    """When __init__.py is empty (no re-export), the submodule walk
    still picks the class up."""
    pkg = tmp_pkg / "ep_walk"
    pkg.mkdir()
    (pkg / "synth.py").write_text(_make_endpoint_class_source("SynthEp", "synthesize"))
    # __init__.py is empty — no re-export at all.
    (pkg / "__init__.py").write_text("")

    found = find_endpoint_classes(["ep_walk"])
    qualnames = {f.qualname for f in found}
    assert "SynthEp" in qualnames, (
        f"submodule walk did not return class from un-reexported submodule; "
        f"got: {qualnames}"
    )


def test_class_outside_walked_package_is_skipped_with_diagnostic(
    tmp_pkg: Path, caplog
):
    """A class re-exported from an unrelated package must be rejected
    (it's a real third-party leak), and the skip should be logged."""
    other = tmp_pkg / "other_pkg"
    other.mkdir()
    (other / "__init__.py").write_text(
        _make_endpoint_class_source("StrayClass", "stray")
    )

    walked = tmp_pkg / "ep_stray"
    walked.mkdir()
    (walked / "__init__.py").write_text(
        "from other_pkg import StrayClass  # third-party re-export\n"
    )

    with caplog.at_level(logging.INFO, logger="gen_worker.discovery.walk"):
        found = find_endpoint_classes(["ep_stray"])

    qualnames = {f.qualname for f in found}
    assert "StrayClass" not in qualnames, (
        f"third-party class leaked into walked package; got: {qualnames}"
    )
    assert any(
        "Skipping endpoint class" in rec.message
        and "outside the walked package" in rec.message
        for rec in caplog.records
    ), (
        f"expected skip-with-reason log; got: "
        f"{[r.message for r in caplog.records]}"
    )
