"""pgw#506: fail-loud lazy-import stubs for heavy deps + hard-fail walker.

Covers both halves of the issue:

* ``stub_missing_heavy_deps`` — a missing allowlisted heavy root imports as
  a stub (module-top ``import torch`` is free during discovery), while any
  attribute TOUCH raises an actionable ``HeavyDepStubError``; installed
  roots are never stubbed; stubs are removed on exit.
* ``find_endpoints`` hard-fails on ANY other module import error (a broken
  submodule can no longer be silently skipped out of the endpoint.lock).
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from gen_worker.discovery.heavy_deps import (
    DEFAULT_HEAVY_ROOTS,
    HeavyDepStubError,
    _HeavyDepStub,
    stub_missing_heavy_deps,
)
from gen_worker.discovery.walk import EndpointImportError, find_endpoints

# Guaranteed-absent import root standing in for torch in a torch-less env —
# the stub path is identical (find_spec miss -> meta-path stub), only the name
# differs.
FAKE = "gw_fake_heavy_dep"


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _endpoint_src(heavy_import_lines: str = "") -> str:
    return heavy_import_lines + textwrap.dedent("""\
        import msgspec
        from gen_worker import RequestContext, endpoint

        class In_(msgspec.Struct):
            text: str = ""

        class Out_(msgspec.Struct):
            reply: str

        @endpoint
        class Gen:
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(reply=data.text)
    """)


# --------------------------------------------------------------------------- #
# the shim                                                                     #
# --------------------------------------------------------------------------- #


def _stmt_import(src: str) -> dict:
    """Run real ``import`` STATEMENTS (the surface the shim serves —
    ``importlib.import_module`` deliberately bypasses it)."""
    ns: dict = {}
    exec(src, ns)
    return ns


def test_installed_roots_are_never_stubbed() -> None:
    # msgspec is installed -> not in the stubbed set; the real module is used.
    with stub_missing_heavy_deps(extra=("msgspec", FAKE)) as stubbed:
        assert "msgspec" not in stubbed
        assert FAKE in stubbed
        import msgspec

        assert not isinstance(msgspec, _HeavyDepStub)
        assert callable(msgspec.json.encode)  # attribute use works: real module


def test_no_op_when_all_installed(monkeypatch) -> None:
    # With every allowlisted root "installed", the context changes nothing:
    # no __import__ wrapper, no meta-path finder.
    monkeypatch.setattr(
        "gen_worker.discovery.heavy_deps._root_installed", lambda root: True
    )
    import builtins

    before_import = builtins.__import__
    before_meta = list(sys.meta_path)
    with stub_missing_heavy_deps() as stubbed:
        assert stubbed == frozenset()
        assert builtins.__import__ is before_import
        assert sys.meta_path == before_meta


def test_missing_root_imports_as_stub_including_submodules() -> None:
    with stub_missing_heavy_deps(extra=(FAKE,)):
        ns = _stmt_import(
            f"import {FAKE}\n"
            f"import {FAKE}.nn.functional as F\n"
            f"from {FAKE} import nn\n"
        )
        assert isinstance(ns[FAKE], _HeavyDepStub)
        assert isinstance(ns["F"], _HeavyDepStub)
        assert isinstance(ns["nn"], _HeavyDepStub)


def test_find_spec_probes_stay_honest() -> None:
    # Third-party availability probes (transformers is_torchvision_available
    # is find_spec-only) must NOT see a missing dep as installed — a fooled
    # probe unlocks module-scope use of the dep in library code.
    with stub_missing_heavy_deps(extra=(FAKE,)):
        assert importlib.util.find_spec(FAKE) is None
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(FAKE)  # programmatic probe, not stubbed


def test_attribute_touch_raises_actionable_error() -> None:
    with stub_missing_heavy_deps(extra=(FAKE,)):
        mod = _stmt_import(f"import {FAKE}")[FAKE]
        with pytest.raises(HeavyDepStubError) as ei:
            _ = mod.bfloat16
        msg = str(ei.value)
        assert f"{FAKE}.bfloat16" in msg
        assert "not installed" in msg
        assert "setup()" in msg          # names the fix
        assert f"install {FAKE!r}" in msg


def test_defaulted_getattr_probes_degrade_gracefully() -> None:
    # getattr(mod, "__version__", "N/A") style probes (transformers' package
    # fallback) must get the default, not an explosion: HeavyDepStubError is
    # an AttributeError.
    with stub_missing_heavy_deps(extra=(FAKE,)):
        mod = _stmt_import(f"import {FAKE}")[FAKE]
        assert getattr(mod, "__version__", "N/A") == "N/A"
        assert not hasattr(mod, "__file__")


def test_stubs_removed_on_exit() -> None:
    import builtins

    before_import = builtins.__import__
    with stub_missing_heavy_deps(extra=(FAKE,)):
        _stmt_import(f"import {FAKE}.nn")
        assert FAKE in sys.modules
    assert builtins.__import__ is before_import
    assert FAKE not in sys.modules
    assert f"{FAKE}.nn" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        _stmt_import(f"import {FAKE}")


def test_non_heavy_import_errors_pass_through() -> None:
    with stub_missing_heavy_deps(extra=(FAKE,)):
        with pytest.raises(ModuleNotFoundError):
            _stmt_import("import definitely_not_a_real_package_xyz")


def test_torch_is_on_the_default_allowlist() -> None:
    assert "torch" in DEFAULT_HEAVY_ROOTS
    assert "torchvision" in DEFAULT_HEAVY_ROOTS


# --------------------------------------------------------------------------- #
# discovery integration: module-top heavy import in a dep-less env            #
# --------------------------------------------------------------------------- #


def test_discovery_with_top_level_heavy_import_in_bare_env(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_heavy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(_endpoint_src(
        f"import {FAKE}\nfrom {FAKE} import nn\n"
    ))

    fns = discover_functions(
        tmp_pkg, main_module="ep_heavy.main", extra_heavy_deps=(FAKE,)
    )
    assert [f["name"] for f in fns] == ["generate"]
    assert fns[0]["input_schema"]  # schemas still build against the stubs


def test_discovery_module_scope_use_of_missing_heavy_dep_fails_loud(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_heavy_use"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(_endpoint_src(
        f"import {FAKE}\nDTYPE = {FAKE}.bfloat16\n"
    ))

    with pytest.raises(ValueError) as ei:
        discover_functions(
            tmp_pkg, main_module="ep_heavy_use.main", extra_heavy_deps=(FAKE,)
        )
    # The chain surfaces the actionable stub error, not a bare AttributeError.
    assert f"{FAKE}.bfloat16" in str(ei.value)
    cause: BaseException | None = ei.value
    while cause is not None and not isinstance(cause, HeavyDepStubError):
        cause = cause.__cause__
    assert isinstance(cause, HeavyDepStubError)


# --------------------------------------------------------------------------- #
# hard-fail on submodule import errors (the pgw#506 sharpening)                #
# --------------------------------------------------------------------------- #


def test_broken_submodule_hard_fails_the_walk(tmp_pkg: Path) -> None:
    pkg = tmp_pkg / "ep_broken_sub"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(_endpoint_src())
    (pkg / "broken.py").write_text("import definitely_not_a_real_package_xyz\n")

    with pytest.raises(EndpointImportError) as ei:
        find_endpoints(["ep_broken_sub"])
    assert "ep_broken_sub.broken" in str(ei.value)
    assert isinstance(ei.value.__cause__, ModuleNotFoundError)


def test_syntax_error_in_submodule_hard_fails(tmp_pkg: Path) -> None:
    pkg = tmp_pkg / "ep_syntax"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(_endpoint_src())
    (pkg / "oops.py").write_text("def broken(:\n")

    with pytest.raises(EndpointImportError) as ei:
        find_endpoints(["ep_syntax"])
    assert "ep_syntax.oops" in str(ei.value)
    assert isinstance(ei.value.__cause__, SyntaxError)


def test_top_level_import_error_hard_fails(tmp_pkg: Path) -> None:
    pkg = tmp_pkg / "ep_top_broken"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('boom at import')\n")

    with pytest.raises(EndpointImportError) as ei:
        find_endpoints(["ep_top_broken"])
    assert "ep_top_broken" in str(ei.value)
    assert "boom at import" in str(ei.value)


def test_discover_functions_wraps_walk_failure_with_context(tmp_pkg: Path) -> None:
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_wrap"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(_endpoint_src())
    (pkg / "dead.py").write_text("import definitely_not_a_real_package_xyz\n")

    with pytest.raises(ValueError) as ei:
        discover_functions(tmp_pkg, main_module="ep_wrap.main")
    assert "ep_wrap" in str(ei.value)
    cause: BaseException | None = ei.value
    while cause is not None and not isinstance(cause, EndpointImportError):
        cause = cause.__cause__
    assert isinstance(cause, EndpointImportError)
