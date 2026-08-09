"""pgw#1049: the second-writer fence, RED-proven.

`scripts/lint_settings_writers.py` is the CI gate; these tests prove it can
actually go red (a green gate that could never fail proves nothing) and that
its authority set cannot drift from the authority module's own declaration.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]


def _load_lint() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lint_settings_writers", REPO / "scripts" / "lint_settings_writers.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scan_tree(tmp_path: Path, source: str) -> dict:
    lint = _load_lint()
    (tmp_path / "offender.py").write_text(textwrap.dedent(source))
    return lint.scan(root=tmp_path)


def test_direct_backend_write_is_red(tmp_path: Path) -> None:
    sites = _scan_tree(tmp_path, """
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
    """)
    assert any("torch.backends.cuda.matmul.allow_tf32" in site
               for _, site in sites), sites


def test_aliased_config_write_is_red(tmp_path: Path) -> None:
    sites = _scan_tree(tmp_path, """
        import torch._inductor.config as inductor_config
        inductor_config.cpp.march = "native"
    """)
    assert any("torch._inductor.config.cpp.march" in site
               for _, site in sites), sites


def test_dynamo_config_write_and_setter_are_red(tmp_path: Path) -> None:
    sites = _scan_tree(tmp_path, """
        import torch
        import torch._dynamo
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch.set_float32_matmul_precision("highest")
    """)
    joined = {site for _, site in sites}
    assert "torch._dynamo.config.automatic_dynamic_shapes" in joined
    assert "torch.set_float32_matmul_precision" in joined


def test_watched_env_write_is_red_including_constants(tmp_path: Path) -> None:
    sites = _scan_tree(tmp_path, """
        import os
        _KNOB = "TORCHINDUCTOR_MAX_AUTOTUNE"
        os.environ["TRITON_PTXAS_PATH"] = "/tmp/ptxas"
        os.environ[_KNOB] = "1"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "native")
    """)
    joined = {site for _, site in sites}
    assert "os.environ[TRITON_PTXAS_PATH]" in joined
    assert "os.environ[TORCHINDUCTOR_MAX_AUTOTUNE]" in joined
    assert "os.environ.setdefault(PYTORCH_CUDA_ALLOC_CONF)" in joined


def test_unclassified_site_reports_and_classified_passes(tmp_path: Path) -> None:
    lint = _load_lint()
    sites = _scan_tree(tmp_path, """
        import torch
        torch.backends.cudnn.benchmark = True
    """)
    problems = lint.check(sites, {})
    assert problems and "UNCLASSIFIED" in problems[0]
    key = next(iter(sites))
    assert lint.check(sites, {key: "SCOPED"}) == []


def test_stale_allowlist_row_is_red() -> None:
    lint = _load_lint()
    problems = lint.check({}, {("src/gen_worker/gone.py", "torch.backends.x"): "SCOPED"})
    assert problems and "stale allowlist row" in problems[0]


def test_real_tree_is_green() -> None:
    """The shipped tree + shipped allowlist: zero problems — and this stays
    meaningful because the tests above prove the scanner can go red."""
    lint = _load_lint()
    problems = lint.load_allowlist()[1] + lint.check(
        lint.scan(), lint.load_allowlist()[0])
    assert problems == [], problems


def test_fence_authority_set_matches_the_module() -> None:
    from gen_worker import settings_authority as sa

    lint = _load_lint()
    assert set(lint.AUTHORITY_FILES) == set(sa.AUTHORITY_MODULES)
