"""The conversion-before-compute fence uses TCG's public identity boundary.

The historical worker key implementation is intentionally absent from these
tests.  TCG 0.4 owns graph-class declaration and the three-axis compiled-graph
identity; the worker fence only discovers callers and forbids layout demand or
conversion state from reaching them.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from torch_compiled_graphs import IdentityError
from torch_compiled_graphs.identity import from_axes

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "lint_cell_key_layout_fence.py"
SRC = REPO / "src" / "gen_worker"


def _fence_module():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_fence(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_tcg_is_the_three_axis_identity_authority() -> None:
    key = from_axes({"graph": "g", "sm": "sm_89", "toolchain": "t"})
    assert key.as_dict() == {"graph": "g", "sm": "sm_89", "toolchain": "t"}
    with pytest.raises(IdentityError, match="unknown identity axes"):
        from_axes({"graph": "g", "sm": "sm_89", "toolchain": "t", "layout": "forbidden"})


def test_fence_discovers_current_tcg_producer_callers() -> None:
    fenced = {path.name: why for path, why in _fence_module().fenced_modules(SRC).items()}
    assert {
        "aot_compile_child.py",
        "aot_mint.py",
        "boot_key.py",
        "boot_trace_child.py",
        "fleet_cells.py",
        "receipts.py",
    } <= set(fenced)
    assert "cell_key.py" not in fenced
    assert all("canonical compiled-graph producer" in why for why in fenced.values())


def test_entrypoint_red_proof_catches_layout_at_identity_boundary(tmp_path: Path) -> None:
    (tmp_path / "rogue.py").write_text(
        textwrap.dedent(
            """
            from torch_compiled_graphs.identity import from_axes
            from gen_worker.convert.layout_converters import LayoutId


            def rogue_identity(layout: LayoutId):
                return from_axes({
                    "graph": "g",
                    "sm": "sm_89",
                    "toolchain": layout.render(),
                })
            """
        ),
        encoding="utf-8",
    )
    result = _run_fence("--src", str(tmp_path))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "reads the layout vocabulary" in result.stdout
    assert "never widen or locally restate compiled-graph identity" in result.stdout


def test_layout_words_in_prose_do_not_trip_the_fence(tmp_path: Path) -> None:
    (tmp_path / "producer.py").write_text(
        textwrap.dedent(
            '''
            from torch_compiled_graphs.identity import from_axes


            def identity():
                """LayoutId and conversion_provenance cannot reach this boundary."""
                return from_axes({"graph": "g", "sm": "sm_89", "toolchain": "t"})
            '''
        ),
        encoding="utf-8",
    )
    result = _run_fence("--src", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_relation_literal_detector_has_its_own_red_proof(tmp_path: Path) -> None:
    fence = _fence_module()
    relation = tmp_path / "relation.py"
    bodies = []
    for name in fence.RELATION_FUNCTIONS:
        value = '"weights.safetensors@1"' if name == "classify_layout" else "None"
        bodies.append(f"def {name}():\n    return {value}\n")
    relation.write_text("\n".join(bodies), encoding="utf-8")
    literals, missing = fence._relation_handle_literals(relation)
    assert not missing
    assert literals == [(2, "classify_layout: 'weights.safetensors@1'")]


def test_shipped_tree_passes_conversion_before_compute_fence() -> None:
    result = _run_fence()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compiled graphs never key on layout" in result.stdout
