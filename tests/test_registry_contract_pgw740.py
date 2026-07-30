"""pgw#740: the registry-contract gate discriminates artifact truth from suite truth.

Two arms, deliberately capped:
1. the contract holds from an INSTALLED wheel (the artifact we ship, not the
   source tree the suite normally imports);
2. the gate goes RED when a consumer's declarations silently fail to register —
   the exact failure the ordinary suite cannot see, because a test populates
   what it asserts on with its own imports.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_registry_contract.py"


@pytest.mark.integration
def test_contract_holds_from_installed_wheel(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    dist = tmp_path / "dist"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(dist)],
        cwd=REPO, check=True, capture_output=True,
    )
    wheel = next(dist.glob("gen_worker-*.whl"))
    venv = tmp_path / "venv"
    subprocess.run(
        [uv, "venv", "--python", "3.12", str(venv)], check=True, capture_output=True
    )
    py = venv / "bin" / "python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(py), str(wheel)],
        check=True, capture_output=True,
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [str(py), str(SCRIPT), "--installed"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "registry contract holds" in proc.stdout


def test_gate_reds_when_registrations_are_silently_dropped(tmp_path: Path) -> None:
    harness = (
        "import gen_worker.convert as c\n"
        "def _noop(*a, **k):\n"
        "    return None\n"
        "c.register_repackage_family = _noop\n"
        "c.register_layout = _noop\n"
        "c.declare_foreign_family_map = _noop\n"
        "import runpy, sys\n"
        "sys.argv = ['check_registry_contract.py']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(
        [sys.executable, "-c", harness],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "registry contract BROKEN" in out
    assert "repackage" in out
