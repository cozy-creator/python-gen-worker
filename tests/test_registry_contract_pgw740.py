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


# pgw#1310 deleted the `_vendored_requirements()` helper that stood here. It
# supplied the two deleted projects as explicit git requirements so this test
# could go green — which made the ONE test that installs the artifact stop
# measuring whether the artifact is installable. The wheel stayed unresolvable
# for every consumer (ie#738, te#221, e2e#1893) with this test passing. The
# install below is bare ON PURPOSE: it must resolve the wheel's own metadata
# against the real index, exactly as a consumer does.


# The three `uv` calls below build an isolated environment, which means
# PyPI — the one remaining third party in this suite. It is the SAME PyPI the
# job's own `uv sync --locked` already required, so this adds no new dependency;
# what it did add was a way for a transient 503 mid-run to turn a required check
# red on a fetch that has nothing to do with the change under test. A third
# party may degrade this row to a counted skip and may not fail the build. The
# match list is deliberately narrow: anything that is not recognisably a network
# failure still reds, because that is a real defect in the wheel.
_FETCH_FAILURES = (
    "Failed to fetch", "error sending request", "Could not connect",
    "Temporary failure in name resolution", "Network is unreachable",
    "operation timed out", "429 Too Many Requests", "503 Service Unavailable",
)


def _uv(argv: list[str], **kw: object) -> subprocess.CompletedProcess:
    proc = subprocess.run(argv, capture_output=True, **kw)  # type: ignore[call-overload]
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace")
        if any(marker in err for marker in _FETCH_FAILURES):
            pytest.skip(f"PyPI unreachable, so the wheel was never built: "
                        f"{err.strip().splitlines()[-1][:200]}")
        raise AssertionError(f"{argv[1]} failed ({proc.returncode}):\n{err}")
    return proc


@pytest.mark.integration
def test_contract_holds_from_installed_wheel(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    dist = tmp_path / "dist"
    _uv([uv, "build", "--wheel", "--out-dir", str(dist)], cwd=REPO)
    wheel = next(dist.glob("gen_worker-*.whl"))
    venv = tmp_path / "venv"
    _uv([uv, "venv", "--python", "3.12", str(venv)])
    py = venv / "bin" / "python"
    _uv([uv, "pip", "install", "--python", str(py), str(wheel)])
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
