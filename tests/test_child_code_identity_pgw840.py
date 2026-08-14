"""The entry child must BE the parent's own gen_worker.

``python -m gen_worker.aot_compile_child`` does not mean "this gen_worker" —
same venv, same interpreter, and the child can still resolve to a DIFFERENT
gen_worker off ``sys.path``. The visible symptom is attribution (an entry table
with no child spans, the whole compile falling into ``reap_lag_s``) while the
compile itself succeeds and returns files that exist.

The defect is that the process which compiles the files a cell publishes was
chosen by ``sys.path``, while every gate runs in the parent against the
parent's program. Two assertions: the child RESOLVES to the parent's code, and
a child that did not is REFUSED by name rather than believed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from gen_worker import aot_compile_pool as pool

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _decoy_tree(root: Path) -> Path:
    """A second, importable ``gen_worker`` — the thing this box is full of."""
    pkg = root / "gen_worker"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("DECOY = True\n")
    (pkg / "aot_compile_child.py").write_text("raise SystemExit(0)\n")
    return root


# ---------------------------------------------------------------------------
# 1. Resolution: the child imports the parent's files, not the path's
# ---------------------------------------------------------------------------


def test_the_child_resolves_the_parents_own_gen_worker(tmp_path: Path) -> None:
    """Driven through the real ``child_env`` and a real interpreter.

    Both loopholes at once: a ``gen_worker`` inherited on ``PYTHONPATH`` and a
    ``gen_worker`` sitting in the cwd (which outranks ``PYTHONPATH`` under
    ``-m``). Before pgw#840 the child took whichever of those it found first.
    """
    decoy = _decoy_tree(tmp_path / "stale")
    base = dict(os.environ, PYTHONPATH=str(decoy))
    env = pool.child_env(str(tmp_path / "cache"), base=base)

    out = subprocess.run(
        [sys.executable, "-c",
         "import gen_worker, sys; print(gen_worker.__file__)"],
        env=env, cwd=str(decoy), capture_output=True, text=True, check=True)
    resolved = Path(out.stdout.strip()).resolve()

    assert resolved == Path(pool.PACKAGE_ROOT) / "gen_worker" / "__init__.py", (
        f"the entry child resolved gen_worker to {resolved} while the parent "
        f"runs from {pool.PACKAGE_ROOT}. That child compiles the loose files "
        f"the cell publishes, and every gate runs in the parent against the "
        f"parent's program — the assignment is only sound while both are the "
        f"same code (pgw#840)")


# ---------------------------------------------------------------------------
# 2. Proof, not assumption: a skewed child is refused BY NAME
# ---------------------------------------------------------------------------


_STALE_CHILD = '''#!{python}
"""A pre-pgw#840 entry child: compiles nothing, reports the OLD shape."""
import json, sys
from pathlib import Path

job = json.loads(Path(sys.argv[-1]).read_bytes())
loose = Path(job["report"]).parent / "fake.wrapper.cpp"
loose.write_text("// as far as the parent can tell, a compiled entry\\n")
Path(job["report"]).write_text(json.dumps({{
    "entry": job["entry"], "status": "compiled", "files": [str(loose)],
    "detail": "1 loose file(s)", "elapsed_s": 0.01, "peak_rss_bytes": 1,
    "phases": {{}},
}}))
sys.exit(0)
'''


def _program() -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(8, 8)

        def forward(self, x: Any) -> Any:
            return torch.relu(self.a(x))

    return torch.export.export(Tiny(), (torch.randn(2, 8),))


def test_a_child_from_other_code_is_refused_by_name(tmp_path: Path) -> None:
    """The real pool, a real spawn, a child that is not this gen_worker.

    Before pgw#840 this run SUCCEEDED: the pool took the files, recorded
    ``compile_s`` with no child spans inside it, absorbed the entire compile
    into ``reap_lag_s`` and published an entry compiled by code the parent
    never ran. The only visible trace was pgw#830's invariant going red — the
    symptom that got filed, one level below the defect.
    """
    stale = tmp_path / "stale-child.py"
    stale.write_text(_STALE_CHILD.format(python=sys.executable))
    stale.chmod(0o755)

    width = pool.entry_workers(
        1, limit=1, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    box = pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=str(stale))

    with pytest.raises(pool.EntryCompileFailed) as caught:
        box.compile([("unet/adapter=true/dim=0", _program())])

    assert caught.value.entry == "unet/adapter=true/dim=0"
    detail = str(caught.value)
    assert "DIFFERENT" in detail and "too old to report one" in detail, detail
    assert pool.CODE_DIGEST in detail, detail
    # And it never became an attribution puzzle: the entry has no table at all
    # rather than one whose members quietly do not add up.
    assert "unet/adapter=true/dim=0" not in box.entry_phases


def test_the_reports_identity_survives_the_wire(tmp_path: Path) -> None:
    """``code_digest``/``code_dir`` are the report's, not the parent's guess.

    A default-valued field is exactly what an old child produces, so the check
    must distinguish "reported nothing" from "reported a match" — the empty
    digest is a REFUSAL, never a pass.
    """
    encoded = msgspec.json.encode(pool.EntryReport(
        entry="unet/dim=0", status=pool.COMPILED, files=["/dev/null"],
        code_digest=pool.CODE_DIGEST, code_dir=pool.PACKAGE_ROOT))
    back = msgspec.json.decode(encoded, type=pool.EntryReport)
    assert back.code_digest == pool.CODE_DIGEST and back.code_digest
    assert back.code_dir == pool.PACKAGE_ROOT

    old = msgspec.json.decode(
        b'{"entry":"unet/dim=0","status":"compiled","files":["/dev/null"]}',
        type=pool.EntryReport)
    assert old.code_digest == "" and old.spans == {}
