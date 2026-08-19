"""The entry child must BE the parent's own gen_worker.

``python -m gen_worker.aot_compile_child`` does not mean "this gen_worker" —
same venv, same interpreter, and the child can still resolve to a DIFFERENT
gen_worker off ``sys.path``. The visible symptom is attribution (an entry table
with no child spans, the whole compile falling into ``reap_lag_s``) while the
compile itself succeeds and returns files that exist.

The defect is that the process which compiles the files a compiled graph publishes was
chosen by ``sys.path``, while every gate runs in the parent against the
parent's program. Two assertions: the child RESOLVES to the parent's code, and
a child that did not is REFUSED by name rather than believed.
"""

from __future__ import annotations

import importlib.util

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
        f"the compiled graph publishes, and every gate runs in the parent against the "
        f"parent's program — the assignment is only sound while both are the "
        f"same code (pgw#840)")


# ---------------------------------------------------------------------------
# 2. Proof, not assumption: a skewed child is refused BY NAME
# ---------------------------------------------------------------------------


_STALE_CHILD = '''#!{python}
"""A pre-pgw#840 compile child: compiles nothing, reports the OLD shape."""
import json, sys
from pathlib import Path

job = json.loads(Path(sys.argv[-1]).read_bytes())
artifact = Path(job["report"]).parent / "fake.tar.gz"
artifact.write_text("// as far as the parent can tell, a packed graph class\\n")
Path(job["report"]).write_text(json.dumps({{
    "entry": job["share"], "status": "compiled",
    "classes": [{{"name": "unet/adapter=true/dim=0", "key": "ek1-fake",
                 "artifact": str(artifact)}}],
    "declared_classes": 1,
    "detail": "1 packed graph class", "elapsed_s": 0.01, "peak_rss_bytes": 1,
    "phases": {{}},
}}))
sys.exit(0)
'''


def _template(tmp_path: Path) -> pool.EntryJob:
    """The recipe half of a job — never reached, because this child is refused
    on its code identity before anything it produced is believed."""
    return pool.EntryJob(
        function="txt2img", modules=("nowhere",),
        out_dir=str(tmp_path / "artifacts"))


#: pgw#1438: `aot_compile_pool` spawns `python -m gen_worker.aot_compile_child`
#: (`ENTRY_CHILD_MODULE`), and cd46c957 deleted that module while keeping the
#: pool. The tier is not RETIRED — pgw#1373 states it "stays for pgw#1371/#1372"
#: — it is a parent with no child until one of those lanes wires it to
#: `gen_worker.serving` or deletes it, and which of those happens is their
#: decision, not this lane's. So the rows that need a live child are STAGED on
#: that decision (census disposition, counted and printed), never deleted: the
#: subject still exists and a deleted row would have to be rewritten from
#: scratch the day it lands.
_NO_ENTRY_CHILD = pytest.mark.skipif(
    importlib.util.find_spec(pool.ENTRY_CHILD_MODULE) is None,
    # Kept under the census normalizer's 120-char key truncation (conftest.py
    # `_norm`), which otherwise cuts mid-phrase and leaves a trailing space the
    # census FILE's own `.strip()` then removes — a key that can never match.
    reason="pgw#1371/#1372: gen_worker.aot_compile_child is absent, so the entry pool can spawn no child",
)

@_NO_ENTRY_CHILD
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
        box.compile(_template(tmp_path))

    assert caught.value.entry == "share-000"
    detail = str(caught.value)
    assert "DIFFERENT" in detail and "too old to report one" in detail, detail
    assert pool.CODE_DIGEST in detail, detail
    # And it never became an attribution puzzle: the share has no table at all
    # rather than one whose members quietly do not add up.
    assert "share-000" not in box.entry_phases


@_NO_ENTRY_CHILD
def test_the_reports_identity_survives_the_wire(tmp_path: Path) -> None:
    """``code_digest``/``code_dir`` are the report's, not the parent's guess.

    A default-valued field is exactly what an old child produces, so the check
    must distinguish "reported nothing" from "reported a match" — the empty
    digest is a REFUSAL, never a pass.
    """
    encoded = msgspec.json.encode(pool.EntryReport(
        entry="share-000", status=pool.COMPILED,
        classes=[pool.PackedGraphClass(
            name="unet/dim=0", key="ek1-x", artifact="/dev/null")],
        code_digest=pool.CODE_DIGEST, code_dir=pool.PACKAGE_ROOT))
    back = msgspec.json.decode(encoded, type=pool.EntryReport)
    assert back.code_digest == pool.CODE_DIGEST and back.code_digest
    assert back.code_dir == pool.PACKAGE_ROOT
    assert [c.name for c in back.classes] == ["unet/dim=0"]

    old = msgspec.json.decode(
        b'{"entry":"share-000","status":"compiled"}',
        type=pool.EntryReport)
    assert old.code_digest == "" and old.spans == {}


# --- pgw#1446: the missing-contract refusal, and what it must NOT do --------


def _guard_row() -> Any:
    """The minimum `_Running` the guard reads: it only uses `row.entry`."""
    return type("_Row", (), {"entry": "share-000"})()


def test_an_unverifiable_child_report_is_refused_when_the_contract_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#840's hazard, reached through pgw#1444's recorded cause.

    A child REPORTED code the parent cannot check, because a module
    `_CONTRACT_MODULES` names is absent. That is a broken install, not the
    frozen case, and the artifact must not be packed.
    """
    monkeypatch.setattr(pool, "_CONTRACT_MISSING", ["aot_compile_child.py"])
    box = object.__new__(pool.EntryCompilePool)
    report = pool.EntryReport(entry="share-000", code_digest="deadbeefdeadbeef")

    with pytest.raises(pool.EntryCompileFailed, match="CANNOT CHECK IT"):
        pool.EntryCompilePool._verify_child_code(box, _guard_row(), report)


def test_a_missing_contract_alone_does_not_refuse_a_child_that_reported_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE pgw#1446 REGRESSION, and it is about measurements.

    pgw#1444 refused on `_CONTRACT_MISSING` unconditionally. `entry_phases` is
    banked AFTER this gate, so that raise emptied it and broke pgw#1189's law
    — the attempt that FAILED is exactly the attempt whose measurement the
    next one sizes against (pgw#877 states it three lines above the call
    site). It also fired on a tier Paul PARKED on pgw#1371/#1372 (`c4f2caaa`),
    which cannot run at all, so it protected nothing while breaking staged
    tests.

    No reported digest means no artifact of unknown origin — nothing to
    refuse, and the measurements survive.
    """
    monkeypatch.setattr(pool, "_CONTRACT_MISSING", ["aot_compile_child.py"])
    monkeypatch.setattr(pool, "CODE_DIGEST", "")
    box = object.__new__(pool.EntryCompilePool)
    report = pool.EntryReport(entry="share-000", code_digest="")

    pool.EntryCompilePool._verify_child_code(box, _guard_row(), report)  # no raise


def test_the_missing_contract_cause_is_recorded_rather_than_silent() -> None:
    """The half of pgw#1444 that was right and stays: an empty digest whose
    cause is a DELETED module is inspectable, where before it was
    indistinguishable from zipimport."""
    assert pool.CODE_DIGEST == "" and pool._CONTRACT_MISSING == [
        "aot_compile_child.py"
    ], (pool.CODE_DIGEST, pool._CONTRACT_MISSING)
