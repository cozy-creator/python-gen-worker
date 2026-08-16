"""pgw#1283 — the STORE split: TCG owns the bytes, the worker owns the policy.

The two NO-SHIP criteria this unit had to CLOSE, and the one whose deviation is
flagged for a ruling.

**Criterion 4 — TCG's storage quarantine must stay separate from the worker's.**
Two directions, and the cutover created the second one:

* a TCG CAS quarantine is a fact about a stored record and is repaired by
  re-storing the bytes, so writing it into this worker's ``verdict`` — which
  §1.3.4 makes terminal, "kept for forensics, never served" — would strand a
  healthy cell forever on a defect a re-store fixes;
* the worker's own quarantine used to be unreachable by anything else, because
  the refused bytes lived only in ``local_cell_store``. They are now in the CAS
  ``aot_serve.arm_compiled_graph`` loads runners from, so the load path must ask
  before it resolves, or it arms a cell this worker refused.

**Criterion 7 — no crash-retry alias loss.** ``store`` is idempotent across
every one of TCG's success outcomes. A crash between the record write and the
memo write leaves bytes TCG reports ``PRESENT`` on the retry; if that outcome
were allowed to skip the rest of the function, the arm-token alias would be
gone permanently and a machine with no boot-key route would re-mint on every
boot forever. Which outcome came back is a fact for the log, never a branch.

**Criterion 6 — collision-safe memo persistence.** RESOLVED BY DEFAULT, and
flagged. The review demanded MANDATORY collision-safe memo persistence; this
module's stated design is that a memo is a shortcut and never load-bearing.
What landed: the WRITE is collision-safe (a per-write temp name, fsync, atomic
replace), and PERSISTENCE stays non-load-bearing — a failed memo write is
logged loudly and ``store`` still reports the cell stored. Making it mandatory
would mean a full disk on the cache partition fails a mint that succeeded.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

import tcg_artifacts
from gen_worker import aot_serve, local_cell_store
from gen_worker.compile_cache import AdoptError

_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="pgw1283-split-"))
atexit.register(shutil.rmtree, _FIXTURE_DIR, True)
ARTIFACT = tcg_artifacts.build(_FIXTURE_DIR / "cell.tar.gz", witness="a" * 16)
KEY = tcg_artifacts.key_of(ARTIFACT)
ARM = "arm2-" + "1" * 40


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-cells"
    monkeypatch.setenv(local_cell_store.ENV_STORE_DIR, str(root))
    return root


@pytest.fixture()
def cas(tmp_path: Path) -> Path:
    return tmp_path / "cas"


def _staged(tmp_path: Path) -> Path:
    p = tmp_path / "mint" / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT, p)
    return p


def _cas_blob(cas: Path) -> Path:
    """The store object holding the artifact — the bytes TCG vouches for."""
    blobs = sorted((p for p in cas.rglob("*") if p.is_file()),
                   key=lambda p: p.stat().st_size)
    assert blobs, f"no CAS objects under {cas}"
    return blobs[-1]


# ---------------------------------------------------------------------------
# Criterion 4a — the load path must not bypass the WORKER's quarantine
# ---------------------------------------------------------------------------


def test_a_worker_quarantined_key_is_refused_before_the_cas_is_asked(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """The defect the cutover would otherwise have created, by name.

    Before the split, a cell this worker quarantined was unreachable by any
    other route: its bytes lived only in ``local_cell_store`` and nothing else
    could address them. After it, they are in the CAS
    ``aot_serve.arm_compiled_graph`` resolves from, and TCG has no concept of a
    worker parity/arm refusal — so without the guard the runner loads a cell
    §1.3.4 says must never be served.

    The refusal is TYPED, and it fires before ``open_worker_engine`` is called
    at all: a quarantined cell must not even be unpacked by the serving path.
    """
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="micro-diffusion", arm_token=ARM,
        cas_root=cas) is not None
    local_cell_store.mark(KEY, verdict=local_cell_store.VERDICT_QUARANTINED)
    assert local_cell_store.is_quarantined(KEY)

    with pytest.raises(AdoptError) as raised:
        aot_serve.arm_compiled_graph(object(), object(), KEY, cas)
    assert raised.value.reason == "compiled_graph_worker_quarantined", (
        "the load path reached the CAS for a cell this worker refused")


def test_the_guard_refuses_ONLY_a_worker_quarantine_and_not_an_unproven_cell(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """The discriminating half — the guard must not become "refuse anything".

    §1.5 stores a cell BEFORE the gate that proves it, so an ``unverified`` row
    is the normal state of a cell that is about to be armed by the very call
    this guard sits in front of. And a key this worker never recorded is every
    hub-delivered cell there is. Refusing either would turn the criterion-4
    guard into an outage.
    """
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="micro-diffusion",
        verdict=local_cell_store.VERDICT_UNVERIFIED, cas_root=cas) is not None

    def _reason() -> str:
        """Why the load path refused, or "" when it got past every gate.

        It cannot COMPLETE here — the fixture's ``model.so`` is a shaped ELF
        with no runnable code — so what is asserted is which refusal came back,
        never that the arm succeeded. A pass/fail assertion would read green
        for the wrong reason.
        """
        try:
            aot_serve.arm_compiled_graph(object(), object(), KEY, cas)
        except AdoptError as exc:
            return str(exc.reason)
        except Exception:
            return "reached-tcg"
        return ""

    for state in (local_cell_store.VERDICT_UNVERIFIED,
                  local_cell_store.VERDICT_ADMITTED):
        local_cell_store.mark(KEY, verdict=state)
        assert _reason() != "compiled_graph_worker_quarantined", (
            f"a {state!r} cell was refused as if a gate had rejected it")

    local_cell_store.drop(KEY)
    assert _reason() != "compiled_graph_worker_quarantined", (
        "a key this worker never recorded is not a key it quarantined")


# ---------------------------------------------------------------------------
# Criterion 4b — a TCG storage quarantine is NOT this worker's verdict
# ---------------------------------------------------------------------------


def test_cas_rot_never_becomes_the_workers_verdict_and_a_repair_restores_it(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """The other direction, and the one the review named outright: *"TCG CAS
    repair could leave worker state permanently quarantined"*.

    Rot in the CAS makes TCG quarantine its own record. This worker records
    NOTHING about that — its verdict is a statement about a gate it ran — so
    re-storing the same artifact repairs TCG and the cell serves again, with no
    mint and no manual intervention. Had the worker mirrored the quarantine,
    the repair would be unreachable: :func:`mark` can move the verdict back,
    but nothing in the tree ever calls it that way, because a worker quarantine
    is terminal by design.
    """
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        cas_root=cas) is not None
    assert local_cell_store.lookup(KEY, cas_root=cas) is not None

    blob = _cas_blob(cas)
    rotted = bytearray(blob.read_bytes())
    rotted[0] ^= 0xFF
    blob.write_bytes(bytes(rotted))
    (store / "aot-cells" / KEY / "cell.tar.gz").unlink()

    assert local_cell_store.lookup(KEY, cas_root=cas) is None, (
        "bytes TCG cannot verify must not be served")
    assert local_cell_store.verdict_of(KEY) == local_cell_store.VERDICT_ADMITTED, (
        "a CAS-storage quarantine was written into this worker's verdict")
    assert not local_cell_store.is_quarantined(KEY)
    assert [c.key for c in local_cell_store.quarantined_cells()] == [], (
        "the forensics listing is the WORKER's refusals; a CAS repair case in "
        "it is a support ticket about a defect that repaired itself")

    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        cas_root=cas) is not None, "re-storing the same artifact must repair"
    healed = local_cell_store.lookup(KEY, cas_root=cas)
    assert healed is not None and healed.key == KEY
    assert healed.artifact.read_bytes() == ARTIFACT.read_bytes()


def test_a_worker_quarantine_survives_everything_the_cas_does(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """The symmetric guarantee: a gate refusal is not repairable by re-storing.

    §1.3.4 keeps a refused cell for forensics precisely because it is the one
    object that can explain its own refusal. Re-storing its bytes proves
    something about the BYTES; it says nothing about the gate that refused
    them, so it must not clear the verdict.
    """
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        cas_root=cas) is not None
    local_cell_store.mark(KEY, verdict=local_cell_store.VERDICT_QUARANTINED)

    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        verdict=local_cell_store.VERDICT_QUARANTINED, cas_root=cas) is not None
    assert local_cell_store.lookup(KEY, cas_root=cas) is None
    assert [c.key for c in local_cell_store.quarantined_cells()] == [KEY]


# ---------------------------------------------------------------------------
# Criterion 7 — no crash-retry alias loss
# ---------------------------------------------------------------------------


def test_a_retry_after_a_crash_between_the_record_and_the_memo_restores_the_alias(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """The crash-retry alias loss, at the cut where it happens.

    ``store`` is three durable steps: the bytes into TCG, the policy record,
    the arm-token alias. A crash after step 2 leaves a cell TCG will report
    ``PRESENT`` for on the retry — and if ``PRESENT`` were treated as "already
    done, nothing to do", the alias would never be written again. The bytes
    survive and the SHORTCUT does not, so a machine with no boot-key route
    (§4.28's cozy-local box, which never traces) pays a full compile on every
    boot for a cell it is holding.

    Every step therefore runs on every call. This drives the crash by deleting
    exactly what step 3 wrote, which is the same on-disk state a kill between
    the two ``os.replace``s leaves.
    """
    staged = _staged(tmp_path)
    assert local_cell_store.store(
        staged, key=KEY, family="f", arm_token=ARM, cas_root=cas) is not None

    memo = local_cell_store.memo_path(ARM)
    assert memo.is_file(), "setup: the first store wrote the alias"
    memo.unlink()                                    # the crash cut
    assert local_cell_store.lookup_for_arm(ARM, cas_root=cas) is None

    retried = local_cell_store.store(
        staged, key=KEY, family="f", arm_token=ARM, cas_root=cas)

    assert retried is not None, "the retry must report the cell stored"
    hit = local_cell_store.lookup_for_arm(ARM, cas_root=cas)
    assert hit is not None and hit.key == KEY, (
        "the retry left the arm alias lost — every later boot re-mints a cell "
        "this machine is holding")


def test_a_retry_after_a_crash_before_the_record_restores_the_whole_cell(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """The earlier cut, and the ownership rule that makes it safe.

    Bytes in the CAS with no policy record are a clean MISS, not a half-cell:
    nothing vouches for arming them, and "the CAS has it" is not this worker's
    permission. The retry re-records against bytes TCG reports ``PRESENT``, so
    the recovery costs a metadata write rather than a mint.
    """
    staged = _staged(tmp_path)
    assert local_cell_store.store(
        staged, key=KEY, family="f", arm_token=ARM, cas_root=cas) is not None
    shutil.rmtree(local_cell_store.cell_dir(KEY))     # the crash cut
    local_cell_store.memo_path(ARM).unlink(missing_ok=True)

    assert local_cell_store.lookup(KEY, cas_root=cas) is None
    assert local_cell_store.verdict_of(KEY) == ""

    assert local_cell_store.store(
        staged, key=KEY, family="f", arm_token=ARM, cas_root=cas) is not None
    assert local_cell_store.lookup(KEY, cas_root=cas) is not None
    assert local_cell_store.lookup_for_arm(ARM, cas_root=cas) is not None


def test_store_reports_every_tcg_outcome_and_branches_on_none_of_them(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """Structural half of criterion 7: the outcome is LOGGED, never a branch.

    A reader adding ``if outcome is PRESENT: return record`` recreates the
    alias loss above, and it reads like an optimisation. The only ``StoreOutcome``
    comparison in the module is the ``DIVERGENT`` refusal — which is not a
    success outcome at all.
    """
    import ast

    tree = ast.parse(Path(local_cell_store.__file__).read_text())
    compared = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "StoreOutcome"
    }
    assert compared == {"DIVERGENT"}, (
        f"the store branches on TCG success outcomes {sorted(compared)}; every "
        f"step must run on every call or a crash-retry loses the alias")


def test_a_key_tcg_binds_to_different_bytes_is_refused_and_records_nothing(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """``DIVERGENT`` is the one TCG outcome that is not a success.

    Two artifacts state the SAME key over different bytes — the envelope is
    deterministic, so this takes an inert change to the wrapper source that
    moves no fact TCG derives. TCG keeps the first bytes and quarantines the
    newcomer rather than choosing between them, and this module must then
    record nothing: a sidecar written here would file worker policy against
    bytes nothing will ever resolve.
    """
    divergent = tcg_artifacts.build(
        tmp_path / "divergent" / "cell.tar.gz", witness="a" * 16,
        filler="the same facts, different bytes")
    assert tcg_artifacts.key_of(divergent) == KEY
    assert divergent.read_bytes() != ARTIFACT.read_bytes()

    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        cas_root=cas) is not None
    first = local_cell_store.lookup(KEY, cas_root=cas)
    assert first is not None
    stored_at = first.stored_at

    assert local_cell_store.store(
        divergent, key=KEY, family="other", cas_root=cas) is None, (
        "TCG refused to rebind the key; the store must report that failure")

    kept = local_cell_store.lookup(KEY, cas_root=cas)
    assert kept is not None, "the resident cell must survive a refused newcomer"
    assert kept.family == "f" and kept.stored_at == stored_at, (
        "the refusal rewrote worker policy for bytes TCG did not accept")
    assert kept.artifact.read_bytes() == ARTIFACT.read_bytes()


# ---------------------------------------------------------------------------
# Criterion 6 — the memo WRITE is collision-safe; its PERSISTENCE is not
#               load-bearing. See this module's header: resolved by default.
# ---------------------------------------------------------------------------


def test_two_writes_to_one_memo_never_share_a_temporary_file(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 6's collision-safety half, deterministically.

    The temp name used to be ``<name>.tmp-<pid>`` — one name per PROCESS, which
    is not a distinct name for two THREADS of one process. The memo is written
    from the mint path and from the arm path, and those do run concurrently, so
    two writers shared one temp file: the second truncates and rewrites what
    the first is still filling, and the first then ``os.replace``s a torn
    interleaving under a name that says it is atomic.

    Asserted on the NAMES rather than on a race, because a race that only
    sometimes reproduces is not calibration.
    """
    temporaries: List[str] = []
    real_replace = os.replace

    def _record(src: Any, dst: Any) -> None:
        temporaries.append(str(src))
        real_replace(src, dst)

    # The publication point itself is the instrument: whatever name reaches
    # ``os.replace`` IS the file the two writers were filling. Restored the
    # instant the two writes are done, so nothing else in the run is observed.
    monkeypatch.setattr(os, "replace", _record)
    try:
        assert local_cell_store.note_memo(ARM, KEY) is True
        assert local_cell_store.note_memo(ARM, KEY) is True
    finally:
        monkeypatch.undo()

    assert len(temporaries) == 2
    assert temporaries[0] != temporaries[1], (
        "two writers of one memo shared a temporary file; the 'atomic' replace "
        "then publishes whatever the loser left behind")


def test_concurrent_memo_writers_never_publish_a_torn_memo(
    store: Path, tmp_path: Path,
) -> None:
    """The corroborating end-to-end property: readers only ever see one of the
    written answers, never a splice of two, and no temp residue survives."""
    keys = [tcg_artifacts.key_of(ARTIFACT), "cg-key-v1-" + "b" * 56]
    seen: List[Dict[str, Any]] = []
    barrier = threading.Barrier(8)
    stop = threading.Event()

    def _writer(which: int) -> None:
        barrier.wait()
        for _ in range(40):
            local_cell_store.note_memo(ARM, keys[which % 2])
        stop.set()

    def _reader() -> None:
        barrier.wait()
        while not stop.is_set():
            try:
                raw = local_cell_store.memo_path(ARM).read_text()
            except OSError:
                continue
            seen.append(json.loads(raw))     # a torn write raises here

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=_reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert seen, "the readers never observed a published memo"
    assert {row["compiled_graph_key"] for row in seen} <= set(keys)
    residue = list((store / "aot-cells" / local_cell_store.MEMO_DIRNAME
                    ).glob("*.tmp-*"))
    assert not residue, f"temporary memo files survived: {residue}"


def test_a_memo_that_cannot_be_written_is_LOUD_and_still_a_stored_cell(
    store: Path, cas: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Criterion 6's DEVIATION, asserted rather than left implicit.

    The review asked for memo persistence to be MANDATORY — part of store
    success. It is not, and this is the reason stated as a test: the memo is a
    shortcut whose absence costs one lookup, while a mandatory memo makes a
    full cache partition fail a mint that has already succeeded. What the
    deviation buys instead of silence is VOLUME: the failure is logged at
    warning with both the token and the key.

    Flagged for Paul in the PR body under "Criterion 6 — resolved by default".
    """
    import logging

    real = local_cell_store._write_json_atomic

    def _fail_only_the_memo(path: Path, payload: Dict[str, Any]) -> None:
        if local_cell_store.MEMO_DIRNAME in path.parts:
            raise OSError("no space left on device")
        real(path, payload)

    monkeypatch.setattr(
        local_cell_store, "_write_json_atomic", _fail_only_the_memo)

    with caplog.at_level(logging.WARNING, logger=local_cell_store.__name__):
        stored = local_cell_store.store(
            _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
            cas_root=cas)

    assert stored is not None, (
        "a failed SHORTCUT reported the whole cell unstored — the defect "
        "pgw#1283 opened on")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert ARM in joined and KEY in joined, (
        "a non-load-bearing failure must still name what was lost, or "
        "'best-effort' is indistinguishable from 'silent'")
    monkeypatch.setattr(local_cell_store, "_write_json_atomic", real)
    assert local_cell_store.lookup(KEY, cas_root=cas) is not None


# ---------------------------------------------------------------------------
# The split itself — the duplication that WAS the unit
# ---------------------------------------------------------------------------


def test_the_store_computes_no_digest_of_its_own(store: Path) -> None:
    """The unit, stated as an absence.

    ``local_cell_store`` used to hash the artifact on the way in and re-hash it
    on every lookup — twice re-deriving what the CAS derived when it ingested
    the bytes, over an artifact measured in hundreds of megabytes. TCG owns the
    digest now, and the module has no hashing left to drift out of agreement
    with it.
    """
    import ast

    source = Path(local_cell_store.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "hashlib" not in imported, (
        "the store grew a second digest authority over bytes it does not own")
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for word in ("sha256", "hexdigest", "digest"):
        assert not {n for n in names if word in n.lower()}, (
            f"{word!r} survives in the store; the duplication IS the unit")


def test_the_owed_scan_stays_metadata_only(
    store: Path, cas: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 5, re-proved against the NEW custody (it was already met).

    ``cells_owed_to_sink`` reads sidecars. Under the split the tempting bug is
    new: materializing each candidate to size or verify it would make "is
    anything owed?" cost an export per resident cell, on every reconnect.
    """
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        sink=local_cell_store.SINK_OWED, cas_root=cas) is not None

    def _no_engine(*a: Any, **k: Any) -> Any:
        raise AssertionError("an owed SCAN asked TCG for bytes")

    monkeypatch.setattr(local_cell_store, "_engine", _no_engine)

    owed = local_cell_store.cells_owed_to_sink()
    assert [c.key for c in owed] == [KEY]
    assert not hasattr(owed[0], "artifact"), (
        "a scan row must not carry a materialized path — that is what makes "
        "the cheap scan impossible to use expensively by accident")


def test_dropping_a_cell_forgets_the_POLICY_and_never_the_cas_bytes(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """``drop`` narrowed, deliberately.

    It used to delete the only copy of an artifact. The bytes are now
    content-addressed and may be exactly the bytes another route resolves, so a
    worker decision about ONE arm identity must not destroy them. What goes is
    the permission to serve — and a later store of the same artifact re-records
    it against a TCG ``PRESENT``, at no mint cost.
    """
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        cas_root=cas) is not None
    blob = _cas_blob(cas)

    local_cell_store.drop(KEY)

    assert not local_cell_store.cell_dir(KEY).exists()
    assert local_cell_store.lookup(KEY, cas_root=cas) is None
    assert blob.is_file(), "a worker policy decision deleted CAS bytes"
    assert local_cell_store.store(
        _staged(tmp_path), key=KEY, family="f", arm_token=ARM,
        cas_root=cas) is not None
    assert local_cell_store.lookup(KEY, cas_root=cas) is not None
