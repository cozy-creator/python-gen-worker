"""pgw#1283 — absence and corruption are DIFFERENT facts, and a stored compiled graph
is stored even when its shortcut is not.

Two defects that predate the pgw#1232 program and cost real money on master:

* ``_read_json`` caught ``OSError`` and ``ValueError`` alike and returned
  ``None`` for both, so a corrupt ``record.json`` was indistinguishable from an
  empty store. ``lookup`` then reported a resident compiled graph as absent — a **silent
  re-mint**, i.e. a GPU pod run nobody could see in a log — and ``mark`` quietly
  dropped a verdict/sink transition, losing a ``delivered`` sink into a
  redundant re-upload.
* ``store`` wrote the memo INSIDE the same ``try`` as the record, *after* the
  record was already durable. A failure to write a shortcut therefore returned
  ``None`` while the artifact and record sat on disk, and
  ``fleet_compiled_graphs._stage_durable`` emitted ``local_compiled_graph_store_failed``
  for a compiled graph ``lookup`` would happily find.

WHAT A CORRUPT RECORD MEANS TO ``lookup`` — decided here, deliberately. It
still answers "absent", because the digest that vouches for the bytes lived in
the record and this store never arms bytes it cannot verify (the module's trust
boundary: a compiled graph is user-generated EXECUTABLE code). What changes is that the
corruption is now SAID and the unusable directory is dropped, exactly as the
long-standing digest-mismatch path already does. The re-mint is correct; its
silence was the defect.
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

import tcg_artifacts
from gen_worker import local_compiled_graph_store

# pgw#1283: real TCG envelopes. The store no longer holds opaque bytes, so a
# hand-typed key and a hand-rolled tarball are refused at the seam — see this
# file's sibling note in ``tests/tcg_artifacts.py``.
_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="pgw1283-corruption-"))
atexit.register(shutil.rmtree, _FIXTURE_DIR, True)
ARTIFACT_A = tcg_artifacts.build(_FIXTURE_DIR / "a.tar.gz", witness="a" * 16)
ARTIFACT_B = tcg_artifacts.build(_FIXTURE_DIR / "b.tar.gz", witness="b" * 16)
KEY_A = tcg_artifacts.key_of(ARTIFACT_A)
KEY_B = tcg_artifacts.key_of(ARTIFACT_B)
ARM_A = "arm2-" + "1" * 40


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-compiled graphs"
    monkeypatch.setenv(local_compiled_graph_store.ENV_STORE_DIR, str(root))
    return root


@pytest.fixture()
def cas(tmp_path: Path) -> Path:
    return tmp_path / "cas"


def _artifact(tmp_path: Path, *, source: Path = ARTIFACT_A,
              name: str = "mint") -> Path:
    """A packed compiled graph carrying its own stamp, as a real mint produces."""
    p = tmp_path / name / "cell.tar.gz"  # cell-spelling: on-disk artifact name read by cozy-local's compiled-graphs CLI
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, p)
    return p


def _stored(tmp_path: Path, cas: Path, *, source: Path = ARTIFACT_A,
            key: str = KEY_A, arm_token: str = "",
            name: str = "mint") -> local_compiled_graph_store.CompiledGraphRecord:
    graph = local_compiled_graph_store.store(
        _artifact(tmp_path, source=source, name=name), key=key,
        family="micro-diffusion", arm_token=arm_token, cas_root=cas)
    assert graph is not None, "fixture setup: the compiled graph must store"
    return graph


def _corrupt(path: Path) -> None:
    """Leave something at ``path`` that is not a usable record."""
    path.write_text("{ this is not json")


# ---------------------------------------------------------------- defect 1


def test_read_json_separates_absence_from_corruption(tmp_path: Path) -> None:
    """The unit-level distinction the two callers below depend on.

    Absence stays ``None`` — the empty store is the normal case and must not
    become noisy. Corruption raises, so no caller can read it as "nothing was
    ever stored" by accident.
    """
    missing = tmp_path / "no-such-file.json"
    assert local_compiled_graph_store._read_json(missing) is None

    corrupt = tmp_path / "record.json"
    _corrupt(corrupt)
    with pytest.raises(local_compiled_graph_store.RecordUnreadable):
        local_compiled_graph_store._read_json(corrupt)

    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[1, 2, 3]")
    with pytest.raises(local_compiled_graph_store.RecordUnreadable):
        local_compiled_graph_store._read_json(not_an_object)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"verdict": "admitted"}))
    assert local_compiled_graph_store._read_json(good) == {"verdict": "admitted"}


def test_lookup_says_so_and_drops_when_a_record_is_unreadable(
    store: Path, cas: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt record beside intact bytes is not silence.

    The answer is still ``None`` — without the recorded digest nothing can
    vouch for executable bytes — but it is LOGGED at error, and the directory
    that can no longer be used is dropped, so the corruption costs one honest
    re-mint rather than an invisible one.
    """
    _stored(tmp_path, cas)
    record = local_compiled_graph_store.compiled_graph_dir(KEY_A) / local_compiled_graph_store.RECORD_NAME
    _corrupt(record)

    with caplog.at_level(logging.ERROR, logger=local_compiled_graph_store.__name__):
        assert local_compiled_graph_store.lookup(KEY_A, cas_root=cas) is None

    assert any("unreadable" in r.message.lower() or "unreadable" in r.getMessage().lower()
               for r in caplog.records), (
        "a corrupt record must be reported, not swallowed into a silent miss")
    assert not local_compiled_graph_store.compiled_graph_dir(KEY_A).exists(), (
        "the unusable compiled graph directory must be dropped, as the digest-mismatch "
        "path already does")


def test_mark_reports_the_transition_it_loses(
    store: Path, cas: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The expensive silent case: a lost ``delivered`` means a re-upload.

    ``mark`` still refuses — it cannot merge into a record it cannot read —
    but a dropped state transition is now an error line naming the verdict and
    sink that were not applied.
    """
    _stored(tmp_path, cas)
    record = local_compiled_graph_store.compiled_graph_dir(KEY_A) / local_compiled_graph_store.RECORD_NAME
    _corrupt(record)

    with caplog.at_level(logging.ERROR, logger=local_compiled_graph_store.__name__):
        assert local_compiled_graph_store.mark(
            KEY_A, sink=local_compiled_graph_store.SINK_DELIVERED) is False

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "unreadable" in joined.lower(), (
        "losing a sink transition to corruption must be visible")
    assert local_compiled_graph_store.SINK_DELIVERED in joined, (
        "the log must name the transition that was lost, or it cannot be "
        "reconciled after the fact")


def test_one_corrupt_record_does_not_blank_the_listing(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """``stored_compiled_graphs`` is what a ``cozy compiled graphs``-style CLI reads.

    One bad file must cost that entry, never the whole inventory.
    """
    _stored(tmp_path, cas, source=ARTIFACT_A, key=KEY_A, name="a")
    _stored(tmp_path, cas, source=ARTIFACT_B, key=KEY_B, name="b")
    _corrupt(local_compiled_graph_store.compiled_graph_dir(KEY_A) / local_compiled_graph_store.RECORD_NAME)

    listed = local_compiled_graph_store.stored_compiled_graphs()
    assert [c.key for c in listed] == [KEY_B], (
        "the good compiled graph must survive a corrupt sibling")


def test_an_unreadable_memo_is_discarded_rather_than_shadowing(
    store: Path, cas: Path, tmp_path: Path,
) -> None:
    """A memo is a shortcut, never an authority — but a corrupt one left in
    place is paid for on every boot, because ``note_memo`` only rewrites it
    once a compiled graph arms."""
    _stored(tmp_path, cas, arm_token=ARM_A)
    path = local_compiled_graph_store.memo_path(ARM_A)
    assert path.is_file(), "fixture setup: store writes the memo"
    _corrupt(path)

    assert local_compiled_graph_store.lookup_for_arm(ARM_A, cas_root=cas) is None
    assert not path.exists(), (
        "the corrupt shortcut must be cleared so evidence can rebuild it")


def test_a_corrupt_trust_class_reads_as_not_yet_known(
    store: Path,
) -> None:
    """"Not yet known" is the self-healing answer: the next mint attempts the
    publish and re-learns the class from the hub's own refusal. It must not
    raise out of a plain accessor."""
    root = local_compiled_graph_store.store_root()
    root.mkdir(parents=True, exist_ok=True)
    _corrupt(root / local_compiled_graph_store.TRUST_CLASS_NAME)

    assert local_compiled_graph_store.trust_class() == ""
    assert local_compiled_graph_store.keeps_compiled_graphs_locally() is False


# ---------------------------------------------------------------- defect 2


def test_a_failed_memo_write_does_not_report_the_store_as_failed(
    store: Path, cas: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store``'s return value means "artifact and record are durable".

    The memo is a shortcut whose failure costs a lookup and nothing else, so it
    must not be able to turn a stored compiled graph into a reported failure — which is
    what ``fleet_compiled_graphs._stage_durable`` would then log as
    ``local_compiled_graph_store_failed`` for a compiled graph that is, in fact, stored.
    """
    real = local_compiled_graph_store._write_json_atomic

    def fail_only_the_memo(path: Path, payload: Dict[str, Any]) -> None:
        if local_compiled_graph_store.MEMO_DIRNAME in path.parts:
            raise OSError("no space left on device")
        real(path, payload)

    monkeypatch.setattr(
        local_compiled_graph_store, "_write_json_atomic", fail_only_the_memo)

    graph = local_compiled_graph_store.store(
        _artifact(tmp_path), key=KEY_A, family="micro-diffusion",
        arm_token=ARM_A, cas_root=cas)

    assert graph is not None, (
        "the record was durable before the memo was attempted; reporting "
        "failure here is the alias hazard")
    assert graph.key == KEY_A
    assert not local_compiled_graph_store.memo_path(ARM_A).exists(), (
        "the memo genuinely did not get written — the point is that the COMPILED GRAPH "
        "is still reported stored")

    monkeypatch.setattr(local_compiled_graph_store, "_write_json_atomic", real)
    found = local_compiled_graph_store.lookup(KEY_A, cas_root=cas)
    assert found is not None and found.key == KEY_A, (
        "lookup and store must agree about whether the compiled graph exists")
