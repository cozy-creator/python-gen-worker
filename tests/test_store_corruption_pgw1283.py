"""pgw#1283 — absence and corruption are DIFFERENT facts, and a stored cell
is stored even when its shortcut is not.

Two defects that predate the pgw#1232 program and cost real money on master:

* ``_read_json`` caught ``OSError`` and ``ValueError`` alike and returned
  ``None`` for both, so a corrupt ``record.json`` was indistinguishable from an
  empty store. ``lookup`` then reported a resident cell as absent — a **silent
  re-mint**, i.e. a GPU pod run nobody could see in a log — and ``mark`` quietly
  dropped a verdict/sink transition, losing a ``delivered`` sink into a
  redundant re-upload.
* ``store`` wrote the memo INSIDE the same ``try`` as the record, *after* the
  record was already durable. A failure to write a shortcut therefore returned
  ``None`` while the artifact and record sat on disk, and
  ``fleet_cells._stage_durable`` emitted ``local_compiled_graph_store_failed``
  for a cell ``lookup`` would happily find.

WHAT A CORRUPT RECORD MEANS TO ``lookup`` — decided here, deliberately. It
still answers "absent", because the digest that vouches for the bytes lived in
the record and this store never arms bytes it cannot verify (the module's trust
boundary: a cell is user-generated EXECUTABLE code). What changes is that the
corruption is now SAID and the unusable directory is dropped, exactly as the
long-standing digest-mismatch path already does. The re-mint is correct; its
silence was the defect.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import local_cell_store

KEY_A = "cg-key-v1-" + "a" * 56
KEY_B = "cg-key-v1-" + "b" * 56
ARM_A = "arm2-" + "1" * 40


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cozy-cells"
    monkeypatch.setenv(local_cell_store.ENV_STORE_DIR, str(root))
    return root


def _artifact(tmp_path: Path, *, key: str = KEY_A, name: str = "mint") -> Path:
    """A packed cell carrying its own stamp, as a real mint produces."""
    p = tmp_path / name / "cell.tar.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"kind": "aot-inductor", "compiled_graph_key": key,
         "family": "micro-diffusion"}
    ).encode()
    with tarfile.open(p, mode="w:gz") as tar:
        info = tarfile.TarInfo("metadata.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return p


def _stored(tmp_path: Path, *, key: str = KEY_A, arm_token: str = "",
            name: str = "mint") -> local_cell_store.LocalCell:
    cell = local_cell_store.store(
        _artifact(tmp_path, key=key, name=name), key=key,
        family="micro-diffusion", arm_token=arm_token)
    assert cell is not None, "fixture setup: the cell must store"
    return cell


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
    assert local_cell_store._read_json(missing) is None

    corrupt = tmp_path / "record.json"
    _corrupt(corrupt)
    with pytest.raises(local_cell_store.RecordUnreadable):
        local_cell_store._read_json(corrupt)

    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[1, 2, 3]")
    with pytest.raises(local_cell_store.RecordUnreadable):
        local_cell_store._read_json(not_an_object)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"verdict": "admitted"}))
    assert local_cell_store._read_json(good) == {"verdict": "admitted"}


def test_lookup_says_so_and_drops_when_a_record_is_unreadable(
    store: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt record beside intact bytes is not silence.

    The answer is still ``None`` — without the recorded digest nothing can
    vouch for executable bytes — but it is LOGGED at error, and the directory
    that can no longer be used is dropped, so the corruption costs one honest
    re-mint rather than an invisible one.
    """
    _stored(tmp_path)
    record = local_cell_store.cell_dir(KEY_A) / local_cell_store.RECORD_NAME
    artifact = local_cell_store.cell_dir(KEY_A) / local_cell_store.ARTIFACT_NAME
    assert artifact.is_file(), "the bytes are intact; only the record rots"
    _corrupt(record)

    with caplog.at_level(logging.ERROR, logger=local_cell_store.__name__):
        assert local_cell_store.lookup(KEY_A) is None

    assert any("unreadable" in r.message.lower() or "unreadable" in r.getMessage().lower()
               for r in caplog.records), (
        "a corrupt record must be reported, not swallowed into a silent miss")
    assert not local_cell_store.cell_dir(KEY_A).exists(), (
        "the unusable cell directory must be dropped, as the digest-mismatch "
        "path already does")


def test_mark_reports_the_transition_it_loses(
    store: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The expensive silent case: a lost ``delivered`` means a re-upload.

    ``mark`` still refuses — it cannot merge into a record it cannot read —
    but a dropped state transition is now an error line naming the verdict and
    sink that were not applied.
    """
    _stored(tmp_path)
    record = local_cell_store.cell_dir(KEY_A) / local_cell_store.RECORD_NAME
    _corrupt(record)

    with caplog.at_level(logging.ERROR, logger=local_cell_store.__name__):
        assert local_cell_store.mark(
            KEY_A, sink=local_cell_store.SINK_DELIVERED) is False

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "unreadable" in joined.lower(), (
        "losing a sink transition to corruption must be visible")
    assert local_cell_store.SINK_DELIVERED in joined, (
        "the log must name the transition that was lost, or it cannot be "
        "reconciled after the fact")


def test_one_corrupt_record_does_not_blank_the_listing(
    store: Path, tmp_path: Path,
) -> None:
    """``stored_cells`` is what a ``cozy cells``-style CLI reads.

    One bad file must cost that entry, never the whole inventory.
    """
    _stored(tmp_path, key=KEY_A, name="a")
    _stored(tmp_path, key=KEY_B, name="b")
    _corrupt(local_cell_store.cell_dir(KEY_A) / local_cell_store.RECORD_NAME)

    listed = local_cell_store.stored_cells()
    assert [c.key for c in listed] == [KEY_B], (
        "the good cell must survive a corrupt sibling")


def test_an_unreadable_memo_is_discarded_rather_than_shadowing(
    store: Path, tmp_path: Path,
) -> None:
    """A memo is a shortcut, never an authority — but a corrupt one left in
    place is paid for on every boot, because ``note_memo`` only rewrites it
    once a cell arms."""
    _stored(tmp_path, arm_token=ARM_A)
    path = local_cell_store.memo_path(ARM_A)
    assert path.is_file(), "fixture setup: store writes the memo"
    _corrupt(path)

    assert local_cell_store.lookup_for_arm(ARM_A) is None
    assert not path.exists(), (
        "the corrupt shortcut must be cleared so evidence can rebuild it")


def test_a_corrupt_trust_class_reads_as_not_yet_known(
    store: Path,
) -> None:
    """"Not yet known" is the self-healing answer: the next mint attempts the
    publish and re-learns the class from the hub's own refusal. It must not
    raise out of a plain accessor."""
    root = local_cell_store.store_root()
    root.mkdir(parents=True, exist_ok=True)
    _corrupt(root / local_cell_store.TRUST_CLASS_NAME)

    assert local_cell_store.trust_class() == ""
    assert local_cell_store.keeps_cells_locally() is False


# ---------------------------------------------------------------- defect 2


def test_a_failed_memo_write_does_not_report_the_store_as_failed(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store``'s return value means "artifact and record are durable".

    The memo is a shortcut whose failure costs a lookup and nothing else, so it
    must not be able to turn a stored cell into a reported failure — which is
    what ``fleet_cells._stage_durable`` would then log as
    ``local_compiled_graph_store_failed`` for a cell that is, in fact, stored.
    """
    real = local_cell_store._write_json_atomic

    def fail_only_the_memo(path: Path, payload: Dict[str, Any]) -> None:
        if local_cell_store.MEMO_DIRNAME in path.parts:
            raise OSError("no space left on device")
        real(path, payload)

    monkeypatch.setattr(
        local_cell_store, "_write_json_atomic", fail_only_the_memo)

    cell = local_cell_store.store(
        _artifact(tmp_path), key=KEY_A, family="micro-diffusion",
        arm_token=ARM_A)

    assert cell is not None, (
        "the record was durable before the memo was attempted; reporting "
        "failure here is the alias hazard")
    assert cell.key == KEY_A
    assert not local_cell_store.memo_path(ARM_A).exists(), (
        "the memo genuinely did not get written — the point is that the CELL "
        "is still reported stored")

    monkeypatch.setattr(local_cell_store, "_write_json_atomic", real)
    found = local_cell_store.lookup(KEY_A)
    assert found is not None and found.key == KEY_A, (
        "lookup and store must agree about whether the cell exists")
