"""pgw#973 (DESIGN-RULINGS §4.24): ONE bound on the safetensors header length.

The threat is named in ``gen_worker/models/safetensors_header.py``: an 8-byte
declared header length, read from the file before anything has validated it,
sizes a read and a JSON parse in whichever process opened the file.

A second copy of the bound that disagrees is not harmless: a writer accepting
headers the loader refuses emits a shard the serving path cannot open — same
bytes, two verdicts.

These tests drive every real entry point against real files on disk — no mocks,
no monkeypatching of the bound. They prove the one bound refuses the runaway
everywhere, and that writer and loader agree.

RED-verified: restoring ``_MAX_HEADER_BYTES = 512 * 1024 * 1024`` in
convert/writer.py fails ``test_writer_and_loader_agree_on_the_same_file``
(the writer parses a 200 MiB-declared header the loader rejects).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from gen_worker.convert.ingest import detect_snapshot_dtype
from gen_worker.convert.writer import read_safetensors_header, component_stored_tensor_names
from gen_worker.models.loading import safetensors_file_valid
from gen_worker.models.safetensors_header import MAX_HEADER_BYTES, header_len_ok
from gen_worker.models.svdq import _read_safetensors_metadata
from gen_worker.models.w4a4 import _read_header as w4a4_read_header
from gen_worker.models.w8a8 import _read_header as w8a8_read_header
import sys

# pgw#1310: one home for "which subtrees a guard may not judge" —
# scripts/_lint_scope.py, shared with the CI lint scanners.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _lint_scope import is_unowned  # noqa: E402


# One tensor of one F32 element, so the header describes something real.
_HEADER = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}


def _write_safetensors(path: Path, *, declared_len: int | None = None) -> Path:
    """Write a real safetensors file. ``declared_len`` overrides ONLY the
    8-byte prefix, which is exactly what a hostile or corrupt file does: it
    promises a header it never has to back with bytes."""
    body = json.dumps(_HEADER).encode()
    n = len(body) if declared_len is None else declared_len
    path.write_bytes(struct.pack("<Q", n) + body + b"\x00" * 4)
    return path


# --------------------------------------------------------------------------
# The bound itself
# --------------------------------------------------------------------------

def test_the_bound_refuses_the_runaway_and_absence():
    assert header_len_ok(1)
    assert header_len_ok(MAX_HEADER_BYTES)
    # The runaway: a length no file could back.
    assert not header_len_ok(2**63 - 1)
    assert not header_len_ok(MAX_HEADER_BYTES + 1)
    # §4.24 item 4: zero is an absent header, not an empty one.
    assert not header_len_ok(0)
    assert not header_len_ok(-1)


def test_the_bound_is_stated_exactly_once():
    """Structural pin: no module may re-declare a safetensors header cap.

    Deleting five copies is only worth doing if a sixth cannot quietly come
    back.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "gen_worker"
    owner = src / "models" / "safetensors_header.py"
    offenders = []
    for py in src.rglob("*.py"):
        if py == owner or is_unowned(py, src):
            continue
        for i, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "HEADER_BYTES" in stripped and "=" in stripped and "import" not in stripped:
                offenders.append(f"{py.relative_to(src)}:{i}: {stripped}")
    assert not offenders, (
        "a second safetensors header bound is back — §4.24: one threat, one "
        "number; import it from models/safetensors_header.py:\n" + "\n".join(offenders)
    )


# --------------------------------------------------------------------------
# Every entry point that used to carry its own copy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("declared", [2**63 - 1, MAX_HEADER_BYTES + 1, 0])
def test_every_reader_refuses(tmp_path: Path, declared: int):
    """The five readers that each carried their own 100 MiB copy."""
    f = _write_safetensors(tmp_path / "m.safetensors", declared_len=declared)

    assert safetensors_file_valid(str(f)) is False
    assert w8a8_read_header(f) == {}
    assert w4a4_read_header(f) == {}
    assert _read_safetensors_metadata(f) in ({}, None)
    with pytest.raises(ValueError):
        fd = os.open(f, os.O_RDONLY)
        try:
            read_safetensors_header(fd)
        finally:
            os.close(fd)


def test_ingest_refuses(tmp_path: Path):
    """convert/ingest's snapshot dtype detector — the sixth copy."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    _write_safetensors(snapshot / "m.safetensors", declared_len=2**63 - 1)
    # No parseable header anywhere => no dtype could be detected.
    assert detect_snapshot_dtype(snapshot) in (None, "")


def test_a_legitimate_file_still_parses_everywhere(tmp_path: Path):
    """The deletion must not have made anything stricter than it was."""
    f = _write_safetensors(tmp_path / "ok.safetensors")

    assert safetensors_file_valid(str(f)) is True
    assert w8a8_read_header(f) == _HEADER
    assert w4a4_read_header(f) == _HEADER
    fd = os.open(f, os.O_RDONLY)
    try:
        header, _ = read_safetensors_header(fd)
    finally:
        os.close(fd)
    assert header == _HEADER


def test_writer_and_loader_agree_on_the_same_file(tmp_path: Path):
    """The defect the 512 MiB outlier caused, pinned.

    A 200 MiB declared header sits between the old writer cap (512 MiB) and
    the reader cap (100 MiB). Before this change the writer parsed it and the
    loader refused it — a shard the re-shard path would emit and the serving
    path could not open. Now both refuse.
    """
    between = 200 * 1024 * 1024
    assert MAX_HEADER_BYTES < between < 512 * 1024 * 1024
    f = _write_safetensors(tmp_path / "between.safetensors", declared_len=between)

    assert safetensors_file_valid(str(f)) is False, "premise: the loader refuses it"
    fd = os.open(f, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="implausible header_length"):
            read_safetensors_header(fd)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# The two readers the CENSUS MISSED — they had no cap at all
# --------------------------------------------------------------------------
# pgw#973 wave 2 follow-up. The census enumerated *bounds* and adjudicated each
# one, so a reader carrying NO bound was invisible to it. Two such readers
# existed since 2026-07-24 (`6714ad8b`), both doing
# `json.loads(f.read(header_len))` straight off an unvalidated 8-byte prefix:
# `convert/writer.component_stored_tensor_names` and
# `models/loading` in the deshard path. Same threat, zero guard.
#
# Recorded as a METHOD defect, not just two fixes: "census every bound" does
# not find "the place a bound should be and isn't". Absence is invisible to an
# inventory of what is present.

def test_writer_component_scan_refuses_unbounded_header(tmp_path: Path):
    comp = tmp_path / "component"
    comp.mkdir()
    _write_safetensors(comp / "m.safetensors", declared_len=2**63 - 1)
    with pytest.raises(ValueError, match="implausible header_length"):
        component_stored_tensor_names(comp)


def test_writer_component_scan_still_reads_a_good_file(tmp_path: Path):
    comp = tmp_path / "component"
    comp.mkdir()
    _write_safetensors(comp / "m.safetensors")
    assert component_stored_tensor_names(comp) == frozenset({"t"})
