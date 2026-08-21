from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from gen_worker._vendor.tensorfs import gguf
from gen_worker._vendor.tensorfs.local import LocalCAS
from gen_worker._vendor.tensorfs.manifest import FileEntry
from gen_worker.cas import ingest_file
from gen_worker.cas.planner import (
    BLOB_V1,
    GGUF_V1,
    MAX_OBJECT_SIZE,
    SAFETENSORS_V1,
    Region,
    plan,
    plan_chunks,
)
from gen_worker._vendor.tensorfs.writer import TensorWriter

VECTORS = Path(__file__).resolve().parent / "testdata" / "planner-vectors"
UPSTREAM = Path(
    os.environ.get("TENSORFS_REPO", Path.home() / "cozy" / "tensorfs")
).expanduser()

PLANNER_VECTORS_DIGEST = "c180e5ebf28b96b475f75059ef914acaecf886e26d3b1e0f2dfb7a3f32449983"


def _corpus_digest(root: Path) -> str:
    rolling = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rolling.update(path.relative_to(root).as_posix().encode())
        rolling.update(b"\0")
        rolling.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        rolling.update(b"\n")
    return rolling.hexdigest()


def _cases() -> list[dict]:
    return json.loads((VECTORS / "planner-vectors.json").read_text())["cases"]


def _source(case: dict) -> bytes:
    raw = bytes.fromhex((VECTORS / case["fixture"]).read_text().strip())
    return raw + b"\0" * int(case.get("zero_tail") or 0)


def _Bytes(data: bytes) -> BinaryIO:
    return io.BytesIO(data)


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_every_planner_vector_agrees_object_for_object(case: dict) -> None:
    raw = _source(case)
    expected = case["expected"]
    planned = plan(_Bytes(raw), len(raw))

    assert planned.planner == expected["planner"], (
        f"{case['name']}: chose {planned.planner}, upstream chooses "
        f"{expected['planner']}"
    )
    assert planned.file_size == expected["file_size"]

    got = [(region.offset, region.length, region.kind) for region in planned.regions]
    want = [(o["offset"], o["length"], o["kind"]) for o in expected["objects"]]
    assert got == want, (
        f"{case['name']}: object boundaries differ from the Rust planner.\n"
        f"  planned  {got}\n  upstream {want}"
    )

    for region, declared in zip(planned.regions, expected["objects"], strict=True):
        digest = hashlib.sha256(raw[region.offset : region.offset + region.length]).hexdigest()
        assert f"sha256:{digest}" == declared["digest"], (
            f"{case['name']}: object at {region.offset} hashes to {digest}, "
            f"upstream declares {declared['digest']}"
        )


def test_the_corpus_is_the_one_upstream_released() -> None:
    """The vendored corpus is pinned by digest AND compared to the sibling."""

    assert _corpus_digest(VECTORS) == PLANNER_VECTORS_DIGEST, (
        "the vendored planner-vector corpus was edited. It is upstream's "
        "released conformance corpus and is re-vendored, never patched."
    )

    upstream = UPSTREAM / "spec" / "v1" / "planner-vectors"
    if not UPSTREAM.is_dir():
        pytest.skip("no tensorfs checkout beside this repo (set TENSORFS_REPO)")
    assert upstream.is_dir(), (
        f"{UPSTREAM} exists but {upstream} does not. The corpus moved or the "
        f"repo was renamed; this gate cannot be satisfied by ignoring it."
    )
    compared = 0
    for mine in sorted(p for p in VECTORS.rglob("*") if p.is_file()):
        theirs = upstream / mine.relative_to(VECTORS)
        assert theirs.is_file(), f"{theirs} is missing upstream"
        assert mine.read_bytes() == theirs.read_bytes(), f"{mine.name} drifted from upstream"
        compared += 1
    assert compared == 15, f"compared {compared} files, expected the whole 15-file corpus"


def test_the_retired_greedy_pack_is_gone() -> None:
    """The concrete drift: two small tensors are two objects, not one pack."""

    case = next(c for c in _cases() if c["name"] == "safetensors-two-tensors")
    raw = _source(case)
    assert plan_chunks(_Bytes(raw), len(raw)) == (113, 3, 5)


@pytest.mark.parametrize("name", ["gguf-v2-f32", "gguf-v3-q4-0"])
def test_gguf_is_planned_as_gguf_and_not_as_a_raw_blob(name: str) -> None:

    case = next(c for c in _cases() if c["name"] == name)
    raw = _source(case)
    planned = plan(_Bytes(raw), len(raw))
    assert planned.planner == GGUF_V1
    assert len(planned.regions) > 1


def _assemble(cas: LocalCAS, entry: FileEntry) -> bytes:
    return b"".join(cas.object_path(chunk.digest).read_bytes() for chunk in entry.chunks)


def _write_safetensors(
    cas: LocalCAS, tensors: dict[str, bytes], path: str = "model.safetensors"
) -> FileEntry:
    writer = TensorWriter(cas, path)
    for name, payload in tensors.items():
        writer.add(name, "U8", (len(payload),), payload)
    return writer.finish()


def test_the_writer_and_the_planner_cut_in_the_same_places(tmp_path: Path) -> None:
    """`TensorWriter` emits the seal planner's grid (upstream asserts that against the real Rust planner)."""

    cas = LocalCAS(tmp_path / "cas")
    entry = _write_safetensors(
        cas,
        {
            "small_a": b"\x01" * 3,
            "small_b": b"\x02" * 5,
            "big": bytes(MAX_OBJECT_SIZE + 4096),
        },
    )
    raw = _assemble(cas, entry)
    assert len(raw) == entry.size_bytes

    planned = plan(_Bytes(raw), len(raw))
    assert planned.planner == SAFETENSORS_V1
    assert planned.lengths() == tuple(chunk.length for chunk in entry.chunks)
    for region, chunk in zip(planned.regions, entry.chunks, strict=True):
        digest = hashlib.sha256(raw[region.offset : region.offset + region.length]).hexdigest()
        assert digest == chunk.digest.digest


def test_the_writer_and_the_planner_agree_on_gguf_padding(tmp_path: Path) -> None:
    """GGUF's per-tensor alignment padding is its own object."""

    cas = LocalCAS(tmp_path / "cas")
    header = gguf.GGUFHeader(
        version=3,
        alignment=32,
        metadata_count=0,
        metadata=b"",
        directory_start=0,
        directory_end=0,
        data_start=0,
        tensors=(),
    )
    writer = TensorWriter(cas, "model.gguf", gguf_header=header)
    writer.add("a", "I8", (7,), b"\x01" * 7)
    writer.add("b", "I8", (13,), b"\x02" * 13)
    entry = writer.finish()

    raw = _assemble(cas, entry)
    planned = plan(_Bytes(raw), len(raw))
    assert planned.planner == GGUF_V1
    assert planned.lengths() == tuple(chunk.length for chunk in entry.chunks)
    assert 7 in planned.lengths() and 13 in planned.lengths(), (
        "a tensor's unpadded extent is its own object"
    )


def test_a_below_grid_blob_still_packs_as_one_chunkless_object() -> None:
    """The control."""

    payload = b"{\n  \"not\": \"a tensor container\"\n}\n" * 100
    assert plan(_Bytes(payload), len(payload)).planner == BLOB_V1
    assert plan_chunks(_Bytes(payload), len(payload)) == ()
    assert plan_chunks(_Bytes(b""), 0) == ()


def test_the_oversized_blob_deviation_is_the_only_one() -> None:
    """The plan is faithful; the v1-manifest ADAPTER deviates above 64 MiB, and only there."""

    size = MAX_OBJECT_SIZE * 2 + 7

    class _Sparse:

        _at = 0

        def seek(self, at: int) -> int:
            self._at = at
            return at

        def read(self, length: int) -> bytes:
            return b"\0" * min(length, size - self._at)

    sparse = cast(BinaryIO, _Sparse())
    planned = plan(sparse, size)
    assert planned.planner == BLOB_V1
    assert planned.regions == (Region(0, size, "blob"),), (
        "the PLAN stays blob-v1: one region of any size"
    )
    assert plan_chunks(cast(BinaryIO, _Sparse()), size) == (MAX_OBJECT_SIZE, MAX_OBJECT_SIZE, 7)


def test_the_duplicate_metadata_key_gap_is_named() -> None:
    """The one structural refusal this pure-Python port cannot see."""

    entry = gguf.encode_symbol(b"k") + struct.pack("<II", 4, 0)
    raw = gguf.encode_prefix(3, 0, 2) + entry + entry
    planned = plan(_Bytes(raw), len(raw))
    assert planned.planner == GGUF_V1, (
        "the duplicate-metadata-key gap closed; update the module docstring "
        "and this test together"
    )


def test_ingest_file_admits_on_the_planner_grid(tmp_path: Path) -> None:
    """`hubio/client.py::publish_v2` calls exactly this."""

    case = next(c for c in _cases() if c["name"] == "safetensors-two-tensors")
    raw = _source(case)
    source = tmp_path / "model.safetensors"
    source.write_bytes(raw)

    cas = LocalCAS(tmp_path / "cas")
    entry = ingest_file(cas, source, manifest_path="model.safetensors")
    assert tuple(chunk.length for chunk in entry.chunks) == (113, 3, 5)
    for chunk, declared in zip(entry.chunks, case["expected"]["objects"], strict=True):
        assert f"sha256:{chunk.digest.digest}" == declared["digest"]


def test_a_second_publish_moves_only_the_tensors_that_changed(tmp_path: Path) -> None:
    """The dedup consequence, measured rather than asserted."""

    small = {f"t{i}": bytes([i + 1]) * 1024 for i in range(16)}
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"

    build = LocalCAS(tmp_path / "build")
    a.write_bytes(_assemble(build, _write_safetensors(build, small, "a.safetensors")))
    changed = dict(small)
    changed["t7"] = b"\xff" * 1024
    b.write_bytes(_assemble(build, _write_safetensors(build, changed, "b.safetensors")))

    cas = LocalCAS(tmp_path / "cas")
    first = ingest_file(cas, a, manifest_path="model.safetensors")
    resident = {chunk.digest.digest for chunk in first.chunks}
    second = ingest_file(cas, b, manifest_path="model.safetensors")

    new = [chunk for chunk in second.chunks if chunk.digest.digest not in resident]
    assert [chunk.length for chunk in new] == [1024], (
        f"a one-tensor edit should move exactly that tensor; moved {new}"
    )
    tensors = list(second.chunks[1:])
    assert len(tensors) == 16, "one object per tensor"
    assert all(chunk.length == 1024 for chunk in tensors)
    assert sum(chunk.length for chunk in tensors) == 16384
    assert len([c for c in tensors if c.digest.digest in resident]) == 15, (
        "15 of the 16 tensor objects deduplicate; the retired pack shared none"
    )
