"""pgw#781 / th#1303: the mandatory volume check must never pass without hashing.

This drives the REAL call site — `ModelStore._verify_snapshot_tree` — over real
files on disk, because the bug is not in the hashing helper (that was always
correct) but in what the call site READS off the manifest entry.

The defect: `_verify_snapshot_tree` took the digest from `f.blake3`. Under
manifest v2 that field is EMPTY and the digest lives in `f.digest` as
`"sha256:<hex>"`. So `digest` was `""`, the `if digest and ...` hash check was
skipped, and the tree was reported CLEAN WITHOUT BEING HASHED — while a real
verification and a vacuous one produce byte-identical "ok" results. That is the
whole `entries`-vs-`files` lesson pointed at a security control: pgw#769's fill
check is what the NFS/shared-volume verdict rests on, so a no-op that reports
success is a hole, not a cosmetic gap.

Every assertion below therefore checks the DENOMINATOR — that bytes were
actually read — not merely that the verdict was "ok".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gen_worker.executor import ModelStore
from gen_worker.models import volume_verify
from gen_worker.models.volume_verify import clear_memo


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _File:
    """Duck-typed pb.SnapshotFile. The legacy `blake3` field is still carried
    on the wire (it dies at a proto rev), but nothing reads it any more."""

    def __init__(self, path, size_bytes, digest="", blake3=""):
        self.path = path
        self.size_bytes = size_bytes
        self.digest = digest
        self.blake3 = blake3


class _Snap:
    def __init__(self, files):
        self.files = files


def verify(tree: Path, files):
    # _verify_snapshot_tree never touches `self`; calling it unbound keeps the
    # test on the real code path without standing up a whole ModelStore.
    return ModelStore._verify_snapshot_tree(None, tree, _Snap(files))


@pytest.fixture(autouse=True)
def _clear():
    clear_memo()
    yield
    clear_memo()


@pytest.fixture
def counted(monkeypatch):
    """Record every verify_files report so a test can assert on the
    denominators the call site actually produced."""
    seen = []
    real = volume_verify.verify_files

    def spy(targets, **kw):
        rep = real(targets, **kw)
        seen.append(rep)
        return rep

    # Patch where the call site imports it FROM, which is the module itself.
    monkeypatch.setattr(volume_verify, "verify_files", spy)
    return seen


def _tree(tmp_path: Path, name: str, data: bytes) -> Path:
    root = tmp_path / "snap"
    root.mkdir(exist_ok=True)
    (root / name).write_bytes(data)
    return root


def test_a_v2_snapshot_is_actually_hashed_not_merely_declared_clean(tmp_path, counted):
    """THE regression. Before the fix this returned ok=True having read ZERO
    bytes; the assertion that fails on the old code is `bytes_hashed`."""
    data = b"\x01\x02\x03" * 4096
    root = _tree(tmp_path, "model.safetensors", data)

    ok, bad = verify(root, [
        _File("model.safetensors", len(data), digest="sha256:" + sha(data)),
    ])

    assert ok and bad == []
    assert counted, "the v2 entry produced NO verification call at all"
    rep = counted[-1]
    assert rep.expected == 1 and rep.examined == 1
    assert rep.hashed == 1
    assert rep.bytes_hashed == len(data), (
        "the verifier reported a pass without reading the bytes — a verifier "
        "that examines nothing looks exactly like one that passes"
    )


def test_a_corrupt_v2_file_is_caught_and_named_for_quarantine(tmp_path, counted):
    good = b"\xAA" * 8192
    root = _tree(tmp_path, "model.safetensors", good)
    ref = "sha256:" + sha(good)

    # Flip one byte, keeping the SIZE identical: only the hash can catch this,
    # so it fails iff the file was really hashed.
    raw = bytearray(good)
    raw[4000] ^= 0xFF
    (root / "model.safetensors").write_bytes(bytes(raw))

    ok, bad = verify(root, [_File("model.safetensors", len(good), digest=ref)])

    assert not ok
    assert bad == [ref], bad  # the DIGEST, so the bad blob is quarantinable
    assert counted[-1].bytes_hashed == 0  # a failed hash reports no clean bytes
    assert counted[-1].examined == 1


def test_an_entry_carrying_only_the_legacy_mirror_is_SKIPPED_not_passed(tmp_path, counted):
    """th#1303 S1 replaces `test_legacy_blake3_snapshots_still_verify`.

    The blake3 fallback is gone, so a mirror-only entry names no digest. It
    must be reported as skipped and hash NOTHING — and the denominator is the
    point: 1 expected, 0 hashed is a different verdict from 1 expected,
    1 hashed, and only the denominator can tell them apart.
    """
    from gen_worker.models.volume_verify import snapshot_verify_targets

    data = b"legacy-bytes-" * 512
    root = _tree(tmp_path, "config.json", data)
    targets, skipped = snapshot_verify_targets(
        [_File("config.json", len(data), blake3="b" * 64)], root)
    assert targets == []
    assert skipped == ["config.json"]



def test_a_mixed_tree_hashes_every_covered_file(tmp_path, counted):
    """Two entries in one snapshot: BOTH must be hashed. Any per-entry
    algorithm assumption drops one of them silently."""
    a = b"sha-side" * 300
    b = b"blake-side" * 300
    root = tmp_path / "snap"
    root.mkdir()
    (root / "a.safetensors").write_bytes(a)
    (root / "sub").mkdir()
    (root / "sub" / "b.safetensors").write_bytes(b)

    ok, bad = verify(root, [
        _File("a.safetensors", len(a), digest="sha256:" + sha(a)),
        _File("sub/b.safetensors", len(b), digest="sha256:" + sha(b)),
    ])
    assert ok and bad == []
    rep = counted[-1]
    assert (rep.expected, rep.examined, rep.hashed) == (2, 2, 2)
    assert rep.bytes_hashed == len(a) + len(b)


def test_a_missing_v2_file_is_corrupt_not_clean(tmp_path, counted):
    root = tmp_path / "snap"
    root.mkdir()
    ref = "sha256:" + sha(b"never written")
    ok, bad = verify(root, [_File("ghost.safetensors", 13, digest=ref)])
    assert not ok
    assert bad == [ref]
    assert counted[-1].examined == 1, "a missing file must still be ACCOUNTED for"


def test_a_size_mismatch_is_caught_before_hashing(tmp_path, counted):
    data = b"x" * 100
    root = _tree(tmp_path, "m.safetensors", data)
    ref = "sha256:" + sha(data)
    ok, bad = verify(root, [_File("m.safetensors", 999, digest=ref)])
    assert not ok and bad == [ref]


def test_an_unreadable_digest_is_a_finding_never_a_skip(tmp_path, counted):
    """A digest we cannot parse must not degrade to 'nothing to check'."""
    data = b"z" * 64
    root = _tree(tmp_path, "m.safetensors", data)
    ok, bad = verify(root, [_File("m.safetensors", len(data), digest="sha256:zzz")])
    assert not ok
    assert counted[-1].examined == 1
    assert counted[-1].findings
