"""th#1303 empty-guard class, fail-closed half (pgw#821).

`ModelStore.component_digests` read `f.blake3` — EMPTY on every manifest-v2
entry — so every file of a v2 snapshot was skipped and component sharing
(gw#479) was silently OFF for the whole repointed corpus. Revert-turns-red:
restore the `f.blake3`-only read and the v2 test returns {} again.
"""

from __future__ import annotations

from pathlib import Path

from gen_worker.executor import ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


def _snapshot(files: list[pb.SnapshotFile]) -> pb.Snapshot:
    snap = pb.Snapshot(digest="sha256:" + "ab" * 32)
    snap.files.extend(files)
    return snap


def test_v2_snapshot_yields_component_digests(tmp_path: Path) -> None:
    """A digest-tagged (v2) snapshot must turn component sharing ON: every
    component of the snapshot appears, none silently vanishes."""
    store = ModelStore(_noop_send, cache_dir=tmp_path)
    store.bank_snapshot("o/repo", _snapshot([
        pb.SnapshotFile(path="transformer/model.safetensors", size_bytes=4,
                        digest="sha256:" + "11" * 32),
        pb.SnapshotFile(path="vae/model.safetensors", size_bytes=4,
                        digest="sha256:" + "22" * 32),
        pb.SnapshotFile(path="model_index.json", size_bytes=2,
                        digest="sha256:" + "33" * 32),
    ]))
    digests = store.component_digests("o/repo")
    assert set(digests) == {"transformer", "vae", ""}
    assert all(digests.values())


def test_v1_snapshot_still_yields_component_digests(tmp_path: Path) -> None:
    """Pre-repoint (blake3-mirror-only) snapshots keep sharing until S1
    retires the arm — the fix must be additive, not a flip."""
    store = ModelStore(_noop_send, cache_dir=tmp_path)
    store.bank_snapshot("o/legacy", _snapshot([
        pb.SnapshotFile(path="transformer/model.safetensors", size_bytes=4,
                        blake3="44" * 32),
        pb.SnapshotFile(path="vae/model.safetensors", size_bytes=4,
                        blake3="55" * 32),
    ]))
    digests = store.component_digests("o/legacy")
    assert set(digests) == {"transformer", "vae"}
    assert all(digests.values())
