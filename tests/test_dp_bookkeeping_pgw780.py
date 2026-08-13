"""pgw#780 + pgw#776/DPA-6: the per-group bookkeeping pgw#748 CLAIMED.

Each row is the cheap unit assertion the issue's acceptance names — wire it
or delete the claim. Items 3 (per-group preloader/boot warm) and 5 (per-
(function, group) disables) are recorded in the tracker as dissolved by the
pgw#783 process split (each child IS one group) and are deliberately absent
here.
"""

from __future__ import annotations

from gen_worker.models.store import ModelStore
from gen_worker.models import staging
from gen_worker.topology import ExecutionTopology


def _store() -> ModelStore:
    return ModelStore(emit=None)  # type: ignore[arg-type]


def test_bind_topology_wires_the_pinned_pool_fair_share():
    """pgw#780 item 1: `PinnedPool.set_group_count` was called nowhere in
    src/ — the cap/G fair share was dead code and group 0 could claim the
    whole pinned budget on a G=4 pod."""
    pool = staging.pinned_pool()
    assert pool.group_count == 1
    _store().bind_topology(ExecutionTopology(gpu_count=4, gpus_per_execution_group=1))
    assert pool.group_count == 4


def test_bind_topology_creates_every_groups_registry_eagerly(tmp_path):
    """pgw#780 item 2: registries were created lazily on first dispatch, so
    the boot disk re-track (a union over all_residencies()) was a no-op for
    groups 1..G-1 — their eviction/preserve views started blind."""
    store = _store()
    store.bind_topology(ExecutionTopology(gpu_count=4, gpus_per_execution_group=1))
    regs = store.all_residencies()
    assert len(regs) == 4
    # And the union walk actually sees a disk ref banked before group 3 ever
    # serves a job — the exact re-track e3d5af5 claimed.
    snap = tmp_path / "snap"
    snap.mkdir()
    store.residency_for(3).track_disk("hub:fam/model", snap)
    assert "hub:fam/model" in store.disk_refs()


def test_residency_snapshot_unions_every_group(tmp_path):
    """pgw#776/DPA-6: `residency_snapshot()` read the CURRENT group's registry,
    and on the event-loop thread that is always group 0 — the hub saw 1/G of
    the resident set, so cache-aware victims, keep-warm objectives and warm
    routing decided on a quarter of the truth."""
    store = _store()
    store.bind_topology(ExecutionTopology(gpu_count=2, gpus_per_execution_group=1))
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    store.residency_for(0).track_disk("hub:fam/zero", a)
    store.residency_for(1).track_disk("hub:fam/one", b)
    # Snapshot taken WITHOUT any group scope — the event-loop shape.
    refs = {m.ref for m in store.residency_snapshot()}
    assert refs == {"hub:fam/zero", "hub:fam/one"}


def test_residency_snapshot_merges_a_ref_at_its_best_tier(tmp_path):
    store = _store()
    store.bind_topology(ExecutionTopology(gpu_count=2, gpus_per_execution_group=1))
    d = tmp_path / "d"
    d.mkdir()
    store.residency_for(0).track_disk("hub:fam/m", d)
    reg1 = store.residency_for(1)
    reg1.track_disk("hub:fam/m", d)
    reg1.track_vram("hub:fam/m", object(), vram_bytes=7)
    rows = [m for m in store.residency_snapshot() if m.ref == "hub:fam/m"]
    assert len(rows) == 1, "one row per ref, not one per group"
    from gen_worker.pb import worker_scheduler_pb2 as pb

    assert rows[0].tier == pb.RESIDENCY_TIER_VRAM, "best tier wins the merge"
    assert rows[0].vram_bytes == 7


def test_copy_streams_are_keyed_by_device(monkeypatch):
    """pgw#780 item 4: the H2D copy stream was a device-0 singleton — a
    promote onto cuda:3 queued its copies under card 0's stream context and
    synchronized card 0 (a no-op for the card actually copying). One stream
    per device, keyed by the target."""

    class _Stream:
        def __init__(self, device=None):
            self.device = device

    class _Cuda:
        Stream = _Stream

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

    class _Device:
        def __init__(self, spec):
            if isinstance(spec, _Device):
                self.type, self.index = spec.type, spec.index
                return
            spec = str(spec)
            self.type, _, idx = spec.partition(":")
            self.index = int(idx) if idx else None

    class _Torch:
        cuda = _Cuda
        device = _Device

    import sys

    monkeypatch.setitem(sys.modules, "torch", _Torch)
    monkeypatch.setattr(staging, "_streams", {})

    s0 = staging.copy_stream("cuda:0")
    s3 = staging.copy_stream("cuda:3")
    assert s0 is not s3, "one stream per device, never a singleton"
    assert s3.device == 3, "the stream is created ON the target card"
    assert staging.copy_stream("cuda:3") is s3, "cached per device"
    assert staging.copy_stream(None) is s0, "bare 'cuda'/None = current device"
