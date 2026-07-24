"""pgw#636 hot-GPU mandate — worker half.

Paul's contract: ``Resources(vram_gb=N)`` is a placement MINIMUM ("a machine
with at least N GB"), never a usage cap; the worker packs real card capacity
with checkpoints (all resident where they fit), serves resident models NOW
while absent models download in the background, and LRU-evicts exclusive
entries (per-pick UNets) before genuinely shared components (TE/VAE).

Layers under test, all real logic (fakes only at the torch module boundary):
1. ``_estimate_setup_need`` — the pre-load headroom ask that used to reserve
   the declared ``vram_gb`` wholesale for every never-seen pick (the live
   9.8/24 GB one-pipeline incident).
2. ``Residency`` holders semantics — shared components are demotable while
   idle, never evictable while referenced, and multi-holder entries sort
   last in LRU victim order.
3. The packing juggle: a 24 GB budget holds several ~5 GB picks hot; only
   real pressure demotes the LRU one (planner logic real; modules faked).
4. ``Slot.share_components`` declaration surface.
5. Serve-while-downloading over the real worker + hub-double: a staged
   background download completes WHILE a tenant job holds the worker busy
   (the old tenant-idle gate + run_job preemption starved staged downloads
   at 0%% for minutes).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import msgspec
import pytest

from gen_worker import Slot
from gen_worker.executor import _estimate_setup_need
from gen_worker.models.residency import (
    LoadedComponentKey,
    Residency,
    Tier,
)
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness import toy_endpoints
from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_model_event, is_ready, is_result_for
from harness.toy_endpoints import EchoIn, EchoOut

_GiB = 1024 ** 3


# ---------------------------------------------------------------------------
# 1. _estimate_setup_need: vram_gb is a placement hint, never the ask.
# ---------------------------------------------------------------------------


def test_estimate_prefers_measured_hint_over_everything() -> None:
    # (hint, snapshot_bytes): the measured 5.1 GiB footprint wins.
    need = _estimate_setup_need([(int(5.1 * _GiB), int(4.7 * _GiB))], 12.0)
    assert need == int(5.1 * _GiB)


def test_estimate_uses_snapshot_bytes_for_never_seen_picks() -> None:
    # The live incident shape: a never-seen 4.7 GB sdxl pick + a tiny VAE
    # with a prior hint. The old code reserved max(sum, 12 GiB) = 12 GiB and
    # evicted the resident pipeline on a 24 GB card; the honest ask is the
    # snapshot byte total.
    need = _estimate_setup_need(
        [(0, int(4.7 * _GiB)), (int(0.2 * _GiB), 0)], 12.0
    )
    assert need == int(4.7 * _GiB) + int(0.2 * _GiB)
    assert need < 12 * _GiB


def test_estimate_falls_back_to_declared_only_when_no_facts() -> None:
    assert _estimate_setup_need([(0, 0)], 12.0) == 12 * _GiB
    # No declaration either: 0 (make_room becomes a no-op; the load ladder
    # still sizes honestly at load time).
    assert _estimate_setup_need([(0, 0)], 0.0) == 0


# ---------------------------------------------------------------------------
# 2 + 3. Residency holders semantics and the packing juggle.
# ---------------------------------------------------------------------------


class _FakeModule:
    """Torch-boundary fake: movable, no parameters (device walks no-op)."""

    def __init__(self) -> None:
        self.device = "cuda"

    def to(self, device: str) -> "_FakeModule":
        self.device = device
        return self


def _budget_residency(gib: int) -> Residency:
    moved: List[Tuple[Any, str]] = []

    def move(obj: Any, device: str) -> None:
        moved.append((obj, device))
        to = getattr(obj, "to", None)
        if callable(to):
            to(device)

    return Residency(vram_budget_bytes=gib * _GiB, move_fn=move)


def _key(name: str, digest: str) -> LoadedComponentKey:
    return LoadedComponentKey.for_component(
        content_digest=digest, component=name, label=name
    )


def test_shared_holders_do_not_block_demote_but_block_evict() -> None:
    res = _budget_residency(24)
    key = _key("text_encoder", "d" * 32)
    obj = _FakeModule()
    got = res.acquire_shared(key, lambda: obj, vram_bytes=2 * _GiB)
    assert got is obj
    assert res.shared_refcount(key) == 1
    ref = key.cache_id()

    # Referenced by one record, idle: demotable under pressure.
    assert res.demote(ref) is True
    assert res.tier(ref) is Tier.RAM
    assert obj.device == "cpu"
    # ...and promotable back before the owner executes.
    assert res.promote(ref) is True
    assert res.tier(ref) is Tier.VRAM
    assert obj.device == "cuda"

    # The registry must never drop its handle while records alias the module.
    assert res.evict(ref) is False
    assert res.release_to_disk(ref) is False
    assert res.release_shared(key) == 0
    assert res.evict(ref) is True


def test_executing_pin_blocks_demote_of_shared_component() -> None:
    res = _budget_residency(24)
    key = _key("vae", "e" * 32)
    res.acquire_shared(key, _FakeModule, vram_bytes=_GiB)
    ref = key.cache_id()
    with res.executing(ref):
        assert res.demote(ref) is False
    assert res.demote(ref) is True


def test_lru_victims_order_multi_holder_shared_last() -> None:
    res = _budget_residency(24)
    shared = _key("text_encoder", "f" * 32)
    res.acquire_shared(shared, _FakeModule, vram_bytes=2 * _GiB)
    res.acquire_shared(shared, lambda: pytest.fail("must hit"))  # 2nd holder
    assert res.shared_refcount(shared) == 2
    # Exclusive UNet lanes, touched AFTER the shared entry (more recent).
    res.track_vram("unet-a", _FakeModule(), vram_bytes=3 * _GiB)
    res.track_vram("unet-b", _FakeModule(), vram_bytes=3 * _GiB)
    victims = res.lru_vram_victims()
    # Both exclusive lanes come first despite the shared entry being LRU.
    assert victims[-1] == shared.cache_id()
    assert set(victims[:-1]) == {"unet-a", "unet-b"}


def test_release_shared_leaves_entry_resident_for_the_next_pick() -> None:
    # pgw#636: no eager drain — a hot GPU keeps components resident; the
    # next pick with equal bytes aliases them for free.
    res = _budget_residency(24)
    key = _key("text_encoder", "a" * 32)
    obj = _FakeModule()
    res.acquire_shared(key, lambda: obj, vram_bytes=2 * _GiB)
    assert res.release_shared(key) == 0
    assert res.tier(key.cache_id()) is Tier.VRAM  # still hot
    # Next pick, same bytes: pure hit.
    assert res.acquire_shared(key, lambda: pytest.fail("must hit")) is obj
    stats = res.shared_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    # drain reclaims only unreferenced entries.
    assert res.release_shared(key) == 0
    assert res.drain_shared() == 1


def test_packing_juggle_24gb_card_holds_multiple_picks() -> None:
    """The mandate's core arithmetic on REAL make_room logic: ~5 GB picks
    pack a 24 GiB card until real pressure, then the LRU pick demotes —
    never the multi-holder shared entry, never the executing pick."""
    res = _budget_residency(24)
    picks = ["pick-a", "pick-b", "pick-c", "pick-d"]
    objs = {}
    for pick in picks:
        need = _estimate_setup_need([(0, 5 * _GiB)], 12.0)
        assert need == 5 * _GiB  # honest per-pick ask, not 12 GiB
        assert res.make_room(need) is True
        objs[pick] = _FakeModule()
        res.track_vram(pick, objs[pick], vram_bytes=5 * _GiB)

    # Four picks hot: 20 GiB of 24 in use — nothing was evicted to get here.
    assert [res.tier(p) for p in picks] == [Tier.VRAM] * 4
    assert res.free_vram_bytes() == 4 * _GiB

    # Fifth pick: 5 + 2 margin > 4 free -> exactly ONE LRU victim demotes.
    res.touch("pick-a")  # a is hot again; b becomes LRU
    assert res.make_room(5 * _GiB) is True
    assert res.tier("pick-b") is Tier.RAM
    assert [res.tier(p) for p in ("pick-a", "pick-c", "pick-d")] == [Tier.VRAM] * 3
    res.track_vram("pick-e", _FakeModule(), vram_bytes=5 * _GiB)
    assert res.tier("pick-e") is Tier.VRAM

    # Warm swap back: pick-b promotes when room allows (b's 5 GiB demand
    # demotes the new LRU, pick-c).
    assert res.promote("pick-b") is True
    assert res.tier("pick-b") is Tier.VRAM
    assert objs["pick-b"].device == "cuda"


# ---------------------------------------------------------------------------
# 4. Slot.share_components declaration.
# ---------------------------------------------------------------------------


def test_slot_share_components_declaration() -> None:
    class _Pipe:
        @classmethod
        def from_pretrained(cls, path: str) -> "_Pipe":  # pragma: no cover
            return cls()

    slot = Slot(_Pipe, share_components=("text_encoder", " vae ", ""))
    assert slot.share_components == ("text_encoder", "vae")
    with pytest.raises(ValueError, match="duplicate"):
        Slot(_Pipe, share_components=("vae", "vae"))
    assert Slot(_Pipe).share_components == ()


# ---------------------------------------------------------------------------
# 5. Serve-while-downloading (real worker + hub-double + real blob host).
# ---------------------------------------------------------------------------

_STAGED_REF = "harness/pgw636-staged-checkpoint"


def test_staged_download_completes_while_tenant_job_runs(tmp_path) -> None:
    """A runnable tenant job must never starve a staged background
    download (and vice versa): with `hold` provably in flight, a HelloAck
    disk staging still reaches ON_DISK before the job finishes. Under the
    pre-pgw#636 behavior (tenant-idle gate + run_job preemption of the
    reconcile loop) the ON_DISK wait times out — revert-red verified."""
    blobs = BlobHost(tmp_path)
    toy_endpoints.HOLD_STARTED.clear()
    toy_endpoints.HOLD_RELEASE.clear()
    try:
        snapshot = blobs.one_file_snapshot("snap-636", "blob", b"pgw636-weights")
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            # Occupy the worker with a job that will NOT finish on its own.
            conn.send(run_job=pb.RunJob(
                request_id="r-hold", attempt=1, function_name="hold",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
            assert toy_endpoints.HOLD_STARTED.wait(timeout=15.0)

            # Stage a background disk download mid-job.
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[_STAGED_REF],
                    snapshots={_STAGED_REF: snapshot},
                ),
            ))
            on_disk = conn.wait_for(
                is_model_event(_STAGED_REF, pb.MODEL_STATE_ON_DISK),
                timeout=15.0,
            ).model_event
            assert on_disk.snapshot_digest == "snap-636"
            # The job is still in flight — the download did not wait for idle.
            assert not any(
                m.WhichOneof("msg") == "job_result"
                and m.job_result.request_id == "r-hold"
                for m in conn.received
            )

            toy_endpoints.HOLD_RELEASE.set()
            res = conn.wait_for(is_result_for("r-hold")).job_result
            assert res.status == pb.JOB_STATUS_OK
            out = msgspec.msgpack.decode(res.inline, type=EchoOut)
            assert out.response == "released"
    finally:
        toy_endpoints.HOLD_RELEASE.set()
        blobs.shutdown()


# ---------------------------------------------------------------------------
# 6. THE JUGGLE: N checkpoints hot in ONE dynamic slot (real worker end to end).
# ---------------------------------------------------------------------------


def _juggle(conn, rid: str, ref: str, snap) -> str:
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name="juggle-echo",
        input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
        models=[pb.ModelBinding(slot="pipeline", ref=ref)],
        snapshots={ref: snap},
    ))
    res = conn.wait_for(is_result_for(rid), timeout=30.0).job_result
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    return msgspec.msgpack.decode(res.inline, type=EchoOut).response


def test_four_checkpoints_stay_hot_in_one_dynamic_slot(tmp_path) -> None:
    """pgw#636 / pgw#641 Stage 1, end to end through the REAL worker.

    One `selected_by="model"` slot, four different checkpoints, interleaved.
    Every instance must be RETAINED: the second round of the same four picks
    runs zero additional `setup()` calls, each pick still serves its OWN
    bytes, and all four records stay ready simultaneously. Under a
    one-instance-per-function retention policy the second round would re-setup
    all four (setups=8) and each hop would be a cold reload.
    """
    blobs = BlobHost(tmp_path)
    toy_endpoints.JUGGLE_SETUPS.clear()
    picks = {}
    for name in ("alpha", "beta", "gamma", "delta"):
        picks[name] = (
            f"harness/juggle-{name}:prod",
            blobs.snapshot(f"snap-{name}", [blobs.file(
                f"w-{name}", f"weights-{name}".encode(),
                path_in_snapshot="transformer/weights.txt")]),
        )
    try:
        with hub_double() as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            executor = harness.worker.executor

            for i, (name, (ref, snap)) in enumerate(picks.items()):
                assert _juggle(conn, f"r1-{i}", ref, snap) == (
                    f"weights-{name}|setups={i + 1}")
            assert toy_endpoints.JUGGLE_SETUPS == [
                "weights-alpha", "weights-beta", "weights-gamma", "weights-delta"]

            # Four distinct instance records, all ready at once — nothing was
            # rehomed onto one live instance, nothing was torn down: there was
            # no VRAM pressure, so retention is the ONLY correct outcome.
            def _juggle_records():
                return [
                    r for r in executor._classes.values()
                    if r.ready and r.cls is toy_endpoints.JuggleEndpoint
                ]

            ready = _juggle_records()
            assert len(ready) == 4, executor._classes
            assert len({id(r.instance) for r in ready}) == 4
            assert {r.instance.pipe.weights for r in ready} == {
                f"weights-{n}" for n in picks}

            # Round two, same four picks, reverse order: every one is a warm
            # hit — no reload, no re-setup, and no cross-talk between picks.
            for i, name in enumerate(reversed(list(picks))):
                ref, snap = picks[name]
                assert _juggle(conn, f"r2-{i}", ref, snap) == (
                    f"weights-{name}|setups=4")
            assert len(toy_endpoints.JUGGLE_SETUPS) == 4
            assert len(_juggle_records()) == 4
    finally:
        blobs.shutdown()
