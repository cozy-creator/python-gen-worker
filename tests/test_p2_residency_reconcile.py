"""P2 (th#960/pgw#609 design table): HelloAck full-replace DesiredResidency
-> download -> verify -> ModelEvent chain, over a real hub-double + real
blake3 blob host. Events are fenced by snapshot digest + generation (a late
event after a tag-move never misattributes to the new identity); a mutable
tag can move to new bytes under the SAME wire ref.

Not covered here (documented deviation, one line): "adopt-to-self no-op
re-arm" (gw#604/gw#607) is compile-cell/fleet_cells adoption machinery, not
model residency — its worker-side tests live in test_executor_adopt.py,
currently under a different lane's active dirty WIP per the th#960 tracker
note ("foreign WIP ... compile_cache/executor/fleet_cells ... left alone").
Duplicating that surface here risks colliding with in-flight work; the
th#960 checkpoint records this as an open follow-up for whoever owns that
lane next.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost, CorruptingBlobHost
from harness.hub_double import (
    hub_double,
    is_exact_model_event,
    is_model_event,
    is_ready,
    is_result_for,
)
from harness.toy_endpoints import EchoIn, EchoOut

_MODEL_REF = "harness/residency-tiny"
_BROKEN_REF = "harness/residency-broken"
_RAM_PIPELINE_REF = "harness/ram-pressure-pipeline"
_RAM_VAE_REF = "harness/ram-pressure-shared-vae"


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


# Module-level (not test-local): gen_worker's setup-slot construction check
# (Executor._worker_loaded_slots) resolves `from __future__ import
# annotations` string annotations via `typing.get_type_hints`, which needs
# these names in the DEFINING MODULE's globals — a test-function-local class
# resolves to nothing and silently falls back to str-path injection.
class _WeightsCorruptionIn(msgspec.Struct):
    x: str = ""


class _WeightsPipe:
    """Loads only when on-disk weights match `expected` (test-scoped via
    a per-test subclass) — the exact J17 failure shape otherwise."""

    expected: bytes = b""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        data = (Path(path) / "model.safetensors").read_bytes()
        if data != cls.expected:
            raise OSError("Unable to load weights from checkpoint file")
        return cls()

    def to(self, device):
        return self


class _WeightsEndpoint:
    def setup(self, m: _WeightsPipe) -> None:
        self.m = m

    def run(self, ctx, payload: _WeightsCorruptionIn):  # pragma: no cover
        return payload


def test_desired_residency_downloads_warms_and_serves(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        payload = b"tiny-weights"
        snapshot = blobs.one_file_snapshot("snap-1", "blob", payload)
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            ready = conn.wait_for(is_ready)
            # Gated until its model loads: present in loading, not available.
            assert "model-echo" not in ready.state_delta.available_functions
            assert "model-echo" in ready.state_delta.loading_functions

            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[_MODEL_REF],
                    hot=[pb.DesiredInstance(
                        function_name="model-echo",
                        models=[pb.ModelBinding(slot="model", ref=_MODEL_REF)],
                    )],
                    snapshots={_MODEL_REF: snapshot},
                ),
            ))
            downloading = conn.wait_for(
                is_model_event(_MODEL_REF, pb.MODEL_STATE_DOWNLOADING)
            ).model_event
            on_disk = conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK)).model_event
            in_ram = conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_IN_RAM)).model_event
            for event in (downloading, on_disk, in_ram):
                assert event.snapshot_digest == "snap-1"
                assert event.residency_generation == 1
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "model-echo" in m.state_delta.available_functions
                and m.state_delta.observed_residency_generation == 1
            )
            conn.send(run_job=pb.RunJob(
                request_id="r1", attempt=1, function_name="model-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
            res = conn.wait_for(is_result_for("r1")).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert _decode(res.inline).response == payload.decode()
    finally:
        blobs.shutdown()


def test_mutable_tag_move_fences_events_by_digest_and_generation(tmp_path) -> None:
    """A moved tag keeps the SAME wire ref but new bytes/digest; every event
    (and the resumed declarative baseline) must carry the NEW generation —
    a late event never silently downgrades to a stale identity."""
    blobs = BlobHost(tmp_path)
    try:
        payload_a = b"tiny-weights-a"
        payload_b = b"moved-tag-weights-b"
        snap_a = blobs.one_file_snapshot("snap-a", "blob-a", payload_a)
        snap_b = blobs.one_file_snapshot("snap-b", "blob-b", payload_b)

        with hub_double() as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1, disk_refs=[_MODEL_REF],
                    hot=[pb.DesiredInstance(
                        function_name="model-echo",
                        models=[pb.ModelBinding(slot="model", ref=_MODEL_REF)],
                    )],
                    snapshots={_MODEL_REF: snap_a},
                ),
            ))
            conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_IN_RAM))

            # Disk-only move to B (no hot instance): tensorhub prepositions
            # disk residency ahead of any dispatch.
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=2, disk_refs=[_MODEL_REF], snapshots={_MODEL_REF: snap_b},
                ),
            ))
            moved_disk = conn.wait_for(
                lambda m: is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK)(m)
                and m.model_event.snapshot_digest == "snap-b"
            ).model_event
            assert moved_disk.residency_generation == 2

            # A late RunJob may legitimately still ask for A while desired
            # residency has moved to B/gen2; afterward the resumed
            # declarative loop restores B WITH generation 2, not gen0.
            resumed_b = is_exact_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK, "snap-b", 2)
            b_events_before = conn.count(resumed_b)
            conn.send(run_job=pb.RunJob(
                request_id="r-priority-a", attempt=1, function_name="model-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
                snapshots={_MODEL_REF: snap_a}))
            conn.wait_for(lambda m: (
                is_model_event(_MODEL_REF, pb.MODEL_STATE_IN_RAM)(m)
                and m.model_event.snapshot_digest == "snap-a"
                and m.model_event.residency_generation == 0
            ))
            priority_a = conn.wait_for(is_result_for("r-priority-a")).job_result
            assert priority_a.status == pb.JOB_STATUS_OK
            assert _decode(priority_a.inline).response == payload_a.decode()
            conn.wait_for_count(resumed_b, b_events_before + 1)
            assert harness.worker.executor.store.resident_identity(_MODEL_REF) == ("snap-b", 2)
    finally:
        blobs.shutdown()


def test_corrupt_blob_is_never_trusted_silently(tmp_path) -> None:
    """A blob that fails its blake3 verify on download must not be served —
    the model either quarantines/retries or the function stays unavailable;
    it never silently loads mismatched bytes. Real corruption via a real
    HTTP server serving wrong bytes for the declared digest (no mocking of
    the download/verify path)."""
    blobs = CorruptingBlobHost(tmp_path, corrupt="blob")
    try:
        real_payload = b"tiny-weights-that-will-be-corrupted"
        snapshot = blobs.one_file_snapshot("snap-corrupt", "blob", real_payload)
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1, disk_refs=[_MODEL_REF],
                    hot=[pb.DesiredInstance(
                        function_name="model-echo",
                        models=[pb.ModelBinding(slot="model", ref=_MODEL_REF)],
                    )],
                    snapshots={_MODEL_REF: snapshot},
                ),
            ))
            # The verify failure must surface as a typed FAILED ModelEvent —
            # never a silent IN_RAM with mismatched bytes.
            failed_or_ram = conn.wait_for(lambda m: (
                m.WhichOneof("msg") == "model_event" and m.model_event.ref == _MODEL_REF
                and m.model_event.state in (pb.MODEL_STATE_FAILED, pb.MODEL_STATE_IN_RAM)
            ))
            assert failed_or_ram.model_event.state == pb.MODEL_STATE_FAILED, (
                "corrupted bytes must never reach MODEL_STATE_IN_RAM"
            )
    finally:
        blobs.shutdown()


# ---------------------------------------------------------------------------
# th#960/pgw#609 Phase 2b additions, absorbed before deleting their source
# files (per-issue single-purpose test files superseded by these rows).
# ---------------------------------------------------------------------------


def test_setup_failure_emits_fn_unavailable_and_recovers(tmp_path) -> None:
    """Absorbed from test_worker_grpc_e2e.py (#365/th#581): a function whose
    pipeline setup raises must not sit in loading_functions forever under a
    READY phase — it leaves BOTH lists and a terminal FnUnavailable{
    setup_failed} reaches the hub. Re-sending the same desired generation
    retries setup and re-advertises it after recovery."""
    import harness.toy_endpoints as ep

    blobs = BlobHost(tmp_path)
    ep.BREAK_SETUP.set()
    try:
        payload = b"tiny-weights"
        snapshot = blobs.one_file_snapshot("snap-broken", "blob", payload)

        def is_fn_unavailable(m: pb.WorkerMessage) -> bool:
            return (
                m.WhichOneof("msg") == "fn_unavailable"
                and m.fn_unavailable.function_name == "broken-echo"
                and m.fn_unavailable.reason == "setup_failed"
            )

        with hub_double() as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            ready = conn.wait_for(is_ready)
            assert "broken-echo" in ready.state_delta.loading_functions

            desired = pb.DesiredResidency(
                generation=1, disk_refs=[_BROKEN_REF],
                hot=[pb.DesiredInstance(
                    function_name="broken-echo",
                    models=[pb.ModelBinding(slot="model", ref=_BROKEN_REF)],
                )],
                snapshots={_BROKEN_REF: snapshot},
            )
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT, desired_residency=desired,
            ))
            sig = conn.wait_for(is_fn_unavailable).fn_unavailable
            assert "pipeline exploded" in sig.detail
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "broken-echo" not in m.state_delta.available_functions
                and "broken-echo" not in m.state_delta.loading_functions
            )
            conn.wait_for(is_model_event(_BROKEN_REF, pb.MODEL_STATE_FAILED))

            # Same-generation full replacement is a retry; setup succeeds
            # once the toggle clears.
            ep.BREAK_SETUP.clear()
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT, desired_residency=desired,
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "broken-echo" in m.state_delta.available_functions
                and m.state_delta.observed_residency_generation == 1
            )
            assert "broken-echo" not in harness.worker.executor.unavailable

            conn.send(run_job=pb.RunJob(
                request_id="r-broken", attempt=1, function_name="broken-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
            res = conn.wait_for(is_result_for("r-broken")).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert _decode(res.inline).response == payload.decode()
    finally:
        ep.BREAK_SETUP.set()
        blobs.shutdown()


def test_host_ram_failure_precedes_retryable_result_on_wire(tmp_path, monkeypatch) -> None:
    """Absorbed from test_worker_grpc_e2e.py (th#807): host-RAM admission
    failure crosses the real worker transport BEFORE the retry result. Only
    the largest staged ref fails; a smaller shared ref stays usable."""
    from gen_worker.models import disk_gc
    from gen_worker.models import residency as residency_mod

    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 8.0)

    blobs = BlobHost(tmp_path)
    try:
        pipeline_payload = b"large-pipeline"
        vae_payload = b"small-shared-vae"
        pipeline_snap = blobs.one_file_snapshot("ram-pipeline", "pipeline", pipeline_payload)
        vae_snap = blobs.one_file_snapshot("ram-vae", "vae", vae_payload)

        def _tree_bytes(path) -> int:
            data = (path / "model.safetensors").read_bytes()
            return (6 if data == pipeline_payload else 1) * 1024**3

        monkeypatch.setattr(disk_gc, "tree_bytes", _tree_bytes)

        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-host-ram", attempt=1, function_name="ram-pressure",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
                snapshots={_RAM_PIPELINE_REF: pipeline_snap, _RAM_VAE_REF: vae_snap},
            ))
            result = conn.wait_for(is_result_for("r-host-ram")).job_result
            assert result.status == pb.JOB_STATUS_RETRYABLE

            received = list(conn.received)
            failed = [
                (i, m.model_event) for i, m in enumerate(received)
                if m.WhichOneof("msg") == "model_event"
                and m.model_event.state == pb.MODEL_STATE_FAILED
            ]
            assert [(e.ref, e.error) for _, e in failed] == [
                (_RAM_PIPELINE_REF, "insufficient_host_ram"),
            ]
            result_index = next(
                i for i, m in enumerate(received)
                if m.WhichOneof("msg") == "job_result" and m.job_result.request_id == "r-host-ram"
            )
            assert failed[0][0] < result_index, "the FAILED event must precede the retryable result"
            assert all(e.ref != _RAM_VAE_REF for _, e in failed)
    finally:
        blobs.shutdown()


def test_corrupt_load_failure_refetches_and_retries_once(tmp_path, monkeypatch) -> None:
    """Absorbed from test_snapshot_corruption.py (gw#408, J17 flood: a pod-
    churn-interrupted write left truncated safetensors trusted forever). A
    corruption-shaped load failure digest-verifies, quarantines, re-
    downloads, and retries ONCE — setup succeeds on the second attempt.
    Network boundary faked (real ModelStore/Executor, no hub_double needed
    here — this bug lives entirely below the wire)."""
    import asyncio
    import json
    import struct

    from blake3 import blake3

    import gen_worker.models.cozy_snapshot as snap_mod
    from gen_worker.api.binding import Hub as HubRef
    from gen_worker.api.decorators import Resources
    from gen_worker.executor import Executor, ModelStore
    from gen_worker.registry import EndpointSpec

    def _tiny_safetensors(tag: bytes = b"\x00\x01\x02\x03") -> bytes:
        header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(tag)]}}
        hb = json.dumps(header, separators=(",", ":")).encode()
        return struct.pack("<Q", len(hb)) + hb + tag

    content = _tiny_safetensors()
    digest = blake3(content).hexdigest()
    snap = pb.Snapshot(digest="corrupt-quarantine", files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=len(content), blake3=digest,
        url="http://example.invalid/blob",
    )])
    calls = {"n": 0}

    async def _fake_dl(url, dst, expected_size, expected_blake3, on_bytes=None):
        calls["n"] += 1
        dst.write_bytes(content)

    monkeypatch.setattr(snap_mod, "_download_one_file", _fake_dl)
    monkeypatch.setattr(_WeightsPipe, "expected", content)

    ref = "harness/snapshot-corruption-quarantine"
    spec = EndpointSpec(
        name="ep", method=_WeightsEndpoint.run, kind="inference",
        payload_type=_WeightsCorruptionIn, output_mode="single", cls=_WeightsEndpoint,
        attr_name="run", models={"m": HubRef(ref)}, resources=Resources(),
    )

    async def _run() -> None:
        store = ModelStore(lambda m: asyncio.sleep(0), cache_dir=tmp_path)
        ex = Executor([spec], lambda m: asyncio.sleep(0), store=store)
        path = await store.ensure_local(ref, snap)
        (path / "model.safetensors").write_bytes(b"garbage-that-parses-as-nothing")
        store._verified.discard(ref)

        inst = await ex.ensure_setup(spec, {ref: snap})
        assert isinstance(inst.m, _WeightsPipe)
        assert (path / "model.safetensors").read_bytes() == content

    asyncio.run(_run())
    assert calls["n"] == 2, "quarantine must trigger exactly one re-download"
