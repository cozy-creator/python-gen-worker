"""The weight download's byte POSITION, off the real fetch loop."""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gen_worker import activity, weight_position
from gen_worker import config as gw_config
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef, WireRef, normalize_model_ref
from gen_worker.models.store import ModelStore
from gen_worker.serving.reserved_repos import materialize_reserved_inputs_async
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.weight_position import MIB, FetchPosition

OBJECT_BYTES = 3 * MIB // 2
OBJECTS = 6


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        key = self.path.rsplit("/", 1)[-1]
        body = self.server.blobs.get(key)  # type: ignore[attr-defined]
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        chunk = len(body) // 4 or len(body)
        for start in range(0, len(body), chunk):
            self.wfile.write(body[start:start + chunk])
            self.wfile.flush()
            time.sleep(0.01)


class _Origin:

    def __init__(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.blobs = {}  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def put(self, data: bytes) -> str:
        digest = _sha(data)
        self.server.blobs[digest] = data  # type: ignore[attr-defined]
        host, port = self.server.server_address[:2]
        return f"http://{host!s}:{port!s}/{digest}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class _Wire:

    def __init__(self) -> None:
        self.updates: list[pb.ActivityUpdate] = []
        self.model_events: list[pb.ModelEvent] = []

    async def send(self, msg: pb.WorkerMessage) -> None:
        which = msg.WhichOneof("msg")
        if which == "activity_update":
            self.updates.append(msg.activity_update)
        elif which == "model_event":
            self.model_events.append(msg.model_event)

    def positions(self) -> list[pb.ActivityUpdate]:
        return [u for u in self.updates if u.kind == activity.KIND_WEIGHT_FETCH]

    def open_download_records(self) -> dict[str, pb.ModelEvent]:
        """The hub's own bookkeeping rule, replayed over what we emitted."""
        open_rows: dict[str, pb.ModelEvent] = {}
        for event in self.model_events:
            if event.state == pb.MODEL_STATE_DOWNLOADING:
                open_rows[event.ref] = event
            elif event.state in (
                pb.MODEL_STATE_ON_DISK,
                pb.MODEL_STATE_FAILED,
                pb.MODEL_STATE_EVICTED,
            ):
                open_rows.pop(event.ref, None)
        return open_rows


def _detail(update: pb.ActivityUpdate) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", update.detail))


def _resolved(files: list[tuple[str, bytes, str]]) -> WorkerResolvedRepo:
    fingerprint = hashlib.sha256(
        b"|".join(f"{path}:{_sha(body)}".encode() for path, body, _url in files)
    ).hexdigest()
    return WorkerResolvedRepo(
        snapshot_digest="sha256:" + fingerprint,
        files=[
            WorkerResolvedRepoFile(path, len(body), url, digest=_sha(body))
            for path, body, url in files
        ],
    )


def _pull(wire: _Wire, base: Path, repo: str, resolved: WorkerResolvedRepo,
          position: FetchPosition) -> None:

    async def run() -> None:
        activity.bind_sink(wire.send, asyncio.get_running_loop())
        position.open()
        try:
            await ensure_snapshot_async(
                base_dir=base,
                ref=TensorhubRef(owner="acme", repo=repo, release="latest"),
                resolved=resolved,
                progress=position.progress,
            )
        finally:
            position.close(ok=True)
        for _ in range(4):
            await asyncio.sleep(0)

    asyncio.run(run())


@pytest.fixture(autouse=True)
def _clean_sink() -> Any:
    activity.reset_for_tests()
    yield
    activity.reset_for_tests()


@pytest.fixture()
def _fine_cadence(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(weight_position, "STRIDE_MIB", 1)
    monkeypatch.setattr(weight_position, "MIN_INTERVAL_S", 0.0)
    yield


def test_the_position_advances_while_the_fetch_runs(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """THE FACT rental #6 did not have."""
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        resolved = _resolved([(p, b, origin.put(b)) for p, b in blobs])
        wire = _Wire()
        position = FetchPosition("tensorhub:acme/model-a@latest",
                                 total_bytes=OBJECTS * OBJECT_BYTES)
        _pull(wire, tmp_path / "pod", "model-a", resolved, position)

        rows = wire.positions()
        phases = [r.phase for r in rows]
        assert phases[0] == weight_position.PHASE_STARTED
        assert phases[-1] == weight_position.PHASE_FETCHED
        steps = [r.step for r in rows]
        assert steps[0] == 0, "the opening row states position zero"
        assert steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB
        assert len(rows) >= 4, f"a 9 MiB trickle left only {len(rows)} position(s)"
        running = [r.step for r in rows if r.phase != weight_position.PHASE_FETCHED]
        assert all(b > a for a, b in zip(running, running[1:])), (
            f"positions must strictly increase; got {steps}")
        assert steps[-1] >= running[-1]
        assert steps[-1] > steps[0]

        for row in rows:
            assert row.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
            got = _detail(row)
            assert got["ref"] == "tensorhub:acme/model-a@latest"
            assert int(got["pos_mib"]) == row.step
            assert int(got["total_mib"]) == row.total_steps
    finally:
        origin.close()


def test_a_sub_mib_fetch_still_leaves_rows_at_zero(tmp_path: Path) -> None:
    """Absence renders as a row that says zero, at the PRODUCTION cadence (no fixture here)."""
    origin = _Origin()
    try:
        body = b"c" * 4096
        resolved = _resolved([("config/model.safetensors", body, origin.put(body))])
        wire = _Wire()
        position = FetchPosition("tensorhub:acme/tiny@latest", total_bytes=len(body))
        _pull(wire, tmp_path / "pod", "tiny", resolved, position)

        rows = wire.positions()
        assert [r.phase for r in rows] == [
            weight_position.PHASE_STARTED, weight_position.PHASE_FETCHED]
        assert [r.step for r in rows] == [0, 0]
        assert [r.total_steps for r in rows] == [0, 0]
    finally:
        origin.close()


def test_the_position_is_integral_mib_never_a_fraction() -> None:
    position = FetchPosition("tensorhub:acme/big@latest", total_bytes=7 * 1024 * MIB)
    for done, expected in (
        (0, 0),
        (MIB - 1, 0),
        (MIB, 1),
        (4212 * MIB + 900_000, 4212),
        (7 * 1024 * MIB, 7168),
    ):
        position._pos_bytes = done
        assert position.position_mib == expected
        assert isinstance(position.position_mib, int)


def test_a_frozen_or_regressing_position_emits_nothing_new(_fine_cadence: Any) -> None:
    """The hub advances on STRICT INCREASE, so the worker must never manufacture an increase."""
    wire: list[pb.ActivityUpdate] = []
    original = weight_position.FetchPosition._emit

    def spy(self: FetchPosition, phase: str) -> None:
        wire.append(pb.ActivityUpdate(phase=phase, step=self.position_mib))
        original(self, phase)

    position = FetchPosition("tensorhub:acme/model-a@latest", total_bytes=8 * MIB)
    try:
        weight_position.FetchPosition._emit = spy  # type: ignore[method-assign]
        position.open()
        position.progress(4 * MIB, 8 * MIB)
        emitted = len(wire)
        position.progress(4 * MIB, 8 * MIB)
        position.progress(4 * MIB + MIB - 1, 8 * MIB)
        position.progress(1 * MIB, 8 * MIB)
        assert len(wire) == emitted, f"{wire[emitted:]} should not have been emitted"
        position.progress(6 * MIB, 8 * MIB)
        assert [u.step for u in wire] == [0, 4, 6]
    finally:
        weight_position.FetchPosition._emit = original  # type: ignore[method-assign]


_REF = WireRef("acme/model-a")


def _pb_snapshot(files: list[tuple[str, bytes, str]]) -> pb.Snapshot:
    resolved = _resolved(files)
    return pb.Snapshot(
        digest=resolved.snapshot_digest,
        files=[
            pb.SnapshotFile(
                path=f.path, size_bytes=f.size_bytes, digest=f.digest, url=f.url,
            )
            for f in resolved.files
        ],
    )


def _store(wire: _Wire, cas: Path) -> ModelStore:
    return ModelStore(wire.send, cache_dir=cas)


def test_a_real_fetch_opens_advances_and_closes_the_record(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """THE CONTROL, and it must keep passing: a genuine transfer through `ModelStore` still opens a download record, advances positions while the bytes move, and closes the record with ON_DISK."""
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            await store.ensure_local(_REF, snapshot)
            for _ in range(8):
                await asyncio.sleep(0)

        asyncio.run(run())

        states = [e.state for e in wire.model_events]
        assert pb.MODEL_STATE_DOWNLOADING in states, (
            f"a real transfer must declare itself; got {states}")
        assert pb.MODEL_STATE_ON_DISK in states
        assert wire.open_download_records() == {}, (
            "the record a completed fetch opened must be closed")
        steps = [r.step for r in wire.positions()]
        assert steps and steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB
        assert steps[-1] > steps[0]
    finally:
        origin.close()


def test_a_resident_ref_opens_no_download_record(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            await store.ensure_local(_REF, snapshot)
            for _ in range(8):
                await asyncio.sleep(0)
            assert store.residency.tier(_REF) is not None
            wire.updates.clear()
            wire.model_events.clear()

            store.disk_local_path = lambda ref: None  # type: ignore[method-assign]
            store._verified.discard(_REF)
            await store.ensure_local(_REF, snapshot)
            for _ in range(8):
                await asyncio.sleep(0)

        asyncio.run(run())

        assert wire.open_download_records() == {}, (
            "a ref the pod ALREADY HOLDS must not leave a `downloading` record: "
            f"{[(e.ref, e.bytes_done, e.bytes_total) for e in wire.open_download_records().values()]}"
        )
        assert not [
            e for e in wire.model_events
            if e.state == pb.MODEL_STATE_DOWNLOADING
        ], "a transfer that moves no bytes must not be declared at all"
        phases = [r.phase for r in wire.positions()]
        assert weight_position.PHASE_ALREADY_RESIDENT in phases, phases
    finally:
        origin.close()


def test_the_record_opens_LAZILY_when_the_resident_check_was_wrong(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """THE SAFETY VALVE, and it must be exercised, not asserted about."""
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store.residency.track_ram(_REF, object())
            wire.model_events.clear()
            wire.updates.clear()
            await store.ensure_local(_REF, snapshot)
            for _ in range(8):
                await asyncio.sleep(0)

        asyncio.run(run())

        downloading = [
            e for e in wire.model_events
            if e.state == pb.MODEL_STATE_DOWNLOADING
        ]
        assert downloading, (
            "a fetch that really moved bytes must declare itself even when the "
            "resident pre-check said otherwise")
        assert downloading[0].bytes_done > 0, (
            "the lazy open states the position that provoked it")
        assert wire.open_download_records() == {}, "and it must still close"
        steps = [r.step for r in wire.positions()]
        assert steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB
    finally:
        origin.close()


def test_a_transfer_that_dies_midway_leaves_no_open_record(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The same class, reached the other way: a download whose PROCESS ends mid-flight."""
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            task = asyncio.ensure_future(store.ensure_local(_REF, snapshot))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.1)

        asyncio.run(run())

        states = [e.state for e in wire.model_events]
        assert pb.MODEL_STATE_DOWNLOADING in states, (
            f"the transfer should have declared itself before dying; got {states}")
        assert wire.open_download_records() == {}, (
            "a cancelled transfer must close the record it opened")
        assert [
            e.error for e in wire.model_events
            if e.state == pb.MODEL_STATE_FAILED
        ] == ["download_canceled"]
        assert weight_position.PHASE_ABANDONED in [
            r.phase for r in wire.positions()]
    finally:
        origin.close()


class _Ctx:

    def __init__(self) -> None:
        self.source_path = ""

    def _set_source_path(self, path: str) -> None:
        self.source_path = path

    def raise_if_cancelled(self, _reason: str) -> None:
        return None


class _Payload:

    def __init__(self, ref: str) -> None:
        self.source = {"ref": ref}


def test_the_job_planes_reserved_repo_fetch_reports_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fine_cadence: Any
) -> None:
    """A reserved-repo materialization leaves the same advancing byte positions a serving-path fetch does."""
    origin = _Origin()
    try:
        monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "cache"))
        gw_config.reload_for_test()
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        resolved = _resolved(files)
        ref = normalize_model_ref("acme/model-a")
        wire = _Wire()
        ctx, payload = _Ctx(), _Payload("acme/model-a")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            await materialize_reserved_inputs_async(ctx, payload, {ref: resolved})
            for _ in range(8):
                await asyncio.sleep(0)

        asyncio.run(run())

        assert ctx.source_path, "the reserved path must still be filled"
        assert str(tmp_path) in ctx.source_path, (
            "the CAS override did not take; this test would be measuring a\n"
            f"shared cache: {ctx.source_path}")
        rows = wire.positions()
        assert rows, (
            "the job plane's reserved-repo fetch emitted NO weight_fetch rows — "
            "20.5 GB of silence is the defect this closes")
        phases = [r.phase for r in rows]
        assert phases[0] == weight_position.PHASE_STARTED
        assert phases[-1] == weight_position.PHASE_FETCHED
        steps = [r.step for r in rows]
        assert steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB
        assert steps[-1] > steps[0], f"the position must advance; got {steps}"
        assert all(_detail(r)["ref"] == str(ref) for r in rows)
    finally:
        gw_config.reload_for_test()
        origin.close()
