from __future__ import annotations

import asyncio
import hashlib
import http.server
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from gen_worker import activity, snapshot_pull
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.snapshot_pull import SnapshotPullStats


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
        with self.server.lock:  # type: ignore[attr-defined]
            self.server.wire[0] += len(body)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Origin:

    def __init__(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.blobs = {}  # type: ignore[attr-defined]
        self.server.wire = [0]  # type: ignore[attr-defined]
        self.server.lock = threading.Lock()  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def put(self, data: bytes) -> str:
        digest = _sha(data)
        self.server.blobs[digest] = data  # type: ignore[attr-defined]
        address = self.server.server_address
        return f"http://{address[0]!s}:{address[1]!s}/{digest}"

    @property
    def wire_bytes(self) -> int:
        with self.server.lock:  # type: ignore[attr-defined]
            return int(self.server.wire[0])  # type: ignore[attr-defined]

    def reset(self) -> None:
        with self.server.lock:  # type: ignore[attr-defined]
            self.server.wire[0] = 0  # type: ignore[attr-defined]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class _Wire:

    def __init__(self) -> None:
        self.updates: list[pb.ActivityUpdate] = []

    async def send(self, msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") == "activity_update":
            self.updates.append(msg.activity_update)

    def pulls(self) -> list[pb.ActivityUpdate]:
        return [
            u for u in self.updates if u.kind == activity.KIND_SNAPSHOT_PULL
        ]


def _detail(update: pb.ActivityUpdate) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", update.detail))


def _ints(update: pb.ActivityUpdate) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in _detail(update).items():
        try:
            out[key] = int(value)
        except ValueError:
            continue
    return out


def _ref(repo: str) -> TensorhubRef:
    return TensorhubRef(owner="acme", repo=repo, release="latest")


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


def _pull(
    wire: _Wire,
    base: Path,
    repo: str,
    resolved: WorkerResolvedRepo,
    *,
    fill_source_dir: Path | None = None,
) -> pb.ActivityUpdate:

    async def run() -> None:
        activity.bind_sink(wire.send, asyncio.get_running_loop())
        await ensure_snapshot_async(
            base_dir=base,
            ref=_ref(repo),
            resolved=resolved,
            progress=None,
            fill_source_dir=fill_source_dir,
        )
        for _ in range(4):
            await asyncio.sleep(0)

    before = len(wire.pulls())
    asyncio.run(run())
    fresh = wire.pulls()[before:]
    assert len(fresh) == 1, f"one pull, {len(fresh)} snapshot_pull event(s)"
    return fresh[0]


@pytest.fixture(autouse=True)
def _clean_sink() -> Any:
    activity.reset_for_tests()
    yield
    activity.reset_for_tests()


def test_a_cold_pull_reports_the_bytes_that_crossed_the_wire(tmp_path: Path) -> None:
    """The number that did not exist."""
    origin = _Origin()
    try:
        te = b"text-encoder-" + b"t" * 4000
        unet = b"unet-a-" + b"u" * 9000
        resolved = _resolved(
            [
                ("text_encoder/model.safetensors", te, origin.put(te)),
                ("unet/model.safetensors", unet, origin.put(unet)),
            ]
        )
        wire = _Wire()
        event = _pull(wire, tmp_path / "pod", "model-a", resolved)

        assert event.phase == snapshot_pull.PHASE_PULLED
        assert event.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
        got = _ints(event)
        assert got["requested_objects"] == 2
        assert got["fetched_objects"] == 2
        assert got["resident_objects"] == 0
        assert got["tree_bytes"] == len(te) + len(unet)
        assert got["fetched_bytes"] == origin.wire_bytes == len(te) + len(unet)
        assert _detail(event)["snapshot"] == resolved.snapshot_digest
    finally:
        origin.close()


def test_warm_boot_dedup_is_a_query_over_two_overlapping_models(
    tmp_path: Path,
) -> None:
    origin = _Origin()
    try:
        shared = b"shared-text-encoder-" + b"s" * 20000
        unet_a = b"unet-a-" + b"a" * 2000
        unet_b = b"unet-b-" + b"b" * 2000
        shared_url = origin.put(shared)
        model_a = _resolved(
            [
                ("text_encoder/model.safetensors", shared, shared_url),
                ("unet/model.safetensors", unet_a, origin.put(unet_a)),
            ]
        )
        model_b = _resolved(
            [
                ("text_encoder/model.safetensors", shared, shared_url),
                ("transformer/model.safetensors", unet_b, origin.put(unet_b)),
            ]
        )
        assert model_a.snapshot_digest != model_b.snapshot_digest

        pod = tmp_path / "pod"
        wire = _Wire()
        cold = _ints(_pull(wire, pod, "model-a", model_a))
        origin.reset()
        warm_event = _pull(wire, pod, "model-b", model_b)
        warm = _ints(warm_event)

        assert cold["fetched_objects"] == 2
        assert cold["resident_objects"] == 0
        assert cold["fetched_bytes"] == len(shared) + len(unet_a)

        assert warm["requested_objects"] == 2
        assert warm["fetched_objects"] == 1
        assert warm["resident_objects"] == 1
        assert warm["resident_bytes"] == len(shared)
        assert warm["tree_bytes"] == len(shared) + len(unet_b)
        assert warm["fetched_bytes"] == len(unet_b) == origin.wire_bytes
        assert warm["fetched_bytes"] * 4 < warm["tree_bytes"]
    finally:
        origin.close()


def test_a_fully_resident_snapshot_reports_a_zero_wire(tmp_path: Path) -> None:
    """The boundary."""
    origin = _Origin()
    try:
        blob = b"weights-" + b"w" * 5000
        files = [("unet/model.safetensors", blob, origin.put(blob))]
        pod = tmp_path / "pod"
        wire = _Wire()
        _pull(wire, pod, "model-a", _resolved(files))
        origin.reset()
        again = _resolved(files + [("README.md", b"x", origin.put(b"x"))])
        warm = _ints(_pull(wire, pod, "model-a2", again))

        assert warm["requested_objects"] == 2
        assert warm["resident_objects"] == 1
        assert warm["resident_bytes"] == len(blob)
        assert warm["fetched_objects"] == 1
        assert warm["fetched_bytes"] == 1 == origin.wire_bytes
    finally:
        origin.close()


def test_every_requested_object_is_accounted_for(tmp_path: Path) -> None:
    """A count that does not add up is a count nobody can query against."""
    origin = _Origin()
    try:
        blobs = [bytes([65 + i]) * (300 + i) for i in range(5)]
        files = [
            (f"c{i}/f.safetensors", body, origin.put(body))
            for i, body in enumerate(blobs)
        ]
        pod = tmp_path / "pod"
        wire = _Wire()
        first = _ints(_pull(wire, pod, "m", _resolved(files[:3])))
        second = _ints(_pull(wire, pod, "m2", _resolved(files)))
        for row in (first, second):
            assert row["accounted_objects"] == row["requested_objects"]
        assert second["resident_objects"] == 3
        assert second["fetched_objects"] == 2
    finally:
        origin.close()


def test_an_endpoint_volume_fill_is_not_pod_local_dedup(tmp_path: Path) -> None:
    """The number that would flatter itself."""
    origin = _Origin()
    try:
        from gen_worker.models.cache_paths import open_worker_cas

        on_volume = b"volume-supplied-" + b"v" * 3000
        over_wire = b"wire-only-" + b"w" * 1000
        volume = tmp_path / "endpoint-volume"
        open_worker_cas(volume).put_bytes(on_volume)

        wire = _Wire()
        got = _ints(
            _pull(
                wire,
                tmp_path / "pod",
                "m",
                _resolved(
                    [
                        ("te/model.safetensors", on_volume, origin.put(on_volume)),
                        ("unet/model.safetensors", over_wire, origin.put(over_wire)),
                    ]
                ),
                fill_source_dir=volume,
            )
        )
        assert got["filled_objects"] == 1
        assert got["filled_bytes"] == len(on_volume)
        assert got["resident_objects"] == 0
        assert got["resident_bytes"] == 0
        assert got["fetched_objects"] == 1
        assert got["fetched_bytes"] == len(over_wire) == origin.wire_bytes
        assert got["accounted_objects"] == got["requested_objects"] == 2
    finally:
        origin.close()


def test_the_detail_grammar_survives_a_reader(tmp_path: Path) -> None:
    """The row is only worth emitting if the harness that reads it can parse it."""
    stats = SnapshotPullStats(
        requested_objects=9,
        tree_bytes=1234,
        fetched_objects=2,
        fetched_bytes=56,
        resident_objects=7,
        resident_bytes=1178,
    )
    detail = stats.detail(snapshot="sha256:abc", key="sha256:abc__c9f", components=2)
    parsed = dict(re.findall(r"(\w+)=(\S+)", detail))
    assert parsed["snapshot"] == "sha256:abc"
    assert parsed["key"] == "sha256:abc__c9f"
    assert int(parsed["fetched_bytes"]) == 56
    assert int(parsed["resident_objects"]) == 7
    assert len(detail.split()) == len(parsed), "a token in `detail` is not a k=v pair"

    blank = stats.detail(snapshot="", key="a b", components=0)
    assert dict(re.findall(r"(\w+)=(\S+)", blank))["snapshot"] == "-"
    assert dict(re.findall(r"(\w+)=(\S+)", blank))["key"] == "ab"
    assert len(blank.split()) == len(re.findall(r"(\w+)=(\S+)", blank))


def test_the_pull_event_is_an_event_and_never_a_running_activity(
    tmp_path: Path,
) -> None:
    """It is a self-contained roll-up of finished work."""
    origin = _Origin()
    try:
        blob = b"z" * 900
        wire = _Wire()
        event = _pull(
            wire,
            tmp_path / "pod",
            "m",
            _resolved([("unet/f.safetensors", blob, origin.put(blob))]),
        )
        assert event.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
        assert event.counter == "" and event.counter_total == 0
        assert not event.self_stalled
        assert event.duration_ms >= 0
    finally:
        origin.close()
