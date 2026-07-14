"""gw#425: dataset materialization — payload.datasets → local parquet shards.

Real codepaths: a local HTTP server plays the tensorhub datasets API
(list → materialize manifest with presigned URLs → shard bytes) and
``resolve_dataset`` streams real pyarrow-written parquet to disk with
blake3 verification and bounded retries.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

import blake3
import msgspec
import pytest

from gen_worker.api.errors import AuthError, SnapshotBuildFailedError
from gen_worker.api.types import DatasetRef
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.request_context import TrainingContext
from gen_worker.request_context import _datasets as datasets_mod

# th#641 production shape: the hub rewrites payload.datasets[].ref to the
# dataset UUID at submit and mints the read_dataset grant by UUID.
_DATASET_UUID = "0b6e0c33-8f5e-4a8e-9d0f-2f6f19e1c9aa"
# A capability token carrying ONLY a read_dataset grant — can materialize by
# id, cannot list.
_GRANT_TOKEN = "cap-tok-read-grant"


def _tiny_parquet_bytes() -> bytes:
    """HF-datasets columnar shard per th#642: embedded image bytes + caption."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    image = pa.array(
        [{"bytes": b"\x89PNG-fake-0", "path": "0.png"},
         {"bytes": b"\x89PNG-fake-1", "path": "1.png"}],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    caption = pa.array(["a cat", "a dog"], type=pa.string())
    table = pa.table({"image": image, "caption": caption})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


class _FakeHub:
    """Datasets list + materialize + presigned shard bytes over real HTTP."""

    def __init__(self) -> None:
        import uuid

        self.snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        self.shard = _tiny_parquet_bytes()
        self.shard_digest = "blake3:" + blake3.blake3(self.shard).hexdigest()
        self.fail_first_n_shard_gets = 0
        self.lie_about_digest = False
        self.shard_requests = 0
        self.list_requests = 0
        self.seen_auth: List[str] = []
        # DATASET-V2 async snapshot contract (th#691 / gw#457).
        self.materialize_202s = 0  # 202 {building} responses before the 200
        self.build_failed = False  # 503 typed snapshot_build_failed
        self.materialize_requests = 0
        self.seen_wait: List[str] = []
        # Rows the list endpoint knows + ids materialize serves. th#641
        # production refs are bare UUIDs the list endpoint never saw.
        self.known_ids = {"ds-1", _DATASET_UUID}

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:  # noqa: N802
                pass

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                auth = self.headers.get("Authorization") or ""
                outer.seen_auth.append(auth)
                parts = parsed.path.strip("/").split("/")
                if parsed.path == "/api/v1/datasets":
                    outer.list_requests += 1
                    # Grant-scoped read_dataset capability tokens cannot list
                    # (tensorhub resolveTargetTenant requires create_dataset).
                    if auth == f"Bearer {_GRANT_TOKEN}":
                        self._status(403)
                        return
                    # The hub reads ?tenant= — ?owner= is silently ignored
                    # there, which here is a hard failure.
                    q = urllib.parse.parse_qs(parsed.query)
                    if q.get("tenant", [""])[0] != "acme":
                        self._status(400)
                        return
                    self._json({"items": [
                        {"dataset_id": "ds-1", "tenant": "acme", "name": "faces"},
                        {"dataset_id": "ds-2", "tenant": "acme", "name": "other"},
                    ]})
                elif (len(parts) == 5 and parts[:3] == ["api", "v1", "datasets"]
                      and parts[4] == "materialize"):
                    ds_id = parts[3]
                    if ds_id not in outer.known_ids:
                        self._status(404)
                        return
                    outer.materialize_requests += 1
                    q = urllib.parse.parse_qs(parsed.query)
                    outer.seen_wait.append(q.get("wait", [""])[0])
                    if outer.build_failed:
                        self._json({"error": "snapshot_build_failed",
                                    "error_code": "encode_error"}, status=503)
                        return
                    if outer.materialize_requests <= outer.materialize_202s:
                        self._json({"status": "building", "state_version": 7,
                                    "retry_after": 0.01}, status=202)
                        return
                    digest = outer.shard_digest
                    if outer.lie_about_digest:
                        digest = "blake3:" + "0" * 64
                    self._json({
                        "dataset_id": ds_id,
                        "format": "parquet_ref",
                        "snapshot_id": outer.snapshot_id,
                        "entries": [
                            {
                                "path": "schema/dataset_info.json",
                                "inline_text": json.dumps({"features": {"caption": {"_type": "Value"}}}),
                            },
                            {
                                "path": "rows/train-00000.parquet",
                                "url": f"http://127.0.0.1:{outer.port}/shard0",
                                "size_bytes": len(outer.shard),
                                "checksum": digest,
                            },
                        ],
                    })
                elif parsed.path == "/shard0":
                    outer.shard_requests += 1
                    if outer.shard_requests <= outer.fail_first_n_shard_gets:
                        body = outer.shard[: len(outer.shard) // 2]  # truncated
                    else:
                        body = outer.shard
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _status(self, code: int) -> None:
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _json(self, payload: Dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def hub():
    h = _FakeHub()
    yield h
    h.close()


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(datasets_mod, "_DOWNLOAD_BACKOFF_S", 0.01)
    monkeypatch.setattr(datasets_mod, "_POLL_BACKOFF_START_S", 0.01)
    monkeypatch.setattr(datasets_mod, "_POLL_BACKOFF_CAP_S", 0.02)


def _ctx(hub: _FakeHub) -> TrainingContext:
    return TrainingContext(
        request_id="r1",
        file_api_base_url=hub.base,
        worker_capability_token="cap-tok",
    )


def _assert_snapshot(root: Path, hub: _FakeHub) -> None:
    import pyarrow.parquet as pq

    shard = root / "rows" / "train-00000.parquet"
    assert shard.is_file()
    assert shard.read_bytes() == hub.shard
    table = pq.read_table(shard)
    assert table.column_names == ["image", "caption"]
    assert table.column("caption").to_pylist() == ["a cat", "a dog"]
    assert table.column("image").to_pylist()[0]["bytes"] == b"\x89PNG-fake-0"
    assert (root / "schema" / "dataset_info.json").is_file()
    assert not list(root.rglob("*.tmp"))


def test_resolve_dataset_downloads_verified_parquet(hub) -> None:
    ctx = _ctx(hub)
    root = Path(ctx.resolve_dataset("acme/faces"))
    _assert_snapshot(root, hub)
    assert ctx.dataset_paths["acme/faces"] == str(root)
    assert hub.seen_auth[0] == "Bearer cap-tok"
    # Second resolve is a cache hit — no extra shard fetch.
    n = hub.shard_requests
    assert ctx.resolve_dataset("acme/faces") == str(root)
    assert hub.shard_requests == n


def test_resolve_dataset_retries_truncated_shard(hub) -> None:
    hub.fail_first_n_shard_gets = 2  # size mismatch twice, then clean
    ctx = _ctx(hub)
    root = Path(ctx.resolve_dataset("acme/faces"))
    _assert_snapshot(root, hub)
    assert hub.shard_requests == 3


def test_resolve_dataset_digest_mismatch_fails_loudly(hub) -> None:
    hub.lie_about_digest = True
    ctx = _ctx(hub)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        ctx.resolve_dataset("acme/faces")
    assert hub.shard_requests == 3  # exhausted retries
    assert "acme/faces" not in ctx.dataset_paths


def test_resolve_dataset_unknown_name(hub) -> None:
    ctx = _ctx(hub)
    with pytest.raises(RuntimeError, match="not found"):
        ctx.resolve_dataset("acme/nope")


def test_resolve_dataset_uuid_ref_skips_list(hub) -> None:
    """th#641 production path: bare UUID ref + grant-scoped token → straight
    to materialize; the list endpoint (403 for this token) is never touched."""
    ctx = TrainingContext(
        request_id="r1",
        file_api_base_url=hub.base,
        worker_capability_token=_GRANT_TOKEN,
    )
    root = Path(ctx.resolve_dataset(_DATASET_UUID))
    _assert_snapshot(root, hub)
    assert ctx.dataset_paths[_DATASET_UUID] == str(root)
    assert hub.list_requests == 0
    assert f"Bearer {_GRANT_TOKEN}" in hub.seen_auth
    # Cache hit on second resolve.
    n = hub.shard_requests
    assert ctx.resolve_dataset(_DATASET_UUID) == str(root)
    assert hub.shard_requests == n


def test_grant_scoped_token_cannot_use_owner_name_refs(hub) -> None:
    """owner/name refs need the list endpoint, which read-only grant tokens
    can't call — AuthError, not a silent wrong answer."""
    ctx = TrainingContext(
        request_id="r1",
        file_api_base_url=hub.base,
        worker_capability_token=_GRANT_TOKEN,
    )
    with pytest.raises(AuthError):
        ctx.resolve_dataset("acme/faces")


# ---- DATASET-V2 async snapshot contract (th#691 / gw#457) ------------------


def test_resolve_dataset_rides_202_until_ready(hub) -> None:
    """202 {building, retry_after} twice, then 200 → manifest resolves; the
    worker sent the ?wait long-poll hint on every attempt."""
    hub.materialize_202s = 2
    ctx = _ctx(hub)
    root = Path(ctx.resolve_dataset("acme/faces"))
    _assert_snapshot(root, hub)
    assert hub.materialize_requests == 3
    assert all(w == "30" for w in hub.seen_wait)


def test_snapshot_build_failed_is_typed(hub) -> None:
    hub.build_failed = True
    ctx = _ctx(hub)
    with pytest.raises(SnapshotBuildFailedError, match="encode_error") as ei:
        ctx.resolve_dataset("acme/faces")
    assert ei.value.error_code == "encode_error"
    assert hub.materialize_requests == 1  # no retry loop on a typed failure
    assert "acme/faces" not in ctx.dataset_paths


def test_materialize_budget_exhaustion(hub) -> None:
    hub.materialize_202s = 10**9  # never ready
    ctx = _ctx(hub)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        ctx.resolve_dataset("acme/faces", budget_s=0.05)
    assert hub.materialize_requests >= 1


def test_executor_rides_202_window(hub) -> None:
    """The executor pre-materialization path (gw#425) survives a live 202
    window — same helper, same polling."""
    hub.materialize_202s = 2
    res = _run_training_job(hub, _TrainIn(datasets=[DatasetRef(ref="acme/faces")]))
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_TrainOut)
    assert out.shard_exists
    assert hub.materialize_requests == 3


# ---- executor: payload.datasets materialized before the handler runs -------


class _TrainIn(msgspec.Struct):
    datasets: List[DatasetRef] = msgspec.field(default_factory=list)
    steps: int = 1


class _TrainOut(msgspec.Struct):
    dataset_paths: Dict[str, str]
    shard_exists: bool


def _train(ctx, payload: _TrainIn) -> _TrainOut:
    paths = dict(ctx.dataset_paths)
    first = next(iter(paths.values()), "")
    return _TrainOut(
        dataset_paths=paths,
        shard_exists=bool(first) and (Path(first) / "rows" / "train-00000.parquet").is_file(),
    )


def _run_training_job(hub: _FakeHub, payload: _TrainIn, *, token: str = "cap-tok") -> pb.JobResult:
    async def _go() -> pb.JobResult:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        spec = EndpointSpec(
            name="train-lora", method=_train, kind="training",
            payload_type=_TrainIn, output_mode="single",
        )
        ex = Executor([spec], _send)
        ex.file_base_url = hub.base
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="train-lora",
            capability_token=token,
            input_payload=msgspec.msgpack.encode(payload)))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert results
        assert job.renew_task is None  # cancelled at _finish
        return results[-1]

    return asyncio.run(_go())


def test_executor_materializes_payload_datasets(hub) -> None:
    res = _run_training_job(hub, _TrainIn(datasets=[DatasetRef(ref="acme/faces")]))
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_TrainOut)
    assert "acme/faces" in out.dataset_paths
    assert out.shard_exists


def test_executor_dataset_failure_is_job_failure(hub) -> None:
    hub.lie_about_digest = True
    res = _run_training_job(hub, _TrainIn(datasets=[DatasetRef(ref="acme/faces")]))
    assert res.status != pb.JOB_STATUS_OK


def test_executor_materializes_uuid_payload_datasets(hub) -> None:
    """The gw#425 pre-materialization path must accept th#641's rewritten
    UUID refs with a grant-scoped token — the production shape."""
    res = _run_training_job(
        hub, _TrainIn(datasets=[DatasetRef(ref=_DATASET_UUID)]), token=_GRANT_TOKEN,
    )
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_TrainOut)
    assert _DATASET_UUID in out.dataset_paths
    assert out.shard_exists
    assert hub.list_requests == 0
