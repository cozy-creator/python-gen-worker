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
from typing import Dict, List, Optional

import blake3
import msgspec
import pytest

from gen_worker.api.types import DatasetRef
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.request_context import TrainingContext
from gen_worker.request_context import _datasets as datasets_mod


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
        self.seen_auth: List[str] = []

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:  # noqa: N802
                pass

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                outer.seen_auth.append(self.headers.get("Authorization") or "")
                if parsed.path == "/api/v1/datasets":
                    self._json({"items": [
                        {"dataset_id": "ds-1", "tenant": "acme", "name": "faces"},
                        {"dataset_id": "ds-2", "tenant": "acme", "name": "other"},
                    ]})
                elif parsed.path == "/api/v1/datasets/ds-1/materialize":
                    digest = outer.shard_digest
                    if outer.lie_about_digest:
                        digest = "blake3:" + "0" * 64
                    self._json({
                        "dataset_id": "ds-1",
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

            def _json(self, payload: Dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
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


def _run_training_job(hub: _FakeHub, payload: _TrainIn) -> pb.JobResult:
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
            capability_token="cap-tok",
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
