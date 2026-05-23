"""Tests for per-request model-download lifecycle events (gen-orchestrator #348).

When a request's assigned worker has to fetch a model that isn't on disk yet,
the worker emits ``model.download.{started,progress,completed,failed}`` events
tagged with the triggering ``request_id``. They ride the existing WorkerEvent
fabric and surface on the request's SSE stream
(GET /v1/requests/:id/events).

These tests drive ``Worker._run_download_with_request_progress`` directly:
  - A fake ``download_fn`` that writes blob files to disk so the on-disk byte
    sampler observes progress (the real downloader has no byte callback).
  - A fake resolved manifest entry that declares per-file ``size_bytes`` +
    ``blake3`` digests — the catalog data the worker sums for total bytes.
  - ``_send_message`` captured so the test introspects every emitted event.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker.run_metrics_v1 import _blob_path, _cozy_blobs_root
from gen_worker.worker import Worker


class _FakeFile:
    def __init__(self, blake3: str, size_bytes: int) -> None:
        self.blake3 = blake3
        self.size_bytes = size_bytes


class _FakeResolvedEntry:
    def __init__(self, files: List[_FakeFile]) -> None:
        self.files = files


def _digest_for(name: str) -> str:
    return hashlib.blake2b(name.encode(), digest_size=16).hexdigest()


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    w._sent_messages = []
    w._send_message = lambda msg: w._sent_messages.append(msg)
    return w


def _download_events(w: Worker) -> List[dict]:
    out = []
    for msg in w._sent_messages:
        if msg.WhichOneof("msg") != "worker_event":
            continue
        ev = msg.worker_event
        if not ev.event_type.startswith("model.download."):
            continue
        payload = json.loads(ev.payload_json.decode("utf-8")) if ev.payload_json else {}
        out.append({"request_id": ev.request_id, "type": ev.event_type, "payload": payload})
    return out


def test_per_request_download_emits_started_progress_completed(tmp_path: Path) -> None:
    cache_dir = tmp_path
    blobs_root = _cozy_blobs_root(cache_dir)

    # Manifest: 5 files * 20MB = 100MB total (matches the issue's 100MB test).
    files = [_FakeFile(_digest_for(f"f{i}"), 20 * 1024 * 1024) for i in range(5)]
    entry = _FakeResolvedEntry(files)

    w = _bare_worker()

    def fake_download() -> str:
        # Stream the blobs onto disk one at a time with a small delay so the
        # sampler thread (polls every ~1s) observes byte growth.
        for f in files:
            p = _blob_path(blobs_root, f.blake3)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\0" * f.size_bytes)
            time.sleep(0.45)
        return str(cache_dir / "snapshots" / "abc")

    local = w._run_download_with_request_progress(
        request_id="req-348-x",
        model_id="acme/diffuse@sha256:abc",
        cache_dir=str(cache_dir),
        resolved_entry=entry,
        download_fn=fake_download,
    )
    assert local

    events = _download_events(w)
    types = [e["type"] for e in events]

    # started first, completed last, at least one progress in between.
    assert types[0] == "model.download.started", types
    assert types[-1] == "model.download.completed", types
    assert "model.download.progress" in types, types

    # All tagged with the triggering request_id.
    for e in events:
        assert e["request_id"] == "req-348-x", e

    started = events[0]["payload"]
    assert started["estimated_total_bytes"] == 100 * 1024 * 1024, started

    # Progress events carry byte progress + ETA + throughput, and are
    # rate-limited to the hard cap of 25.
    progress = [e for e in events if e["type"] == "model.download.progress"]
    assert 1 <= len(progress) <= 25, len(progress)
    for p in progress:
        pay = p["payload"]
        assert pay["bytes_total"] == 100 * 1024 * 1024
        assert 0 <= pay["bytes_downloaded"] <= pay["bytes_total"]
        assert "eta_remaining_seconds" in pay
        assert "throughput_bytes_per_sec" in pay
        assert 0.0 <= pay["percent_complete"] <= 100.0

    # bytes_downloaded must be monotonic non-decreasing across progress events.
    seen = [p["payload"]["bytes_downloaded"] for p in progress]
    assert seen == sorted(seen), seen

    completed = events[-1]["payload"]
    assert completed["bytes_total"] == 100 * 1024 * 1024
    assert "duration_ms" in completed


def test_download_failure_emits_failed_with_error_type(tmp_path: Path) -> None:
    entry = _FakeResolvedEntry([_FakeFile(_digest_for("f0"), 10 * 1024 * 1024)])
    w = _bare_worker()

    class _BoomError(RuntimeError):
        pass

    def fake_download() -> str:
        raise _BoomError("upstream 500")

    try:
        w._run_download_with_request_progress(
            request_id="req-348-fail",
            model_id="acme/diffuse",
            cache_dir=str(tmp_path),
            resolved_entry=entry,
            download_fn=fake_download,
        )
        assert False, "expected the download error to propagate"
    except _BoomError:
        pass

    events = _download_events(w)
    types = [e["type"] for e in events]
    assert "model.download.started" in types, types
    assert types[-1] == "model.download.failed", types

    failed = events[-1]["payload"]
    assert failed["request_id"] == "req-348-fail"
    # error_type carries the actual exception class from the download path.
    assert failed["error_type"] == "_BoomError", failed
    assert failed["provider"] == "tensorhub"
    assert failed["source"] == "tensorhub_cas"
    assert "duration_ms" in failed


@pytest.mark.parametrize(
    ("provider", "source"),
    [
        ("hf", "huggingface"),
        ("civitai", "civitai"),
    ],
)
def test_provider_progress_failure_emits_provider_source(
    tmp_path: Path,
    provider: str,
    source: str,
) -> None:
    w = _bare_worker()

    class _ProviderBoom(RuntimeError):
        pass

    def fake_download(progress_cb):  # noqa: ANN001
        progress_cb(1, 10)
        raise _ProviderBoom("download failed")

    with pytest.raises(_ProviderBoom):
        w._run_download_with_request_progress(
            request_id=f"req-{provider}-fail",
            model_id="provider/ref",
            cache_dir=str(tmp_path),
            resolved_entry=None,
            download_fn=lambda: "not-used",
            provider=provider,
            source=source,
            progress_download_fn=fake_download,
        )

    events = _download_events(w)
    failed = events[-1]["payload"]
    assert events[-1]["type"] == "model.download.failed"
    assert failed["error_type"] == "_ProviderBoom"
    assert failed["provider"] == provider
    assert failed["source"] == source


def test_requestless_fetch_emits_no_request_events(tmp_path: Path) -> None:
    # request_id="" is the worker-lifecycle/prewarm scope — out of #348's
    # request-scoped fabric. The wrapper must just run the download silently.
    entry = _FakeResolvedEntry([_FakeFile(_digest_for("f0"), 1024)])
    w = _bare_worker()
    ran = {"n": 0}

    def fake_download() -> str:
        ran["n"] += 1
        return "ok"

    out = w._run_download_with_request_progress(
        request_id="",
        model_id="acme/diffuse",
        cache_dir=str(tmp_path),
        resolved_entry=entry,
        download_fn=fake_download,
    )
    assert out == "ok"
    assert ran["n"] == 1
    assert _download_events(w) == []


def test_bootstrap_fallback_no_manifest_sizes(tmp_path: Path) -> None:
    # Catalog unreachable / no per-file sizes: estimated_total_bytes=0,
    # estimated_eta_seconds=-1; started/completed still fire, no byte progress.
    entry = _FakeResolvedEntry([])  # no files -> total bytes unknown
    w = _bare_worker()

    out = w._run_download_with_request_progress(
        request_id="req-348-boot",
        model_id="acme/diffuse",
        cache_dir=str(tmp_path),
        resolved_entry=entry,
        download_fn=lambda: "ok",
    )
    assert out == "ok"

    events = _download_events(w)
    types = [e["type"] for e in events]
    assert types == ["model.download.started", "model.download.completed"], types
    started = events[0]["payload"]
    assert started["estimated_total_bytes"] == 0
    assert started["estimated_eta_seconds"] == -1


@pytest.mark.parametrize(
    ("provider", "source", "model_id"),
    [
        ("hf", "huggingface", "owner/model#bf16"),
        ("civitai", "civitai", "123456"),
    ],
)
def test_provider_progress_sink_emits_same_shape_without_tensorhub_manifest(
    tmp_path: Path,
    provider: str,
    source: str,
    model_id: str,
) -> None:
    w = _bare_worker()
    seen_callbacks: list[tuple[int, int | None]] = []

    def fake_download(progress_cb):
        progress_cb(0, 100)
        seen_callbacks.append((0, 100))
        time.sleep(1.05)
        progress_cb(50, 100)
        seen_callbacks.append((50, 100))
        time.sleep(1.05)
        progress_cb(100, 100)
        seen_callbacks.append((100, 100))
        return "ok"

    out = w._run_download_with_request_progress(
        request_id="req-hf-progress",
        model_id=model_id,
        cache_dir=str(tmp_path),
        resolved_entry=None,
        download_fn=lambda: "not-used",
        provider=provider,
        source=source,
        progress_download_fn=fake_download,
    )
    assert out == "ok"
    assert seen_callbacks == [(0, 100), (50, 100), (100, 100)]

    events = _download_events(w)
    assert [e["type"] for e in events][0] == "model.download.started"
    assert [e["type"] for e in events][-1] == "model.download.completed"
    progress = [e for e in events if e["type"] == "model.download.progress"]
    assert progress, events
    for e in events:
        payload = e["payload"]
        assert payload["provider"] == provider
        assert payload["source"] == source
    assert progress[-1]["payload"]["bytes_downloaded"] == 100
