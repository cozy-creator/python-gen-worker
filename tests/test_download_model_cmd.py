"""Tests for the DownloadModelCommand handler (gen-orchestrator #339).

The orchestrator sends a `DownloadModelCommand` to a worker when an
inference request needs a model that isn't on any worker's disk yet — the
worker should fetch the bytes, advertise the ref in `disk_models` on the
next heartbeat, and emit `WorkerModelReadySignal{DOWNLOAD_COMPLETED}` so
the scheduler can re-dispatch the parked job.

These tests exercise `Worker._run_download_model_cmd` directly (the
synchronous body extracted from `_handle_download_model_cmd` for exactly
this reason). We:
  - Mock the downloader so no real bytes move.
  - Mock _send_message to capture the outgoing pb messages.
  - Mock _register_worker to confirm the force-heartbeat path fires.
  - Use a real ModelCache so the disk_models query reflects the
    handler's writes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from gen_worker.models.cache import ModelCache
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.run_metrics_v1 import _blob_path, _cozy_blobs_root
from gen_worker.worker import Worker


def _bare_worker(tmp_path: Path) -> Worker:
    """Build a Worker instance without running __init__.

    Only the attributes touched by `_run_download_model_cmd` are stubbed.
    Other Worker attributes that getattr() falls through into are left
    unset so any unexpected dependency surfaces as AttributeError.
    """
    w = Worker.__new__(Worker)
    # Captured wire-out messages — _send_message appends here so the test
    # can introspect every signal the handler emits.
    w._sent_messages = []
    w._send_message = lambda msg: w._sent_messages.append(msg)
    # Force-heartbeat: we record the call count rather than running the
    # real registration path (it would try to talk to a stub).
    w._register_worker_calls = []
    w._register_worker = lambda is_heartbeat=False: w._register_worker_calls.append(
        bool(is_heartbeat)
    )
    # Real ModelCache so get_disk_models() returns what the handler wrote.
    w._model_cache = ModelCache()
    w._downloader = None  # tests set this per case
    w._resolved_repos_by_id_baseline = None
    # If the snapshot pre-existence check fires it must not blow up.
    w._try_find_existing_cozy_snapshot_dir = lambda canon, cache_dir: None
    return w


def _last_load_model_result(w: Worker) -> pb.LoadModelResult:
    for msg in reversed(w._sent_messages):
        if msg.WhichOneof("msg") == "load_model_result":
            return msg.load_model_result
    raise AssertionError("no LoadModelResult was sent")


def _model_ready_signals(w: Worker) -> List[pb.WorkerModelReadySignal]:
    return [
        msg.worker_model_ready
        for msg in w._sent_messages
        if msg.WhichOneof("msg") == "worker_model_ready"
    ]


def _worker_events(w: Worker) -> List[dict]:
    events: List[dict] = []
    for msg in w._sent_messages:
        if msg.WhichOneof("msg") != "worker_event":
            continue
        ev = msg.worker_event
        payload = json.loads(ev.payload_json.decode("utf-8")) if ev.payload_json else {}
        events.append({"request_id": ev.request_id, "type": ev.event_type, "payload": payload})
    return events


class _FakeFile:
    def __init__(self, blake3: str, size_bytes: int) -> None:
        self.blake3 = blake3
        self.size_bytes = size_bytes


class _FakeResolvedEntry:
    def __init__(self, files: List[_FakeFile]) -> None:
        self.files = files


def _digest_for(name: str) -> str:
    return hashlib.blake2b(name.encode(), digest_size=16).hexdigest()


# Use a non-canonicalized ref so we also exercise the canonicalize+echo
# round-trip the handler does. ParsedModelRef accepts owner/repo with no
# tag and the canonicalizer leaves it alone.
TEST_REF = "acme/test-model"


def test_download_invokes_downloader_with_canonical_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler must call downloader.download(ref, cache_dir) exactly once
    with the canonical model ref the orchestrator asked for.
    """
    # Point cache_dir into tmp_path so the handler doesn't touch $HOME.
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)

    download_calls: List[Any] = []
    downloaded_dir = tmp_path / "snapshot-1"
    downloaded_dir.mkdir()

    def fake_download(ref: str, dest_dir: str, filename: str | None = None) -> str:
        download_calls.append((ref, dest_dir, filename))
        return str(downloaded_dir)

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = fake_download

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    assert len(download_calls) == 1, f"downloader.download called {len(download_calls)} times, want 1"
    called_ref, called_dest, _ = download_calls[0]
    assert called_ref == TEST_REF
    assert called_dest == str(tmp_path)


def test_download_success_sends_load_model_result_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On successful download the worker MUST send LoadModelResult with
    success=True so the orchestrator clears its pendingDownloadsByModel
    tracker for this ref.
    """
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    downloaded_dir = tmp_path / "snapshot-ok"
    downloaded_dir.mkdir()

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = lambda ref, dest_dir, filename=None: str(downloaded_dir)

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    result = _last_load_model_result(w)
    assert result.model_id == TEST_REF
    assert result.success is True
    assert result.error_message == ""


def test_download_success_emits_download_completed_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker must emit WorkerModelReadySignal{MODEL_AVAILABILITY_DOWNLOAD_COMPLETED}.
    handleWorkerModelReady() in connect_worker.go reads this kind to
    immediately append the ref to info.Resources.DiskModels without
    waiting for the next heartbeat resource snapshot.
    """
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    downloaded_dir = tmp_path / "snapshot-signal"
    downloaded_dir.mkdir()

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = lambda ref, dest_dir, filename=None: str(downloaded_dir)

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    signals = _model_ready_signals(w)
    kinds = {s.kind for s in signals}
    assert pb.MODEL_AVAILABILITY_DOWNLOAD_COMPLETED in kinds, (
        f"expected DOWNLOAD_COMPLETED signal, got kinds={kinds}"
    )
    # Every emitted signal must echo the model_id the orchestrator sent.
    for s in signals:
        assert s.model_id == TEST_REF


def test_download_success_emits_worker_lifecycle_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    downloaded_dir = tmp_path / "snapshot-lifecycle"
    downloaded_dir.mkdir()

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = lambda ref, dest_dir, filename=None: str(downloaded_dir)

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    events = _worker_events(w)
    types = [e["type"] for e in events]
    assert "worker.model.download.started" in types, types
    assert "worker.model.download.completed" in types, types
    assert "model.download.started" not in types, types
    assert "model.download.completed" not in types, types
    for e in events:
        if e["type"].startswith("worker.model.download."):
            assert e["request_id"] == ""
            assert e["payload"]["model_id"] == TEST_REF
            assert e["payload"]["provider"] == "tensorhub"
            assert e["payload"]["source"] == "tensorhub_cas"


def test_download_success_emits_worker_lifecycle_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    blobs_root = _cozy_blobs_root(tmp_path)
    files = [_FakeFile(_digest_for(f"cmd-{i}"), 10 * 1024 * 1024) for i in range(3)]
    entry = _FakeResolvedEntry(files)
    downloaded_dir = tmp_path / "snapshot-progress"
    downloaded_dir.mkdir()

    w = _bare_worker(tmp_path)
    w._resolved_repos_by_id_baseline = {TEST_REF: entry}
    w._downloader = MagicMock()

    def fake_download(ref: str, dest_dir: str, filename: str | None = None) -> str:
        for f in files:
            p = _blob_path(blobs_root, f.blake3)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\0" * f.size_bytes)
            time.sleep(0.45)
        return str(downloaded_dir)

    w._downloader.download = fake_download

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    events = _worker_events(w)
    types = [e["type"] for e in events]
    assert "worker.model.download.started" in types, types
    assert "worker.model.download.progress" in types, types
    assert "worker.model.download.completed" in types, types
    progress = [e for e in events if e["type"] == "worker.model.download.progress"]
    assert 1 <= len(progress) <= 25
    assert progress[-1]["payload"]["bytes_total"] == 30 * 1024 * 1024


@pytest.mark.parametrize(
    ("provider", "source", "ref"),
    [
        ("hf", "huggingface", "owner/model#bf16"),
        ("civitai", "civitai", "123456"),
    ],
)
def test_download_model_command_provider_progress_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    source: str,
    ref: str,
) -> None:
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    downloaded_dir = tmp_path / f"snapshot-{provider}"
    downloaded_dir.mkdir()

    class _ProviderDownloader:
        def download(self, model_ref: str, dest_dir: str, filename: str | None = None) -> str:  # noqa: ARG002
            return str(downloaded_dir)

        def download_with_progress(self, model_ref: str, dest_dir: str, filename: str | None = None, progress_callback=None) -> str:  # noqa: ANN001, ARG002
            assert model_ref == ref
            if progress_callback is not None:
                progress_callback(0, 100)
                time.sleep(1.05)
                progress_callback(50, 100)
                time.sleep(1.05)
                progress_callback(100, 100)
            return str(downloaded_dir)

    w = _bare_worker(tmp_path)
    w._provider_by_ref_index = {ref: provider}
    w._downloader = _ProviderDownloader()

    w._run_download_model_cmd(echo_model_id=ref, canonical_ref=ref)

    events = [e for e in _worker_events(w) if e["type"].startswith("worker.model.download.")]
    assert [e["type"] for e in events][0] == "worker.model.download.started"
    assert [e["type"] for e in events][-1] == "worker.model.download.completed"
    progress = [e for e in events if e["type"] == "worker.model.download.progress"]
    assert progress
    for e in events:
        assert e["payload"]["provider"] == provider
        assert e["payload"]["source"] == source
    assert progress[-1]["payload"]["bytes_total"] == 100


def test_download_success_forces_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a successful download the worker must trigger an immediate
    `_register_worker(is_heartbeat=True)` so the next wire heartbeat
    reflects the new disk_models entry without waiting 10s.
    """
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    downloaded_dir = tmp_path / "snapshot-hb"
    downloaded_dir.mkdir()

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = lambda ref, dest_dir, filename=None: str(downloaded_dir)

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    assert w._register_worker_calls, "expected at least one _register_worker call after download"
    # The force-heartbeat path must be is_heartbeat=True; is_heartbeat=False
    # makes the orchestrator treat it as a fresh registration.
    assert any(hb is True for hb in w._register_worker_calls), (
        f"expected is_heartbeat=True, got {w._register_worker_calls}"
    )


def test_download_success_updates_disk_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After success the model_cache's get_disk_models() must include the
    new ref — this is what the next heartbeat's WorkerResources.disk_models
    snapshot reads from.
    """
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)
    downloaded_dir = tmp_path / "snapshot-disk"
    downloaded_dir.mkdir()

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = lambda ref, dest_dir, filename=None: str(downloaded_dir)

    assert TEST_REF not in w._model_cache.get_disk_models()  # sanity precondition

    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    assert TEST_REF in w._model_cache.get_disk_models(), (
        f"expected {TEST_REF!r} in disk_models, got {w._model_cache.get_disk_models()}"
    )


def test_download_failure_sends_load_model_result_with_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the downloader raises, the worker must NOT crash and MUST send
    LoadModelResult{success=False, error_message=<reason>} so the
    orchestrator's pending tracker can be cleared / surfaced.
    """
    monkeypatch.setattr("gen_worker.worker.tensorhub_cas_dir", lambda: tmp_path)

    def boom(ref: str, dest_dir: str, filename: str | None = None) -> str:
        raise RuntimeError("upstream 403")

    w = _bare_worker(tmp_path)
    w._downloader = MagicMock()
    w._downloader.download = boom

    # Must not raise.
    w._run_download_model_cmd(echo_model_id=TEST_REF, canonical_ref=TEST_REF)

    result = _last_load_model_result(w)
    assert result.success is False
    assert "upstream 403" in result.error_message
    # No DOWNLOAD_COMPLETED signal should have fired.
    kinds = {s.kind for s in _model_ready_signals(w)}
    assert pb.MODEL_AVAILABILITY_DOWNLOAD_COMPLETED not in kinds
    # No force-heartbeat on failure (avoids advertising a ref that isn't there).
    assert not any(hb is True for hb in w._register_worker_calls)

    events = _worker_events(w)
    types = [e["type"] for e in events]
    assert "worker.model.download.started" in types, types
    assert "worker.model.download.failed" in types, types
    assert "model.download.failed" not in types, types
    failed = [e for e in events if e["type"] == "worker.model.download.failed"][-1]["payload"]
    assert failed["provider"] == "tensorhub"
    assert failed["source"] == "tensorhub_cas"


def test_download_empty_ref_short_circuits_with_error(tmp_path: Path) -> None:
    """A DownloadModelCommand with neither model_id nor ref is a
    structural orchestrator bug; the worker should reject it via
    LoadModelResult without invoking the downloader.
    """
    w = _bare_worker(tmp_path)
    download_calls: List[Any] = []
    w._downloader = MagicMock()
    w._downloader.download = lambda *a, **kw: (download_calls.append(a) or "")  # type: ignore[func-returns-value]

    cmd = pb.DownloadModelCommand(model_id="", ref="")
    w._handle_download_model_cmd(cmd)

    assert not download_calls, "downloader should not be invoked for empty ref"
    result = _last_load_model_result(w)
    assert result.success is False
    assert result.error_message  # non-empty reason


def test_download_handler_dispatches_via_process_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The message-dispatch switch in _process_message routes the
    `download_model_cmd` oneof variant to `_handle_download_model_cmd`.
    Without this wire-up the orchestrator could send the command and the
    worker would silently no-op.
    """
    w = _bare_worker(tmp_path)
    seen: List[Any] = []
    w._handle_download_model_cmd = lambda cmd: seen.append(cmd)

    cmd = pb.DownloadModelCommand(model_id=TEST_REF, ref=TEST_REF)
    msg = pb.WorkerSchedulerMessage(download_model_cmd=cmd)
    w._process_message(msg)

    assert len(seen) == 1
    assert seen[0].model_id == TEST_REF
    assert seen[0].ref == TEST_REF
