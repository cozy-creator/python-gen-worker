from __future__ import annotations

import os
import threading
import json
from pathlib import Path

import requests
import pytest
import gen_worker.request_context as rc
from gen_worker.api.types import Tensors
from gen_worker.worker import RequestContext


def test_request_context_local_output_backend(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid1",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    ref = "jobs/rid1/outputs/hello.bin"
    asset = ctx.save_bytes(ref, b"abc")
    assert asset.ref == ref
    assert asset.local_path is not None
    p = Path(asset.local_path)
    assert p.exists()
    assert p.read_bytes() == b"abc"


def test_request_context_save_checkpoint_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid2",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    local = tmp_path / "converted.safetensors"
    local.write_bytes(b"weights")
    tensors = ctx.save_checkpoint("jobs/rid2/outputs/final.safetensors", str(local))
    assert isinstance(tensors, Tensors)
    assert tensors.ref == "jobs/rid2/outputs/final.safetensors"
    assert tensors.local_path is not None
    assert tensors.format == "safetensors"
    assert tensors.sha256


def test_request_context_open_output_stream_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid3",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    with ctx.open_output_stream("jobs/rid3/outputs/chunked.bin") as stream:
        stream.write(b"ab")
        stream.write(b"cd")
        assert stream.bytes_written == 4
        asset = stream.finalize()
    assert asset.ref == "jobs/rid3/outputs/chunked.bin"
    assert asset.local_path is not None
    assert Path(asset.local_path).read_bytes() == b"abcd"


def test_request_context_open_checkpoint_stream_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid4",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    with ctx.open_checkpoint_stream("jobs/rid4/outputs/final.safetensors") as stream:
        stream.write(b"we")
        stream.write(b"ights")
        tensors = stream.finalize()
    assert isinstance(tensors, Tensors)
    assert tensors.format == "safetensors"
    assert tensors.local_path is not None
    assert tensors.stream_mode == "local_fallback"
    assert Path(tensors.local_path).read_bytes() == b"weights"


def test_request_context_stream_checkpoint_matches_save_checkpoint(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid5",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh")

    saved_file = ctx.save_checkpoint("jobs/rid5/outputs/by-file.safetensors", str(source))
    with ctx.open_checkpoint_stream("jobs/rid5/outputs/by-stream.safetensors", format="safetensors") as stream:
        stream.write(b"abcd")
        stream.write(b"efgh")
        saved_stream = stream.finalize()

    assert saved_file.local_path is not None
    assert saved_stream.local_path is not None
    assert Path(saved_file.local_path).read_bytes() == Path(saved_stream.local_path).read_bytes()


def test_request_context_stream_finalize_failure_cleans_temp_file(tmp_path: Path, monkeypatch) -> None:
    stream_tmp = tmp_path / "stream-tmp.bin"

    def _mkstemp(*, prefix: str, suffix: str) -> tuple[int, str]:
        _ = prefix
        _ = suffix
        fd = os.open(stream_tmp, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
        return fd, str(stream_tmp)

    monkeypatch.setattr(rc.tempfile, "mkstemp", _mkstemp)

    ctx = RequestContext("rid6", local_output_dir=str(tmp_path), owner="o1", invoker_id="u1")

    def _fail_save_checkpoint(ref: str, local_path: str, format: str | None = None) -> Tensors:
        _ = ref
        _ = local_path
        _ = format
        raise RuntimeError("boom")

    monkeypatch.setattr(ctx, "save_checkpoint", _fail_save_checkpoint)

    writer = ctx.open_checkpoint_stream("jobs/rid6/outputs/fail.safetensors", format="safetensors")
    writer.write(b"weights")
    with pytest.raises(RuntimeError, match="boom"):
        writer.finalize()
    assert not stream_tmp.exists()


def test_request_context_save_file_network_error_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "upload.bin"
    src.write_bytes(b"abc")
    ctx = RequestContext(
        "rid7",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )

    def _fail_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        _ = kwargs
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "post", _fail_post)

    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        ctx.save_file("jobs/rid7/outputs/upload.bin", str(src))


def test_request_context_open_checkpoint_stream_remote_presigned_upload(monkeypatch) -> None:
    from gen_worker.presigned_upload import PresignedUploadResult
    import gen_worker.presigned_upload as pu

    ctx = RequestContext(
        "rid8",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )

    upload_calls: list[dict] = []

    def _mock_presigned_upload(*, file_path, base_url, endpoint_path, headers, create_payload, blake3_hex, size_bytes, **kwargs):
        upload_calls.append({"file_path": file_path, "blake3": blake3_hex, "size": size_bytes, "endpoint": endpoint_path})
        return PresignedUploadResult(meta={"size_bytes": size_bytes, "sha256": "abcd", "blake3": blake3_hex}, dedup=False)

    monkeypatch.setattr(pu, "presigned_upload_file", _mock_presigned_upload)

    writer = ctx.open_checkpoint_stream("jobs/rid8/outputs/final.safetensors", format="safetensors")
    writer.write(b"ab")
    writer.write(b"cd")
    out = writer.finalize()

    assert out.ref == "jobs/rid8/outputs/final.safetensors"
    assert out.local_path is None
    assert out.format == "safetensors"
    assert out.stream_mode == "presigned"
    assert len(upload_calls) == 1
    assert upload_calls[0]["size"] == 4
    assert upload_calls[0]["endpoint"] == "/api/v1/media/uploads"


def test_request_context_open_checkpoint_stream_requires_repo_job_scope_for_conversion() -> None:
    ctx = RequestContext(
        "rid8-scope",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
        execution_hints={"kind": "conversion"},
    )
    with pytest.raises(RuntimeError, match="tensor upload requires repo job scope"):
        ctx.open_checkpoint_stream("jobs/rid8-scope/outputs/final.safetensors")


def test_request_context_save_checkpoint_requires_repo_job_scope_for_training(tmp_path: Path) -> None:
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"abcd")
    ctx = RequestContext(
        "rid8-save-scope",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
        execution_hints={"kind": "training"},
    )
    with pytest.raises(RuntimeError, match="tensor upload requires repo job scope"):
        ctx.save_checkpoint("jobs/rid8-save-scope/outputs/final.safetensors", str(src))


def test_request_context_open_checkpoint_stream_emits_upload_progress(monkeypatch) -> None:
    pytest.skip("TODO(#213): rewrite for presigned multipart upload flow")
    events: list[dict[str, object]] = []

    def _emit(evt: dict[str, object]) -> None:
        events.append(evt)

    ctx = RequestContext(
        "rid8-progress",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
        emitter=_emit,
    )

    offset = 0

    class _Resp:
        def __init__(self, status_code: int, body: dict[str, object] | None = None, headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self._body = body or {}
            self.headers = headers or {}

        def json(self) -> dict[str, object]:
            return dict(self._body)

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal offset
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/media/uploads"):
            return _Resp(201, {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 8 * 1024 * 1024})
        if method == "PATCH" and url.endswith("/api/v1/media/uploads/sess-1"):
            assert headers.get("Upload-Offset") == str(offset)
            offset += len(data)
            return _Resp(204, headers={"Upload-Offset": str(offset)})
        if method == "POST" and url.endswith("/api/v1/media/uploads/sess-1/complete"):
            return _Resp(200, {"size_bytes": 4, "sha256": "abcd"})
        if method == "DELETE" and url.endswith("/api/v1/media/uploads/sess-1"):
            return _Resp(200, {"ok": True})
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)
    monkeypatch.setenv("WORKER_STREAM_PROGRESS_INTERVAL_S", "0")

    writer = ctx.open_checkpoint_stream("jobs/rid8-progress/outputs/final.safetensors", format="safetensors")
    writer.write(b"ab")
    writer.write(b"cd")
    out = writer.finalize()
    assert out.stream_mode == "remote_append"

    progress_events = [e for e in events if e.get("type") == "request.upload_progress"]
    assert progress_events
    last_payload = progress_events[-1].get("payload") or {}
    assert last_payload.get("stream_mode") == "remote_append"
    assert int(last_payload.get("bytes_uploaded") or 0) >= 4
    assert int(last_payload.get("chunks_uploaded") or 0) >= 1
    assert int(last_payload.get("chunks_written") or 0) >= 1
    assert float(last_payload.get("upload_bps") or 0.0) >= 0.0
    assert float(last_payload.get("finalize_elapsed_s") or 0.0) >= 0.0


def test_request_context_open_output_stream_remote_network_error_is_deterministic(monkeypatch) -> None:
    pytest.skip("TODO(#213): rewrite for presigned multipart upload flow")
    ctx = RequestContext(
        "rid9",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        _ = method
        _ = url
        _ = headers
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/media/uploads"):
            class _Created:
                status_code = 201
                headers: dict[str, str] = {}

                @staticmethod
                def json() -> dict[str, object]:
                    return {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 8 * 1024 * 1024}

            return _Created()
        if method == "PATCH" and url.endswith("/api/v1/media/uploads/sess-1"):
            raise requests.ConnectionError("network down")
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_output_stream("jobs/rid9/outputs/upload.bin")
    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        writer.write(b"abcd")
        writer.finalize()


def test_request_context_open_checkpoint_stream_remote_replay_recovery(monkeypatch) -> None:
    pytest.skip("TODO(#213): replay mechanism removed in presigned flow")
    ctx = RequestContext(
        "rid10",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )
    calls = {"patch": 0}
    offset = 0

    class _Resp:
        def __init__(self, status_code: int, body: dict[str, object] | None = None, headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self._body = body or {}
            self.headers = headers or {}

        def json(self) -> dict[str, object]:
            return dict(self._body)

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal offset
        _ = headers
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/media/uploads"):
            return _Resp(201, {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 8 * 1024 * 1024})
        if method == "PATCH" and url.endswith("/api/v1/media/uploads/sess-1"):
            calls["patch"] += 1
            if calls["patch"] <= 2:
                # Initial uploader exhausts retries; replay then succeeds.
                raise requests.ConnectionError("transient network")
            offset += len(data)
            return _Resp(204, headers={"Upload-Offset": str(offset)})
        if method == "POST" and url.endswith("/api/v1/media/uploads/sess-1/complete"):
            return _Resp(200, {"size_bytes": 4, "sha256": "abcd"})
        if method == "DELETE" and url.endswith("/api/v1/media/uploads/sess-1"):
            return _Resp(200, {"ok": True})
        if method == "HEAD" and url.endswith("/api/v1/media/uploads/sess-1"):
            return _Resp(204, {}, headers={"Upload-Offset": str(offset)})
        if method == "PATCH":
            raise requests.ConnectionError("transient network")
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)
    monkeypatch.setenv("WORKER_STREAM_UPLOAD_RETRY_ATTEMPTS", "2")

    writer = ctx.open_checkpoint_stream("jobs/rid10/outputs/final.safetensors", format="safetensors")
    try:
        writer.write(b"ab")
        writer.write(b"cd")
    except RuntimeError:
        pass
    out = writer.finalize()
    assert out.ref == "jobs/rid10/outputs/final.safetensors"
    assert out.stream_mode == "remote_append"
    assert calls["patch"] >= 3


def test_request_context_open_checkpoint_stream_cancel_aborts_remote(monkeypatch) -> None:
    pytest.skip("TODO(#213): rewrite for presigned multipart upload flow")
    events: list[dict[str, object]] = []

    def _emit(evt: dict[str, object]) -> None:
        events.append(evt)

    ctx = RequestContext(
        "rid11",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
        emitter=_emit,
    )

    offset = 0

    class _Resp:
        def __init__(self, status_code: int, body: dict[str, object] | None = None, headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self._body = body or {}
            self.headers = headers or {}

        def json(self) -> dict[str, object]:
            return dict(self._body)

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal offset
        _ = method
        _ = headers
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/media/uploads"):
            return _Resp(201, {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 8 * 1024 * 1024})
        if method == "PATCH" and url.endswith("/api/v1/media/uploads/sess-1"):
            offset += len(data)
            return _Resp(204, {}, headers={"Upload-Offset": str(offset)})
        if method == "DELETE" and url.endswith("/api/v1/media/uploads/sess-1"):
            return _Resp(200, {"ok": True})
        raise RuntimeError(f"unexpected request completion: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_checkpoint_stream("jobs/rid11/outputs/final.safetensors", format="safetensors")
    writer.write(b"ab")
    ctx.cancel()
    with pytest.raises(InterruptedError, match="canceled"):
        writer.write(b"cd")
    writer.close()

    progress_events = [e for e in events if e.get("type") == "request.upload_progress"]
    assert progress_events
    assert any((e.get("payload") or {}).get("stage") in {"stream_canceled", "stream_aborted"} for e in progress_events)


def test_request_context_open_output_stream_remote_finalize_recovery_failure_is_deterministic(monkeypatch) -> None:
    pytest.skip("TODO(#213): replay mechanism removed in presigned flow")
    ctx = RequestContext(
        "rid12",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        _ = method
        _ = url
        _ = headers
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/media/uploads"):
            class _Created:
                status_code = 201
                headers: dict[str, str] = {}

                @staticmethod
                def json() -> dict[str, object]:
                    return {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 8 * 1024 * 1024}

            return _Created()
        if method == "PATCH" and url.endswith("/api/v1/media/uploads/sess-1"):
            raise requests.ConnectionError("network down")
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "request", _req)
    monkeypatch.setenv("WORKER_STREAM_UPLOAD_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("WORKER_STREAM_UPLOAD_RETRY_BACKOFF_MS", "0")

    writer = ctx.open_output_stream("jobs/rid12/outputs/upload.bin")
    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        writer.write(b"abcd")
        writer.finalize()


def test_request_context_open_checkpoint_stream_session_append_finalize(monkeypatch) -> None:
    pytest.skip("TODO(#213): rewrite for presigned multipart upload flow")
    ctx = RequestContext(
        "rid13",
        job_id="run13",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )
    seen_chunks: list[tuple[int, bytes]] = []
    finalize_payloads: list[dict[str, object]] = []
    current_offset = 0

    class _Resp:
        def __init__(self, code: int, body: dict[str, object], headers: dict[str, str] | None = None) -> None:
            self.status_code = code
            self._body = body
            self.headers = headers or {}

        def json(self) -> dict[str, object]:
            return dict(self._body)

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal current_offset
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/media/uploads"):
            payload = json.loads(data)
            assert payload["ref"] == "jobs/rid13/outputs/final.safetensors"
            assert payload["request_id"] == "rid13"
            assert payload["job_id"] == "run13"
            assert int(payload.get("upload_length") or 0) == 4
            return _Resp(201, {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 2})
        if method == "PATCH" and url.endswith("/api/v1/media/uploads/sess-1"):
            seq = current_offset // 2
            assert headers.get("Upload-Offset") == str(current_offset)
            assert isinstance(data, (bytes, bytearray))
            seen_chunks.append((seq, bytes(data)))
            current_offset += len(data)
            return _Resp(204, {}, headers={"Upload-Offset": str(current_offset)})
        if method == "POST" and url.endswith("/api/v1/media/uploads/sess-1/complete"):
            payload = json.loads(data)
            finalize_payloads.append(payload)
            return _Resp(200, {"size_bytes": 4, "sha256": payload["final_sha256"], "mime_type": "application/octet-stream"})
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_checkpoint_stream(
        "jobs/rid13/outputs/final.safetensors",
        format="safetensors",
        expected_size_bytes=4,
    )
    assert str(getattr(writer, "_replay_path", "")).strip() != ""
    writer.write(b"abcd")
    out = writer.finalize()
    assert out.ref == "jobs/rid13/outputs/final.safetensors"
    assert out.stream_mode == "remote_append"
    assert seen_chunks == [(0, b"ab"), (1, b"cd")]
    assert len(finalize_payloads) == 1
    assert int(finalize_payloads[0].get("final_size_bytes") or 0) == 4


def test_request_context_open_checkpoint_stream_repo_job_upload(monkeypatch) -> None:
    pytest.skip("TODO(#213): rewrite for presigned multipart upload flow")
    ctx = RequestContext(
        "rid13-job",
        job_id="job13",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="alice",
        execution_hints={
            "kind": "conversion",
            "destination_repo": "alice/model-a",
            "job_id": "job13",
        },
    )
    seen_chunks: list[bytes] = []
    finalize_payloads: list[dict[str, object]] = []
    current_offset = 0

    class _Resp:
        def __init__(self, code: int, body: dict[str, object], headers: dict[str, str] | None = None) -> None:
            self.status_code = code
            self._body = body
            self.headers = headers or {}

        def json(self) -> dict[str, object]:
            return dict(self._body)

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal current_offset
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/repos/alice/model-a/jobs/job13/uploads"):
            payload = json.loads(data)
            assert payload["path"] == "jobs/rid13-job/outputs/final.safetensors"
            assert payload["request_id"] == "rid13-job"
            assert payload.get("job_id") is None
            assert int(payload.get("upload_length") or 0) == 4
            return _Resp(201, {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 2})
        if method == "PATCH" and url.endswith("/api/v1/repos/alice/model-a/jobs/job13/uploads/sess-1"):
            assert headers.get("Upload-Offset") == str(current_offset)
            assert isinstance(data, (bytes, bytearray))
            seen_chunks.append(bytes(data))
            current_offset += len(data)
            return _Resp(204, {}, headers={"Upload-Offset": str(current_offset)})
        if method == "POST" and url.endswith("/api/v1/repos/alice/model-a/jobs/job13/uploads/sess-1/complete"):
            payload = json.loads(data)
            finalize_payloads.append(payload)
            assert int(payload.get("final_size_bytes") or 0) == 4
            assert isinstance(payload.get("final_blake3"), str) and len(str(payload["final_blake3"])) == 64
            return _Resp(
                200,
                {
                    "size_bytes": 4,
                    "sha256": "abcd",
                    "blake3": str(payload["final_blake3"]),
                    "blob_domain": "private",
                    "blob_path": "alice/blobs/abcd",
                    "snapshot_digest": "blake3:snapshot1234",
                    "blob_digest": f"blake3:{payload['final_blake3']}",
                },
            )
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_checkpoint_stream(
        "jobs/rid13-job/outputs/final.safetensors",
        format="safetensors",
        expected_size_bytes=4,
    )
    writer.write(b"abcd")
    out = writer.finalize()
    assert out.ref == "jobs/rid13-job/outputs/final.safetensors"
    assert out.stream_mode == "remote_append"
    assert out.blob_digest is not None and out.blob_digest.startswith("blake3:")
    assert out.snapshot_digest == "blake3:snapshot1234"
    assert seen_chunks == [b"ab", b"cd"]
    assert len(finalize_payloads) == 1


def test_request_context_save_checkpoint_repo_job_upload(monkeypatch, tmp_path: Path) -> None:
    pytest.skip("TODO(#213): rewrite for presigned multipart upload flow")
    src = tmp_path / "model.safetensors"
    src.write_bytes(b"abcd")
    ctx = RequestContext(
        "rid13-save-job",
        job_id="job13save",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="alice",
        execution_hints={
            "kind": "training",
            "destination_repo": "alice/model-a",
            "job_id": "job13save",
        },
    )
    calls: list[tuple[str, str]] = []

    class _Resp:
        def __init__(self, code: int, body: dict[str, object], headers: dict[str, str] | None = None) -> None:
            self.status_code = code
            self._body = body
            self.headers = headers or {}

        def json(self) -> dict[str, object]:
            return dict(self._body)

    offset = 0

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal offset
        _ = headers
        _ = timeout
        calls.append((method, url))
        if method == "POST" and url.endswith("/api/v1/repos/alice/model-a/jobs/job13save/uploads"):
            payload = json.loads(data)
            assert payload["path"] == "jobs/rid13-save-job/outputs/final.safetensors"
            return _Resp(201, {"upload_id": "sess-1", "upload_offset": 0, "max_chunk_bytes": 2})
        if method == "PATCH" and url.endswith("/api/v1/repos/alice/model-a/jobs/job13save/uploads/sess-1"):
            offset += len(data)
            return _Resp(204, {}, headers={"Upload-Offset": str(offset)})
        if method == "POST" and url.endswith("/api/v1/repos/alice/model-a/jobs/job13save/uploads/sess-1/complete"):
            payload = json.loads(data)
            assert isinstance(payload.get("final_blake3"), str)
            assert int(payload.get("final_size_bytes") or 0) == 4
            return _Resp(
                200,
                {
                    "size_bytes": 4,
                    "sha256": "abcd",
                    "blake3": str(payload["final_blake3"]),
                    "blob_digest": f"blake3:{payload['final_blake3']}",
                    "snapshot_digest": "blake3:snapshot5678",
                },
            )
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)

    out = ctx.save_checkpoint("jobs/rid13-save-job/outputs/final.safetensors", str(src))
    assert out.ref == "jobs/rid13-save-job/outputs/final.safetensors"
    assert out.stream_mode == "remote_append"
    assert out.blob_digest is not None and out.blob_digest.startswith("blake3:")
    assert out.snapshot_digest == "blake3:snapshot5678"
    assert any(url.endswith("/api/v1/repos/alice/model-a/jobs/job13save/uploads") for _, url in calls)


def test_request_context_mirror_dedupe_or_run_hit_copies_and_writes_claim(monkeypatch) -> None:
    ctx = RequestContext("rid14", owner="alice")
    calls: dict[str, object] = {
        "search": None,
        "copy": None,
        "upsert": None,
        "miss": 0,
    }

    def _search_metadata_claims(**kwargs):  # type: ignore[no-untyped-def]
        calls["search"] = dict(kwargs)
        return {
            "items": [
                {
                    "claim_id": 11,
                    "owner": "alice",
                    "repo": "seed-model",
                    "version_id": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "metadata_json": {
                        "result": {
                            "primary_artifact_ref": "jobs/seed/outputs/weights.bin",
                            "primary_artifact_format": "bin",
                        }
                    },
                }
            ]
        }

    def _copy_repo_by_reference(**kwargs):  # type: ignore[no-untyped-def]
        calls["copy"] = dict(kwargs)
        return {
            "ok": True,
            "copied_version_id": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }

    def _upsert_version_metadata_claim(**kwargs):  # type: ignore[no-untyped-def]
        calls["upsert"] = dict(kwargs)
        return {"ok": True, "claim_id": 101}

    def _on_miss():
        calls["miss"] = int(calls["miss"] or 0) + 1
        return {
            "version_id": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        }

    monkeypatch.setattr(ctx, "search_metadata_claims", _search_metadata_claims)
    monkeypatch.setattr(ctx, "copy_repo_by_reference", _copy_repo_by_reference)
    monkeypatch.setattr(ctx, "upsert_version_metadata_claim", _upsert_version_metadata_claim)

    out = ctx.mirror_dedupe_or_run(
        source_identity={
            "provider": "huggingface",
            "source_ref": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "source_revision": "main",
            "identity_hash": "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "dedupe_supported": True,
        },
        destination_repo="alice/model-a",
        destination_repo_tags=["prod"],
        on_miss=_on_miss,
    )

    assert out.get("result_code") == "dedupe_copy_hit"
    assert out.get("copied_from_repo") == "alice/seed-model"
    assert out.get("copied_version_id") == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert out.get("claim_written") is True
    assert calls["miss"] == 0
    assert isinstance(calls["search"], dict)
    assert isinstance(calls["copy"], dict)
    assert isinstance(calls["upsert"], dict)
    assert (calls["search"] or {}).get("identity_hash") == "sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    assert (calls["copy"] or {}).get("destination_repo") == "alice/model-a"
    assert (calls["upsert"] or {}).get("destination_repo") == "alice/model-a"


def test_request_context_mirror_dedupe_or_run_invalidates_stale_claim_and_runs_miss(monkeypatch) -> None:
    ctx = RequestContext("rid15", owner="alice")
    deleted: list[dict[str, object]] = []
    upserts: list[dict[str, object]] = []

    def _search_metadata_claims(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return {
            "items": [
                {
                    "claim_id": 9,
                    "owner": "alice",
                    "repo": "stale-model",
                    "version_id": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                }
            ]
        }

    def _copy_repo_by_reference(**kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        raise RuntimeError("mirror_copy_forbidden:source_version_not_found")

    def _delete_version_metadata_claim(**kwargs):  # type: ignore[no-untyped-def]
        deleted.append(dict(kwargs))
        return {"ok": True, "deleted": True}

    def _upsert_version_metadata_claim(**kwargs):  # type: ignore[no-untyped-def]
        upserts.append(dict(kwargs))
        return {"ok": True, "claim_id": 88}

    monkeypatch.setattr(ctx, "search_metadata_claims", _search_metadata_claims)
    monkeypatch.setattr(ctx, "copy_repo_by_reference", _copy_repo_by_reference)
    monkeypatch.setattr(ctx, "delete_version_metadata_claim", _delete_version_metadata_claim)
    monkeypatch.setattr(ctx, "upsert_version_metadata_claim", _upsert_version_metadata_claim)

    out = ctx.mirror_dedupe_or_run(
        source_identity={
            "provider": "huggingface",
            "source_ref": "hf/model",
            "source_revision": "resolved-rev",
            "identity_hash": "sha256:9999999999999999999999999999999999999999999999999999999999999999",
            "dedupe_supported": True,
        },
        destination_repo="alice/model-a",
        destination_repo_tags=["prod"],
        on_miss=lambda: {
            "version_id": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "primary_artifact_ref": "jobs/rid15/outputs/ingested-source.bin",
            "primary_artifact_format": "bin",
        },
    )

    assert out.get("result_code") == "dedupe_miss"
    assert out.get("claim_written") is True
    assert out.get("invalidated_claim_ids") == [9]
    assert len(deleted) == 1
    assert deleted[0]["destination_repo"] == "alice/stale-model"
    assert len(upserts) == 1
    assert upserts[0]["destination_repo"] == "alice/model-a"
    assert upserts[0]["version_id"] == "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
