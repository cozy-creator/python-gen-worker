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


def test_action_context_local_output_backend(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid1",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    ref = "runs/rid1/outputs/hello.bin"
    asset = ctx.save_bytes(ref, b"abc")
    assert asset.ref == ref
    assert asset.local_path is not None
    p = Path(asset.local_path)
    assert p.exists()
    assert p.read_bytes() == b"abc"


def test_action_context_save_checkpoint_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid2",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    local = tmp_path / "converted.safetensors"
    local.write_bytes(b"weights")
    tensors = ctx.save_checkpoint("runs/rid2/outputs/final.safetensors", str(local))
    assert isinstance(tensors, Tensors)
    assert tensors.ref == "runs/rid2/outputs/final.safetensors"
    assert tensors.local_path is not None
    assert tensors.format == "safetensors"
    assert tensors.sha256


def test_action_context_open_output_stream_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid3",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    with ctx.open_output_stream("runs/rid3/outputs/chunked.bin") as stream:
        stream.write(b"ab")
        stream.write(b"cd")
        assert stream.bytes_written == 4
        asset = stream.finalize()
    assert asset.ref == "runs/rid3/outputs/chunked.bin"
    assert asset.local_path is not None
    assert Path(asset.local_path).read_bytes() == b"abcd"


def test_action_context_open_checkpoint_stream_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid4",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    with ctx.open_checkpoint_stream("runs/rid4/outputs/final.safetensors") as stream:
        stream.write(b"we")
        stream.write(b"ights")
        tensors = stream.finalize()
    assert isinstance(tensors, Tensors)
    assert tensors.format == "safetensors"
    assert tensors.local_path is not None
    assert tensors.stream_mode == "local_fallback"
    assert Path(tensors.local_path).read_bytes() == b"weights"


def test_action_context_stream_checkpoint_matches_save_checkpoint(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid5",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh")

    saved_file = ctx.save_checkpoint("runs/rid5/outputs/by-file.safetensors", str(source))
    with ctx.open_checkpoint_stream("runs/rid5/outputs/by-stream.safetensors", format="safetensors") as stream:
        stream.write(b"abcd")
        stream.write(b"efgh")
        saved_stream = stream.finalize()

    assert saved_file.local_path is not None
    assert saved_stream.local_path is not None
    assert Path(saved_file.local_path).read_bytes() == Path(saved_stream.local_path).read_bytes()


def test_action_context_stream_finalize_failure_cleans_temp_file(tmp_path: Path, monkeypatch) -> None:
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

    writer = ctx.open_checkpoint_stream("runs/rid6/outputs/fail.safetensors", format="safetensors")
    writer.write(b"weights")
    with pytest.raises(RuntimeError, match="boom"):
        writer.finalize()
    assert not stream_tmp.exists()


def test_action_context_save_file_network_error_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "upload.bin"
    src.write_bytes(b"abc")
    ctx = RequestContext(
        "rid7",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )

    def _fail_put(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        _ = kwargs
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "put", _fail_put)

    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        ctx.save_file("runs/rid7/outputs/upload.bin", str(src))


def test_action_context_open_checkpoint_stream_remote_upload_streams_before_finalize(monkeypatch) -> None:
    ctx = RequestContext(
        "rid8",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )

    saw_first_chunk = threading.Event()
    uploaded: list[bytes] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"size_bytes": 4, "sha256": "abcd"}

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        assert method == "PUT"
        assert "/api/v1/file/runs/rid8/outputs/final.safetensors" in url
        _ = headers
        _ = timeout
        for chunk in data:
            uploaded.append(bytes(chunk))
            saw_first_chunk.set()
        return _Resp()

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_checkpoint_stream("runs/rid8/outputs/final.safetensors", format="safetensors")
    writer.write(b"ab")
    assert saw_first_chunk.wait(timeout=1.0)
    writer.write(b"cd")
    out = writer.finalize()

    assert out.ref == "runs/rid8/outputs/final.safetensors"
    assert out.local_path is None
    assert out.format == "safetensors"
    assert out.stream_mode == "remote_append"
    assert b"".join(uploaded) == b"abcd"


def test_action_context_open_checkpoint_stream_emits_upload_progress(monkeypatch) -> None:
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

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"size_bytes": 4, "sha256": "abcd"}

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        _ = method
        _ = url
        _ = headers
        _ = timeout
        for _chunk in data:
            pass
        return _Resp()

    monkeypatch.setattr(requests, "request", _req)
    monkeypatch.setenv("WORKER_STREAM_PROGRESS_INTERVAL_S", "0")

    writer = ctx.open_checkpoint_stream("runs/rid8-progress/outputs/final.safetensors", format="safetensors")
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


def test_action_context_open_output_stream_remote_network_error_is_deterministic(monkeypatch) -> None:
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
        # Ensure generator starts consuming chunks.
        for _chunk in data:
            break
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_output_stream("runs/rid9/outputs/upload.bin")
    writer.write(b"abcd")
    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        writer.finalize()


def test_action_context_open_checkpoint_stream_remote_replay_recovery(monkeypatch) -> None:
    ctx = RequestContext(
        "rid10",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )
    calls = {"n": 0}

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"size_bytes": 4, "sha256": "abcd"}

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        _ = method
        _ = url
        _ = headers
        _ = timeout
        calls["n"] += 1
        if calls["n"] == 1:
            # Live stream fails after first consumed chunk.
            for _chunk in data:
                break
            raise requests.ConnectionError("transient network")
        # Replay path should send the full payload.
        if hasattr(data, "read"):
            assert data.read() == b"abcd"
        else:
            assert b"".join(data) == b"abcd"
        return _Resp()

    monkeypatch.setattr(requests, "request", _req)
    monkeypatch.setenv("WORKER_STREAM_UPLOAD_RETRY_ATTEMPTS", "2")

    writer = ctx.open_checkpoint_stream("runs/rid10/outputs/final.safetensors", format="safetensors")
    writer.write(b"ab")
    writer.write(b"cd")
    out = writer.finalize()
    assert out.ref == "runs/rid10/outputs/final.safetensors"
    assert out.stream_mode == "remote_append"
    assert calls["n"] == 2


def test_action_context_open_checkpoint_stream_cancel_aborts_remote(monkeypatch) -> None:
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

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        _ = method
        _ = url
        _ = headers
        _ = timeout
        for _chunk in data:
            pass
        raise RuntimeError("unexpected request completion")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_checkpoint_stream("runs/rid11/outputs/final.safetensors", format="safetensors")
    writer.write(b"ab")
    ctx.cancel()
    with pytest.raises(InterruptedError, match="canceled"):
        writer.write(b"cd")
    writer.close()

    progress_events = [e for e in events if e.get("type") == "request.upload_progress"]
    assert progress_events
    assert any((e.get("payload") or {}).get("stage") in {"stream_canceled", "stream_aborted"} for e in progress_events)


def test_action_context_open_output_stream_remote_finalize_recovery_failure_is_deterministic(monkeypatch) -> None:
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
        if hasattr(data, "read"):
            data.read()
        else:
            for _chunk in data:
                pass
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "request", _req)
    monkeypatch.setenv("WORKER_STREAM_UPLOAD_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("WORKER_STREAM_UPLOAD_RETRY_BACKOFF_MS", "0")

    writer = ctx.open_output_stream("runs/rid12/outputs/upload.bin")
    writer.write(b"abcd")
    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        writer.finalize()


def test_action_context_open_checkpoint_stream_session_append_finalize(monkeypatch) -> None:
    monkeypatch.setenv("WORKER_STREAM_APPEND_SESSION_ENABLED", "1")
    ctx = RequestContext(
        "rid13",
        run_id="run13",
        file_api_base_url="https://files.example.test",
        worker_capability_token="token",
        owner="o1",
    )
    seen_chunks: list[tuple[int, bytes]] = []
    finalize_payloads: list[dict[str, object]] = []

    class _Resp:
        def __init__(self, code: int, body: dict[str, object]) -> None:
            self.status_code = code
            self._body = body

        def json(self) -> dict[str, object]:
            return dict(self._body)

    def _req(*, method: str, url: str, headers: dict[str, str], data, timeout: int):  # type: ignore[no-untyped-def]
        _ = headers
        _ = timeout
        if method == "POST" and url.endswith("/api/v1/file/upload-sessions"):
            payload = json.loads(data)
            assert payload["ref"] == "runs/rid13/outputs/final.safetensors"
            assert payload["request_id"] == "rid13"
            assert payload["run_id"] == "run13"
            assert int(payload.get("expected_size_bytes") or 0) == 4
            return _Resp(200, {"session_id": "sess-1", "max_chunk_bytes": 2})
        if method == "PUT" and "/api/v1/file/upload-sessions/sess-1/chunks/" in url:
            seq = int(url.rsplit("/", 1)[-1])
            assert isinstance(data, (bytes, bytearray))
            seen_chunks.append((seq, bytes(data)))
            return _Resp(200, {"ok": True})
        if method == "POST" and url.endswith("/api/v1/file/upload-sessions/sess-1/finalize"):
            payload = json.loads(data)
            finalize_payloads.append(payload)
            return _Resp(200, {"size_bytes": 4, "sha256": payload["final_sha256"], "mime_type": "application/octet-stream"})
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(requests, "request", _req)

    writer = ctx.open_checkpoint_stream(
        "runs/rid13/outputs/final.safetensors",
        format="safetensors",
        expected_size_bytes=4,
    )
    assert str(getattr(writer, "_replay_path", "")).strip() != ""
    writer.write(b"abcd")
    out = writer.finalize()
    assert out.ref == "runs/rid13/outputs/final.safetensors"
    assert out.stream_mode == "remote_append"
    assert seen_chunks == [(0, b"ab"), (1, b"cd")]
    assert len(finalize_payloads) == 1
    assert int(finalize_payloads[0].get("final_size_bytes") or 0) == 4


def test_action_context_mirror_dedupe_or_run_hit_copies_and_writes_claim(monkeypatch) -> None:
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
                            "primary_artifact_ref": "runs/seed/outputs/weights.bin",
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


def test_action_context_mirror_dedupe_or_run_invalidates_stale_claim_and_runs_miss(monkeypatch) -> None:
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
            "primary_artifact_ref": "runs/rid15/outputs/ingested-source.bin",
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
