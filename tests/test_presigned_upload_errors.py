from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import gen_worker.presigned_upload as presigned_upload_module
import gen_worker.s3_transfer as s3_transfer_module
from gen_worker._upload_transport import TransportError
from gen_worker.api.errors import ArtifactTransferError
from gen_worker.presigned_upload import presigned_upload_file
from gen_worker.s3_transfer import S3TransferGrant, S3TransferResult, download_file_with_grant
from gen_worker.worker import Worker


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("invalid json")
        return self._payload


def test_presigned_part_failure_surfaces_transfer_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "weights.safetensors"
    src.write_bytes(b"abcdef")

    post_calls = 0
    aborts: list[str] = []

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        nonlocal post_calls
        post_calls += 1
        assert post_calls == 1
        return _FakeResponse(
            200,
            {
                "upload_id": "upload-1",
                "part_urls": ["https://r2.example.test/part-1"],
                "part_size": 6,
                "total_parts": 1,
            },
            text="{}",
        )

    def fake_delete(url: str, **kwargs: Any) -> _FakeResponse:
        aborts.append(url)
        return _FakeResponse(204, {}, "")

    def fake_upload_part_to_presigned_url(**kwargs: Any) -> str:
        raise TransportError("S3 part upload tls error: bad record mac", retryable=True)

    monkeypatch.setattr(presigned_upload_module.requests, "post", fake_post)
    monkeypatch.setattr(presigned_upload_module.requests, "delete", fake_delete)
    monkeypatch.setattr(presigned_upload_module, "upload_part_to_presigned_url", fake_upload_part_to_presigned_url)

    with pytest.raises(ArtifactTransferError) as exc_info:
        presigned_upload_file(
            file_path=src,
            base_url="https://tensorhub.example.test",
            endpoint_path="/api/v1/media/root/uploads",
            headers={"Authorization": "Bearer token"},
            create_payload={"path": "weights.safetensors"},
            blake3_hex="abc",
            size_bytes=6,
        )

    err = exc_info.value
    assert err.provider == "tensorhub"
    assert err.phase == "put"
    assert err.retryable is True
    assert "bad record mac" in str(err)
    assert aborts == ["https://tensorhub.example.test/api/v1/media/root/uploads/upload-1"]


def test_presigned_finalize_failure_keeps_status_and_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "weights.safetensors"
    src.write_bytes(b"abcdef")

    responses = [
        _FakeResponse(
            200,
            {
                "upload_id": "upload-1",
                "part_urls": ["https://r2.example.test/part-1"],
                "part_size": 6,
                "total_parts": 1,
            },
            text="{}",
        ),
        _FakeResponse(503, {}, "r2 complete throttled"),
        _FakeResponse(503, {}, "r2 complete throttled"),
        _FakeResponse(503, {}, "r2 complete throttled"),
        _FakeResponse(503, {}, "r2 complete throttled"),
        _FakeResponse(503, {}, "r2 complete throttled"),
    ]

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(presigned_upload_module.requests, "post", fake_post)
    monkeypatch.setattr(presigned_upload_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(presigned_upload_module, "upload_part_to_presigned_url", lambda **_kwargs: '"etag"')

    with pytest.raises(ArtifactTransferError) as exc_info:
        presigned_upload_file(
            file_path=src,
            base_url="https://tensorhub.example.test",
            endpoint_path="/api/v1/media/root/uploads",
            headers={"Authorization": "Bearer token"},
            create_payload={"path": "weights.safetensors"},
            blake3_hex="abc",
            size_bytes=6,
        )

    err = exc_info.value
    assert err.phase == "complete"
    assert err.status_code == 503
    assert err.retryable is True
    assert "r2 complete throttled" in str(err)


def test_model_weight_upload_requires_transfer_grant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "weights.safetensors"
    src.write_bytes(b"abcdef")

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            200,
            {
                "upload_id": "upload-1",
                "part_urls": ["https://r2.example.test/part-1"],
                "part_size": 6,
                "total_parts": 1,
            },
            text="{}",
        )

    monkeypatch.setattr(presigned_upload_module.requests, "post", fake_post)

    with pytest.raises(ArtifactTransferError) as exc_info:
        presigned_upload_file(
            file_path=src,
            base_url="https://tensorhub.example.test",
            endpoint_path="/api/v1/repos/root/model/uploads",
            headers={"Authorization": "Bearer token"},
            create_payload={"path": "weights.safetensors"},
            blake3_hex="abc",
            size_bytes=6,
        )

    err = exc_info.value
    assert err.provider == "tensorhub"
    assert err.phase == "create"
    assert err.retryable is False
    assert "transfer_grant" in str(err)


def test_worker_maps_artifact_transfer_error_to_retryable_result() -> None:
    worker = Worker.__new__(Worker)
    error_type, retryable, safe_message, internal = worker._map_exception(
        ArtifactTransferError(
            "tensorhub R2 multipart PUT failed: tls retry exhausted",
            provider="tensorhub",
            phase="put",
            retryable=True,
            cause_type="TransportError",
        )
    )

    assert error_type == "artifact_transfer"
    assert retryable is True
    assert "tls retry exhausted" in safe_message
    assert "ArtifactTransferError" in internal


def test_presigned_upload_uses_sdk_transfer_grant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "weights.safetensors"
    src.write_bytes(b"abcdef")

    complete_payloads: list[dict[str, Any]] = []
    responses = [
        _FakeResponse(
            200,
            {
                "upload_id": "upload-1",
                "transfer_grant": {
                    "endpoint_url": "https://account.r2.cloudflarestorage.com",
                    "bucket": "repo-cas",
                    "key": "uploads/req-1/weights.safetensors",
                    "access_key_id": "access",
                    "secret_access_key": "secret",
                    "session_token": "session",
                },
            },
            text="{}",
        ),
        _FakeResponse(200, {"ok": True}, text="{}"),
    ]

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        if len(responses) == 1:
            complete_payloads.append(presigned_upload_module.json.loads(kwargs["data"]))
        return responses.pop(0)

    def fake_upload_file_with_grant(**kwargs: Any) -> S3TransferResult:
        grant = kwargs["grant"]
        assert grant.bucket == "repo-cas"
        assert grant.key == "uploads/req-1/weights.safetensors"
        return S3TransferResult(bucket=grant.bucket, key=grant.key, size_bytes=6, blake3="abc", etag='"etag"')

    monkeypatch.setattr(presigned_upload_module.requests, "post", fake_post)
    monkeypatch.setattr(s3_transfer_module, "upload_file_with_grant", fake_upload_file_with_grant)

    result = presigned_upload_file(
        file_path=src,
        base_url="https://tensorhub.example.test",
        endpoint_path="/api/v1/repos/root/model/uploads",
        headers={"Authorization": "Bearer token"},
        create_payload={"path": "weights.safetensors"},
        blake3_hex="abc",
        size_bytes=6,
    )

    assert result.meta == {"ok": True}
    assert complete_payloads == [
        {
            "transfer": {
                "mode": "s3_sdk",
                "bucket": "repo-cas",
                "key": "uploads/req-1/weights.safetensors",
                "size_bytes": 6,
                "blake3": "abc",
                "etag": '"etag"',
            }
        }
    ]


def test_transfer_grant_requires_scoped_credentials() -> None:
    with pytest.raises(ArtifactTransferError, match="missing required fields"):
        S3TransferGrant.from_mapping({"bucket": "repo-cas", "key": "objects/a"})


def test_sdk_transfer_config_uses_bounded_threaded_multipart() -> None:
    config = s3_transfer_module._transfer_config()
    retry_config = s3_transfer_module._transfer_config(max_concurrency=1)

    assert config.max_concurrency == 10
    assert retry_config.max_concurrency == 1
    assert config.use_threads is True


def test_sdk_upload_uses_boto_transfer_with_isolated_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "weights.safetensors"
    src.write_bytes(b"abcdefghijkl")
    state: dict[str, Any] = {"uploads": [], "closed": 0}

    class _FakeClient:
        def upload_file(self, filename: str, bucket: str, key: str, Config: Any, Callback: Any) -> None:
            state["uploads"].append((filename, bucket, key, Config.max_concurrency))
            Callback(5)
            Callback(7)

        def close(self) -> None:
            state["closed"] += 1

    monkeypatch.setattr(s3_transfer_module, "_s3_client", lambda _grant: _FakeClient())
    progress: list[int] = []

    result = s3_transfer_module.upload_file_with_grant(
        file_path=src,
        grant=S3TransferGrant(
            endpoint_url="https://account.r2.cloudflarestorage.com",
            bucket="repo-cas",
            key="objects/a",
            access_key_id="access",
            secret_access_key="secret",
            session_token="session",
        ),
        blake3_hex="abc",
        size_bytes=12,
        on_progress=lambda _done, _total, current: progress.append(current),
    )

    assert state["uploads"] == [(str(src), "repo-cas", "objects/a", 10)]
    assert state["closed"] == 1
    assert progress[-1] == 12


def test_sdk_upload_retries_with_lower_multipart_concurrency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "weights.safetensors"
    src.write_bytes(b"abcdefghijkl")
    state: dict[str, Any] = {"configs": [], "clients": 0, "sleeps": []}

    class _FakeClient:
        def upload_file(self, filename: str, bucket: str, key: str, Config: Any, Callback: Any) -> None:
            state["configs"].append(Config.max_concurrency)
            if len(state["configs"]) < 3:
                raise RuntimeError("tls alert bad record mac")
            Callback(12)

        def close(self) -> None:
            state["clients"] += 1

    monkeypatch.setattr(s3_transfer_module, "_s3_client", lambda _grant: _FakeClient())
    monkeypatch.setattr(s3_transfer_module.time, "sleep", lambda seconds: state["sleeps"].append(seconds))

    result = s3_transfer_module.upload_file_with_grant(
        file_path=src,
        grant=S3TransferGrant(
            endpoint_url="https://account.r2.cloudflarestorage.com",
            bucket="repo-cas",
            key="objects/a",
            access_key_id="access",
            secret_access_key="secret",
            session_token="session",
        ),
        blake3_hex="abc",
        size_bytes=12,
        on_progress=lambda *_args: None,
    )

    assert result.size_bytes == 12
    assert state["configs"] == [10, 5, 1]
    assert state["clients"] == 3
    assert state["sleeps"] == [1, 2]


def test_sdk_download_rejects_blake3_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeClient:
        def download_file(self, bucket: str, key: str, filename: str, Config: Any) -> None:
            Path(filename).write_bytes(b"not the expected content")

    monkeypatch.setattr(s3_transfer_module, "_s3_client", lambda _grant: _FakeClient())
    monkeypatch.setattr(s3_transfer_module, "_transfer_config", lambda: object())

    with pytest.raises(ArtifactTransferError, match="BLAKE3 mismatch"):
        download_file_with_grant(
            grant=S3TransferGrant(
                endpoint_url="https://account.r2.cloudflarestorage.com",
                bucket="repo-cas",
                key="objects/a",
                access_key_id="access",
                secret_access_key="secret",
                session_token="session",
            ),
            dest_path=tmp_path / "out.safetensors",
            expected_blake3="0" * 64,
        )
