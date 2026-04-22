"""Tests for presigned_upload_batch — batch mode that collapses N single-item
presigns into one batch create + one batch complete."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gen_worker.presigned_upload import (
    PresignedBatchItemResult,
    presigned_upload_batch,
)


def _fake_response(status: int, body: dict | None = None, headers: dict | None = None):
    resp = MagicMock()
    resp.status_code = status
    resp.text = json.dumps(body) if body is not None else ""
    resp.json = lambda: (body or {})
    resp.headers = headers or {}
    return resp


def test_presigned_batch_empty_returns_empty(tmp_path):
    assert presigned_upload_batch(
        files=[],
        base_url="https://hub",
        batch_endpoint_path="/api/v1/datasets/x/uploads/batch",
        headers={},
    ) == []


def test_presigned_batch_all_dedup(tmp_path):
    """If every item dedups on create, no PUT/complete call is needed."""
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"\x00" * 16)
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"\x11" * 16)

    create_body = {
        "items": [
            {"index": 0, "dedup": True, "key": "blake3/aaa", "blake3": "a" * 64},
            {"index": 1, "dedup": True, "key": "blake3/bbb", "blake3": "b" * 64},
        ]
    }

    with patch("gen_worker.presigned_upload.requests") as mock_requests:
        mock_requests.post.return_value = _fake_response(200, create_body)
        mock_requests.RequestException = Exception

        results = presigned_upload_batch(
            files=[
                (str(f1), "a" * 64, 16, {"domain": "private"}),
                (str(f2), "b" * 64, 16, {"domain": "private"}),
            ],
            base_url="https://hub",
            batch_endpoint_path="/api/v1/datasets/x/uploads/batch",
            headers={"Authorization": "Bearer x"},
        )
    assert len(results) == 2
    assert all(r.dedup for r in results)
    assert all(r.ok for r in results)
    # One call only (create); no complete since everything dedupe'd.
    assert mock_requests.post.call_count == 1


def test_presigned_batch_mixed_dedup_and_upload(tmp_path):
    """One item dedups, one uploads. Expect 1 create + 1 PUT + 1 complete."""
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"X" * 32)
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"Y" * 32)

    create_body = {
        "items": [
            {"index": 0, "dedup": True, "key": "blake3/aaa"},
            {
                "index": 1,
                "upload_id": "u-2",
                "part_urls": ["https://s3/part-0"],
                "part_size": 64 * 1024 * 1024,
                "total_parts": 1,
            },
        ]
    }
    put_resp = _fake_response(200, headers={"ETag": '"etag-2"'})
    complete_body = {
        "items": [
            {"index": 0, "upload_id": "u-2", "ok": True, "key": "blake3/bbb"},
        ]
    }

    with patch("gen_worker.presigned_upload.requests") as mock_requests:
        mock_requests.RequestException = Exception
        mock_requests.post.side_effect = [
            _fake_response(200, create_body),
            _fake_response(200, complete_body),
        ]
        mock_requests.put.return_value = put_resp

        results = presigned_upload_batch(
            files=[
                (str(f1), "a" * 64, 32, {}),
                (str(f2), "b" * 64, 32, {}),
            ],
            base_url="https://hub",
            batch_endpoint_path="/api/v1/datasets/x/uploads/batch",
            headers={"Authorization": "Bearer x"},
        )
    assert results[0].dedup is True
    assert results[1].ok is True
    assert results[1].dedup is False
    assert mock_requests.post.call_count == 2  # create + complete
    assert mock_requests.put.call_count == 1


def test_presigned_batch_per_item_error_does_not_abort(tmp_path):
    """A per-item error in create surfaces on that result; others still succeed."""
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"X" * 16)
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"Y" * 16)

    create_body = {
        "items": [
            {"index": 0, "error": "request_too_large", "message": "too big"},
            {
                "index": 1,
                "upload_id": "u-2",
                "part_urls": ["https://s3/part-0"],
                "part_size": 64 * 1024 * 1024,
                "total_parts": 1,
            },
        ]
    }
    complete_body = {"items": [{"index": 0, "upload_id": "u-2", "ok": True}]}

    with patch("gen_worker.presigned_upload.requests") as mock_requests:
        mock_requests.RequestException = Exception
        mock_requests.post.side_effect = [
            _fake_response(200, create_body),
            _fake_response(200, complete_body),
        ]
        mock_requests.put.return_value = _fake_response(200, headers={"ETag": '"e"'})

        results = presigned_upload_batch(
            files=[
                (str(f1), "a" * 64, 16, {}),
                (str(f2), "b" * 64, 16, {}),
            ],
            base_url="https://hub",
            batch_endpoint_path="/api/v1/datasets/x/uploads/batch",
            headers={},
        )
    assert results[0].ok is False
    assert "request_too_large" in (results[0].error or "")
    assert results[1].ok is True


def test_presigned_batch_create_auth_failure_raises(tmp_path):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"x")

    from gen_worker.api.errors import AuthError

    with patch("gen_worker.presigned_upload.requests") as mock_requests:
        mock_requests.RequestException = Exception
        mock_requests.post.return_value = _fake_response(403, {"error": "forbidden"})

        with pytest.raises(AuthError):
            presigned_upload_batch(
                files=[(str(f1), "a" * 64, 1, {})],
                base_url="https://hub",
                batch_endpoint_path="/api/v1/datasets/x/uploads/batch",
                headers={},
            )
