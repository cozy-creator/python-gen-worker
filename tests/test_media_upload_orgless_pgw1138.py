from __future__ import annotations

import base64
import json
from typing import Any, Dict

import pytest

from gen_worker import RequestContext
from gen_worker.api.errors import ArtifactTransferError

from harness.upload_sink import (
    MEDIA_UPLOADS_PATH,
    DedupUploadSink,
    is_media_upload_route,
    reset_upload_sink,
    serve_upload_sink,
)


def _unsigned_jwt(claims: Dict[str, Any]) -> str:
    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


def test_save_bytes_uploads_against_the_org_less_route_family() -> None:
    """The real `ctx.save_bytes` codepath, against a server that serves only tensorhub's org-less routes."""
    httpd, base_url = serve_upload_sink()
    try:
        token = _unsigned_jwt(
            {"tenant": "019f4c33-f3a5-705b-9848-0b3b0863c416", "request_id": "req-1138"}
        )
        ctx = RequestContext(
            request_id="req-1138",
            owner="tensorhub",
            file_api_base_url=base_url,
            worker_capability_token=token,
        )
        asset = ctx.save_bytes("samples/pair-000.bin", b"payload")

        assert asset.ref == "samples/pair-000.bin"
        assert not DedupUploadSink.rejected, (
            "the client addressed a route tensorhub does not serve: "
            f"{DedupUploadSink.rejected}"
        )
        assert DedupUploadSink.requests_seen, "the real upload sink must have been hit"
        path, body = DedupUploadSink.requests_seen[-1]
        assert path == MEDIA_UPLOADS_PATH, (
            "the org comes from the credential, never from the path"
        )
        assert body["ref"] == "samples/pair-000.bin"
    finally:
        httpd.shutdown()
        reset_upload_sink()


def test_the_upload_carries_no_owner_header() -> None:
    """`X-Cozy-Owner` was write-only: no tensorhub version reads it."""
    httpd, base_url = serve_upload_sink()
    try:
        ctx = RequestContext(
            request_id="req-1138-hdr",
            owner="org",
            file_api_base_url=base_url,
            worker_capability_token=_unsigned_jwt({"tenant": "org-uuid"}),
        )
        ctx.save_bytes("samples/hdr.bin", b"payload")
        assert DedupUploadSink.headers_seen
        sent = {k.lower() for k in DedupUploadSink.headers_seen[-1]}
        assert "authorization" in sent, "the credential is what carries the org"
        assert "x-cozy-owner" not in sent
    finally:
        httpd.shutdown()
        reset_upload_sink()


def test_an_upload_with_no_derivable_owner_is_no_longer_a_failure_mode() -> None:
    httpd, base_url = serve_upload_sink()
    try:
        ctx = RequestContext(
            request_id="req-1138-noowner",
            owner="",
            file_api_base_url=base_url,
            worker_capability_token=_unsigned_jwt({"request_id": "req-1138-noowner"}),
        )
        asset = ctx.save_bytes("samples/no-owner.bin", b"payload")
        assert asset.ref == "samples/no-owner.bin"
        assert DedupUploadSink.requests_seen
        assert DedupUploadSink.requests_seen[-1][0] == MEDIA_UPLOADS_PATH
    finally:
        httpd.shutdown()
        reset_upload_sink()


def test_the_sink_would_have_caught_the_old_shape() -> None:
    """The harness is only evidence if it can go red."""
    assert is_media_upload_route(MEDIA_UPLOADS_PATH)
    assert is_media_upload_route(MEDIA_UPLOADS_PATH + "/u1/complete")
    assert not is_media_upload_route("/api/v1/media/some-org-uuid/uploads")
    assert not is_media_upload_route("/api/v1/media/some-org-uuid/uploads/u1/complete")


def test_the_sink_404s_a_route_tensorhub_does_not_serve() -> None:
    import requests

    httpd, base_url = serve_upload_sink()
    try:
        resp = requests.post(
            f"{base_url}/api/v1/media/some-org/uploads", json={"ref": "x"}, timeout=5
        )
        assert resp.status_code == 404
        assert DedupUploadSink.rejected == ["/api/v1/media/some-org/uploads"]
        assert not DedupUploadSink.requests_seen
    finally:
        httpd.shutdown()
        reset_upload_sink()


def test_a_create_404_raises_a_typed_transfer_error() -> None:
    """Guards the RED signal itself: had the sink answered every path, the tests above would pass on master."""
    httpd, base_url = serve_upload_sink()
    try:
        ctx = RequestContext(
            request_id="req-1138-404",
            owner="org",
            file_api_base_url=base_url + "/nope",
            worker_capability_token=_unsigned_jwt({"tenant": "org-uuid"}),
        )
        with pytest.raises(ArtifactTransferError):
            ctx.save_bytes("samples/dead.bin", b"payload")
    finally:
        httpd.shutdown()
        reset_upload_sink()
