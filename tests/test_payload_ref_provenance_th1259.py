"""th#1259: a bad address in the REQUEST PAYLOAD must fail the request, not
the release.

The filed chain: `score_benchmark` passed the ref-stem of a two-address image
instead of its content digest, `ctx.materialize_blob` raised a bare
`RuntimeError: blob fetch 404`, the executor mapped that to JOB_STATUS_FATAL,
and the hub counted a fatal as model-health evidence — 503 release_broken /
model_load_failure_streak for every caller of the release.

These run the REAL fetch path (a real HTTP server, the real
`_download_blob_by_digest`, the real `_map_exception`), because the defect
lived in exactly the seam between them.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest
from blake3 import blake3

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.api.errors import (
    AuthError,
    BlobForbiddenError,
    BlobNotFoundError,
    DatasetNotFoundError,
    PayloadRefError,
    ValidationError,
)
from gen_worker.executor import _map_exception
from gen_worker.request_context import (
    REF_ORIGIN_PAYLOAD,
    REF_ORIGIN_PLATFORM,
    ConversionContext,
)

BLOB_BYTES = b"real blob bytes"
# pgw#1013: the digest ADDRESSES the bytes, and `_download_blob_by_digest`
# now verifies that. A rig that serves one blob under an arbitrary digest is
# describing a hub that cannot exist, so the "good" address is derived from the
# bytes it names.
GOOD_DIGEST = "blake3:" + blake3(BLOB_BYTES).hexdigest()
MISSING_DIGEST = "sha256:6f3306ab3b849905dd21c6e3073a2f88a4ae34ac4ee5f3af4bda597f559e9d17"
FORBIDDEN_DIGEST = "blake3:" + "bb" * 32


class _Hub(BaseHTTPRequestHandler):
    """The blob-read and dataset surfaces, answering exactly like the hub."""

    def log_message(self, *_a: object) -> None:  # keep pytest output clean
        pass

    def _send(self, code: int, body: bytes = b"") -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if "/blobs/" in path:
            if MISSING_DIGEST.replace(":", "%3A") in path or MISSING_DIGEST in path:
                return self._send(404, b"no such blob")
            if FORBIDDEN_DIGEST.replace(":", "%3A") in path or FORBIDDEN_DIGEST in path:
                return self._send(403, b"forbidden")
            return self._send(200, BLOB_BYTES)
        if "/materialize" in path:
            return self._send(404, b"no such dataset")
        if "/datasets" in path:
            return self._send(200, b'{"items": []}')
        return self._send(404, b"")


@pytest.fixture()
def hub() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Hub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _ctx(hub_url: str) -> ConversionContext:
    ctx = ConversionContext(request_id="r-th1259")
    ctx._file_api_base_url = hub_url
    ctx._worker_capability_token = "test-token"
    return ctx


def test_payload_digest_404_is_a_typed_request_error(hub: str, tmp_path: Path) -> None:
    """The filed repro. A well-formed but non-existent CALLER-SUPPLIED digest
    fails the request with a typed 4xx class — and the executor maps it to
    JOB_STATUS_INVALID, which is what keeps it out of every health signal."""
    ctx = _ctx(hub)

    with pytest.raises(BlobNotFoundError) as ei:
        ctx.materialize_blob(MISSING_DIGEST, tmp_path / "out.webp")

    exc = ei.value
    assert exc.code == "blob_not_found"
    assert exc.ref == MISSING_DIGEST
    assert isinstance(exc, PayloadRefError)
    assert isinstance(exc, ValidationError)  # -> the INVALID branch

    status, message = _map_exception(exc)
    assert status == pb.JOB_STATUS_INVALID, (
        "a payload-origin miss must never reach the hub as FATAL — FATAL is "
        "what fed model_load_failure_streak in th#1259"
    )
    # The machine-readable class survives the safe_message hop.
    assert message.startswith("blob_not_found: ")
    assert MISSING_DIGEST in message


def test_payload_digest_403_is_a_typed_request_error(hub: str, tmp_path: Path) -> None:
    ctx = _ctx(hub)
    with pytest.raises(BlobForbiddenError) as ei:
        ctx.materialize_blob(FORBIDDEN_DIGEST, tmp_path / "out.bin")
    assert ei.value.code == "blob_forbidden"
    assert _map_exception(ei.value)[0] == pb.JOB_STATUS_INVALID


def test_platform_origin_404_stays_fatal(hub: str, tmp_path: Path) -> None:
    """The other half of the provenance rule: an address the PLATFORM
    produced (a hub dataset manifest) that 404s is a platform fault and keeps
    its fatal classification, so genuine breakage still reaches the breaker."""
    ctx = _ctx(hub)

    with pytest.raises(RuntimeError) as ei:
        ctx.materialize_blob(
            MISSING_DIGEST, tmp_path / "out.bin", origin=REF_ORIGIN_PLATFORM,
        )
    assert not isinstance(ei.value, PayloadRefError)
    assert _map_exception(ei.value)[0] == pb.JOB_STATUS_FATAL

    # And the same address forbidden platform-side is still an AuthError.
    with pytest.raises(AuthError):
        ctx.materialize_blob(
            FORBIDDEN_DIGEST, tmp_path / "out2.bin", origin=REF_ORIGIN_PLATFORM,
        )


def test_platform_blob_helper_is_bound_to_platform_origin(hub: str, tmp_path: Path) -> None:
    """The dataset downloader is wired to `_fetch_platform_blob`; prove that
    helper really carries platform provenance rather than the default."""
    ctx = _ctx(hub)
    with pytest.raises(RuntimeError) as ei:
        ctx._fetch_platform_blob(MISSING_DIGEST, tmp_path / "out.bin")
    assert not isinstance(ei.value, PayloadRefError)


def test_payload_dataset_ref_404_is_a_typed_request_error(hub: str) -> None:
    """Same rule on the second caller-supplied resolution path."""
    ctx = _ctx(hub)
    with pytest.raises(DatasetNotFoundError) as ei:
        ctx.resolve_dataset("00000000-0000-0000-0000-000000000000")
    assert ei.value.code == "dataset_not_found"
    assert _map_exception(ei.value)[0] == pb.JOB_STATUS_INVALID

    # owner/name refs take the list-lookup leg and classify identically.
    with pytest.raises(DatasetNotFoundError):
        ctx.resolve_dataset("acme/nope")


def test_good_digest_still_downloads(hub: str, tmp_path: Path) -> None:
    ctx = _ctx(hub)
    out = ctx.materialize_blob(GOOD_DIGEST, tmp_path / "ok.bin")
    assert out.read_bytes() == BLOB_BYTES
    assert REF_ORIGIN_PAYLOAD == "payload"
