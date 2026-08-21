"""The size check belongs INSIDE the download loop."""

from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterator, NamedTuple, Optional

import pytest
import requests
from blake3 import blake3

from gen_worker.bounded_stream import StreamTooLarge, copy_bounded, free_space_bound
from gen_worker.request_context import JobContext
from gen_worker.request_context._datasets import _download_url_streamed

BODY_BYTES = 32 << 20
DECLARED_BYTES = 1 << 20

class _Rig(http.server.ThreadingHTTPServer):

    daemon_threads = True
    allow_reuse_address = True

    routes: Dict[str, "_Route"]
    served: Dict[str, int]
    finished: Dict[str, bool]
    lock: threading.Lock


class _Route(NamedTuple):
    payload: bytes
    declared: Optional[int] = None
    chunked: bool = False
    encoding: str = ""
    status: int = 200


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        srv: _Rig = self.server  # type: ignore[assignment]
        path = urllib.parse.urlparse(self.path).path
        route = srv.routes.get(path)
        if route is None:
            body = b'{"error":{"code":"not_found"}}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        payload, declared, chunked, encoding, status = route
        if not chunked and declared is not None and declared != len(payload):
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header(
                "Content-Length",
                str(len(payload) if declared is None else declared),
            )
        self.end_headers()
        sent = 0
        done = False
        block = 64 << 10
        try:
            for off in range(0, len(payload), block):
                piece = payload[off : off + block]
                if chunked:
                    self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
                else:
                    self.wfile.write(piece)
                sent += len(piece)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
            done = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            with srv.lock:
                srv.served[path] = max(srv.served.get(path, 0), sent)
                srv.finished[path] = srv.finished.get(path, False) or done


@pytest.fixture()
def rig() -> Iterator[_Rig]:
    srv = _Rig(("127.0.0.1", 0), _Handler)
    srv.routes = {}
    srv.served = {}
    srv.finished = {}
    srv.lock = threading.Lock()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _url(rig: _Rig, path: str) -> str:
    return f"http://127.0.0.1:{rig.server_port}{path}"


def _serve(
    rig: _Rig,
    path: str,
    payload: bytes,
    *,
    declared: Optional[int] = None,
    chunked: bool = False,
    encoding: str = "",
    status: int = 200,
) -> str:
    rig.routes[path] = _Route(payload, declared, chunked, encoding, status)
    return _url(rig, path)


def _served(rig: _Rig, path: str) -> int:
    with rig.lock:
        return rig.served.get(path, 0)


def _finished(rig: _Rig, path: str) -> bool:
    with rig.lock:
        return rig.finished.get(path, False)


def _aborted_early(rig: _Rig, path: str) -> None:
    assert not _finished(rig, path), (
        f"{path}: the server wrote the whole {BODY_BYTES}-byte body and the "
        f"client read all of it ({_served(rig, path)} bytes counted), so the "
        f"size check ran AFTER the download loop, not inside it — which is "
        f"pgw#1013's defect exactly: by then the bytes are on disk, the disk "
        f"may be full, and the pod may be gone"
    )


def test_copy_bounded_refuses_a_zero_bound() -> None:
    """The vacuous-guard shape (`if expected_size and total > expected_size`) became unbounded whenever the manifest omitted the size."""
    sink: list[bytes] = []
    for bad in (0, -1):
        with pytest.raises(ValueError) as ei:
            copy_bounded([b"x"], sink.append, limit_bytes=bad, what="t")
        assert "positive byte bound" in str(ei.value)
    assert sink == []


def test_copy_bounded_stops_at_the_byte_that_passes_the_cap() -> None:
    sink: list[bytes] = []
    with pytest.raises(StreamTooLarge):
        copy_bounded([b"a" * 8, b"b" * 8, b"c" * 8], sink.append,
                     limit_bytes=10, what="t")
    assert b"".join(sink) == b"a" * 8


def test_free_space_bound_is_positive_and_below_the_disk(tmp_path: Path) -> None:
    import shutil

    bound = free_space_bound(tmp_path)
    assert 0 < bound < shutil.disk_usage(tmp_path).free


BLOB = b"the addressed bytes" * 64
BLOB_DIGEST = "sha256:" + hashlib.sha256(BLOB).hexdigest()


def _blob_path(digest: str) -> str:
    return f"/api/v1/blobs/{urllib.parse.quote(digest, safe=':')}/content"


def _ctx(rig: _Rig) -> JobContext:
    ctx = JobContext(request_id="r-pgw1013")
    ctx._file_api_base_url = _url(rig, "")
    ctx._worker_capability_token = "test-token"
    return ctx


def test_blob_legitimate_transfer_is_unaffected(rig: _Rig, tmp_path: Path) -> None:
    path = _blob_path(BLOB_DIGEST)
    _serve(rig, path, BLOB)
    out = _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "ok.bin")
    assert out.read_bytes() == BLOB
    assert not (tmp_path / "ok.bin.part").exists()


def test_blob_expanding_body_is_refused_at_the_declared_length(
    rig: _Rig, tmp_path: Path
) -> None:
    """The filed defect: no cap at all, on the path documented as untrusted."""
    path = _blob_path(BLOB_DIGEST)
    bomb = gzip.compress(b"\0" * BODY_BYTES)
    _serve(rig, path, bomb, encoding="gzip")
    assert len(bomb) < BODY_BYTES // 100, "rig is not actually a compression bomb"

    with pytest.raises(StreamTooLarge) as ei:
        _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "bomb.bin")

    assert ei.value.limit_bytes == len(bomb)
    assert ei.value.delivered <= len(bomb) + (1 << 20), (
        "the refusal must land at the cap, not after the body expanded"
    )
    assert not (tmp_path / "bomb.bin").exists()
    assert not (tmp_path / "bomb.bin.part").exists()


def test_blob_without_a_declared_length_is_refused(rig: _Rig, tmp_path: Path) -> None:
    """The uncapped case."""
    path = _blob_path(BLOB_DIGEST)
    _serve(rig, path, BLOB, chunked=True)

    with pytest.raises(RuntimeError) as ei:
        _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "nolen.bin")

    assert "no declared length" in str(ei.value)
    assert not (tmp_path / "nolen.bin").exists()
    assert not (tmp_path / "nolen.bin.part").exists(), (
        "the refusal must come before the file is opened, not after it is filled"
    )


def test_blob_digest_mismatch_is_refused(rig: _Rig, tmp_path: Path) -> None:
    """The other missing half: the digest ADDRESSES the bytes, and nothing checked that the bytes hashed to it."""
    path = _blob_path(BLOB_DIGEST)
    _serve(rig, path, b"different bytes entirely")

    with pytest.raises(RuntimeError) as ei:
        _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "bad.bin")

    assert "digest mismatch" in str(ei.value)
    assert not (tmp_path / "bad.bin").exists()
    assert not (tmp_path / "bad.bin.part").exists()


def test_blob_truncated_stream_leaves_nothing_behind(rig: _Rig, tmp_path: Path) -> None:
    """Under the declared size rather than over it."""
    path = _blob_path(BLOB_DIGEST)
    _serve(rig, path, BLOB[:20], declared=len(BLOB))

    with pytest.raises(Exception):
        _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "short.bin")

    assert not (tmp_path / "short.bin").exists()
    assert not (tmp_path / "short.bin.part").exists()


def test_blob_empty_blob_is_a_legal_object(rig: _Rig, tmp_path: Path) -> None:
    """Zero bytes is a real object with a real digest, and zero is not a bound `copy_bounded` accepts."""
    digest = "sha256:" + hashlib.sha256(b"").hexdigest()
    path = _blob_path(digest)
    _serve(rig, path, b"")
    out = _ctx(rig).materialize_blob(digest, tmp_path / "empty.bin")
    assert out.read_bytes() == b""


def test_blob_verification_covers_the_blake3_namespace(rig: _Rig, tmp_path: Path) -> None:
    digest = "blake3:" + blake3(BLOB).hexdigest()
    path = _blob_path(digest)
    _serve(rig, path, BLOB)
    assert _ctx(rig).materialize_blob(digest, tmp_path / "b3.bin").read_bytes() == BLOB


SHARD = b"shard payload" * 32
SHARD_DIGEST = "sha256:" + hashlib.sha256(SHARD).hexdigest()


def test_shard_legitimate_transfer_is_unaffected(rig: _Rig, tmp_path: Path) -> None:
    url = _serve(rig, "/shard-ok", SHARD)
    dest = tmp_path / "train.jsonl"
    _download_url_streamed(
        url, dest, expected_digest=SHARD_DIGEST, expected_size=len(SHARD))
    assert dest.read_bytes() == SHARD


def test_shard_oversized_stream_is_abandoned_mid_transfer(rig: _Rig, tmp_path: Path) -> None:
    """The manifest entry says 1 MiB; the presigned URL hands back 32 MiB."""
    url = _serve(rig, "/shard-big", b"\0" * BODY_BYTES)
    dest = tmp_path / "big.jsonl"

    with pytest.raises(StreamTooLarge):
        _download_url_streamed(
            url, dest, expected_digest=SHARD_DIGEST, expected_size=DECLARED_BYTES)

    _aborted_early(rig, "/shard-big")
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_shard_without_a_declared_size_is_still_bounded(rig: _Rig, tmp_path: Path) -> None:
    """`size_bytes` is optional on a manifest entry, so refusing is not available — the destination filesystem is the bound instead, and the digest still has the final word."""
    url = _serve(rig, "/shard-nosize", SHARD)
    dest = tmp_path / "nosize.jsonl"
    _download_url_streamed(
        url, dest, expected_digest=SHARD_DIGEST, expected_size=None)
    assert dest.read_bytes() == SHARD


WEIGHTS = b"safetensors-ish" * 100
WEIGHTS_SHA = hashlib.sha256(WEIGHTS).hexdigest()


def _civitai(url: str, dst: Path, *, expected_size: int, sha: str = "") -> int:
    from gen_worker.models.download import _civitai_stream_one

    written, _observed = _civitai_stream_one(
        url, dst, api_key="", expected_size=expected_size,
        expected_sha256=sha or WEIGHTS_SHA, on_bytes=lambda _n: None,
    )
    return written


def test_civitai_legitimate_transfer_is_unaffected(rig: _Rig, tmp_path: Path) -> None:
    url = _serve(rig, "/civ-ok", WEIGHTS)
    assert _civitai(url, tmp_path / "m.safetensors", expected_size=len(WEIGHTS))
    assert (tmp_path / "m.safetensors").read_bytes() == WEIGHTS


def test_civitai_keeps_its_rounded_size_tolerance(rig: _Rig, tmp_path: Path) -> None:
    """`sizeBytes` is derived from `sizeKB`, a rounded float, so the true count is legitimately off by up to a kilobyte (live: wan i2v ...441 vs ...442)."""
    url = _serve(rig, "/civ-round", WEIGHTS)
    assert _civitai(url, tmp_path / "r.safetensors", expected_size=len(WEIGHTS) - 900)


def test_civitai_oversized_stream_is_abandoned_mid_transfer(rig: _Rig, tmp_path: Path) -> None:
    """civitai's API says `sizeBytes` is 1 MiB and its CDN hands back 32 MiB — the third-party origin choosing both halves, which is exactly why the check could not be left until the body had landed."""
    url = _serve(rig, "/civ-big", b"\0" * BODY_BYTES)
    dst = tmp_path / "big.safetensors"

    with pytest.raises(StreamTooLarge):
        _civitai(url, dst, expected_size=DECLARED_BYTES)

    _aborted_early(rig, "/civ-big")
    assert not dst.exists()
    assert not dst.with_suffix(dst.suffix + ".part").exists()


def test_civitai_sizeless_stream_is_still_bounded(rig: _Rig, tmp_path: Path) -> None:
    """civitai really does omit `sizeBytes`, which made the old post-loop check vacuous."""
    url = _serve(rig, "/civ-nosize", WEIGHTS)
    assert _civitai(url, tmp_path / "n.safetensors", expected_size=0) == len(WEIGHTS)


ARTIFACT = b"compiled graph-tarball-bytes" * 64
ARTIFACT_DIGEST = "sha256:" + hashlib.sha256(ARTIFACT).hexdigest()


def test_the_ordering_predicate_FIRES_on_a_post_loop_check(
    rig: _Rig, tmp_path: Path
) -> None:
    """A guard that cannot fire is worse than no guard, and rewriting one is exactly when that gets introduced."""
    url = _serve(rig, "/post-loop", b"\0" * BODY_BYTES)
    dest = tmp_path / "post-loop.bin"

    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        written = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1 << 16):
                written += len(chunk)
                fh.write(chunk)
    assert written > DECLARED_BYTES, "the rig must have handed over the excess"

    assert _finished(rig, "/post-loop")
    with pytest.raises(AssertionError, match="ran AFTER the download loop"):
        _aborted_early(rig, "/post-loop")


def test_the_ordering_predicate_PASSES_on_a_real_in_loop_bound(
    rig: _Rig, tmp_path: Path
) -> None:
    """The other side of severance: the shipping bounded copy, on the same rig, must satisfy the predicate — so a green row means "the bound held", not "the predicate cannot fire"."""
    url = _serve(rig, "/in-loop", b"\0" * BODY_BYTES)
    dest = tmp_path / "in-loop.bin"

    with pytest.raises(StreamTooLarge):
        _download_url_streamed(
            url, dest, expected_digest=SHARD_DIGEST,
            expected_size=DECLARED_BYTES)

    assert not _finished(rig, "/in-loop")
    _aborted_early(rig, "/in-loop")
