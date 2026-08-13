"""pgw#1013 — the size check moves INSIDE the download loop, at four sites.

THE DEFECT. Seven downloaders in this repo stream a remote body to disk. Three
compared the running byte count to the declared size inside the loop and
aborted at the first excess byte. Four wrote the whole body first and compared
sizes after the loop had ended, which is not a bound: by the time the
comparison runs the bytes are on disk, the disk may be full, and the pod may be
gone. The worst of the four, `RequestContext._download_blob_by_digest`, had
NEITHER a cap NOR any verification, on the path its own public wrapper
(`materialize_blob`) documents as "the untrusted case".

WHAT THESE TESTS ASSERT, and why it is not just "an error is raised". An error
after the fact is exactly what the old code produced for three of the four
sites. So every refusal below is checked for WHEN it happened — and since
pgw#1204 that is observed as an ORDERING rather than as a byte count: the rig
records whether it wrote a body to its LAST byte with the client still reading.
An in-loop bound kills the connection mid-write, so the server never finishes;
a post-loop check reads all 32 MiB, so it does. That difference is the entire
issue, and it is a fact about order — unlike "how many bytes got out before the
abort", which is a fact about the runner's scheduler and flaked five cuts.

No mocks: every case runs a real `requests` client against a real
`ThreadingHTTPServer` over a real socket, through the shipping download
functions.
"""

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
from gen_worker.request_context import ConversionContext
from gen_worker.request_context._datasets import _download_url_streamed

# The rig always OFFERS this much body. A site that only checks after the loop
# writes all of it; a site that checks in-loop abandons the connection early.
BODY_BYTES = 32 << 20
DECLARED_BYTES = 1 << 20

# WHY THE OVERSIZE CASES DECLARE AN HONEST Content-Length. urllib3 enforces
# Content-Length itself and stops reading at it, so a server that UNDER-declares
# cannot be used to demonstrate anything — the excess never reaches our code.
# The runaway that actually exists is the one these tests build: the transport
# is honest about a large body, and it is the MANIFEST (the dataset entry, the
# civitai `sizeBytes`, the cell entry's `size_bytes`) that says the file is
# small. That declaration is what the four sites compared against after the
# loop and now compare against inside it.
#
# `materialize_blob` is the exception and gets its own shape: its declaration
# IS the transport length, so the only way past it is a `Content-Encoding` that
# expands — see the gzip-bomb case.
# ---------------------------------------------------------------------------
# Rig
# ---------------------------------------------------------------------------

class _Rig(http.server.ThreadingHTTPServer):
    """Serves scripted bodies and records, per path, whether the response was
    written to COMPLETION before the client hung up.

    ``finished`` is the ordering observation this file turns on.
    ``served`` is kept as a DIAGNOSTIC — it makes a failure message concrete —
    and no assertion reads it, because how many bytes a server got onto the
    wire before an abort is a fact about the runner's scheduler, not about the
    code under test.
    """

    daemon_threads = True
    allow_reuse_address = True

    routes: Dict[str, "_Route"]
    served: Dict[str, int]
    finished: Dict[str, bool]
    lock: threading.Lock


class _Route(NamedTuple):
    payload: bytes
    declared: Optional[int] = None   # a Content-Length that lies about payload
    chunked: bool = False            # no Content-Length at all
    encoding: str = ""               # a Content-Encoding the client decodes
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
            # A response whose body disagrees with its own Content-Length can
            # only end by closing the connection — keep-alive would leave the
            # client waiting for bytes that are never coming, which is the rig
            # hanging rather than the code under test being wrong.
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if chunked:
            # No Content-Length at all — the shape a hub arm never produces.
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
            # Reached only when every byte went out and the client was still
            # reading. THIS is the ordering fact: the client's check cannot
            # have run inside the loop, because the loop ran to the end.
            done = True
        except (BrokenPipeError, ConnectionResetError):
            # The client refused mid-transfer. That is the assertion.
            self.close_connection = True
        finally:
            with srv.lock:
                srv.served[path] = max(srv.served.get(path, 0), sent)
                # Sticky: a client that completed the body even ONCE read all
                # of it, whatever a retry did afterwards. Sticky in the
                # direction that makes `_aborted_early` fire, never hide.
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
    """Whether the server wrote this body to its LAST byte with the client
    still reading."""
    with rig.lock:
        return rig.finished.get(path, False)


def _aborted_early(rig: _Rig, path: str) -> None:
    """The refusal happened DURING the transfer, not after it.

    OBSERVED AS AN ORDERING, NOT AS A BYTE COUNT. This predicate
    used to assert `served < 8 MiB` against a 32 MiB body — a threshold, on a
    quantity nothing in the code under test controls. How far a server gets
    before an abort lands is decided by the runner's scheduler and by kernel
    socket buffers on both ends, so the assertion converted CI timing into a
    boolean and flaked: it billed five cuts, twice at ~8.7 MiB against the 8
    MiB line, and each time the same tree passed on rerun.

    The real property was always ordinal — *did the client's check run inside
    the loop or after it?* — and the server can answer it exactly: if the
    client aborted mid-transfer, the write loop died on a broken pipe and
    never reached its last byte. If the check ran after the loop, the client
    read all 32 MiB and the server completed. That is a fact about ORDER, it
    is the same on a loaded runner as on an idle one, and there is no
    threshold to tune.

    The byte count survives in the MESSAGE, where a number belongs.
    """
    assert not _finished(rig, path), (
        f"{path}: the server wrote the whole {BODY_BYTES}-byte body and the "
        f"client read all of it ({_served(rig, path)} bytes counted), so the "
        f"size check ran AFTER the download loop, not inside it — which is "
        f"pgw#1013's defect exactly: by then the bytes are on disk, the disk "
        f"may be full, and the pod may be gone"
    )


# ---------------------------------------------------------------------------
# The shared helper: an unbounded copy has no spelling
# ---------------------------------------------------------------------------

def test_copy_bounded_refuses_a_zero_bound() -> None:
    """The vacuous-guard shape (`if expected_size and total > expected_size`)
    became unbounded whenever the manifest omitted the size. Here a caller
    without a declaration cannot reach the loop at all."""
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
    # The chunk that would have crossed the cap never reached the sink.
    assert b"".join(sink) == b"a" * 8


def test_free_space_bound_is_positive_and_below_the_disk(tmp_path: Path) -> None:
    import shutil

    bound = free_space_bound(tmp_path)
    assert 0 < bound < shutil.disk_usage(tmp_path).free


# ---------------------------------------------------------------------------
# materialize_blob — the worst of the four
# ---------------------------------------------------------------------------

BLOB = b"the addressed bytes" * 64
BLOB_DIGEST = "sha256:" + hashlib.sha256(BLOB).hexdigest()


def _blob_path(digest: str) -> str:
    return f"/api/v1/blobs/{urllib.parse.quote(digest, safe=':')}/content"


def _ctx(rig: _Rig) -> ConversionContext:
    ctx = ConversionContext(request_id="r-pgw1013")
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
    """The filed defect: no cap at all, on the path documented as untrusted.

    `Content-Encoding` is how a body outgrows the length that declared it — 32
    MiB of zeros ships as a few KiB and the client decompresses it. The old
    reader wrote every one of those 32 MiB to disk and compared nothing, ever.
    The cap now stops it at the declared length, so the bomb costs the pod one
    read block instead of its whole disk.
    """
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
    """The uncapped case. Both arms of the hub's by-digest route declare a
    length — a 302 to a presigned object, or `DataFromReader` with the object's
    size — so a response that declares none is not the responder this contract
    describes, and there is nothing left to bound the transfer with."""
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
    """The other missing half: the digest ADDRESSES the bytes, and nothing
    checked that the bytes hashed to it. A content-addressed read that accepts
    whatever arrives is not content-addressed."""
    path = _blob_path(BLOB_DIGEST)
    _serve(rig, path, b"different bytes entirely")

    with pytest.raises(RuntimeError) as ei:
        _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "bad.bin")

    assert "digest mismatch" in str(ei.value)
    assert not (tmp_path / "bad.bin").exists()
    assert not (tmp_path / "bad.bin.part").exists()


def test_blob_truncated_stream_leaves_nothing_behind(rig: _Rig, tmp_path: Path) -> None:
    """Under the declared size rather than over it.

    Which refusal fires is not the point and is not asserted: urllib3 enforces
    Content-Length itself and usually raises first, with the post-loop
    `total != declared` check as the backstop for a stream that ends cleanly
    short. What IS the invariant, and what the old code did not have, is that
    a refused fetch leaves no file — not the destination, and not a `.part`
    that a later resume check would read as complete.
    """
    path = _blob_path(BLOB_DIGEST)
    _serve(rig, path, BLOB[:20], declared=len(BLOB))

    with pytest.raises(Exception):
        _ctx(rig).materialize_blob(BLOB_DIGEST, tmp_path / "short.bin")

    assert not (tmp_path / "short.bin").exists()
    assert not (tmp_path / "short.bin.part").exists()


def test_blob_empty_blob_is_a_legal_object(rig: _Rig, tmp_path: Path) -> None:
    """Zero bytes is a real object with a real digest, and zero is not a bound
    `copy_bounded` accepts. The declared-length refusal must not swallow it —
    a cap that refuses the empty file is a cap that broke a legal read."""
    digest = "sha256:" + hashlib.sha256(b"").hexdigest()
    path = _blob_path(digest)
    _serve(rig, path, b"")
    out = _ctx(rig).materialize_blob(digest, tmp_path / "empty.bin")
    assert out.read_bytes() == b""


def test_blob_verification_covers_the_blake3_namespace(rig: _Rig, tmp_path: Path) -> None:
    """Both CAS algorithms the address parser accepts must be hashable here —
    a digest this repo can address but not verify would be a download taken on
    trust, which is the shape th#1303 S1 and pgw#882 both refused."""
    digest = "blake3:" + blake3(BLOB).hexdigest()
    path = _blob_path(digest)
    _serve(rig, path, BLOB)
    assert _ctx(rig).materialize_blob(digest, tmp_path / "b3.bin").read_bytes() == BLOB


# ---------------------------------------------------------------------------
# dataset shards
# ---------------------------------------------------------------------------

SHARD = b"shard payload" * 32
SHARD_DIGEST = "sha256:" + hashlib.sha256(SHARD).hexdigest()


def test_shard_legitimate_transfer_is_unaffected(rig: _Rig, tmp_path: Path) -> None:
    url = _serve(rig, "/shard-ok", SHARD)
    dest = tmp_path / "train.jsonl"
    _download_url_streamed(
        url, dest, expected_digest=SHARD_DIGEST, expected_size=len(SHARD))
    assert dest.read_bytes() == SHARD


def test_shard_oversized_stream_is_abandoned_mid_transfer(rig: _Rig, tmp_path: Path) -> None:
    """The manifest entry says 1 MiB; the presigned URL hands back 32 MiB. The
    old code wrote all 32 and then said so."""
    url = _serve(rig, "/shard-big", b"\0" * BODY_BYTES)
    dest = tmp_path / "big.jsonl"

    with pytest.raises(StreamTooLarge):
        _download_url_streamed(
            url, dest, expected_digest=SHARD_DIGEST, expected_size=DECLARED_BYTES)

    _aborted_early(rig, "/shard-big")
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_shard_without_a_declared_size_is_still_bounded(rig: _Rig, tmp_path: Path) -> None:
    """`size_bytes` is optional on a manifest entry, so refusing is not
    available — the destination filesystem is the bound instead, and the
    digest still has the final word."""
    url = _serve(rig, "/shard-nosize", SHARD)
    dest = tmp_path / "nosize.jsonl"
    _download_url_streamed(
        url, dest, expected_digest=SHARD_DIGEST, expected_size=None)
    assert dest.read_bytes() == SHARD


# ---------------------------------------------------------------------------
# civitai — the third-party origin
# ---------------------------------------------------------------------------

WEIGHTS = b"safetensors-ish" * 100
WEIGHTS_SHA = hashlib.sha256(WEIGHTS).hexdigest()


def _civitai(url: str, dst: Path, *, expected_size: int, sha: str = "") -> int:
    from gen_worker.models.download import _civitai_stream_one

    # the observed digest now rides back beside the byte count so the
    # manifest can distinguish a verified download from an unverified one.
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
    """`sizeBytes` is derived from `sizeKB`, a rounded float, so the true count
    is legitimately off by up to a kilobyte (live: wan i2v ...441 vs ...442).
    The in-loop cap must not be tighter than the acceptance rule, or it refuses
    files the next line would accept."""
    url = _serve(rig, "/civ-round", WEIGHTS)
    assert _civitai(url, tmp_path / "r.safetensors", expected_size=len(WEIGHTS) - 900)


def test_civitai_oversized_stream_is_abandoned_mid_transfer(rig: _Rig, tmp_path: Path) -> None:
    """civitai's API says `sizeBytes` is 1 MiB and its CDN hands back 32 MiB —
    the third-party origin choosing both halves, which is exactly why the check
    could not be left until the body had landed."""
    url = _serve(rig, "/civ-big", b"\0" * BODY_BYTES)
    dst = tmp_path / "big.safetensors"

    with pytest.raises(StreamTooLarge):
        _civitai(url, dst, expected_size=DECLARED_BYTES)

    _aborted_early(rig, "/civ-big")
    assert not dst.exists()
    assert not dst.with_suffix(dst.suffix + ".part").exists()


def test_civitai_sizeless_stream_is_still_bounded(rig: _Rig, tmp_path: Path) -> None:
    """civitai really does omit `sizeBytes`, which made the old post-loop check
    vacuous. The transfer still completes — and is still bounded, by the disk
    it is landing on."""
    url = _serve(rig, "/civ-nosize", WEIGHTS)
    assert _civitai(url, tmp_path / "n.safetensors", expected_size=0) == len(WEIGHTS)


# ---------------------------------------------------------------------------
# AOT cell artifacts
# ---------------------------------------------------------------------------

ARTIFACT = b"cell-tarball-bytes" * 64
ARTIFACT_DIGEST = "sha256:" + hashlib.sha256(ARTIFACT).hexdigest()


def _resolve_route(rig: _Rig, repo: str, entry: Dict[str, Any]) -> None:
    body = json.dumps({"files": [entry]}).encode()
    rig.routes[f"/api/v1/repos/{repo}/resolve"] = _Route(body)


def _fetch_cell(rig: _Rig, cache: Path, entry: Dict[str, Any],
                digest: str) -> Optional[Path]:
    """pgw#904: cell bytes arrive as the EXACT named artifact from the
    grant's transport (`aot_delivery`), never a discovery fetch — the same
    bounded-read invariants, at the delivery seam that replaced it."""
    from types import SimpleNamespace

    from gen_worker import aot_delivery

    presigned = SimpleNamespace(files=[SimpleNamespace(
        path=str(entry.get("path") or ""),
        size_bytes=int(entry.get("size_bytes") or 0),
        digest=str(entry.get("digest") or ""),
        url=str(entry.get("url") or ""),
        chunk_size_bytes=0,
        chunks=(),
    )])
    try:
        return aot_delivery.materialize_named_artifact(
            "root/family-fam#ck", digest, presigned,
            cache_dir=cache, what="stream-bounds rig")
    except aot_delivery.NamedArtifactUnavailable:
        return None


def _repo_for(family: str = "fam") -> str:
    from gen_worker import compile_cache as cc

    return cc.system_repo(family)


def test_cell_artifact_legitimate_transfer_is_unaffected(rig: _Rig, tmp_path: Path) -> None:
    url = _serve(rig, "/cell.tar.gz", ARTIFACT)
    entry = {"path": "cell.tar.gz", "url": url,
             "digest": ARTIFACT_DIGEST, "size_bytes": len(ARTIFACT)}
    out = _fetch_cell(rig, tmp_path, entry, ARTIFACT_DIGEST)
    assert out is not None and out.read_bytes() == ARTIFACT


def test_cell_artifact_without_size_bytes_is_a_typed_miss(rig: _Rig, tmp_path: Path) -> None:
    """The chunked sibling branch passes this exact field to
    `download_chunked_file` as `total_size`; the whole-file branch had it and
    ignored it. An entry that cannot say how big it is now costs the pilot lane
    a miss rather than an unbounded fetch — a cell miss self-mints, so failing
    closed here is free."""
    url = _serve(rig, "/cell-nosize.tar.gz", ARTIFACT)
    entry = {"path": "cell-nosize.tar.gz", "url": url, "digest": ARTIFACT_DIGEST}
    assert _fetch_cell(rig, tmp_path, entry, ARTIFACT_DIGEST) is None
    assert _served(rig, "/cell-nosize.tar.gz") == 0, "no bytes fetched at all"


def test_cell_artifact_oversized_stream_is_abandoned_mid_transfer(
    rig: _Rig, tmp_path: Path
) -> None:
    url = _serve(rig, "/cell-big.tar.gz", b"\0" * BODY_BYTES)
    entry = {"path": "cell-big.tar.gz", "url": url,
             "digest": ARTIFACT_DIGEST, "size_bytes": DECLARED_BYTES}

    assert _fetch_cell(rig, tmp_path, entry, ARTIFACT_DIGEST) is None
    _aborted_early(rig, "/cell-big.tar.gz")
    hexname = ARTIFACT_DIGEST.split(":", 1)[-1]
    assert not (tmp_path / "aot-cells" / f"{hexname}.tar.gz").exists()
    assert not (tmp_path / "aot-cells" / f"{hexname}.part").exists()


# ---------------------------------------------------------------------------
# the SEVERANCE check — `_aborted_early` must still be able to fail
# ---------------------------------------------------------------------------


def test_the_ordering_predicate_FIRES_on_a_post_loop_check(
    rig: _Rig, tmp_path: Path
) -> None:
    """A guard that cannot fire is worse than no guard, and rewriting one is
    exactly when that gets introduced.

    So: reproduce pgw#1013's ACTUAL defect — read the whole body, then compare
    sizes — against the same rig, and prove `_aborted_early` calls it. This is
    the row that would have caught a rewrite that made the predicate vacuous,
    and it is why the threshold could be deleted rather than merely widened
    (widening a flaky threshold buys silence, not a signal).
    """
    url = _serve(rig, "/post-loop", b"\0" * BODY_BYTES)
    dest = tmp_path / "post-loop.bin"

    # The four sites pgw#1013 fixed, in miniature: the whole body lands, and
    # only then is its size compared against what the manifest declared.
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        written = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1 << 16):
                written += len(chunk)
                fh.write(chunk)
    assert written > DECLARED_BYTES, "the rig must have handed over the excess"

    # The server ran to its last byte, so the check cannot have been in-loop.
    assert _finished(rig, "/post-loop")
    with pytest.raises(AssertionError, match="ran AFTER the download loop"):
        _aborted_early(rig, "/post-loop")


def test_the_ordering_predicate_PASSES_on_a_real_in_loop_bound(
    rig: _Rig, tmp_path: Path
) -> None:
    """The other side of severance: the shipping bounded copy, on the same
    rig, must satisfy the predicate — so a green row means "the bound held",
    not "the predicate cannot fire"."""
    url = _serve(rig, "/in-loop", b"\0" * BODY_BYTES)
    dest = tmp_path / "in-loop.bin"

    with pytest.raises(StreamTooLarge):
        _download_url_streamed(
            url, dest, expected_digest=SHARD_DIGEST,
            expected_size=DECLARED_BYTES)

    assert not _finished(rig, "/in-loop")
    _aborted_early(rig, "/in-loop")
