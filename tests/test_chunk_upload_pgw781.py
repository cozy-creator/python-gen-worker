"""pgw#781 / th#1303: the chunked UPLOAD client.

Drives real PUTs against a real localhost HTTP server that ENFORCES the
checksum the way R2 does — refusing a body that disagrees with the
x-amz-checksum-sha256 header (400, and the object is not stored) and refusing a
request whose header was changed from the one the grant signed (403). That
enforcement is the whole premise of the design, so the test server implements
it rather than accepting everything.

The happy paths (declaration shape, chunk boundaries, dedup/resume) are pinned
end-to-end through HubClient.publish_v2 in
tests/convert/test_publish_v2_pgw781.py; this file keeps only the client-side
refusal properties that an end-to-end pass cannot distinguish.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import threading

import pytest

from gen_worker.models.chunk_upload import (
    UploadGrant,
    hash_file_and_chunks,
    sources_from_declarations,
    upload_grants,
)

CS = 4096


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def b64(hexdigest: str) -> str:
    return base64.b64encode(bytes.fromhex(hexdigest)).decode()


def make(total: int, seed: int = 0) -> bytes:
    out = bytearray(total)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(total):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


class _Store(http.server.BaseHTTPRequestHandler):
    """Minimal R2-shaped enforcing store."""

    def log_message(self, *a):
        pass

    def do_PUT(self):
        srv = self.server
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        claimed = self.headers.get("x-amz-checksum-sha256")
        signed = srv.signed.get(self.path)
        with srv.lock:
            srv.attempts[self.path] = srv.attempts.get(self.path, 0) + 1
        # The checksum is INSIDE the signature: changing the claim breaks it.
        if signed is not None and claimed != signed:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"<Code>SignatureDoesNotMatch</Code>")
            return
        # And the store enforces the claim against the bytes.
        if claimed is None or b64(sha(body)) != claimed:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<Code>BadDigest</Code>")
            return
        with srv.lock:
            srv.objects[self.path] = body
        self.send_response(200)
        self.end_headers()


class Store:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Store)
        self.httpd.objects = {}
        self.httpd.signed = {}
        self.httpd.attempts = {}
        self.httpd.lock = threading.Lock()
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self):
        h, p = self.httpd.server_address[:2]
        return f"http://{h}:{p}"

    def grant(self, digest: str, size: int) -> UploadGrant:
        hexd = digest.split(":", 1)[-1]
        path = f"/staging/{hexd}"
        self.httpd.signed[path] = b64(hexd)
        return UploadGrant(
            digest=digest, size_bytes=size,
            put_url=self.base + path,
            headers={"x-amz-checksum-sha256": b64(hexd)},
            staging_key=path.lstrip("/"),
        )

    @property
    def objects(self):
        with self.httpd.lock:
            return dict(self.httpd.objects)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_a_substituted_checksum_claim_is_refused_by_the_store(tmp_path):
    """The checksum is inside the signature. Changing the claim must 403 — and
    the client must NOT retry a terminal 4xx into a storm."""
    data = make(CS, seed=13)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store = Store()
    try:
        g = store.grant("sha256:" + sha(data), len(data))
        tampered = UploadGrant(
            digest=g.digest, size_bytes=g.size_bytes, put_url=g.put_url,
            headers={"x-amz-checksum-sha256": b64(sha(b"other bytes"))},
        )
        rep = upload_grants([tampered], lambda dg: (f, 0, len(data)), parallel=1)
        assert not rep.ok
        assert "403" in rep.failures[0], rep.failures
        assert store.objects == {}
        with store.httpd.lock:
            assert sum(store.httpd.attempts.values()) == 1, "a terminal 4xx must not be retried"
    finally:
        store.close()


def test_bytes_that_disagree_with_the_digest_never_leave_the_client(tmp_path):
    """A local mismatch is a LOCAL bug and must be named as one, not shipped
    and reported back as an opaque 400."""
    data = make(CS, seed=17)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    store = Store()
    try:
        # A grant for a digest the local bytes do not have.
        wrong = "sha256:" + sha(b"completely different")
        g = store.grant(wrong, len(data))
        rep = upload_grants([g], lambda dg: (f, 0, len(data)), parallel=1)
        assert not rep.ok
        assert "refusing to upload" in rep.failures[0], rep.failures
        with store.httpd.lock:
            assert store.httpd.attempts == {}, "nothing should have been sent"
    finally:
        store.close()


def test_shared_chunks_upload_once(tmp_path):
    """Chunk-granular dedup is local too: identical spans resolve to one
    source entry, so the same bytes are never PUT twice."""
    block = make(CS, seed=23)
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(block * 2 + b"AA")
    b.write_bytes(block * 2 + b"BB")
    da = hash_file_and_chunks(a, chunk_size=CS, rel_path="a.bin")
    db = hash_file_and_chunks(b, chunk_size=CS, rel_path="b.bin")
    src = sources_from_declarations([da, db], {"a.bin": a, "b.bin": b})
    # The two full leading chunks are identical across both files.
    assert da.chunks[0].sha256 == db.chunks[0].sha256 == sha(block)
    digests = {"sha256:" + c.sha256 for c in list(da.chunks) + list(db.chunks)}
    assert len(digests) == 3, "two shared chunks + two distinct tails collapse to 3"
    assert set(src) == digests


