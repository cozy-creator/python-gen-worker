"""pgw#781 / th#1303: the chunked UPLOAD client.

Drives real PUTs against a real localhost HTTP server that ENFORCES the
checksum the way R2 does — refusing a body that disagrees with the
x-amz-checksum-sha256 header (400, and the object is not stored) and refusing a
request whose header was changed from the one the grant signed (403). That
enforcement is the whole premise of the design, so the test server implements
it rather than accepting everything.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import threading
from pathlib import Path

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


def test_one_pass_produces_the_whole_hash_and_every_chunk(tmp_path):
    data = make(CS * 3 + 11, seed=5)
    f = tmp_path / "big.safetensors"
    f.write_bytes(data)

    d = hash_file_and_chunks(f, chunk_size=CS, rel_path="t/big.safetensors")
    assert d.digest == "sha256:" + sha(data)
    assert d.size_bytes == len(data)
    assert len(d.chunks) == 4
    # Boundaries are fixed multiples; the chunk hashes are of the real spans.
    for i, c in enumerate(d.chunks):
        assert c.offset == i * CS
        span = data[c.offset:c.offset + c.length]
        assert c.sha256 == sha(span)
        assert c.length == (CS if i < 3 else 11)
    assert sum(c.length for c in d.chunks) == len(data)
    # And the wire shape matches the hub's manifest v2 entry.
    wire = d.to_wire()
    assert wire["digest"].startswith("sha256:")
    assert wire["chunks"][0] == {"digest": d.chunks[0].sha256, "len": CS}


def test_files_at_or_below_the_threshold_stay_whole(tmp_path):
    for size in (1, CS - 1, CS):
        data = make(size, seed=size)
        f = tmp_path / f"s{size}.json"
        f.write_bytes(data)
        d = hash_file_and_chunks(f, chunk_size=CS)
        assert d.chunks == [], f"{size} bytes must be stored whole"
        assert d.digest == "sha256:" + sha(data)
        assert "chunks" not in d.to_wire()
    # One byte over and it chunks.
    data = make(CS + 1, seed=99)
    f = tmp_path / "over.bin"
    f.write_bytes(data)
    assert len(hash_file_and_chunks(f, chunk_size=CS).chunks) == 2


def test_upload_puts_each_chunk_once_and_the_store_accepts_them(tmp_path):
    data = make(CS * 3, seed=7)
    f = tmp_path / "m.safetensors"
    f.write_bytes(data)
    d = hash_file_and_chunks(f, chunk_size=CS, rel_path="m.safetensors")
    src = sources_from_declarations([d], {"m.safetensors": f})

    store = Store()
    try:
        grants = [store.grant("sha256:" + c.sha256, c.length) for c in d.chunks]
        rep = upload_grants(grants, lambda dg: src[dg], parallel=3)
        assert rep.ok, rep.failures
        assert (rep.granted, rep.uploaded) == (3, 3)
        assert rep.bytes_uploaded == len(data)
        # Reassembling what the store holds must reproduce the file exactly.
        got = b"".join(store.objects[f"/staging/{c.sha256}"] for c in d.chunks)
        assert got == data
        assert sha(got) == d.digest.split(":")[1]
    finally:
        store.close()


def test_resume_uploads_only_the_shrunken_need_set(tmp_path):
    """Resume is not session state: the hub simply grants fewer objects the
    second time, because the ones that landed are now resident."""
    data = make(CS * 4, seed=11)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    d = hash_file_and_chunks(f, chunk_size=CS, rel_path="m.bin")
    src = sources_from_declarations([d], {"m.bin": f})

    store = Store()
    try:
        first = [store.grant("sha256:" + c.sha256, c.length) for c in d.chunks[:2]]
        rep = upload_grants(first, lambda dg: src[dg], parallel=2)
        assert rep.ok and rep.uploaded == 2

        # Re-declare: only the two that did not land come back as needs.
        rest = [store.grant("sha256:" + c.sha256, c.length) for c in d.chunks[2:]]
        rep2 = upload_grants(rest, lambda dg: src[dg], parallel=2)
        assert rep2.ok and rep2.granted == 2 and rep2.uploaded == 2

        got = b"".join(store.objects[f"/staging/{c.sha256}"] for c in d.chunks)
        assert got == data
        # Nothing was uploaded twice.
        with store.httpd.lock:
            assert all(v == 1 for v in store.httpd.attempts.values()), store.httpd.attempts
    finally:
        store.close()


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


def test_declaration_self_check_catches_a_local_inconsistency(tmp_path):
    data = make(CS * 2, seed=29)
    f = tmp_path / "m.bin"
    f.write_bytes(data)
    d = hash_file_and_chunks(f, chunk_size=CS)
    from gen_worker.models.chunk_upload import _assert_consistent

    d.chunks[0].__class__  # frozen dataclass; rebuild to mutate
    from gen_worker.models.chunk_upload import ChunkPlan

    d.chunks[0] = ChunkPlan(sha256=d.chunks[0].sha256, offset=0, length=7)
    with pytest.raises(ValueError):
        _assert_consistent(d, CS)
