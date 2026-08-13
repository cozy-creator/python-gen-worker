"""pgw#781 / th#1303: the PUBLISH side — `HubClient.publish_v2` over the hub's
chunked-sha256 routes.

This is the half that makes the conversion/ingest endpoint a v2 PRODUCER.
Before it, `models/chunk_upload.py` had zero production callers and every
publish — including every model mirrored through the conversion endpoint —
still declared blake3 and uploaded S3 multipart.

The stub hub ENFORCES LIKE R2, because that enforcement is the entire point of
the design and a stub that accepts anything would prove nothing:
  * a PUT whose body does not hash to the key's sha256 -> 400 BadDigest, and
    the object DOES NOT EXIST afterwards;
  * a PUT whose `x-amz-checksum-sha256` header differs from the one the hub
    signed -> 403 SignatureDoesNotMatch;
  * a re-declare answers `have` for what already landed.

Real sockets, real files, real threads. Run:
    pytest tests/convert/test_publish_v2_pgw781.py -q
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from gen_worker.hubio.client import CommitFile, HubClient, HubPublishError
from gen_worker.models.chunk_upload import hash_file_and_chunks

CS = 4096  # test chunk size; production is 64 MiB and pinned elsewhere


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def payload(n: int, seed: int = 1) -> bytes:
    out = bytearray(n)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


# ---------------------------------------------------------------------------


class _Hub(http.server.BaseHTTPRequestHandler):
    """Minimal but HONEST implementation of the th#1303 v2 publish contract."""

    def log_message(self, *a):  # noqa: D102
        pass

    # -- helpers ----------------------------------------------------------
    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _send(self, code: int, obj) -> None:
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _plan(self, srv, declared: list) -> dict:
        """Residency comes from the STORE, never from a client claim."""
        need, have, seen = [], [], set()
        for f in declared:
            objs = (
                [(c["digest"], c["len"]) for c in f["chunks"]]
                if f.get("chunks")
                else [(f["digest"].split(":", 1)[-1], f["size_bytes"])]
            )
            for d, n in objs:
                if d in seen:
                    continue
                seen.add(d)
                if d in srv.store:
                    have.append("sha256:" + d)
                    continue
                need.append({
                    "digest": "sha256:" + d,
                    "size_bytes": n,
                    "staging_key": f"staging/sha256/{d}",
                    # The checksum is a SIGNED HEADER, never hoisted into the
                    # query string: measured against R2, the hoisted form is
                    # IGNORED, so wrong bytes would be accepted.
                    "put_url": f"{srv.base}/put/{d}",
                    "headers": {
                        "x-amz-checksum-sha256":
                            base64.b64encode(bytes.fromhex(d)).decode(),
                    },
                })
        return {"have": have, "need": need,
                "distinct_objects": len(seen), "resident_objects": len(have)}

    # -- routes -----------------------------------------------------------
    def do_POST(self):  # noqa: N802
        srv = self.server
        body = self._body()
        if self.path.endswith("/publishes"):
            declared = body.get("files") or []
            srv.declared_bodies.append(body)
            pid = f"pub{len(srv.sessions) + 1}"
            srv.sessions[pid] = declared
            out = {"publish_id": pid, "stage": "uploading",
                   "declared_files": len(declared)}
            out.update(self._plan(srv, declared))
            return self._send(201, out)
        if self.path.endswith("/grants"):
            pid = self.path.split("/publishes/")[1].split("/")[0]
            srv.replans += 1
            return self._send(200, self._plan(srv, srv.sessions[pid]))
        if self.path.endswith("/complete"):
            pid = self.path.split("/publishes/")[1].split("/")[0]
            plan = self._plan(srv, srv.sessions[pid])
            if plan["need"]:
                return self._send(409, {"error": {"code": "upload_incomplete"}})
            manifest = {"version": 2, "entries": srv.sessions[pid]}
            srv.manifests[pid] = manifest
            return self._send(200, {
                "stage": "promoted", "terminal": True,
                "checkpoint": {"checkpoint_id":
                               "sha256:" + sha(json.dumps(manifest, sort_keys=True).encode())},
                "checks_unavailable": ["15", "16", "17", "18", "19"],
            })
        return self._send(404, {"error": {"code": "not_found"}})

    def do_PUT(self):  # noqa: N802
        srv = self.server
        key = self.path.rsplit("/", 1)[-1]
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        claimed = self.headers.get("x-amz-checksum-sha256") or ""
        srv.puts.append(key)
        # 403: the claim differs from what the hub signed into the URL.
        if claimed != base64.b64encode(bytes.fromhex(key)).decode():
            return self._send(403, {"error": {"code": "SignatureDoesNotMatch"}})
        # 400: the bytes do not hash to the key. Nothing is stored.
        if sha(data) != key:
            return self._send(400, {"error": {"code": "BadDigest"}})
        srv.store[key] = data
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):  # noqa: N802
        self.server.aborts.append(self.path)
        return self._send(200, {"stage": "repudiated"})


@pytest.fixture()
def hub():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    srv.store, srv.sessions, srv.manifests = {}, {}, {}
    srv.puts, srv.aborts, srv.declared_bodies = [], [], []
    srv.replans = 0
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def client(hub) -> HubClient:
    return HubClient(base_url=hub.base, token="cap-token")


def write(tmp: Path, name: str, data: bytes) -> CommitFile:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return CommitFile(path=name, local_path=p, size_bytes=len(data))


# ---------------------------------------------------------------------------
# The outcome
# ---------------------------------------------------------------------------


def test_publish_v2_uploads_every_object_and_mints_a_checkpoint(hub, tmp_path, monkeypatch):
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    big, small = payload(CS * 3 + 9), payload(300, seed=4)
    res = client(hub).publish_v2(
        destination_repo="org/model",
        files=[write(tmp_path, "transformer/w.safetensors", big),
               write(tmp_path, "config.json", small)],
        tags=["prod"], mode="replace",
    )
    assert res.checkpoint_id.startswith("sha256:")
    # 4 chunks + 1 whole file, each PUT once.
    assert len(hub.store) == 5
    assert res.uploaded == 5 and res.deduped == 0
    # Every stored object's bytes hash to its own key — that is the invariant
    # the whole design exists to make structural.
    for k, v in hub.store.items():
        assert sha(v) == k


def test_the_declaration_carries_per_chunk_DIGEST_AND_LENGTH(hub, tmp_path, monkeypatch):
    """th#1303's explicit schema requirement — without lengths, a byte range
    cannot be mapped to (chunk, offset) and targeted reads degrade to blind
    fetches. It is called out as painful to retrofit."""
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    data = payload(CS * 2 + 7)
    client(hub).publish_v2(
        destination_repo="org/model",
        files=[write(tmp_path, "w.safetensors", data)], tags=["prod"],
    )
    entry = hub.declared_bodies[0]["files"][0]
    assert entry["digest"] == "sha256:" + sha(data)
    assert entry["size_bytes"] == len(data)
    lens = [c["len"] for c in entry["chunks"]]
    assert lens == [CS, CS, 7]
    assert sum(lens) == len(data)
    # Cumulative offsets are readable from the manifest alone.
    off = 0
    for c in entry["chunks"]:
        assert sha(data[off:off + c["len"]]) == c["digest"]
        off += c["len"]


def test_small_files_are_declared_WHOLE_not_chunked(hub, tmp_path, monkeypatch):
    """Object count only grows for weight shards; configs/tokenizers stay one
    object each."""
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    client(hub).publish_v2(
        destination_repo="org/model",
        files=[write(tmp_path, "config.json", payload(CS))], tags=["prod"],
    )
    assert "chunks" not in hub.declared_bodies[0]["files"][0]
    assert len(hub.store) == 1


def test_dedup_answers_from_the_store_so_resident_chunks_are_never_uploaded(
    hub, tmp_path, monkeypatch,
):
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    data = payload(CS * 3)
    f = write(tmp_path, "w.safetensors", data)
    decl = hash_file_and_chunks(Path(f.local_path), chunk_size=CS, rel_path=f.path)
    # Pre-seed two of the three chunks, as a prior publish would have.
    for c in decl.chunks[:2]:
        hub.store[c.sha256] = data[c.offset:c.offset + c.length]

    res = client(hub).publish_v2(
        destination_repo="org/model", files=[f], tags=["prod"],
    )
    assert res.deduped == 2 and res.uploaded == 1
    assert len(hub.puts) == 1  # bytes moved exactly once, for the missing chunk


def test_republishing_the_same_bytes_uploads_NOTHING(hub, tmp_path, monkeypatch):
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    f = write(tmp_path, "w.safetensors", payload(CS * 2))
    c = client(hub)
    c.publish_v2(destination_repo="org/model", files=[f], tags=["prod"])
    first = len(hub.puts)
    res = c.publish_v2(destination_repo="org/model", files=[f], tags=["prod"])
    assert len(hub.puts) == first  # zero new PUTs
    assert res.uploaded == 0 and res.deduped == 2
    assert res.checkpoint_id.startswith("sha256:")


# ---------------------------------------------------------------------------
# The properties that make it safe
# ---------------------------------------------------------------------------


def test_grant_headers_are_sent_VERBATIM(hub, tmp_path, monkeypatch):
    """The checksum lives INSIDE the signature. Reconstructing or "improving"
    the headers is the one refactor that silently removes enforcement — the
    SDK's own presigner hoists the checksum into the query string, and R2
    IGNORES that form (measured, th#1303 checkpoint 2)."""
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    data = payload(CS * 2)
    client(hub).publish_v2(
        destination_repo="org/model",
        files=[write(tmp_path, "w.safetensors", data)], tags=["prod"],
    )
    # The stub 403s any request whose header does not match what it signed;
    # every PUT landed, so every header went through untouched.
    assert len(hub.store) == 2
    assert all(sha(v) == k for k, v in hub.store.items())


def test_corrupt_local_bytes_are_refused_BEFORE_the_transfer(hub, tmp_path, monkeypatch):
    """A local mismatch is a LOCAL bug and should say so, not arrive as an
    opaque 400 after paying for the upload."""
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)
    f = write(tmp_path, "w.safetensors", payload(CS * 2))
    hub_c = client(hub)

    orig = Path(f.local_path).read_bytes()
    # Declare, then mutate the file underneath the plan (same length).
    def mutate(*_a, **_k):
        raw = bytearray(orig)
        raw[10] ^= 0xFF
        Path(f.local_path).write_bytes(bytes(raw))
    import gen_worker.models.chunk_upload as cu
    real = cu.upload_grants

    def wrapped(*a, **k):
        mutate()
        return real(*a, **k)
    monkeypatch.setattr(cu, "upload_grants", wrapped)

    with pytest.raises(HubPublishError, match="failed to upload"):
        hub_c.publish_v2(destination_repo="org/model", files=[f], tags=["prod"])
    # Nothing corrupt was stored.
    assert all(sha(v) == k for k, v in hub.store.items())
    # pgw#1002 B: and the session is LEFT ALONE. `DELETE /publishes/:id`
    # deletes every staged chunk hub-side, so it is answered only for a refusal
    # the HUB classified terminal — never for a client-side fault, where the
    # objects that did land are exactly what a retry wants. Sessions nobody
    # comes back to are the staging lifecycle's problem (th#1319).
    assert hub.aborts == []


def test_by_reference_adds_are_REFUSED(hub, tmp_path):
    """v2's guarantee is that a digest is PROVEN from bytes in hand. A
    by-reference add has no bytes, so accepting one would reintroduce exactly
    the claimed-digest trust this protocol removes (th#1305)."""
    with pytest.raises(HubPublishError, match="by-reference"):
        client(hub).publish_v2(
            destination_repo="org/model",
            files=[CommitFile(path="w.safetensors", size_bytes=10)],
            tags=["prod"],
        )


def test_a_hub_without_v2_routes_FAILS_rather_than_downgrading(tmp_path):
    """No silent fallback to blake3. A downgrade would make "did this publish
    v2?" unanswerable from outside, and th#715 is the standing reminder that a
    404 from a proxy is not a 404 from the hub."""
    class _NoV2(_Hub):
        def do_POST(self):  # noqa: N802
            self._body()
            self._send(404, {"error": {"code": "not_found"}})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoV2)
    srv.store, srv.sessions, srv.manifests = {}, {}, {}
    srv.puts, srv.aborts, srv.declared_bodies = [], [], []
    srv.replans = 0
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(HubPublishError, match="publish declare failed"):
            HubClient(base_url=srv.base, token="t").publish_v2(
                destination_repo="org/model",
                files=[write(tmp_path, "config.json", b"x" * 32)], tags=["prod"],
            )
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_a_grant_for_something_never_declared_is_refused(hub, tmp_path, monkeypatch):
    """The client PUTs only what it can prove it declared. A hub (or a
    man-in-the-middle) that grants an unrelated digest must not get bytes."""
    monkeypatch.setattr("gen_worker.models.chunk_upload.CAS_CHUNK_SIZE_BYTES", CS)

    class _Rogue(_Hub):
        def _plan(self, srv, declared):
            out = _Hub._plan(self, srv, declared)
            out["need"].append({
                "digest": "sha256:" + "ee" * 32, "size_bytes": 16,
                "put_url": f"{srv.base}/put/{'ee' * 32}", "headers": {},
            })
            return out

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Rogue)
    srv.store, srv.sessions, srv.manifests = {}, {}, {}
    srv.puts, srv.aborts, srv.declared_bodies = [], [], []
    srv.replans = 0
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(HubPublishError, match="never declared"):
            HubClient(base_url=srv.base, token="t").publish_v2(
                destination_repo="org/model",
                files=[write(tmp_path, "config.json", payload(100))], tags=["prod"],
            )
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_the_worker_addable_provenance_subset_reaches_the_declare(hub, tmp_path):
    """th#1331: the v2 declare now stamps lineage through v1's OWN resolver, so
    a mirror's `upstream_revision` must actually arrive on the wire.

    This test replaced one asserting the opposite. `publish_v2` used to REFUSE a
    non-empty provenance, because the route wrote `Provenance: []byte("{}")`
    unconditionally and accepting-then-dropping would have blanked, silently,
    the field naming which upstream commit we copied. The route learned the
    field; the refusal went with it."""
    res = client(hub).publish_v2(
        destination_repo="org/model",
        files=[write(tmp_path, "config.json", payload(64))],
        tags=["prod"],
        provenance={"upstream_revision": "abc123", "quantization_method": "",
                    "step_number": 7},
    )
    assert res.checkpoint_id.startswith("sha256:")
    sent = hub.declared_bodies[0]["provenance"]
    assert sent == {"upstream_revision": "abc123", "step_number": 7}, sent


def test_forgeable_lineage_fields_are_the_HUB_s_refusal_not_ours(hub, tmp_path):
    """The client does not police this and must not pretend to. `parents` /
    `derivation_op` / `upstream_ref` / `job_id` are orchestrator-derived and
    signed into the capability token; the hub 400s them NAMING the field. A
    client-side filter here would silently drop a forgery attempt instead of
    surfacing it, which is strictly worse — the caller would believe it had
    stamped what it sent."""
    client(hub).publish_v2(
        destination_repo="org/model",
        files=[write(tmp_path, "config.json", payload(64))],
        tags=["prod"], provenance={"upstream_ref": "hf:someone/else"},
    )
    assert hub.declared_bodies[0]["provenance"] == {"upstream_ref": "hf:someone/else"}


