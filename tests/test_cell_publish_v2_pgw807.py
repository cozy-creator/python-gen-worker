"""pgw#807 item 3: the cell self-mint publisher ships over CHUNKED SHA-256.

The frozen v1 (blake3) commit route answers a cell publish with
410 ``unsupported_digest_algorithm``, which is where the first AOT mint in
platform history died with a complete, self-adopt-verified artifact in hand.
These drive the REAL :meth:`fleet_cells.CellPublisher.publish` — real sockets,
real multi-MB bytes, real chunk arithmetic, real threads — against a localhost
server that implements tensorhub's v2 publish contract INCLUDING the property
that makes it worth having: R2 refuses bytes that do not hash to the digest
signed into the presigned PUT. Nothing about the transport is stubbed, because
every property under test (which route is taken, what a second publish
uploads, what a corrupted chunk does) is a property of the IO.

The hub-side halves that cannot run here — the th#1340 cell writes and the
receipt mint — are proven against a real tensorhub separately (see the issue's
scratch-stack record); what is pinned here is the CLIENT contract that talks
to them.

Run: pytest tests/test_cell_publish_v2_pgw807.py -q
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import urllib.parse
import uuid
from pathlib import Path

import pytest
from hashrepo import TransferReport

import gen_worker.hubio.client as hub_client
from torch_compiled_graphs.identity import from_artifact_metadata, toolchain_axis_digest

from gen_worker import fleet_cells as fc
from gen_worker.hubio.client import HubPublishError
from gen_worker.procsplit import actions
from harness.cell_meta import CLASS_HASH as _CLASS_HASH
from harness.cell_meta import exported_cell_meta

FAMILY = "sdxl"

def _blob(total: int, seed: int = 7) -> bytes:
    out = bytearray(total)
    x = (seed * 2654435761 + 1) & 0xFFFFFFFF
    for i in range(total):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (x >> 24) & 0xFF
    return bytes(out)


class _Hub(http.server.BaseHTTPRequestHandler):
    """tensorhub's v2 publish contract, plus the R2 digest enforcement."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102
        pass

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {}

    # -- PUT: the CAS object store ------------------------------------------
    def do_PUT(self):  # noqa: N802
        srv = self.server
        digest = urllib.parse.urlparse(self.path).path.rsplit("/", 1)[-1]
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        with srv.lock:
            if digest in srv.corrupt:
                # A wire/host corruption: one byte differs from what the
                # publisher read. The store, not the uploader, is what catches
                # it — integrity must not depend on the sender's honesty.
                body = bytes(bytearray(body[:1]) + b"\x00" + body[2:]) if len(body) > 2 else b""
        # The digest is signed into the grant: bytes that do not hash to it are
        # refused and NO object exists afterwards. This is the property the
        # whole protocol is for, so the test store really enforces it.
        if hashlib.sha256(body).hexdigest() != digest:
            self._json(400, {"error": {"code": "digest_mismatch"}})
            return
        with srv.lock:
            srv.objects["sha256:" + digest] = body
            srv.puts.append(digest)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        srv = self.server
        path = urllib.parse.urlparse(self.path).path
        body = self._body()
        with srv.lock:
            srv.calls.append((path, body))

        if path.endswith("/v1/worker/compiled-graphs/publish-intent"):
            # pgw#1224: one answer per entry, in request order, ONE TOKEN EACH.
            entries = body.get("entries") or []
            self._json(200, {
                "object": "compiled_graph_publish_intent_batch",
                "repo": f"root/family-{body.get('family')}",
                "family": body.get("family"),
                "granted": len(entries),
                "answers": [
                    {"compiled_graph_key": e.get("compiled_graph_key"),
                     "status": "granted",
                     "capability_token": f"cap-token-{i}",
                     "expires_at_unix": 4102444800}
                    for i, e in enumerate(entries)],
            })
            return
        if path.endswith("/v1/worker/compiled-graphs/publish-complete"):
            self._json(200, {"recorded": True})
            return
        if path.endswith("/commits"):
            # The frozen v1 route (th#1303 phase 3.5), verbatim.
            self._json(410, {"error": {
                "code": "unsupported_digest_algorithm",
                "message": "the v1 (blake3) publish protocol is frozen on this "
                           "hub — no new v1 commit is accepted."}})
            return
        if path.endswith("/publishes"):
            files = body.get("files") or []
            need, have = [], []
            declared = {}
            for f in files:
                for c in f.get("chunks") or []:
                    declared["sha256:" + c["digest"]] = int(c["len"])
                if not f.get("chunks"):
                    declared[f["digest"]] = int(f["size_bytes"])
            with srv.lock:
                for digest, size in declared.items():
                    if digest in srv.objects:
                        have.append(digest)
                    else:
                        need.append({
                            "digest": digest, "size_bytes": size,
                            "put_url": f"{srv.base}/cas/{digest.split(':', 1)[1]}",
                            "headers": {"x-amz-checksum-sha256": digest},
                        })
                pid = str(uuid.uuid4())
                srv.sessions[pid] = declared
                srv.declares.append(dict(body))
            self._json(201, {"publish_id": pid, "have": have, "need": need,
                             "distinct_objects": len(declared),
                             "resident_objects": len(have)})
            return
        if path.endswith("/grants"):
            pid = path.split("/publishes/")[1].split("/")[0]
            with srv.lock:
                declared = srv.sessions.get(pid) or {}
                need = [
                    {"digest": d, "size_bytes": s,
                     "put_url": f"{srv.base}/cas/{d.split(':', 1)[1]}",
                     "headers": {}}
                    for d, s in declared.items() if d not in srv.objects
                ]
            self._json(200, {"need": need, "have": []})
            return
        if path.endswith("/complete"):
            pid = path.split("/publishes/")[1].split("/")[0]
            with srv.lock:
                declared = srv.sessions.get(pid) or {}
                missing = [d for d in declared if d not in srv.objects]
            if missing:
                # A th#1301 projection refusal, not an error envelope.
                self._json(409, {"status": {
                    "stage": "repudiated", "terminal": True,
                    "failure": {"code": "objects_missing", "retryable": False,
                                "message": f"{len(missing)} object(s) never landed"}}})
                return
            self._json(200, {"checkpoint": {"checkpoint_id": "sha256:" + "c" * 64}})
            return
        self._json(404, {"error": {"code": "not_found"}})


class _Server:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
        self.httpd.lock = threading.Lock()
        self.httpd.objects = {}
        self.httpd.corrupt = set()
        self.httpd.sessions = {}
        self.httpd.calls = []
        self.httpd.declares = []
        self.httpd.puts = []
        self.httpd.base = self.base
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def hub():
    s = _Server()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """Opaque exported bytes for the transport/dedup contract.

    TCG owns artifact construction and admission; this test exercises only the
    worker's HashRepo upload adapter, so fabricating a second archive format
    here would weaken both boundaries.
    """
    out = tmp_path / "compiled-graph.aoti"
    out.write_bytes(b"".join(_blob(20_000, i) for i in range(6)))
    return out


# pgw#1046: a REAL exported-cell envelope, not the identity-less stub this
# fixture used to carry. The publish path now recomputes the cell's key from
# these blocks and refuses anything that cannot state one, so a stub here would
# only prove that a cell the fleet can never arm still uploads.
META = exported_cell_meta(
    family=FAMILY,
    sku="l4",
    gen_worker="0.87.0",
    weight_lane="w8a8",
    lora_bucket=64,
)
CELL_KEY = META["compiled_graph_key"]

#: The shape every cell in the corpus was published under before pgw#1046 —
#: `_identity_axes`' six-axis FALLBACK, carrying neither identity digest. Kept
#: as a fixture so the refusal below is pinned against the real historical row,
#: not against an invented one.
TODAYS_FALLBACK_META = {
    "compiled_graph_key": "cg-key-v1-" + "b" * 56,
    "family": FAMILY, "sku": "l4", "sm": "89",
    "gen_worker": "0.87.0", "kind": "aot-inductor", "format": "pt2",
    "compile_mode": "regional", "weight_lane": "w8a8", "lora_bucket": 64,
}


def _publisher(hub) -> fc.CellPublisher:
    return fc.CellPublisher(base_url=hub.base, worker_jwt=lambda: "worker-jwt",
                            image_digest="sha256:" + "1" * 64)


def _events(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(
        fc.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append((kind, phase)))
    return seen


def test_publish_takes_the_v2_route_and_never_the_frozen_v1_one(
        hub, artifact, monkeypatch):
    """GREEN. The publisher declares sha256, PUTs every chunk, completes — and
    the frozen v1 route is never touched. (RED: this same server answers
    /commits with the live 410, which is what the v1 publisher got.)"""
    seen = _events(monkeypatch)
    ckpt = _publisher(hub).publish(FAMILY, artifact, dict(META),
                                   mint_duration_ms=347_940)
    assert ckpt == "sha256:" + "c" * 64

    paths = [p for p, _ in hub.httpd.calls]
    assert not any(p.endswith("/commits") for p in paths), (
        "the frozen v1 blake3 route must never be reached")
    assert any(p.endswith("/publishes") for p in paths)

    decl = hub.httpd.declares[0]
    (f,) = decl["files"]
    assert f["digest"].startswith("sha256:")
    assert f["size_bytes"] == artifact.stat().st_size
    # This small control-plane artifact is one HashRepo object.
    assert "chunks" not in f
    assert hub.httpd.puts == [f["digest"].split(":", 1)[1]]
    assert decl["mode"] == "replace"
    # pgw#1159: the cell key is the token's claim, never a body field.
    assert "flavor" not in decl
    assert "tags" not in decl and "default_flavor" not in decl
    # th#1340: the cell identity is hub-derived and rides the token.
    for forbidden in ("compiled_graph_publish", "compiled_graph_key", "family",
                      "owning_endpoint_id", "axes"):
        assert forbidden not in decl

    # The reassembled object is byte-identical to the artifact.
    joined = hub.httpd.objects[f["digest"]]
    assert joined == artifact.read_bytes()
    assert hashlib.sha256(joined).hexdigest() == f["digest"].split(":", 1)[1]

    assert [p for k, p in seen if k == "self_mint_publish"] == [
        "declared", "uploading", "committing", "committed"]


def test_intent_carries_the_identity_axes_and_the_mint_cost(
    hub, artifact, monkeypatch
):
    monkeypatch.setattr(fc.cc, "runtime_key", lambda: {"sku": "l4"})
    monkeypatch.setattr(fc.cc, "gen_worker_version", lambda: "0.87.0")
    _publisher(hub).publish(FAMILY, artifact, dict(META), mint_duration_ms=347_940)
    intent = next(b for p, b in hub.httpd.calls if p.endswith("publish-intent"))
    # pgw#1224: the three ATTESTED axes are batch-level (all three are
    # properties of the POD); the key, the identity axes and the mint cost are
    # per ENTRY — the last one because under the old whole-cell shape ONE
    # number covered 36 entries and could not answer which class is expensive.
    assert intent["axes"] == {"sku": "l4", "image_digest": "sha256:" + "1" * 64,
                              "gen_worker": "0.87.0"}
    (entry,) = intent["entries"]
    assert entry["compiled_graph_key"] == CELL_KEY
    assert entry["mint_duration_ms"] == 347_940
    assert entry["identity_axes"] == {
        "graph": _CLASS_HASH,
        "sm": "89",
        "toolchain": toolchain_axis_digest(META["toolchain"]),
    }

    complete = next(b for p, b in hub.httpd.calls if p.endswith("publish-complete"))
    assert complete["ok"] is True
    # pgw#711's artifact_digest/manifest_digest are gone: the hub's route
    # decodes no such fields, so sending them was a blake3 hash pass whose
    # result nothing read.
    assert set(complete) == {
        "family", "compiled_graph_key", "checkpoint_id", "ok"
    }


# ---------------------------------------------------------------------------
# pgw#1046 — what the hub needs to ARM the cell it just accepted
#
# th#1457's producer builds the worker's ExecutionSpec out of this exact map:
# `ArtifactFromCellRecord` reads `toolchain`/`env_seal`, `ArmFromVerifiedCell`
# reads `graph_contract`, and pgw#904's landed consumer refuses an
# ArtifactIdentity missing any of them. Before this, EVERY published cell fell
# to a six-axis subset that carried none — so the whole corpus was structurally
# unarmable, and it cost a mint per pod to find out.
# ---------------------------------------------------------------------------


def test_publish_intent_states_the_full_arming_identity(hub, artifact):
    """GREEN. The three axes the hub cannot recompute reach it, each recomputed
    from the artifact's OWN recorded blocks — plus the complete ck axis set.

    RED before the fix: `identity_axes` was
    ``{family, kind, format, sm, mode, lane}`` and this asserts on three keys
    that were not in it."""
    _publisher(hub).publish(FAMILY, artifact, dict(META))
    intent = next(b for p, b in hub.httpd.calls if p.endswith("publish-intent"))
    (entry,) = intent["entries"]
    axes = entry["identity_axes"]

    # Derived from the recorded blocks, never from a second stamp.
    assert axes["toolchain"] == toolchain_axis_digest(META["toolchain"])

    # ...and the KEY axes hash to the key the cell is published under, so
    # the hub's row and its flavor cannot describe two different cells. The
    # map also carries the wire facts (graph_contract, env_seal) and the
    # demoted store metadata (family, lane) — pgw#1059: neither is identity.
    assert from_artifact_metadata(META).value == entry["compiled_graph_key"] == CELL_KEY
    assert set(axes) == {"graph", "sm", "toolchain"}
    assert axes["graph"] == _CLASS_HASH


def test_the_pre_fix_fallback_row_shape_can_no_longer_be_published(hub, artifact):
    """RED-CHAIN PIN. The exact envelope the corpus was published from is now
    refused BEFORE a byte moves, by name, instead of producing the six-axis row
    tensorhub's `TestTH1457TodaysFallbackAxesRowRefusesTyped` proves unarmable.

    That hub-side test stays valid and must not be weakened: it guards the
    REFUSAL of a row shape that already exists in the store. This test guards
    the other end — that the worker can no longer create one."""
    with pytest.raises(fc.CellPublishRefused) as exc:
        _publisher(hub).publish(FAMILY, artifact, dict(TODAYS_FALLBACK_META))
    assert "no TCG identity" in str(exc.value)
    assert not hub.httpd.calls, "refused before the intent left the pod"

    # And the shape itself is unreachable: the one public entry builder cannot
    # emit an axis map without the TCG identity.
    with pytest.raises(fc.CellPublishRefused):
        fc.intent_entry(dict(TODAYS_FALLBACK_META))


def test_a_stamp_that_disagrees_with_the_recorded_axes_is_refused(hub, artifact):
    """A cell whose `cell_key` does not describe its own blocks is a cell the
    hub would index under one identity and the worker would fence on another."""
    forged = dict(META)
    forged["compiled_graph_key"] = "cg-key-v1-" + "9" * 56
    with pytest.raises(fc.CellPublishRefused, match="disagrees"):
        _publisher(hub).publish(FAMILY, artifact, forged)
    assert not hub.httpd.calls


def test_a_cell_with_no_class_hash_is_refused(hub, artifact):
    """The refusal that survives the inversion above: an entry that cannot
    state its own graph axis has no identity, so it would be stored under a
    flavor nothing can request."""
    hollow = dict(META)
    hollow["graph_class"] = {**META["graph_class"], "class_hash": ""}
    with pytest.raises(fc.CellPublishRefused):
        _publisher(hub).publish(FAMILY, artifact, hollow)
    assert not hub.httpd.calls


def test_seam_authorizes_the_live_publish_payloads(hub, artifact):
    """The delta-1 allowlist REFUSES an unlisted body key, and the compute
    child is the process that publishes. So the table and the payloads are one
    contract: drive the publisher, then authorize exactly what it sent."""
    _publisher(hub).publish(FAMILY, artifact, dict(META), mint_duration_ms=1)
    for path, body in hub.httpd.calls:
        if "/v1/worker/compiled-graphs/" not in path:
            continue
        actions.authorize({"method": "POST", "path": path, "json": body})


def test_republish_of_identical_bytes_uploads_nothing(hub, artifact):
    """Staged/dedup semantics: the second publish's need set is empty, so a
    pod re-run after a lost publish costs no transfer at all."""
    pub = _publisher(hub)
    pub.publish(FAMILY, artifact, dict(META))
    first = list(hub.httpd.puts)
    assert first
    hub.httpd.puts.clear()
    pub.publish(FAMILY, artifact, dict(META))
    assert hub.httpd.puts == [], "a re-publish of resident bytes re-uploaded"


def test_a_corrupted_chunk_is_refused_by_the_store_and_fails_the_publish(
        hub, artifact, monkeypatch):
    """A chunk whose bytes arrive altered is REFUSED by the store (400, and no
    object exists afterwards), so the publish fails loudly instead of minting a
    checkpoint over bytes nothing can arm. The refusal survives the client's
    re-plan passes, and the hub hears `ok=false`."""
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    hub.httpd.corrupt.add(digest)

    seen = _events(monkeypatch)
    with pytest.raises(HubPublishError) as exc:
        _publisher(hub).publish(FAMILY, artifact, dict(META))
    assert "failed to upload" in str(exc.value)
    # The poisoned object never landed; every honest one did.
    assert ("sha256:" + digest) not in hub.httpd.objects
    # The hub still hears about it: a failed publish files ok=false.
    complete = next(b for p, b in hub.httpd.calls if p.endswith("publish-complete"))
    assert complete["ok"] is False
    assert "declared" in [p for k, p in seen if k == "self_mint_publish"]


def test_complete_over_missing_objects_is_a_typed_repudiation(hub, artifact,
                                                              monkeypatch):
    """The other half of the refusal: when `/complete` answers with a th#1301
    PROJECTION rather than an error envelope, the client reads the hub's own
    `code` + `retryable` bit and puts THAT on the wire as the failure phase."""
    monkeypatch.setattr(hub_client, "upload", lambda *a, **k: TransferReport())
    with pytest.raises(HubPublishError) as exc:
        _publisher(hub).publish(FAMILY, artifact, dict(META))
    assert exc.value.code == "objects_missing"
    assert exc.value.retryable is False
    assert fc._publish_failure_phase(exc.value) == "objects_missing"


def test_publish_failure_phase_prefers_the_hubs_own_code():
    assert fc._publish_failure_phase(
        HubPublishError("x", status=410, code="unsupported_digest_algorithm")
    ) == "unsupported_digest_algorithm"
    assert fc._publish_failure_phase(HubPublishError("x", status=503)) == "http_503"
    assert fc._publish_failure_phase(RuntimeError("x")) == "RuntimeError"
