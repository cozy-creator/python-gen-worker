"""th#1645 + pgw#987: the cell publish declare is CONTROL, and a 413 is final.

Two defects, one incident. On 2026-08-06 a fully-minted AOT cell — sealed,
key byte-identical across two cards — could not be published by any worker on
any card:

  * **th#1645.** The declare body carried the cell's ENTIRE envelope. On a real
    published sdxl cell the `guard_manifest` block alone measured 13,092,487 of
    13,377,167 metadata bytes (98%) against a 69 MB artifact; at a ~200 MB AOT
    cell the body crossed the hub's 32 MiB route cap and
    `POST /api/v1/repos/:org/:name/publishes` answered **413 in 25 µs**, before
    reading a byte. The bytes themselves were never the problem — they go
    worker -> R2 over presigned PUTs and always did.

  * **pgw#987.** The client read those 32 definite refusals as *"network, no
    definite hub answer for 609s"* and re-sent a permanently-refused body for
    ten minutes of a paid pod, because gin's `AbortWithStatusJSON` envelope
    carries a STRING `error` with no `message` and matched neither shape
    `http_origin.response_is_from_hub` knows.

Everything here is real: a real socket, a real body cap enforced on the
declared `Content-Length` exactly as the hub's middleware does, a real
~200 MB artifact hashed and chunked by the real chunker and uploaded over real
presigned PUTs by the real `CellPublisher.publish`. Nothing about the transport
is stubbed, because every property under test is a property of the IO.

Run: pytest tests/test_cell_declare_bound_th1645_pgw987.py -q
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import threading
import urllib.parse
import uuid
from pathlib import Path

import pytest

# pgw#1181: `guard_closure.MANIFEST_KEY` went with `closure_manifest`, the
# only writer of this block, when the `torch-inductor-cache` format was
# deleted. The BLOCK NAME stays spelled out here because
# `fleet_cells._UNBOUNDED_ENVELOPE_BLOCKS` still lists it as a literal:
# the control-plane cap is a defensive filter over whatever an envelope
# carries, and what these rows prove — that an unbounded block is dropped
# before the hub sees it, and that a 200 MB cell still publishes — is a
# property of the CAP, not of any one producer.
GUARD_MANIFEST_BLOCK = "guard_manifest"


from harness.cell_meta import exported_cell_meta
from hashrepo import MAX_CHUNK_SIZE

from gen_worker import fleet_cells as fc
from gen_worker import http_origin
from gen_worker.hubio.client import HubPublishError

FAMILY = "sdxl"
# pgw#1046: computed from `_meta()`'s identity blocks, never invented — the
# publish path refuses a stamp its recorded axes do not describe.
COMPILED_GRAPH_KEY = exported_cell_meta(
    family=FAMILY, sku="rtx-4090", gen_worker="0.91.0",
    weight_lane="w8a8", lora_bucket=64)["compiled_graph_key"]

# The hub's group-wide default (internal/api/api.go: `maxRequestBodyMiddleware(32 << 20)`).
HUB_BODY_CAP = 32 << 20

# Attempt twenty-two's two mints measured 202,5xx,xxx and 204,7xx,xxx bytes.
# The artifact is REAL at that size here: a declare that fits is only half the
# proof if the bytes it describes were never moved.
ARTIFACT_BYTES = 202_500_000


class _Hub(http.server.BaseHTTPRequestHandler):
    """tensorhub's v2 publish contract behind the REAL body cap.

    `maxRequestBodyMiddleware` aborts on the DECLARED `Content-Length` alone,
    before reading the body — which is why the hub answered in 25 µs and why
    the client, still writing 200 MB into a stream the peer had finished with,
    saw an SSL EOF and called it a network fault. Reproduced faithfully: the
    cap is checked against the header, and the request body is NOT drained.
    """

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

    def _over_cap(self) -> bool:
        declared = int(self.headers.get("Content-Length") or 0)
        cap = self.server.body_cap
        if declared <= cap:
            return False
        with self.server.lock:
            self.server.refusals.append((self.path, declared))
        # VERBATIM the shape the deployed hub emits today (gin
        # `AbortWithStatusJSON`): a bare string `error`, no `message`. The
        # client must terminate on this WITHOUT needing a hub deploy — old
        # hubs exist, and a classifier that can only read the fixed envelope
        # would have left this incident live until both sides shipped.
        self._json(413, {"error": "request_body_too_large",
                         "max_body_bytes": cap,
                         "declared_length": declared})
        self.close_connection = True
        return True

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {}

    def do_PUT(self):  # noqa: N802
        srv = self.server
        digest = urllib.parse.urlparse(self.path).path.rsplit("/", 1)[-1]
        n = int(self.headers.get("Content-Length") or 0)
        h = hashlib.sha256()
        left = n
        while left > 0:
            block = self.rfile.read(min(1 << 20, left))
            if not block:
                break
            h.update(block)
            left -= len(block)
        # The digest is signed into the grant: R2 refuses bytes that do not
        # hash to it. Enforced here for the same reason it is enforced there.
        if h.hexdigest() != digest:
            self._json(400, {"error": {"code": "digest_mismatch"}})
            return
        with srv.lock:
            srv.objects.add("sha256:" + digest)
            srv.put_bytes += n
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        srv = self.server
        path = urllib.parse.urlparse(self.path).path
        if path.endswith("/publishes") and self._over_cap():
            return
        body = self._body()

        if path.endswith("/v1/worker/compiled-graphs/publish-intent"):
            entries = body.get("entries") or []
            self._json(200, {
                "repo": f"root/family-{body.get('family')}",
                "granted": len(entries),
                "answers": [
                    {"compiled_graph_key": e.get("compiled_graph_key"), "status": "granted",
                     "capability_token": f"cap-token-{i}"}
                    for i, e in enumerate(entries)]})
            return
        if path.endswith("/v1/worker/compiled-graphs/publish-complete"):
            with srv.lock:
                srv.completes.append(dict(body))
            self._json(200, {"recorded": True})
            return
        if path.endswith("/publishes"):
            declared = {}
            for f in body.get("files") or []:
                for c in f.get("chunks") or []:
                    declared["sha256:" + c["digest"]] = int(c["len"])
                if not f.get("chunks"):
                    declared[f["digest"]] = int(f["size_bytes"])
            with srv.lock:
                srv.declares.append(dict(body))
                srv.declare_lengths.append(
                    int(self.headers.get("Content-Length") or 0))
                have = [d for d in declared if d in srv.objects]
                need = [
                    {"digest": d, "size_bytes": s,
                     "put_url": f"{srv.base}/cas/{d.split(':', 1)[1]}",
                     "headers": {"x-amz-checksum-sha256": d}}
                    for d, s in declared.items() if d not in srv.objects
                ]
                pid = str(uuid.uuid4())
                srv.sessions[pid] = declared
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
                missing = [d for d in (srv.sessions.get(pid) or {})
                           if d not in srv.objects]
            if missing:
                self._json(409, {"status": {
                    "stage": "repudiated", "terminal": True,
                    "failure": {"code": "objects_missing", "retryable": False,
                                "message": f"{len(missing)} object(s) never landed"}}})
                return
            self._json(200, {"checkpoint": {"checkpoint_id": "sha256:" + "c" * 64}})
            return
        self._json(404, {"error": {"code": "not_found"}})


class _Server:
    def __init__(self, body_cap: int = HUB_BODY_CAP):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
        self.httpd.lock = threading.Lock()
        self.httpd.body_cap = body_cap
        self.httpd.objects = set()
        self.httpd.sessions = {}
        self.httpd.declares = []
        self.httpd.declare_lengths = []
        self.httpd.refusals = []
        self.httpd.completes = []
        self.httpd.put_bytes = 0
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


def _guard_manifest(graphs: int, rows: int) -> dict:
    """A guard manifest the size the real one is.

    Not padding: the shape is the real one (`graphs` -> per-graph guard rows),
    scaled from the MEASURED block on a real published sdxl cell — checkpoint
    sha256:926bc9f5…, 13,092,487 bytes of guard manifest against a 69,045,459
    byte artifact. The envelope grows with the artifact, so attempt
    twenty-two's ~202.5 MB cell puts it near 38 MB, which is the number that
    crossed the 32 MiB cap. That is what this reproduces.
    """
    return {
        "v": 2,
        "graphs": {
            f"graph_{g}": {
                "guards": [
                    f"L['kwargs']['x'].size()[{i}] == 128  # {'g' * 96}"
                    for i in range(rows)
                ],
                "code_hash": hashlib.sha256(str(g).encode()).hexdigest(),
            }
            for g in range(graphs)
        },
        "composition": {f"mod_{i}": "sha256:" + "d" * 64 for i in range(400)},
    }


def _meta() -> dict:
    """A real cell envelope: the 34 keys a published sdxl cell carries, with
    the two unbounded blocks at their measured magnitudes."""
    # pgw#1046: the identity blocks are REAL (the publish path recomputes the
    # key from them and refuses a cell that cannot state one); everything after
    # them is the measured bulk this test exists to size.
    meta = exported_cell_meta(
        family=FAMILY, sku="rtx-4090", gen_worker="0.91.0",
        weight_lane="w8a8", lora_bucket=64)
    meta |= {
        "torch": "2.9.0+cu128", "triton": "3.5.0", "cuda": "12.8",
        "cuda_driver": "570.86", "storage_dtype": "fp8", "source_ref": "root/sdxl",
        "source_digest": "", "family_reason": "declared-by-endpoint",
        "low_vram_mode": False, "shapes": {"h": 1024, "w": 1024},
        "shape_contract": {"strategy": "dynamic"}, "targets": ["unet"],
        "guidance_scales": [7.5], "content_keys": ["unet", "vae"],
        "libs": ["diffusers"], "image_digest": "sha256:" + "e" * 64,
        "graph_signature": "sha256:" + "f" * 32,
        "loaded_libs": {f"lib{i}": f"1.{i}.0" for i in range(40)},
        "code_closure": {f"fn_{i}": "sha256:" + "b" * 64 for i in range(60)},
        # The two unbounded blocks, at measured magnitude.
        GUARD_MANIFEST_BLOCK: _guard_manifest(graphs=700, rows=400),
        "weight_contract": {f"unet.block{i}.weight": [1280, 1280, "fp8"]
                            for i in range(700)},
    }
    return meta


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """A REAL ~200 MB cell artifact — the size attempt twenty-two produced."""
    out = tmp_path / f"{COMPILED_GRAPH_KEY}.tar.gz"
    # Every megabyte distinct. A repeating block would make all four 64 MiB
    # chunks hash the same, the CAS would dedup three of them, and the test
    # would silently prove a 68 MB upload instead of a 200 MB one.
    block = bytearray(os.urandom(1 << 20))
    with out.open("wb") as fh:
        written = 0
        while written < ARTIFACT_BYTES:
            block[:8] = (written // (1 << 20)).to_bytes(8, "big")
            n = min(len(block), ARTIFACT_BYTES - written)
            fh.write(bytes(block[:n]))
            written += n
    return out


def _publisher(hub: _Server) -> fc.CellPublisher:
    return fc.CellPublisher(
        base_url=hub.base,
        worker_jwt=lambda: "worker-jwt",
        image_digest="sha256:" + "e" * 64,
    )


# --------------------------------------------------------------------------
# th#1645 — the declare is CONTROL
# --------------------------------------------------------------------------


def test_the_envelope_that_broke_the_hub_is_over_the_cap():
    """The premise, measured rather than asserted: the OLD declare body — the
    whole envelope, which is what shipped — really does exceed 32 MiB."""
    old_body = {k: v for k, v in _meta().items() if v is not None}
    encoded = len(json.dumps({"mode": "replace", "files": [], "flavor": COMPILED_GRAPH_KEY,
                              "metadata": old_body}).encode())
    assert encoded > HUB_BODY_CAP, (
        f"the fixture no longer reproduces the incident: {encoded} bytes is "
        f"under the {HUB_BODY_CAP}-byte cap")


def test_control_plane_metadata_drops_only_the_unbounded_blocks():
    meta = _meta()
    kept = fc.control_plane_metadata(meta)

    for block in (GUARD_MANIFEST_BLOCK, "weight_contract"):
        assert block not in kept, f"{block} is data; it must not ride the declare"
    # Everything else survives byte-identically — this is a projection, not a
    # rewrite, so the hub still receives every fact it ever received that is
    # bounded in size.
    for key, value in meta.items():
        if key in fc._UNBOUNDED_ENVELOPE_BLOCKS:
            continue
        assert kept[key] == value, f"{key} was altered on the way to the declare"

    encoded = len(json.dumps(kept, sort_keys=True, default=str).encode())
    assert encoded < fc.CELL_DECLARE_MAX_BYTES
    # And by two orders of magnitude under the hub's route cap, so the bound
    # that broke A1 is no longer anywhere near the traffic.
    assert encoded * 100 < HUB_BODY_CAP


def test_a_new_unbounded_block_is_refused_on_the_pod_and_named():
    """The bound has to fail LOUDLY and name the culprit. The alternative is
    what happened: a 413 that named no key, ten minutes into a paid pod, and a
    P0 filed against the wrong subsystem."""
    meta = _meta()
    meta["autotune_log"] = {f"kernel_{i}": "x" * 512 for i in range(20_000)}

    with pytest.raises(fc.CellPublishRefused) as excinfo:
        fc.control_plane_metadata(meta)
    assert "autotune_log" in str(excinfo.value)
    assert "th#1645" in str(excinfo.value)
    # Its own groupable token: a code defect must not land in the same bucket
    # as the hub's trust-tier and quota refusals.
    assert excinfo.value.code == fc.COMPILED_GRAPH_DECLARE_OVERSIZE_CODE
    assert fc._publish_failure_phase(excinfo.value) == "compiled_graph_declare_oversize"


def test_a_real_200mb_cell_publishes_through_the_real_cap(hub, artifact, monkeypatch):
    """THE proof: the exact artifact size and the exact envelope that produced
    32 × 413, now landing — declare accepted under the real cap, every chunk
    uploaded and digest-verified, publish-complete recorded ok."""
    monkeypatch.setattr(fc.broker, "active", lambda: False)
    assert artifact.stat().st_size == ARTIFACT_BYTES

    checkpoint = _publisher(hub).publish(FAMILY, artifact, _meta(), 354_450)

    assert checkpoint == "sha256:" + "c" * 64
    assert hub.httpd.refusals == [], (
        f"the cap refused the publish: {hub.httpd.refusals}")

    # The declare was CONTROL-sized...
    assert len(hub.httpd.declares) == 1
    assert hub.httpd.declare_lengths[0] < fc.CELL_DECLARE_MAX_BYTES
    declared_meta = hub.httpd.declares[0]["metadata"]
    assert GUARD_MANIFEST_BLOCK not in declared_meta
    assert declared_meta["compiled_graph_key"] == COMPILED_GRAPH_KEY

    # ...and the DATA still moved, all of it, over presigned PUTs, every
    # object refused unless it hashed to the digest signed into its grant.
    assert hub.httpd.put_bytes == ARTIFACT_BYTES
    expected_chunks = -(-ARTIFACT_BYTES // MAX_CHUNK_SIZE)
    assert len(hub.httpd.objects) == expected_chunks

    assert hub.httpd.completes[-1]["ok"] is True
    assert hub.httpd.completes[-1]["compiled_graph_key"] == COMPILED_GRAPH_KEY


# --------------------------------------------------------------------------
# pgw#987 — a 413 is a verdict, not silence
# --------------------------------------------------------------------------


def test_the_hubs_413_envelope_is_a_definite_answer():
    """Directly against the shape the deployed hub emits. Before this, both
    envelope tests failed on it and the loop treated a verdict as silence."""

    class _Resp:
        status_code = 413
        headers = {"Content-Type": "application/json"}
        text = '{"error":"request_body_too_large"}'

        def json(self):
            return {"error": "request_body_too_large",
                    "max_body_bytes": HUB_BODY_CAP, "declared_length": 39_000_000}

    resp = _Resp()
    # The envelope test still says "not the hub's shape" — that is the hub's
    # bug (fixed in tensorhub separately) and NOT what makes this terminal.
    assert http_origin.response_is_from_hub(resp) is False
    # Terminal anyway: 413 is determined by our own Content-Length, which is
    # byte-identical on every retry, so no origin can make retrying useful.
    assert http_origin.is_definite_hub_answer(resp) is True


def test_a_proxy_404_is_still_indefinite():
    """The guard on the pgw#743 doctrine this must not weaken: ngrok's offline
    page cost two 58-minute clones when it was read as a verdict."""

    class _Resp:
        status_code = 404
        headers = {"Content-Type": "text/html"}
        text = "<!DOCTYPE html><html>ngrok</html>"

        def json(self):
            raise ValueError("not json")

    assert http_origin.is_definite_hub_answer(_Resp()) is False


def test_an_oversized_publish_stops_on_the_first_refusal(artifact, monkeypatch):
    """One attempt, a typed terminal naming the hub's own code, no retry.

    Attempt twenty-two sent this body 23 times over 609 seconds and reported
    `network, no definite hub answer`. The pod cost is the smaller half; the
    larger half is that the typed event named the wrong subsystem and a P0 was
    filed against a config-roll drain that had nothing to do with it.
    """
    monkeypatch.setattr(fc.broker, "active", lambda: False)
    # A cap BELOW the (now bounded) declare, so the refusal is real rather
    # than simulated: the hub answers 413 on Content-Length exactly as it does
    # in production, while the client is still writing.
    hub = _Server(body_cap=2048)
    try:
        with pytest.raises(HubPublishError) as excinfo:
            _publisher(hub).publish(FAMILY, artifact, _meta(), 0)
    finally:
        hub.close()

    exc = excinfo.value
    assert exc.status == 413
    assert exc.code == "request_body_too_large", (
        "the typed event must carry the hub's own code — `http_413` cannot be "
        "grouped and a transport string is a lie")
    assert fc._publish_failure_phase(exc) == "request_body_too_large"
    assert len(hub.httpd.refusals) == 1, (
        f"a permanent refusal was retried {len(hub.httpd.refusals)} times")
