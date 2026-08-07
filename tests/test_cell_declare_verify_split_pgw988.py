"""pgw#988: the publish declare and the discovery filter are ONE contract.

th#1645 shrank the cell publish declare to control-plane size by stripping the
four blocks that grow with the model — correct about SIZE, and it fixed a real
413. But `aot_cells._candidates` verified the hub LISTING metadata, which IS
that declare body, with `aot_serve.verify` — the full check, `entries` map and
all. So from `c2e52f5f` every AOT cell published by any worker was rejected on
every pod as::

    malformed declared contract: metadata declares no entries map

and a pod that finds no cell mints its own. The fleet re-minted forever, per
cold boot, and the failure presented as COST rather than as an error.

The rig (pgw#978) found it because it runs the whole seam — mint, publish, and a
SECOND process adopting. This file is that seam as a CI row, minus the card: a
real `CellPublisher.publish` over a real socket into a hub that stores the
declare it actually received, then a real `aot_cells.discover` reading it back
off a real checkpoint listing and pulling the real artifact bytes the publisher
uploaded. Nothing between the two halves is stubbed, because the defect lived
exactly between them — each half was correct alone.

Only the two hardware probes are pinned (`aot_serve.runtime_key` and the torch
read inside `host_isa.stamp`): CI has neither a GPU nor torch, and the property
under test is a contract between two modules, not device detection. The ISA
stamp is still THIS host's real one, so `host_isa_reason` rules on it for real.

Run: pytest tests/test_cell_declare_verify_split_pgw988.py -q
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gen_worker import aot_cells, aot_serve, compile_cache as cc
from gen_worker import fleet_cells as fc
from gen_worker import guard_closure, host_isa

FAMILY = "sdxl"
CELL_KEY = "ck5-" + ("9f4c1d" * 10)[:56]
RUNTIME = {"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130", "cuda": "13.0"}

INPUTS = [
    {"name": "sample", "position": 0, "dtype": "bfloat16",
     "shape": [2, 4, 128, 128]},
    {"name": "timestep", "position": 1, "dtype": "int64", "shape": []},
]
CONSTANTS = [
    {"fqn": "conv_in.weight", "source": aot_serve.SOURCE_STATE_DICT,
     "dtype": "bfloat16", "shape": [320, 4, 3, 3]},
]


# ---------------------------------------------------------------------------
# One hub: the publish surface AND the catalog read surface, same store
# ---------------------------------------------------------------------------


class _Hub(http.server.BaseHTTPRequestHandler):
    """tensorhub's cell surfaces against ONE state.

    The point of serving both from one object: the metadata `aot_cells` reads
    back is the metadata `CellPublisher` sent, byte for byte, rather than a
    fixture that could stay green while the wire disagreed. That gap is the
    whole defect — every existing discovery test hands `_candidates` a full
    envelope no publisher has ever put on the wire.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:  # noqa: D102
        pass

    def _json(self, code: int, body: dict) -> None:
        self._send(code, json.dumps(body).encode(), "application/json")

    def _send(self, code: int, raw: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    # -- read surface (what a booting pod sees) -----------------------------

    def do_GET(self) -> None:  # noqa: N802
        srv = self.server
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.endswith("/checkpoints"):
            with srv.lock:
                items = list(srv.checkpoints)
            self._json(200, {"items": items})
            return
        if parsed.path.endswith("/resolve"):
            query = urllib.parse.parse_qs(parsed.query)
            digest = (query.get("digest") or [""])[0]
            with srv.lock:
                row = srv.manifests.get(digest)
            if row is None:
                self._json(404, {"error": {"code": "not_found"}})
                return
            self._json(200, {"files": [{
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                # th#1303: manifest v2 — algorithm-tagged, never a bare hex
                # mirror. A cell artifact is compiled code; the digest the
                # consumer checks against must state its own algorithm.
                "digest": row["digest"],
                "url": f"{srv.base}/blob/{row['digest'].split(':', 1)[1]}",
            }]})
            return
        if parsed.path.startswith("/blob/"):
            want = parsed.path[len("/blob/"):]
            with srv.lock:
                blob = srv.assemble(want)
            if blob is None:
                self._json(404, {"error": {"code": "not_found"}})
                return
            self._send(200, blob, "application/octet-stream")
            return
        self._json(404, {"error": {"code": "not_found"}})

    # -- write surface (what a minting pod does) ----------------------------

    def do_PUT(self) -> None:  # noqa: N802
        srv = self.server
        digest = urllib.parse.urlparse(self.path).path.rsplit("/", 1)[-1]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        # The digest is signed into the grant: R2 refuses bytes that do not
        # hash to it, and so does this.
        if hashlib.sha256(raw).hexdigest() != digest:
            self._json(400, {"error": {"code": "digest_mismatch"}})
            return
        with srv.lock:
            srv.objects["sha256:" + digest] = raw
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        srv = self.server
        path = urllib.parse.urlparse(self.path).path
        body = self._body()

        if path.endswith("/v1/worker/cells/publish-intent"):
            self._json(200, {"capability_token": "cap-token",
                             "repo": cc.system_repo(str(body.get("family")))})
            return
        if path.endswith("/v1/worker/cells/publish-complete"):
            with srv.lock:
                srv.completes.append(dict(body))
            self._json(200, {"recorded": True})
            return
        if path.endswith("/publishes"):
            need, plan = [], {}
            for f in body.get("files") or []:
                chunks = f.get("chunks") or []
                whole = str(f.get("digest") or "")
                plan[whole] = {
                    "path": str(f.get("path") or ""),
                    "size_bytes": int(f.get("size_bytes") or 0),
                    "parts": ["sha256:" + c["digest"] for c in chunks] or [whole],
                }
                for want, size in (
                    [("sha256:" + c["digest"], int(c["len"])) for c in chunks]
                    or [(whole, int(f.get("size_bytes") or 0))]
                ):
                    need.append({
                        "digest": want, "size_bytes": size,
                        "put_url": f"{srv.base}/cas/{want.split(':', 1)[1]}",
                        "headers": {"x-amz-checksum-sha256": want}})
            pid = str(uuid.uuid4())
            with srv.lock:
                srv.declares.append(dict(body))
                srv.declare_lengths.append(
                    int(self.headers.get("Content-Length") or 0))
                srv.sessions[pid] = plan
            self._json(201, {"publish_id": pid, "have": [], "need": need,
                             "distinct_objects": len(need),
                             "resident_objects": 0})
            return
        if path.endswith("/grants"):
            self._json(200, {"need": [], "have": []})
            return
        if path.endswith("/complete"):
            pid = path.split("/publishes/")[1].split("/")[0]
            with srv.lock:
                plan = srv.sessions.get(pid) or {}
                missing = [d for row in plan.values() for d in row["parts"]
                           if d not in srv.objects]
                if missing:
                    self._json(409, {"status": {
                        "stage": "repudiated", "terminal": True,
                        "failure": {"code": "objects_missing",
                                    "retryable": False,
                                    "message": f"{len(missing)} never landed"}}})
                    return
                checkpoint = "sha256:" + hashlib.sha256(pid.encode()).hexdigest()
                # The checkpoint the catalog will serve carries the DECLARE
                # body as its metadata — exactly what tensorhub stores and
                # hands back on the listing route.
                declare = srv.declares[-1]
                for whole, row in plan.items():
                    srv.manifests[checkpoint] = {**row, "digest": whole}
                srv.checkpoints.append({
                    "checkpoint_id": checkpoint,
                    "updated_at": "2026-08-06T00:00:00Z",
                    "metadata": dict(declare.get("metadata") or {}),
                })
            self._json(200, {"checkpoint": {"checkpoint_id": checkpoint}})
            return
        self._json(404, {"error": {"code": "not_found"}})


class _Server:
    def __init__(self) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
        self.httpd.lock = threading.Lock()
        self.httpd.objects = {}
        self.httpd.sessions = {}
        self.httpd.manifests = {}
        self.httpd.checkpoints = []
        self.httpd.declares = []
        self.httpd.declare_lengths = []
        self.httpd.completes = []
        self.httpd.base = self.base
        self.httpd.assemble = self._assemble
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _assemble(self, hex_digest: str) -> Optional[bytes]:
        """The whole file, from the objects that actually landed in the CAS."""
        for row in self.httpd.manifests.values():
            if row["digest"] != "sha256:" + hex_digest:
                continue
            parts = [self.httpd.objects.get(d) for d in row["parts"]]
            if any(p is None for p in parts):
                return None
            return b"".join(p for p in parts if p is not None)
        return None

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


@pytest.fixture(autouse=True)
def _isa(monkeypatch: pytest.MonkeyPatch) -> None:
    """`host_isa.stamp` reads `torch._inductor.config.cpp` for the mint's
    imposed march; CI has no torch. The MACHINE and LEVEL stay this host's real
    ones, so the ISA gate that refuses an unexecutable cell still rules for
    real — only the unclamped-march read is stood in for."""
    monkeypatch.setattr(host_isa, "stamp", lambda: {
        "machine": host_isa.machine(), "march": "", "simdlen": 0,
        "level": host_isa.host_level() if host_isa.machine() == "x86_64" else "",
    })


@pytest.fixture()
def _runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME))


@pytest.fixture()
def _lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "cell_base_execution_lane", lambda pipe: "w8a8")


class _FakePipe:
    pass


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 64


def _meta() -> Dict[str, Any]:
    """A real AOT cell envelope: `artifact_metadata` builds it, so `entries`
    carries stamped `range_digest`/`class_hash` and the ck6 combined hash —
    the very block th#1645 removed from the declare."""
    meta = aot_serve.artifact_metadata(
        family=FAMILY, precision="bf16", cell_key=CELL_KEY,
        entries={"unet/g": {
            "target": "unet", "fork": [], "class_dims": [],
            "inputs": [dict(r) for r in INPUTS], "symbols": {},
            "constants": [dict(r) for r in CONSTANTS], "graph": {},
        }},
        lora_bucket=64)
    meta["weight_lane"] = "w8a8"
    meta["gen_worker"] = "0.93.1"
    # The other unbounded block, at a magnitude that would have re-broken the
    # 413 had the fix been "put it all back": one real sdxl cell measured
    # 13,092,487 bytes of guard manifest.
    meta[guard_closure.MANIFEST_KEY] = {
        "v": 2,
        "graphs": {f"graph_{g}": {"guards": [f"L['x'].size()[{i}] == 128"
                                             for i in range(200)]}
                   for g in range(400)},
    }
    return meta


def _artifact(meta: Dict[str, Any], tmp_path: Path) -> Path:
    content = tmp_path / "content"
    content.mkdir(parents=True, exist_ok=True)
    (content / aot_serve.PACKAGE_NAME).write_bytes(b"fake-pt2" * 4096)
    return aot_serve.pack(content, tmp_path / f"{CELL_KEY}.tar.gz", meta)


def _publish(hub: _Server, artifact: Path, meta: Dict[str, Any]) -> str:
    return fc.CellPublisher(
        base_url=hub.base,
        worker_jwt=lambda: "worker-jwt",
        image_digest="sha256:" + "e" * 64,
    ).publish(FAMILY, artifact, meta, 1234)


# ---------------------------------------------------------------------------
# THE ROW — publish through the real publisher, adopt through the real filter
# ---------------------------------------------------------------------------


def test_a_published_cell_is_adopted_and_not_re_minted(
    hub: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    _runtime: None, _lane: None,
) -> None:
    """The rig's `adopt ok`, as a CI row. RED on `c2e52f5f`: discovery returned
    None with `verify:malformed declared contract: metadata declares no entries
    map`, and the caller's miss policy is a self-mint."""
    monkeypatch.setattr(fc.broker, "active", lambda: False)
    meta = _meta()
    artifact = _artifact(meta, tmp_path)

    checkpoint = _publish(hub, artifact, meta)
    assert checkpoint
    assert hub.httpd.completes[-1]["ok"] is True

    adopted = aot_cells.discover(
        _FakePipe(), _Cfg(), base_url=hub.base,
        worker_jwt=lambda: "worker-jwt", cache_dir=tmp_path / "cache")

    assert adopted is not None, (
        "the cell this very test published is undiscoverable — pgw#988: the "
        "publisher declares a bounded subset and the consumer verifies a "
        "superset of it, so every pod re-mints")
    assert adopted.cell_key == CELL_KEY
    assert adopted.family == FAMILY
    # The bytes the pod pulled are the bytes the publisher uploaded, whole-file
    # digest verified by the real `verify_file_digest` on the way in.
    assert adopted.artifact.read_bytes() == artifact.read_bytes()
    assert adopted.snapshot_digest == "sha256:" + hashlib.sha256(
        artifact.read_bytes()).hexdigest()


def test_the_adopted_declare_still_never_carries_the_unbounded_blocks(
    hub: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    _runtime: None, _lane: None,
) -> None:
    """The fix is a SPLIT, not a revert. th#1645's property has to hold on the
    same wire that now yields an adoption: had `entries` gone back into the
    declare, this cell would publish and adopt and the next 200 MB one would
    413 again."""
    monkeypatch.setattr(fc.broker, "active", lambda: False)
    meta = _meta()
    _publish(hub, _artifact(meta, tmp_path), meta)

    declared = hub.httpd.declares[-1]["metadata"]
    for block in sorted(fc._UNBOUNDED_ENVELOPE_BLOCKS):
        assert block not in declared, f"{block} is data; it must not ride the declare"
    assert hub.httpd.declare_lengths[-1] < fc.CELL_DECLARE_MAX_BYTES
    # And the listing a pod reads is that same body — the premise of the row
    # above, asserted rather than assumed.
    assert hub.httpd.checkpoints[-1]["metadata"] == declared

    adopted = aot_cells.discover(
        _FakePipe(), _Cfg(), base_url=hub.base,
        worker_jwt=lambda: "worker-jwt", cache_dir=tmp_path / "cache")
    assert adopted is not None


# ---------------------------------------------------------------------------
# The contract is ONE thing now, and it is checked
# ---------------------------------------------------------------------------


def test_the_declared_and_the_verified_key_sets_are_one_contract() -> None:
    """Two computations of one fact with no single authority is the defect
    class (pgw#985, one layer up). The two sets are now checked against each
    other at import — this asserts the check exists and holds."""
    fc.assert_declare_contract()
    assert not (fc._UNBOUNDED_ENVELOPE_BLOCKS & aot_cells.DECLARE_CONTRACT_KEYS)
    # Everything the pre-download filter reads survives the projection.
    kept = fc.control_plane_metadata(_meta())
    assert aot_cells.DECLARE_CONTRACT_KEYS <= set(kept)


def test_moving_a_verified_block_out_of_the_declare_now_fails_loudly() -> None:
    """The acceptance condition: the next block that moves fails a test run,
    not a fleet. This is th#1645's exact edit, replayed."""
    with pytest.raises(RuntimeError) as excinfo:
        fc.assert_declare_contract(
            dropped=fc._UNBOUNDED_ENVELOPE_BLOCKS | {"host_isa"},
            read=aot_cells.DECLARE_CONTRACT_KEYS)
    assert "host_isa" in str(excinfo.value)
    assert "pgw#988" in str(excinfo.value)


def test_the_entries_contract_is_verified_on_the_artifact_not_the_declare(
    tmp_path: Path, _runtime: None,
) -> None:
    """The check MOVED; it did not go away. A declare-shaped metadata (no
    `entries`) passes the pre-download filter, and the same metadata is still
    refused by the full check — which is what the staged artifact gets."""
    meta = _meta()
    declare = fc.control_plane_metadata(meta)

    assert aot_serve.verify_declared(declare, family=FAMILY) == ""
    assert "no entries map" in aot_serve.verify(declare, family=FAMILY)

    # And a tampered contract inside the artifact is still refused at staging,
    # by name, before anything is dlopen'd.
    bad = dict(meta)
    bad["entries"] = {
        name: {**block, "class_hash": "0" * 16}
        for name, block in meta["entries"].items()}
    artifact = _artifact(bad, tmp_path / "bad")
    with pytest.raises(aot_serve.AdoptError) as excinfo:
        aot_serve.stage_artifact(artifact, FAMILY, cache_dir=tmp_path / "stage")
    assert "class_hash" in str(excinfo.value)


def test_a_declare_with_no_entries_is_no_longer_a_discovery_rejection(
    _runtime: None,
) -> None:
    """The rejection class that ate the fleet, asserted absent at the unit
    seam: `_candidates` counted `verify:malformed declared contract: metadata
    declares no entries map=N` for every published cell."""
    declare = fc.control_plane_metadata(_meta())
    rejected: Dict[str, int] = {}
    rows = aot_cells._candidates(
        [{"checkpoint_id": "cafe01", "updated_at": "2026-08-06T00:00:00Z",
          "metadata": declare}],
        FAMILY, "w8a8-lora64", rejected)
    assert rejected == {}, rejected
    assert [r[1] for r in rows] == ["cafe01"]


def test_a_genuinely_unloadable_cell_is_still_refused_before_download(
    _runtime: None,
) -> None:
    """The split must not have loosened the pre-download filter into a pass.
    Every axis a declare CAN carry still refuses, by its own class, with no
    bytes moved."""
    base = fc.control_plane_metadata(_meta())
    cases = {
        "sm": ({**base, "sm": "sm_80"}, "verify:sm"),
        "torch": ({**base, "torch": "2.9.0+cu128"}, "verify:torch"),
        "kind": ({**base, "kind": "torch-inductor-cache"}, "not_an_aot_cell"),
        "host_isa": ({k: v for k, v in base.items() if k != "host_isa"},
                     aot_serve.NO_HOST_ISA_STAMP),
        "baked": ({**base, "package_constants_in_so": True},
                  "verify:artifact was minted"),
        "lane": ({**base, "weight_lane": "w8a16"}, "lane_mismatch"),
    }
    for axis, (meta, want) in cases.items():
        rejected: Dict[str, int] = {}
        rows = aot_cells._candidates(
            [{"checkpoint_id": "x", "updated_at": "2026-08-06T00:00:00Z",
              "metadata": meta}], FAMILY, "w8a8-lora64", rejected)
        assert rows == [], f"{axis} was admitted"
        classes: List[str] = list(rejected)
        assert any(c.startswith(want) for c in classes), (axis, classes)
