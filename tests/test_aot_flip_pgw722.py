"""pgw#722 pilot flip seams (SDXL-AOT-PILOT-RUNBOOK.md §3) — F1/F2/F3.

One flag (``Settings.compile_prefer_aot`` / ``GEN_WORKER_PREFER_AOT``),
default OFF. The red half of every seam asserts flag-off behavior is
IDENTICAL to today's: no discovery call, no lifted install, buffer-copy
adapter attach. The green half asserts each seam's contract:

* **F1** — fetch-and-filter cell discovery: list the family repo's
  checkpoints, keep only receipt-shaped ``aot-inductor`` cells this
  runtime can serve (runtime key + contract via ``aot_serve.verify``,
  canonical lane/bucket), newest wins; the downloaded artifact feeds the
  EXISTING ``provision.enable_compiled`` choke point and the adopted
  identity rides ``ArmOutcome.self_mint`` so the advertisement never
  names bytes the pipe does not serve.
* **F2** — lifted-binding install at arm: ``install_lifted_lora_forward``
  on the artifact's target module AFTER the branch containers exist and
  BEFORE ``aot_serve.enable`` (the C2 pod-10 order); rolled back when the
  arm refuses.
* **F3** — lifted-aware adapter attach: with a binding installed, attach
  and clear write through the binding views (the flat pair the artifact
  reads); the canonical module buffers stay untouched.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pytest

from gen_worker import aot_cells, aot_serve, cell_key, fleet_cells
from gen_worker.config import get_settings
from gen_worker.models.chunk_cas import sha256_file

FAMILY = "sdxl"
RUNTIME = {"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
           "cuda": "13.0"}

INPUTS = [
    {"name": "sample", "position": 0, "dtype": "bfloat16",
     "shape": [2, 4, 128, 128]},
    {"name": "timestep", "position": 1, "dtype": "int64", "shape": []},
    {"name": "lora_a", "position": 2, "dtype": "bfloat16", "shape": [64]},
    {"name": "lora_b", "position": 3, "dtype": "bfloat16", "shape": [64]},
]
CONSTANTS = [
    {"fqn": "conv_in.weight", "source": aot_serve.SOURCE_STATE_DICT,
     "dtype": "bfloat16", "shape": [320, 4, 3, 3]},
]


def _key(seed: str) -> str:
    return "ck5-" + (seed * 56)[:56]


@pytest.fixture()
def _runtime_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME))


def _pilot_meta(key: str, *, lane: str = "w8a8", bucket: int = 64,
                **over: Any) -> Dict[str, Any]:
    """A published-cell metadata dict shaped like the real (format-2,
    multi-graph) cells — one entry is simply the N=1 case."""
    meta = aot_serve.artifact_metadata(
        family=FAMILY, precision="bf16", cell_key=key,
        entries={"unet/g": {
            "target": "unet", "fork": [], "class_dims": [],
            "inputs": [dict(r) for r in INPUTS], "symbols": {},
            "constants": [dict(r) for r in CONSTANTS], "graph": {},
        }},
        lora_bucket=bucket)
    meta["weight_lane"] = lane
    meta.update(over)
    return meta


def _flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_PREFER_AOT", "1")
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def test_flag_defaults_off() -> None:
    assert get_settings().compile_prefer_aot is False
    assert aot_cells.prefer_aot() is False


def test_flag_rides_typed_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _flag_on(monkeypatch)
    assert get_settings().compile_prefer_aot is True
    assert aot_cells.prefer_aot() is True


def test_flag_never_joins_the_env_seal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flip switch selects an artifact; it must not change cell identity.
    Flipping it must leave the env_seal digest byte-identical, or every
    published JIT cell strands fleet-wide (the runbook's rollback contract)."""
    from gen_worker import env_seal

    assert "compile_prefer_aot" not in env_seal.CANONICAL_CONFIG
    before = env_seal.seal_digest(env_seal.effective_seal())
    _flag_on(monkeypatch)
    assert env_seal.seal_digest(env_seal.effective_seal()) == before


# ---------------------------------------------------------------------------
# F1 — candidate filter
# ---------------------------------------------------------------------------


def test_candidates_filter_and_newest_wins(_runtime_key: None) -> None:
    good_old = _pilot_meta(_key("a"))
    good_new = _pilot_meta(_key("b"))
    wrong_kind = _pilot_meta(_key("c"), kind="torch-inductor-cache")
    wrong_lane = _pilot_meta(_key("d"), lane="w8a16")
    wrong_bucket = _pilot_meta(_key("e"), bucket=0)
    wrong_sm = _pilot_meta(_key("f"), sm="sm_80")
    unkeyed = _pilot_meta(_key("1"), cell_key="not-a-key")
    items = [
        {"checkpoint_id": "ck-old", "updated_at": "2026-07-27T00:00:00Z",
         "metadata": good_old},
        {"checkpoint_id": "ck-new", "updated_at": "2026-07-28T00:00:00Z",
         "metadata": good_new},
        {"checkpoint_id": "x1", "updated_at": "2026-07-29T00:00:00Z",
         "metadata": wrong_kind},
        {"checkpoint_id": "x2", "updated_at": "2026-07-29T00:00:00Z",
         "metadata": wrong_lane},
        {"checkpoint_id": "x3", "updated_at": "2026-07-29T00:00:00Z",
         "metadata": wrong_bucket},
        {"checkpoint_id": "x4", "updated_at": "2026-07-29T00:00:00Z",
         "metadata": wrong_sm},
        {"checkpoint_id": "x5", "updated_at": "2026-07-29T00:00:00Z",
         "metadata": unkeyed},
        {"checkpoint_id": "x6", "updated_at": "2026-07-29T00:00:00Z"},
    ]
    # pgw#765: a cross-SKU same-sm cell is a CANDIDATE (it sorts behind the
    # same-SKU rows, which is the preference, not a filter).
    other_sku = _pilot_meta(_key("9"), sku="a40")
    items.append({"checkpoint_id": "ck-a40",
                  "updated_at": "2026-07-29T00:00:00Z",
                  "metadata": other_sku})
    rows = aot_cells._candidates(items, FAMILY, "w8a8-lora64")
    assert [r[1] for r in rows] == ["ck-new", "ck-old", "ck-a40"]


# ---------------------------------------------------------------------------
# F1 — discovery end to end against a stub hub
# ---------------------------------------------------------------------------


class _StubHub:
    """Minimal catalog read surface: checkpoints listing + resolve + bytes."""

    def __init__(self, items: List[Dict[str, Any]],
                 artifacts: Dict[str, bytes], *,
                 refuse_bearer: bool = False,
                 fail_all: bool = False,
                 entry_for: Optional[Any] = None) -> None:
        self.items = items
        self.artifacts = artifacts  # checkpoint_id -> tarball bytes
        self.refuse_bearer = refuse_bearer
        self.fail_all = fail_all
        # (digest, blob, blob_url) -> the manifest file entry to serve. The
        # default is a manifest v1 entry; th#1303 tests hand in v2 shapes.
        self.entry_for = entry_for or _v2_entry
        self.requests: List[Tuple[str, bool]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler
                parsed = urlparse(self.path)
                has_auth = "Authorization" in self.headers
                outer.requests.append((parsed.path, has_auth))
                if outer.fail_all:
                    self.send_response(500)
                    self.end_headers()
                    return
                if outer.refuse_bearer and has_auth:
                    self.send_response(401)
                    self.end_headers()
                    return
                if parsed.path.endswith("/checkpoints"):
                    body = json.dumps({"items": outer.items}).encode()
                elif parsed.path.endswith("/resolve"):
                    digest = dict(
                        p.split("=", 1) for p in parsed.query.split("&") if "=" in p
                    ).get("digest", "")
                    blob = outer.artifacts.get(digest)
                    if blob is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = json.dumps({"files": [outer.entry_for(
                        digest, blob,
                        f"http://{self.headers['Host']}/blob/{digest}",
                    )]}).encode()
                elif parsed.path.startswith("/blob/"):
                    body = outer.artifacts.get(parsed.path[len("/blob/"):], b"")
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _blake3_bytes(blob: bytes) -> str:
    import blake3

    return str(blake3.blake3(blob).hexdigest())


def _sha256_bytes(blob: bytes) -> str:
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def _v1_entry(digest: str, blob: bytes, url: str) -> Dict[str, Any]:
    """A manifest v1 file entry: the bare-hex ``blake3`` mirror, no ``digest``.

    th#1303 S1 kept this fixture ONLY to prove such an entry is now REFUSED —
    nothing here may adopt from it. The default hub entry is ``_v2_entry``.
    """
    return {"path": f"{digest}.tar.gz", "size_bytes": len(blob),
            "blake3": _blake3_bytes(blob), "url": url}


def _v2_entry(digest: str, blob: bytes, url: str) -> Dict[str, Any]:
    """A manifest v2 file entry: algorithm-tagged ``digest``, NO ``blake3``."""
    return {"path": f"{digest}.tar.gz", "size_bytes": len(blob),
            "digest": "sha256:" + _sha256_bytes(blob), "url": url}


def _tarball(meta: Dict[str, Any], tmp_path: Path) -> bytes:
    content = tmp_path / f"content-{meta['cell_key'][:12]}"
    content.mkdir(exist_ok=True)
    (content / aot_serve.PACKAGE_NAME).write_bytes(b"fake-pt2")
    out = aot_serve.pack(
        content, tmp_path / f"{meta['cell_key']}.tar.gz", meta)
    return out.read_bytes()


class _FakePipe:
    pass


@dataclass
class _Cfg:
    family: str = FAMILY
    lora_bucket: int = 64


@pytest.fixture()
def _plain_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker import compile_cache as cc

    monkeypatch.setattr(cc, "cell_base_lane", lambda pipe: "w8a8")


def test_discover_downloads_newest_and_registers_key(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
) -> None:
    key = _key("9")
    meta = _pilot_meta(key)
    blob = _tarball(meta, tmp_path)
    hub = _StubHub(
        [{"checkpoint_id": "cafe01", "updated_at": "2026-07-28T00:00:00Z",
          "metadata": meta}],
        {"cafe01": blob})
    try:
        adopted = aot_cells.discover(
            _FakePipe(), _Cfg(),
            base_url=hub.base_url, worker_jwt=lambda: "jwt-token",
            cache_dir=tmp_path / "cache")
        assert adopted is not None
        assert adopted.cell_key == key
        assert adopted.ref == f"root/family-{FAMILY}#{key}"
        assert adopted.artifact.is_file()
        import hashlib
        assert sha256_file(adopted.artifact) == hashlib.sha256(blob).hexdigest()
        assert adopted.snapshot_digest == (
            "sha256:" + hashlib.sha256(blob).hexdigest())
        # The executor's kind dispatch (#734/#735) must now classify the
        # ck5-flavored ref as an exported cell.
        assert aot_serve.is_aot_ref(adopted.ref)
        # Re-discovery serves from the local cache (no second download).
        blob_requests = [p for p, _ in hub.requests if p.startswith("/blob/")]
        assert len(blob_requests) == 1
        again = aot_cells.discover(
            _FakePipe(), _Cfg(),
            base_url=hub.base_url, worker_jwt=lambda: "jwt-token",
            cache_dir=tmp_path / "cache")
        assert again is not None and again.artifact == adopted.artifact
        assert len([p for p, _ in hub.requests if p.startswith("/blob/")]) == 1
    finally:
        hub.close()


def test_discover_names_a_refusal_and_never_retries_anonymously(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
    monkeypatch: Any,
) -> None:
    """th#1335 (was: test_discover_retries_anonymously_on_401).

    The anonymous retry is retired. th#1310 made the family cell repos private
    and named the retry as the hazard that would turn one visibility flip into
    unauthenticated cross-tenant enumeration; the worker JWT now carries a
    hub-issued read grant instead. So a 401/403 must (a) NOT be re-issued
    without the bearer, and (b) surface as a NAMED refusal — th#1230's lesson:
    an authorization outcome the worker cannot tell apart from "this family has
    no cells" is silently fatal to the whole lane.
    """
    key = _key("8")
    meta = _pilot_meta(key)
    blob = _tarball(meta, tmp_path)
    hub = _StubHub(
        [{"checkpoint_id": "cafe02", "updated_at": "2026-07-28T00:00:00Z",
          "metadata": meta}],
        {"cafe02": blob}, refuse_bearer=True)
    seen: List[Tuple[str, str]] = []
    monkeypatch.setattr(
        aot_cells, "_emit",
        lambda phase, detail: seen.append((phase, detail)))
    try:
        adopted = aot_cells.discover(
            _FakePipe(), _Cfg(),
            base_url=hub.base_url, worker_jwt=lambda: "jwt-token",
            cache_dir=tmp_path / "cache")
        assert adopted is None
    finally:
        hub.close()
    phases = [p for p, _ in seen]
    assert "not_authorized" in phases, seen
    assert "list_failed" not in phases, (
        "an authz refusal must not be laundered into the generic listing "
        f"failure the worker treats as absence: {seen}")
    # Not one request went out without the credential.
    assert all(has_auth for _path, has_auth in hub.requests), hub.requests


def test_discover_never_raises(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
) -> None:
    hub = _StubHub([], {}, fail_all=True)
    try:
        assert aot_cells.discover(
            _FakePipe(), _Cfg(),
            base_url=hub.base_url, worker_jwt=lambda: "jwt",
            cache_dir=tmp_path) is None
    finally:
        hub.close()


def _discover_against(
    entry_for: Any, tmp_path: Path, *, key_seed: str,
    seen: Optional[List[Tuple[str, str]]] = None,
    monkeypatch: Optional[pytest.MonkeyPatch] = None,
) -> Tuple[Optional[aot_cells.AdoptedAotCell], Path]:
    """Run one discovery against a stub hub serving ``entry_for`` manifests.

    Returns the adoption (or None) and the path the artifact would cache at,
    so a test can assert about BYTES ON DISK — the only assertion that can
    tell a real verification from one that never ran.
    """
    key = _key(key_seed)
    meta = _pilot_meta(key)
    blob = _tarball(meta, tmp_path)
    cache = tmp_path / "cache"
    hub = _StubHub(
        [{"checkpoint_id": "cafe03", "updated_at": "2026-07-28T00:00:00Z",
          "metadata": meta}],
        {"cafe03": blob}, entry_for=entry_for)
    if seen is not None and monkeypatch is not None:
        monkeypatch.setattr(
            aot_cells, "_emit",
            lambda phase, detail: seen.append((phase, detail)))
    try:
        adopted = aot_cells.discover(
            _FakePipe(), _Cfg(),
            base_url=hub.base_url, worker_jwt=lambda: "jwt",
            cache_dir=cache)
    finally:
        hub.close()
    return adopted, cache / "aot-cells" / f"{key}.tar.gz"


def test_discover_refuses_a_v1_blake3_only_cell_th1303(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1303 S1 replaces `test_discover_digest_mismatch_is_a_miss`.

    A manifest v1 entry names its bytes only through the legacy ``blake3``
    mirror. With `resolved_entry_digest`'s mirror arm gone that entry carries
    NO digest, so the cell is refused before a byte is executed. Restoring the
    mirror arm makes this adopt again, which is the revert-turns-red.
    """
    seen: List[Tuple[str, str]] = []
    adopted, dest = _discover_against(
        _v1_entry, tmp_path, key_seed="7", seen=seen, monkeypatch=monkeypatch)
    assert adopted is None, "a blake3-only cell must never arm at S1"
    assert not dest.exists() and not dest.with_suffix(".part").exists()
    assert "resolve_failed" in [p for p, _ in seen], seen



def test_discover_refuses_a_v2_cell_whose_bytes_do_not_match_th1303(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1303 landmine 2 — the empty-guard class, on EXECUTABLE bytes.

    A manifest v2 entry carries ``digest: "sha256:…"`` and NO ``blake3`` key.
    The old code read ``entry["blake3"]``, got "", and both integrity guards
    (``if want_b3 and …``) evaporated — so a v2 cell artifact was written to
    the arming cache having been checked against nothing, and reported as a
    successful fetch. Assert about the bytes, not about "ok": a happy path
    returning a Path is exactly what the broken code also did.
    """
    def wrong_v2(digest: str, blob: bytes, url: str) -> Dict[str, Any]:
        ent = _v2_entry(digest, blob, url)
        ent["digest"] = "sha256:" + "0" * 64
        return ent

    seen: List[Tuple[str, str]] = []
    adopted, dest = _discover_against(
        wrong_v2, tmp_path, key_seed="6", seen=seen, monkeypatch=monkeypatch)
    assert adopted is None, "a v2 cell that hashes wrong must never arm"
    assert not dest.exists() and not dest.with_suffix(".part").exists()
    assert "digest_mismatch" in [p for p, _ in seen], seen


def test_discover_verifies_a_v2_cell_with_sha256_th1303(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
) -> None:
    """The green half: a correct v2 entry is verified UNDER SHA-256 and armed.

    Hashing a sha256 digest with blake3 is the mirror-image defect, so the
    honest v2 path has to be pinned alongside the refusal.
    """
    adopted, dest = _discover_against(_v2_entry, tmp_path, key_seed="5")
    assert adopted is not None
    assert adopted.artifact == dest and dest.is_file()


def test_discover_refuses_an_entry_with_no_digest_at_all_th1303(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No digest is a REFUSAL, never a skip. A cell artifact is compiled code
    the pod will execute; "the manifest didn't say" is not permission."""
    def no_digest(digest: str, blob: bytes, url: str) -> Dict[str, Any]:
        ent = _v2_entry(digest, blob, url)
        ent.pop("digest")
        return ent

    seen: List[Tuple[str, str]] = []
    adopted, dest = _discover_against(
        no_digest, tmp_path, key_seed="4", seen=seen, monkeypatch=monkeypatch)
    assert adopted is None
    assert not dest.exists()
    assert "resolve_failed" in [p for p, _ in seen], seen


def test_discover_refetches_a_cached_artifact_that_stopped_matching_th1303(
    tmp_path: Path, _runtime_key: None, _plain_lane: None,
) -> None:
    """The cache-hit guard is the SECOND site the empty digest disarmed.

    A corrupted (or substituted) cache file must be re-fetched, not served —
    under v2, where the old ``dest.is_file() and want_b3 and …`` short-circuit
    would have taken neither branch and fallen through to a re-download by
    accident rather than by rule.
    """
    key = _key("3")
    meta = _pilot_meta(key)
    blob = _tarball(meta, tmp_path)
    cache = tmp_path / "cache"
    dest = cache / "aot-cells" / f"{key}.tar.gz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not the cell you published")
    hub = _StubHub(
        [{"checkpoint_id": "cafe03", "updated_at": "2026-07-28T00:00:00Z",
          "metadata": meta}],
        {"cafe03": blob}, entry_for=_v2_entry)
    try:
        adopted = aot_cells.discover(
            _FakePipe(), _Cfg(), base_url=hub.base_url,
            worker_jwt=lambda: "jwt", cache_dir=cache)
        assert adopted is not None
        assert dest.read_bytes() == blob, "the poisoned cache file was served"
        assert [p for p, _ in hub.requests if p.startswith("/blob/")]
    finally:
        hub.close()


# ---------------------------------------------------------------------------
# F1 — the fleet arming hook
# ---------------------------------------------------------------------------


class _StubPublisher:
    base_url = "http://hub.invalid"

    def enabled(self) -> bool:
        return True

    def worker_jwt(self) -> str:
        return "jwt"


@pytest.fixture()
def _armable(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker import compile_cache as cc

    monkeypatch.setattr(cc, "has_compile_target", lambda pipe, cfg: True)


def test_flag_off_never_discovers(
    monkeypatch: pytest.MonkeyPatch, _armable: None,
) -> None:
    """RED: flag off => byte-identical arming — discovery must not run."""
    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("discovery ran with the flag off")

    monkeypatch.setattr(aot_cells, "discover", _forbidden)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: True)
    outcome = fleet_cells.enable_compiled(
        _FakePipe(), _Cfg(), publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is None


def test_flag_on_arms_discovered_cell_and_advertises_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _armable: None,
) -> None:
    _flag_on(monkeypatch)
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"artifact")
    adopted = aot_cells.AdoptedAotCell(
        family=FAMILY, cell_key=_key("5"),
        ref=f"root/family-{FAMILY}#{_key('5')}",
        snapshot_digest="blake3:" + "5" * 64, artifact=art)
    monkeypatch.setattr(
        aot_cells, "discover", lambda *a, **k: adopted)
    armed_with: List[Optional[Path]] = []

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any) -> bool:
        armed_with.append(artifact)
        # The real path wraps the module; is_armed then reads the marker.
        pipe._cozy_aot = {"state": {"failed": False}}
        return True

    monkeypatch.setattr(fleet_cells.provision, "enable_compiled", _enable)
    outcome = fleet_cells.enable_compiled(
        _FakePipe(), _Cfg(), publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed
    assert outcome.self_mint is adopted
    assert armed_with == [art]


def test_flag_on_failed_aot_arm_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _armable: None,
) -> None:
    """A discovered cell that refuses to arm must leave the boot on the
    ordinary policy with the originally delivered artifact."""
    _flag_on(monkeypatch)
    art = tmp_path / "cell.tar.gz"
    art.write_bytes(b"artifact")
    delivered = tmp_path / "delivered.tar.gz"
    delivered.write_bytes(b"dynamo")
    adopted = aot_cells.AdoptedAotCell(
        family=FAMILY, cell_key=_key("4"),
        ref=f"root/family-{FAMILY}#{_key('4')}",
        snapshot_digest="blake3:" + "4" * 64, artifact=art)
    monkeypatch.setattr(aot_cells, "discover", lambda *a, **k: adopted)
    calls: List[Optional[Path]] = []

    def _enable(pipe: Any, cfg: Any, cache_dir: Any, artifact: Any) -> bool:
        calls.append(artifact)
        return artifact == delivered

    monkeypatch.setattr(fleet_cells.provision, "enable_compiled", _enable)
    outcome = fleet_cells.enable_compiled(
        _FakePipe(), _Cfg(), artifact=delivered,
        publisher=_StubPublisher())  # type: ignore[arg-type]
    assert outcome.armed and outcome.self_mint is None
    assert calls == [art, delivered]


def test_flag_on_without_publisher_skips_discovery(
    monkeypatch: pytest.MonkeyPatch, _armable: None,
) -> None:
    _flag_on(monkeypatch)

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("discovery ran without a hub sink")

    monkeypatch.setattr(aot_cells, "discover", _forbidden)
    monkeypatch.setattr(
        fleet_cells.provision, "enable_compiled",
        lambda pipe, cfg, cache_dir, artifact: True)
    outcome = fleet_cells.enable_compiled(_FakePipe(), _Cfg(), publisher=None)
    assert outcome.armed and outcome.self_mint is None


# ---------------------------------------------------------------------------
# F2 — lifted-binding install at arm (torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from gen_worker.models import lora_lifted, provision, w8a8_lora  # noqa: E402

DIM = 16
BUCKET = 16  # smallest canonical rank bucket (w8a8_lora.RANK_BUCKETS)


class _Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.to_q = nn.Linear(DIM, DIM)
        self.to_v = nn.Linear(DIM, DIM)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.to_v(self.to_q(x))


class _Pipe:
    def __init__(self) -> None:
        self.unet = _Denoiser()


@dataclass
class _ArmCfg:
    family: str = FAMILY
    lora_bucket: int = BUCKET


def _arm_aot(
    monkeypatch: pytest.MonkeyPatch, pipe: _Pipe, *,
    enable_result: bool,
) -> List[bool]:
    """Drive provision.enable_compiled at an AOT artifact; returns the
    lifted-binding observations recorded at aot_serve.enable call time."""
    from gen_worker import compile_cache as cc
    from gen_worker import trt_engine

    monkeypatch.setattr(cc, "has_compile_target", lambda p, c: True)
    monkeypatch.setattr(cc, "enable", lambda *a, **k: False)
    monkeypatch.setattr(
        trt_engine, "unpack_metadata",
        lambda p: {"kind": aot_serve.ARTIFACT_KIND, "module": "unet"})
    observed: List[bool] = []

    def _enable(p: Any, c: Any, cache_dir: Any, artifact: Any) -> bool:
        observed.append(
            lora_lifted.lifted_binding(p.unet) is not None)
        return enable_result

    monkeypatch.setattr(aot_serve, "enable", _enable)
    provision.enable_compiled(
        pipe, _ArmCfg(), None, Path("/nonexistent/cell.tar.gz"))
    return observed


def test_f2_flag_off_never_installs_lifting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED: flag off => the module reaches aot_serve.enable UNLIFTED,
    exactly as 0.76.x ships (a lifted artifact then refuses by name)."""
    pipe = _Pipe()
    observed = _arm_aot(monkeypatch, pipe, enable_result=False)
    assert observed == [False]
    assert lora_lifted.lifted_binding(pipe.unet) is None


def test_f2_flag_on_installs_before_enable_in_proven_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flag_on(monkeypatch)
    pipe = _Pipe()
    observed = _arm_aot(monkeypatch, pipe, enable_result=True)
    # Installed BEFORE the arm (assert_lifted_contract sees a lifted module)
    assert observed == [True]
    binding = lora_lifted.lifted_binding(pipe.unet)
    assert binding is not None and binding.plan.bucket == BUCKET
    # ... and ON the canonical branch containers apply_lora_lane allocated.
    assert w8a8_lora.branch_bucket(pipe.unet) == BUCKET


def test_f2_failed_arm_rolls_the_install_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _flag_on(monkeypatch)
    pipe = _Pipe()
    observed = _arm_aot(monkeypatch, pipe, enable_result=False)
    assert observed == [True]
    # The dynamo fallthrough must never trace a lifted forward.
    assert lora_lifted.lifted_binding(pipe.unet) is None


# ---------------------------------------------------------------------------
# F3 — lifted-aware adapter attach (torch)
# ---------------------------------------------------------------------------

from gen_worker.utils.lora import AdapterResidency, PreparedAdapter  # noqa: E402


def _adapter(model: nn.Module, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    sd: Dict[str, Any] = {}
    for path, mod in w8a8_lora.branch_modules(model).items():
        sd[f"unet.{path}.lora_A.weight"] = (
            torch.randn(BUCKET, mod.in_features) * 0.05)
        sd[f"unet.{path}.lora_B.weight"] = (
            torch.randn(mod.out_features, BUCKET) * 0.05)
    return sd


def _prepared(sd: Dict[str, Any], ref: str) -> PreparedAdapter:
    return PreparedAdapter(
        slot="pipeline", ref=ref, cache_key=ref, name=f"a{abs(hash(ref)) % 997}",
        weight=1.0, state_dict=sd)


def _view_delta(binding: "lora_lifted.LiftedLoraBinding", path: str) -> Any:
    a, b = binding.tensors
    for slot_path, av, bv in binding.plan.views(a, b):
        if slot_path == path:
            return bv.detach() @ av.detach()
    raise AssertionError(f"no slot for {path}")


def test_f3_flag_off_attach_is_the_buffer_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED: no binding => today's buffer-copy path, byte-identical."""
    pipe = _Pipe()
    from gen_worker import compile_cache as cc

    cc.apply_lora_lane(pipe, BUCKET)
    sd = _adapter(pipe.unet, seed=3)
    AdapterResidency().activate(
        "ref", pipe, [_prepared(sd, "hub/adapter@1")], request_id="off")
    mod = pipe.unet.to_q
    delta = mod.lora_b.detach() @ mod.lora_a.detach()
    want = sd["unet.to_q.lora_B.weight"] @ sd["unet.to_q.lora_A.weight"]
    assert torch.allclose(delta, want, atol=1e-6)


def test_f3_lifted_attach_writes_through_the_binding_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _Pipe()
    from gen_worker import compile_cache as cc

    cc.apply_lora_lane(pipe, BUCKET)
    binding = lora_lifted.install_lifted_lora_forward(pipe.unet, BUCKET)
    sd = _adapter(pipe.unet, seed=4)
    residency = AdapterResidency()
    residency.activate(
        "ref", pipe, [_prepared(sd, "hub/adapter@2")], request_id="on")

    # The ARTIFACT-visible flat pair carries the adapter ...
    delta = _view_delta(binding, "to_q")
    want = sd["unet.to_q.lora_B.weight"] @ sd["unet.to_q.lora_A.weight"]
    assert torch.allclose(delta, want, atol=1e-6)
    # ... and the orphaned canonical buffers stay zero — a buffer copy here
    # is exactly the "adapter change the artifact never sees" defect.
    mod = pipe.unet.to_q
    assert torch.count_nonzero(mod.lora_b.detach()) == 0

    # Deactivation zeroes B in the flat pair (zero addend, same graph).
    residency.deactivate("ref", pipe, request_id="off")
    a, b = binding.tensors
    assert torch.count_nonzero(b) == 0
