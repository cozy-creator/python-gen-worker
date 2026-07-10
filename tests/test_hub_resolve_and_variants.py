"""#379 client-side Hub resolve + #380 variant auto-selection.

#379: ``_resolve_local_path`` on a tensorhub ref resolves against a REAL local
HTTP server speaking tensorhub's public resolve route (th#560) and downloads
blobs into the blake3 CAS via the shared cozy_snapshot path — no mocks on the
unit under test, only a stdlib HTTP server standing in for the hub + R2.

#380: ``select_variant`` policy (SM/library gates, VRAM ladder, base
fallback), listing fit verdicts, and ``--variant auto`` selection.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import pytest
from blake3 import blake3

import gen_worker.cli.run as run_mod
from gen_worker import RequestContext, endpoint
from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.models.hub_client import (
    HubRepoNotFoundError,
    HubResolveError,
    resolve_repo,
)
from gen_worker.models.hub_policy import (
    TensorhubWorkerCapabilities,
    select_variant,
    variant_fit,
)
from gen_worker.models.refs import parse_model_ref


# ---------------------------------------------------------------------------
# Local hub double: resolve route + blob GETs over real HTTP
# ---------------------------------------------------------------------------

class _HubState:
    def __init__(self) -> None:
        self.blobs: Dict[str, bytes] = {}
        self.snapshot_digest = ""
        self.manifest: List[Dict[str, Any]] = []
        self.resolves = 0
        self.auth_headers: List[str] = []
        self.status = 200
        self.blob_status: Dict[str, int] = {}  # digest -> forced status once


def _make_handler(state: _HubState, base_url_holder: List[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/v1/repos/") and "/resolve" in self.path:
                state.resolves += 1
                state.auth_headers.append(self.headers.get("Authorization") or "")
                if state.status != 200:
                    self.send_response(state.status)
                    self.end_headers()
                    return
                files = []
                for ent in state.manifest:
                    e = dict(ent)
                    e["url"] = f"{base_url_holder[0]}/blobs/{e['blake3']}"
                    files.append(e)
                body = json.dumps({
                    "tenant": "root", "name": "tiny", "tag": "latest",
                    "snapshot_digest": state.snapshot_digest,
                    "files": files,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/blobs/"):
                digest = self.path.rsplit("/", 1)[-1]
                forced = state.blob_status.pop(digest, 0)
                if forced:
                    self.send_response(forced)
                    self.end_headers()
                    return
                blob = state.blobs.get(digest)
                if blob is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


@pytest.fixture()
def local_hub():
    state = _HubState()
    holder: List[str] = [""]
    srv = HTTPServer(("127.0.0.1", 0), _make_handler(state, holder))
    holder[0] = f"http://127.0.0.1:{srv.server_port}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield holder[0], state
    finally:
        srv.shutdown()
        srv.server_close()


def _seed(state: _HubState, files: Dict[str, bytes]) -> None:
    manifest = []
    for path, data in files.items():
        d = blake3(data).hexdigest()
        state.blobs[d] = data
        manifest.append({"path": path, "size_bytes": len(data), "blake3": d})
    state.manifest = manifest
    state.snapshot_digest = blake3(b"".join(sorted(files.values()))).hexdigest()


def _events() -> Tuple[List[Dict[str, Any]], Any]:
    seen: List[Dict[str, Any]] = []
    return seen, seen.append


# ---------------------------------------------------------------------------
# #379 resolve → download → CAS
# ---------------------------------------------------------------------------

def test_hub_ref_resolves_and_lands_in_cas(local_hub, monkeypatch, tmp_path):
    base, state = local_hub
    _seed(state, {"unet/model.safetensors": b"weights-bytes", "config.json": b"{}"})
    monkeypatch.setenv("TENSORHUB_URL", base)
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    monkeypatch.delenv("TENSORHUB_TOKEN", raising=False)

    seen, emit = _events()
    local = run_mod._resolve_local_path(
        ref="root/tiny", provider="tensorhub", offline=False, emit=emit,
    )
    assert local.endswith(state.snapshot_digest)
    snap = tmp_path / "snapshots" / state.snapshot_digest
    assert (snap / "unet" / "model.safetensors").read_bytes() == b"weights-bytes"
    assert (snap / "config.json").read_bytes() == b"{}"
    # Blobs stored content-addressed.
    d = blake3(b"weights-bytes").hexdigest()
    assert (tmp_path / "blobs" / "blake3" / d[:2] / d[2:4] / d).exists()
    # Anonymous pull: no Authorization header was sent.
    assert state.auth_headers == [""]
    kinds = [e.get("kind") for e in seen]
    assert "model_fetch.started" in kinds and "model_fetch.completed" in kinds
    assert all(e.get("provider") == "tensorhub" for e in seen)

    # Second resolve is a no-download cache hit (snapshot dir short-circuits).
    local2 = run_mod._resolve_local_path(
        ref="root/tiny", provider="tensorhub", offline=False, emit=lambda e: None,
    )
    assert local2 == local


def test_hub_token_sent_as_bearer(local_hub, monkeypatch, tmp_path):
    base, state = local_hub
    _seed(state, {"a.bin": b"aa"})
    monkeypatch.setenv("TENSORHUB_URL", base)
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    monkeypatch.setenv("TENSORHUB_TOKEN", "oat_secret")
    run_mod._resolve_local_path(
        ref="root/tiny:latest", provider="tensorhub", offline=False, emit=lambda e: None,
    )
    assert state.auth_headers == ["Bearer oat_secret"]


def test_hub_404_is_typed_not_found(local_hub, monkeypatch, tmp_path):
    base, state = local_hub
    state.status = 404
    monkeypatch.setenv("TENSORHUB_URL", base)
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    ref = parse_model_ref("root/ghost", provider="tensorhub").tensorhub
    with pytest.raises(HubRepoNotFoundError, match="not found"):
        resolve_repo(ref, base_url=base)
    with pytest.raises(run_mod._ModelResolutionError, match="not found"):
        run_mod._resolve_local_path(
            ref="root/ghost", provider="tensorhub", offline=False, emit=lambda e: None,
        )


def test_hub_offline_is_cas_only(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    monkeypatch.setenv("TENSORHUB_URL", "http://127.0.0.1:9")  # must not be dialed
    with pytest.raises(run_mod._ModelResolutionError, match="--offline"):
        run_mod._resolve_local_path(
            ref="root/tiny", provider="tensorhub", offline=True, emit=lambda e: None,
        )
    # Digest-pinned ref whose snapshot IS pre-seeded works offline.
    snap = tmp_path / "snapshots" / "abcd1234"
    snap.mkdir(parents=True)
    local = run_mod._resolve_local_path(
        ref="root/tiny@blake3:abcd1234", provider="tensorhub", offline=True,
        emit=lambda e: None,
    )
    assert local == str(snap)


def test_hub_offline_reuses_remembered_tag_ref(local_hub, monkeypatch, tmp_path):
    base, state = local_hub
    _seed(state, {"w.bin": b"tag-ref-weights"})
    monkeypatch.setenv("TENSORHUB_URL", base)
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    online = run_mod._resolve_local_path(
        ref="root/tiny:latest", provider="tensorhub", offline=False, emit=lambda e: None,
    )
    # Now fully offline (hub unreachable): the tag->digest memory serves it.
    monkeypatch.setenv("TENSORHUB_URL", "http://127.0.0.1:9")
    offline = run_mod._resolve_local_path(
        ref="root/tiny:latest", provider="tensorhub", offline=True, emit=lambda e: None,
    )
    assert offline == online
    # A never-fetched tag still misses with the typed error.
    with pytest.raises(run_mod._ModelResolutionError, match="--offline"):
        run_mod._resolve_local_path(
            ref="root/tiny:other", provider="tensorhub", offline=True, emit=lambda e: None,
        )


def test_hub_no_base_url_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))
    monkeypatch.delenv("TENSORHUB_URL", raising=False)
    with pytest.raises(run_mod._ModelResolutionError, match="TENSORHUB_URL"):
        run_mod._resolve_local_path(
            ref="root/tiny", provider="tensorhub", offline=False, emit=lambda e: None,
        )


def test_url_expired_triggers_one_reresolve(local_hub, monkeypatch, tmp_path):
    base, state = local_hub
    _seed(state, {"model.bin": b"expiring-blob"})
    d = blake3(b"expiring-blob").hexdigest()
    state.blob_status[d] = 403  # first GET: presign "expired"
    monkeypatch.setenv("TENSORHUB_URL", base)
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path))

    seen, emit = _events()
    local = run_mod._resolve_local_path(
        ref="root/tiny", provider="tensorhub", offline=False, emit=emit,
    )
    assert state.resolves == 2  # initial + one re-resolve
    assert (tmp_path / "snapshots" / state.snapshot_digest / "model.bin").exists()
    assert local.endswith(state.snapshot_digest)
    assert any(e.get("kind") == "model_fetch.reresolve" for e in seen)


def test_resolve_repo_flavor_and_digest_params(local_hub, monkeypatch, tmp_path):
    base, state = local_hub
    _seed(state, {"m.bin": b"fp8"})
    ref = parse_model_ref("root/tiny#fp8", provider="tensorhub").tensorhub
    assert ref.flavor == "fp8"
    out = resolve_repo(ref, base_url=base)
    assert out.snapshot_digest == state.snapshot_digest
    assert out.files[0].url.startswith(base)


# ---------------------------------------------------------------------------
# #380 select_variant policy
# ---------------------------------------------------------------------------

_CAPS_4070 = TensorhubWorkerCapabilities(
    cuda_version="12.8", gpu_sm=89, torch_version="2.11", installed_libs=["torchao"],
)
_CAPS_B200 = TensorhubWorkerCapabilities(
    cuda_version="12.8", gpu_sm=100, torch_version="2.11", installed_libs=["torchao"],
)
_CAPS_CPU = TensorhubWorkerCapabilities(
    cuda_version="", gpu_sm=0, torch_version="", installed_libs=[],
)

_FLUX_VARIANTS = [
    ("flux-bf16", Resources(vram_gb=24)),
    ("flux-fp8", Resources(vram_gb=14)),
    ("flux-nvfp4", Resources(vram_gb=12, compute_capability=10.0)),
]


def test_sm_gate_rejects_nvfp4_on_sm89():
    fit, reason = variant_fit(Resources(vram_gb=12, compute_capability=10.0), _CAPS_4070, 8.0)
    assert fit == "incompatible" and "compute capability" in reason


def test_library_gate():
    fit, reason = variant_fit(Resources(vram_gb=4, libraries=("transformer_engine",)), _CAPS_4070, 8.0)
    assert fit == "incompatible" and "transformer_engine" in reason


def test_vram_ladder_picks_largest_fitting():
    # 16 GB free on SM89: bf16 (24) offloads, fp8 (14) fits, nvfp4 SM-gated.
    choice = select_variant(_FLUX_VARIANTS, _CAPS_4070, 16.0)
    assert choice is not None and choice.name == "flux-fp8" and choice.fit == "fits"
    # 30 GB free: prefer the largest precision that fits.
    choice = select_variant(_FLUX_VARIANTS, _CAPS_4070, 30.0)
    assert choice.name == "flux-bf16"
    # SM100 with 12.5 GB free: nvfp4 (12 GB card rec) unlocked and the only
    # fit — fp8's 14 GB-card recommendation implies a >=13 GB free floor.
    choice = select_variant(_FLUX_VARIANTS, _CAPS_B200, 12.5)
    assert choice.name == "flux-nvfp4"


def test_base_fallback_below_every_floor():
    # 8 GB free: nothing fits resident, but fp8's runtime fp8-storage
    # estimate (14 * 0.55 = 7.7) does -> the automatic runtime rungs
    # (gw#420 / th#683) outrank offload.
    choice = select_variant(
        _FLUX_VARIANTS, _CAPS_4070, 8.0, base=("flux", Resources(vram_gb=24)),
    )
    assert choice.name == "flux-fp8" and choice.fit == "emergency_fp8"
    # 5 GB free: even 4-bit estimates too big -> base binding + offload ladder.
    choice = select_variant(
        _FLUX_VARIANTS, _CAPS_4070, 5.0, base=("flux", Resources(vram_gb=24)),
    )
    assert choice.name == "flux" and choice.fit == "offload"
    # No base declared -> smallest compatible variant, offloaded.
    choice = select_variant(_FLUX_VARIANTS, _CAPS_4070, 5.0)
    assert choice.name == "flux-fp8" and choice.fit == "offload"


def test_cpu_box_has_no_routable_gpu_variant():
    assert select_variant(_FLUX_VARIANTS, _CAPS_CPU, 0.0) is None
    fit, _ = variant_fit(Resources(), _CAPS_CPU, 0.0)
    assert fit == "fits"  # CPU-only endpoints still fit


# ---------------------------------------------------------------------------
# #380 CLI surfaces: listing fit verdicts + --variant auto selection
# ---------------------------------------------------------------------------

class _VIn(msgspec.Struct):
    prompt: str = ""


class _VOut(msgspec.Struct):
    ok: bool


def _variant_module():
    import types

    mod = types.ModuleType("_test_variants")

    @endpoint(
        kind="inference",
        resources=Resources(vram_gb=24),
        models={"pipeline": HF("org/base")},
        variants={
            "gen-fp8": (HF("org/base-fp8"), Resources(vram_gb=14)),
            "gen-nvfp4": (HF("org/base-nvfp4"), Resources(vram_gb=12, compute_capability=10.0)),
        },
    )
    class Gen:
        def setup(self, pipeline: object) -> None:
            pass

        def generate(self, ctx: RequestContext, payload: _VIn) -> _VOut:
            return _VOut(ok=True)

    mod.Gen = Gen
    return mod


def _patched_hw(monkeypatch, *, sm: int, free_gb: float, libs=()):  # helpers
    import gen_worker.models.hub_policy as hp
    import gen_worker.models.memory as mem

    caps = TensorhubWorkerCapabilities(
        cuda_version="12.8" if sm else "", gpu_sm=sm,
        torch_version="2.11", installed_libs=list(libs),
    )
    monkeypatch.setattr(hp, "detect_worker_capabilities", lambda **_k: caps)
    monkeypatch.setattr(mem, "get_available_vram_gb", lambda *a, **k: free_gb)


def test_listing_carries_fit_and_variant_of(monkeypatch):
    from gen_worker.cli.listing import build_description

    _patched_hw(monkeypatch, sm=89, free_gb=16.0)
    mod = _variant_module()
    candidates = run_mod.discover_candidates(mod)
    doc = build_description(main_module="_test_variants", candidates=candidates)

    assert doc["detected"]["gpu_sm"] == 89
    assert doc["detected"]["free_vram_gb"] == 16.0
    by_name = {f["name"]: f for f in doc["functions"]}
    assert set(by_name) == {"generate", "gen-fp8", "gen-nvfp4"}
    # 24 GB > 16 free, but 24 * 0.55 fits -> automatic fp8-storage rung (th#683)
    assert by_name["generate"]["fit"] == "emergency_fp8"
    assert "variant_of" not in by_name["generate"]
    assert by_name["gen-fp8"]["fit"] == "fits"
    assert by_name["gen-fp8"]["variant_of"] == "generate"
    assert by_name["gen-nvfp4"]["fit"] == "incompatible"
    assert by_name["gen-nvfp4"]["resources"]["vram_gb"] == 12.0


def test_variant_auto_picks_fp8_on_sm89(monkeypatch):
    _patched_hw(monkeypatch, sm=89, free_gb=16.0)
    mod = _variant_module()
    candidates = run_mod.discover_candidates(mod)
    picked = run_mod.select_function_with_variant(
        candidates, cls_name=None, method_name="generate",
        default_name=None, variant="auto",
    )
    assert picked.fn_name == "gen-fp8"


def test_variant_auto_base_fallback_below_floor(monkeypatch):
    # 8 GB free: gen-fp8's 4-bit estimate fits -> emergency rung (gw#420).
    _patched_hw(monkeypatch, sm=89, free_gb=8.0)
    mod = _variant_module()
    candidates = run_mod.discover_candidates(mod)
    picked = run_mod.select_function_with_variant(
        candidates, cls_name=None, method_name="generate",
        default_name=None, variant="auto",
    )
    assert picked.fn_name == "gen-fp8"
    # 5 GB free: no 4-bit estimate fits -> base + offload ladder.
    _patched_hw(monkeypatch, sm=89, free_gb=5.0)
    picked = run_mod.select_function_with_variant(
        candidates, cls_name=None, method_name="generate",
        default_name=None, variant="auto",
    )
    assert picked.fn_name == "generate"


def test_variant_by_name_and_unknown(monkeypatch):
    mod = _variant_module()
    candidates = run_mod.discover_candidates(mod)
    picked = run_mod.select_function_with_variant(
        candidates, cls_name=None, method_name="generate",
        default_name=None, variant="gen-fp8",
    )
    assert picked.fn_name == "gen-fp8"
    with pytest.raises(run_mod._UsageError, match="available"):
        run_mod.select_function_with_variant(
            candidates, cls_name=None, method_name="generate",
            default_name=None, variant="nope",
        )
