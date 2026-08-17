"""Media transport and output integrity: one put table, one envelope rule.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.parse import unquote

import msgspec
import numpy as np
import pytest
from blake3 import blake3
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.media_bytes_th2082 import MediaIn, MediaOut
from harness.toy_endpoints import EchoIn
from harness.upload_sink import (
    MEDIA_UPLOADS_PATH,
    DedupUploadSink,
    reset_upload_sink,
    serve_upload_sink,
)

from gen_worker import content_credentials as cc
from gen_worker import io as gw_io
from gen_worker import output_integrity as oi
from gen_worker.api.decorators import Compile
from gen_worker.api.errors import (
    ArtifactTransferError,
    BlobDigestMalformedError,
    OutputIntegrityError,
    PayloadRefError,
    ValidationError,
)
from gen_worker.api.export_contract import DeclarationError, Dim, GraphClass, Input
from gen_worker.executor import _map_exception
from gen_worker.hubio import transport
from gen_worker.hubio.transport import (
    PUT_EXPIRED,
    PUT_OK,
    PUT_TERMINAL,
    PUT_TRANSIENT,
    put_verdict,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.request_context import (
    REF_ORIGIN_PLATFORM,
    JobContext,
)

# ============================================================================
# pgw#1206 — B — the One-transport contract.
# ============================================================================

def test_put_verdict_is_the_one_table() -> None:
    assert put_verdict(200) == PUT_OK
    assert put_verdict(204) == PUT_OK
    # 403 past expires_at is a re-plan, never a repudiation.
    assert put_verdict(403, presign_expired=True) == PUT_EXPIRED
    assert put_verdict(403) == PUT_TERMINAL
    assert put_verdict(400) == PUT_TERMINAL
    assert put_verdict(404) == PUT_TERMINAL
    for status in (408, 429, 500, 503):
        assert put_verdict(status) == PUT_TRANSIENT, status


def test_media_put_path_projects_from_it() -> None:
    """pgw#1206: Worker-owned media transport keeps one status table."""
    import inspect

    assert "put_verdict" in inspect.getsource(transport._classify_response_status)
    # And the projections agree on the transient set (408 included — the
    # engine's old private table called it terminal).
    err = transport._classify_response_status(408, "")
    assert err is not None and err.retryable


def test_the_credentialed_lane_stays_dead() -> None:
    """pgw#1206: The hub has no transfer_grant producers; the worker holds no store credentials."""
    with pytest.raises(ModuleNotFoundError):
        import gen_worker.s3_transfer  # noqa: F401

    repo = Path(__file__).resolve().parents[1]
    pyproject = (repo / "pyproject.toml").read_text()
    assert "boto3" not in pyproject
    hits = [
        p for p in (repo / "src" / "gen_worker").rglob("*.py")
        if "import boto3" in p.read_text() or "transfer_grant" in p.read_text()
    ]
    assert hits == [], f"credentialed-lane residue: {hits}"


def _put_403(phase: str = "put", status: int = 403) -> ArtifactTransferError:
    return ArtifactTransferError(
        "S3 part upload terminal status (403)", provider="tensorhub",
        phase=phase, retryable=False, status_code=status,
    )


def test_a_403_part_put_replans_the_session_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """pgw#1206: RED against 1b282a82: the media path failed terminally on the first 403'd part PUT."""
    from gen_worker import presigned_upload as pu

    calls: list[int] = []

    def fake_scoped(**kw: object) -> pu.PresignedUploadResult:
        calls.append(1)
        if len(calls) == 1:
            raise _put_403()
        return pu.PresignedUploadResult(meta={"ok": True})

    monkeypatch.setattr(pu, "_presigned_upload_file_scoped", fake_scoped)
    monkeypatch.setattr(pu, "control_plane_session", lambda base: (object(), True))
    f = tmp_path / "a.bin"; f.write_bytes(b"x")
    result = pu.presigned_upload_file(
        file_path=f, base_url="http://hub.invalid", endpoint_path="/api/v1/media/uploads",
        headers={}, create_payload={}, blake3_hex="00", size_bytes=1,
    )
    assert result.meta == {"ok": True}
    assert len(calls) == 2, "exactly one re-plan"


def test_a_403_on_the_fresh_presigns_is_terminal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gen_worker import presigned_upload as pu

    calls: list[int] = []

    def fake_scoped(**kw: object) -> pu.PresignedUploadResult:
        calls.append(1)
        raise _put_403()

    monkeypatch.setattr(pu, "_presigned_upload_file_scoped", fake_scoped)
    monkeypatch.setattr(pu, "control_plane_session", lambda base: (object(), True))
    f = tmp_path / "a.bin"; f.write_bytes(b"x")
    with pytest.raises(ArtifactTransferError):
        pu.presigned_upload_file(
            file_path=f, base_url="http://hub.invalid", endpoint_path="/api/v1/media/uploads",
            headers={}, create_payload={}, blake3_hex="00", size_bytes=1,
        )
    assert len(calls) == 2, "one re-plan, then terminal — never a loop"


def test_non_403_failures_do_not_replan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """pgw#1206: The re-plan is FOR expired presigns; a create-phase failure or a 400 keeps its one-shot semanti..."""
    from gen_worker import presigned_upload as pu

    calls: list[int] = []

    def fake_scoped(**kw: object) -> pu.PresignedUploadResult:
        calls.append(1)
        raise _put_403(phase="create", status=500)

    monkeypatch.setattr(pu, "_presigned_upload_file_scoped", fake_scoped)
    monkeypatch.setattr(pu, "control_plane_session", lambda base: (object(), True))
    f = tmp_path / "a.bin"; f.write_bytes(b"x")
    with pytest.raises(ArtifactTransferError):
        pu.presigned_upload_file(
            file_path=f, base_url="http://hub.invalid", endpoint_path="/api/v1/media/uploads",
            headers={}, create_payload={}, blake3_hex="00", size_bytes=1,
        )
    assert len(calls) == 1


# ============================================================================
# pgw#767 — MEDIA_BYTES_INLINE must not decide whether the result
#   envelope's blob actually exists.
# ============================================================================

def test_inline_dispatch_over_the_envelope_ceiling_still_really_uploads() -> None:
    """pgw#767: RED before the fix: the sink is never hit and the returned blob_ref names nothing."""
    org_id = "00000000-0000-0000-0000-000000000001"
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(file_base_url=base_url) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id="r-inline-big", attempt=1, function_name="large-usage",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
                org=org_id, capability_token="cap-token",
                media_bytes=pb.MEDIA_BYTES_INLINE,
            ))
            res = conn.wait_for(is_result_for("r-inline-big")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert res.blob_ref, "over the envelope ceiling the result ships a ref"
            assert not res.inline
            assert DedupUploadSink.requests_seen, (
                "a returned blob_ref must name a blob that was REALLY uploaded — "
                "the inline media hint must not reach the result envelope"
            )
            path, body = DedupUploadSink.requests_seen[-1]
            assert path == MEDIA_UPLOADS_PATH
            assert body["size_bytes"] > 64 * 1024
    finally:
        httpd.shutdown()
        reset_upload_sink()


def test_save_bytes_still_inlines_media_for_the_client() -> None:
    """pgw#767: The fix is scoped to the envelope: the public media path keeps the `Prefer: bytes=inline` shortc..."""
    from gen_worker import RequestContext

    ctx: Any = RequestContext(
        request_id="req-inline", owner="org",
        execution_hints={"output_format": "inline"},
    )
    asset = ctx.save_bytes("samples/small.bin", b"payload")
    assert asset.inline_bytes == b"payload"


def test_result_envelope_helper_ignores_the_inline_hint() -> None:
    """pgw#767: The envelope helper never takes the shortcut: it either uploads or refuses, but it must not hand..."""
    from gen_worker import RequestContext

    ctx: Any = RequestContext(
        request_id="req-envelope", owner="org",
        execution_hints={"output_format": "inline"},
    )
    assert hasattr(ctx, "_save_result_envelope")
    try:
        asset = ctx._save_result_envelope("results/req-envelope.msgpack", b"payload")
    except Exception:
        return  # refused with no upload endpoint configured — correct
    assert not asset.inline_bytes


# ============================================================================
# th#1307 — no C2PA private key ever lives in a pod; the hub signs.
# ============================================================================

@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    d = tmp_path_factory.mktemp("c2pa-1307")
    run = lambda *a: subprocess.run(a, check=True, capture_output=True)  # noqa: E731
    run("openssl", "ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", str(d / "raw.pem"))
    run("openssl", "pkcs8", "-topk8", "-nocrypt", "-in", str(d / "raw.pem"), "-out", str(d / "key.pem"))
    run("openssl", "req", "-x509", "-new", "-key", str(d / "key.pem"),
        "-subj", "/O=Cozy Test/CN=cozy signer", "-days", "365", "-sha256",
        "-addext", "basicConstraints=CA:FALSE",
        "-addext", "keyUsage=critical,digitalSignature",
        "-addext", "extendedKeyUsage=emailProtection", "-out", str(d / "cert.pem"))
    return {"cert": d / "cert.pem", "key": d / "key.pem"}


def _reset(monkeypatch):
    monkeypatch.setattr(cc, "_configured", False)
    monkeypatch.setattr(cc, "_config", None)
    monkeypatch.setattr(cc, "_remote", None)


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (7, 90, 210)).save(buf, format="PNG")
    return buf.getvalue()


def _big_png() -> bytes:
    """~1 MiB of incompressible noise: a stand-in for a real generated asset."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1307)
    arr = rng.integers(0, 256, size=(700, 700, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _configure_cert_only(chain: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch)
    cc.configure(SimpleNamespace(  # type: ignore[arg-type]
        c2pa_cert_pem=chain["cert"].read_text(), c2pa_cert_path="",
        c2pa_alg="es256", c2pa_ta_url="",
    ))


class _HubSigner:
    """Minimal stand-in for POST /v1/worker/c2pa/sign (see the Go route)."""

    def __init__(self, key_pem: bytes, status: int = 200):
        from cryptography.hazmat.primitives import serialization

        self.key: Any = serialization.load_pem_private_key(key_pem, password=None)
        self.status = status
        self.claim_sizes: list[int] = []
        self.tokens: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import ec
                from cryptography.hazmat.primitives.asymmetric import utils as au

                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.tokens.append((self.headers.get("Authorization") or "").removeprefix("Bearer "))
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    return
                claim = base64.b64decode(body["claim_b64"])
                outer.claim_sizes.append(len(claim))
                der = outer.key.sign(claim, ec.ECDSA(hashes.SHA256()))
                r, s = au.decode_dss_signature(der)
                sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
                out = json.dumps({"alg": "es256",
                                  "signature_b64": base64.b64encode(sig).decode()}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *_a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


@pytest.mark.parametrize("env_name", ["GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH"])
def test_private_key_in_pod_env_is_refused(chain, monkeypatch, env_name):
    """th#1307: A hub regression that re-injects the key must kill the pod, loudly."""
    _reset(monkeypatch)
    monkeypatch.setenv(env_name, chain["key"].read_text())
    with pytest.raises(cc.C2paSigningError) as e:
        cc.configure(SimpleNamespace(  # type: ignore[arg-type]
            c2pa_cert_pem=chain["cert"].read_text(), c2pa_cert_path="",
            c2pa_alg="es256", c2pa_ta_url="",
        ))
    assert env_name in str(e.value)
    # And the refusal is not swallowed by the lazy path either: a caller that
    # merely asks "is signing on?" gets the same hard error, never a quiet
    # False that would ship media unsigned.
    with pytest.raises(cc.C2paSigningError):
        cc.enabled()
    _reset(monkeypatch)


@pytest.mark.parametrize("env_name", ["GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH"])
def test_key_bearing_pod_is_refused_even_if_configure_was_never_called(chain, monkeypatch, env_name):
    """th#1307: The refusal belongs to the POD, not to one entry point."""
    _reset(monkeypatch)
    monkeypatch.setenv(env_name, chain["key"].read_text())

    for call in (
        lambda: cc.enabled(),
        lambda: cc.sign_media_bytes(_png(), ref="outputs/a.png"),
        lambda: cc.sign_media_file("/nonexistent.png", ref="outputs/a.png"),
    ):
        with pytest.raises(cc.C2paSigningError) as e:
            call()
        assert env_name in str(e.value)
    _reset(monkeypatch)


def test_key_delivered_after_a_clean_configure_is_still_refused(chain, monkeypatch):
    """th#1307: Memoisation must not outrank the ratchet."""
    _configure_cert_only(chain, monkeypatch)
    try:
        assert cc.enabled()
        monkeypatch.setenv("GEN_WORKER_C2PA_KEY_PEM", chain["key"].read_text())
        with pytest.raises(cc.C2paSigningError) as e:
            cc.enabled()
        assert "GEN_WORKER_C2PA_KEY_PEM" in str(e.value)
    finally:
        _reset(monkeypatch)


def test_settings_have_no_private_key_field_at_all():
    """The ratchet in the config layer: nothing can carry a key into Settings."""
    from gen_worker.config.loader import _ENV_TO_FIELD
    from gen_worker.config.settings import Settings

    for env_name in ("GEN_WORKER_C2PA_KEY_PEM", "GEN_WORKER_C2PA_KEY_PATH"):
        assert env_name not in _ENV_TO_FIELD, f"{env_name} must not map into Settings (th#1307)"
    fields = set(getattr(Settings, "__dataclass_fields__", {}) or Settings.__annotations__)
    assert "c2pa_key_pem" not in fields
    assert "c2pa_key_path" not in fields
    # The PUBLIC half is still configurable.
    assert "c2pa_cert_pem" in fields


def test_hub_signs_and_media_verifies(chain, monkeypatch):
    import c2pa

    hub = _HubSigner(chain["key"].read_bytes())
    try:
        _configure_cert_only(chain, monkeypatch)
        cc.configure_remote_signer(hub.base_url, lambda: "worker-jwt-abc")
        raw = _big_png()
        signed = cc.sign_media_bytes(raw, ref="outputs/a.png", request_id="req-1307")
        assert signed != raw
        with c2pa.Reader("image/png", io.BytesIO(signed)) as reader:
            assert str(reader.get_validation_state()) in ("Valid", "Trusted")

        assert hub.claim_sizes, "the hub was never asked to sign"
        assert hub.tokens == ["worker-jwt-abc"], "the worker JWT must authenticate the oracle"
        # The media never transits the hub: a claim is hashes + assertions, so
        # its size is bounded and INDEPENDENT of the asset (here ~1 KiB of
        # claim for a ~1 MiB asset). This is why hub-side signing does not
        # drag multi-GB video through the control plane.
        assert max(hub.claim_sizes) < 8 * 1024, f"claim is {max(hub.claim_sizes)} bytes"
        assert max(hub.claim_sizes) * 20 < len(raw), (
            f"claim bytes {max(hub.claim_sizes)} vs media {len(raw)} — only the claim may travel"
        )
    finally:
        hub.close()
        _reset(monkeypatch)


def test_unarmed_signer_fails_closed(chain, monkeypatch):
    """th#1307: Cert configured, hub transport not armed yet: raise, never return the unsigned bytes as if they ..."""
    _configure_cert_only(chain, monkeypatch)
    try:
        assert cc.enabled()
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_bytes(_png(), ref="outputs/a.png")
    finally:
        _reset(monkeypatch)


def test_hub_refusal_fails_closed(chain, monkeypatch):
    hub = _HubSigner(chain["key"].read_bytes(), status=503)
    try:
        _configure_cert_only(chain, monkeypatch)
        cc.configure_remote_signer(hub.base_url, lambda: "worker-jwt-abc")
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_bytes(_png(), ref="outputs/a.png")
    finally:
        hub.close()
        _reset(monkeypatch)


def test_unreachable_hub_fails_closed(chain, monkeypatch):
    _configure_cert_only(chain, monkeypatch)
    try:
        # Port 1 is never listening.
        cc.configure_remote_signer("http://127.0.0.1:1", lambda: "worker-jwt-abc")
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_bytes(_png(), ref="outputs/a.png")
    finally:
        _reset(monkeypatch)


def test_file_variant_also_fails_closed(
    chain: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    _configure_cert_only(chain, monkeypatch)
    try:
        with pytest.raises(cc.C2paSigningError):
            cc.sign_media_file(str(src), ref="outputs/a.png")
        assert src.read_bytes() == _png(), "source must never be mutated"
    finally:
        _reset(monkeypatch)


# ============================================================================
# pgw#1094 — the serve-path output-integrity floor.
# ============================================================================

H, W = 256, 448


def _scene(h: int = H, w: int = W) -> np.ndarray:
    """One textured still: gradients + a checker + a bright disc."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (yy / h) * 120.0 + (xx / w) * 90.0
    checker = ((yy // 16 + xx // 16) % 2) * 40.0
    disc = ((yy - h * 0.4) ** 2 + (xx - w * 0.45) ** 2) < (min(h, w) * 0.18) ** 2
    img = base + checker + disc * 70.0
    rgb = np.stack([img, img * 0.85 + 20.0, img * 0.6 + 40.0], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def clean_clip(frames: int = 24) -> np.ndarray:
    """pgw#1094: A real render's shape: one scene panning, plus the per-frame high-frequency variation every rea..."""
    still = _scene(H, W + frames).astype(np.float32)
    rng = np.random.default_rng(4242)
    clip = np.stack([still[:, t:t + W] for t in range(frames)])
    clip = clip + rng.normal(0.0, 13.0, size=clip.shape).astype(np.float32)
    return np.clip(clip, 0, 255).astype(np.uint8)


def noise_clip(frames: int = 24) -> np.ndarray:
    """ie#634's candidate: every frame independently sampled, so consecutive frames are unrelated."""
    rng = np.random.default_rng(1094)
    return rng.integers(0, 256, size=(frames, H, W, 3), dtype=np.uint8)


def black_clip(frames: int = 24) -> np.ndarray:
    return np.zeros((frames, H, W, 3), dtype=np.uint8)


def melted_clip(frames: int = 24) -> np.ndarray:
    """pgw#1094: The fp8-melt class: the clean clip with its high-frequency detail smeared away."""
    clip = clean_clip(frames).astype(np.float32)
    for _ in range(6):  # separable box blur, repeated -> heavy smear
        clip = (clip + np.roll(clip, 1, axis=1) + np.roll(clip, -1, axis=1)) / 3.0
        clip = (clip + np.roll(clip, 1, axis=2) + np.roll(clip, -1, axis=2)) / 3.0
    return np.clip(clip, 0, 255).astype(np.uint8)


def cut_clip(frames: int = 24) -> np.ndarray:
    """pgw#1094: Two unrelated real scenes spliced: one adjacent pair correlates at ~0 and the MEDIAN is what ke..."""
    a = clean_clip(frames // 2)
    b = np.stack([_scene(H, W)[::-1, ::-1]] * (frames // 2))
    b = (b.astype(np.float32) * 0.7 + 30.0).astype(np.uint8)
    b = np.stack([np.roll(f, i * 3, axis=1) for i, f in enumerate(b)])
    return np.concatenate([a, b])


def test_clean_clip_passes():
    r = oi.check_frames(clean_clip())
    assert r.verdict == oi.PASS, r.summary()
    assert r.adjacent_frame_corr > oi.NOISE_CORR_FLOOR
    assert oi.SCOPE_NOTE in r.detail  # a PASS always carries its scope limit


def test_noise_clip_is_rejected_as_noise():
    r = oi.check_frames(noise_clip())
    assert r.verdict == oi.NOISE, r.summary()
    assert r.rejected and not r.ok
    # ie#634 measured 0.29 on the production clip; independent sampling here is
    # even lower. Either way it is nowhere near the 0.6 floor.
    assert r.adjacent_frame_corr < 0.3, r.summary()


def test_black_clip_is_rejected_as_blank_not_noise():
    r = oi.check_frames(black_clip())
    assert r.verdict == oi.BLANK, r.summary()
    assert r.frame_std_min < oi.BLANK_STD_FLOOR


def test_nonfinite_pixels_are_their_own_verdict():
    """pgw#1094: A NaN latent decodes to NaN pixels — the save-path-observable form of the pre-decode NaN check."""
    clip = clean_clip().astype(np.float32) / 255.0
    clip[0] = np.nan
    r = oi.check_frames(clip)
    assert r.verdict == oi.NONFINITE, r.summary()


def test_a_hard_cut_still_serves():
    """pgw#1094: The MEDIAN over spread pairs is what makes this safe — a cut drives one pair to ~0 while the re..."""
    r = oi.check_frames(cut_clip())
    assert r.verdict == oi.PASS, r.summary()
    assert min(r.corr_series) < 0.6, "the fixture must actually contain a cut"


def test_melt_blindness_is_the_scope_boundary():
    """pgw#1094: THE PIN. A melted render PASSES and scores HIGHER than a clean one."""
    clean = oi.check_frames(clean_clip())
    melted = oi.check_frames(melted_clip())
    assert melted.verdict == oi.PASS, melted.summary()
    assert melted.adjacent_frame_corr > clean.adjacent_frame_corr, (
        f"melt inversion is the documented blind spot: "
        f"melted {melted.adjacent_frame_corr:.4f} vs clean "
        f"{clean.adjacent_frame_corr:.4f}"
    )


def test_single_frame_is_judged_on_the_blank_half_alone():
    flat = oi.check_frames(np.zeros((1, H, W, 3), np.uint8))
    assert flat.verdict == oi.BLANK
    good = oi.check_frames(_scene()[None])
    assert good.verdict == oi.PASS
    assert np.isnan(good.adjacent_frame_corr)


def test_image_floor_rejects_a_flat_render_and_passes_a_real_one():
    from PIL import Image

    assert oi.check_image(Image.new("RGB", (512, 512), (0, 0, 0))).verdict == oi.BLANK
    assert oi.check_image(Image.fromarray(_scene())).verdict == oi.PASS


def test_unmeasurable_output_is_never_a_pass():
    r = oi.check_frames(np.zeros((4, 8, 8, 7), np.uint8))  # not RGB
    assert r.verdict == oi.UNMEASURED
    assert not r.ok and not r.rejected  # confesses, does not refuse


def test_the_verdict_is_decimation_invariant():
    """pgw#1094: The ~96-row decimation is what makes this affordable on the serve path, and it does not change ..."""
    for fixture, want in ((clean_clip(), oi.PASS), (noise_clip(), oi.NOISE),
                          (black_clip(), oi.BLANK)):
        full_h = fixture.shape[1]
        assert oi.check_frames(fixture).verdict == want
        assert oi.check_frames(fixture, target_h=full_h).verdict == want
    clip = clean_clip()
    small = oi.check_frames(clip).adjacent_frame_corr
    full = oi.check_frames(clip, target_h=clip.shape[1]).adjacent_frame_corr
    assert abs(small - full) < 0.05, (small, full)


def test_cost_on_a_full_size_clip_is_single_digit_ms(capsys):
    """121 frames at 1344x768 uint8 — 0.37 GB of pixels, the ie#634 shape."""
    rng = np.random.default_rng(7)
    still = rng.integers(0, 256, size=(768, 1344 + 121, 3), dtype=np.uint8)
    clip = np.stack([still[:, t:t + 1344] for t in range(121)])
    oi.check_frames(clip)  # warm numpy
    best = min(oi.check_frames(clip).seconds for _ in range(5))
    with capsys.disabled():
        print(f"\npgw#1094 integrity floor: {best * 1000:.2f} ms on "
              f"{clip.shape} uint8 ({clip.nbytes / 1e9:.2f} GB)")
    # Naive full-resolution on this clip is 555 ms. The bound is deliberately
    # loose for CI hardware; the printed number is the measurement.
    assert best < 0.060, f"{best * 1000:.1f} ms"


def _stream(clip: np.ndarray, chunk: int = 6) -> oi.OutputIntegrity:
    c = oi.StreamCollector()
    for i in range(0, len(clip), chunk):
        c.observe(clip[i:i + chunk])
    return c.verdict()


def test_streaming_collector_rejects_noise_and_passes_a_render():
    assert _stream(clean_clip(48)).verdict == oi.PASS
    assert _stream(noise_clip(48)).verdict == oi.NOISE
    assert _stream(black_clip(48)).verdict == oi.BLANK


def test_single_chunk_stream_equals_the_buffered_answer():
    clip = clean_clip(24)
    assert _stream(clip, chunk=len(clip)).adjacent_frame_corr == pytest.approx(
        oi.check_frames(clip).adjacent_frame_corr)


def test_streaming_cost_stays_bounded_on_a_long_clip():
    """pgw#1094: The clip length is unknown until the producer is done, so the collector thins: kept pairs halve..."""
    clip = clean_clip(64)
    c = oi.StreamCollector()
    for i in range(0, 4000, 4):  # 1000 chunks of the same 4 frames
        c.observe(clip[i % 60:i % 60 + 4])
    r = c.verdict()
    assert r.verdict == oi.PASS
    assert r.frames_sampled < 12 * oi.STREAM_PAIR_BUDGET, r.frames_sampled
    assert r.seconds < 0.5, r.seconds


class _Ctx:
    """The save surface ``io.write_video`` / ``io.write_image`` touch."""

    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_video(self, path, ref, format="mp4"):
        self.saved.append(ref)
        from gen_worker.api.types import VideoAsset
        return VideoAsset(ref=ref, owner="t")

    def save_bytes(self, ref, data):
        self.saved.append(ref)
        from gen_worker.api.types import Asset
        return Asset(ref=ref, owner="t", size_bytes=len(data))


def test_write_video_refuses_noise_before_it_can_be_uploaded():
    ctx = _Ctx()
    with pytest.raises(OutputIntegrityError) as exc:
        gw_io.write_video(ctx, "outputs/r/video.mp4", noise_clip(), fps=24)
    assert exc.value.verdict == oi.NOISE
    assert "output-integrity floor" in str(exc.value)
    assert ctx.saved == [], "a rejected render must never reach the upload"


def test_write_video_refuses_a_black_clip():
    ctx = _Ctx()
    with pytest.raises(OutputIntegrityError) as exc:
        gw_io.write_video(ctx, "outputs/r/video.mp4", black_clip(), fps=24)
    assert exc.value.verdict == oi.BLANK
    assert ctx.saved == []


def test_write_video_still_serves_a_real_render():
    ctx = _Ctx()
    asset = gw_io.write_video(ctx, "outputs/r/video.mp4", clean_clip(), fps=24)
    assert asset.ref == "outputs/r/video.mp4"
    assert ctx.saved == ["outputs/r/video.mp4"]


def test_write_video_screens_the_streaming_seam_too():
    ctx = _Ctx()
    clip = noise_clip(48)
    chunks = (clip[i:i + 8] for i in range(0, len(clip), 8))
    with pytest.raises(OutputIntegrityError):
        gw_io.write_video(ctx, "outputs/r/video.mp4", chunks, fps=24)
    assert ctx.saved == []


def test_write_image_refuses_a_flat_render():
    from PIL import Image

    ctx = _Ctx()
    with pytest.raises(OutputIntegrityError) as exc:
        gw_io.write_image(ctx, "outputs/r/image", Image.new("RGB", (256, 256)))
    assert exc.value.verdict == oi.BLANK
    assert ctx.saved == []
    assert gw_io.write_image(ctx, "outputs/r/image",
                             Image.fromarray(_scene())).ref.endswith(".webp")


def test_the_fault_maps_fatal_and_never_invalid():
    """pgw#1094: BLAME: a render is produced by release code, model state AND payload together, so this is neith..."""
    from gen_worker import executor
    from gen_worker.api.errors import RetryableError, ValidationError
    from gen_worker.pb import worker_scheduler_pb2 as pb

    exc = OutputIntegrityError(oi.NOISE, ref="outputs/r/video.mp4",
                               kind="video", summary="integrity noise")
    assert not isinstance(exc, (ValidationError, RetryableError))
    status, msg = executor._map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert msg.startswith("OutputIntegrityError:")


def test_reject_and_unmeasured_emit_a_typed_event_and_pass_does_not():
    from gen_worker import activity

    rows: list = []
    activity.reset_for_tests()
    try:
        for result, kind in (
            (oi.check_frames(noise_clip()), oi.NOISE),
            (oi.check_frames(np.zeros((4, 8, 8, 7), np.uint8)), oi.UNMEASURED),
            (oi.check_frames(clean_clip()), oi.PASS),
        ):
            before = len(rows)
            with _capture(activity, rows):
                try:
                    oi.enforce(result, ref="outputs/r/out", kind="video")
                except OutputIntegrityError:
                    pass
            if kind is oi.PASS:
                assert len(rows) == before, "a PASS buys no row"
            else:
                assert len(rows) == before + 1
                assert rows[-1].kind == activity.KIND_OUTPUT_INTEGRITY
                assert rows[-1].phase == kind
                assert "adjacent_frame_corr" in rows[-1].detail
    finally:
        activity.reset_for_tests()


class _capture:
    def __init__(self, activity, rows):
        self._a, self._rows = activity, rows

    def __enter__(self):
        self._prev = self._a._sink
        self._a._sink = self._rows.append
        return self

    def __exit__(self, *exc):
        self._a._sink = self._prev
        return False


def test_measured_floors_match_the_eval_half():
    """pgw#1094: The eval half (cozy-eval ce#10, metric_set @7) and this serve half share the floors by construc..."""
    assert oi.NOISE_CORR_FLOOR == 0.6
    assert oi.BLANK_STD_FLOOR == 0.01
    assert oi.INTEGRITY_PAIRS == 5
    assert oi.INTEGRITY_TARGET_H == 96


# ============================================================================
# pgw#991 — a bare-hex blob address is refused, not tagged
#   `blake3:`.
# ============================================================================

BLOB_BYTES = b"real blob bytes"


BARE_HEX = hashlib.sha256(BLOB_BYTES).hexdigest()


SHA256_DIGEST = f"sha256:{BARE_HEX}"


BLAKE3_DIGEST = "blake3:" + blake3(BLOB_BYTES).hexdigest()


_SUPPORTED = {"blake3": 64, "sha256": 64}


_REQUESTED: list[str] = []


def _hub_parse_digest(ref: str) -> str | None:
    """tensorhub `internal/storage/cas_paths.go` ParseDigest, transcribed."""
    ref = ref.strip()
    if not ref:
        return None
    algo, sep, hexpart = ref.partition(":")
    if not sep:
        return None  # "bare hex is refused"
    algo = algo.strip().lower()
    hexpart = hexpart.strip()
    width = _SUPPORTED.get(algo)
    if width is None:
        return None
    if any(c not in "0123456789abcdefABCDEF" for c in hexpart):
        return None
    if len(hexpart) != width:
        return None
    return f"{algo}:{hexpart.lower()}"


class _Hub(BaseHTTPRequestHandler):
    def log_message(self, *_a: object) -> None:
        pass

    def _send(self, code: int, body: bytes = b"") -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if "/blobs/" not in path or not path.endswith("/content"):
            return self._send(404, b"")
        raw = unquote(path.split("/blobs/", 1)[1][: -len("/content")])
        _REQUESTED.append(raw)
        if _hub_parse_digest(raw) is None:
            # api.WriteError(c, http.StatusBadRequest, "invalid_digest", "")
            return self._send(400, b'{"error":{"code":"invalid_digest"}}')
        return self._send(200, BLOB_BYTES)


@pytest.fixture()
def hub() -> Iterator[str]:
    _REQUESTED.clear()
    srv = HTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _ctx(hub_url: str) -> JobContext:
    ctx = JobContext(request_id="r-pgw991")
    ctx._file_api_base_url = hub_url
    ctx._worker_capability_token = "test-token"
    return ctx


def test_bare_hex_is_refused_before_any_request(hub: str, tmp_path: Path) -> None:
    """pgw#991: The filed defect."""
    ctx = _ctx(hub)

    with pytest.raises(BlobDigestMalformedError) as ei:
        ctx.materialize_blob(BARE_HEX, tmp_path / "out.bin")

    exc = ei.value
    assert exc.code == "blob_digest_malformed"
    assert exc.ref == BARE_HEX
    assert isinstance(exc, PayloadRefError)
    assert isinstance(exc, ValidationError)
    # The refusal names the fix, in the hub's own words.
    assert "sha256:<64 hex>" in str(exc)
    assert _REQUESTED == [], "a malformed address must not reach the network"


def test_bare_hex_maps_to_INVALID_not_FATAL(hub: str, tmp_path: Path) -> None:
    """th#1259's rule holds for this class too: a caller's bad address is never model-health evidence."""
    ctx = _ctx(hub)
    with pytest.raises(BlobDigestMalformedError) as ei:
        ctx.materialize_blob(BARE_HEX, tmp_path / "out.bin")
    status, message = _map_exception(ei.value)
    assert status == pb.JOB_STATUS_INVALID
    assert message.startswith("blob_digest_malformed: ")


def test_the_old_blake3_guess_is_what_the_hub_rejects(hub: str, tmp_path: Path) -> None:
    """pgw#991: Guard on the regression itself."""
    ctx = _ctx(hub)
    out = ctx.materialize_blob(SHA256_DIGEST, tmp_path / "ok.bin")
    assert out.read_bytes() == BLOB_BYTES
    assert _REQUESTED == [SHA256_DIGEST], (
        "the digest must reach the hub verbatim and algorithm-tagged; "
        f"a blake3: guess would have sent blake3:{BARE_HEX}"
    )


def test_blake3_still_works_for_the_dataset_cas(hub: str, tmp_path: Path) -> None:
    """pgw#991: blake3 is not dead — it is the dataset-CAS contract."""
    ctx = _ctx(hub)
    out = ctx.materialize_blob(BLAKE3_DIGEST, tmp_path / "ds.bin")
    assert out.read_bytes() == BLOB_BYTES
    assert _REQUESTED == [BLAKE3_DIGEST]


def test_platform_origin_malformed_stays_fatal(hub: str, tmp_path: Path) -> None:
    """pgw#991: A malformed address the PLATFORM produced is a platform fault, so it keeps the fatal classificat..."""
    ctx = _ctx(hub)
    with pytest.raises(RuntimeError) as ei:
        ctx.materialize_blob(BARE_HEX, tmp_path / "p.bin", origin=REF_ORIGIN_PLATFORM)
    assert not isinstance(ei.value, PayloadRefError)
    assert "malformed platform blob digest" in str(ei.value)
    assert _REQUESTED == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        BARE_HEX,                       # not algorithm-tagged
        f"md5:{BARE_HEX}",              # unsupported algorithm
        "sha256:" + "zz" * 32,          # non-hex character
        "sha256:" + "ab" * 16,          # wrong width
        "sha256:",                      # empty hex
    ],
)
def test_worker_refusal_matches_the_hub_exactly(hub: str, tmp_path: Path, bad: str) -> None:
    """pgw#991: Parity, asserted rather than asserted-about: every address the worker refuses is one the hub's P..."""
    assert _hub_parse_digest(bad) is None, "rig disagrees with the hub's rule"
    ctx = _ctx(hub)
    with pytest.raises(BlobDigestMalformedError):
        ctx.materialize_blob(bad, tmp_path / "x.bin")
    assert _REQUESTED == []


@pytest.mark.parametrize(
    "good",
    [SHA256_DIGEST, SHA256_DIGEST.upper().replace("SHA256", "sha256"), BLAKE3_DIGEST],
)
def test_worker_accepts_everything_the_hub_accepts(hub: str, tmp_path: Path, good: str) -> None:
    canonical = _hub_parse_digest(good)
    assert canonical is not None
    ctx = _ctx(hub)
    ctx.materialize_blob(good, tmp_path / "y.bin")
    # Canonicalised the same way the hub canonicalises it (lowercased hex).
    assert _REQUESTED == [canonical]


# ============================================================================
# th#2082 — the worker's ONE read of ``RunJob.media_bytes``,
#   fenced on the value it RECEIVED rather than on the value the hub set.
# ============================================================================

_ORG = "00000000-0000-0000-0000-000000000001"


_MODULES = ("harness.media_bytes_th2082",)


def _render(request_id: str, **run_job_kwargs: object) -> MediaOut:
    """Dispatch one small-media job and return what the handler observed."""
    reset_upload_sink()
    httpd, base_url = serve_upload_sink()
    try:
        with hub_double(modules=_MODULES, file_base_url=base_url) as (scheduler, _h):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(run_job=pb.RunJob(
                request_id=request_id, attempt=1, function_name="render",
                input_payload=msgspec.msgpack.encode(MediaIn(text="x")),
                org=_ORG, capability_token="cap-token",
                **run_job_kwargs,  # type: ignore[arg-type]
            ))
            res = conn.wait_for(is_result_for(request_id)).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            return msgspec.msgpack.decode(res.inline, type=MediaOut)
    finally:
        httpd.shutdown()


def test_inline_preference_is_honoured_by_the_worker_that_received_it() -> None:
    """th#2082: RED when the consumer stops reading field 9: the bytes get uploaded and the client's `Prefer: by..."""
    out = _render("r-2082-inline", media_bytes=pb.MEDIA_BYTES_INLINE)

    assert out.inline, (
        "the worker received MEDIA_BYTES_INLINE and still returned a ref-only "
        "Asset — `Prefer: bytes=inline` degraded to URL delivery, which is the "
        "SILENT failure this fence exists for"
    )
    assert not DedupUploadSink.requests_seen, (
        "MEDIA_BYTES_INLINE must skip the tensorhub upload entirely; the sink "
        f"was hit {len(DedupUploadSink.requests_seen)} time(s)"
    )
    assert out.size_bytes == 2048


def test_url_preference_is_honoured_by_the_worker_that_received_it() -> None:
    out = _render("r-2082-url", media_bytes=pb.MEDIA_BYTES_URL)

    assert not out.inline, "MEDIA_BYTES_URL must upload and return a ref"
    assert DedupUploadSink.requests_seen, (
        "MEDIA_BYTES_URL must really upload — a ref naming bytes that never "
        "left the process is pgw#767's defect"
    )


def test_unspecified_falls_to_the_workers_own_default() -> None:
    """th#2082: No `Prefer:` header = no hub opinion; the worker's default (upload) wins."""
    out = _render("r-2082-unset")

    assert not out.inline
    assert DedupUploadSink.requests_seen


def test_the_rename_did_not_move_the_wire() -> None:
    """th#2082: Field NUMBER 9 and enum numbers 0/1/2 are what travel."""
    wire = pb.RunJob(
        request_id="r-2082-wire", media_bytes=pb.MEDIA_BYTES_INLINE,
    ).SerializeToString()

    # tag = (9 << 3) | 0 (varint) = 0x48; MEDIA_BYTES_INLINE = 2.
    assert b"\x48\x02" in wire, (
        f"field 9 does not carry varint 2 on the wire ({wire!r}) — the number "
        "moved, which IS a wire break even though the rename was not"
    )

    received = pb.RunJob()
    received.ParseFromString(wire)
    assert received.media_bytes == pb.MEDIA_BYTES_INLINE
    assert int(pb.MEDIA_BYTES_UNSPECIFIED) == 0
    assert int(pb.MEDIA_BYTES_URL) == 1
    assert int(pb.MEDIA_BYTES_INLINE) == 2


def _grep(pattern: str) -> str:
    import subprocess
    done = subprocess.run(
        ["git", "grep", "--untracked", "-nI", "-e", pattern, "--",
         "*.py", "*.proto"],
        capture_output=True, text=True, check=False,
    )
    return done.stdout.strip()


def test_retired_wire_enum_spelling_is_gone() -> None:
    # th#2082. The searched strings are ASSEMBLED so this file is not its own
    # hit — the old `:!tests/test_media_bytes_th2082.py` exclusion blinded the
    # guard to a whole file, and after the pgw#1362 merge that file holds
    # neighbours. The grep now covers the entire tree with no exclusion.
    for retired in ("OUTPUT_" "MODE_", "Output" "Mode", "run." "output_mode"):
        hits = _grep(retired)
        assert not hits, (
            f"th#2082: {retired!r} is back. The client's `Prefer: bytes=` "
            f"preference is `media_bytes`/`MediaBytes`; `EndpointSpec."
            f"output_mode` is a DIFFERENT question and keeps its own name "
            f"until pgw#1320:\n{hits}"
        )


def test_the_scanner_can_actually_find_things() -> None:
    """th#2082: An absent-string fence passes just as well when its scanner is broken."""
    assert _grep("MEDIA_BYTES_INLINE"), (
        "th#2082 fence scanner found no `MEDIA_BYTES_INLINE` — the scanner is "
        "broken, so the absence fence above proves nothing"
    )


# ============================================================================
# pgw#1320 — the manifest carries ONE spelling of output
#   cardinality.
# ============================================================================

LIVE_KEY = "incremental_output"


DEAD_KEY = "output_mode"


def _endpoint_tree(root: Path) -> None:
    """pgw#1320: A toy endpoint with BOTH cardinalities — a struct-returning function and an Iterator-returning ..."""
    (root / "pyproject.toml").write_text(textwrap.dedent("""
        [project]
        name = "ep1320"

        [tool.gen_worker]
        main = "ep1320.main"
    """))
    src = root / "ep1320"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(textwrap.dedent("""
        from typing import Iterator

        import msgspec
        from gen_worker import RequestContext, Resources, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str = ""

        @endpoint(resources=Resources(gpu=False))
        class Both:
            def setup(self) -> None: ...

            def whole(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_()

            def streamed(self, ctx: RequestContext, data: In_) -> Iterator[Out_]:
                yield Out_()
    """))


@pytest.fixture()
def functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    from gen_worker.discovery.discover import discover_manifest

    _endpoint_tree(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    return list(discover_manifest(tmp_path)["functions"])


def test_the_walk_is_not_vacuous(functions: list[dict]) -> None:
    """pgw#1320: The control for the absence assertion below."""
    by_name = {fn["name"]: fn for fn in functions}
    assert set(by_name) == {"whole", "streamed"}, by_name.keys()
    assert by_name["whole"][LIVE_KEY] is False
    assert by_name["streamed"][LIVE_KEY] is True


def test_the_manifest_carries_no_output_mode(functions: list[dict]) -> None:
    """pgw#1320: the dead key is gone from a REAL discovery manifest."""
    carriers = [fn["name"] for fn in functions if DEAD_KEY in fn]
    assert carriers == [], (
        f"{DEAD_KEY!r} is back on {carriers} — the hub has never decoded it "
        f"({DEAD_KEY} appears nowhere in tensorhub/internal/builder); "
        f"{LIVE_KEY} is the fact and the only spelling the hub reads"
    )


def test_no_manifest_block_reintroduces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1320: Scoped to the function rows above, the fence would miss the key reappearing on a sibling manife..."""
    import json

    from gen_worker.discovery.discover import discover_manifest

    _endpoint_tree(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    doc = json.dumps(discover_manifest(tmp_path))

    assert LIVE_KEY in doc, "the walk emitted no cardinality at all"
    assert f'"{DEAD_KEY}"' not in doc, (
        f"{DEAD_KEY!r} reappeared somewhere in the manifest document"
    )


# ============================================================================
# pgw#1158 — A declared target that NO Input row reaches is refused.
# ============================================================================

def _decl(**over: Any) -> Compile:
    base = dict(
        family="fam", text_len=512,
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        shape_strategy="static-rows", warm_changes_key=False)
    base.update(over)
    return Compile(**base)  # type: ignore[arg-type]


def test_a_target_no_input_row_reaches_is_REFUSED() -> None:
    """RED on master: ACCEPTED, and it mints a plan for the starved target."""
    with pytest.raises(DeclarationError, match="NO Input row reaches it"):
        _decl(
            targets=("transformer", "vae.decode"),
            inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                          targets=("transformer",)),))


def test_the_refusal_names_the_TARGET_and_where_the_rows_went() -> None:
    """pgw#1158: An author reading it must be able to act without re-deriving the scope map: the starved target ..."""
    with pytest.raises(DeclarationError) as err:
        _decl(
            targets=("transformer", "vae.decode"),
            inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                          targets=("transformer",)),))
    detail = str(err.value)
    assert "'vae.decode'" in detail, detail
    assert "['transformer']" in detail, detail
    # ...and all three ways out, because a refusal with one exit gets routed
    # around by whichever exit the author thought of first.
    assert "Scope a row" in detail and "drop the target" in detail, detail
    assert "untargeted" in detail, detail


def test_an_UNTARGETED_row_reaching_every_target_is_untouched() -> None:
    """pgw#1158: Today's defaulting — the half that is NOT ruled here."""
    decl = _decl(
        targets=("transformer", "vae.decode"),
        inputs=(Input("hidden_states", shape=("B", 4), dtype="model"),))
    assert decl.targets == ("transformer", "vae.decode")


def test_a_declaration_with_NO_inputs_at_all_is_a_different_case() -> None:
    """pgw#1158: `inputs=()` is a declaration that states no ingress anywhere — already handled elsewhere and le..."""
    assert _decl(targets=("transformer",), inputs=()).inputs == ()


def test_every_target_scoped_explicitly_is_fine() -> None:
    """The shape the guard exists to require."""
    decl = _decl(
        targets=("transformer", "vae.decode"),
        inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                      targets=("transformer",)),
                Input("latent", shape=("B", 4), dtype="model",
                      targets=("vae.decode",))))
    assert len(decl.inputs) == 2


def test_the_guard_FIRES_on_the_shape_master_accepted() -> None:
    """pgw#1158: The severance experiment, run on this guard: master accepted the construction below and minted ..."""
    with pytest.raises(DeclarationError):
        _decl(
            targets=("transformer", "vae.decode"),
            inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                          targets=("transformer",)),))
        pytest.fail("the guard did not fire — it has been severed")
