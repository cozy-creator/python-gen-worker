"""C2PA sign+verify round trips (pgw#658 restores gw#518 coverage, th#714).

EU AI Act Art. 50 legal-critical path: every generated media asset must leave
``RequestContext.save_bytes/save_file`` carrying a signed Content-Credentials
manifest when signing is configured. The pgw#609 test sweep deleted the
original gw#518 suite; this is its greenfield replacement.

th#1307: the private key is HUB-SIDE. Every round trip below therefore signs
through a real HTTP signing oracle (``fake_hub_signer``) that holds the key and
answers ``POST /v1/worker/c2pa/sign`` exactly like the hub route does —
base64 COSE fixed-width r||s. The worker only ever holds the public chain.

Chain fixture mirrors the production cert profile: ES256, keyUsage
digitalSignature, EKU emailProtection, key in PKCS#8 — generated with openssl
so a cert Paul buys per the th#714 runbook exercises the exact same code path.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker import content_credentials as cc


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture(scope="module")
def es256_chain(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Self-signed ES256 signing cert + PKCS#8 key via openssl."""
    d = tmp_path_factory.mktemp("c2pa-chain")
    key_raw = d / "key-raw.pem"
    key = d / "key.pem"  # PKCS#8
    cert = d / "cert.pem"
    run = lambda *args: subprocess.run(  # noqa: E731
        args, check=True, capture_output=True
    )
    run("openssl", "ecparam", "-genkey", "-name", "prime256v1", "-noout",
        "-out", str(key_raw))
    run("openssl", "pkcs8", "-topk8", "-nocrypt", "-in", str(key_raw),
        "-out", str(key))
    run("openssl", "req", "-x509", "-new", "-key", str(key),
        "-subj", "/O=Cozy Test/CN=cozy-gen-worker test signer",
        "-days", "365", "-sha256",
        "-addext", "basicConstraints=CA:FALSE",
        "-addext", "keyUsage=critical,digitalSignature",
        "-addext", "extendedKeyUsage=emailProtection",
        "-out", str(cert))
    return {"cert": cert, "key": key}


@pytest.fixture(scope="module")
def fake_hub_signer(es256_chain):
    """A stand-in for the hub's POST /v1/worker/c2pa/sign route.

    Holds the private key (as the hub does), returns base64 COSE r||s (as
    internal/orchestrator/c2pasign does), and refuses an unauthenticated
    caller (as the worker-JWT gate does). This is the whole th#1307 contract
    from the worker side.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils

    key = serialization.load_pem_private_key(
        es256_chain["key"].read_bytes(), password=None
    )
    state = {"signatures": 0, "tokens": [], "claim_bytes": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != cc.SIGN_PATH:
                self.send_response(404); self.end_headers(); return
            authz = self.headers.get("Authorization") or ""
            if not authz.startswith("Bearer "):
                self.send_response(401); self.end_headers(); return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            claim = base64.b64decode(body["claim_b64"])
            der = key.sign(claim, ec.ECDSA(hashes.SHA256()))
            r, sv = asym_utils.decode_dss_signature(der)
            sig = r.to_bytes(32, "big") + sv.to_bytes(32, "big")
            state["signatures"] += 1
            state["tokens"].append(authz.removeprefix("Bearer "))
            state["claim_bytes"].append(len(claim))
            out = json.dumps({"alg": "es256",
                              "signature_b64": base64.b64encode(sig).decode()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


@pytest.fixture()
def signer_configured(es256_chain, fake_hub_signer, monkeypatch):
    """Configure the worker exactly as a pod is configured: PUBLIC cert only,
    signatures from the hub. Resets module state afterwards."""
    settings = SimpleNamespace(
        c2pa_cert_pem="",
        c2pa_cert_path=str(es256_chain["cert"]),
        c2pa_alg="es256",
        c2pa_ta_url="",
    )
    cc.configure(settings)
    cc.configure_remote_signer(fake_hub_signer["base_url"], lambda: "worker-jwt-test")
    yield settings
    # Plain assignment, NOT monkeypatch (pgw#772 release lane): this runs during
    # finalization, and `monkeypatch`'s own undo -- which recorded the globals
    # AFTER cc.configure() above had already armed them -- would restore the armed
    # values right back. That leaked a configured signer, pointed at this
    # now-shutdown fake hub, into every later test in the session; th#1307 made
    # signing failures fatal, so 7 downstream tests went JOB_STATUS_FATAL.
    cc._configured = False
    cc._config = None
    cc._remote = None


def _reset(monkeypatch):
    monkeypatch.setattr(cc, "_configured", False)
    monkeypatch.setattr(cc, "_config", None)
    monkeypatch.setattr(cc, "_remote", None)


def _unconfigure(monkeypatch):
    cc.configure(
        SimpleNamespace(c2pa_cert_pem="", c2pa_cert_path="",
                        c2pa_alg="es256", c2pa_ta_url="")
    )


# ---------------------------------------------------------------------------
# sample media


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 120, 40)).save(buf, format="PNG")
    return buf.getvalue()


def _webp() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 90, 160)).save(buf, format="WEBP")
    return buf.getvalue()


def _jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (60, 60, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 1600)
    return buf.getvalue()


def _mp4(tmp_path: Path) -> bytes:
    import av
    import numpy as np

    p = tmp_path / "clip.mp4"
    with av.open(str(p), "w") as container:
        stream = container.add_stream("h264", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return p.read_bytes()


def _verify(mime: str, data: bytes) -> dict:
    """Read the embedded manifest back with the library Reader."""
    import c2pa

    with c2pa.Reader(mime, io.BytesIO(data)) as reader:
        state = reader.get_validation_state()
        # No trust list is configured in tests, so the self-signed issuer
        # yields Valid or Trusted depending on library defaults — anything
        # but Invalid means hard bindings + signature verified.
        assert str(state) in ("Valid", "Trusted"), state
        return json.loads(reader.json())


# ---------------------------------------------------------------------------
# round trips


@pytest.mark.parametrize(
    "make,ref,mime",
    [
        (_png, "outputs/a.png", "image/png"),
        (_webp, "outputs/a.webp", "image/webp"),
        (_jpeg, "outputs/a.jpg", "image/jpeg"),
        (_wav, "outputs/a.wav", "audio/wav"),
    ],
    ids=["png", "webp", "jpeg", "wav"],
)
def test_sign_verify_round_trip(signer_configured, make, ref, mime):
    raw = make()
    signed = cc.sign_media_bytes(
        raw, ref=ref, request_id="req-123", models=["cozy/sdxl-base@v5"]
    )
    assert signed != raw
    manifest_store = _verify(mime, signed)
    active = manifest_store["manifests"][manifest_store["active_manifest"]]
    actions = next(
        a for a in active["assertions"] if a["label"].startswith("c2pa.actions")
    )
    action = actions["data"]["actions"][0]
    assert action["action"] == "c2pa.created"
    assert action["digitalSourceType"].endswith("trainedAlgorithmicMedia")
    cozy = next(a for a in active["assertions"] if a["label"] == "com.cozy.generation")
    assert cozy["data"]["models"] == ["cozy/sdxl-base@v5"]
    assert cozy["data"]["request_sha256"] == hashlib.sha256(b"req-123").hexdigest()
    # Privacy line (th#1047-consistent): no prompts, no user identity.
    blob = json.dumps(active)
    assert "prompt" not in blob
    assert "req-123" not in blob


def test_sign_verify_round_trip_mp4(signer_configured, tmp_path):
    raw = _mp4(tmp_path)
    signed = cc.sign_media_bytes(raw, ref="outputs/clip.mp4", request_id="req-v")
    assert signed != raw
    _verify("video/mp4", signed)


def test_sign_media_file_round_trip(signer_configured, tmp_path):
    src = tmp_path / "img.webp"
    src.write_bytes(_webp())
    out_path = cc.sign_media_file(str(src), ref="outputs/img.webp", request_id="r")
    assert out_path is not None
    try:
        signed = Path(out_path).read_bytes()
        assert src.read_bytes() == _webp()  # source untouched
        _verify("image/webp", signed)
    finally:
        os.unlink(out_path)


# ---------------------------------------------------------------------------
# config semantics


def test_inline_pem_config_signs(es256_chain, fake_hub_signer, monkeypatch):
    """GEN_WORKER_C2PA_CERT_PEM (hub-injected pod env, th#714) is sufficient —
    the cert is the ONLY material a pod gets (th#1307)."""
    _reset(monkeypatch)
    cc.configure(
        SimpleNamespace(
            c2pa_cert_pem=es256_chain["cert"].read_text(),
            c2pa_cert_path="",
            c2pa_alg="es256",
            c2pa_ta_url="",
        )
    )
    cc.configure_remote_signer(fake_hub_signer["base_url"], lambda: "worker-jwt-test")
    try:
        assert cc.enabled()
        signed = cc.sign_media_bytes(_png(), ref="o.png")
        _verify("image/png", signed)
    finally:
        _reset(monkeypatch)


def test_inline_pem_beats_broken_paths(es256_chain, monkeypatch):
    """Inline PEM wins: a bogus *_PATH value is not even read."""
    _reset(monkeypatch)
    cc.configure(
        SimpleNamespace(
            c2pa_cert_pem=es256_chain["cert"].read_text(),
            c2pa_cert_path="/nonexistent/cert.pem",
            c2pa_alg="es256",
            c2pa_ta_url="",
        )
    )
    try:
        assert cc.enabled()
    finally:
        _reset(monkeypatch)


def test_unconfigured_is_noop(monkeypatch):
    _reset(monkeypatch)
    _unconfigure(monkeypatch)
    try:
        raw = _png()
        assert cc.sign_media_bytes(raw, ref="o.png") is raw
        assert cc.sign_media_file(__file__, ref="o.png") is None
        assert not cc.enabled()
    finally:
        _reset(monkeypatch)


def test_non_media_passthrough(signer_configured):
    payload = b'{"not": "media"}'
    assert cc.sign_media_bytes(payload, ref="outputs/result.json") is payload


def test_garbage_pem_refuses(monkeypatch):
    """Configured-but-broken must refuse (worker startup fails loudly)."""
    _reset(monkeypatch)
    with pytest.raises(cc.C2paSigningError):
        cc.configure(
            SimpleNamespace(
                c2pa_cert_pem="not a pem",
                c2pa_cert_path="",
                c2pa_alg="es256",
                c2pa_ta_url="",
            )
        )
    _reset(monkeypatch)


# ---------------------------------------------------------------------------
# format sniffing


@pytest.mark.parametrize(
    "head,ref,expect",
    [
        (b"\x89PNG\r\n\x1a\n########", "a.png", "image/png"),
        (b"RIFF\x00\x00\x00\x00WEBP####", "a.webp", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVE####", "a.wav", "audio/wav"),
        (b"\x00\x00\x00\x18ftypisom####", "a.mp4", "video/mp4"),
        (b"\x00\x00\x00\x18ftypqt  ####", "a.mov", "video/quicktime"),
        (b'{"json": true}##', "a.json", None),
        (b"\x89PNG\r\n\x1a\n########", "no-extension", "image/png"),
    ],
)
def test_sniff(head, ref, expect):
    assert cc.sniff_media_mime(head, ref) == expect
