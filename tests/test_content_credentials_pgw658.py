"""C2PA sign+verify round trips (pgw#658 restores gw#518 coverage, th#714).

EU AI Act Art. 50 legal-critical path: every generated media asset must leave
``RequestContext.save_bytes/save_file`` carrying a signed Content-Credentials
manifest when signing is configured. The pgw#609 test sweep deleted the
original gw#518 suite; this is its greenfield replacement.

Chain fixture mirrors the production cert profile: ES256, keyUsage
digitalSignature, EKU emailProtection, key in PKCS#8 — generated with openssl
so a cert Paul buys per the th#714 runbook exercises the exact same code path.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import wave
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


@pytest.fixture()
def signer_configured(es256_chain, monkeypatch):
    """Install the test signer process-wide; reset module state afterwards."""
    settings = SimpleNamespace(
        c2pa_cert_pem="",
        c2pa_key_pem="",
        c2pa_cert_path=str(es256_chain["cert"]),
        c2pa_key_path=str(es256_chain["key"]),
        c2pa_alg="es256",
        c2pa_ta_url="",
    )
    cc.configure(settings)
    yield settings
    _reset(monkeypatch)


def _reset(monkeypatch):
    monkeypatch.setattr(cc, "_configured", False)
    monkeypatch.setattr(cc, "_config", None)


def _unconfigure(monkeypatch):
    cc.configure(
        SimpleNamespace(c2pa_cert_pem="", c2pa_key_pem="",
                        c2pa_cert_path="", c2pa_key_path="",
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


def test_inline_pem_config_signs(es256_chain, monkeypatch):
    """GEN_WORKER_C2PA_*_PEM (hub-injected pod env, th#714) is sufficient."""
    _reset(monkeypatch)
    cc.configure(
        SimpleNamespace(
            c2pa_cert_pem=es256_chain["cert"].read_text(),
            c2pa_key_pem=es256_chain["key"].read_text(),
            c2pa_cert_path="",
            c2pa_key_path="",
            c2pa_alg="es256",
            c2pa_ta_url="",
        )
    )
    try:
        assert cc.enabled()
        signed = cc.sign_media_bytes(_png(), ref="o.png")
        _verify("image/png", signed)
    finally:
        _reset(monkeypatch)


def test_inline_pem_beats_broken_paths(es256_chain, monkeypatch):
    """Inline PEM wins: bogus *_PATH values are not even read."""
    _reset(monkeypatch)
    cc.configure(
        SimpleNamespace(
            c2pa_cert_pem=es256_chain["cert"].read_text(),
            c2pa_key_pem=es256_chain["key"].read_text(),
            c2pa_cert_path="/nonexistent/cert.pem",
            c2pa_key_path="/nonexistent/key.pem",
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


def test_cert_without_key_refuses(monkeypatch, es256_chain):
    """Configured-but-broken must refuse (worker startup fails loudly)."""
    _reset(monkeypatch)
    with pytest.raises(cc.C2paSigningError):
        cc.configure(
            SimpleNamespace(
                c2pa_cert_pem=es256_chain["cert"].read_text(),
                c2pa_key_pem="",
                c2pa_cert_path="",
                c2pa_key_path="",
                c2pa_alg="es256",
                c2pa_ta_url="",
            )
        )
    _reset(monkeypatch)


def test_garbage_pem_refuses(monkeypatch):
    _reset(monkeypatch)
    with pytest.raises(cc.C2paSigningError):
        cc.configure(
            SimpleNamespace(
                c2pa_cert_pem="not a pem",
                c2pa_key_pem="also not a pem",
                c2pa_cert_path="",
                c2pa_key_path="",
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
