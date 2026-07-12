"""C2PA content-credential signing at the media-finalize seam (th#714).

Real sign + verify round trips with a self-signed test cert (openssl-generated
ES256 chain), through the actual production codepaths: RequestContext.save_image
-> save_bytes (png) and io.write_video -> save_video -> save_file (mp4).
Verification uses the c2pa library's Reader.
"""

from __future__ import annotations

import io
import json
import hashlib
import subprocess
from pathlib import Path

import pytest

c2pa = pytest.importorskip("c2pa")
PIL_Image = pytest.importorskip("PIL.Image")

from gen_worker import RequestContext, content_credentials
from gen_worker.config import Settings


TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)


@pytest.fixture(scope="session")
def test_cert(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Self-signed ES256 chain matching the C2PA cert profile (digitalSignature
    keyUsage + emailProtection EKU), key in PKCS#8."""
    d = tmp_path_factory.mktemp("c2pa-certs")

    def run(*args: str) -> None:
        subprocess.run(args, cwd=d, check=True, capture_output=True)

    run("openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", "ca.key")
    run(
        "openssl", "req", "-x509", "-new", "-key", "ca.key", "-sha256", "-days", "365",
        "-subj", "/O=Cozy Test/CN=Cozy Test C2PA Root",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        "-out", "ca.pem",
    )
    run("openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", "leaf.sec1")
    run("openssl", "pkcs8", "-topk8", "-nocrypt", "-in", "leaf.sec1", "-out", "leaf.key")
    run(
        "openssl", "req", "-new", "-key", "leaf.key",
        "-subj", "/O=Cozy Test/CN=Cozy Test C2PA Signer", "-out", "leaf.csr",
    )
    (d / "leaf.ext").write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature\n"
        "extendedKeyUsage=emailProtection\n"
    )
    run(
        "openssl", "x509", "-req", "-in", "leaf.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
        "-CAcreateserial", "-days", "365", "-sha256", "-extfile", "leaf.ext", "-out", "leaf.pem",
    )
    (d / "chain.pem").write_bytes((d / "leaf.pem").read_bytes() + (d / "ca.pem").read_bytes())
    return {"cert": str(d / "chain.pem"), "key": str(d / "leaf.key")}


@pytest.fixture()
def signing_on(test_cert: dict[str, str]):
    content_credentials._reset_for_tests()
    content_credentials.configure(
        Settings(c2pa_cert_path=test_cert["cert"], c2pa_key_path=test_cert["key"])
    )
    yield
    content_credentials._reset_for_tests()


@pytest.fixture()
def signing_off():
    content_credentials._reset_for_tests()
    content_credentials.configure(Settings())
    yield
    content_credentials._reset_for_tests()


def _ctx(tmp_path: Path, **kw) -> RequestContext:
    return RequestContext(
        request_id="req-c2pa-1",
        local_output_dir=str(tmp_path / "outputs"),
        models=kw.pop("models", {"main": "ltx-video-cloud:prod"}),
        **kw,
    )


def _read_manifest(path: str, mime: str) -> dict:
    with open(path, "rb") as f:
        reader = c2pa.Reader(mime, f)
        report = json.loads(reader.json())
    assert report.get("validation_state") == "Valid"
    return report["manifests"][report["active_manifest"]]


def _actions(active: dict) -> list[dict]:
    for a in active["assertions"]:
        if a["label"].startswith("c2pa.actions"):
            return a["data"]["actions"]
    raise AssertionError(f"no c2pa.actions assertion in {active['assertions']}")


def test_png_sign_verify_roundtrip_via_save_image(tmp_path: Path, signing_on) -> None:
    ctx = _ctx(tmp_path)
    img = PIL_Image.new("RGB", (64, 64), (200, 30, 90))
    asset = ctx.save_image(img, "outputs/img", format="png")
    assert asset.local_path and asset.local_path.endswith(".png")

    active = _read_manifest(asset.local_path, "image/png")
    action = _actions(active)[0]
    assert action["action"] == "c2pa.created"
    assert action["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA
    assert active["claim_generator_info"][0]["name"] == "cozy-gen-worker"
    assert active["signature_info"]["issuer"] == "Cozy Test"

    cozy = [a for a in active["assertions"] if a["label"] == "com.cozy.generation"][0]["data"]
    assert cozy["request_sha256"] == hashlib.sha256(b"req-c2pa-1").hexdigest()
    assert cozy["models"] == ["ltx-video-cloud:prod"]
    # No PII / raw identifiers embedded anywhere in the manifest.
    assert "req-c2pa-1" not in json.dumps(active)

    # Asset metadata reflects the SIGNED bytes.
    signed = Path(asset.local_path).read_bytes()
    assert asset.size_bytes == len(signed)
    assert asset.sha256 == hashlib.sha256(signed).hexdigest()


def test_webp_and_jpeg_sign_via_save_image(tmp_path: Path, signing_on) -> None:
    ctx = _ctx(tmp_path)
    img = PIL_Image.new("RGB", (32, 32), (5, 5, 5))
    for fmt, mime in (("webp", "image/webp"), ("jpeg", "image/jpeg")):
        asset = ctx.save_image(img, f"outputs/img-{fmt}", format=fmt)
        active = _read_manifest(asset.local_path, mime)
        assert _actions(active)[0]["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA


def test_mp4_sign_verify_roundtrip_via_write_video(tmp_path: Path, signing_on) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    from gen_worker import io as gw_io

    ctx = _ctx(tmp_path)
    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    frames[:, 16:48, 16:48] = 200
    asset = gw_io.write_video(ctx, "outputs/clip", frames, fps=8.0)
    assert asset.local_path and asset.local_path.endswith(".mp4")

    active = _read_manifest(asset.local_path, "video/mp4")
    action = _actions(active)[0]
    assert action["action"] == "c2pa.created"
    assert action["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA
    cozy = [a for a in active["assertions"] if a["label"] == "com.cozy.generation"][0]["data"]
    assert cozy["request_sha256"] == hashlib.sha256(b"req-c2pa-1").hexdigest()


def test_save_file_signs_media_without_mutating_source(tmp_path: Path, signing_on) -> None:
    ctx = _ctx(tmp_path)
    img = PIL_Image.new("RGB", (16, 16), (1, 2, 3))
    src = tmp_path / "src.png"
    img.save(src, format="PNG")
    original = src.read_bytes()

    asset = ctx.save_file("outputs/copy.png", src)
    assert src.read_bytes() == original  # caller's file untouched
    active = _read_manifest(asset.local_path, "image/png")
    assert _actions(active)[0]["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA


def test_non_media_bytes_pass_through_unsigned(tmp_path: Path, signing_on) -> None:
    ctx = _ctx(tmp_path)
    payload = json.dumps({"result": "ok"}).encode()
    asset = ctx.save_bytes("outputs/result.json", payload)
    assert Path(asset.local_path).read_bytes() == payload


def test_unconfigured_is_noop_with_loud_warning(
    tmp_path: Path, signing_off, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="gen_worker.content_credentials"):
        content_credentials._reset_for_tests()
        content_credentials.configure(Settings())
    warned = [r for r in caplog.records if "C2PA" in r.getMessage()]
    assert warned and "DISABLED" in warned[0].getMessage()
    assert not content_credentials.enabled()

    ctx = _ctx(tmp_path)
    img = PIL_Image.new("RGB", (16, 16), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    asset = ctx.save_bytes("outputs/plain.png", buf.getvalue())
    # Bytes unchanged, and no embedded manifest.
    assert Path(asset.local_path).read_bytes() == buf.getvalue()
    with pytest.raises(Exception):
        with open(asset.local_path, "rb") as f:
            reader = c2pa.Reader("image/png", f)
            assert json.loads(reader.json())["manifests"] == {}


def test_configured_but_broken_fails_startup(tmp_path: Path, test_cert) -> None:
    content_credentials._reset_for_tests()
    try:
        with pytest.raises(content_credentials.C2paSigningError):
            content_credentials.configure(Settings(c2pa_cert_path="/nonexistent/cert.pem",
                                                   c2pa_key_path="/nonexistent/key.pem"))
        with pytest.raises(content_credentials.C2paSigningError):
            content_credentials.configure(Settings(c2pa_cert_path=test_cert["cert"]))
        # Garbage PEM fails the probe-signer construction.
        bad = tmp_path / "bad.pem"
        bad.write_text("not a pem")
        with pytest.raises(content_credentials.C2paSigningError):
            content_credentials.configure(
                Settings(c2pa_cert_path=str(bad), c2pa_key_path=str(bad))
            )
    finally:
        content_credentials._reset_for_tests()


def test_sniff_media_mime() -> None:
    sniff = content_credentials.sniff_media_mime
    assert sniff(b"\x89PNG\r\n\x1a\n", "x") == "image/png"
    assert sniff(b"\xff\xd8\xff\xe0" + b"\0" * 12, "x") == "image/jpeg"
    assert sniff(b"RIFF\0\0\0\0WEBPVP8 ", "x") == "image/webp"
    assert sniff(b"RIFF\0\0\0\0WAVEfmt ", "x") == "audio/wav"
    assert sniff(b"\0\0\0\x20ftypisom\0\0\0\0", "clip.mp4") == "video/mp4"
    assert sniff(b"\0\0\0\x14ftypqt  \0\0\0\0", "clip.mov") == "video/quicktime"
    assert sniff(b'{"a": 1}', "result.json") is None
    assert sniff(b"PK\x03\x04" + b"\0" * 12, "archive.zip") is None
    # safetensors header must never look like media
    assert sniff((8).to_bytes(8, "little") + b'{"a":"b"}', "model.safetensors") is None
