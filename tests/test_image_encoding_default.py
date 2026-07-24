"""Paul's ruling (2026-07-24): image encoding defaults to WebP always, with
PNG and JPEG as first-class alternatives (th#1126 item 4 follow-up).

Real PIL round trips through the ONE encode core both surfaces now share
(``gw_io.encode_image``), plus ``ctx.save_image`` driven over a real
RequestContext and a real local media-upload sink — same pattern as
tests/test_p9_result_upload_metrics.py, no encode mocking.
"""

from __future__ import annotations

import base64
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, ClassVar, Dict, List, Tuple

import pytest

from gen_worker import io as gw_io
from gen_worker.api.errors import ValidationError

Image = pytest.importorskip("PIL.Image")


# ---- real encode core -------------------------------------------------------


def _img(mode: str = "RGB", size: Tuple[int, int] = (64, 48)) -> Any:
    img = Image.new(mode, size)
    # Non-uniform content so a lossy codec has something to lose.
    for x in range(size[0]):
        for y in range(size[1]):
            v = (x * 4 + y * 2) % 256
            img.putpixel((x, y), (v, 255 - v, (v * 3) % 256) if mode == "RGB"
                         else (v, 255 - v, (v * 3) % 256, 255) if mode == "RGBA"
                         else v)
    return img


def test_the_framework_default_is_webp() -> None:
    assert gw_io.DEFAULT_IMAGE_FORMAT == "webp"
    payload, ext = gw_io.encode_image(_img())
    assert ext == ".webp"
    # RIFF....WEBP container magic.
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"
    assert Image.open(BytesIO(payload)).size == (64, 48)


@pytest.mark.parametrize(
    "fmt,ext,magic",
    [
        ("webp", ".webp", b"RIFF"),
        ("png", ".png", b"\x89PNG\r\n\x1a\n"),
        ("jpg", ".jpg", b"\xff\xd8\xff"),
        ("jpeg", ".jpg", b"\xff\xd8\xff"),
        ("WEBP", ".webp", b"RIFF"),
        ("  PNG  ", ".png", b"\x89PNG\r\n\x1a\n"),
    ],
)
def test_every_supported_format_decodes_with_the_right_magic_and_extension(
    fmt: str, ext: str, magic: bytes
) -> None:
    payload, got_ext = gw_io.encode_image(_img(), format=fmt)
    assert got_ext == ext
    assert payload.startswith(magic)
    decoded = Image.open(BytesIO(payload))
    decoded.load()
    assert decoded.size == (64, 48)


def test_jpeg_of_a_transparent_image_converts_instead_of_exploding() -> None:
    """PIL raises OSError on RGBA->JPEG; the core converts to RGB first."""
    payload, ext = gw_io.encode_image(_img("RGBA"), format="jpg")
    assert ext == ".jpg"
    decoded = Image.open(BytesIO(payload))
    decoded.load()
    assert decoded.mode == "RGB"
    # Palette mode too (the other JPEG-hostile mode).
    payload, _ = gw_io.encode_image(_img("RGB").convert("P"), format="jpeg")
    assert payload.startswith(b"\xff\xd8\xff")


def test_unknown_format_raises_a_typed_error_not_a_pil_traceback() -> None:
    with pytest.raises(ValidationError) as exc:
        gw_io.encode_image(_img(), format="tiff")
    assert "tiff" in str(exc.value)
    assert "webp" in str(exc.value), "the error must name the supported formats"


def test_webp_lossless_is_reachable_and_pixel_exact() -> None:
    src = _img()
    lossy, _ = gw_io.encode_image(src, format="webp", quality=95)
    lossless, _ = gw_io.encode_image(src, format="webp", lossless=True)
    assert list(Image.open(BytesIO(lossless)).convert("RGB").getdata()) == \
        list(src.getdata()), "lossless=True must round-trip exactly"
    assert list(Image.open(BytesIO(lossy)).convert("RGB").getdata()) != \
        list(src.getdata()), "the default must actually be the lossy encoder"


def test_quality_reaches_the_encoder() -> None:
    small, _ = gw_io.encode_image(_img(), format="webp", quality=10)
    big, _ = gw_io.encode_image(_img(), format="webp", quality=100)
    assert len(small) < len(big)


def test_both_encode_surfaces_share_one_default(monkeypatch: Any) -> None:
    """The 90-vs-95 drift that motivated this lane must not come back."""
    from gen_worker.request_context import RequestContext

    for fn in (gw_io.write_image, RequestContext.save_image):
        params = inspect.signature(fn).parameters
        assert params["format"].default == gw_io.DEFAULT_IMAGE_FORMAT == "webp"
        assert params["quality"].default == gw_io.DEFAULT_IMAGE_QUALITY == 95

    # ...and they must route through the same core object, not their own copy.
    import gen_worker.request_context as rc

    assert rc.encode_image is gw_io.encode_image

    calls: List[Dict[str, Any]] = []
    real = gw_io.encode_image

    def _spy(image: Any, **kw: Any) -> Tuple[bytes, str]:
        calls.append(kw)
        return real(image, **kw)

    monkeypatch.setattr(gw_io, "encode_image", _spy)

    class _Ctx:
        def save_bytes(self, ref: str, data: bytes) -> Dict[str, Any]:
            return {"ref": ref, "size": len(data)}

    gw_io.write_image(_Ctx(), "out", _img())
    assert calls == [{"format": "webp", "quality": 95}]


def test_write_image_derives_the_extension_when_the_ref_has_none() -> None:
    """A webp payload stored under a bare `image` ref defeats mime inference."""
    saved: List[str] = []

    class _Ctx:
        def save_bytes(self, ref: str, data: bytes) -> Dict[str, Any]:
            saved.append(ref)
            return {"ref": ref, "size": len(data)}

    gw_io.write_image(_Ctx(), "image", _img())
    gw_io.write_image(_Ctx(), "image", _img(), format="jpg")
    gw_io.write_image(_Ctx(), "already.png", _img(), format="png")
    assert saved == ["image.webp", "image.jpg", "already.png"]


# ---- ctx.save_image over a real context + real upload sink ------------------


class _UploadSink(BaseHTTPRequestHandler):
    """Real stand-in for tensorhub's /api/v1/media/:owner/uploads (dedup
    response, so no S3 part PUT scripting is needed)."""

    requests_seen: ClassVar[List[Dict[str, Any]]] = []

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests_seen.append(body)
        resp = json.dumps({
            "dedup": True, "ref": body.get("ref") or "",
            "filename": (body.get("ref") or "").rsplit("/", 1)[-1],
            "blake3": body.get("blake3") or "",
            "size_bytes": body.get("size_bytes") or 0,
            "mime_type": "application/octet-stream", "media_id": "m1",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def _unsigned_jwt(claims: Dict[str, Any]) -> str:
    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


def test_ctx_save_image_defaults_to_webp_over_the_real_upload_path() -> None:
    from gen_worker import RequestContext

    owner = "019f4c33-f3a5-705b-9848-0b3b0863c416"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _UploadSink)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        ctx = RequestContext(
            request_id="r-img", owner="tensorhub", file_api_base_url=base_url,
            worker_capability_token=_unsigned_jwt(
                {"tenant": owner, "request_id": "r-img"}),
        )
        default = ctx.save_image(_img())
        assert default.ref == "outputs/r-img/image.webp", (
            "the default output ref must carry the webp extension")

        png = ctx.save_image(_img(), "outputs/r-img/explicit", format="png")
        jpg = ctx.save_image(_img("RGBA"), "outputs/r-img/explicit", format="jpg")
        assert png.ref.endswith(".png") and jpg.ref.endswith(".jpg")

        sizes = {b["ref"]: b["size_bytes"] for b in _UploadSink.requests_seen}
        assert sizes["outputs/r-img/image.webp"] > 0
        assert sizes["outputs/r-img/explicit.png"] > 0
        assert len(_UploadSink.requests_seen) == 3, "every save must really upload"

        with pytest.raises(ValidationError):
            ctx.save_image(_img(), format="gif")
    finally:
        httpd.shutdown()
        _UploadSink.requests_seen = []
