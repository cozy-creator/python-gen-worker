from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from PIL import Image

from gen_worker import Asset, ImageAsset, InputTooLargeError, ValidationError
from gen_worker.worker import Worker


class _Payload(msgspec.Struct):
    image: ImageAsset


def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def http_url() -> Any:
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self._send(send_body=False)

        def do_GET(self) -> None:
            self._send(send_body=True)

        def _send(self, *, send_body: bool) -> None:
            status, headers, body = routes.get(
                self.path,
                (404, {"Content-Type": "text/plain"}, b"missing"),
            )
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def make_url(path: str, body: bytes, content_type: str = "application/octet-stream") -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        routes[path] = (200, {"Content-Type": content_type}, body)
        return f"http://127.0.0.1:{server.server_port}{path}"

    make_url.routes = routes  # type: ignore[attr-defined]
    yield make_url
    server.shutdown()
    thread.join(timeout=5)


def _worker() -> Worker:
    w = Worker.__new__(Worker)
    w.owner = "tester"
    return w


def _ctx(request_id: str = "req") -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, job_id=None, owner="tester")


def test_approved_image_url_materializes_to_local_path(monkeypatch: pytest.MonkeyPatch, http_url: Any) -> None:
    monkeypatch.setattr("gen_worker.worker._url_is_blocked", lambda url: False)
    url = http_url("/image.png", _png_bytes(), "image/png")
    asset = ImageAsset(ref=url, url_allowed_mime_types=("image/png",), url_max_pixels=16)

    _worker()._materialize_asset(_ctx("approved-image"), asset)

    assert asset.local_path
    assert asset.owner == "tester"
    assert asset.mime_type == "image/png"
    assert asset.size_bytes and asset.size_bytes > 0
    assert asset.sha256 and len(asset.sha256) == 64


def test_media_specific_asset_recurses_like_asset(monkeypatch: pytest.MonkeyPatch, http_url: Any) -> None:
    monkeypatch.setattr("gen_worker.worker._url_is_blocked", lambda url: False)
    payload = _Payload(image=ImageAsset(ref=http_url("/nested.png", _png_bytes(), "image/png")))

    _worker()._materialize_assets(_ctx("nested"), payload)

    assert payload.image.local_path


def test_redirect_to_private_target_is_rejected(monkeypatch: pytest.MonkeyPatch, http_url: Any) -> None:
    routes = http_url.routes
    url = http_url("/start", b"", "text/plain")
    private_url = url.rsplit("/", 1)[0] + "/private"
    routes["/start"] = (302, {"Location": private_url}, b"")
    routes["/private"] = (200, {"Content-Type": "image/png"}, _png_bytes())
    monkeypatch.setattr(
        "gen_worker.worker._url_is_blocked",
        lambda value: str(value).endswith("/private"),
    )

    with pytest.raises(ValidationError, match="redirect target blocked"):
        _worker()._materialize_asset(_ctx("redirect"), ImageAsset(ref=url))


def test_too_large_streaming_download_is_rejected(monkeypatch: pytest.MonkeyPatch, http_url: Any) -> None:
    monkeypatch.setattr("gen_worker.worker._url_is_blocked", lambda url: False)
    asset = ImageAsset(ref=http_url("/large.png", _png_bytes(), "image/png"), url_max_bytes=4)

    with pytest.raises(InputTooLargeError):
        _worker()._materialize_asset(_ctx("large"), asset)


def test_mime_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch, http_url: Any) -> None:
    monkeypatch.setattr("gen_worker.worker._url_is_blocked", lambda url: False)
    asset = ImageAsset(ref=http_url("/text.txt", b"not an image", "text/plain"))

    with pytest.raises(ValidationError, match="not valid for ImageAsset"):
        _worker()._materialize_asset(_ctx("mime"), asset)


def test_oversized_image_dimensions_are_rejected(monkeypatch: pytest.MonkeyPatch, http_url: Any) -> None:
    monkeypatch.setattr("gen_worker.worker._url_is_blocked", lambda url: False)
    asset = ImageAsset(ref=http_url("/wide.png", _png_bytes(width=8, height=2), "image/png"), url_max_width=4)

    with pytest.raises(InputTooLargeError, match="input image width"):
        _worker()._materialize_asset(_ctx("dims"), asset)


def test_cache_key_includes_download_token_and_validation_context(
    monkeypatch: pytest.MonkeyPatch,
    http_url: Any,
) -> None:
    monkeypatch.setattr("gen_worker.worker._url_is_blocked", lambda url: False)
    url = http_url("/token.png", _png_bytes(), "image/png")
    first = Asset(ref=url, download_token="token-a", url_validation_context="ctx")
    second = Asset(ref=url, download_token="token-b", url_validation_context="ctx")
    third = Asset(ref=url, download_token="token-a", url_validation_context="other")
    w = _worker()

    w._materialize_asset(_ctx("cache"), first)
    w._materialize_asset(_ctx("cache"), second)
    w._materialize_asset(_ctx("cache"), third)

    assert first.local_path != second.local_path
    assert first.local_path != third.local_path
