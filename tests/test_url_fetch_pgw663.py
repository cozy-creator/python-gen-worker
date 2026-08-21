from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

import pytest

from gen_worker import url_fetch
from gen_worker.api.errors import ValidationError


class _Origin(BaseHTTPRequestHandler):

    state: dict[str, Any] = {}

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        st = _Origin.state
        st.setdefault("paths", []).append(self.path)
        if self.path.startswith("/redirect-to-"):
            target = st["targets"][self.path.rsplit("-", 1)[-1]]
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/no-length":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * 5000)
            return
        if st.get("status"):
            self.send_error(int(st["status"]))
            return
        payload = st.get("body", b"hello")
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def origin():
    _Origin.state = {"targets": {}}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Origin)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, _Origin.state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def public(monkeypatch):
    """Treat every 127.0.0.1 URL as public EXCEPT ones marked ``blocked``."""
    def _blocked(url: str) -> bool:
        return "blocked" in url

    monkeypatch.setattr(url_fetch, "_url_is_blocked", _blocked)
    return _blocked


def test_plain_fetch(origin, public) -> None:
    base, state = origin
    state["body"] = b"a caption input"
    got = url_fetch.fetch_bytes(f"{base}/ok")
    assert got.data == b"a caption input"
    assert got.size_bytes == 15


def test_redirect_into_a_blocked_destination_is_refused(origin, public) -> None:
    """THE bug this issue exists for."""
    base, state = origin
    state["targets"]["meta"] = f"{base}/blocked-metadata"
    with pytest.raises(ValidationError) as exc:
        url_fetch.fetch_bytes(f"{base}/redirect-to-meta")
    assert "redirect hop 1" in str(exc.value)
    assert not any("blocked-metadata" in p for p in state["paths"])


def test_a_permitted_redirect_still_works(origin, public) -> None:
    base, state = origin
    state["body"] = b"redirected"
    state["targets"]["ok"] = f"{base}/final"
    assert url_fetch.fetch_bytes(f"{base}/redirect-to-ok").data == b"redirected"


def test_redirect_loops_terminate(origin, public) -> None:
    base, state = origin
    state["targets"]["loop"] = f"{base}/redirect-to-loop"
    with pytest.raises(ValidationError, match="redirects"):
        url_fetch.fetch_bytes(f"{base}/redirect-to-loop", max_redirects=2)


def test_non_http_scheme_is_refused(public) -> None:
    with pytest.raises(ValidationError, match="http"):
        url_fetch.fetch_bytes("file:///etc/passwd")


def test_blocked_host_is_refused(origin, public) -> None:
    base, _ = origin
    with pytest.raises(ValidationError, match="non-public address"):
        url_fetch.fetch_bytes(f"{base}/blocked")


def test_caller_allowlist(origin, public) -> None:
    base, state = origin
    state["body"] = b"ok"
    with pytest.raises(ValidationError, match="caller's allowlist"):
        url_fetch.fetch_bytes(f"{base}/ok", allowed_hosts=("cdn.example.com",))
    assert url_fetch.fetch_bytes(f"{base}/ok", allowed_hosts=("127.0.0.1",)).data == b"ok"


def test_deployment_allowlist_is_an_outer_bound(origin, public, monkeypatch) -> None:
    base, state = origin
    state["body"] = b"ok"
    monkeypatch.setenv(url_fetch.ALLOWED_HOSTS_ENV, "cdn.example.com")
    with pytest.raises(ValidationError, match="allowlist"):
        url_fetch.fetch_bytes(f"{base}/ok", allowed_hosts=("127.0.0.1",))


def test_declared_oversize_is_refused_before_reading(origin, public) -> None:
    base, state = origin
    state["body"] = b"x" * 4096
    with pytest.raises(ValidationError, match="declares"):
        url_fetch.fetch_bytes(f"{base}/big", max_bytes=100)


def test_a_body_with_no_declared_length_is_still_capped(origin, public) -> None:
    """The cap cannot depend on the server declaring a size — the per-chunk accounting is what actually bounds the read."""
    base, _ = origin
    with pytest.raises(ValidationError, match="exceeds its"):
        url_fetch.fetch_bytes(f"{base}/no-length", max_bytes=1000)


def test_an_error_status_is_a_typed_refusal_for_callers(origin, public) -> None:
    """`fetch_bytes` maps a bad status to a caller refusal — but the OPENER re-raises it untouched, so `input_assets` keeps its own (retryable) reading of a 404 on an authorized transport."""
    import urllib.error

    base, state = origin
    state["status"] = 404
    with pytest.raises(ValidationError, match="HTTP 404"):
        url_fetch.fetch_bytes(f"{base}/gone")
    with pytest.raises(urllib.error.HTTPError):
        with url_fetch.open_guarded_stream(f"{base}/gone"):
            pass


def test_mime_allowlist_uses_the_sniffed_type(origin, public) -> None:
    base, state = origin
    state["body"] = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    assert url_fetch.fetch_bytes(f"{base}/x.png", allowed_mime_types=("image/png",))
    with pytest.raises(ValidationError, match="content type"):
        url_fetch.fetch_bytes(f"{base}/x.png", allowed_mime_types=("image/jpeg",))


