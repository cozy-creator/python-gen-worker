"""The hub CONTROL plane keeps its connection across saves; the R2 DATA plane still does not."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import presigned_upload
from gen_worker.api.errors import ArtifactTransferError
from gen_worker.presigned_upload import presigned_upload_file, reset_control_plane_sessions


class _CountingServer(ThreadingHTTPServer):

    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.connections = 0
        self.requests_seen: List[str] = []
        self.cookie_headers: List[Optional[str]] = []
        self.lock = threading.Lock()
        self.poison_after_response = False
        self.drop_create_requests = 0
        self.set_cookie = False

    def get_request(self) -> Any:
        conn = super().get_request()
        with self.lock:
            self.connections += 1
        return conn


class _HubHandler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: Any) -> None:
        pass

    @property
    def hub(self) -> _CountingServer:
        return self.server  # type: ignore[return-value]

    def _json(self, code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if self.hub.set_cookie:
            self.send_header("Set-Cookie", "hubstate=1; Path=/")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        leg = "complete" if self.path.endswith("/complete") else "create"
        with self.hub.lock:
            self.hub.requests_seen.append(leg)
            self.hub.cookie_headers.append(self.headers.get("Cookie"))
            drop = leg == "create" and self.hub.drop_create_requests > 0
            if drop:
                self.hub.drop_create_requests -= 1
        if drop:
            self.close_connection = True
            return
        if leg == "complete":
            self._json(200, {
                "ref": "outputs/r/abc.webp",
                "media_id": "m1",
                "blake3": "",
                "sha256": "",
                "size_bytes": 0,
                "mime_type": "image/webp",
            })
        else:
            self._json(201, {
                "upload_id": "u1",
                "put_url": f"{self.hub.r2_base}/final/abc.webp",  # type: ignore[attr-defined]
                "put_headers": {"x-amz-checksum-sha256": "AAA="},
            })
        if self.hub.poison_after_response:
            self.close_connection = True


class _R2Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a: Any) -> None:
        pass

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self.send_response(200)
        self.send_header("ETag", '"deadbeef"')
        self.send_header("Content-Length", "0")
        self.end_headers()


def _serve(handler: Any) -> Tuple[_CountingServer, str]:
    httpd = _CountingServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture()
def stack(tmp_path):
    """A hub and an R2, both counting connections, plus a file to save."""
    reset_control_plane_sessions()
    hub, hub_base = _serve(_HubHandler)
    r2, r2_base = _serve(_R2Handler)
    hub.r2_base = r2_base  # type: ignore[attr-defined]
    src = tmp_path / "out.webp"
    src.write_bytes(b"x" * 4096)

    def save() -> Any:
        return presigned_upload_file(
            file_path=str(src),
            base_url=hub_base,
            endpoint_path="/api/v1/media/o/uploads",
            headers={"Authorization": "Bearer t"},
            create_payload={"ref": "out.webp", "sha256": "0" * 64},
            blake3_hex="0" * 64,
            size_bytes=4096,
        )

    try:
        yield hub, r2, hub_base, save
    finally:
        reset_control_plane_sessions()
        hub.shutdown()
        r2.shutdown()


def test_saves_share_one_hub_connection_and_never_share_the_r2_pool(stack) -> None:
    """The whole point, and its guardrail, in one assertion pair."""
    hub, r2, _base, save = stack
    for _ in range(3):
        assert save().dedup is False

    assert hub.requests_seen.count("create") == 3
    assert hub.requests_seen.count("complete") == 3
    assert hub.connections == 1, (
        f"control plane opened {hub.connections} connections for 3 saves — "
        "the process-scoped session is not being reused"
    )
    assert r2.connections == 3, (
        f"data plane opened {r2.connections} connections for 3 saves — the R2 "
        "pool must stay SAVE-scoped (issue #13, SSLV3_ALERT_BAD_RECORD_MAC)"
    )


def test_a_dead_keepalive_socket_retries_once_instead_of_failing_the_save(stack) -> None:
    """The failure mode reuse buys: the money path must survive it."""
    hub, _r2, _base, save = stack
    assert save().dedup is False

    hub.drop_create_requests = 1
    assert save().dedup is False

    assert hub.requests_seen.count("create") == 3
    assert hub.connections == 2


def test_a_hub_that_hangs_up_after_answering_costs_one_reconnect_not_a_failure(stack) -> None:
    """The realistic shape of staleness: the peer FINs an idle socket while the pod denoises."""
    hub, _r2, _base, save = stack
    hub.poison_after_response = True
    for _ in range(3):
        assert save().dedup is False
    assert hub.requests_seen.count("create") == 3
    assert hub.requests_seen.count("complete") == 3


def test_a_fresh_session_does_not_invent_a_retry_create_never_had(stack) -> None:
    """The retry is scoped to STALENESS, not to "the hub is down"."""
    hub, _r2, _base, save = stack
    hub.drop_create_requests = 99

    with pytest.raises(ArtifactTransferError) as first:
        save()
    assert first.value.phase == "create"
    assert hub.requests_seen.count("create") == 1, "a fresh session must not retry"

    with pytest.raises(ArtifactTransferError):
        save()
    assert hub.requests_seen.count("create") == 3, "a reused session retries exactly once"


def test_the_keepalive_session_carries_sockets_only_never_server_state(stack) -> None:
    """A process-scoped session that accumulated cookies would leak one save's server state into the next request's identity."""
    hub, _r2, _base, save = stack
    hub.set_cookie = True
    save()
    save()
    assert all(c is None for c in hub.cookie_headers), hub.cookie_headers


def test_eviction_is_by_identity_so_a_sibling_replacement_survives() -> None:
    """Two threads can fail on the same socket at once."""
    reset_control_plane_sessions()
    try:
        first, fresh = presigned_upload.control_plane_session("http://127.0.0.1:9/x")
        assert fresh is True
        again, fresh_again = presigned_upload.control_plane_session("http://127.0.0.1:9/y")
        assert again is first and fresh_again is False

        presigned_upload._evict_control_session("http://127.0.0.1:9/x", first)
        replacement, _ = presigned_upload.control_plane_session("http://127.0.0.1:9/x")
        assert replacement is not first

        presigned_upload._evict_control_session("http://127.0.0.1:9/x", first)
        still, _ = presigned_upload.control_plane_session("http://127.0.0.1:9/x")
        assert still is replacement
    finally:
        reset_control_plane_sessions()


def test_sessions_are_per_origin_so_one_dead_hub_does_not_evict_another() -> None:
    reset_control_plane_sessions()
    try:
        a, _ = presigned_upload.control_plane_session("http://hub-a:3867/api")
        b, _ = presigned_upload.control_plane_session("http://hub-b:3867/api")
        assert a is not b
        presigned_upload._evict_control_session("http://hub-a:3867/api", a)
        assert presigned_upload.control_plane_session("http://hub-b:3867/api")[0] is b
    finally:
        reset_control_plane_sessions()


def test_reuse_does_not_slow_the_second_save(stack) -> None:
    """Sanity, not a benchmark: the reused-socket save must not be SLOWER than the first."""
    _hub, _r2, _base, save = stack
    t0 = time.monotonic()
    save()
    first = time.monotonic() - t0
    t1 = time.monotonic()
    save()
    second = time.monotonic() - t1
    assert second <= first + 0.25
