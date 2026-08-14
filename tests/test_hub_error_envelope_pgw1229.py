"""pgw#1229: the hub's typed refusal survives all the way onto the wire.

Measured defect. Two production `dj-pipeline/make-video` invokes died `fatal`,
and the ENTIRE diagnosis that reached `request_state.error_message_safe` was:

    HTTPError: 403 Client Error: Forbidden for url: .../api/v1/media/urls

What the hub had actually sent (`internal/api/media_v1.go:461`) was a code and
a sentence naming the correct surface. `resp.raise_for_status()` threw both
away one stack frame from where they were needed.

The rig is a real HTTP server answering the hub's exact 403 envelope, and the
assertion is on the string `executor._map_exception` puts on the wire — not on
the raised exception, because the exception is not what a human reads hours
later with no pod logs. `test_raise_for_status_is_the_defect` is the control:
it pins what the old call still produces, so this file fails loudly if the two
paths are ever confused for each other.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest
import requests

from gen_worker.executor import _map_exception
from gen_worker.hub_error import (
    HubApiError,
    parse_hub_error,
    raise_for_hub_error,
)
from gen_worker.pb import worker_scheduler_pb2 as pb

HUB_CODE = "forbidden"
HUB_MESSAGE = "worker capabilities must use the exact input-asset resolver"
HUB_REQUEST_ID = "req_01J8ZQ"


class _Hub(BaseHTTPRequestHandler):
    """The routes this rig serves, each answering exactly like tensorhub."""

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/v1/media/urls":
            body = json.dumps({
                "error": {
                    "code": HUB_CODE,
                    "message": HUB_MESSAGE,
                    "request_id": HUB_REQUEST_ID,
                }
            }).encode()
            self._send(403, body, "application/json")
        elif self.path == "/no-body":
            self._send(403, b"", "application/json")
        elif self.path == "/proxy-outage":
            # ngrok with no healthy backend: HTML, not our envelope.
            self._send(503, b"<!DOCTYPE html><html>tunnel offline</html>",
                       "text/html")
        elif self.path == "/ok":
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b'{"error":{"code":"not_found"}}', "application/json")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:  # noqa: D102
        pass


@pytest.fixture(scope="module")
def hub() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Hub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_raise_for_status_is_the_defect(hub: str) -> None:
    """The control: what the shipped call site produced, verbatim."""
    resp = requests.post(f"{hub}/api/v1/media/urls", json={"refs": ["x"]}, timeout=10)
    with pytest.raises(requests.HTTPError) as caught:
        resp.raise_for_status()
    _status, safe = _map_exception(caught.value)
    assert HUB_CODE not in safe
    assert HUB_MESSAGE not in safe
    assert "403 Client Error" in safe


def test_typed_body_reaches_the_wire(hub: str) -> None:
    """The fix: code AND message in the exception, and in `safe_message`."""
    resp = requests.post(f"{hub}/api/v1/media/urls", json={"refs": ["x"]}, timeout=10)
    with pytest.raises(HubApiError) as caught:
        raise_for_hub_error(resp, what="presign input assets")
    exc = caught.value

    assert exc.status_code == 403
    assert exc.code == HUB_CODE
    assert exc.message == HUB_MESSAGE
    assert exc.request_id == HUB_REQUEST_ID

    status, safe = _map_exception(exc)
    # A hub-authored 403 is a refusal, not a blip: never retried.
    assert status == pb.JOB_STATUS_FATAL
    assert HUB_CODE in safe
    assert HUB_MESSAGE in safe
    assert "presign input assets" in safe
    # One line — `_map_exception`'s generic arm keeps only splitlines()[0], and
    # a remedy split across lines is a remedy half-delivered.
    assert "\n" not in safe


def test_absent_body_degrades_to_the_status_line(hub: str) -> None:
    """The failure mode of an error path is silence, never a second error.

    A body that is not the hub's envelope is treated as proxy-origin and so
    RETRYABLE — `http_origin`'s standing bias, not a judgement invented here:
    an unrecognised answer is more often a tunnel with no healthy backend than
    a verdict, and a retry costs a bounded backoff.
    """
    resp = requests.post(f"{hub}/no-body", timeout=10)
    with pytest.raises(HubApiError) as caught:
        raise_for_hub_error(resp, what="presign input assets")
    status, safe = _map_exception(caught.value)
    assert status == pb.JOB_STATUS_RETRYABLE
    assert "403" in safe
    assert "presign input assets" in safe


def test_proxy_outage_is_retryable(hub: str) -> None:
    """An answer that is not the hub's envelope came from in front of it."""
    resp = requests.post(f"{hub}/proxy-outage", timeout=10)
    with pytest.raises(HubApiError) as caught:
        raise_for_hub_error(resp, what="presign input assets")
    assert caught.value.retryable is True
    status, _safe = _map_exception(caught.value)
    assert status == pb.JOB_STATUS_RETRYABLE


def test_success_passes_through(hub: str) -> None:
    resp = requests.post(f"{hub}/ok", timeout=10)
    assert raise_for_hub_error(resp) is resp


@pytest.mark.parametrize(
    "body,code,message",
    [
        ('{"error":{"code":"not_found","message":"no such repo"}}',
         "not_found", "no such repo"),
        # pgw#987 publish/gin shape: the code is a bare string beside `message`.
        ('{"error":"publish_repudiated","message":"audit findings"}',
         "publish_repudiated", "audit findings"),
        # A bare token with no prose is still the code...
        ('{"error":"insufficient_scope"}', "insufficient_scope", ""),
        # ...but prose is not a code.
        ('{"error":"the body could not be parsed"}',
         "", "the body could not be parsed"),
        ("<!DOCTYPE html>", "", "<!DOCTYPE html>"),
        ("", "", ""),
    ],
)
def test_envelope_shapes(body: str, code: str, message: str) -> None:
    err = parse_hub_error(body)
    assert err.code == code
    assert err.message == message
