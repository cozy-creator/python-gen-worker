"""Unified request cancellation (issue #352).

A cancel (control frame -> ``interrupt_request``) trips the canonical
``ctx.cancel()`` for ONE in-flight request; the handler observes it and unwinds,
the server keeps running and serves the next request. Driven at the
``_Endpoint`` level (deterministic, no socket timing) plus frame-parser and
client-canceler unit checks.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Iterator

import msgspec
import pytest

import gen_worker.cli.run as run_mod
import gen_worker.cli.serve as serve_mod
from gen_worker import RequestContext, inference


class _In(msgspec.Struct):
    text: str = ""


class _Out(msgspec.Struct):
    response: str


def _slow_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    @inference()
    class Slow:
        def setup(self) -> None:
            pass

        @inference.function(name="slow")
        def slow(self, ctx: RequestContext, data: _In) -> Iterator[_Out]:
            # Stream one item, then block-poll until canceled (the cooperative
            # idiom). raise_if_canceled turns the cancel into a CanceledError.
            yield _Out(response="started")
            while not ctx.is_canceled():
                time.sleep(0.005)
            ctx.raise_if_canceled()

    Slow.__module__ = name
    mod.Slow = Slow
    sys.modules[name] = mod
    return mod


def _dispatch_async(ep, fn, payload, rid):
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(env=ep.dispatch(fn, payload, request_id=rid)))
    t.start()
    return t, box


def test_interrupt_cancels_inflight_and_server_keeps_serving() -> None:
    mod = _slow_module("_cancel_slow")
    ep = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep.boot(run_mod.discover_candidates(mod))

    # First in-flight request.
    t1, box1 = _dispatch_async(ep, "slow", {"text": "x"}, "req-1")
    deadline = time.time() + 5
    while "req-1" not in ep._active and time.time() < deadline:
        time.sleep(0.005)
    assert "req-1" in ep._active  # registered while running

    assert ep.interrupt_request("req-1") is True
    t1.join(timeout=5)
    assert not t1.is_alive()
    assert box1["env"]["ok"] is False
    assert box1["env"]["error"]["kind"] == "canceled"
    assert "req-1" not in ep._active  # unregistered in finally

    # Server still serves a SECOND request (cancel was per-request).
    t2, box2 = _dispatch_async(ep, "slow", {"text": "y"}, "req-2")
    while "req-2" not in ep._active and time.time() < time.time() + 5:
        time.sleep(0.005)
    assert ep.interrupt_request("req-2") is True
    t2.join(timeout=5)
    assert box2["env"]["error"]["kind"] == "canceled"

    assert ep.interrupt_request("nope") is False  # unknown id
    ep.shutdown()


def test_parse_frame_request_and_cancel() -> None:
    import json

    req = serve_mod._parse_frame(json.dumps({"request_id": "a", "function": "f", "payload": {"x": 1}}).encode())
    assert req == {"kind": "request", "function": "f", "payload": {"x": 1}, "request_id": "a"}

    can = serve_mod._parse_frame(json.dumps({"cancel": {"request_id": "a"}}).encode())
    assert can == {"kind": "cancel", "request_id": "a"}

    bad = serve_mod._parse_frame(b"{not json")
    assert bad["kind"] == "error"

    nofn = serve_mod._parse_frame(json.dumps({"payload": {}}).encode())
    assert nofn["kind"] == "error"


def test_client_canceler_sends_cancel_frame(tmp_path) -> None:
    """The client canceler dials the socket and writes a cancel control frame
    for its request_id (server-side effect verified separately)."""
    import json
    import socket as _socket

    from gen_worker.cli.invoke import _ClientCanceler

    sock_path = tmp_path / "c.sock"
    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(1)
    srv.settimeout(5.0)

    got: dict = {}

    def _accept():
        conn, _ = srv.accept()
        data = conn.recv(4096)
        got["frame"] = json.loads(data.decode().splitlines()[0])
        conn.close()

    th = threading.Thread(target=_accept)
    th.start()

    c = _ClientCanceler(sock_path, "rid-123")
    c._send_cancel()
    th.join(timeout=5)
    srv.close()

    assert got["frame"] == {"cancel": {"request_id": "rid-123"}}
