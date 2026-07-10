"""gw#425 (client half of tensorhub #561): capability-token renewal loop.

A real local HTTP server plays the hub's /v1/worker/capability/renew route;
the loop renews at ~80% TTL, swaps the stored token, and stops loudly on a
terminal denial.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

import pytest

from gen_worker.capability_renewal import (
    RenewDenied,
    renew_capability_while_running,
    renew_once,
)


def _jwt(claims: Dict) -> str:
    def _b64(obj: Dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{_b64({'alg': 'none'})}.{_b64(claims)}.sig"


class _RenewServer:
    def __init__(self, *, deny_with: int = 0) -> None:
        self.deny_with = deny_with
        self.requests: List[Dict] = []
        self.auth_headers: List[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:  # noqa: N802
                pass

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/worker/capability/renew":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.requests.append(json.loads(body))
                outer.auth_headers.append(self.headers.get("Authorization") or "")
                if outer.deny_with:
                    self.send_response(outer.deny_with)
                    self.end_headers()
                    return
                exp = int(time.time()) + 3600
                payload = json.dumps({
                    "capability_token": _jwt({"exp": exp, "iat": int(time.time()),
                                              "cap_kind": "worker_capability"}),
                    "expires_at_unix": exp,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_loop_renews_at_80pct_ttl_and_swaps_token() -> None:
    srv = _RenewServer()
    try:
        now = time.time()
        tokens = [_jwt({"exp": now + 3.0, "iat": now, "request_id": "r1"})]

        async def _go() -> None:
            task = asyncio.create_task(renew_capability_while_running(
                file_base_url=srv.base,
                request_id="r1",
                attempt=2,
                get_worker_jwt=lambda: "worker-jwt",
                get_token=lambda: tokens[-1],
                set_token=tokens.append,
            ))
            for _ in range(80):  # renewal due ~2.4s in
                if len(tokens) > 1:
                    break
                await asyncio.sleep(0.1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(_go())
        assert len(tokens) == 2, "token was not renewed"
        assert srv.requests == [{
            "request_id": "r1", "attempt": 2, "capability_token": tokens[0],
        }]
        assert srv.auth_headers == ["Bearer worker-jwt"]
        # New token verifies as a fresh long-TTL JWT.
        claims = json.loads(base64.urlsafe_b64decode(
            tokens[-1].split(".")[1] + "=="))
        assert claims["exp"] > time.time() + 3000
    finally:
        srv.close()


def test_loop_stops_on_terminal_denial_without_clobbering_token() -> None:
    srv = _RenewServer(deny_with=409)
    try:
        now = time.time()
        tokens = [_jwt({"exp": now + 2.0, "iat": now})]

        async def _go() -> None:
            await asyncio.wait_for(renew_capability_while_running(
                file_base_url=srv.base,
                request_id="r1",
                attempt=1,
                get_worker_jwt=lambda: "worker-jwt",
                get_token=lambda: tokens[-1],
                set_token=tokens.append,
            ), timeout=30.0)

        asyncio.run(_go())  # returns (does not loop) after the denial
        assert len(tokens) == 1
        assert len(srv.requests) == 1
    finally:
        srv.close()


def test_loop_is_a_noop_for_opaque_tokens() -> None:
    async def _go() -> None:
        await asyncio.wait_for(renew_capability_while_running(
            file_base_url="http://127.0.0.1:9",  # never contacted
            request_id="r1",
            attempt=1,
            get_worker_jwt=lambda: "worker-jwt",
            get_token=lambda: "not-a-jwt",
            set_token=lambda t: pytest.fail("must not renew"),
        ), timeout=5.0)

    asyncio.run(_go())


def test_renew_once_transient_failure_raises_runtime_error() -> None:
    srv = _RenewServer(deny_with=500)
    try:
        with pytest.raises(RuntimeError):
            renew_once(
                file_base_url=srv.base, worker_jwt="w", request_id="r1",
                attempt=1, capability_token="t",
            )
    finally:
        srv.close()


def test_renew_once_denial_raises_renew_denied() -> None:
    srv = _RenewServer(deny_with=403)
    try:
        with pytest.raises(RenewDenied):
            renew_once(
                file_base_url=srv.base, worker_jwt="w", request_id="r1",
                attempt=1, capability_token="t",
            )
    finally:
        srv.close()
