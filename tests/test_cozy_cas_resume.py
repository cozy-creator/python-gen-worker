from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pytest
from aiohttp import web
from blake3 import blake3

from gen_worker.cozy_cas import _download_one_file


@dataclass
class _Server:
    base_url: str
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    runner: web.AppRunner


def _start_server(handler: web.StreamResponse) -> _Server:
    loop = asyncio.new_event_loop()
    runner: Optional[web.AppRunner] = None
    base_url: Dict[str, str] = {}

    def run() -> None:
        nonlocal runner
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            nonlocal runner
            app = web.Application()
            app.router.add_route("*", "/file", handler)  # type: ignore[arg-type]
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = list(site._server.sockets)[0].getsockname()[1]  # type: ignore[attr-defined]
            base_url["v"] = f"http://127.0.0.1:{port}"

        loop.run_until_complete(_run())
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while "v" not in base_url:
        pass
    assert runner is not None
    return _Server(base_url=base_url["v"], thread=t, loop=loop, runner=runner)


def _stop_server(srv: _Server) -> None:
    fut = asyncio.run_coroutine_threadsafe(srv.runner.cleanup(), srv.loop)
    fut.result(timeout=5)
    srv.loop.call_soon_threadsafe(srv.loop.stop)
    srv.thread.join(timeout=5)


def _parse_range(h: str) -> int:
    # Very small parser for "bytes=<start>-"
    _, _, rest = h.partition("=")
    start_s, _, _ = rest.partition("-")
    return int(start_s)


def test_cozy_cas_resume_with_range(tmp_path: Path) -> None:
    data = (b"0123456789abcdef" * 65536)  # 1MB-ish
    expected_size = len(data)
    expected_b3 = blake3(data).hexdigest()

    state = {"calls": 0, "saw_range": False}

    async def handler(req: web.Request) -> web.StreamResponse:
        state["calls"] += 1
        rng = req.headers.get("Range")
        if rng:
            state["saw_range"] = True
            start = _parse_range(rng)
            body = data[start:]
            resp = web.Response(status=206, body=body)
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Range"] = f"bytes {start}-{len(data)-1}/{len(data)}"
            return resp

        # First request: return a truncated response so the client has to resume.
        if state["calls"] == 1:
            return web.Response(status=200, body=data[:200_000], headers={"Accept-Ranges": "bytes"})
        return web.Response(status=200, body=data, headers={"Accept-Ranges": "bytes"})

    srv = _start_server(handler)  # type: ignore[arg-type]
    try:
        dst = tmp_path / "out.bin"
        asyncio.run(_download_one_file(f"{srv.base_url}/file", dst, expected_size=expected_size, expected_blake3=expected_b3))
        assert dst.read_bytes() == data
        assert state["saw_range"] is True
    finally:
        _stop_server(srv)


def test_cozy_cas_resume_falls_back_if_no_range_support(tmp_path: Path) -> None:
    data = b"x" * 1024 * 256
    expected_size = len(data)
    expected_b3 = blake3(data).hexdigest()

    state = {"saw_range": False}

    async def handler(req: web.Request) -> web.StreamResponse:
        if req.headers.get("Range"):
            state["saw_range"] = True
        # Ignore Range and always return 200 with full body
        return web.Response(status=200, body=data)

    srv = _start_server(handler)  # type: ignore[arg-type]
    try:
        dst = tmp_path / "out.bin"
        # Create a partial .part file to force resume logic path.
        part = dst.with_suffix(dst.suffix + ".part")
        part.write_bytes(data[:12345])

        asyncio.run(_download_one_file(f"{srv.base_url}/file", dst, expected_size=expected_size, expected_blake3=expected_b3))
        assert dst.read_bytes() == data
        assert state["saw_range"] is True
    finally:
        _stop_server(srv)
