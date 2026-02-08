import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pytest
from aiohttp import web
from blake3 import blake3

from gen_worker.cozy_cas import CozyHubClient, CozySnapshotDownloader
from gen_worker.model_refs import CozyRef


@dataclass
class _Server:
    base_url: str
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    runner: web.AppRunner


def _start_server(routes: Dict[str, web.StreamResponse]) -> _Server:
    loop = asyncio.new_event_loop()

    async def make_app() -> web.Application:
        app = web.Application()
        for path, handler in routes.items():
            app.router.add_route("*", path, handler)
        return app

    runner: Optional[web.AppRunner] = None
    site: Optional[web.TCPSite] = None
    base_url: Dict[str, str] = {}

    def run() -> None:
        nonlocal runner, site
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            nonlocal runner, site
            app = await make_app()
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            # fetch port
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
    async def _stop() -> None:
        await srv.runner.cleanup()

    srv.loop.call_soon_threadsafe(srv.loop.stop)
    srv.loop.call_soon_threadsafe(lambda: asyncio.create_task(_stop()))


def _json(data: dict) -> web.StreamResponse:
    async def handler(_req: web.Request) -> web.Response:
        return web.json_response(data)

    return handler  # type: ignore[return-value]


def _bytes(data: bytes) -> web.StreamResponse:
    async def handler(_req: web.Request) -> web.Response:
        return web.Response(body=data, headers={"Content-Type": "application/octet-stream"})

    return handler  # type: ignore[return-value]


@pytest.mark.parametrize("use_tag", [True, False])
def test_cozy_snapshot_downloader_materializes(tmp_path: Path, use_tag: bool) -> None:
    obj_digest = "obj1"
    snap_digest = "snap1"

    config_bytes = b'{"foo":"bar"}\n'
    config_b3 = blake3(config_bytes).hexdigest()

    model_index_bytes = b'{"_class_name":"X"}\n'
    model_index_b3 = blake3(model_index_bytes).hexdigest()

    base = ""  # filled after server starts
    routes: Dict[str, web.StreamResponse] = {}

    # resolve endpoint
    routes["/api/v1/repos/o/r/resolve"] = _json({"digest": snap_digest})

    # snapshot manifest endpoint
    def snapshot_manifest() -> dict:
        return {
            "snapshot_digest": snap_digest,
            "objects": {"transformer": obj_digest},
            "files": [
                {
                    "path": "model_index.json",
                    "size_bytes": len(model_index_bytes),
                    "blake3": model_index_b3,
                    "url": f"{base}/files/model_index.json",
                }
            ],
        }

    async def snapshot_handler(_req: web.Request) -> web.Response:
        return web.json_response(snapshot_manifest())

    routes[f"/api/v1/repos/o/r/snapshots/{snap_digest}/manifest"] = snapshot_handler  # type: ignore[assignment]

    # object manifest
    def object_manifest() -> dict:
        return {
            "object_digest": obj_digest,
            "files": [
                {
                    "path": "config.json",
                    "size_bytes": len(config_bytes),
                    "blake3": config_b3,
                    "url": f"{base}/files/config.json",
                }
            ],
        }

    async def obj_handler(_req: web.Request) -> web.Response:
        return web.json_response(object_manifest())

    routes[f"/api/v1/objects/{obj_digest}/manifest"] = obj_handler  # type: ignore[assignment]

    # file bytes
    routes["/files/config.json"] = _bytes(config_bytes)
    routes["/files/model_index.json"] = _bytes(model_index_bytes)

    srv = _start_server(routes)
    try:
        base = srv.base_url
        client = CozyHubClient(base_url=base)
        dl = CozySnapshotDownloader(client)

        if use_tag:
            ref = CozyRef(owner="o", repo="r", tag="latest", digest=None)
        else:
            ref = CozyRef(owner="o", repo="r", tag="latest", digest=snap_digest)

        local = asyncio.run(dl.ensure_snapshot(tmp_path, ref))
        assert (local / "transformer" / "config.json").read_bytes() == config_bytes
        assert (local / "model_index.json").read_bytes() == model_index_bytes
        assert (tmp_path / "cozy" / "objects" / obj_digest / ".cozy-object.json").exists()
    finally:
        _stop_server(srv)
