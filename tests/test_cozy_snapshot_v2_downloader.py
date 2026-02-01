import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from aiohttp import web
from blake3 import blake3

from gen_worker.cozy_snapshot_v2_downloader import CozySnapshotV2Downloader
from gen_worker.cozy_hub_v2 import CozyHubV2Client
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

    srv.loop.call_soon_threadsafe(lambda: asyncio.create_task(_stop()))
    srv.loop.call_soon_threadsafe(srv.loop.stop)


def test_snapshot_v2_downloader_materializes(tmp_path: Path) -> None:
    snap = "a" * 64

    b1 = b"hello"
    b1_digest = blake3(b1).hexdigest()
    b2 = b"world"
    b2_digest = blake3(b2).hexdigest()

    base = ""  # populated after server starts
    routes: Dict[str, web.StreamResponse] = {}

    async def resolve_artifact(req: web.Request) -> web.Response:
        body = await req.json()
        assert body["tag"] == "latest"
        return web.json_response(
            {
                "repo_revision_seq": 1,
                "snapshot_digest": snap,
                "artifact": {
                    "label": "safetensors-fp16",
                    "file_layout": "diffusers",
                    "file_type": "safetensors",
                    "quantization": "fp16",
                },
                "snapshot_manifest": {
                    "version": 1,
                    "files": [
                        {
                            "path": "cozy.pipeline.yaml",
                            "size_bytes": len(b1),
                            "blake3": b1_digest,
                            "url": f"{base}/files/pipeline",
                        },
                        {
                            "path": "unet/config.json",
                            "size_bytes": len(b2),
                            "blake3": b2_digest,
                            "url": f"{base}/files/unet_config",
                        },
                    ],
                },
            }
        )

    async def get_pipeline(_req: web.Request) -> web.Response:
        return web.Response(body=b1, headers={"Content-Type": "application/octet-stream"})

    async def get_unet_config(_req: web.Request) -> web.Response:
        return web.Response(body=b2, headers={"Content-Type": "application/octet-stream"})

    routes["/api/v1/repos/o/r/resolve_artifact"] = resolve_artifact  # type: ignore[assignment]
    routes["/files/pipeline"] = get_pipeline  # type: ignore[assignment]
    routes["/files/unet_config"] = get_unet_config  # type: ignore[assignment]

    srv = _start_server(routes)
    try:
        base = srv.base_url
        client = CozyHubV2Client(base_url=base)
        dl = CozySnapshotV2Downloader(client)
        ref = CozyRef(org="o", repo="r", tag="latest")
        local = asyncio.run(dl.ensure_snapshot(tmp_path, ref))
        assert (local / "cozy.pipeline.yaml").read_bytes() == b1
        assert (local / "unet" / "config.json").read_bytes() == b2
        blob1 = tmp_path / "cozy" / "blobs" / "blake3" / b1_digest[:2] / b1_digest[2:4] / b1_digest
        assert blob1.exists()
    finally:
        _stop_server(srv)

