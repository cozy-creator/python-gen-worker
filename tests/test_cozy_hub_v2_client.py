import asyncio
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import pytest
from aiohttp import web

from gen_worker.cozy_hub_v2 import CozyHubNoCompatibleArtifactError, CozyHubV2Client


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

    srv.loop.call_soon_threadsafe(srv.loop.stop)
    srv.loop.call_soon_threadsafe(lambda: asyncio.create_task(_stop()))


def test_resolve_artifact_success() -> None:
    async def handler(req: web.Request) -> web.Response:
        body = await req.json()
        assert body["tag"] == "latest"
        assert body["include_urls"] is True
        return web.json_response(
            {
                "repo_revision_seq": 1,
                "snapshot_digest": "a" * 64,
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
                            "size_bytes": 123,
                            "blake3": "b" * 64,
                            "url": "http://example.test/file",
                        }
                    ],
                },
            }
        )

    srv = _start_server({"/api/v1/repos/o/r/resolve_artifact": handler})  # type: ignore[arg-type]
    try:
        client = CozyHubV2Client(srv.base_url)
        res = asyncio.run(
            client.resolve_artifact(
                owner="o",
                repo="r",
                tag="latest",
                include_urls=True,
                preferences={"file_type_preference": ["safetensors"]},
                capabilities={"installed_libs": []},
            )
        )
        assert res.repo_revision_seq == 1
        assert res.snapshot_digest == "a" * 64
        assert res.artifact.label == "safetensors-fp16"
        assert res.files[0].url is not None
    finally:
        _stop_server(srv)


def test_resolve_artifact_no_compatible() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.json_response({"error": "no_compatible_artifact", "debug": {"why": "missing_lib"}}, status=409)

    srv = _start_server({"/api/v1/repos/o/r/resolve_artifact": handler})  # type: ignore[arg-type]
    try:
        client = CozyHubV2Client(srv.base_url)
        with pytest.raises(CozyHubNoCompatibleArtifactError) as e:
            asyncio.run(
                client.resolve_artifact(
                    owner="o",
                    repo="r",
                    tag="latest",
                    include_urls=False,
                    preferences={},
                    capabilities={},
                )
            )
        assert "no compatible artifact" in str(e.value).lower()
        assert isinstance(e.value.debug, dict)
    finally:
        _stop_server(srv)
