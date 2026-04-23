import asyncio
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import pytest
from aiohttp import web

from gen_worker.models.hub_client import CozyHubError, CozyHubNoCompatibleArtifactError, CozyHubPublicModelPendingError, CozyHubV2Client


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
    fut = asyncio.run_coroutine_threadsafe(srv.runner.cleanup(), srv.loop)
    fut.result(timeout=5)
    srv.loop.call_soon_threadsafe(srv.loop.stop)


def test_resolve_artifact_no_compatible() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.json_response({"error": "no_compatible_artifact", "debug": {"why": "missing_lib"}}, status=409)

    srv = _start_server({"/api/v1/inference/repos/o/r/resolve": handler})  # type: ignore[arg-type]
    try:
        client = CozyHubV2Client(srv.base_url)
        with pytest.raises(CozyHubNoCompatibleArtifactError) as e:
            asyncio.run(
                client.resolve_artifact(
                    owner="o",
                    repo="r",
                    tag="latest",
                    digest=None,
                    include_urls=False,
                    preferences={},
                    capabilities={},
                )
            )
        assert "no compatible artifact" in str(e.value).lower()
        assert isinstance(e.value.debug, dict)
    finally:
        _stop_server(srv)


def test_request_public_model_pending_202() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.json_response({"ingest_job_id": "sess1"}, status=202)

    srv = _start_server({"/api/v1/public/models/request": handler})  # type: ignore[arg-type]
    try:
        client = CozyHubV2Client(srv.base_url)
        with pytest.raises(CozyHubPublicModelPendingError) as e:
            asyncio.run(client.request_public_model(model_ref="hf:o/r@main", include_urls=True))
        assert e.value.ingest_job_id == "sess1"
    finally:
        _stop_server(srv)


def test_request_public_model_hf_missing_route_raises_cozy_error() -> None:
    async def handler(_req: web.Request) -> web.Response:
        return web.json_response({"error": "not_found"}, status=404)

    srv = _start_server({"/api/v1/public/models/request": handler})  # type: ignore[arg-type]
    try:
        client = CozyHubV2Client(srv.base_url)
        with pytest.raises(CozyHubError):
            asyncio.run(client.request_public_model(model_ref="hf:o/r@main", include_urls=True))
    finally:
        _stop_server(srv)
