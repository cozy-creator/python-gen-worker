from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web
import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.worker import ActionContext, Worker


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    man = json.loads(raw)
    if not isinstance(man, dict):
        raise ValueError("manifest must be a JSON object")
    return man


def _get_modules_from_manifest(manifest: dict[str, Any]) -> list[str]:
    modules: set[str] = set()
    for fn in manifest.get("functions", []) or []:
        if not isinstance(fn, dict):
            continue
        mod = str(fn.get("module") or "").strip()
        if mod:
            modules.add(mod)
    return sorted(modules)


def _maybe_add_pythonpath(root: Optional[str]) -> None:
    if not root:
        return
    p = Path(root).expanduser().resolve()
    if not p.exists():
        return
    root_s = str(p)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    src = p / "src"
    if src.exists():
        src_s = str(src)
        if src_s not in sys.path:
            sys.path.insert(0, src_s)


@dataclass
class DevRunResult:
    run_id: str
    success: bool
    output: Any
    error_type: str
    safe_message: str
    error_message: str
    events: list[dict[str, Any]]


class DevWorker(Worker):
    """
    A Worker that never connects to a scheduler.

    It reuses the real execution engine but captures messages in-process
    so we can expose a small dev HTTP API for local testing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dev_q: "queue.Queue[pb.WorkerSchedulerMessage]" = queue.Queue()

    def _send_message(self, message: pb.WorkerSchedulerMessage) -> None:  # type: ignore[override]
        self._dev_q.put_nowait(message)

    def _drain_messages(self) -> list[pb.WorkerSchedulerMessage]:
        out: list[pb.WorkerSchedulerMessage] = []
        while True:
            try:
                out.append(self._dev_q.get_nowait())
            except queue.Empty:
                break
        return out

    def run_task_sync(
        self,
        *,
        function_name: str,
        payload_obj: Any,
        run_id: Optional[str] = None,
        owner: str = "",
        user_id: str = "",
        timeout_ms: int = 0,
        required_variant_refs: Optional[list[str]] = None,
        resolved_cozy_models_by_id: Optional[dict[str, Any]] = None,
        local_output_dir: Optional[str] = None,
    ) -> DevRunResult:
        rid = (run_id or "").strip() or str(uuid.uuid4())
        fn = (function_name or "").strip()
        spec = self._task_specs.get(fn)
        if spec is None:
            raise web.HTTPNotFound(text="unknown function")

        self._drain_messages()

        raw = msgspec.msgpack.encode(payload_obj)
        req = pb.TaskExecutionRequest(
            run_id=rid,
            function_name=fn,
            input_payload=raw,
            required_variant_refs=[str(v).strip() for v in (required_variant_refs or []) if str(v).strip()],
            timeout_ms=int(timeout_ms or 0),
            owner=str(owner or ""),
            user_id=str(user_id or ""),
            file_base_url="",  # local-only
            file_token="",  # local-only
        )

        # Mirror _handle_run_request's ctx construction, but use local output backend.
        ctx = ActionContext(
            rid,
            emitter=self._emit_progress_event,
            owner=str(owner or "") or None,
            user_id=str(user_id or "") or None,
            timeout_ms=int(timeout_ms or 0) or None,
            file_api_base_url=None,
            file_api_token=None,
            local_output_dir=local_output_dir,
            resolved_cozy_models_by_id=resolved_cozy_models_by_id or None,
            required_models=[str(v).strip() for v in (required_variant_refs or []) if str(v).strip()],
        )

        # Execute synchronously.
        self._execute_task(ctx, spec, req.input_payload)

        msgs = self._drain_messages()
        events: list[dict[str, Any]] = []
        result: Optional[pb.TaskExecutionResult] = None
        for m in msgs:
            mt = m.WhichOneof("msg")
            if mt == "worker_event" and m.worker_event is not None:
                ev = m.worker_event
                try:
                    payload = json.loads(bytes(ev.payload_json or b"{}").decode("utf-8"))
                except Exception:
                    payload = {}
                events.append(
                    {
                        "run_id": str(ev.run_id or ""),
                        "event_type": str(ev.event_type or ""),
                        "payload": payload,
                    }
                )
            if mt == "run_result" and m.run_result is not None:
                result = m.run_result

        if result is None:
            raise RuntimeError("missing run_result (dev worker internal error)")

        out_obj: Any = None
        if result.success and result.output_payload:
            try:
                out_obj = msgspec.msgpack.decode(bytes(result.output_payload))
            except Exception:
                out_obj = None

        return DevRunResult(
            run_id=rid,
            success=bool(result.success),
            output=out_obj,
            error_type=str(result.error_type or ""),
            safe_message=str(result.safe_message or ""),
            error_message=str(result.error_message or ""),
            events=events,
        )


def _json_response(obj: Any, *, status: int = 200) -> web.Response:
    raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return web.Response(body=raw, status=status, content_type="application/json")


def _parse_run_body(data: Any) -> tuple[Any, dict[str, Any]]:
    """
    Accept either:
      - raw payload object, OR
      - envelope: {payload: <obj>, ...}
    """
    if isinstance(data, dict) and "payload" in data and isinstance(data.get("payload"), (dict, list)):
        env = dict(data)
        payload = env.pop("payload")
        return payload, env
    return data, {}


async def serve_http(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="gen-worker dev serve-http")
    ap.add_argument("--listen", default=os.getenv("GEN_WORKER_HTTP_LISTEN", "127.0.0.1:8081"))
    ap.add_argument("--manifest", default=os.getenv("GEN_WORKER_MANIFEST_PATH", "/app/.cozy/manifest.json"))
    ap.add_argument("--project-root", default=os.getenv("GEN_WORKER_PROJECT_ROOT", "/app"))
    ap.add_argument("--outputs", default=os.getenv("GEN_WORKER_OUTPUT_DIR", "/outputs"))
    args = ap.parse_args(list(argv) if argv is not None else None)

    listen = str(args.listen).strip()
    if ":" not in listen:
        raise SystemExit("--listen must be host:port")
    host, port_s = listen.rsplit(":", 1)
    port = int(port_s)

    manifest_path = Path(str(args.manifest)).expanduser()
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)

    _maybe_add_pythonpath(str(args.project_root))
    user_modules = _get_modules_from_manifest(manifest)
    if not user_modules:
        raise SystemExit("manifest contains no function modules")

    out_dir = Path(str(args.outputs)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Do not verify JWT in dev HTTP runner. Provide a dummy token.
    w = DevWorker(
        scheduler_addr="dev",
        scheduler_addrs=[],
        user_module_names=user_modules,
        worker_id="dev-worker",
        worker_jwt="dev",
        use_tls=False,
        reconnect_delay=0,
        max_reconnect_attempts=0,
        manifest=manifest,
    )

    routes = web.RouteTableDef()

    @routes.get("/v1/status")
    async def status(_req: web.Request) -> web.Response:
        fns = sorted(list(w._task_specs.keys()))
        stats = w._model_cache.get_stats().to_dict() if getattr(w, "_model_cache", None) is not None else {}
        return _json_response(
            {
                "ok": True,
                "functions": fns,
                "model_cache": stats,
            }
        )

    @routes.get("/v1/models/status")
    async def models_status(_req: web.Request) -> web.Response:
        stats = w._model_cache.get_stats().to_dict() if getattr(w, "_model_cache", None) is not None else {}
        return _json_response({"ok": True, "model_cache": stats})

    @routes.post("/v1/models/unload")
    async def models_unload(req: web.Request) -> web.Response:
        body = await req.json()
        model_id = str((body or {}).get("model_id") or "").strip()
        if not model_id:
            return _json_response({"error": "missing_model_id"}, status=400)
        cmd = pb.UnloadModelCommand(model_id=model_id)
        w._handle_unload_model_cmd(cmd)
        msgs = w._drain_messages()
        ok = False
        err = ""
        for m in msgs:
            if m.WhichOneof("msg") == "unload_model_result" and m.unload_model_result is not None:
                ok = bool(m.unload_model_result.success)
                err = str(m.unload_model_result.error_message or "")
                break
        return _json_response({"ok": ok, "error_message": err})

    @routes.post("/v1/models/load")
    async def models_load(req: web.Request) -> web.Response:
        body = await req.json()
        model_id = str((body or {}).get("model_id") or "").strip()
        if not model_id:
            return _json_response({"error": "missing_model_id"}, status=400)
        cmd = pb.LoadModelCommand(model_id=model_id)
        w._handle_load_model_cmd(cmd)
        msgs = w._drain_messages()
        ok = False
        err = ""
        for m in msgs:
            if m.WhichOneof("msg") == "load_model_result" and m.load_model_result is not None:
                ok = bool(m.load_model_result.success)
                err = str(m.load_model_result.error_message or "")
                break
        return _json_response({"ok": ok, "error_message": err})

    @routes.post("/v1/models/prefetch")
    async def models_prefetch(req: web.Request) -> web.Response:
        body = await req.json()
        models = (body or {}).get("models") or []
        if not isinstance(models, list) or not models:
            return _json_response({"error": "missing_models"}, status=400)
        cache_dir = Path(os.getenv("WORKER_MODEL_CACHE_DIR") or "/tmp/cozy/models")
        cache_dir.mkdir(parents=True, exist_ok=True)

        out: list[dict[str, Any]] = []
        for raw in models:
            mid = str(raw or "").strip()
            if not mid:
                continue
            t0 = time.monotonic()
            try:
                # Best-effort download into the shared cache dir.
                local_path = w._downloader.download(mid, str(cache_dir)) if getattr(w, "_downloader", None) else ""
                lp = Path(str(local_path))
                if not local_path or not lp.exists():
                    raise RuntimeError(f"download returned missing path: {local_path!r}")
                if getattr(w, "_model_cache", None) is not None:
                    w._model_cache.mark_cached_to_disk(mid, lp)
                out.append({"model_id": mid, "ok": True, "duration_ms": int((time.monotonic() - t0) * 1000)})
            except Exception as e:
                out.append({"model_id": mid, "ok": False, "error_type": type(e).__name__, "duration_ms": int((time.monotonic() - t0) * 1000)})
        return _json_response({"results": out})

    @routes.post("/v1/run/{function_name}")
    async def run(req: web.Request) -> web.Response:
        fn = str(req.match_info.get("function_name") or "").strip()
        try:
            body = await req.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)

        payload, env = _parse_run_body(body)

        rid = str(env.get("run_id") or "").strip() or str(uuid.uuid4())
        timeout_ms = int(env.get("timeout_ms") or 0)
        owner = str(env.get("owner") or "").strip()
        user_id = str(env.get("user_id") or "").strip()
        required_variant_refs = env.get("required_variant_refs")
        if required_variant_refs is None:
            required_variant_refs = env.get("required_models")  # alias
        rvr: list[str] = []
        if isinstance(required_variant_refs, list):
            rvr = [str(v).strip() for v in required_variant_refs if str(v).strip()]

        resolved = env.get("resolved_cozy_models_by_id")
        if resolved is not None and not isinstance(resolved, dict):
            return _json_response({"error": "resolved_cozy_models_by_id must be an object"}, status=400)

        t0 = time.monotonic()
        try:
            res = w.run_task_sync(
                function_name=fn,
                payload_obj=payload,
                run_id=rid,
                owner=owner,
                user_id=user_id,
                timeout_ms=timeout_ms,
                required_variant_refs=rvr,
                resolved_cozy_models_by_id=resolved if isinstance(resolved, dict) else None,
                local_output_dir=str(out_dir),
            )
        except web.HTTPException:
            raise
        except Exception as e:
            return _json_response({"run_id": rid, "success": False, "error_type": type(e).__name__, "safe_message": str(e)}, status=500)

        return _json_response(
            {
                "run_id": res.run_id,
                "success": res.success,
                "output": res.output,
                "error_type": res.error_type,
                "safe_message": res.safe_message,
                "error_message": res.error_message,
                "events": res.events,
                "timing_ms": {"total_ms": int((time.monotonic() - t0) * 1000)},
            },
            status=200 if res.success else 400,
        )

    @routes.post("/v1/warmup/{function_name}")
    async def warmup(req: web.Request) -> web.Response:
        # Alias of /v1/run that discards output but forces init/compiles.
        fn = str(req.match_info.get("function_name") or "").strip()
        body = await req.json()
        payload, env = _parse_run_body(body)
        rid = str(env.get("run_id") or "").strip() or str(uuid.uuid4())
        t0 = time.monotonic()
        res = w.run_task_sync(
            function_name=fn,
            payload_obj=payload,
            run_id=rid,
            timeout_ms=int(env.get("timeout_ms") or 0),
            required_variant_refs=[str(v).strip() for v in (env.get("required_variant_refs") or []) if str(v).strip()]
            if isinstance(env.get("required_variant_refs"), list)
            else [],
            local_output_dir=str(out_dir),
        )
        return _json_response({"ok": res.success, "run_id": rid, "duration_ms": int((time.monotonic() - t0) * 1000)})

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    print(f"dev http runner listening on http://{host}:{port}", file=sys.stderr)

    # Run forever.
    import asyncio

    while True:
        await asyncio.sleep(3600)


def main(argv: Optional[list[str]] = None) -> None:
    # asyncio entrypoint wrapper (avoid importing asyncio at module import time in prod).
    import asyncio

    asyncio.run(serve_http(argv=argv))


if __name__ == "__main__":
    main()
