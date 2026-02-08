import time
import unittest

from gen_worker.worker import ActionContext, RealtimeSocket, Worker, pb
from gen_worker.decorators import worker_websocket, ResourceRequirements


@worker_websocket(ResourceRequirements())
async def echo_ws(ctx: ActionContext, sock: RealtimeSocket) -> None:
    await sock.send_json({"status": "ready"})
    async for data in sock.iter_bytes():
        await sock.send_bytes(data)


class TestRealtimeSocket(unittest.TestCase):
    def _make_worker(self) -> Worker:
        w = Worker.__new__(Worker)
        import threading

        w.owner = "tenant-1"
        w._emit_progress_event = lambda e: None  # type: ignore[method-assign]
        w._runtime_loaders = {}
        w._custom_runtime_cache = {}
        w._custom_runtime_locks = {}
        w._release_model_id_by_key = {}
        w._release_allowed_model_ids = None
        w._model_manager = None
        w._realtime_sessions = {}
        w._realtime_lock = threading.Lock()
        w._sent = []
        w._send_message = lambda msg: w._sent.append(msg)  # type: ignore[method-assign]
        w._ws_specs = {}
        w._discovered_resources = {}
        w._inspect_websocket_spec = Worker._inspect_websocket_spec.__get__(w, Worker)  # type: ignore[attr-defined]
        w._resolve_injected_value = Worker._resolve_injected_value.__get__(w, Worker)  # type: ignore[attr-defined]
        w._handle_realtime_open_cmd = Worker._handle_realtime_open_cmd.__get__(w, Worker)  # type: ignore[attr-defined]
        w._handle_realtime_frame = Worker._handle_realtime_frame.__get__(w, Worker)  # type: ignore[attr-defined]
        w._handle_realtime_close_cmd = Worker._handle_realtime_close_cmd.__get__(w, Worker)  # type: ignore[attr-defined]
        return w

    def test_realtime_echo(self) -> None:
        w = self._make_worker()
        spec = w._inspect_websocket_spec(echo_ws)  # type: ignore[arg-type]
        w._ws_specs[spec.name] = spec

        w._handle_realtime_open_cmd(
            pb.RealtimeOpenCommand(session_id="s1", function_name=spec.name, owner="tenant-1")
        )

        # Wait for ready JSON frame.
        deadline = time.time() + 2.0
        ready = False
        while time.time() < deadline:
            for m in list(w._sent):
                if getattr(m, "realtime_frame", None) and m.realtime_frame.is_text:
                    if b"ready" in m.realtime_frame.data:
                        ready = True
                        break
            if ready:
                break
            time.sleep(0.01)
        self.assertTrue(ready)

        # Send binary bytes frame and expect an echoed binary frame back.
        w._handle_realtime_frame(pb.RealtimeFrame(session_id="s1", data=b"abc", is_text=False))

        deadline = time.time() + 2.0
        echoed = False
        while time.time() < deadline:
            for m in list(w._sent):
                if getattr(m, "realtime_frame", None) and not m.realtime_frame.is_text:
                    if m.realtime_frame.data == b"abc":
                        echoed = True
                        break
            if echoed:
                break
            time.sleep(0.01)
        self.assertTrue(echoed)

        w._handle_realtime_close_cmd(pb.RealtimeCloseCommand(session_id="s1", reason="end"))
