"""Server-subprocess runtime: boot / health-wait / abort / shutdown."""

from __future__ import annotations

import sys
import textwrap

import msgspec
import pytest

from gen_worker.api.streaming import BatchItemDelta
from gen_worker.runtimes.server import (
    ServerBootError,
    ServerProcess,
    free_port,
    llama_server,
    vllm_server,
)

_HEALTH_SERVER = textwrap.dedent("""
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
        def log_message(self, *a): pass

    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
""")


def test_boot_health_wait_and_shutdown(tmp_path) -> None:
    script = tmp_path / "srv.py"
    script.write_text(_HEALTH_SERVER)
    port = free_port()
    proc = ServerProcess(
        [sys.executable, str(script), str(port)],
        health_url=f"http://127.0.0.1:{port}/health",
        base_url=f"http://127.0.0.1:{port}",
        boot_timeout_s=20,
    )
    handle = proc.start()
    try:
        assert handle.alive
        assert handle.base_url.endswith(str(port))
    finally:
        handle.stop()
    assert not handle.alive
    handle.stop()  # idempotent


def test_boot_failure_raises_and_reaps() -> None:
    port = free_port()
    proc = ServerProcess(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        health_url=f"http://127.0.0.1:{port}/health",
        boot_timeout_s=5,
    )
    with pytest.raises(ServerBootError, match="exited during boot"):
        proc.start()


def test_health_timeout_kills_process() -> None:
    port = free_port()
    proc = ServerProcess(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        health_url=f"http://127.0.0.1:{port}/health",
        boot_timeout_s=1.0,
    )
    with pytest.raises(ServerBootError, match="health check"):
        proc.start()


def test_engine_command_shapes() -> None:
    vp = vllm_server("/models/x", port=1234, extra_args=("--max-model-len", "8192"))
    assert vp.command[:3] == ["vllm", "serve", "/models/x"]
    assert vp.health_url == "http://127.0.0.1:1234/health"
    lp = llama_server("/models/x.gguf", port=4321)
    assert lp.command[:3] == ["llama-server", "-m", "/models/x.gguf"]


def test_batch_item_delta_is_first_class_and_binary_safe() -> None:
    from gen_worker.executor import Executor

    delta = BatchItemDelta(
        index=1, total=4, item_id="a", finished=True,
        chunk=b"\x00\xffbinary", content_type="audio/ogg",
    )
    data, ctype = Executor._encode_chunk(object(), delta)  # self unused
    assert ctype == "application/x-batch-item+msgpack"
    decoded = msgspec.msgpack.decode(data, type=BatchItemDelta)
    assert decoded == delta
