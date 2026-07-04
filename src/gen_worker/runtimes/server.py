"""First-class server-subprocess runtime: boot / health-wait / abort /
shutdown for engine-hosting servers (vLLM, llama-server).

``@endpoint(runtime="vllm")`` (or ``"llama-server"``) makes the worker boot
the server around ``setup()`` and inject a :class:`ServerHandle` into any
setup parameter annotated ``ServerHandle``. Endpoints can also drive
:class:`ServerProcess` directly for custom engines.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_BOOT_TIMEOUT_S = 600.0
_TERM_GRACE_S = 10.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class ServerHandle:
    """A running engine server: base URL + process control."""

    base_url: str
    process: subprocess.Popen = field(repr=False)

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, *, grace_s: float = _TERM_GRACE_S) -> None:
        """SIGTERM, wait ``grace_s``, then SIGKILL. Idempotent."""
        proc = self.process
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            logger.warning("engine server ignored SIGTERM; killing")
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception:
            logger.exception("engine server stop failed")


class ServerBootError(RuntimeError):
    """The engine server exited or failed health checks during boot."""


class ServerProcess:
    """Boot an HTTP server subprocess and wait for it to become healthy.

    ::

        handle = ServerProcess(
            ["vllm", "serve", model_path, "--port", str(port)],
            health_url=f"http://127.0.0.1:{port}/health",
        ).start()
        ...
        handle.stop()
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        health_url: str,
        base_url: str = "",
        boot_timeout_s: float = _DEFAULT_BOOT_TIMEOUT_S,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.command = [str(c) for c in command]
        self.health_url = health_url
        self.base_url = base_url or health_url.rsplit("/", 1)[0]
        self.boot_timeout_s = float(boot_timeout_s)
        self.env = env

    def start(self) -> ServerHandle:
        logger.info("booting engine server: %s", shlex.join(self.command))
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        proc = subprocess.Popen(self.command, env=env, start_new_session=True)
        handle = ServerHandle(base_url=self.base_url, process=proc)
        try:
            self._wait_healthy(proc)
        except BaseException:
            handle.stop()
            raise
        logger.info("engine server healthy at %s", self.base_url)
        return handle

    def _wait_healthy(self, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.boot_timeout_s
        delay = 0.25
        while True:
            code = proc.poll()
            if code is not None:
                raise ServerBootError(
                    f"engine server exited during boot (code {code}): "
                    f"{shlex.join(self.command)}"
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise ServerBootError(
                    f"engine server failed health check within "
                    f"{self.boot_timeout_s:.0f}s at {self.health_url}"
                )
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)


def vllm_server(
    model_path: str,
    *,
    port: Optional[int] = None,
    extra_args: Sequence[str] = (),
    boot_timeout_s: float = _DEFAULT_BOOT_TIMEOUT_S,
) -> ServerProcess:
    """``vllm serve <model_path>`` with an OpenAI-compatible API + /health."""
    p = port or free_port()
    return ServerProcess(
        ["vllm", "serve", model_path, "--host", "127.0.0.1", "--port", str(p),
         *extra_args],
        health_url=f"http://127.0.0.1:{p}/health",
        base_url=f"http://127.0.0.1:{p}",
        boot_timeout_s=boot_timeout_s,
    )


def llama_server(
    gguf_path: str,
    *,
    port: Optional[int] = None,
    extra_args: Sequence[str] = (),
    boot_timeout_s: float = _DEFAULT_BOOT_TIMEOUT_S,
) -> ServerProcess:
    """``llama-server -m <gguf>`` with the built-in /health endpoint."""
    p = port or free_port()
    return ServerProcess(
        ["llama-server", "-m", gguf_path, "--host", "127.0.0.1",
         "--port", str(p), *extra_args],
        health_url=f"http://127.0.0.1:{p}/health",
        base_url=f"http://127.0.0.1:{p}",
        boot_timeout_s=boot_timeout_s,
    )


RUNTIME_FACTORIES = {"vllm": vllm_server, "llama-server": llama_server}


__all__ = [
    "RUNTIME_FACTORIES",
    "ServerBootError",
    "ServerHandle",
    "ServerProcess",
    "free_port",
    "llama_server",
    "vllm_server",
]
