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
from typing import Callable, Dict, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 10.0
_RUNTIME_NAMES = frozenset(("vllm", "llama-server"))


@dataclass(frozen=True)
class VLLMRuntime:
    """Typed vLLM process configuration owned by the worker lifecycle.

    Startup has no wall-clock deadline: while the process is alive but not yet
    healthy, the worker advertises the function in ``StateDelta.loading_functions``.
    Process exit becomes a typed setup failure; readiness becomes availability.
    """

    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("VLLMRuntime.max_model_len must be positive")
        if self.gpu_memory_utilization is not None and not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("VLLMRuntime.gpu_memory_utilization must be in (0, 1]")

    @property
    def engine(self) -> str:
        return "vllm"

    def command_args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self.max_model_len is not None:
            args.extend(("--max-model-len", str(self.max_model_len)))
        if self.gpu_memory_utilization is not None:
            args.extend(("--gpu-memory-utilization", str(self.gpu_memory_utilization)))
        return tuple(args)


RuntimeSpec = Union[str, VLLMRuntime]


def runtime_name(runtime: RuntimeSpec) -> str:
    """Wire/discovery name for a validated runtime declaration."""
    if isinstance(runtime, VLLMRuntime):
        return runtime.engine
    if isinstance(runtime, str) and runtime in _RUNTIME_NAMES:
        return runtime
    raise ValueError(
        f"runtime must be 'vllm', 'llama-server', or VLLMRuntime(...), got {runtime!r}"
    )


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
        boot_timeout_s: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.command = [str(c) for c in command]
        self.health_url = health_url
        self.base_url = base_url or health_url.rsplit("/", 1)[0]
        if boot_timeout_s is not None and boot_timeout_s <= 0:
            raise ValueError("boot_timeout_s must be positive when provided")
        self.boot_timeout_s = float(boot_timeout_s) if boot_timeout_s is not None else None
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
        deadline = (
            time.monotonic() + self.boot_timeout_s if self.boot_timeout_s is not None else None
        )
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
            if deadline is not None and time.monotonic() >= deadline:
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
    boot_timeout_s: Optional[float] = None,
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


class DegradingBoot:
    """Try candidate server commands in order; first healthy one wins.

    The llama.cpp fit ladder rides this: planned ``-ngl``, then half, then
    CPU-only — a VRAM-tight boot degrades instead of crashing.
    """

    def __init__(self, candidates: Sequence[ServerProcess]) -> None:
        if not candidates:
            raise ValueError("DegradingBoot needs at least one candidate")
        self.candidates = list(candidates)

    def start(self) -> ServerHandle:
        last: Optional[ServerBootError] = None
        for proc in self.candidates:
            try:
                return proc.start()
            except ServerBootError as exc:
                last = exc
                logger.warning("engine boot failed (%s); trying degraded rung", exc)
        assert last is not None
        raise last


_NGL_FLAGS = {"-ngl", "--n-gpu-layers", "--gpu-layers"}
_CTX_FLAGS = {"-c", "--ctx-size"}


def llama_server(
    model_source: str,
    *,
    port: Optional[int] = None,
    extra_args: Sequence[str] = (),
    boot_timeout_s: Optional[float] = None,
    vram_budget_gb: Optional[float] = None,
    n_ctx: Optional[int] = None,
) -> "ServerProcess | DegradingBoot":
    """``llama-server -m <gguf>`` with the built-in /health endpoint.

    ``model_source`` may be the ``.gguf`` file or a snapshot dir (the
    Hub()-injected path) — dirs resolve to their single GGUF model. Unless
    the caller pins ``-ngl``/``-c`` in ``extra_args``, ``-ngl`` and context
    are sized to the free-VRAM budget (gw#402) and the boot degrades
    through fewer GPU layers rather than failing.
    """
    from .llama import plan_for, resolve_gguf

    gguf_path = str(resolve_gguf(model_source))
    args = [str(a) for a in extra_args]

    def _proc(fit_args: Sequence[str]) -> ServerProcess:
        p = port or free_port()
        return ServerProcess(
            ["llama-server", "-m", gguf_path, "--host", "127.0.0.1",
             "--port", str(p), *fit_args, *args],
            health_url=f"http://127.0.0.1:{p}/health",
            base_url=f"http://127.0.0.1:{p}",
            boot_timeout_s=boot_timeout_s,
        )

    if any(a in _NGL_FLAGS for a in args):
        return _proc(())
    plan = plan_for(gguf_path, vram_budget_gb=vram_budget_gb, n_ctx=n_ctx)
    if plan is None:
        return _proc(())
    ctx_args = [] if any(a in _CTX_FLAGS for a in args) else ["-c", str(plan.n_ctx)]
    rungs = [plan.n_gpu_layers]
    if plan.n_gpu_layers > 1:
        rungs.append(plan.n_gpu_layers // 2)
    if plan.n_gpu_layers > 0:
        rungs.append(0)
    candidates = [_proc(["-ngl", str(n), *ctx_args]) for n in rungs]
    return candidates[0] if len(candidates) == 1 else DegradingBoot(candidates)


def process_vram_bytes(pid: int) -> int:
    """Measured VRAM of one process via nvidia-smi. 0 when unavailable —
    engine subprocesses are invisible to torch's allocator, so residency
    accounting for server runtimes books this number instead."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return 0
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                total += int(parts[1]) * 1024 * 1024
            except ValueError:
                pass
    return total


RUNTIME_FACTORIES: Dict[str, Callable[[str], "ServerProcess | DegradingBoot"]] = {
    "vllm": vllm_server, "llama-server": llama_server,
}


def runtime_process(runtime: RuntimeSpec, model_path: str) -> "ServerProcess | DegradingBoot":
    """Build the process represented by a typed endpoint runtime declaration."""
    name = runtime_name(runtime)
    if isinstance(runtime, VLLMRuntime):
        return vllm_server(model_path, extra_args=runtime.command_args())
    return RUNTIME_FACTORIES[name](model_path)


__all__ = [
    "DegradingBoot",
    "RUNTIME_FACTORIES",
    "RuntimeSpec",
    "ServerBootError",
    "ServerHandle",
    "ServerProcess",
    "VLLMRuntime",
    "free_port",
    "llama_server",
    "process_vram_bytes",
    "runtime_name",
    "runtime_process",
    "vllm_server",
]
