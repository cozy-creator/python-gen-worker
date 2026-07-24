"""Worker wiring: registry -> executor -> lifecycle -> transport, one asyncio
loop. See transport.py / executor.py / lifecycle.py for the moving parts and
proto/CONTRACT.md for the wire semantics.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from .config import Settings
from .executor import Executor, ModelStore
from .lifecycle import Lifecycle
from .pb import worker_scheduler_pb2 as pb
from .registry import collect_endpoints
from .transport import FatalTransportError, Transport

logger = logging.getLogger(__name__)

_SIGNAL_DRAIN_DEADLINE_MS = 30_000


class UnexpectedWorkerExit(RuntimeError):
    """The run loop ended without a hub Drain or a shutdown signal (gw#640)."""


class _LoopStallWatchdog:
    """Forensics for gw#407: a host in RAM reclaim-thrash stalls the whole
    process — the event loop AND the gRPC C threads that answer h2 keepalive
    pings — and the hub reaps the worker as dead within ~30s. This thread
    pings the loop and logs LOUDLY (with available host RAM) when the ping
    isn't serviced within ``warn_after_s``, so the stall episode is visible
    in worker logs instead of only as a hub-side disconnect."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        interval_s: float = 5.0,
        warn_after_s: float = 10.0,
    ) -> None:
        self._loop = loop
        self._interval = interval_s
        self._warn_after = warn_after_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="loop-stall-watchdog", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            ping = threading.Event()
            t0 = time.monotonic()
            try:
                self._loop.call_soon_threadsafe(ping.set)
            except RuntimeError:
                return  # loop closed
            while not ping.wait(self._warn_after):
                if self._stop.is_set() or self._loop.is_closed():
                    return
                lag = time.monotonic() - t0
                avail_gb = 0.0
                try:
                    from .models.memory import get_available_ram_gb

                    avail_gb = get_available_ram_gb()
                except Exception:
                    pass
                logger.warning(
                    "event loop stalled for %.1fs (available host RAM %.1fGiB) — "
                    "host under memory/IO pressure; hub keepalive may lapse (gw#407)",
                    lag, avail_gb,
                )


class Worker:
    def __init__(
        self,
        settings: Settings,
        user_module_names: List[str],
        *,
        manifest: Optional[Dict[str, Any]] = None,
        gpu_slots: int = 1,
        queue_maxsize: int = 1024,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
    ) -> None:
        if not (settings.orchestrator_public_addr or "").strip():
            raise ValueError("Settings.orchestrator_public_addr is required")
        self.settings = settings

        if manifest:
            from .models.download import (
                build_provider_index_from_manifest,
                set_provider_index,
            )

            set_provider_index(build_provider_index_from_manifest(manifest))

        specs = collect_endpoints(list(user_module_names))
        if not specs:
            raise ValueError(
                f"no endpoint classes found in modules {list(user_module_names)!r}"
            )
        store = ModelStore(
            self._send, hf_home=settings.hf_home, hf_token=settings.hf_token
        )
        self.executor = Executor(
            specs, self._send, settings=settings, store=store, gpu_slots=gpu_slots
        )
        if (settings.config_snapshot_path or "").strip():
            from .runtime_config import ConfigStore

            self.executor.runtime_config = ConfigStore(
                settings.config_snapshot_path.strip()
            )
        self.lifecycle = Lifecycle(settings, self.executor)
        self.executor._on_state_change = self.lifecycle.state_changed
        self.transport = Transport(
            settings,
            self.lifecycle,
            queue_maxsize=queue_maxsize,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
        )
        self.lifecycle.transport = self.transport
        # Capability renewal presents the freshest worker JWT (contract §1
        # rotation), not the boot-time settings token.
        self.executor.worker_jwt_provider = lambda: self.transport.current_worker_jwt

    async def _send(self, msg: pb.WorkerMessage) -> None:
        await self.transport.send(msg)

    def run(self) -> int:
        """Always returns an exit code. gw#640: a fatal end to the run loop is
        reported to the HUB here (sync context, own loop) before returning —
        pod stdout is unreadable on RunPod, so this is the only channel that
        survives the process."""
        try:
            return asyncio.run(self.arun())
        except (FatalTransportError, UnexpectedWorkerExit) as exc:
            from .worker_fatal import report_worker_fatal

            logger.error("worker exiting on a fatal: %s", exc, exc_info=True)
            report_worker_fatal(self.settings, "run_loop", exc, exit_code=1)
            return 1

    _loop: Optional[asyncio.AbstractEventLoop] = None
    _stop_requested: bool = False

    def stop(self) -> None:
        """Thread-safe stop (tests / embedding); production exits via Drain."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.transport.stop)

    async def arun(self) -> int:
        loop = self._loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, self.lifecycle.start_drain, _SIGNAL_DRAIN_DEADLINE_MS
                )
            except (NotImplementedError, RuntimeError):
                pass

        watchdog = _LoopStallWatchdog(loop)
        watchdog.start()
        startup = asyncio.create_task(self.lifecycle.startup(), name="startup")
        transport_task = asyncio.create_task(self.transport.run(), name="transport")
        try:
            await transport_task
        except FatalTransportError as exc:
            # gw#640: surfaced to the entrypoint so the cause reaches the hub
            # instead of only this pod's stdout.
            logger.error("worker exiting: %s", exc)
            raise
        finally:
            watchdog.stop()
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
        if self.lifecycle.drained.is_set():
            logger.info("worker drained; exiting 0")
            return 0
        if self._stop_requested:
            return 0
        # gw#640: the reconnect loop is supposed to run until a hub Drain or a
        # signal. Falling out of it any other way ended the process with a
        # clean exit 0 and NOTHING on the wire — the hub saw only a stream
        # close and a young-worker death. An unexplained exit is a fatal.
        raise UnexpectedWorkerExit(
            "transport loop ended without a Drain command or shutdown signal "
            f"(connected={self.transport.connected} draining={self.lifecycle.draining})"
        )
