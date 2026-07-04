"""Worker wiring: registry -> executor -> lifecycle -> transport, one asyncio
loop. See transport.py / executor.py / lifecycle.py for the moving parts and
proto/CONTRACT.md for the wire semantics.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any, Dict, List, Optional

from .config import Settings
from .executor import Executor, ModelStore
from .lifecycle import Lifecycle
from .pb import worker_scheduler_pb2 as pb
from .registry import collect_endpoints
from .transport import FatalTransportError, Transport

logger = logging.getLogger(__name__)

_SIGNAL_DRAIN_DEADLINE_MS = 30_000


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
            from .models.ref_downloader import (
                build_provider_index_from_manifest,
                set_provider_by_ref_global,
            )

            set_provider_by_ref_global(build_provider_index_from_manifest(manifest))

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

    async def _send(self, msg: pb.WorkerMessage) -> None:
        await self.transport.send(msg)

    def run(self) -> int:
        return asyncio.run(self.arun())

    async def arun(self) -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, self.lifecycle.start_drain, _SIGNAL_DRAIN_DEADLINE_MS
                )
            except (NotImplementedError, RuntimeError):
                pass

        startup = asyncio.create_task(self.lifecycle.startup(), name="startup")
        transport_task = asyncio.create_task(self.transport.run(), name="transport")
        try:
            await transport_task
        except FatalTransportError as exc:
            logger.error("worker exiting: %s", exc)
            return 1
        finally:
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
        if self.lifecycle.drained.is_set():
            logger.info("worker drained; exiting 0")
            return 0
        return 0
