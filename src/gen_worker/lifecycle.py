"""Connection lifecycle: Hello/HelloAck + in_flight reconcile, edge-triggered
StateDelta full-replace snapshots, FnUnavailable emission, startup phases, and
a drain that actually drains (stop admitting -> finish in-flight -> ship
results -> close stream -> exit 0).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from .config import Settings
from .executor import Executor
from .pb import worker_scheduler_pb2 as pb
from .transport import PROTOCOL_VERSION, Transport

logger = logging.getLogger(__name__)

_VRAM_QUANTUM_FRACTION = 0.05  # quantize free-VRAM deltas to 5% of total


def probe_hardware() -> Dict[str, Any]:
    """Static hardware facts + gate inputs. torch is optional."""
    info: Dict[str, Any] = {
        "gpu_count": 0, "gpu_total_mem": 0, "gpu_free_mem": 0,
        "gpu_name": "", "gpu_sm": "", "installed_libs": [],
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info["gpu_total_mem"] = int(total_mem)
            info["gpu_free_mem"] = int(free_mem)
    except Exception:
        pass
    try:
        from .models.hub_policy import detect_worker_capabilities

        caps = detect_worker_capabilities()
        info["installed_libs"] = list(caps.installed_libs or [])
        if caps.gpu_sm:
            info["gpu_sm"] = str(int(caps.gpu_sm))
    except Exception:
        pass
    return info


def free_vram_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            free_mem, _total = torch.cuda.mem_get_info(0)
            return int(free_mem)
    except Exception:
        pass
    return 0


class Lifecycle:
    """Transport handlers + worker state machine. One instance per process."""

    def __init__(self, settings: Settings, executor: Executor) -> None:
        self._settings = settings
        self.executor = executor
        self.transport: Optional[Transport] = None  # set by worker wiring
        # Identity: JWT claims are authoritative (sub = worker_id, release_id);
        # Hello echoes them for dev mode / cross-checking only.
        claims: Dict[str, Any] = {}
        if (settings.worker_jwt or "").strip():
            from .request_context import _decode_unverified_jwt_claims

            claims = _decode_unverified_jwt_claims(settings.worker_jwt.strip())
        self.worker_id = (
            settings.worker_id or str(claims.get("sub") or "").strip()
            or f"py-worker-{os.getpid()}"
        )
        self.release_id = str(claims.get("release_id") or "").strip()
        self.phase = pb.WORKER_PHASE_BOOTING
        self.hardware = probe_hardware()
        self.draining = False
        self.drained = asyncio.Event()  # set when drain completed -> exit 0
        self._last_delta: Optional[bytes] = None
        self._emitted_unavailable: set[str] = set()
        self._drain_task: Optional[asyncio.Task] = None

    # ---- snapshots -----------------------------------------------------------

    def _state_delta(self) -> pb.StateDelta:
        free = free_vram_bytes()
        total = int(self.hardware.get("gpu_total_mem") or 0)
        quantum = max(1, int(total * _VRAM_QUANTUM_FRACTION)) if total else 1
        return pb.StateDelta(
            phase=self.phase,
            available_functions=self.executor.available_functions(),
            loading_functions=self.executor.loading_functions(),
            free_vram_bytes=(free // quantum) * quantum,
        )

    def build_resources(self) -> pb.WorkerResources:
        hw = self.hardware
        return pb.WorkerResources(
            gpu_count=int(hw.get("gpu_count") or 0),
            vram_total_bytes=int(hw.get("gpu_total_mem") or 0),
            gpu_name=str(hw.get("gpu_name") or ""),
            gpu_sm=str(hw.get("gpu_sm") or ""),
            installed_libs=[str(x) for x in (hw.get("installed_libs") or [])],
            image_digest=os.environ.get("WORKER_IMAGE_DIGEST", ""),
            git_commit=os.environ.get("WORKER_GIT_COMMIT", ""),
            instance_id=self._settings.runpod_pod_id or "",
        )

    # ---- transport handlers --------------------------------------------------

    def build_hello(self) -> pb.Hello:
        in_flight = {k for k in self.executor.in_flight_keys()}
        if self.transport is not None:
            in_flight.update(self.transport.queue.pending_result_keys)
        return pb.Hello(
            protocol_version=PROTOCOL_VERSION,
            worker_id=self.worker_id,
            release_id=self.release_id,
            resources=self.build_resources(),
            state=self._state_delta(),
            models=self.executor.store.residency_snapshot(),
            in_flight=[
                pb.InFlightJob(request_id=rid, attempt=att)
                for rid, att in sorted(in_flight)
            ],
        )

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        # Full-replace config: file base URL + disk-retention keep set.
        self.executor.file_base_url = ack.file_base_url or ""
        self.executor.store.keep = set(ack.keep)
        # New connection: per-worker fn disables were wiped by Hello; re-emit
        # any that still hold, then re-baseline dynamic state.
        self._emitted_unavailable.clear()
        self._last_delta = None
        await self._emit_unavailable()
        await self.maybe_send_state_delta()

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        which = msg.WhichOneof("msg")
        if which == "run_job":
            await self.executor.handle_run_job(msg.run_job)
        elif which == "cancel_job":
            self.executor.handle_cancel(msg.cancel_job)
        elif which == "model_op":
            # Model work must never block the receive path.
            asyncio.create_task(self._model_op_then_delta(msg.model_op))
        elif which == "drain":
            self.start_drain(int(msg.drain.deadline_ms or 0))

    async def _model_op_then_delta(self, op: pb.ModelOp) -> None:
        try:
            await self.executor.handle_model_op(op)
        finally:
            await self.maybe_send_state_delta()

    async def on_disconnect(self) -> None:
        self._last_delta = None
        self._emitted_unavailable.clear()

    # ---- state emission --------------------------------------------------------

    def state_changed(self) -> None:
        """Sync callback from the executor; coalesced onto the loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.maybe_send_state_delta())

    async def maybe_send_state_delta(self) -> None:
        if self.transport is None or not self.transport.connected:
            return
        delta = self._state_delta()
        raw = delta.SerializeToString(deterministic=True)
        if raw == self._last_delta:
            return
        self._last_delta = raw
        await self.transport.send(pb.WorkerMessage(state_delta=delta))
        await self._emit_unavailable()

    async def _emit_unavailable(self) -> None:
        if self.transport is None:
            return
        for name, (reason, detail, axes) in list(self.executor.unavailable.items()):
            if name in self._emitted_unavailable:
                continue
            self._emitted_unavailable.add(name)
            await self.transport.send(pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
                function_name=name, reason=reason, detail=detail, axes=axes)))

    async def set_phase(self, phase: int) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        await self.maybe_send_state_delta()

    # ---- startup ---------------------------------------------------------------

    async def startup(self) -> None:
        """Gate functions, prefetch worker-fetchable models with retry/backoff,
        set up endpoints, advance phases. Never raises: failures gate
        individual functions, not the process."""
        self.executor.gate_functions(self.hardware)

        prefetch_refs: List[str] = []
        for spec in self.executor.specs.values():
            if spec.name in self.executor.unavailable:
                continue
            for repo in spec.fixed_models.values():
                provider = getattr(repo, "provider", "tensorhub")
                if provider != "tensorhub" and repo.ref not in prefetch_refs:
                    # hf/civitai refs need no orchestrator snapshot; tensorhub
                    # refs arrive via ModelOp{DOWNLOAD} after HelloAck (§7).
                    prefetch_refs.append(repo.ref)

        if prefetch_refs:
            await self.set_phase(pb.WORKER_PHASE_DOWNLOADING_MODELS)
            for ref in prefetch_refs:
                try:
                    await self.executor.store.ensure_local(ref)
                except Exception as exc:
                    logger.error("startup prefetch of %s failed terminally: %s", ref, exc)

        await self.set_phase(pb.WORKER_PHASE_LOADING_PIPELINES)
        for spec in list(self.executor.specs.values()):
            if spec.name in self.executor.unavailable:
                continue
            fetchable = all(
                getattr(r, "provider", "tensorhub") != "tensorhub"
                or self.executor.store.local_path(r.ref) is not None
                for r in spec.fixed_models.values()
            )
            if not fetchable:
                continue  # waits for ModelOp / RunJob snapshots
            try:
                await self.executor.ensure_setup(spec)
            except Exception as exc:
                logger.error("startup setup of %s failed: %s", spec.name, exc)

        await self.set_phase(pb.WORKER_PHASE_READY)

    # ---- drain -------------------------------------------------------------------

    def start_drain(self, deadline_ms: int) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(
                self.drain(deadline_ms), name="drain"
            )

    async def drain(self, deadline_ms: int = 0) -> None:
        """stop admitting -> finish in-flight -> ship buffered results ->
        close the stream -> signal exit 0."""
        if self.draining:
            return
        self.draining = True
        self.executor.draining = True
        logger.info("drain started (deadline_ms=%d)", deadline_ms)
        await self.maybe_send_state_delta()

        deadline_s = (deadline_ms / 1000.0) if deadline_ms > 0 else None
        finished = await self.executor.wait_idle(timeout=deadline_s)
        if not finished:
            logger.warning("drain deadline expired; aborting remaining jobs as RETRYABLE")
            await self.executor.abort_all(safe_message="worker draining")

        await self.executor.shutdown_instances()
        if self.transport is not None:
            await self.transport.close_after_flush(timeout=30.0)
        self.drained.set()
        logger.info("drain complete")
