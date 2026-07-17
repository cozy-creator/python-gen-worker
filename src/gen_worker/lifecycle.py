"""Connection lifecycle: Hello/HelloAck + in_flight reconcile, edge-triggered
StateDelta full-replace snapshots, FnUnavailable + FnDegraded emission,
startup phases, and a drain that actually drains (stop admitting -> finish
in-flight -> ship results -> close stream -> exit 0).
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
    """Free VRAM summed across ALL CUDA devices (StateDelta.free_vram_bytes)."""
    try:
        import torch

        if torch.cuda.is_available():
            return sum(
                int(torch.cuda.mem_get_info(i)[0])
                for i in range(torch.cuda.device_count())
            )
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
        # fn name -> the "ran" rung last reported. Keyed on the rung (not
        # mere membership) so a runtime ladder demotion (gw#463) re-emits.
        self._emitted_degraded: dict[str, str] = {}
        self._drain_task: Optional[asyncio.Task] = None
        self._drain_deadline_at: Optional[float] = None
        self._desired_residency: Optional[pb.DesiredResidency] = None
        self._residency_task: Optional[asyncio.Task] = None
        self._observed_residency_generation = 0

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
            # gw#516: encode/upload tails past the GPU-slot release. The hub
            # must not drain/retire this worker on GPU-idleness alone.
            finalizing_jobs=self.executor.finalizing_jobs(),
            observed_residency_generation=self._observed_residency_generation,
        )

    def build_resources(self) -> pb.WorkerResources:
        hw = self.hardware
        # gw#550 boot host canary: measured once per process (cached), so
        # reconnect Hellos re-ship the same boot-time facts. Never fatal.
        canary = None
        try:
            from .host_canary import get_host_canary

            c = get_host_canary()
            canary = pb.HostCanary(
                memcpy_gbps=c.memcpy_gbps,
                h2d_gbps=c.h2d_gbps,
                d2h_gbps=c.d2h_gbps,
                pinned_alloc_ok=c.pinned_alloc_ok,
                cpu_single_mbps=c.cpu_single_mbps,
                cpu_multi_mbps=c.cpu_multi_mbps,
                vcpus=c.vcpus,
                ram_total_gb=c.ram_total_gb,
                duration_ms=c.duration_ms,
            )
        except Exception:
            logger.warning("host canary failed; Hello ships without it", exc_info=True)
        return pb.WorkerResources(
            host_canary=canary,
            gpu_count=int(hw.get("gpu_count") or 0),
            vram_total_bytes=int(hw.get("gpu_total_mem") or 0),
            gpu_name=str(hw.get("gpu_name") or ""),
            gpu_sm=str(hw.get("gpu_sm") or ""),
            installed_libs=[str(x) for x in (hw.get("installed_libs") or [])],
            image_digest=self._settings.worker_image_digest,
            # git_commit intentionally unpopulated (pgw#514/P4): no launcher
            # ever set WORKER_GIT_COMMIT and Go never read WorkerResources
            # .git_commit — dead on both ends. Field stays on the wire
            # (deleting it needs a coordinated tensorhub proto update).
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
        # Full-replace config: file base URL + desired model residency.
        self.executor.file_base_url = ack.file_base_url or ""
        # th#697: apply the hub's precision-ladder picks for THIS card
        # (full-replace: refs absent from the map revert to declared).
        self.executor.apply_model_resolutions({
            r.ref: (r.resolved_ref, r.cast) for r in ack.resolutions
        })
        desired = pb.DesiredResidency()
        desired.CopyFrom(ack.desired_residency)
        generation = int(desired.generation)
        if generation < self._observed_residency_generation:
            logger.info(
                "ignoring stale desired residency generation %d (observed %d)",
                generation, self._observed_residency_generation,
            )
        else:
            self._observed_residency_generation = generation
            self.executor.store.keep = list(dict.fromkeys(ref for ref in desired.disk_refs if ref))
            self.executor.store.replace_desired_snapshots(
                dict(desired.snapshots), generation=generation,
            )
            self._replace_residency_reconcile(desired)
        # New connection: per-worker fn disables/degradations were wiped by
        # Hello. Capacity evidence has causal priority over retained results;
        # other finite baseline messages follow it in the same prepend lane.
        self._emitted_unavailable.clear()
        self._emitted_degraded.clear()
        capacity_replay = await self.executor.host_ram_capacity_replay()
        if capacity_replay:
            if self.transport is not None:
                # Nonblocking reconnect lane: active failures, then undelivered progress,
                # both ahead of durable results. The queue dedupes a repeated
                # midstream HelloAck within this connection epoch.
                await self.transport.prepend_reconnect(capacity_replay)
            else:
                # Unit/embedded lifecycle without a Transport still observes
                # the same ordered replay contract.
                for message in capacity_replay:
                    await self.executor._send(message)
        await self._emit_unavailable(hello_ack=True)
        self._last_delta = None
        await self.maybe_send_state_delta(hello_ack=True)

    def _replace_residency_reconcile(self, desired: "pb.DesiredResidency") -> None:
        self._cancel_residency_reconcile()
        self._desired_residency = desired
        self._resume_residency_reconcile()

    def _cancel_residency_reconcile(self) -> None:
        task = getattr(self, "_residency_task", None)
        if task is not None:
            task.cancel()
        self._residency_task = None

    def _resume_residency_reconcile(self) -> None:
        desired = getattr(self, "_desired_residency", None)
        if desired is None or self.draining:
            return
        self._residency_task = asyncio.create_task(
            self._reconcile_residency(desired),
            name=f"residency-{int(desired.generation)}",
        )

    async def _reconcile_residency(self, desired: "pb.DesiredResidency") -> None:
        """Converge in declared order while tenant work has first claim."""
        snapshots = dict(desired.snapshots)
        try:
            for ref in desired.disk_refs:
                if not ref:
                    continue
                await self.executor.wait_idle()
                if self.draining:
                    return
                try:
                    await self.executor.revalidate_snapshot_identity(
                        ref, snapshots.get(ref)
                    )
                    await self.executor.store.ensure_local(ref, snapshots.get(ref))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("desired disk residency failed for %s: %s", ref, exc)

            for instance in desired.hot:
                await self.executor.wait_idle()
                if self.draining:
                    return
                try:
                    await self.executor.ensure_desired_instance(instance, snapshots)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "desired hot residency failed for %s: %s",
                        instance.function_name, exc,
                    )
        except asyncio.CancelledError:
            return

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        which = msg.WhichOneof("msg")
        if which == "run_job":
            # A tenant request preempts unrelated background transfer/setup.
            # Re-running desired state after idle is cheap: local refs and
            # ready instances short-circuit through the existing dedupe paths.
            self._cancel_residency_reconcile()
            try:
                await self.executor.handle_run_job(msg.run_job)
            finally:
                self._resume_residency_reconcile()
        elif which == "cancel_job":
            self.executor.handle_cancel(msg.cancel_job)
        elif which == "model_op":
            # Compile-cache adoption must never block the receive path.
            asyncio.create_task(self._adopt_compile_cache_then_delta(msg.model_op))
        elif which == "drain":
            self.start_drain(int(msg.drain.deadline_ms or 0))

    async def _adopt_compile_cache_then_delta(self, op: pb.ModelOp) -> None:
        try:
            await self.executor.handle_model_op(op)
        finally:
            await self.maybe_send_state_delta()

    async def on_disconnect(self) -> None:
        self._last_delta = None
        self._emitted_unavailable.clear()
        self._emitted_degraded.clear()

    async def on_message_shipped(self, msg: pb.WorkerMessage) -> None:
        """Retire delivery-owned capacity progress after stream.write succeeds."""
        if msg.WhichOneof("msg") == "model_event":
            await self.executor.host_ram_capacity_delivered(msg.model_event)

    # ---- state emission --------------------------------------------------------

    def state_changed(self) -> None:
        """Sync callback from the executor; coalesced onto the loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.maybe_send_state_delta())

    async def _send_state_message(
        self, message: pb.WorkerMessage, *, hello_ack: bool = False,
    ) -> None:
        if self.transport is None:
            return
        prepend = getattr(self.transport, "prepend_reconnect", None)
        if hello_ack and prepend is not None:
            await prepend([message])
            return
        await self.transport.send(message)

    async def maybe_send_state_delta(self, *, hello_ack: bool = False) -> None:
        if self.transport is None or not self.transport.connected:
            return
        delta = self._state_delta()
        raw = delta.SerializeToString(deterministic=True)
        if raw != self._last_delta:
            self._last_delta = raw
            await self._send_state_message(
                pb.WorkerMessage(state_delta=delta), hello_ack=hello_ack,
            )
        # Deduped internally; must run even on an unchanged delta — a runtime
        # ladder demotion (gw#463) changes the ServePlan, not the delta bytes.
        await self._emit_unavailable(hello_ack=hello_ack)
        await self._emit_degraded(hello_ack=hello_ack)

    async def _emit_unavailable(self, *, hello_ack: bool = False) -> None:
        if self.transport is None:
            return
        # A recovered function leaves executor.unavailable; drop it from the
        # dedupe set so a later re-failure re-emits.
        self._emitted_unavailable &= set(self.executor.unavailable)
        for name, (reason, detail, axes) in list(self.executor.unavailable.items()):
            if name in self._emitted_unavailable:
                continue
            self._emitted_unavailable.add(name)
            await self._send_state_message(
                pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
                    function_name=name, reason=reason, detail=detail, axes=axes,
                )),
                hello_ack=hello_ack,
            )

    async def _emit_degraded(self, *, hello_ack: bool = False) -> None:
        """th#683 P3: per served function running degraded, tell the
        orchestrator STRUCTURALLY (FnDegraded, not just a log line) so
        placement learns "this release degraded on this card — it wants a
        bigger one". Emitting rides the orchestrator transport, so cozy-local
        (no orchestrator) never sends it — there the honest-guidance advisory
        on the terminal is the surface."""
        if self.transport is None:
            return
        for name, plan in sorted(self.executor.serve_plans.items()):
            if not plan.degraded or name in self.executor.unavailable:
                continue
            ran = plan.ran or plan.run_mode
            if self._emitted_degraded.get(name) == ran:
                continue
            self._emitted_degraded[name] = ran
            await self._send_state_message(
                pb.WorkerMessage(fn_degraded=pb.FnDegraded(
                    function_name=name,
                    wanted=plan.wanted,
                    ran=ran,
                    reason=plan.warning,
                    est_latency_multiplier=float(plan.est_latency_multiplier),
                    recommended_vram_gb=float(plan.recommended_vram_gb or 0.0),
                )),
                hello_ack=hello_ack,
            )

    async def set_phase(self, phase: "pb.WorkerPhase") -> None:
        if phase == self.phase:
            return
        self.phase = phase
        await self.maybe_send_state_delta()

    # ---- startup ---------------------------------------------------------------

    async def startup(self) -> None:
        """Gate functions, prefetch worker-fetchable models with retry/backoff,
        set up endpoints, advance phases. Never raises: failures gate
        individual functions, not the process."""
        # Disk truth first: after a restart the CAS dir is full while Residency
        # starts empty — rescan so Hello.models (and disk GC) see reality.
        self.executor.store.rescan_disk()
        self.executor.gate_functions(self.hardware)

        from .api.binding import wire_ref

        prefetch_refs: List[str] = []
        for spec in self.executor.specs.values():
            if spec.name in self.executor.unavailable:
                continue
            for slot, binding in spec.models.items():
                if slot in spec.slots:
                    # pgw#532: a declared Slot's default_checkpoint is a SEED
                    # for the hub mapping, not a load instruction — the hub
                    # resolves the slot (registered binding / per-request
                    # pick) and drives DOWNLOADs itself. A hub-connected
                    # worker never self-fetches the raw upstream default
                    # (mirror-first, gw#465 — the fc157 civitai_not_found
                    # boot failure).
                    continue
                ref = wire_ref(binding)
                if binding.source != "tensorhub" and ref not in prefetch_refs:
                    # hf/civitai refs need no orchestrator snapshot; tensorhub
                    # refs arrive via DesiredResidency / RunJob (§7).
                    prefetch_refs.append(ref)

        if prefetch_refs:
            await self.set_phase(pb.WORKER_PHASE_DOWNLOADING_MODELS)
            for ref in prefetch_refs:
                try:
                    await self.executor.store.ensure_local(ref)
                except Exception as exc:
                    logger.error("startup prefetch of %s failed terminally: %s", ref, exc)

        await self.set_phase(pb.WORKER_PHASE_LOADING_PIPELINES)
        awaiting_hub: Dict[str, List[str]] = {}
        dynamic: List[str] = []
        for spec in list(self.executor.specs.values()):
            if spec.name in self.executor.unavailable:
                continue
            if spec.slots:
                # pgw#532: dynamic slots materialize the HUB-resolved ref
                # (DesiredResidency pre-warms / RunJob supplies snapshots),
                # per dispatch, instance-per-pick. Setting up eagerly here
                # would load the code seed — the exact fc157 setup-failure
                # bug (raw civitai default -> civitai_not_found -> every
                # function load_failed).
                dynamic.append(spec.name)
                continue
            missing = sorted({
                wire_ref(b) for b in spec.models.values()
                if b.source == "tensorhub"
                and self.executor.store.local_path(wire_ref(b)) is None
            })
            if missing:
                # Waits for DesiredResidency / RunJob snapshots (§7). This wait
                # is unbounded and hub-driven: a release without resolved
                # desired bindings leaves the function in loading_functions
                # forever (ie#455 z-image fns=[]), so say so loudly instead of
                # dropping the function in silence.
                awaiting_hub[spec.name] = missing
                continue
            try:
                await self.executor.ensure_setup(spec)
            except Exception as exc:
                logger.error("startup setup of %s failed: %s", spec.name, exc)
        if dynamic:
            logger.info(
                "dynamic-slot functions serve hub-resolved picks per dispatch "
                "(pgw#532; no boot-time setup): %s", ", ".join(sorted(dynamic)))
        if awaiting_hub:
            logger.warning(
                "functions waiting on hub-supplied snapshots (DesiredResidency, "
                "contract §7) and NOT yet servable: %s — if these never arrive, "
                "check that the release has resolved desired bindings for these refs",
                "; ".join(f"{fn} <- {', '.join(refs)}" for fn, refs in sorted(awaiting_hub.items())),
            )

        await self.set_phase(pb.WORKER_PHASE_READY)

    # ---- drain -------------------------------------------------------------------

    def start_drain(self, deadline_ms: int) -> None:
        if self.draining:
            return
        self._begin_drain(deadline_ms)
        self._drain_task = asyncio.create_task(self._finish_drain(), name="drain")

    async def drain(self, deadline_ms: int = 0) -> None:
        """stop admitting -> finish in-flight -> ship buffered results ->
        close the stream -> signal exit 0. Zero waits without a cutoff."""
        if self.draining:
            return
        self._begin_drain(deadline_ms)
        await self._finish_drain()

    def _begin_drain(self, deadline_ms: int) -> None:
        """Synchronously stop admission and anchor the deadline at receipt."""
        self.draining = True
        self.executor.draining = True
        self._cancel_residency_reconcile()
        logger.info("drain started (deadline_ms=%d)", deadline_ms)
        deadline_s = (deadline_ms / 1000.0) if deadline_ms > 0 else None
        loop = asyncio.get_running_loop()
        self._drain_deadline_at = loop.time() + deadline_s if deadline_s is not None else None
    async def _finish_drain(self) -> None:
        deadline_at = self._drain_deadline_at
        loop = asyncio.get_running_loop()
        await self.maybe_send_state_delta()

        wait_timeout = None if deadline_at is None else max(0.0, deadline_at - loop.time())
        finished = await self.executor.wait_idle(timeout=wait_timeout)
        if not finished:
            logger.warning("drain deadline expired; aborting remaining jobs as RETRYABLE")
            await self.executor.abort_all(safe_message="worker draining")

        await self.executor.shutdown_instances()
        if self.transport is not None:
            flush_timeout = None if deadline_at is None else max(0.0, deadline_at - loop.time())
            await self.transport.close_after_flush(timeout=flush_timeout)
        self.drained.set()
        logger.info("drain complete")
