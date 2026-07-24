"""Connection lifecycle: Hello/HelloAck + in_flight reconcile, edge-triggered
StateDelta full-replace snapshots, FnUnavailable + FnDegraded emission,
startup phases, and a drain that actually drains (stop admitting -> finish
in-flight -> ship results -> close stream -> exit 0).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from . import activity as activity_mod
from .config import Settings
from .executor import Executor
from .pb import worker_scheduler_pb2 as pb
from .runtime_config import extract_config_push
from .transport import PROTOCOL_VERSION, Transport

logger = logging.getLogger(__name__)

# th#1087 echo fields, present once the A+C lane's proto lands; guarded so
# the worker runs against either proto vintage.
_DELTA_HAS_CONFIG_GEN = "observed_config_generation" in pb.StateDelta.DESCRIPTOR.fields_by_name

_VRAM_QUANTUM_FRACTION = 0.05  # quantize free-VRAM deltas to 5% of total

# gw#591 boot-setup watcher: poll cadence for hub-delivered snapshots of
# functions the startup scan left awaiting (store lookups are local + cheap).
_BOOT_SETUP_WATCH_INTERVAL_S = 2.0

# th#965 layer 2: universal app-level heartbeat cadence, declared in Hello.
# The beat task lives on the asyncio event loop — the control loop that owes
# all progress, never a detached timer thread — so beats stop exactly when
# that loop wedges. Each tick force-sends the full StateDelta (unchanged
# bytes included). A stuck coroutine leaves the loop (and beats) alive; the
# hub's layer-3 obligation invariant covers that mode.
# 10s beat / 6 misses (Paul, 2026-07-21): one missed beat costs 10s of slack
# and a transient stall (GC pause, scheduler hiccup) never reads as death,
# while detection stays ~60s. CONTRACT §3 event-loop discipline follows:
# worker code must never block the loop longer than the miss window — long
# synchronous work (torch.compile, model loads, CUDA sync) runs in executor
# threads (executor._to_thread_complete / asyncio.to_thread).
HEARTBEAT_INTERVAL_MS = 10_000

# pgw#610 disk stats keep their original 30s measurement cadence: the report
# rides every beat, but the statvfs/ref-index scan is recomputed at most
# every _DISK_REPORT_TTL_S — a beat between refreshes re-ships the cached
# report (identical bytes, generation unchanged).
_DISK_REPORT_TTL_S = 30.0


def probe_hardware() -> Dict[str, Any]:
    """Static hardware facts + gate inputs. torch is optional."""
    info: Dict[str, Any] = {
        "gpu_count": 0, "gpu_total_mem": 0, "gpu_free_mem": 0,
        "gpu_name": "", "gpu_sm": "", "torch_version": "", "installed_libs": [],
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
        info["torch_version"] = str(caps.torch_version or "")
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


def _snapshot_content_key(snap: Optional["pb.Snapshot"]) -> tuple:
    """Snapshot CONTENT identity: digest + per-file (path, size, blake3).

    Presigned URL bytes are deliberately excluded — the hub refreshes
    expiring snapshot URLs on every release-config rebuild (~15s TTL) and
    its own HelloAck semantic hash excludes them for exactly this reason
    (tensorhub desiredResidencySemanticHash). Hashing URLs made every
    URL rotation look like a model change and cancelled in-flight warmup
    loads at ~15s / ~10min cadence (gw#623)."""
    if snap is None:
        return ()
    return (
        snap.digest,
        tuple(sorted(
            (f.path, f.size_bytes, f.blake3) for f in snap.files
        )),
    )


def _semantic_model_key(
    ack: "pb.HelloAck", desired: "pb.DesiredResidency",
) -> tuple:
    """gw#614: order-independent identity of the MODEL content of an ack —
    resolutions, disk refs, snapshot content, hot instances. Generation,
    presigned URL bytes, and other non-model fields are excluded so benign
    plan rewrites compare equal."""
    return (
        tuple(sorted(
            (r.ref, r.resolved_ref, r.cast, r.lane) for r in ack.resolutions
        )),
        tuple(sorted(ref for ref in desired.disk_refs if ref)),
        tuple(sorted(
            (ref, _snapshot_content_key(snap))
            for ref, snap in desired.snapshots.items()
        )),
        tuple(sorted(
            inst.SerializeToString(deterministic=True) for inst in desired.hot
        )),
    )


def _instance_refs(instance: "pb.DesiredInstance") -> list:
    refs = []
    for model in instance.models:
        refs.append(model.ref)
        refs.extend(lora.ref for lora in model.loras)
    return refs


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
        self._boot_setup_watch: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._disk_report_at = 0.0
        self._disk_report_refresh_task: Optional[asyncio.Task] = None
        self._drain_deadline_at: Optional[float] = None
        self._desired_residency: Optional[pb.DesiredResidency] = None
        self._residency_task: Optional[asyncio.Task] = None
        self._observed_residency_generation = 0
        # gw#623: the reconcile loop's currently-loading work item
        # ((kind, identity, context)) + the level-trigger that re-runs a
        # convergence pass instead of cancelling an in-flight load.
        self._reconcile_active: Optional[tuple] = None
        self._residency_restart: Optional[asyncio.Event] = None
        self._model_resolutions: Dict[str, Tuple[str, str, str]] = {}
        # th#1087: first config generation seen this boot — the hub's env-
        # class staleness input (envs are boot-injected; the worker never
        # mutates them, it just keeps echoing what it booted with).
        self._boot_config_generation = 0

    # ---- snapshots -----------------------------------------------------------

    def _state_delta(self) -> pb.StateDelta:
        free = free_vram_bytes()
        total = int(self.hardware.get("gpu_total_mem") or 0)
        quantum = max(1, int(total * _VRAM_QUANTUM_FRACTION)) if total else 1
        # th#1087/th#1085 worker half: echo the applied config generation
        # (and the boot-time one for env-class staleness) once the proto
        # carries the fields.
        config_echo: Dict[str, int] = {}
        if _DELTA_HAS_CONFIG_GEN:
            config_echo["observed_config_generation"] = (
                self.executor.runtime_config.generation
            )
            if "boot_config_generation" in pb.StateDelta.DESCRIPTOR.fields_by_name:
                config_echo["boot_config_generation"] = self._boot_config_generation
        return pb.StateDelta(
            **config_echo,
            phase=self.phase,
            available_functions=self.executor.available_functions(),
            loading_functions=self.executor.loading_functions(),
            free_vram_bytes=(free // quantum) * quantum,
            # gw#516: encode/upload tails past the GPU-slot release. The hub
            # must not drain/retire this worker on GPU-idleness alone.
            finalizing_jobs=self.executor.finalizing_jobs(),
            observed_residency_generation=self._observed_residency_generation,
            compile_targets=self.executor.compile_targets(),
            # th#883 pull-by-key: worker-computed cell keys the hub may look
            # up in its store (boot attach) — never hub-side selection input.
            cell_lookups=self.executor.cell_lookups(),
            # pgw#610/th#962: measured per-tier disk telemetry. Rides every
            # StateDelta (and thus Hello.state). boothang fix: this ONLY
            # reads ModelStore's cache (never blocks) — the actual statvfs
            # measurement is a fire-and-forget background task kicked by
            # _kick_disk_usage_refresh_if_stale, gated to at most every
            # _DISK_REPORT_TTL_S. A synchronous (or awaited-inline) refresh
            # here could freeze the event loop — and thus the th#965
            # heartbeat sharing the same loop — for as long as a stalled
            # provider VOLUME mount's statvfs() call blocks (the 0.40.7
            # LTX post-seal_publish hang).
            disk_usage=self.executor.store.disk_usage_report(),
        )

    def _kick_disk_usage_refresh_if_stale(self) -> None:
        """FIRE-AND-FORGET off-loop (asyncio.to_thread) disk-usage refresh,
        gated to at most every _DISK_REPORT_TTL_S. Called from
        maybe_send_state_delta — never awaited there and never inline in
        _state_delta() (see boothang fix note above): a send must not wait
        on the measurement any more than the event loop should. A stalled
        provider VOLUME mount just means this boot's disk telemetry stays
        stale until the mount recovers — every StateDelta/heartbeat/RunJob
        keeps flowing on schedule regardless."""
        now = time.monotonic()
        if self._disk_report_at and now - self._disk_report_at < _DISK_REPORT_TTL_S:
            return
        if (self._disk_report_refresh_task is not None
                and not self._disk_report_refresh_task.done()):
            return  # a refresh is already in flight
        self._disk_report_at = now

        async def _run() -> None:
            try:
                await self.executor.store.refresh_disk_usage_report()
            except Exception:
                logger.warning(
                    "disk-usage measurement failed; keeping the last "
                    "cached report", exc_info=True)

        self._disk_report_refresh_task = asyncio.create_task(
            _run(), name="disk-usage-refresh")

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
        # gw#587/th#910: the hub attests the self-mint publish's gen_worker
        # axis against THIS Hello-reported version (absent => publish fails
        # closed, which old workers never hit — they don't publish).
        try:
            from .compile_cache import gen_worker_version

            gw_version = gen_worker_version()
        except Exception:
            gw_version = ""
        return pb.WorkerResources(
            host_canary=canary,
            gpu_count=int(hw.get("gpu_count") or 0),
            vram_total_bytes=int(hw.get("gpu_total_mem") or 0),
            gpu_name=str(hw.get("gpu_name") or ""),
            gpu_sm=str(hw.get("gpu_sm") or ""),
            torch_version=str(hw.get("torch_version") or ""),
            installed_libs=[str(x) for x in (hw.get("installed_libs") or [])],
            gen_worker_version=gw_version,
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
            # th#965 layer 2: promise the beat cadence; the hub reaps after
            # 6 consecutive misses (~60s). Hello counts as the first beat.
            heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
        )

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        # Full-replace config: file base URL + desired model residency.
        self.executor.file_base_url = ack.file_base_url or ""
        # th#1087 class-1 parameters: update memory + rewrite the local
        # snapshot file (per-invoke subprocesses read it). Bindings (class 2)
        # ride the desired-residency reconcile below; envs (class 3) are
        # boot-only — the worker just keeps echoing its boot generation.
        push = extract_config_push(ack)
        if push is not None:
            if self._boot_config_generation == 0:
                self._boot_config_generation = push.config_generation
            self.executor.runtime_config.apply(
                generation=push.config_generation,
                parameters=push.parameters,
                release_id=push.release_id or self.release_id,
            )
        # th#697: apply the hub's precision-ladder picks for THIS card
        # (full-replace: refs absent from the map revert to declared).
        resolutions = {
            r.ref: (r.resolved_ref, r.cast, r.lane) for r in ack.resolutions
        }
        self.executor.apply_model_resolutions(resolutions)
        # gw#623: the reconcile's active-work context compares against the
        # resolutions actually applied to the executor.
        self._model_resolutions = resolutions
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
            self._replace_residency_reconcile(
                desired, model_key=_semantic_model_key(ack, desired))
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

    def _replace_residency_reconcile(
        self, desired: "pb.DesiredResidency", *, model_key: Any = None,
    ) -> None:
        # gw#614 (th#961 defense in depth): an ack whose semantic model set
        # is unchanged must not kill an in-flight reconcile — a running
        # self_mint_compile needs its full window.
        task = getattr(self, "_residency_task", None)
        running = task is not None and not task.done()
        if (
            model_key is not None
            and running
            and getattr(self, "_residency_model_key", None) == model_key
        ):
            self._desired_residency = desired
            return
        self._residency_model_key = model_key
        self._desired_residency = desired
        if running and self._active_work_still_desired(desired):
            # gw#623: the set changed, but the model this reconcile is
            # loading RIGHT NOW is still wanted with the same identity and
            # resolution. Cancelling would discard minutes of load over a
            # benign update (sibling ref added/removed, plan rewrite);
            # signal the level-triggered loop to re-converge against the
            # new set once the current item completes.
            self._signal_residency_restart()
            if not task.done():  # type: ignore[union-attr]
                return
            # The loop finished before observing the signal; fall through
            # and start a fresh reconcile against the new desired state.
        self._cancel_residency_reconcile()
        self._resume_residency_reconcile()

    def _signal_residency_restart(self) -> None:
        event = getattr(self, "_residency_restart", None)
        if event is None:
            event = self._residency_restart = asyncio.Event()
        event.set()

    def _work_context(
        self, refs: Any, desired: "pb.DesiredResidency",
    ) -> tuple:
        """Identity facts an in-flight load depends on, per involved ref:
        snapshot content (URL-free) + the applied precision resolution."""
        resolutions = getattr(self, "_model_resolutions", {})
        return tuple(
            (
                ref,
                _snapshot_content_key(desired.snapshots.get(ref)),
                resolutions.get(ref),
            )
            for ref in sorted({r for r in refs if r})
        )

    def _active_work_still_desired(
        self, desired: "pb.DesiredResidency",
    ) -> bool:
        """Whether the reconcile loop's CURRENT work item survives ``desired``
        unchanged — the gw#623 cancel test: only cancel an in-flight load
        when the model it is FOR left the set (or changed identity or
        resolution), never on unrelated set churn."""
        active = getattr(self, "_reconcile_active", None)
        if active is None:
            return True  # between items: the restart signal re-converges
        kind, ident, ctx = active
        if kind == "disk":
            if ident not in set(desired.disk_refs):
                return False
            return ctx == self._work_context((ident,), desired)
        for instance in desired.hot:
            if instance.SerializeToString(deterministic=True) == ident:
                return ctx == self._work_context(
                    _instance_refs(instance), desired)
        return False

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
            self._reconcile_residency(),
            name=f"residency-{int(desired.generation)}",
        )

    async def _reconcile_residency(self) -> None:
        """Converge to the LATEST desired state while tenant work has first
        claim. Level-triggered (gw#623): each pass reads the freshest set; a
        benign mid-load update re-runs the pass instead of cancelling the
        in-flight item (already-satisfied refs/instances short-circuit
        through the ordinary dedupe paths, so a re-pass is cheap)."""
        restart = getattr(self, "_residency_restart", None)
        if restart is None:
            restart = self._residency_restart = asyncio.Event()
        try:
            while True:
                restart.clear()
                desired = self._desired_residency
                if desired is None or self.draining:
                    return
                await self._reconcile_pass(desired, restart)
                if self.draining or not restart.is_set():
                    return
        except asyncio.CancelledError:
            return
        finally:
            self._reconcile_active = None

    async def _reconcile_pass(
        self, desired: "pb.DesiredResidency", restart: asyncio.Event,
    ) -> None:
        """One convergence pass over ``desired`` in declared order."""
        snapshots = dict(desired.snapshots)
        for ref in desired.disk_refs:
            if not ref:
                continue
            if restart.is_set():
                return
            await self.executor.wait_idle()
            if self.draining:
                return
            self._reconcile_active = (
                "disk", ref, self._work_context((ref,), desired))
            try:
                await self.executor.revalidate_snapshot_identity(
                    ref, snapshots.get(ref)
                )
                await self.executor.store.ensure_local(ref, snapshots.get(ref))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("desired disk residency failed for %s: %s", ref, exc)
            finally:
                self._reconcile_active = None

        for instance in desired.hot:
            if restart.is_set():
                return
            await self.executor.wait_idle()
            if self.draining:
                return
            self._reconcile_active = (
                "hot",
                instance.SerializeToString(deterministic=True),
                self._work_context(_instance_refs(instance), desired),
            )
            try:
                await self.executor.ensure_desired_instance(instance, snapshots)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "desired hot residency failed for %s: %s",
                    instance.function_name, exc,
                )
            finally:
                self._reconcile_active = None

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

    async def maybe_send_state_delta(
        self, *, hello_ack: bool = False, force: bool = False,
    ) -> None:
        if self.transport is None or not self.transport.connected:
            return
        self._kick_disk_usage_refresh_if_stale()
        delta = self._state_delta()
        raw = delta.SerializeToString(deterministic=True)
        # force (th#965 layer 2): the heartbeat tick re-sends an unchanged
        # snapshot — receipt IS the beat the hub timestamps.
        if force or raw != self._last_delta:
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
        # th#965 layer 2: the beat starts BEFORE any boot work so a hang
        # anywhere in setup is still covered — the task shares this event
        # loop, so it beats iff the loop is servicing tasks.
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="heartbeat")
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
            if spec.slots or spec.compile is not None:
                # pgw#532 (slots) + gw#584 (compile): Slot picks and compile
                # cells are set up ONLY on hub delivery — Hot DesiredInstance
                # or RunJob, both of which rebind through _effective_spec
                # with the hub-stamped refs. th#938: th#912's watcher ran
                # ensure_setup on the class-table spec here, materializing
                # the image-baked code default over the hub-stamped release
                # binding (sdxl's Civitai default -> civitai_not_found ->
                # both fns setup_failed). The code default is the hub-less
                # bootstrap fallback only.
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
                "hub-resolved functions (dynamic slots / compile cells) set up "
                "on delivery, not at boot (pgw#532, gw#584): %s",
                ", ".join(sorted(dynamic)))
        if awaiting_hub:
            logger.warning(
                "functions waiting on hub-supplied snapshots (DesiredResidency, "
                "contract §7) and NOT yet servable: %s — if these never arrive, "
                "check that the release has resolved desired bindings for these refs",
                "; ".join(f"{fn} <- {', '.join(refs)}" for fn, refs in sorted(awaiting_hub.items())),
            )
            # gw#591: the hub's desired-disk plan delivers those snapshots
            # seconds later, but nothing re-ran setup — the function sat in
            # loading_functions forever while the hub dispatches only to
            # advertised functions (each side waiting on the other; found
            # live, ie#519). Finish boot setup the moment the refs land.
            self._boot_setup_watch = asyncio.create_task(
                self._setup_awaiting_functions(awaiting_hub),
                name="boot-setup-watch")

        await self.set_phase(pb.WORKER_PHASE_READY)

    async def _heartbeat_loop(self) -> None:
        """th#965 layer 2 + pgw#610: force-send the full StateDelta (which
        carries the refreshed measured-disk report) every beat, in EVERY
        state including drain — the beat only ends when the process does."""
        while not self.drained.is_set():
            await asyncio.sleep(HEARTBEAT_INTERVAL_MS / 1000.0)
            await self.maybe_send_state_delta(force=True)
            # gw#621: progress counters piggyback on the same beat — one
            # counter-carrying ActivityUpdate per tick while an activity is
            # open, plus the typed self-diagnosis on a stalled registry.
            activity_mod.on_beat()

    async def _setup_awaiting_functions(
        self, awaiting: Dict[str, List[str]]
    ) -> None:
        """Complete boot setup for functions whose tensorhub snapshots arrive
        via hub delivery after the startup scan (gw#591), then push a
        StateDelta so ``available_functions`` advertises them."""
        pending = {fn: list(refs) for fn, refs in awaiting.items()}
        while pending and not self.draining:
            for fn in sorted(pending):
                left = [r for r in pending[fn]
                        if self.executor.store.local_path(r) is None]
                if left:
                    pending[fn] = left
                    continue
                del pending[fn]
                spec = self.executor.specs.get(fn)
                if spec is None or fn in self.executor.unavailable:
                    continue
                try:
                    await self.executor.ensure_setup(spec)
                    logger.info(
                        "boot setup of %s completed after hub snapshot "
                        "delivery (gw#591)", fn)
                except Exception as exc:
                    logger.error("startup setup of %s failed: %s", fn, exc)
                await self.maybe_send_state_delta()
            if pending:
                await asyncio.sleep(_BOOT_SETUP_WATCH_INTERVAL_S)

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
        # th#965: the heartbeat deliberately keeps beating through drain — a
        # worker hung mid-drain must still be detectable as dead. It is
        # cancelled after the stream closes in _finish_drain.
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
        # getattr: stubbed Lifecycles skip __init__ (same convention as
        # _cancel_residency_reconcile, pgw#610).
        beat_task = getattr(self, "_heartbeat_task", None)
        if beat_task is not None:
            beat_task.cancel()
            self._heartbeat_task = None
        logger.info("drain complete")
