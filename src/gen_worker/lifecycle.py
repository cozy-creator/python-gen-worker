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
from . import boot_phases as boot_mod
from . import content_credentials
from . import fleet_cells
from . import receipts
from . import serve_posture
from . import progress as progress_mod
from .models.records import shutdown_instances
from .models.refs import WireRef
from .wire_snapshots import index_snapshots
from .config import Settings
from .config.settings import BOOT_CONFIG_GENERATION_ABSENT
from .executor import Executor
from .lifecycle_intents import IntentRegistry, UnreportedIntentWait
from .pb import worker_scheduler_pb2 as pb
from .runtime_config import ConfigSnapshotWriteError, extract_config_push
from .transport import (
    FatalTransportError,
    KEEPALIVE_TIMEOUT_S,
    PROTOCOL_VERSION,
    Transport,
)
from .api.binding import wire_ref
# module import: tests monkeypatch worker_fatal.report_worker_error_async; stay late-bound.
from . import worker_fatal
from .topology import TopologyError, delivered_topology
from . import hostfacts

logger = logging.getLogger(__name__)

_VRAM_QUANTUM_FRACTION = 0.05  # quantize free-VRAM deltas to 5% of total

# How often the drain re-asks the progress registry while it waits for
# in-flight work. A CADENCE, not a verdict — it bounds latency-to-notice and can
# never end anything; the give-up test is `progress.self_diagnosis()`.
_DRAIN_PROGRESS_POLL_S = 1.0

# Poll cadence for hub-delivered snapshots of functions the startup scan left
# awaiting (store lookups are local + cheap).
_BOOT_SETUP_WATCH_INTERVAL_S = 2.0

# Universal app-level heartbeat cadence, declared in Hello. The beat task lives
# on the asyncio event loop, never a detached timer thread, so beats stop
# exactly when that loop wedges; each tick force-sends the full StateDelta.
# 10s beat / 6 misses (Paul, 2026-07-21): one missed beat costs 10s of slack so
# a transient stall (GC pause, scheduler hiccup) never reads as death, while
# detection stays ~60s. Worker code must therefore never block the loop longer
# than the miss window — long synchronous work (torch.compile, model loads,
# CUDA sync) runs in executor threads.
HEARTBEAT_INTERVAL_MS = 10_000

# The disk report rides every beat, but the statvfs/ref-index scan is
# recomputed at most every _DISK_REPORT_TTL_S — a beat between refreshes
# re-ships the cached report (identical bytes, generation unchanged).
_DISK_REPORT_TTL_S = 30.0


def probe_hardware() -> hostfacts.HostFacts:
    """Measure this host ONCE into one immutable :class:`HostFacts`.

    The single producer of the facts the fleet acts on. ``vram_free_bytes`` is
    the input `gate_functions` turns into `unavailable` + `serve_plans` for
    EVERY function, so on a wide pod it must be the free pool of the group with
    the LEAST room, not card 0's. The MIN is the honest single scalar until the
    fit ladder is per-rank: it never promises a group room it does not have.
    """
    gpu_count = 0
    vram_total = 0
    vram_free = 0
    gpu_name = ""
    gpu_sm = ""
    torch_version = ""
    cuda_version = ""
    driver_version = ""
    installed_libs: tuple[str, ...] = ()
    # The HOST driver, read from NVML rather than torch: it must stay readable
    # when the CUDA runtime is not. A cu130 build on a 570.x host imports fine,
    # reports every version string correctly, and dies on its first allocation.
    try:
        from .hardware_report import _nvidia_smi_driver_and_gpu

        driver_version, gpu_name = _nvidia_smi_driver_and_gpu()
    except Exception:
        pass
    try:
        count = hostfacts.device_count()
        if count:
            gpu_count = count
            gpu_name = hostfacts.device_identity(0)[0] or gpu_name
            vram_total = hostfacts.total_vram_bytes(0) or 0
            card0_free = hostfacts.free_vram_bytes(0) or 0
            vram_free = card0_free
            worst = _worst_group_free_vram_bytes()
            if worst and worst != card0_free:
                logger.info(
                    "fit inputs: gating on the least-free group (%d bytes) "
                    "instead of card 0 (%d bytes) — pgw#776",
                    int(worst), int(card0_free))
                vram_free = int(worst)
    except TopologyError:
        # A WEDGED fabric is the one topology fault that must reach the caller:
        # `delivered_topology` raises it so the hub re-packs instead of this
        # worker booting and hanging every collective. Swallowed here it read
        # as "card 0, fine" and the pod went on to serve.
        raise
    except Exception:
        pass
    try:
        from .models.hub_policy import detect_worker_capabilities

        caps = detect_worker_capabilities()
        installed_libs = tuple(str(x) for x in (caps.installed_libs or []))
        torch_version = str(caps.torch_version or "")
        cuda_version = str(caps.cuda_version or "")
        if caps.gpu_sm:
            gpu_sm = str(int(caps.gpu_sm))
    except Exception:
        pass
    return hostfacts.HostFacts(
        gpu_count=gpu_count,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        gpu_name=gpu_name,
        gpu_sm=gpu_sm,
        torch_version=torch_version,
        cuda_version=cuda_version,
        driver_version=driver_version,
        installed_libs=installed_libs,
    )


_MULTI_GPU_NO_TOPOLOGY_WARNED = False


def _worst_group_free_vram_bytes() -> int:
    """The free pool of the least-roomy execution group, or 0 if unknowable.

    Reads the DELIVERED topology: on a pod this worker itself demoted for a
    non-NVLink fabric, ``from_env`` describes a packing that is not being
    served, so reporting from it makes reported and served topologies disagree.
    """

    groups = delivered_topology().all_groups()
    return int(min((g.free_vram_bytes() for g in groups), default=0))


def free_vram_bytes() -> int:
    """Free VRAM the hub may admit against (``StateDelta.free_vram_bytes``).

    Never a sum across devices and never the MAX: VRAM is not fungible between
    cards, and the hub admits against this single scalar, so it can admit G
    concurrent jobs whose sum is G times what any one group has. The only safe
    answer is the MIN across groups — MIN within a replicated group too, since
    every member holds a full copy — so every job the hub may admit fits where
    it lands. At G == 1, MIN and MAX coincide.

    Multi-GPU host with NO delivered topology (every harness attach and
    `cli serve`): the worker has ONE slot which places on the thread-current
    device, so card 0 is the honest number and the other cards are invisible to
    the hub. Named and logged once rather than left to be rediscovered.
    """
    try:
        worst = _worst_group_free_vram_bytes()
        if worst:
            _warn_once_if_gpus_are_invisible()
            return int(worst)
    except TopologyError:
        raise  # see `probe_hardware` — a wedged fabric is never a zero.
    except Exception:
        pass
    return hostfacts.free_vram_bytes(0) or 0


def _warn_once_if_gpus_are_invisible() -> None:
    global _MULTI_GPU_NO_TOPOLOGY_WARNED
    if _MULTI_GPU_NO_TOPOLOGY_WARNED:
        return
    try:
        from .topology import ENV_VAR

        if os.environ.get(ENV_VAR):
            return
        count = hostfacts.device_count()
        if count > 1 and delivered_topology().gpu_count == 1:
            _MULTI_GPU_NO_TOPOLOGY_WARNED = True
            logger.warning(
                "this host has %d CUDA devices and NO delivered %s: the worker "
                "serves ONE slot on card 0 and reports card 0's free VRAM. The "
                "other %d card(s) are invisible to the hub (they are not "
                "summed — VRAM is not fungible across cards). pgw#776",
                count, ENV_VAR, count - 1)
    except Exception:
        pass


def _snapshot_content_key(snap: Optional["pb.Snapshot"]) -> tuple:
    """Snapshot CONTENT identity: digest + per-file (path, size, digest).

    Presigned URL bytes are deliberately excluded — the hub refreshes expiring
    snapshot URLs on every release-config rebuild (~15s TTL) and its own
    HelloAck semantic hash excludes them for the same reason. Hashing them
    makes every URL rotation look like a model change and cancels in-flight
    warmup loads."""
    if snap is None:
        return ()
    return (
        snap.digest,
        tuple(sorted((f.path, f.size_bytes, f.digest) for f in snap.files)),
    )


def _semantic_model_key(
    ack: "pb.HelloAck",
    desired: "pb.DesiredResidency",
) -> tuple:
    """Order-independent identity of the MODEL content of an ack — resolutions,
    disk refs, snapshot content, hot instances. Generation, presigned URL bytes
    and other non-model fields are excluded so benign plan rewrites compare
    equal."""
    return (
        tuple(sorted((r.ref, r.resolved_ref, r.cast, r.lane) for r in ack.resolutions)),
        tuple(sorted(ref for ref in desired.disk_refs if ref)),
        tuple(
            sorted((ref, _snapshot_content_key(snap)) for ref, snap in desired.snapshots.items())
        ),
        tuple(sorted(inst.SerializeToString(deterministic=True) for inst in desired.hot)),
    )


def _desired_snapshots(desired: "pb.DesiredResidency") -> dict:
    """th#1941: the digest-keyed wire map, resolved to the ref-keyed view the
    residency layer materializes from."""
    return index_snapshots(
        desired.snapshots,
        [m for inst in desired.hot for m in inst.models],
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
        # The BOOTSTRAP token by intent: this runs before any stream exists and
        # reads IDENTITY claims (sub / release_id), which rotation never
        # changes. Anything reading a credential to AUTHENTICATE must use
        # `worker_credential.current()` instead.
        _boot_jwt = (settings.bootstrap_worker_jwt or "").strip()
        if _boot_jwt:
            from .request_context._helpers import _decode_unverified_jwt_claims

            claims = _decode_unverified_jwt_claims(_boot_jwt)
        self.worker_id = (
            settings.worker_id or str(claims.get("sub") or "").strip() or f"py-worker-{os.getpid()}"
        )
        # In the split's compute child there is no JWT to read claims from —
        # the parent decodes its own and hands the release id down as a plain
        # value. A claim is not a credential.
        self.release_id = (
            str(claims.get("release_id") or "").strip()
            or settings.worker_release_id.strip()
        )
        self.intent_registry = IntentRegistry(
            self.release_id,
            executor.specs.keys(),
            # Pass the value THROUGH, sentinel and all: a missing
            # WORKER_CONFIG_GENERATION means "no boot-only env was injected"
            # (-1), never "generation 0".
            boot_config_generation=int(getattr(
                settings, "boot_config_generation",
                BOOT_CONFIG_GENERATION_ABSENT,
            )),
            on_change=self.state_changed,
        )
        self.executor.bind_intent_registry(self.intent_registry)
        self.phase = pb.WORKER_PHASE_BOOTING
        self.hardware = probe_hardware()
        self.draining = False
        self.drained = asyncio.Event()  # set when drain completed -> exit 0
        self._last_delta: Optional[bytes] = None
        self._last_lifecycle_snapshot: Optional[bytes] = None
        self._emitted_unavailable: set[str] = set()
        # fn name -> the "ran" rung last reported. Keyed on the rung (not
        # mere membership) so a runtime ladder demotion re-emits.
        self._emitted_degraded: dict[str, str] = {}
        self._drain_task: Optional[asyncio.Task] = None
        self._boot_setup_watch: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._disk_report_at = 0.0
        self._disk_report_refresh_task: Optional[asyncio.Task] = None
        self._drain_deadline_at: Optional[float] = None
        self._drain_work_deadline_at: Optional[float] = None
        self._desired_residency: Optional[pb.DesiredResidency] = None
        self._residency_task: Optional[asyncio.Task] = None
        self._observed_residency_generation = 0
        # The reconcile loop's currently-loading work item
        # ((kind, identity, context)) + the level-trigger that re-runs a
        # convergence pass instead of cancelling an in-flight load.
        self._reconcile_active: Optional[tuple] = None
        self._residency_restart: Optional[asyncio.Event] = None
        self._model_resolutions: Dict[str, Tuple[Any, ...]] = {}
        # declared -> resolved for every disk_ref already reported as
        # superseded, so the typed event fires on transitions, not per ack.
        self._superseded_reported: Dict[str, str] = {}
        self.executor.runtime_config.set_projection_callbacks(
            on_parameter_snapshot=self._config_snapshot_applied,
            on_snapshot_failure=self._config_snapshot_failed,
        )

    # ---- snapshots -----------------------------------------------------------

    def _state_delta(self) -> pb.StateDelta:
        free = free_vram_bytes()
        total = int(self.hardware.vram_total_bytes)
        quantum = max(1, int(total * _VRAM_QUANTUM_FRACTION)) if total else 1
        return pb.StateDelta(
            # echo DesiredResidency.config_generation.
            observed_config_generation=self.executor.runtime_config.generation,
            phase=self.phase,
            available_functions=self.executor.available_functions(),
            loading_functions=self.executor.loading_functions(),
            free_vram_bytes=(free // quantum) * quantum,
            # encode/upload tails past the GPU-slot release. The hub
            # must not drain/retire this worker on GPU-idleness alone.
            finalizing_jobs=self.executor.finalizing_jobs(),
            observed_residency_generation=self._observed_residency_generation,
            # `cell_lookups` is NOT filled: a target states only the identity
            # it IS serving. The wire field retires with the RunJob cut.
            compile_targets=self.executor.compile_targets(),
            # Per-tier disk telemetry, riding every StateDelta (and thus
            # Hello.state). This ONLY reads ModelStore's cache and never
            # blocks — the statvfs measurement is a fire-and-forget background
            # task kicked by _kick_disk_usage_refresh_if_stale. A synchronous
            # (or awaited-inline) refresh here would freeze the event loop —
            # and thus the heartbeat sharing it — for as long as a stalled
            # provider VOLUME mount's statvfs() call blocks.
            disk_usage=self.executor.store.disk_usage_report(),
        )

    def _kick_disk_usage_refresh_if_stale(self) -> None:
        """FIRE-AND-FORGET off-loop disk-usage refresh, gated to at most every
        _DISK_REPORT_TTL_S. Called from maybe_send_state_delta — never awaited
        there and never inline in _state_delta(): a send must not wait on the
        measurement any more than the event loop should. A stalled provider
        VOLUME mount just leaves this boot's disk telemetry stale until the
        mount recovers; StateDelta/heartbeat/RunJob keep flowing regardless."""
        now = time.monotonic()
        if self._disk_report_at and now - self._disk_report_at < _DISK_REPORT_TTL_S:
            return
        if self._disk_report_refresh_task is not None and not self._disk_report_refresh_task.done():
            return  # a refresh is already in flight
        self._disk_report_at = now

        async def _run() -> None:
            try:
                await self.executor.store.refresh_disk_usage_report()
            except Exception:
                logger.warning(
                    "disk-usage measurement failed; keeping the last cached report", exc_info=True
                )

        self._disk_report_refresh_task = asyncio.create_task(_run(), name="disk-usage-refresh")

    # ---- transport handlers --------------------------------------------------

    def build_hello(self) -> pb.Hello:
        in_flight = {k for k in self.executor.in_flight_keys()}
        if self.transport is not None:
            in_flight.update(self.transport.queue.pending_result_keys)
        self.intent_registry.refresh_projection(
            self.executor,
            self._desired_residency,
            self._model_resolutions,
        )
        # NO `resources` field. `WorkerResources` has exactly ONE builder,
        # `procsplit.parent.ParentControl._parent_resources`, which measures
        # the silicon in a process that has imported no tenant code and stamps
        # it onto every relayed Hello. This process HAS imported tenant code,
        # so anything it measured about the host would be replaced wholesale
        # before the hub saw it — 53 lines computing a value the wire
        # discarded, and the fleet condemned SKUs on the other one (pgw#898).
        return pb.Hello(
            protocol_version=PROTOCOL_VERSION,
            worker_id=self.worker_id,
            release_id=self.release_id,
            state=self._state_delta(),
            models=self.executor.store.residency_snapshot(),
            in_flight=[
                pb.InFlightJob(request_id=rid, attempt=att) for rid, att in sorted(in_flight)
            ],
            # promise the beat cadence; the hub reaps after
            # 6 consecutive misses (~60s). Hello counts as the first beat.
            heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
            worker_session_id=self.intent_registry.worker_session_id,
            lifecycle_snapshot=self.intent_registry.snapshot(),
        )

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        # The stream exists now: bind the boot-phase sink and FLUSH everything
        # recorded before it did. This is the earliest point a boot row can
        # reach the hub at all — activity's own sink is not bound until
        # Executor.ensure_setup, after weights are on disk. The hub upserts on
        # (boot_id, ordinal), so the reconnect re-flush is harmless.
        try:
            boot_mod.bind_sink(self.executor._send, asyncio.get_running_loop())
            # ONCE per process: this handler runs again on every reconnect, and
            # "process start -> hello" measured on a reconnect hours later is
            # not a boot number.
            boot_mod.mark_once(boot_mod.PHASE_HELLO, since_process_start=True)
        except Exception:  # telemetry must never break the handshake
            logger.debug("boot phase sink bind failed", exc_info=True)
        desired_command: Optional[pb.DesiredStateCommand] = None
        if ack.HasField("desired_state_command"):
            desired_command = ack.desired_state_command
            receipt = self.intent_registry.apply_command(
                desired_command,
                current_config_generation=self.executor.runtime_config.generation,
            )
            await self._send_state_message(
                pb.WorkerMessage(goal_receipt=receipt),
                hello_ack=True,
            )
            if receipt.status == pb.GOAL_RECEIPT_STATUS_REJECTED:
                # A rejected v5 command never silently falls through to the
                # colocated legacy DesiredResidency. Mandatory rejection also
                # withdraws aggregate readiness until the pod is replaced.
                if self.intent_registry.protocol_rejected:
                    self._enter_error_phase(
                        "mandatory desired-state command rejected: "
                        + str(receipt.detail or "")[:512]
                    )
                await self._send_lifecycle_snapshot(hello_ack=True, force=True)
                self._last_delta = None
                await self.maybe_send_state_delta(hello_ack=True)
                return
        # Full-replace config: file base URL + desired model residency.
        self.executor.file_base_url = ack.file_base_url or ""
        # Arm the cell-receipt gate the moment the hub wiring exists: every
        # delivered compile cell is signature-verified before it may arm.
        if self.executor.file_base_url:
            receipts.configure(
                base_url=self.executor.file_base_url,
                worker_jwt=self.executor.worker_jwt_provider,
            )
            # The platform private key never enters this pod — the hub signs
            # the claim bytes. Until this call lands, a signing-configured
            # worker FAILS media requests rather than shipping them unsigned.
            content_credentials.configure_remote_signer(
                base_url=self.executor.file_base_url,
                worker_jwt=self.executor.worker_jwt_provider,
            )
            # First moment this pod CAN discharge an upload it still owes: a
            # cell minted by a process that died mid-upload is durable in the
            # local CAS with its record still `pending`. Off the critical path,
            # and a no-op on the pods that owe nothing.
            try:
                fleet_cells.resume_owed_publishes(
                    self.executor._cell_publisher())
            except Exception:  # noqa: BLE001 — never blocks the handshake
                logger.warning(
                    "owed cell uploads could not be re-attempted", exc_info=True)
        # The desired state advertises (release_id, config_gen). Observe it
        # (memory + snapshot file rewrite); parameter VALUES ride RunJob stamps
        # (class 1), bindings ride the desired-residency reconcile below
        # (class 2), envs are boot-only (class 3 — the hub drain-rolls; the
        # worker never mutates its own env).
        push = extract_config_push(ack)
        if push is not None:
            gen, release_id = push
            self.intent_registry.receive_config_generation(gen)
            config_intent = self.intent_registry.ensure_intent(pb.DESIRED_INTENT_KIND_CONFIG_APPLY)
            if config_intent:
                self.intent_registry.transition(
                    config_intent,
                    pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    pb.LIFECYCLE_INTENT_STAGE_CONFIG_MATERIALIZING,
                )
            try:
                if (
                    desired_command is not None
                    and int(desired_command.config_generation) == gen
                    and bytes(desired_command.parameter_snapshot)
                ):
                    self.executor.runtime_config.apply_parameter_snapshot(
                        bytes(desired_command.parameter_snapshot),
                        gen,
                        release_id=(
                            str(desired_command.release_id or "").strip()
                            or release_id
                            or self.release_id
                        ),
                    )
                else:
                    self.executor.runtime_config.observe(
                        gen,
                        release_id=release_id or self.release_id,
                    )
            except ConfigSnapshotWriteError as exc:
                self.intent_registry.config_snapshot_failed(str(exc))
                self._enter_error_phase("config snapshot write failed", exc)
                await self._send_lifecycle_snapshot(hello_ack=True, force=True)
                self._last_delta = None
                await self.maybe_send_state_delta(hello_ack=True)
                return
        # The hub's precision-ladder picks for THIS card. Full-replace: refs
        # absent from the map revert to declared.
        resolutions = {
            r.ref: (r.resolved_ref, r.cast, r.lane, bool(r.lane_pinned))
            for r in ack.resolutions
        }
        self.executor.apply_model_resolutions(resolutions)
        # The reconcile's active-work context compares against the
        # resolutions actually applied to the executor.
        self._model_resolutions = resolutions
        desired = pb.DesiredResidency()
        desired.CopyFrom(ack.desired_residency)
        generation = int(desired.generation)
        if generation < self._observed_residency_generation:
            logger.info(
                "ignoring stale desired residency generation %d (observed %d)",
                generation,
                self._observed_residency_generation,
            )
        else:
            self._observed_residency_generation = generation
            # A superseded declared spelling must not be GC-protected either:
            # an unloaded base outranking genuinely cold refs in the preserve
            # set is the same waste one eviction later.
            _superseded = self._superseded_disk_refs(desired)
            self.executor.store.keep = list(dict.fromkeys(
                ref for ref in desired.disk_refs if ref and ref not in _superseded))
            self.executor.store.replace_desired_snapshots(
                _desired_snapshots(desired),
                generation=generation,
            )
            self._replace_residency_reconcile(desired, model_key=_semantic_model_key(ack, desired))
            # The SAME desired set feeds the background stager, which — unlike
            # the reconcile above — is not tenant-idle-gated and not cancelled
            # by run_job: it stages the NEXT instance while jobs compute
            # (fence-conflicting refs are left to the idle-gated pass).
            self.executor.preloader.update_desired(
                list(desired.hot), _desired_snapshots(desired), generation)
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
        self,
        desired: "pb.DesiredResidency",
        *,
        model_key: Any = None,
    ) -> None:
        # An ack whose semantic model set is unchanged must not kill an
        # in-flight reconcile — a running self_mint_compile needs its full
        # window.
        task = getattr(self, "_residency_task", None)
        running = task is not None and not task.done()
        if (
            model_key is not None
            and running
            and getattr(self, "_residency_model_key", None) == model_key
        ):
            self._desired_residency = desired
            # Not cancelling is not the same as doing nothing: the ack already
            # opened a republish epoch (replace_desired_snapshots) and only a
            # reconcile pass reads it, so returning silently would drop the
            # hub's redrive with no later event to heal it. Signal the
            # level-triggered loop instead — it finishes the item it is on and
            # then re-converges, re-announcing the held identities under the
            # new epoch. The re-pass is cheap and cannot spam: satisfied refs
            # short-circuit, and the epoch guard emits at most one re-report
            # per ref per applied ack.
            self._signal_residency_restart()
            wanted = {instance.function_name for instance in desired.hot if instance.function_name}
            if wanted.issubset(set(self.executor.available_functions())):
                self.intent_registry.bindings_applied(int(desired.config_generation))
            return
        self._residency_model_key = model_key
        self._desired_residency = desired
        if running and self._active_work_still_desired(desired):
            # The set changed, but the model this reconcile is loading RIGHT
            # NOW is still wanted with the same identity and resolution.
            # Cancelling would discard minutes of load over a benign update;
            # signal the level-triggered loop to re-converge against the new
            # set once the current item completes.
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

    def _superseded_disk_refs(self, desired: "pb.DesiredResidency") -> set:
        """Declared disk_refs the hub's OWN resolutions have already replaced,
        and whose replacement is in the same list.

        The hub addresses workers in resolved ref space, but a moment with no
        pick — a transient flavor resolve, a refusal, an unminted-cell gate —
        seeds the BARE spelling into desired.disk_refs, and no hub path removes
        it once the pick returns. Materializing both costs a full extra copy of
        weights that are never loaded.

        Skipping is safe exactly when the resolved twin is ALSO listed: the hub
        is already asking for the artifact this ref resolves to, so the
        declared spelling can only add bytes. When the resolved twin is NOT
        listed the ref is materialized normally — a lane override that keeps
        the canonical spelling is never skipped."""
        resolutions = getattr(self, "_model_resolutions", {}) or {}
        if not resolutions:
            return set()
        listed = {ref for ref in desired.disk_refs if ref}
        superseded = set()
        for ref in listed:
            pick = resolutions.get(ref)
            resolved = (pick[0] if pick else "") or ""
            if resolved and resolved != ref and resolved in listed:
                superseded.add(ref)
        for ref in sorted(superseded):
            resolved = resolutions[ref][0]
            if self._superseded_reported.get(ref) == resolved:
                continue
            self._superseded_reported[ref] = resolved
            logger.info(
                "skipping superseded disk_ref %s (resolved to %s, also desired)",
                ref, resolved,
            )
            activity_mod.emit_event(
                "residency_ref_superseded",
                f"declared disk_ref {ref} is superseded by its own resolution "
                f"{resolved}, which is desired in the same generation — not "
                f"materializing the declared spelling (th#1330)",
                phase="skipped",
            )
        for ref in list(self._superseded_reported):
            if ref not in superseded:
                del self._superseded_reported[ref]
        return superseded

    def _work_context(
        self,
        refs: Any,
        desired: "pb.DesiredResidency",
    ) -> tuple:
        """Identity facts an in-flight load depends on, per involved ref:
        snapshot content (URL-free) + the applied precision resolution."""
        resolutions = getattr(self, "_model_resolutions", {})
        return tuple(
            (
                ref,
                _snapshot_content_key(_desired_snapshots(desired).get(ref)),
                resolutions.get(ref),
            )
            for ref in sorted({r for r in refs if r})
        )

    def _active_work_still_desired(
        self,
        desired: "pb.DesiredResidency",
    ) -> bool:
        """Whether the reconcile loop's CURRENT work item survives ``desired``
        unchanged. Only cancel an in-flight load when the model it is FOR left
        the set (or changed identity or resolution), never on unrelated
        churn."""
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
                return ctx == self._work_context(_instance_refs(instance), desired)
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
        claim. Level-triggered: each pass reads the freshest set; a
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
        except UnreportedIntentWait as exc:
            self._enter_error_phase("residency reconcile", exc)
            self._last_delta = None
            await self.maybe_send_state_delta(force=True)
        finally:
            self._reconcile_active = None

    async def _reconcile_pass(
        self,
        desired: "pb.DesiredResidency",
        restart: asyncio.Event,
    ) -> None:
        """One convergence pass over ``desired`` in declared order."""
        snapshots = _desired_snapshots(desired)
        # Never materialize a declared spelling the hub's own
        # resolutions have replaced with another ref in this same list.
        superseded = self._superseded_disk_refs(desired)
        for ref in desired.disk_refs:
            if not ref or ref in superseded:
                continue
            if restart.is_set():
                return
            intent_id = self.intent_registry.ensure_intent(
                pb.DESIRED_INTENT_KIND_MATERIALIZE,
                ref=ref,
            )
            # Do NOT remove this tenant-idle gate to cure staging starvation.
            # The pass's first act is to fence a stale resident identity for
            # this ref, and a tenant job may legitimately be re-materializing
            # OLDER bytes concurrently; without the gate the fence races the
            # job's load and the worker reports the stale identity until the
            # next HelloAck. Guarded by tests/test_p2_residency_reconcile.py::
            # test_mutable_tag_move_fences_events_by_digest_and_generation,
            # which does not converge within 10s without it — and is NOT
            # fixable by re-verifying after the job, because handle_run_job
            # returns before the job's load completes. The real fix is
            # transfers becoming their own tasks with their own lifecycle, so
            # staging never sits inside this serialized pass.
            await self.intent_registry.reported_await(
                intent_id,
                self.executor.wait_idle(),
                operation=f"reconcile idle for {ref}",
                status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_TENANT_IDLE,
                reason=pb.LIFECYCLE_WAIT_REASON_TENANT_WORK,
            )
            if self.draining:
                return
            self._reconcile_active = ("disk", ref, self._work_context((ref,), desired))
            failure_stage = pb.LIFECYCLE_INTENT_STAGE_VERIFYING
            try:
                snapshot = snapshots.get(ref)
                await self.intent_registry.reported_await(
                    intent_id,
                    self.executor.revalidate_snapshot_identity(ref, snapshot),
                    operation=f"snapshot validation for {ref}",
                    status=pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                    stage=pb.LIFECYCLE_INTENT_STAGE_VERIFYING,
                )
                failure_stage = pb.LIFECYCLE_INTENT_STAGE_FETCHING
                with self.executor.store.materialize_intent(intent_id):
                    await self.intent_registry.reported_await(
                        intent_id,
                        self.executor.store.ensure_local(ref, snapshot),
                        operation=f"snapshot materialization for {ref}",
                        status=pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                        stage=pb.LIFECYCLE_INTENT_STAGE_FETCHING,
                    )
                if intent_id:
                    self.intent_registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                        pb.LIFECYCLE_INTENT_STAGE_ON_DISK,
                        actual_digest=(
                            snapshot.digest.encode()
                            if snapshot is not None and snapshot.digest
                            else b""
                        ),
                    )
            except (asyncio.CancelledError, UnreportedIntentWait):
                raise
            except Exception as exc:
                if intent_id:
                    self.intent_registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_FAILED,
                        failure_stage,
                        error_code=(
                            pb.LIFECYCLE_ERROR_CODE_SNAPSHOT_IDENTITY_MISSING
                            if snapshots.get(ref) is None
                            else pb.LIFECYCLE_ERROR_CODE_UNSPECIFIED
                        ),
                        detail=str(exc)[:512],
                    )
                logger.warning("desired disk residency failed for %s: %s", ref, exc)
            finally:
                self._reconcile_active = None

        bindings_converged = True
        for instance in desired.hot:
            if restart.is_set():
                return
            intent_id = self.intent_registry.ensure_intent(
                pb.DESIRED_INTENT_KIND_FUNCTION_READY,
                function_name=instance.function_name,
            )
            await self.intent_registry.reported_await(
                intent_id,
                self.executor.wait_idle(),
                operation=f"reconcile idle for {instance.function_name}",
                status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                stage=pb.LIFECYCLE_INTENT_STAGE_WAIT_TENANT_IDLE,
                reason=pb.LIFECYCLE_WAIT_REASON_TENANT_WORK,
            )
            if self.draining:
                return
            self._reconcile_active = (
                "hot",
                instance.SerializeToString(deterministic=True),
                self._work_context(_instance_refs(instance), desired),
            )
            try:
                if intent_id:
                    self.intent_registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                        pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                    )
                await self.executor.ensure_desired_instance(instance, snapshots)
                if intent_id:
                    self.intent_registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                        pb.LIFECYCLE_INTENT_STAGE_READY,
                    )
            except (asyncio.CancelledError, UnreportedIntentWait):
                raise
            except Exception as exc:
                bindings_converged = False
                if intent_id:
                    self.intent_registry.transition(
                        intent_id,
                        pb.LIFECYCLE_INTENT_STATUS_FAILED,
                        pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
                        detail=str(exc)[:512],
                    )
                logger.warning(
                    "desired hot residency failed for %s: %s",
                    instance.function_name,
                    exc,
                )
            finally:
                self._reconcile_active = None
        if bindings_converged and not restart.is_set() and self._desired_residency is desired:
            self.intent_registry.bindings_applied(int(desired.config_generation))

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        which = msg.WhichOneof("msg")
        if which == "run_job":
            if self.intent_registry.protocol_rejected:
                await self.executor._send_result(
                    msg.run_job.request_id,
                    msg.run_job.attempt,
                    pb.JOB_STATUS_RETRYABLE,
                    safe_message="worker rejected mandatory desired state",
                )
                return
            # A tenant request preempts unrelated background transfer/setup.
            # Re-running desired state after idle is cheap: local refs and
            # ready instances short-circuit through the existing dedupe paths.
            # This cancel is LOAD-BEARING for the mutable-tag identity fence —
            # see the tenant-idle gate comment in _reconcile_pass. Making
            # downloads survive preemption needs transfers to be first-class
            # tasks.
            self._cancel_residency_reconcile()
            try:
                await self.executor.handle_run_job(msg.run_job)
            finally:
                self._resume_residency_reconcile()
        elif which == "cancel_job":
            self.executor.handle_cancel(msg.cancel_job)
        elif which == "model_op":
            # The sole ModelOp kind (ADOPT_COMPILE_CACHE) is retired. Ignore
            # rather than refuse: a hub still sending one must not lose its
            # stream over a message we simply no longer act on.
            logger.warning(
                "ignoring retired ModelOp (pgw#1032): hot compile-cache "
                "adoption is gone; a cell arrives only through §4.27 "
                "boot-adopt — the hub never pushes one")
        elif which == "serve_posture":
            # The operator's eager-only command, applied SYNCHRONOUSLY and
            # unconditionally: it touches one process-global order and never
            # blocks, so no worker is ever "too busy" to stop using a broken
            # cell, and in-flight requests pick it up at their next compiled
            # call. Idempotent, so a reconnect replay is free.
            serve_posture.apply_command(
                bool(msg.serve_posture.eager_only),
                actor=str(msg.serve_posture.actor or ""),
                reason=str(msg.serve_posture.reason or ""),
            )
        elif which == "drain":
            # The hub asked, explicitly, for this budget on this drain.
            self.start_drain(
                int(msg.drain.deadline_ms or 0), work_deadline=True)
        elif which is None:
            raise FatalTransportError("scheduler sent an unknown mandatory command")
        else:
            raise FatalTransportError(f"scheduler sent unsupported command {which!r}")

    async def on_disconnect(self) -> None:
        self._last_delta = None
        self._last_lifecycle_snapshot = None
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

    def _config_snapshot_applied(self, generation: int) -> None:
        self.intent_registry.config_snapshot_applied(generation)

    def _config_snapshot_failed(self, detail: str) -> None:
        self.intent_registry.config_snapshot_failed(detail)
        self._enter_error_phase(f"config snapshot failed: {detail}")
        self.state_changed()

    def _enter_error_phase(
        self, reason: str, exc: Optional[BaseException] = None,
    ) -> None:
        """Every WORKER_PHASE_ERROR names its cause on a durable channel: the
        hub persists only the bare phase flip, and a worker's lifecycle
        snapshot can be dropped by hub shadow validation. Dials the
        worker-fatal carrier once per process, best-effort; the hub stores it
        as a queryable pod_events row."""

        if exc is not None:
            detail = worker_fatal.build_fatal_detail(f"phase_error: {reason}", exc, exit_code=0)
        else:
            detail = f"phase=phase_error: {reason}"
        logger.error("entering WORKER_PHASE_ERROR: %s", detail)
        self.phase = pb.WORKER_PHASE_ERROR
        # getattr: stubbed Lifecycles skip __init__.
        if getattr(self, "_error_phase_reported", False):
            return
        self._error_phase_reported = True
        settings = getattr(self, "_settings", None)

        async def _dial() -> None:
            await worker_fatal.report_worker_error_async(settings, detail)

        try:
            asyncio.get_running_loop().create_task(_dial(), name="phase-error-report")
        except RuntimeError:
            pass

    async def _send_state_message(
        self,
        message: pb.WorkerMessage,
        *,
        hello_ack: bool = False,
    ) -> None:
        if self.transport is None:
            return
        prepend = getattr(self.transport, "prepend_reconnect", None)
        if hello_ack and prepend is not None:
            await prepend([message])
            return
        await self.transport.send(message)

    async def maybe_send_state_delta(
        self,
        *,
        hello_ack: bool = False,
        force: bool = False,
    ) -> None:
        if self.transport is None or not self.transport.connected:
            return
        self.intent_registry.refresh_projection(
            self.executor,
            self._desired_residency,
            self._model_resolutions,
        )
        self._kick_disk_usage_refresh_if_stale()
        delta = self._state_delta()
        raw = delta.SerializeToString(deterministic=True)
        # force: the heartbeat tick re-sends an unchanged snapshot — receipt IS
        # the beat the hub timestamps.
        if force or raw != self._last_delta:
            self._last_delta = raw
            await self._send_state_message(
                pb.WorkerMessage(state_delta=delta),
                hello_ack=hello_ack,
            )
            # THE boot close, and its only owner. "Servable" means the hub may
            # dispatch to this worker, and the hub dispatches to
            # `available_functions` — so the fact is "a StateDelta advertising
            # at least one function has gone out on a connected stream", which
            # is observable here and nowhere else. Marking it from
            # `Lifecycle.startup()` is wrong: that runs concurrently with the
            # transport and closes the boot before `hello`.
            if delta.available_functions and not self.draining:
                self._mark_servable()
        await self._send_lifecycle_snapshot(
            hello_ack=hello_ack,
            force=force,
        )
        # Deduped internally; must run even on an unchanged delta — a runtime
        # ladder demotion changes the ServePlan, not the delta bytes.
        await self._emit_unavailable(hello_ack=hello_ack)
        await self._emit_degraded(hello_ack=hello_ack)

    async def _send_lifecycle_snapshot(
        self,
        *,
        hello_ack: bool = False,
        force: bool = False,
    ) -> None:
        snapshot = self.intent_registry.snapshot()
        raw = snapshot.SerializeToString(deterministic=True)
        if force or raw != self._last_lifecycle_snapshot:
            self._last_lifecycle_snapshot = raw
            await self._send_state_message(
                pb.WorkerMessage(lifecycle_snapshot=snapshot),
                hello_ack=hello_ack,
            )

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
                pb.WorkerMessage(
                    fn_unavailable=pb.FnUnavailable(
                        function_name=name,
                        reason=reason,
                        detail=detail,
                        axes=axes,
                    )
                ),
                hello_ack=hello_ack,
            )

    async def _emit_degraded(self, *, hello_ack: bool = False) -> None:
        """Per served function running degraded, tell the orchestrator
        STRUCTURALLY (FnDegraded, not just a log line) so placement learns
        "this release degraded on this card". Rides the orchestrator transport,
        so cozy-local (no orchestrator) never sends it — there the
        honest-guidance advisory on the terminal is the surface."""
        if self.transport is None:
            return
        for name, plan in sorted(self.executor.serve_plans.items()):
            if not plan.degraded or name in self.executor.unavailable:
                continue
            ran = plan.ran or plan.run_mode
            # `ran` stays inside the hub's exact-match RunMode vocabulary, so a
            # deeper demotion re-reports via the WARNING text — dedupe on both
            # or an escalation inside one run mode goes silent.
            emitted_key = f"{ran}\x1f{plan.warning}"
            if self._emitted_degraded.get(name) == emitted_key:
                continue
            self._emitted_degraded[name] = emitted_key
            await self._send_state_message(
                pb.WorkerMessage(
                    fn_degraded=pb.FnDegraded(
                        function_name=name,
                        wanted=plan.wanted,
                        ran=ran,
                        reason=plan.warning,
                        est_latency_multiplier=float(plan.est_latency_multiplier),
                    )
                ),
                hello_ack=hello_ack,
            )

    async def set_phase(self, phase: "pb.WorkerPhase") -> None:
        if self.intent_registry.protocol_rejected and phase != pb.WORKER_PHASE_ERROR:
            return
        if phase == self.phase:
            return
        self.phase = phase
        await self.maybe_send_state_delta()

    # ---- startup ---------------------------------------------------------------

    async def startup(self) -> None:
        """Gate functions, prefetch worker-fetchable models with retry/backoff,
        set up endpoints, advance phases. Never raises: failures gate
        individual functions, not the process."""
        # the beat starts BEFORE any boot work so a hang anywhere in setup is
        # still covered — the task shares this event loop, so it beats iff the
        # loop is servicing tasks.
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
        # Disk truth first: after a restart the CAS dir is full while Residency
        # starts empty — rescan so Hello.models (and disk GC) see reality.
        self.executor.store.rescan_disk()
        self.executor.gate_functions(self.hardware)

        prefetch_refs: List[WireRef] = []
        for spec in self.executor.specs.values():
            if spec.name in self.executor.unavailable:
                continue
            for slot, binding in spec.models.items():
                if slot in spec.slots:
                    # A declared Slot's default_checkpoint is a SEED for the hub
                    # mapping, not a load instruction: the hub resolves the
                    # slot and drives DOWNLOADs itself. A hub-connected worker
                    # never self-fetches the raw upstream default.
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
                    # The weights_fetch boot span lives in
                    # `ModelStore._materialize_local`, NOT here: this call site
                    # only covers hf/civitai prefetch refs, while the tensorhub
                    # refs that own most of a real cold boot arrive later via
                    # DesiredResidency/RunJob. The materializer is also the only
                    # layer that can stamp bytes AND source (network vs warm
                    # volume vs local CAS).
                    await self.executor.store.ensure_local(ref)
                except Exception as exc:
                    # `_materialize_local` has already retried the transient
                    # classes with backoff, so reaching here means the model is
                    # not coming: the functions that need it go unavailable NOW
                    # rather than resurfacing as a per-job load failure on paid
                    # GPU.
                    gated = self.executor.mark_ref_unmaterializable(
                        ref, f"{type(exc).__name__}: {exc}")
                    logger.error(
                        "startup prefetch of %s failed terminally: %s — gating %s "
                        "(pgw#655: a worker never advertises a function whose "
                        "model is absent)",
                        ref, exc, ", ".join(gated) or "no function",
                    )

        await self.set_phase(pb.WORKER_PHASE_LOADING_PIPELINES)
        awaiting_hub: Dict[str, List[WireRef]] = {}
        dynamic: List[str] = []
        for spec in list(self.executor.specs.values()):
            if spec.name in self.executor.unavailable:
                continue
            if spec.slots or spec.compile is not None:
                # Slot picks and compile cells are set up ONLY on hub delivery —
                # Hot DesiredInstance or RunJob, both of which rebind through
                # _effective_spec with the hub-stamped refs. Running
                # ensure_setup on the class-table spec here would materialize
                # the image-baked code default over the hub-stamped release
                # binding; that default is the hub-less bootstrap fallback only.
                dynamic.append(spec.name)
                continue
            missing = sorted(
                {
                    wire_ref(b)
                    for b in spec.models.values()
                    if b.source == "tensorhub"
                    and self.executor.store.local_path(wire_ref(b)) is None
                }
            )
            if missing:
                # Waits for DesiredResidency / RunJob snapshots (§7). This wait
                # is unbounded and hub-driven: a release without resolved
                # desired bindings leaves the function in loading_functions
                # forever, so say so loudly instead of dropping it in silence.
                awaiting_hub[spec.name] = missing
                continue
            try:
                # The `pipeline_load` span belongs to `Executor.ensure_setup`,
                # not to this call site: a release declaring `Compile` never
                # takes this path (those specs route to `dynamic` above).
                # Instrument the layer all callers pass through.
                await self.executor.ensure_setup(spec)
            except Exception as exc:
                logger.error("startup setup of %s failed: %s", spec.name, exc)
        if dynamic:
            logger.info(
                "hub-resolved functions (dynamic slots / compile cells) set up "
                "on delivery, not at boot (pgw#532, gw#584): %s",
                ", ".join(sorted(dynamic)),
            )
        if awaiting_hub:
            logger.warning(
                "functions waiting on hub-supplied snapshots (DesiredResidency, "
                "contract §7) and NOT yet servable: %s — if these never arrive, "
                "check that the release has resolved desired bindings for these refs",
                "; ".join(
                    f"{fn} <- {', '.join(refs)}" for fn, refs in sorted(awaiting_hub.items())
                ),
            )
            # The hub's desired-disk plan delivers those snapshots seconds
            # later, but nothing else re-runs setup — and the hub dispatches
            # only to advertised functions, so each side would wait on the
            # other forever. Finish boot setup the moment the refs land.
            self._boot_setup_watch = asyncio.create_task(
                self._setup_awaiting_functions(awaiting_hub), name="boot-setup-watch"
            )

        # FIRST-REQUEST-SERVABLE is NOT marked here, on any path: `startup()`
        # runs concurrently with (and usually before) the transport's hello.
        # The milestone has exactly one owner, `maybe_send_state_delta`, which
        # marks it on the fact itself — a StateDelta advertising a function
        # went out. If no function is ever advertised the boot has no closing
        # milestone, and that is the finding.
        await self.set_phase(pb.WORKER_PHASE_READY)

    def _mark_servable(self) -> None:
        """Close the boot: mark first-request-servable from process start.

        `mark_once` keeps it once-per-process — it is called on every StateDelta
        send. The reconciliation detail is filled by the recorder, which is the
        only thing that knows the instant the row is actually released."""
        if not boot_mod.mark_once(
            boot_mod.PHASE_FIRST_REQUEST_SERVABLE,
            since_process_start=True,
        ):
            return
        # The pod reports its OWN decomposition once, at the instant the boot
        # closes, and says so when it cannot explain itself. `SHAPE_ENTRYPOINT`
        # is the MINIMUM every pod boot produces (the cell phases depend on
        # what the boot did), so `missing` here is always an instrument defect,
        # never a boot-shape difference.
        try:
            verdict = boot_mod.completeness(boot_mod.SHAPE_ENTRYPOINT)
            logger.info(
                "[boot] servable in %dms — decomposition:\n%s",
                verdict.total_ms, boot_mod.render_phase_table())
            if not verdict.complete:
                logger.warning(
                    "[boot] this boot cannot fully explain itself: %s",
                    verdict.explain())
        except Exception:  # telemetry must never break the boot it measures
            logger.debug("boot decomposition report failed", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        """Force-send the full StateDelta (which carries the measured-disk
        report) every beat, in EVERY state including drain — the beat only ends
        when the process does."""
        while not self.drained.is_set():
            await asyncio.sleep(HEARTBEAT_INTERVAL_MS / 1000.0)
            await self.maybe_send_state_delta(force=True)
            # Progress counters piggyback on the same beat: one
            # counter-carrying ActivityUpdate per tick while an activity is
            # open, plus the typed self-diagnosis on a stalled registry.
            activity_mod.on_beat()

    async def _setup_awaiting_functions(
        self, awaiting: Dict[str, List[WireRef]],
    ) -> None:
        """Complete boot setup for functions whose tensorhub snapshots arrive
        via hub delivery after the startup scan, then push a
        StateDelta so ``available_functions`` advertises them."""
        pending = {fn: list(refs) for fn, refs in awaiting.items()}
        while pending and not self.draining:
            for fn in sorted(pending):
                left = [r for r in pending[fn] if self.executor.store.local_path(r) is None]
                if left:
                    pending[fn] = left
                    continue
                del pending[fn]
                spec = self.executor.specs.get(fn)
                if spec is None or fn in self.executor.unavailable:
                    continue
                try:
                    # span owned by `Executor.ensure_setup`.
                    await self.executor.ensure_setup(spec)
                    logger.info(
                        "boot setup of %s completed after hub snapshot delivery (gw#591)", fn
                    )
                except Exception as exc:
                    logger.error("startup setup of %s failed: %s", fn, exc)
                await self.maybe_send_state_delta()
            if pending:
                await asyncio.sleep(_BOOT_SETUP_WATCH_INTERVAL_S)
        # No servable mark here: the `maybe_send_state_delta` above advertises
        # these functions and owns the milestone.

    # ---- drain -------------------------------------------------------------------

    def start_drain(self, deadline_ms: int, *, work_deadline: bool = False) -> None:
        if self.draining:
            return
        self._begin_drain(deadline_ms, work_deadline=work_deadline)
        self._drain_task = asyncio.create_task(self._finish_drain(), name="drain")

    async def drain(self, deadline_ms: int = 0, *, work_deadline: bool = True) -> None:
        """stop admitting -> finish in-flight -> ship buffered results ->
        close the stream -> signal exit 0. Zero waits without a cutoff."""
        if self.draining:
            return
        self._begin_drain(deadline_ms, work_deadline=work_deadline)
        await self._finish_drain()

    def _begin_drain(self, deadline_ms: int, *, work_deadline: bool = False) -> None:
        """Synchronously stop admission and anchor the deadline(s) at receipt.

        ``work_deadline`` says whether the caller's budget may bound TENANT
        WORK as well as the shutdown around it — true only for a `Drain` that
        arrived on the wire carrying one.
        """
        self.draining = True
        self.executor.draining = True
        self._cancel_residency_reconcile()
        self.executor.preloader.stop()  # no staging into a drain
        # The heartbeat deliberately keeps beating through drain — a worker
        # hung mid-drain must still be detectable as dead. It is cancelled
        # after the stream closes in _finish_drain.
        logger.info("drain started (deadline_ms=%d)", deadline_ms)
        deadline_s = (deadline_ms / 1000.0) if deadline_ms > 0 else None
        loop = asyncio.get_running_loop()
        self._drain_deadline_at = loop.time() + deadline_s if deadline_s is not None else None
        # Two deadlines, because they bound two different things and
        # collapsing them is the defect. `_drain_deadline_at` bounds the
        # SHUTDOWN — the result flush and the stop time reported to the hub —
        # and every drain has one. `_drain_work_deadline_at` bounds the TENANT
        # WORK, and only a caller that explicitly asked for one gets it: a
        # `Drain` carrying `deadline_ms` is an operator budget on a command,
        # whereas the SIGTERM handler's fleet default would abandon long mints.
        self._drain_work_deadline_at = (
            self._drain_deadline_at if work_deadline else None)
        self.intent_registry.set_drain(
            pb.DRAIN_LIFECYCLE_STATUS_DRAINING,
            deadline_at_unix_ms=(int(time.time() * 1000) + deadline_ms if deadline_ms > 0 else 0),
        )

    async def _await_tenant_idle(self) -> bool:
        """True when every in-flight job finished; False when the work STOPPED
        ADVANCING (or an EXPLICIT operator budget ran out) and the drain gave
        up on the remainder.

        A drain is a COMMAND, so a deadline on it is legitimate — but the
        deadline bounds acknowledgement and shutdown, never the tenant work
        underneath. Admission already stopped in ``_begin_drain``; what ends
        this wait is the work ceasing to advance, which
        ``progress.self_diagnosis()`` answers.

        Only a ``Drain`` that arrives on the wire carrying ``deadline_ms`` is
        an explicit operator budget; the worker never INVENTS one, so the
        SIGTERM handler's `_SIGNAL_DRAIN_DEADLINE_MS` is not a work budget and
        a pod stopping on a signal does not abandon a running mint. Even under
        an explicit budget the progress test applies and fires FIRST: a wedged
        job is given up on when it wedges, not when the clock runs out.

        Two properties worth stating because they are deliberate:

        * **No counters open at all is NOT a stall.** It is the same "keep
          waiting" the in-call stall loop takes (``executor``'s
          ``_STALL_POLL_S`` arm: *"advancing (or evidence-free): keep
          waiting"*), and the outer authority — the hub's terminate, or the
          container runtime's SIGKILL — is what bounds the pathological case.
          The worker's job is to not throw away work that was about to finish.
        * ``self_diagnosis()`` is registry-wide, so a background mint's
          advancing counter can currently defer this verdict for a wedged
          tenant job.
        """
        loop = asyncio.get_running_loop()
        budget_at = self._drain_work_deadline_at
        idle = asyncio.ensure_future(self.executor.wait_idle())
        try:
            while True:
                poll = _DRAIN_PROGRESS_POLL_S
                if budget_at is not None:
                    remaining = budget_at - loop.time()
                    if remaining <= 0:
                        logger.warning(
                            "drain: the caller's explicit deadline_ms budget "
                            "ran out with work still in flight")
                        return False
                    poll = min(poll, remaining)
                try:
                    await asyncio.wait_for(asyncio.shield(idle), poll)
                    return True
                except asyncio.TimeoutError:
                    stalled = progress_mod.self_diagnosis()
                    if stalled is None:
                        continue
                    logger.warning(
                        "drain: counter %r (%s) has not advanced for %.0fs "
                        "(window %.0fs) — the remaining work is not making "
                        "progress", stalled.name, stalled.unit,
                        stalled.age_s, stalled.window_s)
                    return False
        finally:
            if not idle.done():
                idle.cancel()

    async def _finish_drain(self) -> None:
        deadline_at = self._drain_deadline_at
        loop = asyncio.get_running_loop()
        await self.maybe_send_state_delta()
        drain_intent = self.intent_registry.intent_id(pb.DESIRED_INTENT_KIND_DRAIN)
        try:
            finished = await self.intent_registry.reported_await(
                drain_intent,
                self._await_tenant_idle(),
                operation="drain tenant-idle wait",
                status=pb.LIFECYCLE_INTENT_STATUS_WAITING,
                stage=pb.LIFECYCLE_INTENT_STAGE_DRAINING,
                reason=pb.LIFECYCLE_WAIT_REASON_TENANT_WORK,
            )
            if not finished:
                logger.warning(
                    "in-flight work stopped advancing during drain; aborting "
                    "the remainder as RETRYABLE")
                await self.intent_registry.guard_await(
                    drain_intent,
                    self.executor.abort_all(safe_message="worker draining"),
                    operation="drain job abort",
                )

            self.intent_registry.set_drain(
                pb.DRAIN_LIFECYCLE_STATUS_FINALIZING,
            )
            await self.maybe_send_state_delta()
            await self.intent_registry.guard_await(
                drain_intent,
                shutdown_instances(self.executor.teardown_seam),
                operation="drain instance shutdown",
            )
            if self.transport is not None:
                self.intent_registry.set_drain(
                    pb.DRAIN_LIFECYCLE_STATUS_FLUSHING,
                )
                await self.maybe_send_state_delta()
                # Never `max(0.0, deadline_at - loop.time())`: that is provably
                # zero on every path reaching here after the tenant wait
                # expired — precisely when `abort_all` has just produced the
                # typed RETRYABLE results whose purpose is to tell the hub to
                # re-dispatch. The deadline governs waiting for TENANT WORK;
                # shipping results we already hold is not that, so the floor is
                # the channel's own peer-dead threshold.
                flush_timeout = (
                    None if deadline_at is None
                    else max(KEEPALIVE_TIMEOUT_S, deadline_at - loop.time())
                )
                await self.intent_registry.guard_await(
                    drain_intent,
                    self.transport.close_after_flush(timeout=flush_timeout),
                    operation="drain transport flush",
                )
            self.intent_registry.set_drain(
                pb.DRAIN_LIFECYCLE_STATUS_DRAINED,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.intent_registry.set_drain(
                pb.DRAIN_LIFECYCLE_STATUS_FAILED,
                detail=str(exc)[:512],
                error_code=pb.LIFECYCLE_ERROR_CODE_DRAIN_FAILED,
            )
            self._enter_error_phase("drain failed", exc)
            try:
                await self.maybe_send_state_delta()
            except Exception:
                logger.exception("failed to report drain failure")
            raise
        self.drained.set()
        # getattr: stubbed Lifecycles skip __init__.
        beat_task = getattr(self, "_heartbeat_task", None)
        if beat_task is not None:
            beat_task.cancel()
            self._heartbeat_task = None
        logger.info("drain complete")
