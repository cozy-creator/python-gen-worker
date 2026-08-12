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
from .config import Settings
from .config.settings import BOOT_CONFIG_GENERATION_ABSENT
from .executor import Executor
from .intent_registry import IntentRegistry, UnreportedIntentWait
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
from .topology import delivered_topology

logger = logging.getLogger(__name__)

_VRAM_QUANTUM_FRACTION = 0.05  # quantize free-VRAM deltas to 5% of total

# pgw#887: how often the drain re-asks the progress registry while it waits
# for in-flight work. A CADENCE, not a verdict — it bounds latency-to-notice
# and can never end anything; the give-up test is `progress.self_diagnosis()`,
# which is silence-keyed and per-phase.
_DRAIN_PROGRESS_POLL_S = 1.0

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
    """Static hardware facts + gate inputs. torch is optional.

    ``gpu_free_mem`` is the input `gate_functions` turns into `unavailable` +
    `serve_plans` for EVERY function, so on a wide pod it must be the free pool
    of the group with the LEAST room, not card 0's (pgw#776/DPA-7). Card 0 is
    always the least free card in practice — it carries group 0's weights plus
    the allocator cache (measured: qwen 96.5 GiB on card 0 vs ~68 on 1-3) — so
    reading it planned offload/fp8-fallback rungs for groups 1..G-1 from a card
    they will never touch, and a genuinely-unfit card 0 refused functions the
    other three could serve. The MIN is the honest single scalar until the fit
    ladder is per-rank: it never promises a group room it does not have.
    """
    info: Dict[str, Any] = {
        "gpu_count": 0,
        "gpu_total_mem": 0,
        "gpu_free_mem": 0,
        "gpu_name": "",
        "gpu_sm": "",
        "torch_version": "",
        "installed_libs": [],
        "driver_version": "",
    }
    # pgw#1129/th#1798: the HOST driver, read from NVML rather than torch —
    # the whole point of this fact is that it stays readable when the CUDA
    # runtime is not. A cu130 build on a 570.x host imports fine, reports every
    # version string correctly, and dies on its first allocation, so the driver
    # is the only field in this dict that can tell the hub which hosts its
    # placement filter actually landed on.
    try:
        from .hardware_report import _nvidia_smi_driver_and_gpu

        driver, gpu = _nvidia_smi_driver_and_gpu()
        info["driver_version"] = driver
        if gpu and not info["gpu_name"]:
            info["gpu_name"] = gpu
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info["gpu_total_mem"] = int(total_mem)
            info["gpu_free_mem"] = int(free_mem)
            worst = _worst_group_free_vram_bytes()
            if worst and worst != int(free_mem):
                logger.info(
                    "fit inputs: gating on the least-free group (%d bytes) "
                    "instead of card 0 (%d bytes) — pgw#776",
                    int(worst), int(free_mem))
                info["gpu_free_mem"] = int(worst)
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


_MULTI_GPU_NO_TOPOLOGY_WARNED = False


def _worst_group_free_vram_bytes() -> int:
    """The free pool of the least-roomy execution group, or 0 if unknowable.

    Reads the DELIVERED topology (pgw#776/DPA-5): on a pod this worker itself
    demoted for a non-NVLink fabric, ``from_env`` describes a packing that is
    not being served, so reporting from it makes the reported and served
    topologies disagree.
    """

    groups = delivered_topology().all_groups()
    return int(min((g.free_vram_bytes() for g in groups), default=0))


def free_vram_bytes() -> int:
    """Free VRAM the hub may admit against (``StateDelta.free_vram_bytes``).

    This used to be the sum across every CUDA device, which is the reporting
    half of the bug pgw#648 fixed in the accounting half: VRAM is not fungible
    between cards, so a 2x24GB pod that reports 48GB invites the hub to admit
    a 30GB model that fits on neither.

    It then became the MAX across groups, which is honest PER JOB and dishonest
    per POD (pgw#776/DPA-5): the hub admits against a single scalar and can
    admit G concurrent jobs whose sum is G times what any one group has. With
    one scalar on the wire the only safe answer is the MIN across groups — MIN
    within a replicated group too, since every member holds a full copy — so
    every one of the G jobs the hub may admit fits where it lands. At G == 1,
    which is every pod today, MIN and MAX are the same number.

    Multi-GPU host with NO delivered topology (every harness attach and
    `cli serve`): the worker has ONE slot which places on the thread-current
    device, so card 0 is the honest number and the other cards are invisible to
    the hub. That is a real behaviour change from the historical all-device sum
    (the pgw#748 G=1 invariant's one exception), so it is named and logged once
    rather than left to be rediscovered.
    """
    try:
        worst = _worst_group_free_vram_bytes()
        if worst:
            _warn_once_if_gpus_are_invisible()
            return int(worst)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count():
            return int(torch.cuda.mem_get_info(0)[0])
    except Exception:
        pass
    return 0


def _warn_once_if_gpus_are_invisible() -> None:
    global _MULTI_GPU_NO_TOPOLOGY_WARNED
    if _MULTI_GPU_NO_TOPOLOGY_WARNED:
        return
    try:
        import torch

        from .topology import ENV_VAR

        if os.environ.get(ENV_VAR):
            return
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
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
        tuple(sorted((f.path, f.size_bytes, f.digest) for f in snap.files)),
    )


def _semantic_model_key(
    ack: "pb.HelloAck",
    desired: "pb.DesiredResidency",
) -> tuple:
    """gw#614: order-independent identity of the MODEL content of an ack —
    resolutions, disk refs, snapshot content, hot instances. Generation,
    presigned URL bytes, and other non-model fields are excluded so benign
    plan rewrites compare equal."""
    return (
        tuple(sorted((r.ref, r.resolved_ref, r.cast, r.lane) for r in ack.resolutions)),
        tuple(sorted(ref for ref in desired.disk_refs if ref)),
        tuple(
            sorted((ref, _snapshot_content_key(snap)) for ref, snap in desired.snapshots.items())
        ),
        tuple(sorted(inst.SerializeToString(deterministic=True) for inst in desired.hot)),
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
        # pgw#848: the BOOTSTRAP token by intent, not by accident. This runs at
        # construction, before any stream exists, and it reads IDENTITY claims
        # (sub / release_id) which rotation never changes — so the frozen copy is
        # the right and only available answer here. Anything reading a
        # credential to AUTHENTICATE must use `worker_credential.current()`.
        _boot_jwt = (settings.bootstrap_worker_jwt or "").strip()
        if _boot_jwt:
            from .request_context import _decode_unverified_jwt_claims

            claims = _decode_unverified_jwt_claims(_boot_jwt)
        self.worker_id = (
            settings.worker_id or str(claims.get("sub") or "").strip() or f"py-worker-{os.getpid()}"
        )
        # pgw#763 delta 1: in the split's compute child there is no JWT to read
        # claims from — the parent decodes its own and hands the release id
        # down as a plain value. A claim is not a credential.
        self.release_id = (
            str(claims.get("release_id") or "").strip()
            or settings.worker_release_id.strip()
        )
        self.intent_registry = IntentRegistry(
            self.release_id,
            executor.specs.keys(),
            # gw#668: pass the value THROUGH, sentinel and all — a missing
            # WORKER_CONFIG_GENERATION is "no boot-only env was injected"
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
        # mere membership) so a runtime ladder demotion (gw#463) re-emits.
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
        # gw#623: the reconcile loop's currently-loading work item
        # ((kind, identity, context)) + the level-trigger that re-runs a
        # convergence pass instead of cancelling an in-flight load.
        self._reconcile_active: Optional[tuple] = None
        self._residency_restart: Optional[asyncio.Event] = None
        self._model_resolutions: Dict[str, Tuple[Any, ...]] = {}
        # th#1330: declared -> resolved for every disk_ref already reported as
        # superseded, so the typed event fires on transitions, not per ack.
        self._superseded_reported: Dict[str, str] = {}
        self.executor.runtime_config.set_projection_callbacks(
            on_parameter_snapshot=self._config_snapshot_applied,
            on_snapshot_failure=self._config_snapshot_failed,
        )

    # ---- snapshots -----------------------------------------------------------

    def _state_delta(self) -> pb.StateDelta:
        free = free_vram_bytes()
        total = int(self.hardware.get("gpu_total_mem") or 0)
        quantum = max(1, int(total * _VRAM_QUANTUM_FRACTION)) if total else 1
        return pb.StateDelta(
            # th#1087/th#1085: echo DesiredResidency.config_generation.
            observed_config_generation=self.executor.runtime_config.generation,
            phase=self.phase,
            available_functions=self.executor.available_functions(),
            loading_functions=self.executor.loading_functions(),
            free_vram_bytes=(free // quantum) * quantum,
            # gw#516: encode/upload tails past the GPU-slot release. The hub
            # must not drain/retire this worker on GPU-idleness alone.
            finalizing_jobs=self.executor.finalizing_jobs(),
            observed_residency_generation=self._observed_residency_generation,
            # pgw#1032: `cell_lookups` is NOT filled. It advertised COMPUTED
            # (kind="inductor") keys, a space with no producer since pgw#1010,
            # so the store lookups it fed could never resolve. A target now
            # states only the identity it IS serving. The wire field retires
            # with the RunJob cut (th#1457/pgw#891).
            compile_targets=self.executor.compile_targets(),
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
                d2h_gbps=c.d2h_gbps,
                pinned_alloc_ok=c.pinned_alloc_ok,
                cpu_single_mbps=c.cpu_single_mbps,
                cpu_multi_mbps=c.cpu_multi_mbps,
                vcpus=c.vcpus,
                ram_total_gb=c.ram_total_gb,
                duration_ms=c.duration_ms,
                # pgw#748: the measured GPU fabric, alongside gpu_count.
                interconnect=c.interconnect,
                peer_gbps=c.peer_gbps,
                peer_access=c.peer_access,
                topo_link=c.topo_link,
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
            # pgw#1129/th#1798: the host driver, so the hub can answer
            # "can the host we landed on run this pod's CUDA line?" from a
            # SUCCESSFUL boot instead of only from a corpse.
            driver_version=str(hw.get("driver_version") or ""),
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
        self.intent_registry.refresh_projection(
            self.executor,
            self._desired_residency,
            self._model_resolutions,
        )
        return pb.Hello(
            protocol_version=PROTOCOL_VERSION,
            worker_id=self.worker_id,
            release_id=self.release_id,
            resources=self.build_resources(),
            state=self._state_delta(),
            models=self.executor.store.residency_snapshot(),
            in_flight=[
                pb.InFlightJob(request_id=rid, attempt=att) for rid, att in sorted(in_flight)
            ],
            # th#965 layer 2: promise the beat cadence; the hub reaps after
            # 6 consecutive misses (~60s). Hello counts as the first beat.
            heartbeat_interval_ms=HEARTBEAT_INTERVAL_MS,
            worker_session_id=self.intent_registry.worker_session_id,
            lifecycle_snapshot=self.intent_registry.snapshot(),
        )

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        # pgw#764: the stream exists now, so bind the boot-phase sink and FLUSH
        # everything recorded before it did. This is the earliest point a boot
        # row can reach the hub at all: activity's own sink is not bound until
        # Executor.ensure_setup, which is after weights are on disk, and the
        # send queue drops queued events on every reconnect. Binding here also
        # re-flushes after a reconnect, and the hub upserts on
        # (boot_id, ordinal), so a duplicate delivery is harmless.
        try:
            boot_mod.bind_sink(self.executor._send, asyncio.get_running_loop())
            # process start -> hello: the first boot number, and the one the
            # hub could previously only infer from pod create -> connect.
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
        # pgw#709: arm the cell-receipt gate the moment the hub wiring
        # exists — every delivered compile cell is signature-verified
        # before it may arm. Same wiring moment as the CellPublisher's.
        if self.executor.file_base_url:
            receipts.configure(
                base_url=self.executor.file_base_url,
                worker_jwt=self.executor.worker_jwt_provider,
            )
            # th#1307: arm the hub-side C2PA signer at the same moment. The
            # platform private key never enters this pod — the hub signs the
            # claim bytes. Until this call lands, a signing-configured worker
            # FAILS media requests rather than shipping them unsigned.
            content_credentials.configure_remote_signer(
                base_url=self.executor.file_base_url,
                worker_jwt=self.executor.worker_jwt_provider,
            )
            # pgw#1183 / §1.5: the same wiring moment is the first moment this
            # pod CAN discharge an upload it still owes. A cell minted by a
            # process that died mid-upload — or by a pod retired mid-upload —
            # is durable in the local CAS with its record still `pending`;
            # this ships it. Off the critical path, and a no-op on the pods
            # that owe nothing, which is nearly all of them.
            try:
                fleet_cells.resume_owed_publishes(
                    self.executor._cell_publisher())
            except Exception:  # noqa: BLE001 — never blocks the handshake
                logger.warning(
                    "owed cell uploads could not be re-attempted", exc_info=True)
        # th#1087: the desired state advertises (release_id, config_gen).
        # Observe it (memory + snapshot file rewrite); parameter VALUES ride
        # RunJob stamps (class 1), bindings ride the desired-residency
        # reconcile below (class 2), envs are boot-only (class 3 — the hub
        # drain-rolls; the worker never mutates its own env).
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
        # th#697: apply the hub's precision-ladder picks for THIS card
        # (full-replace: refs absent from the map revert to declared).
        resolutions = {
            r.ref: (r.resolved_ref, r.cast, r.lane, bool(r.lane_pinned))
            for r in ack.resolutions
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
                generation,
                self._observed_residency_generation,
            )
        else:
            self._observed_residency_generation = generation
            # th#1330: a superseded declared spelling must not be GC-protected
            # either — an unloaded base outranking genuinely cold refs in the
            # preserve set is the same waste one eviction later.
            _superseded = self._superseded_disk_refs(desired)
            self.executor.store.keep = list(dict.fromkeys(
                ref for ref in desired.disk_refs if ref and ref not in _superseded))
            self.executor.store.replace_desired_snapshots(
                dict(desired.snapshots),
                generation=generation,
            )
            self._replace_residency_reconcile(desired, model_key=_semantic_model_key(ack, desired))
            # pgw#674 rotation preload: the SAME desired set also feeds the
            # background stager, which — unlike the reconcile above — is not
            # tenant-idle-gated and not cancelled by run_job: it stages the
            # NEXT instance while jobs compute (fence-conflicting refs are
            # left to the idle-gated pass).
            self.executor.preloader.update_desired(
                list(desired.hot), dict(desired.snapshots), generation)
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
            # pgw#955: not cancelling is not the same as doing nothing. The
            # ack already opened a republish epoch (replace_desired_snapshots),
            # and only a reconcile pass reads it — so returning silently here
            # dropped the hub's redrive on the floor, with no cancel, no
            # restart, and no later event to heal it. Signal the level-
            # triggered loop instead: it finishes the item it is on (gw#614
            # intact) and then re-converges, which re-announces the held
            # identities under the new epoch. The re-pass is cheap and cannot
            # spam — satisfied refs short-circuit, and the epoch guard emits
            # at most one re-report per ref per applied ack.
            self._signal_residency_restart()
            wanted = {instance.function_name for instance in desired.hot if instance.function_name}
            if wanted.issubset(set(self.executor.available_functions())):
                self.intent_registry.bindings_applied(int(desired.config_generation))
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

    def _superseded_disk_refs(self, desired: "pb.DesiredResidency") -> set:
        """th#1330/th#1316: declared disk_refs the hub's OWN resolutions have
        already replaced, and whose replacement is in the same list.

        The hub addresses workers in resolved ref space (th#736), but a moment
        with no pick — a transient flavor resolve, a refusal, an
        unminted-cell gate — seeds the BARE spelling into desired.disk_refs,
        and no hub path removes it once the pick returns (declared and
        effective collapse to one key in the removal layer). The worker then
        materializes both: measured on prod, a 6.94 GB bf16 base staged ahead
        of the 4.38 GB fp8 variant it was meant to replace, ~144 s of a 270 s
        cold boot, for weights that are never loaded.

        Skipping is safe exactly when the resolved twin is ALSO listed: the
        hub is already asking for the artifact this ref resolves to, so the
        declared spelling can only add bytes. When the resolved twin is NOT
        listed the ref is materialized normally — a lane override that keeps
        the canonical spelling (th#913) is never skipped."""
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
                _snapshot_content_key(desired.snapshots.get(ref)),
                resolutions.get(ref),
            )
            for ref in sorted({r for r in refs if r})
        )

    def _active_work_still_desired(
        self,
        desired: "pb.DesiredResidency",
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
        snapshots = dict(desired.snapshots)
        # th#1330: never materialize a declared spelling the hub's own
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
            # pgw#638 (REVERTED, evidence in the tracker): removing this
            # tenant-idle gate is the obvious fix for the measured staging
            # starvation (4.7 GB staged downloads stuck at 0% for minutes on
            # busy workers) and it is WRONG. The pass's first act is to fence
            # a stale resident identity for this ref, and a tenant job may
            # legitimately be re-materializing OLDER bytes concurrently; with
            # the gate gone the fence races the job's load and the worker
            # reports the stale identity until the next HelloAck (th#1066
            # drift). Proven by
            # tests/test_p2_residency_reconcile.py::
            # test_mutable_tag_move_fences_events_by_digest_and_generation,
            # which does not converge within 10s without this gate — and is
            # NOT fixable by re-verifying after the job, because handle_run_job
            # returns before the job's load completes. The real fix is
            # pgw#641 Stage 4: transfers become their own tasks with their own
            # lifecycle, so staging never sits inside this serialized pass.
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
            # pgw#638 note: this cancel is LOAD-BEARING for the mutable-tag
            # identity fence — see the reverted-gate comment in
            # _reconcile_pass. Making downloads survive preemption needs
            # transfers to be first-class tasks (pgw#641 Stage 4).
            self._cancel_residency_reconcile()
            try:
                await self.executor.handle_run_job(msg.run_job)
            finally:
                self._resume_residency_reconcile()
        elif which == "run_attempt":
            # pgw#904: the Plan head. Same admission bracket as run_job — a
            # tenant request preempts unrelated background transfer/setup.
            attempt = msg.run_attempt
            if self.intent_registry.protocol_rejected:
                if attempt.HasField("attempt") and attempt.attempt.request_id:
                    await self.executor._send_result(
                        attempt.attempt.request_id,
                        int(attempt.attempt.attempt),
                        pb.JOB_STATUS_RETRYABLE,
                        safe_message="worker rejected mandatory desired state",
                    )
                return
            self._cancel_residency_reconcile()
            try:
                await self.executor.handle_run_attempt(attempt)
            finally:
                self._resume_residency_reconcile()
        elif which == "cancel_job":
            self.executor.handle_cancel(msg.cancel_job)
        elif which == "model_op":
            # pgw#1032: the sole ModelOp kind (ADOPT_COMPILE_CACHE) is retired.
            # Its hub-side push was keyed off the COMPUTED cell key, a space
            # with no producer since pgw#1010, so no stack has ever dispatched
            # one. Ignore rather than refuse: a hub that has not yet deployed
            # th#1702 must not lose its stream over a message we simply no
            # longer act on.
            logger.warning(
                "ignoring retired ModelOp (pgw#1032): hot compile-cache "
                "adoption is gone; a cell arrives only as a Plan's exact "
                "Arm.artifact (pgw#904) — the hub never pushes one")
        elif which == "serve_posture":
            # pgw#1142 / §4.32 item 4: the operator's eager-only command.
            # Applied SYNCHRONOUSLY and unconditionally — it touches one
            # process-global order and never blocks, so there is no state in
            # which a worker is "too busy" to stop using a broken cell, and
            # in-flight requests pick it up at their next compiled call.
            # Idempotent, so the hub replaying it on reconnect is free.
            serve_posture.apply_command(
                bool(msg.serve_posture.eager_only),
                actor=str(msg.serve_posture.actor or ""),
                reason=str(msg.serve_posture.reason or ""),
            )
        elif which == "drain":
            # The hub asked, explicitly, for this budget on this drain — an
            # operator budget on a command, which pgw#887 keeps.
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
        """pgw#654 observability: every WORKER_PHASE_ERROR names its cause on
        a durable channel. The hub persists only the bare phase flip
        ("worker phase reported error" -> release breaker), and a worker's
        lifecycle snapshot can be dropped by hub shadow validation — the
        combination made the 0.58.0 fleet-wide breakage undiagnosable
        (ie#544: signature="" detail="worker phase reported error"). Dial
        the gw#640 worker-fatal carrier once per process, best-effort; the
        hub stores it as a queryable pod_events row."""

        if exc is not None:
            detail = worker_fatal.build_fatal_detail(f"phase_error: {reason}", exc, exit_code=0)
        else:
            detail = f"phase=phase_error: {reason}"
        logger.error("entering WORKER_PHASE_ERROR: %s", detail)
        self.phase = pb.WORKER_PHASE_ERROR
        # getattr: stubbed Lifecycles skip __init__ (same convention as
        # _cancel_residency_reconcile, pgw#610).
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
        # force (th#965 layer 2): the heartbeat tick re-sends an unchanged
        # snapshot — receipt IS the beat the hub timestamps.
        if force or raw != self._last_delta:
            self._last_delta = raw
            await self._send_state_message(
                pb.WorkerMessage(state_delta=delta),
                hello_ack=hello_ack,
            )
            # pgw#797: THE boot close, and its only owner. "Servable" means the
            # hub may dispatch to this worker, and the hub dispatches to
            # `available_functions` — so the fact is "a StateDelta advertising
            # at least one function has gone out on a connected stream", which
            # is observable here and nowhere else.
            #
            # pgw#789 marked it from `Lifecycle.startup()` instead, once per
            # code path. That missed the path EVERY real release takes: a spec
            # declaring `Compile` (or slots) is routed to `dynamic` and set up
            # on hub delivery, so `awaiting_hub` was empty, the milestone fired
            # at the end of `startup()` — which `worker.arun` runs CONCURRENTLY
            # with the transport — and the boot closed before `hello`. Live on
            # chaos/0.78.0: ordinal 1 `first_request_servable` 8.9s, ordinal 2
            # `hello` 13.2s, and every span after it suppressed by `in_boot()`.
            if delta.available_functions and not self.draining:
                self._mark_servable()
        await self._send_lifecycle_snapshot(
            hello_ack=hello_ack,
            force=force,
        )
        # Deduped internally; must run even on an unchanged delta — a runtime
        # ladder demotion (gw#463) changes the ServePlan, not the delta bytes.
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
                pb.WorkerMessage(
                    fn_degraded=pb.FnDegraded(
                        function_name=name,
                        wanted=plan.wanted,
                        ran=ran,
                        reason=plan.warning,
                        est_latency_multiplier=float(plan.est_latency_multiplier),
                        recommended_vram_gb=float(plan.recommended_vram_gb or 0.0),
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
        # th#965 layer 2: the beat starts BEFORE any boot work so a hang
        # anywhere in setup is still covered — the task shares this event
        # loop, so it beats iff the loop is servicing tasks.
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
        # Disk truth first: after a restart the CAS dir is full while Residency
        # starts empty — rescan so Hello.models (and disk GC) see reality.
        self.executor.store.rescan_disk()
        self.executor.gate_functions(self.hardware)

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
                    # pgw#789: the weights_fetch boot span lives in
                    # `ModelStore._materialize_local`, NOT here. This call site
                    # only covers hf/civitai prefetch refs; the tensorhub refs
                    # that own the ~230s of a real cold boot arrive later via
                    # DesiredResidency/RunJob and never pass through here, so a
                    # span here measured the cheap path and left the expensive
                    # one invisible. The materializer is also the only layer
                    # that can stamp bytes AND source (network vs warm volume
                    # vs local CAS), which is what the fetch rate needs.
                    await self.executor.store.ensure_local(ref)
                except Exception as exc:
                    # pgw#655: this used to log and walk on to READY with an
                    # unmaterialized model, so the failure resurfaced as a
                    # per-job load failure on paid GPU. `_materialize_local`
                    # has already retried the transient classes with backoff;
                    # reaching here means the model is not coming, so the
                    # functions that need it go unavailable NOW and the hub
                    # stops routing to them.
                    gated = self.executor.mark_ref_unmaterializable(
                        ref, f"{type(exc).__name__}: {exc}")
                    logger.error(
                        "startup prefetch of %s failed terminally: %s — gating %s "
                        "(pgw#655: a worker never advertises a function whose "
                        "model is absent)",
                        ref, exc, ", ".join(gated) or "no function",
                    )

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
                # forever (ie#455 z-image fns=[]), so say so loudly instead of
                # dropping the function in silence.
                awaiting_hub[spec.name] = missing
                continue
            try:
                # pgw#797: the `pipeline_load` span moved INTO
                # `Executor.ensure_setup`. It was here and at
                # `_setup_awaiting_functions`, i.e. on the two boot paths a
                # release declaring `Compile` never takes — those specs are
                # routed to `dynamic` above and set up from hub delivery
                # (`_ensure_hot_instance` / RunJob), so the span measured the
                # paths that cost nothing and missed every real load. Same
                # doctrine as pgw#789 moving `weights_fetch` into the
                # materializer: instrument the layer all callers pass through.
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
            # gw#591: the hub's desired-disk plan delivers those snapshots
            # seconds later, but nothing re-ran setup — the function sat in
            # loading_functions forever while the hub dispatches only to
            # advertised functions (each side waiting on the other; found
            # live, ie#519). Finish boot setup the moment the refs land.
            self._boot_setup_watch = asyncio.create_task(
                self._setup_awaiting_functions(awaiting_hub), name="boot-setup-watch"
            )

        # pgw#764: FIRST-REQUEST-SERVABLE. Boot is over here — READY is the
        # moment the hub may dispatch to this worker, whether it got there
        # eager (pgw#671 eager-first boot, mint still running in the
        # background) or fully compiled. Measured from OS process start, so it
        # is the same quantity the autoscaler's cold-boot horizon prices
        # against, and everything after it is optimization, not boot.
        #
        # pgw#797: NOT marked here, on any path. pgw#789 special-cased
        # `awaiting_hub` and left the `dynamic` (compile-cell / slot) path —
        # i.e. every real release — closing the boot at the end of `startup()`,
        # concurrently with and usually BEFORE the transport's own hello. The
        # milestone now has exactly one owner, `maybe_send_state_delta`, which
        # marks it on the fact itself: a StateDelta advertising a function went
        # out. If no function is ever advertised the boot has no closing
        # milestone, and that is the finding (same doctrine as an open span).
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
        # pgw#1087: the pod reports its OWN decomposition, once, at the instant
        # the boot closes — and says so when it cannot explain itself. A
        # measurement whose only consumer is a hub table nobody has queried yet
        # is how `cell_discover` survived four releases with no producer; the
        # boot that took the time is the one place a hole is cheap to notice.
        # `SHAPE_ENTRYPOINT` is the MINIMUM every pod boot produces (the cell
        # phases depend on what the boot did), so `missing` here is always an
        # instrument defect, never a boot-shape difference.
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

    async def _setup_awaiting_functions(self, awaiting: Dict[str, List[str]]) -> None:
        """Complete boot setup for functions whose tensorhub snapshots arrive
        via hub delivery after the startup scan (gw#591), then push a
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
                    # pgw#797: span owned by `Executor.ensure_setup`.
                    await self.executor.ensure_setup(spec)
                    logger.info(
                        "boot setup of %s completed after hub snapshot delivery (gw#591)", fn
                    )
                except Exception as exc:
                    logger.error("startup setup of %s failed: %s", fn, exc)
                await self.maybe_send_state_delta()
            if pending:
                await asyncio.sleep(_BOOT_SETUP_WATCH_INTERVAL_S)
        # pgw#797: no mark here either. The `maybe_send_state_delta` above is
        # the one that advertises these functions, and it owns the milestone.

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
        arrived on the wire carrying one (pgw#887).
        """
        self.draining = True
        self.executor.draining = True
        self._cancel_residency_reconcile()
        self.executor.preloader.stop()  # pgw#674: no staging into a drain
        # th#965: the heartbeat deliberately keeps beating through drain — a
        # worker hung mid-drain must still be detectable as dead. It is
        # cancelled after the stream closes in _finish_drain.
        logger.info("drain started (deadline_ms=%d)", deadline_ms)
        deadline_s = (deadline_ms / 1000.0) if deadline_ms > 0 else None
        loop = asyncio.get_running_loop()
        self._drain_deadline_at = loop.time() + deadline_s if deadline_s is not None else None
        # pgw#887: two deadlines, because they bound two different things and
        # collapsing them is the defect. `_drain_deadline_at` bounds the
        # SHUTDOWN — the result flush (pgw#845) and the stop time this worker
        # reports to the hub — and every drain has one. `_drain_work_deadline_at`
        # bounds the TENANT WORK, and only a caller that explicitly asked for
        # one gets it: a `Drain` carrying `deadline_ms` is an operator budget
        # on a command (th#1235), while the SIGTERM handler's fleet default is
        # exactly what abandoned a 29-minute mint as `abandoned_shutdown`.
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

        pgw#887 / gw#666 / DESIGN-RULINGS §2. This was
        ``wait_idle(timeout=deadline_at - now)``, so at 30 s (SIGTERM) or 60 s
        (the hub's cluster drain, which is what BOTH standing stacks run) the
        drain logged "drain deadline expired" and called ``abort_all()``.
        Nothing consulted progress: a render at step 30/50 and a render at
        step 1/50 were aborted identically, and a MINT — which has no partial
        result to requeue — was abandoned outright. ``self_mint_abort`` has 52
        rows on chaos and 64 on master; one was recorded verbatim:
        *"attempt sixteen ran 29 minutes and died
        ``self_mint_abort/abandoned_shutdown``"*.

        A drain is a COMMAND, so a deadline on it is legitimate — but the
        deadline must bound acknowledgement and shutdown, not the tenant work
        underneath. Admission already stopped in ``_begin_drain``; what ends
        this wait is the work ceasing to advance, which
        ``progress.self_diagnosis()`` already answers and the 10 s beat
        already reports to the hub. Nothing new is invented here.

        What stayed, and why. A ``Drain`` that arrives on the wire carrying
        ``deadline_ms`` IS the "explicit operator budget on a specific command"
        the ruling allows (th#1235) — the caller chose it, for this drain. What
        is deleted is the worker INVENTING one: the SIGTERM handler's
        `_SIGNAL_DRAIN_DEADLINE_MS` is no longer a work budget, so a pod
        stopping on a signal no longer abandons a mint at 30 s. The hub's own
        60 s cluster window is a hub-side default and is routed to tensorhub
        rather than reinterpreted here. Even under an explicit budget the
        progress test still applies and fires FIRST: a wedged job is given up
        on when it wedges, not when the clock happens to run out.

        Two properties worth stating because they are deliberate:

        * **No counters open at all is NOT a stall.** It is the same "keep
          waiting" the in-call stall loop takes (``executor``'s
          ``_STALL_POLL_S`` arm: *"advancing (or evidence-free): keep
          waiting"*), and the outer authority — the hub's terminate, or the
          container runtime's SIGKILL — is what bounds the pathological case.
          The worker's job is to not throw away work that was about to finish.
        * ``self_diagnosis()`` is registry-wide, so a background mint's
          advancing counter can currently defer this verdict for a wedged
          tenant job. That cross-talk is pgw#894's, is named there, and is
          strictly better than the wall clock this replaces.
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
                self.executor.shutdown_instances(),
                operation="drain instance shutdown",
            )
            if self.transport is not None:
                self.intent_registry.set_drain(
                    pb.DRAIN_LIFECYCLE_STATUS_FLUSHING,
                )
                await self.maybe_send_state_delta()
                # pgw#845: this used to be `max(0.0, deadline_at - loop.time())`,
                # which is exactly 0.0 on every path that gets here after the
                # tenant wait expired — i.e. precisely when `abort_all` above has
                # just produced the typed RETRYABLE results whose entire purpose
                # is to tell the hub to re-dispatch. A bound that is provably
                # zero is not a bound. The deadline governs waiting for TENANT
                # WORK; shipping results we already hold is not that, so the
                # floor is the channel's own peer-dead threshold: we try for as
                # long as the transport would still call the hub alive.
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
        # getattr: stubbed Lifecycles skip __init__ (same convention as
        # _cancel_residency_reconcile, pgw#610).
        beat_task = getattr(self, "_heartbeat_task", None)
        if beat_task is not None:
            beat_task.cancel()
            self._heartbeat_task = None
        logger.info("drain complete")
