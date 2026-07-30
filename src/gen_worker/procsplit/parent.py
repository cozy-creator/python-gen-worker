"""Parent (control-plane) side of the pgw#763 split.

Owns: the gRPC stream + identity/JWT (the real ``Transport``), the compute
child's lifetime (spawn / respawn-on-failure / crash-loop detection /
watchdog), the durable SendQueue, and job attribution across child deaths.
Never imports torch.

Supervision primitives are deliberately systemd's (Paul, 2026-07-29):
``Restart=on-failure`` (always respawn), ``StartLimitBurst``/
``StartLimitIntervalSec`` (loop DETECTION over a window — reported typed, never
cap-and-brick), ``WatchdogSec``+``sd_notify`` (child frames are the liveness
pings; loop silence ARMS a hang verdict that the child's accounted work
DECIDES — pgw#771), socket activation (the hub
connection outlives the process doing the work). What systemd cannot do — and
the reason this is not s6/supervisord in the container — is job attribution:
"request X died of OOM, the release is healthy, the worker is alive, send the
next job" requires holding the stream, the JWT, and the in-flight table here.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import shlex
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings
from ..pb import worker_scheduler_pb2 as pb
from ..transport import FatalTransportError, Transport
from .. import postmortem
from .. import worker_fatal
from . import (
    ENV_CHILD,
    ENV_CHILD_CMD,
    ENV_LIVENESS_FD,
    ENV_SOCKET,
    ENV_WATCHDOG_PING_S,
    actions,
    frames,
)

logger = logging.getLogger(__name__)

# The typed death label. Deliberately NOT in the hub's th#1288
# declaredFaultLabels allowlist: a child death can be payload-driven (an OOM
# this payload caused), so it must not classify as release-declared evidence.
# The hub's per-request blame-probe ladder handles it correctly as a FATAL.
DEATH_LABEL = "ComputeProcessDied"

_DEFAULT_START_LIMIT_BURST = 3          # StartLimitBurst
_DEFAULT_START_LIMIT_INTERVAL_S = 600.0  # StartLimitIntervalSec
_DEFAULT_WATCHDOG_BUDGET_S = 60.0        # WatchdogSec (matches th#965's reap budget)
_DEFAULT_RESPAWN_BACKOFF_BASE_S = 1.0
_DEFAULT_RESPAWN_BACKOFF_CAP_S = 60.0
_BACKOFF_RESET_AFTER_ALIVE_S = 60.0
_CRASH_LOOP_REPORT_MIN_INTERVAL_S = 300.0
_DEATH_FLUSH_GRACE_S = 2.0
# TimeoutStopSec: after the parent forwards SIGTERM, a child that has not
# exited is SIGKILLed rather than holding the pod open forever.
_DEFAULT_STOP_TIMEOUT_S = 120.0
# Bounded flush on a DELIBERATE exit (drain / stop): the queue's durable
# results get a real chance to ship before the parent leaves.
_STOP_FLUSH_TIMEOUT_S = 30.0
# After the child process is reaped, its last frames may still be sitting in
# the socket buffer (a JobResult written microseconds before death). Closing
# the link before draining them loses the result AND mis-attributes the job.
_LINK_SETTLE_TIMEOUT_S = 3.0
# The transport's drain close half-closes and waits for the peer to end the
# call. Cancelling that wait RSTs the call and discards already-retired writes.
_CLEAN_CLOSE_WAIT_S = 12.0
_REPORTED_DEAD_CAP = 512
# Minimum evidence advance that counts as life, mirroring activity._EVIDENCE_EPS
# (not imported: the parent's import graph stays minimal and torch-free).
_EVIDENCE_EPS = 0.05
# delta 1: pod-launch envs that must not survive into the compute child. Every
# one of them is a platform credential or the platform's identity claim, and
# the child imports tenant code — so `os.environ` in that process is a public
# noticeboard. WORKER_JWT is the signing identity; HF_TOKEN is the endpoint
# author's own credential and legitimately belongs to the code that pulls
# weights, so it deliberately stays.
_CHILD_FORBIDDEN_ENVS = ("WORKER_JWT",)
# A mediated hub call must not hold the parent's control loop open forever.
_ACTION_HARD_TIMEOUT_S = 120.0
_ACTION_REFUSAL_REPORT_MIN_INTERVAL_S = 300.0
# The host canary is a real benchmark (memcpy/D2H/CPU); on a cold pod with a
# large card it is seconds, not milliseconds. Generous, and bounded.
_MEASURE_TIMEOUT_S = 180.0


def _http_call(
    method: str,
    url: str,
    token: str,
    query: Dict[str, str],
    body: Optional[Dict[str, Any]],
    timeout: float,
) -> Tuple[int, str]:
    """The parent's half of a mediated call: the ONLY place the worker JWT is
    put on the wire on the child's behalf. Runs in a thread — the control loop
    never blocks on the hub."""
    import requests

    resp = requests.request(
        method,
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=query or None,
        json=body if method == "POST" else None,
        timeout=timeout,
    )
    return resp.status_code, resp.text


class _ChildLink:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = frames.FrameWriter(writer)
        self.saw_hello = False


class ParentControl:
    """The control process: real Transport + child supervision + attribution."""

    def __init__(
        self,
        settings: Settings,
        *,
        child_cmd: Optional[List[str]] = None,
        child_env: Optional[Dict[str, str]] = None,
        socket_path: Optional[str] = None,
        measure_cmd: Optional[List[str]] = None,
        respawn_backoff_base_s: float = _DEFAULT_RESPAWN_BACKOFF_BASE_S,
        respawn_backoff_cap_s: float = _DEFAULT_RESPAWN_BACKOFF_CAP_S,
        start_limit_burst: int = _DEFAULT_START_LIMIT_BURST,
        start_limit_interval_s: float = _DEFAULT_START_LIMIT_INTERVAL_S,
        watchdog_budget_s: float = _DEFAULT_WATCHDOG_BUDGET_S,
        stop_timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
        stop_flush_timeout_s: float = _STOP_FLUSH_TIMEOUT_S,
        beat_interval_s: float = 0.0,   # 0 = adopt the child's declared cadence
        transport_backoff_base_s: float = 1.0,
        transport_backoff_cap_s: float = 30.0,
    ) -> None:
        self._settings = settings
        env_cmd = os.environ.get(ENV_CHILD_CMD, "").strip()
        self._child_cmd = list(
            child_cmd
            if child_cmd is not None
            else (shlex.split(env_cmd) if env_cmd else [sys.executable, "-m", "gen_worker.entrypoint"])
        )
        self._child_env = dict(child_env or {})
        self._socket_path = socket_path or f"/tmp/gen-worker-compute-{os.getpid()}.sock"
        self._backoff_base = respawn_backoff_base_s
        self._backoff_cap = respawn_backoff_cap_s
        self._start_limit_burst = max(1, int(start_limit_burst))
        self._start_limit_interval = start_limit_interval_s
        self._watchdog_budget = watchdog_budget_s
        self._stop_timeout = stop_timeout_s
        self._stop_flush_timeout = stop_flush_timeout_s
        self.transport = Transport(
            settings,
            self,
            backoff_base_s=transport_backoff_base_s,
            backoff_cap_s=transport_backoff_cap_s,
        )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping = asyncio.Event()
        self._link: Optional[_ChildLink] = None
        self._link_ready = asyncio.Event()
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._hello_waiter: Optional[asyncio.Future] = None
        # (request_id, attempt) -> function_name, set at RunJob relay,
        # cleared when the child's JobResult passes back through.
        self._in_flight: Dict[Tuple[str, int], str] = {}
        self._death_times: collections.deque = collections.deque(maxlen=64)
        self._deaths_before_hello = 0
        # Survives link teardown (socket EOF races proc.wait), so a
        # post-Hello death is never misattributed as a boot death.
        self._child_saw_hello = False
        self._spawn_count = 0
        self._last_frame_at = time.monotonic()
        self._relaying = False
        self._watchdog_fired = False
        # pgw#771 liveness (thread-sourced, loop-independent).
        self._liveness_task: Optional[asyncio.Task] = None
        self._last_liveness_at = 0.0
        self._liveness_evidence: Optional[float] = None
        self._liveness_evidence_at = 0.0
        self._liveness_activity = ""
        self._hang_armed_at: Optional[float] = None
        self._hang_hold_reported = False
        self._stall_reported = False
        # pgw#771: the app beat the hub reaps on. The CHILD declares the
        # cadence and used to be the only sender, so a starved child stopped
        # beating and the hub killed the pod at ~6 misses — the split fixed
        # nothing there. The PARENT is the control loop now, so it originates
        # the beat: the last state the child published, re-sent on the child's
        # promised cadence, exactly the "periodic unchanged re-send" the hub
        # already treats as proof the control loop is alive.
        self._last_state_delta: Optional[pb.WorkerMessage] = None
        self._last_state_delta_at = 0.0
        self._beat_interval = beat_interval_s
        self.parent_beats_sent = 0  # observability + tests
        # How long an OPEN activity may go without accruing CPU/IO before its
        # hold lapses: the child's ping cadence is the clock.
        raw_ping = (
            self._child_env.get(ENV_WATCHDOG_PING_S)
            or os.environ.get(ENV_WATCHDOG_PING_S)
            or ""
        )
        try:
            ping_s = float(raw_ping) or 5.0
        except ValueError:
            ping_s = 5.0
        self._evidence_hold_window = max(3.0 * ping_s, 2.0)
        self._draining = False
        self._terminating = False
        self._child_exited_clean = False
        self._last_crash_loop_report_at = 0.0
        self.crash_loop_reports = 0  # observability + tests
        # Set once the child's link read loop has finished (EOF drained), so
        # death attribution never races the child's last frames.
        self._link_closed = asyncio.Event()
        self._link_closed.set()
        # Held CLEAR from the moment the child is reaped until its in-flight
        # jobs have been attributed into the durable queue. A concurrent drain
        # flush must not declare the queue empty before the death report is in
        # it (the FATAL would be dropped by the flush's own stop).
        self._death_report_done = asyncio.Event()
        self._death_report_done.set()
        # (request_id, attempt) already terminal-reported by the death path:
        # a late cancel for one of these is dropped, not relayed to the fresh
        # child that never heard of the request.
        self._reported_dead: collections.OrderedDict = collections.OrderedDict()
        self._stop_deadline_task: Optional[asyncio.Task] = None
        self._reported_unretired = False
        self.unretired_results_at_exit = 0  # observability + tests
        # delta 1: parent-mediated action accounting (observability + tests).
        self._jwt_rotations = 0
        self.actions_performed = 0
        self.actions_refused = 0
        self._last_action_refusal_report_at = 0.0
        # The hub base the parent will direct mediated calls at. Captured from
        # the HelloAck it relays — the CHILD never names a host.
        self._file_base_url = ""
        self._identity_cache: Optional[Tuple[str, str]] = None
        # delta 2: the parent's own pre-import host measurement.
        self._measure_cmd = list(
            measure_cmd
            if measure_cmd is not None
            else [sys.executable, "-m", "gen_worker.procsplit.measure"]
        )
        self._measurement: Optional[Dict[str, Any]] = None
        self._measured = asyncio.Event()
        self._measure_task: Optional[asyncio.Task] = None

    # ---- hardware + canary (parent-owned, PRE-IMPORT) ---------------------

    async def _measure_host(self) -> None:
        """Measure the silicon in a process that has imported no tenant code.

        Runs once, at boot, concurrently with the first child spawn — the child
        cannot produce a Hello before this finishes anyway (it has models to
        find), so the measurement is off the critical path in practice and
        bounded here regardless.
        """
        cmd = list(self._measure_cmd)
        env = dict(os.environ)
        for name in _CHILD_FORBIDDEN_ENVS:
            env.pop(name, None)   # it measures hardware; it needs no credential
        env.pop(ENV_CHILD, None)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, env=env, stdout=asyncio.subprocess.PIPE,
            )
            raw, _ = await asyncio.wait_for(
                proc.communicate(), _MEASURE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.error(
                "host measurement did not finish within %.0fs; the Hello will "
                "ship without parent-measured resources", _MEASURE_TIMEOUT_S,
            )
            return
        except Exception:
            logger.warning("host measurement subprocess failed", exc_info=True)
            return
        finally:
            self._measured.set()
        try:
            self._measurement = json.loads(raw.decode() or "{}")
        except ValueError:
            logger.warning("host measurement produced no usable JSON")
            return
        hw = self._measurement.get("hardware") or {}
        logger.info(
            "parent-measured host: gpu=%s x%s sm=%s torch=%s canary=%s "
            "(measured before any endpoint import)",
            hw.get("gpu_name") or "-", hw.get("gpu_count") or 0,
            hw.get("gpu_sm") or "-", hw.get("torch_version") or "-",
            "yes" if self._measurement.get("canary") else "no",
        )

    def _parent_resources(self) -> Optional[pb.WorkerResources]:
        """``Hello.resources`` built from the PARENT's own measurement."""
        m = self._measurement
        if m is None:
            return None
        hw = m.get("hardware") or {}
        canary = None
        c = m.get("canary")
        if isinstance(c, dict):
            canary = pb.HostCanary(
                memcpy_gbps=float(c.get("memcpy_gbps") or 0.0),
                d2h_gbps=float(c.get("d2h_gbps") or 0.0),
                pinned_alloc_ok=bool(c.get("pinned_alloc_ok")),
                cpu_single_mbps=float(c.get("cpu_single_mbps") or 0.0),
                cpu_multi_mbps=float(c.get("cpu_multi_mbps") or 0.0),
                vcpus=int(c.get("vcpus") or 0),
                ram_total_gb=float(c.get("ram_total_gb") or 0.0),
                duration_ms=int(c.get("duration_ms") or 0),
                interconnect=str(c.get("interconnect") or ""),
                peer_gbps=float(c.get("peer_gbps") or 0.0),
                peer_access=bool(c.get("peer_access")),
                topo_link=str(c.get("topo_link") or ""),
            )
        return pb.WorkerResources(
            host_canary=canary,
            gpu_count=int(hw.get("gpu_count") or 0),
            vram_total_bytes=int(hw.get("gpu_total_mem") or 0),
            gpu_name=str(hw.get("gpu_name") or ""),
            gpu_sm=str(hw.get("gpu_sm") or ""),
            torch_version=str(hw.get("torch_version") or ""),
            installed_libs=[str(x) for x in (hw.get("installed_libs") or [])],
            gen_worker_version=str(m.get("gen_worker_version") or ""),
            # Platform-delivered facts: they identify the pod to the hub, so
            # they belong to the process holding the pod's credential.
            image_digest=self._settings.worker_image_digest,
            instance_id=self._settings.runpod_pod_id or "",
        )

    # ---- identity (parent-owned) -----------------------------------------

    def _identity(self) -> Tuple[str, str]:
        """(worker_id, release_id) from the JWT THIS process holds.

        The credential and the identity it asserts stay together: the child
        cannot derive either (delta 1), and the parent refuses to relay a Hello
        that claims a different one (``build_hello``).
        """
        if self._identity_cache is not None:
            return self._identity_cache
        worker_id = (self._settings.worker_id or "").strip()
        release_id = ""
        token = (self._settings.worker_jwt or "").strip()
        if token:
            try:
                from ..request_context import _decode_unverified_jwt_claims

                claims = _decode_unverified_jwt_claims(token)
                worker_id = worker_id or str(claims.get("sub") or "").strip()
                release_id = str(claims.get("release_id") or "").strip()
            except Exception:
                logger.warning("could not decode the worker JWT claims", exc_info=True)
        self._identity_cache = (worker_id, release_id)
        return self._identity_cache

    # ---- Transport handlers ---------------------------------------------

    async def build_hello(self) -> pb.Hello:
        """Fetch the CHILD's fresh Hello, then merge the parent's durable
        pending-result keys so the hub's in-flight reconcile sees results that
        outlived a child (or a stream)."""
        # delta 2: never assemble a Hello before the parent's own measurement
        # has had its chance — a first Hello that shipped the child's numbers
        # would be exactly the forgery this closes, arriving once per boot.
        try:
            await asyncio.wait_for(self._measured.wait(), _MEASURE_TIMEOUT_S + 5.0)
        except asyncio.TimeoutError:
            pass
        while True:
            link = self._link
            if link is not None:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                self._hello_waiter = fut
                try:
                    await link.writer.frame(frames.T_HELLO_REQ, frames.pack_meta({}))
                    raw = await fut
                except (ConnectionError, OSError, asyncio.CancelledError):
                    raise
                finally:
                    if self._hello_waiter is fut:
                        self._hello_waiter = None
                hello = pb.Hello.FromString(raw)
                # delta 1: identity is the credential holder's to assert. The
                # child builds the serving state; the worker/release it claims
                # to BE comes from the JWT in this process, so a child that
                # rewrote either (tenant code reaches the whole Hello) cannot
                # make the hub attribute its state to another worker.
                worker_id, release_id = self._identity()
                if worker_id:
                    hello.worker_id = worker_id
                if release_id:
                    hello.release_id = release_id
                # delta 2: and the HARDWARE the Hello asserts. Every field here
                # is a fleet-wide verdict key (th#1310) — HardwareUnsuitable
                # machine fences, HostCanary SKU condemnation, the gpu_name that
                # chooses which key gets written — and the child measured them
                # after importing tenant code. Replace, don't merge: a partial
                # overwrite leaves whichever axis we forgot forgeable.
                resources = self._parent_resources()
                if resources is not None:
                    hello.resources.CopyFrom(resources)
                elif hello.HasField("resources"):
                    logger.error(
                        "no parent-side host measurement is available; DROPPING "
                        "the child's self-reported resources rather than "
                        "relaying tenant-reachable numbers the fleet condemns "
                        "SKUs on (pgw#763 delta 2 / th#1310)"
                    )
                    hello.ClearField("resources")
                if self._beat_interval <= 0 and hello.heartbeat_interval_ms > 0:
                    # The child's own promise is the cadence the parent keeps.
                    self._beat_interval = hello.heartbeat_interval_ms / 1000.0
                seen = {(j.request_id, j.attempt) for j in hello.in_flight}
                for rid, att in self.transport.queue.pending_result_keys:
                    if (rid, att) not in seen:
                        hello.in_flight.add(request_id=rid, attempt=att)
                for (rid, att) in self._in_flight:
                    if (rid, att) not in seen:
                        hello.in_flight.add(request_id=rid, attempt=att)
                return hello
            # No compute child yet (booting, or respawning after a death):
            # wait. The transport's own handshake deadline bounds this dial.
            await self._link_ready.wait()

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        # delta 1: the hub's own base URL, for the parent-mediated actions the
        # child asks for. Taking it here (rather than from a child-supplied
        # argument) is what stops a compromised child from aiming the pod's
        # worker JWT at a host of its choosing (th#1312).
        if ack.file_base_url:
            self._file_base_url = ack.file_base_url.rstrip("/")
        link = self._link
        if link is None:
            return
        # CONNECTED before the ack, mirroring Transport's _connected ordering.
        await link.writer.frame(frames.T_CONNECTED)
        await link.writer.frame(frames.T_HELLO_ACK, ack.SerializeToString())

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        which = msg.WhichOneof("msg")
        link = self._link
        if which == "run_job":
            run = msg.run_job
            if link is None:
                await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                    request_id=run.request_id,
                    attempt=run.attempt,
                    status=pb.JOB_STATUS_RETRYABLE,
                    safe_message="compute process restarting",
                )))
                return
            self._in_flight[(run.request_id, run.attempt)] = run.function_name
            try:
                await link.writer.frame(frames.T_SCHED, msg.SerializeToString())
            except (ConnectionError, OSError):
                # The child died under the relay: the job never started, so it
                # is retryable — not a handler death.
                self._in_flight.pop((run.request_id, run.attempt), None)
                await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                    request_id=run.request_id,
                    attempt=run.attempt,
                    status=pb.JOB_STATUS_RETRYABLE,
                    safe_message="compute process restarting",
                )))
            return
        if which == "cancel_job":
            key = (msg.cancel_job.request_id, msg.cancel_job.attempt)
            if link is None or key in self._reported_dead:
                # Already terminal-reported by the death path: relaying it to a
                # fresh child that never heard of the request is noise at best.
                return
            await link.writer.frame(frames.T_SCHED, msg.SerializeToString())
            return
        if which == "drain":
            self._draining = True
            if link is None:
                asyncio.create_task(self._drain_without_child(), name="drain-no-child")
                return
            await link.writer.frame(frames.T_SCHED, msg.SerializeToString())
            return
        if link is None:
            logger.warning("dropping %s command while compute child is down", which)
            return
        await link.writer.frame(frames.T_SCHED, msg.SerializeToString())

    async def on_message_shipped(self, msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") != "model_event":
            return
        link = self._link
        if link is not None:
            await link.writer.frame(frames.T_SHIPPED, msg.SerializeToString())

    async def on_disconnect(self) -> None:
        link = self._link
        if link is not None:
            try:
                await link.writer.frame(frames.T_DISCONNECTED)
            except Exception:
                pass

    async def on_token_refresh(self, token: str, expires_at_unix: int) -> None:
        """The rotated worker JWT stays HERE (delta 1).

        This used to write a ``T_TOKEN`` frame, i.e. hand the pod's signing
        identity to the process that imports tenant endpoint code — which is
        the whole of th#1311's credential-laundering material, delivered on a
        schedule. The transport keeps the token; the child asks for the narrow
        actions it needs (``procsplit/actions.py``) and never sees the bearer.
        """
        self._jwt_rotations += 1

    # ---- child link (unix socket server) ---------------------------------

    async def _on_child_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        old = self._link
        link = _ChildLink(reader, writer)
        self._link = link
        if old is not None:
            old.writer.close()
        self._last_frame_at = time.monotonic()
        self._link_closed.clear()
        self._link_ready.set()
        logger.info("compute child connected on %s", self._socket_path)
        try:
            while True:
                ftype, payload = await frames.read_frame(reader)
                self._last_frame_at = time.monotonic()
                await self._on_child_frame(link, ftype, payload)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            if self._link is link:
                self._link = None
                self._link_ready.clear()
            waiter = self._hello_waiter
            if waiter is not None and not waiter.done():
                waiter.set_exception(ConnectionError("compute child link lost"))
            link.writer.close()
            self._link_closed.set()

    async def _on_child_frame(self, link: _ChildLink, ftype: int, payload: bytes) -> None:
        if ftype == frames.T_WATCHDOG:
            return  # the timestamp update in the read loop IS the handling
        if ftype == frames.T_HELLO:
            link.saw_hello = True
            self._child_saw_hello = True
            waiter = self._hello_waiter
            if waiter is not None and not waiter.done():
                waiter.set_result(payload)
            return
        if ftype == frames.T_WORKER_MSG:
            msg = pb.WorkerMessage.FromString(payload)
            which = msg.WhichOneof("msg")
            if which == "job_result":
                r = msg.job_result
                self._in_flight.pop((r.request_id, r.attempt), None)
            elif which == "state_delta":
                # The freshest truth the child published; the beat re-sends it.
                self._last_state_delta = msg
                self._last_state_delta_at = time.monotonic()
            # SendQueue.put can backpressure (stream down, event lane full);
            # while the READ LOOP is blocked here the child's pings cannot be
            # read, so the watchdog must not mistake parent-side backpressure
            # for a wedged child.
            self._relaying = True
            try:
                await self.transport.send(msg)
            finally:
                self._relaying = False
                self._last_frame_at = time.monotonic()
            return
        if ftype == frames.T_ACTION_REQ:
            # Off the read loop: a mediated call is a network round trip, and
            # blocking here would stop the child's frames (results, watchdog
            # pings) from being read at all.
            asyncio.create_task(
                self._serve_action(link, frames.unpack_meta(payload)),
                name="parent-action",
            )
            return
        if ftype == frames.T_PREPEND:
            msgs = [pb.WorkerMessage.FromString(b) for b in frames.unpack_meta(payload)]
            await self.transport.prepend_reconnect(msgs)
            return
        if ftype == frames.T_FLUSH_REQ:
            meta = frames.unpack_meta(payload)
            timeout = meta.get("timeout")
            # The child only asks for this at the END of its own drain, so the
            # shutdown is deliberate however it was triggered (hub Drain,
            # child-side signal). Recording it here is what stops the parent
            # from respawning into a drain, or exiting 1 on a clean drain.
            self._draining = True
            flushed = await self.transport.close_after_flush(
                timeout=None if timeout is None else float(timeout)
            )
            try:
                await link.writer.frame(
                    frames.T_FLUSH_ACK, frames.pack_meta({"flushed": bool(flushed)})
                )
            except Exception:
                pass
            return
        logger.warning("unknown child frame type %d ignored", ftype)

    # ---- parent-mediated actions (delta 1) -------------------------------

    async def _serve_action(self, link: _ChildLink, req: Dict[str, Any]) -> None:
        """Decide and perform ONE action the child asked for.

        The child holds no credential, so every identity-bearing hub call it
        needs arrives here. The parent chooses the host, attaches the JWT, and
        answers with the response body and nothing else — no headers, no
        bearer, no redirect target.
        """
        rid = req.get("id")
        try:
            result = await self._perform_action(req)
        except actions.ActionRefused as exc:
            self.actions_refused += 1
            logger.error(
                "REFUSED parent-mediated action from the compute child: %s", exc
            )
            # A refusal is a security event: the child asked for authority it is
            # not allowed to borrow. Bank it durably, throttled by the same
            # report path every other typed worker event uses.
            await self._report_action_refusal(str(exc))
            reply: Dict[str, Any] = {"id": rid, "ok": False, "error": str(exc)}
        except Exception as exc:
            logger.warning("parent-mediated action failed", exc_info=True)
            reply = {"id": rid, "ok": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        else:
            self.actions_performed += 1
            reply = {"id": rid, "ok": True, **result}
        try:
            await link.writer.frame(frames.T_ACTION_RESP, frames.pack_meta(reply))
        except (ConnectionError, OSError):
            pass

    async def _perform_action(self, req: Dict[str, Any]) -> Dict[str, Any]:
        named = str(req.get("action") or "")
        if named:
            if named != actions.ACTION_REPORT_DETAIL:
                raise actions.ActionRefused(f"unknown action {named!r}")
            # th#1310: a worker report is a fleet-wide verdict key, so it is
            # worth more from the process that runs no tenant code. The child
            # supplies the text; the parent supplies the identity and the dial.
            detail = str(req.get("detail") or "")[:8000]
            delivered = await asyncio.to_thread(
                worker_fatal.report_worker_detail,
                self._settings,
                f"[compute-child] {detail}",
            )
            return {"result": {"delivered": bool(delivered)}}

        action, query, body = actions.authorize(req)
        base = self._file_base_url or (self._settings.tensorhub_public_url or "").rstrip("/")
        if not base:
            raise actions.ActionRefused(
                f"{action.name}: no hub base URL is known yet (no HelloAck)"
            )
        if action.scoped_to_job:
            body = self._narrow_job_scoped_action(action, dict(body or {}))
        token = self.transport.current_worker_jwt
        if not token:
            raise actions.ActionRefused(f"{action.name}: this pod holds no worker JWT")
        timeout = min(float(req.get("timeout") or action.timeout_s),
                      action.timeout_s, _ACTION_HARD_TIMEOUT_S)
        status, text = await asyncio.to_thread(
            _http_call, action.method, base + str(req.get("path") or ""),
            token, query, body, timeout,
        )
        logger.info(
            "parent-mediated %s -> %d (child holds no credential)", action.name, status
        )
        return self._post_action(action, status, text)

    def _narrow_job_scoped_action(
        self, action: "actions.HubAction", body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hook for the per-job authority policy (delta 4). Delta 1 enforces the
        one check that needs no token parsing: the parent will not renew a
        capability for a request it never dispatched."""
        if action.name != "capability.renew":
            return body
        rid = str(body.get("request_id") or "")
        try:
            attempt = int(body.get("attempt") or 0)
        except (TypeError, ValueError):
            attempt = -1
        if (rid, attempt) not in self._in_flight:
            raise actions.ActionRefused(
                f"capability.renew for {rid}#{attempt}: not an in-flight job on "
                "this worker — the parent renews only what it dispatched"
            )
        return body

    def _post_action(
        self, action: "actions.HubAction", status: int, text: str,
    ) -> Dict[str, Any]:
        """Last look at a response before it crosses back. Delta 4 narrows the
        capability token here."""
        return {"status": int(status), "body": text}

    async def _report_action_refusal(self, detail: str) -> None:
        now = time.monotonic()
        if now - self._last_action_refusal_report_at < _ACTION_REFUSAL_REPORT_MIN_INTERVAL_S:
            return
        self._last_action_refusal_report_at = now
        await self._dial_detail(
            f"phase=compute_action_refused refusals={self.actions_refused} "
            f"detail={detail[:400]} — the compute child asked the control parent "
            "for authority outside the allowlisted action table"
        )

    # ---- child lifetime --------------------------------------------------

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        env.update(self._child_env)
        # delta 1: the compute child gets NO signing identity. Deleting the
        # T_TOKEN frame is only half of it — the JWT also arrives at pod-launch
        # in WORKER_JWT, and `os.environ` is the first place tenant code looks.
        # Strip it from the child's environment so the credential's absence is
        # a property of the process, not of the code paths we remembered to
        # change. The PARENT keeps its own os.environ copy, so its Settings and
        # its Transport are unaffected.
        for name in _CHILD_FORBIDDEN_ENVS:
            env.pop(name, None)
        # ...but the child still needs its IDENTITY, which is not a credential:
        # the intent registry and every lifecycle snapshot are keyed on the
        # release id, and it used to be read out of the JWT the child no longer
        # has. The parent decodes its own token and passes the two claims down.
        worker_id, release_id = self._identity()
        if worker_id:
            env["WORKER_ID"] = worker_id
        if release_id:
            env["WORKER_RELEASE_ID"] = release_id
        env[ENV_CHILD] = "1"
        env[ENV_SOCKET] = self._socket_path
        # The gw#640 flight-recorder fork is redundant under this parent.
        env["GEN_WORKER_SUPERVISOR"] = "0"
        # pgw#771: a dedicated pipe for THREAD-sourced process liveness, so a
        # compile that starves the child's event loop cannot look like a hang.
        read_fd, write_fd = os.pipe()
        env[ENV_LIVENESS_FD] = str(write_fd)
        self._spawn_count += 1
        logger.info(
            "spawning compute child #%d: %s", self._spawn_count, " ".join(self._child_cmd)
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._child_cmd, env=env, pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)   # the child owns it now
        await self._start_liveness_reader(read_fd)
        return proc

    async def _start_liveness_reader(self, read_fd: int) -> None:
        old = self._liveness_task
        if old is not None and not old.done():
            old.cancel()
        self._last_liveness_at = 0.0
        self._liveness_evidence = None
        self._liveness_evidence_at = time.monotonic()
        self._liveness_activity = ""
        self._liveness_task = asyncio.create_task(
            self._liveness_loop(read_fd), name="parent-liveness"
        )

    async def _liveness_loop(self, read_fd: int) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        pipe = os.fdopen(read_fd, "rb", 0)
        try:
            transport, _ = await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), pipe
            )
        except Exception:
            logger.warning("liveness pipe unavailable; hang decisions fall back "
                           "to loop silence alone", exc_info=True)
            pipe.close()
            return
        try:
            while True:
                ftype, payload = await frames.read_frame(reader)
                if ftype != frames.T_LIVENESS:
                    continue
                meta = frames.unpack_meta(payload)
                self._last_liveness_at = time.monotonic()
                # Sticky by design: the flag is a fact about the last time the
                # child could speak. A GIL-starved thread stops speaking; the
                # activity it last reported is still the one running.
                self._liveness_activity = (
                    str(meta.get("kind") or "") if meta.get("act") else ""
                )
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            transport.close()

    async def _child_loop(self) -> None:
        backoff = self._backoff_base
        while not self._stopping.is_set():
            self._watchdog_fired = False
            self._child_saw_hello = False
            self._hang_armed_at = None
            self._hang_hold_reported = False
            self._stall_reported = False
            oom_before = postmortem.oom_kill_count()
            started = time.monotonic()
            self._last_frame_at = started
            try:
                proc = await self._spawn_child()
            except OSError as exc:
                logger.error("compute child spawn failed: %s", exc)
                await self._sleep_or_stop(backoff)
                backoff = min(backoff * 2, self._backoff_cap)
                continue
            self._proc = proc
            rc = await proc.wait()
            lifetime = time.monotonic() - started
            self._proc = None
            # From here until attribution is enqueued, a concurrent drain flush
            # must not conclude the queue is empty.
            self._death_report_done.clear()
            saw_hello = self._child_saw_hello
            # Drain the child's LAST frames before touching the link: a
            # JobResult written microseconds before death is still in the
            # socket buffer, and closing the writer here would discard it and
            # then mis-report that finished job as ComputeProcessDied.
            await self._settle_link()
            if self._stopping.is_set():
                self._death_report_done.set()
                return
            deliberate = self._terminating or (rc == 0 and not self._watchdog_fired)
            if deliberate:
                await self._finish_deliberate_exit(rc, lifetime_s=lifetime)
                return
            await self._handle_child_death(
                rc, oom_before=oom_before, lifetime_s=lifetime, saw_hello=saw_hello,
            )
            if self._draining:
                # Restart=on-failure does not apply to a pod that has been told
                # to go away: the death is reported and shipped, but respawning
                # into a drain would re-advertise capacity the hub retired.
                logger.info(
                    "compute child died during drain; not respawning — flushing and exiting"
                )
                await self._finish_shutdown_flush(reason="death_during_drain")
                return
            if lifetime >= _BACKOFF_RESET_AFTER_ALIVE_S:
                backoff = self._backoff_base
            await self._sleep_or_stop(backoff)
            backoff = min(backoff * 2, self._backoff_cap)

    async def _sleep_or_stop(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), delay)
        except asyncio.TimeoutError:
            pass

    async def _settle_link(self) -> None:
        """Let the reaped child's buffered frames finish relaying, then close."""
        try:
            await asyncio.wait_for(self._link_closed.wait(), _LINK_SETTLE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "compute child link did not settle within %.1fs after exit; "
                "closing it (late frames may be lost)", _LINK_SETTLE_TIMEOUT_S,
            )
        link = self._link
        if link is not None:
            link.writer.close()
            self._link = None
            self._link_ready.clear()
            self._link_closed.set()

    async def _await_child_exit(self, timeout: float) -> bool:
        """TimeoutStopSec: wait for a deliberate child exit, then SIGKILL."""
        proc = self._proc
        if proc is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "compute child did not exit within %.0fs of shutdown "
                "(TimeoutStopSec) — SIGKILL", timeout,
            )
            await self._dial_detail(
                f"phase=compute_stop_timeout timeout_s={timeout:.0f} "
                f"spawns={self._spawn_count} — child SIGKILLed after a "
                "deliberate shutdown request"
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return False

    async def _stop_deadline(self) -> None:
        await self._await_child_exit(self._stop_timeout)

    async def _report_in_flight_dead(self, cause: str) -> Dict[Tuple[str, int], str]:
        """One typed FATAL per open job, into the DURABLE queue.

        Ships on the live stream now, or survives to the next one — either way
        the hub learns WHICH job died, not merely that a pod blinked.
        """
        died_jobs = dict(self._in_flight)
        self._in_flight.clear()
        try:
            for (rid, att), fn in sorted(died_jobs.items()):
                self._reported_dead[(rid, att)] = fn
                while len(self._reported_dead) > _REPORTED_DEAD_CAP:
                    self._reported_dead.popitem(last=False)
                await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                    request_id=rid,
                    attempt=att,
                    status=pb.JOB_STATUS_FATAL,
                    safe_message=(
                        f"{DEATH_LABEL}: cause={cause} function={fn or 'unknown'} "
                        f"(handler process died; worker alive and respawning)"
                    )[:512],
                )))
        finally:
            self._death_report_done.set()
        return died_jobs

    async def _finish_shutdown_flush(self, *, reason: str) -> None:
        """Bounded flush of the durable queue on a deliberate parent exit.

        Unretired results at this point can only be lost, so the loss is
        reported typed instead of vanishing with the process.
        """
        self._child_exited_clean = True   # the shutdown was deliberate, not a crash
        # With no live stream there is no send loop to retire anything, so
        # waiting the full budget would only delay the exit.
        flushed = await self.transport.close_after_flush(
            timeout=self._stop_flush_timeout if self.transport.connected else 1.0
        )
        self._reported_unretired = True
        pending = list(self.transport.queue.pending_result_keys)
        self.unretired_results_at_exit = len(pending)
        if not flushed or pending:
            await self._dial_detail(
                f"phase=compute_parent_exit reason={reason} flushed={flushed} "
                f"unretired_results={len(pending)} "
                f"keys={sorted(f'{r}#{a}' for (r, a) in pending)[:16]} "
                f"spawns={self._spawn_count}"
            )
        self._stopping.set()

    async def _finish_deliberate_exit(self, rc: int, *, lifetime_s: float) -> None:
        """The child left on purpose (drain, or a forwarded SIGTERM)."""
        self._child_exited_clean = True
        cause = f"exit:{rc}" if rc >= 0 else self._death_cause(rc, 0)[0]
        died = await self._report_in_flight_dead(cause)
        if died:
            # A deliberate exit with jobs still open is a should-never-happen
            # (drain finishes tenant work first) — report it, never swallow it.
            logger.error(
                "compute child exited deliberately (rc=%s) with %d job(s) still "
                "in flight; attributed typed", rc, len(died),
            )
            await self._dial_detail(postmortem.format_detail(
                phase="compute_process_exit",
                verdict={"signaled": rc < 0, "exit_code": rc if rc >= 0 else 128 - rc},
                limits=postmortem.container_limits(),
                oom_kill_delta=0,
                lifetime_s=lifetime_s,
                extra={"cause": cause, "deliberate": True,
                       "in_flight": sorted(f"{r}#{a}" for (r, a) in died)},
            ))
        else:
            logger.info("compute child exited cleanly (rc=%s); parent exiting", rc)
        postmortem.clear_inflight()   # nothing of this child may attribute the next
        await self._finish_shutdown_flush(
            reason="terminating" if self._terminating else "drain"
        )

    def _death_cause(self, rc: int, oom_delta: int) -> Tuple[str, Dict[str, Any]]:
        if self._watchdog_fired:
            return "watchdog_hang", {"signaled": True, "signal": signal.SIGKILL,
                                     "signal_name": "SIGKILL", "exit_code": 128 + signal.SIGKILL}
        if rc < 0:
            sig = -rc
            try:
                name = signal.Signals(sig).name
            except ValueError:
                name = f"SIG{sig}"
            verdict = {"signaled": True, "signal": sig, "signal_name": name,
                       "exit_code": 128 + sig}
            if sig == signal.SIGKILL and oom_delta > 0:
                return "oom", verdict
            return f"signal:{name}", verdict
        return f"exit:{rc}", {"signaled": False, "exit_code": rc}

    async def _handle_child_death(
        self, rc: int, *, oom_before: int, lifetime_s: float, saw_hello: bool,
    ) -> None:
        now = time.monotonic()
        oom_delta = max(0, postmortem.oom_kill_count() - oom_before)
        cause, verdict = self._death_cause(rc, oom_delta)
        self._death_times.append(now)
        if not saw_hello:
            self._deaths_before_hello += 1
        else:
            self._deaths_before_hello = 0

        # 1) Attribution first (durable, before any flush can conclude).
        died_jobs = await self._report_in_flight_dead(cause)
        logger.error(
            "compute child died: cause=%s rc=%s lifetime=%.1fs in_flight=%s "
            "(respawning; stream identity kept)",
            cause, rc, lifetime_s, sorted(r for r, _ in died_jobs) or "none",
        )

        # 2) Post-mortem dial (gw#640 typed exit capture, carried forward).
        # pgw#676/pgw#714 parity: a signal death CONSUMES the child's on-disk
        # in-flight markers, attaches the faulthandler tail, and records the
        # per-function native-crash streak the respawned child's own boot gate
        # refuses on. Skipping it would leave stale markers to misattribute the
        # next death and would silently disarm that gate in split mode.
        extra: Dict[str, Any] = {}
        if verdict.get("signaled"):
            try:
                extra.update(postmortem.attribute_signal_death(
                    signal_name=str(verdict.get("signal_name") or "")
                ))
            except Exception:
                logger.warning("signal-death attribution failed", exc_info=True)
        else:
            postmortem.clear_inflight()
        detail = postmortem.format_detail(
            phase="compute_process_exit",
            verdict=verdict,
            limits=postmortem.container_limits(),
            oom_kill_delta=oom_delta,
            lifetime_s=lifetime_s,
            extra={
                **extra,
                "cause": cause,
                "in_flight": sorted(f"{r}#{a}" for (r, a) in died_jobs),
                "spawn_count": self._spawn_count,
                "saw_hello": saw_hello,
            },
        )
        await self._dial_detail(detail)

        # 3) StartLimitBurst / StartLimitIntervalSec: DETECT the loop, report
        # it typed, and keep respawning (Paul: "infinite respawn is fine; the
        # worker can see if its children are crash-looping").
        recent = [t for t in self._death_times if now - t <= self._start_limit_interval]
        looping = len(recent) >= self._start_limit_burst or self._deaths_before_hello >= 2
        if looping and now - self._last_crash_loop_report_at >= _CRASH_LOOP_REPORT_MIN_INTERVAL_S:
            self._last_crash_loop_report_at = now
            self.crash_loop_reports += 1
            await self._dial_detail(
                f"phase=compute_crash_loop deaths={len(recent)} "
                f"window_s={self._start_limit_interval:.0f} "
                f"deaths_before_hello={self._deaths_before_hello} last_cause={cause} "
                f"spawns={self._spawn_count} — respawn continues; no serving Hello "
                "is advertised while the child is down"
            )

        # 4) Give the live stream a moment to ship the FATALs, then cycle the
        # connection so the respawned child re-syncs via a fresh Hello. (If
        # the stream is already down, the durable queue + Hello.in_flight
        # merge carry the attribution to the next connection.)
        try:
            await self.transport.queue.wait_empty(timeout=_DEATH_FLUSH_GRACE_S)
        except Exception:
            pass
        if self._draining or self._terminating or self._stopping.is_set():
            # No child will follow, so no fresh Hello is owed. Cycling here
            # would tear down the very stream the shutdown flush needs.
            return
        self.transport.cycle_connection()

    async def _dial_detail(self, detail: str) -> None:
        logger.error("compute.postmortem\n%s", detail)
        try:
            delivered = await asyncio.to_thread(
                worker_fatal.report_worker_detail, self._settings, detail
            )
            logger.info("compute post-mortem wire report delivered=%s", delivered)
        except Exception:
            logger.warning("compute post-mortem wire report failed", exc_info=True)

    # ---- watchdog (WatchdogSec) -----------------------------------------

    async def _watchdog_loop(self) -> None:
        """Missed beats ARM the verdict; the open activity DECIDES it.

        pgw#771 (th#1299 one layer down): the child's frame ping rides its event
        loop, and an inductor compile starves that loop. Killing on frame
        silence alone SIGKILLs a live compile and labels it ``watchdog_hang`` —
        strictly worse than the hub-side bug it was meant to catch, because no
        hub-side hold can rescue a child this parent already killed. So frame
        silence only ARMS: the decision is the child tree's kernel-accounted
        work, measured HERE from /proc (a child starved of the GIL cannot be a
        witness for itself), plus which activity the child last reported open.
        Tolerance is the activity clock — an open activity still accruing
        CPU/IO — never no clock, the same shape as the hub half's
        ActivityFreshnessWindow hold.

        The boundary between the two halves of "alive": this parent kills only
        what is provably NOT RUNNING. A child that runs but serves nothing is
        the hub's stall/activity clock to reap, exactly as it was before the
        split — so there is one verdict, not two that disagree.
        """
        interval = max(0.25, self._watchdog_budget / 4.0)
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            proc = self._proc
            if proc is None or self._link is None or self._relaying:
                continue
            if self._draining or self._terminating:
                # A deliberate teardown (unloading instances, final flush) is
                # allowed to be quiet; TimeoutStopSec bounds it instead. Killing
                # it here would fabricate a watchdog_hang out of a clean drain.
                continue
            now = time.monotonic()
            # Witness the child's accounted work on EVERY tick, not only once
            # the loop has gone quiet: the parent beat now keeps the pod
            # reachable, so the honest report of what the child is DOING is the
            # only signal left that distinguishes working from stalled. A
            # frozen child whose loop still turns would otherwise be invisible.
            self._sample_child_evidence(proc.pid, now)
            silent_for = now - self._last_frame_at
            if silent_for <= self._watchdog_budget:
                self._hang_armed_at = None
                continue
            if self._hang_armed_at is None:
                self._hang_armed_at = now
            # Only a child whose LOOP has gone silent can be called stalled.
            # Accrued work alone cannot carry that claim: an async wait
            # (marco-polo-slow's `await asyncio.sleep`, a throttled download,
            # any I/O-bound leg) burns no CPU and moves no disk by design —
            # this reported a perfectly healthy 15s job as stalled on the first
            # real-stack run. It is the same trap activity.note_progress exists
            # for. Loop silence ARMS, accrued work DECIDES: one ladder, used
            # for the report exactly as for the kill.
            await self._report_stall_if_any(now)
            verdict = self._hang_verdict(now)
            if verdict is None:
                continue
            if verdict == "held":
                if not self._hang_hold_reported:
                    self._hang_hold_reported = True
                    logger.warning(
                        "compute child loop silent for %.1fs but activity %r is "
                        "alive (evidence advanced %.1fs ago) — hang verdict HELD",
                        silent_for, self._liveness_activity,
                        now - self._liveness_evidence_at,
                    )
                    await self._dial_detail(
                        f"phase=compute_hang_verdict_held "
                        f"loop_silent_s={silent_for:.0f} "
                        f"activity={self._liveness_activity} "
                        f"evidence_age_s={now - self._liveness_evidence_at:.1f} "
                        f"evidence={self._liveness_evidence:.1f} "
                        f"ping_age_s={now - self._last_liveness_at:.1f} "
                        f"budget_s={self._watchdog_budget:.0f} — the child's event "
                        "loop is starved by accounted work, not hung; not killing"
                    )
                continue
            logger.error(
                "compute child silent for %.1fs (budget %.1fs, verdict=%s) — "
                "killing the wedged child (WatchdogSec analog)",
                silent_for, self._watchdog_budget, verdict,
            )
            self._watchdog_fired = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _child_evidence(self, pid: int) -> Optional[float]:
        """The child tree's kernel-accounted work: process+LIVE-children CPU
        seconds plus process disk I/O MB.

        The same combination ``activity._default_evidence`` already trusts —
        measured HERE, from /proc, because a child starved of the GIL (dynamo
        tracing) cannot be a witness for itself. Either source advancing proves
        genuine life: an inductor compile burns child CPU with the GPU idle and
        I/O flat; an on-disk model load moves bytes while CPU-light; a true hang
        advances neither.
        """
        try:
            import psutil
        except Exception:
            return None
        try:
            proc = psutil.Process(pid)
            times = proc.cpu_times()
            total = float(times.user + times.system)
            try:
                io = proc.io_counters()
                total += (io.read_bytes + io.write_bytes) / float(1 << 20)
            except (psutil.Error, AttributeError, NotImplementedError):
                pass
            for child in proc.children(recursive=True):
                try:
                    ct = child.cpu_times()
                except psutil.Error:
                    continue
                total += float(ct.user + ct.system)
            return total
        except psutil.Error:
            return None

    def _sample_child_evidence(self, pid: int, now: float) -> None:
        evidence = self._child_evidence(pid)
        if evidence is None:
            return
        previous = self._liveness_evidence
        if previous is None or evidence - previous >= _EVIDENCE_EPS:
            self._liveness_evidence = evidence
            self._liveness_evidence_at = now
            self._stall_reported = False

    async def _report_stall_if_any(self, now: float) -> None:
        """Say so when the child owes work and is accruing none.

        The parent's beat keeps the pod reachable, so silence is no longer how a
        stall reaches the hub — an honest report is. This is also the only claim
        about the child's progress that is not self-reported by the code being
        measured (the security driver: a value tenant code produces is a hint;
        a parent-side /proc measurement is evidence).

        Callers must have ARMED first (the child's loop is silent past the
        watchdog budget). A child that is still sending frames is waiting, not
        stalled, however little CPU it burns.
        """
        if self._liveness_evidence is None or self._stall_reported:
            return
        if not self._in_flight and not self._liveness_activity:
            # Nothing owed: an idle child legitimately accrues nothing, and
            # calling that a stall would be noise, not truth.
            return
        age = now - self._liveness_evidence_at
        if age <= self._evidence_hold_window:
            return
        self._stall_reported = True
        logger.warning(
            "compute child has accrued no CPU/IO for %.1fs while %d job(s) and "
            "activity %r are open — reporting the stall (stream and beat kept)",
            age, len(self._in_flight), self._liveness_activity,
        )
        await self._dial_detail(
            f"phase=compute_child_stalled evidence_age_s={age:.1f} "
            f"activity={self._liveness_activity or 'none'} "
            f"in_flight={sorted(f'{r}#{a}' for (r, a) in self._in_flight)} "
            f"loop_silent_s={now - self._last_frame_at:.1f} "
            f"window_s={self._evidence_hold_window:.0f} — measured by the parent "
            "from /proc, not self-reported by the child"
        )

    # ---- the app beat (pgw#771) ------------------------------------------

    async def _beat_loop(self) -> None:
        """The PARENT originates the app heartbeat.

        The hub's layer-2 reap counts StateDelta receipts: ~6 missed beats and
        the pod dies. Before this, the only sender was the child's
        ``_heartbeat_loop`` asyncio task, so a child whose loop is starved by
        compute stopped beating and the hub killed a live pod at ~60s — the
        split relayed the silence instead of curing it. Two claims, two
        channels, per WORKER-CONTRACTS.md §1:

        * THIS beat = "the worker is alive and reachable". The parent is the
          control plane — stream, queue, JWT, no torch — so nothing here can be
          starved, which makes it the truthful claimant of that question. What
          it re-sends is the last state the CHILD published: the child cannot
          change state without this relay carrying it, so an unchanged re-send
          is true by construction (and is the very shape the hub documents as
          proof the control loop lives).
        * Progress = a SEPARATE claim, the child's own activity/progress
          evidence, witnessed by the parent's /proc measurement. A wedged child
          keeps the pod alive on this beat while its progress goes quiet, and
          the hub's stall clock decides — no hub-side patience is assumed
          (th#1299's hold was reverted; an open activity buys ZERO tolerance),
          and no mint-specific keepalive is invented.
        """
        while not self._stopping.is_set():
            interval = self._beat_interval if self._beat_interval > 0 else 10.0
            await self._sleep_or_stop(max(0.25, interval / 2.0))
            if self._stopping.is_set() or self._child_exited_clean:
                return
            msg = self._last_state_delta
            if msg is None or self._link is None or not self.transport.connected:
                # Nothing published yet, or no child to speak for: the hub's
                # own machinery owns that gap. Never beat for an absent child.
                continue
            if time.monotonic() - self._last_state_delta_at < interval:
                continue  # the child is beating for itself
            self._last_state_delta_at = time.monotonic()
            self.parent_beats_sent += 1
            try:
                await self.transport.send(msg)
            except Exception:
                logger.debug("parent beat send failed", exc_info=True)

    def _hang_verdict(self, now: float) -> Optional[str]:
        """``None`` = no decision yet, ``"held"`` = alive-but-starved,
        otherwise the reason the child is being killed."""
        if self._liveness_evidence is None:
            # No evidence source at all (/proc unreadable, psutil missing): the
            # only signal left is loop silence, so fall back to it rather than
            # never reaping a genuinely wedged child.
            return "no_evidence_source"
        if now - self._liveness_evidence_at > self._evidence_hold_window:
            # The child tree has stopped accruing CPU and I/O: SIGSTOP, a kernel
            # wedge, a native deadlock. This is the real hang — and it is the
            # same non-advancement activity.watchdog stops heartbeating on.
            return "no_work_accrued"
        if not self._liveness_activity:
            # Burning CPU with nothing open to justify it (a bare `while True`
            # in a handler): alive as a process, useless as a worker, and no
            # clock would ever end it.
            return "loop_wedged_no_activity"
        return "held"

    # ---- drain / signals -------------------------------------------------

    async def _drain_without_child(self) -> None:
        # A child reaped moments ago may still be attributing its in-flight
        # jobs. Flushing before that FATAL is enqueued would retire the queue
        # empty and drop the report with the process.
        try:
            await asyncio.wait_for(
                self._death_report_done.wait(), _LINK_SETTLE_TIMEOUT_S + 2.0
            )
        except asyncio.TimeoutError:
            logger.warning("draining without waiting further for death attribution")
        await self._finish_shutdown_flush(reason="drain_without_child")

    def _forward_signal(self, signum: int) -> None:
        proc = self._proc
        if proc is not None:
            # Mark intent BEFORE the signal lands: the child's death by this
            # signal is deliberate, so it must not respawn, must not count
            # toward the crash-loop window, and must not exit the parent 1.
            self._terminating = True
            if self._stop_deadline_task is None:
                self._stop_deadline_task = asyncio.create_task(
                    self._stop_deadline(), name="parent-stop-deadline"
                )
            try:
                proc.send_signal(signum)
                return
            except ProcessLookupError:
                pass
        # No child to drain: flush and stop.
        self._draining = True
        asyncio.create_task(self._drain_without_child(), name="signal-drain")

    # ---- run -------------------------------------------------------------

    def stop(self) -> None:
        """Thread-safe stop (tests / embedding)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        def _stop() -> None:
            self._stopping.set()
            self.transport.stop()
            proc = self._proc
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        loop.call_soon_threadsafe(_stop)

    async def arun(self) -> int:
        self._loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._loop.add_signal_handler(sig, self._forward_signal, sig)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass
        server = await asyncio.start_unix_server(
            self._on_child_connect, path=self._socket_path
        )
        # delta 2: measure the host BEFORE any endpoint import can have
        # happened — the first child has not been spawned yet.
        self._measure_task = asyncio.create_task(
            self._measure_host(), name="parent-measure"
        )
        transport_task = asyncio.create_task(self.transport.run(), name="parent-transport")
        child_task = asyncio.create_task(self._child_loop(), name="parent-child-loop")
        watchdog_task = asyncio.create_task(self._watchdog_loop(), name="parent-watchdog")
        beat_task = asyncio.create_task(self._beat_loop(), name="parent-beat")
        try:
            done, _ = await asyncio.wait(
                (transport_task, child_task), return_when=asyncio.FIRST_COMPLETED
            )
            if transport_task in done:
                transport_task.result()  # re-raise FatalTransportError
                # The stream ended first. On a DELIBERATE shutdown the child
                # still owns the rest of it (instance teardown, last telemetry,
                # its own exit code) — waiting for it is what makes a drain
                # exit 0 instead of SIGKILLing a cleanly draining child and
                # reporting 1.
                if (self._draining or self._terminating) and not child_task.done():
                    await self._await_child_exit(self._stop_timeout)
                    try:
                        await asyncio.wait_for(asyncio.shield(child_task), 15.0)
                    except asyncio.TimeoutError:
                        logger.warning("child supervision loop did not settle after drain")
                    except Exception:
                        pass
            else:
                # The supervision loop finished first — a deliberate exit, and
                # it has already retired the queue through the send loop. The
                # transport is now inside its half-close (done_writing + wait
                # for the peer to end the call): CANCELLING it here RSTs the
                # call and discards writes the queue already retired, which is
                # how a typed death FATAL gets lost after a drain. Let it end.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(transport_task), _CLEAN_CLOSE_WAIT_S
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "transport did not finish its clean close within %.0fs",
                        _CLEAN_CLOSE_WAIT_S,
                    )
                except Exception:
                    pass  # re-raised (or already logged) by the gather below
            return 0 if (
                self._child_exited_clean
                or self._draining
                or self._terminating
                or self._stopping.is_set()
            ) else 1
        finally:
            if not self._reported_unretired:
                pending = list(self.transport.queue.pending_result_keys)
                self.unretired_results_at_exit = len(pending)
                if pending:
                    await self._dial_detail(
                        f"phase=compute_parent_exit reason=abrupt "
                        f"unretired_results={len(pending)} "
                        f"keys={sorted(f'{r}#{a}' for (r, a) in pending)[:16]} "
                        f"spawns={self._spawn_count}"
                    )
            self._stopping.set()
            proc = self._proc
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            self.transport.stop()
            tasks = [transport_task, child_task, watchdog_task, beat_task]
            for extra in (self._stop_deadline_task, self._liveness_task,
                          self._measure_task):
                if extra is not None:
                    tasks.append(extra)
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server.close()
            try:
                # py3.12 waits for handler completion; never hang shutdown on it.
                await asyncio.wait_for(server.wait_closed(), 5.0)
            except asyncio.TimeoutError:
                pass
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass

    def run(self) -> int:
        try:
            return asyncio.run(self.arun())
        except FatalTransportError as exc:
            logger.error("control parent exiting on a fatal: %s", exc, exc_info=True)
            try:
                worker_fatal.report_worker_fatal(
                    self._settings, "parent_run_loop", exc, exit_code=1
                )
            except Exception:
                logger.warning("parent fatal wire report failed", exc_info=True)
            return 1


def run_parent() -> int:
    """Production entry (called from entrypoint BEFORE any heavy import)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Carry forward the gw#640 previous-container-death report + boot record.
    from ..supervisor import report_previous_container_death

    report_previous_container_death()
    postmortem.clear_inflight()
    postmortem.write_boot_record()
    from ..config import get_settings

    settings = get_settings()
    code = ParentControl(settings).run()
    if code == 0:
        postmortem.clear_boot_record()
        postmortem.clear_inflight()
    return code
