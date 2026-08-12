"""Parent (control-plane) side of the pgw#763 split, generalised to N execution
groups (pgw#783).

Owns: the ONE gRPC stream + identity/JWT (the real ``Transport``), the durable
SendQueue, the parent-side security boundary (deltas 0-5: JWT never in a child,
mediated hub actions, parent-measured hardware, billing attestation, per-job
capability decisions), and the supervision of a GROUP of compute children — one
``_ChildSlot`` per execution group. Never imports torch.

pgw#782 measured why there is one child PER GROUP and not one child total: four
groups in ONE interpreter serve 0.94x of serial (21% per card); four PROCESSES
one group each serve **4.00x** at 91-93%. So the child of the split is the
EXECUTION GROUP, and this parent supervises G of them.

**At G == 1 this is byte-identical to the single-child parent the security
deltas shipped.** Every worker-level aggregation point (Hello assembly, the
frame relay, the dispatch rewrite) takes an explicit ``groups == 1`` fast path
that runs the original code verbatim; the per-child supervision machinery
(``_ChildSlot``) is the same code for one child or four. That identity is the
whole safety property and it is a regression test (``test_group_processes``).

Supervision primitives are deliberately systemd's (Paul, 2026-07-29):
``Restart=on-failure``, ``StartLimitBurst``/``StartLimitIntervalSec``,
``WatchdogSec``+``sd_notify`` (child frames are the liveness
pings; loop silence ARMS a hang verdict the child's accounted work DECIDES —
pgw#771), socket activation (the hub connection outlives the process doing the
work). What systemd cannot do — job attribution — needs the stream, the JWT and
the in-flight table in the survivor, which is this parent. At G>1 one child's
death is attributed to ITS request and respawns ITS group; siblings never see
it (**a group where one of four children dies is not a dead group**).

pgw#826 — WHICH DEATHS ARE RETRYABLE. Respawn is for a child that PROVED it
can boot (reached Hello): its death may be payload-driven, so it respawns with
backoff and post-Hello loops stay typed DETECTION (the hub's liveness/stall
clocks own a sick-but-serving pod). A child that dies before Hello has served
nothing and can owe nothing: after ``boot_death_limit`` consecutive pre-Hello
deaths — or ONE terminal typed boot verdict (``T_BOOT_FATAL``, e.g. a
HardwareUnsuitable CUDA-probe refusal, true for every child this pod could
ever spawn) — the parent propagates the report and exits 1. Before this, a
hardware-unsuitable pod crash-looped its child forever: every respawn emitted
output, so no silence window (pgw#795) could bound it, and the pod billed
indefinitely while the hub never saw the typed refusal.
"""

from __future__ import annotations

import asyncio
import collections
import faulthandler
import json
import logging
import os
import shlex
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec

from ..config import Settings
from ..pb import worker_scheduler_pb2 as pb
from ..transport import FatalTransportError, Transport
from ..topology import ExecutionTopology
from .. import postmortem
from .. import proc_evidence
from .. import config, worker_credential, worker_identity
from .. import worker_fatal
from . import (
    ENV_CHILD,
    ENV_CHILD_CMD,
    ENV_LIVENESS_FD,
    ENV_SESSION_ID,
    ENV_SOCKET,
    ENV_WATCHDOG_PING_S,
    actions,
    attest,
    capability,
    frames,
    merge,
    privdrop,
)
from .group import ChildGroup, GroupPlan
from .seam import SeamAccountant

logger = logging.getLogger(__name__)

# The typed death label. Deliberately NOT in the hub's th#1288
# declaredFaultLabels allowlist: a child death can be payload-driven (an OOM
# this payload caused), so it must not classify as release-declared evidence.
# The hub's per-request blame-probe ladder handles it correctly as a FATAL.
DEATH_LABEL = "ComputeProcessDied"

_DEFAULT_START_LIMIT_BURST = 3          # StartLimitBurst
_DEFAULT_START_LIMIT_INTERVAL_S = 600.0  # StartLimitIntervalSec
# pgw#826: consecutive pre-Hello deaths before the parent gives up and exits 1.
_DEFAULT_BOOT_DEATH_LIMIT = 3
_DEFAULT_WATCHDOG_BUDGET_S = 60.0        # WatchdogSec (matches th#965's reap budget)
_DEFAULT_RESPAWN_BACKOFF_BASE_S = 1.0
_DEFAULT_RESPAWN_BACKOFF_CAP_S = 60.0
_BACKOFF_RESET_AFTER_ALIVE_S = 60.0
_CRASH_LOOP_REPORT_MIN_INTERVAL_S = 300.0
# "no report has ever been sent" — NOT 0.0. Every throttle below compares
# against time.monotonic(), which on Linux is time since BOOT, so a 0.0
# sentinel means "reported at boot" and silently swallows the FIRST report of
# each class on any host whose uptime is under the interval. That is every
# freshly-started worker pod, and the first five minutes is exactly when a
# crash loop or an allowlist probe most needs to reach the hub.
_NEVER_REPORTED = float("-inf")
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
# noticeboard. WORKER_JWT is the signing identity; RUNPOD_API_KEY is injected by
# RunPod into every pod and th#1380 verified it is ACCOUNT-scoped in authority
# (it enumerates our fleet, reads our balance, lists 90 registry credential
# records) and cannot be suppressed at the create call; PUBLIC_KEY is our
# operator SSH key. HF_TOKEN is the endpoint author's OWN credential and
# legitimately belongs to the code that pulls weights, so it deliberately stays.
#
# This list only became load-bearing with pgw#858: until the child ran as its
# own uid, deleting a name here was cosmetic — tenant code read the same value
# out of `/proc/<ppid>/environ` (WORKER_JWT) or `/proc/1/environ` (the RunPod
# key). Both, and the strip, are guarded by test_pod_privilege_isolation_pgw858.
_CHILD_FORBIDDEN_ENVS = ("WORKER_JWT", "RUNPOD_API_KEY", "PUBLIC_KEY")
_ACTION_REFUSAL_REPORT_MIN_INTERVAL_S = 300.0
# A mediated hub call runs on a parent thread-pool slot, and the CHILD supplies
# the request's `timeout` field — so without a ceiling tenant code pins slots
# for as long as it likes and the mediation surface dies for everything else.
# `HubAction.timeout_s` in the allowlist IS that ceiling (`_perform_action`
# min()s the child's number against it), which is why pgw#973 deleted the
# separate `_ACTION_HARD_TIMEOUT_S = 120.0` that sat beside it: every declared
# action is 30 s or 60 s, so a third term 60 s above the highest declared value
# could only ever reject nothing (§4.24 item 1 — say which bound is
# load-bearing and delete the rest). The count axis is bounded here:
_MAX_CONCURRENT_ACTIONS = 16
# pgw#973 (§4.24 item 4): the parent beat's cadence when NOBODY declared one —
# `beat_interval_s=0.0` means "adopt the child's" and every child Hello may
# carry `heartbeat_interval_ms=0`. The fallback used to be a bare `10.0` at the
# loop, i.e. a real bound reachable only by reading the loop body. It is the
# hub's own liveness expectation (a worker silent for a multiple of this is
# called dead), so it is a DECLARED default, not a guess.
_BEAT_INTERVAL_FALLBACK_S = 10.0
# The host canary is a real benchmark (memcpy/D2H/CPU); on a cold pod with a
# large card it is seconds, not milliseconds.
#
# gw#666 exemption, stated rather than assumed (§4.24): the measure subprocess
# reports through `communicate()` — one write, at the end — so it emits NO
# progress signal a SilenceWindow could key on, and giving up here KILLS NO
# WORK: the Hello ships without parent-measured resources and the pod serves.
# A progress signal would have to be invented in the canary's own protocol,
# which is a change to it and not to this bound.
_MEASURE_TIMEOUT_S = 180.0
_MEASURE_BEFORE_SPAWN_S = 60.0
_ATTESTATION_REPORT_MIN_INTERVAL_S = 300.0
_CAPABILITY_REPORT_MIN_INTERVAL_S = 300.0
# Bounded: an observation is dropped when its result passes back OR when the
# death path attributes its job (pgw#937), so nothing but a live job holds one.
_OBSERVATION_CAP = 512
# The WorkerMessage kinds the fan-in reconciles across groups into one worker
# view. Everything else is per-request or per-object and forwards verbatim.
# Only a LIVE group may contribute one of these (see "down-group semantics").
_WORKER_SCOPED_MSGS = frozenset(
    {"state_delta", "activity_update", "fn_unavailable", "fn_degraded"}
)
# pgw#833: the child's stderr is captured by the parent (teed through to the
# container log unchanged) so a pre-Hello death carries its OWN crash text in
# the post-mortem dial. RunPod exposes no container-logs API (gw#640), so a
# child that dies before it can dial — and it never can, it holds no JWT —
# was a bare `exit:1` on the wire; diagnosing pgw#833 took three paid probe
# pods for want of these bytes. Ring-buffered, bounded, reset per spawn.
_STDERR_TAIL_CAP_BYTES = 32768
_STDERR_TAIL_DIAL_CHARS = 3000
# pgw#858: the compute uid's home — HF cache, ~/.triton, ~/.nv, TMPDIR and the
# .pyc prefix all hang off it. On disk, not in the world-writable /tmp, and
# owned by the compute uid rather than shared with the control parent.
_COMPUTE_HOME = "/var/lib/gen-worker/compute"
# models/cache_paths.py's and runtime_config.py's defaults, duplicated rather
# than imported: importing either package pulls the model layer, and the parent
# never imports torch.
_DEFAULT_TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"
_DEFAULT_CONFIG_SNAPSHOT_PATH = "/app/.tensorhub/runtime_config.msgpack"


def _tee_stderr_chunk(chunk: bytes) -> None:
    """Blocking tee of one child-stderr chunk to the parent's stderr.

    Runs in a worker thread (asyncio.to_thread) — never on the event loop."""
    try:
        buf = getattr(sys.stderr, "buffer", None)
        if buf is not None:
            buf.write(chunk)
            buf.flush()
        else:
            sys.stderr.write(chunk.decode("utf-8", errors="replace"))
            sys.stderr.flush()
    except Exception:
        pass  # the ring still keeps the bytes for the dial


def _close_transport(proc: asyncio.subprocess.Process) -> None:
    """Tear a reaped child's transport down while its loop is still alive.

    pgw#833 gave the child a stderr PIPE, which gave BaseSubprocessTransport a
    pipe protocol to close in ``__del__``. If GC reaches it after the loop has
    closed, ``__del__`` calls ``call_soon`` on a dead loop and raises
    "RuntimeError: Event loop is closed" as an unraisable — cosmetic, but it
    surfaced in CI's warnings summary against unrelated passing tests. Closing
    it here, with the child already reaped, removes the GC race entirely.
    """
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except Exception:
        pass


def _http_call(
    method: str,
    url: str,
    token: str,
    query: Dict[str, Any],
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


# pgw#783: PR_SET_PDEATHSIG — make every compute child die with the parent.
# pgw#858 moved the implementation into privdrop, which owns the whole
# post-fork/pre-exec sequence: the parent-death signal must be re-established
# AFTER the credential change, not before, so the two cannot be set in the
# wrong order by accident. This name stays as the module's export of it.
_set_pdeathsig = privdrop.set_pdeathsig


class _ChildLink:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = frames.FrameWriter(writer)
        self.saw_hello = False


class _ChildSlot:
    """One execution group's compute child, plus every per-child supervision
    fact. At G == 1 there is exactly one slot and every field here was a
    ``ParentControl`` field before pgw#783 — the single-slot path is unchanged.

    A slot is a CONTAINMENT boundary (delta security): the child holds no worker
    JWT, no identity credential, no signing key. It owns its group's cards
    (``CUDA_VISIBLE_DEVICES`` in ``group.env``), its own CUDA context, its own
    inductor cache, its own mint. A sibling's death cannot reach it.
    """

    def __init__(self, parent: "ParentControl", group: ChildGroup) -> None:
        self.p = parent
        self.ordinal = group.ordinal
        self.devices = group.devices
        self.socket_path = group.socket_path
        # The per-group env DELTA (empty at G==1): CUDA_VISIBLE_DEVICES, the
        # DxD-rewritten topology, the sibling count. Applied over the parent's
        # shared child env at spawn.
        self.group_env: Dict[str, str] = dict(group.env)
        self.inflight_marker_path = (
            postmortem.group_inflight_path(group.ordinal, parent._postmortem_dir)
            if parent._topology.execution_groups > 1 else None
        )
        self.fault_dump_path = (
            postmortem.group_fault_dump_path(group.ordinal, parent._postmortem_dir)
            if parent._topology.execution_groups > 1 else None
        )
        # pgw#1041: the load path's breadcrumb — consumed on a signal death
        # so a SIGKILL mid-load names its phase/component and byte counts.
        self.load_progress_path = (
            postmortem.group_load_progress_path(
                group.ordinal, parent._postmortem_dir)
            if parent._topology.execution_groups > 1 else None
        )

        self.server: Optional[asyncio.AbstractServer] = None
        self.link: Optional[_ChildLink] = None
        self.link_ready = asyncio.Event()
        self.hello_waiter: Optional[asyncio.Future] = None
        self.proc: Optional[asyncio.subprocess.Process] = None
        # (request_id, attempt) -> function_name for THIS group's jobs, set at
        # RunJob relay, cleared when the child's JobResult passes back.
        self.in_flight: Dict[Tuple[str, int], str] = {}
        self.death_times: collections.deque = collections.deque(maxlen=64)
        self.deaths_before_hello = 0
        self.child_saw_hello = False
        # pgw#826: a terminal typed boot verdict from the dying child.
        self.boot_fatal: Optional[Dict[str, Any]] = None
        # pgw#833: ring buffer of THIS child's most recent stderr bytes.
        self.stderr_tail: collections.deque = collections.deque()
        self.stderr_tail_len = 0
        self.stderr_task: Optional[asyncio.Task] = None
        self.spawn_count = 0
        self.last_frame_at = time.monotonic()
        self.relaying = False
        self.watchdog_fired = False
        # pgw#771 liveness (thread-sourced, loop-independent), per child.
        self.liveness_task: Optional[asyncio.Task] = None
        self.last_liveness_at = 0.0
        self.liveness_evidence: Optional[float] = None
        self.liveness_evidence_at = 0.0
        self.liveness_activity = ""
        self.hang_armed_at: Optional[float] = None
        self.hang_hold_reported = False
        self.stall_reported = False
        # This group's freshest published StateDelta; the worker-level beat
        # re-sends the merge of all groups'.
        self.last_state_delta: Optional[pb.WorkerMessage] = None
        self.last_state_delta_at = 0.0
        # pgw#937 / DESIGN-RULINGS §4.15: a respawn is a NEW GENERATION of this
        # same group. `generation` counts the incarnations that have spoken; it
        # is stamped into the retirement dial so a fact attributed to the wrong
        # incarnation is visible rather than inferred.
        self.generation = 0
        # PARTICIPATION is the fan-in's liveness predicate: do this group's
        # facts count as the worker's? It goes False when an incarnation stops
        # being able to speak and True again when the next one connects. A group
        # that has not spoken YET starts True — it has no facts to be wrong
        # about, and its absence from the dicts is the live-group default. The
        # defect is only ever a group that HAS spoken and then died. See
        # "down-group semantics" in the fan-in section.
        self.participating = True
        self.last_crash_loop_report_at = _NEVER_REPORTED
        # Set once the link read loop has finished (EOF drained), so death
        # attribution never races the child's last frames.
        self.link_closed = asyncio.Event()
        self.link_closed.set()
        # CLEAR from the moment the child is reaped until its in-flight jobs are
        # attributed into the durable queue: a concurrent drain flush must not
        # declare the queue empty before the death report is in it.
        self.death_report_done = asyncio.Event()
        self.death_report_done.set()
        # (request_id, attempt) already terminal-reported by the death path.
        self.reported_dead: collections.OrderedDict = collections.OrderedDict()

    @property
    def label(self) -> str:
        return f"g{self.ordinal}"

    def begin_generation(self) -> None:
        """A NEW incarnation of this group starts speaking (pgw#937, §4.15).

        It enters the merge with EMPTY fan-in state — the death path retired the
        previous generation's — so "absent" once again means the live-group
        default and never a dead incarnation's last frame.
        """
        self.generation += 1
        self.participating = True

    # ---- unix socket server (one per group) ------------------------------

    async def start_server(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        self.server = await asyncio.start_unix_server(
            self._on_child_connect, path=self.socket_path
        )
        # pgw#858: connecting to a unix socket needs WRITE on its inode, so a
        # root-created socket under the default umask is unreachable by the
        # compute child's uid. Handing it to that uid at 0600 is also strictly
        # tighter than the 0755 the split shipped with.
        if self.p._drop_plan is not None:
            privdrop.grant_socket(self.p._drop_plan, self.socket_path)

    async def close_server(self) -> None:
        if self.server is None:
            return
        self.server.close()
        try:
            await asyncio.wait_for(self.server.wait_closed(), 5.0)
        except asyncio.TimeoutError:
            pass
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

    async def _on_child_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        old = self.link
        link = _ChildLink(reader, writer)
        self.link = link
        if old is not None:
            old.writer.close()
        self.last_frame_at = time.monotonic()
        self.link_closed.clear()
        self.link_ready.set()
        self.begin_generation()
        logger.info("compute child %s connected on %s", self.label, self.socket_path)
        # pgw#783: at G>1 a RESPAWNED child (not the first boot) comes up empty
        # and needs the hub to re-drive its desired residency. The death path
        # deliberately did NOT cycle the shared stream (siblings kept serving);
        # now that this group's link is back and every slot is connected, cycle
        # to re-sync the whole worker via a fresh, re-aggregated Hello. The
        # hub's reconcile is idempotent, so the siblings are undisturbed. (At
        # G==1 the death path already cycled — never double-cycle here.)
        if (self.p.execution_groups > 1 and self.spawn_count > 1
                and not (self.p._draining or self.p._terminating
                         or self.p._stopping.is_set())):
            try:
                self.p.transport.cycle_connection()
            except Exception:
                logger.debug("re-sync cycle on %s reconnect failed", self.label,
                             exc_info=True)
        try:
            while True:
                ftype, payload = await frames.read_frame(reader)
                self.last_frame_at = time.monotonic()
                await self._on_child_frame(link, ftype, payload)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            if self.link is link:
                self.link = None
                self.link_ready.clear()
                # pgw#937: participation ends the moment this generation stops
                # being able to speak, not when the OS finally reaps it. A group
                # the parent already answers RETRYABLE for must not still be
                # voting in the worker's merged view. (`self.link is link` keeps
                # a superseded link's teardown from retiring the live one.)
                self.participating = False
                self.p._note_state_delta()
            waiter = self.hello_waiter
            if waiter is not None and not waiter.done():
                waiter.set_exception(ConnectionError("compute child link lost"))
            link.writer.close()
            self.link_closed.set()

    async def _on_child_frame(self, link: _ChildLink, ftype: int, payload: bytes) -> None:
        if ftype == frames.T_WATCHDOG:
            return  # the timestamp update in the read loop IS the handling
        if ftype == frames.T_HELLO:
            link.saw_hello = True
            self.child_saw_hello = True
            waiter = self.hello_waiter
            if waiter is not None and not waiter.done():
                waiter.set_result(payload)
            return
        if ftype == frames.T_WORKER_MSG:
            msg = pb.WorkerMessage.FromString(payload)
            which = msg.WhichOneof("msg")
            if which == "job_result":
                r = msg.job_result
                self.in_flight.pop((r.request_id, r.attempt), None)
                # delta 3: the billable numbers pass through the one component
                # that watched this job from outside the process doing it.
                await self.p._attest_result(r, self)
            elif which == "state_delta" and self.participating:
                # pgw#937: a retired generation's late frame is a dead process's
                # claim about a worker it has left. Recording it would put the
                # group back into the merge without a live child behind it.
                self.last_state_delta = msg
                self.last_state_delta_at = time.monotonic()
                self.p._note_state_delta()
            # SendQueue.put can backpressure (stream down, event lane full);
            # while the READ LOOP is blocked here the child's pings cannot be
            # read, so the watchdog must not mistake parent-side backpressure
            # for a wedged child.
            self.relaying = True
            try:
                out = self.p._fan_in(self, msg)
                if out is not None:
                    # Account the relayed frame: the control-not-data invariant
                    # is that job payload never crosses the parent. `len(payload)`
                    # is the child's serialized WorkerMessage — the exact bytes
                    # that crossed the seam.
                    self.p.seam.record(which or "", len(payload), group=self.ordinal)
                    await self.p.transport.send(out)
            finally:
                self.relaying = False
                self.last_frame_at = time.monotonic()
            return
        if ftype == frames.T_ACTION_REQ:
            # Off the read loop: a mediated call is a network round trip, and
            # blocking here would stop the child's frames from being read.
            asyncio.create_task(
                self.p._serve_action(link, frames.unpack_meta(payload)),
                name=f"parent-action-{self.label}",
            )
            return
        if ftype == frames.T_PREPEND:
            msgs = [pb.WorkerMessage.FromString(b) for b in frames.unpack_meta(payload)]
            await self.p.transport.prepend_reconnect(msgs)
            return
        if ftype == frames.T_BOOT_FATAL:
            # pgw#826: consumed by child_loop after the child is reaped.
            self.boot_fatal = frames.unpack_meta(payload)
            report = (self.boot_fatal or {}).get("report") or {}
            logger.error(
                "compute child %s reported a TERMINAL boot verdict: kind=%s "
                "reason_class=%s", self.label,
                (self.boot_fatal or {}).get("kind"), report.get("reason_class"),
            )
            # pgw#833 (the pgw#826 follow-on race): ack AFTER the verdict is
            # recorded, so a child that waits for this ack can only exit once
            # the parent has consumed the frame — the respawn decision can no
            # longer race the socket buffer on a slow host.
            try:
                await link.writer.frame(frames.T_BOOT_FATAL_ACK, frames.pack_meta({}))
            except Exception:
                logger.debug("boot-fatal ack write failed (child may already "
                             "be gone)", exc_info=True)
            return
        if ftype == frames.T_FLUSH_REQ:
            meta = frames.unpack_meta(payload)
            timeout = meta.get("timeout")
            # The child asks for this at the END of its own drain, so the
            # shutdown is deliberate however it was triggered.
            self.p._draining = True
            flushed = await self.p.transport.close_after_flush(
                timeout=None if timeout is None else float(timeout)
            )
            try:
                await link.writer.frame(
                    frames.T_FLUSH_ACK, frames.pack_meta({"flushed": bool(flushed)})
                )
            except Exception:
                pass
            return
        logger.warning("unknown child frame type %d ignored (%s)", ftype, self.label)

    # ---- child lifetime --------------------------------------------------

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        env.update(self.p._child_env)
        # pgw#783: the per-GROUP env delta — CUDA_VISIBLE_DEVICES scoping the
        # child to its own cards, the DxD topology rewrite (locally G==1), the
        # sibling count. Empty at G==1.
        env.update(self.group_env)
        # delta 1: the compute child gets NO signing identity. Deleting the
        # T_TOKEN frame is only half of it — the JWT also arrives at pod-launch
        # in WORKER_JWT, and `os.environ` is the first place tenant code looks.
        for name in _CHILD_FORBIDDEN_ENVS:
            env.pop(name, None)
        # pgw#858: the uid the child is about to exec as has no account of its
        # own to inherit these from, and `~`, getpass.getuser(), TMPDIR and the
        # .pyc path all resolve through them.
        if self.p._drop_plan is not None:
            env.update(privdrop.child_env(self.p._drop_plan))
        # ...but the child still needs its IDENTITY, which is not a credential.
        worker_id, release_id = self.p._identity()
        if worker_id:
            env["WORKER_ID"] = worker_id
        if release_id:
            env["WORKER_RELEASE_ID"] = release_id
        env[ENV_CHILD] = "1"
        env[ENV_SOCKET] = self.socket_path
        # pgw#783: the parent's stable session id, so a respawned child keeps the
        # worker's shadow-state session instead of minting a fresh uuid the hub
        # would reject.
        env[ENV_SESSION_ID] = self.p._worker_session_id
        # The gw#640 flight-recorder fork is redundant under this parent.
        env["GEN_WORKER_SUPERVISOR"] = "0"
        # pgw#771: a dedicated pipe for THREAD-sourced process liveness.
        read_fd, write_fd = os.pipe()
        env[ENV_LIVENESS_FD] = str(write_fd)
        self.spawn_count += 1
        plan = self.p._drop_plan
        logger.info(
            "spawning compute child %s #%d (devices=%s, as=%s): %s",
            self.label, self.spawn_count, ",".join(str(d) for d in self.devices),
            plan.describe() if plan is not None else f"uid {os.geteuid()} (no drop)",
            " ".join(self.p._child_cmd),
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.p._child_cmd, env=env, pass_fds=(write_fd,),
                # pgw#833: capture the child's stderr so its death carries its
                # own last words in the post-mortem dial. The pump tees every
                # byte straight back to the parent's stderr, so the container
                # log (and the pgw#639 SIGUSR2 stack dumps) are unchanged.
                stderr=asyncio.subprocess.PIPE,
                # pgw#858 + pgw#783, in that order and in ONE hook: drop to the
                # unprivileged compute uid, prove the drop took, then re-arm
                # PR_SET_PDEATHSIG so a crashed group never strands its VRAM as
                # an orphaned torch process. Post-fork/pre-exec, so tenant code
                # has never run in this process when the credential changes.
                preexec_fn=(
                    privdrop.preexec(plan) if sys.platform == "linux" else None
                ),
            )
        finally:
            os.close(write_fd)   # the child owns it now
        self._start_stderr_pump(proc)
        await self._start_liveness_reader(read_fd)
        return proc

    # ---- child stderr capture (pgw#833) ----------------------------------

    def _start_stderr_pump(self, proc: asyncio.subprocess.Process) -> None:
        old = self.stderr_task
        if old is not None and not old.done():
            old.cancel()
        self.stderr_tail.clear()
        self.stderr_tail_len = 0
        self.stderr_task = asyncio.create_task(
            self._stderr_pump(proc), name=f"parent-stderr-{self.label}"
        )

    async def _stderr_pump(self, proc: asyncio.subprocess.Process) -> None:
        """Tee the child's stderr to the parent's own (the container log keeps
        every byte, exactly as before the pipe) while ring-buffering the tail
        for the death dial. Reading continuously also keeps the pipe from ever
        backpressuring the child."""
        stream = proc.stderr
        if stream is None:
            return
        while True:
            try:
                chunk = await stream.read(8192)
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception:
                return
            if not chunk:
                return
            # The tee write happens OFF the event loop: the parent's own
            # stderr can be a pipe whose consumer stalls (pytest capture, a
            # throttled container-log collector), and a blocking flush() on
            # the loop thread freezes signal handling and the shutdown path —
            # measured as test_sigterm_is_forwarded_to_the_worker timing out
            # on a contended host (2-core repro; CI 2/2).
            await asyncio.to_thread(_tee_stderr_chunk, chunk)
            self.stderr_tail.append(chunk)
            self.stderr_tail_len += len(chunk)
            while (self.stderr_tail_len > _STDERR_TAIL_CAP_BYTES
                    and len(self.stderr_tail) > 1):
                dropped = self.stderr_tail.popleft()
                self.stderr_tail_len -= len(dropped)

    def stderr_tail_text(self, max_chars: int = _STDERR_TAIL_DIAL_CHARS) -> str:
        raw = b"".join(self.stderr_tail)
        if not raw:
            return ""
        text = raw.decode("utf-8", errors="replace")
        return text[-max_chars:]

    async def _start_liveness_reader(self, read_fd: int) -> None:
        old = self.liveness_task
        if old is not None and not old.done():
            old.cancel()
        self.last_liveness_at = 0.0
        self.liveness_evidence = None
        self.liveness_evidence_at = time.monotonic()
        self.liveness_activity = ""
        self.liveness_task = asyncio.create_task(
            self._liveness_loop(read_fd), name=f"parent-liveness-{self.label}"
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
                self.last_liveness_at = time.monotonic()
                # Sticky: the flag is a fact about the last time the child could
                # speak. A GIL-starved thread stops speaking; the activity it
                # last reported is still the one running.
                self.liveness_activity = (
                    str(meta.get("kind") or "") if meta.get("act") else ""
                )
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            transport.close()

    async def child_loop(self) -> None:
        """Supervise THIS group's child: spawn, wait, attribute, respawn. Runs
        forever until a worker-level shutdown sets ``_stopping`` or this group's
        child exits deliberately (which, since a child exits cleanly only on a
        worker drain/terminate, sets ``_stopping`` and drains all slots)."""
        p = self.p
        backoff = p._backoff_base
        # delta 2: let the host measurement finish before ANY child exists.
        try:
            await asyncio.wait_for(p._measured.wait(), _MEASURE_BEFORE_SPAWN_S)
        except asyncio.TimeoutError:
            logger.warning(
                "host measurement still running after %.0fs; spawning compute "
                "child %s anyway", _MEASURE_BEFORE_SPAWN_S, self.label,
            )
        while not p._stopping.is_set():
            self.watchdog_fired = False
            self.child_saw_hello = False
            self.boot_fatal = None
            self.hang_armed_at = None
            self.hang_hold_reported = False
            self.stall_reported = False
            oom_before = postmortem.oom_kill_count()
            started = time.monotonic()
            self.last_frame_at = started
            try:
                proc = await self._spawn_child()
            except OSError as exc:
                logger.error("compute child %s spawn failed: %s", self.label, exc)
                await p._sleep_or_stop(backoff)
                backoff = min(backoff * 2, p._backoff_cap)
                continue
            self.proc = proc
            rc = await proc.wait()
            lifetime = time.monotonic() - started
            self.proc = None
            self.death_report_done.clear()
            saw_hello = self.child_saw_hello
            await self._settle_link()
            # AFTER the settle: _settle_link drains the stderr pipe to EOF so
            # the death dial carries the child's actual last words. Closing the
            # transport first would cut that drain short.
            _close_transport(proc)
            if p._stopping.is_set():
                self.death_report_done.set()
                return
            deliberate = p._terminating or (rc == 0 and not self.watchdog_fired)
            if deliberate:
                await self._finish_deliberate_exit(rc, lifetime_s=lifetime)
                return
            cause = await self._handle_child_death(
                rc, oom_before=oom_before, lifetime_s=lifetime, saw_hello=saw_hello,
            )
            if p._draining:
                logger.info(
                    "compute child %s died during drain; not respawning — "
                    "flushing and exiting", self.label,
                )
                await p._finish_shutdown_flush(reason="death_during_drain")
                return
            # pgw#826: terminal boot outcomes never respawn (module docstring).
            fatal = self.boot_fatal
            if fatal is not None and fatal.get("terminal"):
                await p._fail_boot_fatal(self, fatal)
                return
            if not saw_hello and self.deaths_before_hello >= p._boot_death_limit:
                await p._fail_boot_loop(self, cause)
                return
            if lifetime >= _BACKOFF_RESET_AFTER_ALIVE_S:
                backoff = p._backoff_base
            await p._sleep_or_stop(backoff)
            backoff = min(backoff * 2, p._backoff_cap)

    async def _settle_link(self) -> None:
        """Let the reaped child's buffered frames finish relaying, then close."""
        # pgw#833: drain the stderr pipe to EOF first (bounded) so the death
        # dial below carries the child's actual last words, not a prefix. A
        # grandchild holding the fd open cannot stall the death path: on
        # timeout the ring simply keeps what has arrived so far.
        task = self.stderr_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), 1.0)
            except (asyncio.TimeoutError, Exception):
                pass
        try:
            await asyncio.wait_for(self.link_closed.wait(), _LINK_SETTLE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "compute child %s link did not settle within %.1fs after exit; "
                "closing it (late frames may be lost)",
                self.label, _LINK_SETTLE_TIMEOUT_S,
            )
        link = self.link
        if link is not None:
            link.writer.close()
            self.link = None
            self.link_ready.clear()
            self.link_closed.set()
        # pgw#937: whatever route got us here, this generation is done speaking.
        self.participating = False

    async def await_exit(self, timeout: float) -> bool:
        """TimeoutStopSec: wait for THIS child to exit, then SIGKILL it."""
        proc = self.proc
        if proc is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "compute child %s did not exit within %.0fs of shutdown "
                "(TimeoutStopSec) — SIGKILL", self.label, timeout,
            )
            await self.p._dial_detail(
                f"phase=compute_stop_timeout group={self.ordinal} "
                f"timeout_s={timeout:.0f} spawns={self.spawn_count} — child "
                "SIGKILLed after a deliberate shutdown request"
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return False

    async def _report_in_flight_dead(self, cause: str) -> Dict[Tuple[str, int], str]:
        """One typed FATAL per open job in THIS group, into the DURABLE queue.

        Ships on the live stream now, or survives to the next one. Only this
        group's jobs are attributed — a sibling's in-flight is untouched.
        """
        died_jobs = dict(self.in_flight)
        self.in_flight.clear()
        try:
            for (rid, att), fn in sorted(died_jobs.items()):
                self.reported_dead[(rid, att)] = fn
                # pgw#937: the death path is the OTHER exit from the parent's
                # dispatch-time observation. `_attest_result` pops it on the
                # child's JobResult; a job whose child died never produces one,
                # and 512 orphans FIFO-evict LIVE jobs' observations — which
                # silently stops billing attestation on a crash-looping pod.
                self.p._observations.pop((rid, att), None)
                while len(self.reported_dead) > _REPORTED_DEAD_CAP:
                    self.reported_dead.popitem(last=False)
                await self.p.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                    request_id=rid,
                    attempt=att,
                    status=pb.JOB_STATUS_FATAL,
                    safe_message=(
                        f"{DEATH_LABEL}: cause={cause} function={fn or 'unknown'} "
                        f"(handler process died; worker alive and respawning)"
                    )[:512],
                )))
        finally:
            self.death_report_done.set()
        return died_jobs

    async def _finish_deliberate_exit(self, rc: int, *, lifetime_s: float) -> None:
        """This group's child left on purpose (drain, or a forwarded SIGTERM)."""
        p = self.p
        p._child_exited_clean = True
        cause = f"exit:{rc}" if rc >= 0 else self._death_cause(rc, 0)[0]
        died = await self._report_in_flight_dead(cause)
        if died:
            logger.error(
                "compute child %s exited deliberately (rc=%s) with %d job(s) "
                "still in flight; attributed typed", self.label, rc, len(died),
            )
            await p._dial_detail(postmortem.format_detail(
                phase="compute_process_exit",
                verdict={"signaled": rc < 0, "exit_code": rc if rc >= 0 else 128 - rc},
                limits=postmortem.container_limits(),
                oom_kill_delta=0,
                lifetime_s=lifetime_s,
                extra={"cause": cause, "deliberate": True, "group": self.ordinal,
                       "in_flight": sorted(f"{r}#{a}" for (r, a) in died)},
            ))
        else:
            logger.info(
                "compute child %s exited cleanly (rc=%s)", self.label, rc,
            )
        # Nothing of this child may attribute the next generation. At G>1 the
        # explicit path cannot unlink a live sibling's marker (pgw#938).
        postmortem.clear_inflight(path=self.inflight_marker_path)
        await p._finish_shutdown_flush(
            reason="terminating" if p._terminating else "drain"
        )

    def _death_cause(self, rc: int, oom_delta: int) -> Tuple[str, Dict[str, Any]]:
        if self.watchdog_fired:
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
    ) -> str:
        p = self.p
        now = time.monotonic()
        oom_delta = max(0, postmortem.oom_kill_count() - oom_before)
        cause, verdict = self._death_cause(rc, oom_delta)
        self.death_times.append(now)
        if not saw_hello:
            self.deaths_before_hello += 1
        else:
            self.deaths_before_hello = 0

        # 1) Attribution first (durable, before any flush can conclude).
        died_jobs = await self._report_in_flight_dead(cause)
        logger.error(
            "compute child %s died: cause=%s rc=%s lifetime=%.1fs in_flight=%s "
            "(respawning ITS group; stream identity kept, siblings untouched)",
            self.label, cause, rc, lifetime_s,
            sorted(r for r, _ in died_jobs) or "none",
        )

        # 1b) pgw#937: RETIRE this generation from the worker view before
        # anything else ships. Until this runs, the dead child's last frame is
        # still a vote — it can pin an activity RUNNING, retire a function a
        # live group serves, and (merge.py's `all(...)`) veto a live group's
        # `self_stalled` confession, which is the one that costs money.
        for out in p._retire_group_generation(self):
            try:
                await p.transport.send(out)
            except Exception:
                logger.debug("retirement message send failed", exc_info=True)

        # 2) Post-mortem dial (gw#640 typed exit capture; pgw#676/714 parity).
        extra: Dict[str, Any] = {"group": self.ordinal, "generation": self.generation}
        if verdict.get("signaled"):
            try:
                extra.update(postmortem.attribute_signal_death(
                    signal_name=str(verdict.get("signal_name") or ""),
                    inflight_path=self.inflight_marker_path,
                    dump_path=self.fault_dump_path,
                    load_progress_path=self.load_progress_path,
                ))
            except Exception:
                logger.warning("signal-death attribution failed", exc_info=True)
        else:
            postmortem.clear_inflight(path=self.inflight_marker_path)
        # pgw#833: the dying child's own stderr tail rides the dial — the only
        # forensic channel that survives a pre-Hello death on a provider with
        # no container-logs API.
        stderr_tail = self.stderr_tail_text()
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
                "spawn_count": self.spawn_count,
                "saw_hello": saw_hello,
                **({"child_stderr_tail": stderr_tail} if stderr_tail else {}),
            },
        )
        await p._dial_detail(detail)

        # 3) StartLimitBurst / StartLimitIntervalSec: DETECT the loop for THIS
        # group and report it typed. Post-Hello loops keep respawning; pre-Hello
        # loops are bounded by child_loop's boot_death_limit check (pgw#826).
        recent = [t for t in self.death_times if now - t <= p._start_limit_interval]
        looping = len(recent) >= p._start_limit_burst or self.deaths_before_hello >= 2
        if looping and now - self.last_crash_loop_report_at >= _CRASH_LOOP_REPORT_MIN_INTERVAL_S:
            self.last_crash_loop_report_at = now
            p.crash_loop_reports += 1
            await p._dial_detail(
                f"phase=compute_crash_loop group={self.ordinal} "
                f"deaths={len(recent)} window_s={p._start_limit_interval:.0f} "
                f"deaths_before_hello={self.deaths_before_hello} last_cause={cause} "
                f"spawns={self.spawn_count} generation={self.generation} — "
                "respawn continues; this group advertises no serving capacity "
                "while its child is down, and contributes no fact to the "
                "worker's merged view either (pgw#937 down-group semantics: "
                "a down group is EXCLUDED from the fan-in, not defaulted)"
            )

        # 4) Give the live stream a moment to ship the FATALs.
        try:
            await p.transport.queue.wait_empty(timeout=_DEATH_FLUSH_GRACE_S)
        except Exception:
            pass
        if p._draining or p._terminating or p._stopping.is_set():
            return cause
        # Re-sync the desired state to the respawned child. At G==1 this is the
        # proven path: cycle the connection NOW so the fresh Hello re-drives
        # residency (byte-identical to the single-child parent). At G>1 the
        # OTHER groups are still serving on this same stream — cycling here would
        # stall the healthy siblings until this group reboots, and build_hello
        # would block on the down slot. So DON'T cycle on death; the re-sync
        # happens when THIS group's respawned child reconnects (_on_child_connect
        # triggers the cycle then, with every slot's link back up).
        if p.execution_groups == 1:
            p.transport.cycle_connection()
        return cause

    # ---- watchdog (WatchdogSec), per child -------------------------------

    async def watchdog_loop(self) -> None:
        """Missed beats ARM the verdict; the open activity DECIDES it (pgw#771).

        Per child: the parent witnesses THIS child's /proc, because a child
        starved of the GIL cannot witness for itself. The parent kills only what
        is provably NOT RUNNING; a child that runs but serves nothing is the
        hub's stall clock to reap. One child's kill never touches a sibling.
        """
        p = self.p
        interval = max(0.25, p._watchdog_budget / 4.0)
        while not p._stopping.is_set():
            await asyncio.sleep(interval)
            proc = self.proc
            if proc is None or self.link is None or self.relaying:
                continue
            if p._draining or p._terminating:
                continue
            now = time.monotonic()
            self._sample_child_evidence(proc.pid, now)
            silent_for = now - self.last_frame_at
            if silent_for <= p._watchdog_budget:
                self.hang_armed_at = None
                continue
            if self.hang_armed_at is None:
                self.hang_armed_at = now
            await self._report_stall_if_any(now)
            verdict = self._hang_verdict(now)
            if verdict is None:
                continue
            if verdict == "held":
                if not self.hang_hold_reported:
                    self.hang_hold_reported = True
                    logger.warning(
                        "compute child %s loop silent for %.1fs but activity %r "
                        "is alive (evidence advanced %.1fs ago) — hang HELD",
                        self.label, silent_for, self.liveness_activity,
                        now - self.liveness_evidence_at,
                    )
                    await p._dial_detail(
                        f"phase=compute_hang_verdict_held group={self.ordinal} "
                        f"loop_silent_s={silent_for:.0f} "
                        f"activity={self.liveness_activity} "
                        f"evidence_age_s={now - self.liveness_evidence_at:.1f} "
                        f"evidence={self.liveness_evidence:.1f} "
                        f"ping_age_s={now - self.last_liveness_at:.1f} "
                        f"budget_s={p._watchdog_budget:.0f} — the child's event "
                        "loop is starved by accounted work, not hung; not killing"
                    )
                continue
            logger.error(
                "compute child %s silent for %.1fs (budget %.1fs, verdict=%s) — "
                "killing the wedged child (WatchdogSec analog)",
                self.label, silent_for, p._watchdog_budget, verdict,
            )
            self.watchdog_fired = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _child_evidence(self, pid: int) -> Optional[float]:
        """This child tree's kernel-accounted work — see
        :func:`gen_worker.proc_evidence.tree_evidence`, which this grew into
        and which `parallel.group` now shares (pgw#892)."""
        return proc_evidence.tree_evidence(pid)

    def _sample_child_evidence(self, pid: int, now: float) -> None:
        evidence = self._child_evidence(pid)
        if evidence is None:
            return
        previous = self.liveness_evidence
        if previous is None or evidence - previous >= _EVIDENCE_EPS:
            self.liveness_evidence = evidence
            self.liveness_evidence_at = now
            self.stall_reported = False

    async def _report_stall_if_any(self, now: float) -> None:
        """Say so when THIS child owes work and is accruing none — measured by
        the parent from /proc, not self-reported by the child (the security
        driver: a tenant-produced number is a hint, a parent-side measurement is
        evidence)."""
        if self.liveness_evidence is None or self.stall_reported:
            return
        if not self.in_flight and not self.liveness_activity:
            return
        age = now - self.liveness_evidence_at
        if age <= self.p._evidence_hold_window:
            return
        self.stall_reported = True
        logger.warning(
            "compute child %s has accrued no CPU/IO for %.1fs while %d job(s) and "
            "activity %r are open — reporting the stall (stream and beat kept)",
            self.label, age, len(self.in_flight), self.liveness_activity,
        )
        await self.p._dial_detail(
            f"phase=compute_child_stalled group={self.ordinal} "
            f"evidence_age_s={age:.1f} activity={self.liveness_activity or 'none'} "
            f"in_flight={sorted(f'{r}#{a}' for (r, a) in self.in_flight)} "
            f"loop_silent_s={now - self.last_frame_at:.1f} "
            f"window_s={self.p._evidence_hold_window:.0f} — measured by the parent "
            "from /proc, not self-reported by the child"
        )

    def _hang_verdict(self, now: float) -> Optional[str]:
        """``None`` = no decision yet, ``"held"`` = alive-but-starved, otherwise
        the reason the child is being killed."""
        if self.liveness_evidence is None:
            return "no_evidence_source"
        if now - self.liveness_evidence_at > self.p._evidence_hold_window:
            return "no_work_accrued"
        if not self.liveness_activity:
            return "loop_wedged_no_activity"
        return "held"


class ParentControl:
    """The control process: real Transport + the security boundary + supervision
    of a GROUP of compute children (one ``_ChildSlot`` per execution group)."""

    def __init__(
        self,
        settings: Settings,
        *,
        child_cmd: Optional[List[str]] = None,
        child_env: Optional[Dict[str, str]] = None,
        socket_path: Optional[str] = None,
        measure_cmd: Optional[List[str]] = None,
        topology: Optional[ExecutionTopology] = None,
        respawn_backoff_base_s: float = _DEFAULT_RESPAWN_BACKOFF_BASE_S,
        respawn_backoff_cap_s: float = _DEFAULT_RESPAWN_BACKOFF_CAP_S,
        start_limit_burst: int = _DEFAULT_START_LIMIT_BURST,
        start_limit_interval_s: float = _DEFAULT_START_LIMIT_INTERVAL_S,
        boot_death_limit: int = _DEFAULT_BOOT_DEATH_LIMIT,
        watchdog_budget_s: float = _DEFAULT_WATCHDOG_BUDGET_S,
        stop_timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
        stop_flush_timeout_s: float = _STOP_FLUSH_TIMEOUT_S,
        beat_interval_s: float = 0.0,   # 0 = adopt the child's declared cadence
        transport_backoff_base_s: float = 1.0,
        transport_backoff_cap_s: float = 30.0,
    ) -> None:
        self._settings = settings
        # pgw#931 follow-up: the CONTROL PARENT is the process that holds the
        # worker credential — the compute child is deliberately stripped of it
        # (`_CHILD_FORBIDDEN_ENVS`) and signs through this process. So the boot
        # token is installed HERE, where the parent's `Settings` arrive, not
        # only in `run_parent`.
        #
        # It was in `run_parent` alone, and that was wrong: every other way of
        # building a control parent — the split harness, the group-process
        # tests, any embedder — got a parent with no credential, and the
        # mediated C2PA sign refused with "this pod holds no worker JWT".
        # Deriving a fact at ONE of several entry points is the §4.22 defect:
        # the fact and its carrier have to be established together.
        worker_credential.install_bootstrap(settings)
        env_cmd = os.environ.get(ENV_CHILD_CMD, "").strip()
        self._child_cmd = list(
            child_cmd
            if child_cmd is not None
            else (shlex.split(env_cmd) if env_cmd else [sys.executable, "-m", "gen_worker.entrypoint"])
        )
        self._child_env = dict(child_env or {})
        # pgw#931: was `child_env or os.environ or default` — the same env read
        # through two doors, neither of which was the loader. `Settings` owns the
        # value now; the child_env override stays because it is this parent
        # deliberately pointing THIS child somewhere else.
        record_path = (
            self._child_env.get("GEN_WORKER_BOOT_RECORD")
            or self._settings.boot_record_path
            or str(postmortem.BOOT_RECORD_PATH)
        )
        self._postmortem_dir = Path(record_path).parent
        self._socket_path = socket_path or f"/tmp/gen-worker-compute-{os.getpid()}.sock"
        self._backoff_base = respawn_backoff_base_s
        self._backoff_cap = respawn_backoff_cap_s
        self._start_limit_burst = max(1, int(start_limit_burst))
        self._start_limit_interval = start_limit_interval_s
        self._boot_death_limit = max(1, int(boot_death_limit))
        self._watchdog_budget = watchdog_budget_s
        self._stop_timeout = stop_timeout_s
        self._stop_flush_timeout = stop_flush_timeout_s
        self.transport = Transport(
            settings,
            self,
            backoff_base_s=transport_backoff_base_s,
            backoff_cap_s=transport_backoff_cap_s,
        )

        # pgw#783: the delivered packing decides how many children exist. Pure
        # env parse — no torch, no CUDA (the parent stays a control plane). The
        # plan gives each group its devices, socket, and env delta; at G==1 the
        # plan is one slot with an EMPTY env delta and the original socket path.
        self._topology = topology if topology is not None else ExecutionTopology.from_env()
        self._plan = GroupPlan.for_topology(self._topology, socket_path=self._socket_path)

        # pgw#858: decide the compute uid and grant it what it needs BEFORE any
        # slot exists, so no child is ever spawned into a half-prepared pod.
        self._drop_plan = self._prepare_privilege_drop()

        self._slots: List[_ChildSlot] = [
            _ChildSlot(self, group) for group in self._plan.children
        ]

        # pgw#783: the worker session id is minted ONCE, here, and passed to
        # every child — so it survives child respawns (child-minted it changed
        # on each respawn and the hub rejected the cross-session shadow state, a
        # latent defect even at G=1) and is shared across groups (one worker,
        # one session).
        self._worker_session_id = uuid.uuid4().hex

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping = asyncio.Event()
        self._beat_interval = beat_interval_s
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
        # Worker-level beat state: the last (merged) StateDelta and when any
        # group last published, so the beat re-sends the worker's freshest truth.
        self._last_state_delta: Optional[pb.WorkerMessage] = None
        self._last_state_delta_at = 0.0
        self.parent_beats_sent = 0  # observability + tests
        self._draining = False
        self._terminating = False
        self._child_exited_clean = False
        self._shutdown_flushed = False
        self.crash_loop_reports = 0  # observability + tests
        # pgw#826: set when a terminal boot outcome makes the parent exit 1.
        self._terminal_exit = False
        self.terminal_exit_reason = ""  # observability + tests
        self._stop_deadline_task: Optional[asyncio.Task] = None
        self._reported_unretired = False
        self.unretired_results_at_exit = 0  # observability + tests
        # delta 1: parent-mediated action accounting (observability + tests).
        # pgw#876 §2: `_jwt_rotations` and `actions_performed` lived here too
        # and were WRITE-ONLY — incremented, never logged, never asserted.
        # Deleted rather than wired: nothing wanted them.
        self.actions_refused = 0
        self._last_action_refusal_report_at = _NEVER_REPORTED
        self._action_slots = asyncio.Semaphore(_MAX_CONCURRENT_ACTIONS)
        self._file_base_url = ""
        self._identity_cache: Optional[Tuple[str, str]] = None
        # delta 3: (request_id, attempt) -> what the parent watched.
        self._observations: collections.OrderedDict = collections.OrderedDict()
        self.metric_divergences = 0  # observability + tests
        self._last_attestation_report_at = _NEVER_REPORTED
        # delta 4: per-job grant decisions (observability + tests).
        self.capability_withheld = 0
        self.capability_notes = 0
        self._last_capability_report_at = _NEVER_REPORTED
        # delta 2: the parent's own pre-import host measurement.
        self._measure_cmd = list(
            measure_cmd
            if measure_cmd is not None
            else [sys.executable, "-m", "gen_worker.procsplit.measure"]
        )
        self._measurement: Optional[Dict[str, Any]] = None
        self._measured = asyncio.Event()
        self._measure_task: Optional[asyncio.Task] = None
        # pgw#771 fan-in: per-group activity + fn signals, reconciled to ONE
        # worker view before the stream (Paul: the hub sees one worker).
        self._group_activities: Dict[int, Dict[str, pb.ActivityUpdate]] = {}
        self._activity_seq = 0
        self._group_fn_unavail: Dict[int, Dict[str, pb.FnUnavailable]] = {}
        self._group_fn_degraded: Dict[int, Dict[str, pb.FnDegraded]] = {}
        # pgw#783 THE INVARIANT: account every relayed frame so a job whose DATA
        # crosses the parent's interpreter is VISIBLE (a control ceiling on
        # job_result bytes). If the seam ever carries data the GIL bottleneck
        # pgw#782 measured reappears one layer up. Reported, never fatal.
        self.seam = SeamAccountant()

    @property
    def execution_groups(self) -> int:
        return len(self._slots)

    # ---- pgw#858: the compute uid ----------------------------------------

    def _prepare_privilege_drop(self) -> Optional[privdrop.DropPlan]:
        """Decide the compute child's uid, then hand that uid everything it
        needs — while this process is still root and can.

        The list is deliberately explicit (``privdrop.writable_paths`` plus the
        pod's cache roots): the answer to a permission error the child hits is
        another entry here, never giving the child root back. Anything NOT in
        it stays root-owned and read-only to tenant code, which is the point.
        """
        try:
            plan = privdrop.plan_drop(_COMPUTE_HOME)
        except Exception:
            logger.error(
                "refusing to spawn a compute child: the pgw#858 privilege drop "
                "could not be planned", exc_info=True,
            )
            raise
        if plan is None:
            return None
        extra = [
            # weights/CAS: written by the child (it does the fetching), empty on
            # a cold pod so the chown is free, metadata-only when warm. Read
            # from env/Settings rather than models.cache_paths — importing that
            # package pulls the model layer, and this process never imports torch.
            # pgw#931 VIOLATION-A #1: `_settings.tensorhub_cache_dir` and a raw
            # read of the SAME env sat in adjacent lines of this expression —
            # four sources for one fact, two of them the same variable through
            # different doors. `Settings` also loads from yaml, /run/secrets and
            # `.env`, so the raw read could only ever WIN where the loader had
            # already lost. Deleted.
            self._child_env.get("TENSORHUB_CACHE_DIR", "")
            or self._settings.tensorhub_cache_dir
            or _DEFAULT_TENSORHUB_CACHE_DIR,
            self._child_env.get("TENSORHUB_CAS_DIR", "")
            or self._settings.tensorhub_cas_dir,
            # post-mortem markers: the CHILD writes inflight/fault-dump/streaks
            # and this parent takes them, so the dir is genuinely shared. Both
            # sides of it, because the pod points the child at a durable
            # carrier (GEN_WORKER_BOOT_RECORD) while the parent's own default
            # may still be the volatile one.
            str(postmortem.BOOT_RECORD_PATH.parent),
            os.path.dirname(
                self._child_env.get("GEN_WORKER_BOOT_RECORD", "")
                or self._settings.boot_record_path
            ),
            # th#1087's mutable-config snapshot: the CHILD atomically rewrites
            # it on every config-generation push (tmp file in the SAME dir plus
            # os.replace, so the directory itself must be writable), and unlike
            # the post-mortem markers that writer RAISES on failure. It lives
            # in the image at /app/.tensorhub, which is root-owned.
            os.path.dirname(
                self._child_env.get("GEN_WORKER_CONFIG_SNAPSHOT_PATH", "")
                or self._settings.config_snapshot_path
                or _DEFAULT_CONFIG_SNAPSHOT_PATH
            ),
        ]
        granted = privdrop.grant_paths(plan, privdrop.writable_paths(plan, extra))
        privdrop.grant_devices(plan)
        # The datacenter-warm fill source is a mounted network volume we only
        # ever READ. It is deliberately not in the granted set — say so rather
        # than let a later reader assume it was missed.
        fill = self._settings.tensorhub_fill_source_dir
        if fill and not os.access(fill, os.R_OK | os.X_OK):
            logger.warning(
                "fill source %s is not readable by the compute uid; warm fill "
                "will fall back to R2", fill,
            )
        logger.info(
            "compute child will run as %s; writable: %s",
            plan.describe(), ", ".join(granted),
        )
        return plan

    # Single-slot conveniences: the per-child process/spawn state moved into
    # _ChildSlot (pgw#783), but at G==1 there is exactly one, and the G=1
    # identity suite reads these. They intentionally name slot 0.
    @property
    def _proc(self) -> Optional[asyncio.subprocess.Process]:
        return self._slots[0].proc if self._slots else None

    @property
    def _spawn_count(self) -> int:
        return self._slots[0].spawn_count if self._slots else 0

    # ---- worker-level in-flight helpers ----------------------------------

    def _all_in_flight(self) -> Dict[Tuple[str, int], str]:
        merged: Dict[Tuple[str, int], str] = {}
        for slot in self._slots:
            merged.update(slot.in_flight)
        return merged

    def _is_in_flight(self, rid: str, attempt: int) -> bool:
        return any((rid, attempt) in slot.in_flight for slot in self._slots)

    def _slot_for_request(self, key: Tuple[str, int]) -> Optional[_ChildSlot]:
        for slot in self._slots:
            if key in slot.in_flight:
                return slot
        return None

    def _route_slot(self, run: pb.RunJob) -> Optional[_ChildSlot]:
        """Which slot serves this dispatch. At G==1 always the one slot; at G>1
        route by the hub-picked rank-0 device, refusing a mis-dispatch (never
        flooring onto group 0 — pgw#779)."""
        if self.execution_groups == 1:
            return self._slots[0]
        gpu_index = run.compute.gpu_index if run.HasField("compute") else None
        try:
            ordinal = self._plan.route(gpu_index)
        except (ValueError, Exception) as exc:  # noqa: BLE001 - typed refusal below
            logger.error("cannot route dispatch %s: %s", run.request_id, exc)
            return None
        return self._slots[ordinal] if 0 <= ordinal < self.execution_groups else None

    # ---- hardware + canary (parent-owned, PRE-IMPORT) ---------------------

    async def _measure_host(self) -> None:
        """Measure the silicon in a process that has imported no tenant code."""
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
            # pgw#1129/th#1798: the host driver, so the hub can answer
            # "can the host we landed on run this pod's CUDA line?" from a
            # SUCCESSFUL boot instead of only from a corpse.
            driver_version=str(hw.get("driver_version") or ""),
            installed_libs=[str(x) for x in (hw.get("installed_libs") or [])],
            gen_worker_version=str(m.get("gen_worker_version") or ""),
            image_digest=self._settings.worker_image_digest,
            instance_id=self._settings.runpod_pod_id or "",
        )

    # ---- identity (parent-owned) -----------------------------------------

    def _identity(self) -> Tuple[str, str]:
        """(worker_id, release_id) from the JWT THIS process holds."""
        if self._identity_cache is not None:
            return self._identity_cache
        worker_id = (self._settings.worker_id or "").strip()
        release_id = ""
        # pgw#848: IDENTITY claims (sub / release_id), which rotation never
        # changes — the bootstrap copy is correct here and is cached anyway.
        # Anything AUTHENTICATING must read `worker_credential.current()`.
        token = (self._settings.bootstrap_worker_jwt or "").strip()
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

    async def _request_slot_hello(self, slot: _ChildSlot) -> Optional[pb.Hello]:
        """Round-trip a fresh Hello out of ONE group's child (waiting for its
        link). Returns None only if stopping."""
        while not self._stopping.is_set():
            link = slot.link
            if link is not None:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                slot.hello_waiter = fut
                try:
                    await link.writer.frame(frames.T_HELLO_REQ, frames.pack_meta({}))
                    raw = await fut
                except (ConnectionError, OSError, asyncio.CancelledError):
                    raise
                finally:
                    if slot.hello_waiter is fut:
                        slot.hello_waiter = None
                return pb.Hello.FromString(raw)
            await slot.link_ready.wait()
        return None

    async def build_hello(self) -> pb.Hello:
        """Assemble the worker's Hello. delta 2: never before the parent's own
        measurement has had its chance. pgw#783: at G>1 merge every group's
        child Hello into one worker view (Paul: the hub sees one worker)."""
        try:
            await asyncio.wait_for(self._measured.wait(), _MEASURE_TIMEOUT_S + 5.0)
        except asyncio.TimeoutError:
            pass

        if self.execution_groups == 1:
            # BYTE-IDENTICAL to the single-child parent: request the one child's
            # Hello and apply the delta identity/resources overrides + in-flight
            # merge inline, exactly as before pgw#783.
            hello = await self._request_slot_hello(self._slots[0])
            if hello is None:
                return pb.Hello()
            self._apply_identity_and_resources(hello)
            # pgw#783: the parent owns the session id (the child reads it from
            # env, but assert it here too so a stale child can never regress it).
            hello.worker_session_id = self._worker_session_id
            if self._beat_interval <= 0 and hello.heartbeat_interval_ms > 0:
                self._beat_interval = hello.heartbeat_interval_ms / 1000.0
            seen = {(j.request_id, j.attempt) for j in hello.in_flight}
            for rid, att in self.transport.queue.pending_result_keys:
                if (rid, att) not in seen:
                    hello.in_flight.add(request_id=rid, attempt=att)
            for (rid, att) in self._slots[0].in_flight:
                if (rid, att) not in seen:
                    hello.in_flight.add(request_id=rid, attempt=att)
            return hello

        # G>1: gather every group's child Hello, then merge to one worker view.
        hellos: List[pb.Hello] = []
        for slot in self._slots:
            h = await self._request_slot_hello(slot)
            if h is not None:
                hellos.append(h)
        if not hellos:
            return pb.Hello()
        extra = list(self.transport.queue.pending_result_keys)
        extra += list(self._all_in_flight().keys())
        hello = merge.merge_hello(
            hellos,
            worker_session_id=self._worker_session_id,
            extra_in_flight=extra,
        )
        self._apply_identity_and_resources(hello)
        promised = [h.heartbeat_interval_ms for h in hellos if h.heartbeat_interval_ms]
        if self._beat_interval <= 0 and promised:
            self._beat_interval = min(promised) / 1000.0
        return hello

    def _apply_identity_and_resources(self, hello: pb.Hello) -> None:
        """delta 1 + delta 2: the worker/release the Hello claims to BE, and the
        hardware it asserts, come from the credential-holding parent — never
        from a child that imports tenant code."""
        worker_id, release_id = self._identity()
        if worker_id:
            hello.worker_id = worker_id
        if release_id:
            hello.release_id = release_id
        resources = self._parent_resources()
        if resources is not None:
            hello.resources.CopyFrom(resources)
        elif hello.HasField("resources"):
            logger.error(
                "no parent-side host measurement is available; DROPPING the "
                "child's self-reported resources rather than relaying "
                "tenant-reachable numbers the fleet condemns SKUs on "
                "(pgw#763 delta 2 / th#1310)"
            )
            hello.ClearField("resources")

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        # delta 1: the hub's own base URL, for parent-mediated actions.
        if ack.file_base_url:
            self._file_base_url = ack.file_base_url.rstrip("/")
        # CONNECTED before the ack, mirroring Transport's _connected ordering.
        # Broadcast to every group's child.
        payload = ack.SerializeToString()
        for slot in self._slots:
            link = slot.link
            if link is None:
                continue
            await link.writer.frame(frames.T_CONNECTED)
            await link.writer.frame(frames.T_HELLO_ACK, payload)

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        which = msg.WhichOneof("msg")
        if which == "run_job":
            await self._dispatch_run_job(msg)
            return
        if which == "cancel_job":
            key = (msg.cancel_job.request_id, msg.cancel_job.attempt)
            slot = self._slot_for_request(key)
            if self.execution_groups == 1:
                slot = self._slots[0]  # identity: relay even if not yet tracked
            if slot is None or slot.link is None:
                return
            if key in slot.reported_dead:
                return
            await slot.link.writer.frame(frames.T_SCHED, msg.SerializeToString())
            return
        if which == "drain":
            self._draining = True
            payload = msg.SerializeToString()
            any_link = False
            for slot in self._slots:
                if slot.link is not None:
                    any_link = True
                    await slot.link.writer.frame(frames.T_SCHED, payload)
            if not any_link:
                asyncio.create_task(self._drain_without_child(), name="drain-no-child")
            return
        # hello_ack handled above; everything else (model_op, token_refresh,
        # serve_posture, …) is worker-wide desired state: broadcast to every
        # group. pgw#1142's eager-only order is worker-wide by definition —
        # the compiled serving it suppresses lives in the CHILDREN, and every
        # one of them has to hear it, so it takes this path unchanged.
        payload = msg.SerializeToString()
        delivered = False
        for slot in self._slots:
            if slot.link is not None:
                await slot.link.writer.frame(frames.T_SCHED, payload)
                delivered = True
        if not delivered:
            logger.warning("dropping %s command while all compute children are down", which)

    async def _dispatch_run_job(self, msg: pb.SchedulerMessage) -> None:
        run = msg.run_job
        slot = self._route_slot(run)
        if slot is None or slot.link is None:
            await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                request_id=run.request_id,
                attempt=run.attempt,
                status=pb.JOB_STATUS_RETRYABLE,
                safe_message="compute process restarting",
            )))
            return
        # delta 4: the parent DECIDES on the per-job grant before it reaches
        # tenant code — forward, or withhold and refuse.
        if not await self._authorize_run_job(run):
            return
        key = (run.request_id, run.attempt)
        # delta 3: what the parent watched, recorded BEFORE the job exists in
        # the child. The in-flight count is the parent's own dispatch-time
        # observation (all groups), not a child claim.
        self._observations[key] = attest.JobObservation(
            function=run.function_name,
            relayed_at=time.monotonic(),
            concurrency_at_relay=len(self._all_in_flight()),
        )
        while len(self._observations) > _OBSERVATION_CAP:
            self._observations.popitem(last=False)
        slot.in_flight[key] = run.function_name
        # pgw#783: at G>1 the child's world starts at cuda:0 under
        # CUDA_VISIBLE_DEVICES, so rewrite the dispatched rank-0 device to the
        # child-local 0. At G==1 there is no rewrite (identity).
        relay = msg
        if self.execution_groups > 1 and run.HasField("compute"):
            relay = pb.SchedulerMessage()
            relay.CopyFrom(msg)
            relay.run_job.compute.gpu_index = self._plan.local_gpu_index(slot.ordinal)
        try:
            await slot.link.writer.frame(frames.T_SCHED, relay.SerializeToString())
        except (ConnectionError, OSError):
            slot.in_flight.pop(key, None)
            await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                request_id=run.request_id,
                attempt=run.attempt,
                status=pb.JOB_STATUS_RETRYABLE,
                safe_message="compute process restarting",
            )))

    async def on_message_shipped(self, msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") != "model_event":
            return
        payload = msg.SerializeToString()
        for slot in self._slots:
            if slot.link is not None:
                await slot.link.writer.frame(frames.T_SHIPPED, payload)

    async def on_disconnect(self) -> None:
        for slot in self._slots:
            link = slot.link
            if link is not None:
                try:
                    await link.writer.frame(frames.T_DISCONNECTED)
                except Exception:
                    pass

    async def on_token_refresh(self, token: str, expires_at_unix: int) -> None:
        """The rotated worker JWT stays HERE (delta 1) — never sent to a child.

        Present so the non-forward is DECLARED rather than accidental: the
        transport calls this hook if it exists, and its absence would read as
        an oversight. The body is empty on purpose — the compute child holds
        no credential (`ChildTransport.current_worker_jwt` is always ""), and
        every identity-bearing call it makes goes through `procsplit.broker`,
        where the parent presents `worker_credential.current()`.
        """

    # ---- fan-in: N children -> ONE worker view (pgw#783, Paul ruling 2) ----
    #
    # DOWN-GROUP SEMANTICS (pgw#937 ruling; the rule `_handle_child_death`'s
    # crash-loop dial has always claimed in prose and nothing implemented).
    #
    # **A group without a live child is not a participant in the worker view,
    # and its facts are UNKNOWN — which is neither "still true" nor the
    # live-group default.** Every fan-in structure here is generation-scoped:
    # it holds what the CURRENT incarnation of that group said, and it is
    # dropped the moment that incarnation ends (§4.15 — a respawn is a new
    # generation of the same group, under the same worker session).
    #
    # The vocabulary has three states per (group, function/kind), and the bug
    # class this fixes is the §4.22 one — a *default* being made to carry a
    # missing *fact*:
    #
    #   present entry -> a fact the live incarnation reported
    #   absent entry  -> the live-group DEFAULT (serves it / no activity open)
    #   not a member  -> UNKNOWN: this group has no live child
    #
    # so the identity element of a down group is **exclusion from the merge**,
    # in every one of the four aggregations:
    #
    #   last_state_delta   -> dropped from `merge_state_deltas`: the group
    #                         contributes no functions to the union, no free
    #                         VRAM to the sum, and no vote to the phase min.
    #   _group_fn_unavail  -> dropped from `worker_fn_unavailable`'s mapping.
    #                         NOT set to None: None means "this group serves
    #                         it", so popping alone would make a dead group
    #                         read as serving EVERYTHING — strictly worse than
    #                         the stale entry it replaced.
    #   _group_fn_degraded -> dropped from both the mapping and the
    #                         `served_native_somewhere` scan, for the same
    #                         reason (absence there means "serves it native").
    #   _group_activities  -> dropped from `reconcile_activity_kind`, so a dead
    #                         group can no longer pin a kind RUNNING or veto a
    #                         live group's `self_stalled` confession.
    #
    # Exclusion is chosen over inserting an explicit "unknown" sentinel because
    # it carries the same information without teaching four merge functions a
    # third case and without touching the wire vocabulary.
    #
    # A down group is not silently forgotten, though — that would replace one
    # missing fact with another. `_retire_group_generation` EMITS the
    # consequences: a terminal for every activity kind the dead generation held
    # open that no live group still runs, and the recomputed worker StateDelta.
    # The hub learns the capacity dropped; it never has to infer it from an
    # update that stopped arriving.

    def _live_slots(self) -> List["_ChildSlot"]:
        """The groups whose facts are the worker's, i.e. the participating ones."""
        return [s for s in self._slots if s.participating]

    def _note_state_delta(self) -> None:
        """Recompute the worker's freshest StateDelta after a group published.
        At G==1 it IS the child's message (byte-identical beat); at G>1 it is the
        merge of every LIVE group's latest."""
        self._last_state_delta_at = time.monotonic()
        if self.execution_groups == 1:
            self._last_state_delta = self._slots[0].last_state_delta
            return
        deltas = [s.last_state_delta.state_delta for s in self._live_slots()
                  if s.last_state_delta is not None]
        if deltas:
            self._last_state_delta = pb.WorkerMessage(
                state_delta=merge.merge_state_deltas(deltas)
            )
            return
        # Every group is down. Advertising the last live group's function set
        # would keep the hub dispatching into `_dispatch_run_job`'s "compute
        # process restarting" RETRYABLE — on a loop, with no end the hub can
        # see. The honest worker-level fact is: nothing is served, and the pod
        # is coming back.
        self._last_state_delta = pb.WorkerMessage(
            state_delta=pb.StateDelta(phase=pb.WORKER_PHASE_BOOTING)
        )

    def _retire_group_generation(self, slot: _ChildSlot) -> List[pb.WorkerMessage]:
        """This group's child is gone: end its generation's participation.

        Drops every fan-in fact the dead incarnation reported (see "down-group
        semantics" above) and returns the worker-level messages the hub must
        receive as a consequence — the terminals for activity kinds only that
        group had open, then the recomputed worker StateDelta.

        At G==1 the fan-in structures are never populated (`_fan_in` returns the
        child's message verbatim), so this is a no-op there BY CONSTRUCTION, and
        the single-child parent stays byte-identical.
        """
        if self.execution_groups == 1:
            return []
        slot.participating = False
        slot.last_state_delta = None
        open_kinds = self._group_activities.pop(slot.ordinal, {})
        self._group_fn_unavail.pop(slot.ordinal, None)
        self._group_fn_degraded.pop(slot.ordinal, None)

        out: List[pb.WorkerMessage] = []
        live_ordinals = {s.ordinal for s in self._live_slots()}
        for kind in sorted(open_kinds):
            per_group = {
                ordinal: kinds[kind]
                for ordinal, kinds in self._group_activities.items()
                if kind in kinds and ordinal in live_ordinals
            }
            self._activity_seq += 1
            if per_group:
                # A live group still runs this kind: re-state the worker's
                # activity WITHOUT the dead group's progress and without its
                # vote on `self_stalled`.
                merged = merge.reconcile_activity_kind(
                    per_group, seq=self._activity_seq
                )
            else:
                # Nobody runs it any more. It did not complete — the process
                # doing it died — so the terminal is FAILED, not COMPLETED.
                merged = pb.ActivityUpdate()
                merged.CopyFrom(open_kinds[kind])
                merged.state = pb.ACTIVITY_STATE_FAILED
                merged.seq = self._activity_seq
                merged.self_stalled = False
                merged.stalled_for_ms = 0
                if not merged.error:
                    merged.error = "compute process died with this activity open"
                merged.detail = (
                    f"{merged.detail + '; ' if merged.detail else ''}"
                    "the execution group running this activity died "
                    "(worker alive; the group respawns)"
                )[:512]
            out.append(pb.WorkerMessage(activity_update=merged))

        self._note_state_delta()
        merged_delta = self._last_state_delta
        if merged_delta is not None:
            out.append(merged_delta)
        return out

    def _fan_in(
        self, slot: _ChildSlot, msg: pb.WorkerMessage,
    ) -> Optional[pb.WorkerMessage]:
        """The message to actually put on the stream for one child frame.

        At G==1 this is the child's message VERBATIM — byte-identical to the
        pre-pgw#783 relay. At G>1 the hub must see one worker, so worker-scoped
        signals (state_delta, activity_update, fn_(un)available) are reconciled
        across groups here and per-request signals (job_result/progress/accepted,
        model_event) forward verbatim.
        """
        if self.execution_groups == 1:
            return msg
        which = msg.WhichOneof("msg")
        if which in _WORKER_SCOPED_MSGS and not slot.participating:
            # pgw#937: a retired generation cannot speak for the worker. Its
            # per-request frames still forward (they are about a job, and the
            # death path has already attributed the ones it knows about); its
            # WORKER-scoped claims are a dead process's view of a worker it has
            # left, and re-recording them would resurrect it into the merge.
            return None
        if which == "state_delta":
            merged = self._last_state_delta
            return merged if merged is not None else msg
        if which == "activity_update":
            return self._reconcile_activity(slot, msg)
        if which == "fn_unavailable":
            return self._reconcile_fn_unavailable(slot, msg)
        if which == "fn_degraded":
            return self._reconcile_fn_degraded(slot, msg)
        # Per-request or per-object: forward verbatim (already request-scoped).
        return msg

    def _reconcile_activity(
        self, slot: _ChildSlot, msg: pb.WorkerMessage,
    ) -> Optional[pb.WorkerMessage]:
        act = msg.activity_update
        by_kind = self._group_activities.setdefault(slot.ordinal, {})
        if act.state in (pb.ACTIVITY_STATE_COMPLETED, pb.ACTIVITY_STATE_FAILED):
            by_kind.pop(act.kind, None)
        else:
            by_kind[act.kind] = act
        # Only LIVE groups vote. A dead group's last frame must not pin the kind
        # RUNNING, and must not outvote a live group's `self_stalled` confession
        # (merge.py's `all(...)`) — pgw#937.
        live_ordinals = {s.ordinal for s in self._live_slots()}
        per_group = {
            ordinal: kinds[act.kind]
            for ordinal, kinds in self._group_activities.items()
            if act.kind in kinds and ordinal in live_ordinals
        }
        if not per_group:
            # Every group's activity of this kind is terminal: emit the terminal
            # as the worker's, re-stamped with the parent's seq.
            self._activity_seq += 1
            out = pb.ActivityUpdate()
            out.CopyFrom(act)
            out.seq = self._activity_seq
            return pb.WorkerMessage(activity_update=out)
        self._activity_seq += 1
        merged = merge.reconcile_activity_kind(per_group, seq=self._activity_seq)
        return pb.WorkerMessage(activity_update=merged)

    def _reconcile_fn_unavailable(
        self, slot: _ChildSlot, msg: pb.WorkerMessage,
    ) -> Optional[pb.WorkerMessage]:
        fu = msg.fn_unavailable
        by_fn = self._group_fn_unavail.setdefault(slot.ordinal, {})
        by_fn[fu.function_name] = fu
        # THE CONVENTION, stated at the call site because writing it backwards
        # is the pgw#937 defect: `merge.worker_fn_unavailable` reads a `None`
        # value as "this group SERVES the function". So a group is put in the
        # mapping only while it is live — a down group is EXCLUDED, never
        # entered as `None`, which would make it read as serving everything.
        per_group: Dict[int, Optional[pb.FnUnavailable]] = {}
        for s in self._live_slots():
            per_group[s.ordinal] = self._group_fn_unavail.get(
                s.ordinal, {}
            ).get(fu.function_name)
        worker_level = merge.worker_fn_unavailable(per_group)
        if worker_level is None:
            # Some group still serves it: the worker serves it. Emit nothing.
            return None
        return pb.WorkerMessage(fn_unavailable=worker_level)

    def _reconcile_fn_degraded(
        self, slot: _ChildSlot, msg: pb.WorkerMessage,
    ) -> Optional[pb.WorkerMessage]:
        fd = msg.fn_degraded
        by_fn = self._group_fn_degraded.setdefault(slot.ordinal, {})
        by_fn[fd.function_name] = fd
        live = self._live_slots()
        per_group: Dict[int, Optional[pb.FnDegraded]] = {}
        for s in live:
            per_group[s.ordinal] = self._group_fn_degraded.get(
                s.ordinal, {}
            ).get(fd.function_name)
        # A group serves this function NATIVE when it reports neither degraded
        # nor unavailable for it. Absence means native here too, so a DOWN group
        # is excluded rather than scanned — otherwise a dead group would veto a
        # live group's degradation report (pgw#937).
        served_native = any(
            fd.function_name not in self._group_fn_degraded.get(s.ordinal, {})
            and fd.function_name not in self._group_fn_unavail.get(s.ordinal, {})
            for s in live
        )
        worker_level = merge.worker_fn_degraded(
            per_group, served_native_somewhere=served_native
        )
        if worker_level is None:
            return None
        return pb.WorkerMessage(fn_degraded=worker_level)

    # ---- per-job capability policy (delta 4) -----------------------------

    async def _authorize_run_job(self, run: pb.RunJob) -> bool:
        """Decide on this job's capability token. False = refused, not relayed."""
        worker_id, _release_id = self._identity()
        decision = capability.decide(
            run.capability_token,
            request_id=run.request_id,
            attempt=run.attempt,
            function_name=run.function_name,
            worker_id=worker_id,
        )
        if decision.note:
            logger.warning(
                "job %s#%d: %s", run.request_id, run.attempt, decision.note
            )
            self.capability_notes += 1
        if decision.forward:
            return True
        self.capability_withheld += 1
        logger.error(
            "WITHHELD the capability token for %s#%d: %s",
            run.request_id, run.attempt, decision.reason,
        )
        run.capability_token = ""
        await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
            request_id=run.request_id,
            attempt=run.attempt,
            status=(
                pb.JOB_STATUS_RETRYABLE if decision.retryable
                else pb.JOB_STATUS_FATAL
            ),
            safe_message=(
                f"CapabilityWithheld: {decision.reason} (the control parent "
                "refused to hand this grant to handler code)"
            )[:512],
        )))
        await self._report_capability_withheld(run, decision)
        return False           # answered typed; never relayed

    async def _report_capability_withheld(
        self, run: pb.RunJob, decision: "capability.Decision",
    ) -> None:
        now = time.monotonic()
        if now - self._last_capability_report_at < _CAPABILITY_REPORT_MIN_INTERVAL_S:
            return
        self._last_capability_report_at = now
        await self._dial_detail(
            f"phase=compute_capability_withheld request={run.request_id} "
            f"attempt={run.attempt} function={run.function_name} "
            f"withheld={self.capability_withheld} retryable={decision.retryable} "
            f"reason={decision.reason[:400]} — the control parent decides which "
            "per-job grant reaches handler code"
        )

    # ---- billing attestation (delta 3) -----------------------------------

    async def _attest_result(self, result: pb.JobResult, slot: _ChildSlot) -> None:
        """Replace the child's self-reported billables with what the parent
        observed, for the quantities the parent can observe."""
        obs = self._observations.pop((result.request_id, result.attempt), None)
        if obs is None or not result.HasField("metrics"):
            return
        proc = slot.proc
        rss = self._child_rss(proc.pid) if proc is not None else 0
        try:
            divergences = attest.attest(
                result.metrics, obs,
                now=time.monotonic(),
                child_rss_bytes=rss,
                status_ok=result.status == pb.JOB_STATUS_OK,
            )
        except Exception:
            logger.warning("metric attestation failed; relaying unattested",
                           exc_info=True)
            return
        if not divergences:
            return
        self.metric_divergences += 1
        logger.warning(
            "job %s#%d: the child's billing self-report diverged from the "
            "parent's observation: %s",
            result.request_id, result.attempt, "; ".join(divergences),
        )
        now = time.monotonic()
        if now - self._last_attestation_report_at < _ATTESTATION_REPORT_MIN_INTERVAL_S:
            return
        self._last_attestation_report_at = now
        await self._dial_detail(
            f"phase=compute_billing_attestation request={result.request_id} "
            f"attempt={result.attempt} function={obs.function} "
            f"divergences={len(divergences)} total={self.metric_divergences} "
            f"detail={'; '.join(divergences)[:600]} — billable quantities are "
            "attested by the control parent, not by the code being billed"
        )

    @staticmethod
    def _child_rss(pid: int) -> int:
        try:
            import psutil

            return int(psutil.Process(pid).memory_info().rss)
        except Exception:
            return 0

    # ---- parent-mediated actions (delta 1) -------------------------------

    async def _serve_action(self, link: _ChildLink, req: Dict[str, Any]) -> None:
        """Decide and perform ONE action a child asked for."""
        rid = req.get("id")
        try:
            async with self._action_slots:
                result = await self._perform_action(req)
        except actions.ActionRefused as exc:
            self.actions_refused += 1
            logger.error(
                "REFUSED parent-mediated action from a compute child: %s", exc
            )
            await self._report_action_refusal(str(exc))
            reply: Dict[str, Any] = {"id": rid, "ok": False, "error": str(exc)}
        except Exception as exc:
            logger.warning("parent-mediated action failed", exc_info=True)
            reply = {"id": rid, "ok": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        else:
            reply = {"id": rid, "ok": True, **result}
        try:
            await link.writer.frame(frames.T_ACTION_RESP, frames.pack_meta(reply))
        except (ConnectionError, OSError):
            pass

    async def _perform_action(self, req: Dict[str, Any]) -> Dict[str, Any]:
        named = str(req.get("action") or "")
        if named:
            if named == actions.ACTION_VIEWER_IDENTITY:
                return {"result": self._viewer_identity()}
            if named != actions.ACTION_REPORT_DETAIL:
                raise actions.ActionRefused(f"unknown action {named!r}")
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
        # The child's number is advisory and may only ever LOWER the call's
        # budget; the allowlist's own `timeout_s` is the ceiling.
        timeout = min(float(req.get("timeout") or action.timeout_s),
                      action.timeout_s)
        status, text = await asyncio.to_thread(
            _http_call, action.method, base + str(req.get("path") or ""),
            token, query, body, timeout,
        )
        logger.info(
            "parent-mediated %s -> %d (child holds no credential)", action.name, status
        )
        return self._post_action(action, status, text)

    def _viewer_identity(self) -> Dict[str, str]:
        """pgw#1122: name this pod for the child — the CLAIMS, not the token.

        The compute child holds no credential by construction, so it cannot
        decode its own identity; the receipt trust gate that tried refused
        every org-tier cell on every real serving pod. The parent holds the
        credential and answers from it, exactly as it does for the resolve and
        the publish. Nothing in the request is read: the child names no field
        here, so it cannot ask to be somebody else.
        """
        token = (self.transport.current_worker_jwt or "").strip()
        if not token:
            token = (self._settings.bootstrap_worker_jwt or "").strip()
        if not token:
            raise actions.ActionRefused(
                f"{actions.ACTION_VIEWER_IDENTITY}: this pod holds no worker "
                "credential, so nothing here can name the endpoint or org it "
                "serves")
        identity = worker_identity.from_token(token)
        return {
            "endpoint_id": identity.endpoint_id,
            "org_id": identity.org_id,
        }

    def _narrow_job_scoped_action(
        self, action: "actions.HubAction", body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """The narrowing that needs PARENT STATE: the parent will not renew a
        capability for a request it never dispatched (on any group)."""
        if action.name != "capability.renew":
            return body
        rid = str(body.get("request_id") or "")
        try:
            attempt = int(body.get("attempt") or 0)
        except (TypeError, ValueError):
            attempt = -1
        if not self._is_in_flight(rid, attempt):
            raise actions.ActionRefused(
                f"capability.renew for {rid}#{attempt}: not an in-flight job on "
                "this worker — the parent renews only what it dispatched"
            )
        return body

    def _post_action(
        self, action: "actions.HubAction", status: int, text: str,
    ) -> Dict[str, Any]:
        """Last look at a response before it crosses back (delta 4)."""
        if action.name != "capability.renew" or status != 200:
            return {"status": int(status), "body": text}
        try:
            renewed = str((json.loads(text or "{}") or {}).get("capability_token") or "")
        except ValueError:
            renewed = ""
        if renewed:
            rid, attempt = capability.scope_of(renewed)
            if rid and not self._is_in_flight(rid, attempt):
                self.capability_withheld += 1
                logger.error(
                    "WITHHELD a renewed capability token scoped to %s#%d, which "
                    "is not in flight on this worker", rid, attempt,
                )
                raise actions.ActionRefused(
                    f"the hub returned a capability token scoped to {rid}#{attempt}, "
                    "which this worker is not running — not handing it to the child"
                )
        return {"status": int(status), "body": text}

    async def _report_action_refusal(self, detail: str) -> None:
        now = time.monotonic()
        if now - self._last_action_refusal_report_at < _ACTION_REFUSAL_REPORT_MIN_INTERVAL_S:
            return
        self._last_action_refusal_report_at = now
        await self._dial_detail(
            f"phase=compute_action_refused refusals={self.actions_refused} "
            f"detail={detail[:400]} — a compute child asked the control parent "
            "for authority outside the allowlisted action table"
        )

    # ---- terminal boot outcomes (pgw#826) --------------------------------

    async def _fail_boot_fatal(self, slot: _ChildSlot, fatal: Dict[str, Any]) -> None:
        """A terminal typed boot verdict: relay the child's HardwareUnsuitable
        report on the parent's credential and exit 1 — a hardware verdict holds
        for every child this parent could ever spawn."""
        report_raw = fatal.get("report") or {}
        reason_class = str(report_raw.get("reason_class") or "unknown")
        delivered = False
        try:
            from ..hardware_report import HardwareReport, deliver_hardware_report

            report = msgspec.convert(report_raw, HardwareReport, strict=False)
            delivered = await asyncio.to_thread(
                deliver_hardware_report, self._settings, report
            )
        except Exception:
            logger.warning("terminal boot report relay failed", exc_info=True)
        await self._dial_detail(
            f"phase=compute_boot_fatal group={slot.ordinal} terminal=true "
            f"reason_class={reason_class} report_delivered={delivered} "
            f"spawns={slot.spawn_count} — a hardware verdict is not a "
            "transient fault; the parent exits 1 instead of respawning"
        )
        self._give_up(f"boot_fatal:{reason_class}")

    async def _fail_boot_loop(self, slot: _ChildSlot, cause: str) -> None:
        """N consecutive pre-Hello deaths: the child has never served and never
        will — report typed and exit 1 rather than billing a respawn loop."""
        tail = slot.stderr_tail_text(1500)
        await self._dial_detail(
            f"phase=compute_boot_crash_loop group={slot.ordinal} "
            f"deaths_before_hello={slot.deaths_before_hello} "
            f"limit={self._boot_death_limit} last_cause={cause} "
            f"spawns={slot.spawn_count} — a child that repeatedly dies before "
            "Hello will never serve; the parent exits 1 instead of respawning"
            + (f"\nlast child stderr:\n{tail}" if tail else "")
        )
        self._give_up(f"boot_crash_loop:{cause}")

    def _give_up(self, reason: str) -> None:
        logger.error("control parent giving up (%s): exiting 1, no respawn", reason)
        self._terminal_exit = True
        self.terminal_exit_reason = reason
        self._stopping.set()
        for slot in self._slots:
            proc = slot.proc
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        self.transport.stop()

    async def _dial_detail(self, detail: str) -> None:
        logger.error("compute.postmortem\n%s", detail)
        try:
            delivered = await asyncio.to_thread(
                worker_fatal.report_worker_detail, self._settings, detail
            )
            logger.info("compute post-mortem wire report delivered=%s", delivered)
        except Exception:
            logger.warning("compute post-mortem wire report failed", exc_info=True)

    # ---- the app beat (pgw#771) ------------------------------------------

    async def _beat_loop(self) -> None:
        """The PARENT originates the app heartbeat — one for the whole worker.

        At G>1 it re-sends the MERGED worker state (the union/aggregate of every
        group's latest), so a single group cannot make the worker's beat regress
        or advance on its own. The claim is "the worker is alive and reachable",
        made by the control plane that nothing tenant-side can starve.
        """
        while not self._stopping.is_set():
            interval = (
                self._beat_interval if self._beat_interval > 0
                else _BEAT_INTERVAL_FALLBACK_S)
            await self._sleep_or_stop(max(0.25, interval / 2.0))
            if self._stopping.is_set() or self._child_exited_clean:
                return
            msg = self._last_state_delta
            if msg is None or not self._any_link() or not self.transport.connected:
                continue
            if time.monotonic() - self._last_state_delta_at < interval:
                continue  # a child is beating for itself
            self._last_state_delta_at = time.monotonic()
            self.parent_beats_sent += 1
            try:
                await self.transport.send(msg)
            except Exception:
                logger.debug("parent beat send failed", exc_info=True)

    def _any_link(self) -> bool:
        return any(slot.link is not None for slot in self._slots)

    # ---- drain / signals -------------------------------------------------

    async def _sleep_or_stop(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), delay)
        except asyncio.TimeoutError:
            pass

    async def _finish_shutdown_flush(self, *, reason: str) -> None:
        """Bounded flush of the durable queue on a deliberate parent exit. Runs
        ONCE however many groups reach it."""
        if self._shutdown_flushed:
            self._stopping.set()
            return
        self._shutdown_flushed = True
        self._child_exited_clean = True   # the shutdown was deliberate, not a crash
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
                f"groups={self.execution_groups}"
            )
        self._stopping.set()

    async def _drain_without_child(self) -> None:
        # A child reaped moments ago may still be attributing its in-flight
        # jobs. Flush must not retire the queue empty before that FATAL is in it.
        waits = [slot.death_report_done.wait() for slot in self._slots]
        try:
            await asyncio.wait_for(
                asyncio.gather(*waits), _LINK_SETTLE_TIMEOUT_S + 2.0
            )
        except asyncio.TimeoutError:
            logger.warning("draining without waiting further for death attribution")
        await self._finish_shutdown_flush(reason="drain_without_child")

    def _forward_signal(self, signum: int) -> None:
        # Mark intent BEFORE the signals land: the children's deaths by this
        # signal are deliberate, so they must not respawn, count toward the
        # crash-loop window, or exit the parent 1.
        self._terminating = True
        any_proc = False
        for slot in self._slots:
            proc = slot.proc
            if proc is not None:
                any_proc = True
                try:
                    proc.send_signal(signum)
                except ProcessLookupError:
                    pass
        if any_proc:
            if self._stop_deadline_task is None:
                self._stop_deadline_task = asyncio.create_task(
                    self._stop_deadline(), name="parent-stop-deadline"
                )
            return
        # No children to drain: flush and stop.
        self._draining = True
        asyncio.create_task(self._drain_without_child(), name="signal-drain")

    async def _stop_deadline(self) -> None:
        await self._await_all_children_exit(self._stop_timeout)

    async def _await_all_children_exit(self, timeout: float) -> bool:
        results = await asyncio.gather(
            *(slot.await_exit(timeout) for slot in self._slots)
        )
        return all(results)

    # ---- run -------------------------------------------------------------

    def stop(self) -> None:
        """Thread-safe stop (tests / embedding)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        def _stop() -> None:
            self._stopping.set()
            self.transport.stop()
            for slot in self._slots:
                proc = slot.proc
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
        # pgw#639 under the split: SIGUSR2 dumps every process's stacks and
        # kills none. The forward is installed first, then faulthandler with
        # chain=True, so one signal yields parent + children stacks in the pod
        # log (children inherit stderr and register their own dump handler).
        def _forward_usr2(signum: int, _frame: object) -> None:
            for slot in self._slots:
                proc = slot.proc
                if proc is not None and proc.pid is not None:
                    try:
                        os.kill(proc.pid, signum)
                    except (ProcessLookupError, OSError):
                        pass
        try:
            signal.signal(signal.SIGUSR2, _forward_usr2)
            faulthandler.register(signal.SIGUSR2, all_threads=True, chain=True)
        except (AttributeError, ValueError, OSError):
            pass
        # One unix server per group's child socket.
        for slot in self._slots:
            await slot.start_server()
        # delta 2: measure the host BEFORE any endpoint import can have happened.
        self._measure_task = asyncio.create_task(
            self._measure_host(), name="parent-measure"
        )
        transport_task = asyncio.create_task(self.transport.run(), name="parent-transport")
        child_tasks = [
            asyncio.create_task(slot.child_loop(), name=f"parent-child-loop-{slot.label}")
            for slot in self._slots
        ]
        watchdog_tasks = [
            asyncio.create_task(slot.watchdog_loop(), name=f"parent-watchdog-{slot.label}")
            for slot in self._slots
        ]
        beat_task = asyncio.create_task(self._beat_loop(), name="parent-beat")
        # The worker is done when the stream ends, or when EVERY group's
        # supervision loop has exited (which only happens on a deliberate,
        # worker-wide shutdown — a single group respawns forever).
        # gather() returns a Future, already awaitable — asyncio.wait accepts it.
        children_done = asyncio.gather(*child_tasks)
        try:
            done, _ = await asyncio.wait(
                (transport_task, children_done), return_when=asyncio.FIRST_COMPLETED
            )
            if transport_task in done:
                transport_task.result()  # re-raise FatalTransportError
                if (self._draining or self._terminating) and not children_done.done():
                    await self._await_all_children_exit(self._stop_timeout)
                    try:
                        await asyncio.wait_for(asyncio.shield(children_done), 15.0)
                    except asyncio.TimeoutError:
                        logger.warning("child supervision loops did not settle after drain")
                    except Exception:
                        pass
            else:
                # Supervision loops finished first — a deliberate exit, and the
                # queue is retired through the send loop. Let the transport end
                # its clean half-close rather than RST it.
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
                    pass
            if self._terminal_exit:
                return 1   # pgw#826: a terminal boot outcome, already reported
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
                        f"groups={self.execution_groups}"
                    )
            self._stopping.set()
            for slot in self._slots:
                proc = slot.proc
                if proc is not None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            self.transport.stop()
            tasks: List["asyncio.Future[Any]"] = [
                transport_task, children_done, beat_task, *watchdog_tasks
            ]
            for extra in (self._stop_deadline_task, self._measure_task):
                if extra is not None:
                    tasks.append(extra)
            for slot in self._slots:
                if slot.liveness_task is not None:
                    tasks.append(slot.liveness_task)
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for slot in self._slots:
                await slot.close_server()

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
    postmortem.clear_all_inflight()
    postmortem.write_boot_record()
    # §1.18: the bootstrap-owned load for the CONTROL-PARENT process entry.
    settings = config.install(config.load_settings())
    # The boot credential is installed by ParentControl.__init__, which is the
    # seam every parent goes through — not just this entry point.
    code = ParentControl(settings).run()
    if code == 0:
        postmortem.clear_boot_record()
        postmortem.clear_all_inflight()
    return code
