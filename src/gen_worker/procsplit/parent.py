"""Parent (control-plane) side of the process split, generalised to N execution groups."""

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
from .. import hostfacts
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
    EXIT_JOB_RECYCLE,
    actions,
    attest,
    capability,
    frames,
    liveness,
    merge,
    privdrop,
    procdiag,
)
from .group import ChildGroup, GroupPlan
from .seam import SeamAccountant

logger = logging.getLogger(__name__)

DEATH_LABEL = "ComputeProcessDied"

_DEFAULT_START_LIMIT_BURST = 3
_DEFAULT_START_LIMIT_INTERVAL_S = 600.0
_DEFAULT_BOOT_DEATH_LIMIT = 3
# pgw#1630: A SAMPLING CADENCE, NOT A VERDICT INPUT. Nothing decides a child's
# life from it; the watchdog loop divides it by four to choose how often to read
# /proc. The flatness window that DOES decide is derived per child from that
# child's own observed inter-progress gaps (procsplit/liveness.py).
_DEFAULT_WATCHDOG_BUDGET_S = 60.0
_DEFAULT_RESPAWN_BACKOFF_BASE_S = 1.0
_DEFAULT_RESPAWN_BACKOFF_CAP_S = 60.0
_BACKOFF_RESET_AFTER_ALIVE_S = 60.0
_CRASH_LOOP_REPORT_MIN_INTERVAL_S = 300.0
_NEVER_REPORTED = float("-inf")
_DEATH_FLUSH_GRACE_S = 2.0
_DEFAULT_STOP_TIMEOUT_S = 120.0
_STOP_FLUSH_TIMEOUT_S = 30.0
_LINK_SETTLE_TIMEOUT_S = 3.0
_CLEAN_CLOSE_WAIT_S = 12.0
_REPORTED_DEAD_CAP = 512
_EVIDENCE_EPS = 0.05
_CHILD_FORBIDDEN_ENVS = ("WORKER_JWT", "RUNPOD_API_KEY", "PUBLIC_KEY")
_ACTION_REFUSAL_REPORT_MIN_INTERVAL_S = 300.0
_MAX_CONCURRENT_ACTIONS = 16
_BEAT_INTERVAL_FALLBACK_S = 10.0
_MEASURE_TIMEOUT_S = 180.0
_CENSUS_SPAWNS = 3
_CENSUS_SPAWN_BACKOFF_S = 2.0
_MEASURE_BEFORE_SPAWN_S = 60.0
_ATTESTATION_REPORT_MIN_INTERVAL_S = 300.0
_CAPABILITY_REPORT_MIN_INTERVAL_S = 300.0
_OBSERVATION_CAP = 512
_WORKER_SCOPED_MSGS = frozenset(
    {"state_delta", "activity_update", "fn_unavailable", "fn_degraded"}
)
_STDERR_TAIL_CAP_BYTES = 32768
_STDERR_TAIL_DIAL_CHARS = 3000
_COMPUTE_HOME = "/var/lib/gen-worker/compute"
_DEFAULT_TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"

_COMPILED_GRAPH_STORE_DIRNAME = "compiled-graph-store"
_DEFAULT_CONFIG_SNAPSHOT_PATH = "/app/.tensorhub/runtime_config.msgpack"


def _tee_stderr_chunk(chunk: bytes) -> None:
    try:
        buf = getattr(sys.stderr, "buffer", None)
        if buf is not None:
            buf.write(chunk)
            buf.flush()
        else:
            sys.stderr.write(chunk.decode("utf-8", errors="replace"))
            sys.stderr.flush()
    except Exception:
        pass


def _close_transport(proc: asyncio.subprocess.Process) -> None:
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
    import requests

    resp = requests.request(
        method,
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=query or None,
        json=body if method in ("POST", "PUT", "PATCH") else None,
        timeout=timeout,
    )
    return resp.status_code, resp.text


_set_pdeathsig = privdrop.set_pdeathsig


def is_grpc_fork_abort(
    *, cause: str, saw_hello: bool, oom_delta: int, stderr_tail: str,
) -> bool:
    """The written discriminator for gRPC's fork abort: gRPC registers pthread_atfork handlers that SKIP when another thread is inside gRPC, and the forked-but-not-yet-exec'd child then aborts out of the polling engine on an fd it must not touch — self-inflicted, nothing about the pod, image, card or tenant implicated, and it has been re-diagnosed from first principles at least five times. Deliberately NOT keyed on the ~0.8 s lifetime (a duration inside a classifier is magic-timeout-as-evidence and would misclassify on a slower box). Narrow on purpose: a SIGABRT with no Hello and no OOM that does not carry gRPC's fork marks is a different, unexplained defect and must keep saying so."""
    if saw_hello or oom_delta > 0 or cause != "signal:SIGABRT":
        return False
    tail = stderr_tail or ""
    if "fork_posix" in tail or "skipping fork() handlers" in tail:
        return True
    return "Epoll1Poller" in tail and "Bad file descriptor" in tail


class _ChildLink:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = frames.FrameWriter(writer)
        self.saw_hello = False


class _ChildSlot:

    def __init__(self, parent: "ParentControl", group: ChildGroup) -> None:
        self.p = parent
        self.ordinal = group.ordinal
        self.devices = group.devices
        self.socket_path = group.socket_path
        self.group_env: Dict[str, str] = dict(group.env)
        self.inflight_marker_path = (
            postmortem.group_inflight_path(group.ordinal, parent._postmortem_dir)
            if parent._topology.execution_groups > 1 else None
        )
        self.fault_dump_path = (
            postmortem.group_fault_dump_path(group.ordinal, parent._postmortem_dir)
            if parent._topology.execution_groups > 1 else None
        )
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
        self.in_flight: Dict[Tuple[str, int], str] = {}
        self.death_times: collections.deque = collections.deque(maxlen=64)
        self.deaths_before_hello = 0
        self.child_saw_hello = False
        self.boot_fatal: Optional[Dict[str, Any]] = None
        self.stderr_tail: collections.deque = collections.deque()
        self.stderr_tail_len = 0
        self.stderr_task: Optional[asyncio.Task] = None
        self.spawn_count = 0
        self.last_frame_at = time.monotonic()
        self.relaying = False
        self.watchdog_fired = False
        # pgw#1630: THE ONE INPUT to the kill decision. Kernel-accounted
        # evidence, its observation-derived flatness window, and this child's
        # position on the report -> diagnose -> TERM -> KILL ladder.
        self.evidence = liveness.EvidenceTrack(
            floor_s=parent._liveness_floor_s, eps=_EVIDENCE_EPS,
        )
        # Liveness (thread-sourced, loop-independent), per child. Everything
        # below is TELEMETRY since pgw#1630: it labels a stall report and
        # decides nothing. `liveness_evidence*` mirror `self.evidence` for the
        # existing readers.
        self.liveness_task: Optional[asyncio.Task] = None
        self.last_liveness_at = 0.0
        self.liveness_evidence: Optional[float] = None
        self.liveness_evidence_at = 0.0
        self.liveness_activity = ""
        # This group's freshest published StateDelta; the worker-level beat
        # re-sends the merge of all groups'.
        self.last_state_delta: Optional[pb.WorkerMessage] = None
        self.last_state_delta_at = 0.0
        self.generation = 0
        self.participating = True
        self.last_crash_loop_report_at = _NEVER_REPORTED
        self.link_closed = asyncio.Event()
        self.link_closed.set()
        self.death_report_done = asyncio.Event()
        self.death_report_done.set()
        self.reported_dead: collections.OrderedDict = collections.OrderedDict()

    @property
    def label(self) -> str:
        return f"g{self.ordinal}"

    def begin_generation(self) -> None:
        """A NEW incarnation of this group starts speaking."""
        self.generation += 1
        self.participating = True

    async def start_server(self) -> None:
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        self.server = await asyncio.start_unix_server(
            self._on_child_connect, path=self.socket_path
        )
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
                self.participating = False
                self.p._note_state_delta()
            waiter = self.hello_waiter
            if waiter is not None and not waiter.done():
                waiter.set_exception(ConnectionError("compute child link lost"))
            link.writer.close()
            self.link_closed.set()

    async def _on_child_frame(self, link: _ChildLink, ftype: int, payload: bytes) -> None:
        if ftype == frames.T_WATCHDOG:
            return
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
                await self.p._attest_result(r, self)
            elif which == "state_delta" and self.participating:
                self.last_state_delta = msg
                self.last_state_delta_at = time.monotonic()
                self.p._note_state_delta()
            self.relaying = True
            try:
                out = self.p._fan_in(self, msg)
                if out is not None:
                    self.p.seam.record(which or "", len(payload), group=self.ordinal)
                    await self.p.transport.send(out)
            finally:
                self.relaying = False
                self.last_frame_at = time.monotonic()
            return
        if ftype == frames.T_ACTION_REQ:
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
            self.boot_fatal = frames.unpack_meta(payload)
            report = (self.boot_fatal or {}).get("report") or {}
            logger.error(
                "compute child %s reported a TERMINAL boot verdict: kind=%s "
                "reason_class=%s", self.label,
                (self.boot_fatal or {}).get("kind"), report.get("reason_class"),
            )
            try:
                await link.writer.frame(frames.T_BOOT_FATAL_ACK, frames.pack_meta({}))
            except Exception:
                logger.debug("boot-fatal ack write failed (child may already "
                             "be gone)", exc_info=True)
            return
        if ftype == frames.T_FLUSH_REQ:
            meta = frames.unpack_meta(payload)
            timeout = meta.get("timeout")
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

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        env.update(self.p._child_env)
        env.update(self.group_env)
        for name in _CHILD_FORBIDDEN_ENVS:
            env.pop(name, None)
        if self.p._drop_plan is not None:
            env.update(privdrop.child_env(self.p._drop_plan))
        worker_id, release_id = self.p._identity()
        if worker_id:
            env["WORKER_ID"] = worker_id
        if release_id:
            env["WORKER_RELEASE_ID"] = release_id
        env[ENV_CHILD] = "1"
        env[ENV_SOCKET] = self.socket_path
        env[ENV_SESSION_ID] = self.p._worker_session_id
        env["GEN_WORKER_SUPERVISOR"] = "0"
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
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=(
                    privdrop.preexec(plan) if sys.platform == "linux" else None
                ),
            )
        finally:
            os.close(write_fd)
        self._start_stderr_pump(proc)
        await self._start_liveness_reader(read_fd)
        return proc

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
        """Supervise THIS group's child: spawn, wait, attribute, respawn."""
        p = self.p
        backoff = p._backoff_base
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
            # A RESPAWN is a new process: its predecessor's high-water evidence,
            # its demonstrated gaps and its ladder position all describe a pid
            # that no longer exists.
            self.evidence = liveness.EvidenceTrack(
                floor_s=p._liveness_floor_s, eps=_EVIDENCE_EPS,
            )
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
            _close_transport(proc)
            if p._stopping.is_set():
                self.death_report_done.set()
                return
            deliberate = p._terminating or (rc == 0 and not self.watchdog_fired)
            if deliberate:
                await self._finish_deliberate_exit(rc, lifetime_s=lifetime)
                return
            if (rc == EXIT_JOB_RECYCLE and saw_hello and not self.watchdog_fired
                    and not p._draining):
                await self._report_in_flight_dead("job_recycle")
                postmortem.clear_inflight(path=self.inflight_marker_path)
                logger.info(
                    "compute child %s recycled after a job (run-once "
                    "lifecycle, rc=%s); respawning", self.label, rc)
                backoff = p._backoff_base
                continue
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
        died_jobs = dict(self.in_flight)
        self.in_flight.clear()
        try:
            for (rid, att), fn in sorted(died_jobs.items()):
                self.reported_dead[(rid, att)] = fn
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

        died_jobs = await self._report_in_flight_dead(cause)
        logger.error(
            "compute child %s died: cause=%s rc=%s lifetime=%.1fs in_flight=%s "
            "(respawning ITS group; stream identity kept, siblings untouched)",
            self.label, cause, rc, lifetime_s,
            sorted(r for r, _ in died_jobs) or "none",
        )

        for out in p._retire_group_generation(self):
            try:
                await p.transport.send(out)
            except Exception:
                logger.debug("retirement message send failed", exc_info=True)

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
        stderr_tail = self.stderr_tail_text()
        known = is_grpc_fork_abort(
            cause=cause, saw_hello=saw_hello, oom_delta=oom_delta,
            stderr_tail=stderr_tail,
        )
        if known:
            logger.error(
                "compute child %s: this is pgw#932 — gRPC's fork handlers were "
                "skipped because another thread was inside gRPC, and the "
                "forked child aborted out of the polling engine before exec. "
                "SELF-INFLICTED and transient: nothing about this pod, image, "
                "card or tenant is implicated, the respawn below is the "
                "correct response, and a CI run that fails on it wants a "
                "rerun, not a diagnosis. The real fix is the exec-first "
                "launcher, gated on pgw#909.", self.label,
            )
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
                **({"known_defect": "pgw#932:grpc_fork_abort"} if known else {}),
                **({"child_stderr_tail": stderr_tail} if stderr_tail else {}),
            },
        )
        await p._dial_detail(detail)

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

        try:
            await p.transport.queue.wait_empty(timeout=_DEATH_FLUSH_GRACE_S)
        except Exception:
            pass
        if p._draining or p._terminating or p._stopping.is_set():
            return cause
        if p.execution_groups == 1:
            p.transport.cycle_connection()
        return cause

    async def watchdog_loop(self) -> None:
        """KERNEL EVIDENCE DECIDES, and nothing else is an input (pgw#1630).

        The parent witnesses THIS child's /proc, because a child starved of the
        GIL cannot witness for itself. Evidence advancing means HELD,
        unconditionally: the parent kills only what is provably NOT RUNNING, and
        a child that runs but serves nothing is the hub's stall clock to reap.
        Loop pings and declared activities are REPORTING ONLY — they label a
        stall, they do not decide one. Ladder in procsplit/liveness.py. One
        child's kill never touches a sibling.
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
            await self._walk_liveness_ladder(proc, now)

    def _child_evidence(self, pid: int) -> Optional[float]:
        return proc_evidence.tree_evidence(pid)

    def _sample_child_evidence(self, pid: int, now: float) -> None:
        """One sample into the ladder. A `None` reading is an instrument
        failure, never a flat reading — `EvidenceTrack.observe` keeps the two
        apart, which is what stops an unreadable `/proc` accruing flatness
        against a healthy child."""
        self.evidence.observe(self._child_evidence(pid), now)
        # Mirrored for the reporters and for every existing reader; the ladder
        # itself reads only `self.evidence`.
        self.liveness_evidence = self.evidence.value
        self.liveness_evidence_at = self.evidence.advanced_at

    async def _walk_liveness_ladder(self, proc: Any, now: float) -> None:
        """report -> diagnose -> SIGTERM -> SIGKILL, one window apart.

        Every rung requires flatness to have CONTINUED: the rung is recomputed
        from `flat_for(now)` on every sample and any evidence advance clears the
        fired set, so a wedge that un-wedges mid-ladder was never a wedge and
        starts again from zero if it re-wedges.
        """
        rung = self.evidence.verdict(now)
        if rung == liveness.RUNG_ALIVE:
            return
        if rung == liveness.RUNG_UNMEASURABLE:
            # Absence of instrument is not guilt. This branch REPLACES
            # `no_evidence_source -> KILL`.
            if self.evidence.claim(rung):
                logger.warning(
                    "compute child %s: cannot read kernel evidence (%s) — "
                    "HOLDING and reporting; failure to measure is never a kill",
                    self.label, self.evidence.describe(now),
                )
                await self.p._dial_detail(
                    f"phase=compute_liveness_unmeasurable group={self.ordinal} "
                    f"{self.evidence.describe(now)} {self._liveness_labels(now)} "
                    "— the parent cannot measure this child; holding, never killing"
                )
            return

        if rung == liveness.RUNG_REPORT:
            if self.evidence.claim(rung):
                logger.warning(
                    "compute child %s has accrued NO kernel-accounted work for "
                    "%.1fs (window %.1fs) — reporting the stall (stream, beat and "
                    "process all kept)",
                    self.label, self.evidence.flat_for(now), self.evidence.window_s,
                )
                await self.p._dial_detail(
                    f"phase=compute_child_stalled group={self.ordinal} "
                    f"{self.evidence.describe(now)} {self._liveness_labels(now)} "
                    "— measured by the parent from /proc, not self-reported by "
                    "the child; the labels are TELEMETRY and decided nothing"
                )
            return

        if rung == liveness.RUNG_DIAGNOSE:
            if self.evidence.claim(rung):
                report = await asyncio.to_thread(
                    procdiag.capture, int(proc.pid), self.p._postmortem_dir,
                )
                logger.error(
                    "compute child %s flat for %.1fs (2x window) — captured a "
                    "diagnosis before anything is signalled:\n%s",
                    self.label, self.evidence.flat_for(now), report,
                )
                await self.p._dial_detail(
                    f"phase=compute_liveness_diagnosis group={self.ordinal} "
                    f"{self.evidence.describe(now)} {self._liveness_labels(now)}\n"
                    f"{report}"
                )
            return

        if rung == liveness.RUNG_TERM:
            if self.evidence.claim(rung):
                logger.error(
                    "compute child %s flat for %.1fs (3x window %.1fs) — SIGTERM; "
                    "it gets one more window to unwind before SIGKILL",
                    self.label, self.evidence.flat_for(now), self.evidence.window_s,
                )
                await self.p._dial_detail(
                    f"phase=compute_liveness_term group={self.ordinal} "
                    f"{self.evidence.describe(now)} {self._liveness_labels(now)} "
                    "— evidence has been flat for three consecutive windows; "
                    "asking the child to exit"
                )
                self.watchdog_fired = True
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            return

        # RUNG_KILL — four windows flat, and the SIGTERM went unanswered.
        if not self.evidence.claim(rung):
            return
        logger.error(
            "compute child %s flat for %.1fs (4x window %.1fs) and unresponsive "
            "to SIGTERM — SIGKILL. %s",
            self.label, self.evidence.flat_for(now), self.evidence.window_s,
            self.evidence.describe(now),
        )
        await self.p._dial_detail(
            f"phase=compute_liveness_kill group={self.ordinal} "
            f"{self.evidence.describe(now)} {self._liveness_labels(now)} "
            "— provably NOT RUNNING: no CPU second and no byte of I/O across "
            "four consecutive observation-derived windows"
        )
        self.watchdog_fired = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def _liveness_labels(self, now: float) -> str:
        """The cooperative signals, as LABELS on a report.

        pgw#1630 demoted every one of these. They make a stall report and the
        hub's attribution honest — "flat while `boot_materialize` was open, with
        two jobs in flight" is a far better bug than "flat" — and they decide
        nothing. A path that forgets to declare an activity loses a label, not a
        process.
        """
        return (
            f"label_activity={self.liveness_activity or 'none'} "
            f"label_in_flight={sorted(f'{r}#{a}' for (r, a) in self.in_flight)} "
            f"label_loop_silent_s={now - self.last_frame_at:.1f} "
            f"label_ping_age_s={now - self.last_liveness_at:.1f}"
        )


class ParentControl:
    """The control process: real Transport + the security boundary + supervision of a GROUP of compute children (one ``_ChildSlot`` per execution group)."""

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
        # pgw#1630: the flatness FLOOR, in seconds. 0 = read `Settings`, then
        # the module default. Deliberately SEPARATE from `watchdog_budget_s`,
        # which is now only a /proc sampling cadence: narrowing how often the
        # parent LOOKS must not silently narrow how long a child may be flat,
        # because those are answers to different questions and conflating them
        # is how the old single-budget cliff worked.
        liveness_floor_s: float = 0.0,
        stop_timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
        stop_flush_timeout_s: float = _STOP_FLUSH_TIMEOUT_S,
        beat_interval_s: float = 0.0,
        transport_backoff_base_s: float = 1.0,
        transport_backoff_cap_s: float = 30.0,
    ) -> None:
        self._settings = settings
        worker_credential.install_bootstrap(settings)
        env_cmd = os.environ.get(ENV_CHILD_CMD, "").strip()
        self._child_cmd = list(
            child_cmd
            if child_cmd is not None
            else (shlex.split(env_cmd) if env_cmd else [sys.executable, "-m", "gen_worker.entrypoint"])
        )
        self._child_env = dict(child_env or {})
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

        self._topology = topology if topology is not None else ExecutionTopology.from_env()
        self._plan = GroupPlan.for_topology(self._topology, socket_path=self._socket_path)

        self._drop_plan = self._prepare_privilege_drop()

        # pgw#1630: the FLOOR under each child's flatness window. The window
        # itself is derived from that child's own observed inter-progress gaps
        # (`procsplit/liveness.py`); this is only the lower bound, and it is a
        # Settings knob because pgw#1613 proved the operator lever must exist.
        # Resolved BEFORE the slots, because each slot builds its own track.
        #
        # The old `_evidence_hold_window = 3 x the child's PING CADENCE` is
        # DELETED with the rung that read it: a ping is a self-report, and how
        # often a child says "still here" says nothing about how long real work
        # can legitimately go without touching the kernel.
        # Precedence: an explicit constructor argument (a harness or an
        # embedder that KNOWS its child's shape), then the operator's Settings,
        # then the derived default.
        floor = float(liveness_floor_s or 0.0)
        if floor <= 0:
            floor = float(getattr(settings, "watchdog_flatness_floor_s", 0.0) or 0.0)
        self._liveness_floor_s = (
            floor if floor > 0 else liveness.DEFAULT_FLATNESS_FLOOR_S
        )

        self._slots: List[_ChildSlot] = [
            _ChildSlot(self, group) for group in self._plan.children
        ]

        self._worker_session_id = uuid.uuid4().hex

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping = asyncio.Event()
        self._beat_interval = beat_interval_s
        # Worker-level beat state: the last (merged) StateDelta and when any
        # group last published, so the beat re-sends the worker's freshest truth.
        self._last_state_delta: Optional[pb.WorkerMessage] = None
        self._last_state_delta_at = 0.0
        self.parent_beats_sent = 0
        self._draining = False
        self._terminating = False
        self._child_exited_clean = False
        self._shutdown_flushed = False
        self.crash_loop_reports = 0
        self._terminal_exit = False
        self.terminal_exit_reason = ""
        self._stop_deadline_task: Optional[asyncio.Task] = None
        self._reported_unretired = False
        self.unretired_results_at_exit = 0
        self.actions_refused = 0
        self._last_action_refusal_report_at = _NEVER_REPORTED
        self._action_slots = asyncio.Semaphore(_MAX_CONCURRENT_ACTIONS)
        self._file_base_url = ""
        self._identity_cache: Optional[Tuple[str, str]] = None
        self._observations: collections.OrderedDict = collections.OrderedDict()
        self.metric_divergences = 0
        self._last_attestation_report_at = _NEVER_REPORTED
        self.capability_withheld = 0
        self.capability_notes = 0
        self._last_capability_report_at = _NEVER_REPORTED
        self._measure_cmd = list(
            measure_cmd
            if measure_cmd is not None
            else [sys.executable, "-m", "gen_worker.procsplit.measure"]
        )
        self._measurement: Optional[Dict[str, Any]] = None
        self._measured = asyncio.Event()
        self._measure_task: Optional[asyncio.Task] = None
        self._group_activities: Dict[int, Dict[str, pb.ActivityUpdate]] = {}
        self._activity_seq = 0
        self._group_fn_unavail: Dict[int, Dict[str, pb.FnUnavailable]] = {}
        self._group_fn_degraded: Dict[int, Dict[str, pb.FnDegraded]] = {}
        self.seam = SeamAccountant()

    @property
    def execution_groups(self) -> int:
        return len(self._slots)

    def _prepare_privilege_drop(self) -> Optional[privdrop.DropPlan]:
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
            self._child_env.get("TENSORHUB_CACHE_DIR", "")
            or self._settings.tensorhub_cache_dir
            or _DEFAULT_TENSORHUB_CACHE_DIR,
            str(postmortem.BOOT_RECORD_PATH.parent),
            os.path.dirname(
                self._child_env.get("GEN_WORKER_BOOT_RECORD", "")
                or self._settings.boot_record_path
            ),
            os.path.join(
                self._child_env.get("TENSORHUB_CACHE_DIR", "")
                or self._settings.tensorhub_cache_dir
                or _DEFAULT_TENSORHUB_CACHE_DIR,
                _COMPILED_GRAPH_STORE_DIRNAME,
            ),
            os.path.dirname(
                self._child_env.get("GEN_WORKER_CONFIG_SNAPSHOT_PATH", "")
                or self._settings.config_snapshot_path
                or _DEFAULT_CONFIG_SNAPSHOT_PATH
            ),
        ]
        granted = privdrop.grant_paths(plan, privdrop.writable_paths(plan, extra))
        privdrop.grant_devices(plan)
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

    @property
    def _proc(self) -> Optional[asyncio.subprocess.Process]:
        return self._slots[0].proc if self._slots else None

    @property
    def _spawn_count(self) -> int:
        return self._slots[0].spawn_count if self._slots else 0

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
        if self.execution_groups == 1:
            return self._slots[0]
        gpu_index = run.compute.gpu_index if run.HasField("compute") else None
        try:
            ordinal = self._plan.route(gpu_index)
        except (ValueError, Exception) as exc:  # noqa: BLE001 - typed refusal below
            logger.error("cannot route dispatch %s: %s", run.request_id, exc)
            return None
        return self._slots[ordinal] if 0 <= ordinal < self.execution_groups else None

    async def _measure_host(self) -> None:
        best: Optional[Dict[str, Any]] = None
        for spawn in range(_CENSUS_SPAWNS):
            await self._measure_host_once()
            current = self._measurement or {}
            if best is None or len(current.get("census_gaps") or []) < len(
                best.get("census_gaps") or []
            ):
                if current or best is None:
                    best = current
            gaps = list(current.get("census_gaps") or [])
            if "capability" not in gaps or spawn == _CENSUS_SPAWNS - 1:
                break
            logger.warning(
                "host census still missing CAPABILITY after spawn %d/%d — "
                "torch freezes a failed CUDA init per PROCESS, so re-measuring "
                "in a FRESH interpreter (pgw#1436)",
                spawn + 1, _CENSUS_SPAWNS,
            )
            await asyncio.sleep(_CENSUS_SPAWN_BACKOFF_S * (spawn + 1))
        if best is not None:
            self._measurement = best

    async def _measure_host_once(self) -> None:
        cmd = list(self._measure_cmd)
        env = dict(os.environ)
        for name in _CHILD_FORBIDDEN_ENVS:
            env.pop(name, None)
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
        m = self._measurement
        if m is None:
            return None
        try:
            hw = msgspec.convert(m.get("hardware") or {}, hostfacts.HostFacts)
        except msgspec.ValidationError:
            logger.error("host measurement is not a HostFacts", exc_info=True)
            hw = hostfacts.HostFacts()
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
            gpu_count=hw.gpu_count,
            vram_total_bytes=hw.vram_total_bytes,
            gpu_name=hw.gpu_name,
            gpu_sm=hw.gpu_sm,
            torch_version=hw.torch_version,
            driver_version=hw.driver_version,
            cuda_version=hw.cuda_version,
            capability_reason_class=str(m.get("capability_reason_class") or ""),
            capability_detail=str(m.get("capability_detail") or ""),
            installed_libs=list(hw.installed_libs),
            gen_worker_version=str(m.get("gen_worker_version") or ""),
            image_digest=self._settings.worker_image_digest,
            instance_id=self._settings.runpod_pod_id or "",
        )

    def _identity(self) -> Tuple[str, str]:
        if self._identity_cache is not None:
            return self._identity_cache
        worker_id = (self._settings.worker_id or "").strip()
        release_id = ""
        token = (self._settings.bootstrap_worker_jwt or "").strip()
        if token:
            try:
                from ..request_context._helpers import _decode_unverified_jwt_claims

                claims = _decode_unverified_jwt_claims(token)
                worker_id = worker_id or str(claims.get("sub") or "").strip()
                release_id = str(claims.get("release_id") or "").strip()
            except Exception:
                logger.warning("could not decode the worker JWT claims", exc_info=True)
        self._identity_cache = (worker_id, release_id)
        return self._identity_cache

    async def _request_slot_hello(self, slot: _ChildSlot) -> Optional[pb.Hello]:
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
        """Assemble the worker's Hello."""
        try:
            await asyncio.wait_for(self._measured.wait(), _MEASURE_TIMEOUT_S + 5.0)
        except asyncio.TimeoutError:
            pass

        if self.execution_groups == 1:
            hello = await self._request_slot_hello(self._slots[0])
            if hello is None:
                return pb.Hello()
            self._apply_identity_and_resources(hello)
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
        worker_id, release_id = self._identity()
        if worker_id:
            hello.worker_id = worker_id
        if release_id:
            hello.release_id = release_id
        resources = self._parent_resources()
        if resources is not None:
            hello.resources.CopyFrom(resources)
            return
        logger.error(
            "no parent-side host measurement is available; this Hello ships "
            "with NO resources — the hub sees an unmeasured host"
        )
        hello.ClearField("resources")

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        if ack.file_base_url:
            self._file_base_url = ack.file_base_url.rstrip("/")
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
                slot = self._slots[0]
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
        if not await self._authorize_run_job(run):
            return
        key = (run.request_id, run.attempt)
        self._observations[key] = attest.JobObservation(
            function=run.function_name,
            relayed_at=time.monotonic(),
            concurrency_at_relay=len(self._all_in_flight()),
        )
        while len(self._observations) > _OBSERVATION_CAP:
            self._observations.popitem(last=False)
        slot.in_flight[key] = run.function_name
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
        """The rotated worker JWT stays HERE (delta 1) — never sent to a child."""

    def _live_slots(self) -> List["_ChildSlot"]:
        return [s for s in self._slots if s.participating]

    def _note_state_delta(self) -> None:
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
        self._last_state_delta = pb.WorkerMessage(
            state_delta=pb.StateDelta(phase=pb.WORKER_PHASE_BOOTING)
        )

    def _retire_group_generation(self, slot: _ChildSlot) -> List[pb.WorkerMessage]:
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
                merged = merge.reconcile_activity_kind(
                    per_group, seq=self._activity_seq
                )
            else:
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
        if self.execution_groups == 1:
            return msg
        which = msg.WhichOneof("msg")
        if which in _WORKER_SCOPED_MSGS and not slot.participating:
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
        live_ordinals = {s.ordinal for s in self._live_slots()}
        per_group = {
            ordinal: kinds[act.kind]
            for ordinal, kinds in self._group_activities.items()
            if act.kind in kinds and ordinal in live_ordinals
        }
        if not per_group:
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
        per_group: Dict[int, Optional[pb.FnUnavailable]] = {}
        for s in self._live_slots():
            per_group[s.ordinal] = self._group_fn_unavail.get(
                s.ordinal, {}
            ).get(fu.function_name)
        worker_level = merge.worker_fn_unavailable(per_group)
        if worker_level is None:
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

    async def _authorize_run_job(self, run: pb.RunJob) -> bool:
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
        return False

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

    async def _attest_result(self, result: pb.JobResult, slot: _ChildSlot) -> None:
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

    async def _serve_action(self, link: _ChildLink, req: Dict[str, Any]) -> None:
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

    async def _fail_boot_fatal(self, slot: _ChildSlot, fatal: Dict[str, Any]) -> None:
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

    async def _beat_loop(self) -> None:
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
                continue
            self._last_state_delta_at = time.monotonic()
            self.parent_beats_sent += 1
            try:
                await self.transport.send(msg)
            except Exception:
                logger.debug("parent beat send failed", exc_info=True)

    def _any_link(self) -> bool:
        return any(slot.link is not None for slot in self._slots)

    async def _sleep_or_stop(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), delay)
        except asyncio.TimeoutError:
            pass

    async def _finish_shutdown_flush(self, *, reason: str) -> None:
        if self._shutdown_flushed:
            self._stopping.set()
            return
        self._shutdown_flushed = True
        self._child_exited_clean = True
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
        waits = [slot.death_report_done.wait() for slot in self._slots]
        try:
            await asyncio.wait_for(
                asyncio.gather(*waits), _LINK_SETTLE_TIMEOUT_S + 2.0
            )
        except asyncio.TimeoutError:
            logger.warning("draining without waiting further for death attribution")
        await self._finish_shutdown_flush(reason="drain_without_child")

    def _forward_signal(self, signum: int) -> None:
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
        self._draining = True
        asyncio.create_task(self._drain_without_child(), name="signal-drain")

    async def _stop_deadline(self) -> None:
        await self._await_all_children_exit(self._stop_timeout)

    async def _await_all_children_exit(self, timeout: float) -> bool:
        results = await asyncio.gather(
            *(slot.await_exit(timeout) for slot in self._slots)
        )
        return all(results)

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
        for slot in self._slots:
            await slot.start_server()
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
        children_done = asyncio.gather(*child_tasks)
        try:
            done, _ = await asyncio.wait(
                (transport_task, children_done), return_when=asyncio.FIRST_COMPLETED
            )
            if transport_task in done:
                transport_task.result()
                if (self._draining or self._terminating) and not children_done.done():
                    await self._await_all_children_exit(self._stop_timeout)
                    try:
                        await asyncio.wait_for(asyncio.shield(children_done), 15.0)
                    except asyncio.TimeoutError:
                        logger.warning("child supervision loops did not settle after drain")
                    except Exception:
                        pass
            else:
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
                return 1
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
    from ..supervisor import report_previous_container_death

    report_previous_container_death()
    postmortem.clear_all_inflight()
    postmortem.write_boot_record()
    settings = config.install(config.load_settings())
    code = ParentControl(settings).run()
    if code == 0:
        postmortem.clear_boot_record()
        postmortem.clear_all_inflight()
    return code
