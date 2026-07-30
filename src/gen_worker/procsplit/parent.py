"""Parent (control-plane) side of the pgw#763 split.

Owns: the gRPC stream + identity/JWT (the real ``Transport``), the compute
child's lifetime (spawn / respawn-on-failure / crash-loop detection /
watchdog), the durable SendQueue, and job attribution across child deaths.
Never imports torch.

Supervision primitives are deliberately systemd's (Paul, 2026-07-29):
``Restart=on-failure`` (always respawn), ``StartLimitBurst``/
``StartLimitIntervalSec`` (loop DETECTION over a window — reported typed, never
cap-and-brick), ``WatchdogSec``+``sd_notify`` (child frames are the liveness
pings; a silent child is killed and respawned), socket activation (the hub
connection outlives the process doing the work). What systemd cannot do — and
the reason this is not s6/supervisord in the container — is job attribution:
"request X died of OOM, the release is healthy, the worker is alive, send the
next job" requires holding the stream, the JWT, and the in-flight table here.
"""

from __future__ import annotations

import asyncio
import collections
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
from . import ENV_CHILD, ENV_CHILD_CMD, ENV_SOCKET, frames

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
        respawn_backoff_base_s: float = _DEFAULT_RESPAWN_BACKOFF_BASE_S,
        respawn_backoff_cap_s: float = _DEFAULT_RESPAWN_BACKOFF_CAP_S,
        start_limit_burst: int = _DEFAULT_START_LIMIT_BURST,
        start_limit_interval_s: float = _DEFAULT_START_LIMIT_INTERVAL_S,
        watchdog_budget_s: float = _DEFAULT_WATCHDOG_BUDGET_S,
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
        self._draining = False
        self._child_exited_clean = False
        self._last_crash_loop_report_at = 0.0
        self.crash_loop_reports = 0  # observability + tests

    # ---- Transport handlers ---------------------------------------------

    async def build_hello(self) -> pb.Hello:
        """Fetch the CHILD's fresh Hello, then merge the parent's durable
        pending-result keys so the hub's in-flight reconcile sees results that
        outlived a child (or a stream)."""
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
            if link is None:
                return  # job already terminal-reported by the death path
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
        link = self._link
        if link is not None:
            await link.writer.frame(
                frames.T_TOKEN,
                frames.pack_meta({"token": token, "exp": int(expires_at_unix)}),
            )

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
            if msg.WhichOneof("msg") == "job_result":
                r = msg.job_result
                self._in_flight.pop((r.request_id, r.attempt), None)
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
        if ftype == frames.T_PREPEND:
            msgs = [pb.WorkerMessage.FromString(b) for b in frames.unpack_meta(payload)]
            await self.transport.prepend_reconnect(msgs)
            return
        if ftype == frames.T_FLUSH_REQ:
            meta = frames.unpack_meta(payload)
            timeout = meta.get("timeout")
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

    # ---- child lifetime --------------------------------------------------

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        env.update(self._child_env)
        env[ENV_CHILD] = "1"
        env[ENV_SOCKET] = self._socket_path
        # The gw#640 flight-recorder fork is redundant under this parent.
        env["GEN_WORKER_SUPERVISOR"] = "0"
        self._spawn_count += 1
        logger.info(
            "spawning compute child #%d: %s", self._spawn_count, " ".join(self._child_cmd)
        )
        return await asyncio.create_subprocess_exec(*self._child_cmd, env=env)

    async def _child_loop(self) -> None:
        backoff = self._backoff_base
        while not self._stopping.is_set():
            self._watchdog_fired = False
            self._child_saw_hello = False
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
            saw_hello = self._child_saw_hello
            link = self._link
            if link is not None:
                link.writer.close()
                self._link = None
                self._link_ready.clear()
            if self._stopping.is_set():
                return
            if rc == 0 and not self._watchdog_fired:
                # Deliberate exit (drain / stop): the pod's work is done.
                logger.info("compute child exited cleanly; parent exiting")
                self._child_exited_clean = True
                self.transport.stop()
                return
            await self._handle_child_death(
                rc, oom_before=oom_before, lifetime_s=lifetime, saw_hello=saw_hello,
            )
            if lifetime >= _BACKOFF_RESET_AFTER_ALIVE_S:
                backoff = self._backoff_base
            await self._sleep_or_stop(backoff)
            backoff = min(backoff * 2, self._backoff_cap)

    async def _sleep_or_stop(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), delay)
        except asyncio.TimeoutError:
            pass

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

        # 1) Attribution first: a typed FATAL per in-flight job, into the
        # DURABLE queue (ships on the live stream now, or survives to the
        # next one — either way the hub learns which job died, not merely
        # that a pod blinked).
        died_jobs = dict(self._in_flight)
        self._in_flight.clear()
        for (rid, att), fn in sorted(died_jobs.items()):
            await self.transport.send(pb.WorkerMessage(job_result=pb.JobResult(
                request_id=rid,
                attempt=att,
                status=pb.JOB_STATUS_FATAL,
                safe_message=(
                    f"{DEATH_LABEL}: cause={cause} function={fn or 'unknown'} "
                    f"(handler process died; worker alive and respawning)"
                )[:512],
            )))
        logger.error(
            "compute child died: cause=%s rc=%s lifetime=%.1fs in_flight=%s "
            "(respawning; stream identity kept)",
            cause, rc, lifetime_s, sorted(r for r, _ in died_jobs) or "none",
        )

        # 2) Post-mortem dial (gw#640 typed exit capture, carried forward).
        detail = postmortem.format_detail(
            phase="compute_process_exit",
            verdict=verdict,
            limits=postmortem.container_limits(),
            oom_kill_delta=oom_delta,
            lifetime_s=lifetime_s,
            extra={
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
        interval = max(0.25, self._watchdog_budget / 4.0)
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            proc = self._proc
            if proc is None or self._link is None or self._relaying:
                continue
            silent_for = time.monotonic() - self._last_frame_at
            if silent_for > self._watchdog_budget:
                logger.error(
                    "compute child silent for %.1fs (budget %.1fs) — killing "
                    "the wedged child (WatchdogSec analog)",
                    silent_for, self._watchdog_budget,
                )
                self._watchdog_fired = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

    # ---- drain / signals -------------------------------------------------

    async def _drain_without_child(self) -> None:
        await self.transport.close_after_flush(timeout=30.0)
        self._stopping.set()

    def _forward_signal(self, signum: int) -> None:
        proc = self._proc
        if proc is not None:
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
        transport_task = asyncio.create_task(self.transport.run(), name="parent-transport")
        child_task = asyncio.create_task(self._child_loop(), name="parent-child-loop")
        watchdog_task = asyncio.create_task(self._watchdog_loop(), name="parent-watchdog")
        try:
            done, _ = await asyncio.wait(
                (transport_task, child_task), return_when=asyncio.FIRST_COMPLETED
            )
            if transport_task in done:
                transport_task.result()  # re-raise FatalTransportError
            return 0 if (self._child_exited_clean or self._stopping.is_set()) else 1
        finally:
            self._stopping.set()
            proc = self._proc
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            self.transport.stop()
            for t in (transport_task, child_task, watchdog_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(
                transport_task, child_task, watchdog_task, return_exceptions=True
            )
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
