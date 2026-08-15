"""RankGroup — the D−1 rank siblings that execute ONE execution sequence.

The ruling this lives inside: a pod is ONE worker multiplexing G execution
sequences, with one connection to the orchestrator for jobs and results.

The rank siblings are **not workers**. They hold no hub connection, no store,
no output path, no lifecycle, no heartbeat, no capability tokens, no receipts.
They are an implementation detail of one execution sequence, the same way a
CUDA stream is: the worker is still one worker multiplexing G sequences;
sequence ``g`` merely happens to be executed by D OS processes when D > 1.

Leader-plus-followers, not ``torchrun``: N symmetric processes would each
register with the hub, each claim a worker identity, each own a store, each
report residency — and every hub-facing invariant we have is
one-worker-per-pod.

**Process-group discipline.** The worker process is rank 0 of EVERY
group, so no group may ever touch the default torch.distributed process
group: two groups would collide in one world and corrupt each other's
collectives. Every group therefore gets its own NON-default ProcessGroup —
its own TCPStore (the master socket owns the port, so allocation is
race-free), a unique group name, and an explicit handle passed to every
collective. Teardown destroys exactly that handle; a sibling group and the
process-local default are untouchable by construction.

**Command channel.** Rank-0 -> follower commands ride per-follower
mp queues, NOT collectives: an idle follower parks on ``queue.get`` (no
collective timeout to trip), teardown is a queue put + join/terminate (no
collective that can block the event loop), and command payloads are encoded
EAGERLY on rank 0 so an uncrossable argument is a typed per-request error
instead of a feeder-thread mystery. Collectives happen only INSIDE the model
call (the CP hooks) and carry the process group's timeout, so a rank that
raises mid-call strands its peers for at most ``collective_timeout_s``, after
which they fail loudly and the group is condemned — never a silent wedge.

The channel's bytes are msgspec msgpack (:mod:`.wire`), NEVER pickle: the
writer is the process that imports tenant endpoint code and marshals
tenant-supplied model-call arguments, so a follower that unpickled would be a
deserialization gadget on tenant-reachable input.
"""

from __future__ import annotations

import logging
import os
import queue as queue_mod
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast
import multiprocessing as mp

from . import wire
from .. import proc_evidence, settings_authority
from ..stall import SilenceWindow

logger = logging.getLogger(__name__)

# Group formation is bounded by SILENCE, never by a wall clock: a flat deadline
# condemns WORK, and a cold pipeline materialization legitimately runs for hours.
# What distinguishes a wedged follower from a slow one is whether it is still
# doing work, and `check_alive()` already covers the DEAD case typed-and-
# immediately. So the bound is a silence window over
# `proc_evidence.tree_evidence` — the follower's own kernel-accounted CPU and
# I/O, which needs no cooperation from a process that has no protocol between
# spawn and ready. A follower that keeps advancing runs as long as its work
# does; one that stops advancing is condemned by name, at any elapsed time.
#
# The window is a statement about the CHANNEL, not a budget for the job: how
# long a materializing process may plausibly show neither a CPU tick nor a
# byte. Sized at 4x `progress.STALL_WINDOW_S["load"]` (240 s), the window the
# in-process load phase is already judged by, because /proc sampling is
# coarser and a follower that is genuinely blocked on a slow remote read shows
# nothing until the read returns.
_STAGING_SILENCE_WINDOW_S = 4.0 * 240.0

# How often the two loops re-ask. A cadence, never a verdict: it bounds
# latency-to-notice, and lengthening it cannot condemn anything.
_STAGING_POLL_S = 1.0

# Ceiling on any single in-call collective wait: legitimate skew between
# ranks executing the same denoise step is seconds; a peer that raised (or
# died) mid-call leaves this as the unwedging mechanism.
_COLLECTIVE_TIMEOUT_S = 300.0

# A follower's wait for rank 0's FIRST command. Rank 0 sends `arm` within
# milliseconds of `form()` returning, so this bounds nothing anyone does; the
# threat it names is a follower parked forever because rank 0 died in a way
# `_die_with_rank0`'s PR_SET_PDEATHSIG cannot cover (non-Linux hosts, and the
# pre-prctl spawn window). It bounds a WAIT ON A PEER, never work.
_FIRST_COMMAND_WAIT_S = 1800.0

# The TCPStore's own socket-connect budget. This bounds a CONNECT to a
# localhost port whose listener already exists (rank 0 owns the master socket
# before any follower spawns), so it either succeeds in milliseconds or the
# port is wrong — it can never be the thing a slow load runs out of, and it
# is not the store-arrival wait, which is `_await_arrivals` above.
_STORE_CONNECT_TIMEOUT_S = 180.0


class RankGroupError(RuntimeError):
    """The group could not form, or a rank died. Fatal for the request; the
    worker keeps serving its other groups."""


@dataclass(frozen=True)
class RankSpec:
    """What one rank needs to join. Deliberately tiny and picklable — it
    crosses the SPAWN boundary, which is multiprocessing's own pickle of a
    fixed struct this process built, not the command queue (that is `wire`,
    and it carries tenant-reachable payloads). ``group_name`` is unique per
    arm so a process that arms many groups over its lifetime never collides
    in the torch.distributed registries."""

    rank: int
    world_size: int
    device: int
    master_addr: str
    master_port: int
    backend: str  # "nccl" on GPUs; "gloo" is the CPU test rig
    group_name: str = ""
    collective_timeout_s: float = _COLLECTIVE_TIMEOUT_S


@dataclass
class FollowerChannel:
    """A follower's half of the command plane: commands in, readiness out."""

    commands: Any  # mp.Queue of msgpack-encoded `wire.Command` bytes
    ready: Any     # mp.Queue; follower puts its rank when armed

    def next_command(self, timeout: Optional[float] = None) -> wire.Command:
        """The next command, decoded as one of exactly three types. A payload
        cannot name a class, so it cannot reach a constructor."""
        raw = self.commands.get(timeout=timeout)
        return wire.decode(raw)

    def report_ready(self, rank: int) -> None:
        self.ready.put(int(rank))


def _ensure_local_default_group() -> None:
    """A process-LOCAL, network-free default PG.

    torch.distributed's bookkeeping (`get_rank(group)`, subgroup registries)
    requires *a* default group to exist. It must never be the rendezvous for
    anything: world size 1 over a HashStore, no sockets, no peers. Installed
    identically in the worker and in every follower so group construction is
    symmetric across ranks.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return
    dist.init_process_group(
        backend="gloo", rank=0, world_size=1, store=dist.HashStore()
    )


def arrive_key(spec: RankSpec) -> str:
    """The store key a rank sets the moment it is ready to join the group.

    Rank 0 waits on these keys (pollable, so it can check follower liveness
    between polls) instead of entering the backend rendezvous blind: a follower
    that dies BEFORE joining would otherwise park rank 0 inside the backend's
    connect for the whole collective timeout.
    """
    return f"{spec.group_name or 'gwsp'}/arrive/{int(spec.rank)}"


_NVLS_ENV = "NCCL_NVLS_ENABLE"


def _refuse_nvls_multicast() -> None:
    """Turn NVLink SHARP (NVLS) multicast OFF before NCCL builds a communicator.

    Measured live on a 4xH100-80GB-HBM3 SXM pod (NV18, 366.7 GB/s peer,
    NCCL 2.29.7): the group forms, CP installs, and then the FIRST all-to-all
    of every arm — degree 2, degree 4 and both groups of a 2x2 — dies::

        ncclUnhandledCudaError: Call to CUDA function failed.
        Failed to bind NVLink SHARP (NVLS) Multicast memory of size 2097152 :
        CUDA error 401 'the operation cannot be performed in the present state'

    NCCL >= 2.2x enables NVLS by default on NVSwitch hosts, and binding
    multicast memory needs a privilege our containers do not have. It is a
    total failure, not a slowdown: sequence parallelism does not work at all on
    a stock 4xH100 pod without this.

    Switching it off costs nothing measurable HERE: Ulysses is all-to-all, and
    NVLS accelerates switch-side reductions (all-reduce/reduce-scatter), not
    all-to-all.

    The write is UNCONDITIONAL and immediately precedes communicator creation —
    never "only when unset", or an image or operator could turn NVLS back on
    and take every Ulysses arm down with CUDA 401. The env survives only as
    NCCL's own handoff mechanism (it reads env at init and offers no other
    API); it is nobody's choice. A future collective that can benefit from NVLS
    needs a measured capability, and may not revive this as a gate.

    Removing an override removes an escape hatch, so the override is never
    silently discarded: whoever set it is TOLD what was dropped, at the moment
    it is dropped.
    """
    previous = os.environ.get(_NVLS_ENV)
    # The write is the settings authority's (NCCL_NVLS_ENABLE=0 is in
    # DECLARED_ENV); this site keeps the drop-an-override warning below.
    settings_authority.impose_process_env()
    if previous not in (None, "0"):
        logger.warning(
            "%s was %r; overwritten to 0. NVLS multicast cannot be bound in "
            "our containers (measured: ncclUnhandledCudaError / CUDA 401 on "
            "the first all-to-all of every group) and Ulysses does not use it "
            "— this is not an operator choice (pgw#929). If you have a "
            "collective and a container that genuinely bind NVLS multicast, "
            "that is a measured capability and a new issue, not this env: "
            "re-enabling it here takes down every arm of every group.",
            _NVLS_ENV, previous,
        )


def init_rank(spec: RankSpec, store: Any = None) -> Any:
    """Join ``spec``'s group as a NON-default process group; returns the
    ProcessGroup handle every collective must be given explicitly.

    Called in EVERY rank — rank 0 in the worker process (which passes the
    master store it already owns), followers in their spawned ones.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    # Before any communicator exists in this process: NCCL reads its env at
    # communicator creation, and after that it is too late.
    _refuse_nvls_multicast()
    if spec.backend == "nccl":
        torch.cuda.set_device(spec.device)
    _ensure_local_default_group()
    if store is None:
        store = dist.TCPStore(
            spec.master_addr,
            spec.master_port,
            spec.world_size,
            is_master=False,
            timeout=timedelta(seconds=_STORE_CONNECT_TIMEOUT_S),
        )
        # Announce arrival BEFORE the backend rendezvous, so rank 0 knows
        # this rank is real and about to join.
        store.set(arrive_key(spec), b"1")
    pg, _prefix_store = c10d._new_process_group_helper(
        spec.world_size,
        spec.rank,
        [c10d.get_rank()],
        spec.backend,
        store,
        group_name=cast(Any, spec.group_name or f"gwsp-{uuid.uuid4().hex[:8]}"),
        timeout=timedelta(seconds=spec.collective_timeout_s),
    )
    # get_rank(pg)/get_group_rank need the world-rank -> group-rank mapping;
    # the process-local default world has exactly one rank (this process).
    c10d._world.pg_group_ranks[pg] = {c10d.get_rank(): spec.rank}
    return pg


_PR_SET_PDEATHSIG = 1


def _die_with_rank0(rank0_pid: int) -> None:
    """A follower is a GRANDCHILD of the process-split parent, whose
    ``PR_SET_PDEATHSIG`` covers parent -> compute child and does not cascade.
    ``daemon=True`` only reaps through ``multiprocessing``'s atexit hook, which
    an abort (``rc=-6``, NCCL) never runs: the followers would keep a full
    weight replica on cards 1..D-1 for their own 300 s collective timeout while
    the parent respawns the group onto those same cards in ~1 s — a crash loop
    seeded by its own orphans.

    So every follower asks the kernel for the same contract the compute child
    has: SIGKILL when its parent (rank 0) dies, abort included. Linux-only;
    elsewhere container death reaps.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        return
    # Close the spawn race: if rank 0 died BEFORE the prctl landed, no death
    # signal is coming — the reparent already happened, so exit now.
    if rank0_pid and os.getppid() != rank0_pid:
        os._exit(1)


def _follower_main(
    spec: RankSpec,
    entry: Callable[[RankSpec, FollowerChannel], None],
    channel: FollowerChannel,
    error_q: Any,
    rank0_pid: int = 0,
) -> None:  # pragma: no cover - runs in a spawned process
    """A follower's whole life: join, run the narrow loop, report a fatal.

    Nothing here talks to the hub, and nothing here decides anything — every
    adaptive choice arrives from rank 0 over the channel.
    """
    _die_with_rank0(rank0_pid)
    try:
        entry(spec, channel)
    except BaseException as exc:  # noqa: BLE001 — the whole point is to report
        try:
            error_q.put((spec.rank, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        raise


class RankGroup:
    """The D ranks that execute one group. Rank 0 is this process.

    Lifecycle is explicitly not the worker's lifecycle: a group is formed for
    a materialization and torn down with it. ``close()`` is idempotent, never
    performs a collective, and is the ONLY way a follower is supposed to
    exit.
    """

    def __init__(
        self,
        devices: Sequence[int],
        *,
        backend: str = "nccl",
        entry: Optional[Callable[[RankSpec, FollowerChannel], None]] = None,
        collective_timeout_s: float = _COLLECTIVE_TIMEOUT_S,
    ) -> None:
        self.devices: Tuple[int, ...] = tuple(int(d) for d in devices)
        if not self.devices:
            raise ValueError("a rank group needs at least one device")
        self.backend = backend
        self._entry = entry
        self._collective_timeout_s = float(collective_timeout_s)
        self._procs: List[Any] = []
        #: pid -> high-water evidence mark.
        self._staging_peaks: dict[int, float] = {}
        self._channels: List[FollowerChannel] = []
        self._error_q: Any = None
        self._ready_q: Any = None
        self._store: Any = None  # master TCPStore: its socket owns the port
        self._pg: Any = None
        self._formed = False

    @property
    def degree(self) -> int:
        return len(self.devices)

    @property
    def process_group(self) -> Any:
        """The NON-default ProcessGroup handle, or None at degree 1."""
        return self._pg

    def form(self) -> RankSpec:
        """Spawn the D−1 siblings and join as rank 0. Returns rank 0's spec.

        Degree 1 forms nothing: a single-device group is exactly today's
        worker, and it must not pay a process-group tax for a collective it
        will never make.
        """
        if self.degree == 1:
            self._formed = True
            return RankSpec(0, 1, self.devices[0], "127.0.0.1", 0, self.backend)

        import torch.distributed as dist

        if self._entry is None:
            raise RankGroupError("a degree>1 group needs a follower entry point")

        ctx = mp.get_context("spawn")
        self._error_q = ctx.Queue()
        self._ready_q = ctx.Queue()
        group_name = f"gwsp-{uuid.uuid4().hex[:8]}"
        # Race-free rendezvous: bind the master store FIRST (port 0 = the OS
        # picks a free one and the socket holds it), then hand the bound port
        # to the followers. No probe-release-rebind window, no env vars.
        self._store = dist.TCPStore(
            "127.0.0.1",
            0,
            self.degree,
            is_master=True,
            timeout=timedelta(seconds=_STORE_CONNECT_TIMEOUT_S),
            wait_for_workers=False,
        )
        port = int(self._store.port)
        specs = [
            RankSpec(
                r, self.degree, self.devices[r], "127.0.0.1", port,
                self.backend, group_name=group_name,
                collective_timeout_s=self._collective_timeout_s,
            )
            for r in range(self.degree)
        ]
        for spec in specs[1:]:
            channel = FollowerChannel(commands=ctx.Queue(), ready=self._ready_q)
            proc = ctx.Process(
                target=_follower_main,
                # The follower's death is tied to THIS process (rank 0) in its
                # own bootstrap — daemon=True cannot survive an abort, and the
                # process-split parent must not know about ranks.
                args=(spec, self._entry, channel, self._error_q, os.getpid()),
                name=f"sp-{group_name}-rank{spec.rank}",
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
            self._channels.append(channel)
        logger.info(
            "sequence-parallel group forming: name=%s degree=%d devices=%s "
            "backend=%s port=%d", group_name, self.degree,
            list(self.devices), self.backend, port,
        )
        try:
            self._await_arrivals(specs)
            self._pg = init_rank(specs[0], store=self._store)
        except BaseException as exc:  # noqa: BLE001
            self.close()
            if isinstance(exc, RankGroupError):
                raise
            raise RankGroupError(
                f"rank 0 failed to join the group: {type(exc).__name__}: {exc}"
            ) from exc
        self._formed = True
        return specs[0]

    def _await_arrivals(self, specs: Sequence[RankSpec]) -> None:
        """Wait for every follower to announce arrival, polling liveness.

        Rank 0 must never enter the backend rendezvous blind: gloo/NCCL block
        in ``connect`` for the whole collective timeout when a peer is absent,
        so a follower that crashed on import (bad card, missing wheel, OOM
        during spawn) would turn "the group cannot form" into a multi-minute
        stall inside ``form()``. Polling the store's arrive keys makes it a
        bounded, TYPED failure that names the dead rank.
        """
        keys = [arrive_key(s) for s in specs[1:]]
        silence = self._staging_silence()
        while True:
            self.check_alive()
            if self._store.check(keys):
                return
            if self._followers_advanced():
                silence.touch()
            elif silence.stalled():
                missing = [
                    s.rank for s in specs[1:]
                    if not self._store.check([arrive_key(s)])
                ]
                raise RankGroupError(
                    f"followers {missing} have not reached the rendezvous and "
                    f"have shown no CPU or I/O for {silence.silent_for():.0f}s "
                    f"— the group cannot form. A follower still importing torch "
                    f"is not condemned by this; one that stopped working is."
                )
            time.sleep(_STAGING_POLL_S)

    # ---- command plane (never a collective) --------------------------------

    def send(self, command: wire.Command) -> None:
        """Deliver one command to every follower.

        Encoded EAGERLY here so an uncrossable payload raises a typed error on
        THIS thread with the group still coherent — the followers are parked
        at ``queue.get`` and simply never see the command.
        """
        try:
            raw = wire.encode(command)
        except Exception as exc:
            raise RankGroupError(
                f"command cannot cross the rank boundary: {type(exc).__name__}: "
                f"{exc}"
            ) from exc
        for channel in self._channels:
            channel.commands.put(raw)

    def _staging_silence(self) -> SilenceWindow:
        """A fresh silence window over follower work, with the high-water
        marks it compares against reset."""
        self._staging_peaks = {}
        return SilenceWindow(_STAGING_SILENCE_WINDOW_S)

    def _followers_advanced(self) -> bool:
        """Whether ANY follower's process tree has done measurable work since
        the last sample.

        High-water marks, not deltas: a descendant's CPU migrates into its
        parent's ``cutime``/``cstime`` on reap, so a live-only sum falls when
        a subprocess finishes. A follower whose evidence cannot be read at all
        contributes nothing here and is left to ``check_alive()``, which is
        the honest split — an unreadable process is not a stalled one, and it
        is not this method's job to condemn it.
        """
        advanced = False
        for proc in self._procs:
            pid = int(getattr(proc, "pid", 0) or 0)
            if pid <= 0:
                continue
            evidence = proc_evidence.tree_evidence(pid)
            if evidence is None:
                continue
            if evidence > self._staging_peaks.get(pid, -1.0):
                self._staging_peaks[pid] = evidence
                advanced = True
        return advanced

    def wait_armed(self) -> None:
        """Block until every follower reported ready, failing loudly on a
        dead or SILENT follower. Queue-based — never a collective, so a
        follower that dies mid-materialization cannot park us.

        Deliberately NOT a wall-clock timeout: `check_alive()` already fails a
        DEAD follower typed and immediately, so a duration adds nothing death
        detection does not give and subtracts every slow cold load.
        """
        if self.degree == 1:
            return
        ready: set = set()
        silence = self._staging_silence()
        while len(ready) < self.degree - 1:
            self.check_alive()
            try:
                ready.add(int(self._ready_q.get(timeout=_STAGING_POLL_S)))
                silence.touch()
                continue
            except queue_mod.Empty:
                pass
            if self._followers_advanced():
                silence.touch()
            elif silence.stalled():
                raise RankGroupError(
                    f"followers have shown no CPU or I/O for "
                    f"{silence.silent_for():.0f}s while arming "
                    f"(ready={sorted(ready)} of ranks 1..{self.degree - 1}) — "
                    f"a cold materialization that is still working is not "
                    f"condemned by this"
                )

    def check_alive(self) -> None:
        """A dead follower must fail the request LOUDLY and immediately —
        never let the group park on a collective that cannot complete."""
        fatal = self.drain_errors()
        if fatal:
            raise RankGroupError("; ".join(fatal))
        for proc in self._procs:
            if not proc.is_alive():
                raise RankGroupError(
                    f"{proc.name} exited with code {proc.exitcode} — the "
                    "sequence-parallel group is broken"
                )

    def drain_errors(self) -> List[str]:
        out: List[str] = []
        if self._error_q is None:
            return out
        while True:
            try:
                rank, msg = self._error_q.get_nowait()
            except Exception:
                break
            out.append(f"rank {rank}: {msg}")
        return out

    def barrier(self) -> None:
        import torch.distributed as dist

        if self.degree == 1 or self._pg is None:
            return
        self.check_alive()
        dist.barrier(group=self._pg)

    def close(self, *, grace_s: float = 10.0) -> None:
        """Idempotent teardown of THIS group only. No collectives: a CLOSE
        command, a bounded join, then terminate/kill — safe to run against a
        group whose followers are stuck in a collective (they are killed and
        their peers' collectives time out typed). Destroys only this group's
        process-group handle; the process-local default and every sibling
        group are untouched."""
        for channel in self._channels:
            try:
                channel.commands.put(wire.encode(wire.Close()))
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        deadline = time.monotonic() + grace_s
        for proc in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                logger.warning("%s did not exit; terminating", proc.name)
                proc.terminate()
                proc.join(timeout=5.0)
            if proc.is_alive():  # pragma: no cover - kernel-level stuck
                proc.kill()
        self._procs = []
        self._channels = []
        pg, self._pg = self._pg, None
        if pg is not None:
            try:
                import torch.distributed as dist

                dist.destroy_process_group(pg)
            except Exception:
                logger.warning(
                    "destroying the group's process group failed", exc_info=True
                )
        self._store = None
        self._formed = False
