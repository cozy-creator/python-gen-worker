"""RankGroup — the D−1 rank siblings that execute ONE execution sequence.

Paul's ruling (2026-07-28), and the frame this lives inside:

> I'm thinking it'll be seen as one worker, which is multiplexing 4x or 8x
> execution sequences separately. It only needs one connection to the
> orchestrator to get jobs and send results back.

The rank siblings are **not workers**. They hold no hub connection, no store,
no output path, no lifecycle, no heartbeat, no capability tokens, no receipts.
They are an implementation detail of one execution sequence, the same way a
CUDA stream is: the worker is still one worker multiplexing G sequences;
sequence ``g`` merely happens to be executed by D OS processes when D > 1.

Leader-plus-followers, not ``torchrun``: N symmetric processes would each
register with the hub, each claim a worker identity, each own a store, each
report residency — and every hub-facing invariant we have is
one-worker-per-pod.

**Process-group discipline (pgw#773).** The worker process is rank 0 of EVERY
group, so no group may ever touch the default torch.distributed process
group: two groups would collide in one world and corrupt each other's
collectives. Every group therefore gets its own NON-default ProcessGroup —
its own TCPStore (the master socket owns the port, so allocation is
race-free), a unique group name, and an explicit handle passed to every
collective. Teardown destroys exactly that handle; a sibling group and the
process-local default are untouchable by construction.

**Command channel (pgw#774).** Rank-0 -> follower commands ride per-follower
mp queues, NOT collectives: an idle follower parks on ``queue.get`` (no
collective timeout to trip), teardown is a queue put + join/terminate (no
collective that can block the event loop), and command payloads are pickled
EAGERLY on rank 0 so an unpicklable argument is a typed per-request error
instead of a feeder-thread mystery. Collectives happen only INSIDE the model
call (the CP hooks) and carry the process group's timeout, so a rank that
raises mid-call strands its peers for at most ``collective_timeout_s``, after
which they fail loudly and the group is condemned — never a silent wedge.
"""

from __future__ import annotations

import logging
import pickle
import queue as queue_mod
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# A follower that has not reached the store rendezvous by this deadline is
# dead weight: the group cannot form and the request must fail loudly rather
# than park on a rendezvous that will never complete. A constant, not an env
# knob (DPA-22: the old GEN_WORKER_SP_RENDEZVOUS_S changed nothing and is
# gone).
_RENDEZVOUS_TIMEOUT_S = 180.0

# Ceiling on any single in-call collective wait: legitimate skew between
# ranks executing the same denoise step is seconds; a peer that raised (or
# died) mid-call leaves this as the unwedging mechanism.
_COLLECTIVE_TIMEOUT_S = 300.0

# Arming covers a follower's full pipeline materialization (a cold model
# load), so its budget is the staging budget, not the collective one.
_ARM_TIMEOUT_S = 1800.0

_OP_RUN = "run"
_OP_CLOSE = "close"


class RankGroupError(RuntimeError):
    """The group could not form, or a rank died. Fatal for the request; the
    worker keeps serving its other groups."""


@dataclass(frozen=True)
class RankSpec:
    """What one rank needs to join. Deliberately tiny and picklable — it
    crosses a spawn boundary. ``group_name`` is unique per arm so a process
    that arms many groups over its lifetime never collides in the
    torch.distributed registries."""

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

    commands: Any  # mp.Queue of pickled command bytes
    ready: Any     # mp.Queue; follower puts its rank when armed

    def next_command(self, timeout: Optional[float] = None) -> Any:
        raw = self.commands.get(timeout=timeout)
        return pickle.loads(raw)

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
    between polls) instead of entering the backend rendezvous blind: a
    follower that dies BEFORE joining used to park rank 0 inside the
    backend's connect for the whole collective timeout.
    """
    return f"{spec.group_name or 'gwsp'}/arrive/{int(spec.rank)}"


def init_rank(spec: RankSpec, store: Any = None) -> Any:
    """Join ``spec``'s group as a NON-default process group; returns the
    ProcessGroup handle every collective must be given explicitly.

    Called in EVERY rank — rank 0 in the worker process (which passes the
    master store it already owns), followers in their spawned ones.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    if spec.backend == "nccl":
        torch.cuda.set_device(spec.device)
    _ensure_local_default_group()
    if store is None:
        store = dist.TCPStore(
            spec.master_addr,
            spec.master_port,
            spec.world_size,
            is_master=False,
            timeout=timedelta(seconds=_RENDEZVOUS_TIMEOUT_S),
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
        group_name=spec.group_name or f"gwsp-{uuid.uuid4().hex[:8]}",
        timeout=timedelta(seconds=spec.collective_timeout_s),
    )
    # get_rank(pg)/get_group_rank need the world-rank -> group-rank mapping;
    # the process-local default world has exactly one rank (this process).
    c10d._world.pg_group_ranks[pg] = {c10d.get_rank(): spec.rank}
    return pg


def _follower_main(
    spec: RankSpec,
    entry: Callable[[RankSpec, FollowerChannel], None],
    channel: FollowerChannel,
    error_q: Any,
) -> None:  # pragma: no cover - runs in a spawned process
    """A follower's whole life: join, run the narrow loop, report a fatal.

    Nothing here talks to the hub, and nothing here decides anything — every
    adaptive choice arrives from rank 0 over the channel (pgw#748 §5.4).
    """
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

        import multiprocessing as mp

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
            timeout=timedelta(seconds=_RENDEZVOUS_TIMEOUT_S),
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
                args=(spec, self._entry, channel, self._error_q),
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
        during spawn) turned "the group cannot form" into a multi-minute stall
        inside ``form()``. Polling the store's arrive keys makes it a bounded,
        TYPED failure that names the dead rank.
        """
        keys = [arrive_key(s) for s in specs[1:]]
        deadline = time.monotonic() + _RENDEZVOUS_TIMEOUT_S
        while True:
            self.check_alive()
            if self._store.check(keys):
                return
            if time.monotonic() >= deadline:
                missing = [
                    s.rank for s in specs[1:]
                    if not self._store.check([arrive_key(s)])
                ]
                raise RankGroupError(
                    f"followers {missing} never reached the rendezvous within "
                    f"{_RENDEZVOUS_TIMEOUT_S:.0f}s — the group cannot form"
                )
            time.sleep(0.05)

    # ---- command plane (never a collective) --------------------------------

    def send(self, command: Any) -> None:
        """Deliver one command to every follower.

        Pickled EAGERLY here so an unpicklable payload (a closure callback,
        a live handle) raises a typed error on THIS thread with the group
        still coherent — the followers are parked at ``queue.get`` and simply
        never see the command.
        """
        try:
            raw = pickle.dumps(command)
        except Exception as exc:
            raise RankGroupError(
                f"command cannot cross the rank boundary: {type(exc).__name__}: "
                f"{exc} — pass only picklable model-call arguments (closures/"
                "callbacks cannot be broadcast to follower ranks)"
            ) from exc
        for channel in self._channels:
            channel.commands.put(raw)

    def wait_armed(self, timeout_s: float = _ARM_TIMEOUT_S) -> None:
        """Block until every follower reported ready, failing loudly on a
        dead or overdue follower. Queue-based — never a collective, so a
        follower that dies mid-materialization cannot park us."""
        if self.degree == 1:
            return
        deadline = time.monotonic() + float(timeout_s)
        ready: set = set()
        while len(ready) < self.degree - 1:
            self.check_alive()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RankGroupError(
                    f"followers not armed after {timeout_s:.0f}s "
                    f"(ready={sorted(ready)} of ranks 1..{self.degree - 1})"
                )
            try:
                ready.add(int(self._ready_q.get(timeout=min(1.0, remaining))))
            except queue_mod.Empty:
                continue

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
        group are untouched (pgw#773)."""
        for channel in self._channels:
            try:
                channel.commands.put(pickle.dumps({"op": _OP_CLOSE}))
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
