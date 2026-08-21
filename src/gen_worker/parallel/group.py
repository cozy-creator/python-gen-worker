"""RankGroup — the D-1 rank siblings that execute ONE execution sequence. They are NOT workers: no hub connection, no store, no lifecycle, no credentials. Process-group discipline: the worker process is rank 0 of EVERY group, so no group may ever touch the default torch.distributed process group — each gets its own NON-default ProcessGroup (own TCPStore, unique name, explicit handle to every collective); teardown destroys exactly that handle. Rank-0 -> follower commands ride per-follower mp queues, NEVER collectives (an idle follower parks on queue.get; teardown cannot hang; payloads are encoded eagerly on rank 0 so an uncrossable argument is a typed error). Collectives happen only inside the model call, bounded by the group's timeout. The channel's bytes are msgspec msgpack, NEVER pickle: rank 0 marshals tenant-supplied arguments, and a follower that unpickled would be a deserialization gadget on tenant-reachable input."""

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

_STAGING_SILENCE_WINDOW_S = 4.0 * 240.0

_STAGING_POLL_S = 1.0

_COLLECTIVE_TIMEOUT_S = 300.0

_FIRST_COMMAND_WAIT_S = 1800.0

_STORE_CONNECT_TIMEOUT_S = 180.0


class RankGroupError(RuntimeError):
    """The group could not form, or a rank died."""


@dataclass(frozen=True)
class RankSpec:
    """What one rank needs to join."""

    rank: int
    world_size: int
    device: int
    master_addr: str
    master_port: int
    backend: str
    group_name: str = ""
    collective_timeout_s: float = _COLLECTIVE_TIMEOUT_S


@dataclass
class FollowerChannel:
    """A follower's half of the command plane: commands in, readiness out."""

    commands: Any
    ready: Any

    def next_command(self, timeout: Optional[float] = None) -> wire.Command:
        """The next command, decoded as one of exactly three types."""
        raw = self.commands.get(timeout=timeout)
        return wire.decode(raw)

    def report_ready(self, rank: int) -> None:
        self.ready.put(int(rank))


def _ensure_local_default_group() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        return
    dist.init_process_group(
        backend="gloo", rank=0, world_size=1, store=dist.HashStore()
    )


def arrive_key(spec: RankSpec) -> str:
    """The store key a rank sets the moment it is ready to join the group."""
    return f"{spec.group_name or 'gwsp'}/arrive/{int(spec.rank)}"


_NVLS_ENV = "NCCL_NVLS_ENABLE"


def _refuse_nvls_multicast() -> None:
    """Turn NVLink SHARP (NVLS) multicast OFF before NCCL builds a communicator. NCCL >= 2.2x enables NVLS by default on NVSwitch hosts, binding multicast memory needs a privilege our containers do not have, and the FIRST all-to-all of every arm then dies with CUDA error 401 — a total failure, not a slowdown (sequence parallelism does not work at all on a stock 4xH100 pod without this). Costs nothing here: NVLS accelerates switch-side reductions, not all-to-all. The write is UNCONDITIONAL and immediately precedes communicator creation — never "only when unset", or an image or operator export could turn NVLS back on and take every Ulysses arm down; a dropped override is reported to whoever set it."""
    previous = os.environ.get(_NVLS_ENV)
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
    """Join ``spec``'s group as a NON-default process group; returns the ProcessGroup handle every collective must be given explicitly."""
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

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
    c10d._world.pg_group_ranks[pg] = {c10d.get_rank(): spec.rank}
    return pg


_PR_SET_PDEATHSIG = 1


def _die_with_rank0(rank0_pid: int) -> None:
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        return
    if rank0_pid and os.getppid() != rank0_pid:
        os._exit(1)


def _follower_main(
    spec: RankSpec,
    entry: Callable[[RankSpec, FollowerChannel], None],
    channel: FollowerChannel,
    error_q: Any,
    rank0_pid: int = 0,
) -> None:  # pragma: no cover - runs in a spawned process
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
    """The D ranks that execute one group."""

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
        self._staging_peaks: dict[int, float] = {}
        self._channels: List[FollowerChannel] = []
        self._error_q: Any = None
        self._ready_q: Any = None
        self._store: Any = None
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
        """Spawn the D−1 siblings and join as rank 0."""
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

    def send(self, command: wire.Command) -> None:
        """Deliver one command to every follower."""
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
        self._staging_peaks = {}
        return SilenceWindow(_STAGING_SILENCE_WINDOW_S)

    def _followers_advanced(self) -> bool:
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
        """Block until every follower reported ready, failing loudly on a dead or SILENT follower."""
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
        """A dead follower must fail the request LOUDLY and immediately — never let the group park on a collective that cannot complete."""
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
        """Idempotent teardown of THIS group only."""
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
