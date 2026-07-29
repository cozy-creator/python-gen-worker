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
one-worker-per-pod (the Hello bind is a blind last-writer-wins UPDATE with no
CAS on worker_id, so two workers on one pod silently corrupt each other).

The sibling loop is narrow and fixed: join the group, receive a
materialization plan, load from the SAME on-disk content-addressed store, then
``{await command → run → barrier}``.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# A follower that has not reached the rendezvous by this deadline is dead
# weight: the group cannot form and the request must fail loudly rather than
# park on a collective that will never complete.
_RENDEZVOUS_TIMEOUT_S = float(os.environ.get("GEN_WORKER_SP_RENDEZVOUS_S", "180"))


class RankGroupError(RuntimeError):
    """The group could not form, or a rank died. Fatal for the request; the
    worker keeps serving its other groups."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass(frozen=True)
class RankSpec:
    """What one rank needs to join. Deliberately tiny and picklable — it
    crosses a spawn boundary."""

    rank: int
    world_size: int
    device: int
    master_addr: str
    master_port: int
    backend: str  # "nccl" on GPUs; "gloo" is the CPU test rig


def init_rank(spec: RankSpec) -> None:
    """Join the process group as ``spec.rank``. Called in EVERY rank —
    rank 0 in the worker process, followers in their spawned ones."""
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = spec.master_addr
    os.environ["MASTER_PORT"] = str(spec.master_port)
    if spec.backend == "nccl":
        torch.cuda.set_device(spec.device)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=spec.backend,
            rank=spec.rank,
            world_size=spec.world_size,
        )


def _follower_main(
    spec: RankSpec,
    entry: Callable[[RankSpec], None],
    error_q: Any,
) -> None:  # pragma: no cover - runs in a spawned process
    """A follower's whole life: join, run the narrow loop, report a fatal.

    Nothing here talks to the hub, and nothing here decides anything — every
    adaptive choice arrives as a broadcast GroupPlan (pgw#748 §5.4).
    """
    try:
        init_rank(spec)
        entry(spec)
    except BaseException as exc:  # noqa: BLE001 — the whole point is to report
        try:
            error_q.put((spec.rank, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        raise


class RankGroup:
    """The D ranks that execute one group. Rank 0 is this process.

    Lifecycle is explicitly not the worker's lifecycle: a group is formed for
    a materialization and torn down with it. ``close()`` is idempotent and is
    the ONLY way a follower is supposed to exit.
    """

    def __init__(
        self,
        devices: Sequence[int],
        *,
        backend: str = "nccl",
        entry: Optional[Callable[[RankSpec], None]] = None,
    ) -> None:
        self.devices: Tuple[int, ...] = tuple(int(d) for d in devices)
        if not self.devices:
            raise ValueError("a rank group needs at least one device")
        self.backend = backend
        self._entry = entry
        self._procs: List[Any] = []
        self._error_q: Any = None
        self._formed = False

    @property
    def degree(self) -> int:
        return len(self.devices)

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

        if self._entry is None:
            raise RankGroupError("a degree>1 group needs a follower entry point")

        ctx = mp.get_context("spawn")
        self._error_q = ctx.Queue()
        port = _free_port()
        specs = [
            RankSpec(r, self.degree, self.devices[r], "127.0.0.1", port,
                     self.backend)
            for r in range(self.degree)
        ]
        for spec in specs[1:]:
            proc = ctx.Process(
                target=_follower_main,
                args=(spec, self._entry, self._error_q),
                name=f"sp-rank{spec.rank}",
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
        logger.info(
            "sequence-parallel group forming: degree=%d devices=%s backend=%s "
            "port=%d", self.degree, list(self.devices), self.backend, port,
        )
        try:
            init_rank(specs[0])
        except BaseException as exc:  # noqa: BLE001
            self.close()
            raise RankGroupError(
                f"rank 0 failed to join the group: {type(exc).__name__}: {exc}"
            ) from exc
        self._formed = True
        return specs[0]

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

        if self.degree == 1 or not dist.is_initialized():
            return
        self.check_alive()
        dist.barrier()

    def close(self, *, grace_s: float = 10.0) -> None:
        import torch.distributed as dist

        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            logger.warning("destroying the process group failed", exc_info=True)
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
        self._formed = False
