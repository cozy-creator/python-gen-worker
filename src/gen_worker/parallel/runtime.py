"""SequenceRuntime — the seam where a `D>1` execution group becomes D ranks.

**The pipeline call is the SPMD unit, not the handler.** The endpoint handler
runs on rank 0 ONLY: it owns the payload, `ctx`, VAE decode, mp4 export and the
output path, exactly as SEQPAR-DESIGN §5.1 draws the line. What every rank must
execute identically is the *model call*, because that is where the collectives
live — so the runtime wraps the pipeline's ``__call__`` and, when rank 0 enters
it, sends the call over the command channel and has the followers execute the
same call on their own cards. Endpoint code never sees a rank, a degree or a
device.

Why not run the handler on every rank: `RequestContext` is executor-owned, it
carries the job's capability token, deferred outputs, progress channel and
cancellation — a follower holding one would be a second worker, which is the
thing Paul's one-connection ruling forbids. Why not wrap only the transformer:
the denoise LOOP has to stay in lockstep, and it lives in the pipeline.

**Error doctrine.** An exception cannot interrupt an SPMD region
symmetrically: when rank 0 raises mid-call (OOM, deadline, cancel, endpoint
bug) its peers are somewhere inside the same call waiting on collectives rank
0 will never issue. There is no barrier that repairs that — only two honest
outcomes: the peers' collectives time out TYPED (the process group carries a
real timeout), and the group is CONDEMNED — the request fails with the
original error, the runtime refuses further calls by name, teardown runs off
the event loop, and the record goes stale so the next setup re-arms a fresh
group. The worker keeps serving its other groups throughout.

Restriction, refused typed rather than half-supported: sequence parallelism
requires a CLASS-annotated slot (``Slot(SomePipeline)``), because the follower
materializes its own copy through `provision.load_slot` from the pod's shared
CAS path. A self-loading (str/Path-slot) endpoint builds its pipeline inside
`setup()` with code the runtime cannot re-run on a follower.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Tuple

from . import wire
from .cp import (
    ContextParallelUnavailable,
    CpComms,
    gated_call,
    install_context_parallel,
    refuse_unless_shard_invariant_quant,
)
from .group import (
    FollowerChannel,
    RankGroup,
    RankGroupError,
    RankSpec,
    _FIRST_COMMAND_WAIT_S,
    init_rank,
)
from .plan import BootPlan, GroupPlan
from .cp import w8a8_gemm_mode
from ..models import provision
from ..registry import collect_endpoints
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Call marshalling lives in `wire`: the model call must arrive at every rank
# IDENTICAL, and its bytes must not be able to execute anything.
# ---------------------------------------------------------------------------


def _rank_device(device: int) -> str:
    """A rank's device string. Falls back to CPU only where there is no CUDA
    at all — i.e. the gloo test rig; a real follower always has its card."""
    return f"cuda:{int(device)}" if cuda_ready() else "cpu"


def _force_collectives(value: Any) -> Any:
    """Materialize any lazy functional-collective tensors in a call result.

    diffusers' gather hook returns ``AsyncCollectiveTensor``s whose actual
    all-gather runs on first use. The gated call is the LAST point where the
    group's failure domain is visible — the handler that consumes the output
    is plain endpoint code — so completion is forced here: a peer that left
    the collective surfaces as a typed in-call error (bounded by the process
    group's timeout), never as a hang at some later `.cpu()`.
    """
    import torch

    try:
        from torch.distributed._functional_collectives import (
            AsyncCollectiveTensor,
        )
    except Exception:  # pragma: no cover - ancient torch
        return value
    if isinstance(value, AsyncCollectiveTensor):
        return value.wait()
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        return [_force_collectives(v) for v in value]
    if isinstance(value, tuple):
        out = tuple(_force_collectives(v) for v in value)
        return type(value)(*out) if hasattr(value, "_fields") else out
    if isinstance(value, dict):
        return {k: _force_collectives(v) for k, v in value.items()}
    for attr in ("frames", "images", "videos", "audios"):
        try:
            inner = getattr(value, attr, None)
        except Exception:  # pragma: no cover - exotic output objects
            continue
        if inner is not None:
            try:
                setattr(value, attr, _force_collectives(inner))
            except Exception:  # pragma: no cover - frozen output objects
                pass
    return value


# The gated-call flag lives with the hooks it guards (cp.py); re-exported here
# because the gate is what enters it.
_gated_call = gated_call


# ---------------------------------------------------------------------------
# The follower's whole life.
# ---------------------------------------------------------------------------


def sequence_rank_main(spec: RankSpec, channel: FollowerChannel) -> None:  # pragma: no cover - spawned
    """join -> obey the plan -> materialize -> {await command, run}.

    No hub connection, no store, no lifecycle, no receipts, no output path.
    Every adaptive decision arrives from rank 0; nothing is decided here. A
    follower that cannot honour the plan raises RankDivergence — fatal for
    the group, never a local fallback (§5.4).
    """
    import torch

    pg = init_rank(spec)
    if spec.backend == "nccl":
        torch.cuda.set_device(spec.device)

    first = channel.next_command(timeout=_FIRST_COMMAND_WAIT_S)
    if not isinstance(first, wire.Arm):
        raise RankGroupError(
            f"rank {spec.rank}: expected the arm command first, got {first!r}")
    boot, plan = first.boot, first.plan
    plan.refuse_unless_cp_safe()
    pipe = _materialize(boot, spec)
    refuse_unless_shard_invariant_quant(pipe, degree=plan.sp_degree)
    # This rank's own view of the group facts, held against rank 0's
    # broadcast plan: a disagreement (mismatched card, different toolchain)
    # fails the group LOUDLY — a rank never adapts locally.

    mine = GroupPlan(
        precision_execution_lane=plan.precision_execution_lane,
        gemm_mode=w8a8_gemm_mode(pipe) or plan.gemm_mode,
        degraded_plan=plan.degraded_plan,
        compile_armed=plan.compile_armed,
        compiled_graph_key=plan.compiled_graph_key,
        loras=plan.loras,
        sp_degree=spec.world_size,
    )
    mine.assert_agrees(plan, rank=spec.rank)
    install_context_parallel(
        pipe, degree=plan.sp_degree,
        comms=CpComms(pg=pg, rank=spec.rank, device=_rank_device(spec.device)),
    )
    channel.report_ready(spec.rank)

    while True:
        cmd = channel.next_command()
        if isinstance(cmd, wire.Close):
            return
        if not isinstance(cmd, wire.Run):
            logger.warning("rank %d: unknown command %r", spec.rank, cmd)
            continue
        args, kwargs = wire.run_call(cmd, device=_rank_device(spec.device))
        with torch.no_grad(), _gated_call():
            pipe(*args, **kwargs)   # output discarded: rank 0 owns the output


def _materialize(boot: BootPlan, spec: RankSpec) -> Any:  # pragma: no cover

    if boot.cache_dir:
        os.environ["TENSORHUB_CACHE_DIR"] = boot.cache_dir
    specs = collect_endpoints(list(boot.modules))
    match = next((s for s in specs if s.name == boot.function_name), None)
    if match is None:
        raise RankGroupError(
            f"rank {spec.rank}: function {boot.function_name!r} not found in "
            f"{list(boot.modules)} — the group cannot be SPMD")
    binding = match.models.get(boot.slot)
    annotation = _slot_annotation(match, boot.slot)
    load = provision.load_slot(
        annotation, boot.path, binding=binding, slot=boot.slot,
        device=_rank_device(spec.device),
    )
    if not load.is_pipeline:
        raise ContextParallelUnavailable(
            f"slot {boot.slot!r} is not a class-annotated pipeline slot; "
            "sequence parallelism cannot rebuild a self-loaded pipeline on a "
            "follower rank")
    return load.obj


def _slot_annotation(spec: Any, slot: str) -> Any:
    decl = spec.slots.get(slot)
    target = getattr(decl, "annotation", None) or getattr(decl, "type", None)
    if target is not None:
        return target
    binding = spec.models.get(slot)
    return getattr(binding, "annotation", None)


# ---------------------------------------------------------------------------
# Rank 0's handle on the group.
# ---------------------------------------------------------------------------


class SequenceRuntime:
    """One armed degree-D group. Owned by the executor record it serves."""

    def __init__(
        self, devices: Tuple[int, ...], *, entry: Optional[Any] = None,
        backend: str = "nccl",
        collective_timeout_s: Optional[float] = None,
    ) -> None:
        self.devices = tuple(int(d) for d in devices)
        self.degree = len(self.devices)
        # The follower entry point and gloo backend are overridable ONLY so
        # an acceptance probe or the CPU rig can drive the shipped rank group
        # against a bare model (no endpoint, no CAS) — production always uses
        # `sequence_rank_main` over NCCL.
        self._entry = entry or sequence_rank_main
        self._backend = backend
        self._collective_timeout_s = collective_timeout_s
        self._group: Optional[RankGroup] = None
        self._pipe: Any = None
        self._armed = False
        self._broken = ""       # non-empty = condemned, with the reason
        self._close_lock = threading.Lock()

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def broken(self) -> str:
        return self._broken

    @property
    def process_group(self) -> Any:
        return self._group.process_group if self._group is not None else None

    def arm(self, pipe: Any, boot: BootPlan, plan: GroupPlan) -> Tuple[str, ...]:
        """Form the group, deliver the plan, install CP on every rank.

        Ordering is forced and it is the whole reason this is one method:
        `set_attention_backend` (the endpoint's, already done by the time a
        pipeline is materialized) -> `enable_parallelism` -> (compile —
        disabled at degree>1).
        """
        if self.degree <= 1:
            return ()
        plan.refuse_unless_cp_safe()
        refuse_unless_shard_invariant_quant(pipe, degree=self.degree)

        kwargs: dict = {"backend": self._backend, "entry": self._entry}
        if self._collective_timeout_s is not None:
            kwargs["collective_timeout_s"] = self._collective_timeout_s
        self._group = RankGroup(self.devices, **kwargs)
        try:
            spec0 = self._group.form()
            self._group.send(wire.Arm(boot=boot, plan=plan))
            installed = install_context_parallel(
                pipe, degree=self.degree,
                comms=CpComms(
                    pg=self._group.process_group,
                    rank=0,
                    device=_rank_device(spec0.device),
                ),
            )
            # Queue-based readiness: a follower still materializing its
            # pipeline parks nobody on a collective, and one that died is a
            # loud typed failure here.
            self._group.wait_armed()
        except BaseException:
            group, self._group = self._group, None
            if group is not None:
                group.close()
            raise
        self._pipe = pipe
        self._armed = True
        logger.info(
            "sequence-parallel group ARMED degree=%d devices=%s components=%s",
            self.degree, list(self.devices), list(installed))
        return installed

    def call_with(
        self, base_call: Any, pipe: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Run ONE model call across the group. Rank 0's result is the
        request's; the followers' identical results are discarded."""
        if self._broken:
            raise RankGroupError(
                f"sequence group condemned, refusing the call: {self._broken}")
        if not self._armed:
            raise RankGroupError("sequence group is not armed")
        group = self._group
        assert group is not None
        group.check_alive()
        # Marshalled BEFORE the queue, on this thread: an argument outside the
        # wire's vocabulary is a typed per-request error with the group
        # coherent, never a follower that decoded half a command.
        try:
            command = wire.run_command(args, kwargs)
        except wire.UncrossableArgument as exc:
            raise RankGroupError(
                f"command cannot cross the rank boundary: {exc}") from exc
        group.send(command)
        try:
            with _gated_call():
                out = _force_collectives(base_call(pipe, *args, **kwargs))
        except BaseException as exc:
            # Rank 0 left the SPMD region mid-call; the followers are inside
            # it waiting on collectives that will never come. No barrier can
            # repair that — condemn the group (their collectives time out
            # typed; teardown off-thread) and surface the ORIGINAL error.
            self._condemn(
                f"rank 0 raised mid-call: {type(exc).__name__}: {exc}")
            raise
        try:
            # A follower that died mid-call must surface HERE, loudly, rather
            # than as output silently rendered at a lower effective degree.
            group.check_alive()
        except RankGroupError as exc:
            self._condemn(str(exc))
            raise
        return out

    def _condemn(self, reason: str) -> None:
        """Mark the group unusable by name and tear it down OFF this thread's
        critical path. Idempotent; the record owning this runtime goes stale
        via the executor's failure handling and re-arms a fresh group."""
        if self._broken:
            return
        self._broken = reason[:500]
        self._armed = False
        logger.error("sequence group CONDEMNED: %s", self._broken)
        threading.Thread(
            target=self.close, name="sp-condemn-close", daemon=True
        ).start()

    def close(self) -> None:
        """Teardown without collectives: CLOSE over the command channel, then
        join/terminate. Safe from any thread; must not run ON the event loop
        (the executor calls it via a worker thread) because it joins
        processes."""
        with self._close_lock:
            group, self._group = self._group, None
            self._armed = False
            self._pipe = None
            if group is None:
                return
            group.close(grace_s=2.0 if self._broken else 10.0)


_SP_FLAG = "_gen_worker_sp_gated"
_SP_ATTR = "_gen_worker_sp_runtime"


def arm_sequence_gate(pipe: Any, runtime: "SequenceRuntime") -> bool:
    """Route this pipeline's ``__call__`` through the group.

    Same dynamic-subclass technique as the LaneGate (object identity and
    isinstance preserved, idempotent), and it COMPOSES with it: each wrap
    subclasses the previous class, so a lane-gated pipeline stays lane-gated.
    The endpoint's handler calls ``pipe(...)`` exactly as it always has and
    never learns that D processes ran it.
    """
    if pipe is None or runtime.degree <= 1:
        return False
    if getattr(type(pipe), _SP_FLAG, False):
        object.__setattr__(pipe, _SP_ATTR, runtime)
        return True
    cls = type(pipe)
    if not any("__call__" in vars(k) for k in cls.__mro__):
        return False
    base_call = cls.__call__

    def _sp_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        rt = getattr(self, _SP_ATTR, None)
        if rt is None or not rt.armed:
            return base_call(self, *args, **kwargs)
        return rt.call_with(base_call, self, *args, **kwargs)

    try:
        gated = type(cls.__name__, (cls,), {
            "__call__": _sp_call,
            _SP_FLAG: True,
            "__module__": cls.__module__,
        })
        pipe.__class__ = gated
    except Exception:  # noqa: BLE001 - a slotted/immutable pipeline cannot be
        # gated; refuse loudly rather than serve degree-1 against a degree-D
        # promise.
        logger.exception("could not arm the sequence gate on %s", cls.__name__)
        return False
    object.__setattr__(pipe, _SP_ATTR, runtime)
    return True
