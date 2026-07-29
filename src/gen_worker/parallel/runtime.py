"""SequenceRuntime — the seam where a `D>1` execution group becomes D ranks.

**The pipeline call is the SPMD unit, not the handler.** The endpoint handler
runs on rank 0 ONLY: it owns the payload, `ctx`, VAE decode, mp4 export and the
output path, exactly as SEQPAR-DESIGN §5.1 draws the line. What every rank must
execute identically is the *model call*, because that is where the collectives
live — so the runtime wraps the pipeline's ``__call__`` and, when rank 0 enters
it, broadcasts the call and has the followers execute the same call on their own
cards. Endpoint code never sees a rank, a degree or a device.

Why not run the handler on every rank: `RequestContext` is executor-owned, it
carries the job's capability token, deferred outputs, progress channel and
cancellation — a follower holding one would be a second worker, which is the
thing Paul's one-connection ruling forbids. Why not wrap only the transformer:
the denoise LOOP has to stay in lockstep, and it lives in the pipeline.

Restriction, refused typed rather than half-supported: sequence parallelism
requires a CLASS-annotated slot (``Slot(SomePipeline)``), because the follower
materializes its own copy through `provision.load_slot` from the pod's shared
CAS path. A self-loading (str/Path-slot) endpoint builds its pipeline inside
`setup()` with code the runtime cannot re-run on a follower.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cp import (
    ContextParallelUnavailable,
    install_context_parallel,
    refuse_unless_shard_invariant_quant,
)
from .group import RankGroup, RankGroupError, RankSpec, init_rank
from .plan import GroupPlan, broadcast_plan

logger = logging.getLogger(__name__)

_OP_RUN = "run"
_OP_CLOSE = "close"


@dataclass
class BootPlan:
    """Everything a follower needs to build the SAME pipeline. Small and
    picklable by construction — it crosses a spawn boundary and a collective."""

    modules: Tuple[str, ...] = ()
    function_name: str = ""
    slot: str = ""
    # slot -> the pod-shared CAS path. One copy of the bytes, N mappings.
    path: str = ""
    cache_dir: str = ""
    degree: int = 1
    dtype: str = ""
    storage_dtype: str = ""


def _send(obj: Any) -> None:
    import torch.distributed as dist

    box: List[Any] = [obj]
    dist.broadcast_object_list(box, src=0)


def _recv() -> Any:
    import torch.distributed as dist

    box: List[Any] = [None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


# ---------------------------------------------------------------------------
# Call marshalling: the model call must arrive at every rank IDENTICAL.
# ---------------------------------------------------------------------------


def _dehydrate(value: Any) -> Any:
    """Make one call argument crossable. CUDA tensors go via CPU (the
    followers own different cards); a `torch.Generator` is not picklable at
    all and is replaced by its seed, which is the only part that has to
    agree — every rank rebuilds an identical one."""
    import torch

    if isinstance(value, torch.Tensor):
        return ("__tensor__", value.detach().to("cpu"))
    if isinstance(value, torch.Generator):
        return ("__generator__", int(value.initial_seed()))
    if isinstance(value, (list, tuple)):
        marshalled = [_dehydrate(v) for v in value]
        return ("__list__", marshalled) if isinstance(value, list) else (
            "__tuple__", marshalled)
    if isinstance(value, dict):
        return ("__dict__", {k: _dehydrate(v) for k, v in value.items()})
    return value


def _rank_device(device: int) -> str:
    """A rank's device string. Falls back to CPU only where there is no CUDA
    at all — i.e. the gloo test rig; a real follower always has its card."""
    import torch

    return f"cuda:{int(device)}" if torch.cuda.is_available() else "cpu"


def _rehydrate(value: Any, device: int) -> Any:
    import torch

    dev = _rank_device(device)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        tag, payload = value
        if tag == "__tensor__":
            return payload.to(dev)
        if tag == "__generator__":
            return torch.Generator(device=dev).manual_seed(int(payload))
        if tag == "__list__":
            return [_rehydrate(v, device) for v in payload]
        if tag == "__tuple__":
            return tuple(_rehydrate(v, device) for v in payload)
        if tag == "__dict__":
            return {k: _rehydrate(v, device) for k, v in payload.items()}
    return value


# ---------------------------------------------------------------------------
# The follower's whole life.
# ---------------------------------------------------------------------------


def sequence_rank_main(spec: RankSpec) -> None:  # pragma: no cover - spawned
    """join -> obey the plan -> materialize -> {await call, run, barrier}.

    No hub connection, no store, no lifecycle, no receipts, no output path.
    Every adaptive decision arrives as a broadcast; nothing is decided here.
    """
    import torch
    import torch.distributed as dist

    from ..topology import set_device_group

    init_rank(spec)
    set_device_group(spec.rank)  # this rank's own bookkeeping identity
    torch.cuda.set_device(spec.device)

    plan = broadcast_plan(None, rank=spec.rank)
    boot: BootPlan = _recv()
    pipe = _materialize(boot, spec)
    refuse_unless_shard_invariant_quant(pipe, degree=plan.sp_degree)
    install_context_parallel(pipe, degree=plan.sp_degree)
    dist.barrier()

    while True:
        cmd = _recv()
        if not isinstance(cmd, dict) or cmd.get("op") == _OP_CLOSE:
            return
        if cmd.get("op") != _OP_RUN:
            logger.warning("rank %d: unknown command %r", spec.rank, cmd)
            continue
        args = tuple(_rehydrate(a, spec.device) for a in cmd.get("args", ()))
        kwargs = {k: _rehydrate(v, spec.device)
                  for k, v in (cmd.get("kwargs") or {}).items()}
        with torch.no_grad():
            pipe(*args, **kwargs)   # output discarded: rank 0 owns the output
        dist.barrier()


def _materialize(boot: BootPlan, spec: RankSpec) -> Any:  # pragma: no cover
    from ..models import provision
    from ..registry import collect_endpoints

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
        device=f"cuda:{spec.device}",
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
    ) -> None:
        self.devices = tuple(int(d) for d in devices)
        self.degree = len(self.devices)
        # The follower entry point. Overridable ONLY so an acceptance probe can
        # drive the shipped rank group against a bare model (no endpoint, no
        # CAS) — production always uses `sequence_rank_main`.
        self._entry = entry or sequence_rank_main
        self._group: Optional[RankGroup] = None
        self._pipe: Any = None
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self, pipe: Any, boot: BootPlan, plan: GroupPlan) -> Tuple[str, ...]:
        """Form the group, broadcast the plan, install CP on every rank.

        Ordering is forced and it is the whole reason this is one method:
        `set_attention_backend` (the endpoint's, already done by the time a
        pipeline is materialized) -> `enable_parallelism` -> `torch.compile`.
        """
        import torch.distributed as dist

        if self.degree <= 1:
            return ()
        plan.refuse_unless_cp_safe()
        refuse_unless_shard_invariant_quant(pipe, degree=self.degree)

        self._group = RankGroup(self.devices, backend="nccl",
                                entry=self._entry)
        self._group.form()
        broadcast_plan(plan, rank=0)
        _send(boot)
        installed = install_context_parallel(pipe, degree=self.degree)
        dist.barrier()
        self._pipe = pipe
        self._armed = True
        logger.info(
            "sequence-parallel group ARMED degree=%d devices=%s components=%s",
            self.degree, list(self.devices), list(installed))
        return installed

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Run ONE model call across the group. Rank 0's result is the
        request's; the followers' identical results are discarded."""
        import torch.distributed as dist

        if not self._armed:
            raise RankGroupError("sequence group is not armed")
        self._group.check_alive()  # type: ignore[union-attr]
        _send({"op": _OP_RUN,
               "args": tuple(_dehydrate(a) for a in args),
               "kwargs": {k: _dehydrate(v) for k, v in kwargs.items()}})
        try:
            out = self._pipe(*args, **kwargs)
        finally:
            # A rank that died mid-call must surface HERE, loudly, rather
            # than as a barrier that never returns.
            self._group.check_alive()  # type: ignore[union-attr]
        dist.barrier()
        return out

    def close(self) -> None:
        if self._group is None:
            return
        try:
            if self._armed:
                _send({"op": _OP_CLOSE})
        except Exception:  # noqa: BLE001 — teardown must not raise
            logger.warning("sequence group close broadcast failed", exc_info=True)
        finally:
            self._armed = False
            self._group.close()
            self._group = None
            self._pipe = None


_SP_FLAG = "_gen_worker_sp_gated"
_SP_ATTR = "_gen_worker_sp_runtime"


def arm_sequence_gate(pipe: Any, runtime: "SequenceRuntime") -> bool:
    """Route this pipeline's ``__call__`` through the group.

    Same dynamic-subclass technique as gw#551's LaneGate (object identity and
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


def _sequence_call_with(
    self: "SequenceRuntime", base_call: Any, pipe: Any, *args: Any, **kwargs: Any
) -> Any:
    import torch.distributed as dist

    if not self._armed:
        raise RankGroupError("sequence group is not armed")
    self._group.check_alive()  # type: ignore[union-attr]
    _send({"op": _OP_RUN,
           "args": tuple(_dehydrate(a) for a in args),
           "kwargs": {k: _dehydrate(v) for k, v in kwargs.items()}})
    try:
        out = base_call(pipe, *args, **kwargs)
    finally:
        # A rank that died mid-call must surface HERE, loudly, rather than as
        # a barrier that never returns.
        self._group.check_alive()  # type: ignore[union-attr]
    dist.barrier()
    return out


SequenceRuntime.call_with = _sequence_call_with  # type: ignore[attr-defined]
