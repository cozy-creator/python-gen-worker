"""Installing context parallelism on a pipeline — the runtime's job, never the
endpoint's.

The endpoint declares *how many devices one instance needs*; the worker owns
placement, and CP **is** placement. This is the same line ``Resources``'s own
docstring draws (*"endpoint code NEVER picks devices"*), so nothing in an
endpoint ever sees a degree or a rank.

**The ordering is forced:** ``set_attention_backend`` → ``enable_parallelism``
→ ``torch.compile``. CP is installed as diffusers ``ModelHook``s
(``ContextParallelSplitHook.pre_forward`` splits,
``ContextParallelGatherHook.post_forward`` gathers). Compiling *after* the
hooks puts the split/all-to-all/gather inside the traced graph; compiling first
and installing hooks after wraps a compiled callable in Python hooks — correct,
but the collectives fall outside the graph and the comm/compute overlap is
forfeit.

**``enable_parallelism`` is a ``ModelMixin`` method, not a pipeline one.**
Wan 2.2 A14B runs two experts (``transformer`` high-noise, ``transformer_2``
low-noise) with a mid-schedule handoff, so BOTH need the call — exactly as the
endpoint's own ``_use_cudnn_attention`` already iterates its expert tuple. A
group where one expert is sharded and the other is not diverges silently.

**CP and CPU offload do not compose** (diffusers #12533: *"Enabling CPU offload
before enabling parallelism will raise a shape error after the first pipe
call"*). Offload/expert-swap is our standard answer below 80 GB, so sequence
parallelism is an 80 GB-class-card feature and the two levers must never be
armed together — a second, independent reason consumer multi-GPU is out.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The gated-call flag.
#
# Installing CP puts split/all-to-all/gather hooks ON THE MODULES, so from that
# moment EVERY forward through them issues collectives — and the only seam that
# supplies participants on the other ranks is the pipeline-level ``__call__``
# gate. A forward from anywhere else (a hot-swap warm compile, a mint seed, a
# proof warmup, an activation/degraded probe, an endpoint calling a component
# by hand) runs on rank 0 alone and HANGS the whole group in NCCL.
#
# The flag is thread-local because that is exactly the failure boundary: the
# gate wraps the call on the requesting thread, while every stray forward comes
# from a different thread (the shared shape-warm thread, a background mint turn)
# or from outside the gate on the same one.
# ---------------------------------------------------------------------------

_tls = threading.local()


class gated_call:
    """Mark the dynamic extent in which collectives have participants."""

    def __enter__(self) -> None:
        _tls.active = getattr(_tls, "active", 0) + 1

    def __exit__(self, *exc: object) -> None:
        _tls.active = max(0, getattr(_tls, "active", 1) - 1)


def in_gated_call() -> bool:
    return bool(getattr(_tls, "active", 0))

# Every module diffusers can shard declares a plan. No plan means the model was
# never adapted for CP, and installing anyway would shard something whose
# attention does not expect it.
_CP_PLAN_ATTR = "_cp_plan"


class ContextParallelUnavailable(RuntimeError):
    """This pipeline cannot be sharded, named exactly. A typed refusal — the
    hub demotes the pod; the worker never silently serves degree 1 against a
    degree-2 promise."""


class UngatedShardedForward(ContextParallelUnavailable):
    """A forward through a CP-sharded module outside the group's call gate.
    Raised BY NAME, with the caller's thread, instead of hanging the group in a
    collective no other rank will ever join."""


@dataclass(frozen=True)
class CpComms:
    """The GROUP's communication facts, passed explicitly.

    ``pg`` is the group's NON-default ProcessGroup (from
    ``parallel.group.init_rank``), ``rank`` this process's rank within THAT
    group, ``device`` the rank's own card. CP is never installed against the
    default process group again: the worker process is rank 0 of every
    group, so "the default group" is not a well-defined thing to shard over.
    """

    pg: Any
    rank: int
    device: Any  # torch.device or str


class _GroupMesh:
    """The exact DeviceMesh surface diffusers' CP machinery consumes, backed
    by ONE explicit process group.

    diffusers ``ContextParallelConfig.setup`` performs
    ``mesh["ring", "ulysses"]._flatten()`` / ``mesh["ring"]`` /
    ``mesh["ulysses"]`` / ``get_local_rank()``, and the hooks + attention
    dispatch then use only ``.size()``, ``.get_group()`` and
    ``dist.get_rank/get_world_size(group)``. Ring degree is always 1 here
    (Ulysses only), so the ring axis is a size-1 shim whose ``get_group`` is
    a typed refusal — nothing may ever run a ring collective through it. A test
    pins this contract against the installed diffusers so an upstream change
    breaks loudly, not silently.
    """

    def __init__(self, pg: Any, size: int, local_rank: int, *, axis: str = "ulysses") -> None:
        self._pg = pg
        self._size = int(size)
        self._local_rank = int(local_rank)
        self._axis = axis

    def size(self) -> int:
        return self._size

    def get_group(self) -> Any:
        if self._axis == "ring":
            raise ContextParallelUnavailable(
                "ring attention is not supported: the platform installs "
                "Ulysses sequence parallelism only (ring_degree=1)"
            )
        return self._pg

    def get_local_rank(self) -> int:
        return self._local_rank

    def _flatten(self) -> "_GroupMesh":
        return _GroupMesh(self._pg, self._size, self._local_rank)

    def __getitem__(self, key: Any) -> "_GroupMesh":
        if isinstance(key, tuple):
            return _GroupMesh(self._pg, self._size, self._local_rank)
        if key == "ring":
            return _GroupMesh(self._pg, 1, 0, axis="ring")
        if key == "ulysses":
            return _GroupMesh(self._pg, self._size, self._local_rank)
        raise KeyError(key)

    def __repr__(self) -> str:  # pragma: no cover - logging sugar
        return f"_GroupMesh(size={self._size}, rank={self._local_rank}, axis={self._axis})"


def _sharding_candidates(pipeline: Any) -> List[Tuple[str, Any]]:
    """Every component that declares a CP plan, in declaration order.

    Discovered from the objects themselves rather than from a hard-coded
    expert tuple: a two-expert MoE, a single DiT and a future three-stage
    model are all just "the components that declare a plan".
    """
    out: List[Tuple[str, Any]] = []
    candidates = list(getattr(pipeline, "components", {}) or {})
    for extra in ("transformer", "transformer_2"):
        if extra not in candidates:
            candidates.append(extra)
    for name in candidates:
        comp = getattr(pipeline, name, None)
        if comp is None:
            continue
        if getattr(comp, _CP_PLAN_ATTR, None) and hasattr(
            comp, "enable_parallelism"
        ):
            if not any(c is comp for _n, c in out):
                out.append((name, comp))
    return out


def install_context_parallel(
    pipeline: Any, *, degree: int, comms: Optional[CpComms] = None,
) -> Tuple[str, ...]:
    """Install Ulysses sequence parallelism at ``degree`` on every expert.

    Returns the component names that were sharded. Degree 1 installs nothing
    and returns ``()`` — a single-device group must be byte-identical to a
    worker that has never heard of this module.

    ``comms`` is REQUIRED at degree>1: installation binds the CP
    hooks to that group's explicit process group. diffusers'
    ``enable_parallelism`` reads rank/world/device from the DEFAULT process
    group, which in the worker process (rank 0 of every group) is a size-1
    local group — so this replicates its install sequence with group-correct
    facts instead of calling it.
    """
    if int(degree) <= 1:
        return ()
    components = _sharding_candidates(pipeline)
    if not components:
        raise ContextParallelUnavailable(
            "no pipeline component declares a `_cp_plan`; this model has not "
            "been adapted for context parallelism and sharding it would "
            "produce silently wrong output"
        )
    _refuse_if_offloaded(pipeline)
    if comms is None:
        raise ContextParallelUnavailable(
            "context parallelism at degree>1 requires the group's explicit "
            "process group (CpComms); installing against the default process "
            "group is exactly the pgw#773 cross-group corruption"
        )

    installed: List[str] = []
    for name, comp in components:
        # diffusers #12536, at the one moment it costs nothing: the all-to-all
        # shards the HEAD dimension, so an expert whose declared head count
        # does not divide the degree fails as an inscrutable shape mismatch
        # mid-denoise, on a paying request, on a pod that has already rented.
        # The head count is a property of the MODEL and is knowable here; the
        # sequence length is a property of a REQUEST and is not, so it is not
        # stated (see `refuse_unless_divisible`).
        refuse_unless_divisible(
            tokens=0, heads=_declared_heads(comp), degree=int(degree))
        _install_on_component(comp, degree=int(degree), comms=comms)
        _install_gate_guard(comp, name)
        installed.append(name)
    logger.info(
        "context parallelism installed: ulysses_degree=%d components=%s "
        "group_rank=%d", int(degree), installed, int(comms.rank),
    )
    return tuple(installed)


_GUARD_ATTR = "_gen_worker_cp_gate_guard"


def _install_gate_guard(comp: Any, name: str) -> None:
    """Refuse an ungated forward through a sharded module BY NAME.

    The hooks are on the module, so the module is where the failure domain is
    knowable. A forward pre-hook is the cheapest possible check (one thread-
    local read per call, no wrapper object, no change to the module's class, so
    `torch.compile`'s class keying is untouched) and it fires on rank 0 exactly
    where a stray forward would otherwise enter an unmatched collective and
    hang every rank of the group.
    """
    if getattr(comp, _GUARD_ATTR, None) is not None:
        return

    def _guard(module: Any, args: Any, kwargs: Any = None) -> None:
        if in_gated_call():
            return
        raise UngatedShardedForward(
            f"forward through the context-parallel component {name!r} outside "
            "the group's call gate: this rank would issue collectives no other "
            "rank is in. Every forward at degree>1 must go through the gated "
            "pipeline call (compile warmups, mint seeds, proof warmups, "
            "activation/degraded probes and direct component calls are "
            "disabled under context parallelism — pgw#775)"
        )

    handle = comp.register_forward_pre_hook(_guard, with_kwargs=True)
    try:
        setattr(comp, _GUARD_ATTR, handle)
    except Exception:  # noqa: BLE001 - frozen modules keep the hook anyway
        pass


def _install_on_component(comp: Any, *, degree: int, comms: CpComms) -> None:
    """The group-aware half of diffusers' ``ModelMixin.enable_parallelism``.

    Same sequence, same objects: backend-compatibility check, a ParallelConfig
    set up with THIS group's rank/world/device/mesh, ``_parallel_config``
    stamped on the model and every attention processor, then
    ``apply_context_parallel`` installs the split/gather hooks. The only
    deltas are the mesh (an explicit-group ``_GroupMesh``, never
    ``init_device_mesh`` over the default world) and rank/world/device (the
    group's, never ``dist.get_rank()``'s). Pinned to diffusers 0.39.x by the
    serving image; a surface change raises typed here instead of corrupting.
    """
    import torch

    try:
        from diffusers.hooks.context_parallel import apply_context_parallel
        from diffusers.models._modeling_parallel import (
            ContextParallelConfig,
            ParallelConfig,
        )
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
        )
        from diffusers.models.attention_processor import Attention
    except Exception as exc:  # noqa: BLE001 - version drift must be typed
        raise ContextParallelUnavailable(
            f"diffusers CP surface unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        from diffusers.models.attention_processor import MochiAttention

        attention_classes: tuple = (Attention, MochiAttention, AttentionModuleMixin)
    except Exception:  # pragma: no cover - model-zoo dependent
        attention_classes = (Attention, AttentionModuleMixin)

    cp_config = ContextParallelConfig(ulysses_degree=int(degree))
    config = ParallelConfig(context_parallel_config=cp_config)

    # Backend compatibility, exactly as upstream checks it: the active
    # attention backend must support CP or the hooks shard tensors an
    # incompatible kernel then mis-reads.
    for module in comp.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_attention_backend"):
            continue
        backend = processor._attention_backend
        if backend is None:
            backend, _ = _AttentionBackendRegistry.get_active_backend()
        else:
            backend = AttentionBackendName(backend)
        if not _AttentionBackendRegistry._is_context_parallel_available(backend):
            supported = sorted(_AttentionBackendRegistry._supports_context_parallel)
            raise ContextParallelUnavailable(
                f"attention backend {backend.value!r} does not support context "
                f"parallelism; set one of {supported} before arming"
            )
        break

    device = comms.device if isinstance(comms.device, torch.device) else torch.device(str(comms.device))
    mesh = _GroupMesh(comms.pg, int(degree), int(comms.rank))
    config.setup(int(comms.rank), int(degree), device, mesh=cast(Any, mesh))
    comp._parallel_config = config
    for module in comp.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_parallel_config"):
            processor._parallel_config = config

    plan = getattr(comp, "_cp_plan", None)
    if not plan:
        raise ContextParallelUnavailable(
            f"{type(comp).__name__} lost its `_cp_plan` between candidate "
            "discovery and install"
        )
    apply_context_parallel(comp, cp_config, plan)


def _refuse_if_offloaded(pipeline: Any) -> None:
    """diffusers #12533: CPU offload enabled BEFORE parallelism raises a shape
    error after the first pipe call. Catch it here, named, instead of as an
    inscrutable shape mismatch mid-denoise."""
    markers = (
        "_all_hooks",              # accelerate cpu_offload bookkeeping
        "_offload_gpu_id",
        "hf_device_map",
    )
    for marker in markers:
        if getattr(pipeline, marker, None):
            raise ContextParallelUnavailable(
                f"pipeline carries {marker!r}: CPU offload and context "
                "parallelism do not compose (diffusers #12533). Sequence "
                "parallelism is an 80GB-class-card feature; a group that "
                "needs offload must serve at degree 1."
            )


def _declared_heads(comp: Any) -> int:
    """A sharding candidate's declared attention-head count, or 0.

    Read off the DECLARATION (the model config), never counted from modules: a
    module walk answers a different question (how many attention blocks) and
    would produce a confident wrong number. 0 means the component declares
    none, and an undeclared count is not a divisibility failure.
    """
    config = getattr(comp, "config", None)
    for attr in ("num_attention_heads", "num_heads", "attention_heads"):
        for holder in (config, comp):
            if holder is None:
                continue
            try:
                heads = int(getattr(holder, attr, 0) or 0)
            except (TypeError, ValueError):
                continue
            if heads > 0:
                return heads
    return 0


def refuse_unless_divisible(
    *, tokens: int, heads: int, degree: int
) -> None:
    """The diffusers #12536 assertion, checked BEFORE a pod is committed.

    Ulysses shards the sequence across ranks and the head dimension inside the
    all-to-all, so both must divide the degree.

    Either term may be 0 — "this caller cannot state it" — and 0 asserts
    nothing rather than asserting falsely. ``install_context_parallel`` is
    exactly such a caller: the head count is a property of the MODEL and is
    knowable before the pod commits to a packing; the sequence length is a
    property of a REQUEST and is not knowable at install.
    """
    d = int(degree)
    if d <= 1:
        return
    if int(tokens) % d:
        raise ContextParallelUnavailable(
            f"sequence length {tokens} is not divisible by ulysses_degree {d}"
        )
    if int(heads) % d:
        raise ContextParallelUnavailable(
            f"attention head count {heads} is not divisible by "
            f"ulysses_degree {d}"
        )


def refuse_unless_shard_invariant_quant(pipeline: Any, *, degree: int) -> None:
    """W8A8 composes with CP — with one exception that must be typed.

    ``rowwise`` derives a scale per TOKEN, which is a function of that token
    alone and therefore identical whichever rank holds it. ``pertensor``
    derives one scalar from the whole activation, i.e. from the LOCAL shard —
    so two ranks quantize the same logical tensor differently and the group
    produces silently wrong output. That is the quiet failure mode, which is
    why this refuses rather than warns.
    """
    if int(degree) <= 1:
        return
    bad: List[str] = []
    for name, module in _named_modules(pipeline):
        mode = getattr(module, "gemm_mode", "")
        if mode and mode != "rowwise":
            bad.append(f"{name}={mode}")
            if len(bad) >= 4:
                break
    if bad:
        raise ContextParallelUnavailable(
            "w8a8 gemm_mode must be `rowwise` under context parallelism: a "
            "per-tensor activation scale is derived from the local sequence "
            "shard, so each rank quantizes differently and the output is "
            f"silently wrong. Found {', '.join(bad)}"
        )


def _named_modules(pipeline: Any) -> Sequence[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    seen: set = set()
    comps = getattr(pipeline, "components", None) or {}
    for cname, comp in (comps.items() if hasattr(comps, "items") else []):
        named = getattr(comp, "named_modules", None)
        if not callable(named):
            continue
        for mname, module in named():
            key = id(module)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"{cname}.{mname}" if mname else cname, module))
    return out


def w8a8_gemm_mode(pipeline: Any) -> str:
    """The one w8a8 GEMM mode this pipeline's quantized linears run, or "".

    Reported into the GroupPlan so the mode is a GROUP fact that rank 0
    decides, not something each rank re-derives from its own SM (which is
    exactly how two ranks end up quantizing differently)."""
    modes = {
        m for _n, mod in _named_modules(pipeline)
        if (m := str(getattr(mod, "gemm_mode", "") or ""))
    }
    if len(modes) == 1:
        return modes.pop()
    return "" if not modes else "mixed"
