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
group where one expert is sharded and the other is not is the divergence class
§5.4 is about.

**CP and CPU offload do not compose** (diffusers #12533: *"Enabling CPU offload
before enabling parallelism will raise a shape error after the first pipe
call"*). Offload/expert-swap is our standard answer below 80 GB, so sequence
parallelism is an 80 GB-class-card feature and the two levers must never be
armed together — a second, independent reason consumer multi-GPU is out.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Every module diffusers can shard declares a plan. No plan means the model was
# never adapted for CP, and installing anyway would shard something whose
# attention does not expect it.
_CP_PLAN_ATTR = "_cp_plan"


class ContextParallelUnavailable(RuntimeError):
    """This pipeline cannot be sharded, named exactly. A typed refusal — the
    hub demotes the pod; the worker never silently serves degree 1 against a
    degree-2 promise."""


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


def install_context_parallel(pipeline: Any, *, degree: int) -> Tuple[str, ...]:
    """Install Ulysses sequence parallelism at ``degree`` on every expert.

    Returns the component names that were sharded. Degree 1 installs nothing
    and returns ``()`` — a single-device group must be byte-identical to a
    worker that has never heard of this module.
    """
    if int(degree) <= 1:
        return ()

    from diffusers import ContextParallelConfig  # type: ignore

    components = _sharding_candidates(pipeline)
    if not components:
        raise ContextParallelUnavailable(
            "no pipeline component declares a `_cp_plan`; this model has not "
            "been adapted for context parallelism and sharding it would "
            "produce silently wrong output"
        )
    _refuse_if_offloaded(pipeline)

    config = ContextParallelConfig(ulysses_degree=int(degree))
    installed: List[str] = []
    for name, comp in components:
        comp.enable_parallelism(config=config)
        installed.append(name)
    logger.info(
        "context parallelism installed: ulysses_degree=%d components=%s",
        int(degree), installed,
    )
    return tuple(installed)


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


def refuse_unless_divisible(
    *, tokens: int, heads: int, degree: int
) -> None:
    """The diffusers #12536 assertion, checked BEFORE a pod is committed.

    Ulysses shards the sequence across ranks and the head dimension inside the
    all-to-all, so both must divide the degree. Our T2V shapes divide cleanly
    (40 heads, 75,600 tokens at 2/4/8); I2V's 769-token concat is what
    ``_cp_plan``'s own comment disables splitting for.
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
