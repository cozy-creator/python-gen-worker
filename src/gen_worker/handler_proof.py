"""pgw#1199: the does-it-run proof belongs to the process that HOLDS the weights.

WHAT THIS REPLACES, AND WHY IT IS A CORRECTION AND NOT A CLEANUP
----------------------------------------------------------------
pgw#984 proves the endpoint's own handler RUNS before a cell seals, because
``torch.export`` traces the declared graph classes off the modules directly and
never enters the handler — so a mint's phase table could read green for an
endpoint whose handler could not run at all (pgw#969: ``ctx.slots["pipeline"]``,
0.0 s into ``warmup_forward``).

It ran in the MINT CHILD, which on a weight-free mint holds no weights — so the
child first materialised REAL random values for every virtual parameter. That is
one full checkpoint at compute dtype, in the child, **concurrently with the
parent's resident copy**. Measured on pod ``729431an6ugbvq`` (H100-80, wan-2.2):
56.2 GB wanted, 15.5 GiB free, `CUDA out of memory` twice, no cell, eager for
life. §4.33's *"compile weight-free … VRAM cost is negligible"* was true of the
compile and false of the mint, and the ~8 GiB figure that anchored the subsystem
was measuring exactly this allocation for an sdxl-sized family.

§4.33 steps 4-5 already say where verification goes: *"load the cell into the
LIVE pipeline — already running eager — and verify it works"*, against weights
that are resident and are not paid for twice. The proof was in the wrong
process. This module is that correction, and the SDK had already made the same
call once: ``mint_child.execution_lane_verdict_for`` skips the kernel-lane A/B
on a meta mint because *"running it would put weight-scale values back in the
one process this slice exists to keep empty"*.

WHAT A PROOF IS
---------------
One warm forward through the endpoint's OWN handler, on the RESIDENT pipeline,
with REAL checkpoint values — strictly stronger than the child's random-value
re-run, and free on the fleet path, where the executor's boot warm plan already
runs every declared handler before a mint is ever delegated.

The record is a per-PROCESS ledger keyed by function name, because that is the
scope of the claim: *this* process ran *that* handler against the weights it is
about to mint for. It never crosses a process boundary as a ledger — the mint
request carries the PROVENANCE string, so the child's report says which proof
stood behind the cell it sealed rather than asserting one it cannot see.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: function name -> how it was proven (the provenance that rides the request).
_PROVEN: Dict[str, str] = {}
_LOCK = threading.Lock()


class HandlerProofFailed(Exception):
    """The endpoint's own handler does not run on the resident pipeline.

    A cell must not seal for a handler that cannot serve — pgw#984's sentence,
    unchanged. What moved is only WHERE the sentence is spoken: here, in the
    process that holds the weights, instead of in a child that had to allocate
    a second copy of them to say it.
    """


def record(function: str, how: str) -> None:
    """Note that ``function``'s handler ran to completion in THIS process."""
    name = str(function or "").strip()
    if not name:
        return
    with _LOCK:
        _PROVEN.setdefault(name, str(how or "handler ran"))


def provenance(function: str) -> str:
    """How ``function`` was proven here, or "" if it never was."""
    with _LOCK:
        return _PROVEN.get(str(function or "").strip(), "")


def proven(function: str) -> bool:
    return bool(provenance(function))


def _forget(function: str = "") -> None:
    """Drop the ledger (a whole process's, or one function's).

    Private, and it is meant to stay that way: production never un-proves a
    handler that ran, so a public name here would be public surface with no
    production caller. Tests reach it deliberately.
    """
    with _LOCK:
        if function:
            _PROVEN.pop(str(function), None)
        else:
            _PROVEN.clear()


# ---------------------------------------------------------------------------
# Running one, for the callers that do not get it for free
# ---------------------------------------------------------------------------


def warm_jobs(specs: Sequence[Any]) -> List[Any]:
    """The endpoint's own declared warm plan.

    Moved here from ``mint_child`` unchanged: the plan a proof runs and the
    plan the executor warms on must be the SAME derivation, or the two
    processes prove different graphs.
    """
    from . import warmup as warmup_mod
    from .api.decorators import ATTR as DECL_ATTR

    if not specs:
        raise HandlerProofFailed("no endpoint spec to derive a warm plan from")
    decl = getattr(specs[0].cls, DECL_ATTR, None)
    if decl is None:
        raise HandlerProofFailed(
            "the endpoint class carries no @endpoint declaration, so no warm "
            "plan can be derived")
    jobs, _skips = warmup_mod.plan(
        specs, decl_warmup=decl.warmup, has_warmup_method=False)
    jobs, _mode = warmup_mod.select_runs(jobs, tracing=True)
    if not jobs:
        raise HandlerProofFailed(
            "the derived warm plan is empty — there is nothing to prove")
    return jobs


def run_warm_job(
    instance: Any, job: Any, config: Dict[str, Any], execution_lane: str,
    origin: str = "",
) -> None:
    """One warm forward through the endpoint's OWN handler.

    Mirrors the executor's ``_invoke_warmup`` (bound handler, ctx+payload
    kwargs, stream consumed) — deliberately the same call shape, because the
    graphs this exercises are the graphs the pod will later have to hit.
    """
    import asyncio

    from . import warmup

    spec = job.spec
    with tempfile.TemporaryDirectory(prefix="gw-proof-") as tmp:
        payload = job.build(tmp)
        if payload is None:
            return
        # pgw#828: the SAME construction the executor's warm path uses. This
        # was three hand-rolled contexts, and one of them had no slots at all —
        # `ctx.slots["pipeline"]` raised `KeyError: 'pipeline'` on a real L4
        # after a 16.45 s load.
        ctx = warmup.warm_context(
            spec, request_id=f"handler-proof-{spec.name}",
            local_output_dir=tmp, execution_lane=execution_lane, config=config,
            origin=origin)
        bound = getattr(instance, spec.attr_name)
        kwargs = {spec.ctx_param: ctx, spec.payload_param: payload}
        if spec.is_async_gen:
            async def _drain() -> None:
                async for _ in bound(**kwargs):
                    pass

            asyncio.run(_drain())
        elif spec.is_async:
            asyncio.run(bound(**kwargs))
        else:
            out = bound(**kwargs)
            if spec.output_mode == "stream":
                for _ in out:
                    pass


def prove(
    instance: Any, specs: Sequence[Any], function: str, *,
    execution_lane: str = "", config: Optional[Dict[str, Any]] = None,
    origin: str = "",
) -> str:
    """Run ONE warm forward on the resident pipeline and record the proof.

    Returns the provenance string. Idempotent: a function already proven in
    this process costs nothing, which is what makes the fleet path free — the
    executor's boot warm plan has already run every declared handler by the
    time a mint is delegated.

    Raises :class:`HandlerProofFailed` when the handler does not run. That is
    fatal to the MINT and to nothing else: the caller keeps serving eager, and
    a pod whose handler cannot run was never going to serve a request either.
    """
    name = str(function or "").strip()
    already = provenance(name)
    if already:
        return already
    jobs = warm_jobs(specs)
    job = jobs[0]
    try:
        run_warm_job(
            instance, job, dict(config or {}), execution_lane, origin=origin)
    except Exception as exc:
        raise HandlerProofFailed(
            f"the endpoint's own warm plan does not run on the resident "
            f"pipeline — warm job {job.spec.name!r} raised "
            f"{type(exc).__name__}: {exc}. A cell must not seal for a handler "
            f"that cannot serve.") from exc
    how = f"resident warm forward {job.spec.name!r} (real weights)"
    record(name, how)
    return how


__all__ = [
    "HandlerProofFailed",
    "prove",
    "provenance",
    "proven",
    "record",
    "run_warm_job",
    "warm_jobs",
]
