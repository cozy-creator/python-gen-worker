"""pgw#784: the mint child — one cell, one process, then exit.

``python -m gen_worker.mint_child <request.json>``

Run by ``mint_process.run_mint`` on a compile-cell cache MISS while the
serving process keeps serving eager and heartbeating. Also runnable by hand
from the request file a failed mint leaves behind, which is the whole reason
the boundary is a file.

What it does, and why in this order
-----------------------------------
1. **Seal the environment** exactly as a worker boot does. The cell key
   digests the inductor config; a child that sealed differently would stamp a
   cell the parent's ``verify()`` then rejects on an axis nobody changed.
2. **Cap its own VRAM** (``torch.cuda.set_per_process_memory_fraction``). The
   parent reserved a share for this child and kept the rest for the tenant;
   this is what makes that reservation an ENFORCED bound instead of a hope.
   A child that wants more OOMs ITSELF — a failed mint reported by a live
   worker — rather than stealing the peak out from under a live request.
3. **Install the parent's RESOLVED slot bindings** (pgw#969) onto the specs
   this process rediscovered, before a weight is read. Rediscovery yields
   only what ``@endpoint`` DECLARED, and a hub-catalog slot
   (``Slot(selected_by=...)`` with no ``default_checkpoint=``) declares
   nothing — so without this the endpoint's own handler reaches
   ``ctx.slots["pipeline"]`` unbound and the mint dies 0.0 s into
   ``warmup_forward``. A request that still cannot resolve a declared slot
   REFUSES here, by name, rather than nine seconds later inside the endpoint.
4. **Load the endpoint's own pipeline**, through ``cli.run.run_setup``: the
   endpoint's real ``setup()``/``warmup()``, the production ``provision``
   loader, the same already-materialized weights on local disk. No network.
5. **Arm COLD and drive the endpoint's OWN derived warm plan** — never
   ``mint_artifact``'s producer-style ``_compile_and_warm``. That distinction
   is gw#586/gw#587's whole lesson: a synthetic single-stage warm call can
   trace DIFFERENT FX graphs than a conditioned/two-stage endpoint's real
   warmup, and a cell packed from the wrong graphs bricks every adopting boot.
   So the child runs ``warmup.plan`` over the same sibling spec set the parent
   would have, through the endpoint's own handler. Same code, same shapes,
   same seal — a different PROCESS, not a different execution.

   The AOT recipe traces with ``torch.export`` instead, so its cell is not
   derived from these forwards — but it runs ONE of them anyway (pgw#984),
   before exporting. A recipe that never enters the handler cannot tell a
   working endpoint from one whose forward dies on its first request, and a
   green mint that sealed a cell for the latter is the shape pgw#969 cost
   four hours to find on a pod.
6. **Pack** the exported cell and write a typed report.

The parity claim is checked, not asserted
-----------------------------------------
The parent adopts this artifact through the ordinary DELIVERED-cell path and
then runs its own warmup proof against it. If the child's traced graphs are
not the ones the parent's serving code compiles, the parent's proof MISSES and
the cell is neither advertised nor published — the same gw#607 per-object
proof that gates a hub-delivered cell. In-process capture made that proof
tautological (the capture was byte-derived from the proof); out-of-process
makes it load-bearing. A parity gap therefore degrades to "eager, cell
absent", never to a poisoned published cell.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

from . import warm_spans, worker_goals
from .api.errors import ValidationError
from .config import load_settings
from .mint_process import (
    EXIT_BAD_REQUEST,
    EXIT_MINTED,
    EXIT_REFUSED,
    EXIT_RESOURCE,
    MintReport,
    MintRequest,
    MintSlot,
    frame_line,
)

logger = logging.getLogger(__name__)

#: pgw#1010: the child builds ONE artifact kind. The dynamo recipe it used to
#: also run produced a cell with no consumer, and it is deleted rather than
#: kept behind a request field nobody may set.
RECIPE_AOT = "aot"


class MintChildRefused(RuntimeError):
    """A named, deterministic reason this mint cannot happen here.

    Never retried by the parent: re-running a named refusal buys a second
    billed compile for the same sentence.

    ``mint_phases`` carries the refusing mint's PARTIAL phase table when it
    got far enough to have one (pgw#825) — the entries it exported and
    compiled before refusing are real minutes and must reach the hub.
    """

    def __init__(
        self, *args: Any, mint_phases: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(*args)
        self.mint_phases: Dict[str, Any] = dict(mint_phases or {})


def _declaration_refusal(exc: ValidationError) -> MintChildRefused:
    """pgw#1075: a declared value this SDK's own vocabulary rejects is a
    REFUSAL, not a crash — and the author's fix is already in the message.

    ``api.errors.ValidationError`` means, verbatim, "bad user input; do not
    retry". Inside this process the "user input" IS the declaration: the
    family's ``Compile`` block, the mint request's spec, the axis values the
    endpoint declares. Every property the parent's retry policy reads off a
    refusal already holds for it — deterministic, identical on the next card,
    fixed by editing the declaration and by nothing else — so it took the one
    exit that says none of that.

    Measured on the rig 2026-08-09: a vehicle declaring ``lora_bucket=8``
    (``RANK_BUCKETS = (16, 32, 64, 128)``) makes ``enable_lora_branches``
    raise its typed ``ValidationError`` — the refusal is CORRECT — and the
    child reported ``mint-child crashed: the mint process exited 1`` with a
    truncated traceback tail. The sentence naming the fix, and the fact that
    it was a refusal at all, both died at the process boundary. pgw#999's
    rule: refusals carry a class.

    The wrapper is the same shape ``aot_mint.export_program`` already applies
    to a custom op with no fake kernel (pgw#1062) — the message is the
    authoring contract, so it is carried whole and never summarised.
    """
    return MintChildRefused(
        f"declaration refused: {type(exc).__name__}: {exc}",
        mint_phases=getattr(exc, "mint_phases", None))


# th#1322: per-phase spans, measured HERE because this process owns the clock
# the phases run on. `frame()` is the single funnel every phase transition goes
# through, so accumulating on transition needs no second bookkeeping path that
# could drift out of step with what the parent sees.
_PHASE_SPANS: Dict[str, float] = {}
_PHASE_OPEN: Tuple[str, float] = ("", 0.0)


def _rotate_phase(phase: str) -> None:
    """Close the open phase's span and open ``phase``'s. Repeat frames for the
    SAME phase (step/note updates) keep the one span running."""
    global _PHASE_OPEN
    name, started = _PHASE_OPEN
    if name == phase:
        return
    if name:
        _PHASE_SPANS[name] = round(
            _PHASE_SPANS.get(name, 0.0) + (time.monotonic() - started), 3)
    _PHASE_OPEN = (phase, time.monotonic())


def _close_phases() -> Dict[str, float]:
    """Close the last open phase and return the measured table."""
    _rotate_phase("")
    return dict(_PHASE_SPANS)


def frame(
    phase: str = "", step: int = 0, total: int = 0, note: str = "",
) -> None:
    """Emit one progress frame on stdout.

    Reporting only. The parent's liveness verdict comes from MEASURED
    evidence (this process tree's CPU, the capture dir's bytes), never from a
    frame — a wedged child can still print.
    """
    if phase:
        _rotate_phase(phase)
    sys.stdout.write(frame_line(phase=phase, step=step, total=total, note=note))
    sys.stdout.flush()


def _write_report(path: Path, report: MintReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(msgspec.json.encode(report))
    os.replace(tmp, path)


def cap_vram(device: int, cap_bytes: int) -> str:
    """Bound this process's device allocations to its reservation.

    Returns a human note, ALWAYS non-empty, so the caller frames it. The
    fraction is computed from the card's REAL total, so the bound is the
    parent's byte reservation and not a fraction anybody guessed.

    pgw#973 (§4.24 item 4): every path that does NOT cap now says so. A mint
    child sharing a card with the serving process runs uncapped whenever
    `MintRequest.vram_cap_bytes` is 0 — which is the legitimate value for an
    unprobeable card (`mint_budget.co_residency` reports `probed=False`) and
    equally the value a budget that computed to nothing produces. Both used to
    return "" here, the caller framed nothing, and the phase table recorded a
    capped mint and an uncapped mint identically. "Uncapped" is a state to
    STATE, not an absence to infer.
    """
    if cap_bytes <= 0:
        note = (
            "vram cap NOT applied: the request carries no cap "
            "(vram_cap_bytes=0) — this child may allocate the whole card")
        logger.warning("mint-child: %s", note)
        return note
    try:
        import torch

        if not torch.cuda.is_available():
            return "vram cap NOT applied: no CUDA in this process"
        # pgw#877 #6. `mint_process.child_env` pins CUDA_VISIBLE_DEVICES only
        # when `request.device >= 0`, so "the pin already chose for us" is
        # true for a named device and FALSE for -1 — and this used to cap
        # ordinal 0 either way, written `0 if device < 0 else 0`: a ternary
        # with one arm, which read like a decision and was not one.
        #
        # A cap applied to the wrong card is worse than no cap: it neither
        # bounds the child nor protects the tenant, and it reports a note
        # saying it did both. So an ambiguous request refuses to cap and SAYS
        # so, rather than capping a card nobody named.
        if device < 0 and torch.cuda.device_count() > 1:
            note = (
                f"vram cap NOT applied: the request named no device and "
                f"{torch.cuda.device_count()} cards are visible, so there is "
                f"no ordinal this process can honestly cap — capping cuda:0 "
                f"would bound a card the pipeline may not be on")
            logger.warning("mint-child: %s", note)
            return note
        dev = torch.cuda.current_device()
        torch.cuda.set_device(dev)
        _free, total = torch.cuda.mem_get_info(dev)
        if total <= 0:
            return (f"vram cap NOT applied: cuda:{dev} reports no total "
                    f"memory, so there is no fraction to compute")
        fraction = min(1.0, max(0.01, cap_bytes / float(total)))
        torch.cuda.set_per_process_memory_fraction(fraction, dev)
        return (
            f"vram cap {cap_bytes / (1 << 30):.2f}GiB "
            f"({fraction:.3f} of {total / (1 << 30):.2f}GiB)")
    except Exception as exc:  # noqa: BLE001 — an uncappable child still mints
        logger.warning("mint-child: could not cap VRAM (%s)", exc)
        return f"vram cap NOT applied: {type(exc).__name__}: {exc}"


def select_specs(
    specs: Sequence[Any], function: str,
) -> Tuple[Any, List[Any]]:
    """The requested function's spec plus every sibling sharing its instance.

    Siblings matter: the warm plan is CLASS-scoped (pgw#654 — one cell per
    class, so turbo and base seed the same capture). Minting from one
    function's view is how a sibling lane ends up absent from the cell and
    every adopting boot fails its per-object proof (gw#612).
    """
    chosen = next((s for s in specs if s.name == function), None)
    if chosen is None:
        raise MintChildRefused(
            f"function {function!r} is not in this image "
            f"(have: {sorted(s.name for s in specs)})")
    if chosen.cls is None:
        raise MintChildRefused(
            f"function {function!r} is function-shaped — it has no instance "
            "to compile")
    siblings = [
        s for s in specs
        if s.cls is chosen.cls and s.instance_key == chosen.instance_key
    ]
    if chosen not in siblings:
        siblings.append(chosen)
    return chosen, siblings


def mint_identity(request: MintRequest) -> str:
    """One sentence naming WHICH mint a message belongs to (pgw#969).

    A mint child holds no orchestrator session, so its only channel is the
    text of whatever it raises. ``ValueError: slot 'pipeline': no resolved
    model ref`` named a symptom and no mint: family, cell key, lane and
    function all had to be reconstructed from a truncated activity row plus a
    DB read before anyone could say which of a pod's cells had died.
    """
    return (
        f"mint family={request.family!r} arm_key={request.arm_token!r} "
        f"lane={request.execution_lane or '(unset)'!r} "
        f"fn={request.function!r}")


def bind_slots(specs: Sequence[Any], resolved: Mapping[str, MintSlot]) -> None:
    """Install the PARENT's resolved slot bindings onto rediscovered specs.

    The child re-runs discovery in a fresh interpreter, so ``spec.models``
    comes back holding only what ``@endpoint`` declared. For a hub-catalog
    slot — ``Slot(cls, selected_by="model")`` with no ``default_checkpoint=``,
    which is sdxl's shape and every multi-checkpoint family's — the decorator
    declares no ref at all, and the resolution chain then has nothing to
    resolve: ``ctx.slots["pipeline"]`` raises before a graph is traced.

    Applied to the WHOLE sibling set, never one spec: ``instance_key`` is a
    live property over ``spec.models``, so binding the chosen function alone
    would move its key out from under its siblings and silently narrow the
    class-scoped warm plan (pgw#654) to one lane.
    """
    for spec in specs:
        for slot, res in resolved.items():
            if slot in spec.slots or slot in spec.models:
                spec.models[slot] = res.ref


def assert_slots_resolvable(
    specs: Sequence[Any],
    slots: Mapping[str, MintSlot],
    *,
    what: str = "",
) -> None:
    """pgw#969: refuse a request whose slots cannot resolve — before the load.

    ``slots`` + ``what`` rather than the whole ``MintRequest`` (pgw#1089): the
    boot-trace child resolves the same slots for the same reason and must get
    the same refusal, and a second copy of this check would be a second answer
    to "can this process trace the checkpoint the parent serves".

    Two distinct failures, both named rather than deferred to a handler's
    first dereference nine seconds later:

    * the parent sent no binding for a declared, non-optional slot (the wire
      gap this issue exists for), and
    * a slot the parent DID bind still fails the resolution chain here, which
      is real divergence between the two processes and never a normal path.

    An OPTIONAL slot the parent did not bind is deliberately silent: the
    parent's own warm context resolves it exactly this way (the deploy chose
    not to serve that lane), and refusing here would refuse a mint the parent
    can serve.
    """
    from . import warmup as warmup_mod

    bound = set(slots)
    problems: List[str] = []
    for spec in specs:
        missing = sorted(
            name for name, slot in spec.slots.items()
            if name not in bound
            and name not in spec.models
            and not getattr(slot, "optional", False)
        )
        for name in missing:
            problems.append(
                f"slot {name!r} of {spec.name!r}: the parent sent no resolved "
                f"binding for it and the endpoint declares no "
                f"default_checkpoint, so this child has no checkpoint to "
                f"trace (MintRequest.slots carries "
                f"{sorted(bound) or '(nothing)'})")
        errors = warmup_mod.resolved_slots_kwargs(spec, None)["slot_errors"]
        for name, why in sorted(errors.items()):
            resolved = slots.get(name)
            if resolved is not None:
                problems.append(
                    f"slot {name!r} of {spec.name!r}: the parent's binding "
                    f"{resolved.ref.path!r} "
                    f"crossed but does not resolve in this process — {why}")
    if not problems:
        return
    raise MintChildRefused(
        f"{what or 'slot resolution'}: cannot build the warmup request — "
        + "; ".join(problems)
        + ". A mint cannot trace a graph for a pipeline it was never told "
          "to load.")


def assert_composable(resolved: Mapping[str, MintSlot]) -> None:
    """pgw#816: refuse a request that describes a tree this child cannot load.

    A materialized snapshot path is not self-describing except in one
    direction: ``snapshot_dir_key`` stamps ``__x`` on a tree fetched with an
    overridden component EXCLUDED (th#1330 B2). Handed such a path and no
    override for it, diffusers walks into the absent subfolder and reports
    ``no file named config.json found in directory <the tree's ROOT>`` — which
    names neither the component nor the cause, and cost the first delegated
    mint in production two attempts to say nothing.

    So the wiring gap is caught HERE, before a single weight is read, as a
    named REFUSAL: deterministic, terminal, and it points at the parent that
    built the request rather than at the loader that tripped over it.
    """
    from .models.cozy_snapshot import dir_key_excludes_components

    bad = sorted(
        slot for slot, res in resolved.items()
        if dir_key_excludes_components(res.path) and not res.component_paths
    )
    if not bad:
        return
    raise MintChildRefused(
        f"slot(s) {bad} were materialized as override-narrowed trees "
        f"(the overridden component's files were excluded from the fetch) "
        f"but this request carries no component override for them, so the "
        f"composition cannot be rebuilt: "
        + "; ".join(f"{slot}={resolved[slot].path}" for slot in bad))


def pick_compile_target(loaded: Dict[str, Any], cfg: Any) -> Tuple[str, Any]:
    """The loaded slot that actually carries the declared compile target(s).

    Public since pgw#1089: the boot-trace child must pick the SAME slot this
    child picks, and two pickers would be two compile targets.
    """
    from . import compile_cache as cc

    for slot, obj in loaded.items():
        try:
            if cc.has_compile_target(obj, cfg):
                return slot, obj
        except Exception:
            continue
    raise MintChildRefused(
        f"no compile target resolved on any loaded slot "
        f"({sorted(loaded) or '(none)'}) for family {cfg.family!r} "
        f"targets={list(cfg.targets)}")


def _warm_jobs(specs: Sequence[Any]) -> List[Any]:
    from . import warmup as warmup_mod
    from .api.decorators import ATTR as DECL_ATTR

    decl = getattr(specs[0].cls, DECL_ATTR, None)
    if decl is None:
        raise MintChildRefused(
            "the endpoint class carries no @endpoint declaration, so no warm "
            "plan can be derived")
    jobs, _skips = warmup_mod.plan(
        specs, decl_warmup=decl.warmup, has_warmup_method=False)
    jobs, _mode = warmup_mod.select_runs(jobs, tracing=True)
    if not jobs:
        raise MintChildRefused(
            "the derived warm plan is empty — there is nothing to compile")
    return jobs


def _run_warm_job(
    instance: Any, job: Any, config: Dict[str, Any], execution_lane: str,
    origin: str = "",
) -> None:
    """One warm forward through the endpoint's OWN handler.

    Mirrors the executor's ``_invoke_warmup`` (bound handler, ctx+payload
    kwargs, stream consumed) — deliberately the same call shape, because the
    graphs this traces are the graphs the parent will later have to hit.
    """
    import asyncio

    from . import warmup

    spec = job.spec
    with tempfile.TemporaryDirectory(prefix="gw-mintchild-") as tmp:
        payload = job.build(tmp)
        if payload is None:
            return
        # pgw#828: the SAME construction the executor's warm path uses. This
        # was three hand-rolled contexts, and the child's had no slots at
        # all — `ctx.slots["pipeline"]` raised `KeyError: 'pipeline'` on a
        # real L4 after a 16.45 s load, so the dynamo mint route published
        # nothing.
        ctx = warmup.warm_context(
            spec, request_id=f"mint-child-{spec.name}",
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


def _drive_warm_plan(
    instance: Any, jobs: Sequence[Any], request: MintRequest, *,
    proof_only: bool = False,
) -> warm_spans.WarmLedger:
    """Run the endpoint's OWN warm plan, framed as ``warmup_forward``.

    Returns the plan's own cost ledger (pgw#989). On the dynamo recipe these
    forwards ARE the compile, so ``warmup_forward`` is 97.6 % of the mint under
    a single name — measured beside an ``inductor_compile`` row reading 0.0 s.
    The ledger splits it into compile and forward per job. It is measured on
    the AOT recipe's proof forward too: that job costs real seconds and an
    unmeasured cost is how this one hid.

    ``proof_only`` runs ONE job (pgw#984, the AOT recipe); otherwise the whole
    plan runs (the dynamo recipe, where these forwards ARE the compile).

    Failure classification is the same on both recipes (pgw#985). A resource
    shortfall re-raises untouched so ``main`` can exit ``EXIT_RESOURCE`` and
    the parent can re-budget; everything else is a REFUSAL. The endpoint's
    handler, its declared warm plan and the parent's slot bindings are all
    fixed for the life of this request file, so a forward that raised once
    raises identically on attempt two — the hub already books this event as
    ``(phase=warmup_forward, deterministic)`` while the worker was still
    calling it ``crashed`` and buying the second pod.
    """
    ledger = warm_spans.WarmLedger()
    total = 1 if proof_only else len(jobs)
    frame(phase="warmup_forward", step=0, total=total)
    for index, job in enumerate(jobs[:total], start=1):
        frame(phase="warmup_forward", step=index, total=total,
              note=job.spec.name)
        try:
            with ledger.job(job.spec.name):
                _run_warm_job(
                    instance, job,
                    dict(request.configs.get(job.spec.name) or {}),
                    request.execution_lane, origin=mint_identity(request))
        except BaseException as exc:
            if _is_resource_error(exc) or not isinstance(exc, Exception):
                raise
            raise MintChildRefused(
                f"{mint_identity(request)}: the endpoint's own warm plan does "
                f"not run — warm job {job.spec.name!r} raised "
                f"{type(exc).__name__}: {exc}. A cell must not seal for a "
                f"handler that cannot serve.") from exc
    return ledger


def _release() -> None:
    """Reclaim a probe pipeline's device memory. The caller has already
    dropped its references; this collects the cycles and returns the cached
    blocks so the next candidate loads onto an empty card."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — best effort
        logger.debug("mint-child: empty_cache failed", exc_info=True)


def _measure_execution_lane(execution_lane: str, load: Any) -> Any:
    """Load, build and time ONE candidate lane.

    Its own frame on purpose: the probe pipeline and its compiled step must
    be unreachable the moment this returns, or the next candidate loads onto
    a card the previous one is still resident on.
    """
    from . import aot_mint, kernel_path

    kernel_path.pin(execution_lane, "mint-time A/B probe (pgw#947)")
    try:
        _instance, pipe, spec = load(execution_lane)
        return kernel_path.measure(execution_lane, aot_mint.bench_step(pipe, spec))
    except Exception as exc:  # noqa: BLE001 — a candidate that cannot be
        # built or measured drops out of the ranking; it never fails the
        # mint, and the reason is recorded with the verdict.
        logger.warning("mint-child: kernel-lane %s not buildable — %s",
                       execution_lane, exc)
        return kernel_path.Measurement(
            execution_lane=execution_lane, unavailable=f"build: {type(exc).__name__}: {exc}")


def execution_lane_verdict_for(
    load: Any, *, meta_mint: bool = False,
) -> Tuple[Any, Any, Any, Any]:
    """MEASURE which serving-kernel lane wins on THIS card (pgw#947).

    Returns ``(verdict, endpoint instance, pipeline, spec)`` — the winning
    lane's loaded pipeline and its export spec, ready to mint, so the artifact
    and the verdict inside it can never describe different code. The INSTANCE
    rides along because the AOT recipe now proves the endpoint's own warm plan
    runs before it exports (pgw#984), and the handler is a method on it.

    A "lane" is the pgw#863 COMBINATION of both axes (``fused+packed``,
    ``baseline+packed``, ...), so the ranking prices the modulation lane's
    residency win and the linear lane's throughput win against each other
    under one rule instead of a hand tuple per axis.

    The A/B has to happen HERE and not inside ``aot_mint``: the lane is
    chosen when the checkpoint's linears are swapped, which is model LOAD, so
    comparing lanes means loading once per candidate. ``load(lane)`` does one
    full endpoint load with that lane pinned and hands back
    ``(pipeline, export_spec)``.

    Replaces the hand-maintained SM tuples this used to be. Whether each
    kernel EXISTS on a card is still a capability question
    (``kernel_lane.candidate_axes``), and an axis with one buildable value
    contributes no candidates — a non-Blackwell card therefore has ONE
    combination and pays no benchmark at all. Whether an armed value is
    FASTER on a qualifying card is now measured there instead of edited into
    a tuple after a $12 campaign.

    Never fails a mint: any gap leaves the default lane pinned, records no
    verdict, and says why — a cell with no verdict is the documented
    conservative-default case on the serving side.
    """
    from . import kernel_path

    candidates = kernel_path.candidates_here()
    if meta_mint and len(candidates) >= 2:
        # pgw#1080, coordinator ruling 2026-08-10: option (a). The A/B is a
        # WHOLE-MODEL benchmark (`bench_step` times a real pipeline step), so
        # running it would put weight-scale values back in the one process
        # this slice exists to keep empty — to buy a verdict with no consumer
        # below Blackwell, where `fused_candidate_gap` leaves one candidate
        # and no A/B runs at all. Typed absence, with its reason.
        verdict = kernel_path.unmeasured(
            candidates[0],
            "meta-mint: this child holds no weights (pgw#1080) and the lane "
            "A/B is a whole-model benchmark; the serving side treats a cell "
            "with no verdict as the documented conservative default")
        kernel_path.pin(verdict.winner, f"meta-mint: {verdict.detail}")
        frame(phase="load",
              note=f"kernel lane {verdict.winner} (unmeasured, meta-mint)")
        instance, pipe, spec = load(verdict.winner)
        return verdict, instance, pipe, spec

    if len(candidates) < 2:
        _axes, gaps = kernel_path.candidate_axes()
        detail = "; ".join(
            f"{axis}: {gap}" for axis, gap in sorted(gaps.items()))
        verdict = kernel_path.sole(
            candidates[0] if candidates else kernel_path.DEFAULT_EXECUTION_LANE,
            f"only one lane combination is buildable on this card — "
            f"{detail or 'no rival'}")
        kernel_path.pin(verdict.winner, f"sole candidate: {verdict.detail}")
        frame(phase="load", note=f"kernel lane {verdict.winner} (sole)")
        instance, pipe, spec = load(verdict.winner)
        return verdict, instance, pipe, spec

    measurements = []
    for index, execution_lane in enumerate(candidates, start=1):
        frame(phase="load", step=index, total=len(candidates),
              note=f"kernel-lane probe: {execution_lane}")
        measurements.append(_measure_execution_lane(execution_lane, load))
        # Each candidate is loaded onto an EMPTY card and torn down before the
        # next one: two resident pipelines would make the probe itself the
        # thing that OOMs a mint pod sized for one, and the peak this measures
        # has to be one lane's peak, not two lanes' sum. The probe pipeline
        # dies with `_measure_execution_lane`'s frame; this reclaims it.
        _release()

    total, name, sm = kernel_path.device_facts()
    verdict = kernel_path.select(
        measurements, device_total_bytes=total, device_name=name, sm=sm)
    frame(phase="load",
          note=f"kernel lane {verdict.winner} ({verdict.binding})")
    kernel_path.pin(verdict.winner, f"mint verdict: {verdict.detail}")
    # The winner is loaded FRESH rather than kept from its probe pass: the
    # probe ran `torch.compile` over the denoiser, and the graph that gets
    # exported must come from a pipeline nothing has warmed or specialized.
    instance, pipe, spec = load(verdict.winner)
    return verdict, instance, pipe, spec


def _mint_aot(
    request: MintRequest, pipe: Any, cfg: Any, target: Path, *,
    started: float, sha256_file: Any,
    execution_lane_verdict: Any = None,
    spec: Any = None,
    footprint: Optional[Dict[str, Any]] = None,
) -> MintReport:
    """pgw#805: the AOT recipe — torch.export + AOTInductor over the family's
    whole declared graph-class set, packed as ONE multi-graph cell (pgw#758).

    This is the wire that never existed. ``aot_mint.mint`` has been complete
    and operator-driven since pgw#723/#758, and discovery filtered for its
    artifact kind since pgw#722 — but no serving-pod code
    path imported ``aot_mint``, so a discovery MISS could only ever fall
    through to the dynamo recipe, whose cell that filter rejects. Every pod
    missed, "re-minted" the wrong kind, and the next pod missed identically.

    Runs against the pipeline the child ALREADY loaded through the endpoint's
    own ``setup()``, so the exported graphs are the serving graphs.
    """
    from . import aot_mint, aot_resume, fleet_cells

    # pgw#848 item 5: install the cross-attempt resume bank before anything is
    # exported. Process-global rather than a parameter threaded through
    # `aot_mint.mint` -> `_mint_cell` -> `_compile_entries_parallel`: the bank
    # is opened by the entry pool, three call frames down, and the intervening
    # signatures describe WHAT to compile rather than where a previous attempt
    # left its work. Empty request field = no bank, and the mint runs exactly
    # as it did before.
    aot_resume.set_root(request.resume)

    frame(phase="trace_graph", note=f"export declaration for {cfg.family!r}")
    if spec is None:
        spec = fleet_cells.aot_export_spec(pipe, cfg)
    out_dir = target.parent / "aot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pgw#824: `aot_mint.mint` used to be ONE opaque call spanning the family's
    # whole declared class set (sdxl: 18), so this function framed `trace_graph`
    # once and then said nothing at all until `seal_publish` below. A real
    # export measured ~5 minutes of complete wire silence; the parent's only
    # evidence that the child was working was that its CPU was warm, which
    # proves alive, not progressing. Every entry now rides the frame protocol
    # that already exists -- no new wire, and the parent's `_on_frame` lands it
    # on the same self_mint_compile activity the hub already reads.
    def _progress(phase: str, step: int, total: int, note: str) -> None:
        frame(phase=phase, step=step, total=total, note=note)

    try:
        result = aot_mint.mint(
            pipe, spec, out_dir, on_progress=_progress,
            # pgw#848: banked by the parent from a previous mint on this pod.
            # 0 on a pod that has never minted this (family, lane).
            entry_peak_rss_bytes=int(
                getattr(request, "entry_peak_rss_bytes", 0) or 0),
            # pgw#877: and the DEVICE half. Read off the request rather than a
            # module global, because a module global here is always empty —
            # the parent is the only process that banks.
            entry_device_peak_bytes=int(
                getattr(request, "entry_device_peak_bytes", 0) or 0),
            # pgw#848: rewritten on every beat, so a mint this process is
            # KILLED in still leaves its measurements on disk for the parent.
            phase_snapshot=(
                Path(request.phases_snapshot)
                if request.phases_snapshot else None),
            # pgw#947: the MEASURED serving-kernel lane for this card. The
            # discrete verdict lands in the packed envelope (serving reads it
            # instead of an SM tuple); the numbers ride the result metadata.
            execution_lane_verdict=execution_lane_verdict,
            # pgw#1053: this process exits when the mint ends — its pipeline
            # serves nobody after the last export, so surrender it and let
            # the compile pool re-derive K against the freed card.
            release_residents=True)
    except aot_mint.MintRefused as exc:
        # A named export refusal is a REFUSAL, not a crash: the parent must
        # not retry it, and the sentence is the whole diagnostic on a pod
        # that exposes no logs (pgw#760). pgw#825: the seconds it spent before
        # refusing ride WITH the sentence — a refusal after four paid compiles
        # and a refusal in the first second are not the same event.
        raise MintChildRefused(
            f"aot mint refused: {exc}",
            mint_phases=getattr(exc, "mint_phases", None)) from exc

    frame(phase="seal_publish", note=f"packed {result.artifact.name}")
    artifact = Path(result.artifact)
    if artifact != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(artifact, target)
    frame(phase="finalize", note=f"cell {result.cell_key}")

    # pgw#1053: the release resets the allocator's high-water so the pool can
    # regrant K — the TRUE peak was banked into the timings first, and the
    # report must carry it or `record_child_peak` learns the post-release
    # residue instead of what the mint really held.
    peak = max(
        _peak_vram(),
        int(result.timings.get("peak_vram_before_release_bytes", 0) or 0))
    # pgw#1080: the EXPORT half of the footprint, measured from the peak reset
    # that follows the warm proof. This is the ceiling the pgw#992 census must
    # size the next export child against — the whole point of the slice is
    # that it is autotune-scratch scale and not weight scale.
    marks = dict(footprint or {})
    marks.setdefault("export_peak_bytes", peak)
    return MintReport(
        status="minted",
        artifact=str(target),
        digest=sha256_file(target),
        cell_key=str(result.cell_key),
        detail=(
            f"exported {len((result.metadata.get('entries') or {}))} graph "
            f"class(es) for family {cfg.family!r} as one aot-inductor cell"),
        phase="finalize",
        peak_vram_bytes=max(
            peak, int(marks.get("warm_proof_peak_bytes", 0) or 0)),
        elapsed_s=time.monotonic() - started,
        phases=_close_phases(),
        mint_phases=dict(result.metadata.get("mint_phases") or {}),
        structure_only_components=tuple(
            marks.get("structure_only_components") or ()),
        structure_refusal=str(marks.get("structure_refusal") or ""),
        virtual_param_bytes=int(marks.get("virtual_param_bytes", 0) or 0),
        structure_real_bytes=int(marks.get("structure_real_bytes", 0) or 0),
        warm_proof_peak_bytes=int(marks.get("warm_proof_peak_bytes", 0) or 0),
        warm_proof_values=str(marks.get("warm_proof_values") or ""),
        export_peak_bytes=int(marks.get("export_peak_bytes", 0) or 0),
    )


def mint(request: MintRequest) -> MintReport:
    """Build the cell. Raises ``MintChildRefused`` for a named refusal."""
    # The two views ``cli.run.run_setup`` takes, both DERIVED from the one
    # resolution (pgw#974) rather than carried beside it.
    paths = {slot: res.path for slot, res in request.slots.items()}
    overrides = {
        slot: dict(res.component_paths)
        for slot, res in request.slots.items() if res.component_paths
    }
    # pgw#816: the request's SHAPE is checked before anything heavy is
    # imported or a single weight is read — a composition this child cannot
    # rebuild is a wiring refusal, not a load crash eight seconds in.
    assert_composable(request.slots)

    from . import compile_cache as cc
    from . import env_seal
    from .cli.run import run_setup
    from .models.chunk_cas import sha256_file
    from .registry import collect_endpoints

    started = time.monotonic()
    target = Path(request.target)

    frame(phase="load", note="env seal")
    env_seal.establish()
    # Always framed: `cap_vram` states the uncapped cases too (pgw#973).
    frame(phase="load", note=cap_vram(request.device, request.vram_cap_bytes))

    if not cc.toolchain_present():
        raise MintChildRefused(
            "no C toolchain (cc/gcc/clang) — inductor cannot link a kernel")
    if not cc.cxx_toolchain_present():
        # pgw#823: AOTInductor's stricter requirement, asserted BEFORE the
        # weights are read. The guard above passes on a C compiler; AOTI needs
        # a C++ one, and without this the miss surfaces 336 s later as an
        # InductorError from inside the linker.
        raise MintChildRefused(
            "no C++ compiler (g++/clang++) — AOTInductor links a shared "
            "object and cannot build one")

    frame(phase="load", note=f"discover {list(request.modules)}")
    specs = collect_endpoints(list(request.modules))
    spec, siblings = select_specs(specs, request.function)
    # pgw#969: the parent's resolution, installed on the rediscovered specs
    # BEFORE anything is loaded — and the refusal, if it cannot be.
    bind_slots(siblings, request.slots)
    assert_slots_resolvable(
        siblings, request.slots, what=mint_identity(request))
    # pgw#1034: the wire struct IS the cfg. It used to be re-inflated into a
    # `registry.CompileCell` to keep `contract_facts()` byte-identical for a
    # key the child computed — the child has computed no key since pgw#758, so
    # the rebuild only manufactured a CompileCell whose contract axes were
    # whatever the wire happened to carry. Every consumer here reads the
    # declared facts by name (family, targets, shapes, text_lens, guidance,
    # lora_bucket) and the spec carries exactly those.
    cfg = request.cfg

    frame(phase="load", note=(
        f"setup {spec.cls.__name__}"
        + (f" (+{sum(len(c) for c in overrides.values())} component "
           f"override(s))" if overrides else "")))

    # pgw#1080: the compile targets are built from CODE + CONFIG, so the
    # process that exports and compiles never holds a checkpoint value. A
    # component that cannot be built that way says so, typed, and that slot
    # loads its weights the old way — the property is then reported as NOT
    # held for this mint rather than silently assumed.
    structure_targets = tuple(cfg.targets)
    structure_refusals: List[str] = []

    def _load(_execution_lane: str) -> Tuple[Any, Any, Any]:
        """One full endpoint load on the currently pinned kernel lane: the
        endpoint instance, its compile-target pipeline, and that pipeline's
        export spec."""
        from . import fleet_cells
        from .models.structure_only import StructureOnlyUnsupported

        obj = spec.cls()
        try:
            got = run_setup(
                obj, dict(paths), arm_compile=False,
                return_loaded=True, component_paths=overrides,
                structure_only=structure_targets) or {}
        except StructureOnlyUnsupported as exc:
            structure_refusals.append(str(exc))
            frame(phase="load", note=f"structure-only declined: {exc}")
            obj = spec.cls()
            got = run_setup(
                obj, dict(paths), arm_compile=False,
                return_loaded=True, component_paths=overrides) or {}
        _slot, loaded_pipe = pick_compile_target(got, cfg)
        frame(phase="load", note=f"compile target on slot {_slot!r}")
        if cfg.lora_bucket:
            cc.apply_lora_execution_lane(loaded_pipe, cfg.lora_bucket)
        return obj, loaded_pipe, fleet_cells.aot_export_spec(loaded_pipe, cfg)

    # pgw#947: MEASURE the serving-kernel lane on this card before the cell is
    # exported, so the cell can carry the verdict instead of the fleet
    # re-deriving it from a hand-maintained SM tuple. The probe loads once per
    # candidate; the winner's pipeline is what gets minted.
    verdict, instance, pipe, aot_spec = execution_lane_verdict_for(
        _load, meta_mint=bool(structure_targets))
    from .models import structure_only

    facts = structure_only.facts_of(pipe)
    virtual_modules = structure_only.modules_of(pipe)
    footprint: Dict[str, Any] = {
        "structure_only_components": tuple(name for name, _m in virtual_modules),
        "virtual_param_bytes": sum(f.virtual_param_bytes for f in facts),
        "structure_real_bytes": sum(f.real_buffer_bytes for f in facts),
        "structure_refusal": "; ".join(structure_refusals)[:400],
    }
    if facts:
        frame(phase="load", note=(
            "structure-only "
            + ", ".join(
                f"{f.component}({f.cls_name}): "
                f"{f.virtual_param_bytes / 2 ** 20:.1f} MiB virtual, "
                f"{f.real_buffer_bytes / 2 ** 20:.1f} MiB real buffers"
                for f in facts)))
    # pgw#984: PROVE the endpoint's own forward runs, before a byte is
    # exported. `torch.export` traces the declared graph classes off the
    # modules directly and never enters the handler, so an AOT mint's
    # phase table read `load / trace_graph / seal_publish / finalize` —
    # green, with no `warmup_forward` row, for an endpoint whose handler
    # could not run at all. That is precisely pgw#969's crash class
    # (`ctx.slots["pipeline"]`, 0.0 s into `warmup_forward`), and it was
    # unreachable on this recipe.
    #
    # ONE forward, not the whole plan: the export derives its own class set, so
    # a full eager pass would buy minutes and prove nothing more than the first
    # job does. Eager — nothing is armed here — so it specializes no graph the
    # export then has to trace around.
    #
    # pgw#1080 (coordinator ruling 2026-08-10): on a structure-only mint the
    # proof runs on RANDOM values. It is a does-it-run proof, not a numerics
    # one, and the ratified variability rule already forbids value-dependent
    # program structure — so a handler whose control flow breaks under random
    # weights is violating that rule and surfacing it here is the point.
    # Random values are NOT checkpoint values, but they ARE weight-scale for
    # the length of the proof, so the window is measured and reported
    # separately from the export's, which is the number the pgw#992 census
    # must size the next export child against.
    device_label = _device_label(request)
    if virtual_modules:
        _reset_peak()
        randomized = sum(
            structure_only.materialize_random(module, device=device_label)
            for _name, module in virtual_modules)
        frame(phase="warmup_forward", note=(
            f"values=random ({randomized / 2 ** 20:.1f} MiB) on "
            f"{[name for name, _m in virtual_modules]}"))
        # The BASELINE for the call-time check below. The lane installs real
        # tensors of its own before any forward runs — LoRA branch containers
        # are the standing example, and they are legitimate (a lifted adapter
        # is a graph INPUT, not a baked constant). What the check is looking
        # for is a tensor that APPEARS during the proof, so it compares
        # against what was already there instead of hardcoding a vocabulary.
        strays_before = {
            f"{name}.{path}"
            for name, module in virtual_modules
            for path, _tensor in structure_only.stray_real_tensors(module)
        }
    _drive_warm_plan(instance, _warm_jobs(siblings), request, proof_only=True)
    if virtual_modules:
        footprint["warm_proof_peak_bytes"] = _peak_vram()
        footprint["warm_proof_values"] = "random"
        for _name, module in virtual_modules:
            structure_only.restore_virtual(module, device=device_label)
        # ie#628's call-time class, caught where it CAN be caught on a
        # weightless mint. Inside a fake-mode export every allocation is fake,
        # so the mode-based gate cannot see a lazily-built pinned table — but
        # the warm proof just ran the handler for real, so if one was built it
        # is now cached on a plain attribute, still real, about to be traced.
        strays = [
            (f"{name}.{path}", tensor)
            for name, module in virtual_modules
            for path, tensor in structure_only.stray_real_tensors(module)
            if f"{name}.{path}" not in strays_before
        ]
        if strays:
            raise MintChildRefused(
                "meta-instantiation gate (ie#628, call-time): after the warm "
                "proof released its random values, "
                + ", ".join(
                    f"{path} still holds a REAL {tuple(t.shape)} "
                    f"{t.dtype} tensor on {t.device}" for path, t in strays[:4])
                + ". A weightless mint traces what the module holds, so this "
                "tensor would be exported as an anonymous graph literal and "
                "the cell would carry values no adopter can rebind. It is "
                "built at CALL time instead of at __init__: register derived "
                "tables with `register_buffer` and NO device pin (ie#630's "
                "`rope_buffers` is the worked example); an explicit "
                "`with torch.device(...)` inside model code is itself the "
                "violation to remove.")
        _release()
        frame(phase="warmup_forward", note=(
            f"random values released; warm-proof device peak "
            f"{footprint['warm_proof_peak_bytes'] / 2 ** 20:.1f} MiB"))
    _reset_peak()
    # Deliberately NOT `aot_mint.compose_for_mint` (which builds a pipeline
    # from a model ref for an operator's mint pod): the graphs this cell must
    # serve are the graphs the ENDPOINT's own composed pipeline runs, and
    # composing a second time is how a mint exports something the serving pod
    # cannot adopt.
    return _mint_aot(
        request, pipe, cfg, target, started=started,
        sha256_file=sha256_file, execution_lane_verdict=verdict, spec=aot_spec,
        footprint=footprint)


def _device_label(request: MintRequest) -> str:
    """Where this child's tensors live — the ordinal the parent chose, or the
    card this process defaults to. ``cpu`` on a cardless box, stated rather
    than assumed: a structure built as "cpu" compiles a CPU cell."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    except Exception:  # noqa: BLE001 — torch-less: nothing to place
        return "cpu"
    ordinal = int(getattr(request, "device", -1) or -1)
    return f"cuda:{ordinal}" if ordinal >= 0 else "cuda"


def _reset_peak() -> None:
    """Start a fresh device high-water window (pgw#1080).

    The warm proof's random-value window and the export's window are two
    different measurements, and reporting their max as one number is what let
    weight residency size an export child that no longer needs it.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — a probe never fails a mint
        logger.debug("mint-child: reset_peak_memory_stats failed",
                     exc_info=True)


def _peak_vram() -> int:
    """This process's device high-water.

    pgw#848: read on EVERY terminus, not just the minted one. A mint that
    died against its own cap is precisely the mint whose peak the next
    attempt has to widen against, and until now the crash and refusal
    reports carried no ``peak_vram_bytes`` at all — so the parent banked
    nothing and re-asked identically, forever.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 — a probe never fails a report
        return 0


def _is_resource_error(exc: BaseException) -> bool:
    from .models.memory import is_cuda_oom

    if is_cuda_oom(exc):
        return True
    # pgw#848: a HOST memory shortfall, classified by whoever measured it.
    # Duck-typed rather than imported: `aot_mint.MintResourceExhausted` sets
    # `mint_resource_shortfall = True`, and this module deliberately pulls in
    # as little of the arming brain as it can. Until this existed the AOT
    # pool's only exit was `MintRefused` -> EXIT_REFUSED -> never retried, so
    # an OOM-killed entry child was reported to the hub as a deterministic
    # refusal and the mint never tried the narrower K that would fix it.
    if getattr(exc, "mint_resource_shortfall", False) is True:
        return True
    return isinstance(exc, MemoryError)


def _install_goals() -> None:
    """Publish THIS process's goal set (pgw#868 A4).

    ``worker_goals`` is a per-process publication with exactly one carrier and
    one moment it is set. Both existing callers of :func:`worker_goals.install`
    are in the SERVING parent (``entrypoint``, ``procsplit.parent``) — and the
    mint runs in a child spawned as ``python -m gen_worker.mint_child``, which
    installed nothing. So ``worker_goals.current()`` fell back to
    :data:`~gen_worker.worker_goals.SERVE_ONLY` in the one process that decides
    the compile pool's width (``aot_compile_pool.entry_workers`` is called from
    ``aot_mint._mint_cell``, three frames down from here), and a mint-only pod
    held back a tenant VRAM reserve, a serving CPU headroom and a tenant
    host-RAM reserve for a tenant that cannot reach it: it accepts no dispatch.

    The fallback is the RIGHT default for a library import with no hub — this
    process has a hub, and its declaration is already in the environment the
    parent handed down (``mint_process.child_env`` copies it), read here the
    same way the parent reads it: off typed `Settings`, never off `os.environ`
    (§1.18). A declaration this build cannot interpret still lands, carrying
    ``declaration_understood=False``, exactly as it does in the parent.

    Never fatal: a child that cannot read its settings keeps the serve-only
    fallback and mints at the narrower width, which is what it does today.

    The settings are LOADED and not ``config.install``ed, deliberately. This
    child is a process entry and §1.18 says a process entry installs — but it
    never has, so every ``config.current_or(default)`` reader inside a mint has
    been answering from its default for the life of this module. Publishing
    them here would flip all of those at once, which is a blast radius this
    change has no way to test and no business taking. It is a real gap and it
    is recorded as one; what this function fixes is the goal set, whose only
    reader inside the child is the width policy.
    """
    try:
        goals = worker_goals.from_settings(load_settings())
    except Exception:  # noqa: BLE001 — a narrower pool beats a dead mint
        logger.warning(
            "mint-child: could not read worker goals; keeping the serve-only "
            "fallback (the pool will hold tenant reserves)", exc_info=True)
        return
    worker_goals.install(goals)
    logger.info(
        "mint-child: goals serve=%s mint=%s (declared %r, understood=%s)",
        goals.serve, goals.mint, goals.declared, goals.declaration_understood)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr)
    if len(args) != 1:
        print(
            f"usage: python -m {__spec__.name if __spec__ else 'gen_worker.mint_child'}"
            " <request.json>", file=sys.stderr)
        return EXIT_BAD_REQUEST
    try:
        request = msgspec.json.decode(
            Path(args[0]).read_bytes(), type=MintRequest)
    except Exception as exc:
        print(f"BAD REQUEST {args[0]}: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST

    _install_goals()
    report_path = Path(request.report)
    started = time.monotonic()
    try:
        try:
            report = mint(request)
        except ValidationError as exc:
            # pgw#1075: classified HERE and not deeper, because every path into
            # this process — spec build, composition check, branch arm, export
            # declaration — can raise it and they all mean the same thing.
            raise _declaration_refusal(exc) from exc
    except MintChildRefused as exc:
        # th#1322: a refused mint's phase table is where it spent the time
        # BEFORE refusing — the most useful half of a failed mint. The OPEN
        # phase is read first on purpose: `_close_phases` closes it, and
        # reading after would report every failure as phase "".
        died_in = _PHASE_OPEN[0]
        _write_report(report_path, MintReport(
            status="refused", detail=str(exc)[:2000],
            elapsed_s=time.monotonic() - started, phases=_close_phases(),
            phase=died_in, peak_vram_bytes=_peak_vram(),
            mint_phases=dict(getattr(exc, "mint_phases", None) or {})))
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except BaseException as exc:  # noqa: BLE001 — every death is classified
        resource_shortfall = _is_resource_error(exc)
        died_in = _PHASE_OPEN[0]
        _write_report(report_path, MintReport(
            status="resource" if resource_shortfall else "failed",
            detail=f"{type(exc).__name__}: {exc}"[:2000],
            elapsed_s=time.monotonic() - started, phases=_close_phases(),
            phase=died_in, peak_vram_bytes=_peak_vram(),
            # pgw#825: a CRASH's paid compiles are measurable too — the mint
            # attaches its partial table to whatever it died with.
            mint_phases=dict(getattr(exc, "mint_phases", None) or {})))
        logger.exception("mint-child: mint failed")
        if resource_shortfall:
            return EXIT_RESOURCE
        if not isinstance(exc, Exception):
            raise
        return 1
    _write_report(report_path, report)
    print(f"MINTED {report.artifact} digest={report.digest}", file=sys.stderr)
    return EXIT_MINTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MintChildRefused", "assert_composable", "assert_slots_resolvable",
           "pick_compile_target",
           "bind_slots", "cap_vram", "frame", "main",
           "mint_identity", "mint", "select_specs"]
