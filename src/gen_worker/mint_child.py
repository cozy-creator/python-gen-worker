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
3. **Load the endpoint's own pipeline**, through ``cli.run.run_setup``: the
   endpoint's real ``setup()``/``warmup()``, the production ``provision``
   loader, the same already-materialized weights on local disk. No network.
4. **Arm COLD and drive the endpoint's OWN derived warm plan** — never
   ``mint_artifact``'s producer-style ``_compile_and_warm``. That distinction
   is gw#586/gw#587's whole lesson: a synthetic single-stage warm call can
   trace DIFFERENT FX graphs than a conditioned/two-stage endpoint's real
   warmup, and a cell packed from the wrong graphs bricks every adopting boot.
   So the child runs ``warmup.plan`` over the same sibling spec set the parent
   would have, through the endpoint's own handler. Same code, same shapes,
   same seal — a different PROCESS, not a different execution.
5. **Pack** with ``finish_fleet_mint`` and write a typed report.

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

from .mint_process import (
    EXIT_BAD_REQUEST,
    EXIT_MINTED,
    EXIT_REFUSED,
    EXIT_RESOURCE,
    MintReport,
    MintRequest,
    frame_line,
)

logger = logging.getLogger(__name__)

#: pgw#805 recipes (mirrors ``fleet_cells.RECIPE_*``; duplicated as literals
#: rather than imported so the child never pulls the whole arming brain in).
RECIPE_DYNAMO = "dynamo"
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

    Returns a human note (empty when there is nothing to cap). The fraction is
    computed from the card's REAL total, so the bound is the parent's byte
    reservation and not a fraction anybody guessed.
    """
    if cap_bytes <= 0:
        return ""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
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
            return ""
        fraction = min(1.0, max(0.01, cap_bytes / float(total)))
        torch.cuda.set_per_process_memory_fraction(fraction, dev)
        return (
            f"vram cap {cap_bytes / (1 << 30):.2f}GiB "
            f"({fraction:.3f} of {total / (1 << 30):.2f}GiB)")
    except Exception as exc:  # noqa: BLE001 — an uncappable child still mints
        logger.warning("mint-child: could not cap VRAM (%s)", exc)
        return ""


def compile_cfg(spec_cfg: Any) -> Any:
    """Rebuild the parent's ``registry.CompileCell`` from the wire form.

    The PARENT owns this: the cell key digests
    ``CompileCell.contract_facts()``, and the class-scoped guidance/text-len
    unions live on the spec rather than the decorator, so a child that
    re-derived the cfg from ``@endpoint`` alone would compute a different key
    and the parent would then refuse its own artifact on a contract axis.
    """
    from .api.decorators import DynamicDim
    from .registry import CompileCell

    return CompileCell(
        shapes=tuple(tuple(int(v) for v in row) for row in spec_cfg.shapes),
        targets=tuple(str(t) for t in spec_cfg.targets),
        family=str(spec_cfg.family or ""),
        regional=bool(spec_cfg.regional),
        text_len=spec_cfg.text_len,
        dynamic=tuple(
            DynamicDim(dim=d.dim, min=d.min, max=d.max)
            for d in spec_cfg.dynamic),
        lora_bucket=int(spec_cfg.lora_bucket or 0),
        guidance_scales=tuple(float(v) for v in spec_cfg.guidance_scales),
        text_lens=tuple(int(v) for v in spec_cfg.text_lens),
    )


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


def assert_composable(
    snapshots: Dict[str, str], overrides: Dict[str, Dict[str, str]],
) -> None:
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
        slot for slot, path in snapshots.items()
        if dir_key_excludes_components(path) and not overrides.get(slot)
    )
    if not bad:
        return
    raise MintChildRefused(
        f"slot(s) {bad} were materialized as override-narrowed trees "
        f"(the overridden component's files were excluded from the fetch) "
        f"but this request carries no component override for them, so the "
        f"composition cannot be rebuilt: "
        + "; ".join(f"{slot}={snapshots[slot]}" for slot in bad))


def _pick_compile_target(loaded: Dict[str, Any], cfg: Any) -> Tuple[str, Any]:
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


def _run_warm_job(instance: Any, job: Any, config: Dict[str, Any], lane: str) -> None:
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
            local_output_dir=tmp, lane=lane, config=config)
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


def _drain_router(pipe: Any, *, poll_s: float = 0.5) -> None:
    """Wait out any hot-swap router queue before packing.

    ``begin_fleet_mint`` arms cold and GUARDED with no router, so the warm
    forwards above compile INLINE on this thread — which is exactly what a
    dedicated mint process is for. But endpoint code is allowed to install a
    router of its own, and a capture packed while a compile is still queued is
    the partial-capture class gw#612 bricks adopting boots with. Failed
    signatures are the caller's problem (``finish_fleet_mint`` gates on the
    expected graph count).
    """
    from . import hot_swap

    router = hot_swap.router_of(pipe)
    if router is None:
        return
    while True:
        _warm, pending, _failed = router.stats()
        if pending == 0:
            return
        frame(phase="inductor_compile", note=f"{pending} compile(s) queued")
        time.sleep(poll_s)


def _mint_aot(
    request: MintRequest, pipe: Any, cfg: Any, target: Path, *,
    started: float, sha256_file: Any,
) -> MintReport:
    """pgw#805: the AOT recipe — torch.export + AOTInductor over the family's
    whole declared graph-class set, packed as ONE multi-graph cell (pgw#758).

    This is the wire that never existed. ``aot_mint.mint`` has been complete
    and operator-driven since pgw#723/#758, and ``aot_cells.discover`` has
    filtered for its artifact kind since pgw#722 — but no serving-pod code
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
                if request.phases_snapshot else None))
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

    peak = _peak_vram()
    return MintReport(
        status="minted",
        artifact=str(target),
        digest=sha256_file(target),
        cell_key=str(result.cell_key),
        detail=(
            f"exported {len((result.metadata.get('entries') or {}))} graph "
            f"class(es) for family {cfg.family!r} as one aot-inductor cell"),
        phase="finalize",
        peak_vram_bytes=peak,
        elapsed_s=time.monotonic() - started,
        phases=_close_phases(),
        mint_phases=dict(result.metadata.get("mint_phases") or {}),
        recipe=request.recipe or RECIPE_AOT,
    )


def mint(request: MintRequest) -> MintReport:
    """Build the cell. Raises ``MintChildRefused`` for a named refusal."""
    overrides = {
        slot: dict(comps)
        for slot, comps in request.component_paths.items() if comps
    }
    # pgw#816: the request's SHAPE is checked before anything heavy is
    # imported or a single weight is read — a composition this child cannot
    # rebuild is a wiring refusal, not a load crash eight seconds in.
    assert_composable(dict(request.snapshots), overrides)

    from . import compile_cache as cc
    from . import env_seal
    from .cli.run import run_setup
    from .models.chunk_cas import sha256_file
    from .registry import collect_endpoints

    started = time.monotonic()
    target = Path(request.target)
    capture = Path(request.capture)
    capture.mkdir(parents=True, exist_ok=True)

    frame(phase="load", note="env seal")
    env_seal.establish()
    note = cap_vram(request.device, request.vram_cap_bytes)
    if note:
        frame(phase="load", note=note)

    if not cc.toolchain_present():
        raise MintChildRefused(
            "no C toolchain (cc/gcc/clang) — inductor cannot link a kernel")
    if request.recipe == RECIPE_AOT and not cc.cxx_toolchain_present():
        # pgw#823: the AOT recipe's stricter requirement, asserted BEFORE the
        # weights are read. The guard above passes on a C compiler; AOTI needs
        # a C++ one, and without this the miss surfaces 336 s later as an
        # InductorError from inside the linker.
        raise MintChildRefused(
            "no C++ compiler (g++/clang++) — AOTInductor links a shared "
            "object and cannot build one; the dynamo recipe does not need "
            "this, the AOT recipe does")

    frame(phase="load", note=f"discover {list(request.modules)}")
    specs = collect_endpoints(list(request.modules))
    spec, siblings = select_specs(specs, request.function)
    cfg = compile_cfg(request.cfg)

    frame(phase="load", note=(
        f"setup {spec.cls.__name__}"
        + (f" (+{sum(len(c) for c in overrides.values())} component "
           f"override(s))" if overrides else "")))
    instance = spec.cls()
    loaded = run_setup(
        instance, dict(request.snapshots), arm_compile=False,
        return_loaded=True, component_paths=overrides) or {}
    slot, pipe = _pick_compile_target(loaded, cfg)
    frame(phase="load", note=f"compile target on slot {slot!r}")

    if cfg.lora_bucket:
        cc.apply_lora_lane(pipe, cfg.lora_bucket)

    if request.recipe == RECIPE_AOT:
        # pgw#805: the SAME loaded pipeline, a different recipe. Deliberately
        # NOT `aot_mint.compose_for_mint` (which builds a pipeline from a
        # model ref for an operator's mint pod): the graphs this cell must
        # serve are the graphs the ENDPOINT's own composed pipeline runs, and
        # composing a second time is how a mint exports something the serving
        # pod cannot adopt.
        return _mint_aot(
            request, pipe, cfg, target, started=started,
            sha256_file=sha256_file)

    jobs = _warm_jobs(siblings)
    # Arm COLD, pointed at our own capture dir: the warm forwards below are
    # the ONLY compile this cell will ever see (gw#587).
    cc.begin_fleet_mint(pipe, cfg, capture)
    miss_before = cc.cache_miss_count(pipe)

    frame(phase="warmup_forward", step=0, total=len(jobs))
    for index, job in enumerate(jobs, start=1):
        frame(
            phase="warmup_forward", step=index, total=len(jobs),
            note=job.spec.name)
        _run_warm_job(
            instance, job, dict(request.configs.get(job.spec.name) or {}),
            request.lane)
    frame(phase="inductor_compile", note="draining any queued compiles")
    _drain_router(pipe)

    if cc.execution_count(pipe) <= 0:
        raise MintChildRefused(
            "the warm plan ran but no compile object served a compiled "
            "call — there is nothing to pack")

    frame(phase="seal_publish", note="packing")
    meta = cc.finish_fleet_mint(
        pipe, cfg, cfg.family, target, capture,
        expected_graphs=max(0, cc.cache_miss_count(pipe) - miss_before))
    frame(phase="finalize", note=f"packed {target.name}")

    peak = _peak_vram()
    from . import cell_key

    try:
        minted_key = cell_key.from_artifact_metadata(meta).digest
    except Exception:
        minted_key = request.cell_key

    return MintReport(
        status="minted",
        artifact=str(target),
        digest=sha256_file(target),
        cell_key=str(minted_key or request.cell_key),
        detail=f"packed {target.name} for family {cfg.family!r}",
        phase="finalize",
        peak_vram_bytes=peak,
        elapsed_s=time.monotonic() - started,
        phases=_close_phases(),
    )


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

    report_path = Path(request.report)
    started = time.monotonic()
    try:
        report = mint(request)
    except MintChildRefused as exc:
        # th#1322: a refused mint's phase table is where it spent the time
        # BEFORE refusing — the most useful half of a failed mint.
        _write_report(report_path, MintReport(
            status="refused", detail=str(exc)[:2000],
            elapsed_s=time.monotonic() - started, phases=_close_phases(),
            phase=_PHASE_OPEN[0], peak_vram_bytes=_peak_vram(),
            mint_phases=dict(getattr(exc, "mint_phases", None) or {})))
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except BaseException as exc:  # noqa: BLE001 — every death is classified
        resource_shortfall = _is_resource_error(exc)
        _write_report(report_path, MintReport(
            status="resource" if resource_shortfall else "failed",
            detail=f"{type(exc).__name__}: {exc}"[:2000],
            elapsed_s=time.monotonic() - started, phases=_close_phases(),
            phase=_PHASE_OPEN[0], peak_vram_bytes=_peak_vram(),
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


__all__ = ["MintChildRefused", "assert_composable", "cap_vram", "compile_cfg",
           "frame", "main", "mint", "select_specs"]
