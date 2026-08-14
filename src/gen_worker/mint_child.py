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
2. **Install the parent's RESOLVED slot bindings** (pgw#969) onto the specs
   this process rediscovered, before a weight is read. Rediscovery yields
   only what ``@endpoint`` DECLARED, and a hub-catalog slot
   (``Slot(selected_by=...)`` with no ``default_checkpoint=``) declares
   nothing — so without this the endpoint's own handler reaches
   ``ctx.slots["pipeline"]`` unbound and the mint dies 0.0 s into
   ``warmup_forward``. A request that still cannot resolve a declared slot
   REFUSES here, by name, rather than nine seconds later inside the endpoint.
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

   The AOT recipe traces with ``torch.export`` instead, so its cell is not
   derived from these forwards — but it runs ONE of them anyway (pgw#984),
   before exporting. A recipe that never enters the handler cannot tell a
   working endpoint from one whose forward dies on its first request, and a
   green mint that sealed a cell for the latter is the shape pgw#969 cost
   four hours to find on a pod.
5. **Pack** the exported cell and write a typed report.

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
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

from . import compile_posture, handler_proof, warm_spans
from .api.errors import ValidationError
from .api.export_contract import blocker_refusal, export_declaration, open_blockers
from .child_contract import MintSlot, frame_line
from .child_preflight import (
    PreflightRefused,
    assert_slots_resolvable,
    bind_slots,
    pick_compile_target,
    select_specs,
)
from .file_hash import sha256_file
from .mint_process import (
    EXIT_BAD_REQUEST,
    EXIT_MINTED,
    EXIT_REFUSED,
    EXIT_RESOURCE,
    MintReport,
    MintRequest,
)

logger = logging.getLogger(__name__)

#: pgw#1010: the child builds ONE artifact kind. The dynamo recipe it used to
#: also run produced a cell with no consumer, and it is deleted rather than
#: kept behind a request field nobody may set.
RECIPE_AOT = "aot"


def _assert_family_mintable(family: str) -> None:
    """Refuse the mint outright while the family declares an open blocker
    (pgw#1115).

    The pattern is pgw#1080's: split the refusal by TYPE and fail closed
    rather than degrading. A blocked family has exactly one legal outcome —
    it serves eager and mints nothing — so there is no fallback to take, and
    a mint that started anyway would publish a cell for a class set the
    declaration says it cannot yet claim.
    """
    if not family:
        return
    try:
        decl = export_declaration(family)
    except Exception as exc:  # noqa: BLE001 — a refusing declaration is a refusal
        raise PreflightRefused(
            f"family {family!r}'s export declaration refuses to build "
            f"({type(exc).__name__}): {exc}") from exc
    if decl is None:
        return
    blocked = open_blockers(decl)
    if blocked:
        raise PreflightRefused(blocker_refusal(family, blocked))


def _declaration_refusal(exc: ValidationError) -> PreflightRefused:
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
    return PreflightRefused(
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


def mint(request: MintRequest) -> MintReport:
    """Build the cell. Raises ``PreflightRefused`` for a named refusal."""
    # The two views ``cli.run.run_setup`` takes, both DERIVED from the one
    # resolution (pgw#974) rather than carried beside it.
    paths = {slot: res.path for slot, res in request.slots.items()}

    from . import compile_cache as cc
    from . import env_seal
    from .cli.run import run_setup
    from .registry import collect_endpoints

    started = time.monotonic()
    target = Path(request.target)

    frame(phase="load", note="env seal")
    env_seal.establish()

    if not cc.toolchain_present():
        raise PreflightRefused(
            "no C toolchain (cc/gcc/clang) — inductor cannot link a kernel")
    if not cc.cxx_toolchain_present():
        # pgw#823: AOTInductor's stricter requirement, asserted BEFORE the
        # weights are read. The guard above passes on a C compiler; AOTI needs
        # a C++ one, and without this the miss surfaces 336 s later as an
        # InductorError from inside the linker.
        raise PreflightRefused(
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

    # pgw#1115: FAIL CLOSED on a declared mint blocker, here — after discovery
    # has registered the declaration and before one weight is read. The parent
    # declines a blocked family in `fleet_cells.mint_recipe`, so a request that
    # reaches a child came from somewhere else (an operator CLI, a delegated
    # request built against a stale declaration). Serving a blocked family
    # eagerly is the declared outcome; minting it is not available at all, and
    # a refusal that only one of the two paths honours is not a refusal.
    _assert_family_mintable(str(getattr(cfg, "family", "") or ""))

    frame(phase="load", note=f"setup {spec.cls.__name__}")

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
        from .models import structure_only
        from .models.structure_only import (
            StructureCapabilityMissing,
            StructureNotHonored,
            StructureOnlyUnsupported,
        )

        obj = spec.cls()
        try:
            got = run_setup(
                obj, dict(paths), arm_compile=False, return_loaded=True,
                # pgw#1208: NO SERVING PLACEMENT on the weight-free path. The
                # pgw#1124 seam, and the same argument one door over: the
                # placement ladder is for a pipeline that will run a forward,
                # and this one never will — it exports and nothing else. Since
                # pgw#1199 the child runs no warm proof either, so the last
                # reason to place it is gone.
                #
                # This is not a tuning choice, it is the whole of the sdxl
                # and z-image export failures. The ladder installs OFFLOAD
                # HOOKS, and an offload hook puts a `@torch.compiler.disable`d
                # function in the traced path — which `torch.export(strict=True)`
                # refuses to inline, fatally and deterministically:
                #
                #   Unsupported: Skip inlining `torch.compiler.disable()`d
                #   function <function ModuleGroup.onload_>
                #
                # The MECHANISM is family-independent; only the hook class
                # differs (`ModuleGroup.onload_` for sdxl's group offload,
                # `CpuOffload.pre_forward` for z-image's). It is NOT a
                # card-capacity story: z-image refused on a 48 GiB A40 holding
                # a 19 GiB model, so offload here is a CONFIGURATION and not a
                # response to memory pressure. `place_pipeline` has exactly one
                # call site, guarded by `if place and ...`, and no endpoint
                # installs offload itself — so this line removes every offload
                # hook for every family. The hooks are not stripped; they are
                # never installed.
                place=False,
                structure_only=structure_targets) or {}
        except StructureNotHonored as exc:
            # pgw#1080 z-image tail: the target WAS built weight-free and the
            # pipeline discarded it and rebuilt from the checkpoint. Falling
            # back here (below) would export ~weight-scale REAL tensors while
            # the child reports weightless — the silent 40 GiB `retryable` OOM
            # (ie#638). This is buildable-but-not-honored, NOT a stranded
            # family, so it FAILS CLOSED with the authoring cause named rather
            # than degrading to a real-weight export the meta gate never sees.
            raise PreflightRefused(
                f"structure-only was requested for {mint_identity(request)} "
                f"and the target built weight-free, but the composed pipeline "
                f"did not carry it: {exc}") from exc
        except StructureOnlyUnsupported as exc:
            structure_refusals.append(str(exc))
            # pgw#1123: still never fatal — a real-weight mint is a correct,
            # more expensive mint. But a CAPABILITY refusal is not this
            # family declining, it is this image being unable to, and the two
            # were one note. Say which, and say it at ERROR: the same image
            # derives no boot key either, so every pod running it re-mints.
            if isinstance(exc, StructureCapabilityMissing):
                logger.error(
                    "the weight-free mint is unavailable in this image, so "
                    "this mint loads real weights and every boot in it will "
                    "self-mint: %s", exc)
            frame(phase="load", note=(
                f"structure-only {structure_only.refusal_token(exc)}: {exc}"))
            obj = spec.cls()
            # The REAL-WEIGHT fallback keeps its placement, deliberately:
            # structure-only was refused for this family, so this process
            # holds the checkpoint AND runs the pgw#984 warm proof — a real
            # forward, which needs a placed pipeline. Only the path that
            # exports and never executes may skip the ladder.
            got = run_setup(
                obj, dict(paths), arm_compile=False,
                return_loaded=True) or {}
        _slot, loaded_pipe = pick_compile_target(got, cfg)
        frame(phase="load", note=f"compile target on slot {_slot!r}")
        assert_traceable_as_loaded(loaded_pipe, request)
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
    # pgw#984's guarantee, discharged where it costs nothing (pgw#1199).
    #
    # The endpoint's own handler must be PROVEN to run before a cell seals:
    # `torch.export` traces the declared graph classes off the modules directly
    # and never enters the handler, so a mint's phase table could read
    # `load / trace_graph / seal_publish / finalize` — green — for an endpoint
    # whose handler could not run at all (pgw#969: `ctx.slots["pipeline"]`,
    # 0.0 s into `warmup_forward`).
    #
    # It used to be proven HERE, which on a weight-free mint meant materialising
    # REAL random values for every virtual parameter first — one full checkpoint
    # at compute dtype, in a process holding none, concurrently with the
    # parent's resident copy. That is 56.2 GB on wan-2.2 against 15.5 GiB free,
    # and it is what §4.33's "~8 GiB" was actually measuring (`materialize_random`
    # for an sdxl-sized family). §4.33 steps 4-5 put verification on the LIVE
    # pipeline that already holds these weights, so the proof travels as the
    # parent's PROVENANCE and this process allocates nothing for it. The SDK had
    # already made the same call for the kernel-lane A/B — see
    # `execution_lane_verdict_for`'s `meta_mint` branch.
    #
    # REFUSED, never re-proven here: a child that proved it itself would
    # reintroduce the allocation, and a caller that cannot prove its handler
    # runs has no business publishing a cell for it.
    if virtual_modules:
        assert_handler_proven(request)
        footprint["handler_proof"] = str(request.handler_proof)
        frame(phase="warmup_forward", note=(
            f"handler proven by the parent, on resident weights: "
            f"{request.handler_proof} — this child allocates nothing for it"))
    else:
        # The real-weight fallback: structure-only was refused for this family
        # (a quantized artifact lane, a class with no config surface), so the
        # checkpoint is already resident IN THIS PROCESS and the proof costs
        # what it has always cost — nothing extra.
        _drive_warm_plan(
            instance, handler_proof.warm_jobs(siblings), request,
            proof_only=True)
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


def assert_traceable_as_loaded(pipeline: Any, request: MintRequest) -> None:
    """Refuse ONCE, before any export, when this pipeline cannot be traced.

    pgw#1208 (a). The weight-free path never places and so never acquires these
    hooks; this is for the REAL-WEIGHT fallback, which loads a checkpoint into
    this process and runs the serving placement ladder over it. An offload hook
    puts a `@torch.compiler.disable`d function in the traced path — diffusers'
    `ModuleGroup.onload_` (group offload) or accelerate's
    `CpuOffload.pre_forward` (model/sequential offload) — and every one of the
    family's declared graph classes then refuses `Unsupported: Skip …` for the
    same reason.

    **This is NOT a capacity verdict, and must never be read as one.** z-image
    refused on a 48 GiB A40 holding a 19 GiB model, so offload is a pipeline
    CONFIGURATION rather than a response to memory pressure — and §1.35 has
    since ruled the whole card-filter concept out of existence (every model runs
    on every GPU, best effort; feasibility is never asked). What this detects is
    exactly what it says: the pipeline AS LOADED carries hooks that cannot be
    traced.

    Without this, pgw#1208's per-entry skip would do exactly what it is supposed
    to do and dutifully skip all 36: thirty-six typed refusals and an hour of
    wall clock to say once what was knowable before the first export began. The
    per-entry skip is for a class that is individually unexportable. This is the
    whole PIPELINE being untraceable as loaded, which is a different fact and
    gets its own sentence.

    Cost is already covered by the attempt-is-the-budget rule (§4.33); what this
    buys is LEGIBILITY — a pod that cannot mint says why, once, in a countable
    typed refusal, instead of emitting a refusal per class and publishing
    nothing.

    No placement logic and no card arithmetic: it asks the object in front of it
    whether it carries disabled work, so it stays true for any future source of
    such hooks rather than only for the offload rung that produced the first.
    """
    from .models import traceability

    reason = traceability.untraceable_reason(pipeline)
    if not reason:
        return
    raise PreflightRefused(
        f"mint_pipeline_not_traceable: {mint_identity(request)} loaded real "
        f"weights and the placement ladder installed offload hooks — {reason}. "
        f"`torch.export` refuses a `torch.compiler.disable`d function rather "
        f"than tracing around it, so EVERY declared graph class would refuse "
        f"identically. Refused once, before the first export, instead of once "
        f"per class. NOT a statement about this card (§1.35: every model runs "
        f"on every GPU, feasibility is never asked) — it is a statement about "
        f"this PIPELINE AS CONFIGURED. This pod serves the family eager "
        f"exactly as before; only minting is unavailable while it is loaded "
        f"this way.")


def assert_handler_proven(request: MintRequest) -> None:
    """A weight-free mint REFUSES unless the parent proved the handler runs.

    pgw#984's guarantee, and the one thing pgw#1199 had to be careful not to
    drop while moving where it is discharged. The child cannot prove it itself
    on this path: it holds no weights, so proving it means materialising one
    full checkpoint at compute dtype beside the parent's resident copy — 56.2
    GB on wan-2.2 against 15.5 GiB free, which is the mint that died on pod
    `729431an6ugbvq`, and which is what §4.33's "~8 GiB" was measuring for an
    sdxl-sized family.

    So the obligation moves to the caller, whole: run ONE warm forward on the
    resident pipeline and say so. It is free on the fleet path (the executor's
    boot warm plan has already run every declared handler before a mint is
    delegated) and one forward on cozy-local. Refusing here is not a fence
    weakened to make a case pass — it is the same sentence, addressed to the
    process that can afford to say it.

    Public because it IS the contract between the two processes, and a caller
    that wants to know whether its request will be accepted should be able to
    ask the same question the child asks.
    """
    if str(getattr(request, "handler_proof", "") or "").strip():
        return
    raise PreflightRefused(
        f"{mint_identity(request)}: this is a WEIGHT-FREE mint and the parent "
        f"sent no handler proof. pgw#984 requires the endpoint's own handler "
        f"to have run before a cell seals, and since pgw#1199 that proof "
        f"belongs to the process that HOLDS the weights — proving it here "
        f"would mean materialising one full checkpoint at compute dtype in a "
        f"process that holds none. Run one warm forward on the resident "
        f"pipeline and declare it (`MintRequest.handler_proof`).")


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


def _install_posture(request: MintRequest) -> None:
    """Adopt the posture the parent DECLARED, and act on it — §4.30.

    Three things happen here and they all belong at this one point, before a
    single graph is exported:

    **1. Publish it**, so ``aot_compile_pool.entry_workers`` — three frames
    down, in this process — sizes K against it. One carrier, one moment
    (§4.22).

    **2. Drop this process's scheduling priority**, which is the whole of the
    CPU half of politeness. It is done HERE, by the child to itself, and NOT
    with a ``preexec_fn`` on the spawn, for the reason
    ``aot_compile_pool.arm_parent_death_signal`` already writes down: a
    ``preexec_fn`` forces ``fork()`` instead of ``posix_spawn()`` for a process
    that has live gRPC threads with ``pthread_atfork`` handlers, which is a
    large blast radius for a one-line guarantee.

    Doing it here also covers strictly more ground than nicing the entry-pool
    spawns would. Since pgw#1080 every production mint is weight-free, and
    ``aot_mint._mint_cell`` forces ``parallel=False`` for a weight-free mint —
    so the pool spawns NO children at all today and the compile runs in THIS
    process. Nice is inherited across ``fork``/``exec``, so this one call
    covers the serial compile, the entry children when parallelism returns
    (pgw#1111), inductor's own compile workers, and every ``cc1plus`` under
    them.

    **3. Arm ``PR_SET_PDEATHSIG``** on a user's machine. ``mint_process`` spawns
    this child with ``start_new_session=True``, so it holds its OWN session and
    a terminal's Ctrl-C or SIGHUP never reaches it. On a pod that is correct —
    the parent reaps the group deliberately when it abandons a mint. On a
    desktop the "parent" is a CLI that a human closes, and without this a
    closed terminal leaves a full-speed compile tree running with nobody left
    to reap it: the exact "my machine is at a crawl and I don't know why"
    support ticket politeness exists to prevent. Nothing is lost by dying —
    ``MintRequest.resume`` points at ``aot_resume``'s cross-attempt bank, which
    lives outside the per-attempt workdir precisely so finished entries survive
    into the next run.

    Never fatal. A kernel that refuses either call leaves a mint that is rude
    rather than no mint at all, and says so.
    """
    posture = request.posture
    compile_posture.install(posture)
    if not posture.user_machine:
        return
    from .aot_compile_pool import arm_parent_death_signal

    level = posture.nice_level()
    applied = -1
    try:
        applied = os.nice(level)
    except (OSError, AttributeError):
        logger.warning(
            "mint-child: could not lower this mint's scheduling priority "
            "(nice %d) — it will compile at ordinary priority on a machine "
            "someone is using", level, exc_info=True)
    reaped = arm_parent_death_signal()
    logger.info(
        "mint-child: §4.30 user-machine posture — nice %d (now %d), "
        "dies-with-parent=%s", level, applied, reaped)


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

    _install_posture(request)
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
    except PreflightRefused as exc:
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


__all__ = ["frame", "main", "mint_identity", "mint"]
