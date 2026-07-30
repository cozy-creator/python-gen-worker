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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


class MintChildRefused(RuntimeError):
    """A named, deterministic reason this mint cannot happen here.

    Never retried by the parent: re-running a named refusal buys a second
    billed compile for the same sentence.
    """


def frame(
    phase: str = "", step: int = 0, total: int = 0, note: str = "",
) -> None:
    """Emit one progress frame on stdout.

    Reporting only. The parent's liveness verdict comes from MEASURED
    evidence (this process tree's CPU, the capture dir's bytes), never from a
    frame — a wedged child can still print.
    """
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
        dev = 0 if device < 0 else 0  # CUDA_VISIBLE_DEVICES already pinned us
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

    from .request_context import RequestContext

    spec = job.spec
    with tempfile.TemporaryDirectory(prefix="gw-mintchild-") as tmp:
        payload = job.build(tmp)
        if payload is None:
            return
        ctx: RequestContext[Any] = RequestContext(
            request_id=f"mint-child-{spec.name}",
            local_output_dir=tmp,
            boot_warmup=True,
        )
        if lane:
            ctx._set_lane(lane)
        if config:
            ctx._set_config(dict(config))
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


def mint(request: MintRequest) -> MintReport:
    """Build the cell. Raises ``MintChildRefused`` for a named refusal."""
    from . import compile_cache as cc
    from . import env_seal
    from .cli.run import run_setup
    from .convert.hub import blake3_file
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

    frame(phase="load", note=f"discover {list(request.modules)}")
    specs = collect_endpoints(list(request.modules))
    spec, siblings = select_specs(specs, request.function)
    cfg = compile_cfg(request.cfg)

    frame(phase="load", note=f"setup {spec.cls.__name__}")
    instance = spec.cls()
    loaded = run_setup(
        instance, dict(request.snapshots), arm_compile=False,
        return_loaded=True) or {}
    slot, pipe = _pick_compile_target(loaded, cfg)
    frame(phase="load", note=f"compile target on slot {slot!r}")

    if cfg.lora_bucket:
        cc.apply_lora_lane(pipe, cfg.lora_bucket)

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

    peak = 0
    try:
        import torch

        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
    except Exception:
        peak = 0
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        rss = 0
    from . import cell_key

    try:
        minted_key = cell_key.from_artifact_metadata(meta).digest
    except Exception:
        minted_key = request.cell_key

    return MintReport(
        status="minted",
        artifact=str(target),
        digest=blake3_file(target),
        cell_key=str(minted_key or request.cell_key),
        detail=f"packed {target.name} for family {cfg.family!r}",
        phase="finalize",
        peak_vram_bytes=peak,
        peak_rss_bytes=rss,
        elapsed_s=time.monotonic() - started,
    )


def _is_resource_error(exc: BaseException) -> bool:
    from .models.memory import is_cuda_oom

    if is_cuda_oom(exc):
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
        _write_report(report_path, MintReport(
            status="refused", detail=str(exc)[:2000],
            elapsed_s=time.monotonic() - started))
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except BaseException as exc:  # noqa: BLE001 — every death is classified
        resource_shortfall = _is_resource_error(exc)
        _write_report(report_path, MintReport(
            status="resource" if resource_shortfall else "failed",
            detail=f"{type(exc).__name__}: {exc}"[:2000],
            elapsed_s=time.monotonic() - started))
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


__all__ = ["MintChildRefused", "cap_vram", "compile_cfg", "frame", "main",
           "mint", "select_specs"]
