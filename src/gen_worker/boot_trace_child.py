"""pgw#1089: ONE boot-trace child — ``python -m gen_worker.boot_trace_child``.

Composes the compile target STRUCTURE-ONLY (pgw#1080's builder, reached through
the loader's own ``components=`` seam), exports its assigned graph classes on
fake tensors, and writes each class's keying block. It holds no checkpoint
value, opens no socket, and returns no artifact — only the facts an entry's
``class_hash`` is computed from.

**This is pgw#1080's owed widening**, discharged the only way that keeps ONE
load path: the boot derivation runs the mint child's own load
(``run_setup(..., structure_only=cfg.targets)``) and the mint's own export loop
(``aot_mint.trace_for_key``), rather than a second boot-flavoured pair. Two
loads would be two compositions, and two compositions trace two graphs — which
is exactly the divergence a derived key exists to rule out.

It holds no card either (pgw#1124)
----------------------------------
The child loads with ``place=False``: the worker's serving placement ladder is
for a pipeline that will run a forward, and this one never will. Running it
here moved every REAL non-target component of the slot — qwen-image's 15.5 GiB
Qwen2.5-VL text encoder, its VAE — onto the card the serving parent is already
resident on, and the CUDA OOM that followed was reported as a rollback failure
rather than as the placement nobody needed. The compile target still claims the
compute device, because a virtual structure allocates nothing and the device it
claims is part of the graph's identity. :func:`off_host_tensors` then proves the
property instead of assuming it.

The structure refusal is a REFUSAL here, not a fallback
-------------------------------------------------------
``mint_child`` falls back to real weights when a component cannot be built from
config, because a mint that must produce an artifact is better weightful than
absent. A BOOT trace has the opposite obligation: loading the checkpoint is the
one thing §4.27 step 1 forbids (*"No weights required, ever, for identity"*),
and a pod that quietly downloaded 42 GiB to state its key would satisfy the
letter of the derivation and destroy its purpose. So a stranded family produces
a typed ``structure_unsupported`` refusal, the pod adopts nothing and mints the
ordinary way, and the reason names the class — which is the signal that says
which family's structure-only build to build next.

The weight-free premise is FENCED, not assumed (pgw#1173)
---------------------------------------------------------
Every VRAM conclusion in the compile loop follows from "compilation is
weight-free", so this child proves it about the pipeline it is about to trace
rather than inferring it from the absence of a complaint. Two guards used to
stand here and neither could see the case they were named for: a
``place=False`` load puts real weights on the HOST, where
:func:`off_host_tensors` cannot see them, and
``structure_only.structure_only_components`` is satisfied by ANY one virtual
component, so a family declaring two targets passed with one of them stranded
on real weights. :func:`gen_worker.models.structure_only.assert_weight_free`
is the ALL-of check over the declared targets, device-blind, typed
``StructureNotHonored`` — the type that already means exactly this and that
this path is required to treat as fatal.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import msgspec

from .boot_key import (
    CODE_DIGEST, EXIT_BAD_JOB, EXIT_OK, EXIT_REFUSED, TraceJob, TraceReport,
)

logger = logging.getLogger(__name__)


def _write(report_path: Path, report: TraceReport) -> None:
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(".tmp")
        tmp.write_bytes(msgspec.json.encode(report))
        tmp.replace(report_path)
    except OSError:
        logger.exception("boot-trace: could not write %s", report_path)


def _fail(report_path: Path, reason: str, detail: str) -> int:
    _write(report_path, TraceReport(
        ok=False, reason=reason, detail=str(detail)[:4000],
        code_digest=CODE_DIGEST))
    return EXIT_REFUSED


def _probe_prop_ms(program: Any) -> int:
    """pgw#847, measured where the graphs are: ``FakeTensorProp`` milliseconds
    over this program's own graph.

    A probe. It mutates a FRESH ``program.module()`` (never the program), no
    decision reads it, and any failure records 0 rather than touching a trace.
    ``.module()`` is excluded from the timing because the collapse being priced
    would pay it too, so the comparison stays like-for-like — the discipline
    ``aot_mint._probe_prop_s`` already keeps for the mint path, aimed at the
    boot path where the saving would be spent.
    """
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        args, _kwargs = program.example_inputs
        gm = program.module(check_guards=False)
        t0 = time.monotonic()
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode:
            fake = tuple(
                mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
                for a in args)
            FakeTensorProp(gm, mode=mode).propagate(*fake)
        return int((time.monotonic() - t0) * 1000)
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return 0


def off_host_tensors(pipeline: Any) -> List[tuple[str, str, str]]:
    """REAL tensors of ``pipeline`` that are not on the host, as
    ``(component, tensor, device)``. Empty is the property holding.

    The pgw#1124 invariant, ENFORCED rather than hoped for: a boot trace needs
    shapes and a graph, so nothing it composes may occupy the card its serving
    parent is resident on. The compile target's virtual parameters claim that
    card and allocate nothing, which is the whole of pgw#1080 — so this is
    exactly ``device_mismatches``, whose walk already excludes what is virtual
    by design. A hit means the composition placed real weights: the child
    refuses by name and the pod mints the ordinary way, instead of OOMing the
    card and reporting it as a rollback failure (which is what the fleet
    census caught it doing on 14/14 qwen-image and 9/9 flux2-klein-9b
    children).
    """
    from .models.memory import device_mismatches

    return device_mismatches(pipeline, "cpu")


def run(job: TraceJob) -> int:
    """Trace this child's share and write its report. Never raises."""
    report_path = Path(job.report)
    if job.code_digest and CODE_DIGEST and job.code_digest != CODE_DIGEST:
        return _fail(
            report_path, "code_drift",
            f"parent contract source {job.code_digest!r} != this child's "
            f"{CODE_DIGEST!r}; the classes this child would hash are not the "
            f"classes the parent's code describes")

    from . import aot_mint, fleet_cells
    from .cli.run import run_setup
    from .registry import collect_endpoints
    from .mint_child import (
        MintChildRefused, assert_slots_resolvable, bind_slots,
        pick_compile_target, select_specs,
    )
    from .models import structure_only

    t_setup = time.monotonic()
    try:
        specs = collect_endpoints(list(job.modules))
        chosen, siblings = select_specs(specs, job.function)
        bind_slots(siblings, job.slots)
        assert_slots_resolvable(
            siblings, job.slots,
            what=f"boot key derivation for {job.function!r}")
    except MintChildRefused as exc:
        return _fail(report_path, "slots_unresolvable", str(exc))

    cfg = job.cfg
    paths = {name: slot.path for name, slot in job.slots.items()}
    overrides = {
        name: dict(slot.component_paths or {})
        for name, slot in job.slots.items() if slot.component_paths}

    instance = chosen.cls()
    try:
        loaded = run_setup(
            instance, dict(paths), arm_compile=False, return_loaded=True,
            component_paths=overrides,
            # pgw#1124: no serving placement, ever — see `off_host_tensors`.
            place=False,
            structure_only=tuple(cfg.targets)) or {}
    except structure_only.StructureOnlyUnsupported as exc:
        # NOT a fallback — see the module docstring. Two tokens, because a
        # stranded FAMILY and an image that cannot meta-instantiate AT ALL are
        # opposite findings that were reported identically (pgw#1123).
        token = structure_only.refusal_token(exc)
        if token == structure_only.TOKEN_CAPABILITY_MISSING:
            logger.error("boot key cannot be derived in this image: %s", exc)
        return _fail(report_path, token, str(exc))

    try:
        _slot, pipeline = pick_compile_target(loaded, cfg)
    except MintChildRefused as exc:
        return _fail(report_path, "no_compile_target", str(exc))

    resident = off_host_tensors(pipeline)
    if resident:
        return _fail(
            report_path, "real_weights_resident",
            f"{type(pipeline).__name__} composed {len(resident)} REAL "
            f"tensor(s) off the host (e.g. {resident[:3]!r}) — a boot trace "
            f"holds no card, so this composition would be competing for VRAM "
            f"with the serving parent that spawned it")

    # The weight-free PREMISE, fenced (pgw#1173). Every declared target this
    # pipeline carries must hold zero real parameters — an ALL-of check, and
    # one that does not care which device the weights are on. The two guards
    # that stood here before could both read clean on a composition that traces
    # real weights: `off_host_tensors` sees nothing because `place=False` puts
    # them on the HOST, and `structure_only_components` is satisfied by ANY one
    # virtual component, so a two-target family with one target stranded passed
    # both. This is the seam every downstream VRAM conclusion rests on, so it
    # fails closed and names the component, the class and the bytes.
    try:
        structure_only.assert_weight_free(
            pipeline, cfg.targets, what=f"the boot trace of {job.function!r}")
    except structure_only.StructureNotHonored as exc:
        return _fail(report_path, "structure_not_honored", str(exc))
    virtual = structure_only.structure_only_components(pipeline)
    if not virtual:
        return _fail(
            report_path, "structure_not_honored",
            f"{type(pipeline).__name__} composed no structure-only component "
            f"for target(s) {list(cfg.targets)!r} — this child would be "
            f"tracing real weights, which §4.27 step 1 forbids")

    if cfg.lora_bucket:
        from . import compile_cache as cc

        # The CONTAINER half and the lane stamp, exactly as `mint_child` arms
        # the pipeline it hands the mint. The LIFTED half belongs to the loop
        # that needs it — `aot_mint.trace_for_key` arms it itself (pgw#1132);
        # a second arm site here is how the two halves drifted apart.
        cc.apply_lora_execution_lane(pipeline, int(cfg.lora_bucket))
    export_spec = fleet_cells.aot_export_spec(pipeline, cfg)
    setup_ms = int((time.monotonic() - t_setup) * 1000)

    decl = aot_mint.export_declaration(export_spec.family)
    if decl is None:
        return _fail(
            report_path, "no_declaration",
            f"family {export_spec.family!r} has no registered export "
            f"declaration — a multi-graph cell derives its class set from it")

    blocks: Dict[str, str] = {}
    nodes: Dict[str, int] = {}
    trace_ms: Dict[str, int] = {}
    prop_ms = 0
    export_ms = 0
    declared = 0
    try:
        t0 = time.monotonic()
        # `trace_for_key` emits the pgw#1087 `trace_for_key` span itself,
        # around the export it measures. The clock here is the child's own
        # per-class wall, which is what the parent's report carries — the two
        # measure the same window from either side of the generator.
        for traced in aot_mint.trace_for_key(
                pipeline, export_spec, decl,
                share_index=job.share_index, share_count=job.share_count):
            elapsed = int((time.monotonic() - t0) * 1000)
            declared = int(traced.declared)
            nodes[traced.name] = int(traced.nodes)
            trace_ms[traced.name] = elapsed
            if not export_ms:
                export_ms = elapsed
                prop_ms = _probe_prop_ms(traced.program)
            blocks[traced.name] = json.dumps(
                traced.block, sort_keys=True, separators=(",", ":"))
            # The program is the largest object this child holds and nothing
            # downstream reads it.
            traced.program = None
            t0 = time.monotonic()
    except aot_mint.MintRefused as exc:
        return _fail(report_path, "trace_refused", str(exc))

    if not blocks:
        # A share with no rows is only legitimate when the declaration has
        # fewer classes than the pool has children, and the parent does not
        # spawn those. Reaching here means the enumeration disagreed.
        return _fail(
            report_path, "empty_share",
            f"share {job.share_index}/{job.share_count} produced no classes "
            f"from a declaration this child did enumerate")

    _write(report_path, TraceReport(
        ok=True,
        blocks=blocks,
        nodes=nodes,
        trace_ms=trace_ms,
        setup_ms=setup_ms,
        declared_classes=declared,
        structure_only=tuple(virtual),
        weight_lane=str(export_spec.weight_lane or ""),
        precision=str(export_spec.precision or ""),
        strict=bool(export_spec.strict),
        lora_bucket=int(export_spec.lora_bucket or 0),
        code_digest=CODE_DIGEST,
        prop_probe_ms=prop_ms,
        export_probe_ms=export_ms,
        device_peak_bytes=_device_peak_bytes(),
    ))
    return EXIT_OK


def _device_peak_bytes() -> int:
    """This child's own device high-water, including its CUDA context.

    pgw#1165: the number the parent's concurrency budget is derived from, and
    it has to be the WHOLE process footprint rather than the allocator's view.
    `max_memory_allocated` reports tensors only; the dominant term here is the
    CUDA context plus the cuBLAS/cuDNN kernel images the export loads, which is
    ~1.3 GiB and belongs to the process, not to any tensor. Measured as
    `total - free` on THIS process's device after the work, minus nothing:
    the parent needs to know what one child costs the card, and every byte of
    it costs the card.

    A trace that never touched CUDA reports 0, which the parent reads as
    "unmeasured" rather than "free" — an absent measurement must never widen a
    pool.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 — telemetry never fails a trace
        return 0
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return 0
        free, total = torch.cuda.mem_get_info()
        used = int(total) - int(free)
        return max(0, used)
    except Exception:  # noqa: BLE001
        return 0


def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if len(argv) != 2:
        sys.stderr.write(
            "usage: python -m gen_worker.boot_trace_child <job.json>\n")
        return EXIT_BAD_JOB
    try:
        job = msgspec.json.decode(Path(argv[1]).read_bytes(), type=TraceJob)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        sys.stderr.write(f"boot-trace: unreadable job {argv[1]}: {exc}\n")
        return EXIT_BAD_JOB
    try:
        return run(job)
    except BaseException as exc:  # noqa: BLE001 — every terminus is reported
        logger.exception("boot-trace: child failed")
        return _fail(
            Path(job.report), "child_error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":  # pragma: no cover — process entrypoint
    raise SystemExit(main(sys.argv))
