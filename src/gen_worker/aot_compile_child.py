"""pgw#809/pgw#1215: one SHARE of a mint's graph classes, one process, then exit.

``python -m gen_worker.aot_compile_child <job.json>``

Run by :class:`aot_compile_pool.EntryCompilePool` K-wide while the serving
process goes on serving eager. The boundary is a file for the same reason
pgw#784's is: a mint that fails on share 2 of 4 leaves the job behind and the
work is reproducible by hand, on the same box, without the parent.

**What changed, and it is the whole of th#1834 Phase 3 step 2b.** This child
used to load ONE already-exported ``ExportedProgram`` off disk and compile it.
The load cost a median of **36.04 s** (pgw#1216, P0-E §5c) — ~22 min of a
36-class sdxl mint spent deserializing a program another process in the same
pod had just serialized. So the program is not serialized at all any more:
this child builds the weight-free pipeline itself, exports its own share, and
compiles the program it is still holding.

The worker now stops at the exported program. ``torchcg`` owns
graph identity, AOTInductor policy, packaging, tensorfs storage and exact-key
reuse. The child supplies one ``GraphClassSpec`` and one sealed
``RuntimeCompatibility`` per class; it does not restate that library's
metadata or key derivation.

Three things died with the round trip, and none of them is a loss:

* ``EntryJob.program`` — the staged ``.pt2`` file;
* ``EntryJob.symbol_values`` / ``symbol_labels`` (pgw#998) — the ShapeEnv
  values ``torch.export``'s save/load rebuilds keyed by size EXPRESSIONS
  instead of free symbols. This process traced the graph, so its ShapeEnv is
  the authority, unrepaired;
* ``structure_only.as_meta_for_save`` / ``revirtualize_from_meta`` (pgw#1111)
  — a FAKE tensor has no storage to serialize, so a weight-free program used
  to cross as META and be re-virtualized on the far side. Nothing crosses.

Packing is INSIDE the trace loop, one class at a time (pgw#1183's ordering,
held at the graph class). A child that accumulated its compiled share and
packed it on the way out would lose every finished class to a crash on class
k — up to 36/K classes at a measured 156-509 s each, which is the
all-or-nothing loss th#1825 spent 1 h 37 m on. Packed means ON DISK: a crash
costs the ONE class in flight, and a refusal at class k reports the k-1
artifacts that already exist rather than discarding them.

What it still does NOT do: touch the hub, warm anything, publish anything, or
decide anything about which classes exist. The declaration decides that; this
child takes ``rows[share_index::share_count]`` of it, exactly as
``boot_trace_child`` shards the boot key derivation, and for the same reason —
the adapter fork is decided by the COMPOSED pipeline, so no parent can
enumerate the class names to hand out.
"""

from __future__ import annotations

import time

#: pgw#830: the FIRST statement of the module, before any gen_worker import.
#: The parent's ``compile_s`` starts at its own ``Popen`` return, so the
#: interpreter's boot and this package's import are inside the measured total;
#: without this stamp they can only ever land in a residual. Wall-clock (not
#: monotonic) because the span it closes is measured across two processes.
MODULE_IMPORT_EPOCH = time.time()

import hashlib
import json
import logging
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
from gen_worker._vendor.torchcg import GRAPH_CLASS_BLOCK
from gen_worker._vendor.torchcg.spans import (
    PARTITION_KEYS,
    PARTITIONS,
    SPANS_V,
    SpanLedger,
    check,
    phase_delta,
    phase_snapshot,
)

from .aot_compile_pool import (
    CLASS_ROW_DIRNAME,
    CODE_DIGEST,
    COMPILED,
    PACKAGE_ROOT,
    POSITION_NAME,
    arm_parent_death_signal,
    EXIT_BAD_JOB,
    EXIT_COMPILED,
    EXIT_REFUSED,
    REFUSED,
    EntryJob,
    EntryReport,
    PackedGraphClass,
)
from . import aot_device_lock
from .hostfacts import cuda_ready
from .child_preflight import (
    PreflightRefused,
    assert_slots_resolvable,
    bind_slots,
    pick_compile_target,
    select_specs,
)
from .api.export_contract import export_declaration

logger = logging.getLogger(__name__)

#: TCG's non-CUDA compile target. Not a placeholder and not an accelerator
#: state: TCG resolves it to ``cpu-<isa>`` and keys the artifact on that, which
#: every CPU-lane host with the same ISA computes identically. Choosing it for
#: a mint that was bought to serve CUDA is the defect — see
#: ``compile_cache.compile_target_block`` and ``mint_child.mint``.
CPU_COMPILE_TARGET = "cpu"


def _install_posture(job: EntryJob) -> None:
    """Install worker-owned host policy before tracing or compiling."""
    from . import compile_posture

    compile_posture.install(job.posture)
    level = job.posture.nice_level()
    if level:
        try:
            os.nice(level)
        except OSError:
            logger.warning(
                "compile child could not apply nice level %d; continuing",
                level,
            )
    arm_parent_death_signal()


def _write(path: Path, report: EntryReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(msgspec.json.encode(report))
    os.replace(tmp, path)


def _mark_position(job: EntryJob, phase: str, *, detail: str = "") -> None:
    """The child's position beat (pgw#1371), rewritten at phase boundaries.

    A share is up to 36/K classes and its report lands once, at the very end
    — so from the parent's side a 46-minute healthy share and a wedged one
    read identically until the terminus (the 2026-08-18 fleet signature).
    This file is the child SAYING where it is: the parent counts a CHANGED
    position as silence-window evidence and quotes the last one when it
    condemns. Best-effort by construction — telemetry never fails a mint.
    """
    try:
        path = Path(job.report).parent / POSITION_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(json.dumps({
            "phase": phase, "detail": detail[:200],
            "epoch": round(time.time(), 3),
        }).encode())
        os.replace(tmp, path)
    except OSError:
        logger.debug("aot-compile: position beat failed", exc_info=True)


def _stream_class_row(job: EntryJob, index: int, packed: PackedGraphClass) -> None:
    """One packed class, streamed to the parent THE MOMENT it is on disk.

    pgw#1371: the artifact itself already lands per class (pgw#1183's
    durability atom); this row is the parent-visible half — it feeds the live
    `class_spans`, the pool ledger's `classes_landed`, the per-class beat the
    hub's stall rule reads, and the silence window. Written atomically and
    named by index so the parent can harvest strictly in order.
    """
    try:
        rows = Path(job.report).parent / CLASS_ROW_DIRNAME
        rows.mkdir(parents=True, exist_ok=True)
        tmp = rows / f".{index:03d}.tmp"
        tmp.write_bytes(msgspec.json.encode(packed))
        os.replace(tmp, rows / f"{index:03d}.json")
    except OSError:
        logger.debug("aot-compile: class row stream failed", exc_info=True)


def _peak_rss() -> int:
    """This share's host high-water — the interpreter AND the compilers it ran.

    pgw#848, MEASURED: ``RUSAGE_SELF`` alone is not the peak, it is the peak of
    everything EXCEPT the peak. The real sdxl wrapper TU compiled with the
    production flags on a quiet box peaks at **2.04 GiB in cc1plus**, a process
    this interpreter never allocates a byte for — ``RUSAGE_SELF`` read
    0.012 GiB across the same 177 s. The number feeds
    ``aot_compile_pool.entry_workers(peak_rss_bytes=...)``, which bounds K, so
    an instrument that cannot see the largest allocation is the difference
    between a pool that fits and a pool the OOM killer sizes.

    SUM, not max: the two overlap by construction — this process is holding the
    ExportedProgram and inductor's IR the whole time g++ runs, which is
    precisely why the pair is what a concurrent slot must reserve.
    ``RUSAGE_CHILDREN`` counts reaped children only, and every compiler process
    is reaped by the time ``aot_compile`` returns.

    ⚠️ pgw#1216 item 3: since one child takes a SHARE rather than one class,
    this is the worker's LIFETIME high-water over every class it traced and
    compiled, not one class's. K divides by it, so the two must not be
    confused — a per-class figure would size the pool against a process that
    does not exist.
    """
    try:
        me = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        kids = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (OSError, ValueError):
        return 0
    return (me + kids) * 1024


def _device_lock_wait_s() -> float:
    """Seconds this child spent BLOCKED on the pool-wide GPU benchmark lock.

    pgw#809 has counted it since the lock existed (``DeviceBenchmarkLock``
    tracks ``waited_s``) and nobody ever read the counter. At K=5 it is time
    inside ``compile_s`` during which the child is doing nothing at all — the
    single most misleading second in the table if it stays anonymous, because
    it looks like compile cost and is actually pool contention.
    """

    lock = aot_device_lock.installed()
    return round(float(getattr(lock, "waited_s", 0.0) or 0.0), 3) if lock else 0.0


def _peak_device() -> tuple:
    """This child's DEVICE high-water — allocated and reserved.

    pgw#868 A4. Telemetry only: no decision reads it, and the width policy is
    deliberately NOT changed in the same breath (pgw#830's rule — instrument
    first, optimise never in the same change). BOTH numbers, because they
    answer different questions: `allocated` is what the work needed, `reserved`
    is what the caching allocator held and therefore what a concurrent sibling
    actually cannot have.
    """
    try:
        import torch

        if not cuda_ready():
            return (0, 0)
        return (int(torch.cuda.max_memory_allocated()),
                int(torch.cuda.max_memory_reserved()))
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return (0, 0)


def _device_fields() -> dict:
    allocated, reserved = _peak_device()
    return {"peak_device_bytes": allocated,
            "peak_device_reserved_bytes": reserved}


_LATENT_RECONCILED = "latent_basis_reconciled"
_LATENT_UNRECONCILED_UNDECLARED = (
    "latent_basis_unreconciled_no_declared_basis"
)
_LATENT_UNRECONCILED_NO_VAE = "latent_basis_unreconciled_no_vae"


def _observed_latent_basis(pipeline: Any) -> Optional[int]:
    """The composed pipeline's real latent divisor, when observable."""
    if getattr(pipeline, "vae", None) is None:
        return None
    value = getattr(pipeline, "vae_scale_factor", None)
    if value is None:
        return None
    try:
        basis = int(value)
    except (TypeError, ValueError):
        return None
    return basis if basis > 0 else None


def _reconcile_latent_basis(pipeline: Any, declaration: Any) -> str:
    """Refuse a declared latent basis that disagrees with this composition.

    The child is the only production path where the loaded composition and
    declaration coexist before export. An absent declaration or VAE is an
    explicit unreconciled state; only a proven mismatch refuses.
    """
    declared = getattr(declaration, "latent_basis", None)
    if declared is None:
        logger.info(
            "aot-compile-child: %s",
            _LATENT_UNRECONCILED_UNDECLARED,
        )
        return _LATENT_UNRECONCILED_UNDECLARED
    observed = _observed_latent_basis(pipeline)
    if observed is None:
        logger.info(
            "aot-compile-child: %s — declared basis %d",
            _LATENT_UNRECONCILED_NO_VAE,
            declared,
        )
        return _LATENT_UNRECONCILED_NO_VAE
    if int(declared) != observed:
        raise PreflightRefused(
            "latent_basis_mismatch: the declaration derived its graph classes "
            f"at a latent divisor of {declared}, and the composed pipeline's "
            f"vae divides by {observed}. Every declared latent extent is "
            "therefore wrong: the declaration does not match this composition "
            "— check the `latent_scale=` passed to the class deriver and any "
            "VAE component override"
        )
    return _LATENT_RECONCILED


def build_pipeline(job: EntryJob) -> Tuple[Any, Any, Any]:
    """``(pipeline, export spec, export declaration)`` for this child's share.

    The mint child's own load, reached through the same four preflight checks
    ``boot_trace_child`` uses — deliberately, because two compositions trace
    two graphs and a compiled graph that does not match the boot key it is
    filed under is the failure the derived key exists to rule out.

    ``place=False`` (pgw#1124): this pipeline never runs a forward, so the
    worker's serving placement ladder must not move its real non-target
    components onto the card the serving parent is resident on. That holds on
    BOTH composition paths below — the fallback loads a checkpoint, it still
    never executes one.

    **Structure-only is PREFERRED, never REQUIRED** (pgw#1215, restoring the
    semantics ``mint_child._load`` has had since pgw#1123). A family whose
    component has no config-only surface — a quantized artifact lane, a class
    the tree's ``model_index.json`` does not name — is stranded on the
    real-weight mint, and a real-weight mint is a correct, more expensive
    mint. Requiring structure-only here would make every such family, which
    minted on the parent-traced path, UNMINTABLE the moment the trace moved
    into this process: a capability regression with no fleet-facing notice.
    ``StructureNotHonored`` is the one refusal that stays fatal, because it
    means the target WAS built weight-free and the pipeline discarded it —
    falling back there exports weight-scale REAL tensors while the child
    reports weightless (ie#638's silent 40 GiB OOM).

    A named function because every one of its refusals is deterministic and
    typed, and because a test can reach the whole preflight without an
    inductor compile — the compile itself cannot be a cheap test.
    """
    from . import compile_cache as cc, fleet_compiled_graphs
    from .cli.run import run_setup
    from .models import structure_only
    from .registry import collect_endpoints

    # AOTInductor always links a C++ wrapper. Refuse before endpoint
    # collection or either pipeline-composition path can read weights; the
    # generic C-compiler predicate is insufficient on images that carry only
    # cc/gcc (pgw#823's measured 336-second late refusal).
    if not cc.cxx_toolchain_present():
        raise PreflightRefused(
            "no C++ compiler for the AOTI compile child")

    cfg = job.cfg
    specs = collect_endpoints(list(job.modules))
    chosen, siblings = select_specs(specs, job.function)
    bind_slots(siblings, job.slots)
    assert_slots_resolvable(
        siblings, job.slots,
        what=f"compile of {job.share or 'share'} for {job.function!r}")

    paths = {name: slot.path for name, slot in job.slots.items()}
    try:
        loaded = run_setup(
            chosen.cls(), dict(paths), arm_compile=False, return_loaded=True,
            place=False,
            structure_only=tuple(cfg.targets)) or {}
    except structure_only.StructureNotHonored as exc:
        raise PreflightRefused(
            f"structure-only was requested for {job.function!r} and the "
            f"target built weight-free, but the composed pipeline did not "
            f"carry it: {exc}") from exc
    except structure_only.StructureOnlyUnsupported as exc:
        # A CAPABILITY refusal is not this family declining, it is this image
        # being unable to (pgw#1123) — and it is louder, because the same
        # image derives no boot key either, so every pod running it re-mints.
        if isinstance(exc, structure_only.StructureCapabilityMissing):
            logger.error(
                "the weight-free compile is unavailable in this image, so "
                "this share loads real weights and every boot in it will "
                "self-mint: %s", exc)
        else:
            logger.warning(
                "structure-only %s: %s — this share compiles from real "
                "weights", structure_only.refusal_token(exc), exc)
        loaded = run_setup(
            chosen.cls(), dict(paths), arm_compile=False, return_loaded=True,
            place=False) or {}
    _slot, pipeline = pick_compile_target(loaded, cfg)

    if cfg.lora_bucket:
        # The CONTAINER half and the lane stamp, exactly as `mint_child` and
        # `boot_trace_child` arm the pipeline they hand the export. The LIFTED
        # half belongs to the loop that needs it — `aot_mint.trace_for_key`
        # arms it itself (pgw#1132); a second arm site here is how the two
        # halves drifted apart.
        cc.apply_lora_execution_lane(pipeline, int(cfg.lora_bucket))
    spec = fleet_compiled_graphs.aot_export_spec(pipeline, cfg)
    decl = export_declaration(str(spec.family or ""))
    if decl is None:
        raise PreflightRefused(
            f"family {spec.family!r} has no registered export declaration — "
            f"a multi-graph compiled graph derives its class set from it")
    _reconcile_latent_basis(pipeline, decl)
    return pipeline, spec, decl


@dataclass(frozen=True)
class _TCGClassResult:
    """One TCG result before it is projected onto the existing pool wire."""

    packed: PackedGraphClass
    compile_s: float
    reuse_s: float
    pack_s: float


def _tcg_runtime(cache_root: Optional[Path] = None) -> Tuple[Any, Any]:
    """Open TCG with sealed compiler facts.

    Production omits ``cache_root`` and therefore uses the worker's canonical
    CAS. The explicit measurement CLI supplies a temporary root so a diagnostic
    run cannot mutate serving state.

    TCG's target vocabulary is exactly ``"cpu"`` or a concrete ``sm_NN``, and
    it validates the CUDA form against the visible device. ``cpu`` is not a
    placeholder there — it keys the artifact ``cpu-<isa>``, an honest identity
    every CPU-lane host with that ISA computes — so this function may name it,
    and the measurement CLI and the CPU-lane compile tests depend on that.

    What it must NOT do is CHOOSE the lane. A mint pod bought to serve CUDA
    that quietly compiles ``cpu-avx512`` has burnt itself: the artifact is
    keyed truthfully and no GPU pod will ever adopt it, so the family re-mints
    forever. That decision belongs to ``mint_child.mint``, which refuses
    deterministically on :func:`compile_cache.compile_target_block` before any
    export — pgw#985. This frame only reports which of the two targets the host
    can state.
    """
    from gen_worker._vendor.torchcg import RuntimeCompatibility

    from . import compile_cache
    from .models.cache_paths import open_worker_engine

    sm = str(compile_cache.runtime_key().get("sm") or "").strip()
    compatibility = RuntimeCompatibility(
        sm or CPU_COMPILE_TARGET,
        toolchain=dict(compile_cache.toolchain_digest()),
    )
    return open_worker_engine(cache_root), compatibility


def _compile_traced_class(
    traced: Any,
    export_spec: Any,
    engine: Any,
    runtime: Any,
    *,
    work: Path,
    out_dir: Path,
) -> _TCGClassResult:
    """Compile/reuse and export exactly one TCG-owned artifact."""
    from . import aot_mint

    spec = aot_mint.tcg_graph_class_spec(traced, export_spec)
    token = hashlib.sha256(str(traced.name).encode()).hexdigest()[:16]
    materialized = work / "tcg-materialized" / token
    started = time.monotonic()
    ensured = engine.compile(spec, runtime, materialized)
    elapsed = time.monotonic() - started
    stored = ensured.compiled_graph
    key = str(stored.key)
    metadata = dict(stored.metadata)
    if metadata.get("compiled_graph_key") != key:
        raise ValueError(
            f"TCG result for {traced.name!r} records key "
            f"{metadata.get('compiled_graph_key')!r}, not selected ref {key!r}"
        )
    graph_class = metadata.get(GRAPH_CLASS_BLOCK)
    if not isinstance(graph_class, dict) or graph_class.get("name") != traced.name:
        raise ValueError(
            f"TCG result for {traced.name!r} records the wrong graph_class"
        )
    t_pack = time.monotonic()
    artifact = engine.export_artifact(key, out_dir / f"{key}.tar.gz")
    pack_s = time.monotonic() - t_pack
    reused = str(getattr(ensured.outcome, "value", ensured.outcome)) == "reused"
    return _TCGClassResult(
        packed=PackedGraphClass(
            name=str(traced.name),
            key=key,
            artifact=str(artifact),
            metadata=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        ),
        compile_s=0.0 if reused else elapsed,
        reuse_s=elapsed if reused else 0.0,
        pack_s=pack_s,
    )


def compile_traced_class(
    traced: Any,
    export_spec: Any,
    engine: Any,
    runtime: Any,
    *,
    work: Path,
    out_dir: Path,
) -> _TCGClassResult:
    try:
        return _compile_traced_class(
            traced, export_spec, engine, runtime, work=work, out_dir=out_dir
        )
    finally:
        # One in-flight class is the loss bound. Release even when spec
        # derivation, admission, compilation, storage or export refuses.
        traced.release()


def _trace_share(
    aot_mint: Any,
    pipeline: Any,
    export_spec: Any,
    declaration: Any,
    job: EntryJob,
) -> Any:
    """Apply the holes and retry filters before export; no worker compile."""
    return aot_mint.trace_for_key(
        pipeline,
        export_spec,
        declaration,
        share_index=int(job.share_index),
        share_count=int(job.share_count),
        have_classes=tuple(job.have_classes),
        hole_classes=tuple(job.hole_classes),
    )


def run(job: EntryJob) -> int:
    from . import aot_mint, env_seal

    started = time.monotonic()
    # pgw#830: the child's own wall, partitioned. `close()` at every exit,
    # including the refusal paths — a share that died still spent its seconds
    # somewhere, and an aborted mint's table is the one a reader needs most.
    ledger = SpanLedger()
    report_path = Path(job.report)
    # Before anything expensive: if the parent dies, this work dies with it. A
    # serving pod must never be left burning CPU on a compiled graph nobody is waiting
    # for any more.
    _install_posture(job)
    # The FIRST beat, before anything that can take minutes: a child whose
    # position file never appears did not reach its own code.
    _mark_position(job, "start")
    # The seal is the parent's — re-established, not re-derived, because this
    # process emits the very bytes the seal describes. A child that sealed
    # differently would produce an artifact the parent's verify() rejects on
    # an axis nobody changed (pgw#784's rule, same reason).
    # NOT reordered to put `import torch` first, tempting as that is for the
    # attribution: `scrub_env()` must run before torch reads the environment
    # (env_seal's whole contract), and the torch imposition imports torch.
    with ledger.span("child_seal_s"):
        env_seal.establish()
    seal_detail = dict(env_seal.LAST_ESTABLISH_SPANS)
    _mark_position(job, "sealed")
    # Before ANY compile touches the card: every inductor GPU benchmark in
    # this process goes through the pool-wide lock, so K concurrent children
    # cannot time kernels against each other and bake contention-chosen
    # configs into a compiled graph whose key would not move (pgw#809).
    with ledger.span("child_devlock_s"):
        if job.device_lock:
            aot_device_lock.install(Path(job.device_lock))

    def _refuse(detail: str, spans: Dict[str, float] | None = None) -> int:
        _write(report_path, EntryReport(
            entry=job.share, status=REFUSED, detail=detail[:8000],
            elapsed_s=round(time.monotonic() - started, 2),
            peak_rss_bytes=_peak_rss(),
            **_device_fields(),
            **_span_fields(ledger, spans or {}, {}, seal_detail)))
        return EXIT_REFUSED

    try:
        # Timed SEPARATELY from the setup below. On a K-wide pool this is paid
        # K times — one cold interpreter per share — and the pool's process
        # boundary is the reason it exists at all, so it has to be a line item
        # before anyone can argue about a persistent worker.
        with ledger.span("child_torch_import_s"):
            import torch
        _mark_position(job, "torch_imported")

        # pgw#1215: this is what replaced `child_program_load_s`. The child
        # composes the weight-free target it is about to trace instead of
        # deserializing one somebody else traced.
        with ledger.span("child_setup_s"):
            pipeline, spec, decl = build_pipeline(job)
            env_seal.assert_seal_unchanged("aot compile child setup")
        _mark_position(job, "pipeline_composed")
    except PreflightRefused as exc:
        return _refuse(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("aot-compile: could not build the compile target")
        return _refuse(
            f"{job.share}: could not build the compile target for "
            f"{job.function!r}: {type(exc).__name__}: {exc}")

    # pgw#757's phase split is read from dynamo's IN-PROCESS counters, so it
    # has to be measured here, in the process that does the compiling.
    # pgw#868 A4: reset so the high-water below is this SHARE'S OWN work, not
    # whatever the seal's torch import allocated first. A probe: it reads and
    # clears a counter and decides nothing.
    try:
        if cuda_ready():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        pass

    before = phase_snapshot()
    # pgw#1183's ordering, held at the graph class (pgw#1215). The rows are
    # packed INSIDE the loop, one class at a time, and never accumulated for a
    # batch pack at the end. That is not a style choice: a child that held its
    # whole compiled share in memory and packed it on the way out would lose
    # every class it had finished to a crash on class k — up to 36/K classes at
    # a measured 156-509 s each — which is exactly the all-or-nothing loss
    # th#1825 spent 1 h 37 m on and pgw#1183 fixed. Packed means ON DISK, so a
    # crash costs the ONE class in flight.
    packed: List[PackedGraphClass] = []
    trace_s = 0.0
    compile_s = 0.0
    pack_s = 0.0
    declared = 0
    refusal = ""
    current_class = job.share
    try:
        engine, runtime = _tcg_runtime()
        for traced in _trace_share(aot_mint, pipeline, spec, decl, job):
            current_class = traced.name
            declared = int(traced.declared) or declared
            _mark_position(
                job, "compile", detail=f"{traced.name} "
                f"({len(packed) + 1} of this share, {declared} declared)")
            timings = dict(traced.timings or {})
            row_trace = float(timings.get("export_s", 0.0))
            trace_s += row_trace
            class_before = phase_snapshot()
            result = compile_traced_class(
                traced,
                spec,
                engine,
                runtime,
                work=Path(job.work),
                out_dir=Path(job.out_dir),
            )
            partition, _overlays, _raw = phase_delta(
                class_before, phase_snapshot())
            compile_s += result.compile_s
            pack_s += result.pack_s
            spans = {
                "export_s": round(row_trace, 3),
                "compile_s": round(result.compile_s, 3),
                "reuse_s": round(result.reuse_s, 3),
                "nodes": float(traced.nodes),
                **{str(k): float(v) for k, v in partition.items()},
            }
            packed.append(PackedGraphClass(
                name=result.packed.name,
                key=result.packed.key,
                artifact=result.packed.artifact,
                metadata=result.packed.metadata,
                spans=spans,
            ))
            # pgw#1371: the parent learns of the class NOW, not at the
            # share's report — the artifact is already on disk, which is the
            # durability bound (pgw#1183) this row makes visible.
            _stream_class_row(job, len(packed) - 1, packed[-1])
            _mark_position(
                job, "packed", detail=f"{traced.name} "
                f"({len(packed)} of this share, {declared} declared)")
    except aot_mint.MintRefused as exc:
        # NOT a discard. Every class this child already packed is on disk and
        # is named in the report — a share is not all-or-nothing, which is the
        # whole point of the per-graph-class atom. The refusal rides beside
        # them so the parent can say WHICH class stopped the share.
        refusal = str(exc)
    except Exception as exc:  # noqa: BLE001 — classified before reporting
        aot_mint.raise_if_device_oom(
            exc, f"TCG compile of {current_class!r}")
        refusal = f"TCG refused {type(exc).__name__}: {exc}"

    if not declared:
        # A share can legitimately yield NOTHING — every class in it is
        # already held (pgw#1215 step 4's retry), or K exceeded the class
        # count. The declared count is what proves the shares cover the
        # declaration, so it cannot be a by-product of having traced
        # something; it is enumerated directly. No export, no compile: this
        # is the same `declared_class_rows` call the trace loop opens with.
        try:
            declared = len(aot_mint.declared_class_rows(pipeline, spec, decl))
        except Exception:  # noqa: BLE001 — the pool refuses on 0, loudly
            logger.exception("aot-compile: could not enumerate declared rows")

    # pgw#1371 holes-only: how many of THIS share's rows the hole list
    # matched, before the have filter — the number the parent sums to prove
    # hole coverage. -1 (no evidence) on a plain mint, and on a share that
    # cannot enumerate — the pool refuses that loudly rather than treating
    # it as zero.
    targeted = -1
    if job.hole_classes:
        try:
            targeted = aot_mint.share_targeted(
                pipeline, spec, decl,
                share_index=int(job.share_index),
                share_count=int(job.share_count),
                hole_classes=tuple(job.hole_classes))
        except Exception:  # noqa: BLE001 — the pool refuses -1, loudly
            logger.exception("aot-compile: could not count targeted rows")

    ledger.mark("child_trace_s", trace_s)
    ledger.mark("compile_wall_s", compile_s)
    ledger.mark("child_pack_s", pack_s)
    partition, overlays, raw = phase_delta(before, phase_snapshot())
    if refusal:
        _write(report_path, EntryReport(
            entry=job.share, status=REFUSED, classes=packed,
            declared_classes=declared,
            targeted_classes=targeted,
            phases=dict(partition),
            detail=(
                f"{len(packed)} graph class(es) packed before the share "
                f"refused: {refusal}")[:8000],
            elapsed_s=round(time.monotonic() - started, 2),
            peak_rss_bytes=_peak_rss(),
            **_device_fields(),
            metrics_raw={k: v for k, v in sorted(raw.items())},
            **_span_fields(ledger, partition, overlays, seal_detail)))
        return EXIT_REFUSED

    _write(report_path, EntryReport(
        entry=job.share, status=COMPILED, classes=packed,
        declared_classes=declared,
        targeted_classes=targeted,
        phases=dict(partition),
        detail=f"{len(packed)} packed graph class(es), {declared} declared",
        elapsed_s=round(time.monotonic() - started, 2),
        peak_rss_bytes=_peak_rss(),
        **_device_fields(),
        metrics_raw={k: v for k, v in sorted(raw.items())},
        **_span_fields(ledger, partition, overlays, seal_detail)))
    return EXIT_COMPILED


def _span_fields(
    ledger: SpanLedger,
    partition: Dict[str, float],
    overlays: Dict[str, float],
    seal_detail: Dict[str, float],
) -> Dict[str, Any]:
    """Seal the child's ledger and shape it for :class:`EntryReport`.

    Order matters: the child's own wall is closed FIRST, against only the
    spans the ledger itself timed, so ``child_other_s`` is a true residual.
    The inductor leaves are folded in AFTERWARDS, as the sub-partition of
    ``compile_wall_s`` — folding them in earlier would let them count twice
    and make the residual negative.

    Neither residual may be implied by subtraction at read time: a reader who
    has to do the subtraction himself is exactly how 44 % went dark.
    """
    # EVERY declared member, seeded to zero before the ledger closes. A member
    # the ledger never touched is not "0" to TCG's `spans.check` — it is
    # MISSING, and the residual silently absorbs it, which is pgw#830's own
    # defect one level down. A child that refuses before it traces has really
    # spent zero seconds tracing, and must say so.
    for label in PARTITIONS["child_wall_s"]:
        if label != "child_other_s":
            ledger.spans.setdefault(label, 0.0)
    spans = ledger.close("child_wall_s", "child_other_s")
    compile_wall = float(spans.get("compile_wall_s", 0.0))
    for label in PARTITION_KEYS:
        spans[label] = round(float(partition.get(label, 0.0)), 3)
    spans["compile_other_s"] = round(compile_wall - sum(
        spans[label] for label in PARTITION_KEYS), 3)
    overlay = dict(overlays)
    # A split of `child_seal_s`, never a member of any partition: the seal's
    # own steps, published by env_seal. `seal_libhash_s` is the identity
    # manifest's SHA-256 pass over every toolchain .so the image ships — paid
    # once per SHARE here, because the pool's worker is a process that exits.
    overlay.update({k: float(v) for k, v in seal_detail.items()})
    waited = _device_lock_wait_s()
    if waited:
        overlay["device_lock_wait_s"] = waited
    violations = check(spans)
    if violations:
        logger.warning(
            "aot-compile: child span partition does not close: %s",
            "; ".join(violations),
        )
    return {
        "spans": spans,
        "overlays": overlay,
        "spans_v": SPANS_V,
        "module_import_epoch": MODULE_IMPORT_EPOCH,
        "run_start_epoch": ledger.start_epoch,
        "report_epoch": time.time(),
        # pgw#840: WHICH gen_worker this was. Carried on every report shape
        # (compiled and both refusals) because a skewed child's refusal is as
        # misleading as its success — it is a verdict from other code.
        "code_digest": CODE_DIGEST,
        "code_dir": PACKAGE_ROOT,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("GEN_WORKER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m gen_worker.aot_compile_child <job.json>",
              file=sys.stderr)
        return EXIT_BAD_JOB
    try:
        job = msgspec.json.decode(Path(args[0]).read_bytes(), type=EntryJob)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        print(f"unreadable job {args[0]!r}: {exc}", file=sys.stderr)
        return EXIT_BAD_JOB
    return run(job)


if __name__ == "__main__":
    raise SystemExit(main())
