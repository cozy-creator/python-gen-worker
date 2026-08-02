"""pgw#809: one entry's AOTI compile, one process, then exit.

``python -m gen_worker.aot_compile_child <job.json>``

Run by :class:`aot_compile_pool.EntryCompilePool` K-wide while the mint child
(pgw#784) goes on exporting the rest of the cell's entries and the serving
process goes on serving eager. The boundary is a file for the same reason
pgw#784's is: a mint that fails on entry 13 of 18 leaves the job behind and
the compile is reproducible by hand, on the same box, without the pipeline.

What it does NOT do is as important as what it does. It does not load the
endpoint, does not touch the hub, does not warm anything and does not decide
anything about the cell. It loads ONE exported program, calls the SAME
``aot_mint.compile_entry_files`` the serial path calls, and reports where the
loose files landed. Every gate — code-only, bindability, constant-set drift,
declared range — stays in the parent, running against the parent's own
in-memory program, which is what makes a divergent child a caught defect
rather than a published one.
"""

from __future__ import annotations

import time

#: pgw#830: the FIRST statement of the module, before any gen_worker import.
#: The parent's ``compile_s`` starts at its own ``Popen`` return, so the
#: interpreter's boot and this package's import are inside the measured total;
#: without this stamp they can only ever land in a residual. Wall-clock (not
#: monotonic) because the span it closes is measured across two processes.
MODULE_IMPORT_EPOCH = time.time()

import logging
import os
import resource
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import msgspec

from . import aot_compile_spans
from .aot_compile_pool import (
    CODE_DIGEST,
    COMPILED,
    PACKAGE_ROOT,
    arm_parent_death_signal,
    EXIT_BAD_JOB,
    EXIT_COMPILED,
    EXIT_REFUSED,
    REFUSED,
    EntryJob,
    EntryReport,
)

logger = logging.getLogger(__name__)


def _write(path: Path, report: EntryReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(msgspec.json.encode(report))
    os.replace(tmp, path)


def _peak_rss() -> int:
    """This entry's host high-water — the interpreter AND the compiler it ran.

    pgw#848, MEASURED: ``RUSAGE_SELF`` alone is not this entry's peak, it is
    the peak of everything EXCEPT the peak. The real sdxl wrapper TU compiled
    with the production flags on a quiet box peaks at **2.04 GiB in cc1plus**,
    a process this interpreter never allocates a byte for — ``RUSAGE_SELF``
    read 0.012 GiB across the same 177 s. The number feeds
    ``aot_compile_pool.entry_workers(peak_rss_bytes=...)``, which bounds K, so
    an instrument that cannot see the largest allocation in the entry is the
    difference between a pool that fits and a pool the OOM killer sizes.

    SUM, not max: the two overlap by construction — this process is holding
    the loaded ``ExportedProgram`` and inductor's IR the whole time g++ runs,
    which is precisely why the pair is what a concurrent slot must reserve.
    ``RUSAGE_CHILDREN`` counts reaped children only, and every compiler
    process is reaped by the time ``aot_compile`` returns.
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
    inside ``compile_s`` during which the entry is doing nothing at all — the
    single most misleading second in the table if it stays anonymous, because
    it looks like compile cost and is actually pool contention.
    """
    from . import aot_device_lock

    lock = aot_device_lock.installed()
    return round(float(getattr(lock, "waited_s", 0.0) or 0.0), 3) if lock else 0.0


def _peak_device() -> tuple:
    """This entry child's DEVICE high-water — allocated and reserved.

    pgw#868 A4. Nothing has ever measured this, and it is the number that
    holds K to 1-2. The pool's per-entry device ask is
    `mint_budget.co_residency().need_bytes`, which is NOT a compile-child
    measurement at all: it is `resident_weights * 1.25 + 5 GiB`, i.e. sdxl's
    4.87 GiB of weights plus an activation term the module's own docstring
    calls "a fraction nobody measured" plus a flat context+workspace constant.
    That estimate is what `per_entry_device_basis: 'measured'` reports as
    "measured" — meaning "the caller handed a probed number", not "somebody
    watched a compile child".

    So this reports what the child ACTUALLY peaked at. Telemetry only: no
    decision reads it, and the width policy is deliberately NOT changed in the
    same breath (pgw#830's rule — instrument first, optimise never in the same
    change). BOTH numbers, because they answer different questions: `allocated`
    is what the compile needed, `reserved` is what the caching allocator held
    and therefore what a concurrent sibling actually cannot have.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return (0, 0)
        return (int(torch.cuda.max_memory_allocated()),
                int(torch.cuda.max_memory_reserved()))
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return (0, 0)


def _device_fields() -> dict:
    allocated, reserved = _peak_device()
    return {"peak_device_bytes": allocated,
            "peak_device_reserved_bytes": reserved}


def run(job: EntryJob) -> int:
    from . import aot_device_lock, aot_mint, env_seal

    started = time.monotonic()
    # pgw#830: the child's own wall, partitioned. `close()` at every exit,
    # including the refusal paths — an entry that died still spent its
    # seconds somewhere, and an aborted mint's table is the one a reader
    # needs most.
    ledger = aot_compile_spans.SpanLedger()
    report_path = Path(job.report)
    # Before anything expensive: if the mint child dies, this compile dies
    # with it. A serving pod must never be left burning CPU on a cell nobody
    # is waiting for any more.
    arm_parent_death_signal()
    # The seal is the parent's — re-established, not re-derived, because this
    # process emits the very bytes the seal describes. A child that sealed
    # differently would produce an artifact the parent's verify() rejects on
    # an axis nobody changed (pgw#784's rule, same reason).
    # NOT reordered to put `import torch` first, tempting as that is for the
    # attribution: `scrub_env()` must run before torch reads the environment
    # (env_seal's whole contract), and `establish_config` imports torch itself.
    # So the seal legitimately OWNS the torch import, and the split below —
    # published by env_seal as telemetry — is how it gets named anyway.
    with ledger.span("child_seal_s"):
        env_seal.establish()
    seal_detail = dict(env_seal.LAST_ESTABLISH_SPANS)
    # Before ANY compile touches the card: every inductor GPU benchmark in
    # this process goes through the pool-wide lock, so K concurrent entries
    # cannot time kernels against each other and bake contention-chosen
    # configs into a cell whose key would not move (pgw#809).
    with ledger.span("child_devlock_s"):
        if job.device_lock:
            aot_device_lock.install(Path(job.device_lock))
    try:
        # Timed SEPARATELY from the program load. On a 72-entry mint this is
        # paid 72 times — one cold interpreter per entry — and the pool's
        # process boundary is the reason it exists at all, so it has to be a
        # line item before anyone can argue about a persistent worker.
        with ledger.span("child_torch_import_s"):
            import torch

        # The other half of the process boundary: pgw#809 hands the child its
        # ExportedProgram on disk (`torch.export.save` in the parent, here the
        # matching `load`). Named because "serialization across the pool's
        # process boundary" was a prime suspect for the dark 44 %.
        with ledger.span("child_program_load_s"):
            program = torch.export.load(job.program)
    except Exception as exc:  # noqa: BLE001
        _write(report_path, EntryReport(
            entry=job.entry, status=REFUSED,
            detail=(
                f"could not load the exported program from {job.program!r}: "
                f"{type(exc).__name__}: {exc}"),
            elapsed_s=round(time.monotonic() - started, 2),
            peak_rss_bytes=_peak_rss(),
            **_device_fields(),
            **_span_fields(ledger, {}, {}, seal_detail)))
        return EXIT_REFUSED

    # pgw#757's phase split is read from dynamo's IN-PROCESS counters, so it
    # has to be measured here, in the process that does the compiling.
    # pgw#868 A4: reset so the high-water below is this ENTRY'S COMPILE, not
    # whatever `torch.export.load` or the seal's own torch import allocated
    # first. A probe: it reads and clears a counter and decides nothing.
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        pass
    before = aot_compile_spans.phase_snapshot()
    try:
        with ledger.span("compile_wall_s"):
            files = aot_mint.compile_entry_files(
                program, job.entry, inductor_configs=job.inductor_configs)
    except aot_mint.MintRefused as exc:
        partition, overlays, _raw = aot_compile_spans.phase_delta(
            before, aot_compile_spans.phase_snapshot())
        _write(report_path, EntryReport(
            entry=job.entry, status=REFUSED, detail=str(exc),
            elapsed_s=round(time.monotonic() - started, 2),
            peak_rss_bytes=_peak_rss(),
            **_device_fields(),
            **_span_fields(ledger, partition, overlays, seal_detail)))
        return EXIT_REFUSED

    partition, overlays, raw = aot_compile_spans.phase_delta(
        before, aot_compile_spans.phase_snapshot())
    _write(report_path, EntryReport(
        entry=job.entry, status=COMPILED, files=[str(f) for f in files],
        phases=dict(partition),
        detail=f"{len(files)} loose file(s)",
        elapsed_s=round(time.monotonic() - started, 2),
        peak_rss_bytes=_peak_rss(),
        **_device_fields(),
        metrics_raw={k: v for k, v in sorted(raw.items())},
        **_span_fields(ledger, partition, overlays, seal_detail)))
    return EXIT_COMPILED


def _span_fields(
    ledger: aot_compile_spans.SpanLedger,
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
    spans = ledger.close("child_wall_s", "child_other_s")
    compile_wall = float(spans.get("compile_wall_s", 0.0))
    for label in aot_compile_spans.PARTITION_KEYS:
        spans[label] = round(float(partition.get(label, 0.0)), 3)
    spans["compile_other_s"] = round(compile_wall - sum(
        spans[label] for label in aot_compile_spans.PARTITION_KEYS), 3)
    overlay = dict(overlays)
    # A split of `child_seal_s`, never a member of any partition: the seal's
    # own steps, published by env_seal. `seal_libhash_s` is the identity
    # manifest's SHA-256 pass over every toolchain .so the image ships — paid
    # once per ENTRY here, because the pool's worker is a process that exits.
    overlay.update({k: float(v) for k, v in seal_detail.items()})
    waited = _device_lock_wait_s()
    if waited:
        overlay["device_lock_wait_s"] = waited
    return {
        "spans": spans,
        "overlays": overlay,
        "spans_v": aot_compile_spans.SPANS_V,
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
