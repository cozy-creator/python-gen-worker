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

import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import msgspec

from .aot_compile_pool import (
    COMPILED,
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
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (OSError, ValueError):
        return 0


def run(job: EntryJob) -> int:
    from . import aot_device_lock, aot_mint, env_seal

    started = time.monotonic()
    report_path = Path(job.report)
    # The seal is the parent's — re-established, not re-derived, because this
    # process emits the very bytes the seal describes. A child that sealed
    # differently would produce an artifact the parent's verify() rejects on
    # an axis nobody changed (pgw#784's rule, same reason).
    env_seal.establish()
    # Before ANY compile touches the card: every inductor GPU benchmark in
    # this process goes through the pool-wide lock, so K concurrent entries
    # cannot time kernels against each other and bake contention-chosen
    # configs into a cell whose key would not move (pgw#809).
    if job.device_lock:
        aot_device_lock.install(Path(job.device_lock))
    try:
        import torch

        program = torch.export.load(job.program)
    except Exception as exc:  # noqa: BLE001
        _write(report_path, EntryReport(
            entry=job.entry, status=REFUSED,
            detail=(
                f"could not load the exported program from {job.program!r}: "
                f"{type(exc).__name__}: {exc}"),
            elapsed_s=round(time.monotonic() - started, 2),
            peak_rss_bytes=_peak_rss()))
        return EXIT_REFUSED

    # pgw#757's phase split is read from dynamo's IN-PROCESS counters, so it
    # has to be measured here, in the process that does the compiling.
    before = aot_mint._phase_snapshot()
    try:
        files = aot_mint.compile_entry_files(
            program, job.entry, inductor_configs=job.inductor_configs)
    except aot_mint.MintRefused as exc:
        _write(report_path, EntryReport(
            entry=job.entry, status=REFUSED, detail=str(exc),
            elapsed_s=round(time.monotonic() - started, 2),
            peak_rss_bytes=_peak_rss()))
        return EXIT_REFUSED

    phases = aot_mint._phase_delta(before, aot_mint._phase_snapshot())
    _write(report_path, EntryReport(
        entry=job.entry, status=COMPILED, files=[str(f) for f in files],
        phases=dict(phases or {}),
        detail=f"{len(files)} loose file(s)",
        elapsed_s=round(time.monotonic() - started, 2),
        peak_rss_bytes=_peak_rss()))
    return EXIT_COMPILED


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
