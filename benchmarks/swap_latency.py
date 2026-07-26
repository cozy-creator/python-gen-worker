"""Repo-side shim (pgw#674): the harness moved INTO the wheel so pods have
it — see ``src/gen_worker/benchmarks/swap_latency.py``. Run it as

    python -m gen_worker.benchmarks.swap_latency ...

or dispatch it as a worker function via ``gen_worker.diagnostics``.
This shim keeps the old ``python -m benchmarks.swap_latency`` invocation
working from a repo checkout."""

from __future__ import annotations

import sys

from gen_worker.benchmarks.swap_latency import (  # noqa: F401
    ALL_CASES,
    Collector,
    OffPodError,
    Row,
    bench_demote_promote,
    bench_load,
    bench_overlap,
    bench_stage,
    bench_swap,
    check_on_pod,
    component_digest,
    main,
    run_cases,
    swap_plan,
)

if __name__ == "__main__":
    sys.exit(main())
