"""pgw#868 A4: export the rows of one adapter arm CONCURRENTLY, in processes.

**The measurement this exists for.** `trace_graph` is ~74 minutes for sdxl's 36
rows (2.06 and 2.07 min/row, two pods, agreeing to two decimals), it runs
SERIALLY in the mint parent, and pgw#809's compile pool divides none of it. The
profile says why parallelism is the lever and not a local fix: >80 % of a row is
single-threaded Python proxy-tensor tracing and fake-tensor dispatch, with flat
self-time (largest frame 5.5 %) — there is no hot path to optimise. And a
controlled four-size series says width buys what it appears to buy: export is
**sub-linear** in node count (exponent 0.71-0.93) with per-call cost flat or
falling, so parallelism divides a real wall rather than fighting a curve.

**The seam, and why "serial" was broader than the constraint.**
``aot_mint._export_entry`` says export *"must stay SERIAL — one live pipeline,
one card, one branch-arm toggle"*. All three hold AT THE BOUNDARY:
``_disarm_branches`` mutates the module in place, exactly ONCE, between the
adapter-bearing rows and the branchless ones. Rows *within* an arm mutate
nothing. So sdxl's 36 rows are **two internally order-independent groups of
18**, and the serialisation across those 18 was inherited from where the model
lives, not required by correctness.

**Processes, not threads:** the bottleneck is the GIL, so a thread pool buys
nothing. Not `fork` either — banned after CUDA init (pgw#784). Each worker is a
fresh interpreter that builds its OWN module copy, which is what makes the
width VRAM-bounded.

**Width is bounded by the EXPORT-phase footprint, which nobody has measured.**
It is NOT the compile pool's `weights x 1.25 + 5 GiB` — that estimate's
activation and workspace terms are *inductor's*, and ~56 % of it was never
observed. Export traces with fake tensors and executes no kernel, so its
footprint should be close to the resident weights alone. `peak_device_bytes`
(pgw#868) measures it per phase on the next mint; **until then this refuses to
guess and stays OFF.**

**The rule, and the gate.** pgw#846 governs: the traced graph may not move. So
this ships behind ``GEN_WORKER_AOT_EXPORT_PARALLEL`` (OFF), and a worker's
artifact must be **byte-identical** to the parent's for the same row — proven,
not asserted, by comparing generated C++ in the SAME cleared build directory
(the build dir and expanded ``-march`` are embedded; different dirs can never be
byte-equal, a trap this program has hit twice).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

#: OFF by default. Turning it on needs the export-phase VRAM measurement first.
ENV_FLAG = "GEN_WORKER_AOT_EXPORT_PARALLEL"

#: Below this a group cannot repay a worker's own module load.
MIN_GROUP = 3


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "on")


def groups(rows: Sequence[Tuple[Any, Any]]) -> List[List[int]]:
    """Split declared rows into runs that share an arm, in declaration order.

    The ONLY safe partition: `_disarm_branches` fires at each arm change, so
    row indices may be reordered *within* a run and never across one. Returns
    index lists so the caller reassembles by position and completion order
    stays unobservable in the artifact.
    """
    out: List[List[int]] = []
    last = object()
    for i, row in enumerate(rows):
        arm = row[1] if isinstance(row, (tuple, list)) and len(row) > 1 else None
        if arm != last:
            out.append([])
            last = arm
        out[-1].append(i)
    return out


def width_for(
    group_size: int, *, free_device_bytes: int, per_export_device_bytes: int,
    cpu_workers: int, ceiling: int = 8,
) -> Dict[str, Any]:
    """K for ONE arm's export, from the EXPORT footprint — never the compile one.

    Refuses to license concurrency it cannot justify: an unmeasured or absent
    per-export footprint yields 1, because the failure mode of guessing here is
    an OOM that kills a 74-minute phase.
    """
    if group_size < MIN_GROUP:
        return {"workers": 1, "binding": "group-too-small"}
    if per_export_device_bytes <= 0:
        return {"workers": 1, "binding": "export-footprint-unmeasured"}
    if free_device_bytes <= 0:
        return {"workers": 1, "binding": "no-device-reading"}
    device_workers = max(1, int(free_device_bytes // per_export_device_bytes))
    workers = max(1, min(device_workers, max(1, cpu_workers), ceiling,
                         group_size))
    binding = min((device_workers, "vram"), (cpu_workers, "cpu"),
                  (ceiling, "ceiling"), (group_size, "group"))[1]
    return {"workers": workers, "binding": binding,
            "device_workers": device_workers, "cpu_workers": cpu_workers,
            "per_export_device_bytes": int(per_export_device_bytes)}


__all__ = ["ENV_FLAG", "MIN_GROUP", "decide", "enabled", "groups",
           "width_for"]


def decide(
    rows: Sequence[Any], timings: Dict[str, Any], *,
    free_device_bytes: int = -1, cpu_workers: int = 0,
) -> Dict[str, float]:
    """The CALL SITE's decision, as flat telemetry. Emitted on EVERY mint.

    pgw#868 A4: `timings["export_peak_device_bytes"]` (shipped 0.90.6) is
    exactly :func:`width_for`'s ``per_export_device_bytes``. Producer and
    consumer were both built and never joined — so the flag was inert and
    "shipped OFF behind a flag" described a seam that had never been cut.
    This is the join, and it records the decision **whether or not the flag is
    on**, because the decision is the OBSERVABLE: a reader sees the width the
    export phase would have run at and which fact bound it, on a mint that
    changed nothing. That is what makes turning it on a measurement rather than
    a leap.
    """
    if cpu_workers <= 0 or free_device_bytes < 0:
        from . import aot_compile_pool

        try:
            if cpu_workers <= 0:
                cpu_workers = max(1, aot_compile_pool.cpu_facts().vcpus // 2)
            if free_device_bytes < 0:
                free_device_bytes = aot_compile_pool.device_facts().free_bytes
        except Exception:  # noqa: BLE001 — telemetry never fails a mint
            cpu_workers = max(1, cpu_workers)
            free_device_bytes = max(0, free_device_bytes)

    per_export = int(float(timings.get("export_peak_device_bytes") or 0))
    runs = groups(rows)
    largest = max((len(g) for g in runs), default=0)
    width = width_for(
        largest, free_device_bytes=int(free_device_bytes),
        per_export_device_bytes=per_export, cpu_workers=int(cpu_workers))
    return {
        "export_parallel_enabled": 1.0 if enabled() else 0.0,
        "export_parallel_groups": float(len(runs)),
        "export_parallel_largest_group": float(largest),
        "export_parallel_width": float(width["workers"]),
        "export_parallel_binding": float(
            {"vram": 1, "cpu": 2, "ceiling": 3, "group": 4,
             "export-footprint-unmeasured": 5, "no-device-reading": 6,
             "group-too-small": 7}.get(str(width["binding"]), 0)),
        "export_parallel_per_export_bytes": float(per_export),
    }
