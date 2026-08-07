"""pgw#868 A4 / pgw#1000: how wide the export phase may run, decided in code.

The phase this is about
-----------------------
``aot_mint._mint_cell`` exports one declared class row at a time, serially, in
the mint parent. MEASURED on the only complete 36-entry sdxl AOT mint that has
ever finished (attempt 26, L40S)::

    export_s 3891.82   compile_s 9153.12   total_s 9546.65

**65 minutes of a 2 h 43 m mint, and pgw#809's K-wide compile pool divides
none of it.** Per row that is 108 s, which agrees with the 125-134 s the
z-image lane measured on an A100. It is the single largest serial term left.

Why this module shipped dark for three releases, and what actually fixed it
--------------------------------------------------------------------------
It was gated on ``GEN_WORKER_AOT_EXPORT_PARALLEL``, default OFF, with the
stated reason: *"width is bounded by the EXPORT-phase footprint, which nobody
has measured... until then this refuses to guess and stays OFF."*

The footprint was not unmeasured. It was measured WRONG, and nothing said so.
``timings["export_peak_device_bytes"]`` is ``max_memory_allocated`` over the
ENTIRE serial loop against a resident pipeline — 15.4 GiB on attempt 26 — so
:func:`width_for` was asked "how many 15.4 GiB workers fit beside a resident
mint child" and correctly answered **1**, on every card, forever. A flag
guarding a computation that cannot return anything but 1 is not a safety
mechanism; it is a feature that never ran.

pgw#1000 measures the right quantity — :class:`aot_mint._ExportFootprint`, one
row's DELTA over the resident baseline, which is what a worker adds to a card
that already holds the module it traces — and the gate is gone. **The width
decision lives here, in code.** Width 1 is the natural floor when the budget
says so, and it says so through named terms, not a flag.

The budget is pgw#992's, unchanged
----------------------------------
An export worker is priced exactly as an entry-compile child is priced, by the
same :class:`~gen_worker.aot_compile_pool.CardCensus`::

    budget = total - resident co-tenant - own device high-water - tenant reserve

That bound exists because dividing a momentary free-VRAM SAMPLE killed the
first AOT mint ever to reach the compile phase (pgw#992). Export has no claim
to a weaker rule than compile: it runs on the same card, beside the same
residents, in the same mint.

The partition, and why it is the only safe one
----------------------------------------------
``_disarm_branches`` mutates the module in place, exactly ONCE, between the
adapter-bearing rows and the branchless ones. Rows *within* an arm mutate
nothing, so sdxl's 36 rows are two internally order-independent groups of 18
and the serialisation across those 18 was inherited from where the model
lives, not required by correctness. :func:`groups` is that partition and
nothing may reorder across it.

Processes, not threads: the bottleneck is the GIL, and ``fork`` is banned
after CUDA init (pgw#784). Each worker is a fresh interpreter holding its own
module copy — which is exactly why the width is VRAM-bounded and why the
divisor above had to be real before an executor could be built against it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Below this a group cannot repay a worker's own module load.
MIN_GROUP = 3

#: Hard ceiling regardless of budget, matching the compile pool's
#: (:data:`~gen_worker.aot_compile_pool.MAX_ENTRY_WORKERS`). Past it the shared
#: inductor cache and the disk holding N saved programs stop behaving, and the
#: remaining serial terms dominate anyway.
CEILING = 8


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
    group_size: int, *, budget_bytes: int, per_export_device_bytes: int,
    cpu_workers: int, ceiling: int = CEILING,
) -> Dict[str, Any]:
    """K for ONE arm's export, from the EXPORT footprint and pgw#992's budget.

    ``budget_bytes`` is what the CARD can hold at once beside its residents —
    never a free-VRAM sample. Refuses to license concurrency it cannot justify:
    an unmeasured footprint or an unreadable budget yields 1, because the
    failure mode of guessing here is an OOM that kills a 65-minute phase.
    """
    if group_size < MIN_GROUP:
        return {"workers": 1, "binding": "group-too-small"}
    if per_export_device_bytes <= 0:
        return {"workers": 1, "binding": "export-footprint-unmeasured"}
    if budget_bytes <= 0:
        return {"workers": 1, "binding": "no-card-budget"}
    device_workers = max(1, int(budget_bytes // per_export_device_bytes))
    workers = max(1, min(device_workers, max(1, cpu_workers), ceiling,
                         group_size))
    binding = min((device_workers, "vram"), (cpu_workers, "cpu"),
                  (ceiling, "ceiling"), (group_size, "group"))[1]
    return {"workers": workers, "binding": binding,
            "device_workers": device_workers, "cpu_workers": cpu_workers,
            "budget_bytes": int(budget_bytes),
            "per_export_device_bytes": int(per_export_device_bytes)}


def decide(
    rows: Sequence[Any], timings: Dict[str, Any], *,
    budget_bytes: int = -1, cpu_workers: int = 0,
    census: Optional[Any] = None,
) -> Dict[str, float]:
    """The width the export phase runs at, as flat telemetry.

    Emitted on EVERY mint. When ``budget_bytes`` is not supplied it is derived
    from pgw#992's census exactly as the compile pool derives it — same terms,
    same card, same residents — so the two phases can never disagree about how
    much room exists.
    """
    from . import aot_compile_pool

    if cpu_workers <= 0:
        try:
            cpu_workers = max(1, aot_compile_pool.cpu_facts().vcpus // 2)
        except Exception:  # noqa: BLE001 — telemetry never fails a mint
            cpu_workers = 1
    if budget_bytes < 0:
        budget_bytes = _card_budget(
            census if census is not None else aot_compile_pool.card_census())

    per_export = int(float(timings.get("per_export_device_bytes") or 0))
    runs = groups(rows)
    largest = max((len(g) for g in runs), default=0)
    width = width_for(
        largest, budget_bytes=int(budget_bytes),
        per_export_device_bytes=per_export, cpu_workers=int(cpu_workers))
    return {
        "export_parallel_groups": float(len(runs)),
        "export_parallel_largest_group": float(largest),
        "export_parallel_width": float(width["workers"]),
        "export_parallel_binding": float(
            {"vram": 1, "cpu": 2, "ceiling": 3, "group": 4,
             "export-footprint-unmeasured": 5, "no-card-budget": 6,
             "group-too-small": 7}.get(str(width["binding"]), 0)),
        "export_parallel_per_export_bytes": float(per_export),
        "export_parallel_budget_bytes": float(max(0, budget_bytes)),
    }


def _card_budget(census: Any) -> int:
    """pgw#992's budget, for export workers instead of entry children.

    ``total - resident co-tenant - own device high-water``. The tenant reserve
    is the compile pool's to apply against a serve goal; export runs inside the
    same mint child, so the two must not each subtract it — the pool re-derives
    the whole budget for itself when it builds.
    """
    from . import aot_compile_pool

    try:
        if not census.readable:
            return -1
        own = max(aot_compile_pool.own_device_high_water(),
                  int(census.own_reserved_bytes))
        return int(census.total_bytes) - int(census.resident_other_bytes) - own
    except Exception:  # noqa: BLE001 — telemetry never fails a mint
        return -1


__all__ = ["CEILING", "MIN_GROUP", "decide", "groups", "width_for"]
