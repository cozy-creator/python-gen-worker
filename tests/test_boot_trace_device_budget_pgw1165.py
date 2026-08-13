"""pgw#1165: the boot trace's fan-out gets a DEVICE budget.

Sizing `trace_workers` from CPU alone (a pod reporting `host_cpu_count=96`) caps
K only by the declared class count, so 18 boot-trace children put 18 CUDA
contexts onto a 24 GB card and all 18 die `OUT OF DEVICE MEMORY` on an
uncontended pod. The same derivation succeeds on a 48 GB A40, which is why it
reads as "small cards cannot derive keys" rather than a pool with no budget.

What the per-child cost IS: one structure-only export's *allocator* high-water
is ~9.8 MiB, but the PROCESS is 1.25-1.29 GiB — a CUDA context plus the
cuBLAS/cuDNN kernel images the export loads. It is not going to shrink, which is
exactly why the answer is a budget rather than a diet.
"""
from __future__ import annotations

from gen_worker import boot_key


# The measured constants of the incident, used as INPUTS to the pure functions
# under test — never as thresholds inside them.
GIB = 1024 ** 3
CHILD_BYTES = int(1.29 * GIB)      # measured, RTX 3090
CARD_3090_FREE = int(22.9 * GIB)   # 23.56 total minus the parent's 578 MiB
CARD_A40_FREE = int(46.8 * GIB)
CARD_A4000_FREE = int(14.5 * GIB)
SDXL_CLASSES = 18


def test_the_3090_incident_no_longer_puts_18_contexts_on_the_card() -> None:
    """The whole point. 18 x 1.29 GiB = 23.2 GiB against 22.9 GiB free is the
    card ending at 3.12 MiB free, which is what happened."""
    width, why = boot_key.concurrency_budget(
        SDXL_CLASSES, free_bytes=CARD_3090_FREE, per_child_bytes=CHILD_BYTES)
    assert width * CHILD_BYTES <= CARD_3090_FREE, (
        f"admitted {width} children at 1.29 GiB onto {CARD_3090_FREE / GIB:.2f} "
        f"GiB — that is the incident, verbatim")
    assert "affordable" in why and "basis=measured" in why


def test_a_16gb_a4000_derives_instead_of_refusing() -> None:
    """The acceptance shape: the card that FAILED must now trace, in waves,
    rather than refuse 18 of 18."""
    width, _ = boot_key.concurrency_budget(
        SDXL_CLASSES, free_bytes=CARD_A4000_FREE, per_child_bytes=CHILD_BYTES)
    assert width >= 1
    assert width * CHILD_BYTES <= CARD_A4000_FREE
    # ...and it must not have become a serial crawl to get there.
    assert width >= 8, (
        f"W={width} on a card with room for {CARD_A4000_FREE // CHILD_BYTES} — "
        f"a budget that collapses is the c9fb5d4a bug with better manners")


def test_a_card_with_room_keeps_its_FULL_width() -> None:
    """THE anti-collapse assertion. `c9fb5d4a`: the mint's entry pool ran K=1
    fleet-wide for weeks, cost every mint 2.4x, and survived because no test
    asserted the achieved width. This is that test for the trace pool."""
    width, _ = boot_key.concurrency_budget(
        SDXL_CLASSES, free_bytes=CARD_A40_FREE, per_child_bytes=CHILD_BYTES)
    assert width == SDXL_CLASSES, (
        f"W={width} on a 48 GB card that fits all {SDXL_CLASSES} — the budget "
        f"must cost NOTHING where the card affords the fan-out")

    # `trace_workers` decides SHARDING and is CPU-bound, so its width depends on
    # the machine (CI runs on 4 cores and correctly returns 3). What must hold
    # everywhere is that the device budget cannot be what narrowed it — the
    # bound was deliberately removed from this function, so `vram` is
    # unreachable here and a regression that re-adds it shows up as this
    # binding appearing.
    w = boot_key.trace_workers(SDXL_CLASSES)
    assert 1 <= w.workers <= SDXL_CLASSES
    assert w.binding in ("classes", "cpu"), (
        f"binding={w.binding!r}: sharding must never be narrowed by the device "
        f"budget — that is concurrency_budget's job, one layer down")


def test_an_unmeasured_card_keeps_EXACTLY_the_old_width() -> None:
    """An absent measurement must never throttle a pod. Zero is 'no evidence',
    not 'no memory' — the direction that fails safe is the old behaviour."""
    width, why = boot_key.concurrency_budget(
        SDXL_CLASSES, free_bytes=0, per_child_bytes=0)
    assert width == SDXL_CLASSES and "basis=unmeasured" in why

    for free, per in ((CARD_3090_FREE, 0), (0, CHILD_BYTES)):
        w, _ = boot_key.concurrency_budget(
            SDXL_CLASSES, free_bytes=free, per_child_bytes=per)
        assert w == SDXL_CLASSES, "a half-measurement is not a measurement"

    w = boot_key.trace_workers(SDXL_CLASSES)
    assert w.workers >= 1 and w.binding in ("classes", "cpu")


def test_the_budget_never_reaches_zero() -> None:
    """A pool of zero derives no key at all. A card that cannot hold ONE child
    is a placement fact for the caller to report, not a width to floor away."""
    width, _ = boot_key.concurrency_budget(
        SDXL_CLASSES, free_bytes=int(0.1 * GIB), per_child_bytes=CHILD_BYTES)
    assert width == 1


def test_the_binding_is_NAMED_so_a_regression_is_assertable() -> None:
    """Prose is not assertable; `binding` is. Each bound must be reachable and
    must say which one it was."""
    assert boot_key.trace_workers(4).binding in ("classes", "cpu")
    assert boot_key.trace_workers(SDXL_CLASSES, limit=2).binding == "cap"
    # The DEVICE bound is not a sharding concern and must not appear here —
    # an unwired parameter is the dead-code defect this repo keeps paying for.
    import inspect
    assert "device" not in str(inspect.signature(boot_key.trace_workers))


def test_sharding_is_untouched_so_the_KEY_cannot_move() -> None:
    """The property that lets this ship without re-keying anything: the budget
    bounds CONCURRENCY, never how the declared classes are divided. `shares`
    is the sharding, and pgw#1165 does not call it differently."""
    for k in (1, 3, 18):
        got = boot_key.shares(SDXL_CLASSES, k)
        assert [i for i, _ in got] == list(range(min(k, SDXL_CLASSES)))
        assert all(c == k for _, c in got)


def test_the_child_reports_its_whole_process_footprint() -> None:
    """`device_peak_bytes` must be the PROCESS cost, not the allocator's view —
    the 1.29 GiB that broke the card is context plus kernel images, and
    `max_memory_allocated` would have reported ~9.8 MiB of it."""
    import inspect

    from gen_worker import boot_trace_child

    src = inspect.getsource(boot_trace_child._device_peak_bytes)
    # The CODE, not the prose: the docstring names the rejected alternative on
    # purpose, and a substring check over the whole source would read that as
    # the thing it forbids.
    body = src.replace(boot_trace_child._device_peak_bytes.__doc__ or "", "")
    assert "mem_get_info" in body, (
        "the budget's input must be total-minus-free on the card; an allocator "
        "high-water misses the context, which IS the cost")
    assert "max_memory_allocated" not in body

    assert "device_peak_bytes" in boot_key.TraceReport.__struct_fields__
    assert boot_key.TraceReport().device_peak_bytes == 0


def test_the_budget_is_sized_on_the_PROCESS_not_the_artifact() -> None:
    """th#1825's lane ruled out per-entry literals as the dominant term by three
    independent bounds — they live inside the artifact, measured at 4.19 MB —
    and found the real cost is the loaded AOTI packages plus device code,
    per-runner workspace and load-time buffers, none of which appear in the
    artifact. A budget sized on artifact or literal bytes under-counts in the
    direction that OOMs, so the input stays `total - free`."""
    import inspect

    from gen_worker import boot_key as bk

    body = inspect.getsource(bk.free_device_bytes)
    assert "mem_get_info" in body
    for wrong in ("artifact", "literal", "max_memory_allocated", "getsize"):
        assert wrong not in body.split('"""')[-1], (
            f"{wrong!r} in the sizing input — it cannot see the loaded packages")
