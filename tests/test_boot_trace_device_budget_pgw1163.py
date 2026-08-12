"""pgw#1163: the boot trace's fan-out gets a DEVICE budget.

Measured on a live pod, not inferred. RTX 3090 `q1y1tx4fa1vjzs` (24 GB, sm_86),
release 6ee9b4d4df2697a53da6f43a, gen-worker 0.112.0, **uncontended** — a fresh
pod, one request, nothing else on the card:

    boot_adopt child_error key=- — 18 of 18 boot-trace child(ren) produced no
    class hashes: MintResourceExhausted: torch.export(strict=True) for
    UNet2DConditionModel: OUT OF DEVICE MEMORY
    (GPU 0 has a total capacity of 23.56 GiB of which 3.12 MiB is free.
     Process 783428 has 578.00 MiB memory in use.
     Process 794409 has 1.29 GiB   Process 794397 has 1.29 GiB
     Process 794400 has 1.29 GiB   Process 794401 has 1.29 GiB
     Process 794403 has 1.29 GiB   Process 794405 has 1.25 GiB
     Process 794404 has 1.26 GiB   …)

Those processes are the trace's own children. `trace_workers` sized the pool
from CPU alone — the pod reported `host_cpu_count=96` — so K was capped only by
the declared class count, 18, and 18 CUDA contexts went onto a 24 GB card. The
same derivation on a 48 GB A40 succeeded twice with a byte-identical key, which
is why this looked like "small cards cannot derive keys" rather than a pool
with no budget.

Note what the per-child cost IS: one structure-only export's *allocator*
high-water really is ~9.8 MiB (pgw#1080, and `trace_workers` said so). The
1.25-1.29 GiB is the PROCESS — a CUDA context plus the cuBLAS/cuDNN kernel
images the export loads. It is not going to shrink, which is exactly why the
answer is a budget rather than a diet.
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
    assert "affordable" in why


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

    w = boot_key.trace_workers(SDXL_CLASSES)
    assert w.workers == SDXL_CLASSES and w.binding == "classes"


def test_an_unmeasured_card_keeps_EXACTLY_the_old_width() -> None:
    """An absent measurement must never throttle a pod. Zero is 'no evidence',
    not 'no memory' — the direction that fails safe is the old behaviour."""
    width, why = boot_key.concurrency_budget(
        SDXL_CLASSES, free_bytes=0, per_child_bytes=0)
    assert width == SDXL_CLASSES and "unmeasured" in why

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
    is the sharding, and pgw#1163 does not call it differently."""
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
