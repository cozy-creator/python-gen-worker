"""Concurrent multi-file upload coordinator (issue #269).

Single source of truth for the worker-side upload fan-out constant and
the small ThreadPoolExecutor helper that wraps it. Loops that previously
did ``for f in files: ctx.save_checkpoint(f)`` should swap in
``parallel_save_checkpoints(ctx, files, upload_fn)`` to get up to
``MAX_CONCURRENT_UPLOADS`` files running their hash-read-PUT-complete
cycle concurrently.

Library-internal. ``_``-prefixed module name. Don't import from tenant
code.

# Why a constant, not a knob

Workers are tenant-written code; we don't expose tuning surface to
tenants. The trade-off is fixed:

  - 1  → serial, easy to reason about, ~113 MB/s on a 10 GbE host
         saturating BLAKE3 single-threaded.
  - 4  → 4× disk readers + 4× HTTP clients in flight; saturates the
         BLAKE3 fan-out (each instance also fans out internally via
         ``max_threads=blake3.AUTO``) without thrashing the read-ahead
         page cache. Empirically lands near wire saturation on a
         single-NIC host with NVMe storage.
  - 8+ → diminishing returns; competes for BLAKE3 worker threads,
         starts to thrash the page cache, and the marginal MB/s gain
         is small relative to the CPU contention cost.

Flip to ``1`` in source to reproduce the prior serial behavior for
debugging.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Hardcoded; do NOT plumb to an env var or config knob. See module
# docstring for the trade-off rationale.
MAX_CONCURRENT_UPLOADS = 4

T = TypeVar("T")
R = TypeVar("R")


def parallel_map_uploads(
    items: Sequence[T],
    upload_fn: Callable[[T], R],
    *,
    max_workers: int = MAX_CONCURRENT_UPLOADS,
    label: str = "upload",
) -> List[R]:
    """Run ``upload_fn`` over ``items`` with bounded concurrency, preserving order.

    Each callable owns one file end-to-end (hash → presigned create →
    multipart PUT → /complete). The pool is sized at ``max_workers``
    (default ``MAX_CONCURRENT_UPLOADS``) — if ``items`` is shorter, only
    that many workers spin up.

    Results are returned in the SAME order as ``items``. If any worker
    raises, the first exception is re-raised after the remaining
    in-flight uploads finish (best-effort drain, no aggressive cancel —
    we don't want a half-cancelled multipart leaking partial parts).

    The S3 multipart PUTs inside each upload already run their own
    inner part-level thread pool; this fan-out is at the FILE level on
    top of that.
    """
    n = len(items)
    if n == 0:
        return []
    # Serial fast path keeps debug behavior 1:1 with the pre-#269 code.
    if max_workers <= 1 or n == 1:
        return [upload_fn(it) for it in items]

    workers = min(max_workers, n)
    results: List[Any] = [None] * n
    first_exc: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"gw-{label}") as pool:
        futures = {pool.submit(upload_fn, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except BaseException as exc:  # noqa: BLE001
                if first_exc is None:
                    first_exc = exc
                logger.warning(
                    "%s_concurrent_item_failed idx=%d err=%r",
                    label, idx, exc,
                )
    if first_exc is not None:
        raise first_exc
    return results


def parallel_save_checkpoints(
    ctx: Any,
    items: Iterable[Tuple[str, str, str]],
    *,
    extra_kwargs_for_index: Callable[[int], dict] | None = None,
) -> List[Any]:
    """Bulk wrapper around ``ctx.save_checkpoint``.

    ``items`` is an iterable of ``(ref, local_path, format)`` tuples.
    Returns the per-item Tensors in input order.

    For per-item lineage metadata (step/epoch/flavor/etc.), pass a
    callable via ``extra_kwargs_for_index`` that yields the kwargs map
    for index i. Most callers in the clone pipeline don't need lineage
    on each shard; the trainer's checkpoint-emission path is the only
    one that uses per-item lineage today, and it uploads one file at a
    time (no fan-out opportunity), so it stays on the direct API.
    """
    materialized = list(items)

    def _one(idx_item: Tuple[int, Tuple[str, str, str]]) -> Any:
        i, (ref, local_path, fmt) = idx_item
        extra: dict = {}
        if extra_kwargs_for_index is not None:
            extra = dict(extra_kwargs_for_index(i) or {})
        return ctx.save_checkpoint(ref, local_path, format=fmt, **extra)

    indexed: List[Tuple[int, Tuple[str, str, str]]] = list(enumerate(materialized))
    return parallel_map_uploads(indexed, _one, label="save-checkpoint")
