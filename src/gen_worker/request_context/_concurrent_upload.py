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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Hardcoded; do NOT plumb to an env var or config knob. See module
# docstring for the trade-off rationale.
MAX_CONCURRENT_UPLOADS = 4


class BudgetExceededError(RuntimeError):
    """Raised when a single file exceeds the per-file or total byte budget.

    Per-request budgets come from the worker_capability_token's
    ``max_total_bytes`` + ``max_bytes_per_file`` claims (issued by tensorhub).
    The pool's per-file fan-out can over-commit if multiple very large
    shards run in parallel — this gate back-pressures starts until in-flight
    bytes fit the aggregate budget.
    """


class BudgetGate:
    """Capability-budget back-pressure for the concurrent upload pool (issue #269).

    The pool runs ``MAX_CONCURRENT_UPLOADS=4`` files in parallel. Tensorhub's
    worker_capability_token caps aggregate in-flight bytes per session via
    ``max_total_bytes`` and per-file via ``max_bytes_per_file``. Without
    back-pressure, four 30 GiB shards would exceed a typical 50 GiB total
    budget; the server would reject them mid-upload, leaving partial parts.

    The gate sits between the pool and the upload entry points
    (``save_file``, ``save_checkpoint``, ``save_file_create``). Each entry
    point computes ``size = os.path.getsize(src)`` and wraps its upload
    work in ``with gate.reserve(size): ...``.

    Reentrancy: ``save_checkpoint``'s non-streaming branch calls
    ``save_file`` internally. The reservation is thread-scoped — a nested
    ``reserve()`` from the same thread is a no-op so the outer reservation
    isn't double-counted.

    Unbounded axis: when ``max_total_bytes <= 0`` or ``max_bytes_per_file <= 0``,
    that axis is treated as unbounded. A gate constructed with both at
    zero is a pure pass-through (no blocking, no rejection).
    """

    def __init__(self, max_total_bytes: int = 0, max_bytes_per_file: int = 0) -> None:
        self._max_total_bytes = int(max_total_bytes) if int(max_total_bytes) > 0 else 0
        self._max_bytes_per_file = int(max_bytes_per_file) if int(max_bytes_per_file) > 0 else 0
        self._inflight = 0
        self._cond = threading.Condition()
        self._tls = threading.local()

    @property
    def max_total_bytes(self) -> int:
        return self._max_total_bytes

    @property
    def max_bytes_per_file(self) -> int:
        return self._max_bytes_per_file

    @property
    def inflight_bytes(self) -> int:
        with self._cond:
            return self._inflight

    def reserve(self, size_bytes: int) -> "_BudgetReservation":
        return _BudgetReservation(self, int(size_bytes))


class _BudgetReservation:
    __slots__ = ("_gate", "_size", "_held")

    def __init__(self, gate: BudgetGate, size_bytes: int) -> None:
        self._gate = gate
        self._size = size_bytes
        self._held = False

    def __enter__(self) -> "_BudgetReservation":
        gate = self._gate
        size = self._size
        # Per-file ceiling — fail fast before any reservation work.
        if gate._max_bytes_per_file > 0 and size > gate._max_bytes_per_file:
            raise BudgetExceededError(
                f"file size {size} exceeds capability max_bytes_per_file {gate._max_bytes_per_file}"
            )
        # Reentrancy guard: nested reserve() from the same thread is a no-op.
        # save_checkpoint -> save_file pattern would otherwise double-count.
        depth = getattr(gate._tls, "depth", 0)
        gate._tls.depth = depth + 1
        if depth > 0:
            return self
        # Acquire aggregate-bytes budget. Unbounded axis (== 0) skips the wait.
        if gate._max_total_bytes > 0:
            with gate._cond:
                while gate._inflight + size > gate._max_total_bytes:
                    # If no other reservation is in flight, this single file is
                    # larger than the total budget — fail rather than deadlock.
                    if gate._inflight == 0:
                        gate._tls.depth = depth  # release the depth bump
                        raise BudgetExceededError(
                            f"file size {size} > capability max_total_bytes {gate._max_total_bytes}"
                        )
                    gate._cond.wait()
                gate._inflight += size
        self._held = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        gate = self._gate
        depth = getattr(gate._tls, "depth", 0)
        if depth > 0:
            gate._tls.depth = depth - 1
        # Only the outermost reservation releases bytes.
        if not self._held:
            return False
        if gate._max_total_bytes > 0:
            with gate._cond:
                gate._inflight = max(0, gate._inflight - self._size)
                gate._cond.notify_all()
        self._held = False
        return False


def budget_gate_from_capability_jwt(token: str) -> BudgetGate:
    """Construct a BudgetGate from a worker_capability_token's budget claims.

    Reads ``max_total_bytes`` and ``max_bytes_per_file`` from the JWT
    payload (unverified — these are advisory caps from the issuer; the
    server enforces them authoritatively). Missing/zero claims yield an
    unbounded gate on that axis.

    Returns a pass-through gate (both axes 0) when the token is missing,
    malformed, or has neither claim set — back-pressure becomes a no-op
    in dev/test paths without a server-issued JWT.
    """
    # Import locally to avoid a module-level circular dep on _helpers.
    from gen_worker.request_context._helpers import _decode_unverified_jwt_claims

    claims = _decode_unverified_jwt_claims(token) if token else {}

    def _int_claim(key: str) -> int:
        raw = claims.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    return BudgetGate(
        max_total_bytes=_int_claim("max_total_bytes"),
        max_bytes_per_file=_int_claim("max_bytes_per_file"),
    )

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
