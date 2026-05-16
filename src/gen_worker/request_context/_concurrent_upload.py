"""Concurrent multi-file upload coordinator (issue #269, refactored #13).

Single source of truth for the worker-side file-level upload fan-out.
Loops that previously did ``for f in files: ctx.save_checkpoint(f)``
should swap in ``parallel_map_uploads(items, upload_fn)`` to pipeline
disk read + BLAKE3 hash + multipart PUT across files instead of
serializing them.

Library-internal. ``_``-prefixed module name. Don't import from tenant
code.

# Adaptive concurrency (issue #11/#13)

The former fixed file fan-out constant is gone. ``parallel_map_uploads``
now owns a per-call adaptive scheduler:

  - start with a small number of in-flight files
  - record elapsed time + best-effort item size as each file finishes
  - ramp while recent throughput is improving
  - hold when it plateaus
  - back off immediately on upload errors

``BudgetGate`` (below) is the authoritative byte-budget back-pressure
and composes with the internal concurrency policy; nothing in the
adaptive scheduler overrides the byte budget.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

_ADAPTIVE_START_WORKERS = 2
_ADAPTIVE_MAX_WORKERS = 16
_ADAPTIVE_WINDOW = 4
_ADAPTIVE_IMPROVEMENT_RATIO = 1.05
_ADAPTIVE_DROP_RATIO = 0.90


@dataclass(frozen=True)
class _UploadSample:
    size_bytes: int
    elapsed_s: float

    @property
    def throughput(self) -> float:
        # Unknown sizes use a synthetic one-byte unit so the controller can
        # still compare relative task completion rates.
        size = max(1, int(self.size_bytes))
        elapsed = max(float(self.elapsed_s), 1e-6)
        return float(size) / elapsed


class _AdaptiveUploadController:
    """Per-call file fan-out controller.

    This is deliberately local to one ``parallel_map_uploads`` call. No
    global state leaks between jobs, and there is no operator-facing knob.
    """

    def __init__(self, item_count: int) -> None:
        count = max(1, int(item_count))
        self._ceiling = min(count, _ADAPTIVE_MAX_WORKERS)
        self._target = min(_ADAPTIVE_START_WORKERS, self._ceiling)
        self._samples: list[_UploadSample] = []

    @property
    def target(self) -> int:
        return max(1, min(self._target, self._ceiling))

    @property
    def ceiling(self) -> int:
        return self._ceiling

    def observe_success(self, *, size_bytes: int, elapsed_s: float) -> None:
        self._samples.append(_UploadSample(size_bytes=size_bytes, elapsed_s=elapsed_s))
        if self._target >= self._ceiling:
            return

        # Warm-up: ramp quickly from the conservative start until we have a
        # meaningful window to compare.
        if len(self._samples) < (_ADAPTIVE_WINDOW * 2):
            self._target += 1
            return

        recent = self._avg_throughput(self._samples[-_ADAPTIVE_WINDOW:])
        previous = self._avg_throughput(self._samples[-(_ADAPTIVE_WINDOW * 2):-_ADAPTIVE_WINDOW])
        if previous <= 0:
            self._target += 1
            return
        ratio = recent / previous
        if ratio >= _ADAPTIVE_IMPROVEMENT_RATIO:
            self._target += 1
        elif ratio < _ADAPTIVE_DROP_RATIO and self._target > 1:
            self._target -= 1

    def observe_failure(self) -> None:
        if self._target > 1:
            self._target -= 1

    @staticmethod
    def _avg_throughput(samples: Sequence[_UploadSample]) -> float:
        if not samples:
            return 0.0
        return sum(s.throughput for s in samples) / float(len(samples))


def optimal_file_concurrency(item_count: int) -> int:
    """Return the internal upper bound for one file-level upload batch."""
    if item_count <= 1:
        return 1
    return min(int(item_count), _ADAPTIVE_MAX_WORKERS)


def _estimate_item_size_bytes(item: Any) -> int:
    """Best-effort size estimator used only for adaptive scheduling.

    Common upload callers pass paths, ``Path`` objects, or tuples/lists
    containing a path. Unknown shapes return 0 and still participate in the
    controller using completion-rate units.
    """
    if isinstance(item, (str, bytes, os.PathLike)):
        try:
            return int(os.path.getsize(os.fspath(item)))
        except OSError:
            return 0
    if isinstance(item, (tuple, list)):
        for value in item:
            size = _estimate_item_size_bytes(value)
            if size > 0:
                return size
    path = getattr(item, "path", None) or getattr(item, "local_path", None)
    if path is not None:
        return _estimate_item_size_bytes(path)
    return 0


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

    The adaptive scheduler may run several files in parallel. Tensorhub's
    worker_capability_token caps aggregate in-flight bytes per session via
    ``max_total_bytes`` and per-file via ``max_bytes_per_file``. Without
    back-pressure, multiple 30 GiB shards could exceed a typical 50 GiB
    total budget; the server would reject them mid-upload, leaving partial
    parts.

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
    label: str = "upload",
) -> List[R]:
    """Run ``upload_fn`` over ``items`` with bounded concurrency, preserving order.

    Each callable owns one file end-to-end (hash → presigned create →
    multipart PUT → /complete). The pool starts conservatively and adjusts
    internal fan-out as files complete.

    Results are returned in the SAME order as ``items``. If any worker
    raises, the first exception is re-raised after the remaining
    in-flight uploads finish. No new uploads are scheduled after the first
    failure.

    The S3 multipart PUTs inside each upload already run their own
    inner part-level thread pool; this fan-out is at the FILE level on
    top of that.
    """
    n = len(items)
    if n == 0:
        return []
    if n == 1:
        return [upload_fn(it) for it in items]

    controller = _AdaptiveUploadController(n)
    max_workers = controller.ceiling
    results: List[Any] = [None] * n
    first_exc: BaseException | None = None
    next_idx = 0
    futures: dict[Future[R], tuple[int, float, int]] = {}

    def _schedule_available(pool: ThreadPoolExecutor) -> None:
        nonlocal next_idx
        while first_exc is None and next_idx < n and len(futures) < controller.target:
            item = items[next_idx]
            fut = pool.submit(upload_fn, item)
            futures[fut] = (next_idx, time.monotonic(), _estimate_item_size_bytes(item))
            next_idx += 1

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"gw-{label}") as pool:
        _schedule_available(pool)
        while futures:
            done, _pending = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                idx, started, size_bytes = futures.pop(fut)
                elapsed_s = time.monotonic() - started
                try:
                    results[idx] = fut.result()
                except BaseException as exc:  # noqa: BLE001
                    controller.observe_failure()
                    if first_exc is None:
                        first_exc = exc
                    logger.warning(
                        "%s_concurrent_item_failed idx=%d target_workers=%d err=%r",
                        label, idx, controller.target, exc,
                    )
                else:
                    controller.observe_success(size_bytes=size_bytes, elapsed_s=elapsed_s)
                    logger.debug(
                        "%s_concurrent_item_done idx=%d size_bytes=%d elapsed_s=%.3f target_workers=%d",
                        label, idx, size_bytes, elapsed_s, controller.target,
                    )
            _schedule_available(pool)
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
