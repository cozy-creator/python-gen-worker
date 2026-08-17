"""Capability byte-budget back-pressure for worker-side uploads.

There is NO file-level upload parallelism: files are uploaded serially and the
fan-out lives *inside* one file (S3 parts in ``presigned_upload.py``, CAS chunks
in tensorfs), each with its own bounded, process-wide PUT
budget.

What lives here is ``BudgetGate``: the ``max_total_bytes`` /
``max_bytes_per_file`` back-pressure derived from the worker capability token.
Library-internal, ``_``-prefixed module name — don't import from tenant code.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from gen_worker.request_context._helpers import _decode_unverified_jwt_claims

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when a single file exceeds the per-file or total byte budget.

    Per-request budgets come from the worker_capability_token's
    ``max_total_bytes`` + ``max_bytes_per_file`` claims (issued by tensorhub).
    The pool's per-file fan-out can over-commit if multiple very large
    shards run in parallel — this gate back-pressures starts until in-flight
    bytes fit the aggregate budget.
    """


class BudgetGate:
    """Capability-budget back-pressure for the concurrent upload pool.

    The upload coordinator may run several files in parallel. Tensorhub's
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

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        gate = self._gate
        depth = getattr(gate._tls, "depth", 0)
        if depth > 0:
            gate._tls.depth = depth - 1
        # Only the outermost reservation releases bytes.
        if not self._held:
            return
        if gate._max_total_bytes > 0:
            with gate._cond:
                gate._inflight = max(0, gate._inflight - self._size)
                gate._cond.notify_all()
        self._held = False


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

    claims = _decode_unverified_jwt_claims(token) if token else {}

    def _int_claim(key: str) -> int:
        raw = claims.get(key)
        try:
            value = int(raw or 0)
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    return BudgetGate(
        max_total_bytes=_int_claim("max_total_bytes"),
        max_bytes_per_file=_int_claim("max_bytes_per_file"),
    )
