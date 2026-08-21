"""Capability byte-budget back-pressure for worker-side uploads."""

from __future__ import annotations

import logging
import threading
from typing import Any

from gen_worker.request_context._helpers import _decode_unverified_jwt_claims

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when a single file exceeds the per-file or total byte budget."""


class BudgetGate:
    """Capability-budget back-pressure for the concurrent upload pool."""

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
        if gate._max_bytes_per_file > 0 and size > gate._max_bytes_per_file:
            raise BudgetExceededError(
                f"file size {size} exceeds capability max_bytes_per_file {gate._max_bytes_per_file}"
            )
        depth = getattr(gate._tls, "depth", 0)
        gate._tls.depth = depth + 1
        if depth > 0:
            return self
        if gate._max_total_bytes > 0:
            with gate._cond:
                while gate._inflight + size > gate._max_total_bytes:
                    if gate._inflight == 0:
                        gate._tls.depth = depth
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
        if not self._held:
            return
        if gate._max_total_bytes > 0:
            with gate._cond:
                gate._inflight = max(0, gate._inflight - self._size)
                gate._cond.notify_all()
        self._held = False


def budget_gate_from_capability_jwt(token: str) -> BudgetGate:
    """Construct a BudgetGate from a worker_capability_token's budget claims."""

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
