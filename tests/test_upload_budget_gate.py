"""Tests for the capability-budget back-pressure gate (issue #269)."""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest

from gen_worker.request_context._concurrent_upload import (
    BudgetExceededError,
    BudgetGate,
    budget_gate_from_capability_jwt,
)


def test_unbounded_gate_passes_through():
    """Both axes at zero -> no blocking, no rejection."""
    gate = BudgetGate(max_total_bytes=0, max_bytes_per_file=0)
    with gate.reserve(10**12):  # 1 TB on a zero-budget gate
        assert gate.inflight_bytes == 0  # zero-axis doesn't track inflight


def test_per_file_ceiling_rejects_oversize():
    gate = BudgetGate(max_total_bytes=0, max_bytes_per_file=100)
    with pytest.raises(BudgetExceededError, match="exceeds capability max_bytes_per_file"):
        with gate.reserve(101):
            pytest.fail("should not have entered")


def test_aggregate_budget_serializes_oversized_pairs():
    """Two reservations whose sum exceeds the total budget must serialize."""
    gate = BudgetGate(max_total_bytes=100, max_bytes_per_file=0)
    order: list[str] = []
    started = threading.Event()

    def a() -> None:
        with gate.reserve(60):
            order.append("a-enter")
            started.set()
            time.sleep(0.05)
            order.append("a-exit")

    def b() -> None:
        started.wait()
        time.sleep(0.01)
        with gate.reserve(60):
            order.append("b-enter")
            order.append("b-exit")

    t1 = threading.Thread(target=a)
    t2 = threading.Thread(target=b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # 'b' could not enter before 'a' exited because 60+60 > 100.
    assert order == ["a-enter", "a-exit", "b-enter", "b-exit"], order


def test_aggregate_budget_allows_fitting_concurrent():
    """Two reservations whose sum fits the total budget run concurrently."""
    gate = BudgetGate(max_total_bytes=100, max_bytes_per_file=0)
    barrier = threading.Barrier(2)
    both_in = threading.Event()
    in_count = [0]
    lock = threading.Lock()

    def worker() -> None:
        with gate.reserve(40):
            with lock:
                in_count[0] += 1
                if in_count[0] == 2:
                    both_in.set()
            barrier.wait(timeout=1.0)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert both_in.is_set(), "both reservations should have been concurrently in-flight (40+40 <= 100)"


def test_single_file_exceeds_total_budget_fails_fast():
    """A lone reservation larger than max_total_bytes raises rather than deadlocks."""
    gate = BudgetGate(max_total_bytes=50, max_bytes_per_file=0)
    with pytest.raises(BudgetExceededError, match="> capability max_total_bytes"):
        with gate.reserve(60):
            pytest.fail("should not have entered")


def test_reentrant_same_thread_does_not_double_count():
    """Nested reserve() from the same thread is a no-op (save_checkpoint→save_file pattern)."""
    gate = BudgetGate(max_total_bytes=100, max_bytes_per_file=0)
    with gate.reserve(60):
        # Outer holds 60 bytes; inner would be 60+60=120 > 100 if not reentrant.
        # Reentrant => no-op => no block, no exception.
        with gate.reserve(60):
            assert gate.inflight_bytes == 60
        assert gate.inflight_bytes == 60
    assert gate.inflight_bytes == 0


def test_inflight_bytes_release_on_exit():
    gate = BudgetGate(max_total_bytes=100, max_bytes_per_file=0)
    assert gate.inflight_bytes == 0
    with gate.reserve(40):
        assert gate.inflight_bytes == 40
    assert gate.inflight_bytes == 0


def test_inflight_bytes_release_on_exception():
    gate = BudgetGate(max_total_bytes=100, max_bytes_per_file=0)
    with pytest.raises(RuntimeError, match="boom"):
        with gate.reserve(40):
            assert gate.inflight_bytes == 40
            raise RuntimeError("boom")
    assert gate.inflight_bytes == 0


def _make_unverified_jwt(payload: dict) -> str:
    b64 = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return ".".join([b64({"alg": "none", "typ": "JWT"}), b64(payload), ""])


def test_budget_gate_from_capability_jwt_reads_claims():
    token = _make_unverified_jwt({
        "max_total_bytes": 50 * 1024 * 1024 * 1024,
        "max_bytes_per_file": 10 * 1024 * 1024 * 1024,
        "max_files": 32,  # ignored — file-count cap isn't byte-budget
    })
    gate = budget_gate_from_capability_jwt(token)
    assert gate.max_total_bytes == 50 * 1024 * 1024 * 1024
    assert gate.max_bytes_per_file == 10 * 1024 * 1024 * 1024


def test_budget_gate_from_capability_jwt_missing_claims_passes_through():
    """No budget claims -> unbounded gate (both axes zero)."""
    token = _make_unverified_jwt({"cap_kind": "worker_capability"})
    gate = budget_gate_from_capability_jwt(token)
    assert gate.max_total_bytes == 0
    assert gate.max_bytes_per_file == 0


def test_budget_gate_from_capability_jwt_handles_empty_token():
    """Empty/missing token -> pass-through gate."""
    assert budget_gate_from_capability_jwt("").max_total_bytes == 0
    assert budget_gate_from_capability_jwt(None).max_total_bytes == 0  # type: ignore[arg-type]
