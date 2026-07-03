"""Worker reconnect robustness (gen-worker #338).

Three compounding worker-side causes wedged an already-connected worker
permanently when the orchestrator restarted:

1. `_emit_ready_if_all_cached` is gated by a once-only `_ready_phase_emitted`
   latch that was never reset on reconnect, so the worker re-registered but
   never re-advertised `ready` -> orchestrator pinned it at connected=0.
2. The reconnect loop retried with ~no backoff (retry storm).
3. A write-only-dead ("half-open") scheduler stream was never detected, so the
   worker settled on a registered-but-dead stream.

These tests drive the real Worker helpers via a bare `Worker.__new__` instance
(the same pattern as test_concurrency_semaphore / test_worker_dispatch) so we
exercise the actual latch / backoff / half-open logic, not a reimplementation.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Tuple


from gen_worker.worker import HEARTBEAT_INTERVAL, Worker


# --------------------------------------------------------------------------- #
# Fix 1: the ready latch resets on (re)connect so `ready` re-fires.
# --------------------------------------------------------------------------- #


def _ready_worker(required: set[str] | None = None, on_disk: set[str] | None = None) -> Tuple[Worker, List[str]]:
    """Bare Worker wired with just enough state for the readiness gate, plus a
    capture list of every startup phase emitted."""
    w = Worker.__new__(Worker)
    w._ready_phase_emitted = False
    w._ready_phase_lock = threading.Lock()
    w._startup_required_refs_canonical = set(required or set())
    w._terminally_failed_refs = set()
    w.scheduler_addr = "sched:50051"

    class _Cache:
        def get_disk_models(self) -> list[str]:
            return list(on_disk or set())

    w._model_cache = _Cache()

    emitted: List[str] = []

    def _capture(phase: str, **_kw: Any) -> None:
        emitted.append(phase)

    w._emit_startup_phase = _capture  # type: ignore[method-assign]
    return w, emitted


def test_ready_emits_once_then_latches_until_reset() -> None:
    # Idle worker (no required refs) -> emits `ready` immediately, then latches.
    w, emitted = _ready_worker(required=set())
    w._emit_ready_if_all_cached()
    assert emitted == ["ready"]
    assert w._ready_phase_emitted is True

    # Second call without a reset is a no-op (the once-only behavior).
    w._emit_ready_if_all_cached()
    assert emitted == ["ready"]


def test_reset_ready_phase_latch_lets_ready_refire_on_reconnect() -> None:
    """The #338 fix-1 core: after the latch is reset (simulating a reconnect),
    `_emit_ready_if_all_cached` re-fires `ready`."""
    w, emitted = _ready_worker(required=set())
    w._emit_ready_if_all_cached()
    assert emitted == ["ready"]

    # Simulate a reconnect: the stream re-established, latch is cleared.
    w._reset_ready_phase_latch()
    assert w._ready_phase_emitted is False

    w._emit_ready_if_all_cached()
    assert emitted == ["ready", "ready"], "ready must re-fire after a reconnect"


def test_reset_ready_phase_latch_safe_without_lock() -> None:
    # bare-Worker path that never connected (no _ready_phase_lock): no crash.
    w = Worker.__new__(Worker)
    w._reset_ready_phase_latch()
    assert w._ready_phase_emitted is False


def test_ready_blocked_until_required_ref_cached_then_refires() -> None:
    # Required ref not on disk -> ready does NOT fire.
    w, emitted = _ready_worker(required={"repo:a"}, on_disk=set())
    w._emit_ready_if_all_cached()
    assert emitted == []
    assert w._ready_phase_emitted is False

    # Ref lands -> ready fires.
    w._model_cache.get_disk_models = lambda: ["repo:a"]  # type: ignore[method-assign]
    w._emit_ready_if_all_cached()
    assert emitted == ["ready"]

    # Reconnect resets the latch; ready re-fires because the ref is still cached.
    w._reset_ready_phase_latch()
    w._emit_ready_if_all_cached()
    assert emitted == ["ready", "ready"]


# --------------------------------------------------------------------------- #
# Fix 2: bounded exponential backoff with full jitter.
# --------------------------------------------------------------------------- #


def _backoff_worker(base: float = 0.5, cap: float = 30.0) -> Worker:
    w = Worker.__new__(Worker)
    w._reconnect_delay_base = base
    w._reconnect_delay_max = cap
    return w


def test_backoff_is_bounded_by_cap() -> None:
    w = _backoff_worker(base=0.5, cap=30.0)
    # Sample many draws across escalating attempts; none may exceed the cap.
    for attempt in range(1, 40):
        for _ in range(50):
            d = w._next_reconnect_delay(attempt)
            assert 0.0 <= d <= 30.0, f"attempt={attempt} delay={d} out of [0,cap]"


def test_backoff_expectation_grows_exponentially_then_saturates() -> None:
    """Full jitter: realized delay is uniform(0, capped_backoff), so the MAX
    achievable delay doubles each attempt until it hits the cap."""
    w = _backoff_worker(base=0.5, cap=30.0)

    def max_draw(attempt: int) -> float:
        return max(w._next_reconnect_delay(attempt) for _ in range(2000))

    # attempt 1 -> ceiling ~0.5, attempt 2 -> ~1.0, attempt 3 -> ~2.0 ...
    m1, m2, m3 = max_draw(1), max_draw(2), max_draw(3)
    assert m1 <= 0.5 + 1e-9
    assert m1 < m2 < m3, f"ceiling must grow: {m1} {m2} {m3}"
    assert m2 <= 1.0 + 1e-9
    assert m3 <= 2.0 + 1e-9

    # Far out the ceiling saturates at the cap, never beyond.
    m_high = max_draw(20)
    assert 25.0 <= m_high <= 30.0, f"should saturate near cap, got {m_high}"


def test_backoff_no_longer_a_tight_retry_storm() -> None:
    """The bug was a ~0.1s constant retry. With full jitter and a 0.5s base, the
    AVERAGE delay over many draws at attempt>=1 must be materially above 0.1s."""
    w = _backoff_worker(base=0.5, cap=30.0)
    draws = [w._next_reconnect_delay(1) for _ in range(5000)]
    avg = sum(draws) / len(draws)
    # uniform(0, 0.5) has mean ~0.25s — comfortably above the 0.1s storm.
    assert avg > 0.15, f"average reconnect delay {avg:.3f}s too tight (storm regression)"


def test_backoff_defaults_set_in_init() -> None:
    """Guard the #338 fix-2 tuning so a future edit can't silently revert the
    base/cap back to the storm-y 5s/60s values."""
    import inspect

    src = inspect.getsource(Worker.__init__)
    assert "_reconnect_delay_max = 30" in src
    assert "_reconnect_delay_base" in src


# --------------------------------------------------------------------------- #
# Fix 3: half-open (write-only-dead) stream detection.
# --------------------------------------------------------------------------- #


def _half_open_worker(timeout_s: float) -> Worker:
    w = Worker.__new__(Worker)
    w._half_open_silence_timeout_s = timeout_s
    w._last_server_msg_at = time.monotonic()
    return w


def test_half_open_not_tripped_while_recently_heard_from() -> None:
    w = _half_open_worker(timeout_s=float(HEARTBEAT_INTERVAL * 6))
    # Just heard from the scheduler -> healthy.
    assert w._stream_is_half_open() is False
    # A small staleness, well under the window -> still healthy.
    w._last_server_msg_at = time.monotonic() - (HEARTBEAT_INTERVAL * 2)
    assert w._stream_is_half_open() is False


def test_half_open_trips_after_silence_window() -> None:
    timeout = float(HEARTBEAT_INTERVAL * 6)
    w = _half_open_worker(timeout_s=timeout)
    # No inbound message for longer than the window -> half-open.
    w._last_server_msg_at = time.monotonic() - (timeout + 1.0)
    assert w._stream_is_half_open() is True


def test_half_open_disabled_when_timeout_zero() -> None:
    w = _half_open_worker(timeout_s=0.0)
    w._last_server_msg_at = time.monotonic() - 10_000.0
    assert w._stream_is_half_open() is False


def test_half_open_signal_triggers_reconnect_teardown() -> None:
    """When the watchdog trips, the heartbeat loop must force a teardown via
    `_handle_connection_error` and stop, rather than settling on the dead
    stream. We mock the health signal + the teardown to assert the wiring."""
    w = Worker.__new__(Worker)
    w._stop_event = threading.Event()
    w._half_open_silence_timeout_s = float(HEARTBEAT_INTERVAL * 6)
    # Force the half-open predicate True regardless of clock.
    w._stream_is_half_open = lambda *a, **k: True  # type: ignore[method-assign]

    torn_down: List[bool] = []
    w._handle_connection_error = lambda: torn_down.append(True)  # type: ignore[method-assign]
    # _register_worker should NOT be called on a half-open tick.
    sent: List[bool] = []
    w._register_worker = lambda is_heartbeat=False: sent.append(True)  # type: ignore[method-assign]

    # Run the loop in a thread; the first HEARTBEAT_INTERVAL wait will fire and
    # the half-open check should break it immediately. Use a tiny patched
    # interval by monkeypatching the wait to return immediately once.
    waited: List[bool] = []
    orig_wait = w._stop_event.wait

    def _fast_wait(_t: float) -> bool:
        # First tick: return False (not stopped) so the loop body runs once.
        if not waited:
            waited.append(True)
            return False
        return orig_wait(0.01)

    w._stop_event.wait = _fast_wait  # type: ignore[method-assign]

    t = threading.Thread(target=w._heartbeat_loop, daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "heartbeat loop must exit after detecting half-open"
    assert torn_down == [True], "half-open must force a reconnect teardown"
    assert sent == [], "no heartbeat should be sent on a half-open tick"


def test_process_message_stamps_liveness_clock() -> None:
    """Every inbound message must refresh the liveness clock the watchdog
    reads, so a busy stream never trips the half-open detector."""
    w = Worker.__new__(Worker)
    w._last_server_msg_at = time.monotonic() - 1000.0

    # Build a minimal message object that _process_message can introspect.
    class _Msg:
        def WhichOneof(self, _field: str) -> str:
            return "unknown_type"  # falls through all branches harmlessly

    before = w._last_server_msg_at
    w._process_message(_Msg())  # type: ignore[arg-type]
    assert w._last_server_msg_at > before
