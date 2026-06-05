"""Production cancel path: orchestrator interrupt -> ctx.cancel() (#346).

The end-to-end cancel contract is one primitive (``ctx.cancel()``) fed by many
sources. The CLI side is covered in test_cli_cancel*; this covers the PRODUCTION
side: an orchestrator ``interrupt_job_cmd`` is dispatched to
``Worker._handle_interrupt_request(request_id)``, which looks the request up in
the ``_active_requests`` registry and trips ``ctx.cancel()`` — the exact
mechanism the CLI mirrors. Driven with a minimal stand-in so it doesn't need a
booted Worker.
"""

from __future__ import annotations

import threading

from gen_worker.worker import Worker


class _FakeCtx:
    def __init__(self) -> None:
        self.canceled = False

    def cancel(self) -> None:
        self.canceled = True


class _FakeWorker:
    """Just the attributes ``_handle_interrupt_request`` touches."""

    def __init__(self) -> None:
        self._active_requests_lock = threading.Lock()
        self._batched_inflight_lock = threading.Lock()
        self._active_requests: dict = {}
        self._batched_inflight: dict = {}
        self._batched_loop = None


def test_interrupt_request_cancels_registered_ctx() -> None:
    w = _FakeWorker()
    ctx = _FakeCtx()
    w._active_requests["req-1"] = ctx

    # Same call the inbound interrupt_job_cmd handler makes (worker.py:5886).
    Worker._handle_interrupt_request(w, "req-1")
    assert ctx.canceled is True


def test_interrupt_request_unknown_id_is_noop() -> None:
    w = _FakeWorker()
    ctx = _FakeCtx()
    w._active_requests["req-1"] = ctx
    Worker._handle_interrupt_request(w, "does-not-exist")  # must not raise
    assert ctx.canceled is False
