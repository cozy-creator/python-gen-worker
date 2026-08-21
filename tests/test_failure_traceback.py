"""The traceback tail a failed job ships, and the catch site that ships it."""

from __future__ import annotations

import asyncio
import types
from typing import List

import pytest

from gen_worker import failure_traceback as ft
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.redact import sanitize, sanitize_credentials
from gen_worker.serving.envelope import EnvelopeError
from gen_worker.worker import Worker


def _raise_keyerror_deep() -> None:

    def collect(state):
        return state["keys"]

    def calibrate(state):
        return collect(state)

    calibrate({})


def test_traceback_tail_names_the_module_the_line_and_the_call() -> None:
    try:
        _raise_keyerror_deep()
    except KeyError as exc:
        tail = ft.traceback_tail(exc)
    assert "KeyError: 'keys'" in tail
    assert "test_failure_traceback.py" in tail, tail
    assert "in collect" in tail, tail
    assert 'return state["keys"]' in tail, tail


def test_the_bound_keeps_the_TAIL_and_says_it_was_cut() -> None:
    def recurse(n: int) -> None:
        if n == 0:
            raise KeyError("keys")
        recurse(n - 1)

    try:
        recurse(400)
    except KeyError as exc:
        tail = ft.traceback_tail(exc, max_bytes=600)

    assert len(tail.encode("utf-8")) <= 600
    assert tail.endswith("KeyError: 'keys'"), tail[-120:]
    assert tail.startswith(ft.TRUNCATED_MARKER), tail[:120]
    for line in tail.splitlines()[1:]:
        assert line == "" or line.startswith((" ", "KeyError", "Traceback")), line


def test_an_unbounded_traceback_is_returned_whole() -> None:
    try:
        _raise_keyerror_deep()
    except KeyError as exc:
        tail = ft.traceback_tail(exc)
    assert ft.TRUNCATED_MARKER not in tail
    assert tail.startswith("Traceback (most recent call last):")


def test_the_frame_cap_is_a_TAIL_cap_too() -> None:
    def recurse(n: int) -> None:
        if n == 0:
            raise KeyError("keys")
        recurse(n - 1)

    try:
        recurse(200)
    except KeyError as exc:
        tail = ft.traceback_tail(exc)
    assert tail.count("in recurse") <= ft.MAX_FRAMES
    assert tail.endswith("KeyError: 'keys'")


def test_credentials_are_scrubbed_and_PATHS_ARE_NOT() -> None:
    line = (
        'HTTPError: 403 for /opt/endpoint/jobs/quantize.py via '
        "https://r2.example/obj?X-Amz-Signature=deadbeefcafe&x=1 "
        "with Bearer sk-live-abcdefghijklmnop"
    )
    scrubbed = sanitize_credentials(line)
    assert "deadbeefcafe" not in scrubbed
    assert "sk-live-abcdefghijklmnop" not in scrubbed
    assert "/opt/endpoint/jobs/quantize.py" in scrubbed

    assert "/opt/endpoint/jobs/quantize.py" not in sanitize(line)
    assert "deadbeefcafe" not in sanitize(line)


def test_a_capability_token_dragged_into_a_message_is_scrubbed() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqb2IifQ.c2lnbmF0dXJlLWhlcmU"
    try:
        raise RuntimeError(f"refused token {jwt}")
    except RuntimeError as exc:
        tail = ft.traceback_tail(exc)
    assert jwt not in tail
    assert "[redacted]" in tail


def test_the_formatter_never_raises_on_a_hostile_exception() -> None:
    class Hostile(Exception):
        def __str__(self) -> str:  # pragma: no cover - exercised via the tail
            raise ValueError("this exception refuses to be formatted")

    try:
        raise Hostile()
    except Hostile as exc:
        tail = ft.traceback_tail(exc)
    assert isinstance(tail, str)


class _CapturedWorker:

    def __init__(self, boom: BaseException) -> None:
        self.sent: List[pb.WorkerMessage] = []
        w = object.__new__(Worker)
        w._jobs = {}
        w._canceled = set()
        w.draining = False
        w.lanes = frozenset()
        w.file_base_url = ""
        w.loaded = types.SimpleNamespace(entrypoints={})  # type: ignore[assignment]

        def _invoke(*_args, **_kwargs):
            raise boom

        w.serve = types.SimpleNamespace(invoke=_invoke)  # type: ignore[assignment]

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        w._send = _send  # type: ignore[method-assign]
        self.w = w

    def run(self, job_id: str = "job-1") -> pb.JobResult:
        run = pb.RunJob(request_id=job_id, attempt=1, function_name="quantize")
        asyncio.run(self.w._run_one(run, (job_id, 1)))
        results = [m.job_result for m in self.sent if m.HasField("job_result")]
        assert len(results) == 1, self.sent
        return results[0]


def test_the_catch_site_ships_the_tail_with_the_fatal_terminal() -> None:
    try:
        _raise_keyerror_deep()
    except KeyError as exc:
        boom = exc

    result = _CapturedWorker(boom).run()

    assert result.status == pb.JOB_STATUS_FATAL
    assert result.safe_message == "KeyError: 'keys'"
    assert "in collect" in result.traceback, result.traceback
    assert 'return state["keys"]' in result.traceback


def test_an_invalid_terminal_ships_one_too() -> None:
    result = _CapturedWorker(EnvelopeError("steps must be an integer")).run()
    assert result.status == pb.JOB_STATUS_INVALID
    assert "EnvelopeError" in result.traceback
    assert "steps must be an integer" in result.traceback


def test_the_wire_field_is_bounded_at_the_hubs_own_ceiling() -> None:
    def recurse(n: int) -> None:
        if n == 0:
            raise KeyError("keys")
        recurse(n - 1)

    try:
        recurse(3000)
    except (KeyError, RecursionError) as exc:
        boom = exc

    result = _CapturedWorker(boom).run()
    assert len(result.traceback.encode("utf-8")) <= ft.MAX_BYTES
    assert result.traceback.endswith(("KeyError: 'keys'", "recursion depth exceeded"))


@pytest.mark.parametrize("field", ["traceback"])
def test_the_field_is_absent_on_a_run_that_did_not_raise(field: str) -> None:
    result = pb.JobResult(request_id="x", attempt=1, status=pb.JOB_STATUS_OK)
    assert getattr(result, field) == ""
