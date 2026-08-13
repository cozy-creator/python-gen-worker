"""gw#640: a message-handler exception must never wear a dropped socket's clothes.

Ten live th#1085 runs read as "the worker keeps dying" because `_recv_loop`
awaits the handlers inline and `run()`'s catch-all logged the raise as
"connection to <addr> failed". The process was alive the whole time, so no
in-process instrument could ever see it.
"""

from __future__ import annotations

import asyncio

import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transport import HandlerError, Transport


class _Handlers:
    """Handlers whose on_message raises the way the live worker's did."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.disconnects = 0

    def build_hello(self):  # pragma: no cover - not reached in these tests
        return pb.Hello()

    async def on_hello_ack(self, ack):
        return None

    async def on_message(self, msg):
        raise self.exc

    async def on_disconnect(self):
        self.disconnects += 1


def _transport(handlers, settings) -> Transport:
    return Transport(settings, handlers)


def _settings():
    from gen_worker import config as gw_config

    return gw_config.current()


def test_handler_exception_is_its_own_class_not_a_transport_error():
    """The wrapper carries WHICH message and the original cause."""
    err = HandlerError("run_job", ValueError("payload decode blew up"))
    assert err.kind == "run_job"
    assert isinstance(err.cause, ValueError)
    assert "run_job" in str(err)
    assert "ValueError" in str(err)
    assert "payload decode blew up" in str(err)


def test_recv_loop_wraps_handler_raise_and_names_the_message(monkeypatch):
    """A RunJob handler raise surfaces as HandlerError(kind='run_job')."""
    boom = RuntimeError("msgpack decode failed")
    handlers = _Handlers(boom)
    t = _transport(handlers, _settings())

    run_job = pb.SchedulerMessage(run_job=pb.RunJob(request_id="r1", attempt=1))

    class _Stream:
        def __init__(self):
            self.reads = [run_job]

        async def read(self):
            return self.reads.pop(0)

    with pytest.raises(HandlerError) as caught:
        asyncio.run(t._recv_loop(_Stream()))
    assert caught.value.kind == "run_job"
    assert caught.value.cause is boom


def test_handler_failure_is_reported_through_the_worker_fatal_carrier(monkeypatch):
    """It dials the hub (bounded), naming the phase after the message kind."""
    t = _transport(_Handlers(RuntimeError("x")), _settings())
    seen = {}

    def _fake_report(settings, phase, exc, *, exit_code):
        seen["phase"] = phase
        seen["exc"] = exc
        seen["exit_code"] = exit_code
        return True

    import gen_worker.worker_fatal as wf

    monkeypatch.setattr(wf, "report_worker_fatal", _fake_report)

    cause = RuntimeError("msgpack decode failed")
    t._report_handler_failure(HandlerError("run_job", cause))
    assert seen["phase"] == "message_handler:run_job"
    assert seen["exc"] is cause

    # deduped: the reconnect loop must not re-dial the same fault every cycle
    seen.clear()
    t._report_handler_failure(HandlerError("run_job", RuntimeError("msgpack decode failed")))
    assert seen == {}

    # a DIFFERENT fault still reports
    t._report_handler_failure(HandlerError("run_job", ValueError("other")))
    assert seen["phase"] == "message_handler:run_job"


def test_transport_failures_are_still_plain_transport_failures():
    """EOF from the scheduler stays a ConnectionError, not a HandlerError."""
    import grpc

    t = _transport(_Handlers(RuntimeError("unused")), _settings())

    class _EofStream:
        async def read(self):
            return grpc.aio.EOF

    with pytest.raises(ConnectionError):
        asyncio.run(t._recv_loop(_EofStream()))


# --- gw#640 secondary half: stop EMITTING malformed shadow state ------------


def test_waiting_intent_always_carries_blocker_retry_or_deadline():
    """The hub's shadow validator requires one of the three on a WAITING state.

    compat-synthesized intents (`compat-materialize-*`, minted by
    ensure_local_intent OUTSIDE a DesiredStateCommand, so they never get the
    command's first_action deadline) carried none of the three. That is the
    exact snapshot the hub rejected for twelve straight th#1085 runs.
    """
    from gen_worker.lifecycle_intents import IntentRegistry
    from gen_worker.pb import worker_scheduler_pb2 as p

    reg = IntentRegistry("release-1", ["artifact-stat"])
    intent_id = reg.ensure_local_intent("materialize", "tensorhub/tiny:prod")
    assert intent_id.startswith("compat-materialize-"), intent_id

    reg.transition(
        intent_id,
        p.LIFECYCLE_INTENT_STATUS_WAITING,
        p.LIFECYCLE_INTENT_STAGE_FETCHING,
        reason=p.LIFECYCLE_WAIT_REASON_NETWORK_RETRY,
    )
    state = next(s for s in reg.snapshot().intents if s.intent_id == intent_id)
    blocked = bool(state.blocker_intent_id) or bool(state.blocker_request.request_id)
    assert blocked or state.next_retry_at_unix_ms > 0 or state.deadline_at_unix_ms > 0, (
        "a WAITING intent with no blocker and no retry must get a fallback "
        "deadline, or the hub rejects the whole snapshot"
    )


def test_explicit_waiting_fields_are_not_overwritten():
    """The fallback must only fill a genuine gap."""
    from gen_worker.lifecycle_intents import IntentRegistry
    from gen_worker.pb import worker_scheduler_pb2 as p

    reg = IntentRegistry("release-1", ["artifact-stat"])
    intent_id = reg.ensure_local_intent("materialize", "tensorhub/tiny:prod")
    reg.transition(
        intent_id,
        p.LIFECYCLE_INTENT_STATUS_WAITING,
        p.LIFECYCLE_INTENT_STAGE_FETCHING,
        reason=p.LIFECYCLE_WAIT_REASON_NETWORK_RETRY,
        next_retry_at_unix_ms=1234,
    )
    state = next(s for s in reg.snapshot().intents if s.intent_id == intent_id)
    assert state.next_retry_at_unix_ms == 1234
    assert state.deadline_at_unix_ms == 0, "an explicit retry time needs no deadline"
