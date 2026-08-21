"""th#2267 §6: a text stream ends on the PRODUCER's death, never on a poll.

`iter_transformers_text_deltas` handed its `timeout` straight to the streamer,
where expiring it raised a bare `queue.Empty` out of the generator. The default
was 0.5 s, and the most ordinary thing a decode does — prefill — takes longer
than that on a long prompt or a large model. The opposite spelling, `None`
(what joycaption passes), is the mirror defect: it blocks the consumer forever
when the generate thread dies, because a dead thread never enqueues the end
sentinel.

Both sides here run on the same fake streamer, with no model and no GPU.
"""

from __future__ import annotations

import queue
import time

import pytest

from gen_worker.serving.deltas import iter_transformers_text_deltas


class _SlowPrefillStreamer:
    """A streamer whose first token is slower than any poll, then streams.

    This is prefill, verbatim: nothing for a while, then tokens.
    """

    def __init__(self, *_a, timeout=None, **_kw):
        self.timeout = timeout
        self._first_at = time.monotonic() + 0.35
        self._left = ["hello", " ", "world"]
        self.polls_expired = 0

    def __iter__(self):
        return self

    def __next__(self):
        if time.monotonic() < self._first_at:
            time.sleep(min(self.timeout or 0.02, 0.02))
            self.polls_expired += 1
            raise queue.Empty
        if not self._left:
            raise StopIteration
        return self._left.pop(0)


class _DeadProducerStreamer:
    """A streamer that never yields, for a generate call that dies at once."""

    def __init__(self, *_a, timeout=None, **_kw):
        self.timeout = timeout

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(min(self.timeout or 0.02, 0.02))
        raise queue.Empty


def _run(streamer_cls, generate, **kw):
    class _Model:
        def generate(self, **kwargs):
            return generate(**kwargs)

    return list(
        iter_transformers_text_deltas(
            model=_Model(),
            tokenizer=object(),
            generation_kwargs={"x": 1},
            streamer_cls=streamer_cls,
            **kw,
        )
    )


def test_prefill_longer_than_a_poll_is_not_the_end_of_the_stream():
    """The false-kill. Every poll expiring while generate RUNS is prefill."""

    def generate(**kwargs):
        time.sleep(0.5)  # the decode is still going

    out = _run(_SlowPrefillStreamer, generate)
    assert out == ["hello", " ", "world"], (
        "a stream whose first token arrived after several expired polls was cut "
        "short — prefill is the most ordinary thing a decode does, and it is not "
        "evidence that anything failed"
    )


def test_a_dead_producer_ends_the_stream_instead_of_hanging_forever():
    """The other side, and the one `timeout=None` could never reach."""

    def generate(**kwargs):
        return None  # returns immediately, enqueues nothing

    started = time.monotonic()
    out = _run(_DeadProducerStreamer, generate, timeout=None)
    assert out == []
    assert time.monotonic() - started < 5.0, (
        "the stream blocked on a producer that had already exited — a dead "
        "thread never enqueues the end sentinel, so nothing but its liveness "
        "can end this wait"
    )


def test_the_producers_exception_survives_and_is_raised_typed():
    """A decode that dies must be attributable, not a bare queue.Empty."""

    class _Boom(RuntimeError):
        pass

    def generate(**kwargs):
        raise _Boom("CUDA out of memory")

    with pytest.raises(_Boom, match="CUDA out of memory"):
        _run(_DeadProducerStreamer, generate, timeout=None)


def test_a_none_timeout_still_polls_so_liveness_can_be_observed():
    """`timeout=None` must NOT reach the streamer as a blocking wait.

    joycaption passes None. A streamer that blocks cannot notice a dead
    producer, which is exactly how that call site hangs.
    """
    seen = {}

    class _Recording(_DeadProducerStreamer):
        def __init__(self, *a, timeout=None, **kw):
            seen["timeout"] = timeout
            super().__init__(*a, timeout=timeout, **kw)

    _run(_Recording, lambda **kw: None, timeout=None)
    assert seen["timeout"] is not None and seen["timeout"] > 0, (
        f"the streamer was built with timeout={seen['timeout']!r} — a blocking "
        "streamer can never observe that its producer died"
    )
