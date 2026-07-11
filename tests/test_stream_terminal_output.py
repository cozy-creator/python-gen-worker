"""gw#475: stream-mode functions must produce a NON-empty terminal output.

Live deltas are droppable by contract (tensorhub #505: in-memory ProgressHub
only, never persisted). The terminal JobResult is the authoritative record:
the executor accumulates yielded deltas and serializes them as the result —
concatenated text for IncrementalTokenDelta, finished items[] for
BatchItemDelta — while the live JobProgress stream is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator, List

import msgspec

from gen_worker.api.streaming import BatchItemDelta, IncrementalTokenDelta
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    n: int = 3


async def _token_stream(ctx, payload: _In) -> AsyncIterator[IncrementalTokenDelta]:
    for i in range(payload.n):
        yield IncrementalTokenDelta(text=f"tok{i} ")


async def _empty_stream(ctx, payload: _In) -> AsyncIterator[IncrementalTokenDelta]:
    if False:  # pragma: no cover
        yield IncrementalTokenDelta(text="")


def _batch_stream(ctx, payload: _In) -> Iterator[BatchItemDelta]:
    # Item 0 arrives in two chunks; item 1 in one finished delta.
    yield BatchItemDelta(index=0, total=2, chunk=b"a red ", content_type="text/plain")
    yield BatchItemDelta(index=0, total=2, chunk=b"apple", content_type="text/plain",
                         finished=True)
    yield BatchItemDelta(index=1, total=2, item_id="b", chunk=b"\x81\xa1x",
                         content_type="application/octet-stream", finished=True)


def _run(method, *, is_async_gen: bool) -> List[pb.WorkerMessage]:
    async def _go() -> List[pb.WorkerMessage]:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        spec = EndpointSpec(
            name="fn", method=method, kind="inference",
            payload_type=_In, output_mode="stream",
            delta_type=IncrementalTokenDelta,
            is_async=is_async_gen, is_async_gen=is_async_gen,
        )
        ex = Executor([spec], _send)
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="fn",
            input_payload=msgspec.msgpack.encode(_In())))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        for _ in range(20):
            await asyncio.sleep(0)
        return sent

    return asyncio.run(_go())


def _result(sent: List[pb.WorkerMessage]) -> pb.JobResult:
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert len(results) == 1
    return results[0]


def _progress(sent: List[pb.WorkerMessage]) -> List[pb.JobProgress]:
    return [m.job_progress for m in sent if m.WhichOneof("msg") == "job_progress"]


def test_token_stream_terminal_output_carries_full_text() -> None:
    sent = _run(_token_stream, is_async_gen=True)
    res = _result(sent)
    assert res.status == pb.JOB_STATUS_OK
    assert res.WhichOneof("output") == "inline"
    out = msgspec.msgpack.decode(res.inline)
    assert out == {"text": "tok0 tok1 tok2 "}
    # Live stream unchanged: one text/plain chunk per delta.
    chunks = _progress(sent)
    assert [c.data for c in chunks] == [b"tok0 ", b"tok1 ", b"tok2 "]
    assert {c.content_type for c in chunks} == {"text/plain"}


def test_batch_stream_terminal_output_accumulates_finished_items() -> None:
    sent = _run(_batch_stream, is_async_gen=False)
    res = _result(sent)
    assert res.status == pb.JOB_STATUS_OK
    assert res.WhichOneof("output") == "inline"
    out = msgspec.msgpack.decode(res.inline)
    items = out["items"]
    assert len(items) == 2
    # Chunked text item: concatenated and decoded to str.
    assert items[0]["index"] == 0
    assert items[0]["content"] == "a red apple"
    assert items[0]["content_type"] == "text/plain"
    assert items[0]["error"] == ""
    # Binary item: bytes preserved verbatim.
    assert items[1]["item_id"] == "b"
    assert items[1]["content"] == b"\x81\xa1x"


def test_empty_stream_keeps_empty_terminal_output() -> None:
    sent = _run(_empty_stream, is_async_gen=True)
    res = _result(sent)
    assert res.status == pb.JOB_STATUS_OK
    assert res.WhichOneof("output") is None
