"""Incremental output, end to end (pgw#1576).

The v1 hardcut deleted token streaming and shipped no successor: three
endpoints (`qwen3.6-35b-a3b`, `qwen3.6-27b-mtp-gguf`, `joycaption`) could not
be ported at all. The successor is NOT the v1 async generator — Python forbids
`return <value>` inside one, so that shape can express only the DROPPABLE half
of the wire. A streaming entrypoint declares its chunk type, emits chunks, and
still RETURNS its terminal struct.

This drives the real path on CPU with no weights and no GPU:

1. the declaration — `streams=` lands on the spec; a generator is refused with
   the migration line;
2. the serve loop — chunks reach a sink in order, the terminal is the return;
3. **the worker** — `Worker._run_one` with a capturing transport: N ordered
   `JobProgress` frames (including the ctx-event lane that was dead) followed
   by exactly one `JobResult` carrying the whole output;
4. the manifest — `incremental_output` + `delta_output_schema`, read off the
   spec without executing author code.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, List, Tuple

import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.serving.deltas import ItemDelta, TokenDelta, frame_of
from gen_worker.serving.entrypoints import (
    ENTRYPOINT_ATTR,
    EntrypointDeclarationError,
)
from gen_worker.serving.loader import load_endpoint_module
from gen_worker.serving.residency import ResidencyManager
from gen_worker.serving.serve_loop import ServeLoop
from gen_worker.worker import EVENT_CONTENT_TYPE, Worker

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "streaming_endpoint"


@pytest.fixture(scope="module")
def streaming() -> Iterator[ModuleType]:
    sys.path.insert(0, str(FIXTURES))
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))


# -- 1. the declaration ------------------------------------------------------


def test_the_chunk_type_lands_on_the_spec(streaming: ModuleType) -> None:
    """`streams=` is a STATIC fact: publish reads it, no body runs."""
    spec = getattr(streaming.complete, ENTRYPOINT_ATTR)
    assert spec.delta_type is TokenDelta and spec.delta_arms == (TokenDelta,)
    # ...and the return contract is UNCHANGED — the terminal is still a struct.
    assert spec.return_type is streaming.Completion

    assert getattr(streaming.caption, ENTRYPOINT_ATTR).delta_type is ItemDelta
    assert getattr(streaming.silent, ENTRYPOINT_ATTR).delta_type is None


def _declare(source: str) -> None:
    """A REAL module-level declaration — @entrypoint refuses nested functions
    first, so an in-function probe would measure the wrong wall."""
    preamble = (
        "import msgspec\n"
        "from typing import AsyncIterator, Iterator\n"
        "from gen_worker import RequestContext, TokenDelta, entrypoint\n"
        "class In(msgspec.Struct): text: str\n"
        "class Out(msgspec.Struct): text: str\n"
    )
    module = types.ModuleType("pgw1576_probe")
    sys.modules["pgw1576_probe"] = module
    try:
        exec(preamble + source, module.__dict__)  # noqa: S102
    finally:
        del sys.modules["pgw1576_probe"]


@pytest.mark.parametrize(
    "source",
    [
        # the v1 shape, verbatim (qwen3.6-35b-a3b's `complete`)
        "@entrypoint\n"
        "async def f(ctx: RequestContext, payload: In) -> AsyncIterator[TokenDelta]:\n"
        "    yield TokenDelta(text='x')\n",
        # ...and the sync one (joycaption's `caption_images`)
        "@entrypoint\n"
        "def f(ctx: RequestContext, payload: In) -> Iterator[TokenDelta]:\n"
        "    yield TokenDelta(text='x')\n",
        # a generator BODY whose annotation lies about it
        "@entrypoint\n"
        "def f(ctx: RequestContext, payload: In) -> Out:\n"
        "    yield Out(text='x')\n",
    ],
)
def test_a_generator_entrypoint_is_refused_with_the_migration(source: str) -> None:
    """The refusal IS the design, so it has to say what to write instead —
    'return type must be a msgspec.Struct' would send the author hunting for a
    schema defect that is not there."""
    with pytest.raises(EntrypointDeclarationError) as excinfo:
        _declare(source)
    message = str(excinfo.value)
    assert "streams=TokenDelta" in message
    assert "ctx.emit" in message
    assert "return" in message


def test_streams_takes_a_struct_type_and_says_so() -> None:
    with pytest.raises(EntrypointDeclarationError, match="streams= takes the"):
        _declare(
            "@entrypoint(streams='TokenDelta')\n"
            "def f(ctx: RequestContext, payload: In) -> Out: ...\n"
        )


def test_a_multi_shape_declaration_must_be_discriminable() -> None:
    """Several untagged arms cannot be told apart in ONE published schema, so
    the ambiguity is refused at import rather than shipped to clients."""
    with pytest.raises(EntrypointDeclarationError, match="carry no msgspec tag"):
        _declare(
            "class D1(msgspec.Struct): a: str = ''\n"
            "class D2(msgspec.Struct): b: str = ''\n"
            "@entrypoint(streams=(D1, D2))\n"
            "def f(ctx: RequestContext, payload: In) -> Out: ...\n"
        )


# -- 2. framing --------------------------------------------------------------


def test_a_token_delta_frames_as_concatenable_text() -> None:
    """The hub renders every non-audio chunk as `payload["delta"] = string(data)`
    inside a JSON SSE envelope, so a token stream must BE text on the wire."""
    assert frame_of(TokenDelta(text="hello ")) == (b"hello ", "text/plain")


def test_an_item_delta_frames_as_json_because_its_metadata_is_the_point() -> None:
    data, content_type = frame_of(ItemDelta(index=2, total=5, text="a cat"))
    assert content_type == "application/json"
    assert msgspec.json.decode(data)["index"] == 2


# -- 3. the serve loop -------------------------------------------------------


class _NoModels:
    def resolve(self, model_cls: type, checkpoint_ref: str) -> Any:
        raise AssertionError("a weightless request resolved a binding")

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        raise AssertionError("a weightless request asked for a default pick")

    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        raise AssertionError("a weightless request sized a residency slot")

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        raise AssertionError("a weightless request reserved activation bytes")


def _loop() -> ServeLoop:
    sys.path.insert(0, str(FIXTURES))
    try:
        loaded = load_endpoint_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))
    return ServeLoop(
        loaded,
        residency=ResidencyManager(1 << 30, _NoModels()),
        resolver=_NoModels(),
    )


def test_the_serve_loop_streams_the_chunks_and_returns_the_terminal(
    streaming: ModuleType,
) -> None:
    chunks: List[Tuple[bytes, str]] = []
    outcome = _loop().invoke(
        "complete",
        {"input": {"prompt": "hi", "max_tokens": 3}},
        request_id="pgw1576-loop",
        context={"chunk_sink": lambda data, ct: chunks.append((data, ct))},
    )

    # The DROPPABLE half: three ordered text frames.
    assert [data for data, _ in chunks] == [b"hi-0 ", b"hi-1 ", b"hi-2 "]
    assert {ct for _, ct in chunks} == {"text/plain"}
    # The AUTHORITATIVE half: the whole completion, typed, as the return value.
    assert isinstance(outcome.result, streaming.Completion)
    assert outcome.result.text == "hi-0 hi-1 hi-2 "
    assert outcome.result.tokens == 3


def test_a_sync_entrypoint_streams_with_the_identical_call(
    streaming: ModuleType,
) -> None:
    """joycaption's handler is a plain `def`; it must not need an async rewrite."""
    chunks: List[Tuple[bytes, str]] = []
    outcome = _loop().invoke(
        "caption",
        {"input": {"items": ["a cat", "", "a dog"]}},
        request_id="pgw1576-sync",
        context={"chunk_sink": lambda data, ct: chunks.append((data, ct))},
    )

    rows = [msgspec.json.decode(data) for data, _ in chunks]
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert rows[1]["error"] == "blank item"          # one ITEM failed...
    assert outcome.result.failed == 1                # ...and the request did not
    assert outcome.result.captions[2] == "a photo of a dog"


def test_one_handler_streams_two_shapes(streaming: ModuleType) -> None:
    """joycaption's case: tokens WHILE an item decodes, an item frame when it
    finishes. Both arms are declared, so both are publishable and emittable."""
    chunks: List[Tuple[bytes, str]] = []
    _loop().invoke(
        "narrate",
        {"input": {"items": ["a cat"]}},
        request_id="pgw1576-union",
        context={"chunk_sink": lambda data, ct: chunks.append((data, ct))},
    )

    assert [(data, ct) for data, ct in chunks[:2]] == [
        (b"a ", "text/plain"), (b"cat ", "text/plain"),
    ]
    terminal = msgspec.json.decode(chunks[-1][0])
    # The JSON arm carries its own discriminator, so a consumer reading one
    # stream of mixed frames never has to guess which shape it holds.
    assert terminal["type"] == "ItemDelta" and terminal["finished"] is True


def test_emit_refuses_a_function_that_declared_no_chunk_type() -> None:
    """An undeclared emitter would stream past a manifest saying
    `incremental_output: false` — one fact, two enforcers."""
    with pytest.raises(RuntimeError, match="declared no chunk type"):
        _loop().invoke(
            "silent",
            {"input": {"items": []}},
            request_id="pgw1576-undeclared",
            context={"chunk_sink": lambda data, ct: None},
        )


def test_emit_refuses_a_chunk_of_another_type() -> None:
    from gen_worker.serving.context import RequestContext

    ctx: RequestContext[Any] = RequestContext(
        "pgw1576-type", streams=(TokenDelta,), chunk_sink=lambda *_: None
    )
    with pytest.raises(TypeError, match="streams=TokenDelta"):
        ctx.emit(ItemDelta(index=0))


# -- 4. the worker: the actual wire -----------------------------------------


class _CapturedWorker:
    """`Worker._run_one` over a REAL `ServeLoop`, with only the socket faked.

    The dispatch, the `to_thread` hop the body really runs on, the per-request
    JobProgress channel, the seq counter and the terminal `JobResult` are all
    the production ones; `_send` collects instead of writing to gRPC.
    """

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        worker = object.__new__(Worker)
        worker._jobs = {}
        worker._canceled = set()
        worker.draining = False
        worker.lanes = frozenset()
        worker.file_base_url = ""
        worker.serve = _loop()
        worker.loaded = worker.serve.loaded

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        worker._send = _send  # type: ignore[method-assign]
        self.worker = worker

    def run(self, function: str, payload: Any, job_id: str = "pgw1576") -> None:
        run = pb.RunJob(
            request_id=job_id,
            attempt=1,
            function_name=function,
            input_payload=msgspec.msgpack.encode(payload),
        )
        asyncio.run(self.worker._run_one(run, (job_id, 1)))

    @property
    def progress(self) -> List[pb.JobProgress]:
        return [m.job_progress for m in self.sent if m.HasField("job_progress")]

    @property
    def result(self) -> pb.JobResult:
        results = [m.job_result for m in self.sent if m.HasField("job_result")]
        assert len(results) == 1, self.sent
        return results[0]


def test_the_wire_carries_ordered_chunks_then_one_terminal_result(
    streaming: ModuleType,
) -> None:
    captured = _CapturedWorker()
    captured.run("complete", {"prompt": "tok", "max_tokens": 3})

    frames = captured.progress
    # `seq` is "strictly increasing per (request_id, attempt)" and is stamped on
    # the loop, so send order and seq order cannot disagree.
    assert [f.seq for f in frames] == list(range(1, len(frames) + 1))
    assert {f.request_id for f in frames} == {"pgw1576"}

    deltas = [f for f in frames if f.content_type == "text/plain"]
    assert [f.data for f in deltas] == [b"tok-0 ", b"tok-1 ", b"tok-2 "]

    result = captured.result
    assert result.status == pb.JOB_STATUS_OK
    decoded = msgspec.msgpack.decode(result.inline, type=streaming.Completion)
    assert decoded.text == "tok-0 tok-1 tok-2 " and decoded.tokens == 3


def test_the_ctx_event_lane_is_alive_on_the_same_seam(streaming: ModuleType) -> None:
    """The adjacent pgw#1576 finding: the v2 worker wired NO emitter, so
    `ctx.progress` hit `_emit_event`'s "no emitter configured" branch on every
    pod and the hub's liveness sweep read positions nobody sent."""
    captured = _CapturedWorker()
    captured.run("complete", {"prompt": "p", "max_tokens": 2})

    events = [
        msgspec.json.decode(f.data)
        for f in captured.progress
        if f.content_type == EVENT_CONTENT_TYPE
    ]
    progress = [e for e in events if e["type"] == "request.progress"]
    assert [e["payload"]["step"] for e in progress] == [1, 2]
    assert progress[-1]["payload"]["stage"] == "decode"
    assert progress[-1]["payload"]["progress"] == 1.0


# -- 5. the manifest ---------------------------------------------------------


def _rows() -> dict:
    from gen_worker.discovery.entrypoints_v2 import discover_entrypoints

    sys.path.insert(0, str(FIXTURES))
    try:
        return {row["name"]: row for row in discover_entrypoints(MODULE)}
    finally:
        sys.path.remove(str(FIXTURES))


def test_publish_reports_the_stream_without_running_the_body() -> None:
    rows = _rows()

    streamed = rows["complete"]
    assert streamed["incremental_output"] is True
    # The hub decodes `delta_output_schema` on the SAME `manifestFunction` an
    # `entrypoints[]` row lands in, so this reaches
    # `endpoint_function_schemas.delta_output_schema` with no hub change.
    delta = streamed["delta_output_schema"]
    assert delta["$ref"] == "#/$defs/TokenDelta"
    assert delta["$defs"]["TokenDelta"]["properties"]["text"]["type"] == "string"
    # ...and the terminal schema is untouched: a non-streaming caller still
    # reads one struct.
    assert set(
        streamed["output_schema"]["$defs"]["Completion"]["properties"]
    ) == {"text", "tokens"}

    caption_delta = rows["caption"]["delta_output_schema"]
    assert "finished" in caption_delta["$defs"]["ItemDelta"]["properties"]

    # Two shapes, one handler: a DISCRIMINATED anyOf, so a client reading the
    # manifest knows both frames and how to tell them apart.
    union = rows["narrate"]["delta_output_schema"]
    assert union["discriminator"]["propertyName"] == "type"
    assert set(union["discriminator"]["mapping"]) == {"TokenDelta", "ItemDelta"}

    plain = rows["silent"]
    assert plain["incremental_output"] is False
    assert "delta_output_schema" not in plain


# -- 6. the tombstone --------------------------------------------------------


@pytest.mark.parametrize(
    "name, successor",
    [
        ("IncrementalTokenDelta", "gen_worker.TokenDelta"),
        ("BatchItemDelta", "gen_worker.ItemDelta"),
    ],
)
def test_the_deleted_names_now_name_their_successor(name: str, successor: str) -> None:
    """They were deleted BY OMISSION — no successor line, so an author got a
    bare ImportError and the gap read as an oversight instead of a ruling."""
    import gen_worker
    from gen_worker.v1_deleted import V1SdkDeleted

    with pytest.raises(V1SdkDeleted) as excinfo:
        getattr(gen_worker, name)
    assert successor in str(excinfo.value)


def test_the_transformers_helper_imports_instead_of_refusing() -> None:
    """`iter_transformers_text_deltas` came back verbatim — same name, same
    signature — so joycaption's call site ports unchanged."""
    import gen_worker

    assert callable(gen_worker.iter_transformers_text_deltas)
