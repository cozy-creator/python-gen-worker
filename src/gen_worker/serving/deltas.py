"""The streamed-chunk vocabulary (pgw#1576) — what ``ctx.emit`` puts on the wire.

An entrypoint declares its chunk type once, on the decorator, and emits chunks
of that type while it works::

    @entrypoint(streams=TokenDelta)
    async def complete(ctx: RequestContext, payload: CompletionInput,
                       model: QwenModel) -> Completion:
        parts: list[str] = []
        async for token in engine.stream(payload.prompt):
            ctx.raise_if_cancelled()
            parts.append(token)
            ctx.emit(TokenDelta(text=token))
        return Completion(text="".join(parts))

The function is an ORDINARY ``async def`` (or ``def``) returning a
``msgspec.Struct``, because the wire has two channels and they carry different
promises: ``JobProgress`` deltas are ordered, live, never persisted and
DROPPABLE by contract, while the returned struct is the single authoritative
``JobResult``. An async-generator entrypoint could express only the first —
Python forbids ``return <value>`` inside one — so it is refused at declaration
with the migration line, and the terminal is a real ``return``.

**TWO delta types, because they answer two different questions.**

* :class:`TokenDelta` — MORE OF ONE OUTPUT. One completion, growing. Frames as
  raw UTF-8 ``text/plain``, so an SSE subscriber concatenates
  ``payload.delta`` and has the answer.
* :class:`ItemDelta` — WHICH OF N OUTPUTS ADVANCED. A batch, item by item, each
  with its own index, terminal flag and per-item error. Frames as
  ``application/json``.

Collapsing them into one struct would hand every token five fields it must
leave at zero. An author who needs a third shape subclasses :class:`Delta`
(default framing: ``application/json``) or overrides :meth:`Delta.frame`.

**One handler may declare several shapes** — ``@entrypoint(streams=(TokenDelta,
ItemDelta))`` — and the manifest publishes a discriminated ``anyOf`` over them.
Note before reaching for it: per-token progress WITHIN one batch item is
``ItemDelta(index=…, text=…, finished=False)``, which is one shape and what
joycaption's own v1 body did (it wrapped every ``iter_transformers_text_deltas``
piece in a ``BatchItemDelta``, never mixing the two). The union is for a stream
that genuinely carries both. A consumer tells them apart by ``content_type``
(``text/plain`` is token text) or by the ``type`` tag inside a JSON frame.

**Framing is UTF-8 text, and that is a wire constraint rather than a
preference.** The hub renders every non-``audio/*`` chunk as
``payload["delta"] = string(data)`` inside a JSON SSE envelope
(``internal/orchestrator/http/requests.go``), so binary framing arrives as
replacement-character mojibake. v1's ``BatchItemDelta.chunk: bytes`` +
``application/x-batch-item+msgpack`` was undeliverable for exactly that reason
and is not restored: binary rides the ``audio/*`` arm (base64'd by the hub as
``audio_chunk``) or the terminal result.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Tuple

import msgspec

#: What a chunk frames to when its type states nothing else.
JSON_CONTENT_TYPE = "application/json"
#: The concatenable token stream every SSE consumer already reads.
TEXT_CONTENT_TYPE = "text/plain"


class Delta(msgspec.Struct, frozen=True, kw_only=True, tag_field="type", tag=True):
    """Base for a streamed chunk. Frames as JSON unless a subclass says else.

    TAGGED on ``type`` (each subclass's own class name), for two reasons and
    neither is decoding: a JSON frame tells a consumer WHICH shape it is
    without a second channel, and ``streams=(A, B)`` — one handler, two shapes,
    which joycaption needs — can only be published as a discriminated
    ``anyOf`` if the arms carry a tag. An untagged multi-arm declaration is
    refused at import rather than published as an ambiguous schema.
    """

    def frame(self) -> Tuple[bytes, str]:
        """This chunk on the wire: ``(data, content_type)``."""
        return msgspec.json.encode(self), JSON_CONTENT_TYPE


class TokenDelta(Delta, frozen=True, kw_only=True):
    """More of ONE output — the successor to v1's ``IncrementalTokenDelta``.

    ``text`` is the increment, never the accumulation: the caller concatenates.
    The complete text belongs in the entrypoint's returned struct, which is what
    a non-streaming caller (and the persisted request record) reads.
    """

    text: str = ""

    def frame(self) -> Tuple[bytes, str]:
        return self.text.encode("utf-8"), TEXT_CONTENT_TYPE


class ItemDelta(Delta, frozen=True, kw_only=True):
    """WHICH of N outputs advanced — the successor to v1's ``BatchItemDelta``.

    ``finished`` marks an item's terminal delta and ``error`` (non-empty) fails
    that ITEM without failing the request; the entrypoint still returns a
    terminal struct describing the whole batch.
    """

    index: int = 0
    total: int = 0
    item_id: str = ""
    text: str = ""
    finished: bool = False
    error: str = ""


def frame_of(chunk: Any) -> Tuple[bytes, str]:
    """``(data, content_type)`` for any chunk — the one encoder.

    A chunk that defines ``frame()`` frames itself; anything else is JSON. Kept
    a free function so the emitter never has to care which it holds.
    """
    framer = getattr(chunk, "frame", None)
    if callable(framer):
        data, content_type = framer()
        return bytes(data), str(content_type)
    return msgspec.json.encode(chunk), JSON_CONTENT_TYPE


def iter_transformers_text_deltas(
    *,
    model: Any,
    tokenizer: Any,
    generation_kwargs: Mapping[str, Any],
    cancel_checker: Callable[[], bool] | None = None,
    skip_prompt: bool = True,
    timeout: float | None = 0.5,
    join_timeout: float = 2.0,
    streamer_cls: type[Any] | None = None,
    decode_kwargs: Mapping[str, Any] | None = None,
) -> Iterator[str]:
    """Stream text chunks out of a ``transformers`` ``model.generate`` call.

    Wraps ``TextIteratorStreamer`` and runs ``generate`` on a background thread,
    yielding progressive text. ``cancel_checker`` is wired into the generation's
    stopping criteria when transformers exposes them, so a cancelled request
    stops the decode rather than draining it. An exception raised inside the
    generate thread is re-raised here.

    Chunks are text SEGMENTS, not guaranteed single tokens.
    """
    if model is None:
        raise ValueError("model is required")
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    if generation_kwargs is None:
        raise ValueError("generation_kwargs is required")

    local_decode_kwargs = dict(decode_kwargs or {})
    local_generation_kwargs = dict(generation_kwargs)

    if streamer_cls is None:
        try:
            from transformers import TextIteratorStreamer
        except Exception as exc:  # pragma: no cover - guarded by endpoint deps
            raise RuntimeError(
                "transformers TextIteratorStreamer is unavailable"
            ) from exc
        streamer_cls = TextIteratorStreamer

    if cancel_checker is not None:
        _checker = cancel_checker
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _CancelStopCriteria(StoppingCriteria):
                def __call__(self, _ids: Any, _scores: Any, **_kw: Any) -> bool:
                    try:
                        return bool(_checker())
                    except Exception:
                        return False

            existing = local_generation_kwargs.get("stopping_criteria")
            cancel_criteria = _CancelStopCriteria()
            if existing is None:
                local_generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [cancel_criteria]
                )
            elif isinstance(existing, StoppingCriteriaList):
                existing.append(cancel_criteria)
            else:
                local_generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    list(existing) + [cancel_criteria]
                )
        except Exception:
            # Streaming still works without cooperative stopping criteria.
            pass

    streamer = streamer_cls(
        tokenizer, skip_prompt=bool(skip_prompt), timeout=timeout,
        **local_decode_kwargs,
    )
    local_generation_kwargs["streamer"] = streamer

    errq: "queue.Queue[BaseException]" = queue.Queue(maxsize=1)

    def _run_generate() -> None:
        try:
            model.generate(**local_generation_kwargs)
        except BaseException as exc:
            try:
                errq.put_nowait(exc)
            except Exception:
                pass

    thread = threading.Thread(
        target=_run_generate, daemon=True, name="hf-generate-stream"
    )
    thread.start()
    try:
        for chunk in streamer:
            if cancel_checker is not None and cancel_checker():
                break
            text = str(chunk or "")
            if text:
                yield text
    finally:
        thread.join(timeout=max(0.0, float(join_timeout)))

    if not errq.empty():
        raise errq.get()


__all__ = [
    "JSON_CONTENT_TYPE",
    "TEXT_CONTENT_TYPE",
    "Delta",
    "ItemDelta",
    "TokenDelta",
    "frame_of",
    "iter_transformers_text_deltas",
]
