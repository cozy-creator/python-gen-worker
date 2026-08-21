"""The streamed-chunk vocabulary — what ctx.emit puts on the wire. The wire has two channels with different promises: JobProgress deltas are ordered, live, never persisted and DROPPABLE by contract; the returned struct is the single authoritative JobResult (an async-generator entrypoint is refused at declaration — Python forbids `return <value>` inside one). TokenDelta = more of ONE output, framed raw UTF-8 text/plain (subscribers concatenate payload.delta); ItemDelta = which of N outputs advanced, framed application/json. Framing must be UTF-8 text, a wire constraint: the hub renders every non-audio/* chunk as payload["delta"] = string(data) inside a JSON SSE envelope, so binary framing arrives as replacement-character mojibake — binary rides the audio/* arm (base64'd by the hub as audio_chunk) or the terminal result."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Tuple

import msgspec

JSON_CONTENT_TYPE = "application/json"
TEXT_CONTENT_TYPE = "text/plain"


class Delta(msgspec.Struct, frozen=True, kw_only=True, tag_field="type", tag=True):
    """Base for a streamed chunk."""

    def frame(self) -> Tuple[bytes, str]:
        """This chunk on the wire: ``(data, content_type)``."""
        return msgspec.json.encode(self), JSON_CONTENT_TYPE


class TokenDelta(Delta, frozen=True, kw_only=True):
    """More of ONE output — the successor to v1's ``IncrementalTokenDelta``."""

    text: str = ""

    def frame(self) -> Tuple[bytes, str]:
        return self.text.encode("utf-8"), TEXT_CONTENT_TYPE


class ItemDelta(Delta, frozen=True, kw_only=True):
    """WHICH of N outputs advanced — the successor to v1's ``BatchItemDelta``."""

    index: int = 0
    total: int = 0
    item_id: str = ""
    text: str = ""
    finished: bool = False
    error: str = ""


# How often the stream RE-LOOKS at whether the generate thread is still alive.
# It ends nothing: every expiry is a re-check, and only the thread's death ends
# the stream. Shorter costs a wakeup, longer delays noticing a dead producer.
_RECHECK_CADENCE_S = 0.5


def frame_of(chunk: Any) -> Tuple[bytes, str]:
    """``(data, content_type)`` for any chunk — the one encoder."""
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
    timeout: float | None = None,
    join_timeout: float = 2.0,
    streamer_cls: type[Any] | None = None,
    decode_kwargs: Mapping[str, Any] | None = None,
) -> Iterator[str]:
    """Stream text chunks out of a ``transformers`` ``model.generate`` call.

    ``timeout`` is a RE-CHECK CADENCE, not a bound on the work (th#2267 §6).
    It used to default to 0.5 s and be handed straight to the streamer, where
    expiring it raised a bare ``queue.Empty`` out of this generator — untyped,
    unattributable, and triggered by the most ordinary thing a decode does:
    prefill. A long prompt or a large model spends whole seconds before the
    first token, and half a second of that killed the stream. The opposite
    spelling was no better: ``timeout=None`` (what joycaption passes) blocks
    the consumer FOREVER when the generate thread dies, because a dead thread
    never enqueues the end sentinel.

    Neither shape asks the question that matters. The question is whether the
    thread producing tokens is still alive, so that is what decides: a poll
    that expires while the thread is running is prefill and the wait continues;
    a poll that expires once the thread is gone is the end, and any exception
    it died of is re-raised. How long the decode takes is governed elsewhere,
    by the in-call progress gate that any ``ctx`` event re-arms.
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
            pass

    # Always poll. A blocking streamer cannot notice a dead producer, and a
    # streamer that gives up cannot survive a prefill; polling separates them.
    poll_s = _RECHECK_CADENCE_S if timeout is None else max(float(timeout), 0.0)
    streamer = streamer_cls(
        tokenizer, skip_prompt=bool(skip_prompt), timeout=poll_s,
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
        stream = iter(streamer)
        while True:
            try:
                chunk = next(stream)
            except StopIteration:
                break
            except queue.Empty:
                # NOT a verdict. It says only that no token arrived inside one
                # poll, which is what prefill looks like from here.
                if thread.is_alive():
                    if cancel_checker is not None and cancel_checker():
                        break
                    continue
                # The producer is gone and the queue is drained: this is the
                # end of the stream, however it ended. If it ended badly the
                # exception is below, typed and attributable.
                break
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
