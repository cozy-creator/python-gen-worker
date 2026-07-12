from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Union

import msgspec


# ============================================================================
# Typed streaming signals.
#
# Tenant streaming handlers (``async def fn(...) -> AsyncIterator[...]``)
# yield these — or their own msgspec.Struct delta types. The executor
# encodes each yielded item as one JobProgress chunk on the wire.
# ============================================================================


class IncrementalTokenDelta(msgspec.Struct, frozen=True, kw_only=True):
    """One incremental token (or token group) emitted by a streaming endpoint.

    ``item_id`` lets a single request multiplex deltas across distinct
    logical outputs.
    """

    text: str = ""
    item_id: str | None = None


class BatchItemDelta(msgspec.Struct, frozen=True, kw_only=True):
    """One item of a multi-item (batch) streaming response.

    First-class struct replacing ad-hoc per-endpoint delta shapes and
    magic field-name peeling. ``chunk`` carries the item's binary payload
    (audio frame, encoded image, ...); ``content_type`` names its encoding.
    ``finished`` marks the item's terminal delta; ``error`` (non-empty)
    fails the item without failing the whole batch.
    """

    index: int = 0
    total: int = 0
    item_id: str | None = None
    finished: bool = False
    error: str = ""
    chunk: bytes = b""
    content_type: str = "application/octet-stream"


class TokenUsage(msgspec.Struct, frozen=True, kw_only=True):
    """Terminal usage signal for token-streaming endpoints.

    Yield one at the end of the stream; the executor folds it into the
    terminal :class:`StreamResult` (billing reads it there) and forwards it
    live as a JSON chunk.
    """

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0


class Done(msgspec.Struct, frozen=True, kw_only=True):
    """Optional end-of-stream marker. Yielding it emits no chunk — the
    dispatcher's ``_encode_chunk`` treats it as a no-op (see executor.py).
    A stream ends naturally when the handler's generator exhausts; yield
    ``Done()`` if you want to terminate explicitly without yielding a
    final partial item.
    """


class Error(msgspec.Struct, frozen=True, kw_only=True):
    """Terminal error signal. Yield Error(message=...) to fail the stream.

    Prefer raising a typed ``gen_worker`` exception (``ValidationError`` /
    etc.) when you have one; Error() is the fallback for engine-internal
    errors the tenant catches and reports inline.
    """

    message: str = ""


# ============================================================================
# Terminal stream output (gw#475): live deltas are droppable by contract, so
# the executor folds them into a StreamResult and serializes it as the
# completed request's authoritative output.
# ============================================================================


class StreamItem(msgspec.Struct, frozen=True, kw_only=True):
    """One accumulated batch item in the terminal output.

    ``content`` is ``str`` for ``text/*`` items (lossless through the JSON
    SSE/webhook decode surfaces), raw ``bytes`` otherwise.
    """

    item_id: str | None = None
    index: int = 0
    # str for text/*, bytes otherwise (msgspec typed decode can't take a
    # str|bytes union; msgpack distinguishes them on the wire regardless).
    content: Any = b""
    content_type: str = "application/octet-stream"
    error: str = ""


class StreamResult(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Terminal output of a stream-mode request.

    ``text`` is the concatenated token stream (default item); ``texts``
    holds per-``item_id`` streams when multiplexed; ``items`` the finished
    batch items; ``usage`` the endpoint's TokenUsage signal. ``truncated``
    marks content dropped past the accumulation cap (metadata is always
    kept). Empty fields are omitted on the wire — a plain token stream
    encodes as exactly ``{"text": ...}``.
    """

    text: str = ""
    texts: dict[str, str] = {}
    items: list[StreamItem] = []
    usage: TokenUsage | None = None
    truncated: bool = False


# Binary/batch content past this cap is dropped from the terminal record
# (item metadata survives); token text is never dropped in practice.
_ACCUM_MAX_BYTES = 64 * 1024 * 1024


class StreamAccumulator:
    """Folds yielded stream signals into the terminal StreamResult."""

    def __init__(self, max_bytes: int = _ACCUM_MAX_BYTES) -> None:
        self._max = max_bytes
        self._size = 0
        self._texts: dict[str, list[str]] = {}
        self._items: dict[tuple[str | None, int], dict[str, Any]] = {}
        self._finished: list[StreamItem] = []
        self._usage: TokenUsage | None = None
        self._truncated = False

    def add(self, item: Any) -> None:
        if isinstance(item, IncrementalTokenDelta):
            if self._book(len(item.text)):
                self._texts.setdefault(item.item_id or "", []).append(item.text)
        elif isinstance(item, BatchItemDelta):
            key = (item.item_id, item.index)
            rec = self._items.get(key)
            if rec is None:
                rec = {"chunks": [], "content_type": item.content_type, "error": ""}
                self._items[key] = rec
            if item.chunk and self._book(len(item.chunk)):
                rec["chunks"].append(item.chunk)
            if item.content_type != "application/octet-stream":
                rec["content_type"] = item.content_type
            if item.error:
                rec["error"] = item.error
            if item.finished or item.error:
                # Only finished/errored items reach the terminal record.
                raw = b"".join(rec["chunks"])
                content: Union[str, bytes] = raw
                if rec["content_type"].startswith("text/"):
                    content = raw.decode("utf-8", errors="replace")
                self._finished.append(StreamItem(
                    item_id=key[0], index=key[1], content=content,
                    content_type=rec["content_type"], error=rec["error"],
                ))
                self._items.pop(key, None)
        elif isinstance(item, TokenUsage):
            self._usage = item

    def _book(self, n: int) -> bool:
        if self._size + n > self._max:
            self._truncated = True
            return False
        self._size += n
        return True

    def result(self) -> StreamResult | None:
        """The terminal record, or None when nothing accumulated (an empty
        stream keeps an output-less OK result)."""
        texts = {k: "".join(v) for k, v in self._texts.items()}
        text = texts.pop("", "")
        if not text and len(texts) == 1:
            text = next(iter(texts.values()))
        if not (text or texts or self._finished or self._usage or self._truncated):
            return None
        return StreamResult(
            text=text,
            texts=texts,
            items=list(self._finished),
            usage=self._usage,
            truncated=self._truncated,
        )


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
    """
    Stream generation text chunks from a Hugging Face model.generate call.

    This wraps `transformers.TextIteratorStreamer` and runs `model.generate`
    in a background thread. The returned iterator yields progressive text
    chunks suitable for worker incremental-output paths.

    Notes:
    - The streamer may emit chunked text segments (not guaranteed single-token).
    - If `cancel_checker` is provided and transformers stopping criteria are
      available, generation is stopped cooperatively.
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
            raise RuntimeError("transformers TextIteratorStreamer is unavailable") from exc
        streamer_cls = TextIteratorStreamer

    # Best-effort cooperative cancellation for long generations.
    if cancel_checker is not None:
        _checker = cancel_checker
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _CancelStopCriteria(StoppingCriteria):
                def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
                    try:
                        return bool(_checker())
                    except Exception:
                        return False

            existing = local_generation_kwargs.get("stopping_criteria")
            cancel_criteria = _CancelStopCriteria()
            if existing is None:
                local_generation_kwargs["stopping_criteria"] = StoppingCriteriaList([cancel_criteria])
            elif isinstance(existing, StoppingCriteriaList):
                existing.append(cancel_criteria)
            else:
                local_generation_kwargs["stopping_criteria"] = StoppingCriteriaList(list(existing) + [cancel_criteria])
        except Exception:
            # Keep streaming behavior even if stopping-criteria wiring isn't available.
            pass

    streamer = streamer_cls(
        tokenizer,
        skip_prompt=bool(skip_prompt),
        timeout=timeout,
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

    thread = threading.Thread(target=_run_generate, daemon=True, name="hf-generate-stream")
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
