from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Union

import msgspec


# ============================================================================
# Typed token-streaming signals (#321 wire-side, #273 SDK-side surface).
#
# These are the Python-level signal types tenant code yields from a
# ``@batched_inference.function`` async generator. They're thin msgspec
# Structs that the BatchedWorker dispatcher maps onto the existing
# IncrementalTokenDelta / IncrementalTokenStreamDone / IncrementalTokenStreamError
# proto messages on the wire (see worker.py ``_emit_incremental_*_typed``).
#
# The wire format already exists — these classes only give tenants a stable
# import:
#
#     from gen_worker import IncrementalTokenDelta, Done, Error
#
# instead of having to define their own msgspec.Struct shapes per endpoint.
# Endpoints that need richer payloads (image bytes, structured tool calls,
# etc.) can still declare their own struct types — the dispatcher checks
# ``isinstance(item, _SIGNAL_TYPES)`` first and falls through to the legacy
# duck-typed path otherwise.
# ============================================================================


class IncrementalTokenDelta(msgspec.Struct, frozen=True, kw_only=True):
    """One incremental token (or token group) emitted by a streaming endpoint.

    Tenants yield instances of this from an
    ``async def fn(...) -> AsyncIterator[Signal]`` method on a
    ``@batched_inference`` class. The worker dispatcher routes ``text``
    into the wire ``delta_text`` slot and emits an ``IncrementalTokenDelta``
    proto message back to the orchestrator. ``item_id`` lets a single
    request multiplex deltas across distinct logical outputs (rarely used
    for chat / captioning; reserved for batched multi-output flows).
    """

    text: str = ""
    item_id: str | None = None


class Done(msgspec.Struct, frozen=True, kw_only=True):
    """End-of-stream marker. Yield exactly one Done() to terminate cleanly.

    The dispatcher converts this into an ``IncrementalTokenStreamDone``
    proto message and the terminal ``JobExecutionResult(success=True)``.
    """


class Error(msgspec.Struct, frozen=True, kw_only=True):
    """Terminal error signal. Yield Error(message=...) to fail the stream.

    Equivalent to raising — the dispatcher emits an
    ``IncrementalTokenStreamError`` proto message and the terminal
    ``JobExecutionResult(success=False)``. Prefer raising a typed
    ``gen_worker`` exception (``ValidationError`` / ``ResourceError`` /
    etc.) when you have one; Error() is the fallback for engine-internal
    errors the tenant catches and reports inline.
    """

    message: str = ""


# Tuple form for runtime ``isinstance`` checks in the dispatcher.
_SIGNAL_TYPES: tuple[type, ...] = (IncrementalTokenDelta, Done, Error)
TokenStreamSignal = Union[IncrementalTokenDelta, Done, Error]


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
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
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
