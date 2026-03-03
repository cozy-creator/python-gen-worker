from __future__ import annotations

import queue
from typing import Any

import pytest

from gen_worker.transformers_streaming import iter_transformers_text_deltas


class FakeStreamer:
    def __init__(self, _tokenizer: Any, timeout: float | None = None, **_kwargs: Any) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self._timeout = max(0.01, float(timeout or 0.1))

    def push(self, chunk: str) -> None:
        self._q.put_nowait(chunk)

    def close(self) -> None:
        self._q.put_nowait(None)

    def __iter__(self):
        while True:
            try:
                item = self._q.get(timeout=self._timeout)
            except queue.Empty:
                return
            if item is None:
                return
            yield item


class FakeModel:
    def __init__(self) -> None:
        self.seen_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> None:
        self.seen_kwargs = kwargs
        streamer = kwargs["streamer"]
        streamer.push("hello")
        streamer.push(" world")
        streamer.close()


class ErrorModel:
    def generate(self, **_kwargs: Any) -> None:
        raise RuntimeError("boom")


def test_iter_transformers_text_deltas_streams_chunks() -> None:
    model = FakeModel()
    out = list(
        iter_transformers_text_deltas(
            model=model,
            tokenizer=object(),
            generation_kwargs={"max_new_tokens": 8},
            streamer_cls=FakeStreamer,
        )
    )
    assert out == ["hello", " world"]
    assert model.seen_kwargs is not None
    assert model.seen_kwargs["max_new_tokens"] == 8
    assert "streamer" in model.seen_kwargs


def test_iter_transformers_text_deltas_surfaces_generate_error() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        list(
            iter_transformers_text_deltas(
                model=ErrorModel(),
                tokenizer=object(),
                generation_kwargs={"max_new_tokens": 8},
                streamer_cls=FakeStreamer,
            )
        )
