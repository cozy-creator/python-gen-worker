"""th#959: ``compile=`` + ``warmup=NoWarmup`` is refused at decoration.

compiled graphs are minted and proven by warmup execution (self-mint boots a
warmup; adoption proof requires the warmup to hit the compiled graph). A
NoWarmup class therefore can never arm a compiled graph — live, the combination struck
a qwen-edit release broken in 3 pods while the auto lane kept picking the
mandatory w8a8+compiled rung. The contradiction now fails at import, the
earliest possible moment. A custom ``warmup()`` method resolves it (proof
runs through that), and NoWarmup without compile= stays legal.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import (
    Compile,
    NoWarmup,
    RequestContext,
    Resources,
    Slot,
    endpoint,
    worker_function,
)


class _In(msgspec.Struct):
    prompt: str = "warm"


class _Out(msgspec.Struct):
    ok: bool = True


def _compile() -> Compile:
    return Compile(
        family="sdxl", targets=("unet",), shapes=((1024, 1024),), text_len=77
    )


def test_compile_plus_nowarmup_refused_at_decoration() -> None:
    with pytest.raises(TypeError, match="th#959"):

        @endpoint(
            models={"pipeline": Slot(str)},
            resources=Resources(gpu=True),
            compile=_compile(),
            warmup=NoWarmup("edit needs a caller-supplied source image"),
        )
        class Contradiction:
            def setup(self, pipeline: str) -> None:
                self.pipeline = pipeline

            @worker_function()
            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()


def test_nowarmup_without_compile_stays_legal() -> None:
    @endpoint(
        models={"pipeline": Slot(str)},
        resources=Resources(gpu=True),
        warmup=NoWarmup("upstream stack manages its own capture"),
    )
    class OptedOut:
        def setup(self, pipeline: str) -> None:
            self.pipeline = pipeline

        @worker_function()
        def generate(self, ctx: RequestContext, p: _In) -> _Out:
            return _Out()


def test_custom_warmup_method_resolves_the_contradiction() -> None:
    @endpoint(
        models={"pipeline": Slot(str)},
        resources=Resources(gpu=True),
        compile=_compile(),
        warmup=NoWarmup("derived plan opted out; custom warmup proves compiled graphs"),
    )
    class CustomWarmup:
        def setup(self, pipeline: str) -> None:
            self.pipeline = pipeline

        def warmup(self) -> None:
            pass

        @worker_function()
        def generate(self, ctx: RequestContext, p: _In) -> _Out:
            return _Out()
