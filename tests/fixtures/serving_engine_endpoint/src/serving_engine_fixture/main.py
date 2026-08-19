"""pgw#1421 fixture: an ENGINE-HOSTED endpoint, written the way the two real
consumers (`qwen3.6-27b-mtp-gguf`, `qwen3.6-35b-a3b`) will be after the wave.

The v1 shape this replaces, verbatim from the shipped endpoints::

    from gen_worker.runtimes.server import ServerHandle, llama_server
    @endpoint(model=Slot(str, layouts_undeclarable="GGUF: ..."), ...)
    class QwenMTPCompletion:
        def setup(self, model: str) -> None:
            self._handle = llama_server(model, extra_args=[...]).start()
        def shutdown(self) -> None:
            self._handle.stop()

The v2 shape below states the same facts in the wave's vocabulary: the model
class is the stateful half, `lanes=()` is eager-permanent (a self-loading
external binary performs no pytorch lane load), `ctx.engine` is the boot seam,
and there is no `shutdown` at all — the host stops the engine structurally.

Two engine arms and a CONTROL: llama.cpp, vLLM, and a plain pytorch model
that must read as hosting NO engine. The arm that actually BOOTS an engine
lives in `serving_engine_host_endpoint` — `EndpointHost.setup()` loads every
referenced model class, so declaring all four here would exec `llama-server`.
"""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import (
    LlamaServer,
    LoadContext,
    Model,
    RequestContext,
    VllmServer,
    entrypoint,
)
from gen_worker.models import SDXL


class In(msgspec.Struct):
    prompt: str = ""


class Out(msgspec.Struct):
    text: str


class FakePipeline:
    """A module-level pipeline class, so the CONTROL arm resolves the way
    every pytorch endpoint in the fleet does."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "FakePipeline":
        return cls()


class GgufModel(
    Model[SDXL],
    lanes=(),
    self_loading="served by llama-server, which self-loads a GGUF; the "
                 "block-quantized container is the external-binary class "
                 "the streaming engine refuses by design",
):
    """llama.cpp / GGUF — `qwen3.6-27b-mtp-gguf`'s shape.

    `self_loading=` (pgw#1431 fix (b)) is not a second declaration beside
    `ctx.engine`: an engine-hosted model is self-loading BY CONSTRUCTION, and
    the marker is where that fact reaches the publish path.
    """

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(LlamaServer(
            n_ctx=32768,
            extra_args=["--alias", "qwen3.6-27b-mtp-gguf", "-ngl", "99",
                        "--cache-type-k", "q8_0", "-fa", "on"],
        ))


class VllmModel(
    Model[SDXL],
    lanes=(),
    self_loading="served by vLLM, which self-loads the checkpoint directory; "
                 "ctx.load is never called",
):
    """vLLM / fp8 — `qwen3.6-35b-a3b`'s shape."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(VllmServer(
            extra_args=["--max-model-len", "16384",
                        "--gpu-memory-utilization", "0.94"],
        ))


class PytorchModel(Model[SDXL], lanes=()):
    """THE CONTROL: boots no engine, so the census must not name it."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(FakePipeline)


@entrypoint
def chat(ctx: RequestContext, payload: In, model: GgufModel) -> Out:
    return Out(text=model.engine.base_url)


@entrypoint
def complete(ctx: RequestContext, payload: In, model: VllmModel) -> Out:
    return Out(text=model.engine.base_url)


@entrypoint
def draw(ctx: RequestContext, payload: In, model: PytorchModel) -> Out:
    return Out(text=type(model.pipe).__name__)
