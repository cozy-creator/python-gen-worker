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
    lane,
)
from gen_worker.demand import MiB, const
from gen_worker.models import SDXL
#: THE REAL RATIFIED PAIR (pgw#1621). A lane is `(topology, quant)`; both
#: halves are documents in the vendored `spec/v2` corpus, so this fixture
#: cannot invent one. The v1 constant it replaces is deleted.
SDXL_DIFFUSERS_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")


class In(msgspec.Struct):
    prompt: str = ""


class Out(msgspec.Struct):
    text: str


class FakePipeline:
    """A module-level pipeline class, so the CONTROL arm resolves the way every pytorch endpoint in the fleet does."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "FakePipeline":
        return cls()


class GgufModel(
    Model[SDXL],
    lanes={SDXL_DIFFUSERS_BF16: lane(request=const(MiB(64)))},
    self_loading="served by llama-server, which self-loads a GGUF; the "
                 "block-quantized container is the external-binary class "
                 "the streaming engine refuses by design",
):
    """llama.cpp / GGUF — `qwen3.6-27b-mtp-gguf`'s shape."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(LlamaServer(
            n_ctx=32768,
            extra_args=["--alias", "qwen3.6-27b-mtp-gguf", "-ngl", "99",
                        "--cache-type-k", "q8_0", "-fa", "on"],
        ))


class VllmModel(
    Model[SDXL],
    lanes={SDXL_DIFFUSERS_BF16: lane(request=const(MiB(64)))},
    self_loading="served by vLLM, which self-loads the checkpoint directory; "
                 "ctx.load is never called",
):
    """vLLM / fp8 — `qwen3.6-35b-a3b`'s shape."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(VllmServer(
            extra_args=["--max-model-len", "16384",
                        "--gpu-memory-utilization", "0.94"],
        ))


class PytorchModel(
    Model[SDXL],
    lanes={SDXL_DIFFUSERS_BF16: lane(request=const(MiB(64)))},
):
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
