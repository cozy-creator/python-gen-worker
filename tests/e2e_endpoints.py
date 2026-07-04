"""Endpoint module served by the fake-scheduler e2e suite (not a test file)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator

import msgspec

from gen_worker import Hub, RequestContext, ValidationError, endpoint


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


@endpoint
class E2EEndpoint:
    def echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        ctx.raise_if_cancelled()
        if (data.text or "").strip().lower() == "marco":
            return EchoOut(response="polo")
        raise ValidationError(f"expected 'marco', got {data.text!r}")

    async def stream3(self, ctx: RequestContext, data: EchoIn) -> AsyncIterator[EchoOut]:
        for i in range(3):
            ctx.raise_if_cancelled()
            yield EchoOut(response=f"chunk-{i}")

    async def slow(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        await asyncio.sleep(30.0)
        return EchoOut(response="late")

    def sleepy(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        time.sleep(0.5)
        return EchoOut(response="done")


@endpoint(model=Hub("e2e/tiny"))
class ModelBoundEndpoint:
    """Tensorhub-bound endpoint: only becomes available after ModelOp LOAD."""

    def setup(self, model: str) -> None:
        self.model_path = model

    def model_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        weights = Path(self.model_path) / "model.safetensors"
        return EchoOut(response=weights.read_text())
