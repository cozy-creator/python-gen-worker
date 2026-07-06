"""Endpoint module served by the fake-scheduler e2e suite (not a test file)."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import msgspec

from gen_worker import Hub, RequestContext, ValidationError, endpoint

# Cross-thread signals for the GPU-slot-yield tests (#382). The fake-scheduler
# suite runs the worker in-process, so tests coordinate through these directly.
SLOT_PROBE_STARTED = threading.Event()
SLOT_PEER_RAN = threading.Event()


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

    def slot_probe(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        """Holds the GPU slot, then waits for `slot_peer` INSIDE the yielded
        window. Only completes if the slot is actually released during
        uploads — with a held slot the peer can never run and this times out."""
        SLOT_PROBE_STARTED.set()
        with ctx._gpu_slot_yielded():
            ok = SLOT_PEER_RAN.wait(timeout=10.0)
        return EchoOut(response="overlapped" if ok else "starved")

    def slot_peer(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        SLOT_PEER_RAN.set()
        return EchoOut(response="peer-done")


@endpoint(model=Hub("e2e/tiny"))
class ModelBoundEndpoint:
    """Tensorhub-bound endpoint: only becomes available after ModelOp LOAD."""

    def setup(self, model: str) -> None:
        self.model_path = model

    def model_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        weights = Path(self.model_path) / "model.safetensors"
        return EchoOut(response=weights.read_text())
