"""Toy endpoints served by the harness's hub-double tests (P1/P2/P3/P6/P9).

No torch, no GPU, no real weights — generalized from ``tests/e2e_endpoints.py``
(#365) plus the gw#583/th#938 fixture endpoints, consolidated so every P-test
loads this one module. Cross-thread ``threading.Event`` globals coordinate
the GPU-slot-yield probes (the hub-double runs the worker in-process).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import msgspec

from gen_worker import Hub, RequestContext, Slot, ValidationError, endpoint
from gen_worker.api.streaming import StreamResult, TokenUsage
from gen_worker.families.base import FamilyDefaults, family


@family("harness-testfam")
class _ToyDefaults(FamilyDefaults, frozen=True):
    steps: int = 7


class EchoIn(msgspec.Struct):
    text: str = ""
    model: str = ""  # selected_by="model" target field for catalog-slot rows


class EchoOut(msgspec.Struct):
    response: str


# ---------------------------------------------------------------------------
# P1/P6: plain dispatch contract + GPU-slot-yield probes.
# ---------------------------------------------------------------------------

SLOT_PROBE_STARTED = threading.Event()
SLOT_PEER_RAN = threading.Event()
FINALIZE_PROBE_STARTED = threading.Event()
FINALIZE_PEER_RAN = threading.Event()


@endpoint
class Basics:
    def echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        ctx.raise_if_cancelled()
        if (data.text or "").strip().lower() == "marco":
            return EchoOut(response="polo")
        raise ValidationError(f"expected 'marco', got {data.text!r}")

    async def stream3(self, ctx: RequestContext, data: EchoIn) -> AsyncIterator[EchoOut]:
        for i in range(3):
            ctx.raise_if_cancelled()
            yield EchoOut(response=f"chunk-{i}")

    async def slow_stream(self, ctx: RequestContext, data: EchoIn) -> AsyncIterator[EchoOut]:
        """Yields slowly enough that a mid-stream cancel is observable (P6)."""
        for i in range(20):
            ctx.raise_if_cancelled()
            yield EchoOut(response=f"slow-chunk-{i}")
            await asyncio.sleep(0.2)

    async def slow(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        await asyncio.sleep(30.0)
        return EchoOut(response="late")

    def sleepy(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        time.sleep(0.5)
        return EchoOut(response="done")

    def slot_probe(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        SLOT_PROBE_STARTED.set()
        with ctx._gpu_slot_yielded():
            ok = SLOT_PEER_RAN.wait(timeout=10.0)
        return EchoOut(response="overlapped" if ok else "starved")

    def slot_peer(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        SLOT_PEER_RAN.set()
        return EchoOut(response="peer-done")


# ---------------------------------------------------------------------------
# P2: Tensorhub-bound (Hub sugar) — desired-residency round trip.
# ---------------------------------------------------------------------------


@endpoint(model=Hub("harness/residency-tiny"))
class ModelBoundEndpoint:
    def setup(self, model: str) -> None:
        self.model_path = model

    def model_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        weights = Path(self.model_path) / "model.safetensors"
        return EchoOut(response=weights.read_text())


# ---------------------------------------------------------------------------
# P3: Slot-declared endpoints (pgw#606/th#938 precedence + pgw#583 identity).
# ---------------------------------------------------------------------------

# Never delivered in ANY P3 test: if boot ever fetches these, the store finds
# no snapshot registered for them and fails fast (no real network involved).
BOOT_UNREACHABLE_PIPELINE = Hub("harness/boot-precedence-pipeline", tag="prod")
BOOT_UNREACHABLE_VAE = Hub("harness/boot-precedence-vae", tag="prod")


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=BOOT_UNREACHABLE_PIPELINE, default_config=_ToyDefaults()),
    "vae": Slot(str, default_checkpoint=BOOT_UNREACHABLE_VAE, default_config=_ToyDefaults()),
})
class SlotBootPrecedenceEndpoint:
    """Both slot defaults are tensorhub-sourced; neither may be boot-set-up
    from the code default — only a hub-stamped Hot DesiredInstance may."""

    def setup(self, pipeline: str, vae: str) -> None:
        self.pipeline_path = pipeline
        self.vae_path = vae

    def slot_boot_precedence(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        weights = Path(self.pipeline_path) / "model.safetensors"
        return EchoOut(response=weights.read_text())


DECLARED_PIPELINE = Hub("harness/slot-identity-declared", tag="prod")


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=DECLARED_PIPELINE, default_config=_ToyDefaults()),
})
class SlotIdentityFixedEndpoint:
    """FIXED slot (no selected_by=): a dispatched pick naming a DIFFERENT
    repo than the declared default must refuse (gw#583)."""

    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def slot_identity_fixed(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        resolved = ctx.slots["pipeline"]
        ref = resolved.ref
        return EchoOut(response=f"{ref.source}:{ref.path}:{ref.tag}#{ref.flavor}")


CATALOG_DEFAULT_PIPELINE = Hub("harness/slot-catalog-default", tag="prod")


@endpoint(models={
    "pipeline": Slot(
        str, selected_by="model", default_checkpoint=CATALOG_DEFAULT_PIPELINE,
        default_config=_ToyDefaults(),
    ),
})
class SlotIdentityCatalogEndpoint:
    """selected_by= catalog slot: a dispatched pick of a DIFFERENT repo is a
    legitimate explicit surface, not an identity mismatch."""

    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def slot_identity_catalog(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        resolved = ctx.slots["pipeline"]
        ref = resolved.ref
        return EchoOut(response=f"{ref.source}:{ref.path}:{ref.tag}#{ref.flavor}")


# ---------------------------------------------------------------------------
# P9: typed billing usage on JobMetrics, inline vs blob_ref by size alone.
# ---------------------------------------------------------------------------


@endpoint
class BillableEndpoint:
    def small_usage(self, ctx: RequestContext, data: EchoIn) -> StreamResult:
        return StreamResult(
            text="ok", usage=TokenUsage(prompt_tokens=12, cached_tokens=2, completion_tokens=5),
        )

    def large_usage(self, ctx: RequestContext, data: EchoIn) -> StreamResult:
        # >64KB msgpack-encoded -> executor's INLINE_RESULT_MAX_BYTES tips
        # this to the blob_ref path automatically (executor._serialize_output).
        return StreamResult(
            text="x" * 200_000,
            usage=TokenUsage(prompt_tokens=4000, cached_tokens=100, completion_tokens=9000),
        )
