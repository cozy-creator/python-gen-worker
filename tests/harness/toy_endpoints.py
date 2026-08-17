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

from gen_worker import (
    ConfigParam, Hub, RequestContext, Resources, Slot, ValidationError, endpoint,
)
from gen_worker.api.streaming import StreamResult, TokenUsage
from gen_worker.families.base import GenerationDefaults, family


def _slot_weight_bytes(slot_dir: object) -> str:
    """What the slot's weights file HOLDS, on any tree shape.

    A harness endpoint is the closest thing this repo has to a pgw#1303 author
    slot: it is handed a DIRECTORY and reads raw weight bytes out of it. After
    pgw#1308 step ⑥ that directory is a projected tree, so the file at the
    path is a ~128 B pointer stub — which is exactly what a real third-party
    loader gets, and exactly why #1303 is a ruling rather than a cleanup.

    So these fixtures go through the SAME seam a gated production site now
    goes through (`models.materialized_view.third_party_dir`), which is what
    makes them evidence about the gate instead of a workaround for it.
    """

    from gen_worker.models.materialized_view import third_party_dir

    real = third_party_dir(Path(str(slot_dir)), why="harness author slot")
    return (real / "model.safetensors").read_text()


@family("harness-testfam")
class _ToyDefaults(GenerationDefaults, frozen=True):
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
        return EchoOut(response=_slot_weight_bytes(self.model_path))


# Toggled by P2's setup-failure test: True => BrokenSetupEndpoint.setup
# raises (the ernie-on-pod shape: import fine, pipeline load fails).
BREAK_SETUP = threading.Event()
BREAK_SETUP.set()


@endpoint(model=Hub("harness/residency-broken"))
class BrokenSetupEndpoint:
    """Setup raises while BREAK_SETUP is set; desired-state retry recovers."""

    def setup(self, model: str) -> None:
        if BREAK_SETUP.is_set():
            raise RuntimeError("pipeline exploded: cannot import FakePipeline")
        self.model_path = model

    def broken_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        return EchoOut(response=_slot_weight_bytes(self.model_path))


class _RamPressurePipeline:
    """Pipeline-shaped annotation for the P2 host-RAM admission wire test."""

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> "_RamPressurePipeline":
        return cls()

    def to(self, device: str) -> "_RamPressurePipeline":
        return self


@endpoint(models={
    "pipeline": Hub("harness/ram-pressure-pipeline"),
    "vae": Hub("harness/ram-pressure-shared-vae"),
})
class RamPressureEndpoint:
    """Two staged refs: only the larger pipeline may fail RAM admission."""

    def setup(
        self, pipeline: _RamPressurePipeline, vae: _RamPressurePipeline,
    ) -> None:
        self.pipeline = pipeline
        self.vae = vae

    def ram_pressure(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        return EchoOut(response="unexpectedly loaded")


# ---------------------------------------------------------------------------
# P3: Slot-declared endpoints (pgw#606/th#938 precedence + pgw#583 identity).
# ---------------------------------------------------------------------------

# Never delivered in ANY P3 test: if boot ever fetches these, the store finds
# no snapshot registered for them and fails fast (no real network involved).
BOOT_UNREACHABLE_PIPELINE = Hub("harness/boot-precedence-pipeline", release="prod")
BOOT_UNREACHABLE_VAE = Hub("harness/boot-precedence-vae", release="prod")


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=BOOT_UNREACHABLE_PIPELINE),
    "vae": Slot(str, default_checkpoint=BOOT_UNREACHABLE_VAE),
})
class SlotBootPrecedenceEndpoint:
    """Both slot defaults are tensorhub-sourced; neither may be boot-set-up
    from the code default — only a hub-stamped Hot DesiredInstance may."""

    def setup(self, pipeline: str, vae: str) -> None:
        self.pipeline_path = pipeline
        self.vae_path = vae

    def slot_boot_precedence(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        return EchoOut(response=_slot_weight_bytes(self.pipeline_path))


DECLARED_PIPELINE = Hub("harness/slot-identity-declared", release="prod")


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=DECLARED_PIPELINE),
})
class SlotIdentityFixedEndpoint:
    """FIXED slot (no selected_by=): a dispatched pick naming a DIFFERENT
    repo than the declared default must refuse (gw#583)."""

    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def slot_identity_fixed(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        resolved = ctx.slots["pipeline"]
        ref = resolved.ref
        return EchoOut(response=f"{ref.source}:{ref.path}@{ref.release}")


CATALOG_DEFAULT_PIPELINE = Hub("harness/slot-catalog-default", release="prod")


@endpoint(models={
    "pipeline": Slot(
        str, selected_by="model", default_checkpoint=CATALOG_DEFAULT_PIPELINE,
    ),
})
class SlotIdentityCatalogEndpoint:
    """selected_by= catalog slot: a dispatched pick of a DIFFERENT repo is a
    legitimate explicit surface, not an identity mismatch."""

    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def slot_identity_catalog(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        resolved = ctx.slots["pipeline"]
        ref = resolved.ref
        return EchoOut(response=f"{ref.source}:{ref.path}@{ref.release}")


# ---------------------------------------------------------------------------
# pgw#617: hierarchical component bindings (load-then-substitute).
# ---------------------------------------------------------------------------


class ToyVae:
    """Component class named by the composed base's model_index.json."""

    def __init__(self, content: str, source: str) -> None:
        self.content = content
        self.source = source

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "ToyVae":
        return cls((Path(path) / "weights.txt").read_text(), source=str(path))


class ToyComposedPipeline:
    """Diffusers-shaped fake: from_pretrained loads components from the tree
    unless a preloaded module is injected via kwargs (the gw#479/pgw#617
    ``components=`` mechanism)."""

    def __init__(self, base_weights: str, vae: ToyVae, injected: bool) -> None:
        self.base_weights = base_weights
        self.vae = vae
        self.vae_injected = injected

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> "ToyComposedPipeline":
        base = (Path(path) / "transformer" / "weights.txt").read_text()
        vae = kwargs.get("vae")
        if isinstance(vae, ToyVae):
            return cls(base, vae, injected=True)
        return cls(base, ToyVae.from_pretrained(str(Path(path) / "vae")), injected=False)

    def to(self, device: str) -> "ToyComposedPipeline":
        return self


COMPOSED_DECLARED = Hub("harness/composed-base", release="prod")
COMPOSED_SETUPS: list = []  # one entry per setup() run (identity-change proof)


@endpoint(models={
    "pipeline": Slot(
        ToyComposedPipeline, default_checkpoint=COMPOSED_DECLARED,
    ),
})
class ComposedEndpoint:
    def setup(self, pipeline: ToyComposedPipeline) -> None:
        COMPOSED_SETUPS.append(id(self))
        self.pipe = pipeline

    def composed_echo(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        p = self.pipe
        return EchoOut(response=(
            f"base={p.base_weights}|vae={p.vae.content}"
            f"|injected={p.vae_injected}|setups={len(COMPOSED_SETUPS)}"
        ))


# ---------------------------------------------------------------------------
# pgw#636: the multi-checkpoint juggle — ONE dynamic ``selected_by=`` slot
# serving N different checkpoints, the sdxl catalog shape.
# ---------------------------------------------------------------------------


class ToyCheckpointPipeline:
    """Minimal pipeline whose identity is the checkpoint bytes it loaded."""

    def __init__(self, weights: str) -> None:
        self.weights = weights

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "ToyCheckpointPipeline":
        return cls((Path(path) / "transformer" / "weights.txt").read_text())

    def to(self, device: str) -> "ToyCheckpointPipeline":
        return self


JUGGLE_DECLARED = Hub("harness/juggle-base", release="prod")
JUGGLE_SETUPS: list = []  # one entry per setup() run, in order


@endpoint(models={
    "pipeline": Slot(
        ToyCheckpointPipeline, selected_by="model",
        default_checkpoint=JUGGLE_DECLARED,
    ),
})
class JuggleEndpoint:
    def setup(self, pipeline: ToyCheckpointPipeline) -> None:
        JUGGLE_SETUPS.append(pipeline.weights)
        self.pipe = pipeline

    def juggle_echo(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        return EchoOut(
            response=f"{self.pipe.weights}|setups={len(JUGGLE_SETUPS)}")


# ---------------------------------------------------------------------------
# P9: typed billing usage on JobMetrics, inline vs blob_ref by size alone.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# th#1050: opt-in declared lane — the endpoint's code BRANCHES on ctx.lane.
# ---------------------------------------------------------------------------


# th#1087: declared config parameters + envs; the handler reads ctx.config.
@endpoint(
    config=[
        ConfigParam("scheduler", str, default="ddim", choices=["ddim", "euler_a"]),
        ConfigParam("default_steps", int, default=30, ge=1, le=150),
    ],
    env=["TOY_API_BASE"],
)
class ConfigKnobsEndpoint:
    def config_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        cfg = ctx.config
        return EchoOut(response=f"{cfg['scheduler']}:{cfg['default_steps']}")


@endpoint(handles=["fp8-w8a8-dynamic"])
class LaneAwareEndpoint:
    def lane_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        # The divergence the declaration marks: an author kernel branch.
        if ctx.execution_lane.startswith("fp8-w8a8-dynamic"):
            return EchoOut(response=f"author-kernel:{ctx.execution_lane}")
        return EchoOut(response=f"reference:{ctx.execution_lane}")


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


# ---------------------------------------------------------------------------
# Optional slots: which lanes a release serves is DEPLOY config (th#980/
# ie#524, th#1226). `edit` has a setup default, so a deploy may bind only
# `t2i` and the endpoint still boots — the unbound lane refuses per-request
# with a typed, client-visible error instead of crashing setup.
# ---------------------------------------------------------------------------

OPTIONAL_T2I = Hub("harness/optional-lane-t2i", release="prod")
OPTIONAL_EDIT = Hub("harness/optional-lane-edit", release="prod")


@endpoint(models={
    "t2i": Slot(str, default_checkpoint=OPTIONAL_T2I, root=True),
    "edit": Slot(str, default_checkpoint=OPTIONAL_EDIT),
})
class OptionalExecutionLaneEndpoint:
    def setup(self, t2i: str, edit: str | None = None) -> None:
        self.t2i_path = t2i
        self.edit_path = edit

    def optional_t2i(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        return EchoOut(response=f"t2i:{_slot_weight_bytes(self.t2i_path)}"
                                f":edit_bound={self.edit_path is not None}")

    def optional_edit(self, ctx: RequestContext[_ToyDefaults], data: EchoIn) -> EchoOut:
        if self.edit_path is None:
            raise ValidationError(
                "slot 'edit' is not bound on this release — this deploy "
                "serves the t2i lane only"
            )
        return EchoOut(response=f"edit:{_slot_weight_bytes(self.edit_path)}")


# ---------------------------------------------------------------------------
# pgw#828: the sdxl SHAPE — a GPU inference endpoint with one declared slot
# whose handler dereferences ``ctx.slots``. The delegated mint child ran this
# with an EMPTY slot table (`KeyError: 'pipeline'` at `sdxl/main.py:326`), so
# a warm job that never touches ``ctx`` could not have caught it.
# ---------------------------------------------------------------------------

WARM_SLOT_PIPELINE = Hub("harness/warm-slot-base", release="prod")


@endpoint(
    models={"pipeline": Slot(str, default_checkpoint=WARM_SLOT_PIPELINE)},
    resources=Resources(gpu=True),
)
class WarmSlotEndpoint:
    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def warm_slot_echo(
        self, ctx: RequestContext[_ToyDefaults], data: EchoIn,
    ) -> EchoOut:
        resolved = ctx.slots["pipeline"]
        return EchoOut(response=f"{resolved.ref.path}:{data.text}")
