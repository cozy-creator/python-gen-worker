"""The declarative endpoint catalog — endpoints as DATA (pgw#808).

This single module replaces the old per-issue harness endpoint modules
(`mint_endpoints_pgw784.py`, `shape_endpoints_pgw789.py`, ...). Scenarios never
declare their own endpoints: they SELECT rows from :data:`CATALOG`, and the
worker loads this one module (:data:`MODULE`). A new behavior is a new row —
a small endpoint method plus one `Row` entry — never a new module.

Contract (kept stable for every later suite):

* ``MODULE`` — the module string handed to the worker / baked manifests.
* ``CATALOG: dict[str, Row]`` — wire-function-name -> declared row. `Row` is
  self-describing: shape, slots, declared default refs, payload/output types,
  and `input_bytes()` / `decode()` codec helpers.
* Wire refs used by model-bound rows are module constants (``HOT_REF``,
  ``PINNED_DEFAULT``, ...) so scenarios and the hub double agree byte-for-byte.
* Cross-process coordination handles live on probe classes (``HoldSetupProbe``)
  with an explicit ``reset()`` — never ambient module state a test forgets.

The table is verified at import: every declared row must match the spec the
real registry extracts (name, stream shape, slot set). The catalog cannot
drift from what the worker actually discovers.

House rules: no torch, no GPU, no network, no real weights — every handler is
production-shaped but tiny, and the CODE PATH exercised is always the real
registry/executor/transport one.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

import msgspec
from PIL import Image

from gen_worker import (
    Hub,
    RequestContext,
    Slot,
    ValidationError,
    diffusers_step_callback,
    endpoint,
)
from gen_worker import io as gw_io
from gen_worker.api.binding import wire_ref
from gen_worker.api.streaming import StreamResult, TokenUsage
from gen_worker.api.types import ImageAsset
from gen_worker.families.base import GenerationDefaults, family

#: The one module string every scenario (and baked manifest) uses.
MODULE = "tests_v2.catalog"

# ---------------------------------------------------------------------------
# Wire refs (module constants so tests and the hub double never restate them).
# ---------------------------------------------------------------------------

HOT_BINDING = Hub("catalog/hot-tiny")
HOT_REF = wire_ref(HOT_BINDING)

PINNED_DEFAULT = Hub("catalog/pinned-default", tag="prod")
PINNED_DEFAULT_REF = wire_ref(PINNED_DEFAULT)

PICKED_DEFAULT = Hub("catalog/picked-default", tag="prod")
PICKED_DEFAULT_REF = wire_ref(PICKED_DEFAULT)


@family("v2-testfam")
class _V2Defaults(GenerationDefaults, frozen=True):
    steps: int = 7


# ---------------------------------------------------------------------------
# Payload / output structs.
# ---------------------------------------------------------------------------


class Ping(msgspec.Struct):
    text: str = ""
    model: str = ""  # `selected_by="model"` target field for picked rows


class Reply(msgspec.Struct):
    response: str


class GenIn(msgspec.Struct):
    prompt: str = ""


class GenOut(msgspec.Struct):
    image: ImageAsset


# ---------------------------------------------------------------------------
# Stage constants for the `staged-generate` row (seconds) — asserted against.
# ---------------------------------------------------------------------------

TEXT_ENCODE_S = 0.06
STEP_S = 0.04
STEPS = 4
DECODE_S = 0.05  # deliberately un-bracketed: must surface as resid.tail


# ---------------------------------------------------------------------------
# Endpoint classes. Each method is one CATALOG row; docstrings say what the
# row is FOR, the table below is the interface.
# ---------------------------------------------------------------------------


@endpoint
class Plain:
    """Model-free rows: dispatch mechanics, streaming, cancellation, timing."""

    def echo(self, ctx: RequestContext, data: Ping) -> Reply:
        ctx.raise_if_cancelled()
        if (data.text or "").strip().lower() == "marco":
            return Reply(response="polo")
        raise ValidationError(f"expected 'marco', got {data.text!r}")

    async def stream3(self, ctx: RequestContext, data: Ping) -> AsyncIterator[Reply]:
        for i in range(3):
            ctx.raise_if_cancelled()
            yield Reply(response=f"chunk-{i}")

    async def slow_stream(self, ctx: RequestContext, data: Ping) -> AsyncIterator[Reply]:
        # Slow enough that a mid-stream cancel is observable.
        for i in range(20):
            ctx.raise_if_cancelled()
            yield Reply(response=f"slow-chunk-{i}")
            await asyncio.sleep(0.2)

    async def slow(self, ctx: RequestContext, data: Ping) -> Reply:
        await asyncio.sleep(30.0)
        return Reply(response="late")

    def sleepy(self, ctx: RequestContext, data: Ping) -> Reply:
        time.sleep(0.5)
        return Reply(response="done")


@endpoint(model=HOT_BINDING)
class HotBound:
    """Hub-bound fixed model: the residency -> load -> serve walk. Serves the
    exact bytes the hub delivered, so a passing dispatch proves the load."""

    def setup(self, model: str) -> None:
        self.model_path = model

    def hot_echo(self, ctx: RequestContext, data: Ping) -> Reply:
        weights = Path(self.model_path) / "model.safetensors"
        return Reply(response=weights.read_text())


@endpoint(models={
    "pipeline": Slot(str, default_checkpoint=PINNED_DEFAULT),
})
class PinnedSlot:
    """FIXED slot (no selected_by=): a dispatch pick naming a DIFFERENT repo
    than the declared default must refuse, naming slot and both refs."""

    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def pinned_echo(self, ctx: RequestContext[_V2Defaults], data: Ping) -> Reply:
        ref = ctx.slots["pipeline"].ref
        return Reply(response=f"{ref.source}:{ref.path}:{ref.tag}#{ref.flavor}")


@endpoint(models={
    "pipeline": Slot(str, selected_by="model", default_checkpoint=PICKED_DEFAULT),
})
class PickedSlot:
    """Catalog slot (selected_by="model"): a different-repo pick is a
    legitimate explicit surface, never an identity mismatch."""

    def setup(self, pipeline: str) -> None:
        self.pipeline_path = pipeline

    def picked_echo(self, ctx: RequestContext[_V2Defaults], data: Ping) -> Reply:
        ref = ctx.slots["pipeline"].ref
        return Reply(response=f"{ref.source}:{ref.path}:{ref.tag}#{ref.flavor}")


@endpoint
class Staged:
    """Image-shaped handler with real stage structure (text encode, stepped
    denoise on the shared diffusers callback, un-bracketed decode gap, real
    write_image encode) so JobResult.metrics.stage_ms is a real map."""

    def staged_generate(self, ctx: RequestContext, data: GenIn) -> GenOut:
        with ctx.stage("text_encode"):
            time.sleep(TEXT_ENCODE_S)
        on_step = diffusers_step_callback(ctx, STEPS)
        for i in range(STEPS):
            time.sleep(STEP_S)
            on_step(None, i, None, {})
        time.sleep(DECODE_S)  # un-bracketed on purpose
        image = Image.effect_noise((512, 512), 64).convert("RGB")
        asset = gw_io.write_image(
            ctx, f"outputs/{ctx.request_id}/image.webp", image,
            format="webp", as_type=ImageAsset,
        )
        return GenOut(image=asset)


@endpoint
class Billable:
    """Typed usage metrics; `large-usage` exceeds INLINE_RESULT_MAX_BYTES so
    the executor's size-alone inline/blob_ref decision tips to blob_ref."""

    def small_usage(self, ctx: RequestContext, data: Ping) -> StreamResult:
        return StreamResult(
            text="ok",
            usage=TokenUsage(prompt_tokens=12, cached_tokens=2, completion_tokens=5),
        )

    def large_usage(self, ctx: RequestContext, data: Ping) -> StreamResult:
        return StreamResult(
            text="x" * 200_000,
            usage=TokenUsage(prompt_tokens=4000, cached_tokens=100, completion_tokens=9000),
        )


class HoldSetupProbe:
    """Coordination handles for the `hold-setup` row (cancelled-setup residue).

    Unarmed (the default, so hub-double boots never block), ``setup()`` is a
    no-op. Armed: attempt 1 parks on RELEASE after allocating a cycle-carrying
    buffer; attempt 2 records whether attempt 1's buffer is STILL ALIVE at the
    exact point a real load would allocate on top of it. Call ``arm()`` before
    use, read ``alive_at_second_attempt`` after, ``reset()`` in teardown.
    """

    ARMED = threading.Event()
    ENTERED = threading.Event()
    RELEASE = threading.Event()
    refs: List[weakref.ref] = []
    alive_at_second_attempt: List[bool] = []

    @classmethod
    def arm(cls) -> None:
        cls.reset()
        cls.ARMED.set()

    @classmethod
    def reset(cls) -> None:
        cls.ARMED.clear()
        cls.ENTERED.clear()
        cls.RELEASE.clear()
        cls.refs.clear()
        cls.alive_at_second_attempt.clear()


class _Buffer:
    """Stand-in for a partially loaded pipeline: carries a reference cycle
    (real pipelines always do), so only a gc pass frees it."""

    def __init__(self) -> None:
        self.cycle = self


@endpoint
class HoldSetup:
    def setup(self) -> None:
        probe = HoldSetupProbe
        if not probe.ARMED.is_set():
            return
        if probe.refs:
            probe.alive_at_second_attempt.append(probe.refs[0]() is not None)
            return
        buf = _Buffer()
        probe.refs.append(weakref.ref(buf))
        self.buf = buf
        probe.ENTERED.set()
        probe.RELEASE.wait(10)

    def hold_setup(self, ctx: RequestContext, data: Ping) -> Reply:
        return Reply(response="held")


# ---------------------------------------------------------------------------
# The table. Scenarios select rows from here; nothing else is interface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One declared worker function, as data.

    ``slots`` maps slot name -> declared default wire ref ("" when the slot
    has no hub-declared default). ``selected_by`` maps slot name -> payload
    field for catalog (dynamic-pick) slots.
    """

    function: str
    shape: str  # "unary" | "stream"
    payload: type = Ping
    output: Optional[type] = Reply
    slots: Mapping[str, str] = field(default_factory=dict)
    selected_by: Mapping[str, str] = field(default_factory=dict)
    behavior: str = ""

    def input_bytes(self, **fields: Any) -> bytes:
        """msgpack-encode a payload for this row (real wire form)."""
        return msgspec.msgpack.encode(self.payload(**fields))

    def decode(self, data: bytes) -> Any:
        """Decode a JobResult.inline payload back to this row's output type."""
        if self.output is None:
            raise TypeError(f"row {self.function!r} declares no typed output")
        return msgspec.msgpack.decode(data, type=self.output)


CATALOG: Dict[str, Row] = {
    "echo": Row(
        "echo", "unary",
        behavior="marco->polo; anything else raises ValidationError (typed INVALID)",
    ),
    "stream3": Row(
        "stream3", "stream",
        behavior="exactly 3 chunks, seq-ordered job_progress",
    ),
    "slow-stream": Row(
        "slow-stream", "stream",
        behavior="20 chunks at 0.2s: mid-stream cancel is observable",
    ),
    "slow": Row(
        "slow", "unary",
        behavior="30s await: cancel target (no wall deadline exists, pgw#904)",
    ),
    "sleepy": Row(
        "sleepy", "unary",
        behavior="0.5s sync sleep: in-flight across reconnect/drain",
    ),
    "hot-echo": Row(
        "hot-echo", "unary", slots={"model": HOT_REF},
        behavior="serves the hub-delivered weight bytes verbatim",
    ),
    "pinned-echo": Row(
        "pinned-echo", "unary", slots={"pipeline": PINNED_DEFAULT_REF},
        behavior="FIXED slot: different-repo pick refuses, naming both refs",
    ),
    "picked-echo": Row(
        "picked-echo", "unary", slots={"pipeline": PICKED_DEFAULT_REF},
        selected_by={"pipeline": "model"},
        behavior="catalog slot: different-repo pick is a legitimate surface",
    ),
    "staged-generate": Row(
        "staged-generate", "unary", payload=GenIn, output=None,
        behavior="real stage_ms map: text_encode + denoise steps + encode",
    ),
    "small-usage": Row(
        "small-usage", "unary", output=None,
        behavior="inline result with typed TokenUsage(12,2,5)",
    ),
    "large-usage": Row(
        "large-usage", "unary", output=None,
        behavior=">64KB output: blob_ref upload with TokenUsage(4000,100,9000)",
    ),
    "hold-setup": Row(
        "hold-setup", "unary",
        behavior="setup parks on HoldSetupProbe: cancelled-setup residue probe",
    ),
}


def row(name: str) -> Row:
    """The declared row for one wire function name (KeyError = not a row)."""
    return CATALOG[name]


# ---------------------------------------------------------------------------
# Import-time self-check: the table IS what the real registry discovers.
# ---------------------------------------------------------------------------


def _verify_catalog() -> None:
    from gen_worker.registry import extract_specs

    classes = (Plain, HotBound, PinnedSlot, PickedSlot, Staged, Billable, HoldSetup)
    specs = {}
    for cls in classes:
        for spec in extract_specs(cls):
            specs[spec.name] = spec

    declared, extracted = set(CATALOG), set(specs)
    if declared != extracted:
        raise RuntimeError(
            f"catalog table drifted from the registry: only-in-table="
            f"{sorted(declared - extracted)} only-in-code={sorted(extracted - declared)}"
        )
    for name, entry in CATALOG.items():
        spec = specs[name]
        want_shape = "stream" if spec.output_mode == "stream" else "unary"
        if entry.shape != want_shape:
            raise RuntimeError(
                f"catalog row {name!r} declares shape={entry.shape!r} but the "
                f"registry extracted output_mode={spec.output_mode!r}"
            )
        if set(entry.slots) - set(spec.models):
            raise RuntimeError(
                f"catalog row {name!r} declares slots {sorted(entry.slots)} the "
                f"registry does not know: {sorted(spec.models)}"
            )


_verify_catalog()
