"""The POST-FOLD shape of a family that may not mint yet (pgw#1115).

ltx-video-2.3 is the family this fixture is drawn from. Its declaration is
COMPLETE — 82 graph classes on H100, 115 on B200 — and it still may not mint,
because three design questions are open (audio_timestep rank, whole-graph OOM
never measured on the served w8a8 lane, live coverage gaps that predate AOT).
Today it says so by registering a THUNK that raises ``MintRefused``
(pgw#853); pgw#1107 folds every declaration onto ``@endpoint(compile=)``,
which takes a ``Compile`` and never a callable, so the refusal has to become
DATA or it disappears in the move.

This module is what that looks like: one endpoint class, one ``Compile``, two
unresolved ``MintBlocker`` rows and one resolved one. Nothing here raises,
nothing here is callable, and the module imports without torch — which is the
whole property the fold needs.

Deliberately isolated (the ``toy_endpoints_slot_only`` precedent): the mint
child walks ``request.modules`` and re-runs discovery over it, so what else
lives beside it is part of the test.
"""

from __future__ import annotations

import msgspec

from gen_worker import (
    Compile, Dim, GraphClass, Hub, Input, MintBlocker, RequestContext,
    Resources, Slot, endpoint,
)
from gen_worker.families.base import GenerationDefaults, register_family

FAMILY = "harness-pgw1115-blocked"
DECLARED_PIPELINE = Hub("harness/pgw1115-blocked", release="prod")

#: The two ids every refusal on this family must name.
OPEN_IDS = ("OQ-2-audio_timestep-rank", "OQ-3-whole-graph-OOM-unmeasured")

BLOCKERS = (
    MintBlocker(
        id=OPEN_IDS[0],
        what="`audio_timestep` is RANK-1 on generate and RANK-2 on edit, and "
             "the pytree input spec pins rank — so the declared T_at dim is "
             "only valid once the audio side is normalized at the endpoint "
             "boundary.",
        evidence="harness fixture standing in for ltx-video-2.3's OQ-2 "
                 "(pipeline_ltx2_audio.py:292 vs pipeline_ltx2_edit.py:294).",
        resolves_when="The endpoint builds (B, 1) instead of (B,) on "
                      "generate/extend/a2v; ONE equivalence test proves it.",
    ),
    MintBlocker(
        id=OPEN_IDS[1],
        what="This declaration assumes WHOLE-transformer export, and the "
             "OOM rationale behind the alternative was measured on a lane "
             "this endpoint no longer serves.",
        evidence="harness fixture standing in for ltx-video-2.3's OQ-3.",
        resolves_when="ONE mint-lane measurement of whole-graph export at the "
                      "largest declared classes.",
    ),
    MintBlocker(
        id="OQ-9-already-settled",
        what="A blocker that has been answered and is kept for the record.",
        evidence="harness fixture: the RESOLVED half of the vocabulary.",
        resolves_when="Measured on the mint lane.",
        resolved=True,
        resolution="Measured 2026-08-11 on the w8a8 lane; the graph is one "
                   "class, not two (this citation is what makes the flip "
                   "reviewable).",
    ),
)

BLOCKED_COMPILE = Compile(
    family=FAMILY,
    targets=("transformer",),
    text_len=128,
    shapes=((64, 64),),
    shape_strategy="static-rows",
    warm_changes_key=False,
    dims=(
        Dim("B", carried_by=(("hidden_states", 0),)),
        Dim("T_txt", carried_by=(("encoder_hidden_states", 1),)),
    ),
    classes=(GraphClass(dims={"B": 1, "T_txt": 128}),),
    inputs=(
        Input("hidden_states", shape=("B", 4, 8, 8), dtype="model"),
        Input("encoder_hidden_states", shape=("B", "T_txt", 16), dtype="model"),
    ),
    blockers=BLOCKERS,
)


class BlockedPipeline:
    """A worker-LOADED slot, so the endpoint is a real compile shape (a
    self-loading `str` slot is refused at discovery for a `compile=` class)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.transformer = object()

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "BlockedPipeline":
        return cls(path)

    def to(self, device: str) -> "BlockedPipeline":
        return self


class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 3


register_family(FAMILY, _Defaults)


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


@endpoint(
    models={"pipeline": Slot(BlockedPipeline, default_checkpoint=DECLARED_PIPELINE)},
    compile=BLOCKED_COMPILE,
    resources=Resources(gpu=True),
)
class BlockedFamilyEndpoint:
    def setup(self, pipeline: BlockedPipeline) -> None:
        self.pipeline = pipeline

    def blocked_echo(
        self, ctx: RequestContext[_Defaults], data: EchoIn,
    ) -> EchoOut:
        return EchoOut(response=f"served:{data.text}")
