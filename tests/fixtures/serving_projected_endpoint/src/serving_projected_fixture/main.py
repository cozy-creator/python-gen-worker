"""A fixture endpoint that loads REAL weights off a REAL projected tree.

pgw#1551. Every other serving fixture in this repo loads a config-only tree
with fake weights, which is precisely the shape that cannot witness the
~21 h fleet outage: a tree with no chunk store behind it never reaches the
streaming engine, so a pod that never bound one looks identical to a pod that
did. This endpoint is deliberately the other case — its checkpoint tree is
projected, its tensor containers are ``TFSSTUB1`` pointer stubs, and the only
way ``load()`` can succeed at all is through the pgw#1380 streaming engine.

``probe`` returns what the load PROVED, not what it intended: whether the
engine bound, what it reported, and whether the parameters are real tensors
rather than meta ones. A pod that fell to the eager bridge cannot answer.
"""

from __future__ import annotations

from typing import Any, cast

import msgspec
import torch
from diffusers import DiffusionPipeline

from gen_worker import LoadContext, Model, RequestContext, entrypoint
from gen_worker._vendor.torchcg import LaneRef

#: bf16, because that is what the fixture writes to disk and the engine's whole
#: contract is that bytes land verbatim in the container's own dtype. A lane
#: naming something else here would make a dtype assertion below vacuous.
STREAM_LANE = LaneRef("fixture.diffusers-bf16@1", dtype=torch.bfloat16)


class TinyPipeline(DiffusionPipeline):
    """A real ``DiffusionPipeline`` — real components, real ``model_index``.

    Matches ``tests/streaming_fixture.py``'s saved article, because the
    skeleton builder resolves each component from the tree's own
    ``model_index.json`` and the class handed to ``ctx.load`` must accept them.
    """

    def __init__(self, unet: Any, vae: Any, text_encoder: Any,
                 text_encoder_2: Any, scheduler: Any) -> None:
        super().__init__()
        self.register_modules(  # type: ignore[attr-defined]
            unet=unet, vae=vae, text_encoder=text_encoder,
            text_encoder_2=text_encoder_2, scheduler=scheduler,
        )


class ProbeInput(msgspec.Struct, forbid_unknown_fields=True):
    """This endpoint takes nothing: the request exists to make the LOAD run."""


class LoadEvidence(msgspec.Struct):
    """What the load actually did. Every field is a MEASUREMENT."""

    #: True when `ctx.load` bound the pgw#1380 streaming engine. On the pod
    #: path before pgw#1544 this was False for every request ever served.
    engine_bound: bool
    #: The engine's own report: how many tensors it walked out of the store.
    tensors_streamed: int
    #: WHICH byte source produced them — `native` (straight out of the CAS) or
    #: `bridge`. Unlabelled the two differ by ~10x and read as one measurement.
    stream_source: str
    #: Parameters still on the meta device. A skeleton whose weights never
    #: arrived reports > 0 here and would generate noise, not an error.
    meta_parameters: int
    #: The dtype the weights came back as, spelled by torch.
    unet_dtype: str
    #: A real parameter's checksum, so "the bytes arrived" is not merely
    #: "a tensor of the right shape exists".
    unet_checksum: float


class StreamModel(Model[Any], lanes={STREAM_LANE: "vram8g"}):
    """The stateful half. ``load`` has exactly one spelling and no fallback."""

    def load(self, ctx: LoadContext[Any]) -> None:
        self.pipe = ctx.load(TinyPipeline)
        # Read off the CONTEXT, which is where pgw#1549 put the one truthful
        # answer: the host no longer holds an engine handle of its own.
        self.engine = ctx.loader_engine


@entrypoint
def probe(ctx: RequestContext, payload: ProbeInput,
          model: StreamModel) -> LoadEvidence:
    """Serve a request that reports its own load's provenance."""

    ctx.raise_if_cancelled()
    # `register_modules` installs these at runtime, so the class carries no
    # static attributes for them — the same shape every real diffusers
    # pipeline has.
    pipe = cast(Any, model.pipe)
    report = getattr(model.engine, "last_report", None)
    modules = (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
    meta = sum(
        1
        for module in modules
        for parameter in module.parameters()
        if parameter.device.type == "meta"
    )
    first = next(pipe.unet.parameters())
    return LoadEvidence(
        engine_bound=model.engine is not None,
        tensors_streamed=int(getattr(report, "tensors", 0) or 0),
        stream_source=str(getattr(report, "source", "") or ""),
        meta_parameters=meta,
        unet_dtype=str(first.dtype),
        unet_checksum=float(first.detach().float().abs().sum().item()),
    )
