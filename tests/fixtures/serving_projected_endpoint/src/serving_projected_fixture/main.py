"""A fixture endpoint that loads REAL weights off a REAL projected tree."""

from __future__ import annotations

from typing import Any, cast

import msgspec
from diffusers import DiffusionPipeline

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const

#: bf16, because that is what the fixture writes to disk and the engine's whole
#: contract is that bytes land verbatim in the container's own dtype. A lane
#: naming something else here would make a dtype assertion below vacuous.
#:
#: pgw#1621: the lane is the `(topology, quant)` stamp pair and the dtype is
#: NOT the fixture's to choose — it is `declared_dtype` on the ratified
#: `plain.bf16@1` rule. The `LaneRef("fixture.diffusers-bf16@1",
#: dtype=torch.bfloat16)` stand-in this replaced could name a topology nobody
#: had banked AND pick its own dtype; both are now refused at class definition.
STREAM_LANE = ("sd15.diffusers@1", "plain.bf16@1")
STREAM_LANE_ID = "sd15.diffusers@1+plain.bf16@1"


class TinyPipeline(DiffusionPipeline):
    """A real ``DiffusionPipeline`` — real components, real ``model_index``."""

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
    """What the load actually did."""

    engine_bound: bool
    tensors_streamed: int
    stream_source: str
    meta_parameters: int
    unet_dtype: str
    unet_checksum: float


class StreamModel(
    Model[Any],
    lanes={STREAM_LANE: lane(request=const(MiB(64)))},
):
    """The stateful half."""

    def load(self, ctx: LoadContext[Any]) -> None:
        self.pipe = ctx.load(TinyPipeline)
        self.engine = ctx.loader_engine


@entrypoint
def probe(ctx: RequestContext, payload: ProbeInput,
          model: StreamModel) -> LoadEvidence:
    """Serve a request that reports its own load's provenance."""

    ctx.raise_if_cancelled()
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
