"""pgw#978: the endpoint the micro-mint rig mints, in the SDXL shape.

Two things about this module are load-bearing and neither is incidental.

**The slot is a CATALOG slot** — ``Slot(TinyDiffusionPipeline, selected_by="model")``
with no ``default_checkpoint=``. That is sdxl's shape and it is the shape pgw#969
died on: the mint child re-runs discovery in a fresh interpreter, so a catalog
slot rediscovers NOTHING and ``ctx.slots["pipeline"]`` raises unless the parent's
``MintSlot`` crossed the boundary intact. A rig whose endpoint declared a code
default would resolve in the child by accident and prove nothing about the wire.

**The handler dereferences ``ctx.slots``.** pgw#828's regression endpoint did the
same and still missed pgw#969, because its slot had a declared default. Here the
dereference is over a catalog slot, so the warm forward the mint child runs is
the exact call that crashed on two L40S pods.

The export declaration is registered at IMPORT time, which is how a real
endpoint package does it — the mint child imports this module from
``MintRequest.modules`` and collection registers the declaration before the
family is traced.
"""

from __future__ import annotations

import os
from typing import List

import msgspec

from gen_worker import RequestContext, Resources, Slot, endpoint
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import (
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
)
from gen_worker.families.base import GenerationDefaults, family

from harness.tiny_diffusion import (
    CONTEXT_DIM,
    LATENT_CHANNELS,
    LATENT_SIZE,
    TEXT_LEN,
    TinyDiffusionPipeline,
    install_synthetic_runtime_if_asked,
)

# pgw#983: the mint child is a fresh interpreter, so import time is the only
# hook the rig has to supply the one runtime fact a cardless box cannot state.
# No-op unless PGW978_SYNTHETIC_RUNTIME asks; see the function's docstring for
# what it does and does not buy.
SYNTHETIC_RUNTIME_INSTALLED = install_synthetic_runtime_if_asked()

FAMILY = "microrig"

#: The pixel shape row the latent grid corresponds to. Part of the cell key's
#: contract axis, so it is stated once here and nowhere else.
PIXEL_SHAPE = (LATENT_SIZE * 8, LATENT_SIZE * 8)

#: The CFG batch. Two, because a classifier-free pass is the fleet's real graph
#: class and a batch-1 export would trace a shape production never runs.
CFG_BATCH = 2

#: Every ref this process resolved, so the rig can prove WHICH checkpoint the
#: child traced rather than trusting that it traced one.
RESOLVED_REFS: List[str] = []


@family(FAMILY)
class MicroRigDefaults(GenerationDefaults, frozen=True):
    steps: int = 2


class RigIn(msgspec.Struct):
    prompt: str = ""
    #: ``selected_by="model"`` binds the catalog slot off this field.
    model: str = ""


class RigOut(msgspec.Struct):
    response: str = ""


DECLARATION = Compile(
    family=FAMILY,
    targets=("unet",),
    shapes=(PIXEL_SHAPE,),
    # A pinned text length, declared rather than discovered — the ie#544 axis.
    text_len=TEXT_LEN,
    dims=(Dim("B", carried_by=(
        ("sample", 0), ("timestep", 0), ("encoder_hidden_states", 0))),),
    classes=(GraphClass(dims={"B": CFG_BATCH}),),
    inputs=(
        Input("sample", shape=("B", LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
              dtype="float32"),
        Input("timestep", shape=("B",), dtype="float32", value=100.0),
        Input("encoder_hidden_states", shape=("B", TEXT_LEN, CONTEXT_DIM),
              dtype="float32"),
    ),
    shape_strategy="static-rows",
    # Measured, never defaulted: whether pre-warm changes the traced graph is a
    # per-family FACT (sdxl False, z-image True). The toy pipeline has no
    # pre-warm at all, so False is the measurement rather than a guess.
    warm_changes_key=False,
)

register_export_declaration(DECLARATION, replace=True)


@endpoint(
    models={"pipeline": Slot(TinyDiffusionPipeline, selected_by="model")},
    compile=Compile(
        family=FAMILY, targets=("unet",), shapes=(PIXEL_SHAPE,),
        text_len=TEXT_LEN),
    resources=Resources(gpu=True),
)
class MicroRigEndpoint:
    def setup(self, pipeline: TinyDiffusionPipeline) -> None:
        self.pipe = pipeline

    def rig_generate(
        self, ctx: RequestContext[MicroRigDefaults], data: RigIn,
    ) -> RigOut:
        """The pgw#969 call, verbatim in shape: the FIRST thing a warm forward
        does is dereference the catalog slot. A child that was handed bytes and
        no identity dies here, 0.0 s in, exactly as production did."""
        import torch

        from harness.tiny_diffusion import example_inputs

        resolved = ctx.slots["pipeline"]
        RESOLVED_REFS.append(str(resolved.ref.path))
        device = next(self.pipe.unet.parameters()).device
        batch = CFG_BATCH
        tensors = example_inputs(batch=batch, device=str(device))
        with torch.no_grad():
            out = self.pipe.unet(
                tensors["sample"], tensors["timestep"],
                tensors["encoder_hidden_states"])
        return RigOut(response=f"{resolved.ref.path}|{tuple(out.shape)}")


def reset() -> None:
    RESOLVED_REFS.clear()


def checkpoint_root() -> str:
    """Where the rig put the generated tree. Env-carried because the mint child
    is a fresh interpreter that shares nothing else with the parent."""
    return os.environ.get("PGW978_CHECKPOINT", "")


__all__ = [
    "CFG_BATCH",
    "DECLARATION",
    "FAMILY",
    "PIXEL_SHAPE",
    "RESOLVED_REFS",
    "MicroRigDefaults",
    "MicroRigEndpoint",
    "RigIn",
    "RigOut",
    "checkpoint_root",
    "reset",
]
