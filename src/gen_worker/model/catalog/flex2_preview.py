"""Flex.2-preview — the self-loading FLUX.1-architecture derivative, declared.

pgw#1346 B5's one image lane, and the only B5 model that shares a directory
with a graph family. ``flux.1-schnell/main.py`` declares two ``@endpoint``
classes: ``Flux1Schnell``, whose tree is DERIVED from ``FluxPipeline`` and
which B1 owns, and ``Flex2Preview``, which ships its pipeline code inside the
repo (``custom_pipeline=``) and loads itself. A ``Slot(str)`` was the declared
way to say "this runtime loads itself"; an eager ``ModelSpec`` is that same
sentence in the surviving vocabulary.

**Why it is eager rather than a ``GraphModelSpec``, and why that is not the F3
carve-out.** Flex.2 is ordinary PyTorch, so unlike vLLM or llama.cpp it is
traceable in principle. What it does not have is a class this repo can
introspect: the pipeline arrives WITH THE CHECKPOINT, so its component tree is
a deploy-time fact and there is nothing at declaration time to write a ``build``
callable against. Nothing here forecloses a later graph declaration; the eager
tier is simply the honest state, and it is the state the endpoint already had.

**It is its own model, not an instance of a Flux1 one.** Three separately
registered hub families exist — ``flux1-dev``, ``flux1-schnell``,
``flex2-preview`` (the endpoint stamps the last on its own compile cell) — and
B1's measured rule is that a differing architecture config is a different
``ModelSpec`` by construction. Flex.2 is a redistill with built-in inpainting
and universal control inputs; its call surface is not schnell's.

**The tuned vocabulary is shared and re-declared, not imported.** The endpoint
registers ONE ``@family("flux1")`` schema and both of its classes read it. Under
the eager tier a schema belongs to the model that owns it, and three hub
families get three ``<root>.schema.json`` documents, so the FIELDS are what is
shared here — two of them — rather than the registration.
"""

from __future__ import annotations

from typing import Final

from ..spec import ModelSpec, TunedValues


class Flex2PreviewTuned(TunedValues, frozen=True):
    """Flex.2-preview's recipe schema.

    The same two fields ``flux.1-schnell``'s ``Flux1Defaults`` declares, with
    the same neutral values — the BFL FLUX.1 base card's numbers, which is what
    the hub stamps when nothing configured the resolved checkpoint. The steps
    field carries the WIRE name so ``RuntimeFormula("a + b*num_inference_steps")``
    resolves it payload-over-stamped.
    """

    num_inference_steps: int = 28
    guidance: float = 3.5


#: ostris/Flex.2-preview.
FLEX2_PREVIEW: Final = ModelSpec(
    name="flex2_preview",
    tuned=Flex2PreviewTuned,
    layouts={"*": ("plain.bf16@1",)},
    # ie#740's floor, BY VALUE from the endpoint's `Slot`: Flex2Preview shares
    # the flux.1-schnell endpoint's scalar, and bf16 is the only lane it serves.
    layout_requirements={"plain.bf16@1": "vram36g"},
)


__all__ = ["FLEX2_PREVIEW", "Flex2PreviewTuned"]
