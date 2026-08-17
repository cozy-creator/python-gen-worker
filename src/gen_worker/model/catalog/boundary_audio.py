"""The audio boundary models: four EAGER declarations.

pgw#1346 B5. These four are eager for a different reason than the LLM lane's:
they ARE PyTorch, but nothing about them is a declared composition today —
``musicgen`` is a transformers autoregressive decoder over EnCodec frames,
``chatterbox`` self-loads from a snapshot path, and the two StableAudio lanes
run a diffusers pipeline the endpoints never split into runners.

That distinction matters and is recorded rather than blurred: under the F3
ruling the LLM lane owes graph classes NEVER, while these four owe them only in
the sense any PyTorch model does — nobody has declared them, and this batch is
not the lane that will. What lands here is the honest state: a family handle,
the tuned schema where one exists, and the layout axes with the ie#740 floors
carried BY VALUE.

``stable_audio_open`` and ``foundation_1`` are TWO declarations of one
architecture, and that is deliberate. They are two hub families (`foundation-1`
is registered; the open checkpoint declares no ``@family`` at all), two
different tuned vocabularies — foundation-1 stamps a 200-step recipe and a
default negative prompt, the open lane stamps nothing — and B1's measured rule
is that an instance carries weights, tuned values and a label only. A model
with a different tuned SCHEMA cannot be an instance of another.
"""

from __future__ import annotations

from typing import Final

from ..spec import ModelSpec, TunedValues


class ChatterboxTuned(TunedValues, frozen=True):
    """Chatterbox's voice recipe, migrated from ``@family("chatterbox")``.

    Schema defaults are the model-card neutral recipe, which is identical to
    the hub's neutral stamp for an unconfigured checkpoint — so a hub-less run
    and a stamped-neutral run agree by construction rather than by luck.
    """

    #: Emotion exaggeration; 0.5 is the model-card neutral default.
    exaggeration: float = 0.5
    #: CFG weight over the voice conditioning (0 disables CFG).
    cfg_weight: float = 0.5
    temperature: float = 0.8


class Foundation1Tuned(TunedValues, frozen=True):
    """Foundation-1's recipe, migrated from ``@family("foundation-1")``.

    ``num_inference_steps`` carries the WIRE spelling (pgw#654 gap #4) so
    ``RuntimeFormula``'s resolved-effective evaluation finds it.
    """

    #: 200 steps: the model card envelope; quality plateaus past ~100.
    num_inference_steps: int = 200
    negative: str = "Low quality."


#: MusicGen stereo. Autoregressive over EnCodec frames; the endpoint loads it
#: with ``AutoProcessor``/``MusicgenForConditionalGeneration`` from a snapshot
#: path, so the declaration names bytes and nothing else.
#:
#: No tuned schema: the endpoint declares no ``@family`` and reads no
#: ``ctx.defaults``. K8 is therefore inapplicable — ``_register()`` publishes
#: nothing under this name and no tensorhub PR is owed for it.
MUSICGEN: Final = ModelSpec(
    name="musicgen",
    layouts={"*": ("plain.bf16@1",)},
    # ie#740's floor, BY VALUE from the endpoint's `Slot`.
    layout_requirements={"plain.bf16@1": "vram12g"},
)

#: Chatterbox TTS. Self-loading (``ChatterboxTTS.from_local``), so no component
#: tree is derivable — but the bytes are plain bf16 and the endpoint says so.
#: No ie#740 floor is declared today, and none is invented here: th#683
#: profiling banked 3.18 GiB (e2e J30) and a floor the endpoint never stated is
#: not this lane's to state.
CHATTERBOX: Final = ModelSpec(
    name="chatterbox",
    tuned=ChatterboxTuned,
    layouts={"*": ("plain.bf16@1",)},
)

#: Stability's Stable Audio Open. No tuned schema — the endpoint declares no
#: ``@family`` and reads no ``ctx.defaults``; every knob it exposes is a wire
#: field with its own default.
STABLE_AUDIO_OPEN: Final = ModelSpec(
    name="stable_audio_open",
    layouts={"*": ("plain.bf16@1",)},
    # ie#740's floor, BY VALUE.
    layout_requirements={"plain.bf16@1": "vram8g"},
)

#: Foundation-1: the same StableAudio architecture, a different checkpoint, and
#: — unlike the open lane — a stamped recipe vocabulary of its own.
FOUNDATION_1: Final = ModelSpec(
    name="foundation_1",
    tuned=Foundation1Tuned,
    layouts={"*": ("plain.bf16@1",)},
    # ie#740's floor, BY VALUE. Measured 2.52 GiB in-VRAM fp16 (e2e J29); 8
    # keeps headroom for 47s-duration activations until th#683 profiles it.
    layout_requirements={"plain.bf16@1": "vram8g"},
)


__all__ = [
    "CHATTERBOX",
    "FOUNDATION_1",
    "MUSICGEN",
    "STABLE_AUDIO_OPEN",
    "ChatterboxTuned",
    "Foundation1Tuned",
]
