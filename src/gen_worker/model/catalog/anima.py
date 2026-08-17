"""Anima, declared — and declared on the EAGER tier, for measured reasons.

Anima (``circlestone-labs/Anima``) is an anime-focused ~2B text-to-image model
on NVIDIA's Cosmos-Predict2-2B backbone, served through DiffSynth-Studio's
``AnimaImagePipeline``: one Qwen3-0.6B text encoder, a T5-XXL tokenizer whose
ids feed an in-model LLM adapter, and the Qwen-Image VAE.

**Why this is a ``ModelSpec`` and not a ``GraphModelSpec``.** Not a scoping
shortcut, and not the eager tier used as a waiting room — two blockers, either
of which alone is sufficient, and neither of which this lane can remove:

1. **The module code is not reachable from the SDK.** A ``Runner.build``
   constructs the module the mint traces, and Anima's module is
   ``diffsynth.models.anima_dit.AnimaDiT``. ``diffsynth`` is not a gen-worker
   dependency in any extra — the catalog's other entries build from
   ``diffusers``, which is. A declaration whose ``build`` cannot run on the
   authoring box cannot produce the committed ``<family>.export.json`` that
   ``ModelExport`` and ``check_model_bindings.py`` require, so the graph tier
   is unreachable here rather than merely unwritten.
2. **The traced call carries an unpinnable text axis.** ``AnimaDiT.forward``
   takes ``t5xxl_ids`` and runs the LLM adapter INSIDE the traced module, and
   the pipeline tokenizes those ids with no padding and no ``max_length`` — so
   their length is the prompt's, variable up to 512. The Qwen3 branch is pinned
   at 512 and the adapter's OUTPUT is padded back to 512, but the ids
   themselves are not, and a free symbol on that axis is what pgw#852 measured
   AOTInductor refusing outright (``CantSplit``). It is also exactly the axis
   the hidream-o1 endpoint names when it explains why IT deleted its compile
   block.

The endpoint independently reached the same place: it declares no ``compile=``
at all, and says so in its own words — torch.compile measured no win on this
compute-bound DiT, and a compile declaration additionally classifies the
function as hub-delivered (gw#584), which deadlocked live against hubs that
only dispatch to advertised functions (ie#519). **So nothing regresses by
declaring the eager tier: it is what the fleet already serves.**

**What is owed, recorded rather than skipped.** Promoting Anima needs, in
order: diffsynth reachable from a mint-side extra (or the DiT vendored), then a
decision on the ``t5xxl_ids`` axis — pin it by padding the ids the way the
Qwen3 branch is already padded, or lift the adapter out of the traced module.
The architecture block below is stated now so that promotion is a diff of one
file rather than a re-derivation.

**And one blocker is not Anima's at all (pgw#1346 K11).** The eager tier has no
generated ``Model`` type: ``ModelExport`` refuses a document with no runners, so
codegen cannot render a binding, so nothing an endpoint can annotate a handler
parameter with exists. ``spec.py`` promises the eager tier is a first-class
citizen on one declaration surface, and it is — as a DECLARATION. Until codegen
grows an eager binding, ``inst.tuned`` is unreachable for this model and the
anima endpoint's ``ctx.defaults`` reads cannot be migrated. That gates every
eager model in the program, not just this one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..spec import ModelSpec
from .anima_serve import (
    LATENT_CHANNELS,
    TEXT_TOKENS,
    TEXT_WIDTH,
    AnimaLoraTuned,
    AnimaTuned,
)

#: Anima's DiT architecture, from ``AnimaDiT.__init__``'s frozen kwargs. It
#: takes NO arguments — the architecture is hard-coded in the model class — so
#: unlike a diffusers family there is no ``config.json`` to read and this block
#: is a transcription of code rather than of data. Recorded because it is the
#: class-level truth a ``Runner.build`` will need, and because
#: ``crossattn_emb_channels`` (1024) is NOT the residual width (2048): the
#: text stream and the image stream are different widths in this model, and
#: conflating them is the mistake a reader is most likely to make.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "in_channels": LATENT_CHANNELS,
    "out_channels": LATENT_CHANNELS,
    "model_channels": 2048,
    "num_blocks": 28,
    "num_heads": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "crossattn_emb_channels": TEXT_WIDTH,
    "pos_emb_cls": "rope3d",
    "use_adaln_lora": True,
    "adaln_lora_dim": 256,
    "concat_padding_mask": True,
    "max_img_h": 240,
    "max_img_w": 240,
    "max_frames": 128,
}

#: Qwen3-0.6B, as Anima's ``text_encoder`` uses it. The stack is read at its
#: LAST hidden state — not an intermediate stack the way FLUX.2-klein reads
#: Qwen3 — which is why the cross-attention width is 1x the encoder width and
#: not 3x.
TEXT_ENCODER: Final[Mapping[str, Any]] = {
    "hidden_size": TEXT_WIDTH,
    "num_hidden_layers": 28,
    "hidden_states_layer": -1,
    "max_length": TEXT_TOKENS,
}

#: The endpoint's preset grid (ie#345), as ``(width, height)``. Trained for
#: 512^2..1536^2, every side a multiple of 16. It is ENDPOINT policy and not a
#: family bucket axis — there are no buckets on the eager tier — but it is
#: recorded here because it is what a ``BucketMap`` will map from, and because
#: ``anima_serve.denoiser_tokens`` over these rows is the bucket axis the
#: promotion gets.
PRESETS: Final[tuple[tuple[int, int], ...]] = (
    (1024, 1024),
    (1536, 1536),
    (1024, 768),
    (768, 1024),
    (1536, 1024),
    (1024, 1536),
    (1360, 768),
    (768, 1360),
)


#: Anima's base checkpoint.
#:
#: The layout axes are preserved BY VALUE from the endpoint's retired ``Slot``
#: (pgw#1346 K1, ie#740). ``sm89+`` is the DECODABLE floor for the rowwise fp8
#: lane — the rowwise GEMM's sm90 is the fast path, not the floor.
#:
#: ⚠️ **The 8 GB bf16 floor is CARRIED, and it is known to be wrong.** The
#: endpoint's own comment records the ie#706 census result: DECLARED 8 against a
#: MEASURED 10.6 GiB peak. It is migrated unchanged and flagged rather than
#: corrected upward, because inventing a number is precisely what a by-value
#: migration exists to refuse, and because an under-declared MINIMUM costs a
#: degrade rung rather than a refusal. Correcting it is th#683 profiling's call,
#: on a measurement, and it must move in the endpoint and here together.
ANIMA: Final = ModelSpec(
    name="anima",
    tuned=AnimaTuned,
    lora_tuned=AnimaLoraTuned,
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram8g",
    },
)

#: Anima's Qwen3 tokenizer tree — the pgw#1346 K5 answer, made concrete.
#:
#: K5 asked whether an auxiliary model an endpoint loads beside its checkpoint
#: (a RIFE interpolator, a latent upsampler, a tokenizer tree) is a ``Model`` or
#: whether the SDK keeps a second, non-family way to name bytes. W1b-1 ruled it
#: is a ``Model``: an eager ``ModelSpec`` with ``tuned=`` omitted, which is what
#: made ``tuned`` optional on that tier. This and its sibling below are the
#: first two in the catalog.
#:
#: ``tuned`` is deliberately absent, and that has a consequence worth naming:
#: ``ModelSpec._register`` publishes NOTHING for a model with no tuned schema,
#: so these two names never reach the hub's family vocabulary. That is the
#: intended behaviour (K8) — a tokenizer answers no inference question, and
#: registering an empty schema under its name would put a word into the hub's
#: vocabulary that nothing can stamp.
#:
#: ``layouts_undeclarable`` is carried BY VALUE from the endpoint's ``Slot``,
#: verbatim (pgw#1346 K2). It is DECLARED undeclarable, not undeclared:
#: discovery refuses a model that names neither, and the family surface's
#: ``DEFAULT_LAYOUT = "bf16"`` would otherwise silently claim a tensor contract
#: for bytes that contain no tensors at all.
_TOKENIZER_UNDECLARABLE: Final = (
    "tokenizer files only — no tensors, so no tensor-layout contract describes "
    "these bytes"
)

ANIMA_QWEN3_TOKENIZER: Final = ModelSpec(
    name="anima_qwen3_tokenizer",
    layouts_undeclarable=_TOKENIZER_UNDECLARABLE,
)

#: Anima's T5-XXL tokenizer tree — vocabulary files only, and NOT a T5 encoder.
#:
#: There is no T5-XXL network anywhere in this pipeline. The ids this tokenizer
#: produces index a learned embedding INSIDE the DiT's LLM adapter, which is
#: also why the endpoint fetches vocab files only rather than the ~9 GB model.
#: Stated because "t5_tokenizer" reads like a slot holding a text encoder, and
#: a reader who believes that will look for a second text tower that does not
#: exist.
ANIMA_T5_TOKENIZER: Final = ModelSpec(
    name="anima_t5_tokenizer",
    layouts_undeclarable=_TOKENIZER_UNDECLARABLE,
)

__all__ = [
    "ANIMA",
    "ANIMA_QWEN3_TOKENIZER",
    "ANIMA_T5_TOKENIZER",
    "PRESETS",
    "TEXT_ENCODER",
    "TRANSFORMER",
]
