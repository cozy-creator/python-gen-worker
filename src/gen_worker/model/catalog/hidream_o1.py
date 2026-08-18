"""HiDream-O1, declared — and the first thing to say is what it is NOT.

**HiDream-O1 is not HiDream-I1.** I1 is a latent MMDiT behind a VAE and a four
tower text stack (CLIP-L, CLIP-G, T5-XXL, Llama-3.1-8B). O1 shares its brand and
nothing else. It is a **unified pixel-space transformer** derived from Qwen3-VL:

* **no VAE.** Noise is initialised in PIXEL space, ``(1, 3, height, width)``, and
  the output needs no decode. ``latent_channels`` is 3 and the scale factor is 1;
* **no external text encoder.** Prompt token ids go straight into the
  transformer's OWN embedding table. There is no encoder forward pass and no
  ``prompt_embeds`` anywhere on the path;
* **one sequence**, carrying text tokens, the target image's vision tokens, any
  reference images' vision tokens and one timestep token, all at width 4096.

Stated at this length because every one of those is a place where a reader who
assumes "diffusion model" will look for a component that does not exist, and
because the endpoint's own README still describes a compile table this code does
not have.

**Why this is a ``ModelSpec`` and not a ``GraphModelSpec``**, measured rather
than chosen — three blockers, the first two of which are the endpoint's own
recorded reasons for deleting its ``compile=`` block:

1. **The text axis cannot be pinned, and cannot honestly be ranged.** O1 has no
   text attention mask, so its sequence cannot be padded to a constant: filler
   would be untrained tokens the model attends to as content. The pipeline
   applies no padding, no ``max_length`` and no truncation at all.
2. **The reference-image axis has no declaration site.** Up to eleven reference
   images append their own vision tokens to the SAME sequence, at
   count-dependent sizes (see ``hidream_o1_serve.reference_edge``). That is
   ie#550's open mechanism, and it is the same shape gap FLUX.2-klein's edit lane
   has. Two free terms in one sequence length is not a bucket axis.
3. **The module code is not reachable from the SDK.** ``Runner.build``
   constructs what the mint traces, and O1's module is DiffSynth-Studio's
   ``Qwen3VLModel``. ``diffsynth`` is not a gen-worker dependency in any extra,
   so a declaration's ``build`` could not run on the authoring box and no
   committed ``<family>.export.json`` could exist.

Nothing regresses: the fleet already serves this endpoint with no compiled graph.

**And one blocker belongs to the SDK, not to this model (pgw#1346 K11).** The
eager tier has no generated ``Model`` type — ``ModelExport`` refuses a document
with no runners, so codegen renders no binding and there is nothing an endpoint
can annotate a handler parameter with. So this declaration publishes a tuned
schema and carries the serving floors, but ``inst.tuned`` is not yet reachable
for it and the endpoint's fourteen ``ctx.defaults`` reads cannot be migrated
until the eager tier gets a binding. That gates every eager model in the
program, including all of B5.

**The one hub-vocabulary divergence, flagged rather than hidden (K11b).** This
model's hub family handle is ``hidream-o1``, with a HYPHEN, and tensorhub
already publishes ``hidream-o1.schema.json`` for it. A ``ModelSpec`` name is a
generated-symbol identifier — ``[a-z][a-z0-9_]*`` — so it cannot carry a hyphen,
and ``ModelSpec._register`` therefore publishes this schema under
``hidream_o1``. The two names must be reconciled before the flip: either the
grammar admits the hub's spelling, or the family is renamed hub-side, or
``ModelSpec`` grows an explicit hub-handle field. FLUX.2-klein has the same
divergence (``flux2_klein_4b`` / ``flux2-klein-4b``) and it is not klein's to
fix either. ``tests/test_hidream_o1_pgw1346.py`` asserts the divergence so that
it is visible rather than discovered on a pod.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..spec import ModelSpec
from .hidream_o1_serve import IMAGE_CHANNELS, PATCH_SIZE, HiDreamO1Tuned

#: O1's transformer architecture, from DiffSynth's ``_build_hidream_config``.
#: Like Anima and unlike every diffusers family here, the architecture is
#: hard-coded in PYTHON rather than published as a ``config.json``, so this
#: block is a transcription of code. There is no cached HiDream repo on any
#: authoring box and none is needed: the config is not checkpoint data.
#:
#: ``num_layers`` is the whole depth — this is a plain 36-block decoder stack,
#: not an MMDiT, so there is no second "single layer" run and no
#: ``joint_attention_dim``: text and image share ONE 4096-wide embedding space.
#: ``mrope_section`` is the rope-axis vocabulary's local spelling; the three
#: sections sum to 64, which is ``head_dim / 2``.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    # Grouped-query attention, 32 query heads over 8 key/value heads. Recorded
    # because it is a live serving hazard rather than trivia: the endpoint
    # fences sage and xformers out of CI precisely because they mishandle this
    # ratio.
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000,
    "mrope_section": (24, 20, 20),
    "patch_size": PATCH_SIZE,
    "in_channels": IMAGE_CHANNELS,
}

#: The vision tower that encodes reference images into the same sequence.
#: Its ``patch_size`` (16) is NOT the diffusion patch size (32) — two different
#: numbers doing two different jobs in one model.
VISION_TOWER: Final[Mapping[str, Any]] = {
    "hidden_size": 1152,
    "depth": 27,
    "num_heads": 16,
    "intermediate_size": 4304,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "out_hidden_size": 4096,
}

#: The endpoint's preset grid (ie#345), as ``(width, height)`` — the eleven
#: TRAINED resolutions, byte-for-byte the upstream ``PREDEFINED_RESOLUTIONS``.
#: Endpoint policy rather than a family bucket axis (there are no buckets on the
#: eager tier), recorded because it is what a ``Compile(shapes=)`` would take
#: the day the two axes above become declarable.
PRESETS: Final[tuple[tuple[int, int], ...]] = (
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
)

#: The maximum reference images one request may carry, and the maximum
#: ``@handle`` references the hub may append. Migrated by value from the
#: endpoint's typed contract.
MAX_REFERENCE_IMAGES: Final = 11


#: HiDream-O1 — one model, two published recipes (a distilled ``dev`` and an
#: undistilled ``full``), which is why the endpoint binds it with
#: ``selected_by="model"``: the CALLER names which one serves. That axis lives
#: on ``Bind`` and not here (pgw#1346 K3) — it names a field of one endpoint's
#: payload, which is not something the model can know about itself.
#:
#: The layout axes are preserved BY VALUE from the endpoint's retired ``Slot``
#: (K1, ie#740): ``sm89+`` is the DECODABLE floor for the rowwise fp8 lane — the
#: rowwise GEMM's sm90 is the fast path, not the floor — and 22 GB is the
#: endpoint's own bf16 scalar for this 8B model.
HIDREAM_O1: Final = ModelSpec(
    name="hidream_o1",
    tuned=HiDreamO1Tuned,
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram22g",
    },
)

#: The hub's own handle for this family, which the declaration above CANNOT
#: spell. Kept as a constant rather than a comment so the divergence is a value
#: a test can assert and a migration can grep for (K11b, see the module header).
HUB_FAMILY: Final = "hidream-o1"

__all__ = [
    "HIDREAM_O1",
    "HUB_FAMILY",
    "MAX_REFERENCE_IMAGES",
    "PRESETS",
    "TRANSFORMER",
    "VISION_TOWER",
]
