# Stable Audio Open 1.0 derive fixture — config-only

Fetched verbatim from the **served** packaging: `tensorhub/stable-audio-open`
@prod, checkpoint `sha256:9938ca7a1b3d2362e65c15fa643432cec8dbb1819e0eae490e7014d6fb92d93b`,
mirrored from `stabilityai/stable-audio-open-1.0` @ `f21265c1e2710b3bd2386596943f0007f55f802e`.
10 JSON files, 34 KB, **zero weight bytes**.

Deliberately excluded: `*.safetensors` (weights), `tokenizer/tokenizer.json`
(2.4 MB vocab blob), `tokenizer/spiece.model` (791 KB binary sentencepiece),
`README.md`, `LICENSE.md`. No `*.safetensors.index.json` shard maps exist in
this tree — every component is a single file.

## Two extra files the sibling tree does not have

`model_config.json` (4.2 KB) and `vae_model_config.json` (3.6 KB) are
**stable-audio-tools** configs that Stability ships alongside the diffusers
layout. `../../../../foundation-1` has neither. They are kept because the tree
is fetched verbatim and they are genuinely part of the served packaging — not
because the derive needs them.

## dtype — the one value a reader will want to "fix"

`float16`, **not** `bfloat16`, and this is the checkpoint's own truth rather
than a transcription slip. Measured two independent ways: the hub's
`validator_metadata.dtype_counts` reports `{F16: 445}` for the transformer, and
recomputing the histogram from a ranged safetensors-header read agrees exactly.
The hub's checkpoint row records `dtype=fp16`.

This endpoint's **v1 slot declared `plain.bf16@1`**, which disagreed with every
byte the fleet ships. That string was an ARTIFACT layout (quant × topology), not
a serving lane, and it died with the se#783 migration. The lane is
`stable-audio.diffusers-fp16@1`.

Consequence worth knowing: a **bf16 repack** of this architecture would NOT be
matched by that document, and is currently captured by an unrelated family's
document. If such a repack is ever minted it needs its own
`stable-audio.diffusers-bf16@1`, the way sdxl carries both a `diffusers-bf16`
and a `diffusers-fp8-rowwise` lane — never a widening of this one's `dtypes`.

## Shared components are undeclared on purpose

`vae/` (`AutoencoderOobleck`) and `text_encoder/` (T5) are byte-identical with
`foundation-1`'s, so the lane document declares the **transformer only**.
Declaring either would make family detection TIE between the two checkpoints,
and this T5 is additionally name-identical with `musicgen`'s (all 99 tensors),
which would tie two unrelated audio families together.
