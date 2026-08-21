# Foundation-1 derive fixture — config-only, and DO NOT "correct" it

Fetched verbatim from the **served** packaging: `tensorhub/foundation-1` @prod,
checkpoint `sha256:d2f53a2ec1adb0d3c07dc15a65742a3573e0f6d98a2a412299a3f79870878aa4`,
mirrored from `tintwotin/Foundation-1-Diffusers` @ `bc4bd07e4c10f8e794e48336582448171f5dbd5f`.
8 JSON files, 27.5 KB, **zero weight bytes**.

Deliberately excluded: `*.safetensors` (weights), `tokenizer/tokenizer.json`
(2.4 MB vocab blob), `tokenizer/spiece.model` (791 KB binary sentencepiece),
`README.md`. There are no `*.safetensors.index.json` shard maps in this tree —
every component is a single file, which is itself a fact worth knowing before
someone goes looking for them.

## Values a reader will want to change, and must not

This endpoint and `../../../../stable-audio-open` serve **one shared lane
document** (`stable-audio.diffusers-fp16@1`) because their transformer
**headers** are identical — same 445 tensor names, shapes and dtypes, same
sha256 `fe6b07a5…`. **That does not make their config trees identical**, and the
differences below are real, measured, and correct:

| key | this tree | `stable-audio-open` | |
|---|---|---|---|
| `transformer/config.json` → `sample_size` | **430.6640625** | 1024.0 | genuinely different |
| `_diffusers_version` (every component) | **0.38.0.dev0** | 0.30.0.dev0 | re-exported later |
| `scheduler/scheduler_config.json` values | — | — | **byte-identical** |

`sample_size = 430.6640625` looks like a typo next to a round 1024.0. It is not.
It is the value the fine-tune was exported with, it is what the hub serves, and
"fixing" it to 1024.0 would make this fixture describe a checkpoint that does
not exist. The same goes for the `_diffusers_version` skew — the two checkpoints
were exported eight minor versions apart.

## What IS shared, and why that is the point

`vae/` (`AutoencoderOobleck`) and `text_encoder/` (T5) are **byte-identical**
between the two checkpoints (`af3c43eb…` and `6cf39d0c…`). Only `transformer/`
and `projection_model/` differ — which is what makes Foundation-1 a genuine
fine-tune rather than a rebrand, and is exactly why the lane document declares
the transformer **only**. Declaring the shared components would make family
detection TIE between these two checkpoints, and the T5 in particular is
name-identical with `musicgen`'s (all 99 tensors), which would tie two unrelated
audio families to each other.

## dtype

`float16`, not `bfloat16`. The transformer histograms `{F16: 445}` and the hub
records `dtype=fp16`. The endpoint's v1 slot declared `plain.bf16@1`; that was
an artifact-layout string that disagreed with the bytes, and it died with this
migration.
