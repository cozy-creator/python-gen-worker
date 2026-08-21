# FLUX.2 Klein 4B config-only checkpoint tree

The `[derive] checkpoint_configs` source for `scripts/lint_author_derive.py --run`
(se#748 tier 2). Verbatim configs from
`black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294`
— **no weights, ever** (weights-locality rule): 8 JSON files, 10 KB.

BFL's FLUX.2 repos are gated per-repo; this one is not (`gated: false`), which
is why the whole klein family could be documented at all (tensorfs#125).

Two things this tree confirms independently of the header read:

* `scheduler/scheduler_config.json` IS `Flux2Klein.canonical_scheduler_config`
  verbatim — `FlowMatchEulerDiscreteScheduler`, `shift 3.0`, `base_shift 0.5`,
  `max_shift 1.15`, `use_dynamic_shifting: true`, and no `beta_*` key at all.
* `transformer/config.json` reproduces the lane document's fusion shares from
  the config side: `num_attention_heads 24 x attention_head_dim 128` = 3072
  (the declared fusion unit), and `mlp_ratio 3.0` on a GATED MLP is
  `2 x 3 x 3072 = 6 x 3072` — so `single_transformer_blocks.{i}.attn.to_qkv_mlp_proj`
  cuts `[q:1, k:1, v:1, mlp:6]`, never equal thirds.

`model_index.json` carries `is_distilled: true`: the open 4B repo is the
step-distilled checkpoint, and the pipeline reads that flag itself
(`do_classifier_free_guidance = guidance_scale > 1 and not config.is_distilled`).
That is the same fact `Flux2KleinDefaults.step_distilled` / `.cfg` carry on the
hub row, which is why this endpoint serves Base and Turbo through one code path.

**Deliberately omitted:** the tokenizer's `vocab.json` / `merges.txt` /
`tokenizer.json`. They carry no weights but they are not "KB of JSON", and the
derive traces on meta tensors with synthesized prompts. If `gen-worker release
derive` turns out to need real vocabs, that is a pgw#1370 question (the derive
should synthesize or stub them) — not a reason to commit megabytes. Same call
sdxl made.
