# FLUX.1-dev checkpoint configs — the SERVED packaging, fetched verbatim

Config-only. **No weights**, and no multi-MB tokenizer vocab blobs. Pulled
whole-file from the hub's own resolve manifest for `tensorhub/flux1-dev`,
which the hub records as a revision-pinned clone of the gated BFL repo — so
these are the bytes the fleet serves, not an HF mirror's.

`author-ci.toml` `[derive] checkpoint_configs` points `gen-worker release
derive` at this tree, and `tests/test_drive.py` builds its real
`FlowMatchEulerDiscreteScheduler` from `scheduler/`. The
`*.safetensors.index.json` shard maps are included deliberately: a derive that
cannot see the shard map cannot enumerate the checkpoint.

## ⚠️ Two values a reader will be tempted to "correct". Do not.

**`scheduler/scheduler_config.json`: `shift` = 3.0, `use_dynamic_shifting` = True.**
FLUX.1-dev and FLUX.1-schnell **DISAGREE** — dev is 3.0/dynamic, schnell is
1.0/static — and FLUX.2 Klein is 3.0/dynamic like dev. So the sibling family
agrees with one arm and not the other, which is exactly how a "harmonising" edit
gets made. Both are upstream's own bytes. This disagreement is also why
`Flux1.canonical_scheduler_config` is `{}` rather than a family constant: no
single value is right for the root (pgw#1428).

**`transformer/config.json`: `guidance_embeds` = True.**
This is the ENTIRE structural difference between the two BFL checkpoints — four
tensors, `time_text_embed.guidance_embedder.linear_1/linear_2` weight and bias
(dev 1160 tensors, schnell 1156). `flux1.diffusers-bf16@1` declares them
OPTIONAL, which is what lets one document serve both at 1160/1160 and 1156/1156
rather than either half-matching.
