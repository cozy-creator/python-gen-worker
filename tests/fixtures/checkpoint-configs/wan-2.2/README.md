# Wan 2.2 TI2V-5B config-only checkpoint tree

The `[derive] checkpoint_configs` source for `scripts/lint_author_derive.py --run`
(se#748 tier 2). Five JSON files, ~3 KB — **no weights, ever** (the
weights-locality rule is absolute: multi-GB artifacts never transit a dev box).

## Which lane, and why this one

wan-2.2 serves THREE checkpoints under one tensorfs lane document
(`wan22.diffusers-bf16@1`). This tree is the **dense TI2V-5B** shape:
one `transformer` (no `transformer_2`), `expand_timesteps: true`, VAE
compression 16 spatial / 4 temporal. It is the cheapest of the three to trace
and the only one whose `vram24g` floor a micro pod can hold.

The A14B pair differs in exactly the ways the endpoint's own code names: a
second byte-identical `transformer_2` component, `expand_timesteps` null/false,
`boundary_ratio` set, and VAE compression 8/4. Adding a second tree is a
follow-on, not a blocker — `[derive] checkpoint_configs` takes one path.

## ⚠️ What is MEASURED here and what is NOT

`_class_name`, the component set, the scheduler class and its flow fields
(`prediction_type: flow_prediction`, `use_flow_sigmas: true`), the patch size
`[1, 2, 2]`, `expand_timesteps: true` and the 16/4 VAE compression are this
repo's own recorded facts, carried across from the deleted
`AOT-EXPORT-DECLARATION.md` and from `src/wan_2_2/main.py`'s measured comments.

The remaining architecture numbers (`num_layers`, `d_model`, `ffn_dim`,
`vocab_size`, `base_dim`, …) are the released repo's published values as
recorded upstream. They have **not** been round-tripped against a real
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` tree on this branch, because doing so means
resolving the repo, which is the pod-side leg. `author-ci.toml` records
`[proof] status = "never-run"` for exactly this reason — the first real
`gen-worker release derive --checkpoint tests/fixtures/checkpoint-configs`
verifies them, and any that are wrong fail loudly there rather than quietly
here.
