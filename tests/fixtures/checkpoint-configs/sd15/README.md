# SD1.5 config-only checkpoint tree

The `[derive] checkpoint_configs` source for `scripts/lint_author_derive.py --run`
(se#748 tier 2). Verbatim configs from
`stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
— **no weights, ever** (weights-locality rule): 9 JSON files, 72 KB, fetched
over HTTPS one file at a time.

This is also the source of the two numbers `Sd15Model.load` derives the canvas
table from: `unet/config.json` `sample_size = 64` and `vae/config.json`'s four
`block_out_channels` (VAE scale factor 8), so the native square edge is 512.
`cross_attention_dim` is 768 and `use_linear_projection` is ABSENT — which is
exactly the rank discriminator `sd15.diffusers-bf16@1` uses to refuse an SD2.x
tree (SD2 sets it true, making the 32 `proj_{in,out}` weights rank-2 Linears
where these are rank-4 1x1 Conv2ds).

**Deliberately omitted:** `tokenizer/vocab.json` and `tokenizer/merges.txt`
(~1.5 MB together). They carry no weights but they are not "KB of JSON", and
the derive traces on meta tensors with synthesized prompts. Same call sdxl's
fixture made, same reasoning: if `gen-worker release derive` turns out to need
real vocabs, that is a pgw#1370 question, not a reason to commit megabytes.
