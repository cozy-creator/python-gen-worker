# SDXL config-only checkpoint tree

The `[derive] checkpoint_configs` source for `scripts/lint_author_derive.py --run`
(se#748 tier 2). Verbatim configs from
`stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b`
— **no weights, ever** (weights-locality rule): 10 JSON files, 72 KB.

**Deliberately omitted:** `tokenizer*/vocab.json` and `tokenizer*/merges.txt`
(~3.2 MB each pair). They carry no weights but they are not "KB of JSON", and
the derive traces on meta tensors with synthesized prompts. If `gen-worker
release derive` turns out to need real vocabs, that is a pgw#1370 question
(the derive should synthesize or stub them), not a reason to commit 6.5 MB —
see se#751.
