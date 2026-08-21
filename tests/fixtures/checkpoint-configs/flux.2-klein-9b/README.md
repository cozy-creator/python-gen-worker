# FLUX.2 Klein config-only checkpoint tree — ⚠️ THESE ARE THE **4B** CONFIGS

The `[derive] checkpoint_configs` source for `scripts/lint_author_derive.py --run`
(se#748 tier 2). **No weights, ever** (weights-locality rule): 8 JSON files, 10 KB.

**They are not 9B's, and nothing here pretends otherwise.** They are verbatim
from `black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294`.
`black-forest-labs/FLUX.2-klein-9B` is HF-gated (auto, 401) and there is no HF
token on this box, so no 9B config — and no 9B safetensors header — has ever
been read here. `transformer/config.json` below states `num_layers 5`,
`num_single_layers 20`, `joint_attention_dim 7680`: those are 4B's numbers and
9B's will differ.

This is the SAME gap that makes `flux2-klein.diffusers-bf16@1` unverified for
9B (see the module docstring in `src/flux2_klein_9b/main.py`), surfacing in a
second place. It is recorded, not papered over:

* the PR-time gate (`lint_author_derive.py`, tier 1) only requires the
  `[derive]` block to name a real config-only tree, which this is;
* tier 2 (`--run`) is not part of any required check, and `[proof] status` is
  `never-run` with this exact blocker written beside it. **Do not backfill a
  proof from this tree** — a derive against 4B's shapes says nothing about 9B's.

Closing it needs a 9B config read from tensorhub (which holds the checkpoints
the fleet actually serves and needs no token) or one authenticated config-only
fetch — tensorfs#124's two routes. An ungated community mirror is explicitly
NOT acceptable: it may not be byte-identical to what ships.

An ungated fact that DOES generalize, because it is family-level rather than
size-level: `scheduler/scheduler_config.json` is `FlowMatchEulerDiscreteScheduler`
with `shift 3.0` / `base_shift 0.5` / `max_shift 1.15` /
`use_dynamic_shifting: true` and no `beta_*` key — i.e.
`Flux2Klein.canonical_scheduler_config` verbatim.

**Deliberately omitted:** the tokenizer's `vocab.json` / `merges.txt` /
`tokenizer.json` — megabytes, no weights, and the derive traces on meta tensors
with synthesized prompts. Same call sdxl made.
