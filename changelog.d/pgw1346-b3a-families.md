## B3a — Qwen-Image, Z-Image and ERNIE-Image enter the catalog

- **Three declarations, and every bucket set is the endpoint's own artifact count.**
  `QwenImage` carries fourteen packed (width, height) coordinates — one per
  `aot/transformer-<w>x<h>.mint.json`; `Ernie` carries seven presets x two CFG arms, the 14 its
  endpoint states it cannot collapse; `ZImage` carries TWO classes, one per CFG arity, with the
  resolution SYMBOLIC inside each — the endpoint's `shape_strategy="dynamic-collapse"`, and the
  first catalog family whose shape axis is not a bucket.

- **CFG is a different KIND of thing on each of the three, and the declarations say so.**
  Qwen-Image runs true CFG (two sequential batch-1 forwards, so guidance is a call count and not an
  axis at all); ERNIE concatenates the latent batch (so `batch` is a bucket axis); Z-Image
  concatenates the pytree (so the arity is a bucket, and the traced call takes STACKED tensors).

- **`shift_terminal` is implemented, in `gen_worker/model/flow_ladders.py`.** Qwen-Image's published
  scheduler config sets it to 0.02 and `FlowMatchEulerDiscrete` does not read it, so the same
  declared block resolves two different ladders — a declaration that carried the key while the math
  ignored it would say the ladder was stretched and walk an unstretched one. Differenced against
  diffusers at B2's instrument (relative 2e-4, never ULP) across every step count the three
  endpoints reach and across the extremes of Qwen's grid, with our own ladder byte-identical under
  `ATEN_CPU_CAPABILITY=default`. It is a separate module because the scheduler SET is K10's; folding
  it onto `FlowMatchEulerDiscrete` there is a move, not a rewrite.

- 🔴 **The batch plan's "explicit-sigma ladders (new)" line is RETIRED for B3a, by measurement.**
  All three pipelines hand `set_timesteps` the same `linspace(1.0, 1/steps, steps)` the SDK already
  synthesizes — ERNIE spells it `linspace(1.0, 0.0, steps + 1)[:-1]`, which is the same points. A
  distilled lane differs by its step COUNT and its shift, never by a table of numbers.

- 🔴 **`ZImageTuned` gains `shift`, and that is a by-value migration DEFECT found here.** The base
  checkpoint publishes `shift: 6.0` and the official DMD Turbo one `3.0`; today each arrives with
  its own weights, but a declared family has ONE scheduler block. Left undeclared, the DMD lane
  would walk the base ladder — measured at >20% on a nine-step walk.

- 🔴 **The edit arm is a different MODEL, not an instance.** `Qwen-Image-Edit-2511` sets
  `zero_cond_t: true`, which IS a constructor parameter and therefore a different traced module.
  (The t2i config's own `pooled_projection_dim` is the mirror case and is NOT a constructor
  parameter — which is why B1's rule needs measuring rather than eyeballing.) It is also not
  authorable yet: its traced class set is 56 token counts derived by a FUNCTION, and pgw#1112 item 3
  requires the boundary-shrink's parity to be proven on a pod before they are declared.

- 🔴 **NEW BLOCKER K12 — a bucket axis cannot carry a PYTREE arity.** (Filed as K11; renumbered
  because B3b and B4 each claimed K11 for something else the same day.) Z-Image's module takes lists,
  and `torch.export` flattens a list into one input per element, so the two CFG arms exported three
  flat inputs and five — refused as `signature_disagreement` (torchcg G2: one runner is one binding
  whose variants differ only in concrete dimensions). Worked around by stacking at the declaration's
  wrapper and unbinding inside the traced region, which makes the arity a concrete leading
  dimension. The two moved lines are the pipeline's own.

- **Z-Image's two graph rewrites move into the catalog** (`z_image_graph.py`, mint-side). Without
  ie#630's rope buffers a fake-tensor export graph-breaks on a lazily built table; without ie#637's
  arithmetic pad the export is refused for an equality guard that pins the very symbols being
  declared. Both are the endpoint's own and both are differenced against the real upstream
  implementation in tests rather than trusted as transcriptions.

- **ie#740 floors migrated BY VALUE** onto all three declarations and asserted as parsed NUMBERS:
  qwen-image `sm89+` / `vram72g`, z-image `sm89+` / `vram40g`, ernie `vram32g` (bf16 only — no fp8
  rung is invented for a family whose endpoint declares one lane).

- **z-image's export carries SYMBOL NAMES, which no earlier catalog entry's did.** Its ingress
  records `2*s20` / `2*s59` with range `[90, 240]` — derived from the preset grid's own extremes —
  so the digest depends on torch's symbol numbering as well as on the declaration. A second export
  on the same box is byte-identical (each variant traces under its own fresh `ShapeEnv`), but it is
  torch-version-coupled in a way a static family's digest is not. Recorded rather than fixed.

- **All three runners declare `component="transformer"`** (W1b-2's serving fact), so a declared
  instance is servable eagerly the moment its endpoint migrates. It is not exported: the digests
  below are byte-identical with and without it, which is the property W1b-2 designed for.

- **Digests**: `qwen_image` `2737764f8a0c7d44bada4df584353fae`, `z_image`
  `5e15b72bafb26382ebc1672825547f34`, `ernie` `d2a7a63d101d0a80c31af374340a8e8a`.
