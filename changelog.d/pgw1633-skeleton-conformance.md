- **pgw#1633: every fleet pipeline class is now asked, on CPU and from its
  configs alone, whether a checkpoint-shaped fill leaves anything on meta.**
  pgw#1626 was found by a rented pod: the meta build breaks the tie a tied
  encoder's checkpoint relies on to OMIT an alias, and without a `retie()` the
  survivor check refused a correct checkpoint — 100% of invokes, on every
  T5-bearing pipeline. It was fixed for the two classes that had been hit. The
  class, though, is "an architecture ties a parameter", and that is a property
  of every architecture the fleet will ever serve. `skeleton.build_modules`
  splits the weight-bearing half of the meta build out of `build` (they share
  one index reader, so the two cannot disagree about what a `model_index.json`
  says), and `tests/test_skeleton_conformance_pgw1633.py` runs meta-build →
  fill-by-the-checkpoint's-key-set → `retie` → `meta_survivors == ∅` over
  twelve vendored config-only trees in 34 s with no GPU and no weights. The tie
  structure is derived from the model after `retie`, by object identity —
  never from the `_tied_weights_keys` class attribute, which lists names a
  class MIGHT tie and would score `Qwen2_5_VLForConditionalGeneration` as
  broken on a checkpoint that carries the tensor. wan-2.2's `UMT5EncoderModel`
  and the Qwen3 family, both recorded by pgw#1626 as forward exposure on their
  held 2.0.0 majors, are pre-cleared by it, each with a red arm proving the
  assertion still has teeth.
