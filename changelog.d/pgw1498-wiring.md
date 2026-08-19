- **GGUF diffusion loads now go through OUR decode, so pgw#1498's kernels are on the serving
  path instead of merely correct.** `models/loading.load_gguf_pipeline` no longer hands the
  checkpoint to diffusers' `GGUFQuantizationConfig`. The denoiser is built from its CONFIG and
  filled with ggml block bytes (`models/gguf_diffusers.build_denoiser` →
  `gguf_torch.install_quantized_weights`), so the weights reside as uint8 buffers on punned
  Linear/Conv/Embedding leaves and each forward decodes its own weight.

  Three things the delegated path could not do and now happen: **convs and embeddings quantize**
  (diffusers puns `nn.Linear` only, so a quantized conv landed as byte-shaped data in a dense
  parameter); **every residency walk reports the QUANTIZED size** (`GGUFParameter` reports the
  dequantized shape, which over-reports a 4-bit denoiser by the compression ratio — the one
  number this lane exists to move); and **the LoRA branch machinery reaches the leaves**.

- **`fp8_storage.BASE_ATTR` is now `_cozy_pun_base` and `gguf_torch` imports it.** "the plain
  class this leaf was punned from" is one concept, so `structural_base` is one function and
  `w8a8_lora.branch_modules` targets a pun it has never heard of by construction. With two
  spellings a GGML Linear was invisible to adapter targeting while every walk still looked
  correct; the red arm is on record in `tests/test_gguf_serving.py`.

- **The `dequant_ahead` tier dial is driven from the residency lease**, through the same
  `apply_low_vram_config(stream_budget_bytes=...)` entry point pgw#1497's `partial_stream` rung
  uses, and capped by real free VRAM the same way. Paul's three-tier ruling in one sentence: a
  worker handed surplus memory decodes as many weights as the surplus pays for ONCE at load
  (largest first, which is the only order that shrinks the per-forward transient), and a
  constrained worker decodes per forward. Nothing is ever re-quantized. The dial runs BEFORE the
  rung is chosen, because turning it changes the footprint the chooser reads.

- **A weight lane and a precision class exist for the storage format.** `"gguf"` joins
  `STAMPABLE_BASE_EXECUTION_LANES` (a GGML denoiser traces differently from a plain one, so it is
  its own compiled-graph family; bucketed branches are `gguf-lora<N>`), and `ladder.CLASS_GGUF`
  joins the producer's declarable precision classes. **Owed and named, not done:** tensorhub's
  `precision.StoredPrecisionOf` does not rank the class yet, and no QUANT layout contract names
  the ggml block encodings — the same tensorhub half as the ingest normalization.

- **A source seam, so the store swap is one constructor.** `SingleFileGguf` is the community
  `.gguf` edge (it borrows diffusers' single-file KEY MAPPING and nothing else);
  `NormalizedTensors` takes tensorfs `TensorView`s straight out of a `LocalCAS`. Both build the
  same denoiser through the same call, asserted bit-identical.

- **Found while proving it:** tensorfs' `gguf-v1` planner caps a tensor name at 63 bytes
  (llama.cpp's `GGML_MAX_NAME`), and diffusers keys blow through it —
  `down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.weight` is 69. So a `.gguf`
  CONTAINER can never carry our key layout, which is an argument FOR the ingest half rather than
  a defect in it: normalize to per-tensor CAS regions with our own metadata and no container in
  the middle.
