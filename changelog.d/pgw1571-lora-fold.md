- **pgw#1571 (correctness): a LoRA on a compiled denoiser served the BASE
  MODEL — fold it into the weights instead.** peft wraps a denoiser's
  submodules, but `aot_serve.wrap_module` replaced its `forward` with the AOTI
  artifact's dispatch and bound the artifact's constants from the base weights
  at arm time, so the wrappers never ran: the request paid for an adapter and
  got the base model, bit-identically, with no refusal and no log (measured —
  eager `max|delta| = 2.2e-02`, compiled-armed `0.0`). Two changes.
  `models.lora_fold.folded(pipe, adapters, rebind=…)` folds the request's
  deltas into the weights IN PLACE (the artifact holds a user-managed pointer,
  so it sees the write), runs the pipeline unchanged, and restores saved
  originals byte for byte — never a delta-subtract, which is the algebraic
  inverse and not the floating-point one. `aot_serve` routes a module carrying
  live peft adapter state to EAGER and says so once, so the silent case cannot
  return. Quantized (fp8/GGML) leaves refuse the fold by name; those lanes keep
  the additive branch. Also settles the LoRA half of pgw#1548: no committed
  lock has a LoRA graph axis (sd15 14 = 2 CFG x 7 aspect, sdxl 18 = 2 x 9, no
  `lora_a`/`lora_b` in any graph), so removing it costs zero specializations.
