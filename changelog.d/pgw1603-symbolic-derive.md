- **pgw#1603: trace count = STRUCTURAL-VARIANT count, never bucket count — and
  the program store banks symbolic PARENTS, not per-bucket near-duplicates.**
  The derive now enumerates ITEMS — (lane × defaults-variant ×
  structural-class) — and runs them in parallel processes
  (`--trace-workers`, default `min(items, cores)`; `GEN_WORKER_TRACE_WORKERS`
  is the config spelling). A declared shape axis is ALWAYS traced
  symbolically; the author's STATIC/DYNAMIC choice controls what is MINTED,
  never how many traces run. STATIC buckets are STAMPED by binding concrete
  shapes to the shared symbolic parent (tcg#88) — byte-identical to the
  per-bucket static exports they replace (graph hash AND ingress; spike-proven
  on real and fake mode, torch 2.13), so no lock re-keys. The store banks the
  PARENT under every bucket identity (content-addressed dedup → one blob per
  structural group) and `strip_diagnostics` drops the per-node
  stack-trace/nn_module_stack strings that were ~60% of every blob; the mint's
  compile seam re-derives the exact requested identity from the parent and
  refuses drift. An axis whose guards refuse the symbolic export drops to
  per-bucket tracing for that axis only, and the LOCK SAYS SO. Measured on
  sd15: 28 bucket traces → 4 structural-variant traces; program store 89 MB /
  28 blobs → single-digit MB.
