- **Model residency lands (`gen_worker.serving.residency`), Paul's
  admission-before-allocation ruling encoded.** Placement is the worker's
  decision: before an instance is even constructed, its exact weight bytes
  (tensorfs manifest per lane) plus an activation-headroom estimate are
  reserved against the VRAM budget; LRU residents demote until it fits; a
  model that can never fit refuses typed at admission — never a CUDA OOM
  mid-load. Loads are serialized per GPU; residency is two-tier
  (VRAM -> host-staged -> chunk store, re-promotion is an H2D copy, never a
  disk walk); tier moves happen only between requests (single-flight per
  instance defines the window) and the author never observes them. The
  `ModelBackend`/`InstanceSizer` protocols are the pgw#1382 seam — the
  SDK-core Model wrapper implements the moves; this engine orders them.
