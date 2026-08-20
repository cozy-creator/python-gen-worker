- **pgw#1546: warm `gen-worker compile` stops paying per-artifact torch
  bookkeeping over bytes that are already here — 140.5 s → seconds.** The
  "warm" run [[pgw#1533]] measured spent 5.47 s per specialization spawning a
  mint child (torch import + `torch.export.load`) purely to re-derive a cg-key
  the engine cache already stores; the actual store work was 0.21 s. `compile`
  now consults torchcg's new torch-free `Engine.reuse_index` and resolves the
  existing mint directly (`reused` outcome), still running the pgw#1533 publish
  + serving-reader read-back per artifact. A fully-warm run (everything
  published) no longer imports the author module or torch at all: presence is
  checked before build policy, the module name comes from a parse-only
  `endpoint_module_name` (still the one reader of `endpoint.toml`'s `main =`),
  and `host_sm` asks `nvidia-smi` before falling back to torch. Boot-time
  adoption stops re-hashing and re-copying verified artifacts on every boot:
  torchcg's `LocalGraphStore` stamps a verified copy-out and serves an
  untouched destination by identity check (tcg#72).
