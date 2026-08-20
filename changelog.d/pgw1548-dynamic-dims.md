- **pgw#1567: the derive's trace dtype comes from the LANE DECLARATION, never
  the mounted checkpoint.** `TraceLoadContext.component_dtype` led its ladder
  with the tree's own safetensors headers, so a derive against a stock fp16
  tree traced fp16 graphs under `sd15.diffusers-bf16@1` and armed 14 graphs a
  bf16 pod could never enter — silently. The lane now answers first for every
  component of the tree it governs; the checkpoint speaks only for a DERIVED
  lane, which has no contract to read a dtype from. That is the serve path's
  own order: the loader is dtype PASSTHROUGH and the STORE converts a tree
  through the layout contract before a pod mounts it. A mounted tree that
  disagrees is NAMED once per component instead of followed.

- **pgw#1548: `--dynamic-axes off|batch|aspect|all`** on `gen-worker lock` and
  `gen-worker release derive` exports the axes that VARY across the observed
  calls as `torch.export.Dim`s, collapsing the shape fan. Measured on a fixture
  with sd15's structure (3 aspect buckets x 2 CFG modes): 6 specializations
  off, 3 with `batch`, 2 with `aspect`, **1** with `all`. `off` is the default
  and stays it until the per-model serve benchmark says otherwise.
