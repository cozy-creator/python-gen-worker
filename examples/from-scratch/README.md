# from-scratch

A `@training_function` that emits an **orphan checkpoint** — a brand-new set of weights with no source repo, no parent lineage. The SDK accepts an empty lineage array and lands the checkpoint as a root node in the lineage DAG.

## What it demonstrates
- The `from-scratch` training kind: `@training_function(kind="from-scratch", concurrency="sequential")` — declares to the runtime that this job genuinely has no upstream model to materialize.
- **`ProducedFlavor`** as the return contract — the function generates weights, writes them to a path, returns `[ProducedFlavor(path=..., flavor=...)]`; the library handles upload + finalize + tag application.
- Tenant code never touches tensorhub's upload API directly — the SDK owns the session lifecycle.

## When to copy it
- Generating random-init weights for a new architecture.
- Producing a "blank" base model that downstream conversion/training jobs can build off.
- Any job that produces checkpoints from nothing (synthesis, distillation from a non-Cozy source, etc.).

## Files
- `from_scratch.py` — the function; uses `torch.manual_seed` for deterministic output.
- `endpoint.toml` — declares CPU-only resources (this example doesn't need GPU).
