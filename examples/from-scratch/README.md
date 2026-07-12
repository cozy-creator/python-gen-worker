# from-scratch

An `@endpoint(kind="conversion")` that publishes an **orphan checkpoint** — a brand-new set of weights with no source repo, no parent lineage.

## What it demonstrates
- **The producer publish contract**: write files locally, call `gen_worker.convert.publish_flavors(ctx, flavors)` — one Tensorhub commit per `ProducedFlavor` — and return a result struct. Nothing publishes implicitly; generator handlers are rejected for producer kinds.
- Tenant code never touches tensorhub's upload API directly — `publish_flavors` owns hashing, presigned part PUTs, dedup, and finalize.

## When to copy it
- Generating random-init weights for a new architecture.
- Producing a "blank" base model that downstream conversion/training jobs can build off.
- Any job that produces checkpoints from nothing (synthesis, distillation from a non-Cozy source, etc.).

## Files
- `from_scratch.py` — the endpoint; uses `torch.manual_seed` for deterministic output.
- `endpoint.toml` — CPU-only build profile (this example doesn't need GPU).
