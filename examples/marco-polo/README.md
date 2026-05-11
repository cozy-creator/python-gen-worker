# marco-polo

The hello-world inference endpoint. Send `{"text": "marco"}`, get back `{"response": "polo"}`. Anything else gets a snarky fallback.

## What it demonstrates
- Minimal `@inference_function` shape — msgspec.Struct in, msgspec.Struct out.
- Cooperative cancellation via `ctx.is_canceled()`.
- No GPU, no models, no dependencies.

## When to copy it
- Smoke-testing a tenant deployment pipeline end-to-end.
- Measuring round-trip latency (no compute = pure overhead measurement).
- Starting a new CPU-only inference endpoint.

## Files
- `src/marco_polo/main.py` — 15 lines of handler.
- `endpoint.toml`, `Dockerfile`, `pyproject.toml` — the standard four files every endpoint ships.
