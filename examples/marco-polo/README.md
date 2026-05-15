# marco-polo

The hello-world inference endpoint. Send `{"text": "marco"}`, get back `{"response": "polo"}`. Anything else gets a snarky fallback.

## What it demonstrates
- Minimal `@inference` shape — msgspec.Struct in, msgspec.Struct out.
- Cooperative cancellation via `ctx.raise_if_canceled()`.
- No GPU, no models, no dependencies.

## When to copy it
- Smoke-testing a tenant deployment pipeline end-to-end.
- Measuring round-trip latency (no compute = pure overhead measurement).
- Starting a new CPU-only inference endpoint.

## Local smoke test

```bash
$ gen-worker run --payload '{"text":"marco"}'
{"event":"result","value":{"response":"polo"}}

$ gen-worker run --payload '{"text":"hello"}'
{"event":"result","value":{"response":"Bro you're supposed to say 'marco'!"}}

$ gen-worker run --payload '{"text":"marco"}' | jq -r .value.response
polo
```

`gen-worker run` invokes the endpoint method in the local Python
interpreter — no docker-compose, no orchestrator. stdout for results,
stderr for events. See [../../docs/local-dev.md](../../docs/local-dev.md).

## Files
- `src/marco_polo/main.py` — 15 lines of handler.
- `endpoint.toml`, `pyproject.toml` — Tensorhub generates the Dockerfile for this endpoint from managed build hints.
