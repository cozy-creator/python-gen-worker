# marco-polo

The hello-world inference endpoint. Send `{"text": "marco"}`, get back
`{"response": "polo"}`. Anything else raises `ValidationError` (exercises the
failed-request billing branch).

## What it demonstrates
- Minimal `@endpoint` class — msgspec.Struct in, msgspec.Struct out.
- Cooperative cancellation via `ctx.raise_if_cancelled()`.
- No GPU, no models, no dependencies.

## When to copy it
- Smoke-testing a tenant deployment pipeline end-to-end.
- Measuring round-trip latency (no compute = pure overhead measurement).
- Starting a new CPU-only inference endpoint.

## Local smoke test

```bash
$ gen-worker run --method marco_polo --payload '{"text":"marco"}'
{"event":"result","value":{"response":"polo"}}

$ gen-worker run --method marco_polo --payload '{"text":"hello"}'
# ValidationError — traceback on stderr, exit 1

$ gen-worker run --method marco_polo --payload '{"text":"marco"}' | jq -r .value.response
polo
```

`gen-worker run` invokes the endpoint method in the local Python
interpreter — no docker-compose, no orchestrator. stdout for results,
stderr for events. See [../../docs/local-dev.md](../../docs/local-dev.md).

## Persistent dev server (warm, no cold start per poke)

`gen-worker run` reloads on every call. To keep models resident and fire many
requests warm, run `gen-worker serve` once and hit it with `gen-worker invoke`:

```bash
# terminal 1 — boot once, models held warm; Ctrl-C to stop
$ gen-worker serve
gen-worker serve: ready

# terminal 2 — address by function NAME; no --class/--method
$ gen-worker invoke marco_polo '{"text":"marco"}'
{"response":"polo"}
$ echo '{"text":"marco"}' | gen-worker invoke marco_polo -   # stdin works too
```

`serve` listens on `./.gen-worker.sock` (override `--socket`) and also reads
NDJSON requests from its own stdin. Transport is NDJSON over stdin/UDS locally
vs gRPC in production — see the fidelity caveat in
[../../docs/local-dev.md](../../docs/local-dev.md).

## Files
- `src/marco_polo/main.py` — sync, async-slow, and streaming variants of the
  same handler.
- `pyproject.toml` — Tensorhub generates the Dockerfile for this endpoint
  from managed build hints.
