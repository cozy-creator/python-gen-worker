# OpenAI Codex example worker

This example runs OpenAI's Codex CLI inside a `gen-worker` function and streams Codex's JSONL events back as incremental deltas.

## Build

This image expects an orchestrator to inject `WORKER_JWT` and a Codex API key at runtime.

```bash
docker build -t cozy-example-openai-codex:dev -f Dockerfile .
```

## Run (local)

Codex `exec` supports `CODEX_API_KEY` in headless mode.

```bash
docker run --rm \
  -e WORKER_JWT="..." \
  -e SCHEDULER_PUBLIC_ADDR="host.docker.internal:8080" \
  -e CODEX_API_KEY="..." \
  cozy-example-openai-codex:dev
```
