# openai-codex

Runs the OpenAI `codex` CLI in headless mode and streams its JSONL event output back to the caller as incremental deltas.

## What it demonstrates
- **Streaming output** — the function returns `Iterator[Delta]`; each yielded struct is flushed to the caller as it's produced (no buffering until completion).
- **CPU-only** endpoint (no GPU declared).
- **Subprocess + secret env vars** — `CODEX_API_KEY` injected at runtime; the function `raise ValueError(...)` cleanly when it's missing, which the SDK maps to a 4xx for the invoker.
- **Multi-profile build** — `[[build.profiles]]` shows how to declare alternative base images (CPU variant only here).

## When to copy it
- Wrapping a CLI tool as an endpoint (anything you'd shell out to).
- Streaming response patterns — generic example you can adapt for token-by-token LLM output, frame-by-frame video, etc.
- Endpoints that need runtime secrets (API keys) injected via environment.

## Files
- `src/openai_codex_worker/main.py` — subprocess spawn + JSONL line iteration.
- `endpoint.toml` — declares CPU-only resources + a multi-profile build matrix.
