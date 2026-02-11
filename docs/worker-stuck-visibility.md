# Worker Stuck/Error Visibility

This document defines how `python-gen-worker` reports startup/runtime progress so
operators can quickly diagnose failures.

## Startup Phase Signals

The worker emits structured startup logs with event name `worker.startup.phase`
and `phase` values:

- `boot`
- `cache_preflight_started`
- `cache_preflight_ok` / `cache_preflight_failed`
- `cache_preflight_fallback_attempt` / `cache_preflight_fallback_enabled`
- `scheduler_connecting`
- `registered`
- `ready`
- `startup_timeout_unregistered`

Each event includes core debug fields:

- `worker_id`
- `scheduler_addr`
- `pid`, `uid`, `gid`, `cwd`
- `elapsed_ms` since process start

## Fatal Crash Signal

On top-level fatal errors, the worker emits `worker.fatal` with:

- `phase`
- `exception_class`
- `exception_message`
- `traceback`
- `exit_code`

This is emitted best-effort to scheduler (if connected) and always to logs.

## Registration Timeout Watchdog

The worker enforces startup registration timeout:

- Env: `WORKER_REGISTER_TIMEOUT_S` (default `90`)
- If registration does not complete in time:
  - emits `worker.startup_timeout_unregistered`
  - logs startup phase `startup_timeout_unregistered`
  - exits non-zero (`RuntimeError("startup_timeout_unregistered")`)

## Task Lifecycle Signals

Per-run events emitted:

- `task.received`
- `task.started`
- `task.model_resolve.started|completed|failed`
- `task.model_load.started|completed|failed`
- `task.inference.started|completed|failed`
- `task.completed` / `task.failed`

## Stuck (Soft Watchdog) Signals

Long-running phase warnings emit:

- `task.model_resolve.stuck`
- `task.model_load.stuck`
- `task.inference.stuck`

Each includes `elapsed_ms`, `warn_after_s`, and phase context.

Env controls:

- `WORKER_WARN_MODEL_RESOLVE_S` (default `30`)
- `WORKER_WARN_MODEL_LOAD_S` (default `60`)
- `WORKER_WARN_INFERENCE_S` (default `60`)

## Orchestrator Surfacing Contract

`gen-orchestrator` should surface these states in logs/API:

- startup: `launch_failed`, `startup_timeout`, `registered`, `ready`
- runtime: `task_stuck`, `task_failed`, `worker_fatal`

At minimum, "waiting for worker connection" should include the last known worker
startup phase and timeout/failure reason when available.

