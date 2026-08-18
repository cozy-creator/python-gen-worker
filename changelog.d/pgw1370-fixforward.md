- **`gen-worker release derive` imports a module that no longer exists.** `_entrypoints` still
  reached for `gen_worker.api.model_base.Model` after pgw#1382 moved the base to
  `gen_worker.serving.model` — a function-local import, so nothing caught it until the command
  ran, and then EVERY derive died at `ModuleNotFoundError` before reading a single entrypoint.
  The module-level `Model` was already the right one; the stale local shadow is deleted.
