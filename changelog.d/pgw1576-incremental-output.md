- **pgw#1576 (feature): incremental output — `@entrypoint(streams=…)` +
  `ctx.emit`, and the ctx-event lane that was dead is alive.** The v1 hardcut
  deleted token streaming with no successor, so `qwen3.6-35b-a3b`,
  `qwen3.6-27b-mtp-gguf` and `joycaption` could not be ported at all. A
  streaming entrypoint now DECLARES its chunk type (one, or a discriminated
  union for a handler that streams several shapes), emits chunks with
  `ctx.emit(TokenDelta(text=…))` from sync or async bodies and any thread, and
  still RETURNS its terminal struct: the wire has a droppable ordered
  `JobProgress` lane and exactly one authoritative `JobResult`, and the
  declaration names both — publish reports `incremental_output` and
  `delta_output_schema` off the spec without executing author code, and no hub
  change was needed. The v1 async-generator shape is refused at import with the
  migration line, because Python forbids `return <value>` inside a generator and
  that shape can therefore express only the droppable half. Found and fixed
  alongside: the v2 worker wired NO emitter at all, so `ctx.progress`,
  `ctx.log`, `ctx.warning` and `ctx.checkpoint` were silently discarded on every
  v2 pod — one JobProgress emitter with one per-(request, attempt) `seq` now
  carries both lanes — and the send queue's two progress-drop sites, silent
  since they were written, each confess a `serve_degrade` row that the hub folds
  into `occurrences`.
