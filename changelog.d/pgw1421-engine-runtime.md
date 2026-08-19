- **pgw#1421: an endpoint whose model is served by an external binary can BOOT
  it again.** The pgw#1373 hardcut deleted `gen_worker.runtimes.{server,llama}`
  — 747 lines of `ServerHandle` / `vllm_server` / `llama_server` /
  `DegradingBoot` — with no v2 successor, while the platform kept ratifying the
  tier around the hole: `Model.unload`'s docstring names *"external server
  processes"* as its reason to exist, `models/materialized_view.py` calls
  `llama-server -m` PERMANENT tier 3, and `serving/streaming/engine.py` refuses
  block-quantized containers into the pytorch path BY DESIGN. Teardown
  survived, the loader deferred to it, and boot was gone. `gen_worker.serving.
  engine_runtime` is the successor, on the post-hardcut surface:
  `ctx.engine(LlamaServer(...))` / `ctx.engine(VllmServer(...))` is the one
  spelling and the sibling of `ctx.load(...)`. Only fleet consumers:
  `qwen3.6-27b-mtp-gguf` and `qwen3.6-35b-a3b` (se#773).

- **pgw#1421: the spec DECLARES, the platform SUPERVISES.** An `EngineSpec` is
  frozen and knows only the engine and its flags — no path, no port, no
  process. `ctx.engine(spec)` supplies the checkpoint tree from the deploy
  binding, allocates the port, walks the boot ladder, waits on real liveness,
  and REGISTERS the handle. `boot_timeout_s` does not exist and will not: a
  boot is bounded by SILENCE on the engine's own output
  (`stall.SilenceWindow`), never by a clock, because a flat deadline whose only
  liveness check is `proc.poll()` cannot tell a healthy 35B cold load from a
  wedge. The red/green pair is in the suite — the SAME 3-second boot under a
  1.5-second window dies when the child goes quiet and succeeds when it keeps
  talking.

- **pgw#1421: engine teardown is STRUCTURAL, not remembered.**
  `EndpointHost.evict` stops every engine the load context started AFTER the
  author's `unload`, whatever that did — the suite's arm has an `unload` that
  RAISES and the process is still reaped. An engine subprocess is invisible to
  torch's allocator, so a stranded one is VRAM the next residency admit can
  neither see nor reclaim. `stop` signals the process GROUP (vLLM's workers
  survive a bare SIGTERM to the parent and keep the card pinned) and is
  idempotent, so an author who stops it anyway pays nothing.

- **pgw#1421: every phase of a boot is a typed, countable row.** New activity
  kind `engine_boot` with a CLOSED phase vocabulary — `engine_planned`
  (the ladder and its rungs), `engine_started` (argv + port), `engine_healthy`
  (with the measured boot wall in the numeric `duration_ms` column),
  `engine_boot_failed`, `engine_stopped` — so "where did this pod's engine boot
  go" groups on `(kind, phase)` with no join and no regex over a sentence.
  `boot_stages` cannot answer it: it decomposes THIS process's cold start and
  an engine subprocess is invisible to every one of its spans. A DEGRADED rung
  additionally confesses on `serve_degrade`, the countable quality channel —
  two questions, two rows, deliberately (the z-image finding that `report_*`
  does not subsume `emit_event`).

- **pgw#1421: llama.cpp degrades instead of dying.** `LlamaServer` resolves the
  checkpoint tree to its single logical GGUF model (split shards count as one;
  several distinct quants fail closed), reads the header, sizes `-ngl` and the
  context to the free-VRAM budget, and steps down through half the GPU layers
  to CPU-only rather than failing the boot. `VllmServer` states ONE rung
  because vLLM sizes itself and refuses rather than degrades — a ladder whose
  lower rungs cannot fire would read as resilience the endpoint does not have.

- **pgw#1421: the endpoint.lock names the engine, read STATICALLY.** Discovery
  parses `ctx.engine(LlamaServer(...))` off the AST — importing nothing, the
  same promise `_pipeline_class` makes — and lifts the answer into a third
  derived census beside `execution_lanes` and `decode_set`:
  `engine_runtimes[] = {entrypoint, slot, model_class, runtime}`. Omitted
  entirely when nothing hosts an engine, so a lock carrying the block IS an
  engine-hosted endpoint, and `entrypoints[]` stays byte-identical to what the
  hub already decodes. The pytorch CONTROL arm asserts the absence.

- **pgw#1421: an engine-hosted model is SELF-LOADING by construction, and adds
  no second surface for saying so.** `self_loading=` landed as pgw#1431 fix (b)
  (`2449c6b1`), and the two engine specs inherit the whole discovery/publish
  path through it unchanged: a marked slot emits `self_loading` instead of
  `pipeline_class` and clears the publish gate. What this lane adds is the
  refusal an UNMARKED one gets. It used to be *"could not read the pipeline
  class … write `ctx.load(StableDiffusionXLPipeline)`"* — a false sentence
  (discovery knows exactly what the slot boots) carrying unfollowable advice
  (the streaming engine refuses a block-quantized container into the pytorch
  path BY DESIGN). It now names the engine and hands over the line to write:
  *"slot 'model' is ENGINE-HOSTED (llama-server) … declare
  `self_loading="served by llama-server; ctx.load drives no part of it"`"*.
  `docs/endpoint-authoring.md` stops teaching the deleted
  `@endpoint(runtime="vllm")` API, and `subproc.py`'s dangling reference to
  `runtimes.server` names its successor.
