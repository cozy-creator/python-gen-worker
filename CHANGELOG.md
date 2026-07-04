# Changelog

## Unreleased

### Breaking

- **New worker <-> orchestrator wire protocol** (`proto/worker_scheduler.proto`,
  package `cozy.scheduler`): ONE bidi `Connect` stream, 12 typed messages,
  single `attempt` fencing token, gRPC HTTP/2 keepalive as the only liveness
  mechanism, results >64KB shipped as `blob_ref`. Full semantics in
  `proto/CONTRACT.md`. No compatibility with the old protocol.
- **Worker core rewritten asyncio-first**: `transport.py` / `registry.py` /
  `executor.py` / `lifecycle.py` / thin `worker.py` replace the old
  ~10k-line `worker.py`. Deleted: aux streams, heartbeats, the JSON
  `worker_event` fabric, `run_metrics_v1`, `api/micro_batch.py`,
  `_worker_support.py`, `wire_protocol.py`. One decorator walker
  (`gen_worker.registry`) now backs the worker, build-time discovery, and
  the CLI.

### Added

- **`gen-worker run` dispatches async handlers** — coroutine and
  async-generator methods run under `asyncio.run`, streaming yields as
  events. marco-polo gained `marco_polo_stream`, an async-generator
  streaming endpoint.

- **`io.write_image` gained `as_type` and `encode_kwargs`.** `as_type`
  re-wraps the returned `Asset` as a typed subclass (e.g. `ImageAsset`) so
  endpoints whose output struct is typed don't round-trip through
  `msgspec.to_builtins`; extra `encode_kwargs` pass through to
  `PIL.Image.save` (e.g. `method=6` for higher-effort WebP).
- **`gen-worker run` handles parametrized (`Case`) functions.** Local method
  collection now enumerates the per-`Case` fan-out functions stamped from a
  single `@invocable` body, binding each row's model + input type.

### Changed

- **Auto-offload ladder gained an `OFF_HEADROOM` threshold.**
  `inference_memory.select_auto_mode` now only drops to fully-unoptimized
  (`"off"`) mode when free VRAM clears a headroom margin, so a
  partially-occupied GPU keeps the vae_only guard against high-resolution
  VAE-decode spikes instead of OOMing.

## 0.8.3

### Fixed

- **Async handler concurrency is no longer capped by the job-executor
  width (~32).** `async def` SerialWorker handlers ran on the shared asyncio
  loop, but each job's dispatcher thread blocked on the coroutine's future, so
  the ThreadPoolExecutor default `min(32, cpu+4)` was the real ceiling for
  async in-flight work (#447). Dispatch is now callback-driven: blocking
  pre-work (GPU semaphore, lazy setup, payload decode, model injection) stays
  on the pooled dispatcher thread, then the coroutine is scheduled onto the
  shared loop and the thread is freed — result encode + send and GPU
  bookkeeping run on the loop at completion. Async streaming handlers are
  driven natively on the loop too (no per-delta cross-thread round-trips).
  Sync handlers are unchanged. Cancellation, GPU-semaphore, and
  one-terminal-result-per-request invariants are preserved.

### Added

- **Workers exit when the capability token is permanently rejected**
  (tensorhub #462-T4). Reconnect already had bounded full-jitter backoff
  (#338), but a worker whose token was revoked/expired spun in that loop
  forever. UNAUTHENTICATED / PERMISSION_DENIED at connect/register or on the
  control stream now counts a CONSECUTIVE auth failure; after
  `GEN_WORKER_MAX_AUTH_FAILURES` (default 10, `0` disables) the worker logs
  `capability token rejected N times; exiting — token is likely
  revoked/expired` and exits so the container is reaped. Any inbound
  scheduler message resets the counter; transient network errors neither
  count nor reset.

## 0.8.2

### Added

- **`gen-worker describe --json`** — machine-readable endpoint introspection
  with no model load: `protocol_version`, `capabilities`, and every function's
  input JSON Schema + model bindings. `serve --list-functions --json` is now a
  thin alias. This is the stable host-integration contract (see
  `docs/host-integration.md`) for tools like cozy.
- **Ergonomic CLI payload args** — `gen-worker run/invoke "a cat" seed=42
  hires=true`: httpie-style `field=value` (coerced to the payload struct's field
  type), `field:=<json>`, `field@file`, and a bare positional for the primary
  field. `--payload '<json>'` still works; tokens merge over it.
- **Request cancellation** — `Ctrl-C` on `run`/`invoke` cancels the in-flight
  request (via `ctx.cancel()`) while a warm `serve` keeps running; a second
  `Ctrl-C` detaches the client. A `{"cancel":{"request_id"}}` control frame is
  the wire mechanism. SIGINT/SIGTERM on `serve` cancel all in-flight requests,
  drain, then shut down.
- **Streamed responses** — `serve` streams each event as produced when a request
  sets `stream:true` (`invoke --stream`, and `run`'s warm-attach), with a
  client-disconnect cancellation backstop.
- **TCP transport** — `serve --listen tcp://0.0.0.0:PORT` + `invoke --socket
  tcp://host:PORT` for cross-process / Docker submission (the Unix socket
  remains the default).
- **`gen-worker repl`** — a load-once interactive single-endpoint session.
- **serve sidecar** — `.gen-worker.serve.json` (pid, listen, protocol_version,
  functions) written on ready and removed on teardown, for host orchestration.
- **`serve --vram-budget GB`** — size the in-process `ModelCache` to a host
  allotment instead of the whole GPU, so several serves co-reside with
  deterministic budgets.
- New docs: `docs/host-integration.md` (the contract) and an expanded
  `docs/local-dev.md` (the three shapes, ergonomic args, Docker topologies).

### Fixed

- Civitai model refs now resolve a MODEL id to its latest version; `.version()`
  pins are honored; a failed lookup fails loud instead of silently downloading
  an unrelated model.
- `describe` accepts the documented `--json` flag (it is the default + only
  format).

## 0.7.21

### Fixed

- **Binding-shape manifests now correctly populate startup readiness
  state.** Pre-fix, gen-worker 0.7.x endpoints (every endpoint built
  with the typed bindings shape from `gen-worker#9`) had no top-level
  `models` / `models_by_function` blocks in their manifest. The 0.7.19
  startup-readiness gate only walked those legacy blocks, so
  `_release_allowed_model_ids` was always `None` for binding-shape
  manifests — the worker emitted `startup_phase=ready` immediately on
  gRPC connect, before any model bytes hit disk. The orchestrator
  flipped `AvailableForRequests=true` and dispatched requests to
  empty-disk workers.
  Fix walks `manifest["functions"][i]["bindings"]` in
  `Worker.__init__`, unions extracted canonical refs into
  `_release_allowed_model_ids`, and builds a per-function
  `_required_refs_by_function` map so `_loading_function_names()`
  computes accurate per-function loading state for binding-shape
  endpoints.
- **HuggingFace ref canonical form now preserves `#flavor`.** Pre-fix,
  `HuggingFaceRef.canonical()` stripped the flavor segment, so
  `disk_models` advertised the bare repo (`owner/repo`) while the
  orchestrator's `RequiredRepoRefs` carried the with-flavor form
  (`owner/repo#bf16`). The cache-locality scorer compared the two with
  exact-string match, always landed on `localityCold`, and parked
  every request waiting for a cold fetch that never satisfied the
  match. FLUX inference requests were observed queued for 249s while
  the worker quietly held the bytes on disk.
  Fix: `HuggingFaceRef` now carries the `flavor` field;
  `parse_model_ref(..., provider="hf")` extracts and preserves it;
  `canonical()` emits `owner/repo[@revision][#flavor]`. The
  orchestrator-side `RequiredRepoRefs` and the worker-side
  `disk_models` now share an identity and route correctly.
- **Terminally-failed required refs no longer block startup readiness.**
  Required refs that fail terminally (HF flavor doesn't exist on the
  repo, 404 / 401 / 403) are now counted as resolved for the
  `_emit_ready_if_all_cached` gate so the worker doesn't sit in
  `models_downloading` forever. Functions whose entire required-ref
  set failed terminally are marked locally unavailable so the dispatch
  gate rejects them with a clear reason.

## 0.7.8

### New

- **`gen_worker.accel` — canonical five-call diffusion acceleration**
  (issue #324). New top-level module exposing the recommended entry
  points for SerialWorker acceleration: `gpu_capability()` (cached
  hardware probe), `compile_diffusion(module)` (torch.compile wrapper
  for the heavy DiT), `apply_fbcache(pipe)` (ParaAttention First-Block
  Cache), `apply_para_attn(pipe)` (ParaAttention sequence-parallel
  adapter), `apply_nvfp4(model)` (NVFP4 weight quantization for
  Blackwell). The lower-level modules (`gen_worker.cache`,
  `gen_worker.compile_helpers`, `gen_worker.quant`,
  `gen_worker.parallelism`) remain available for advanced cases
  (multiple cache backends, multi-precision quant fallbacks, sequence
  parallelism with custom placement); each carries a docstring
  pointing at the corresponding `accel.*` entry point for the common
  case. See [docs/cookbook-acceleration.md](docs/cookbook-acceleration.md).
- **`@batched_inference` class shape** (issue #273). Parallel-to-
  `@inference` decorator for LLM-class workloads (chat / instruct
  models, multimodal captioners, autoregressive TTS). The decorated
  class hosts a single long-lived inference engine; the
  externally-invocable method is an async generator yielding typed
  streaming signals — `IncrementalTokenDelta(text=...)` per delta,
  `Done()` at clean end, `Error(message=...)` for inline failures.
  Worker dispatch leg routes requests through the
  `@batched_inference` codepath without overloading the `@inference`
  function-methods slot. Cooperative cancellation is wired via
  `ctx.cancelled()` — client disconnect (stream-EOF) flips the flag,
  and the tenant's loop calls `engine.abort(request_id)` to release
  the engine slot immediately. **No engine integration yet** — tenants
  construct `AsyncLLMEngine.from_engine_args(...)` (vLLM) or
  `sgl.Engine(...)` (SGLang) in their `setup()`; engine choice,
  precision, and tuning knobs are tenant-owned. See
  [docs/cookbook-batched-llm.md](docs/cookbook-batched-llm.md).
- **`@inference.stage` decorator hardened** (issue #325). Validation
  now fails fast at class-decoration time so tenants see errors
  during `import` rather than at bake / first dispatch:
  - `gpu_class` must be `"small"` or `"large"` — `Literal` typing
    isn't enforced by msgspec at construction, so a typo (`"medium"`,
    `"big"`) now raises `ValueError` with the valid list.
  - `name` (or the method name if not supplied) must produce a non-
    empty slug under the same rules as `@inference.function` wire
    routes; an empty-slug name (`"!!!"`, `""`) raises with the
    slug rules in the error message.
  - Two stages on the same class can't share a name — duplicates
    would silently shadow each other in the manifest's `stages` list
    and in any future remote-dispatch routing table. Cross-class
    duplicates remain legal (stage names are scoped per class).
  - The validated stage spec is plumbed through the manifest as
    `(name, gpu_class, python_name)` so future SDK releases can route
    each stage to a separate worker without endpoint-code changes.

### Breaking

- **`accelerator='cpu'` / `accelerator='gpu'` aliases removed**
  (issue #326). The canonical vocabulary is `'cuda'` (GPU endpoints)
  and `'none'` (CPU-only endpoints — CPU is the *absence* of an
  accelerator, not one). The oxymoronic shorthands were masking typos
  and conflicting with the wire-side meaning of `accelerator`. Both
  legacy spellings now raise at `Resources(...)` construction time
  with a pointer to the canonical value. The check is case-
  insensitive (`'CPU'` / `'GPU'` raise the same way).
- **Discovery-time gate on self-contradictory Resources** (issue
  #326). `Resources(accelerator='none')` paired with any GPU resource
  axis (`requires_gpu=True`, `min_vram_gb`, `min_compute_capability`)
  now raises `ValueError` at decoration time. The combination is
  almost always a copy/paste typo (a CPU port of a GPU endpoint
  where the resources block wasn't pruned) and would otherwise
  silently misroute endpoints. CPU-only endpoints declared cleanly
  (`Resources(accelerator='none')` alone, no GPU axes) continue to
  pass.

### Migration

- **`accelerator='cpu'` → `accelerator='none'`.** CPU-only endpoints
  (small flow-matching audio, CPU-only classifiers) use the
  no-accelerator form. Drop any `requires_gpu=` / `min_vram_gb=` /
  `min_compute_capability=` kwargs from the same `Resources(...)`
  call — they would otherwise trip the new discovery-time gate.
  ```python
  # Before:
  Resources(accelerator='cpu', min_vram_gb=4.0)
  # After:
  Resources(accelerator='none')
  ```
- **`accelerator='gpu'` → `accelerator='cuda'`.** GPU endpoints use
  the explicit CUDA spelling. The `requires_gpu=True` auto-flip on
  the `'cuda'` path is unchanged.
  ```python
  # Before:
  Resources(accelerator='gpu', min_vram_gb=24.0)
  # After:
  Resources(accelerator='cuda', min_vram_gb=24.0)
  ```

## 0.7.7

### Breaking — wire-format hard cut (issue wire-format-bare-refs-typed-provider)

- **No more prefix strings on the wire.** `_wire_ref(binding)` now returns
  `binding.ref` BARE for every provider. The `_binding_to_wire` payload
  carries the typed `provider` field (`"tensorhub"` | `"hf"` | `"civitai"`)
  alongside `ref`; absence of `provider` on a consumer payload defaults
  to `"cozy"` (tensorhub).
- **`parse_model_ref` is no longer LEGACY framing** — the `scheme` alias
  field on `ParsedModelRef` is gone; `provider` is the only field. Every
  internal caller now reads `parsed.provider`.
- **Internal cache key shape changed.** `_resolved_repo_id(ref, ...,
  provider=...)` takes provider explicitly and prefixes non-cozy refs
  with `<provider>::` (double-colon) as an in-process identity tag.
  `cozy` is the implicit default and is elided so existing cozy keys
  round-trip unchanged. This is NOT a wire format — it's an internal
  Python identity string.
- Endpoints rebuilt against 0.7.7 produce manifests that tensorhub
  >= migration 006 accepts. Pre-0.7.7 manifests are rejected with a
  typed migration error pointing at SDK upgrade + endpoint rebuild.

### Cross-repo coordination

- tensorhub migration `006_drop_ref_prefixes.up.sql` strips prefixes
  from existing `function_param_bindings.ref` and `dispatch_table_json`
  entries, populates `provider` from the stripped prefix, marks the
  column NOT NULL, and adds a CHECK constraint forbidding future
  prefixes.
- gen-orchestrator removes prefix-sniff fallback in `BindingProvider`
  and stops `"cozy:" + ref` prepending when sending to workers.
- All 13 inference + 4 training endpoints rebuild against this SDK.

## 0.7.6

### Breaking

- **Class-shape decorators are the only API** (issue #322).
  `@inference` / `@training` / `@dataset` / `@conversion` now decorate a
  class; the function-shape decorators `inference_function`,
  `training_function`, and `realtime_function` are hard-cut migration
  stubs that raise `ImportError` on import or call. The class must
  define `setup(self, **models)` plus one or more methods marked with
  `@inference.function` / `@inference.stage` (and the kind-equivalent
  attributes). All endpoint code must migrate; there is no compat shim
  for the function form.
- **Removed `Repo` string prefixes** in favor of typed provider
  classes (issue #10). `gen_worker.HFRepo("owner/model")` and
  `gen_worker.CivitaiRepo(model_id)` replace prefixed strings (`"hf:…"`,
  `"civitai:…"`). Bare `Repo("owner/repo")` continues to mean tensorhub.
  `Repo.PROVIDER` / `HFRepo.PROVIDER` / `CivitaiRepo.PROVIDER` are
  exposed for introspection.
- **Wire-protocol bump 1.5 → 1.7.** Each binding entry now emits an
  explicit `provider:` field (`cozy` / `hf` / `civitai`). The legacy
  prefix-sniffing path on the orchestrator + worker remains as fallback
  for already-published manifests, but the explicit field is the
  canonical signal going forward. Endpoints must be rebuilt against
  0.7.6 to emit the new field.

### New

- **Acceleration helpers** (issue #324). Four new top-level modules,
  each importable as `gen_worker.<name>`:
  - `gen_worker.cache` — KV-cache + attention-cache helpers.
  - `gen_worker.compile` — `torch.compile` / TensorRT / inductor wrappers
    with hardware-aware fallbacks (e.g. TRT no-ops on pre-Hopper).
  - `gen_worker.quant` — fp8 / int8 / nvfp4 quantization wrappers with
    `fallback="passthrough"` on unsupported hardware (fp8 needs SM 9+,
    nvfp4 needs SM 10+).
  - `gen_worker.parallelism` — xDiT sequence-parallel + tensor-parallel
    helpers, fallback to passthrough on insufficient GPUs.
- **BatchedWorker autoregressive TTS** (issue #327). New
  `gen_worker.runtimes.ar_tts` registry maps AR-TTS models (Chatterbox,
  GPT-SoVITS, Bark, MusicGen, …) onto continuous-batching engines (vLLM
  primary, SGLang where supported). New `gen_worker.engines` module
  hosts the engine wiring.
- **Cross-request micro-batching aggregator**
  (`gen_worker.api.micro_batch`). SerialWorker endpoints opt in by
  declaring `batch_window_ms` + `max_batch` on the `@inference` class
  decorator; payloads arrive at the user method as a list.
- **Decorator-table model bindings** (issue #9). `Repo` + `Dispatch` +
  `Resources` consolidated into the decorator's `models={...}` kwarg.
  `Repo(...).allow_override(*classes)`, `dispatch(field, table)`, and
  reserved `_models` invocation field are stable.
- **Typed provider classes + `_wire_ref` helper** (issue #10). New
  exports from `gen_worker`: `HFRepo`, `CivitaiRepo`, `Binding`,
  `Dispatch`, plus `_wire_ref` for tests.
- **Kind-specific context subclasses**: `ConversionContext` /
  `DatasetContext` / `TrainingContext` extend `RequestContext` with
  only the RPCs appropriate to each endpoint kind.

### Removed

- `inference_function`, `training_function`, `realtime_function`
  function-shape decorators — replaced by class-shape `@inference` /
  `@training` (raise `ImportError` if imported by name and called).
- `ModelRef`, `ModelRefSource`, `Src`, `ScalingHints`,
  `ResourceRequirements` — removed in 0.7.0, still rejected with
  pointer-to-new-API errors.
- `src/gen_worker/conversion/_training_injection.py` and
  `src/gen_worker/conversion/validation.py` (dead code).

## 0.7.0

### New

- Chainable `Repo` + `Dispatch` binding model. Declare model dependencies on
  the decorator's `models={...}` kwarg:
  ```python
  flux = Repo("acme/flux")
  @inference_function(
      resources=Resources(requires_gpu=True, min_vram_gb=22.0),
      models={"pipe": flux.flavor("bf16")},
  )
  def generate(ctx, pipe, payload): ...
  ```
- Payload-driven dispatch via `dispatch(field, table)` — function pins a set
  of picks keyed by a `Literal[...]`-typed payload field.
- `Repo` / `Dispatch` support `.allow_override(*classes)` to permit caller
  substitution within an explicit pipeline-class allowlist.
- Reserved `_models` invocation field — invokers can substitute bindings via
  `{"_models": {"pipe": "owner/repo:tag#flavor"}}` (string or structured
  form). Substitution is atomic.
- `Resources` — merged hardware envelope + cost-shape struct, declared **per
  function**.
- Boot-time self-advertise: the worker compares each function's `Resources`
  against host hardware and marks unavailable functions automatically.

### Breaking

A lot of removed and renamed symbols. There are no compat shims; bare
imports of deleted names raise `ImportError` with a pointer to the new API.
See [docs/endpoint-authoring.md](docs/endpoint-authoring.md) for the full
reference.
