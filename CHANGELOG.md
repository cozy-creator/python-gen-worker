# Changelog

## 0.13.2 (2026-07-10)

- **gw#451: media uploads target the capability-token-bound owner.**
  `/api/v1/media/:owner/uploads` is authorized by the token's `upload_media`
  grant, which is bound to the canonical invoking-org uuid in the token's
  `tenant` claim. The URL owner segment (and `X-Cozy-Owner`) now come from
  that claim instead of the dispatch-stamped `ctx.owner`, which can be a slug
  or a destination-repo owner resolving to a DIFFERENT org. Live failure:
  J19 run34 trained 500/500 steps, then `TrainingContext.save_image` 403'd on
  `/api/v1/media/tensorhub/uploads` (slug) while the grant was bound to the
  invoker-org uuid; inference outputs only worked because their dispatch
  already stamped the uuid. Dev/local paths without a JWT keep `ctx.owner`.

## 0.13.1 (2026-07-10)

- **gw#442: clone workdir flock — concurrent duplicate clones serialize.**
  Two clones of the same (provider, source, destination) share the resumable
  workdir; hf_hub's local-dir download unlinks + re-fetches files the peer
  clone is mid-reading, so the leading clone's convert phase failed with
  `FileNotFoundError` on a shard `snapshot_download` had just written (live:
  e2e J19, crash-recovery re-queue put the same Qwen-Image-Edit-2511 clone on
  one worker twice). `run_clone` now holds an exclusive flock
  (`.clone-<digest>.lock`) for its whole lifetime; a duplicate blocks, then
  (with th#592 banking) publishes by CAS reference without downloading.

## 0.13.0 (2026-07-09)

- **Breaking (gw#424): the standalone trainer runtime is deleted** —
  `src/gen_worker/trainer/`, the `WORKER_MODE=trainer` entrypoint branch,
  `WORKER_MODE`/`TRAINER_JOB_SPEC_PATH` settings, and `examples/training-smoke`
  are gone. Training runs as `@endpoint(kind="training")` through the normal
  executor. The `[trainer]` extra is renamed `[datasets]` (pyarrow, used by
  `gen_worker.convert.dataset`).
- **gw#425: TrainingContext v1 — delegated trainers.** `resolve_dataset` is
  rewritten against the tensorhub datasets materialize route (th#642 wire
  format): presigned parquet shards stream to disk (bounded memory),
  blake3-verified, retried; it lives on `_PublisherMixin` together with new
  `dataset_paths` and `checkpoint_dir`, so Conversion/Dataset/TrainingContext
  all share the producer surface. The executor materializes
  `payload.datasets` (DatasetRef) before the handler runs, mirroring the
  reserved-source contract. New `gen_worker.subproc.run_process` runs a
  delegated trainer subprocess (line-streaming callback, ctx-cancellation →
  SIGTERM process group → SIGKILL). Per-job capability tokens renew in the
  background at ~80% TTL via `POST /v1/worker/capability/renew` (client half
  of tensorhub #561), presenting the transport's rotated worker JWT. Bugfix:
  dataset list/create responses read `dataset_id` (previously `id`, which
  never matched).
- **gw#438: UUID dataset refs + progress emitter everywhere.** Slash-less
  dataset refs (bare UUIDs, the production form after th#641) hit
  `GET /datasets/:id/materialize` directly; `owner/name` refs keep working
  for local/dev (list param fixed to `?tenant=`). The executor now wires a
  progress emitter into every orchestrated context kind, so `ctx.progress` /
  `ctx.log` ride the JobProgress stream into the hub's SSE output; checkpoint
  saves emit `request.checkpoint` events (step_number, output_kind, size).
- feat: `GEN_WORKER_FORBID_CPU_OFFLOAD` veto — refuses CPU-touching inference
  placements (no CUDA, or offload spilling weights to system RAM) on dev boxes.
- note: the gw#424/gw#425 entries above briefly sat under 0.12.3 in this file;
  the published 0.12.3 wheel (tagged at gw#415) does not contain them.

## 0.12.3

- **SVDQuant/nunchaku 4-bit loader mode (gw#415).** A `#svdq-fp4-rN` /
  `#svdq-int4-rN` flavor (diffusers tree whose denoiser dir holds one
  nunchaku single-file checkpoint) is detected from safetensors
  `__metadata__` and served by swapping the nunchaku transformer into the
  standard pipeline (`gen_worker.models.svdq`). Hard (nunchaku, diffusers,
  torch/cu) pin matrix with typed `SvdqStackError` at selection AND load —
  nunchaku 1.2.x requires diffusers>=0.36,<0.37 (gw#405 live crash on
  0.38/0.39). New fit verdicts `svdq_fp4` / `svdq_int4`: on sm_120/121 a
  fitting svdq-fp4 row outranks everything (faster AND smaller than
  fp8-storage, measured); svdq-int4 (sm_75–89) is a fit rung ahead of
  emergency-nf4 only. `variant_fit`/`select_variant` are binding-aware;
  worker capabilities now probe `nunchaku` + `deepcompressor`.
  Convert side: `gen_worker.convert.build_svdq_flavor_tree` /
  `fetch_svdq_checkpoint` build the flavor shape for the mirror + produce
  lanes (>5GB artifacts refused: sharding would strip nunchaku metadata).

## 0.12.2 (2026-07-09)

- fix: CLI str/Path model-slot injection passes the snapshot path instead of loading a pipeline (gw#416)
- fix: promote-or-die exempts object-less RAM ledger entries — no more retry livelock on CUDA hosts (gw#417)
- feat: emergency nf4 quantization is always-on for CUDA hosts (env flag removed); bitsandbytes added to the [torch] extra (gw#420)
- residency tests rewritten against real tiny pipelines (no fake pipes)
- note: 0.12.1 on PyPI was published from a pre-fix tree (gw#427) — use 0.12.2

## 0.12.0

- **`gen_worker.convert`: the cozy-convert workspace package folded into
  gen-worker proper.** The model ETL (HF/Civitai ingest, streaming dtype
  cast / fp8 storage cast, bnb + GGUF quant, singlefile↔diffusers repackage,
  Tensorhub `/commits` publish) is now a standard part of the library:
  `from gen_worker.convert import publish_flavors, ProducedFlavor, Source`.
  `packages/cozy_convert` is deleted; there is no separate `cozy-convert`
  distribution and the staged (never-published) cozy-convert 0.1.0 PyPI
  release is obsolete — publishing gen-worker 0.12.0 supersedes it. New core
  dependency: `gguf>=0.10.0` (small, pure-python). torch/safetensors remain
  optional (`gen-worker[torch]`); `import gen_worker` stays convert-free and
  torch-free (import-graph guard now covers `gen_worker.convert`). Docs:
  `docs/convert.md`.

## 0.11.2

- Republish: the 0.11.0 and 0.11.1 PyPI wheels were both built from a stale local checkout (19 commits behind master, mixed-commit tree) and lack `allow_lora`, `LoraOverlay`, and `inductor_counters`. No code changes vs 0.11.1 master; version bump only, publish from clean `origin/master` HEAD.
- gitignore `.runtime/`.

## 0.11.1

- **Republish from HEAD.** The 0.11.0 PyPI wheel was stale (missing
  `allow_lora` gw#393/ie#358, compile-honesty gw#391); no code changes vs
  HEAD, version bump only to supersede the stale 0.11.0 wheel.

## 0.11.0

- **Per-request LoRA overlays (gw#393).** `ModelBinding.loras` +
  `LoraOverlay{ref, weight}` on the wire; the executor resolves each overlay
  ref (ordinary tensorhub-CAS refs, no upstream fetch), applies them as
  unfused adapters around the handler under the executing() pin, and
  guarantees unload on every exit path (OK / error / cancel / deadline). A
  digest-keyed `AdapterCache` (byte-capped RAM LRU of parsed state dicts)
  makes repeat requests cheap; `ctx.loras` exposes the resolved set
  read-only. `Hub()`/`HF()` gain `allow_lora=` (endpoint opt-in, requires
  `Compile(family=)`); `run --list` surfaces the flag (ie#358).

- **Adapter residency: repeat LoRA requests cost ~50ms of machinery, not
  seconds (#399).** LoRA adapters now stay ATTACHED to the resident pipeline
  (stable `ref@digest`-derived adapter names); each request only toggles the
  ACTIVE set — `set_adapters` + `enable_lora` in, `disable_lora` out on every
  exit path. Zero-leakage becomes explicit activation: adapter-free requests
  run with adapters disabled (and self-protect against a crashed teardown).
  Attached-but-inactive adapters are LRU-evicted under count/byte caps
  (`GEN_WORKER_ATTACHED_LORA_MAX`, `GEN_WORKER_ATTACHED_LORA_MAX_BYTES`);
  demotion out of VRAM drops attachments (new `Residency.pre_demote` hook),
  re-attached lazily from the AdapterCache on next use. Forensics for the
  ie#358 pilot's +5.7s repeat delta (4090, SDXL, the pilot's own 171MB LoRA):
  `load_lora_weights` re-attach was ~1.6-1.9s/request and unfused adapter
  compute adds ~59ms/denoise-step; residency removes the re-attach entirely
  (activate 23ms / disable 24ms / enable 26ms measured on the 4090).

- **Per-SKU TensorRT engine artifacts on the compile-cache rails (gw#390).**
  A second producer/consumer cell kind alongside inductor:
  `trt-<sku>-trt<maj.min>-<precision>` flavors carry a weight-stripped
  refittable engine + a value-matched refit map — one engine serves every
  fine-tune of a family, refit from the weights already resident in VRAM.
  `gen_worker.trt_engine` handles key/verify (full TRT version, plans are
  version-locked), deterministic pack/unpack, and build (ONNX export ->
  STRIP_PLAN|REFIT); the executor's boot-attach/hot-adopt dispatch prefers
  TRT over inductor when both resolve, any refit failure unwraps to eager.
  `hub_policy` advertises `tensorrt==<version>` in `installed_libs` (th#575).

- **Two-format quantization policy: fp8-E4M3 storage + emergency nf4 (gw#389,
  th#546).** Runtime `quantize=` on HF/Hub bindings is removed; the platform
  serves exactly two STORED quantized formats — fp8 E4M3 (universal,
  per-layer upcast to compute dtype via diffusers layerwise casting) and
  nvfp4 (Blackwell). `storage_dtype="fp8"` replaces it; `#fp8`-flavored
  snapshots are detected via safetensors headers and their storage precision
  preserved instead of upcasting into 2x VRAM. An emergency nf4 rung
  (`GEN_WORKER_EMERGENCY_QUANT`, cozy-local only) runtime-quantizes the
  denoiser when even the downloaded flavor can't fit free VRAM, surfaced as
  `runs (emergency quality)`. Ladder: bf16 -> #fp8 -> #nvfp4 -> emergency-nf4
  -> offload.

- **Streaming dtype cast + fp8-E4M3 storage cast in cozy_convert (gw#395,
  gw#396).** `_stream_reencode` casts one tensor at a time (peak anonymous
  RAM ~ largest single tensor regardless of model size; proven on a 22GiB
  fp32 fixture under a 4G cgroup — 513MiB peak for bf16, 1281MiB for fp8).
  `streaming_fp8_storage_cast` produces F8_E4M3-stored weight tensors
  matching the runtime fp8-storage consumption path; weight-only nvfp4 is
  deliberately not shipped (te#44 quality verdict was a hard FAIL). Off-policy
  quant surface (torchao inline paths, awq/gptq, fp8:e5m2/int8) pruned per
  `QUANTIZATION-POLICY.md`; bnb nf4/fp4 inline kept as the emergency-rung
  producer. The old buffering `StreamingWriter` is deleted.

- **Flavor-collapse ref-grammar conformance + producer publish mode=replace
  (#112, th#597).** `parse_model_ref` validates the flavor token against the
  documented grammar `owner/repo[:tag][@sha256|@blake3:<hex>][#flavor]` (one
  lowercased token; multi-`#` refs now raise instead of silently parsing as
  one bogus token) — shared conformance vectors vendored and byte-identical
  with tensorhub. `publish_flavors` now defaults to `mode=replace` (a
  producer's flavor export is a complete tree; the old merge default let a
  `#fp8` export merge with the mirror's fp16 base, the te#44 root cause).
  `mode=merge` stays as an explicit opt-in for overlay publishes.

- **Flashpack format support removed (gw#388).** Dropped everywhere: the
  unsafe-format gate is safetensors-only, the hub capability probe no longer
  advertises flashpack, cozy_convert loses the flashpack converter/extra.
  Evidence (e2e#114): flashpack loses 3.0x cold / 2.7x warm to plain
  safetensors and is dormant upstream.

- **Compile-cell adoption: honest cache-hit proof + rekey (#391).** ADOPTED now
  means the seeded inductor cell actually served the warmup trace: the worker
  reports FX-graph `cache_hits`/`cache_misses` + `warmup_s` in the ADOPTED
  ModelEvent, and a warmup observing zero hits rolls back to eager with
  `adopt_failed:cache_miss` (no `warmup()` = `adopt_failed:no_warmup`).
  Artifact key gains the producer gen-worker version (format 2 — pre-391 cells
  are refused); `build()` prepares the pipeline through the consumer's exact
  path (`place_pipeline`) so producer and consumer trace identical graphs;
  seeding mid-process clears inductor's latched path caches.

- **Video output media metadata (#387).** `VideoAsset` gains optional probed
  container metadata (`duration_s`, `fps`, `width`, `height`, `has_audio`,
  `sample_rate`); `ctx.save_video` fills it via PyAV (best-effort). New
  `io.write_video(ctx, ref, frames, fps=, audio=, audio_sample_rate=)`
  encodes H.264 + AAC-stereo mp4 (PyAV; mirrors diffusers ltx2
  `export_utils.encode_video`) so video endpoints stop hand-rolling
  tempfile + `export_to_video` and stop dropping generated audio. New
  `video` extra (`av` + `numpy`). `ExpectedOutput` gains `duration_s`
  (literal or `input.<field>` ref, seconds) emitted into the discovery
  manifest for submit-time planning; media-seconds settlement is th#572.

- **Residency unification + worker-side VRAM juggling + disk GC (#369,
  #370, #371).** The `Residency` registry now owns the executor's pipelines:
  worker-built pipelines register per ref with their own measured allocator
  delta (multi-model endpoints no longer report `vram_bytes=0`); tenant-loaded
  refs carry the residual. Model loads serialize under a load lock; free-VRAM
  probes sum across all CUDA devices. `ensure_setup` runs `make_room` before
  loading — idle LRU pipelines demote to the warm CPU-RAM tier instead of the
  new load degrading down the offload ladder — and hub `UNLOAD` demotes
  instead of destroying; the next RunJob/LOAD promotes RAM→VRAM in seconds.
  `demote()` only performs transitions it can actually execute (movable
  object + RAM headroom); otherwise the executor tears the owning record down
  and books every ref back to disk. Disk retention exists now: a persisted
  ref index, a pre-download headroom gate, LRU disk GC honoring `keep` +
  in-use pins + a grace window (keep-pressure escape still emits EVICTED),
  fail-fast `insufficient_disk`, and a boot-time rescan so Hello.models
  matches disk truth across restarts.

- **Conversion ETL split out as `cozy-convert` (#367) — breaking.**
  `gen_worker.clone` and `gen_worker.conversion` are gone; the mirror /
  convert / publish ETL and the conversion tenant SDK (`Source`, `Dataset`,
  `ProducedFlavor`, `StreamingWriter`, calibration) now live in the
  `packages/cozy_convert` workspace package (wheel `cozy-convert`).
  `import gen_worker` is torch- and conversion-free (guarded by an
  import-graph test). Inside cozy_convert: one streaming shard writer
  replaces the seven IO modules, a ~300-LOC classifier +
  `snapshot_download(allow_patterns=…)` replaces the 1,324-LOC
  hf_classifier, the hand-rolled GGUF binary parser is replaced by the
  `gguf` package, and clone finalize is ONE path targeting tensorhub's
  HF-shaped `/commits` write API (`mode: merge|replace`) — the
  enumerate-prior-latest-and-delete overwrite hack is gone.
  `ConversionContext` stays in gen_worker but loses `open_output_writer`;
  the `flashpack` dependency moved to cozy_convert.

- **API rewrite (#368) — breaking, no aliases.** ONE `@endpoint` decorator
  (function = stateless, class + optional `setup()` = stateful, `kind=` for
  conversion/training/dataset, async-generator = streaming, `runtime="vllm"`/
  `"llama-server"` = engine-hosted server subprocess with boot/health-wait/
  abort/shutdown). Deleted `@inference`/`@invocable`/`@batched_inference` and
  the per-kind `.function` aliases. Bindings are now single-positional-ref
  `HF(id, revision=, dtype=, subfolder=, files=)` / `Hub(ref, tag=, flavor=)` /
  `Civitai(id, version=)` / `ModelScope(id, ...)`; slot names come only from
  the models-dict key or injected param name; `variants={name: (binding,
  Resources)}` is the one variant mechanism (replaces Case/parametrize +
  dispatch + `.flavor()`/`.dtype()` chainables). `Resources(gpu, vram_gb,
  compute_capability, libraries)` with `vram_gb` implying `gpu`.
  `RequestContext` slimmed to 15 members (`cancelled`/`raise_if_cancelled`,
  typed `save_image/audio/video`, `generator(seed)`); producer methods live on
  the Conversion/Dataset/Training subclasses. The worker owns placement/
  offload (`models.memory.place_pipeline`); `gen_worker.apply_low_vram_config`
  re-export removed. `[tool.gen_worker] main` in pyproject replaces
  endpoint.toml. CLI: `describe` folded into `run --list`; warm-socket attach
  is explicit `--attach`; `_models` payload overrides dropped. New
  `BatchItemDelta` streaming struct. Legacy worker-side capability-claims
  precheck deleted (server enforces).

- models layer rewrite (#366, #358): one async `ensure_local()` download path
  (tensorhub CAS / HF snapshot + small variant selector / civitai fetch), `Residency`
  LRU VRAM/RAM/disk manager (shared components counted once, pin-while-executing,
  free-VRAM-driven eviction), `models.memory` free-VRAM offload ladder, measured
  `ModelEvent.vram_bytes`, snapshot digest-poisoning retry fix, produced-dtype
  stamping for inline conversion. Deleted `pipeline/` (PipelineLoader),
  `inference_memory` (now `gen_worker.models.memory`), and the legacy
  cache/shared_components/hf_selection/hf_downloader/ref_downloader modules.

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
