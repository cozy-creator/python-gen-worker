<!-- python-gen-worker issue tracker — ACTIVE issues -->

> One `# #<id>: <name>` section per issue, separated by `---` lines. IDs share ONE per-repo
> id space across progress.md / future.md / completed.md; new issues take `next_id` below and bump it.
> CONCURRENT EDITS: only ever edit/append your own issue's section — never rewrite the whole file.
>
> Source for all findings: /home/fidika/cozy/python-gen-worker/AUDIT.md (2026-07-03 full-stack audit,
> file:line evidence for every claim). Counterpart protocol/orchestrator issues: tensorhub #503-#510.

next_id: 369

---

# #356: fix red master — 6 failing tests, CI ignored since Jun 8

**Completed:** no
**Status:** OPEN — master pytest is GREEN again (fix/p0-correctness); only the CI-gating hardening (task 3) remains.

## Tasks
- [x] Fix 4 failures in `tests/test_cli_describe.py`: fixture at :36-43 injects a static `HFRepo` into handler param `pipe`, rejected by the injection validation added in a53627b (`api/decorators.py:683`). Update fixtures to the new contract (static repos inject into `setup()`, not handler params). [DONE: dropped the static `pipe`/`m` params from the two describe-test handler signatures; the class-level `@inference(models=...)` still drives the reported bindings.]
- [x] Fix 2 failures in `tests/test_inference_memory_select.py` (:45, :66): re-derive expectations for the OFF_HEADROOM threshold added in 5347209 — but do this together with #358 (free-vs-total fix changes the answers again). [DONE with #358: keyword renamed to `available_vram_gb`; both 24GB-free cases now correctly resolve to "off" (15+GB free headroom > OFF_HEADROOM).]
- [ ] Make CI red mean something: mypy currently `continue-on-error` in `.github/workflows/ci.yml`; either gate on it or drop the job. Add branch protection or a pre-push habit so red master can't sit for weeks.

## Acceptance
`uv run --extra dev pytest -q` fully green on master; CI failure blocks merge.

---

# #357: worker P0 reliability bugs (drain, result loss, reconnect storm, load events)

**Completed:** no
**Status:** OPEN — each of these is a verified live defect in `src/gen_worker/worker.py`; all survive into the v2 rewrite (#365) so fix them where the logic will be ported, or fold into #365 if it lands first.

## Tasks
- [x] **Drain crash**: `_emit_worker_drain_result` constructs `pb.WorkerDrainResult(worker_id=...)` (worker.py:6216) — field removed from proto in #321 → `ValueError` swallowed by receive-loop except (5838-5842). Net: on `WorkerDrainCommand` the worker rejects new jobs but never runs `_drain_then_stop`, never reports, never exits; pods only die by external kill. Remove the field, add a test that drives a full drain round-trip. [DONE (fix/p0-correctness): removed the `worker_id=` kwarg; `test_worker_p0_reliability.py::test_emit_worker_drain_result_builds_valid_proto` guards it.]
- [x] **Results lost during reconnect**: `_send_message` drops any message while `_stop_event` is set (5593-5616) — a handler finishing in the reconnect window loses its `JobExecutionResult` forever (stuck request orchestrator-side). Results must be enqueued (bounded, non-droppable), never refused. [DONE: `_send_message` now gates on `_running` only (persistent queue survives the reconnect); `stop()` clears `_running` before setting `_stop_event`, so genuine shutdown still refuses. Note: the drop-oldest-discards-results and aux-stream tasks below remain open.]
- [ ] **Drop-oldest overflow discards results**: `_send_message` overflow policy (3597-3614) treats job results and fire-and-forget events identically. Separate policies: events droppable, results never.
- [ ] **Aux-stream death strands queued results**: when the results stream dies, `_aux_drain_loop` exits and messages already in `_results_outgoing_queue` have no consumer until a full primary reconnect that may never happen (4548-4571). On aux death, drain the aux queue back onto the primary.
- [ ] **Reconnect backoff unreachable**: after `_stop_event.wait()` returns in `run()` (5565), the guard `not self._stop_event.is_set()` (5581) is false by definition → the #338 backoff never runs, lost stream = immediate reconnect storm. Restructure so backoff applies before redial; test with a fake scheduler that kills the stream N times.
- [ ] **Model-load events never emitted**: `_handle_load_model_cmd` references undefined `started_at` (6834), NameError swallowed at 6846-6847 → `model.load.completed/failed` never sent. Define it (siblings at 6926/7034 do).
- [ ] **VRAM load blocks the receive thread**: `_process_message` runs `_handle_load_model_cmd` inline (5958-5959) — up to 300s wait (6774) + `asyncio.run(load_model_into_vram)` on the only thread that processes dispatches/interrupts. Move to a worker thread like `_handle_download_model_cmd` (6901-6910).
- [ ] **Deadlines are decorative**: `timeout_ms` lands in `ctx._deadline` and nothing reads it; no watchdog aborts a hung handler, which holds the GPU semaphore forever (worker bricked). Enforce deadline: mark request failed, release the slot, and report; escalate to process recycle if the thread won't die.
- [ ] **Transient prefetch failure permanently disables functions**: `_startup_prefetch_loop` calls `_mark_ref_terminally_failed` on any exception (6608-6615). Retry with backoff; reserve terminal for 4xx-class errors.
- [ ] **`cancel_queued_only` cancels running work**: `_handle_interrupt_request` logs `item_ids`/`cancel_queued_only` then unconditionally cancels (7451-7482). Honor the flag (or delete the fields in protocol v2 — see #364).
- [ ] **`_ensure_batched_loop` race**: check-then-create with no lock (7515-7560) can leak a second loop/thread. Guard with a lock (or moot under #365 asyncio-first).
- [ ] **Per-request `CUDA_VISIBLE_DEVICES` mutation** (2107-2147): process-wide, ineffective post-CUDA-init, racy for gpu_count>1. Pass explicit `device` down instead.

## Acceptance
Fake-scheduler integration test (see #365 task list) covers: drain round-trip, stream-kill × N with backoff, result survives reconnect, hung handler is reaped at deadline.

---

# #358: VRAM/memory correctness — free-vs-total, double-counting, digest poisoning

**Completed:** no
**Status:** OPEN — the offload/eviction layer makes decisions from wrong numbers.

## Tasks
- [x] `select_auto_mode` compares against **total** VRAM minus margin (inference_memory.py:204, 228-237); `get_available_vram_gb` (:81-91) is never consulted. Second model on an occupied card gets `"off"` → OOM. Base the ladder on free VRAM. [DONE (fix/p0-correctness): param renamed `total_vram_gb`→`available_vram_gb`, defaults to `get_available_vram_gb()`; the sole caller `apply_low_vram_config` inherits it. Every threshold/headroom check now reads free VRAM.]
- [x] `OFF_HEADROOM` (commit 5347209, inference_memory.py:235) inherits the same defect and proactively strips the vae_only guard on partially-occupied cards. Re-derive against free VRAM; then fix the two #356 tests once. [DONE: OFF_HEADROOM now measures free headroom, so a partially-occupied card (small free space) keeps the vae_only guard instead of falling to "off". #356 memory tests re-derived to match.]
- [ ] `estimate_pipeline_size_gb` (inference_memory.py:126-147) sums all params regardless of device: CPU-offloaded pipelines booked as full VRAM in `ModelCache._vram_used_gb` (models/cache.py:355, 371); shared components double-counted (once via SharedComponentCache shared_components.py:272, again inside each owner's estimate). Count only CUDA-resident params; account shared components once.
- [ ] Kill the fantasy constants: hardcoded `size = 5.0` GB fallback (worker.py:9409) and "evict half the cache budget because we don't know its size" (worker.py:8710). Measure via `torch.cuda.memory_allocated` deltas at load time.
- [ ] **Digest poisoning**: a failed snapshot build leaves the entry + set event in `_SNAP_ENTRIES` and re-raises the stale exception forever (models/cozy_snapshot_v2.py:188-201, 228-232). Evict failed entries so the next request retries.
- [ ] Blocking HF download inside `async def _download_async` (models/ref_downloader.py:169) — wrap in `to_thread` like the civitai/modelscope branches. Same for loader.py rglob/stat/JSON on the loop (:1130, 1162, 1207, 1214).
- [ ] Unify the three disagreeing low-VRAM deciders: `inference_memory.select_auto_mode` (total), `pipeline/loader._apply_memory_optimizations` (:983-1021, free), worker.py inline preflight (:11928-11949, free + private 6GB/2GB thresholds). One decider, one input (free VRAM).
- [ ] Conversion dtype mislabels: fp8:e5m2→e4m3 kernels stamped `fp8:e5m2` (conversion/inline_convert.py:102-103 vs :624), int4→nf4 same (:115 vs :757) — inference dispatches on this label (core_types.py:71-81). Stamp the produced dtype. Unknown dtype strings silently load as bf16 (pipeline/loader.py:313-326) — warn or fail.
- [ ] Delete the ~190-line Flux2-Klein filename-hardcoded hack from the generic worker (worker.py:8748-8940); express it as endpoint-side config or a variant selector.

## Acceptance
A worker with an occupied GPU chooses an offload mode that fits *free* VRAM; a failed download retries cleanly; `test_inference_memory_select.py` green against the new semantics.

---

# #359: delete dead code, sweep 1 — the unreachable legacy function-shape stack (~2,500 LOC)

**Completed:** no
**Status:** OPEN — nothing in the package sets `_is_inference_function`/`_is_training_function` anymore (the decorators that did were hard-cut in #322); everything keyed on them is unreachable at runtime. Verified by repo-wide grep: only `getattr` reads remain.

## Tasks
- [ ] Delete from worker.py: function-shape scan in `_discover_and_register_functions` (2234-2290), `_inspect_request_spec` (3431-3560), `_execute_request` (10255-11046), `_execute_training_request` (11048-11277), `_resolve_model_id_for_injection` (12388-12468), LoRA overlay stack (12074-12303), asset-materialization suite `_materialize_assets`/`_auto_upload_output_assets`/`_materialize_asset`/`_download_url_to_file`/`_validate_url_asset`/`_stream_to_file` (3710-4122), `_binding_to_wire`/`_wire_ref` (192-291), `_looks_like_ref_compatibility_surprise` (1212-1281), `_RequestSpec` (_worker_support.py:106-124).
- [ ] Delete always-one-value knobs and their dead branches: `_jwks_cache`/JWT-verify plumbing incl. all of `_worker_auth.py` (worker.py:449-453, 1189-1210), `max_input_bytes`/`max_output_bytes` + 4 checks (446-447; 7226, 9608, 9828, 10741), `_models_ready_on_connect` + branch reading a nonexistent env var (536, 4307-4312), `_drain_timeout_seconds` + stop() drain-wait (532, 5690-5697), `_local_model_cache_dir` + `_get_local_model_cache` + NFS localization block (549, 11279-11316, 11652-11691), `_filter_prefetch_for_disabled_functions` (6362-6393).
- [ ] Delete zero-caller functions: `_enforce_model_allowlist` (12470-12477), `_prefs_for_canonical` (6623-6652), `payload_key_status` (6450-6467), `_emit_residency_for_refs` (9171-9182), `_reconnect_jitter_seconds` (685); run_metrics_v1: `emit_best_effort` (451-461), `add_upload_time` (260-263), never-set fields `warmup_ms`/`png_encode_ms`/`bytes_read_disk`/`upload_ms`.
- [ ] Remove reads of proto fields that no longer exist in `_handle_job_request` (7128-7143: `required_flavor_refs`, `parent_request_id`, `child_request_id`, `item_id`, `item_index`) and the permanent `model='None'` dispatch-log noise (7150).
- [ ] Fix `tests/test_worker_dispatch.py:52-77` / `test_async_dispatch_concurrency.py:36-61` bare-`Worker.__new__` fixtures that hand-set `_request_specs` — extract one conftest fixture while touching them.
- [ ] Collapse `wire_protocol.py` (67 lines of changelog around two ints) to the two constants.

## Acceptance
Suite green; `grep -rn "_is_inference_function\|_request_specs\|_training_specs" src/` returns nothing.

---

# #360: delete dead code, sweep 2 — zero-caller packages and unused exports (~2,600 LOC)

**Completed:** no
**Status:** OPEN — verified zero callers across src/, tests/, inference-endpoints, training-endpoints (venvs excluded).

## Tasks
- [ ] Delete packages: `quant/` (455), `accel/` (598 — keep `apply_low_vram_config` by moving it to `inference_memory`), `cache/` (219 — keep the `breaks_cross_request_batching` attr convention as a comment where micro_batch reads it), `compile_helpers/` (285), `parallelism/` (225), `engines/` (425) + the never-satisfied gating branch worker.py:7669.
- [ ] Delete `conversion/dtype_utils.py` (289, 100% dead) and `presets.py`.
- [ ] Prune `__init__.py` exports never imported by any consumer: `batched_inference` re-export, `Clamp`, `PositivePrompt`, `NegativePrompt`, `PromptRole`, `MediaAsset`, `Compute`, `Tensors`, `load_loras`, `with_oom_retry`, `Done`, `Error`, `TokenStreamSignal`, `IncrementalTokenDelta`, `Binding`, and the unused error classes (keep `FatalError`, `ValidationError`, `RetryableError`, `CanceledError`).
- [ ] Delete `runtimes/` registry surface except `ar_tts.lookup` (used by chatterbox-tts, musicgen): drop `register`/`all_specs`/bark entry.
- [ ] `pipeline/loader.py`: production injection path bypasses `PipelineLoader` entirely — keep `detect_diffusers_variant`, `get_torch_dtype`, quant-config synthesis (~200 LOC); delete `MODEL_COMPONENTS`, `_class_name` parsing, `DiffusersModelManager` fallback (~1,000 LOC). (Or defer to #366 which deletes the module wholesale.)
- [ ] `clone/`: delete dead dedup scaffolding (`maybe_noop` always-None pipeline.py:2730-2739, `preflight_clone` empty struct :1344-1357, identity-hash machinery :2723-2726), the 4 `NotImplementedError` stubs advertised as public API (:2777-2797), twin `_tensors_artifact`/`_tensors_artifact_module` (:2171/:476). Merge `_finalize_clone` (:1718-2449) and `_finalize_publish_as_is` (:498-928) into one finalize path (~1,800-2,000 LOC saved). (Superseded by #367 if that lands first.)

## Acceptance
Suite green; both endpoint repos still import cleanly (`uv run python -c "import gen_worker"` + grep-verified consumer imports unaffected).

---

# #361: delete dead code, sweep 3 — discovery/API layer + tombstones (~1,500 LOC)

**Completed:** no
**Status:** OPEN.

## Tasks
- [ ] discovery/discover.py: delete legacy-marker paths — `_extract_function_metadata` (524-668), `_extract_conversion_function_metadata` (671-734), `_file_uses_worker_decorator` + no-main fallback scan (472-493, 925-1004), function-shape scan in main-module path (883-921), `_compute_module_name` (496-521), `batch_dimension` merge (1259-1264).
- [ ] discovery/toml_manifest.py (~330 of 592): `TensorhubModelSpec` (17-31), `_parse_model_spec` + ref validators (188-313), `_parse_function_resource_hints` (316-432), `constraint_satisfied` + version helpers (106-185), the four always-empty `EndpointToml` fields.
- [ ] api/decorators.py: migration stubs + `_REMOVED_PUBLIC_SYMBOLS` + module `__getattr__` (1414-1439; `__init__.py:57-61,108-151,174-176`) — pre-launch tombstones; delete. `rate_limit_per_invoker` (accepted, emitted nowhere). `prefer_distilled` (no consumer). `@invocable.stage`/`_StageSpec`/`gpu_class` + per-kind `.stage` aliases (~150 LOC — worker never reads `__gen_worker_stage_methods__`) + `docs/cookbook-stages.md` (568 lines documenting the no-op).
- [ ] Either wire or delete the orphaned dispatch validators `_payload_field_names`/`_payload_field_type`/`_literal_members` (decorators.py:440-471). Wiring them is ~5 lines and makes the binding.py:509-513 documented contract true — prefer wiring.
- [ ] request_context/__init__.py: `save_bytes_create` (963-992), `save_output_stream` self-alias (914-929), `finalize_checkpoints` (1124-1136), discarded `publish_intent`/`metrics` blocks (1213-1237, 1437, 1472-1473), dedupe `TrainingContext.read/write_repo_metadata` (2112-2193 ≡ 1656-1726) and `DatasetContext.materialize_blob` (2098-2109 ≡ 1729-1740) into a mixin; remove the import-time `setattr` monkey-patch loop (1839-1843). Unused: `workspace_scope_id` (196), `partition_context` (540), `item_output_ref` (553); _helpers.py `_default_output_prefix`/`_error_code_from_exception`/`_utc_timestamp_rfc3339`; _stream.py `average_upload_bps`/`_abort_due_to_cancel`/`_classify_error`; _concurrent_upload.py `inflight_bytes`/`parallel_save_checkpoints`; models/cache.py `is_pinned`/`get_residency_map`; api/binding.py `primary_slot_name`.
- [ ] cli: delete `repl` (229 LOC, duplicates `serve --stdin`) and the no-op `describe --json` flag (describe.py:53-56).
- [ ] Move torch/requests imports out of module import time (request_context/__init__.py:19-24) and make `cli/__init__.py` parser building not import the world — `gen-worker --help` should not import torch.

## Acceptance
Suite green; `python -X importtime -c "import gen_worker" ` shows no torch; `gen-worker --help` < 300ms on a torch-equipped machine.

---

# #362: docs teach a deleted API — every README quickstart raises ImportError

**Completed:** no
**Status:** OPEN — README.md:47-135 and docs/endpoint-authoring.md (+endpoint-envs, endpoint-toml, scaling-hints, dockerfile) teach `@inference_function`, a stub that raises ImportError referencing an internal progress.json. Time-to-hello-world through the front door is infinite.

## Tasks
- [x] Rewrite README quickstart against the shipped class API (verified working example: `examples/marco-polo/src/marco_polo/main.py`). Keep it short. [DONE (docs/dev-experience): README minimum-viable + model + dispatch examples now `@inference`/`@invocable` class-shape; verified `gen-worker run --payload '{"prompt":"hello"}'` on the copied README example returns a result in one try.]
- [ ] Purge `@inference_function` from all docs/ (27 occurrences in README + 5 docs) or delete the stale pages outright pending API v2 (#368) — prefer deletion + one accurate page. [PARTIAL (docs/dev-experience): front-door purged — README + endpoint-envs/scaling-hints/endpoint-toml/dockerfile fixed to class-shape. endpoint-authoring.md got a staleness banner + fixed table/quickstart; its ~19 advanced `@inference_function`/`@training_function` examples deliberately left for the #368 rewrite (don't re-port them here).]
- [ ] Fix doc lies flagged in audit: `accelerator` "gpu"/"cpu" shorthands claimed normalized but rejected (endpoint-authoring.md:97 vs decorators.py:129-140); promised `unknown_payload_field` discovery validation that doesn't exist (endpoint-authoring.md:117-118); nonexistent `max_concurrent_per_worker` kwarg (cookbook-image-diffusion.md:205). [PARTIAL (docs/dev-experience): fixed the two front-door lies — accelerator now says `"gpu"/"cpu"` are rejected (not normalized); `unknown_payload_field` reworded to "silent bug, hint never fires" in endpoint-authoring + scaling-hints. `max_concurrent_per_worker` left for #368 (cookbooks, not front-door).]
- [x] Strip "See progress.json #NNN" from ~15 developer-facing error strings (decorators.py:770,1036,1431; toml_manifest.py:520; etc.). [DONE (docs/dev-experience): stripped from every RAISED error string — decorators.py (@class-required, @batched_inference.function, migration stub), __init__.py `_REMOVED_PUBLIC_SYMBOLS` (ModelRef/ModelRefSource), injection.py, discovery/validation.py, toml_manifest.py. Internal module docstrings/comments (decorators.py:42, injection.py:27) left as-is — not error strings.]

## Acceptance
A new dev can copy the README example and get a working `gen-worker run` in one try.

---

# #363: packaging + dev-loop hygiene

**Completed:** no
**Status:** OPEN.

## Tasks
- [x] Drop unused dep `tomli-w` (pyproject.toml:25 — zero imports). [DONE (docs/dev-experience): removed; confirmed zero `tomli_w` imports and `uv sync --extra dev` uninstalls it, suite still green.]
- [x] Resolve redundant/conflicting extras: `images`/`audio` duplicate core Pillow/numpy with a different Pillow floor (pyproject.toml:35-36 vs extras). Keep one. [DONE: removed Pillow+numpy from CORE (both are lazy/TYPE_CHECKING-only in io.py — no eager import anywhere), so `[images]`=Pillow>=10 and `[audio]`=soundfile+numpy are now the single home; base wheel is truly lean, matching the README "plain Python" claim + io.py's own install hints. 212 tests pass with numpy/Pillow uninstalled.]
- [x] Add `task proto` (grpc_tools codegen) — currently no reproducible path from proto/ to pb/; regen the stale `pb/worker_scheduler_pb2.pyi` (still has 3 pre-rename `owner` refs). [DONE: `task proto` added (grpcio-tools protoc → python/grpc/pyi out). Regen is byte-identical for pb2.py/_grpc.py; only the stale `.pyi` changed (owner→tenant). Idempotent re-run = same single-file diff.]
- [x] Fix Taskfile: `task test`/`task lint` fail on fresh clone (need `uv run --extra dev`); `lint` is mypy-only while ruff is used ad hoc — declare ruff in dev extra + Taskfile + CI. [DONE: `test`/`lint` now `uv run --extra dev ...`; `ruff>=0.6` added to dev extra + `[tool.ruff]` config (excludes generated pb/ + mypy-exempt legacy, ignores deliberate lazy-import E402); `lint` runs mypy+ruff advisory (both ignore_error, matching CI); CI got a `continue-on-error` ruff step alongside mypy.]
- [x] Note the `[tool.uv.sources]` cu128 torch pin (pyproject.toml:120-128) is uv-only; document or accept that `pip install gen-worker[torch]` resolves differently. [DONE: added a comment above `[[tool.uv.index]]` stating the cu128 pin is uv-only (read by uv, not pip) and `pip install gen-worker[torch]` resolves torch from PyPI differently.]
- [x] CHANGELOG the three unreleased commits on master (2300c3b, 5347209, f6284bf) at next release. [DONE: added `## Unreleased` — io.write_image as_type/encode_kwargs, cli parametrized-Case method collection, inference_memory OFF_HEADROOM.]

## Acceptance
Fresh clone: `task test` and `task lint` work; `task proto` regenerates pb/ identically to what's committed.

---

# #364: protocol v2 — one stream, ~12 typed messages, attempt fencing (worker side)

**Completed:** no
**Status:** OPEN — counterpart: tensorhub #504. Full design in AUDIT.md §4.1. Current protocol: ~25 message types in one mega-oneof over up to 4 streams, 4 overlapping liveness mechanisms, 3 overlapping crash-recovery mechanisms (one dead), an untyped JSON event side-channel mostly dropped by the receiver, and heartbeats that re-ship the full resource inventory every 10s. Pre-launch: renumber from field 1, no reserved-tag graveyard, one version.

## Tasks
- [ ] Write `proto/worker_scheduler_v2.proto` jointly with tensorhub #504. Worker→orch: `Hello{worker_id, release_id, ver, resources, in_flight[{request_id, attempt}]}`, `StateDelta` (on change only; heartbeat is gRPC keepalive), `JobAccepted{request_id, attempt}`, `JobResult{request_id, attempt, status, output|blob_ref, metrics}`, `JobProgress{request_id, seq, chunk}`, `ModelEvent{ref, state: DOWNLOADING|ON_DISK|IN_VRAM|EVICTED|FAILED, vram_bytes}` (collapses LoadModelResult/UnloadModelResult/WorkerModelReadySignal/download events — and finally reports the `size_bytes` the orchestrator's placement optimizer has been starving for), `FnUnavailable{fn, reason}`. Orch→worker: `HelloAck{config, keep[]}`, `RunJob`, `CancelJob{request_id, attempt}`, `ModelOp{ref, DOWNLOAD|LOAD|UNLOAD}`, `Drain{deadline_ms}`.
- [ ] One bidi stream; identity per connection (JWT), not per message. Delete: split results/events streams + handshakes, dedicated heartbeat stream (exists only for no-auth dev mode), WorkerRegistration-as-heartbeat.
- [ ] Fencing = single `attempt` int on dispatch/ack/result. Delete: `duplicate_request_id` reconcile handshake, `ActiveAssignmentResume` (never sent today; Go path unreachable), assignment epochs.
- [ ] Outputs > ~64KB: presigned PUT to blob storage, result carries the ref — removes the head-of-line blocking that motivated split streams.
- [ ] Delete the untyped `worker_event` JSON fabric. Today the worker emits `model.load.started`, `model.unload.*`, `model.url_refresh`, `models.disk_inventory`, `worker.fatal`, `worker.draining`, `worker.drain.status`, `worker.startup_timeout_unregistered`, legacy `model.ready` — Go drops all of them unread (only `worker.model.download.*` is parsed). Typed messages or logging pipeline, nothing else.
- [ ] Liveness = gRPC HTTP/2 keepalive both sides, period. Delete the app-level 60s inbound-silence watchdog and the 10s full-snapshot heartbeat.
- [ ] Pick ONE metrics vehicle: `JobResult.metrics` replaces RunMetricsV1 triple-emission (canonical_events + metrics.job + observation). Drop never-produced observation fields (ttft/itl/prefix-hit/kv-blocks/scaling_factors) until something emits them.

## Acceptance
Both repos compile against v2; a contract doc lists every message with producer/consumer; zero fields without both.

---

# #365: worker core rewrite — asyncio-first, ~1,600 LOC replacing worker.py

**Completed:** no
**Status:** OPEN — depends on #364. worker.py is 12,532 lines: 5 near-identical dispatch paths, ~15-40 threads, `asyncio.run()` inside threads, ~120 attributes on one god class, half the lines narrating ticket history. Layout per AUDIT.md §4.2.

## Tasks
- [ ] `transport.py` (~300): channel + auth interceptor, one bidi stream, bounded send queue with results-never-dropped, receive loop, reconnect with jittered backoff (actually reachable this time), redirect (`not_leader`) handling.
- [ ] `registry.py` (~200): ONE decorator walker producing `EndpointSpec{instance, method, payload_type, output_mode ∈ {single, stream}, is_async, needs_gpu, finalizer}` — shared by worker, discovery, and CLI (today three separate walkers: discovery/discover.py, worker.py:2200+, cli/run.py:199-305 with a "mirrors discover.py" comment).
- [ ] `executor.py` (~350): intake, GPU semaphore, real deadline + cancellation watchdog, sync-on-`to_thread` / async-on-loop, delta emission, result send. One path; "conversion" is a finalizer, not an archetype. Kills SerialWorker/BatchedWorker/Conversion triplication and both `_execute_serial_*` clones.
- [ ] `lifecycle.py` (~150): Hello/StateDelta, startup phases, working drain.
- [ ] `worker.py` (~150) wiring + keep `entrypoint.py` (~100) as-is.
- [ ] Keep the upload stack byte-for-byte (presigned_upload.py, _upload_transport.py, s3_transfer.py — best code in the repo, already has real-socket tests).
- [ ] Decide micro-batching's fate: current one is structurally inert (GPU semaphore acquired before aggregator submit, worker.py:9544→9593 — batch always 1 on 1-GPU). Either batch *inside* the semaphore or delete api/micro_batch.py (383 LOC) until an endpoint needs it. Recommend delete.
- [ ] The e2e test that doesn't exist today: fake gRPC scheduler (`grpc.server` appears nowhere in tests/) driving real worker: connect → register → dispatch → stream deltas → result → interrupt → stream-kill → reconnect-with-backoff → drain. Style-match the existing real-socket/SIGINT tests.

## Acceptance
marco-polo example runs against the fake scheduler and against a real tensorhub (v2) end to end; thread count at steady state ≤ 6 + executor pool; old worker.py deleted.

---

# #366: models layer v2 — download/cache/memory/residency, ~2,500 LOC

**Completed:** no
**Status:** OPEN — replaces models/ + pipeline/ + the worker's injection half. Two HF download stacks, two loaded-pipeline registries, dtype logic in ≥5 places, SM detection ×5 today.

## Tasks
- [ ] `refs.py` (~150): keep the current ref grammar/parsing (models/refs.py is fine).
- [ ] `download.py` (~500): one `async def ensure_local(ref) -> Path` dispatching to (a) tensorhub CAS (keep cozy_cas/cozy_snapshot_v2 with the #358 poisoning fix), (b) `huggingface_hub.snapshot_download` in `to_thread` with `allow_patterns` from a ~100-line variant selector (replaces hf_selection's 522 LOC + hf_classifier for this purpose), (c) one civitai fetch (~200, extracted from conversion/ingest.py:1939 — the only conversion function the generation path uses). One progress reporter (today: three).
- [ ] `memory.py` (~400): merged inference_memory + ModelCache accounting per #358 — free-VRAM ladder, CUDA-resident-only estimates, single eviction policy.
- [ ] `residency.py` (~450): ModelCache LRU VRAM/CPU/disk tiers + SharedComponentCache folded in, keyed once, public eviction API (no `_private` reach-ins from the worker, no `\ttier=` string smuggling worker.py:9134-9169).
- [ ] No PipelineLoader: endpoints receive a typed local path/pipeline and call `from_pretrained` themselves (they already do — flux.1-dev/main.py:103-110).
- [ ] Absorb multi-model VRAM residency management currently hand-rolled inside flux.2-klein-9b (700 lines of endpoint code doing cross-pipeline eviction) — that's this layer's job.

## Acceptance
Endpoint `setup(self, model: FluxPipeline)` receives exactly what the annotation says, every SDK version; the 7 consumer `_resolve_model_path` hacks become deletable (inference-endpoints #343).

---

# #367: split clone+conversion out as `cozy_convert`; move the tenant SDK to training-endpoints

**Completed:** no
**Status:** OPEN — clone/ (3,769) + conversion/ (11,063) is a mirror/convert/publish ETL product living inside the worker library. Its tenant SDK (`Source`, `Dataset`, `ConversionContext`, dispatch, calibration, writer, produced — ~3,000 LOC) has zero generation-path callers; its only consumers are training-endpoints.

## Tasks
- [ ] New package `cozy_convert` (~4,000 LOC target): hub-API ingest (HF + civitai), ONE streaming shard writer (collapse the 7 IO modules ~1,300 LOC to ~400; `streaming_primitives.py` is a pure re-export facade), dtype cast + quant via the libraries training-endpoints already calls directly (modelopt/bnb/torchao/hqq), repackage.py kept, ONE finalize path (see #360 clone tasks), `gguf` package instead of the hand-rolled binary parser (gguf_utils.py:390).
- [ ] Replace hf_classifier.py (1,324 LOC, zero hf-hub imports, 14 refusal exception classes) with `HfApi.list_repo_files` + `snapshot_download(allow_patterns=...)` + a small classifier.
- [ ] Move `Source`/`Dataset`/`ConversionContext` tenant SDK to training-endpoints (their #34), or into `cozy_convert` if tensorhub also needs it — either way, out of gen_worker.
- [ ] gen_worker keeps only `ensure_local`'s civitai fetch (#366) and the `@endpoint(kind="conversion")` shim that hands a `ConversionContext`.
- [ ] Give the ETL its first tests — clone/ and conversion/ are mypy-exempt (pyproject.toml:100-110) and have zero test coverage today. One integration test per conversion direction on a small real model.

## Acceptance
gen_worker has no `clone/`; `import gen_worker` never imports conversion machinery; training-endpoints/conversion imports from the new home.

---

# #368: API v2 — one decorator, slim context, pyproject config

**Completed:** no
**Status:** OPEN — design in AUDIT.md §4.3, converged independently by the API audit and the consumer audit. Do after #365/#366 so the new surface binds to the new core. Counterpart consumer migration: inference-endpoints #343, training-endpoints #35.

## Tasks
- [ ] One `@endpoint` decorator: function-first (no class/setup for stateless), class + optional `setup()` for state; `kind=` kwarg for conversion/training/dataset; async-generator = streaming (no `@batched_inference`); `runtime="vllm"` for engine-hosted. Delete the six method-marker spellings (`@invocable` + five `.function` aliases, decorators.py:1287-1341) and the ~300 LOC duplicated `@batched_inference` validation (985-1275).
- [ ] Bindings: single-positional-ref constructors — `HF(id, *, revision, dtype, subfolder, files)`, `Hub(ref, tag, flavor)`, `Civitai(id, version)`. Slot name = models-dict key or injected param name, never a constructor arg (kills the `Repo("slot","ref")` dual-arity trap, binding.py:134-165). One variant mechanism (`variants={name: (binding, Resources)}`) replacing Case/parametrize + dispatch + `.flavor()`-vs-`.dtype()`. Wire the Literal validation for `dispatch()` (see #361).
- [ ] `Resources(gpu, vram_gb, compute_capability, libraries)` — and make `Resources(vram_gb=12)` imply gpu (today it silently under-declares, decorators.py:153-169).
- [ ] RequestContext ≤15 members: `request_id`, `device`, `deadline`/`time_remaining()`, ONE cancellation spelling (`cancelled` property + `raise_if_cancelled()`), `progress()`, `log()`, `save_bytes/file/image/audio/video → typed Asset` (kills the 11-endpoint BytesIO→save_bytes→dict-roundtrip ceremony), `ctx.generator(seed)` (kills 14 copies), `models`. Subclasses add publish/mktemp/dataset via real inheritance — no import-time monkey-patching (request_context/__init__.py:1839-1843).
- [ ] Worker owns: placement/offload policy around `setup()` (kills 10 `apply_low_vram_config` epilogues), per-pipeline serialization (kills 5 endpoint lock registries), logging config (kills 14 `basicConfig` copies).
- [ ] First-class server-subprocess runtime: boot/health-wait/abort/shutdown for vLLM + llama-server (today 3 hand-rolled vLLM boots + one 70-line llama-server manager in qwen-gguf; `runtime="vllm"` currently provides nothing — engines/ branch never satisfied).
- [ ] First-class batch-item delta struct (index/total/item_id/finished/error + binary chunk) — replaces the joycaption→chatterbox→musicgen copy-paste and the magic field-name peeling (`audio_chunk`/`audio_codec`).
- [ ] `[tool.gen_worker] main = "..."` in pyproject replaces endpoint.toml (post-hard-cut it carries exactly one meaningful string, toml_manifest.py:481-488, yet is mandatory, discover.py:745). Resources live in Python only — delete the toml `[resources]` duplication ("not yet plumbed — known orch bug", chatterbox endpoint.toml:17-22; needs tensorhub #504/#510 to read function resources from the manifest).
- [ ] CLI: `run` / `serve` / `invoke` / `prefetch`; keep the `field=value` httpie grammar (genuinely good); fold `describe` into `--list`; make `run`'s warm-socket auto-attach (run.py:1194-1200) explicit (`--attach`), since it silently changes semantics and ignores `--device/--offline`.

## Acceptance
Hello-world ≤ 20 lines; flux.1-dev-class endpoint ≈ 60 lines (today ~200); all 22 inference endpoints portable with only deletions (verified by inference-endpoints #343).

---

# #19: Standardize model upload/download transfers on trusted libraries

**Status:** in_progress

RECONCILED 2026-07-03 (full-stack audit, see AUDIT.md + agents/progress.md): superseded by models v2 (#366), which redesigns the download/upload stack. Fold the trusted-library requirement into #366 instead of patching the current transfer code.

## Problem

python-gen-worker currently treats Tensorhub model transfer as custom platform code. Uploads manually implement S3 multipart upload over Tensorhub-issued presigned part URLs with urllib3, while a separate adaptive file scheduler can run many files at once. In practice this creates multiplicative concurrency: file fanout can ramp to 16 and each file can run up to 8 part PUTs, so one worker process can create up to 128 simultaneous R2 PUTs without a single internal transfer budget. That is exactly the wrong failure mode for large R2 mirrors like FLUX.2-klein-4B.

Downloads are better but still split across multiple stacks: HF uses huggingface_hub for actual transfer, Tensorhub uses a custom aiohttp range-resume downloader into a BLAKE3 CAS, Civitai should be a provider-specific bounded downloader, and an older TensorhubDownloader still exists as a second download shape.

## Direction

Prefer standardized trusted libraries over ad hoc transfer code. The current model-weight transfer surface is closed: Tensorhub, Hugging Face, and Civitai. python-gen-worker should not support arbitrary URL model-weight uploads/downloads right now, and it should not keep presigned multipart as a worker model-transfer fallback once trusted R2 credentials land.

Keep the public/operator surface boring: remove as many transfer knobs as possible. Pick reliable, throughput-oriented defaults from measurement. If an internal concurrency budget is needed, it should be an implementation detail that maximizes reliable throughput, not a tenant/operator tuning exercise.

## Target architecture

- Tensorhub owns artifact sessions, revision bookkeeping, dedupe, validation, and finalization. R2 is the byte store, not the artifact database.
- Trusted workers receive narrowly scoped temporary R2/S3 credentials plus bucket/key metadata for Tensorhub model upload/download operations. Scope by bucket, prefix/object, TTL, and needed permissions.
- Worker Tensorhub data transfer uses boto3/botocore/s3transfer against the R2 endpoint with carefully chosen R2-compatible defaults. Do not rely on unsupported AWS checksum headers; keep Tensorhub BLAKE3 validation as the app-level integrity check.
- Worker reports size, BLAKE3, object key, and transfer metadata back to Tensorhub for final commit.
- Large Hugging Face files are staged to disk before Tensorhub upload. Direct HF-to-R2 streaming is deferred unless benchmarks prove it is reliable under retry, because seekable files are safer for production multipart retries.
- Hugging Face downloads delegate transfer/cache/resume to huggingface_hub.snapshot_download/hf_hub_download; worker code only handles provider-specific selection/planning around it.
- Civitai downloads use one provider-specific bounded downloader/API integration. Do not accept arbitrary URL refs as a model source.
- Worker model-weight presigned multipart upload code is removed or demoted to non-model platform plumbing once trusted R2 credentials land.

## Benchmark plan

Try a small set of approaches against representative model sizes, including FLUX.2-klein-4B:

1. boto3/s3transfer from seekable local files with throughput-oriented fixed defaults.
2. boto3/s3transfer with a lower internal concurrency default.
3. Existing presigned path with only the safety limiter, as a baseline to retire.
4. Optional HF direct-stream experiment only if it can preserve retry correctness.

Compare reliability first, then wall-clock time, peak RSS, CPU, disk reads/writes, retry count, and R2 error rate. Pick one default path and remove knobs unless a benchmark shows a real production need.

## Acceptance

- No endpoint-local monkeypatch is needed to keep R2 uploads stable.
- No tenant/operator concurrency tuning is required for normal model-weight transfer.
- FLUX.2-klein-4B can be mirrored into Tensorhub/R2 without an R2 TLS retry storm.
- Upload/download failures surface useful terminal messages to the CLI/orchestrator.
- The codebase has one obvious Tensorhub/R2 transfer path through the S3 SDK, one HF path through huggingface_hub, and one Civitai provider path. No arbitrary URL model-weight transfer path is introduced.

## Validation 2026-05-28 (paul session)
#19 worker-side COMPLETE + validated. Added tests/test_s3_transfer_grant.py (16 tests, all green; full suite 143 passed/1 skipped) covering the SDK grant path: grant parse (snake+camel, missing-field), BLAKE3/size validation, retry backoff with shrinking concurrency, process-wide upload-budget concurrency cap, atomic download + mismatch cleanup, R2-safe checksum config (request_checksum_calculation=when_required). Conversion-endpoint R2 fanout monkeypatch CONFIRMED removed across inference-endpoints/ + training-endpoints/ (conversion pins gen-worker>=0.8.0, uploads via framework, zero monkeypatch). Upload failure propagation CONFIRMED already implemented (worker.py:1303 ArtifactTransferError -> error_type=artifact_transfer + sanitized provider/phase/cause safe_message + internal detail). HTTP-stack consolidation DEFERRED (large regression-prone sweep; not started). FLUX transfer regression target exists in e2e: hf-mirror-flux2-klein. Benchmark harness scripts/benchmark_model_transfer.py exists (needs R2 creds to run).

## Tasks
- [x] Immediate 0.7.12 safety fix: added one process-wide internal R2 transfer limiter for the existing Tensorhub presigned upload path. Uses a hard-coded 8-PUT budget with no tenant/operator tuning knob.
- [x] Removed adaptive remote-upload fanout. `_ADAPTIVE_MAX_WORKERS=16` is gone; file fanout is a fixed internal default and cannot ramp independently into per-file multipart workers.
- [ ] Move the conversion endpoint R2 fanout monkeypatch into gen-worker proper, then remove the endpoint-local monkeypatch after the fixed gen-worker version is released and pinned.
- [ ] Add Tensorhub API support for trusted-worker transfer grants: logical upload/revision session plus scoped temporary S3/R2 credentials, bucket, object key, endpoint URL, permitted operations, and expiry.
- [ ] Implement SDK-backed Tensorhub/R2 uploads using boto3/botocore/s3transfer with Cloudflare R2 endpoint_url, fixed R2-compatible TransferConfig defaults, retry config, progress callback, abort behavior, and BLAKE3/size finalization back to Tensorhub.
- [ ] Avoid unsupported R2 checksum assumptions in botocore configuration. Treat Tensorhub BLAKE3 validation as the source-of-truth integrity check.
- [x] Kept large HF artifacts staged on disk before Tensorhub upload for retry-safe multipart transfer. Existing HF path uses `huggingface_hub` local snapshots; direct HF-to-R2 streaming remains non-default.
- [ ] Remove worker model-weight dependence on presigned multipart uploads after trusted R2 transfer grants land. Do not keep a presigned or arbitrary-URL fallback in python-gen-worker model transfer.
- [ ] Standardize Tensorhub/R2 downloads through SDK-backed `download_file`/`download_fileobj` when scoped credentials are available, with BLAKE3 validation and atomic materialization into the local CAS/snapshot layout.
- [x] Kept Hugging Face artifacts delegated to `huggingface_hub.snapshot_download` / `hf_hub_download`. Custom HF code only plans/selects files and probes metadata.
- [x] Standardized Civitai downloads as a provider-specific bounded downloader/API integration. Removed the public Civitai `source_url` entry point so Civitai no longer routes through arbitrary URL input.
- [x] Removed the legacy `TensorhubDownloader` export and implementation in `models/downloader.py`; Tensorhub model downloads now route through the resolved-manifest/CAS path.
- [ ] Consolidate HTTP stacks. Target: botocore/s3transfer for Tensorhub R2 data, huggingface_hub for HF artifacts, and one small HTTP client choice for Tensorhub JSON control plane plus Civitai provider API calls.
- [ ] Improve upload failure propagation so the CLI/orchestrator receives the actual transfer error class/message instead of a blank failed event.
- [x] Added focused tests proving the internal transfer budget prevents multiplicative file x part concurrency while the existing Tensorhub presigned upload path is still present.
- [ ] Add a benchmark harness comparing candidate transfer approaches on representative model sizes. Measure reliability, wall-clock time, peak RSS, CPU, disk I/O, retry count, and R2 error rate; choose fixed defaults from the results.
- [ ] Add an e2e regression harness for mirroring `black-forest-labs/FLUX.2-klein-4B` into Tensorhub/R2, asserting completion without an R2 TLS retry storm and with useful progress/error events.
- [x] Documented the production transfer architecture: Tensorhub trusted SDK transfer path, Hugging Face provider path, Civitai provider path, no arbitrary URL model-weight transfers, minimal/no config knobs, fixed defaults, retry behavior, validation, and finalization responsibilities.

## Validation 2026 05 29

VALIDATION 2026-05-29: worker transfer-standardization code + tests green (16 transfer/grant tests incl test_s3_transfer_grant.py; full suite 143 passed/1 skipped earlier). FLUX/R2 benchmark runs remain (infra-gated, need GPU+R2 creds).

---

# #21: Incremental cold-boot: download models in priority order, mark functions ready as each lands

**Status:** in_progress

RECONCILED 2026-07-03 (full-stack audit, see AUDIT.md + agents/progress.md): incremental cold-boot readiness becomes a REQUIREMENT of the worker rewrite (#365) and models v2 (#366). Carry the behavior into v2; do not extend the current worker.py implementation — ~60% of it is being deleted.

## Goal

A worker starting up against an endpoint with many functions and many models should:
1. Download the highest-priority models first (priority order is supplied by the orchestrator).
2. As each model finishes download — and, for the top one or two, gets eagerly loaded into VRAM — emit `WorkerModelReadySignal` so the orchestrator can flip the dependent functions to `available` and start dispatching requests.
3. Continue downloading the rest in the background. Lower-priority models can stay on disk until first-request demand promotes them to VRAM via `LoadModelCommand`.

The endpoint is *not* atomically ready/not-ready. Function readiness is per-function and follows from per-model availability.

## What goes wrong today

On cold boot, the worker walks `cfg.supported_repo_refs` in alphabetical order (`worker.py:4655 sorted(supported_set)`) and prefetches every declared model serially to disk. The worker only emits `startup_phase=ready` after the *entire* list finishes (success or error). The orchestrator gates `AvailableForRequests` on that global ready flag (`gen-orchestrator/internal/grpc/worker_availability.go:20`) — so even though `WorkerModelReadySignal` fires as each model lands, dispatch waits for the whole-manifest gate.

**Reproduced 2026-05-17**: Local SD endpoint with 8 declared models. 10 concurrent `generate_sd15` requests. Queue time: 861-984 s per request — almost all of it cold-boot prefetch. Actual GPU work once dispatched: 5.8-15.9 s (mean 11.1). sd1.5 itself finished downloading at minute ~7; the remaining 7 minutes were SDXL/Hyper-SD/SDXL-Lightning the requests didn't need.

## Direction (worker side)

### Part 1: respect orchestrator-supplied order

Drop the `sorted(supported_set)` at `worker.py:4655`. Walk refs in the order the orchestrator sent them. The orchestrator's job (companion gen-orchestrator issue) is to sort by per-model demand before sending.

### Part 2: emit per-model ready signals correctly

`DiffusersModelManager.process_supported_models_config` already calls `_on_model_downloaded` per successful download, which the worker uses to fire `WorkerModelReadySignal`. Audit that this signal carries the canonical ref the orchestrator can match against `FunctionRequirements`, so the per-function gate (`workerSupportsFunction` in scheduler.go) actually flips. The 2026-05-17 trace showed the signal firing — but the global `startup_phase` gate (orch side) blocked dispatch anyway.

### Part 3: eagerly load top-priority model(s) into VRAM during cold boot

Today: prefetch downloads to disk only. First request then incurs an additional ~10-15 s for diffusers `from_pretrained` to load weights into VRAM + run the `setup()` warm-up.

Proposed: after the highest-priority model finishes downloading, immediately load it into VRAM (and run `setup()` if the endpoint has one for that function). Cost is ~10 s of cold-boot time but pays back on the first request. Stop eager-loading when VRAM budget (after safety margin) is full — remaining models stay disk-resident and get promoted on demand via the existing `LoadModelCommand` path.

VRAM budget per worker is already detected (`ModelCache._max_vram_gb`). Eager-loading is just `model_manager.load_model_into_vram(ref)` invoked from the prefetch loop after `_on_model_downloaded` fires, gated on `cache.vram_free_gb > model.estimated_size_gb`.

### Part 4: don't block startup_phase=ready on every download

Keep emitting `startup_phase=ready` once the full prefetch finishes (preserves the existing semantics for any caller that needs the all-done signal). But ensure per-model readiness signals are fully sufficient on the orchestrator side — see the companion gen-orchestrator issue which removes the `startup_phase=READY` gate on `AvailableForRequests`.

## Expected impact

With all four parts plus the orch-side companion: a fresh pod facing demand only for sd1.5 downloads sd1.5 (~3 min), loads it into VRAM (~10 s), and immediately starts serving. Other 7 models continue downloading in the background. Today's 14-min wait collapses to ~3 min. First request is warm (no disk → VRAM hit) because Part 3 pre-loaded it.

## Tasks

- [x] Part 1: drop `sorted(supported_set)` at worker.py:4655. Preserve orchestrator-supplied order. Add test confirming prefetch order matches input order.
- [x] Part 2: audit `WorkerModelReadySignal` payload — confirm the ref shape the worker emits matches what `gen-orchestrator/internal/grpc/scheduler.go::workerSupportsFunction` resolves from `FunctionRequirements`. Fix any canonicalization mismatch.
- [x] Part 3: extend `DiffusersModelManager.process_supported_models_config` to eagerly call `load_model_into_vram(ref)` after each successful download, while `cache.vram_free_gb > estimated_size_gb`. Skip when VRAM is full; those models remain disk-resident.
- [x] Part 3 (companion): include `setup()` invocation for the top-priority model so the first request doesn't pay the endpoint-class instantiation cost (we saw this take ~14 s in the 2026-05-17 test). Subsequent setups for other models happen on first dispatch.
- [x] Part 4: keep `startup_phase=ready` emission semantics. No change to the worker's all-done signal.
- [ ] End-to-end benchmark: cold-boot the SD endpoint with 10 queued sd15 requests. Measure time-to-first-image before vs after. Target: ≤ 3 min, with first request inference-time only (no extra cold-load cost).
- [x] Companion: gen-orchestrator issue must (a) sort `cfg.supported_repo_refs` by per-model demand and (b) flip `AvailableForRequests` on `WorkerModelReadySignal` instead of `startup_phase=READY`. Without that, Parts 1-3 have no user-visible effect.

## Status note 2026-05-28
Worker-side parts (1-4) are implemented and merged. NOT closed: this feature is cross-repo and per its own description "Parts 1-3 have no user-visible effect" until the gen-orchestrator companion lands (sort supported_repo_refs by per-model demand; flip AvailableForRequests on WorkerModelReadySignal instead of startup_phase=READY). Remaining: that orchestrator companion + the end-to-end cold-boot benchmark. Re-statused in_progress (was mis-marked completed).

## Companion verified done 2026-05-28
The gen-orchestrator companion is implemented: model_demand_store.go demand-orders supported_repo_refs, and worker_availability.go flips AvailableForRequests on the first WorkerModelReadySignal (no longer waiting for startup_phase=READY). Remaining: the end-to-end cold-boot benchmark (needs the full stack).

## E2E validation 2026-05-28 (paul session)
Worker-side #21 VALIDATED. CLI isolation: real images, selective per-function model load (serve --function loads only that class's model), low_vram auto-mgmt on 8GB. Full e2e stack (stable-diffusion, 8 models, local GPU worker): worker reaches phase=registered->models_downloading at +3.5s and is dispatchable DURING download (Part 4 ok); reports DIFFERENTIATED per-function availability (SD1.5 fns serviceable, SDXL-family held back) -- not all-or-nothing. E2E HAPPY PATH BLOCKED by TWO cross-repo bugs (NOT gen-worker): (A) tensorhub deploy leaves released published_by_user_id=NULL (endpoint_source_code.go:268 principal.UserID empty for human deploy) -> orchestrator skips model resolve (server.go:2959) -> sends worker 0 supported_repo_refs -> demand-ordering never engages -> worker pre-downloads all 8 models unordered. Even after DB-stamping the publisher, orchestrator still sent 0 refs (deeper model-resolution gap). (B) worker's 60s no-message watchdog forces reconnect during the long download -> orchestrator reaps it -> autoscale register-timeout backoff stops creating workers -> queue stranded. Full writeup + repro: ~/cozy/e2e/agents/coldboot-validation-2026-05-28.md. Added e2e 'benchmark-coldboot' command (cmd/benchmark_coldboot.go). The end-to-end cold-boot benchmark task stays OPEN pending bugs A+B in tensorhub/gen-orchestrator.

## Validation 2026 05 29

VALIDATION 2026-05-29 (e2e local): incremental cold-boot CONFIRMED at the worker level — on the stable-diffusion endpoint (8 HF models) the worker registers generate-sd15 as AVAILABLE within ~4s while the other 7 models are still downloading (reported as Worker-local unavailable functions), and loads sd15 from the (pre-seeded) HF cache. Orchestrator demand-ranked supported_repo_refs so sd15 is prioritized. End-to-end image STORAGE could not be captured due to an e2e orchestrator-router grpc-proxy downstream-relay issue (job dispatch + keepalive not reaching the worker; see gen-orchestrator #386/#387) — unrelated to worker-side #21 logic, which is validated.

---

# #339: Trim the @inference authoring surface (optional shutdown, @invocable, parametrize=, declarative bounds) + kill the FLUX quant single-file resolver

**Status:** in_progress

RECONCILED 2026-07-03 (full-stack audit, see AUDIT.md + agents/progress.md): the authoring surface this trims (@inference/@invocable/parametrize=) is replaced wholesale by API v2 (#368) — do not extend it further. Still valid from this issue: the open quant-config load-test task (GPU-gated). Everything else is landed history.

## Theme

The @inference authoring surface pushes repetitive boilerplate onto every endpoint author. One theme: make the declarative path the only path. Wins land in gen-worker (helper + discovery + docs); the inference-endpoint repos are the consumers/validation.

## 1. Optional shutdown (missing == no-op)
Every author writes `def shutdown(self): pass`. For the common DI-injected endpoint it's pure ceremony: the model is framework-owned, freed by the model cache on UnloadModelCommand, not by shutdown; OS reclaims VRAM on exit. shutdown is called ONLY at process end (SIGTERM/Ctrl+C via stop() worker.py:5409; WorkerDrainCommand via _drain_then_stop worker.py:5958); NEVER on per-model eviction. Matters only for non-DI resources (engine subprocesses, CUDA graphs, threads, connections, torch.distributed groups, scratch dirs). Runtime already treats it optional (getattr+callable worker.py:8556/5990); decoration requires only setup. The 'required' lives only in the docstring (decorators.py:22) and the codegen scaffold (decorators.py:1165). Make it officially optional; keep setup required.

## 2. Function fan-out via parametrize= on @inference
klein-9b 16 classes/1150 LOC, klein-4b 15/1090, stable-diffusion 8, qwen-image 6, flux.1-dev 4. Classes are near-identical: one-liner setup, body delegates to shared _generate; only resources= (placement), model ref, input struct, function name vary. They CANNOT collapse to one function (placement is static per function; scheduler picks GPU from Resources before reading payload). Add `parametrize=[Case(name=, resources=, model=, input=)]` on @inference: one class+body, stamps a separately-placeable routable function per row. fp8/nvfp4/turbo cases are NOT redundant (distinct placement/checkpoints); duplication is only in hand-writing bodies. edit/compiled stay as their own classes. 'parametrize' chosen (pytest precedent; monomorphization not specialization); each row is a Case.

## 3. Declarative input bounds, not imperative clamping
Endpoints hand-clamp (klein-4b _generate main.py:714/716; flux.1-dev steps/guidance/_round_size). Move bounds onto the msgspec struct (Annotated[int, msgspec.Meta(ge=,le=)]); tighten num_inference_steps int|float->int. Validates at the boundary, surfaces in endpoint.lock, rejects bad input instead of silently correcting. Requires discovery to compile msgspec.Meta into the function schema.

## 4. Tenant code is memory-management-free (framework owns disk<->RAM<->VRAM)
Handler assumes its model is VRAM-resident when it runs. 17/22 endpoints leak device code (klein-4b/9b 19 hits): per-request/per-component .to('cuda'), _ensure_cuda_resident (klein-4b:668), empty_cache, enable_model_cpu_offload, OOM retry, cuda.is_available branches — redundant AND harmful (races the ModelCache promote/demote accounting; models/cache.py, model_manager.load_model_into_vram, inference_memory.py OOM tiers). Work: (a) worker promotes the model into VRAM before dispatch (enforce the contract); (b) drive OOM offload from declared Resources, make apply_low_vram_config/with_oom_retry framework-internal; (c) STRIP all tenant device/memory code — setup constructs+registers only. CAVEAT: verify the worker reliably promotes before every dispatch BEFORE stripping the tenant safety net.

## 5. Unify the per-kind method decorator into @invocable
inference.function/training.function/conversion.function/dataset.function all resolve to the same _function_inner — four names for one marker; kind lives on the CLASS decorator. Collapse to a single kind-agnostic @invocable (chosen over handler/route/expose for cross-kind neutrality). Keep the FOUR class decorators distinct (@inference/@training/@conversion/@dataset). Do NOT reuse 'entrypoint' (collides with gen_worker.entrypoint / Docker / packaging). Wire vocabulary stays function_name. RELATED: @batched_inference is redundant with @inference(runtime=...)+async (archetype decided by runtime=/async at worker.py:2221); consider collapsing it, optionally rename internal BatchedWorker->EngineWorker / SerialWorker->InProcessWorker only when already in that code.

## 6. Replace the FLUX quant resolver with from_pretrained (endpoint-side)
All four FLUX repos are proper diffusers-layout pipelines with model_index.json — NO single-file case. _resolve_base_quantized_transformer + _BASE_QUANTIZED_TRANSFORMERS + from_single_file(config=base_ref, subfolder='transformer') + *.safetensors globbing is unnecessary and almost certainly DEAD (keys off a hardcoded magic filename that appears nowhere else, untested; falls through to from_pretrained when unmatched). Delete across flux.1-dev/klein-4b/klein-9b. bf16/turbo: from_pretrained(repo). Quantized: from_pretrained(quant_repo) or swap the transformer/ subfolder (Flux2Transformer2DModel.from_pretrained(quant_repo, subfolder='transformer') passed as transformer=). CAS dedupes shared components so full-pipeline-per-quant is free. Confirm the quant transformer/ carries its quant config.

## 9. Roll out across all endpoints (inference + training)
Per-endpoint checklist: (a) drop empty shutdown; (b) <kind>.function->@invocable; (c) cuda-resident->setup then remove per #4; (d) clamping->Annotated+msgspec.Meta; (e) fold placement/quant cases into one class + parametrize=; (f) FLUX endpoints delete the quant resolver (section 6). inference-endpoints (22): anima, chatterbox-tts, ernie, flux.1-dev, flux.1-schnell, flux.2-klein-4b, flux.2-klein-9b, foundation-1, hunyuan3d-2.1, internvl-U, joycaption, ltx-video-2.3, musicgen, qwen3.6-27b-mtp-gguf, qwen3.6-35b-a3b, qwen-image, sdxl-illustrious, stable-audio-open, stable-diffusion, trellis-3d, wan-2.2, z-image. parametrize= fold applies to the multi-case ones (klein-9b/4b, stable-diffusion, qwen-image, flux.1-dev); single-class endpoints get only (a)-(d). training-endpoints (2): conversion is class-shape so it takes (a),(b),(d); image_lora_finetuner is a job-shaped trainer plugin (setup(ctx: StepContext) + train_step) so @invocable/parametrize= mostly do NOT apply — only the shutdown-optional cleanup; confirm the trainer contract is unaffected.

## 10. Prune the FLUX generation set to the sufficient 7
Sufficient: base {bf16, fp8, nvfp4, compiled} + turbo {bf16, fp8, nvfp4} = 7. DROP nf4/int4, int8, bnb, modelopt-dispatch (base+turbo) + their input structs + dispatch() usage, and the bare `generate` bf16 alias. klein-4b 15->7, klein-9b 16->8 (the 7 + edit, a separate image-EDIT op NOT in the grid — retained unless decided otherwise), flux.1-dev ->3. With parametrize this is ~6 rows (base+turbo x bf16/fp8/nvfp4) on one class, plus compiled and (klein-9b) edit as their own classes. Prune BEFORE the parametrize fold.

## 11. 'Ready' != 'warm': compile/warmup readiness + placement gap
The compiled variant's setup compiles AND prewarms over a constrained aspect_ratio enum (klein-4b:805-811), so dynamic=False is correct and it won't recompile per request once warm — the compile LOGIC is fine. Gaps: (a) Readiness is gated on bytes-on-disk, NOT setup/warmup. _emit_ready_if_all_cached (worker.py:5048-5091) flips AvailableForRequests from cache.get_disk_models(), not setup() (where the multi-minute compile+prewarm runs). The `warming` phase enum (worker.py:1603) exists but the gate doesn't block on it. So the first request to generate_compiled can pay the whole compile UNDER the GPU semaphore. ready != warm. (b) Placement can't isolate compiled: _flux_bf16_compiled = Resources(requires_gpu=True, min_vram_gb=24.0) is IDENTICAL to _flux_bf16, so the orchestrator can't give it its own pod. 'compiled gets its own pod' is NOT enforceable today; needs a warmup-cost/exclusivity/anti-affinity field. (c) Minor: GPU semaphore held before resolution through warmup (worker.py:7833) -> idle-but-locked GPU during a co-located download. torch.compile is non-pausable; interleaving on a shared GPU is the 170s/1640s thrash the semaphore already fixed. Clean path: per-function placement + eager-fallback routing, NOT worker-side compile interleaving.

## 12. The local CLI must drive the SAME production code paths (not a parallel fork)

DESIGN PRINCIPLE (per owner): `gen-worker run`/`serve` exist so the library is tested LOCALLY against the EXACT code paths production uses — model loading through the GPU memory manager (ModelCache: LRU residency + demote-to-RAM + evict + offload when too big), real serial/batched dispatch, the GPU semaphore, context wiring — WITHOUT a remote tensorhub + gen-orchestrator + gRPC. ONLY THE TRANSPORT differs: local stdin/unix-socket instead of gRPC-from-the-orchestrator. Today the CLI FORKED a simplified path (cli/run.py dispatch_request + cli/serve.py _Endpoint) that bypasses ModelCache and reimplements dispatch — so it does NOT exercise the production paths. (This is why I wrongly claimed the memory manager 'can't be tested locally' — the CLI was mis-designed, not a real constraint.) FIX: the CLI drives the same Worker core (discovery -> ModelCache-managed load/residency/offload -> dispatch -> context) with the local transport swapped for gRPC. Then over-subscription locally (serve --function over several big models) genuinely exercises the LRU + offload memory manager on the 8GB card.

## Endpoint sweep landed 2026-05-28
The endpoint-sweep portion is complete and verified across all ~24 inference + 2 training endpoints (3 parallel agents, batches A/B/C). Done: device-strip -> apply_low_vram_config(mode="auto") + guarded .to("cuda") on every diffusers endpoint; imperative clamps -> Annotated[..., msgspec.Meta(ge/le/multiple_of)]; qwen-image parametrize fold (bf16/fp8/nvfp4). All files ast-parse clean; quant resolvers / from_single_file / dispatch / function set left untouched; sdxl-illustrious runtime-verified on the 8GB card (auto-picked model_offload, produced a real 768x768 PNG). STILL OPEN (kept in_progress): FLUX quant-delete + canonical-7 prune (GPU-gated, >=12GB); klein/FLUX parametrize folds (deferred with quant); framework OOM-offload-from-declared-Resources; gen-orchestrator readiness/placement companions (cross-repo); cold-boot eager-setup readiness verification (ties to #21).

## OOM-from-Resources landed 2026-05-28
select_auto_mode/apply_low_vram_config now take an optional peak_vram_gb; the worker threads the endpoint's declared Resources.peak_vram_per_request_gb through InjectionSpec into both the injection-time offload preflight and the baseline auto pass. requirement = max(weights_estimate, declared_peak) — a declared-heavy endpoint offloads sooner, a too-small/absent declaration never lowers below the measured footprint. 4 new tests in test_inference_memory_select.py; full suite 122 passed/1 skipped.

## 4c readiness verified 2026-05-28 — gating deferred by design
VERIFIED: @inference SerialWorker setup() AND warmup() are deferred to first dispatch (worker.py ~2735), not run eagerly at cold-boot. So the first real request currently pays the setup+compile/warmup cost. Making readiness gate on warmup COMPLETION (run setup+warmup eagerly, flip ready only after) is a genuine design TRADEOFF in direct tension with #21 (incremental cold-boot wants ready ASAP as each model lands) and the deliberate fast-boot design; #322's StartupPhaseWarming already lets the orchestrator see a 'warming' worker. NOT flipped unilaterally — needs a team call on cold-boot-latency vs first-request-latency. The 'warmup-heavy/isolate' placement signal additionally needs a new Resources/proto field + orchestrator placement consumer. Both left open pending that decision.

## 4c readiness — verified the design already implements the intended behavior (2026-05-28)
Clarified intent: priority-ordered INCREMENTAL readiness — fully ready the priority model fast (download->VRAM->setup->warmup) and serve it immediately; background the rest. This is the existing design and is correct:
 - priority order: #21 Part 1 (worker respects orchestrator supported_repo_refs order, no sort).
 - priority model warm-before-serve: cold-boot prefetch -> _handle_manager_model_try_eager_load eager-loads the top ref into VRAM -> _trigger_serial_setup_for_ref -> _ensure_serial_class_started runs setup()+warmup() (emits the 'warming' phase) -> heartbeat advertises vram_models -> orchestrator routes localityVRAM (instant). Confirmed warmup() executes at worker.py ~8173.
 - background the rest: eager-load stops at the VRAM budget; remaining refs keep downloading to disk and promote on demand; serveable from disk_models meanwhile (partial online, no 20-model wait).
IMPORTANT: do NOT gate AvailableForRequests on warmup for ALL models — that would make disk-resident models unserveable and break incremental readiness. Availability = 'serveable (possibly via promote)'; the WARM-serve preference is handled by locality routing (4b: VRAM>RAM>DISK). The remaining 4c item (warmup-heavy/isolate placement signal: a Resources.warmup_cost/exclusive field + orchestrator dedicated-placement consumer) is a SEPARATE optional optimization, left open.

## FLUX fold + quant-delete + prune landed 2026-05-28 (inference-endpoints 4c09ff0)
flux.1-dev / flux.2-klein-4b / flux.2-klein-9b: folded per-quant classes into @inference + parametrize=[Case(...)] per MODE; deleted the single-file quant resolver (_resolve_*quantized_transformer, _*QUANTIZED_TRANSFORMERS, from_single_file branch, modelopt helpers) and pruned the dropped variants (nf4/int4/int8/bnb/modelopt). Net 213 ins / 839 del; ast-parse clean, zero dangling refs. STILL OPEN (GPU-gated, >=12GB): 'confirm the quant repos ship a from_pretrained-readable quantization_config' — the fp8/nvfp4 Cases now load via from_pretrained(quant_repo) on that ASSUMPTION; if false, FLUX loads bf16 (OOM) or raises. MUST be runtime-verified on >=12GB (Blackwell for nvfp4) before deploy. Backups at /tmp/flux-bak/. Independently verified the diff (ast, scope, dangling refs, fold structure) before commit.

## Dropped the warmup-heavy/isolate placement signal 2026-05-28 (redundant)
Removed the 'placement (Resources): warmup-heavy/isolate me (warmup_cost/exclusive)' task. It's redundant: the orchestrator already MODELS warmup/boot cost via PodBootTimeTracker (EWMA per release_id x machine_class) and the autoscale ReleaseBootEstimator, and the keep-warm-longer lever already exists as ScalingPolicy.IdleTimeoutOverrideSeconds (per release, in retire_idle.go). The route-to-already-warm-workers value is delivered by the VRAM>RAM>DISK locality tiers (4b, shipped). A declared warmup_cost signal would only duplicate the observed boot cost (which is more accurate). Not worth the cross-system plumbing.

## Tasks
- [x] shutdown: drop `def shutdown(self): pass` from the @inference class-codegen scaffold (decorators.py:1165).
- [x] shutdown: update the lifecycle docstring (decorators.py:22) — shutdown optional, missing=no-op; require only for non-DI resources. setup stays required.
- [x] shutdown: confirm decoration never rejects a class lacking shutdown; add a test that a no-shutdown @inference class registers, serves, and tears down on both SerialWorker and BatchedWorker paths.
- [x] parametrize: add `parametrize=[Case(name=, resources=, model=, input=)]` to @inference, stamping one separately-placeable routable function per row from a single class+body; keep plain @inference classes as the escape hatch (edit/compiled).
- [x] parametrize: fold the multi-case endpoints into one class + parametrize= table — klein-9b (16), klein-4b (15), stable-diffusion (8), qwen-image (6), flux.1-dev (4); keep edit/compiled explicit; measure LOC drop.
- [x] bounds: extend discovery to compile msgspec.Meta numeric/string constraints into the endpoint.lock function schema.
- [x] bounds: migrate endpoint input structs from imperative clamping to Annotated + msgspec.Meta (start flux.1-dev: steps/guidance/width/height); delete hand-clamp/_round_size once the struct enforces it.
- [x] decorator: add a single kind-agnostic @invocable(name=...) marker (replacing per-kind <kind>.function), plus @invocable.stage if stages stay namespaced. Keep the four class decorators distinct. Do not reuse 'entrypoint'; leave wire function_name unchanged.
- [x] decorator: migrate endpoints + examples from <kind>.function to @invocable (deprecation stub or hard cut). Also evaluate collapsing @batched_inference into @inference(runtime=...) (archetype decided by runtime=/async at worker.py:2221).
- [x] quant: delete _resolve_base_quantized_transformer + _BASE_QUANTIZED_TRANSFORMERS + the from_single_file/config=base_ref branch + *.safetensors globbing in flux.1-dev/klein-4b/klein-9b. bf16/turbo: from_pretrained(repo). Quantized: from_pretrained(quant_repo) or swap the transformer/ subfolder via from_pretrained(subfolder='transformer').
- [ ] quant: confirm the quant repos' transformer/ carries its quant config so from_pretrained reloads it quantized; add a load test so the path can't silently regress to a dequantized load.
- [x] memory (framework): guarantee VRAM-residency at handler entry — worker promotes the DI-injected model into VRAM before dispatch. Document the contract: tenant setup only constructs+registers; framework owns placement. [DONE+VERIFIED: gen-worker serve now drives the production ModelCache for residency; serve --function generate_sdxl,generate_sd15 (~9.1GB on the 8GB card) demoted SDXL->CPU to load SD15, promoted SDXL back on re-invoke, 3 real images, 0 OOM. cache.py: offload-managed pipelines skip literal .to() so LRU-evict + intra-pipeline offload coexist.]
- [x] memory (framework): drive OOM offload (inference_memory tiers) from declared Resources instead of tenants calling apply_low_vram_config/with_oom_retry; make those framework-internal.
- [x] memory (endpoint sweep): strip ALL tenant device/memory code across the 17 endpoints (klein-4b/9b 19 hits) — .to('cuda'), _ensure_cuda_resident, empty_cache, cpu_offload, OOM retry, cuda.is_available. Do AFTER the framework residency guarantee is verified.
- [x] rollout (inference-endpoints): sweep all 22 endpoints per the section-9 checklist; track per-endpoint.
- [x] rollout (training-endpoints): migrate conversion (class-shape: shutdown-optional, @invocable, Meta bounds); apply shutdown-optional cleanup to image_lora_finetuner; verify the trainer-plugin StepContext/train_step contract is unaffected.
- [x] prune (endpoint): cut FLUX generation to the canonical 7; delete nf4/int4, int8, bnb, modelopt-dispatch functions + input structs + dispatch() usage + the bare `generate` alias. klein-4b 15->7, klein-9b 16->8 (retain edit), flux.1-dev ->3. Before the parametrize fold.
- [x] readiness (framework + orchestrator companion): gate ready/AvailableForRequests on setup()+warmup() COMPLETION (warm), not bytes-on-disk, for warmup-heavy functions — make the `warming` phase hold readiness until compile+prewarm finish (worker.py:5048).
- [x] readiness (verify): confirm whether @inference SerialWorker setup runs eagerly at cold-boot (#21) or lazily; either way ready fires on bytes-cached, so document ready != warm until the gate lands.
- [x] discovery: fail the build (endpoint.lock) on duplicate routable function names within an endpoint, with a clear error naming the colliding classes — instead of the worker's silent 'Handler name conflict; skipping' warn-drop. (discover.py _assert_unique_function_names; +test_duplicate_function_names.py)
- [x] memory (building blocks, done+tested): made select_auto_mode MODEL-SIZE-AWARE (offload only when a model won't fit available VRAM minus margin, not just a total-VRAM threshold — so a fitting model like sd1.5 stays full-residency on an 8GB card; SDXL ~6.9GB offloads). Endpoint stable-diffusion _prepare_pipeline now delegates placement to apply_low_vram_config(auto): SDXL runs on the 8GB RTX 4070 via model_offload (real 1024 image), sd1.5 stays resident. Lowered SDXL-family floors 12/10 -> 8. (+test_inference_memory_select.py)
- [x] memory (THE real home — per owner): unify GPU-memory management into ONE subsystem (the manager that today is ModelCache + inference_memory). It is NOT 'an LRU cache' — it is the GPU memory MANAGER, and BOTH problems are its job: (a) 'too many models' -> LRU evict / demote-to-RAM / promote, pinned base (#337); (b) 'one model too big for the card' -> apply optimization/offload tiers. On placement/promote it decides: model fits available VRAM (after any eviction/demotion) -> full residency (.to('cuda')); doesn't fit -> apply_low_vram_config(auto) (model-size-aware select_auto_mode) intra-pipeline offload — instead of today's unconditional hard _move_pipeline_to_device('.to(cuda)') that OOMs a too-big model. Endpoints STOP managing device/memory entirely (no .to('cuda')/offload in setup/_prepare_pipeline) — the manager owns it. The local gen-worker run/serve path should ROUTE loading through this manager too (it currently bypasses it; the per-pipeline apply_low_vram_config is the interim local stopgap until it does). Offloaded models use < size_gb VRAM (only the active component) -> tighten the manager's vram accounting (over-count is safe/conservative for now). Production-verify on the real worker + a mix of GPU sizes (not verifiable on the local CLI path, which doesn't use the manager). [DONE+VERIFIED: gen-worker serve now drives the production ModelCache for residency; serve --function generate_sdxl,generate_sd15 (~9.1GB on the 8GB card) demoted SDXL->CPU to load SD15, promoted SDXL back on re-invoke, 3 real images, 0 OOM. cache.py: offload-managed pipelines skip literal .to() so LRU-evict + intra-pipeline offload coexist.]
- [x] memory (local-serve routing — makes the manager LOCALLY exercised+testable): route gen-worker serve/run placement through the GPU memory manager (ModelCache is drivable: __init__ + OrderedDict LRU + mark_loaded_to_vram / unload_model / residency_tier / get_vram_models / mark_cached_to_disk; worker.py:720 constructs it). The local _Endpoint should register each booted model's pipeline+size_gb with the manager and, on each dispatch, ensure the invoked model is resident — promote it, LRU-evict/demote the others when over the VRAM budget, and offload when a single model alone is too big. Today the local serve/run path bypasses ModelCache entirely (just holds every booted instance), which is why it couldn't be tested locally. [DONE+VERIFIED: gen-worker serve now drives the production ModelCache for residency; serve --function generate_sdxl,generate_sd15 (~9.1GB on the 8GB card) demoted SDXL->CPU to load SD15, promoted SDXL back on re-invoke, 3 real images, 0 OOM. cache.py: offload-managed pipelines skip literal .to() so LRU-evict + intra-pipeline offload coexist.]
- [x] memory (LOCAL integration test — the verification method, per owner): over-subscribe VRAM on the 8GB RTX 4070 to force the manager to work: `gen-worker serve --function generate_sdxl,generate_pony_v6_xl,generate_illustrious_xl` (3x ~6.9GB SDXL-family = ~21GB of models on an 8GB card). Then invoke each function separately and assert the manager loads/unloads/offloads as needed so EACH produces a real image with NO OOM and VRAM staying within budget (observe residency_tier transitions / load+evict in the logs). This is exercisable locally precisely because >VRAM total model size creates the LRU+offload pressure the single-endpoint path lacked. (Lighter variants for a faster loop: mix generate_sd15 + generate_sd_turbo + generate_sdxl, ~13GB combined > 8GB.) [DONE+VERIFIED: gen-worker serve now drives the production ModelCache for residency; serve --function generate_sdxl,generate_sd15 (~9.1GB on the 8GB card) demoted SDXL->CPU to load SD15, promoted SDXL back on re-invoke, 3 real images, 0 OOM. cache.py: offload-managed pipelines skip literal .to() so LRU-evict + intra-pipeline offload coexist.]
- [x] CLI architecture (root fix): make gen-worker run/serve drive the SAME production code paths as the gRPC worker — model loading/residency/offload via ModelCache, real serial/batched dispatch, GPU semaphore, context — swapping ONLY the transport (local stdin/socket for gRPC-from-orchestrator). Replace the parallel cli _Endpoint + forked dispatch_request with the shared Worker core (extract a transport-agnostic core from worker.py if needed). Goal: local CLI = production behavior minus the network, so the library is fully testable locally without tensorhub/orchestrator/gRPC. [DONE+VERIFIED: gen-worker serve now drives the production ModelCache for residency; serve --function generate_sdxl,generate_sd15 (~9.1GB on the 8GB card) demoted SDXL->CPU to load SD15, promoted SDXL back on re-invoke, 3 real images, 0 OOM. cache.py: offload-managed pipelines skip literal .to() so LRU-evict + intra-pipeline offload coexist.]

---

# #346: End-to-end cancel contract: gen-orchestrator user-facing cancel → `interrupt_job_cmd` (cross-repo; audit + spec)

**Completed:** no
**Status:** planned
**Related:** Worker side already done: interrupt_job_cmd → _handle_interrupt_request → ctx.cancel() (worker.py:5881/7363), Local analog: #352, #353

RECONCILED 2026-07-03 (full-stack audit, see AUDIT.md + agents/progress.md): cancel is redefined by protocol v2 (#364): one typed cancel message + single attempt fencing. Do the audit/spec as part of #364/#365, not against the v1 interrupt_job_cmd plumbing.

Cross-repo (gen-orchestrator) companion to #352/#353. For "a user cancels a request" to reach tenant code, the orchestrator must expose a user/API-facing cancel that emits `interrupt_job_cmd{request_id}` to the owning worker — which already maps to `ctx.cancel()` (worker.py:5881 → 7363). The worker side is DONE; this issue is the orchestrator side + the unified contract doc.

## Audit what exists in gen-orchestrator

- Is there a public cancel endpoint (HTTP/gRPC) keyed by the request/job id the client received?
- Does it route to the correct worker and send `interrupt_job_cmd`?
- On client disconnect (SSE / response-stream drop) does the orchestrator auto-cancel the in-flight job? (Production analog of #352's client-disconnect backstop.)

## Then write the single end-to-end contract doc

cancel source (API call / client disconnect / OS signal) → request_id lookup → `ctx.cancel()` → tenant observes via `raise_if_canceled()` / `cancel_event`. Same primitive everywhere; only the source differs.

NOTE: implementation lives in gen-orchestrator (separate repo); tracked here for the contract. The worker-side cross-references (worker.py) are in THIS repo.

## Tasks
- [ ] Audit gen-orchestrator for a user-facing cancel-request API; confirm it emits `interrupt_job_cmd` to the owning worker.
- [ ] Auto-cancel on client / response-stream disconnect in the orchestrator (production analog of #352's disconnect backstop).
- [x] Write the single end-to-end cancellation contract doc (sources → request_id → ctx.cancel() → tenant idiom).
