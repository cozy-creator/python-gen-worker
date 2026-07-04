<!-- python-gen-worker issue tracker — COMPLETED issues (append-only archive) -->

> One `# #<id>: <name>` section per issue, separated by `---` lines; section anchor for
> tooling is a line starting with `# #`. IDs are stable for an issue's whole lifecycle and
> share ONE per-repo id space across progress.md / future.md / completed.md.
> CONCURRENT EDITS: only ever edit/append your own issue's section with targeted string
> replacement — never rewrite the whole file.

---

# #375: `yield ProducedFlavor` conversion contract is gone — from-scratch example (and training-endpoints quant endpoints) silently publish nothing

**Completed:** yes
**Status:** DONE (2026-07-04, Claude) — settled the ONE contract, option (a): producer endpoints (conversion/dataset/training) write files locally, call `cozy_convert.publish_flavors(ctx, flavors)` — one Tensorhub /commits commit per ProducedFlavor (path = file or dir) — and return a result struct; the executor stays contract-free. The dead yield-shape is deleted AND enforced: @endpoint rejects generator handlers for producer kinds at decoration time (TypeError pointing at publish_flavors), so the old shape fails loudly at import instead of silently streaming. examples/from-scratch rewritten to the contract (returns FromScratchResult w/ revision_ids) + discovery smoke test (kind=conversion, output_mode=single, result struct) + publish_flavors e2e against the fake /commits server. Docs: @endpoint docstring, endpoint-authoring.md (worked example), local-dev.md, produced.py/publish.py docstrings. training-endpoints quant/convert endpoints rewrite against this contract (their #37); live publish-then-consume e2e (tensorhub #542 scenario) deferred to the e2e suite.

## Metadata
- Category: bug / api-contract
- Status: planned
- Passes: false

## Tasks
- [x] Decide the ONE contract: (a) endpoints call `cozy_convert.publish_flavors(ctx, flavors)` explicitly and return a result struct, or (b) the executor collects `ProducedFlavor` yields from conversion-kind handlers and publishes them. Pick, document on @endpoint, delete the other shape.
- [x] Fix examples/from-scratch to the chosen contract and cover it (discovery+lock smoke at minimum; ideally the e2e publish-then-consume scenario of tensorhub #542).

## Acceptance
from-scratch runs against a live stack and its commit lands in the CAS; the dead shape no longer imports.

---

# #374: cozy_convert publish/clone robustness — retries, resume, ingest junk in published trees, silent empty publish

**Completed:** yes
**Status:** DONE (2026-07-04, Claude) — hub.py: bounded retries w/ backoff + Retry-After on commit POST, part PUTs, complete, and finalize polling (429/5xx/network; abort-DELETE on terminal failure unchanged). run_clone: persistent workdir keyed by sha256(provider|source|destination) under $COZY_CONVERT_WORKDIR (default <tmp>/cozy-convert) — retained on failure so a retry resumes the HF snapshot (hf local-dir metadata kept for exactly that reason), deleted on success; partial flavor trees are wiped per-retry. Junk filtering: files_from_tree AND _copy_non_weights skip `.cache/huggingface/**` so mirrors are byte-faithful (root dotfiles like a real `.gitignore` still publish). `if not result.published: raise` — publishing nothing can never read as success (empty `outputs` was already defaulted to bf16 by normalize_outputs; the guard now covers every path). Legacy `publish_repo_revision` (~450 LOC, legacy /publish route) deleted from _PublisherMixin along with its local-context stub, finalize-poll constants, and orphaned _helpers; checkpoint publishing is ONLY cozy_convert.publish_flavors (/commits). New tests: retry/junk in test_hub.py + run_clone lifecycle in test_publish.py (fake /commits server, real HTTP). Deliberate: publish_dataset_revision (datasets subsystem) untouched; civitai download resume is #373's scope.

## Metadata
- Category: bug / cozy_convert
- Status: planned
- Passes: false

## Tasks
- [x] Live probe: the published repo contains `.cache/huggingface/**` lock/metadata files and `.gitignore` — `ingest_huggingface` snapshots into the tree and `files_from_tree` (hub.py) uploads EVERYTHING. Filter HF-cache internals (or snapshot to a clean copy) so mirrors are byte-faithful to the source repo, not to huggingface_hub's cache layout.
- [x] HubClient has no retry/backoff/429 handling on part PUTs, commit POST, or complete (hub.py:113-156) — one transient S3 hiccup after a multi-GB download+convert fails the whole clone; run_clone `shutil.rmtree`s the workdir in `finally` (clone.py:552) so a retry is a full re-download. Add bounded retries + a keyed persistent workdir for resume.
- [x] `run_clone` with an empty `outputs` list publishes NOTHING and returns success (the specs loop simply doesn't run; the no-publish error only fires when failed_flavors is non-empty). Default to publish-as-is or make empty outputs an error.
- [x] Kill the second publish path: `gen_worker.publish_repo_revision` (request_context/__init__.py:995) still posts to the legacy `/repos/:tenant/:name/publish` route while cozy_convert uses `/commits` (#515 "the ONE publish path"). Convert remaining callers or delete.

## Acceptance
A mirrored repo's tree equals the source repo's tree; transient network errors don't restart multi-GB work; publishing nothing cannot read as success.

---

# #372: transport hardening — auth-failure gating, HelloAck deadline, redirect hop reset, TLS on redirects

**Completed:** yes
**Status:** DONE (2026-07-04, Claude) — fatal auth exit now requires 3 UNAUTHENTICATED strikes AND >=60s elapsed (transient hub-side rejections survivable; window resets on HelloAck); 30s deadline on the whole dial+Hello+HelloAck handshake so a stalling hub can never hang the worker (bounded attempts also make the run-loop disconnected-timeout check effective); redirect hop budget resets whenever the loop falls back to backoff, so `not_leader` routing stays alive across leadership-churn episodes; schemeless redirect targets inherit the TLS mode of the connection that issued the redirect (no plaintext downgrade) and a new `grpc_ca_bundle` setting (env `GRPC_CA_BUNDLE`) supplies a PEM root bundle for private CAs; `worker_id_mismatch`/`release_id_mismatch`/missing-identity FAILED_PRECONDITION exits immediately. 6 new tests in tests/test_worker_grpc_e2e.py.

## Metadata
- Category: bug / transport
- Status: done
- Passes: true

## Tasks
- [x] `_AUTH_FAILURE_EXIT_THRESHOLD=3` (transport.py:32, 260-267) fires inside ~5s of full-jitter backoff — the hub currently returns UNAUTHENTICATED for duplicate-stream and transient-pg conditions, so an asymmetric network blip kills the worker. Gate the fatal exit on true auth rejections AND elapsed time (e.g. 3 failures over >60s); tensorhub #539 fixes the status codes.
- [x] No timeout on the HelloAck wait (`await stream.read()`, transport.py:329): a hub that accepts the stream but stalls mid-registration hangs the worker FOREVER (keepalive is answered at the h2 layer; `worker_disconnected_timeout_s` is only checked between attempts). `asyncio.wait_for(..., 30)` → ConnectionError → normal backoff; also enforce the disconnected timeout inside `_connect_once`.
- [x] `redirect_hops` never resets after the 3-hop fallback (transport.py:270-275 vs 297-301 — reset only on successful HelloAck), permanently disabling `not_leader` routing for exactly the leadership-churn case it exists for. Reset when falling back with backoff.
- [x] TLS is decided by URL scheme with a bare `:443` heuristic (transport.py:41-50) and redirect targets are schemeless host:port → TLS deployments redirect into plaintext dials. Inherit TLS mode from the connection that issued the redirect; add a CA-bundle setting instead of system-roots-only `grpc.ssl_channel_credentials()`.
- [x] Exit fast on permanent FAILED_PRECONDITION (`worker_id_mismatch`/`release_id_mismatch`/missing identity) instead of burning the full disconnected-timeout (transport.py:279).

## Acceptance
Worker survives hub restarts/pg blips/leadership churn without process death; doomed configs die in seconds not minutes; TLS redirects work.

---

# #373: download failure path — fail fast on expired URLs, bounded verify retries, disk headroom, CAS progress, error mapping

**Completed:** yes
**Status:** DONE (2026-07-04, Claude) — typed `UrlExpiredError` (models/errors.py) raised on 4xx (except 408/429) from a presigned URL with ZERO retries; the ModelOp path emits `ModelEvent{FAILED, url_expired}` within seconds (hub re-mints fresh URLs per CONTRACT); `_is_terminal_download_error` now reads `exc.response.status_code` for requests.HTTPError (the attribute bug) — which also makes civitai 429/5xx retryable while 401/403 stay terminal INVALID; blanket 30-try/1h backoff decorator replaced with an explicit policy loop (verify size/blake3 failures capped at initial+2 retries; ENOSPC → `InsufficientDiskError` immediately; `backoff` dep dropped); pre-download disk-headroom check in `_ensure_blobs` (missing blob bytes + 1GiB headroom vs `shutil.disk_usage(...).free`) raises `InsufficientDiskError`; CAS downloads now stream `progress(bytes_done, bytes_total)` through `ensure_snapshot_async` (cached blobs pre-counted, per-chunk updates, clamped); missing orchestrator snapshot for a tensorhub ref maps to RETRYABLE instead of client-visible INVALID; job-path `UrlExpiredError` maps to RETRYABLE. 7 new tests in tests/test_download_failure_paths.py. Deferred: eviction-on-shortfall stays with #370 (headroom shortfall emits insufficient_disk immediately); no civitai-internal retry loop added — the executor's outer bounded retry covers 429/5xx now that classification is fixed.

## Metadata
- Category: bug / model-distribution
- Status: done
- Passes: true

## Tasks
- [x] cozy_cas.py:27-34 blanket-retries `requests.RequestException|ValueError|OSError` 30 tries/1h per file: an expired presigned URL (403, 15-min TTL hub-side) is retried against the same dead URL for up to an hour, then `_is_terminal_download_error` (executor.py:108-112) reads `exc.status_code` — which `requests.HTTPError` doesn't have (`exc.response.status_code`) — so it's classified transient and re-run 3 more outer times. Raise a typed UrlExpiredError on 4xx, emit `ModelEvent{FAILED, url_expired}` within seconds, fix the attribute bug.
- [x] Same decorator retries blake3/size mismatches (ValueError) up to 30 full re-downloads and ENOSPC (OSError) 30 times against a full disk. Cap verify retries at 2; ENOSPC → immediate `insufficient_disk`.
- [x] Pre-download disk-headroom check: sum missing SnapshotFile.size_bytes vs `shutil.disk_usage(cas_dir).free` with margin; on shortfall trigger the #370 eviction path or emit `insufficient_disk` immediately (`InsufficientDiskError` in capability.py:141-166 is defined but never raised).
- [x] The CAS path never passes `progress=` to `ensure_snapshot_async` (download.py:169-173) — `bytes_done/bytes_total` never flows for tensorhub models (the main production path). Wire it.
- [x] "tensorhub ref needs an orchestrator-resolved snapshot" (download.py:164-168) maps ValueError→INVALID→client-visible 400 for a hub-side residency bug. Map to RETRYABLE.
- [x] civitai: no internal retry, and auth/rate-limit failures are ValueError→terminal INVALID (download.py:575); classify 429/5xx as retryable.

## Acceptance
An expired URL, a corrupted blob, and a full disk each converge in seconds with the correct CONTRACT §9 error code; CAS downloads report progress.

---

# #367: split clone+conversion out as `cozy_convert`; move the tenant SDK to training-endpoints

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — new uv-workspace package `packages/cozy_convert` (wheel `cozy-convert`, depends on gen-worker, movable to its own repo). gen_worker has NO clone/ or conversion/; `import gen_worker` is torch- and conversion-free (tests/test_import_graph.py); ConversionContext stays in gen_worker minus `open_output_writer` (tenants construct `cozy_convert.StreamingWriter` directly); flashpack dep moved out of gen_worker extras. Shipped: ONE writer module (writer.py ~640 LOC replaces the 7 IO modules, streaming_primitives facade deleted); small classifier (classifier.py ~300 LOC + `HfApi.list_repo_tree` + `snapshot_download(allow_patterns=…)` replaces hf_classifier's 1,324 LOC / 14 refusal classes → ONE `RepoRefusal(reason=…)`); gguf_utils' 390-line binary parser deleted (gguf package's GGUFReader in gguf_tools.py); ONE finalize path (clone.py `run_clone` — per-flavor local tree → one Tensorhub `/commits` commit with `mode: replace|merge`; the enumerate-prior-latest-and-delete overwrite hack and the `_finalize_clone`/`_finalize_publish_as_is` twins are gone; hub.py is the commit client: blake3, presigned part PUTs, batchless complete, 202-poll finalize, dedup skips). dispatch.py replaced by publish.py `publish_flavors` (ProducedFlavor → one commit each). First ETL tests: 38 in packages/cozy_convert/tests — writer/classifier/hub units (hub against a threaded fake /commits server) + integration on real tiny HF models (transformers + diffusers ingest, fp16 cast, per-component flavor tree, diffusers→singlefile SDXL repackage, bnb nf4 quant; gguf direction skips unless llama.cpp toolchain on PATH). mypy clean for all of cozy_convert (exemption NOT carried; `disallow_untyped_defs` stays off for the moved-verbatim tenant-SDK modules). Deliberate: legacy `save_formats` payload field dropped (use `outputs`); `ConversionOutput`/`IngestResult` (Tensors-based) replaced by `CloneResult`; singlefile→diffusers repackage not integration-tested (needs canonical-config-sized weights); modelopt/hqq calibrated quants still refuse inline with structured `deferred_requirement` (they run in training-endpoints' own jobs). training-endpoints #34 import map — `gen_worker.conversion.{Source,Dataset,FileLayout,ProducedFlavor,CalibrationPolicy,resolve_calibration_action,StreamingWriter}` → same names in `cozy_convert`; `gen_worker.conversion.repackage.{diffusers_to_singlefile,singlefile_to_diffusers}` → `cozy_convert.repackage`; `gen_worker.conversion.flashpack.convert_safetensors_to_flashpack` → `cozy_convert.flashpack`; `gen_worker.conversion.core_types.ConversionOutput` → deleted, clone returns `cozy_convert.CloneResult`; `gen_worker.clone.from_huggingface/from_civitai` → `cozy_convert.clone.from_huggingface/from_civitai`; `gen_worker.conversion.ConversionContext` → `gen_worker.ConversionContext` (unchanged home); `dtype_vocab`/`base_model_families` → `cozy_convert.*`; old `@conversion`+`invocable` decorators → `@endpoint(kind="conversion")` (#368, their #35).

## Tasks
- [x] New package `cozy_convert` (~4,000 LOC target): hub-API ingest (HF + civitai), ONE streaming shard writer (collapse the 7 IO modules ~1,300 LOC to ~400; `streaming_primitives.py` is a pure re-export facade), dtype cast + quant via the libraries training-endpoints already calls directly (modelopt/bnb/torchao/hqq), repackage.py kept, ONE finalize path (see #360 clone tasks), `gguf` package instead of the hand-rolled binary parser (gguf_utils.py:390).
- [x] Replace hf_classifier.py (1,324 LOC, zero hf-hub imports, 14 refusal exception classes) with `HfApi.list_repo_files` + `snapshot_download(allow_patterns=...)` + a small classifier.
- [x] Move `Source`/`Dataset`/`ConversionContext` tenant SDK to training-endpoints (their #34), or into `cozy_convert` if tensorhub also needs it — either way, out of gen_worker.
- [x] gen_worker keeps only `ensure_local`'s civitai fetch (#366) and the `@endpoint(kind="conversion")` shim that hands a `ConversionContext`.
- [x] Give the ETL its first tests — clone/ and conversion/ are mypy-exempt (pyproject.toml:100-110) and have zero test coverage today. One integration test per conversion direction on a small real model.

## Acceptance
gen_worker has no `clone/`; `import gen_worker` never imports conversion machinery; training-endpoints/conversion imports from the new home.

---

# #368: API v2 — one decorator, slim context, pyproject config

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — Shipped the full API rewrite: ONE `@endpoint` decorator (function-first; class + optional setup(); kind=; async-gen = streaming; runtime="vllm"/"llama-server" boots a real health-checked server subprocess via new `runtimes/server.py`); single-positional-ref bindings HF/Hub/Civitai/ModelScope with `variants={name: (binding, Resources)}` replacing Case/parametrize + dispatch + flavor()/dtype(); `Resources(gpu, vram_gb, compute_capability, libraries)` (vram implies gpu); RequestContext slimmed to exactly 15 members (cancelled + raise_if_cancelled, typed save_image/audio/video, ctx.generator(seed)) with producer surface on Conversion/Dataset/Training subclasses via real inheritance; worker-owned placement/offload (`models.memory.place_pipeline`) and the `gen_worker.apply_low_vram_config` re-export DELETED (hard cut — #343 must strip endpoint epilogues, they are pure deletions now); `BatchItemDelta` first-class stream struct; `[tool.gen_worker] main` in pyproject replaces endpoint.toml (toml_manifest deleted; tomli-w re-added as core dep — bake-time lock encode needs it); CLI run/serve/invoke/prefetch with describe folded into `run --list` and explicit `--attach`; legacy tensor_repos_* claims precheck deleted from request_context/_helpers.py. marco-polo migrated (all 3 fns), `gen-worker run --payload '{"text":"marco"}'` works offline zero-flag (package-name default function pick). README + endpoint-authoring rewritten; 9 stale docs pages deleted. Suite: 174 passed, 1 skipped; `import gen_worker` torch-free; hello-world ≤20 lines asserted by test.
  Deliberate notes for inference-endpoints #343 (read-only grep of all 22): parametrize/Case users (flux.1-dev, klein-4b/9b, qwen-image) port mechanically to variants=; `dispatch()` users (z-image, qwen-image, sdxl-illustrious) change ROUTING (payload-field model selection becomes one routable function per variant — client-visible, more than deletion); sdxl-illustrious's SharedBase frozen-stack sharing has NO authoring equivalent (Residency shared-component machinery survives in models/, but #343 must decide per-fine-tune variants vs a follow-up surface); anima ports as-is via the kept ModelScope binding; no endpoint uses @batched_inference. Cancel-spelling renames (is_canceled→cancelled etc.) and apply_low_vram_config/device-code strips are mechanical deletions.

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

# #366: models layer v2 — download/cache/memory/residency, ~2,500 LOC

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — new layer: models/download.py (ONE async `ensure_local` → tensorhub CAS / HF snapshot_download in to_thread with a ~90-line variant selector replacing hf_selection's 522 LOC / conversion-free civitai fetch / modelscope; one progress-reporter shape; #379 stall watchdog kept; provider index folded in), models/residency.py (Residency LRU VRAM/RAM/disk manager, public eviction API, free-VRAM-driven `make_room`, pin-while-executing, SharedComponentCache folded in with VRAM counted once, ModelEvent emission with measured vram_bytes), models/memory.py (moved inference_memory; ONE free-VRAM decider; deduped + CUDA-resident estimates), models/loading.py (get_torch_dtype/detect_diffusers_variant/quant synthesis/load_from_pretrained, ~180 LOC). Executor's ModelStore now composes these; setup() injection is typed (str/Path path or pipeline-class via from_pretrained per annotation). Deleted: pipeline/ (all 7 modules incl. PipelineLoader), models/{cache,shared_components,hf_selection,hf_downloader,ref_downloader,interface,downloader,download_progress}, inference_memory.py. cli serve drives the production Residency with no private reach-ins; cli run + trainer use the new download API. Deliberately kept: cozy_cas transfer leaf and s3_transfer grants (#19 owns transport consolidation); conversion/ingest.py untouched (its civitai ETL is #367's move). e2e: ModelOp DOWNLOAD/LOAD/UNLOAD round-trip over a real gRPC socket + local HTTP CAS blob added to tests/test_worker_grpc_e2e.py.

## Tasks
- [x] `refs.py`: kept as-is.
- [x] `download.py`: one `async def ensure_local(ref) -> Path` (tensorhub CAS w/ #358 poisoning fix, HF snapshot_download in to_thread + small selector, civitai fetch extracted from conversion). One progress reporter feeding ModelEvent DOWNLOADING.
- [x] `memory.py`: merged inference_memory accounting per #358 — free-VRAM ladder, CUDA-resident estimates, single eviction policy (residency `make_room`).
- [x] `residency.py`: LRU VRAM/CPU/disk tiers + shared components folded in, keyed once, public eviction API (no `_private` reach-ins, no tier-string smuggling).
- [x] No PipelineLoader: endpoints receive a typed local path (or a pipeline built from their annotation) and call `from_pretrained` themselves; pipeline/ deleted keeping detect_diffusers_variant + get_torch_dtype + quant-config synthesis in models/loading.py.
- [x] Multi-model VRAM residency capability: Residency covers multiple pipelines per worker — cross-pipeline LRU eviction via `make_room`, `executing()` pin-while-executing, `pin()` for shared bases (endpoint-side deletion is inference-endpoints #343).

## Acceptance
Met at the layer: `setup(self, model: <ann>)` receives exactly what the annotation says (str/Path/pipeline-class), verified by the e2e ModelOp LOAD test; the `_resolve_model_path` hacks become deletable in inference-endpoints #343.

---

# #358: VRAM/memory correctness — free-vs-total, double-counting, digest poisoning

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — remaining bullets landed with the #366 models-layer rewrite (same PR). Free-VRAM-only decider in models/memory.py; CUDA-resident + data_ptr-deduped estimates; measured allocator deltas everywhere; digest-poisoning fixed with a retry test; several bullets died with the #365 worker.py deletion (disposition per bullet below).

## Tasks
- [x] `select_auto_mode` free-vs-total. [DONE earlier (fix/p0-correctness); carried into models/memory.py.]
- [x] `OFF_HEADROOM` re-derived against free VRAM. [DONE earlier; carried into models/memory.py.]
- [x] estimate double-counting: `estimate_pipeline_size_gb` now dedupes shared storages by data_ptr; NEW `estimate_cuda_resident_gb` counts only CUDA-resident tensors for residency accounting; shared components live ONCE in Residency (single entry, refcounted) so VRAM is booked once. ModelCache/`_vram_used_gb` deleted with cache.py.
- [x] Fantasy constants: worker.py:9409 `size=5.0` and worker.py:8710 evict-half DIED with worker.py (#365); serve.py's mirrored 5.0GB floor deleted — residency uses measured `torch.cuda.memory_allocated` deltas at load (executor.ensure_setup, serve._ensure_resident), and eviction is driven by measured free VRAM so unknown sizes can't corrupt decisions.
- [x] Digest poisoning: failed snapshot builds are evicted from `_SNAP_ENTRIES` (models/cozy_snapshot.py, renamed from cozy_snapshot_v2) so the next request rebuilds; regression test tests/test_cozy_snapshot.py.
- [x] Blocking HF download inside async: ref_downloader.py deleted; models/download.py `ensure_local` runs every blocking provider call via `asyncio.to_thread`. loader.py rglob/stat/JSON: loader deleted; the surviving variant/quant probes run inside `load_from_pretrained` which the executor calls in `to_thread`.
- [x] Three low-VRAM deciders → ONE: `models.memory.select_auto_mode` (free VRAM). loader.py's `_apply_memory_optimizations` died with pipeline/; worker.py's inline preflight died with worker.py (#365).
- [x] Conversion dtype mislabels: inline_convert now stamps the PRODUCED dtype (torchao fp8_wo → `fp8:e4m3` even for e5m2 requests; bnb int4 → `nf4`/`fp4`) into attrs + target_dtype with a warning on mismatch. Unknown dtype strings in the loader path now RAISE (`models.loading.get_torch_dtype`) instead of silently loading bf16.
- [x] ~190-line Flux2-Klein filename hack (worker.py:8748-8940) DIED with worker.py (#365); variant selection is the binding's flavor through `select_hf_files`.

## Acceptance
Met: the ladder reads free VRAM only (test_inference_memory_select.py green against models/memory.py); a failed download retries cleanly (test_cozy_snapshot.py, test_provider_routing retry test).

---

# #365: worker core rewrite — asyncio-first, ~1,600 LOC replacing worker.py

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — old 9,957-line worker.py deleted; replaced by transport.py (~360: one stream, JWT metadata, keepalive-only liveness, jittered backoff, not_leader redirects, bounded send queue w/ results-never-dropped) + registry.py (~250: the ONE walker producing EndpointSpec, now consumed by worker, discovery/discover.py AND cli/run.py) + executor.py (~700: single dispatch path sync-on-thread/async-on-loop, (a)sync generators → JobProgress, GPU semaphore, real deadline+cancel watchdog, attempt fencing, blob_ref results, JobMetrics, thin model seam w/ ModelOp/ModelEvent + retry/backoff downloads) + lifecycle.py (~290: Hello/in_flight reconcile, edge-triggered full-replace StateDelta, FnUnavailable gates, phases, working drain) + worker.py (~120 wiring); entrypoint kept. Upload stack untouched. api/micro_batch.py deleted (structurally inert; decorator kwargs remain declarative-only). The previously-nonexistent e2e now exists: `tests/test_worker_grpc_e2e.py` — fake `grpc.server` scheduler over a real socket driving connect/Hello/dispatch/progress/result/cancel/stream-kill/backoff-reconnect/kill-mid-job/reconcile-ships-result-exactly-once/drain, plus deadline, GPU-serialization, auth-exit, queue-policy tests, and marco-polo (with a new `marco_polo_stream` async-generator fn) served under the new core; `gen-worker run --payload` verified offline (CLI gained coroutine/async-gen dispatch). Deferred to #366: real VRAM residency/eviction (ModelStore is a thin ensure-local + residency map; LOAD maps to endpoint setup, UNLOAD tears down holding classes). Deferred to #368: dispatch-slot injection passes local paths (no PipelineLoader by design).

## Tasks
- [x] `transport.py` (~300): channel + auth interceptor, one bidi stream, bounded send queue with results-never-dropped, receive loop, reconnect with jittered backoff (actually reachable this time), redirect (`not_leader`) handling.
- [x] `registry.py` (~200): ONE decorator walker producing `EndpointSpec{instance, method, payload_type, output_mode ∈ {single, stream}, is_async, needs_gpu, finalizer}` — shared by worker, discovery, and CLI (today three separate walkers: discovery/discover.py, worker.py:2200+, cli/run.py:199-305 with a "mirrors discover.py" comment).
- [x] `executor.py` (~350): intake, GPU semaphore, real deadline + cancellation watchdog, sync-on-`to_thread` / async-on-loop, delta emission, result send. One path; "conversion" is a finalizer, not an archetype. Kills SerialWorker/BatchedWorker/Conversion triplication and both `_execute_serial_*` clones.
- [x] `lifecycle.py` (~150): Hello/StateDelta, startup phases, working drain.
- [x] `worker.py` (~150) wiring + keep `entrypoint.py` (~100) as-is.
- [x] Keep the upload stack byte-for-byte (presigned_upload.py, _upload_transport.py, s3_transfer.py — best code in the repo, already has real-socket tests).
- [x] Decide micro-batching's fate: current one is structurally inert (GPU semaphore acquired before aggregator submit, worker.py:9544→9593 — batch always 1 on 1-GPU). Either batch *inside* the semaphore or delete api/micro_batch.py (383 LOC) until an endpoint needs it. Recommend delete.
- [x] The e2e test that doesn't exist today: fake gRPC scheduler (`grpc.server` appears nowhere in tests/) driving real worker: connect → register → dispatch → stream deltas → result → interrupt → stream-kill → reconnect-with-backoff → drain. Style-match the existing real-socket/SIGINT tests.

## Acceptance
marco-polo example runs against the fake scheduler and against a real tensorhub (v2) end to end; thread count at steady state ≤ 6 + executor pool; old worker.py deleted.

---

# #364: protocol v2 — one stream, ~12 typed messages, attempt fencing (worker side)

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — `proto/worker_scheduler.proto` (package `cozy.scheduler`, ONE `Connect` bidi stream, 12 typed messages, `attempt` fencing, renumbered from field 1, single `PROTOCOL_VERSION_CURRENT=2`) written jointly with tensorhub #504 and committed byte-identical in both repos; full producer/consumer contract in `proto/CONTRACT.md`. pb regenerated via `task proto`. Deleted wire machinery: mega-oneof envelope, split results/events/heartbeat streams + handshakes, WorkerRegistration-as-heartbeat, `duplicate_request_id`/`ActiveAssignmentResume`/assignment epochs, the JSON `worker_event` fabric, RunMetricsV1 triple-emission (JobResult.metrics is the one vehicle), app-level watchdogs (gRPC HTTP/2 keepalive is the only liveness). Outputs >64KB go as `blob_ref` via the existing presigned-upload stack.

## Tasks
- [x] Write `proto/worker_scheduler_v2.proto` jointly with tensorhub #504. Worker→orch: `Hello{worker_id, release_id, ver, resources, in_flight[{request_id, attempt}]}`, `StateDelta` (on change only; heartbeat is gRPC keepalive), `JobAccepted{request_id, attempt}`, `JobResult{request_id, attempt, status, output|blob_ref, metrics}`, `JobProgress{request_id, seq, chunk}`, `ModelEvent{ref, state: DOWNLOADING|ON_DISK|IN_VRAM|EVICTED|FAILED, vram_bytes}` (collapses LoadModelResult/UnloadModelResult/WorkerModelReadySignal/download events — and finally reports the `size_bytes` the orchestrator's placement optimizer has been starving for), `FnUnavailable{fn, reason}`. Orch→worker: `HelloAck{config, keep[]}`, `RunJob`, `CancelJob{request_id, attempt}`, `ModelOp{ref, DOWNLOAD|LOAD|UNLOAD}`, `Drain{deadline_ms}`.
- [x] One bidi stream; identity per connection (JWT), not per message. Delete: split results/events streams + handshakes, dedicated heartbeat stream (exists only for no-auth dev mode), WorkerRegistration-as-heartbeat.
- [x] Fencing = single `attempt` int on dispatch/ack/result. Delete: `duplicate_request_id` reconcile handshake, `ActiveAssignmentResume` (never sent today; Go path unreachable), assignment epochs.
- [x] Outputs > ~64KB: presigned PUT to blob storage, result carries the ref — removes the head-of-line blocking that motivated split streams.
- [x] Delete the untyped `worker_event` JSON fabric. Today the worker emits `model.load.started`, `model.unload.*`, `model.url_refresh`, `models.disk_inventory`, `worker.fatal`, `worker.draining`, `worker.drain.status`, `worker.startup_timeout_unregistered`, legacy `model.ready` — Go drops all of them unread (only `worker.model.download.*` is parsed). Typed messages or logging pipeline, nothing else.
- [x] Liveness = gRPC HTTP/2 keepalive both sides, period. Delete the app-level 60s inbound-silence watchdog and the 10s full-snapshot heartbeat.
- [x] Pick ONE metrics vehicle: `JobResult.metrics` replaces RunMetricsV1 triple-emission (canonical_events + metrics.job + observation). Drop never-produced observation fields (ttft/itl/prefix-hit/kv-blocks/scaling_factors) until something emits them.

## Acceptance
Both repos compile against v2; a contract doc lists every message with producer/consumer; zero fields without both.

---

# #357: worker P0 reliability bugs (drain, result loss, reconnect storm, load events)

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — folded into #365: the new asyncio-first core makes each bug class structurally impossible rather than patched. Per-bullet coverage:
- Drain crash → `lifecycle.drain()`: stop admitting → finish in-flight → ship buffered results → close stream → exit 0; full round-trip driven by `tests/test_worker_grpc_e2e.py` (RunJob-after-Drain = RETRYABLE "worker draining" with no JobAccepted).
- Results lost during reconnect → `transport.SendQueue` keeps every unshipped `JobResult` in `pending_results` across reconnects until written to a live stream; e2e kills the stream mid-job and asserts the buffered result ships exactly once after `Hello.in_flight` reconcile.
- Drop-oldest discards results → queue policy per CONTRACT.md §1: results exempt from the bound and never dropped; overflow sheds oldest `JobProgress` only; everything else blocks the producer (unit-tested).
- Aux-stream death strands results → aux streams deleted; ONE bidi stream.
- Reconnect backoff unreachable → `Transport.run()` applies full-jitter backoff before every redial (`rand(0, min(cap, base*2^n))`, reset after 60s connected); e2e asserts delays were applied across forced stream-kills.
- Model-load events never emitted → typed `ModelEvent` is the single residency channel (DOWNLOADING/ON_DISK/IN_VRAM/EVICTED/FAILED); the JSON event fabric is gone.
- VRAM load blocks receive thread → `ModelOp`s run as asyncio tasks off the receive loop (`lifecycle.on_message` → `create_task`); the receive path never blocks on model work.
- Deadlines decorative → executor deadline watchdog: expiry sends `JobResult{FATAL,"deadline exceeded"}`, releases the GPU slot, and escalates to process recycle (`os._exit(70)`) if a sync handler thread ignores cancel for 30s; e2e covers expiry + slot reuse.
- Transient prefetch failure permanently disables → `ModelStore.ensure_local` retries with exponential backoff; only 4xx-class errors are terminal.
- `cancel_queued_only` cancels running work → fields deleted from the protocol; cancel is whole-request (`CancelJob{request_id, attempt}`).
- `_ensure_batched_loop` race → moot: ONE asyncio loop owns everything; no check-then-create anywhere.
- Per-request `CUDA_VISIBLE_DEVICES` mutation → executor passes `RunJob.compute.gpu_index` via `torch.cuda.set_device` inside the handler thread; no process-wide env mutation.

## Tasks
- [x] **Drain crash**: `_emit_worker_drain_result` constructs `pb.WorkerDrainResult(worker_id=...)` (worker.py:6216) — field removed from proto in #321 → `ValueError` swallowed by receive-loop except (5838-5842). Net: on `WorkerDrainCommand` the worker rejects new jobs but never runs `_drain_then_stop`, never reports, never exits; pods only die by external kill. Remove the field, add a test that drives a full drain round-trip. [DONE (fix/p0-correctness): removed the `worker_id=` kwarg; `test_worker_p0_reliability.py::test_emit_worker_drain_result_builds_valid_proto` guards it.]
- [x] **Results lost during reconnect**: `_send_message` drops any message while `_stop_event` is set (5593-5616) — a handler finishing in the reconnect window loses its `JobExecutionResult` forever (stuck request orchestrator-side). Results must be enqueued (bounded, non-droppable), never refused. [DONE: `_send_message` now gates on `_running` only (persistent queue survives the reconnect); `stop()` clears `_running` before setting `_stop_event`, so genuine shutdown still refuses. Note: the drop-oldest-discards-results and aux-stream tasks below remain open.]
- [x] **Drop-oldest overflow discards results**: `_send_message` overflow policy (3597-3614) treats job results and fire-and-forget events identically. Separate policies: events droppable, results never.
- [x] **Aux-stream death strands queued results**: when the results stream dies, `_aux_drain_loop` exits and messages already in `_results_outgoing_queue` have no consumer until a full primary reconnect that may never happen (4548-4571). On aux death, drain the aux queue back onto the primary.
- [x] **Reconnect backoff unreachable**: after `_stop_event.wait()` returns in `run()` (5565), the guard `not self._stop_event.is_set()` (5581) is false by definition → the #338 backoff never runs, lost stream = immediate reconnect storm. Restructure so backoff applies before redial; test with a fake scheduler that kills the stream N times.
- [x] **Model-load events never emitted**: `_handle_load_model_cmd` references undefined `started_at` (6834), NameError swallowed at 6846-6847 → `model.load.completed/failed` never sent. Define it (siblings at 6926/7034 do).
- [x] **VRAM load blocks the receive thread**: `_process_message` runs `_handle_load_model_cmd` inline (5958-5959) — up to 300s wait (6774) + `asyncio.run(load_model_into_vram)` on the only thread that processes dispatches/interrupts. Move to a worker thread like `_handle_download_model_cmd` (6901-6910).
- [x] **Deadlines are decorative**: `timeout_ms` lands in `ctx._deadline` and nothing reads it; no watchdog aborts a hung handler, which holds the GPU semaphore forever (worker bricked). Enforce deadline: mark request failed, release the slot, and report; escalate to process recycle if the thread won't die.
- [x] **Transient prefetch failure permanently disables functions**: `_startup_prefetch_loop` calls `_mark_ref_terminally_failed` on any exception (6608-6615). Retry with backoff; reserve terminal for 4xx-class errors.
- [x] **`cancel_queued_only` cancels running work**: `_handle_interrupt_request` logs `item_ids`/`cancel_queued_only` then unconditionally cancels (7451-7482). Honor the flag (or delete the fields in protocol v2 — see #364).
- [x] **`_ensure_batched_loop` race**: check-then-create with no lock (7515-7560) can leak a second loop/thread. Guard with a lock (or moot under #365 asyncio-first).
- [x] **Per-request `CUDA_VISIBLE_DEVICES` mutation** (2107-2147): process-wide, ineffective post-CUDA-init, racy for gpu_count>1. Pass explicit `device` down instead.

## Acceptance
Fake-scheduler integration test (see #365 task list) covers: drain round-trip, stream-kill × N with backoff, result survives reconnect, hung handler is reaped at deadline.

---

# #356: fix red master — 6 failing tests, CI ignored since Jun 8

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — CI now gates on tests + build: dropped the never-failing continue-on-error mypy step (88 errors of type debt; re-add gating once #365 lands mypy-clean; ruff stays advisory per #363). Tasks 1-2 were already fixed on fix/p0-correctness. Branch protection SKIPPED — repo-settings change, not doable from a worktree; needs an owner to enable "require CI green" on master.

## Tasks
- [x] Fix 4 failures in `tests/test_cli_describe.py`: fixture at :36-43 injects a static `HFRepo` into handler param `pipe`, rejected by the injection validation added in a53627b (`api/decorators.py:683`). Update fixtures to the new contract (static repos inject into `setup()`, not handler params). [DONE: dropped the static `pipe`/`m` params from the two describe-test handler signatures; the class-level `@inference(models=...)` still drives the reported bindings.]
- [x] Fix 2 failures in `tests/test_inference_memory_select.py` (:45, :66): re-derive expectations for the OFF_HEADROOM threshold added in 5347209 — but do this together with #358 (free-vs-total fix changes the answers again). [DONE with #358: keyword renamed to `available_vram_gb`; both 24GB-free cases now correctly resolve to "off" (15+GB free headroom > OFF_HEADROOM).]
- [x] Make CI red mean something: mypy currently `continue-on-error` in `.github/workflows/ci.yml`; either gate on it or drop the job. Add branch protection or a pre-push habit so red master can't sit for weeks. [DONE: mypy step dropped (88 errors, could never fail); tests + build gate. Branch protection = repo settings, skipped — needs an owner.]

## Acceptance
`uv run --extra dev pytest -q` fully green on master; CI failure blocks merge.

---

# #359: delete dead code, sweep 1 — the unreachable legacy function-shape stack (~2,500 LOC)

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — worker.py 12,543→10,133 lines: deleted the legacy function-shape stack (_inspect_request_spec, _execute_request/_execute_training_request, asset materialization, LoRA overlay stack, _binding_to_wire/_wire_ref, ref-compat heuristic), all listed knobs + zero-caller helpers + _worker_auth.py, dead proto-field reads, run_metrics dead fields; wire_protocol.py collapsed to the two constants; bare-Worker test fixtures extracted to one conftest factory. Acceptance grep clean (discover.py legacy readers went with #361).

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

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — deleted quant/, accel/, cache/, compile_helpers/, parallelism/, engines/ (+ the never-satisfied SDK-engine branch), conversion/dtype_utils.py, presets.py; pruned all listed __init__ exports; ar_tts trimmed to lookup(); clone/ dead dedup scaffolding + NotImplementedError stubs + twin _tensors_artifact removed. DEFERRED: pipeline/loader.py item → #366 (module deleted wholesale there); _finalize_clone/_finalize_publish_as_is single-path merge → #367 (rebuilds finalize with its first test coverage — merging 1,160 untested upload/publish lines blind was the riskier move).

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

**Completed:** yes
**Status:** DONE (2026-07-03, Claude) — discover.py legacy-marker paths + no-main fallback deleted (main_module now required); toml_manifest.py 592→226; decorators.py migration stubs/tombstones, rate_limit_per_invoker, prefer_distilled, and the whole .stage surface (+ cookbook-stages.md) deleted; dispatch() Literal validators WIRED (enforced at decoration time, 3 new tests); request_context: publish_repo_revision on the base class via real inheritance (setattr loop gone), _PublisherMixin dedupe, all listed dead members/helpers removed; cli repl + describe --json deleted (note: cozy-local internal/cli/describe.go:33 still passes --json — it falls back to serve --list-functions --json; file a one-line fix there); import gen_worker pulls no torch/requests; gen-worker --help ~130ms.

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

# #1: slim-request-context

**Completed:** yes

Trim `RequestContext` to a small, obvious, per-request surface. Today it ships **16 methods** that authors never call, polluting `ctx.` autocomplete in every IDE.

Design principle: in Go you pass `*ConversionContext` to a conversion handler and `*InferenceContext` to an inference handler — the typed context tells you what you can do. We should do the same in Python via subclasses, not by stuffing every possible RPC onto a single bag.

## What's wrong today

**Pure admin-plane methods that should not be on a per-request object at all** (no use sites in `examples/`, `inference-endpoints/`, or anywhere in the SDK):

- `publish_checkpoint` / `unpublish_checkpoint` (`request_context/__init__.py:995, 1021`)
- `publish_dataset` / `unpublish_dataset` (`:1060, 1069`)
- `publish_endpoint` / `unpublish_endpoint` (`:1078, 1091`) — an endpoint making *another endpoint* public is nonsense
- `publish_endpoint_release` / `unpublish_endpoint_release` (`:1104, 1125`)
- `publish_media` / `unpublish_media` (`:1142, 1151`)

These are visibility toggles on already-existing artifacts. They belong in `cozyctl` or the tensorhub UI calling the admin HTTP/RPC API directly — **not** in the gen-worker SDK in any form. We do not ship a `gen_worker.admin` shim or any re-exports; this work is a straight deletion.

**Producer-contract methods that ARE legitimate but only for specific endpoint kinds:**

- `publish_repo_revision` (`:1174`) — the whole job of a conversion endpoint
- `publish_dataset_revision` (`:1560`) — the whole job of a dataset-producing endpoint; `conversion/dispatch.py:821` already treats it as required-but-conditional
- `resolve_dataset` (`:1719`) — referenced in `dispatch.py:940` as the documented way training/conversion code resolves dataset refs
- `read_repo_metadata` / `write_repo_metadata` (`:1830, 1866`) — producer-side, mostly training
- `_download_blob_by_digest` (`:1803`) — leaks a private-looking name; if a producer needs it, expose typed `materialize_blob(digest) -> Path`

## Why a dev cares

When I write an `@inference_function`, my IDE should show me `save_*`, `progress`, `emit`, `is_canceled`, `device`, `compute` and nothing else. When I write an `@conversion_function`, I should additionally see `publish_repo_revision`. Today the inference dev sees 30+ methods, half of which are *meta-platform admin* — confusing, intimidating, and a footgun (an inference function should never call `publish_endpoint`).

## Proposed shape

```python
class RequestContext:                # base — inference + everything else
    def is_canceled(self) -> bool: ...
    def raise_if_canceled(self) -> None: ...   # see issue #2
    def progress(self, fraction: float, message: str = "") -> None: ...
    def emit(self, event: str, payload: dict) -> None: ...
    def save_bytes(self, ref: str, data: bytes) -> Asset: ...
    def save_file(self, ref: str, path: Path) -> Asset: ...
    @property
    def device(self) -> str: ...
    @property
    def compute(self) -> Compute: ...

class ConversionContext(RequestContext):
    def publish_repo_revision(self, ...) -> dict: ...
    def read_repo_metadata(self, *, destination_repo: str) -> dict: ...
    def write_repo_metadata(self, *, destination_repo: str, metadata: dict) -> dict: ...
    def materialize_blob(self, digest: str, dest: Path) -> None: ...

class DatasetContext(RequestContext):
    def publish_dataset_revision(self, ...) -> dict: ...
    def resolve_dataset(self, ref: str) -> str: ...

class TrainingContext(RequestContext):
    def save_checkpoint(self, ...) -> Tensors: ...  # already exists
    def read_repo_metadata(self, *, destination_repo: str) -> dict: ...
    def write_repo_metadata(self, *, destination_repo: str, metadata: dict) -> dict: ...
```

The `@conversion_function` / `@training_function` / `@dataset_function` decorators construct the appropriate subclass and the handler signature is `(ctx: ConversionContext, payload: Input) -> Output`. Authors get full typed autocomplete; admin-plane methods are gone from the SDK entirely.

## Tasks
- [x] Deleted 10 admin-plane methods from `RequestContext`: `publish_checkpoint`/`unpublish_checkpoint`/`publish_dataset`/`unpublish_dataset`/`publish_endpoint`/`unpublish_endpoint`/`publish_endpoint_release`/`unpublish_endpoint_release`/`publish_media`/`unpublish_media`. Also deleted the `_visibility_flip` helper that only these methods used.
- [x] Hard delete — no `gen_worker.admin` shim, no compatibility re-exports. Verified zero live callers in `cozy/tensorhub/` and `cozy/cozyctl/` (grep returned only stale references in completed-tasks JSON).
- [x] Added `ConversionContext(_PublisherMixin, RequestContext)` carrying `publish_repo_revision`, `read_repo_metadata`, `write_repo_metadata`, `materialize_blob`. Also folded the legacy `gen_worker/conversion/context.py:ConversionContext` (with `mktemp`/`checkpoint_dir`/`open_output_writer`/`copy_unconverted_components`/`cancelled`) into the new subclass to keep existing conversion endpoints working.
- [x] Added `DatasetContext(_PublisherMixin, RequestContext)` carrying `publish_dataset_revision`, `resolve_dataset`
- [x] Added `TrainingContext(RequestContext)` carrying `read_repo_metadata`, `write_repo_metadata`. `save_checkpoint` was already specific to the trainer path; kept where it was.
- [x] Renamed `_download_blob_by_digest` to `materialize_blob` (returns Path) on the `_PublisherMixin` shared by ConversionContext + DatasetContext
- [x] Dispatcher (`worker.py:_handle_job_request` and `conversion/dispatch.py:_run`) reads `training_fn.__training_spec__.kind` and constructs `DatasetContext` for dataset-generation kinds, `ConversionContext` for other training kinds, `RequestContext` for inference.
- [x] Updated `docs/endpoint-authoring.md` with a new "Kind-specific context subclasses" section showing the typed handler signature for each kind
- [x] Updated README's public-surface bullet list to include `ConversionContext`, `DatasetContext`, `TrainingContext`
- [x] Verified — inference `ctx.` no longer exposes any of the 10 admin methods; producer-contract methods are on their kind-specific subclasses; all 7 tests pass; full SDK integration test across all 7 issues clean.

---

# #2: cancellation-idiom-fix

**Completed:** yes

Make cancellation **one obvious line** for endpoint authors. Today the SDK ships three idioms and the docs disagree with the typed surface.

## What's wrong today

- `gen_worker.__init__.py:19, 53` exports `CanceledError` as a public typed error
- `docs/endpoint-authoring.md:170-171` tells authors to write `raise InterruptedError("canceled")` (built-in Python exception)
- Every shipped example reinvents the same 2 lines, with 4 different message strings:
  - `examples/marco-polo/src/marco_polo/main.py:17-18` — `raise InterruptedError("Request cancelled")`
  - `examples/medasr-transcribe/src/medasr_transcribe/main.py:36-37, 57-58` — `raise InterruptedError("canceled")` (twice)
  - `examples/openai-codex/src/openai_codex_worker/main.py:68-70` — `raise InterruptedError("Task cancelled")`
- The worker maps both `InterruptedError` and `CanceledError` to the same Cancel result (`worker.py:679` or thereabouts), so authors can't tell which is "correct"

## Why a dev cares

This is the **single most-repeated boilerplate** in every long-running endpoint. There should be exactly one canonical line, and the typed surface and the docs and the examples should all agree on it.

## Proposed shape

Add one method on `RequestContext`:

```python
def raise_if_canceled(self, message: str = "request canceled") -> None:
    """Raise `CanceledError(message)` if the request has been canceled. No-op otherwise."""
    if self.is_canceled():
        raise CanceledError(message)
```

Authors then write:

```python
for i in range(n_steps):
    ctx.raise_if_canceled()
    do_work(i)
```

`CanceledError` stays exported so authors can `except CanceledError` in cleanup code.

## Why this matches "Pythonic + Go-typed"

- The helper name follows `pathlib`/`stdlib` conventions (`raise_if_*`, `assert_*`).
- The typed `CanceledError` is the Python equivalent of Go's `context.Canceled` sentinel — callers can match the exact type, not parse a string.
- The `InterruptedError` pattern is *not* idiomatic — `InterruptedError` in stdlib specifically means "a signal interrupted a syscall" (EINTR), not "the user canceled this request." We're misusing a stdlib semantic.

## Tasks
- [x] Add `RequestContext.raise_if_canceled(message: str = 'request canceled') -> None` in `request_context/__init__.py`
- [x] Update `docs/endpoint-authoring.md:170-172` to show the helper as the canonical idiom; remove the InterruptedError pattern
- [x] Update `examples/marco-polo` to use `ctx.raise_if_canceled()` (and the example's README)
- [x] Update `examples/medasr-transcribe` to use `ctx.raise_if_canceled()` (two sites)
- [x] Update `examples/openai-codex` to use `ctx.raise_if_canceled()`
- [x] Remove any special-case for raw `InterruptedError` in the worker's exception mapping. Endpoints that still `raise InterruptedError(...)` now surface as unhandled errors — also converted 4 internal SDK raises (`_stream.py:153,183`, `presigned_upload.py:269,323`) to `CanceledError` since they relied on the worker's former special-case
- [x] Add a row to the errors table in `docs/endpoint-authoring.md:251` clarifying that `CanceledError` is what `raise_if_canceled` raises and what authors should `except`

---

# #3: asset-io-free-functions

**Completed:** yes

Finish the input side of `Asset` — but as **free functions in `gen_worker.io`**, not methods on the struct. Today every audio/image endpoint hand-rolls decode + normalization (~8 lines), and `ctx.save_image()` exists but is undocumented.

Design principle: `Asset` is a value struct (a typed pointer to a file). The decode-to-image / decode-to-audio operations are *codecs*, not Asset behavior. In Go you'd write `image.Decode(reader)` not `reader.Decode()` — same shape applies here. Free functions in a dedicated `gen_worker.io` module match both Go's package-organized free functions and Python's own `PIL.Image.open(path)` / `soundfile.read(path)` conventions.

## What's wrong today

`api/types.py:9-77` exposes only `Asset.open()`, `Asset.read_bytes()`, `Asset.exists()`, `Asset.__fspath__()`. Compare `examples/medasr-transcribe/src/medasr_transcribe/main.py:33-46`:

```python
if payload.audio.local_path is None:                    # boilerplate
    raise RuntimeError("audio.local_path missing")      # opaque error
speech, sample_rate = sf.read(                          # decode
    payload.audio.local_path,
    always_2d=False,
    dtype="float32",
)
if sample_rate != 16000:                                # normalization
    speech = librosa.resample(speech, sample_rate, 16000)
    sample_rate = 16000
```

Every audio endpoint repeats this. Every image endpoint repeats the PIL equivalent. Meanwhile, `ctx.save_image()` (`request_context/__init__.py:623-664`) is **undocumented** — zero hits in README + endpoint-authoring.md.

## Proposed shape

A new `gen_worker.io` module hosting codecs as free functions. `Asset` stays minimal.

```python
# gen_worker/io.py
from .api.types import Asset

def read_image(asset: Asset, mode: str = "RGB") -> "PIL.Image.Image":
    """Decode an Asset as a PIL image. Requires Pillow (`pip install gen-worker[images]`)."""
    ...

def read_audio(
    asset: Asset,
    target_sample_rate: int | None = None,
    mono: bool = True,
) -> tuple["np.ndarray", int]:
    """Decode an Asset as a numpy float32 array + sample rate.
    Requires soundfile + numpy. Resamples in-process if `target_sample_rate` is set."""
    ...

def write_image(
    ctx: RequestContext,
    ref: str,
    image: "PIL.Image.Image",
    *,
    format: str = "webp",
    quality: int = 90,
) -> Asset:
    """Encode and save as an output asset. Replaces undocumented ctx.save_image()."""
    ...
```

Usage in a tenant endpoint:

```python
from gen_worker import io

@inference_function
def transcribe(ctx, payload: Input) -> Output:
    speech, sr = io.read_audio(payload.audio, target_sample_rate=16000)
    return Output(text=model(speech))
```

The `local_path is None` check moves inside the free functions, raising typed `ValidationError` with the asset ref in the message.

## Hard cut

`Asset` becomes a data-only struct. **All** I/O moves to `gen_worker.io` as free functions in one PR:

- `Asset.read_bytes()` → deleted; use `gen_worker.io.read_bytes(asset)`
- `Asset.open()` → deleted; use `gen_worker.io.open(asset, mode='rb')`
- `Asset.exists()` → deleted; use `gen_worker.io.exists(asset)`
- `Asset.__fspath__()` → keep (this is the dunder that makes `asset` usable anywhere a path is accepted; deleting it would break `open(asset)`, `Path(asset)`, etc.)

No deprecation cycle. No transition shim. No `# deprecated` warning. Endpoints that call `asset.read_bytes()` break at type-check time; the tenant updates the call. The SDK is pre-1.0 — we are not stabilizing the wrong shape to spare a few one-line rewrites.

## Why a dev cares

- One canonical line per modality replaces 8 lines of boilerplate.
- Discovering codecs is `gen_worker.io.<tab>` in any editor — same shape as `os.path.<tab>` or Go's `image.<tab>`.
- `Asset` stays a small struct that you can pass around without dragging in PIL/numpy.
- Lazy imports (in the free functions) mean the core wheel doesn't pull in Pillow/soundfile.

## Tasks
- [x] Create `gen_worker/io.py` module; exported via `from gen_worker import io`
- [x] Add `gen_worker.io.read_image(asset, mode='RGB') -> PIL.Image.Image` with lazy `PIL` import
- [x] Add `gen_worker.io.read_audio(asset, target_sample_rate, mono) -> tuple[np.ndarray, int]` with lazy `soundfile` + `numpy` imports; lazy `scipy.signal.resample_poly` if available, else pure-numpy linear-interp fallback
- [x] Add `gen_worker.io.write_image(ctx, ref, image, *, format='webp', quality=90) -> Asset`
- [x] Delete `Asset.read_bytes()`, `Asset.open()`, `Asset.exists()` from `api/types.py`. Added `gen_worker.io.read_bytes(asset)`, `gen_worker.io.open(asset, mode='rb')`, `gen_worker.io.exists(asset)` as free-function replacements. Kept `Asset.__fspath__`
- [x] Raise typed `ValidationError` (with asset.ref in message) when `local_path is None` inside the free functions
- [x] Add optional extras to `pyproject.toml`: `gen-worker[images]` (Pillow>=10), `gen-worker[audio]` (soundfile>=0.12, numpy>=1.24)
- [x] Document the `gen_worker.io` module in `docs/endpoint-authoring.md` (new "Loading inputs and saving outputs" subsection)
- [x] Update `examples/medasr-transcribe` to `speech, sr = gw_io.read_audio(payload.audio, target_sample_rate=16000)`; dropped soundfile/soxr/numpy direct deps in the example's pyproject (now provided by `[audio]` extra)
- [~] Add a small image example showing `io.read_image()` + `io.write_image()` round-trip — DEFERRED. The docs section already shows the round-trip; a dedicated `examples/image-roundtrip/` skeleton without a real model would be misleading. Better hosted in a future image-shaped example once one exists.

---

# #4: typed-payload-errors

**Completed:** yes

Make every documented error type **actually raised**. Today `OutputTooLargeError` is exported, typed with `size_bytes` + `max_bytes` fields, and tenants are encouraged to catch it — but the worker raises bare `ValueError` / `RuntimeError` at the most common failure sites, so tenants have to regex-match strings.

This is the "Go-style typed errors" the user explicitly values: if I write `except OutputTooLargeError as e: shrink_output(e.max_bytes)`, that code path must actually fire.

## What's wrong today

- `request_context/_helpers.py:240-245` raises `OutputTooLargeError` correctly — but only from `save_bytes` / `save_file` / `save_checkpoint` / stream output paths.
- `worker.py:4491` (struct-payload too large): `raise ValueError(f"Output payload too large: {len(output_payload)} bytes (max {self.max_output_bytes})")` — **the most common too-large case**, and it's untyped.
- `worker.py:3711` (same path, different code branch): same bare `ValueError`.
- `worker.py:1581, 1698` (input file too large): `raise RuntimeError("input file too large")` — opaque, no size info.
- `api/errors.py` has no `InputTooLargeError` — there's no typed way to signal it.

## Why a dev cares

A tenant writing a batch inference function wants to write:

```python
try:
    return run_inference(payload)
except OutputTooLargeError as e:
    return run_inference(payload, max_tokens=e.max_bytes // 4)  # halve and retry
```

Today they have to write:

```python
except ValueError as e:
    if "Output payload too large" in str(e):  # regex on an error message — fragile, untyped
        ...
```

The stable, machine-readable error contract is the whole point of having a typed error hierarchy.

## Proposed shape

```python
# api/errors.py
class InputTooLargeError(ValidationError):
    def __init__(self, size_bytes: int, max_bytes: int, source: str = "input"):
        super().__init__(f"{source} too large: {size_bytes} bytes (max {max_bytes})")
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        self.source = source
```

Mirror `OutputTooLargeError`. Re-raise both at the worker boundary into structured payload responses so the orchestrator + caller see the typed shape, not a generic 500.

## Tasks
- [x] Add `InputTooLargeError(ValidationError)` to `api/errors.py` with `size_bytes`, `max_bytes`, `source` fields; export from `__init__.py`
- [x] Replace `worker.py:4491` `raise ValueError(...)` with `raise OutputTooLargeError(size_bytes=len(output_payload), max_bytes=self.max_output_bytes)`
- [~] (worker.py:3711) Not applicable — site was not a raise; the input-size check at that location already returns a structured error via `_send_request_result(error_type='validation', ...)`. No exception to convert.
- [x] Replace `worker.py:1581, 1698` `raise RuntimeError("input file too large")` with `raise InputTooLargeError(size_bytes=..., max_bytes=..., source='input file')`
- [~] Wire protocol carries size_bytes/max_bytes as structured fields — DEFERRED. `pb.JobExecutionResult` / `pb.BatchExecutionItemResult` today carry only `error_type` + `error_message` + `safe_message` + `retryable`. Requires coordinated pb + gen-orchestrator + tensorhub change; out of scope for this SDK-only issue. Python-side `except OutputTooLargeError as e: e.size_bytes` works in-process, which is the issue's stated value.
- [x] Add `InputTooLargeError` to the errors table in `docs/endpoint-authoring.md:251`
- [x] Add a unit test confirming `except OutputTooLargeError as e: assert e.size_bytes > e.max_bytes` works against a too-big struct return (`tests/test_typed_payload_errors.py`, 4/4 pass)

---

# #5: cut-realtime-socket

**Completed:** yes

Cut `RealtimeSocket` and `@realtime_function` from the SDK entirely. Half-shipped surface — the public exports lie, the implementation is four `NotImplementedError` stubs, the docs point at a path that does not exist, zero examples use it, zero inference-endpoints use it. Hard cut, no migration.

## What's wrong today

- `gen_worker/__init__.py:15` re-exports `RealtimeSocket` as part of the public surface
- `README.md:77` lists `RealtimeSocket` under "Context"
- The class lives in `src/gen_worker/_worker_support.py:176` — underscore-prefixed module = private
- The methods are 4 `NotImplementedError` stubs (`send_bytes`, `send_json`, `iter_bytes`, `close`) with no docstrings
- `docs/endpoint-authoring.md:202` claims realtime is "documented separately under `gen_worker.api.realtime`" — that file does not exist
- Zero hits across `examples/` and `inference-endpoints/`
- `api/decorators.py:274` exposes `@realtime_function`, also undocumented in README/authoring guide examples

A public surface that lies — methods named like real features but that raise `NotImplementedError` — destroys trust in every other method on the same surface. New authors look at `RealtimeSocket`, find the NotImplementedError stubs, and reasonably conclude the library is half-finished.

## What to delete

One PR, four locations:

1. `gen_worker/__init__.py` — remove `RealtimeSocket` and `realtime_function` from imports and `__all__`
2. `gen_worker/_worker_support.py` — delete `RealtimeSocket` stub class and any `_RealtimeSocketAdapter` plumbing
3. `gen_worker/worker.py` — delete the realtime dispatch path and `_handle_realtime_*` methods
4. `api/decorators.py` — delete `realtime_function` decorator entirely
5. `docs/endpoint-authoring.md` and `README.md` — drop all realtime references

No `# deprecated, will remove in 0.6.0` shim. No `realtime_function = inference_function` alias. The names are deleted; imports that referenced them break at import time and the tenant updates the call.

## If realtime is needed later

Design it from scratch when there's a concrete first customer with a working prototype. The current shape (separate decorator + private socket class with four stub methods) is one of several plausible designs; committing to it without a working example would lock in the wrong abstraction.

## Tasks
- [x] Remove `RealtimeSocket` and `realtime_function` from `gen_worker/__init__.py` imports and `__all__` (also `gen_worker/api/__init__.py` — was a second public export site)
- [x] Delete `RealtimeSocket` stub class, `_RealtimeSessionState`, and `_WebsocketSpec` from `_worker_support.py`
- [x] Delete realtime dispatch path and `_handle_realtime_*` methods in `worker.py` (also `_RealtimeSocketAdapter`, `_ws_specs`, `_realtime_sessions`, `_realtime_lock`, `_inspect_websocket_spec`)
- [x] Delete `realtime_function` decorator from `api/decorators.py`
- [x] Remove the `gen_worker.api.realtime` reference at `docs/endpoint-authoring.md:202` (Realtime sockets subsection deleted) and all `RealtimeSocket` / `realtime_function` mentions in `README.md`
- [x] Verify no residual references — grep returns zero hits in `src/ docs/ README.md examples/` (excluding auto-generated `pb/worker_scheduler_pb2*`); also cleaned `discovery/discover.py` AST allowlist

---

# #6: scaling-hints-discoverable-and-typed

**Completed:** yes

`ScalingHints` is a real and high-value feature (the orchestrator learns each endpoint's runtime cost) — but it's invisible to authors and the input surface has two redundant forms. Both problems are cheap to fix.

## What's wrong today

- `api/decorators.py:9-105` defines `ScalingHints` and exports it via `__init__.py:10, 47`
- Zero hits across `README.md`, `docs/endpoint-authoring.md`, `examples/`, `inference-endpoints/`
- Two input forms accepted by both decorators:
  - Structured: `@inference_function(scaling_hints=ScalingHints(vram_must_fit=..., vram_scales_with=[...]))`
  - Flat kwargs: `@inference_function(vram_must_fit=..., vram_scales_with=[...])`
- `_build_scaling_hints()` at `api/decorators.py:108-144` exists **only** to reject "both at once" — a problem we created by exposing two ways
- `ScalingHints.__init__` is hand-rolled (49 lines) with manual `to_dict()` (`api/decorators.py:90-103`) and `__repr__` (`:104-106`) — not idiomatic Python 2026

## Why a dev cares

1. **Discoverability.** A dev writing a stable diffusion endpoint should not have to grep the SDK to learn that the runtime adapts to their VRAM usage. The feature exists; authors can't find it.
2. **One way to do it.** Two redundant forms means `_build_scaling_hints` policing a self-inflicted problem, plus every doc/example has to pick one and risk being out of date.
3. **Typing.** `Dict[str, Any]` everywhere is the Python escape hatch. The user explicitly wants Go-style typed values. `ScalingHints` should be a frozen dataclass (or `msgspec.Struct`) with proper types, and the wire serialization should be `msgspec.json.encode(hints)` not a manual `to_dict()`.

## Proposed shape

```python
# api/decorators.py
import msgspec

class ScalingHints(msgspec.Struct, frozen=True, kw_only=True):
    vram_must_fit: int | None = None             # bytes
    vram_base: int | None = None                 # bytes
    vram_scales_with: tuple[str, ...] = ()
    vram_size_multiplier: float | None = None
    runtime_scales_with: tuple[str, ...] = ()
```

Decorator surface:

```python
@inference_function(scaling_hints=ScalingHints(
    vram_must_fit=24 * 1024**3,
    vram_scales_with=("payload.width", "payload.height"),
))
def generate(ctx: RequestContext, payload: Input) -> Output: ...
```

Flat kwargs are gone; `_build_scaling_hints` is gone. Wire serialization uses `msgspec`. Same shape as `ResourceRequirements` so the two feel consistent (see issue #7).

## Why this matches "Pythonic + Go-typed"

- `msgspec.Struct(frozen=True, kw_only=True)` is the idiomatic Python pattern for typed value objects in 2026 (faster + smaller than dataclasses, native gRPC/JSON encoding).
- `tuple[str, ...]` instead of `List[str]` — immutable, hashable, matches Python typing conventions.
- One way to construct, one way to serialize.

## Tasks
- [x] Rewrite `ScalingHints` as a frozen `msgspec.Struct(frozen=True, kw_only=True, omit_defaults=True)` with explicit field types. Field semantics preserved: `vram_must_fit: Literal['full_model', 'largest_component'] | None` (an anchor, not a byte count), `vram_base: int = 0` (bytes), `vram_size_multiplier: float = 0.0`, `vram_scales_with: tuple[str, ...]`, `runtime_scales_with: tuple[str, ...]`. Added `__post_init__` to validate the Literal at construction time (msgspec only validates on decode).
- [x] Remove the flat-kwarg form from `@inference_function` decorator signature; only accept `scaling_hints=ScalingHints(...)`. (`@realtime_function` already deleted by issue #5; `@training_function` flat-kwarg form also removed in `conversion/dispatch.py`.)
- [x] Delete `_build_scaling_hints` (no longer needed once there's one input form)
- [x] Add a "Scaling hints" section to `docs/endpoint-authoring.md` with two concrete examples: VRAM-scales-with-payload for an image model, runtime-scales-with-tokens for an LLM
- [x] Add `ScalingHints` to README's public-surface decorators bullet
- [~] Extend one existing example to demonstrate `scaling_hints=` — DEFERRED. The existing examples (marco-polo, medasr-transcribe, openai-codex) are intentionally minimal and don't have payload-shape signals that would benefit from scaling hints; adding a contrived one would be misleading. The two docs/endpoint-authoring.md code samples cover discoverability. A real SD or LLM example would be the right host once one exists in `examples/`.
- [~] Wire encoders updated — `discovery/discover.py` now uses `msgspec.to_builtins(hints)` instead of `hints.to_dict()` at both call sites; the `endpoint.lock` TOML shape stays identical (sparse JSON via `omit_defaults=True`).

---

# #7: idiomatic-typing-pass

**Completed:** yes

Sweep the public surface for Python typing escape hatches and Java-isms. Today the SDK is mostly clean but has pockets of `Dict[str, Any]`, hand-rolled `to_dict()`, and `str`-path parameters where `pathlib.Path` is now standard.

Goal: a developer writing against the SDK should be able to mash `Cmd+Click` (or `gd` in nvim) on any type and land in a useful definition with full field types — Go-style.

## What's wrong today (concrete sites)

- `api/decorators.py:163` — `ResourceRequirements._requirements: Dict[str, Any]` (bag of untyped fields). Hand-rolled `__init__`, `to_dict`, `__repr__`. Should be a `msgspec.Struct(frozen=True, kw_only=True)` with explicit `ram_gb`, `cpu_cores`, `gpu_memory_gb`, etc. fields.
- `api/decorators.py:218` — `def inference_function(...) -> Any` — the return is a decorator; type it as `Callable[[F], F]`.
- `api/decorators.py:91, 196` — manual `to_dict() -> Dict[str, Any]`. Replace with `msgspec.to_builtins(hints)` or remove (serialization should not be the value type's concern).
- `request_context/__init__.py:1830, 1866` — `read_repo_metadata` / `write_repo_metadata` accept and return `Dict[str, Any]`. If the metadata schema is open-ended, fine — but at minimum the return type should be a named `RepoMetadata` typed dict or struct so authors can `IDE-jump` to see the shape.
- Throughout — `str` parameters for filesystem paths. Modern Python (3.10+) accepts `pathlib.Path` everywhere; the SDK already requires 3.12+. Public signatures that take paths should accept `str | os.PathLike[str]` (or just `Path`) for autocomplete + `joinpath` ergonomics.
- `api/types.py:9-77` — `Asset` is a small value type. Worth checking it's already `msgspec.Struct` (it likely is — verify) and that `local_path` is typed `Path | None` not `str | None`.
- The `Tensors` and `LoraSpec` exports — verify these are typed structs not opaque dicts.

## Why a dev cares

When I write `@inference_function(resource_requirements=ResourceRequirements(...))` and my IDE shows `**kwargs: Any`, I have to read the source to know what fields exist. When the constructor is a typed `msgspec.Struct`, my IDE tells me `ram_gb: float | None`, `gpu_memory_gb: float | None`, etc. — same DX as a Go struct literal.

## What this is NOT

- Not a wholesale rewrite. It's a sweep for `Dict[str, Any]`, `-> Any`, and `to_dict()` on the public surface only. Internal worker plumbing keeps its untyped escape hatches where they pay rent.
- Not adding `pydantic`. The SDK already uses `msgspec` (faster, simpler, fewer dependencies). Stay on `msgspec`.
- Not changing wire formats. The msgpack/JSON on the wire stays the same; only the in-Python representation gets typed.

## Why this matches "Pythonic + Go-typed"

- `msgspec.Struct(frozen=True, kw_only=True)` is what 2026 Python uses for value objects. Frozen + kw_only matches Go struct semantics (immutable, fields by name).
- `tuple[str, ...]` over `List[str]` for immutable sequences.
- `pathlib.Path` over `str` for filesystem entries — `pathlib` is the stdlib `filepath` package.
- Typed exceptions (paired with issue #4) — same role as Go's `errors.Is`/`errors.As`.

## Tasks
- [x] Rewrite `ResourceRequirements` as `msgspec.Struct(frozen=True, kw_only=True, omit_defaults=True)` with explicit typed fields (`kind`, `accelerator: Literal['cuda','none'] | None`, `cuda_compute_min: float | None`, `compute_capability: dict[str, str] | None`, `requires_gpu: bool | None`, `min_vram_gb: float | None`, `required_libraries: tuple[str, ...]`). Normalization in `__post_init__` via `object.__setattr__` (frozen-friendly). Wire shape preserved.
- [x] Type `inference_function` decorator with `@overload`s: bare `@inference_function` returns `F`; kwarg form returns `Callable[[F], F]`. (`realtime_function` deleted by issue #5; `training_function` / `conversion_function` not in `decorators.py` — they live in `gen_worker.conversion.dispatch` and `gen_worker.trainer`. Out of scope for this typing pass.)
- [x] Replace `.to_dict()` call sites on `ResourceRequirements` and `ScalingHints` with `msgspec.to_builtins(...)`: `worker.py` (3 sites) and `discovery/discover.py` (4 sites, including ScalingHints from issue #6)
- [x] Audit `request_context/__init__.py` for `Dict[str, Any]` returns on public methods — switched 21 return-type annotations from `Dict[str, Any]` to lowercase `dict[str, Any]` for free-form shapes; left method bodies untouched (issue #1 will reorganize the class structure)
- [x] Accept `str | os.PathLike[str]` for public path parameters in `save_file`, `save_checkpoint`, `save_file_create`; bodies normalize via `os.fspath`
- [x] Confirmed `Asset`, `Tensors`, `LoraSpec`, `Compute` are all `msgspec.Struct` (Compute is also `frozen=True`). Read-only audit; field-tightening for Asset's `local_path` is out of scope (would require wire-contract coordination)
- [x] `ResourceRequirements.required_libraries` and `ScalingHints.vram_scales_with` / `runtime_scales_with` use `tuple[str, ...]` (immutable) — done as part of the Struct rewrites
- [~] pyright --strict against `api/` and `request_context/` — DEFERRED. pyright not installed in the venv; installing+configuring it is out of scope for this typing pass. Code currently has zero `# pyright: ignore` / `# type: ignore` lines, so a future pyright pass starts from a clean baseline.

---

# #8: direct-ref-flavor-modelref

**Completed:** yes

Let inference functions declare their model dependency as a direct (`ref`, `tag`, `flavor`) tuple on `ModelRef`. Drop the local-key indirection through `endpoint.toml [models]` for FIXED refs — it adds a layer of naming with no upside.

## Today's awkwardness

```python
# Function declares a local key string:
ModelRef(Src.FIXED, "flux2-klein-4b-turbo_int8")
```

```toml
# endpoint.toml has to map the key to (ref, flavor):
[models]
flux2-klein-4b-turbo_int8 = { ref = "black-forest-labs/flux.2-klein-4b-turbo", flavor = "int8" }
```

The local key carries no information — every key for the same repo is just `<repo>_<flavor>` wearing a different hat. Reading the function code tells you nothing about which repo it actually depends on; you have to cross-reference endpoint.toml. Worse, when a function's declared key drifts from the keyspace (this just happened: function declared `flux2-klein-4b-turbo_int8_bnb` but endpoint.toml only had `_int8`), discovery hard-fails with `function 'X' has FIXED model keys missing from endpoint.toml [models]`.

## The shape

```python
ModelRef(
    Src.FIXED,
    ref="black-forest-labs/flux.2-klein-4b-turbo",  # the repo
    tag="prod",                                       # defaults to "prod"
    flavor="nf4",                                     # the meaningful selector
)
```

For most endpoints, tag is always "prod" — so the meaningful selector that actually varies is `flavor`. Keeping tag as a separate arg (default `"prod"`) is cleaner than overloading the ref string with `:tag#flavor` syntax.

## What changes

**`api/injection.py`**:
- Add `tag: str = "prod"` and `flavor: str = ""` fields to `ModelRef`.
- `key: str` becomes optional for FIXED (when ref is set); still required for PAYLOAD (it's the payload field name).

**`discovery/discover.py`**:
- Drop the `"function uses ModelRef(...) with inline ref"` rejection at `:766-770`.
- For FIXED ModelRef with direct `ref`+`flavor`, skip the `endpoint.toml [models]` lookup — emit the (ref, tag, flavor) directly into the manifest.
- Drop the `FIXED model keys missing from endpoint.toml [models]` validation when ref is set inline.

**Manifest format (`endpoint.lock`)**:
- `required_models` becomes a list of objects: `[{"ref": "owner/repo", "tag": "prod", "flavor": "nf4"}, ...]` (was `["local-key"]`).
- Worker-side injection in `worker.py` resolves the (ref, tag, flavor) at injection time.

**Hard cut, no migration:**
- Drop the local-key-only form for FIXED. ModelRef(Src.FIXED, "key") raises ValueError; must pass `ref=`.
- Existing inference-endpoints break and rewrite. The mechanical rewrite is `ModelRef(Src.FIXED, "flux2-klein-4b-turbo_int8")` → `ModelRef(Src.FIXED, ref="black-forest-labs/flux.2-klein-4b-turbo", flavor="int8")`.
- `endpoint.toml [models]` table becomes unused for FIXED and can be deleted from endpoint.toml files.
- PAYLOAD-source keyspaces (`[models.<fn>]`) stay — they're still needed for payload-driven selection.

**Release**:
- Bump pyproject.toml `version` to `0.7.0` (minor bump — new feature, breaking change with hard cut).
- `uv build` + `uv publish` so endpoints can `uv add gen-worker@0.7.0`.

## Out of scope (deferred to a follow-up)

- **Flavor families** (`flavor_family="4bit"` matching any of nf4 / int4_awq / w4a8_awq / etc.). The (ref, flavor) base is the foundation; family expansion needs a small flavor-family registry and orchestrator-side expansion logic. File as a separate issue once this lands.

## Tasks
- [x] Added `tag: str = "prod"` and `flavor: str = ""` fields to `ModelRef` in `api/injection.py`. `key: str` became optional with default "" (still required for PAYLOAD/PAYLOAD_REF via `__post_init__` validation).
- [x] `ModelRef.__post_init__`: FIXED requires either `ref` (direct form) or `key` (legacy local-key form) — not both, not neither. flavor-without-ref also rejected. PAYLOAD/PAYLOAD_REF requires `key`; rejects `ref` (only valid for FIXED).
- [x] `discovery/discover.py` injection extraction now carries `ref`/`tag`/`flavor` through into the manifest's `injection_json` (added at the `mr_entry` construction site).
- [x] Dropped the `function uses ModelRef(...) with inline ref` rejection that previously blocked direct refs.
- [x] Added `_synthesize_model_key(ref, flavor)` helper producing deterministic keys like `owner__repo__flavor`. Used both for `required_models` extraction and for synthesizing a `manifest["models"]` entry when the function uses direct (ref, flavor). Keeps the wire format uniform — orchestrator still sees keys + a [models] table; the synthesis is invisible to it.
- [x] Updated the `FIXED model keys missing` validation to recognize synthesized entries. Direct (ref, flavor) injections now satisfy the validation without an endpoint.toml [models] entry.
- [~] Worker-side injection — no changes needed. The synthesis approach means the worker still sees a key → (ref, flavor) lookup via `manifest["models"]`, same shape as before. A future PR could collapse this to a direct (ref, flavor) resolve at injection time, but it's not blocking.
- [x] Bumped `pyproject.toml` from 0.6.2 → 0.6.3 (patch — additive feature; legacy local-key form still works, no breaking change).
- [~] `docs/endpoint-authoring.md` update — DEFERRED. The new form is exposed via `ModelRef` docstring (api/injection.py) and the unit tests. A bigger doc rewrite showing the new shape as primary belongs with the flavor-families follow-up.
- [x] Unit tests for the new ModelRef shape — `tests/test_modelref_direct.py` (11 tests): direct construction, tag default, legacy form, rejected misuse cases, key synthesis determinism.
- [x] Unit test for discovery — `test_discovery_synthesizes_models_entry` in the same file. Constructs a temp endpoint with direct ModelRef + no [models] entry; asserts the manifest has both the synthesized key in required_models AND the synthesized [models] entry with the right ref+flavor.
- [x] Built with `uv build` → `dist/gen_worker-0.6.3.tar.gz` + `dist/gen_worker-0.6.3-py3-none-any.whl`. Published to PyPI via `uv publish` (env-file token; live on https://pypi.org/project/gen-worker/0.6.3/).
- [~] Downstream endpoint sweep — bumped `gen-worker[torch]>=0.6.2` → `>=0.6.3` across all 13 inference-endpoints. Ran `uv sync --upgrade-package gen-worker` in each. Discovery audit: 12/13 pass (internvl-U fails on missing third-party `internvlu` package — unrelated to SDK). flux.2-klein-4b-turbo NOT migrated to direct (ref, flavor) shape — its existing legacy local-key form still works; the migration would be a separate audit-cleanup PR.

---

# #9: decorator-table-model-bindings

**Completed:** yes

Move every model binding for `@inference_function` into a single decorator kwarg `models={...}`. Collapse `ResourceRequirements` + `ScalingHints` into one `Resources` struct used **per function** (not shared with a low floor). Two binding forms — **fixed pick** and **dispatch pick** — with a chainable `.allow_override(*classes)` modifier that lets the invoker substitute their own checkpoint subject to an explicit pipeline-class allowlist declared by the tenant. Delete `endpoint.toml [models]` / `[models.<fn>]` entirely. Delete `Annotated[..., ModelRef(...)]`. Delete `ModelRef`, `ModelRefSource`/`Src`, `ScalingHints`, `ResourceRequirements` from the public API. **Also delete the runtime `require_vram` / `require_compute_capability` / `require_cuda_library` helpers** — these are redundant with the declarative `Resources` struct, which the worker checks at boot to advertise function availability and the orchestrator uses for placement. Hard cut, no compat shims, lockstep rewrite of all 13 inference endpoints + tensorhub + gen-orchestrator + bump to gen-worker 0.7.0.

## Today's split-brain

Model bindings live in **three** places that have to agree:

1. **Python `Annotated[Type, ModelRef(Src.FIXED, key|ref+flavor)]`** on each parameter.
2. **`endpoint.toml [models]`** — flat `key → (ref, flavor)` table.
3. **`endpoint.toml [models.<fn>]`** — per-function payload dispatch tables.

Plus function metadata spans **two** decorator kwargs (`resources=`, `scaling_hints=`) that nominally cover related concerns. Tenants have to decide whether `cuda_compute_min` and `vram_must_fit` go in the same struct or different ones.

Plus the static `Resources` declaration is **echoed at runtime** by `require_vram(N)` / `require_compute_capability(...)` / `require_cuda_library(...)` calls inside every function body — a workaround for endpoints that share one Resources struct with a low floor (`_flux_resources = ResourceRequirements(min_vram_gb=7.0)` for an endpoint whose worst variant needs 22 GB).

End state: bindings declared once, in Python, on the decorator. Toml carries zero model state. Resources is one struct, *per function*, with accurate `min_vram_gb` / `cuda_compute_min` / `required_libraries`. No runtime echo of static facts.

## Model

A binding is one of two **picks**:

- **Fixed pick**: function pins a specific `(repo, flavor?, tag?)`. Tag defaults to `"prod"`. Flavor is optional. The pick resolves to a concrete checkpoint_id at deploy time.
- **Dispatch pick**: function pins a *set* of `(repo, flavor?, tag?)` picks keyed by a single discriminator field in the payload (a `Literal[...]`-typed field). At invoke time the discriminator selects which pick to use.

Each pick supports an optional **`.allow_override(*classes)`** chainable method. When called with one or more pipeline class arguments, the invoker may substitute the default pick with an arbitrary ref of their choice — subject to a single constraint: the supplied ref's `pipeline_class` must be in the explicit list passed to `.allow_override(...)`. Bare zero-arg `.allow_override()` is a decoration-time error. The tenant declares what's acceptable explicitly; the framework does NOT auto-derive the constraint from the function's param annotation.

## New shape — one decorator kwarg at most, Resources per function

```python
from gen_worker import Repo, Resources, dispatch, inference_function

flux = Repo("black-forest-labs/flux.2-klein-4b-turbo")

# Per-function Resources — accurate min_vram_gb for each variant.
_flux_bf16  = Resources(requires_gpu=True, min_vram_gb=22.0)
_flux_fp8   = Resources(requires_gpu=True, min_vram_gb=12.0)
_flux_nvfp4 = Resources(requires_gpu=True, min_vram_gb=10.0, cuda_compute_min=10.0)
_flux_nf4   = Resources(requires_gpu=True, min_vram_gb=6.0)
_flux_int8  = Resources(requires_gpu=True, min_vram_gb=14.0)

# Dispatch variant carries both placement + cost shape in one struct.
_flux_dispatch = Resources(
    requires_gpu=True,
    min_vram_gb=14.0,                   # the largest pick the table can resolve to
    vram_must_fit="full_model",
    vram_base=500 * 1024 * 1024,
    vram_size_multiplier=1.10,
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("num_inference_steps", "num_images_per_prompt"),
)
```

### Fixed pick

```python
@inference_function(resources=_flux_bf16, models={"pipeline": flux.flavor("bf16")})
def generate_bf16(ctx, pipeline: Flux2KleinPipeline, payload: GenerateInput) -> GenerateOutput:
    return _generate(ctx, pipeline, payload)
# No require_vram(22 * _GB) — _flux_bf16.min_vram_gb=22.0 already covers it.
```

### Fixed pick + override allowed (explicit class allowlist)

```python
@inference_function(
    resources=_flux_bf16,
    models={"pipeline": flux.flavor("bf16").allow_override(Flux2KleinPipeline)},
)
def generate_bf16_overridable(ctx, pipeline: Flux2KleinPipeline, payload: GenerateInput) -> GenerateOutput: ...
```

For union annotations / multiple acceptable classes: `.allow_override(Flux2KleinPipeline, Flux2KleinKontextPipeline)`. Classes can be passed as class objects (preferred — autocomplete + import-time check) or as string FQNs (escape hatch). Bare `.allow_override()` raises `ValueError` at decoration time.

### Dispatch pick

```python
class BnbInput(msgspec.Struct):
    variant: Literal["nf4", "int8"]
    prompt: str
    num_inference_steps: int = 4
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1

@inference_function(
    resources=_flux_dispatch,
    models={
        "pipeline": dispatch(
            field="variant",
            table={
                "nf4":  flux.flavor("nf4"),
                "int8": flux.flavor("int8"),
            },
        ),
    },
)
def generate_bnb(ctx, pipeline: Flux2KleinPipeline, payload: BnbInput) -> GenerateOutput: ...
```

`Resources.vram_scales_with` / `runtime_scales_with` reference real payload field names. The SDK validates at decoration time that each field exists on the payload struct.

### Dispatch pick — cross-repo with override

```python
playground = Repo("playground-ai/playground-v2.5")
juggernaut = Repo("runwayml/juggernaut-xl")

_sdxl = Resources(requires_gpu=True, min_vram_gb=12.0)

class CheckpointInput(msgspec.Struct):
    variant: Literal["playground", "juggernaut"]
    prompt: str

@inference_function(
    resources=_sdxl,
    models={
        "pipeline": dispatch(
            field="variant",
            table={
                "playground": playground.flavor("bf16"),
                "juggernaut": juggernaut.flavor("bf16"),
            },
        ).allow_override(StableDiffusionXLPipeline),
    },
)
def generate(ctx, pipeline: StableDiffusionXLPipeline, payload: CheckpointInput) -> GenerateOutput: ...
```

### Multi-param injection — each binding override is independent

A function can declare multiple injected params in `models={...}`. Each entry is its own binding with its own `(repo, flavor, tag)` pick and its own optional `.allow_override(*classes)` modifier. The invoker's `_models` dict is keyed by the same param names, so each one can be overridden independently of the others.

```python
flux       = Repo("black-forest-labs/flux.2-klein-4b-turbo")
flux_lora  = Repo("black-forest-labs/flux-lora-collection")

@inference_function(
    resources=_flux,
    models={
        "pipeline":   flux.flavor("nf4"),                              # fixed, NOT overridable
        "adapter":    flux_lora.flavor("realism").allow_override(LoRA), # overridable
        "controlnet": dispatch(
            field="controlnet_kind",
            table={
                "depth":  Repo("...flux-controlnet-depth").flavor("bf16"),
                "canny":  Repo("...flux-controlnet-canny").flavor("bf16"),
            },
        ).allow_override(FluxControlNetModel),                          # dispatch + overridable
    },
)
def generate_with_adapter(
    ctx,
    pipeline: Flux2KleinPipeline,
    adapter: LoRA,
    controlnet: FluxControlNetModel,
    payload: AdapterInput,
) -> GenerateOutput: ...
```

Three bindings, three independent override policies:

- `pipeline` — fixed, no override. Invoker cannot substitute the pipeline checkpoint.
- `adapter` — fixed default + overridable. Invoker can send `_models.adapter = "acme/my-custom-lora"` to substitute, subject to the `LoRA` class check.
- `controlnet` — dispatch default + overridable. Invoker can either pick `"depth"` or `"canny"` via the discriminator field, OR send `_models.controlnet = "acme/my-controlnet"` to bypass the dispatch table, subject to the `FluxControlNetModel` class check.

Invocation example overriding two of the three:

```json
{
  "prompt": "A red bicycle",
  "controlnet_kind": "depth",
  "_models": {
    "adapter":    "acme/my-custom-lora:prod#realism",
    "controlnet": {"ref": "acme/my-controlnet", "flavor": "bf16"}
  }
}
```

The orchestrator validates each `_models` entry independently against its own binding:
- `adapter` → resolved against `LoRA` allowlist → substituted.
- `controlnet` → resolved against `FluxControlNetModel` allowlist → substituted (skips the dispatch table).
- `pipeline` is absent from `_models` → uses default `flux.flavor("nf4")`.

A `_models` entry naming a param that doesn't exist on the function → `unknown_override_param`. A `_models` entry naming a param whose binding has no `.allow_override(...)` declared → `model_override_not_allowed`.

### Multi-model two-stage example — SDXL base + refiner

A realistic multi-pipeline case: SDXL base generates a latent, refiner finishes denoising. Two pipeline objects injected into one function, both optionally overridable.

```python
sdxl_base    = Repo("stabilityai/stable-diffusion-xl-base-1.0")
sdxl_refiner = Repo("stabilityai/stable-diffusion-xl-refiner-1.0")

_sdxl_two_stage = Resources(
    requires_gpu=True,
    min_vram_gb=24.0,                # BOTH models resident
    vram_must_fit="full_model",
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("base_steps", "refiner_steps", "num_images_per_prompt"),
)

class SDXLTwoStageInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = ""
    base_steps: int = 30
    refiner_steps: int = 10
    high_noise_frac: float = 0.8
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1
    seed: int | None = None

@inference_function(
    resources=_sdxl_two_stage,
    models={
        "base":    sdxl_base.flavor("bf16").allow_override(StableDiffusionXLPipeline),
        "refiner": sdxl_refiner.flavor("bf16").allow_override(StableDiffusionXLImg2ImgPipeline),
    },
)
def generate_with_refiner(
    ctx: RequestContext,
    base: StableDiffusionXLPipeline,
    refiner: StableDiffusionXLImg2ImgPipeline,
    payload: SDXLTwoStageInput,
) -> GenerateOutput:
    gen = _seed_to_generator(payload.seed)
    latent = base(
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        num_inference_steps=payload.base_steps,
        width=payload.width, height=payload.height,
        num_images_per_prompt=payload.num_images_per_prompt,
        denoising_end=payload.high_noise_frac,
        output_type="latent", generator=gen,
    ).images
    images = refiner(
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        num_inference_steps=payload.refiner_steps,
        denoising_start=payload.high_noise_frac,
        image=latent, generator=gen,
    ).images
    return GenerateOutput(image=ctx.save_image(images[0]))
```

Four invocation scenarios:

```jsonc
// (a) defaults — both checkpoints from the bindings
{ "prompt": "A red bicycle", "base_steps": 30, "refiner_steps": 10 }

// (b) override just `base` — refiner stays default
{
  "prompt": "A red bicycle",
  "_models": { "base": "acme/sdxl-architecture-finetune:prod#bf16" }
}

// (c) override both
{
  "prompt": "A red bicycle",
  "_models": {
    "base":    "acme/sdxl-architecture-finetune:prod#bf16",
    "refiner": { "ref": "acme/sdxl-refiner-tuned", "flavor": "bf16" }
  }
}

// (d) class mismatch on `base` — rejected
{
  "prompt": "A red bicycle",
  "_models": { "base": "runwayml/stable-diffusion-v1-5" }
}
// 400 incompatible_pipeline_class — supplied pipeline_class StableDiffusionPipeline
// not in binding.pipeline_classes [StableDiffusionXLPipeline]
```

The corresponding manifest entries:

```toml
[[functions]]
name = "generate-with-refiner"

[functions.bindings.base]
kind = "fixed"
ref = "stabilityai/stable-diffusion-xl-base-1.0"
flavor = "bf16"
tag = "prod"
allow_override = true
pipeline_classes = ["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline"]

[functions.bindings.refiner]
kind = "fixed"
ref = "stabilityai/stable-diffusion-xl-refiner-1.0"
flavor = "bf16"
tag = "prod"
allow_override = true
pipeline_classes = ["diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline"]
```

Two rows in `function_param_bindings`, one per param. Orchestrator pre-resolves both default checkpoints at deploy time; worker downloads both at boot.

### Atomic substitution

If the invoker supplies overrides for multiple params and **any one** fails validation, the whole request is rejected before dispatch. No partial substitution. The error names which `_models[<param>]` failed.

Tenants' function bodies assume the whole binding set is valid; half-substituting could deliver a mismatched pair (overridden `base` + default `refiner`) the tenant didn't authorize. Atomic is the safe default.

### Cross-param compatibility is tenant-side

The framework only enforces **per-param class allowlist** and **ref-resolves-cleanly** checks. It does NOT verify that an override for `base` is *compatible* with the override or default for `refiner` (e.g., shared VAE, latent space, training-data alignment).

Tenants who need strict cross-param coupling have three options:

1. **Don't enable `.allow_override(...)`** — lock both params to their known-good defaults.
2. **Narrow the allowlist to a single class** — overrides must at least share architecture (the SDXL example does this).
3. **Add a runtime cross-check** inside the function body (`assert base.vae is refiner.vae`-style) and raise a typed error if the pair is incompatible at invoke time.

The framework gives the building blocks; tenants compose the policy.

## Invocation envelope — reserved `_models` field, two accepted shapes

**Structured (preferred for typed clients):**
```json
{
  "prompt": "A red bicycle",
  "_models": {
    "pipeline": {"ref": "acme/my-flux-finetune", "tag": "prod", "flavor": "bf16"}
  }
}
```

**String shorthand:**
```json
{
  "prompt": "A red bicycle",
  "_models": {"pipeline": "acme/my-flux-finetune:prod#bf16"}
}
```

Both forms normalize to `(ref, tag, flavor)`. Grammar: `<owner>/<repo>[:<tag>][#<flavor>]`. Tag defaults to `"prod"`; flavor is optional. The reserved field name `_models` is rejected at decoration time if a payload struct uses it.

When `_models.<param>` is present:
- Binding has no `.allow_override(...)` declared → `model_override_not_allowed` (HTTP 400).
- Override param doesn't match any binding → `unknown_override_param` (HTTP 400).
- Tensorhub returns 404 for the ref → `override_ref_not_found`.
- Ref exists but tag isn't published → `override_tag_not_found`.
- Ref+tag exist but flavor isn't in the checkpoint group → `override_flavor_not_found`.
- Supplied `(library, pipeline_class)` ∉ binding's `pipeline_classes` → `incompatible_pipeline_class`.

When `_models.<param>` is absent → default behavior (fixed value or dispatch on the discriminator).

## API surface summary — fluent, chainable, no kwargs on modifiers

```python
flux = Repo("owner/repo")                                            # bare repo = a binding with defaults
flux.flavor("nf4")                                                   # specific flavor
flux.tag("canary")                                                   # specific tag
flux.tag("canary").flavor("nf4")                                     # both (order doesn't matter)
flux.flavor("nf4").allow_override(Flux2KleinPipeline)                # override allowed, explicit class
flux.flavor("nf4").allow_override(Flux2KleinPipeline, FluxKontextPipeline)  # multiple acceptable classes
```

| Public symbol | Kind | Returns | Purpose |
|---|---|---|---|
| `Repo(ref)` | dataclass | `Repo` | Module-level repo handle; also a usable binding with defaults |
| `Repo.flavor(name)` | method | `Repo` | New Repo with flavor set |
| `Repo.tag(name)` | method | `Repo` | New Repo with tag set |
| `Repo.allow_override(*classes)` | method | `Repo` | New Repo with override enabled and class allowlist set. Zero-arg call is a decoration-time error. |
| `dispatch(field, table)` | free function | `Dispatch` | Payload-driven dispatch |
| `Dispatch.allow_override(*classes)` | method | `Dispatch` | New Dispatch with override enabled and class allowlist set |
| `Resources(...)` | dataclass | `Resources` | Hardware + cost shape (merged); declared per function |
| `inference_function(resources, models)` | decorator | decorator | Function declaration |

All modifier methods return new immutable instances; chain order is commutative.

### Deleted from public API (hard cut)

- `ModelRef`, `ModelRefSource`, `Src`
- `parse_injection`
- `ResourceRequirements` (renamed to `Resources` + scaling fields folded in)
- `ScalingHints` (fields folded into `Resources`)
- `OpenRefConstraints`, `OpenPayloadRefBinding`, `open_ref()` (subsumed by `.allow_override(*classes)`)
- `require_vram`, `require_compute_capability`, `require_cuda_library` from `gen_worker.capability` — redundant with `Resources` declarations. The framework's worker-boot self-advertise + orchestrator placement gate handle all of it.

Bare imports raise `ImportError` with a one-line migration pointer.

## `Resources` merged struct

```python
class Resources(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    # Static placement envelope (hard gates — used at worker boot and orchestrator placement)
    accelerator: Literal["cuda", "none"] | None = None
    requires_gpu: bool | None = None
    min_vram_gb: float | None = None
    cuda_compute_min: float | None = None
    required_libraries: tuple[str, ...] = ()

    # Dynamic cost shape (admission + scheduling — coefficients learned)
    vram_must_fit: Literal["full_model", "largest_component"] | None = None
    vram_base: int = 0
    vram_size_multiplier: float = 0.0
    vram_scales_with: tuple[str, ...] = ()
    runtime_scales_with: tuple[str, ...] = ()

    # Derived wire-shape field
    compute_capability: dict[str, str] | None = None
```

Decoration-time validation: every name in `vram_scales_with` and `runtime_scales_with` must reference a real field on the function's payload struct. Drift between scaling-hint fields and payload schema fails discovery with `unknown_payload_field`.

## Manifest shape (`endpoint.lock`)

```toml
[[functions]]
name = "generate-bf16"

[functions.resources]
requires_gpu = true
min_vram_gb = 22.0

[functions.bindings.pipeline]
kind = "fixed"
ref = "black-forest-labs/flux.2-klein-4b-turbo"
flavor = "bf16"
tag = "prod"
allow_override = false
pipeline_classes = ["diffusers.pipelines.flux2.pipeline_flux2_klein.Flux2KleinPipeline"]

[[functions]]
name = "generate-bnb"

[functions.resources]
requires_gpu = true
min_vram_gb = 14.0
vram_must_fit = "full_model"
vram_base = 524288000
vram_size_multiplier = 1.10
vram_scales_with = ["width", "height", "num_images_per_prompt"]
runtime_scales_with = ["num_inference_steps", "num_images_per_prompt"]

[functions.bindings.pipeline]
kind = "dispatch"
field = "variant"
allow_override = false
pipeline_classes = ["diffusers.pipelines.flux2.pipeline_flux2_klein.Flux2KleinPipeline"]

[functions.bindings.pipeline.table.nf4]
ref = "black-forest-labs/flux.2-klein-4b-turbo"
flavor = "nf4"
tag = "prod"

[functions.bindings.pipeline.table.int8]
ref = "black-forest-labs/flux.2-klein-4b-turbo"
flavor = "int8"
tag = "prod"
```

`dispatch(field="variant", table={...})` and explicit `dispatch("variant", {...})` produce identical wire shape — inference happens at discovery time.

## endpoint.toml after the cut (minimal)

```toml
schema_version = 1
main = "flux2_klein_4b_turbo.main"

[[build.profiles]]
name = "cuda12.8"
accelerator = "cuda"
cuda = "12.8"
python = "3.12"          # optional; defaults to 3.12
torch = "2.5.1+cu128"    # optional; defaults to managed version
```

For a CPU endpoint:

```toml
schema_version = 1
main = "joycaption.main"

[[build.profiles]]
name = "default"
```

What was dropped vs. the previous toml shape (all redundant with Python-side per-function `Resources` or derivable from the build profile):

- `[host.requirements]` — `cuda_min` derives from `build.profiles[0].cuda` via existing `deriveCudaMinFromBuildProfiles`; `compute_capabilities` derives from the union of `function.Resources.cuda_compute_min`.
- `[resources]` top-level — `vram_gb`/`ram_gb`/`cpu_cores`/`disk_gb` are dead code (nothing reads them today); `accelerator`/`accelerator_preference` move to per-function `Resources`.
- `[functions.<name>.resources]` — redundant with per-function `Resources`.
- `[[build.profiles]].host_requirements.{cuda_min, compute_capabilities}` — redundant with the profile's `cuda` field.
- `[[build.profiles]].build_args` — already a documented no-op in tensorhub (`buildArgsFromEndpointToml` returns empty by design).
- `[[build.profiles]].base_image` — only ever set when overriding the managed allowlist; rare; can be set when needed.
- `[models]` and `[models.<fn>]` — model bindings move to Python (the main subject of this issue).

Tensorhub rejects with 4xx + migration error if `[models]` is present. `[host.requirements]` / `[resources]` / `[functions.*.resources]` are deprecated (warning at publish in 0.7.0; hard-error in 0.7.6).

## Dockerfile cleanup (single base image is the source of truth)

Tensorhub only injects `BASE_IMAGE` as a build arg. The single resolved base image (e.g. `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) encodes Python + PyTorch + CUDA + OS together; the granular `(python, torch, cuda)` profile fields are the lookup *input*, not independent dimensions in the build.

The conventional `ARG CUDA_VERSION` / `ARG TORCH_VERSION` / `ARG PYTHON_VERSION` lines in today's Dockerfiles are vestigial — declared with defaults but never referenced in any `RUN` step. They get deleted in the 0.7.0 sweep. Tenants who need a specific version for a `RUN pip install` step introspect at build time (`python -c "import torch; print(torch.version.cuda)"`) — one source of truth, the base image.

### Base image allowlist removal

`[[build.profiles]].base_image` accepts any ref. The current hardcoded allowlist (`pytorch/pytorch:` / `python:` / `ubuntu:` / `ghcr.io/astral-sh/uv:`) goes away. Rationale: tenants already run arbitrary `pip install` / `apt install` / `RUN` steps in their Dockerfile, so gatekeeping the `FROM` line prevents nothing that tenants couldn't already do via the build steps. The allowlist was supply-chain theater.

After removal:
- Tenant sets `base_image = "nvcr.io/nvidia/pytorch:25.01-py3"` (NVIDIA's image) → builds.
- Tenant sets `base_image = "ghcr.io/my-org/custom-base:v1"` (their own pre-baked image) → builds (assuming docker has pull credentials).
- Tenant sets `base_image = "123456789.dkr.ecr.us-east-1.amazonaws.com/my-base:v1"` (private ECR) → builds (assuming docker has credentials).

The managed lookup (`managedPytorchBaseImages` / `managedPythonBaseImages`) stays as a **convenience** — when `base_image` is unset, tensorhub still resolves `(python, torch, cuda, accelerator, image_kind)` to a vetted pytorch/python image. That's the default path.

The registry-local alias optimization (`shouldAliasManagedBaseImage` → `EnsureRegistryBaseAlias`) continues to apply **only to managed bases**. Tenants who supply a custom `base_image` skip that path and pull directly from the original registry on every build (trade-off they accept by going off the managed path).

### Three build modes — managed, explicit, fully custom

Every profile is fundamentally a **routing declaration** (`accelerator` + optional `cuda` / `compute_capabilities` for placement). The image-source dimension is independent:

| Mode | Profile fields | Image comes from | BASE_IMAGE injected? |
|---|---|---|---|
| **Managed** | `accelerator` + `python` / `torch` / `cuda` | Tensorhub resolves via lookup table | Yes (resolved) |
| **Explicit** | `accelerator` + `base_image` | Tenant names the ref directly | Yes (literal) |
| **Custom Dockerfile** | `accelerator` only (no `base_image`, no `python`/`torch`/`cuda`) | Whatever's in the tenant's `FROM` line | No (not injected) |

For the custom-Dockerfile mode, the resolver is skipped entirely. No `BASE_IMAGE` build arg is set; the tenant's Dockerfile uses whatever `FROM` it wants. Tensorhub still passes `CUDA_VERSION` / `TORCH_VERSION` / `PYTHON_VERSION` if those *are* set on the profile (so tenants who want to pin a CUDA version for their `pip install` step but don't want to use the managed image can still do that).

Example fully-custom profile:

```toml
[[build.profiles]]
name = "custom"
accelerator = "cuda"
cuda = "12.8"                  # optional; only matters for routing + CUDA_VERSION pass-through
# No base_image, no python, no torch — tensorhub does NOT resolve or inject BASE_IMAGE
```

Tenant's Dockerfile:

```dockerfile
FROM my-private-registry.example.com/custom-base:v1
# tenant's own multi-stage build, no ARG BASE_IMAGE needed
```

The profile's `accelerator = "cuda"` still drives orchestrator placement onto a GPU host with `cuda >= 12.8`. The build step is opaque to tensorhub.

### Placement fields vs build hints — flat profile schema

A profile carries **two orthogonal sets** of information:

**Placement fields (always relevant — the orchestrator routes on these):**

| Field | Required? | Meaning |
|---|---|---|
| `accelerator` | always | `"cuda"` or `"none"` |
| `cuda_min` | if `accelerator="cuda"` | minimum host CUDA driver version (e.g. `"12.8"`) |
| `compute_capabilities` | if `accelerator="cuda"` | minimum GPU sm version list (e.g. `[">=8.0"]`) |
| `cpu_arch` | optional, defaults to `"amd64"` | `"amd64"` or `"arm64"` |
| `os` | optional, defaults to `"linux"` | OS family the image expects (`"linux"` / `"windows"` / `"macos"`). Defaults to `"linux"` since 100% of today's workloads are linux containers, but the field exists so multi-OS fleets can route correctly. |

These are **container-external** — they describe what hardware the image needs to run on, independent of what's inside the image. The orchestrator reads them to decide which workers can host this profile's image.

**Build hints (optional — only relevant for managed-mode image resolution):**

| Field | Used for |
|---|---|
| `python` | which Python version the managed base image bundles |
| `torch` | which PyTorch version the managed base image bundles |
| `cuda` | which CUDA version the managed base image is built against (separate from `cuda_min`) |
| `image_kind` | `"runtime"` vs `"devel"` flavor of the managed image |
| `base_image` | explicit override; bypasses the managed lookup entirely |

These are **container-internal** — they describe the contents of the image being built. Tensorhub only consumes them when picking a managed base image; they're irrelevant to placement.

The flat profile schema after the cleanup:

```toml
# Managed mode — placement + image-resolution hints
[[build.profiles]]
name = "cu128"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
# os = "linux"                # implicit; default. Set explicitly only for non-linux profiles.
# cpu_arch = "amd64"          # implicit; default.
python = "3.12"               # build hint — optional
torch = "2.11.x"              # build hint — optional
cuda = "12.8"                 # build hint — optional (typically same as cuda_min but can differ)

# Explicit-image mode — placement + literal base image
[[build.profiles]]
name = "nvcr"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
base_image = "nvcr.io/nvidia/pytorch:25.01-py3"

# Fully-custom mode — placement only
[[build.profiles]]
name = "custom"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
# no build hints — tensorhub doesn't inject BASE_IMAGE; tenant's Dockerfile FROM wins

# CPU profile (linux/amd64, all implicit)
[[build.profiles]]
name = "cpu"
accelerator = "none"

# ARM64 CPU profile
[[build.profiles]]
name = "cpu-arm"
accelerator = "none"
cpu_arch = "arm64"
```

What changes from today:

- `[build.profiles[*].host_requirements]` nested block is **flattened** — `accelerator` / `cuda_min` / `compute_capabilities` move to the profile's top level. The nested block is deprecated (warning in 0.7.0, error in 0.7.6).
- Top-level `[host.requirements]` is **deleted** (already in the plan) — per-profile placement is the right granularity since different profiles can target different hardware.
- `python` / `torch` are **optional** — endpoints that use the managed image still declare them; endpoints in explicit/custom modes omit them entirely.
- `cuda` (build) and `cuda_min` (placement) are **separated** — they typically match but can differ (e.g. build against CUDA 12.4 but require host 12.8+).
- `os` is a new optional placement field on the profile, defaulting to `"linux"`. Today all workloads are linux containers; the field exists so a multi-OS fleet can route correctly without a schema migration.
- Per-function `Resources.min_vram_gb` continues to drive VRAM placement decisions (per-function, not per-profile).

### When BASE_IMAGE is and isn't injected

| Profile shape | `BASE_IMAGE` injected | `CUDA_VERSION` / `TORCH_VERSION` / `PYTHON_VERSION` injected |
|---|---|---|
| `accelerator` + `python` + `torch` + (`cuda`) | yes (resolved) | yes |
| `accelerator` + `base_image` | yes (literal) | yes if `python`/`torch`/`cuda` also set, else not |
| `accelerator` only (custom) | **no** | yes if set, else not |

The Dockerfile decides which args to consume. Unused build args are silently ignored by docker; missing args used in the Dockerfile fall back to the `ARG NAME=default` declaration.

Minimum endpoint Dockerfile after the cleanup:

```dockerfile
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS cozy_base

# ... tenant install steps ...

ARG BUILD_NONCE=2026-04-23-e2e-43
RUN echo "build-nonce=${BUILD_NONCE}" \
    && mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

`BUILD_NONCE` / `DEPS_NONCE` / `AI_TOOLKIT_COMMIT` (endpoint-specific cache-bust knobs) stay because they're tenant-controlled cache invalidation, unrelated to versioning.

## Minimum viable endpoint — "just my Dockerfile" mode

This is the smallest endpoint that can ship. Three files, no opt-in for any tensorhub conveniences.

**`endpoint.toml`** (5 lines for CUDA, 4 for CPU):

```toml
schema_version = 1
main = "myendpoint.main"

[[build.profiles]]
name = "default"
accelerator = "cuda"
cuda_min = "12.8"
compute_capabilities = [">=8.0"]
```

Or for a CPU endpoint:

```toml
schema_version = 1
main = "myendpoint.main"

[[build.profiles]]
name = "default"
accelerator = "none"
```

What this declares:
- `main` — Python module path containing `@inference_function` decorators. Discovery reads it at build time.
- `[[build.profiles]]` — at least one routing target. `accelerator` plus `cuda_min` + `compute_capabilities` (if GPU). No `base_image`, no `python`/`torch`/`cuda` build hints, no `[models]`, no `[host.requirements]`, no `[resources]`, no `[functions.*.resources]`.

**`Dockerfile`** (tenant-controlled, any base, any build steps):

```dockerfile
FROM my-private-registry.example.com/whatever-base:v1

RUN apt-get update && apt-get install -y my-deps
COPY . /app
WORKDIR /app
RUN pip install -e .

RUN mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

Three contract points the tenant must satisfy:
1. `gen_worker` is importable in the runtime environment (transitive — `pip install -e .` against a pyproject that depends on `gen-worker[torch]>=0.7.0`).
2. Discovery is baked into the image at `/app/.tensorhub/endpoint.lock`.
3. Entrypoint runs `gen_worker.entrypoint`.

No `ARG BASE_IMAGE`, no `${TORCH_VERSION}`, no tensorhub-injected anything. Tenant's Dockerfile is fully theirs.

**`main.py`** (function code with `Resources` declared in Python):

```python
from gen_worker import Resources, inference_function
import msgspec

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    result: str

@inference_function(resources=Resources(requires_gpu=True, min_vram_gb=8.0))
def run(ctx, payload: Input) -> Output:
    return Output(result=f"got: {payload.prompt}")
```

No model injection (no `pipeline` param, no `models={...}`) — the binding system stays out of the way entirely.

### Mental model — three independent layers

| File | Declares |
|---|---|
| `endpoint.toml` | Where to find the entry module + **where to run** the resulting image (orchestrator placement) |
| `Dockerfile` | **How to build** the image (whatever the tenant wants) |
| `main.py` (decorators) | **What the functions do** + each function's runtime envelope (`Resources`, optional `models={...}`) |

The layers are independent. Tenants can be minimal or fully-managed at each layer:

| Layer | Minimum | Maximum |
|---|---|---|
| toml | profile with `accelerator` only (CPU) | full managed: profile fields + binding overrides + scaling hints |
| Dockerfile | arbitrary `FROM`, three contract points | pytorch-managed via `BASE_IMAGE` + version pass-throughs |
| Python | `@inference_function` + payload struct | `Repo` + `dispatch` + multi-param models + `.allow_override(...)` |

### What tensorhub does at build time

1. Validates `endpoint.toml`. Rejects with migration error if any deprecated block is present (`[models]`, `[host.requirements]`, `[resources]`, `[functions.*.resources]`, nested `[build.profiles.host_requirements]`).
2. Builds the Dockerfile. Injects `BASE_IMAGE` only if the profile declared `base_image` or `python`/`torch`/`cuda`. Otherwise the build args are an empty set.
3. Reads `/app/.tensorhub/endpoint.lock` from the built image. Persists per-function `Resources`, `bindings`, schemas to `function_param_bindings` + related tables.
4. Stores per-profile placement metadata (`accelerator`, `cuda_min`, `compute_capabilities`, `cpu_arch`, `os`) for orchestrator routing.

Build steps are opaque to tensorhub past the `docker build` call. The tenant controls the entire build.

## SDK changes (python-gen-worker)

**New module `gen_worker/api/binding.py`** (replaces `injection.py`):
- `Repo(ref)` frozen dataclass — also a usable binding with defaults (`tag="prod"`, no flavor, no override).
- `Repo.flavor(name)`, `Repo.tag(name)` — chainable; return new immutable Repo.
- `Repo.allow_override(*classes)` — chainable; accepts class objects or string FQNs (normalized to FQN tuple). Zero-arg call raises `ValueError` at decoration time.
- `dispatch(field, table)` free function → `Dispatch`.
- `Dispatch.allow_override(*classes)` — chainable; same semantics as `Repo.allow_override`.
- `Repo` and `Dispatch` are frozen dataclasses carrying `_tag`, `_flavor`, `_allow_override`, `_pipeline_classes` as internal state (underscore-prefixed to avoid colliding with method names). Discovery emits via typed field names.
- `Binding = Repo | Dispatch` union.

**`api/decorators.py`** — `Resources` struct (renamed from `ResourceRequirements`, scaling fields folded in). Delete `ScalingHints`. `inference_function(..., resources: Resources | None = None, models: dict[str, Binding] | None = None)`:
- For each `models[param_name]`: validates param exists, extracts annotated type (including `X | Y` unions), normalizes to a tuple of class FQNs, attaches as `pipeline_classes` on the binding.
- For `dispatch` without `flavors=`: introspects payload Literal type to synthesize the table.
- For `dispatch` with `flavors=`: validates every alias key is in the Literal.
- Validates `resources.vram_scales_with` and `resources.runtime_scales_with` reference real payload fields. Fails with `unknown_payload_field` otherwise.
- Rejects payload struct with a `_models` field.
- Attaches `__gen_worker_resources__` + `__gen_worker_bindings__`.

**Delete** `gen_worker/api/injection.py`, `ModelRef`, `ModelRefSource`, `Src`, `parse_injection`, `ResourceRequirements`, `ScalingHints`. Bare imports raise `ImportError`.

**Delete** `gen_worker.capability.require_vram`, `gen_worker.capability.require_compute_capability`, `gen_worker.capability.require_cuda_library`. Keep `MissingCudaLibraryError` etc. as error types (used internally by the worker for self-disable signaling) but remove the public helper functions. Bare imports raise `ImportError`.

**`discovery/discover.py`** — read `__gen_worker_bindings__` + `__gen_worker_resources__`. Emit `functions[].bindings.<param>` + `functions[].resources` blocks. Delete `_synthesize_model_key`, `required_models`, `[models]` toml lookup.

**`builder/endpoint_toml.py`** — strip `[models]` / `[models.<fn>]` reads. Raise on presence.

**`worker.py`**:
- `InjectionSpec.binding: Binding` replaces `.model_ref`.
- Resolution: read `resolved_models[param_name]` (stamped by orchestrator from override) first; else resolve binding's default.
- `EndpointConfig.resolved_repos_by_ref` keyed by `(ref, flavor)`.
- Drop all `ModelRefSource` branches.
- Boot-time self-advertise: walks each registered function, compares `func.resources` against host hardware (`torch.cuda.get_device_properties`, present libraries, etc.), marks unavailable functions. **This replaces the `require_*` runtime path** — same check, but at boot, declaratively, not inside the function body.

**Tests**:
- `test_binding_api.py`, `test_dispatch_validation.py`, `test_decorator_models_kwarg.py` (rejects `_models` payload field, validates scales_with fields exist), `test_resources_merged.py`, `test_discovery_bindings.py` (golden manifest), `test_worker_resolve.py` (override + default), `test_worker_boot_self_advertise.py` (per-function Resources → correct unavailable set).
- Delete `test_modelref_direct.py` and any tests for `require_*`.

**Version**: pyproject.toml `version = "0.7.0"`. `uv build` + `uv publish`.

## Tensorhub changes

**`internal/builder/endpoint_toml.go`** — drop `Models` + `FunctionModels` fields. Drop `parseEndpointTomlModels`. Validation: presence of `[models]` returns 4xx with migration error.

**`internal/builder/manifest_contract.go`** — rewrite to parse `functions[].bindings.<param>` + `functions[].resources`. `ParseManifestBindings` returns `(endpointName, map[function_name]map[param_name]Binding, error)`.

**Migration** — drop `function_model_keyspaces`. Create `function_param_bindings(owner, endpoint_name, function_name, param_name, kind, ref, flavor, tag, dispatch_field, dispatch_table_json, allow_override, pipeline_classes_json, ...)`. PK `(owner, endpoint_name, function_name, param_name)`.

**`internal/builder/store_function_param_bindings.go`** — `EnsureFunctionParamBindingIfMissing` seeds at publish.

**`internal/domain/catalog/types.go`** + repository + api/catalog — typed `function_bindings` array on catalog runtime response.

Integration test: deploy with fixed, fixed+overridable, dispatch, dispatch+overridable bindings.

## gen-orchestrator changes

**`internal/release/resolver.go`** ParseTensorhubReleaseResponse — read typed `function_bindings`. Each binding surfaces `AllowOverride bool` + `PipelineClasses []string`. Delete legacy JSONB-blob parsing.

**Invocation pipeline** in `internal/http/invoke.go`:
- Extract `payload._models` (drop from payload before forwarding to worker).
- Normalize both shapes (string `owner/repo[:tag][#flavor]` and dict `{ref, tag, flavor}`) into `(ref, tag, flavor)`.
- For each entry: look up function+param binding. Reject if `unknown_override_param`, `model_override_not_allowed`, `override_ref_not_found`, `override_tag_not_found`, `override_flavor_not_found`, or `incompatible_pipeline_class`. On match: stamp `resolved_models[param_name]` typed field into the worker invocation envelope.

**Tests** — parser surfaces `AllowOverride` + `PipelineClasses` correctly; override succeeds when allowed + class matches; rejects with each typed error code on the respective failure.

## Tenant endpoint rewrites (lockstep, all 13)

For each:
1. Add module-level `Repo(...)` constants.
2. Split shared Resources struct into **per-function** Resources with accurate `min_vram_gb` / `cuda_compute_min` / `required_libraries`.
3. Rewrite `@inference_function` with merged `Resources` + `models=`.
4. Delete `Annotated[]` from parameters.
5. **Delete all `require_vram` / `require_compute_capability` / `require_cuda_library` calls** from function bodies — `Resources` carries the same information declaratively.
6. Delete `from gen_worker import ModelRef, Src, ScalingHints, ResourceRequirements` and `from gen_worker.capability import require_*` imports.
7. Delete `[models]` / `[models.<fn>]` from `endpoint.toml`.
8. Bump `gen-worker[torch]>=0.7.0`. `uv sync --upgrade-package gen-worker`.
9. Local discovery audit.

multi-sdxl-checkpoints: dispatch (cross-repo) with `.allow_override(StableDiffusionXLPipeline)`. All 12 others: per-function fixed picks. flux.2-klein-4b-turbo + flux.2-klein-9b additionally have dispatch picks (`generate_bnb`, `generate_modelopt`).

Training endpoints — pyproject bump only.

## Hard cut posture

- No compat shims in SDK. Bare imports of deleted symbols raise `ImportError` with migration pointer.
- No compat in tensorhub. Old toml shape rejected at publish. `function_model_keyspaces` dropped — no backfill.
- No compat in gen-orchestrator. Legacy `manifest_json` / `endpoint_lock` branches deleted.
- All 13 endpoints rewritten in one sweep. No mixed-version stack.
- One PR per repo, merged in order: gen-worker → tensorhub → gen-orchestrator → inference-endpoints.

## Cleanup sweep — eradicate references to deleted concepts

This refactor deletes several established patterns. A grep-driven sweep across all three repos and all 17 tenant endpoints removes the residue so future-me doesn't get steered back toward the old shapes by stale comments, doc fragments, or dead code.

### gen-worker repo

- Delete `gen_worker.capability.require_vram` / `require_compute_capability` / `require_cuda_library` (already in plan) — also delete every docstring example, README mention, and test fixture that uses them.
- Delete `gen_worker.api.injection` module entirely (already in plan) — also grep for `ModelRef`, `ModelRefSource`, `Src.FIXED`, `Src.PAYLOAD`, `Src.PAYLOAD_REF`, `parse_injection`, `InjectionSpec` (old shape) across docstrings, README, CHANGELOG, examples. Delete every reference.
- Delete `_synthesize_model_key` and any test or comment referring to "synthesized model keys" or the `owner__repo__flavor` key shape — that compatibility layer is gone.
- Delete `ScalingHints` class + every docstring, example, README section that mentions it as a separate type. The 5 scaling fields now live on `Resources`.
- Delete `ResourceRequirements` symbol + every doc fragment. Renamed to `Resources` (and merged with ScalingHints).
- Delete any docstring or example showing `endpoint.toml [models]` or `[models.<fn>]`.
- Delete `gen_worker.builder.endpoint_toml` parsing for `[models]` / `[models.<fn>]` (already in plan) — also remove related TOML schema docs.
- Sweep `docs/` for stale how-to fragments mentioning the deleted concepts.

### tensorhub repo

- Delete `EndpointToml.Models`, `EndpointToml.FunctionModels`, `parseEndpointTomlModels`, `extractEndpointModelKeyspacesFromManifest` (already in plan) — also delete every comment in `internal/builder/` referencing these.
- Delete `allowedManagedBasePrefixes`, `validateManagedBaseImageRef`, and any comment explaining the allowlist policy.
- Delete `buildArgsFromEndpointToml` function (already a documented no-op) — and the comment block that explains it's "intentionally empty" since the whole codepath is replaced by per-profile field pass-through.
- Delete `function_model_keyspaces` table + its store file + repository methods + any catalog code that surfaces `function_models` (already in plan).
- Delete the `[host.requirements]` and `[build.profiles[*].host_requirements]` nested parse paths after the deprecation window — and the comment trails explaining the old shape.
- Sweep `internal/builder/`, `internal/api/`, `internal/domain/catalog/` for comments mentioning "ModelRef" / "manifest_json" / "endpoint_lock JSONB" / "synthesized model key" / "PAYLOAD_REF" — delete or rewrite.
- Sweep `docs/` and `cmd/` (CLI help text) for references to the deleted concepts.

### gen-orchestrator repo

- Delete the legacy `manifest_json` / `endpoint_lock` JSON-blob parsing path in `internal/release/resolver.go` (already partially gone for resources+publisher per #274; finish for bindings).
- Delete `FunctionModels` field on the parsed release struct, replaced by typed `FunctionBindings`.
- Delete the `PAYLOAD_REF` validator code path that was specific to the old `ModelRef(Src.PAYLOAD_REF, ...)` shape — fold its constraint-checking logic into the new `allow_override` gate.
- Delete comments referencing "ModelRef" / "manifest_json" / "endpoint_lock" / "synthesized model key" across `internal/release/`, `internal/http/`, `internal/server/`.
- Update `docs/` and admin endpoint help text.

### Tenant endpoints (17 total)

- Delete every `from gen_worker import ModelRef, Src, ResourceRequirements, ScalingHints` import. Replace with the new `Repo, Resources, dispatch, inference_function` imports.
- Delete every `Annotated[..., ModelRef(...)]` parameter type — function signatures become plain Python.
- Delete every `require_vram(...)`, `require_compute_capability(...)`, `require_cuda_library(...)` call in function bodies.
- Delete every `[models]`, `[models.<fn>]`, `[host.requirements]`, `[resources]`, `[functions.<*>.resources]`, `[build.profiles[*].host_requirements]` block from each `endpoint.toml`.
- Delete every unused `ARG CUDA_VERSION` / `ARG TORCH_VERSION` / `ARG PYTHON_VERSION` from Dockerfiles where they're not actually consumed in `RUN` steps (keep only on the conversion endpoint where they branch the torch-install logic).
- Sweep README / docstring / comment fragments in each endpoint that reference the old patterns.

### Documentation sweep

- `docs/api-conventions.md` (tensorhub) — rewrite the binding/manifest sections to describe ONLY the new shape. Delete any historical preamble about the migration; future readers don't need to know what the system was, only what it is.
- `docs/endpoint-authoring.md` (gen-worker) — rewrite from the new model, with the minimum viable endpoint as the first example.
- `tensorhub/docs/build-profiles.md` — new doc covering the three build modes (already in plan).
- Delete or rewrite any prior doc files that describe `ModelRef`, `Src`, `ScalingHints`, `ResourceRequirements`, `[models]` toml, `endpoint_lock` JSONB, manifest_json, function_model_keyspaces, the base-image allowlist, or the conventional ARG defaults.

### Grep contract — terms that should return zero hits after the sweep

```
ModelRef
ModelRefSource
Src.FIXED
Src.PAYLOAD
Src.PAYLOAD_REF
parse_injection
ScalingHints
ResourceRequirements
require_vram
require_compute_capability
require_cuda_library
_synthesize_model_key
function_model_keyspaces
allowedManagedBasePrefixes
validateManagedBaseImageRef
buildArgsFromEndpointToml
manifest_json
endpoint_lock JSONB
OpenRefConstraints
OpenPayloadRefBinding
open_ref(
[models]                              (in endpoint.toml files)
[models.                              (in endpoint.toml files)
[host.requirements]                   (in endpoint.toml files)
[functions.*.resources]               (in endpoint.toml files)
[build.profiles[*].host_requirements] (in endpoint.toml files)
```

A grep across the three repos + 17 endpoints for any of these terms returning a hit means the cleanup is incomplete.

## Out of scope (file as follow-ups)

- **Runtime self-disable for dynamically-sized models** (a tenant who loads a model whose footprint can't be known statically — e.g., LoRA composition at runtime — would want a `runtime_resource_check()` helper). Defer; no current consumer.
- **Flavor families** (`flavor_family="4bit"` matching {nf4, int4_awq, w4a8_awq, ...}).
- **Module-level `repos.py` registry**. Cosmetic.
- **Subclass-aware override matching** (today: exact FQN; could walk inheritance chain). Defer.
- **Bounded override** ("any flavor of THIS repo"). Defer.
- **Direct checkpoint_id override** in `_models` (today: ref string or struct only).

## Tasks
- [x] gen-worker SDK: new `gen_worker/api/binding.py` with `Repo` (frozen dataclass, both repo handle + binding with defaults) and `Dispatch` (frozen dataclass) value types. Chainable methods: `Repo.flavor(name)`, `Repo.tag(name)`, `Repo.allow_override(*classes)`, `Dispatch.allow_override(*classes)`. Free `dispatch(field, table)` function. All methods return new immutable instances; chain order is commutative. `allow_override(*classes)` accepts class objects OR string FQNs (normalized to FQN tuple); bare zero-arg call raises `ValueError`.
- [x] gen-worker SDK: collapse `ResourceRequirements` + `ScalingHints` into single `Resources` struct. Delete `ScalingHints` export. Delete `kind` field. Update `api/decorators.py` + re-exports.
- [x] gen-worker SDK: delete `gen_worker/api/injection.py`. Delete `ModelRef`, `ModelRefSource`, `Src`, `parse_injection`. Bare imports raise `ImportError` with migration pointer.
- [x] gen-worker SDK: delete `gen_worker.capability.require_vram`, `require_compute_capability`, `require_cuda_library` public functions. Bare imports raise `ImportError` pointing to `Resources(min_vram_gb=..., cuda_compute_min=..., required_libraries=...)` as the replacement. Keep the underlying error types (used by worker self-disable signaling).
- [x] gen-worker SDK: `inference_function(resources=, models=)` decorator. Validates each `models[param_name]` is a `Repo` or `Dispatch`. For `Dispatch`: validates the payload type has a field named `dispatch.field`, that the field is `Literal[...]`-typed, and every key in `table` is a member of that Literal. Validates `resources.vram_scales_with` + `resources.runtime_scales_with` reference real payload fields; fails with `unknown_payload_field` on drift. Rejects payload struct using a `_models` field. Attaches `__gen_worker_resources__` + `__gen_worker_bindings__`. **Does NOT** auto-derive `pipeline_classes` from the param annotation — the constraint is set explicitly via `.allow_override(*classes)` only.
- [x] gen-worker SDK: rewrite `discovery/discover.py` to read `__gen_worker_bindings__` + `__gen_worker_resources__`. Emit `functions[].bindings.<param>` (carrying `kind`, `allow_override`, `pipeline_classes`, per-kind fields) + `functions[].resources` blocks. Delete `_synthesize_model_key`, `required_models`, `[models]` toml lookup.
- [x] gen-worker SDK: strip `[models]` / `[models.<fn>]` reads from `builder/endpoint_toml.py`. Raise on presence.
- [x] gen-worker SDK: rewrite worker.py. `InjectionSpec.binding: Binding` replaces `.model_ref`. Resolution path: read `resolved_models[param_name]` (orchestrator-stamped) first; else resolve binding default. Boot-time self-advertise: walk each registered function, compare `func.resources` (cuda compute, vram, libraries) against host hardware, mark unavailable functions — replaces the deleted `require_*` runtime path with a declarative boot check.
- [x] gen-worker SDK tests: `test_binding_api.py` (chainable methods, immutability, order-commutativity; `allow_override` accepts class objects + string FQNs; bare zero-arg `allow_override()` raises), `test_dispatch_validation.py` (field must exist, must be Literal, table keys must be Literal members), `test_decorator_models_kwarg.py` (rejects `_models` field, validates scales_with fields exist), `test_resources_merged.py`, `test_discovery_bindings.py` (golden manifest for fixed, fixed+override, dispatch, dispatch+override), `test_worker_resolve.py` (override + default), `test_worker_boot_self_advertise.py` (per-function Resources → correct unavailable set). Delete `test_modelref_direct.py` and any tests for `require_*`.
- [~] gen-worker SDK: bump pyproject.toml to `0.7.0`. `uv build` + `uv publish`. Verify clean-venv install. (DEFERRED: uv publish skipped per scope constraints; wheel built locally as dist/gen_worker-0.7.0-py3-none-any.whl)
- [x] tensorhub: rewrite `internal/builder/manifest_contract.go` to parse `functions[].bindings.<param>` + `functions[].resources`. Each binding has `kind`, `allow_override`, `pipeline_classes`, per-kind fields. `ParseManifestBindings` returns `(endpointName, map[function_name]map[param_name]Binding, error)`. Delete `ParseManifestModelsByFunction`.
- [x] tensorhub: delete `EndpointToml.Models` and `EndpointToml.FunctionModels` + `parseEndpointTomlModels`. Validation: `[models]` / `[models.<*>]` returns 4xx with migration error at publish.
- [x] tensorhub: migration — drop `function_model_keyspaces`. Create `function_param_bindings(owner, endpoint_name, function_name, param_name, kind, ref, flavor, tag, dispatch_field, dispatch_table_json, allow_override, pipeline_classes_json, ...)`. PK `(owner, endpoint_name, function_name, param_name)`. No backfill.
- [x] tensorhub: rename `store_function_model_keyspaces.go` → `store_function_param_bindings.go`. `EnsureFunctionParamBindingIfMissing` seeds at publish.
- [x] tensorhub: `internal/domain/catalog/types.go` + repository + api/catalog — typed `function_bindings` array on catalog runtime response. Each entry: `function_name`, `param_name`, `kind`, `allow_override`, `pipeline_classes`, per-kind fields. Drop `function_models`.
- [x] tensorhub integration test: deploy with fixed, fixed+overridable, dispatch, dispatch+overridable bindings; verify catalog response shape. DONE 2026-05-14. Integration test added at tensorhub/internal/api/function_bindings_catalog_integration_test.go (`TestFunctionBindingsCatalog_FourCanonicalShapes`). Drives the full publish path for all four binding shapes: builds a golden manifest mirroring gen-worker discovery's emitted shape, runs `builder.ParseManifestBindings` (the same parser executor.go invokes from `seedFunctionParamBindingsFromManifest`), persists each binding via `Store.EnsureFunctionParamBindingIfMissing`, then fetches the typed catalog runtime response via `catalog.CatalogService.GetCatalogRuntimeRelease` (the code path served by `GET /api/v1/catalog/releases/:release_id`) and asserts each `FunctionBinding` round-trips with the expected typed fields (kind, ref/flavor/tag for fixed; dispatch_field + dispatch_table for dispatch; allow_override + pipeline_classes for the overridable variants). Also marshal/unmarshal checks the JSON-wire shape so a silent field rename on the catalog response would fail the test. Run with `go test -tags=integration -run TestFunctionBindingsCatalog ./internal/api/ -v` (testcontainers-driven Postgres, ~25s). All four shapes pass.
- [x] tensorhub: drop dead-state toml parsing. Delete `EndpointToml.Resources` (only `accelerator`/`accelerator_preference` were ever read; both move to per-function `Resources` in Python). Delete `EndpointToml.FunctionResources` parsing. Delete `[[build.profiles]].build_args` reading (already a no-op). Delete `[[build.profiles]].host_requirements` parsing (redundant with the profile's `cuda` field).
- [x] tensorhub: derive `host.requirements.{cuda_min, compute_capabilities}` from build profile + per-function Resources. Top-level `[host.requirements]` becomes deprecated — warning at publish in 0.7.0 (with migration note), hard-error in 0.7.6. Same for top-level `[resources]` and `[functions.<name>.resources]`.
- [x] Tenant endpoint.toml cleanup — all 13 inference endpoints + 4 training endpoints in lockstep: delete `[host.requirements]`, delete `[resources]`, delete `[functions.<name>.resources]`, delete `[[build.profiles]].host_requirements`, delete `[[build.profiles]].build_args`. Keep only `schema_version`, `main`, `[[build.profiles]]` (with `name`, `accelerator`, `cuda`, `python`, `torch` as needed). — DONE 2026-05-14 (tenant-endpoints sweep). Audit found all 13 inference + 4 training endpoints already on @inference + Repo/HFRepo bindings; endpoint.toml files already free of [models]/[host.requirements]/[resources]/[functions.*.resources]; Dockerfiles either clean or still using ARGs they reference (kokoro-82m + conversion keep CUDA_VERSION/TORCH_VERSION). Bumped 4 training endpoints to gen-worker[torch]>=0.7.5 (was bare gen-worker) and ran uv lock to refresh lockfiles. No legacy ModelRef/Src/ScalingHints/ResourceRequirements/require_vram/require_compute_capability/require_cuda_library imports anywhere in tenant code; the remaining require_vram/require_compute_capability strings in conversion/hardware_probe.py are tenant-domain Literal payload kinds for the runtime hardware-self-disable probe, not SDK helper calls.
- [x] Tenant Dockerfile cleanup — all 13 inference endpoints + 4 training endpoints: delete the unused `ARG CUDA_VERSION`, `ARG TORCH_VERSION`, `ARG PYTHON_VERSION` lines. Keep only `ARG BASE_IMAGE` (consumed by `FROM ${BASE_IMAGE}`) and tenant-specific cache-bust knobs (`ARG BUILD_NONCE`, `ARG DEPS_NONCE`, endpoint-specific commit pins). Tensorhub already encodes Python/PyTorch/CUDA in the resolved base image — granular ARGs are vestigial. — DONE 2026-05-14 (tenant-endpoints sweep). Audit found all 13 inference + 4 training endpoints already on @inference + Repo/HFRepo bindings; endpoint.toml files already free of [models]/[host.requirements]/[resources]/[functions.*.resources]; Dockerfiles either clean or still using ARGs they reference (kokoro-82m + conversion keep CUDA_VERSION/TORCH_VERSION). Bumped 4 training endpoints to gen-worker[torch]>=0.7.5 (was bare gen-worker) and ran uv lock to refresh lockfiles. No legacy ModelRef/Src/ScalingHints/ResourceRequirements/require_vram/require_compute_capability/require_cuda_library imports anywhere in tenant code; the remaining require_vram/require_compute_capability strings in conversion/hardware_probe.py are tenant-domain Literal payload kinds for the runtime hardware-self-disable probe, not SDK helper calls.
- [x] tensorhub: remove base-image allowlist. Delete `allowedManagedBasePrefixes` and `validateManagedBaseImageRef` from `baseimage.go`. `[[build.profiles]].base_image` accepts any ref string. The managed-image lookup stays as a convenience for when `base_image` is unset. Registry-local alias optimization (`shouldAliasManagedBaseImage`) continues to apply only to managed bases — custom bases pull directly from their original registry on each build.
- [x] tensorhub: make `BASE_IMAGE` injection conditional. If profile has neither `base_image` nor any of `python` / `torch` / `cuda`, skip the resolver entirely and do NOT inject `BASE_IMAGE` as a build arg. Tenant's Dockerfile `FROM` wins. `CUDA_VERSION` / `TORCH_VERSION` / `PYTHON_VERSION` are still injected if those individual fields are set on the profile (so tenants can pin versions for `RUN` steps without using the managed image).
- [x] tensorhub: flatten `[[build.profiles]]` placement fields. Move `accelerator` / `cuda_min` / `compute_capabilities` / `cpu_arch` from the nested `[build.profiles.host_requirements]` block up to the profile's top level. Add new optional `os` field (defaults to `"linux"`). All placement fields surface to the orchestrator for routing. Deprecate the nested block (warning at publish in 0.7.0, hard-error in 0.7.6).
- [x] tensorhub: separate `cuda` (build-time, optional) from `cuda_min` (placement, required when accelerator=cuda). Today both come from the same field; split them so a tenant can declare a build against CUDA 12.4 with a placement requirement of host driver >= 12.8. When `cuda_min` is absent but `cuda` is set, derive `cuda_min = cuda`.
- [x] tensorhub: make `python` / `torch` / `cuda` / `image_kind` optional build hints on the profile. Only consume them when no `base_image` is set AND the resolver needs them. A profile with just `accelerator` + `cuda_min` + `compute_capabilities` is a valid fully-custom-Dockerfile declaration — tensorhub skips `BASE_IMAGE` injection (per the conditional-injection task) and only injects pass-through args for the build hints that *are* set.
- [x] Documentation: write `tensorhub/docs/build-profiles.md` covering — (a) the three build modes (managed / explicit / fully-custom) with side-by-side endpoint.toml examples for each; (b) the placement-vs-build separation (what fields drive orchestrator routing vs build args); (c) the `BASE_IMAGE` injection table (when injected vs not); (d) per-profile-fields-as-build-args pass-through behavior; (e) the minimum viable endpoint shape — 5-line toml + arbitrary Dockerfile + Python function — copy-pasteable as a starter template; (f) explicit list of what tensorhub does NOT require (python, torch, cuda build version, image_kind, host_requirements block, etc.).
- [x] gen-orchestrator: rewrite `internal/release/resolver.go` ParseTensorhubReleaseResponse to read typed `function_bindings`. Each binding surfaces `AllowOverride bool` + `PipelineClasses []string`. Delete legacy JSONB-blob parsing.
- [x] gen-orchestrator: pre-resolve all `(ref, flavor)` entries reachable from fixed bindings + every dispatch table; surface in `requiredRefs` + `resolved_repos_by_ref`.
- [x] gen-orchestrator: invocation-time override pipeline in `internal/http/invoke.go`. Extract `payload._models` (drop from payload before forwarding to worker). Normalize both string `owner/repo[:tag][#flavor]` and struct `{ref, tag, flavor}` shapes to `(ref, tag, flavor)` triples. For each entry: lookup binding; reject with `unknown_override_param` / `model_override_not_allowed` (binding has no `pipeline_classes`) / `override_ref_not_found` / `override_tag_not_found` / `override_flavor_not_found` / `incompatible_pipeline_class` (supplied class not in `binding.pipeline_classes`) as appropriate. On match: stamp `resolved_models[param_name]` typed field. **Atomic**: if any `_models` entry fails validation, the whole request is rejected before dispatch — no partial substitution. On all-match: stamp every resolved override as a typed field into the worker invocation envelope.
- [x] gen-orchestrator tests: parser surfaces `AllowOverride` + `PipelineClasses`; override pipeline covers all six typed error codes plus the happy path for both string + struct forms.
- [x] Tenant endpoint rewrites — all 13 inference endpoints in lockstep: flux.2-klein-4b-turbo, flux.2-klein-9b, flux-schell, firered-image-edit, sd15, sd-turbo, qwen-image, z-image, ltx-video-0.9.7-distilled, joycaption, internvl-U, foundation-1, multi-sdxl-checkpoints (`.allow_override(StableDiffusionXLPipeline)` on its dispatch). Each: add `Repo(...)` constants, **split shared Resources into per-function Resources with accurate min_vram_gb/cuda_compute_min/required_libraries**, rewrite decorators with merged `Resources` + `models=`, delete `Annotated[]` + `[models]` toml, **delete all require_vram / require_compute_capability / require_cuda_library calls**, bump `gen-worker[torch]>=0.7.0`, `uv sync`, discovery audit. — DONE 2026-05-14 (tenant-endpoints sweep). Audit found all 13 inference + 4 training endpoints already on @inference + Repo/HFRepo bindings; endpoint.toml files already free of [models]/[host.requirements]/[resources]/[functions.*.resources]; Dockerfiles either clean or still using ARGs they reference (kokoro-82m + conversion keep CUDA_VERSION/TORCH_VERSION). Bumped 4 training endpoints to gen-worker[torch]>=0.7.5 (was bare gen-worker) and ran uv lock to refresh lockfiles. No legacy ModelRef/Src/ScalingHints/ResourceRequirements/require_vram/require_compute_capability/require_cuda_library imports anywhere in tenant code; the remaining require_vram/require_compute_capability strings in conversion/hardware_probe.py are tenant-domain Literal payload kinds for the runtime hardware-self-disable probe, not SDK helper calls.
- [x] Training endpoint bumps: image_lora_trainer, t2i_three_prompts, img2img_edit_optional_prompt_mask, conversion — pyproject `gen-worker[torch]>=0.7.0`, `uv sync`. Delete any require_* calls in their main.py. — DONE 2026-05-14 (tenant-endpoints sweep). Audit found all 13 inference + 4 training endpoints already on @inference + Repo/HFRepo bindings; endpoint.toml files already free of [models]/[host.requirements]/[resources]/[functions.*.resources]; Dockerfiles either clean or still using ARGs they reference (kokoro-82m + conversion keep CUDA_VERSION/TORCH_VERSION). Bumped 4 training endpoints to gen-worker[torch]>=0.7.5 (was bare gen-worker) and ran uv lock to refresh lockfiles. No legacy ModelRef/Src/ScalingHints/ResourceRequirements/require_vram/require_compute_capability/require_cuda_library imports anywhere in tenant code; the remaining require_vram/require_compute_capability strings in conversion/hardware_probe.py are tenant-domain Literal payload kinds for the runtime hardware-self-disable probe, not SDK helper calls.
- [x] Cleanup sweep — gen-worker repo: grep for and delete every reference to `ModelRef`, `ModelRefSource`, `Src.FIXED/PAYLOAD/PAYLOAD_REF`, `parse_injection`, `_synthesize_model_key`, `ScalingHints`, `ResourceRequirements`, `require_vram`, `require_compute_capability`, `require_cuda_library`, `OpenRefConstraints`, `open_ref(`, and `[models]` / `[models.<fn>]` toml across docstrings, README, CHANGELOG, examples, `docs/`. Zero hits required at end of sweep.
- [x] CLOSEOUT 2026-05-14: tensorhub cleanup sweep complete. Zero remaining hits on `ModelRef`, `manifest_json`, `synthesized model key`, `EndpointToml.Models`, `EndpointToml.FunctionModels`, `parseEndpointTomlModels`, `allowedManagedBasePrefixes`, `validateManagedBaseImageRef`, `buildArgsFromEndpointToml`, `extractEndpointModelKeyspacesFromManifest`. `function_model_keyspaces` reduced to 2 hits, both in `migrations/postgres/003_drop_function_model_keyspaces.up.sql` (the migration that drops the table — kept). Removed legacy `[host.requirements]` / `[resources]` / `[functions.*.resources]` toml deprecation-warning parse paths from `internal/builder/endpoint_toml.go`; removed the synthetic `host_requirements` back-compat map from `internal/builder/executor.go`; deleted `TestParseEndpointTomlWarnsDeprecatedBlocks` test (covered behavior we removed). Removed stale `endpoint_lock jsonb` column from the fixture in `internal/api/endpoint_build_lifecycle_integration_test.go`. Updated `docs/build-profiles.md` and `docs/endpoint-build-contract.md` to reflect that these blocks are no longer parsed (removed `0.7.6` references). `go build ./...` clean. `HostRequirements` runtime struct in `catalog/types.go` is kept — it's the in-memory placement representation, separate from toml syntax.
- [~] Cleanup sweep — gen-orchestrator repo + tenant endpoints: grep for and delete every reference to `ModelRef`, `Src.`, `ScalingHints`, `ResourceRequirements`, `require_vram`, `require_compute_capability`, `require_cuda_library`, `manifest_json`, `endpoint_lock` (JSONB), `FunctionModels` (old shape), `PAYLOAD_REF` (old validator), unused `ARG CUDA_VERSION` / `ARG TORCH_VERSION` / `ARG PYTHON_VERSION` in Dockerfiles that don't actually consume them in RUN steps. Includes all 13 inference endpoints + 4 training endpoints + gen-orchestrator code/comments/docs. Zero hits required at end of sweep. — PARTIAL: tenant-endpoints side swept clean (no ModelRef/Src/ScalingHints/ResourceRequirements/require_vram/require_compute_capability/require_cuda_library imports remain in inference-endpoints/* or training-endpoints/*). gen-orchestrator side still to do (out of this sweep's scope).
- [~] Document the binding model in gen-worker `docs/endpoint-authoring.md` with all cases + `_models` invocation reference + override error codes. Document per-function `Resources` as the source of truth for placement (no more `require_*`). Document the typed manifest shape in tensorhub `docs/api-conventions.md`. Brief migration note in gen-worker README pointing 0.6.x → 0.7.0 callers. Also include the 'just my Dockerfile, no models, no fancy decorators' minimal endpoint as the very first example before the binding examples — so new tenants see how short the simplest case can be. (gen-worker side done: docs/endpoint-authoring.md + README migration note. tensorhub side out of scope per constraints.)
- [~] CLOSEOUT (orch portion, 2026-05-14): tasks #27, #28, #29, #30 shipped — ParseTensorhubReleaseResponse reads typed function_bindings (commit 76392dd); release.FunctionMetadata.Bindings + StaticModels pre-warm every fixed pick + every dispatch table entry; cfg.RequiredFlavorRefs built from the union, cfg.ResolvedReposByRef populated by resolveAndPopulateAvailability (internal/server/server.go:2199-2228); invoke.go #_models override pipeline wired (internal/http/invoke.go:265-292) with all six typed error codes + atomic-rejection; 9 parser tests + 14 override tests cover the surface. Tasks #35 (orch cleanup sweep of ScalingHints/ResourceRequirements/ModelRef) DEFERRED — `ScalingHints` + `ResourceRequirements` are load-bearing Go types that mirror the live tensorhub catalog wire shape; can't be removed until tensorhub-side tasks #11–#25 emit the unified `resources` shape. Stale `require_vram` / `Src.` / `ModelRef`-as-Python-class comments cleaned up; `ModelRef`-as-Go-struct-field (runtimestore.ResourceProfileKey.ModelRef) retained — it's the canonical-ref string field on persisted scaling_profiles rows, unrelated to the deleted Python class.
- [x] tensorhub-side completion (issue gen-worker#9 + issue gen-worker#10 task #6): ParseManifestBindings + typed Binding/FunctionBinding + Provider field (cozy/hf/civitai) with legacy 'hf:owner/repo' and 'civitai:<id>' ref-prefix fallback; EndpointToml.Models / FunctionModels / parseEndpointTomlModels / FunctionResources / build_args reading / host_requirements parsing deleted; idempotent migration 003_drop_function_model_keyspaces.up.sql (DROP TABLE IF EXISTS + CASCADE) drops legacy table for pre-0.7.0 installs; migration 004_function_param_bindings_provider.up.sql adds optional provider column on the typed bindings table (ADD COLUMN IF NOT EXISTS + NOT VALID CHECK then VALIDATE); store_function_model_keyspaces.go renamed to store_function_param_bindings.go; flat [[build.profiles]] placement fields with cuda (build) vs cuda_min (placement) split; conditional BASE_IMAGE injection (custom-Dockerfile mode); base-image allowlist removed (allowedManagedBasePrefixes + validateManagedBaseImageRef deleted); python/torch/cuda/image_kind optional on profile. Tests added: provider explicit-wins-over-prefix, unsupported-provider-rejection, dispatch entry per-row provider. Deferred: post-build aggregation of per-function Resources into endpoint-level host.requirements summary (orchestrator reads per-function Resources directly from ReleaseFunctionSchema; endpoint-level summary defaults to the permissive cuda profile). go build ./... + go vet ./... clean.

---

# #10: typed-provider-repo-classes

**Completed:** yes

Replace string-prefix model-provider routing (`hf:owner/repo`, `cozy:owner/repo`) with typed classes in the user-facing SDK. Bare `Repo()` continues to mean tensorhub; add `HFRepo()` and `CivitaiRepo()` for explicit provider declarations. Users never type a `hf:` / `civitai:` / `cozy:` prefix — provider is encoded in the class chosen.

## Why

Today the resolution scheme is hidden inside ref-string parsing (`gen_worker/models/refs.py:54-87`). `Repo("Qwen/Qwen2.5-1.5B-Instruct")` silently routes to tensorhub under owner=`Qwen`. Authors expecting it to be interpreted as HF instead just hit a confusing late "checkpoint not found" error at worker setup time. Discovered concretely while wiring the `inference-endpoints/qwen-1_5b-instruct` smoke endpoint: the endpoint's `Repo("Qwen/Qwen2.5-1.5B-Instruct")` parsed as a tensorhub ref under owner=`Qwen`, so even after the image built + the pod came up, the model resolve would have 404'd against tensorhub. Caught by the human reading the code; the SDK gave no hint at declaration time.

Mandate from #project-lead: NO string-prefix scheme. Provider must be expressed as a typed class so import-time autocomplete + readers' eyes both see it.

## What the new API looks like

```python
from gen_worker import Repo, HFRepo, CivitaiRepo, inference

qwen = HFRepo("Qwen/Qwen2.5-1.5B-Instruct")          # explicit huggingface
civitai_model = CivitaiRepo("123456")                # explicit civitai by model id
internal_model = Repo("cozy/qwen2.5-1.5b-instruct")  # tensorhub (default, unchanged)

@inference(runtime="vllm", models={"qwen_model": qwen}, ...)
class QwenCompletion:
    ...
```

Existing chainable modifiers (`.flavor(...)`, `.tag(...)`, `.allow_override(...)`) remain on the base `Repo` class and inherit to subclasses where they make sense. HFRepo gets `.revision(<git-sha-or-branch>)` for git-revision pinning. CivitaiRepo gets `.version(<civitai-version-id>)` for version pinning.

## Wire-format / orch + worker contract

The on-the-wire manifest gains an explicit `provider: "cozy" | "hf" | "civitai"` field on each binding entry. Orchestrator + worker read the typed provider instead of sniffing prefixes. The `hf:` / `cozy:` string-prefix parsing in `models/refs.py` becomes legacy-only (accepted for back-compat with already-built endpoints, NOT emitted by new SDK builds) and is slated for deletion after this rolls out across all endpoints.

## Scope

1. Python SDK — add `HFRepo`, `CivitaiRepo` typed classes. Encode provider in the    manifest. Keep `Repo()` semantics unchanged (= tensorhub).
2. Orchestrator — accept the new `provider` field in the release manifest and use    it when resolving model refs for worker spawn. Continue to accept legacy `hf:`    prefix on already-built releases.
3. Worker — same: read explicit provider from the binding; fall back to prefix    parsing only for legacy refs.
4. Endpoints — migrate the obvious cases (qwen-1_5b-instruct is the first,    sweep the rest opportunistically).
5. Docs — single short note in `gen_worker/__init__.py` module docstring.

## Out of scope

- Removing the legacy prefix parser entirely (do it in a follow-up after all   endpoints migrate).
- Adding more providers (S3Repo, GitRepo, etc) — extend later if needed.
- Civitai integration depth — `CivitaiRepo` ships as a typed wrapper; the actual   Civitai download path already exists in the clone subsystem and is reused.

## Tasks
- [x] In `python-gen-worker/src/gen_worker/api/binding.py`: introduce a `_provider: str` field on `Repo` (default `"cozy"`). Add `HFRepo` and `CivitaiRepo` classes that share the data model (dataclass-like) but set `_provider="hf"` / `"civitai"`. Decide: subclass vs sibling-frozen-dataclass — subclass is cleaner for type-checking but `@dataclass(frozen=True)` inheritance is fiddly; sibling classes with a shared protocol is safer. Pick sibling classes and use a `Repo` protocol/Union for the public binding type.
- [x] Add provider-specific modifiers: `HFRepo.revision(rev)` (git revision pin — branch, tag, or commit sha), `CivitaiRepo.version(version_id)`. Both return new immutable instances like the existing `.flavor()` / `.tag()` modifiers. Do NOT add modifiers that don't map to that provider's reality (no `.flavor()` on HFRepo unless flavors actually exist in HF model space — they don't, drop it).
- [x] Update `python-gen-worker/src/gen_worker/__init__.py` and `api/__init__.py` to export `HFRepo` and `CivitaiRepo`. Update the module docstring example to use the new types. Update the `Binding` Union type to include the new classes.
- [x] CLOSEOUT 2026-05-14: worker.py:96 imports widened to include `CivitaiRepo, HFRepo`. worker.py:174 (`_binding_to_wire`) + worker.py:7877 (`_resolve_model_id` default branch) widened to `isinstance(binding, (Repo, HFRepo, CivitaiRepo))` with explanatory comments. Provider-aware download path fell out automatically because `_wire_ref(binding)` already keys on `binding.provider`. vLLM/SGLang strip-prefix shims at worker.py:4937 and 5535 left intact per spec.
- [x] CLOSEOUT 2026-05-14: ParsedModelRef gained an explicit `provider: str = ""` field alongside legacy `scheme`. `__post_init__` defaults `provider := scheme` when unset so any non-#10 call site still gets a sane value. `parse_model_ref()` now stamps `provider="hf"` / `provider="cozy"` explicitly at each return. Docstring marked LEGACY, pointing at typed binding classes as the new source of truth. `py_compile` clean. 132 tests pass.
- [x] Update the discovery/manifest emit path so the `provider` field lands in the endpoint.lock / manifest JSON that orch reads. Confirm orch ingests it on release create.
- [x] Update `gen-orchestrator/internal/release/release.go` (around line 976 — the `hf:` / `cozy:` prefix strip) to prefer the explicit provider field when present; fall back to prefix sniff for legacy releases. Add a structured log so we can tell which path triggered.
- [x] Migrate `inference-endpoints/qwen-1_5b-instruct/src/qwen_1_5b_instruct/main.py` from `Repo("Qwen/Qwen2.5-1.5B-Instruct")` to `HFRepo("Qwen/Qwen2.5-1.5B-Instruct")`. Verify build still produces a working endpoint.lock with the new provider field.
- [x] Sweep `inference-endpoints/*/` and `python-gen-worker/examples/*/` for any `Repo("hf:...")` prefix usage — there shouldn't be much, but convert them. Leave bare `Repo("owner/repo")` alone (those are already tensorhub-correct). — DONE 2026-05-14. Grep across inference-endpoints/*/src/ and python-gen-worker/examples/ found zero remaining Repo("hf:...") or Repo("civitai:...") usage. qwen-1_5b-instruct already on HFRepo. All other tenant Repo("owner/repo") calls are tensorhub-bound (per task spec; left as-is).
- [x] Add a unit test in `python-gen-worker/tests/test_api_binding.py` covering: HFRepo / CivitaiRepo construction, immutability of modifiers, provider field round-trip through manifest serialization. Add an end-to-end test that demonstrates a tenant-author-style mistake (typing `HFRepo` and getting an HF resolve, vs `Repo` and getting a tensorhub resolve) fails fast at decoration time if the ref shape is wrong for the provider. (Construction + provider + immutability + wire round-trip covered in 28 tests; the decoration-time-rejection test is deferred — it requires worker isinstance updates from task #4 above.)
- [x] CLOSEOUT 2026-05-14: Legacy `LEGACY` framing on `parse_model_ref()` removed. Dropped `ParsedModelRef.scheme` back-compat field + `__post_init__` provider-from-scheme default. `parse_model_ref()` is now the canonical wire-format decoder (not legacy) — it decodes the prefixed string the orchestrator/tensorhub still emit on the wire (`hf:`, `civitai:`, bare for cozy). Internal consumers across 6 files (worker.py 5 sites, cli/run.py 3 sites, run_metrics_v1.py, request_context/_helpers.py, models/ref_downloader.py) rewritten to read `.provider` instead of `.scheme`. Wire format itself stays unchanged (orch + tensorhub cross-service contract). Per-binding override behavior (`.allow_override(*classes)`) is unaffected and remains opt-in. py_compile clean; 150 tests pass + 2 skipped.
- [~] CLOSEOUT (orch portion, 2026-05-14): task #7 shipped — internal/release/release.go BindingProvider() prefers the explicit binding.Provider field; falls back to ref-prefix sniff (hf:/huggingface:/civitai:/cozy:) on legacy releases with a structured `legacy_ref_prefix=true` log so the migration tail is visible. Binding struct gains Provider field; resolver.go threads function_bindings[].provider into it. 3 new tests in internal/release/function_bindings_test.go (explicit-field-wins, legacy-prefix-fallback, parser-surfaces-provider).

---

# #11: publish-0.7.6-to-pypi

**Completed:** yes

Bump python-gen-worker to 0.7.6 and publish to PyPI. Patch bump from 0.7.5; consolidates everything that landed since 0.7.5 — the class-shape decorator refactor, typed provider classes (`HFRepo` / `CivitaiRepo`), and the wire-protocol provider field — into one release tenants can pin.

## Additive since 0.7.5

- Issue #10 typed provider classes — `HFRepo`, `CivitaiRepo` exported from `gen_worker`; bare `Repo()` continues to mean tensorhub. No prefix strings in user code.
- Issue #10 wire-format change — `_binding_to_wire` emits an explicit `provider:` field on each binding entry. Orch+worker prefix-sniffing path preserved as fallback for already-published manifests, but the explicit field is the canonical signal.
- Issue #9 closeouts that landed since 0.7.5.
- Class-shape decorator refactor (`@inference` / `@training` / `@dataset` / `@conversion` decorate a class with `setup()` + method handlers).
- Provider-aware `_wire_ref` helper exposed for tests.

## Publish workflow (as executed)

1. Bumped pyproject.toml to 0.7.6.
2. Regenerated `uv.lock`.
3. Built dist: `uv build` -> `dist/gen_worker-0.7.6-*`.
4. Verified wheel imports cleanly with `from gen_worker import HFRepo, CivitaiRepo, Repo`.
5. Published to PyPI (latest = 0.7.6).
6. Bumped 13 inference + 4 training endpoint pyprojects to `gen-worker>=0.7.6` / `gen-worker[torch]>=0.7.6`.
7. CHANGELOG.md updated with the 0.7.6 section.

## Out of scope

- Removing the legacy `hf:` / `cozy:` / `huggingface:` string-prefix parser. Stays in 0.7.x for back-compat with already-published endpoint manifests. Deletion is a future cycle once all endpoints have rebuilt with the new typed wire format.

## Tasks
- [x] Audit `git log v0.7.5..HEAD -- python-gen-worker/` to enumerate every behavior change since 0.7.5; summary in CHANGELOG.md.
- [x] Bumped `python-gen-worker/pyproject.toml` 0.7.5 -> 0.7.6. Refreshed `uv.lock`.
- [x] Ran test suite (`uv run pytest`); all passing before publish.
- [x] Built dist: `uv build`. Wheel + sdist produced as `dist/gen_worker-0.7.6-*`.
- [x] Sanity-checked the built wheel: `python -c "from gen_worker import HFRepo, CivitaiRepo, Repo; print('exports ok')"`.
- [x] Confirmed with user and published to PyPI. 0.7.6 is the latest on PyPI.
- [x] After publish: bumped `gen-worker>=0.7.6` (and `gen-worker[torch]>=0.7.6` for training) across all 13 inference endpoints + 4 training endpoints; uv.locks refreshed.
- [x] Wrote `python-gen-worker/CHANGELOG.md` 0.7.6 section: HFRepo/CivitaiRepo, provider wire field, class-shape decorators, `_wire_ref` helper, plus diffs since 0.7.5.

---

# #12: gen-worker-run-cli

**Completed:** yes

Add a `gen-worker run` subcommand that executes an endpoint's inference (or training) method against the local Python interpreter. Two inputs: which function to call, what payload to send. Everything else — model resolution, payload validation, context wiring — happens automatically, exactly the way the production worker does it.

Greenfield CLI; argparse, no new runtime deps. Ships on the existing 0.7.6 wheel — no version bump. Full design and rationale captured in the implementation PR / commit message.

## Tasks
- [x] pyproject.toml: `[project.scripts] gen-worker = "gen_worker.cli:main"` added; version untouched at 0.7.6.
- [x] src/gen_worker/cli/__init__.py — argparse dispatcher with main(); lazy-imports run subcommand.
- [x] src/gen_worker/cli/run.py — full pipeline: load endpoint.toml, import module, discover class methods, decode payload, resolve models, build LocalRequestContext, instantiate + setup + dispatch.
- [x] endpoint.toml loader honors --config / --module override; sys.path discovery mirrors discover.py's rules (cwd + cwd/src).
- [x] Function/method selection via _collect_class_methods + _select_function; exit 2 on zero/ambiguous with the Class.method list printed to stderr.
- [x] Payload decode: --payload XOR --payload-file; msgspec.json.decode(bytes, type=payload_type); ValidationError -> exit 2 with field path message.
- [x] apply_payload_constraints called post-decode same as worker.py.
- [x] src/gen_worker/cli/local_context.py — LocalRequestContext / LocalConversionContext / LocalDatasetContext / LocalTrainingContext via build_local_context(kind=...).
- [x] LocalRequestContextMixin overrides save_bytes / save_file -> ./.gen-worker-run/outputs/<ref>; returns Asset with local_path set.
- [x] LocalConversionContext.publish_repo_revision / materialize_blob stubbed (print to stderr, return fake response) unless --allow-publish.
- [x] --device flag stashed on GEN_WORKER_LOCAL_DEVICE env var (RequestContext has no .device setter today; tenants read torch directly).
- [x] Model resolution: cozy refs with digest hit _try_find_existing_cozy_snapshot_dir; HF refs auto-fetch via HuggingFaceHubDownloader with stderr 'model_fetch.started' / '.completed' events.
- [x] --offline: HF cache miss -> exit 3 with warm-cache pointer; cozy cache miss -> exit 3 with CAS path printed.
- [x] Override-binding support via payload._models: string shorthand AND structured {ref, tag, flavor} both accepted; rejected when binding has no .allow_override().
- [x] Dispatch-binding support: discriminator field read from decoded payload; table key -> Repo pick resolved at invoke time.
- [x] Class-shape invocation: cls(); setup(**resolved_models) with TypeError fallback to bare setup(); method(ctx, payload); shutdown() in finally block.
- [x] Generator support: yields stream to stdout as {"event":"yield","value":...}; final {"event":"result","value":{"yielded":N}} after.
- [x] SIGINT handler installed for the duration of the call; first Ctrl-C trips ctx._canceled; second within 2s os._exit(130); prior handler restored in finally.
- [x] Exit codes wired: 0/1/2/3/130 per the matrix; constants at top of run.py.
- [x] msgspec.json.encode(result) on stdout (via msgspec.to_builtins routing); --pretty switches to json.dumps(indent=2).
- [x] tests/test_cli_run.py — 18 tests covering selection (1/N/ambiguous), payload errors, _models override (string + structured + reject), --offline (HF + cozy), SIGINT, generator streaming, user exception, ctx.emit on stderr, --help fallback.
- [x] Smoke test against examples/marco-polo: `gen-worker run --payload '{"text":"marco"}'` returns `{"event":"result","value":{"response":"polo"}}` on stdout, exit 0.
- [x] docs/local-dev.md — two-input model, JSON-lines-to-stderr convention, SIGINT story, --offline story, worked examples (marco-polo, streaming, --module override).
- [x] docs/endpoint-authoring.md — `gen-worker run` is now the canonical local-test recipe (the existing python -m gen_worker.entrypoint section kept as a fallback).
- [x] inference-endpoints/qwen-image/README.md updated with `gen-worker run` smoke-test recipe alongside the production deploy section. (No existing endpoint README documented docker-compose/curl; picked the simplest payload shape.)
- [x] README.md — `## Local development` section added pointing at docs/local-dev.md with one-line install + run quickstart.

## Closeout

Closeout: greenfield CLI shipped on the existing 0.7.6 wheel (no version bump). All acceptance criteria from the spec met EXCEPT the cozy-ref auto-fetch flow: cozy refs missing from the local CAS exit 3 with a 'warm via orchestrator' pointer rather than fetching from tensorhub directly. Reason: cozy refs require presigned URLs from the orchestrator's resolve flow (ModelRefDownloader hard-requires resolved_repos_by_id context), and the spec said 'DO NOT edit src/gen_worker/worker.py or src/gen_worker/models/refs.py'. HF refs (HFRepo bindings) auto-fetch on cache miss as designed. marco-polo (no model binding) works end-to-end.

---

# #13: wire-format-bare-refs-typed-provider

**Completed:** yes

The wire format for a model reference becomes:

```json
{ "ref": "Qwen/Qwen2.5-1.5B-Instruct", "provider": "hf", "tag": "prod", "flavor": "bf16" }
```

- `ref` is BARE — never has a prefix.
- `provider` is EXPLICIT. Values: `"cozy"` (tensorhub) | `"hf"` (huggingface) | `"civitai"`.
- **Absence of `provider` in a wire payload defaults to `"cozy"`** (tensorhub).
- Prefix strings `"hf:"`, `"cozy:"`, `"civitai:"`, `"huggingface:"` appear nowhere as ref prefixes — not on the wire, not in DB rows, not in RPC bodies, not in docstrings describing the protocol.

This is a hard cut: no transitional dual-format support. SDK <0.7.7 endpoints stop resolving. The DB migration (tensorhub) updates existing rows to the new shape.

Cross-repo ordering:
1. tensorhub ships first — schema migration + new accept/reject behavior on bindings.
2. gen-orchestrator ships next — removes prefix-sniff fallback, reads bare ref + provider, sends bare ref + provider to workers.
3. python-gen-worker SDK 0.7.7 ships — emits bare ref + explicit provider; decodes bare ref + provider from orchestrator.
4. All 13 inference + 4 training endpoints rebuild against SDK 0.7.7.
5. e2e + cozyctl test fixtures updated.

Related issues filed in lockstep:
- tensorhub#278
- python-gen-worker#13
- gen-orchestrator#334
- e2e#76
- cozyctl#4


## What this repo owns

python-gen-worker is the SDK that emits + decodes the wire format. After this cut:
- `_wire_ref(binding)` returns `binding.ref` BARE (no `hf:` / `civitai:` prepend).
- `_binding_to_wire(...)` emits an explicit `provider` field on every binding entry.
- `parse_model_ref(ref, *, provider)` takes provider as a required keyword argument — no prefix-sniffing.
- All internal callers thread provider context from typed bindings.

Tenant code (`Repo` / `HFRepo` / `CivitaiRepo` classes) is unchanged. Only the wire emit/decode layer changes.

## Concrete changes

1. `src/gen_worker/api/binding.py` — `_wire_ref(binding)` returns bare `binding.ref` always. `_binding_to_wire(...)` puts the explicit `provider` field on the binding entry.

2. `src/gen_worker/models/refs.py`:
   - Delete `_strip_scheme`.
   - `parse_model_ref(raw: str, *, provider: str) -> ParsedModelRef` requires provider keyword. No prefix-routing. Parses just the ref payload (`owner/repo`, `owner/repo:tag`, `owner/repo@blake3:...`, `owner/repo#flavor`).

3. `src/gen_worker/worker.py` (~11 sites) — every `parse_model_ref(canon)` call passes explicit provider from the binding's typed context.

4. `src/gen_worker/cli/run.py` (~5 sites) — same.

5. `src/gen_worker/clone/pipeline.py` (~5 sites) — replace `f"hf:{source_ref}"` parent_checkpoint_id with bare ref + provider field on the tensorhub API call.

6. `src/gen_worker/models/hf_downloader.py` — delete the defensive `hf:` strip (lines ~55-57); after the cut, no caller passes prefixed refs.

7. `src/gen_worker/discovery/toml_manifest.py:277` — legacy prefix check; rewrite to read explicit provider from the manifest entry.

8. `tests/test_api_binding.py` — every `assert ... == "hf:..."` becomes `assert binding["ref"] == "..."` (bare) and `assert binding["provider"] == "hf"`.

9. `tests/test_cli_run.py` — update `_models` override payload examples that contain prefixes.

10. `pyproject.toml`: 0.7.6 → 0.7.7. CHANGELOG.md: new 0.7.7 section.

## `_resolved_repos_by_id` keys

Today the map is keyed by canonical prefixed strings (`"cozy:owner/repo:tag#flavor"`). After the cut, key by a `(provider, ref)` tuple or a frozen dataclass identity. Update every lookup site — these live in `worker.py` and possibly `models/ref_downloader.py`.

## Acceptance criteria

- `_wire_ref(HFRepo("Qwen/..."))` returns `"Qwen/..."` (bare).
- `_binding_to_wire(...)` emits `provider="hf"` (or `"cozy"` / `"civitai"`) as a field on every binding entry; `ref` is always bare.
- `parse_model_ref(raw, provider=...)` signature requires provider; calling without provider is a TypeError.
- No call site passes a prefixed string to `parse_model_ref`.
- 150+ tests pass after wire-format assertion updates.
- pyproject.toml is at 0.7.7; CHANGELOG documents the change.

## Out of scope

- Publishing 0.7.7 to PyPI (user-driven; separate publish issue once the cut is verified end-to-end).
- Rebuilding inference/training endpoints (user-driven after PyPI publish).

## Tasks
- [ ] `src/gen_worker/api/binding.py`: `_wire_ref` returns bare ref always; `_binding_to_wire` emits explicit `provider` field
- [ ] `src/gen_worker/models/refs.py`: delete `_strip_scheme`; rewrite `parse_model_ref(raw, *, provider)` to require explicit provider (no prefix-routing)
- [ ] `src/gen_worker/worker.py`: thread explicit provider through all ~11 `parse_model_ref` callers
- [ ] `src/gen_worker/cli/run.py`: thread explicit provider through all ~5 callers
- [ ] `src/gen_worker/clone/pipeline.py`: replace `f"hf:{source_ref}"` parent_checkpoint_id with bare ref + provider on tensorhub API call
- [ ] `src/gen_worker/models/hf_downloader.py`: delete the defensive `hf:` strip at ~lines 55-57
- [ ] `src/gen_worker/discovery/toml_manifest.py`: rewrite line ~277 prefix check to read explicit provider
- [ ] `_resolved_repos_by_id`: rekey from prefixed canonical strings to `(provider, ref)` tuples (or frozen dataclass); update every lookup site
- [ ] `tests/test_api_binding.py`: update wire-format assertions to bare ref + explicit provider field
- [ ] `tests/test_cli_run.py`: update `_models` override payload examples
- [ ] pyproject.toml: bump 0.7.6 → 0.7.7
- [ ] CHANGELOG.md: new 0.7.7 section documenting the wire-format cut
- [ ] `python -m py_compile` + `uv run pytest`: full suite passes
- [x] CLOSEOUT 2026-05-14: SDK 0.7.7. `_wire_ref(binding)` returns bare `binding.ref` for every provider; `_binding_to_wire` emits explicit `provider` field on every fixed + dispatch entry. `_resolved_repo_id(ref, ..., provider='cozy')` now takes provider explicitly and prefixes non-cozy refs with `<provider>::` (double colon, internal identity tag — NOT a wire format). Worker dispatch + fixed-binding model_id construction calls into the new signature. `tests/test_api_binding.py` updated: `_wire_ref(HFRepo(...))` now asserts bare `"Qwen/..."`, all 6 binding-to-wire assertions assert bare refs + explicit `provider`. `pyproject.toml` bumped 0.7.6 → 0.7.7; `CHANGELOG.md` documents the wire-format cut + cross-repo coordination. 150 tests pass, 2 skipped. NOTE: `_canonicalize_resolved_repos_map` in worker.py still emits `cozy:` tag-based aliases for the cross-process `resolved_repos_by_id` map; that internal identity refactor is deferred to a Phase 2 issue.
- [x] CLOSEOUT 2026-05-14 (PHASE 2): provider value renamed 'cozy' -> 'tensorhub'. Repo.PROVIDER = 'tensorhub' (was 'cozy'); HFRepo / CivitaiRepo unchanged. CozyRef class renamed -> TensorhubRef (refs.py, cozy_snapshot_v2.py, models/__init__.py re-export). ParsedModelRef.cozy attribute stayed named .cozy for now (would require updating all worker.py consumers — deferred as cosmetic, since users see the typed Repo class not the parsed shape). _binding_to_wire OMITS the `provider` field on the wire when value would be 'tensorhub' — absence is the implicit default, matching the wire contract. Tests in test_api_binding.py updated: test_binding_to_wire_tensorhub_fixed asserts `'provider' not in binding`; dispatch-table tensorhub entries assert `'provider' not in entry`. CLI override defaults to 'tensorhub' on absent provider (override does NOT inherit binding provider; consistent with absence==tensorhub contract). 150 tests pass, 2 skipped.
- [x] CLOSEOUT 2026-05-14 (PHASE 3): full hard cut. refs.py rewritten — `parse_model_ref(raw, *, provider='tensorhub')` requires explicit provider (no prefix-sniffing); `TensorhubRef.canonical()` + `HuggingFaceRef.canonical()` + new `CivitaiRef` all return bare refs; `ParsedModelRef.cozy` renamed to `.tensorhub`; added `canonical_id()` helper. All callers updated: worker.py (~7 sites including the `f"cozy:..."` alias construction in `_canonicalize_resolved_repos_map` now bare), cli/run.py (`_resolve_local_path` calls `parse_model_ref(ref, provider=provider)` directly with no canon-string construction), clone/pipeline.py (`normalize_destination_ref` rejects prefixed refs hard), discovery/toml_manifest.py (rejects all four legacy prefixes). Legacy strip-prefix shims at worker.py lines 4951-4955 + 5569-5574 + 7150-7151 deleted (no longer needed after the SDK emits bare refs). Defensive `hf:` strip in hf_downloader.py deleted. Stale docstrings cleaned up across worker.py:499, ref_downloader.py, hf_downloader.py error message. The 12 remaining `"hf:"`/`"cozy:"`/`"civitai:"` literals are all legitimate: CHANGELOG historical record, rejection lists, parent_checkpoint_id external-source encoding (internal tensorhub schema, NOT wire format — documented inline), and test fixtures echoing data back. 150 tests pass + 2 skipped.

---

# #14: Adaptive upload concurrency without operator tuning

**Completed:** yes
**Status:** completed

## Problem

Large conversion uploads need enough concurrency to keep the network busy, but the worker should not expose per-job or environment-level tuning flags for upload fan-out. Concurrency is internal policy that composes with Tensorhub capability budgets.

## Direction

The upload path now starts each file batch conservatively, records file completion time and best-effort byte size, ramps while recent throughput improves, holds when it plateaus, and backs off when an upload fails. The controller is per-call, private to the SDK, and bounded by capability-budget back-pressure.

## Constraints

- No env-configurable upload concurrency flags.
- No caller-provided concurrency or retry overrides in upload APIs.
- No global adaptive state shared across jobs.
- Always compose with `BudgetGate`; byte budgets are hard limits, throughput tuning is secondary.

## Tasks
- [x] Remove the old file-level upload concurrency constant from `_concurrent_upload.py`.
- [x] Remove caller-provided `max_workers` from `parallel_map_uploads`; callers no longer tune file-level upload fan-out.
- [x] Remove caller-provided `max_parallel` from `presigned_upload_file`; callers no longer tune part-level upload fan-out.
- [x] Remove caller-provided upload retry/backoff parameters from `presigned_upload_file`; retry policy is internal to the transport.
- [x] Keep `BudgetGate` as the hard byte-budget back-pressure mechanism for concurrent uploads.
- [x] Add a per-call adaptive scheduler that records elapsed time, best-effort byte size, success, and failure.
- [x] Start file fan-out conservatively, ramp while throughput improves, hold on plateau, and reduce on upload errors.
- [x] Cap adaptive file fan-out with an internal private sanity limit.
- [x] Keep adaptive state local to one upload batch; no global state is shared across jobs.

---

# #15: Consolidate endpoint discovery: one walker shared by build-time and runtime

**Completed:** yes
**Status:** completed

## Problem

Build-time discovery and runtime worker discovery walked endpoint classes through separate code paths. That duplication caused a conversion endpoint outage: build-time discovery found classes re-exported from submodules, while runtime discovery rejected them and booted with zero handlers.

## Direction

Use one shared discovery walker that finds endpoint classes, decorated function methods, submodule re-exports, and package submodules once. Build-time manifest generation and runtime handler registration consume the same `FoundEndpointClass` results so they cannot diverge by construction.

## Acceptance

- `gen_worker.discovery.walk.find_endpoint_classes(...)` is the single source of truth.
- `discover_functions()` and `Worker._discover_and_register_functions()` both call the shared walker.
- Temporary manifest-vs-runtime verification and duplicated fallback walkers are gone.

## Tasks
- [x] Add `gen_worker/discovery/walk.py` with `FoundEndpointClass`, `FoundFunctionMethod`, and `find_endpoint_classes(module_names)`.
- [x] Move the submodule re-export prefix check into the shared walker.
- [x] Move package submodule walking/deduplication into the shared walker instead of keeping it as a runtime-only fallback.
- [x] Preserve class identity deduplication so the same endpoint re-exported from `__init__.py` and found in a submodule registers once.
- [x] Include decorated method discovery in the shared result so build-time and runtime see the same function list.
- [x] Refactor `gen_worker/discovery/discover.py::discover_functions` to serialize results from the shared walker without changing endpoint.lock schema.
- [x] Refactor `Worker._discover_and_register_functions` to instantiate/register classes from the shared walker.
- [x] Delete the stale runtime-only class discovery helpers and manifest coverage verifier; the shared walker owns those behaviors.

---

# #16: Production-grade large-file upload and download path

**Completed:** yes
**Status:** completed

## Problem

Large conversion jobs move 5-10 GB tensor shards through two fixed data-plane paths: Hugging Face/source downloads into the worker, then multipart uploads into Tensorhub/R2. On 2026-05-16 the FLUX.2-klein-4B clone failed repeatedly during the R2 multipart PUT phase with `SSLV3_ALERT_BAD_RECORD_MAC`.

## Direction

Keep the upload contract exactly as designed: the worker asks Tensorhub for a multipart upload session using its worker capability token, Tensorhub returns presigned S3/R2 part URLs, and the worker uploads bytes directly to those URLs. No direct R2 credentials, no boto3 credential handoff, no alternate control plane.

Make that presigned-URL data plane robust: fresh TLS state per part attempt, explicit retry classification, bounded memory, internal adaptive concurrency, and no caller-tuned concurrency flags.

## Acceptance

- Upload callers have no operational concurrency or retry knobs to tune per job.
- Retryable transport failures retry with fresh TLS state; terminal auth/client failures surface immediately.
- The download wrappers defer to Hugging Face/Civitai/client libraries where practical and do not duplicate fragile transport logic.
- Tenant-facing upload APIs stay unchanged.
- The only upload flow is Tensorhub capability token -> Tensorhub presigned URLs -> direct S3/R2 PUT.

## Tasks
- [x] Audit the upload/download data paths used by clone and conversion jobs: `presigned_upload.py`, `_stream.py`, `_upload_session.py`, `_concurrent_upload.py`, `clone/pipeline.py`, `clone/_shared.py`, `conversion/ingest.py`, and model/HF download wrappers.
- [x] Identify the production failure as the long-lived R2 multipart PUT data plane, not Tensorhub control-plane JSON POSTs.
- [x] Replace the data-plane `requests.put` loop with `gen_worker._upload_transport.upload_part_to_presigned_url` using fresh `urllib3.PoolManager` state for every part attempt.
- [x] Reopen the source file per retry with a bounded file reader so failed partial PUTs restart from the correct byte offset without unbounded memory growth.
- [x] Classify TLS, connection, timeout, 429, and 5xx failures as retryable; classify non-429 4xx responses and malformed successful responses as terminal.
- [x] Keep create/complete/abort control-plane calls on ordinary JSON requests while isolating the large byte stream transport.
- [x] Remove caller-facing upload retry/concurrency overrides from `presigned_upload_file`; retries and part concurrency are now internal policy.
- [x] Remove the `parallel_map_uploads(..., max_workers=...)` override path so file-level upload fan-out is internal policy.
- [x] Preserve capability-budget back-pressure via `BudgetGate` so concurrent uploads do not exceed Tensorhub-issued byte budgets.
- [x] Replace fixed file fan-out with the per-call adaptive scheduler from issue #14.
- [x] Scope issue #16 to the existing Tensorhub capability-token -> presigned URL -> direct S3/R2 PUT flow; no direct R2 credentials or boto3 handoff.

---

# #17: HF-provider refs from supported_repo_refs must route to HuggingFace downloader, not tensorhub

**Completed:** yes
**Status:** completed

## Problem

When the orchestrator sends an `EndpointConfig.supported_repo_refs` list to a worker, every ref in the list is parsed by `_process_release_config_async_wrapper` -> `ModelManager.process_supported_models_config` -> `ModelRefDownloader.download(model_ref)` -> `parse_model_ref(model_ref)` *with no `provider=` kwarg*. `parse_model_ref` defaults to `provider="tensorhub"`, so every ref is treated as a tensorhub ref. For an HF-provider binding like `black-forest-labs/FLUX.2-klein-base-4B#bf16`, the downloader then runs the tensorhub branch in `ref_downloader.py:_download_async`, fails the `resolved_repos_by_id` lookup, and raises `RuntimeError("tensorhub ref {canonical!r} not in resolved_repos_by_id — orchestrator must pre-resolve before dispatching the job")`.

Observed live on 2026-05-16 against build `87d8c87a` / release `98f69bb4f8cf3d8f4c601170` on RunPod. The endpoint.lock baked at build time *does* carry `provider: hf` on every binding (we shipped that fix in 0.7.11), so the worker has the routing info — the runtime path is just throwing it away.

The error message itself is a half-implementation: it says "orchestrator must pre-resolve" but the canonical answer for an HF binding is "worker downloads from HF directly", same as the `parsed.provider == "hf"` branch in `_download_async`.

## Direction

Thread the binding's `provider` from the endpoint.lock manifest through to `parse_model_ref` everywhere a bare ref string is decoded.

The endpoint.lock manifest already has the provider per binding (`function_param_bindings` entries with `provider: hf` / `provider: tensorhub` / `provider: civitai`). The worker reads it at startup. Build an index `{ canonical_ref_string -> provider }` from the manifest's bindings, and consult it on every parse:

1. In `_process_release_config_async_wrapper`, before invoking the model manager, build the provider index from the loaded manifest (the same one `_parse_manifest_model_mapping` reads).
2. Change `ModelRefDownloader.download` to consult the index and pass the right `provider=` kwarg into `parse_model_ref`.
3. Same treatment for the other `parse_model_ref` call sites that take a bare ref out of a runtime payload: `worker.py:4498` (resolved-map canonicalization), `worker.py:4786` (snapshot dir lookup), `worker.py:8098/8152/8439` (job-time refs), `run_metrics_v1.py:373`, `request_context/_helpers.py:370`.
4. When the index doesn't have a ref (e.g. an invoker-supplied override that wasn't in the build-time manifest), keep `provider="tensorhub"` as the default — that matches the wire-format contract that tensorhub is the default provider.

Long term, the wire format should carry provider per ref so the worker doesn't need to maintain a side index. Track that as a follow-up (orchestrator gRPC schema change).

## Acceptance

- A worker booted from an image whose endpoint.lock has `provider: hf` bindings, given `supported_repo_refs = ["black-forest-labs/FLUX.2-klein-base-4B#bf16", "black-forest-labs/FLUX.2-klein-4B#bf16"]` by the orchestrator, downloads both refs via `HuggingFaceHubDownloader` (which uses `HF_TOKEN` from the pod env).
- The `tensorhub ref ... not in resolved_repos_by_id` error fires only for genuine tensorhub bindings that the orchestrator failed to pre-resolve — never for hf/civitai bindings.
- No new env vars or knobs; the manifest is the single source of truth.

## Related

- gen-orchestrator companion issue (same flow, orchestrator side): the orchestrator must also stop calling tensorhub's `/internal/v1/repos/resolve` for hf/civitai bindings. Both halves were only half-implemented.
- Live evidence: gen-orchestrator log `endpoint_ref_availability_changed ... ref=black-forest-labs/FLUX.2-klein-base-4B#int8 from=resolved to=invalid_request reason=invalid_request detail="invalid owner"` (orchestrator side) followed by `worker ... load model result: ... err=tensorhub ref 'black-forest-labs/FLUX.2-klein-base-4B:latest#bf16' not in resolved_repos_by_id` (worker side).

## Tasks
- [x] Build a `{canonical_ref -> provider}` index from the loaded endpoint.lock manifest at worker startup.
- [x] Thread provider into `ModelRefDownloader.download` and pass to `parse_model_ref`.
- [x] Audit all `parse_model_ref(...)` call sites in `worker.py`, `run_metrics_v1.py`, `request_context/_helpers.py`; pass the looked-up provider where available, keep `tensorhub` as the default fallback.
- [x] Integration test: simulate a worker registration with `supported_repo_refs` listing an `hf:` provider binding's bare ref; assert HuggingFaceHubDownloader is invoked, not the tensorhub `resolved_repos_by_id` lookup.
- [x] Live re-test the FLUX.2-klein-4b RunPod flow once the orchestrator side ships (downloads complete, first inference succeeds).

---

# #18: Worker: read provider from resolved_models override + enforce safetensors-only on override downloads

**Completed:** yes
**Status:** completed

## Problem

Tenant declares `flux = HFRepo(...).allow_override(Flux2KleinPipeline)`. Invoker overrides with `_models = {"flux": {"ref": "some-other-org/their-flux", "provider": "hf"}}`. Orchestrator validates + stamps `resolved_models["flux"]` on the request. Worker reads it.

Today the worker reads only `{ref, tag, flavor}` from the stamped entry (`worker.py:8785`) and at `worker.py:8831` calls `_resolved_repo_id(ref, flavor=..., tag=...)` *without* `provider=`, which silently defaults to `tensorhub`. So the override always resolves as a tensorhub ref, and the worker fails to download because the ref doesn't exist on tensorhub.

Companion: gen-orchestrator #358 ships the provider field on the wire (proto + stamp). This issue consumes it.

## Constraints from the user

- Cross-provider overrides are allowed. An `HFRepo` binding can be overridden with a tensorhub   or civitai ref, and vice versa. The override's provider is on the wire — use it.
- **Never download pickle files** (.bin / .pt / .ckpt). Only safetensors or flashpack. The   orchestrator validates this pre-dispatch, but the worker should belt-and-braces it on the   download path too — if a stamped override somehow names a pickle file, refuse and fail   with `unsafe_file_format` error.
- Pipeline class stays the same. The binding's declared pipeline class is the load target;   the override only swaps weights. Worker behavior here doesn't change — the existing   `_looks_like_ref_compatibility_surprise` classifier already handles weight/class   mismatches at load time.

## Direction

### 1. Extract provider on the wire

`Worker._resolved_models_for_request` (`worker.py:8760`) iterates `resolved_models` entries and pulls `ref`, `tag`, `flavor`. Also pull `provider` (from the protobuf `ResolvedModel` field or the dict-shape test fixture). Default to `"tensorhub"` when absent (for back-compat with older orchestrators that haven't shipped #358).

### 2. Pass provider into resolution

`_resolve_model_id_for_injection` (`worker.py:8801`) override branch (line 8831): pass `provider=override["provider"]` into `_resolved_repo_id`. Verify `_resolved_repo_id` and `canonical_id` already take `provider` (they do; see `models/refs.py:145`). The model id key in the per-process cache discriminates by provider so cross-provider overrides don't collide.

### 3. Thread provider into the downloader's contextvar

The worker's per-request setup needs to add the override's `(canonical_ref → provider)` to the same `_provider_by_ref` index introduced in #17. That way when `ModelRefDownloader.download` runs, the lookup hits the override's provider (not the binding-time provider, which would be wrong for cross-provider overrides). Update the six `set_resolved_repos_by_id` call sites — they all need the override's provider folded into the index for the duration of the request.

### 4. Safetensors-only enforcement on override downloads

After download, inspect the snapshot's primary weight file. If the largest file (or the file named by the loader's expected pattern) ends in `.bin` / `.pt` / `.ckpt`, raise an error of type `UnsafeFileFormat` with detail naming the file. The error should be classified terminal (no retry — pickle never becomes safe between retries). The classifier check should run *only* for refs that came from `resolved_models` (override path), not for binding-default refs — those go through the build-time validator which already enforces this. (Use the same `_current_payload_ref_keys` signal as `_looks_like_ref_compatibility_surprise`.)

## Acceptance

- An HFRepo binding overridden with `{"ref": "other-org/their-flux", "provider": "hf"}`   downloads from HF (not tensorhub), loads into the binding's declared pipeline class, runs.
- An HFRepo binding overridden with `{"ref": "acme/cozy-mirror", "provider": "tensorhub"}`   downloads via the orchestrator-stamped `resolved_repos_by_id` URLs (existing tensorhub path).
- A stamped override pointing at a snapshot whose primary weight is `.bin` fails fast with   `UnsafeFileFormat: refusing to load <path>; safetensors/flashpack only`. (Defense-in-depth   if orchestrator #358 validation slips.)
- Existing tensorhub-only override tests still pass; binding-default flows unchanged.
- New tests cover: (a) provider extraction from stamped entry, (b) cross-provider override   (hf binding + tensorhub override), (c) safetensors gate trips on .bin / .pt / .ckpt.

## Related

- gen-orchestrator #358 (companion: provider on the wire + pre-dispatch validation).
- gen-worker #17 (provider index from endpoint.lock — extend with per-request override entries).
- gen-orchestrator #357 (circuit breaker if the safetensors gate trips repeatedly).

## Tasks
- [x] Extract `provider` from `ResolvedModel` in `_resolved_models_for_request`.
- [x] Pass `provider=` to `_resolved_repo_id` in `_resolve_model_id_for_injection` override branch.
- [x] Thread override's `(ref → provider)` into `_provider_by_ref` for the request's duration.
- [x] Add `UnsafeFileFormat` error class + post-download primary-weight-file check (override path only).
- [x] Unit test: provider extraction from stamped entry (proto + dict shapes).
- [x] Unit test: cross-provider override (HFRepo binding + tensorhub override + reverse).
- [x] Unit test: safetensors gate trips on .bin / .pt / .ckpt; allows .safetensors / .flashpack.
- [x] Integration test: end-to-end override flow with a mocked HF downloader (no real network).

---

# #20: Pipeline load path: opaque failures, dtype API mismatch, and VRAM accounting for diffusers from_pretrained

**Completed:** yes
**Status:** completed

## Problem

Live testing FLUX.2-klein-4b (release `bca0579831945209d9de2948`) at gen-worker 0.7.21 on RunPod RTX 4090 reaches the actual model load successfully end-to-end up to `from_pretrained`, then fails with the opaque string:

```
load model result: model=black-forest-labs/FLUX.2-klein-base-4B#bf16 success=false
err=MMM.load_model_into_vram failed for 'black-forest-labs/FLUX.2-klein-base-4B#bf16'.
```

The orchestrator + worker can't see the actual exception. Investigation of every layer below `_loader.load(...)` shows everything in the chain is correct except this opacity. The investigation also surfaced two adjacent design issues worth fixing as part of the same release.

## Investigation (what was traced and what's actually fine)

1. **HF download plan** — dry-ran `plan_diffusers_download` + `finalize_diffusers_download` against the live FLUX.2-klein-base-4B repo file list (25 files). Selected file set is **correct**:

   - `model_index.json` ✓
   - `scheduler/scheduler_config.json` ✓
   - `text_encoder/`: both shards (4.7GB + 2.9GB), `model.safetensors.index.json`, `config.json`, `generation_config.json` ✓
   - `tokenizer/`: all 7 tokenizer files ✓
   - `transformer/`: `config.json` + `diffusion_pytorch_model.safetensors` (7.4GB single file) ✓
   - `vae/`: `config.json` + `diffusion_pytorch_model.safetensors` (160MB) ✓
   - Correctly skipped: top-level `flux-2-klein-base-4b.safetensors` (single-file checkpoint variant), `.jpg` cover art, README/LICENSE/.gitattributes.

2. **HF cache layout** — `~/.cache/huggingface/hub/models--<owner>--<repo>/{blobs,refs,snapshots/<sha>/}`. The snapshot dir mirrors the repo: top-level `model_index.json` + per-component subdirs with their own `config.json` + safetensors. After the 0.7.21 partial-cache fix, the snapshot dir is fully populated.

3. **Path supplied to diffusers** — `ModelRefDownloader.download(model_id, dest_dir)` for HF refs ignores `dest_dir` and returns the HF cache snapshot dir path (because `HuggingFaceHubDownloader` uses `huggingface_hub`'s own cache). `PipelineLoader.load(model_id, model_path=<HF snapshot dir>)` passes that path straight to `pipeline_class.from_pretrained(path, torch_dtype=..., ...)`. **This is correct.** Diffusers expects exactly this directory shape.

4. **Class resolution** — `model_index.json` declares `_class_name: Flux2KleinPipeline` with components `Flux2Transformer2DModel`, `AutoencoderKLFlux2`, `Qwen3ForCausalLM`, `Qwen2TokenizerFast`, `FlowMatchEulerDiscreteScheduler`. Verified inside the built image (`tensorhub/root-endpoint-images:flux.2-klein-4b-bca0579831945209d9de2948-linux-cuda-12.8-x86`): every class imports cleanly (diffusers 0.38.0.dev0 @ commit 303c1d8, transformers 5.8.1, torch 2.11.0+cu128). **No import failure.**

5. **Dtype handling on the HF path** — `PipelineLoader.load(model_id="black-forest-labs/FLUX.2-klein-base-4B#bf16", ...)` creates a fresh `PipelineConfig(model_path=path)` with `dtype=None`. The `#bf16` selector is **thrown away** on the HF code path. By coincidence `get_torch_dtype(None, ...)` falls back to `torch.bfloat16` so bf16 works; for `#fp8` / `#nf4` / `#nvfp4` the fallback would silently load wrong precision. **The `#flavor` convention is a tensorhub-side concept (`Repo("owner/repo:tag#flavor")`) and should not leak into the HFRepo surface — see fix 2 below.**

## Three coupled fixes

### Fix 1: propagate the actual load exception

Today `model_manager.py:load_model_into_vram` catches every exception, logs it via `logger.exception(...)` (only the worker's stderr sees it), and returns `bool` `False`. Worker.py:5306 then synthesizes a generic error string with no detail. Every load failure is opaque from the orchestrator side.

Stash the last exception's `repr` + `traceback.format_exc()` on the `DiffusersModelManager` instance (`_last_load_error`, `_last_load_traceback`). In `worker.py` when `success=False`, include those in `LoadModelResult.error_message` so the orchestrator + scheduler logs + #357 broken-release tracker all see the real type + message + traceback.

Scope: ~30 lines across `pipeline/model_manager.py` and `worker.py`. Zero contract change for third-party `ModelManagementInterface` implementations (they just won't populate the new fields).

### Fix 2: separate HFRepo and Repo APIs along their providers' conventions

**Per the user (tenant-API design):** `Repo()` is the tensorhub binding constructor and naturally takes `Repo("owner/repo:tag#flavor")` or `Repo(name=..., tag=..., flavor=...)`. `HFRepo()` is the HuggingFace binding constructor and should use HuggingFace's API surface — **no `flavor`** (HF doesn't have flavor as a ref selector); instead `revision` (git rev), `subfolder`, and `dtype` (which gets passed to `from_pretrained(torch_dtype=...)` not encoded into the ref string).

Today both `Repo` and `HFRepo` share a `.flavor(...)` chainable. That's wrong — it bakes tensorhub-shaped identifiers into the HF surface, then those identifiers leak through `_resolved_repo_id` → `model_id="owner/repo#bf16"` → confuse `PipelineLoader.load` (which strips the `#bf16` and falls back to a default dtype).

Redesign:
- `Repo("owner/repo[:tag][#flavor]")` — unchanged. Tensorhub native.
- `HFRepo("owner/repo", revision=None, dtype=None)` — drop `.flavor()`. Add `.dtype("bf16"|"fp16"|"fp32")` chainable that sets the load-time torch_dtype, NOT a ref selector.
- Wire format: HF refs travel as bare `owner/repo[@revision]` strings. The dtype rides separately on the binding row (alongside `allow_override`, `pipeline_classes`).
- `PipelineLoader.load` consumes `config.dtype` (set from binding's dtype field) and ignores any `#flavor` in the model_id for HF refs.
- Migration: existing endpoints using `.flavor("bf16")` on HFRepo get a deprecation shim that maps `.flavor(x)` → `.dtype(x)` for one release, with a warning. Drop in 0.8.x.

### Fix 3: pre-load VRAM accounting that matches reality

`PipelineLoader.load:1140` enables CPU offload only when `estimate_model_size_gb(path) > self._max_vram_gb`. For FLUX.2-klein-base-4B:

- disk size: 7.4GB transformer + 7.5GB text encoder + 0.16GB vae ≈ 15GB
- effective `_max_vram_gb` on RTX 4090 (24GB total - safety margin ~4GB) ≈ 20GB
- `15 < 20` → offload **not** enabled
- Actual `from_pretrained` peak VRAM ≈ 15GB weights + ~6GB load-time transient (CPU→GPU copy buffers, optimizer state alloc, framework overhead). Easy to push to 22-24GB → OOM kills the process before any offload can be applied at request time.

Two-part fix:
  (a) Loader's auto-offload trigger should account for peak-vs-resident overhead: enable offload when `estimate_model_size_gb(path) * 1.4 > _max_vram_gb` (heuristic for from_pretrained overhead), and ALWAYS enable sequential offload when `estimate >= _max_vram_gb * 0.7`. Numbers calibrated from a small benchmark on representative models.
  (b) Apply `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()` **immediately after** `from_pretrained` returns, before the pipeline is handed to the endpoint's `setup()`. Today the FLUX endpoint code calls `apply_low_vram_config(pipeline, mode="sequential")` inside the function (request time), which is too late if `from_pretrained` already OOM'd.

## Acceptance

- Fix 1: any `from_pretrained` failure that previously logged as `MMM.load_model_into_vram failed for X` now includes the underlying exception type + message + condensed traceback in `LoadModelResult.error_message`. Verify via a unit test that constructs a manager, stubs `_loader.load` to raise a specific exception, and asserts the message is propagated.
- Fix 2: `HFRepo("acme/x").dtype("bf16")` works; `HFRepo("acme/x").flavor("bf16")` emits a DeprecationWarning and behaves the same for one release. `Repo("acme/y:prod#bf16")` unchanged. Wire format: HF refs never carry `#flavor`.
- Fix 3: live FLUX.2-klein-base-4B (bf16) on RTX 4090 (24GB) loads + runs first inference without OOM. CPU offload enabled automatically; no endpoint-code change required.

## Out of scope (file separately if needed)

- The orchestrator's runaway autoscaler when load_model fails fast in a loop (see gen-orchestrator #357 which already handles this for >=3 fast failures across 2 pods).
- Single-file checkpoint loading via top-level `flux-2-klein-base-4b.safetensors` (the planner correctly skips it; we don't need it for the diffusers component path).

## Tasks
- [x] Fix 1: capture last load exception in DiffusersModelManager._last_load_{error,traceback}; worker.py forwards into LoadModelResult.error_message.
- [x] Fix 1 test: stub _loader.load to raise; assert propagation through to error_message.
- [x] Fix 2: HFRepo drops .flavor(), adds .dtype(); wire format change docs.
- [x] Fix 2: deprecation shim mapping .flavor() -> .dtype() with warning for one release.
- [x] Fix 2: PipelineLoader.load consumes config.dtype for HF refs; ignores any stray #flavor in HF model_id.
- [x] Fix 2 tests: HFRepo().dtype() reaches torch_dtype; #flavor on HF model_id is ignored.
- [x] Fix 3a: auto-offload heuristic accounts for from_pretrained peak overhead (~1.4x disk size).
- [x] Fix 3b: apply enable_*_cpu_offload immediately post-from_pretrained, not at request time.
- [x] Fix 3 benchmark: measure peak VRAM for FLUX.2-klein-base-4B / FLUX.2-klein-base-9B / smaller SDXL load on representative GPUs to calibrate the heuristic constants.
- [x] Live regression: FLUX.2-klein-base-4B bf16 on RTX 4090 24GB completes first inference end-to-end.

---

# #22: Clarify Asset/Tensors media typing and remove built-in LoraSpec

**Completed:** yes
**Status:** completed

## Goal

Clean up the SDK payload/model-artifact contract so file/media inputs, model-weight inputs, and LoRA/model-adapter overlays are distinct. This supports upcoming moderation discovery and gives LoRAs a first-class platform surface instead of forcing every endpoint function to invent its own adapter payload plumbing.

## Decisions

1. LoRA/model adapter bytes are tensor/model-weight artifacts, not generic user media. Any endpoint-owned adapter structs should point at `Tensors`, not `Asset`.
2. The SDK should expose specific asset/media types for common content classes instead of using bare `Asset` for every file-like thing.
3. `LoraSpec` should be removed from the core SDK public API. Endpoint authors can define their own adapter structs if they need an escape hatch.
4. The preferred long-term LoRA surface is first-class adapter overlays on model bindings, integrated with the existing custom model override mechanism.

## Target type split

- `Asset`: generic file pointer retained for broad compatibility and non-media files.
- `Tensors`: checkpoint/model-weight artifact pointer; use for LoRAs, PEFT adapters, embeddings, GGUF/safetensors/model shards, and similar model artifacts.
- `ImageAsset`, `VideoAsset`, `AudioAsset`, and possibly `MediaAsset`: semantic media pointer types that still serialize like `Asset` but tell discovery/moderation what the payload field means.
- Endpoint-owned adapter structs: application structs, not built-in SDK types. Useful for custom runtimes or experiments, but not the main platform product surface.

## First-class adapter overlays on model bindings

We already let endpoints opt into custom model overrides through model bindings. LoRAs should extend that same model-binding contract: an endpoint can declare that a binding is adapter-capable, and invocations can provide regular model overrides plus LoRA overlays for that binding.

Example declaration shape:

```python
@inference(
    models={
        "pipeline": HFRepo("black-forest-labs/FLUX.2-klein-4B")
            .dtype("bf16")
            .allow_override(Flux2KleinPipeline)
            .allow_lora()
    }
)
class Generate:
    ...
```

The opt-in should stay intentionally small. `.allow_lora()` means: this binding's injected runtime object may receive compatible LoRA overlays. The SDK/worker should infer compatibility from the runtime object, the pipeline class, and LoRA artifact metadata rather than requiring endpoint authors to repeat `adapter_type`, `family`, or `max_adapters` knobs up front. If we later need operational caps, use platform defaults first and add optional overrides only when there is a measured need.

Example invocation shape:

```json
{
  "prompt": "portrait in embroidered jacket",
  "_models": {
    "pipeline": {
      "ref": "black-forest-labs/FLUX.2-klein-4B",
      "provider": "hf",
      "loras": [
        {
          "ref": "alice/embroidered-style-lora:prod",
          "weight": 0.8
        }
      ]
    }
  }
}
```

In this model, the worker injects `pipeline` already adapted for the current request. The endpoint handler does not need to mention LoRAs. The overlay entry should stay minimal: `ref` is required, `weight` is optional and defaults to `1.0`, and `provider` is optional only when the ref cannot be resolved by the default Tensorhub provider rules. Worker/runtime adapter names are generated internally and are not part of the public request shape.

The platform can validate adapter format and compatibility before dispatch, materialize adapter tensors, apply/unapply adapters consistently, and track billing/lineage/caching in one place.

This should still be opt-in per binding. Not every runtime supports LoRAs, and support is runtime-specific:

- Diffusers: `load_lora_weights`, `set_adapters`, `unload_lora_weights`.
- PEFT/Transformers: adapter APIs differ.
- vLLM/SGLang: LoRA serving has adapter registration/runtime constraints.
- Custom pipelines may not support adapters at all.

So the SDK should discover and expose adapter-capable bindings explicitly rather than treating every function as LoRA-compatible. Compatibility validation should be runtime-driven: for Diffusers, inspect the pipeline/adapter target modules and any Tensorhub lineage metadata; reject hard incompatibilities, but do not force endpoint authors to hand-maintain family strings that can drift.

## Prompt marker types

Add prompt-role marker types as lightweight annotated strings:

```python
PositivePrompt = Annotated[str, PromptRole("positive")]
NegativePrompt = Annotated[str, PromptRole("negative")]
```

Endpoint payloads can then stay self-describing:

```python
class GenerateInput(msgspec.Struct):
    prompt: PositivePrompt
    negative_prompt: NegativePrompt | None = None
    image: ImageAsset | None = None
```

Discovery should read type hints with extras, compile positive/negative prompt fields into endpoint.lock metadata, and validate that the marked fields serialize as strings. Endpoint authors specify moderation dimensions by using prompt marker types and media asset types in their payload structs, not by declaring a separate moderation decorator.

## Moderation relationship

Moderation discovery should infer prompt fields from `PositivePrompt` / `NegativePrompt` marker types and candidate input media from the specific media asset types. There should be no moderation decorator path for the standard endpoint authoring surface, and generic `Asset` should not be auto-moderated. Generic `Asset` is too broad: it may be a PDF, CSV, zip, safetensors file, mask, audio clip, or arbitrary bytes. Media-specific types give endpoint authors a clean way to mark which payload files are image/video/audio media.

`Tensors`, model bindings, and LoRA overlays must not be treated as media moderation inputs.

## Compatibility stance

Prefer a hard cut if the current `LoraSpec` footprint is still limited. It is currently exported by `gen_worker` / `gen_worker.api` and used by the `load_loras` helper, but endpoint code can define the same product-specific shape itself. If a compatibility bridge is needed for one release, keep it short-lived and do not let `LoraSpec` remain the blessed long-term type.

If `load_loras` remains as a helper, it should become structural: it can accept endpoint-defined objects with `tensors` and optional `weight`, or a helper-specific protocol. It should not require importing a core SDK `LoraSpec` type. But the preferred product path is runtime-managed adapter overlays on model bindings, not endpoint payload helper calls.

## Concrete validation: FLUX.2 Klein 4B LoRA inference

Use the existing `~/cozy/inference-endpoints/flux.2-klein-4b` endpoint as the concrete consumer for this SDK cleanup. The endpoint should support request-time FLUX.2 Klein LoRAs so we can see how LoRAs should be passed into custom endpoint functions and how they should be packed into the inference pipeline.

Primary probe: first-class platform-managed overlays on the existing `pipeline` binding using a bare `.allow_lora()` opt-in. The endpoint function should stay unchanged if this works: the worker applies compatible LoRA overlays to the injected `Flux2KleinPipeline` before calling the handler, then unloads them after the request.

Fallback/probe shape if platform-managed overlay needs local mechanics first:

```python
class FluxKleinLora(msgspec.Struct):
    tensors: Tensors
    weight: float = 1.0
    adapter_name: str | None = None

class GenerateWithLoraInput(GenerateInput):
    loras: list[FluxKleinLora] = msgspec.field(default_factory=list)
```

Start with the FLUX.2 Klein 4B bf16/base pipeline because quantized pipelines and accelerate offload can complicate adapter loading. The probe should validate materialization, safetensors adapter loading, adapter weights, unload-on-finally behavior, and a real LoRA-vs-no-LoRA generation comparison.

## Tasks

## Tasks
- [x] Add prompt marker types as annotated strings: `PositivePrompt` and `NegativePrompt`.
- [x] Extend discovery to read `PositivePrompt` / `NegativePrompt` type hints with extras and compile prompt moderation dimensions into endpoint.lock metadata.
- [x] Remove the planned moderation decorator/field-override path; endpoint authors specify moderation dimensions through typed prompt fields and media asset fields only.
- [x] Add media-specific payload types: `ImageAsset`, `VideoAsset`, `AudioAsset`, and decide whether a broad `MediaAsset` alias/type is useful.
- [x] Ensure media-specific asset types preserve the existing `Asset` wire shape and materialization behavior.
- [x] Update discovery/moderation planning so `PositivePrompt` / `NegativePrompt` and media-specific asset types are collected as moderation dimensions, while generic `Asset` is not auto-moderated.
- [x] Remove built-in `LoraSpec` from `gen_worker.api.types` and public exports, or add a short-lived removal stub if release compatibility requires it.
- [x] If `load_loras` remains, make it structural over endpoint-defined objects using `.tensors.local_path` and optional `.weight`, not core SDK `LoraSpec`.
- [x] Add a minimal model-binding API for adapter-capable bindings, e.g. bare `.allow_lora()` on `Repo`/`HFRepo` bindings, without requiring endpoint-authored adapter_type/family/max-count parameters.
- [x] Extend discovery/endpoint.lock metadata to record which bindings are adapter-capable via `.allow_lora()`.
- [x] Extend the model override invocation contract so `_models.<binding>` can carry `loras[]` overlays with minimal ref/provider/weight metadata, with only ref required.
- [x] Add validation so LoRA overlays are accepted only for bindings that opted into adapter support; infer compatibility from runtime/pipeline capabilities, LoRA artifact metadata, and target-module checks rather than endpoint-maintained family strings.
- [x] Implement worker-side Diffusers adapter overlay application for supported bindings: materialize LoRA tensors, load/apply weights before handler invocation, and unload in `finally`.
- [x] Ensure worker-side adapter names are generated internally and are not exposed in the public LoRA overlay request shape.
- [x] Ensure adapter overlays are serialized with the relevant per-pipeline lock and cannot leak across requests.
- [x] In `~/cozy/inference-endpoints/flux.2-klein-4b`, validate the platform-managed LoRA overlay path on the existing bf16/base `pipeline` binding without requiring endpoint handler code to process LoRAs.
- [x] If platform-managed overlays need a stepping stone, add an endpoint-owned `FluxKleinLora`/`GenerateWithLoraInput` probe using `Tensors`, then feed the mechanics back into the worker-managed path.
- [x] Document the supported FLUX.2 Klein LoRA format and minimal example `_models.pipeline.loras[]` request payload in `flux.2-klein-4b/README.md`, emphasizing that endpoint code only uses `.allow_lora()`.
- [x] Add tests proving prompt marker discovery, media asset type discovery, generic `Asset` non-discovery, `Tensors` non-discovery, bare `.allow_lora()` binding metadata, runtime compatibility validation, invalid overlay rejection, and request cleanup after LoRA application.

---

# #23: Materialize approved remote media URLs for typed asset inputs

**Completed:** yes
**Status:** completed

## Goal

Support remote HTTP(S) URL materialization for typed media input assets. gen-orchestrator issue #367 owns typed-field URL acceptance and pre-dispatch approval; python-gen-worker owns turning the approved URL Asset into a local file before endpoint code runs.

The worker already has a generic `Asset.ref` URL downloader path. This issue hardens and aligns that path with the new typed media contract instead of relying on it as an accidental fallback.

## Contract

Endpoint authors should still receive normal media asset objects with `local_path` populated:

```python
class GenerateInput(msgspec.Struct):
    prompt: PositivePrompt
    image: ImageAsset | None = None
```

If the invoker supplied `image` as a remote URL and gen-orchestrator approved it, the worker downloads that URL into the request input directory, sets `local_path`, and preserves/refines metadata such as MIME, size, and hash.

## Worker responsibilities

- Treat URL-backed `Asset` / media-specific asset inputs as first-class materializable inputs.
- Keep SSRF protection in the worker as defense in depth, even though gen-orchestrator preflights the URL.
- Re-check every redirect target, not just the initial URL.
- Enforce the orchestrator-provided byte/type/dimension limits while streaming the download.
- Do not load the whole file into memory; stream to a temp file, hash while streaming, then atomically move into the per-request input directory/cache.
- Verify content type from response headers and sniffed bytes; reject mismatches with typed validation errors.
- For image assets, verify dimensions/pixel count after download before exposing the local path to tenant code.
- Cache by URL plus auth token hash only when safe; do not leak private-token downloads across request/auth contexts.
- Surface clear `ValidationError` / `InputTooLargeError` style errors so gen-orchestrator can map worker failures cleanly if a URL changes between approval and download.

## Type relationship

This builds on issue #22. `ImageAsset`, `VideoAsset`, and `AudioAsset` should preserve the `Asset` wire shape, so the materializer can recurse through them the same way it handles `Asset` today. Generic `Asset` may keep URL materialization for compatibility, but typed media assets are the intended public path for remote image/video/audio URLs.

## Non-goals

- No moderation decisions in python-gen-worker.
- No URL acceptance for arbitrary strings; discovery/orchestrator decide which fields are media assets.
- No Tensorhub user-media import in the worker for v1. If we later want URL inputs persisted as owner uploads, that should be an explicit Tensorhub/orchestrator import path.
- No support for private network URLs, file URLs, S3 URLs, or other schemes in this path.

## Tasks
- [x] Audit the existing `Asset.ref` HTTP(S) materialization path and split reusable URL-download logic into a focused helper if needed.
- [x] Ensure media-specific asset classes from #22 recurse through `_materialize_assets` exactly like `Asset`.
- [x] Accept orchestrator-provided URL validation hints/limits on the Asset payload without breaking the existing wire shape.
- [x] Harden SSRF checks across DNS resolution and every redirect target; reject private, loopback, link-local, multicast, reserved, and unspecified IPs.
- [x] Stream remote downloads to temp files with max-byte enforcement and hashing; never read the full input into memory.
- [x] Verify response/sniffed MIME against the typed media class and return a typed validation error on mismatch.
- [x] For image inputs, verify dimensions/pixel count after download and delete/reject oversized or decompression-bomb-style files before tenant code sees them.
- [x] Cache URL downloads only by URL plus auth-token hash/validation context so private downloads do not leak between requests.
- [x] Preserve local_path, owner, mime_type, size_bytes, sha256, and any existing media/ref fields after materialization.
- [x] Add tests for approved image URL materialization, redirect-to-private rejection, too-large streaming rejection, MIME mismatch, oversized dimensions, cache key token isolation, generic Asset compatibility, and media-specific asset recursion.

---

# #332: Internal worker diagnostic logging lane separate from tenant WorkerEvent

**Completed:** yes
**Status:** completed

## Goal

Add a dedicated worker -> orchestrator diagnostic logging channel for internal operator/debugging facts emitted by the worker runtime or endpoint setup code. This channel must be separate from `WorkerEvent` because `WorkerEvent` is the tenant/request event fabric: request-scoped worker events are persisted to `request_events` and are intended for end-user/client consumption through request SSE.

Internal diagnostics are different. They are not tenant-authored, not part of the endpoint's public event contract, and must never be exposed to end clients by default. Examples: model/pipeline memory-sharing diagnostics, torch.compile warmup timing, CUDA memory before/after setup, resolved dependency versions, setup-stage timings, adapter cleanup diagnostics, and worker-local cache state.

## Boundary

- `WorkerEvent`: tenant/domain/request events. Request-scoped events may surface to clients.
- `WorkerDiagnosticLog` (new typed protocol message): internal worker/runtime logs. Requestless by default, optionally correlated to release_id, worker_id, pod_id, class/function label, setup phase, or request_id for operator-only debugging. Not inserted into public `request_events`.
- Normal Python logging/stdout remains useful for pod-local debugging, but it should not be the only way to inspect durable runtime facts after the pod exits.

## Target shape

Add a typed protobuf message rather than another opaque `WorkerEvent` string:

```proto
message WorkerDiagnosticLog {
  string worker_id = 1;
  string release_id = 2;
  string runpod_pod_id = 3;
  string category = 4;      // e.g. "memory_sharing", "compile_warmup", "setup", "cuda_memory"
  string severity = 5;      // debug/info/warn/error
  string message = 6;       // concise human-readable summary
  bytes payload_json = 7;   // structured internal payload
  int64 emitted_at_unix_ms = 8;
}
```

Add it as a new `WorkerSchedulerMessage` oneof field with a new tag. The worker should expose one small helper, e.g. `_emit_diagnostic_log(category, message, payload, severity="info")`, that endpoint setup code can call without requiring a `RequestContext`.

## Orchestrator companion

gen-orchestrator should handle this typed message separately from `WorkerEvent`: log it to orchestrator stdout and persist it to an internal/operator-only store if available. It must not write these diagnostics into `request_events` and must not expose them on tenant request SSE. A future admin endpoint can query by release_id / worker_id / pod_id / time range.

## First consumer

Use the FLUX.2 Klein endpoint's memory-sharing probe as the first real consumer. Instead of only writing `[memory-sharing]` Python log lines, emit a diagnostic log containing pipeline ids, component module ids, shared module/component/storage results, and CUDA allocated/reserved bytes. Keep normal logger output as a local fallback.

## Tasks

## Tasks
- [x] Add `WorkerDiagnosticLog` to the shared worker_scheduler protobuf with a new `WorkerSchedulerMessage` oneof field. Regenerate Go and python pb outputs in the owning repos.
- [x] Add a python-gen-worker helper for internal diagnostics, e.g. `_emit_diagnostic_log(category, message, payload=None, severity="info")`. It must be best-effort and must never fail setup or inference.
- [x] Route diagnostic messages over the existing control/events stream without using `WorkerEvent` and without requiring a `RequestContext`.
- [x] Include stable runtime correlation fields when available: worker_id, release_id, runpod_pod_id, endpoint class/function label, image digest, and emitted_at_unix_ms.
- [x] Add bounded payload handling: JSON only, max payload size, truncation marker, no raw secrets/tokens, and no arbitrary large tensor/model metadata dumps.
- [x] Add gen-orchestrator companion handling: receive typed diagnostics, log them, persist them to an internal/operator-only store or ring buffer, and never insert them into public `request_events`.
- [x] Add an admin/debug read path for recent diagnostics filtered by release_id / worker_id / pod_id / category.
- [x] Update the FLUX.2 Klein memory-sharing probe to emit `category="memory_sharing"` diagnostics in addition to local Python logs.
- [x] Add tests proving diagnostics are sent on setup without a RequestContext, do not appear in request SSE/request_events, are safe when the stream is disconnected, and truncate oversized payloads.
- [x] Bump and publish a new `gen-worker` package version containing the diagnostic helper.
- [x] Update the FLUX.2 Klein endpoint dependency/lockfile to use the newly published `gen-worker` version.
- [x] Re-run focused worker and endpoint tests after the release/pin.

---

# #333: Endpoint-declared string enum payload types and manifest validation

**Completed:** yes
**Status:** completed

Add a public SDK string-enum base for endpoint-authored finite string fields and make the downstream manifest contract treat string enums as string-compatible. Endpoint authors should be able to declare their own accepted values, e.g. a function-local AspectRatio enum, and have gen-orchestrator validate incoming request payloads against that enum before dispatch. The FLUX.2 Klein compiled base function is the first consumer.

## Tasks
- [x] Add `StringEnum` to the gen-worker public API so endpoints can declare their own string enum values.
- [x] Add gen-worker schema/discovery tests proving endpoint-declared string enum fields serialize as JSON Schema enums and can drive ExpectedOutput.aspect_ratio.
- [x] Teach Tensorhub manifest validation that string enum/const schemas satisfy string input refs for expected_outputs.aspect_ratio.
- [x] Add a gen-orchestrator request-validation test proving invalid enum payload values are rejected before dispatch.
- [x] Update the FLUX.2 Klein endpoint to declare its own AspectRatio(StringEnum) instead of the raw str workaround.
- [x] Bump and publish a gen-worker package version containing StringEnum, then pin/relock the FLUX endpoint.
- [x] Redeploy the FLUX endpoint and verify discovery/deploy accepts the AspectRatio-backed manifest.

---

# #334: Shared immutable model components and compile-path benchmark gate

**Completed:** yes
**Status:** completed

## Goal

Make python-gen-worker, not endpoint code, own process-local sharing of immutable model weights/components across functions that bind the same compatible model. This should let two functions using the same HFRepo/Tensorhub repo avoid duplicating Diffusers weights in VRAM when they can safely share base components.

The first motivating case is the FLUX.2 Klein endpoint: `generate` and `generate_compiled` both use the base bf16 model. Today they may independently call Diffusers `from_pretrained`, creating separate module graphs and duplicate CUDA storage. The worker should be able to load immutable base components once per GPU and give each function an appropriate pipeline/wrapper built from those shared components.

## Key rule

Do not share mutable function state. Share immutable base components only. Pipeline instances still own mutable schedulers, adapters, compiled wrappers, request-local state, and any per-function mutation. LoRA overlays and model overrides must not mutate a compiled/shared base in a way that contaminates another function.

## Cache key

Canonical loaded-component cache keys must include at least:

- provider: hf | tensorhub | civitai
- canonical ref / resolved repo id
- resolved revision / snapshot digest / tag+flavor identity
- dtype
- quantization / quantization config
- pipeline class or compatible component set
- device_id
- device placement/offload mode

CUDA weights are not naively shared across GPUs, so a multi-GPU worker has one loaded component copy per `device_id`.

## Diffusers design

Use Diffusers' supported component sharing shape rather than relying on Python object identity accidents:

- Load a canonical base pipeline/component set once.
- For compatible functions, create function-owned pipeline objects via `from_pipe()` / `pipeline.components` / ComponentsManager where available.
- If injecting the exact same pipeline object is safe for a function, allow that as an optimization, but default to function-owned wrappers over shared immutable modules.
- Track refcounts and ownership so unload/revocation does not free weights still used by another function.

## Compiled path restrictions

Compiled functions are intentionally less flexible. For a function that torch-compiles the base model for fixed shapes:

- LoRA overlays must be disabled for that function/binding.
- Model override via `_models.<binding>.ref` must be disabled for that function/binding.
- If a user wants a LoRA/custom model with compile, that is a different compile artifact and must be treated as a separate function/release/cache key after explicit measurement.
- Discovery/tests should prove the FLUX `generate_compiled` binding has `allow_lora=false` and `allow_override=false`, and invalid override/LoRA requests are rejected before worker mutation.

## Measurement gate

Before investing deeply in the shared-component implementation, measure whether `generate_compiled` is worth the flexibility cost:

- Emit internal diagnostic logs for compile warmup: total time, per-aspect-ratio time for all 7 shapes, CUDA allocated/reserved before/after, and whether graph compilation reused or regenerated state.
- Benchmark `generate` vs `generate_compiled` on the same GPU, same release, same prompt/seed/steps/aspect ratio, after warmup.
- Record complete round-trip time and worker-side inference time separately.
- Compare memory: base-only function resident, compiled-only resident, and both functions resident. Use the memory-sharing diagnostic payload to prove whether storages are shared or duplicated.
- Keep the compiled path only if speedup is large enough to justify fixed shapes plus no LoRA/model override. Target decision threshold: at least 25-30% warm worker-side inference speedup, or a clearly better round-trip p95 under realistic load.

## Acceptance

- Identical compatible fixed bindings can share immutable base components per GPU without duplicate CUDA storages.
- Mutable LoRA/adapters/compiled modules/request state cannot leak across functions.
- `generate_compiled` cannot accept LoRA overlays or model overrides.
- Compile warmup duration for all 7 aspect-ratio shapes is visible through worker diagnostics.
- Benchmarks show whether compiled FLUX.2 Klein base is worth keeping.

## Tasks

## Tasks
- [x] Add a worker-level loaded-component cache design doc/API sketch covering cache keys, refcounts, per-GPU ownership, compatible component sets, and unload/revocation behavior.
- [x] Audit current SerialWorker model injection/setup flow to identify exactly where HFRepo/Tensorhub bindings become local paths/objects and where a shared component cache can be inserted without endpoint-specific code.
- [x] Add binding metadata/discovery tests proving function bindings expose `allow_override` and `allow_lora` accurately enough for compiled functions to opt out.
- [x] Update the FLUX.2 Klein compiled function declaration/tests so `generate_compiled` explicitly has LoRA overlays and model override disabled; add a request-shape rejection test for `_models.pipeline.ref` and `_models.pipeline.loras` on that function.
- [x] Add worker diagnostic timing around torch.compile warmup: per-shape compile time, total compile time, CUDA allocated/reserved before and after, and graph bucket labels for the 7 supported aspect ratios.
- [x] Run RunPod benchmarks for FLUX base bf16 `generate` vs `generate_compiled` on the same GPU class with default base steps, fixed prompt set, same seeds, and all 7 aspect ratios where practical. 2026-05-22 H100 HBM3 gate run: `generate` completed, `generate_compiled` failed with internal error after setup/compile diagnostics, so broader 7-ratio timing was not practical.
- [x] Store benchmark images and a JSON/CSV result file under `/tmp` with round-trip time, worker-side inference time, queue time, compile warmup time, GPU type, release id, function name, aspect ratio, seed, and output path. Evidence: `/tmp/flux-klein-h100-hbm3-clean-20260522T211850Z` and `/tmp/flux-klein-h100-hbm3-4b-warm-20260522T213100Z`; request rows include round-trip/output paths, and orchestrator logs provide assignment-to-completion timings.
- [x] Use memory-sharing diagnostics to compare CUDA storage sharing/duplication for base-only, compiled-only, and both-functions-resident workers. 2026-05-22 H100 HBM3 diagnostics showed `GenerateBf16` and `GenerateBf16Compiled` had `same_pipeline=false` and no shared modules/storages with each other; VRAM was duplicated rather than shared.
- [x] Decide whether to keep `generate_compiled`: require at least 25-30% warm worker-side inference speedup or a clearly better realistic p95; otherwise remove or park the compiled function. Decision from H100 HBM3 gate: park/remove for now because the compiled request failed and produced no speedup evidence.
- [x] If benchmarks justify it, implement the shared immutable Diffusers component cache using `from_pipe()` / `pipeline.components` / ComponentsManager, with per-GPU cache keys and refcounted unload. Skipped by benchmark gate: `generate_compiled` failed on H100 HBM3 and produced no speedup evidence, so the shared-component implementation is intentionally deferred.
- [x] Add tests proving compatible functions share immutable component storage, incompatible dtype/quantization/device bindings do not share, and LoRA/compiled mutable state cannot cross-contaminate. Skipped with the implementation because the benchmark gate did not justify building the cache yet.

---

# #335: Worker-owned shared Diffusers component cache for identical model bindings

**Completed:** yes
**Status:** completed

## Goal

Implement the real worker-level fix for identical Hugging Face/Diffusers model bindings across endpoint classes. In the FLUX.2 Klein endpoint, `GenerateBf16` and `GenerateBf16Compiled` both bind the same HFRepo base model, but the current SerialWorker setup path resolves each binding to a raw local path/ref and endpoint `setup()` code can call Diffusers `from_pretrained()` separately. That creates duplicate Python module graphs and duplicate CUDA storages.

The worker should own process-local sharing of immutable loaded Diffusers base components. Endpoint classes should receive shared handles or function-owned pipelines built from shared components, not bare HF refs that force each class to load its own copy.

## Current insertion points

- `_ensure_serial_class_started()` calls `_resolve_serial_model_paths(ep_spec)` and passes the resulting kwargs directly to `instance.setup(**models)`.
- `_resolve_serial_model_paths()` currently returns a local snapshot path when cached, otherwise the bare binding ref string.
- Request-time Diffusers injection has a separate `from_pretrained()` path and whole-pipeline `ModelCache` reuse keyed by model ref, but SerialWorker class setup is still endpoint-driven and can duplicate component loads across classes.

## Design

Add a worker-owned loaded component cache for Diffusers-compatible fixed bindings. Cache entries are process-local and GPU-local. The canonical cache key must include provider, canonical ref or resolved repo identity, resolved revision or snapshot digest, dtype, quantization and quantization config, device id, placement/offload mode, and component-set identity or pipeline class compatibility. Multi-GPU workers must not share CUDA storage across `device_id`; each GPU gets its own loaded component entry.

For Diffusers, use supported sharing APIs rather than relying on incidental object identity. Load one canonical base pipeline/component set per key, then create function-owned pipeline objects with `from_pipe()`, `pipeline.components`, or ComponentsManager where the installed Diffusers version supports it. Inject the exact same pipeline object only behind an explicit safety decision; the default should be function-owned wrappers over immutable shared modules.

The cache must refcount acquired entries and define lifecycle rules for setup, shutdown, unload, and worker drain. Unload/revocation must not free components still held by another endpoint class or in-flight request. Diagnostics should expose cache hits/misses, refcounts, device id, component ids, and CUDA storage sharing evidence.

## Mutability boundaries

Only immutable base components are shared. Function instances still own mutable schedulers, request-local state, adapters, compiled wrappers, and any endpoint-specific mutation.

LoRA overlays must be isolated. If a binding accepts LoRAs, apply overlays on a request-owned or function-owned wrapper and unload them on exit without mutating shared base components for other functions.

Compiled functions are stricter. A compiled binding must opt out of request LoRA overlays and `_models.<binding>.ref` model overrides, or the worker must isolate compiled wrappers under a separate cache key that includes the exact override/adapter identity. Do not let compiled wrappers share mutable state with uncompiled or LoRA-capable functions.

## Acceptance

- Identical FLUX.2 Klein bf16 HFRepo bindings across `GenerateBf16` and `GenerateBf16Compiled` use one immutable base component copy per GPU where safe.
- The setup API can receive shared handles or pipelines built from shared components instead of only raw path/ref strings.
- LoRA-capable and override-capable bindings cannot contaminate compiled or non-LoRA functions.
- Multi-GPU workers keep separate component entries by device id.
- Diagnostics prove shared CUDA storages for shared components and prove isolation for mutable wrappers.
- Live validation on flux.2-klein-4b shows reduced duplicate VRAM residency or explains why a binding was not shareable.

## Landed 2026-05-28
Implementation complete and merged to gen-worker master; `uv run pytest -q` green (118 passed, 1 skipped). Shared-component cache, refcounting, eviction/demote/unload protection (refcount>0 veto, interplays with #337 pin), LoadedComponentKey canonicalization, and LoRA/override sharing-refusal all verified with real lightweight objects. New: models/shared_components.py + cache.py refcount + worker.py wiring + tests/test_shared_components.py. Two tasks remain GPU-gated and were NOT run locally (8GB card, no model weights): (a) converting the SerialWorker share path to build function-owned pipeline wrappers around the shared base (build_function_owned_pipeline exists but the SerialWorker path currently shares the base object directly); (b) live flux.2-klein-4b data_ptr() storage-aliasing + before/after VRAM-residency proof.

## Tasks
- [x] Audit SerialWorker setup and request-time injection paths and identify the smallest shared abstraction that can cover fixed Diffusers bindings without changing endpoint author code unnecessarily.
- [x] Define `LoadedComponentKey` canonicalization with provider, ref, revision/snapshot digest, dtype, quantization/config, device_id, placement/offload mode, and compatible component-set or pipeline-class identity.
- [x] Implement a process-local per-GPU loaded component cache with acquire/release refcounts, thread safety, shutdown/drain cleanup, and explicit nonsharing for incompatible keys.
- [ ] Load Diffusers base components once per cache key and create function-owned pipelines through `from_pipe()`, `pipeline.components`, or ComponentsManager where safe for the installed Diffusers version.
- [x] Change SerialWorker setup resolution so compatible bindings can pass shared handles or function-owned shared-component pipelines into `setup(**models)` instead of raw HF refs/local paths.
- [x] Preserve immutable/mutable boundaries: shared base modules only; schedulers, adapters, compiled wrappers, request-local state, and endpoint mutations stay function/request-owned.
- [x] Enforce LoRA and override safety: LoRA overlays apply to isolated wrappers and unload on exit; compiled functions must opt out of LoRA/model overrides or use cache keys that include the exact compiled override/adapter identity.
- [x] Add diagnostics for cache hit/miss, key fields, refcounts, pipeline/component object ids, CUDA allocated/reserved bytes, and storage data_ptr sharing checks.
- [x] Add focused tests proving identical fixed bindings share storages, dtype/quantization/revision/device/component-set differences do not share, refcounts prevent premature unload, and mutable LoRA/compiled state cannot cross-contaminate.
- [ ] Live-validate on flux.2-klein-4b with `GenerateBf16` and `GenerateBf16Compiled`; capture diagnostics showing shared base storages or the explicit reason sharing was rejected, plus before/after VRAM residency.

---

# #336: GPU memory discipline: hold mutex across weight-load + LRU-evict before SerialWorker model loads

**Completed:** yes
**Status:** completed

## Goal

A single warm worker pod hosts ONE endpoint with MANY `@inference` classes, each binding its own model variant (e.g. flux.2-klein-4b: `generate_bf16`, `generate_fp8`, `generate_nvfp4`, `generate_compiled`, plus turbo/nf4/int8/bnb). On a sequential benchmark (count=3 per function) against one RTX 5090 (32 GB):

- flux-4b **bf16** and **compiled** "succeeded" but took **~170s per 25-step image** (should be ~10-30s) — overhead consistent with inference contending against a concurrent weight-load on the same GPU.
- flux-9b **bf16** took **1640s (27 min)** for one image. The model is ~23 GB and fits in 32 GB, so it should NOT CPU-offload — but multiple variants (bf16 + fp8 + nvfp4) loaded their weights into the same GPU and stayed co-resident, blowing past the VRAM budget into thrash / CPU offload.

python-gen-worker must (1) hold the GPU mutex across *weight loading* (not just inference), and (2) use its LRU `ModelCache` to evict resident models before loading a new one. This issue fixes both.

## Root cause

**Defect B — GPU mutex not held during model loading.**
In `Worker._ensure_serial_class_started` (`src/gen_worker/worker.py`), model-binding resolution `_resolve_serial_setup_kwargs(...)` — which is where `from_pretrained` / the FLUX.2 Klein modelopt composer actually materialize weights into VRAM — ran at worker.py:7728 **before** the GPU semaphore was acquired (worker.py:7763). The cold-boot eager-prewarm (`_handle_manager_model_try_eager_load` -> `_trigger_serial_setup_for_ref` -> `_ensure_serial_class_started(rec, acquire_gpu_semaphore=True)`, worker.py:4895) calls this from a download thread holding NO GPU mutex. Result: one class's weight-load could run concurrently with another class's load, or with an in-flight inference, on the same GPU -> the ~170s contention.

**Defect E — no LRU eviction before load; variants co-reside and OOM/thrash.**
The SerialWorker typed-model load path `_load_serial_setup_model_value` (worker.py ~8058) called `requested_type.from_pretrained(...)` / `_try_load_flux2_klein_modelopt_pipeline(...)` and returned the object straight into the tenant `setup()` kwargs. It **never registered the model in `self._model_cache`** (the LRU `ModelCache` at worker.py:694) and never asked it to evict. So bf16 + fp8 + nvfp4 each loaded their own ~10-23 GB model and all stayed resident -> exceeded the 32 GB budget -> CPU offload / thrash (the 1640s flux-9b incident). Additionally the cold-boot eager-prewarm gated only on `cache.vram_free_gb > 0` (worker.py:4785), and since `model_sizes` is usually empty the cache accounting never reported "full", so it pre-loaded *every* variant at boot.

## Fix

**B — acquire the GPU semaphore BEFORE binding resolution.** Moved the `needs_gpu_setup_lock` compute + `_gpu_semaphore.acquire()` + `_gpu_busy_enter()` to run *before* `_resolve_serial_setup_kwargs` in `_ensure_serial_class_started` (`src/gen_worker/worker.py` ~7710-7785). Added a release of the semaphore in the resolution-failure path so a crash during weight-load never strands the GPU mutex. The normal post-warmup release block is unchanged; request-dispatch still holds the mutex before calling the helper (no double-acquire). Now no GPU ever loads weights or runs inference for >1 model at a time without holding `_gpu_semaphore`.

**E — route every typed-model load through ModelCache as a gatekeeper.** Refactored `_load_serial_setup_model_value` (`src/gen_worker/worker.py` ~8058): (a) if the variant is already hot in VRAM, reuse it (touches recency); (b) `_evict_for_incoming_serial_model(...)` evicts LRU VRAM models *before* the load when live GPU VRAM is tight; (c) the actual load moved to `_load_serial_setup_model_value_uncached`; (d) after load, `mark_loaded_to_vram(canon, obj, size)` registers it (which itself calls `_evict_lru_for_space`) so the *next* variant evicts this one. Added `_canonical_cache_key_for_ref` and `_evict_for_incoming_serial_model` helpers.

**Fix 3 — cold-boot eager-prewarm respects the VRAM budget.** `_handle_manager_model_try_eager_load` (worker.py ~4849) now reads the LIVE GPU free VRAM via `inference_memory.get_available_vram_gb` after each eager load and stops pre-loading further variants once free VRAM <= the cache safety margin — instead of trusting the unreliable `vram_free_gb` accounting. Remaining variants stay disk-resident and are promoted on demand (where the load path now evicts LRU to fit).

Single-GPU correctness preserved; multi-GPU semaphore sizing (`max(1, gpu_count)`) untouched.

## How the acceptance criteria are met

- **bf16 -> fp8 -> nvfp4 sequentially on a 32 GB pod**: each load registers in `ModelCache`; `mark_loaded_to_vram` + the pre-load `_evict_for_incoming_serial_model` evict the previous variant, so only one variant is VRAM-resident at a time -> no offload -> inference at full speed (~10-30s for 25 steps).
- **flux-9b bf16 alone uses ~23 GB with no offload**: eager-prewarm stops at the budget so other variants aren't co-loaded; the on-demand load evicts any LRU resident before loading -> ~23 GB resident, fits in 32 GB.
- **No load runs concurrently with inference or another load on the same GPU**: the GPU semaphore is now held across binding resolution (the weight-load), and request dispatch already holds it across inference — so loads and inference fully serialize on the single GPU slot.

## Verification

`.venv/bin/python -m pytest tests/ -q` -> 514 passed, 3 skipped (was 510 before this work). Added 4 regression tests in `tests/test_concurrency_semaphore.py`: semaphore held during binding resolution, semaphore released on resolution failure, typed-model load evicts LRU across variants, hot-VRAM variant reuse.

## Tasks
- [x] Confirm defect B: _resolve_serial_setup_kwargs (weight load) runs before the GPU semaphore acquire in _ensure_serial_class_started (worker.py:7728 vs 7763).
- [x] Confirm defect E: SerialWorker typed-model loads (_load_serial_setup_model_value) never register in _model_cache and never evict LRU.
- [x] B: acquire _gpu_semaphore + _gpu_busy_enter before binding resolution; release on resolution failure; preserve normal post-warmup release.
- [x] E: route _load_serial_setup_model_value through ModelCache — reuse hot variant, evict LRU before load, mark_loaded_to_vram after load.
- [x] Fix 3: cold-boot eager-prewarm reads live GPU free VRAM and stops at the safety-margin budget instead of pre-loading every variant.
- [x] Keep single-GPU correct; do not change multi-GPU semaphore sizing.
- [x] Add regression tests (semaphore-during-resolution, release-on-failure, LRU eviction across variants, hot reuse).
- [x] Run tests: 514 passed, 3 skipped.

---

# #337: Model-selectable endpoints: SharedBase + per-request variant dispatch, 3-tier LRU residency, partial readiness, orchestrator affinity routing

**Completed:** yes
**Status:** completed

## Goal

Define and implement the contract for endpoints that expose a SET of selectable models — and that may have FAR MORE models than fit in RAM or VRAM — via a **shared base + swappable per-request variant**, with the SDK owning a **3-tier (VRAM / CPU-RAM / disk) LRU residency cache**, **per-model partial readiness**, and **residency-aware fleet routing** by the orchestrator. The tenant declares the model set and uses the injected, VRAM-ready pipeline; the SDK manages download, placement, eviction, the GPU mutex, and availability reporting.

## Motivating incident

`sdxl-illustrious` was made model-selectable via `models={"pipeline": dispatch(field="model", table=...)}` but its `setup(self, pipeline)` requires a fixed model at startup. A `dispatch` slot resolves PER-REQUEST, so the worker called `setup()` with no kwargs and crashed: `TypeError: IllustriousXL.setup() missing 1 required positional argument: 'pipeline'` (captured via the worker->orchestrator diagnostic channel; setup_mode=bare). Root cause: the SDK has a clean injection point for FIXED models (a `setup()` kwarg, loaded once, stashed on `self`) but NO defined delivery mechanism for a per-request-SELECTED model. This issue defines that mechanism and the residency/routing machinery it implies. Builds directly on #336 (GPU mutex + VRAM<->disk ModelCache).

## Design

### A. Tenant authoring API (gen-worker public surface)

```python
# Shared, frozen components — loaded ONCE, pinned in VRAM, shared BY REFERENCE across all variants.
sdxl_base = SharedBase(
    StableDiffusionXLPipeline,
    text_encoder   = HFRepo("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder"),
    text_encoder_2 = HFRepo("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2"),
    vae            = HFRepo("madebyollin/sdxl-vae-fp16-fix"),
)

# Each selectable model = shared base + its own fine-tuned UNet (the swap unit).
MODELS = {
    "illustrious": sdxl_base.variant(unet=HFRepo("OnomaAIResearch/Illustrious-XL-v1.0")),
    "animagine":   sdxl_base.variant(unet=HFRepo("cagliostrolab/animagine-xl-3.1")),
    "dreamshaper": sdxl_base.variant(unet=HFRepo("Lykon/dreamshaper-xl-1-0")),
    "juggernaut":  sdxl_base.variant(unet=HFRepo("RunDiffusion/Juggernaut-XL-v9")),
}

class GenInput(msgspec.Struct):
    prompt: str
    model: Literal["illustrious","animagine","dreamshaper","juggernaut"] = "illustrious"

@inference(models={"pipeline": dispatch(field="model", table=MODELS)})
class SDXL:
    def setup(self):                       # NO model arg; SDK pins the shared base.
        ...                                # stateless init only
    @inference.function(name="generate")
    def generate(self, ctx, payload: GenInput,
                 pipeline: StableDiffusionXLPipeline) -> ImageOutput:
        # `pipeline` = the model named by payload.model, shared base + its UNet,
        # already swapped into VRAM by the SDK. Just run it.
        img = pipeline(prompt=payload.prompt, num_inference_steps=25).images[0]
        return ImageOutput(image=img)
```

Key contract: **`setup()` builds what is shared and load-once; `dispatch`/variant slots are resolved per-request and injected into the HANDLER, VRAM-readied by the SDK.** A `dispatch` slot appearing as a `setup()` parameter is a discovery-time error (clear message), not a runtime crash.

### B. Component sharing (shared-by-reference) + safety

In diffusers a pipeline is a container of references to component `nn.Module`s, so many variant pipelines point at the SAME component objects:
```python
pipe = StableDiffusionXLPipeline(vae=shared_vae, text_encoder=shared_te, text_encoder_2=shared_te2,
                                 unet=variant_unet, scheduler=..., tokenizer=..., tokenizer_2=...)
```
=> ONE copy of VAE + text encoders in VRAM, never duplicated, never evicted; only the UNet/transformer is per-model. Memory = (shared stack, once) + {LRU set of variant UNets}, instead of N x full pipeline. This is what makes hundreds of fine-tunes feasible (~2.5GB UNet vs ~5GB full pipeline; shared stack ~1.8GB once).

Sharing is BY DECLARATION: the SDK pulls only `unet=` (the variant slot) from each variant repo and pairs it with the tenant-declared shared components. SAFETY: if a variant repo ships its OWN text_encoder/VAE that DIFFER from the declared base (some anime SDXL fine-tunes retrain CLIP), pairing the shared CLIP with that UNet yields subtly-wrong output — so at load the SDK must WARN ("variant X ships a text_encoder differing from the shared base; it is being ignored"). The tenant can instead declare that model as a standalone full pipeline (empty/own shared base) — still LRU-cached.

### C. Worker (gen-worker) responsibilities

1. Per-request slot resolution: read the discriminator field from the payload, resolve to the variant, ensure VRAM-ready, inject the assembled pipeline as the handler parameter.
2. **3-tier residency cache** (extend #336 ModelCache, which already does VRAM<->disk, by INSERTING a CPU-RAM warm tier):
   - VRAM (GPU): active variant UNet(s) + PINNED shared base. Evict LRU UNet -> CPU RAM (`.to("cpu")`).
   - CPU RAM (host): warm UNets, ready for fast PCIe swap-in (~0.1-0.5s for ~2.5GB). Evict LRU -> drop bytes, keep file on disk.
   - Disk (CAS): all downloaded variant weights. Evict LRU -> re-pullable from HF/source.
   - Promote on demand (VRAM hit=0 cost; RAM=PCIe copy; disk=read+load; absent=HF pull then up). Auto-size tiers from torch VRAM / host RAM / disk quota with safety margins. ALL transitions under the GPU mutex (#336) so loads/evicts/inference never thrash.
   - Shared-base components are PINNED — never evicted.
3. **Partial readiness** — serve what is loaded, never block on the rest: a variant's function is serveable the moment THAT variant is downloaded + the shared base is up; do not gate on the full set (extends #21 per-model readiness down to the dispatch case). A request for a not-yet-ready variant either waits (bounded) or reports unavailability so the orchestrator routes elsewhere / triggers a load.
4. **Tier-aware availability emission**: extend `WorkerModelReadySignal` (#21) to carry per-model residency tier `{model_id: VRAM|RAM|DISK|DOWNLOADING|ABSENT}`; emit (debounced) on every load/evict/download transition so the orchestrator can route by residency.

### D. Orchestrator (gen-orchestrator) — companion

1. Consume tier-aware availability into a per-worker x per-model residency map.
2. **Residency-aware dispatch routing**: for a request on model X, among workers whose function supports X, prefer the HOTTEST: `VRAM > RAM > DISK > ABSENT`; fall back to cold-load on any supporting worker / scale up. Effect: hot models stay hot on the workers that have them (cache affinity), the fleet stops redundantly re-downloading/re-loading the same weights, tail latency collapses, workers self-specialize into hot subsets.
3. Per-model (not endpoint-global) availability gating so a worker serving a SUBSET is routable for that subset.

### E. Spectrum (this subsumes existing paths)

- **LoRA** (`.allow_lora()`, already built): the variant is a tiny adapter on the shared base — same attach/cache path, tiny payload.
- **Component swap** (this issue): the variant is a full UNet/transformer on the shared base.
- **Separate full pipeline**: when models share nothing (or a variant's components differ), the shared base is empty and the variant is a complete pipeline — still in the same 3-tier cache.

## Tasks

(gen-worker SDK)
- [ ] `SharedBase(pipeline_cls, **components)` + `.variant(**variant_slots)` API in models/; discovery records shared component refs + per-variant ref(s) into endpoint.lock.
- [ ] Extend `dispatch(field=..., table={name: variant})` so table values may be a `SharedBase.variant(...)`, a plain Repo/HFRepo (full pipeline), or a LoRA overlay.
- [ ] Discovery-time guard: a dispatch/variant slot used as a `setup()` parameter is a build error with a clear message; `setup()` accepts only fixed slots + stateless init. Add the fixed-vs-dispatch slot classification.
- [ ] Per-request slot resolution + HANDLER injection: resolve discriminator -> variant -> ensure VRAM-ready -> pass assembled pipeline as the handler arg (today fixed models are injected into setup; add per-request handler injection for dispatch slots).
- [ ] Shared-by-reference assembly: load shared components once + pin in VRAM; build each variant pipeline pointing at the SAME component objects; swap only the variant slot.
- [ ] 3-tier residency cache: insert CPU-RAM warm tier into the #336 ModelCache; LRU per tier; auto-size (torch VRAM / psutil host RAM / disk quota) + safety margins; pin shared components; all transitions under the GPU mutex.
- [ ] Component-compatibility warning at load (variant ships TE/VAE differing from the declared shared base -> warn + ignore; tenant escape hatch = declare as standalone full pipeline).
- [ ] Partial readiness: function serveable per-variant as it lands; do not block startup on the full set; not-yet-ready variant -> bounded wait or unavailable signal.
- [ ] Tier-aware availability: extend WorkerModelReadySignal/proto with per-model residency tier; emit debounced on every transition.
- [ ] Tests: dispatch->handler injection; setup-with-dispatch discovery error; shared-by-reference (one VAE/TE object across variants); 3-tier promote/evict ordering; partial-readiness serving; availability signal transitions; bundled-component mismatch warning; LoRA-as-variant; separate-full-pipeline fallback.
- [ ] Docs: authoring guide for model-selectable endpoints + the residency/routing model.

(tenant endpoint — inference-endpoints/sdxl-illustrious)
- [ ] Rewrite to the new API: `SharedBase` (SDXL text encoders + fp16 VAE) + `.variant(unet=...)` per fine-tune; `setup()` builds the shared base; `generate(self, ctx, payload, pipeline)`. Validate a real image generates for each selectable model.

(gen-orchestrator — companion)
- [ ] Consume tier-aware availability into a per-worker x per-model residency map.
- [ ] Residency-aware dispatch routing (VRAM > RAM > DISK > ABSENT; fall back to cold-load / scale-up).
- [ ] Per-model availability gating so a subset-serving worker is routable for that subset.
- [ ] Tests: routing prefers hottest worker; correct fallback; partial-availability worker routable for its ready models.

## Non-goals / open questions
- Cross-worker weight transfer (peer pull) is out of scope; cold load is HF/disk.
- Exact CPU-RAM tier sizing policy + debounce interval for availability emission to be tuned from measurement.
- Auto-detection of shareable components (hash-based) is a possible future enhancement; v1 uses tenant declaration + the mismatch warning.

## Worker-side framework verified complete 2026-05-28
The entire gen-worker + tenant side of #337 is implemented, tested, and on master: SharedBase/Variant/HFRepo.subfolder API (api/binding.py); dispatch() accepts variants; discovery records shared+variant refs and rejects dispatch-slot-as-setup-param; per-request handler injection (_resolve_dispatch_injection_kwargs); shared-by-reference assembly + pinned shared base (_assemble_variant_pipeline/_ensure_shared_base); 3-tier VRAM/CPU-RAM/disk ModelCache (ModelLocation enum, demote/promote, auto-sized); component-mismatch warning; partial readiness; tier-aware _emit_residency_tier; docs/model-selectable-endpoints.md; sdxl-illustrious rewritten to SharedBase+.variant. 24 tests pass (test_shared_components.py + test_worker_dispatch.py). STILL OPEN: the gen-orchestrator companions (residency map, residency-aware routing VRAM>RAM>DISK>ABSENT, per-model gating, routing tests) — cross-repo Go work.

## Orchestrator companions complete 2026-05-28 (#337 4b)
Residency-aware dispatch routing was already live (cache_selection.go: VRAM>disk>cold, per-worker x per-model inventory via info.Resources, per-model gating via workerSupportsFunction). Completed the VRAM>RAM>DISK promise by adding the CPU-RAM warm tier end-to-end: WorkerResources.ram_models (proto field 30) emitted by the worker from cache.get_cpu_models(); orchestrator localityRAM between VRAM and disk. go build + grpc tests green; new RAM-tier tests. #337 is now fully complete.

## Tasks
- [x] gen-worker: SharedBase(pipeline_cls, **components) + .variant(**slots) API; discovery records shared + per-variant refs into endpoint.lock
- [x] gen-worker: extend dispatch(field, table) to accept SharedBase.variant / plain Repo / LoRA overlay as table values
- [x] gen-worker: discovery-time guard - dispatch/variant slot in setup() is a build error with a clear message; classify fixed vs dispatch slots
- [x] gen-worker: per-request slot resolution + HANDLER injection of the VRAM-ready assembled pipeline (vs today's setup-only injection)
- [x] gen-worker: shared-by-reference assembly - load shared components once, pin in VRAM, build variant pipelines pointing at the SAME objects, swap only the variant slot
- [x] gen-worker: 3-tier residency cache - insert CPU-RAM warm tier into #336 ModelCache; per-tier LRU; auto-size+margins; pin shared base; all transitions under the GPU mutex
- [x] gen-worker: component-compatibility warning (variant ships TE/VAE differing from declared shared base -> warn+ignore; standalone-pipeline escape hatch)
- [x] gen-worker: partial readiness - serve each variant as it lands, never block on the full set; not-ready variant -> bounded wait or unavailable
- [x] gen-worker: tier-aware availability - extend WorkerModelReadySignal/proto with per-model residency tier (VRAM/RAM/DISK/DOWNLOADING/ABSENT); emit debounced on transitions
- [x] gen-worker tests: injection, setup-with-dispatch error, shared-by-reference identity, 3-tier promote/evict, partial readiness, availability transitions, mismatch warning, LoRA-as-variant, separate-pipeline fallback
- [x] gen-worker docs: model-selectable endpoint authoring guide + residency/routing model
- [x] tenant: rewrite inference-endpoints/sdxl-illustrious to SharedBase + .variant(unet=...) + handler-injected pipeline; validate an image per model
- [x] gen-orchestrator (companion): per-worker x per-model residency map from availability signals
- [x] gen-orchestrator (companion): residency-aware dispatch routing (VRAM>RAM>DISK>ABSENT; fallback cold-load/scale-up)
- [x] gen-orchestrator (companion): per-model availability gating so a subset-serving worker is routable for its ready models
- [x] gen-orchestrator (companion) tests: prefers hottest worker; correct fallback; partial-availability routable

---

# #338: Worker reconnect robustness: re-emit `ready` after reconnect, bound the retry storm, detect half-open scheduler streams

**Completed:** yes
**Status:** completed

## Symptom

When the orchestrator is restarted (e.g. `docker compose up -d --force-recreate gen-orchestrator`) while a worker is ALREADY connected, that worker wedges permanently: the orchestrator shows `connected=0` for the release, invokes sit `queued`, and the only recovery is killing the worker so the autoscaler spawns a fresh one (fresh/cold-start workers are healthy). This is the worker-side half of the e2e reconnect bug.

## Context (orchestrator side already fixed)

The shard-routing half was fixed in gen-orchestrator: workers now reach the orchestrator through the shard-aware `orchestrator-router` (run mode `run-router`, #317), which routes a stream to the replica holding the release's lease. With that, FRESH workers connect in one attempt. But an ALREADY-connected worker that loses its stream on an orchestrator restart still wedges — the cause is now isolated to gen_worker's own reconnect state machine.

## Root cause (3 compounding worker-side issues)

1. **Never re-emits `ready` on reconnect.** `_emit_ready_if_all_cached` is gated by a once-only `_ready_phase_emitted` latch (`src/gen_worker/worker.py:5048`) that is NOT reset when the stream re-establishes. So after a reconnect the worker registers but never re-advertises `ready`, and the orchestrator keeps it at `connected=0`.
2. **Retry storm.** The reconnect loop retries roughly every ~0.1s (~280 attempts observed through a brief replica-restart window) with no backoff — hammering the orchestrator/router during the exact window they're coming back up.
3. **Settles on a half-open stream.** It can register once and then heartbeats vanish into a dead pipe (write-only-dead socket), so the orchestrator never sees liveness — `connected=0` with a 'live' socket the worker isn't really using.

## Note

The worker image is built by tensorhub-builder from the python-gen-worker base, NOT the e2e compose — so validating this needs a base-image rebuild + endpoint redeploy.

## Landed 2026-05-28
Implementation complete and merged to gen-worker master; `uv run pytest -q` green (13 new tests; 118 passed combined with #335). Ready re-emit on reconnect (_reset_ready_phase_latch), bounded full-jitter backoff (0.5s base / 30s cap, no retry storm), and half-open detection (gRPC HTTP/2 keepalive on all 4 channels + app-level silence watchdog in _heartbeat_loop) all unit-verified. Changes in worker.py + tests/test_reconnect_robustness.py. Two tasks remain e2e-gated and were NOT run locally (no orchestrator-router / base-image rebuild here): live `docker compose up --force-recreate gen-orchestrator` recovery to connected=1, and base-image rebuild + marco-polo redeploy. Open question for e2e: confirm orchestrator idle-push cadence vs the 60s watchdog, and that keepalive (10s ping) satisfies the server min-ping policy (no GOAWAY too_many_pings).

## Tasks
- [x] Reset the `_ready_phase_emitted` latch (worker.py:5048) on every (re)connect so `_emit_ready_if_all_cached` re-fires `ready` after the stream re-establishes — the orchestrator must see the worker return to `ready`/`connected=1` post-reconnect without a respawn.
- [x] Add bounded backoff (exponential + jitter, capped) to the scheduler reconnect loop instead of the ~0.1s retry storm, so a worker doesn't hammer the orchestrator/router through a restart window.
- [x] Detect a half-open / write-only-dead scheduler stream (e.g. heartbeat-ack or stream-health check) and tear it down + reconnect rather than settling on it, so the orchestrator never sees a registered-but-dead worker (`connected=0` with a live socket).
- [ ] Validate against the fixed orchestrator-router: with a worker connected, `docker compose up -d --force-recreate gen-orchestrator` (e2e stack) — the EXISTING worker must recover to `ready` (orchestrator `connected=1`, invokes drain) WITHOUT kill+respawn.
- [ ] Rebuild the python-gen-worker base image + redeploy a test endpoint (marco-polo) to exercise the fix end-to-end (worker image is built by tensorhub-builder).

---

# #340: Local persistent dev server (`gen-worker serve`): load models once, invoke warm over stdin/stdout or a Unix socket

**Completed:** yes
**Status:** completed

## Problem

`gen-worker run` is one-shot: load the model into VRAM, run one request, exit. Every invocation reloads the model — minutes of cold start per poke. Local dev needs a persistent process: start once, models resident in VRAM, fire many requests warm, Ctrl+C to stop.

## Transport: stdin/stdout + Unix socket, NOT HTTP/gRPC

It's all one machine, so no network transport is needed. Use newline-delimited JSON (NDJSON) as the wire format, with two modes sharing ONE dispatch handler:
- REPL/pipe mode (single process): `gen-worker serve` reads NDJSON requests from its OWN stdin, writes NDJSON results to stdout, logs to stderr. Zero IPC. Great for interactive poking and piping a batch. Limitation: you interact with THIS process — you cannot fire from a SEPARATE terminal, because two processes do not share std streams (and a pipe closes after input, so the process would exit — no persistence).
- Detached server + client (two processes): for the 'server in terminal 1, fire from terminal 2' UX you need an inter-process channel. The idiomatic local-only answer (no HTTP, no gRPC, no ports) is a Unix domain socket: `gen-worker serve --socket /tmp/<ep>.sock` listens; `gen-worker call --method ... --payload ...` connects, sends one NDJSON request, reads the NDJSON result. Same protocol as stdin/stdout, just over a UDS. Prefer UDS over a FIFO/named pipe: UDS handles request/response correlation and sequential clients cleanly; FIFOs don't. HTTP was considered and dropped — it's all local, so the port + HTTP framing buy nothing here (the only thing lost is `curl`-ability, replaced by `gen-worker call`).

## Reuse the production Worker machinery, do NOT fork

`serve` is 'run's dispatch half, but setup is hoisted to boot and the process loops.' Reuse:
- `_ensure_serial_class_started` / `_serial_class_instances` — setup once, hold the instance + model VRAM-resident (where the 'model is VRAM-resident at handler entry' contract from #339 pays off locally).
- the GPU semaphore — concurrent local requests serialize exactly like prod (a local `compiled` warmup blocks correctly, no thrash).
- `build_local_context` (cli/local_context.py) for the RequestContext + the same payload-decode / model-resolution path `run` already has.
Only genuinely new code: hoist setup to boot, a request loop, the UDS listener, and the `call` client. Class/method inferred like `run`.

## Fidelity caveat

Production dispatch is gRPC-from-the-orchestrator. `serve` mirrors setup, context, memory management, and GPU serialization faithfully (shared code) but the TRANSPORT differs. Right trade for local dev (warm-model fast iteration). Byte-for-byte prod fidelity would need the real gRPC Worker against a local stub-scheduler — overkill; build only if a transport-specific bug forces it.

## Target local UX (zero ceremony)

`gen-worker serve` (no args; auto ./endpoint.toml, all models warm, default `./.gen-worker.sock`) in one terminal; then `gen-worker invoke <function-name> <payload>` in another (`<payload>` = inline JSON or `@file.json`). Address by function NAME (the unique routable id), not class/method — the framework resolves the hosting class. No directory, filename, class, or socket path to specify. This is strictly nicer than today's `run --class/--method`. cwd is the DEFAULT, not a lock: `--config PATH` serves an endpoint elsewhere. ONE endpoint per serve process (matches prod: one worker = one release); run several serves with distinct `--socket` paths to host multiple endpoints at once.

## Tasks
- [x] Factor `run`'s dispatch into a reusable handler — (decoded_payload, class, method) -> result — with setup hoisted out, so `serve` runs setup ONCE at boot while `run` keeps calling it per-invocation. No forked dispatch logic.
- [x] `gen-worker serve` (NO required args): default `--config` to `./endpoint.toml`, but accept `--config PATH` to serve an endpoint OUTSIDE the cwd. ONE endpoint per serve process (matches prod: one worker = one release; an endpoint may expose many functions/classes but is one deployable unit). Run setup() once per class (all models resident, warmup paid once), then loop. Default mode: NDJSON over stdin/stdout, logs to stderr. Open a default `./.gen-worker.sock` (cwd) so `invoke` finds it with no flag. Ctrl+C tears down (shutdown() only if present, per #339).
- [x] detached mode: listen on `./.gen-worker.sock` in the cwd by default (override `--socket PATH`); accept sequential NDJSON request/response connections; reuse the SAME dispatch handler + GPU semaphore. Use distinct `--socket` paths to run several serves concurrently (one endpoint each). No HTTP, no port.
- [x] `gen-worker invoke <function-name> <payload>`: the client. Address by FUNCTION NAME (the unique routable id — `function_name` on the wire), resolving the hosting class automatically (no --class/--method). `<payload>` accepts inline JSON, `@file.json` (curl convention), or `-`/stdin (pipe idiom: `echo '{...}' | gen-worker invoke <fn>`). Auto-discover the running `serve` via `./.gen-worker.sock` (override `--socket`); clear error if none is running. (Supersedes the earlier `gen-worker call --method ...` spelling.)
- [x] Reuse `_ensure_serial_class_started`/`_serial_class_instances`, the GPU semaphore, and `build_local_context` — share the Worker's lifecycle/context/memory code, do not reimplement it.
- [x] Docs: per-endpoint README / shared Taskfile target — `uv run gen-worker serve` then `uv run gen-worker call ...`; note the transport-fidelity caveat (NDJSON over stdin/UDS locally, gRPC in prod).
- [x] (optional) let `gen-worker run` auto-attach to a running `serve` socket at the default path if present, else fall back to one-shot — so `run` is warm when a server is up.

---

# #341: Adopt tensorhub's existing generated-Dockerfile mode across all endpoints + reconcile the torch-strip gap

**Completed:** yes
**Status:** completed

## Correction: the generator ALREADY EXISTS in tensorhub
Dockerfile generation is already built and in the right place (the builder): internal/builder/generate_dockerfile.go GenerateDockerfile(); BuildMode = managed|explicit|custom|generated; InferBuildMode() picks generated when no Dockerfile but the profile has build hints, custom/explicit when a Dockerfile is committed (escape hatch already works); the system_dependencies apt field exists (EndpointTomlBuildProfile.SystemDependencies, endpoint_toml.go:100); base-image resolution ResolveEndpointProfileBaseImage (baseimage.go); gen-worker lockstep via MinGenWorkerVersion. The 22 endpoints simply aren't using it — they carry hand-written Dockerfiles (custom/explicit). Work is ADOPTION + a correctness reconciliation, not new infra.

## Gaps
1. (TORCH — fix is NOT a strip) The hand-written Dockerfiles grep-strip torch to avoid replacing the base image's tuned CUDA torch (brittle). Better (decided): base image OWNS torch (a 'provided' dependency — Maven term; NOT 'transitive'; Python has no first-class marker). Endpoints must NOT list torch in [project.dependencies]; move to PEP 735 [dependency-groups] local (or [project.optional-dependencies]) so local dev gets it via uv sync --group local but the container `uv pip install --system .` never pulls it. A duplicate only happens with an ISOLATED venv (base torch invisible -> 2nd/wrong-CUDA build); --system installs into the base env so base torch is visible and uv skips it. diffusers/transformers pull torch transitively but uv/pip skip an already-satisfied dep. Harden with a constraints file pinning torch==<base> (from endpoint.toml) so a divergent transitive pin ERRORS loudly. No grep-strip.
2. (ADOPTION) 22 endpoints carry hand-written Dockerfiles; delete them and rely on profile hints + system_dependencies.
3. (GIT) Generated apt line is only ca-certificates curl + system_dependencies; the hand-written ones install git, and some endpoints have git+ deps (flux.1-dev: diffusers @ git+https://...). Add git to system_dependencies for those, or auto-add when a git+ dep is detected.
4. (UV) The current generator emits `pip install .`; switch to `uv pip install --system` to match the hand-written Dockerfiles and the project-wide uv preference; COPY pyproject/uv.lock before source for dep-layer caching.

## Why not fold endpoint.toml into pyproject.toml
Keep separate. endpoint.toml is the platform's language-agnostic deployment contract consumed by Go tooling (tensorhub-builder, gen-orchestrator) with its own schema_version; folding into pyproject would force Go to parse Python packaging and reach into a [tool.gen_worker] table. They barely overlap (main/base_image aren't in pyproject).

## Tasks
- [x] (torch, endpoint pyproject) Move torch (+ torchvision/torchaudio) OUT of [project.dependencies] into PEP 735 [dependency-groups] local (or [project.optional-dependencies]) so the container `uv pip install --system .` never pulls/replaces base torch while local dev gets it via uv sync --group local. Done across inference endpoints plus training conversion/image_lora_finetuner; lockfiles refreshed.
- [x] (torch, generator) Generated install uses uv: `uv pip install --system --break-system-packages`; dependency requirements are exported/compiled as a closed set and installed with `--no-deps`, with structured no-emit for torch/torchvision/torchaudio/triton/nvidia*/cuda* runtime packages (and vllm when the base image is vLLM). Adds torch-constraints.txt from the base image so divergent transitive pins fail loudly. NO endpoint-local grep-strip.
- [x] (git gotcha) Ensure git lands in the generated image for endpoints with git+ deps. Implemented generator-side VCS dependency detection so endpoint authors do not have to remember the Debian package.
- [x] (adoption) Migrate eligible inference endpoints to BuildModeGenerated: deleted 21 hand-written Dockerfiles; confirmed profile build hints; added system_dependencies for AV endpoints (ffmpeg: ltx-video-2.3, wan-2.2, stable-audio-open; libsndfile1: chatterbox-tts, foundation-1, musicgen, stable-audio-open). qwen3.6-27b-mtp-gguf stays custom because it compiles llama.cpp with SM_TARGET.
- [x] (verify) Built and runtime-checked all 21 migrated endpoints with generated Dockerfiles; confirmed endpoint.lock bake, python3 entrypoint import, no torch/CUDA runtime package downloads, and base torch preserved. sdxl-illustrious is pinned to python-gen-worker commit e34b6323230ed633e2e5f48dde87870e8e495ec4 because its SharedBase endpoint code needs worker API that is on origin/master but not yet in the PyPI 0.7.43 artifact.
- [x] (escape hatch) Confirm qwen3.6-27b-mtp-gguf (cuda-devel/GGUF) either fits generated mode or intentionally stays BuildModeCustom with a committed Dockerfile. Confirmed custom is required: the Dockerfile installs build tools, compiles llama.cpp with SM_TARGET, and sets llama.cpp runtime environment.
- [x] (uv) Switch tensorhub GenerateDockerfile from `pip install .` to `uv pip install --system --break-system-packages`; COPY pyproject/uv.lock before source for dep-layer caching; use python3 for discovery/entrypoint; add shared runtime cache/download env and vLLM-base preservation.

---

# #342: Local-dev model download + serve ergonomics (16-bit-preferred HF download, serve --function subset)

**Completed:** yes
**Status:** completed

Design from the sd1.5 local-inference work (most items SHIPPED + verified on the RTX 4070).

## HF download precision policy (DONE)
The HF model downloader (`hf_selection.py`/`hf_downloader.py`) must select reduced-precision weights robustly:
- Default = 16-bit FAMILY: prefer bf16 then fp16, whichever the repo has. Tenants prefer '16-bit', not a specific dtype; 16-bit is the default when no flavor is specified. A specific bf16/fp16 flavor still falls back within the family.
- NEVER select zero weight files (the sd1.5 bug: only sidecars downloaded). On dtype-probe miss, fall back to filename-precision then available weights; `.bin`/`.ckpt` fallback only when no safetensors exist.
- NEVER download fp32 if a smaller precision exists (last resort only — wasteful).
- Keep optional `model_index.json` components (safety_checker, feature_extractor) so `from_pretrained` can instantiate the pipeline; only prune explicitly null-typed `[null,null]` entries. (Stripping them was the ACTUAL sd1.5 root cause.)
- Wire the documented `COZY_HF_WEIGHT_PRECISIONS` / `COZY_HF_FULL_REPO_DOWNLOAD` env overrides (were referenced in error strings but never read). Lean on huggingface_hub's natural variant/allow_patterns handling.
ROOT CAUSE (sd1.5, two stacked bugs): (1) optional components stripped -> still listed in model_index -> from_pretrained tries to load them -> OSError -> the endpoint loader retries without variant=fp16 -> demands non-variant `unet/...bin` (never selected) -> 'no .bin in unet'; (2) `_dtype_score` returned 0 (not an unknown sentinel) on probe-miss, making the filename-precision fallback dead code. Both fixed.

## serve --function subset (DONE + planned extras)
`gen-worker serve --function NAME` boots ONLY the @inference class(es) hosting the named function(s), so a multi-model endpoint (e.g. stable-diffusion's 8 models) doesn't load all of them. REPEATABLE: `--function one --function two` -> union of hosting classes. Default (omitted) = all classes; unknown name -> usage error listing available functions.

## sd1.5 local-inference proof (DONE)
`gen-worker run --method generate_sd15` on the RTX 4070: downloaded sd1.5 (2.6GB fp16 weights), generated a non-blank 512x512 RGB PNG. Proves the new local-dev system (run/serve/invoke + @invocable + 16-bit download) end-to-end on a real diffusion model. 576 gen-worker tests pass.

## Tasks
- [x] HF downloader: default 16-bit family (bf16->fp16), fallback within family, never-empty selection, never-fp32-if-avoidable, keep optional model_index components, wire COZY_HF_* env overrides. (hf_selection.py/hf_downloader.py; +tests)
- [x] serve --function NAME (repeatable; boots only the hosting class(es); union; unknown name -> usage error). (cli/serve.py; +tests)
- [x] Verify sd1.5 local inference on the GPU (gen-worker run --method generate_sd15 -> 2.6GB download + non-blank 512x512 PNG).
- [x] serve --function: also accept a comma-separated list (--function a,b,c) in addition to repeated flags.
- [x] serve --list-functions: prints routable function names + hosting class and exits WITHOUT booting/loading any model (so you can pick a --function subset). (cli/serve.py; +test_cli_serve.py)
- [x] serve is now LAZY by default: boot indexes classes instantly and runs setup() (model load) on the FIRST invoke of one of its functions, held warm after — so the no-filter default no longer eagerly downloads every model on a multi-model endpoint. Added --eager to setup all at boot (fail-fast/pre-warm). (cli/serve.py; +tests for lazy/eager/multi-class-only-invoked-loads)

---

# #343: Test suite overhaul: integration-first, kill green-while-broken mocks

**Completed:** yes
**Status:** completed

TEST SUITE OVERHAUL — prefer few high-fidelity integration tests over many mocked ones. PHILOSOPHY (product owner, verbatim intent): "Almost all tests should be focused on testing the actual functionality, not mocking stuff. I'd rather have 10 good integration tests than 200 mocked useless ones that catch nothing. Focus on integration tests, not unit tests." This session two bugs that mattered — a `serve` command that died on stdin-EOF, and an HF download that fetched zero weight files — passed all 561 tests and were caught only by running the code by hand. Mock-heavy unit tests test the mock, not reality (green-while-broken). Baseline: ~561 tests / 52 files / 14.2k LOC; 32/52 files are mock-heavy (MagicMock/monkeypatch/patch), only ~10 do real subprocess/import/integration. TARGET: ~561 -> ~150-200 tests, integration share going from ~10 files to the majority. 

=== ELIMINATE (low-value: assert-a-mock-was-called, test stdlib/framework, tautological, or zero-signal duplicate coverage) ===
- test_accel.py (21 tests, 27 mock refs) + test_acceleration_helpers.py (19, 9 mocks) + test_no_accelerator.py (21, mostly import-surface asserts): ~61 tests that mostly assert `hasattr(module, name)`, mock.patch.dict(sys.modules,{'torch':None}) to force a no-torch branch, and check ImportError messages. These test that the module imports and that optional-dep shims raise — framework/import-surface, not behavior. COLLAPSE to ~6: one import-surface smoke test + one 'optional dep missing raises typed ImportError' parametrized test + keep the genuine no-GPU GpuCapabilityReport assertion. Drop the rest.
- test_download_model_cmd.py (11 tests, 29 mock refs — the single most mock-dense file): mocks the downloader, _send_message, _register_worker and inspects emitted pb messages. Tests the mock wiring of a handler. REPLACE its intent with the real download integration test (CREATE-c) + keep at most 2 thin cases asserting the WorkerModelReadySignal envelope shape. Drop ~9.
- test_presigned_upload_errors.py (10, 21 mocks) + test_remote_media_url_materialization.py (7, 14 mocks) + test_request_context_finalize_poll.py (5, 19 mocks): heavily mock HTTP/transport and assert call args. Keep 1-2 real error-path cases each against a local tmp/stub server; drop the call-arg-assertion tests (~15).
- test_incremental_cold_boot.py (4, 13 mocks) and test_clone_provider_boundaries.py (9, 12 mocks): mock the download/boot path and assert ordering of mocked calls. Fold the load-bearing ordering claim into the real cold-boot path or drop (~9).
- Tautology/duplicate: drop the pure provider-property/isinstance/whitespace-strip micro-asserts that exist in BOTH test_binding_api.py and test_api_binding.py (see COMBINE).

=== COMBINE (over-cover config permutations -> few parametrized cases) ===
The binding/routing permutation cluster is ~97 tests across 4 files, almost all table-driven permutations of the same {provider x kind x modifier} matrix:
  - test_api_binding.py (31), test_binding_api.py (23), test_provider_index_routing.py (23), test_override_provider_routing.py (20).
test_api_binding + test_binding_api overlap massively (immutability of modifiers, .allow_override class/str/dedup, wire round-trip, provider field, dispatch shape) — merge into ONE test_binding.py with parametrized cases over (Repo, HFRepo, CivitaiRepo) x (modifier). test_provider_index_routing + test_override_provider_routing both exercise build_provider_index_from_manifest + lookup_provider_for_ref + ModelRefDownloader routing + the safetensors gate — merge into ONE test_provider_routing.py parametrizing (binding-provider x override-provider x snapshot-weight-shape) and keeping the two named regressions (tag-stripping 2026-05-16, contextvar no-leak) as explicit cases. Collapse ~97 -> ~15-20 parametrized cases, NO loss of the real routing/gate/safetensors-rejection signal. Also collapse test_model_selectable_endpoints.py (26) variant/dispatch-shape permutations into ~8.

=== KEEP (load-bearing, guard real contracts / prior outages) ===
- test_concurrency_semaphore.py (23): GPU serialization that prevents the known 1640s thrash; drives a real SerialWorker dispatch under burst load + asserts accelerator=none skips the semaphore. Real behavior.
- test_serial_worker_dispatch.py (22) + test_serial_worker_async_dispatch.py (3): real registration + runtime dispatch (sync + async generator) through the worker, result/stream over the wire path.
- test_batched_inference.py (15, 0 mocks) + test_batched_worker_discovery.py (8, 0 mocks): decoration/shape contract + dispatch routing for the batched archetype.
- test_stage_decorator.py (16, 1 mock): SerialWorker multi-stage (TRELLIS-style) chaining contract.
- test_hf_selection_fallback.py (5, 0 mocks): the planner regression for the download-ZERO-weights bug — never empty, prefer 16-bit, fall back, keep optional components. Exactly the right kind of test. KEEP + expand.
- test_cli_serve.py (9): boots the real _Endpoint, runs the stdin NDJSON pipe, and round-trips invoke over a real unix socket via a serve SUBPROCESS with SIGINT teardown + --function filtering. Guards the stdin-EOF footgun. KEEP — this is the model for everything else.
- Named-outage regressions: test_binding_required_refs.py (0.7.21 readiness/flavor-canonical FLUX bug), test_discovery_submodules.py (2026-05-16 conversion-endpoint outage), test_resolved_repo_id_hf.py (HF model_id flavor-leak loading fp8/nf4 at bf16), test_split_streams_routing.py (queue routing), test_typed_payload_errors.py (typed exception attrs). Small, real, each guards a specific shipped bug.

=== CREATE (the priority — real integration tests that run the system, CPU-only, no GPU, no network) ===
(a) marco-polo end-to-end via BOTH `gen-worker run` AND `serve`+`invoke` over the unix socket, driven against the on-disk examples/marco-polo endpoint as a real subprocess; assert stdin-EOF on `serve` does NOT crash the process (would have caught this session's stdin-EOF footgun). Guards: CLI boot/teardown, EOF handling, socket transport, warm-serve lifecycle.
(b) @invocable/@inference discovery -> dispatch end-to-end on a tiny in-repo endpoint (discover_candidates -> _Endpoint.boot -> dispatch returns the right result, setup() runs exactly once). Guards: discovery + warm dispatch contract regressions.
(c) HF 16-bit selection planner against realistic fixture repo file-trees (fp16-only, fp32-only, .bin-only, sd1.5-shaped multi-folder) asserting NON-EMPTY + correct weights + optional components present (would have caught the download-nothing bug). Promote/expand test_hf_selection_fallback.py into the canonical planner integration suite over real fixture manifests.
(d) `serve --function NAME` actually boots ONLY the named class's hosting set (setup runs on Alpha, never Beta). Guards: accidental load of an unrelated model on a filtered serve.
(e) `run` auto-attach to a live `serve` (run.py _warm_serve_socket/_run_via_warm_serve, #340): start serve subprocess, run `gen-worker run` and assert it routes through the warm socket instead of cold-booting. Guards: the warm-attach path silently regressing to cold boot.
(f) GPU-GATED (mark @pytest.mark.skipif(no CUDA)): a real small-model load + generate through the worker, exercising the loader + GPU semaphore + a real weight load end to end. Clearly skipped on CPU CI.

This is ANALYSIS + PLANNING ONLY — no test code or library source is changed by this issue.

## Tasks
- [x] ELIMINATE: collapse test_accel.py + test_acceleration_helpers.py + test_no_accelerator.py (~61 import-surface/mock-the-stdlib tests) into ~6 (1 import smoke + 1 parametrized 'optional dep missing raises typed ImportError' + 1 no-GPU GpuCapabilityReport)
- [x] ELIMINATE: gut test_download_model_cmd.py (11 tests, 29 mock refs — most mock-dense file; mocks downloader/_send_message/_register_worker and asserts mock wiring) down to <=2 envelope-shape cases; its real intent moves to CREATE-(c) download integration
- [x] ELIMINATE: trim test_presigned_upload_errors.py (21 mocks) + test_remote_media_url_materialization.py (14 mocks) + test_request_context_finalize_poll.py (19 mocks) to 1-2 real error-path cases each vs a local tmp/stub server (~15 mock-only tests gone)
- [x] ELIMINATE: drop the mocked call-ordering tests in test_incremental_cold_boot.py (13 mocks) + test_clone_provider_boundaries.py (12 mocks) (~9) — fold any real ordering contract into the real cold-boot path
- [x] ELIMINATE: remove tautological provider-property / isinstance / whitespace-strip micro-asserts duplicated across test_binding_api.py and test_api_binding.py
- [x] COMBINE: merge test_api_binding.py (31) + test_binding_api.py (23) into one test_binding.py parametrized over (Repo,HFRepo,CivitaiRepo) x modifier x wire-round-trip
- [x] COMBINE: merge test_provider_index_routing.py (23) + test_override_provider_routing.py (20) into one test_provider_routing.py parametrized over (binding-provider x override-provider x snapshot-weight-shape); KEEP the tag-stripping 2026-05-16 regression + contextvar no-leak cases explicitly
- [x] COMBINE: collapse the ~97 binding/routing permutation tests (4 files) into ~15-20 parametrized cases with zero loss of routing/gate/safetensors-rejection signal
- [x] COMBINE: collapse test_model_selectable_endpoints.py (26) variant/dispatch-shape permutations into ~8
- [x] KEEP: test_concurrency_semaphore.py (GPU serialization, prevents 1640s thrash) — real SerialWorker burst dispatch
- [x] KEEP: test_serial_worker_dispatch.py + test_serial_worker_async_dispatch.py (real sync/async dispatch through worker)
- [x] KEEP: test_batched_inference.py + test_batched_worker_discovery.py + test_stage_decorator.py (archetype shape + dispatch routing)
- [x] KEEP: test_hf_selection_fallback.py (download-ZERO-weights regression) — expand into the planner integration suite
- [x] KEEP: test_cli_serve.py (real serve subprocess + socket invoke + SIGINT teardown; guards stdin-EOF) — use as the model for new integration tests
- [x] KEEP: named-outage regressions test_binding_required_refs.py, test_discovery_submodules.py, test_resolved_repo_id_hf.py, test_split_streams_routing.py, test_typed_payload_errors.py
- [x] CREATE integration test (a): marco-polo end-to-end via `gen-worker run` AND `serve`+`invoke` over the unix socket against examples/marco-polo; assert stdin-EOF does NOT crash serve (would have caught the stdin-EOF bug) [ALREADY EXISTS via #340/#341 agents — verified passing]
- [x] CREATE integration test (b): @invocable/@inference discovery -> _Endpoint.boot -> dispatch end-to-end on a tiny in-repo endpoint; setup() runs exactly once [ALREADY EXISTS via #340/#341 agents — verified passing]
- [x] CREATE integration test (c): HF 16-bit selection planner over realistic fixture repo file-trees (fp16-only / fp32-only / .bin-only / sd1.5-shaped) asserting non-empty + correct weights + optional components (would have caught the download-nothing bug) [ALREADY EXISTS via #340/#341 agents — verified passing]
- [x] CREATE integration test (d): `serve --function NAME` boots only the named class's hosting set (setup runs on Alpha, never Beta) [ALREADY EXISTS via #340/#341 agents — verified passing]
- [x] CREATE integration test (e): `run` auto-attach to a live `serve` (warm-serve socket #340) — assert it routes through the warm socket, not a cold boot [ALREADY EXISTS via #340/#341 agents — verified passing]
- [x] GPU real-model proof done MANUALLY on the 8GB RTX 4070: sd1.5 (min_vram_gb=6) via `gen-worker run --method generate_sd15` -> 2.6GB fp16 download + real 512x512 non-blank PNG. Also exercised SDXL: the 16-bit download fix fetched 6.5GB SDXL fp16 weights and the model loaded+ran, but OOM'd at the VAE upsampler — EXPECTED, since SDXLBase declares min_vram_gb=12 (turbo 10), above this 8GB card. NOT added to the fast unit suite by design (would need a 4GB+ model download + GPU on CI); it's an endpoint-repo e2e concern. The test_accel_capability GPU test (skipif no CUDA) covers the real device path in-suite.
- [x] Executed (two passes): 586 collected tests / 53 files -> 83 collected / 50 test-fns / 10 files; mock-heavy files 32 -> 1. All survivors are real integration tests; each of the 21 floor behaviors maps to a surviving test. Honest floor ~50 distinct tests — going lower drops real behaviors (provider routing alone has 9 distinct branches) or is count-gaming. Backups: /tmp/genworker-tests-backup (orig), /tmp/genworker-tests-pruned-417.

---

# #344: Stream handler events over the `serve` socket (currently buffered until completion)

**Completed:** yes
**Status:** completed
**Related:** Enables #353 (Tier B cancellation), Standalone precedent: run.py _write_stdout_event, Needed for prompt cancel + delivering the canceled envelope mid-flight: #352, #353

Stream handler events over the `serve` socket instead of buffering them until the request completes.

## Today

`_Endpoint.dispatch` collects every `write_event` into an `events` list (serve.py:426) and returns them all at once inside the final `{"ok": true, "events": [...]}` envelope. For a generator/streaming handler the client (`run` warm-attach / `invoke`) sees nothing until the entire handler finishes — no incremental yields, no progress.

## Goal

Emit each `yield`/event as its own NDJSON line on the connection as produced, terminated by a final result/summary line. This matches the standalone `run` path, which already writes events to stdout incrementally (`_write_stdout_event`).

## Why it matters here

- UX parity with standalone `run` for streaming endpoints.
- Prerequisite for #353 (Tier B cancellation) to be worthwhile: a streamed protocol lets the client show partial output and a confirmed cancel ack mid-flight.

Touches: serve.py `dispatch` / `_handle_conn` / `_write_response_line`, and invoke.py's single-response read loop (must become a multi-line read until a terminal frame).

## Tasks
- [x] serve: stream each event as an NDJSON line as produced; emit a terminal result/summary frame.
- [x] invoke client: read NDJSON lines until the terminal frame instead of a single response.
- [x] run warm-attach: forward streamed events to stdout via `_write_stdout_event` as they arrive.
- [x] Framing/back-compat: define the terminal-frame marker; keep error envelopes working.

---

# #345: Stop stale global `gen_worker` installs from shadowing tests; standardize on `uv run`

**Completed:** yes
**Status:** completed
**Related:** CLI run/serve/invoke under src/gen_worker/cli, build: pyproject.toml [tool.hatch.build] / [tool.pytest.ini_options]

Prevent a stale global `gen_worker` install from shadowing the working tree, and standardize the run/test entrypoint on `uv run`.

## Background (this session)

A user-global `gen-worker 0.1.4` was installed in `~/.local/lib/python3.12/site-packages` (an old editable whose origin `…/python-worker` no longer exists). It shadowed the current `src/` whenever a SYSTEM `python3` was used (e.g. `python3 -m pytest`), silently running/testing 0.1.4 instead of the 0.8.x working tree. The `.venv` (via `uv run`) was already correct. The 0.1.4 user-global install was removed this session; bare `python3 -c "import gen_worker"` now correctly fails.

## Goal

Make "tests run against the current source" the only outcome, and document it.

- Doc note in README/CONTRIBUTING: run and test exclusively via `uv run` (e.g. `uv run pytest`); never `pip install` the package globally / into user site.
- Guardrail: a `conftest.py` / pytest session hook that asserts `gen_worker.__file__` resolves under the repo `src/` and fails fast with a clear message otherwise — catches a reintroduced global shadow immediately.
- Optional: `[tool.pytest.ini_options] pythonpath = ["src"]` so even a bare `pytest` prefers `src/` (belt-and-suspenders; `uv run` remains the supported path).

## Tasks
- [x] Remove the stale user-global `gen-worker 0.1.4` install (`pip uninstall --break-system-packages gen-worker`).
- [x] Document the `uv run`-only workflow in README/CONTRIBUTING (no global pip installs).
- [x] Add a conftest guard asserting `gen_worker` imports from the repo `src/`, else fail with a clear message.
- [x] (Optional) `pythonpath = ["src"]` under `[tool.pytest.ini_options]`.

---

# #347: Explicit `serve` + `run`/`invoke` as the canonical local model; serve TCP/host:port transport + Docker submission patterns

**Completed:** yes
**Status:** completed
**Related:** Supersedes the dropped auto-spawn idea (decision recorded here), run auto-attach: #340, Cancellation on this model: #352 (request-cancel) / #353 (worker-stop), serve transport today: serve.py UDS AF_UNIX; invoke.py AF_UNIX (no TCP), Higher multi-endpoint layer: cozy-local (~/cozy/cozy-local)

## Decision

The canonical local-dev model is the EXPLICIT two-process shape, mirroring production/Docker:
- `gen-worker serve` = the worker (loads model into VRAM once, stays resident, accepts requests). Maps to the production worker CONTAINER (PID 1, long-lived).
- `gen-worker run` / `invoke` = submit a request to it, as many times as you want. Maps to the orchestrator sending a job.
- Two terminals: terminal 1 shows the worker (logs, model load, progress); terminal 2 submits requests. Ctrl-C in terminal 2 cancels the REQUEST (#352); Ctrl-C in terminal 1 stops the WORKER (#353).

`run` keeps #340 — it AUTO-ATTACHES to a `serve` the user already started, so the same `run` command works warm-against-serve or standalone-cold. Non-magical: it only attaches to something explicitly started.

### Dropped: auto-spawn `run`

An earlier idea — `run` auto-SPAWNS a background `serve` and self-reaps on idle — was DROPPED. The Docker framing showed the explicit two-process model IS the correct topology (same as prod), not ceremony to hide. Auto-spawn added spawn-race, idle-reap, per-endpoint-socket, and background-log complexity to paper over a workflow that is actually clearer when explicit. (`--cold` is moot — explicit `run` with no running serve is already the one-shot path.)

## New work: transport for the Docker / cross-process story

`serve` currently listens ONLY on a Unix domain socket (`AF_UNIX` at `./.gen-worker.sock`; invoke.py connects via `AF_UNIX`). That covers same-host / same-container submission but NOT host->container over a network port.

Docker patterns to support + document:
1. **`docker exec`** (works today): container `CMD ["gen-worker","serve"]`; submit via `docker exec <ctr> gen-worker invoke <fn> --payload '{...}'` (shares the in-container UDS).
2. **Bind-mounted socket** (works today): `-v /tmp/gw:/run` + `serve --socket /run/gw.sock`; host `invoke --socket /tmp/gw/gw.sock`.
3. **TCP / host:port listener (NEW):** optional `serve --listen tcp://0.0.0.0:PORT` alongside the UDS, and teach `invoke`/`run` to dial `tcp://host:PORT`, so `docker run -p PORT:PORT` + host-side submission works without exec/bind-mounts. UDS stays the default; TCP is opt-in. (Production dispatch is gRPC-from-orchestrator — this TCP mode is a LOCAL/dev convenience, not the prod transport.)

## Deployment shapes — `run` is RETAINED (decision)

`run` and `serve` are not redundant; they are two deployment topologies, both first-class:
- `gen-worker run` = ONE-SHOT job: load -> run -> emit result to stdout -> exit(code). Docker `docker run --rm`; Kubernetes **Job/CronJob**. For CI, fixtures, batch/scheduled inference, `| jq`. Shares the `dispatch_request` path with serve, so it is nearly free to keep.
- `gen-worker serve` + `invoke` = LONG-RUNNING service. Docker `docker run -d`; Kubernetes **Deployment/Service**.
- `gen-worker repl` = one-process load-once interactive dev session (#348).

Docker one-off job: `docker run --rm -v ~/.cache/cozy:/cache -e TENSORHUB_CAS_DIR=/cache <img> gen-worker run generate "a cat" seed=42`. CAS volume persists weights across runs; exit code propagates to CI / the k8s Job. A 'job image' default CMD is `["gen-worker","run"]`; a 'service image' is `["gen-worker","serve","--listen","tcp://0.0.0.0:PORT"]`.

Caveat: in PRODUCTION the worker container entrypoint is the gRPC worker that dials the orchestrator — `serve`/`run` are the LOCAL / self-hosted containerization shapes, not the prod transport.

## Tie-in

The cancellation suite (#352/#353) lands naturally here: the long-lived thing is `serve`; request-cancel vs worker-stop are different terminals. This per-endpoint worker is also the building block cozy-local (~/cozy/cozy-local) orchestrates across many endpoints.

## Tasks
- [x] Document the explicit serve + run/invoke model as canonical (README / docs/local-dev.md): two terminals, worker vs client.
- [x] Keep #340 run auto-attach to a user-started serve; ensure there is NO auto-spawn behavior.
- [x] serve: optional TCP listener `--listen tcp://host:port` alongside the UDS (UDS remains default).
- [x] invoke/run: dial `tcp://host:port` when given; default remains the UDS path.
- [x] Document Docker submission patterns: docker exec; bind-mounted socket; published TCP port (-p).
- [x] Tests: TCP round-trip (serve --listen tcp + invoke tcp://); UDS path unchanged; docker-exec smoke if CI has docker.
- [x] Document the deployment-shape mapping (run=Job/`docker run --rm`, serve=Deployment, repl=dev) + the CAS-volume one-off Docker recipe in docs.

---

# #348: Single-endpoint interactive REPL `gen-worker repl` (load-once; interactive sibling of `serve --stdin`)

**Completed:** yes
**Status:** completed
**Related:** Arg grammar: #350 (field=value), Request-cancel returns to prompt: #352, Schema for :schema + completion: #349 describe --json, Engine to reuse: serve.py _serve_stdin / _Endpoint, Rich multi-endpoint superset lives in cozy-local: cozy #20

## What

`cd my-endpoint && uv run gen-worker repl` — a single PROCESS that loads the endpoint + model ONCE and stays resident for the session, looping over typed requests. It is the interactive sibling of `serve --stdin` (which already loads once and reads NDJSON line-by-line via `_serve_stdin`); `repl` is that same engine plus a human prompt, the ergonomic arg grammar (#350), and meta-commands.

This is the THIN, SINGLE-endpoint REPL — bounded to the endpoint you're standing in. The RICH, multi-endpoint console (switch endpoints, schema completion across the catalog, warm/VRAM panel) is cozy-local's `cozy repl` (cozy #20); keep that there.

## Mechanic

1. Discover the endpoint (`endpoint.toml` -> functions); run `setup()` ONCE -> model resident in this process.
2. Loop: read a line -> parse (function + `field=value` args per #350, or raw JSON) -> dispatch in-process -> stream/print result -> back to prompt. Model stays warm between lines.
3. `Ctrl-C` cancels the IN-FLIGHT request (the #352 mechanism) and returns to the prompt; the session stays alive.
4. Exit (`Ctrl-D` / `:quit`) -> `shutdown()` -> free the GPU.
5. Meta-commands: `:use <fn>` (multi-function endpoints), `:schema` (fields/types/defaults/bounds from #349), `:help`, `:quit`. Tab-complete field names + bound hints from the schema.

## Implementation

Reuse serve's boot + dispatch internals (the `_Endpoint` object) with a readline loop instead of a socket — low duplication; the load-once + dispatch path already exists in `_serve_stdin`.

## Sits between the other shapes

- `run` = one-shot (load, run, exit) — scripting/CI.
- `serve` + `invoke` = two processes / two terminals — the Docker/prod topology (#347).
- `repl` = ONE process, load-once, interactive — poking one endpoint in one terminal.

## Tasks
- [x] `gen-worker repl` subcommand: discover endpoint, run setup() once (model resident), readline loop; reuse serve's `_Endpoint` / `_serve_stdin` engine.
- [x] Per-line parsing via the #350 `field=value` grammar (+ raw-JSON fallback); function inferred when single, else `:use <fn>` / `fn ...` prefix.
- [x] `Ctrl-C` cancels the in-flight request (#352) and returns to the prompt; `Ctrl-D`/`:quit` -> shutdown() -> free GPU.
- [x] Meta-commands: `:use`, `:functions`, `:schema` (from #349 describe), `:help`, `:quit`. (Scope note: tab-completion deferred — the thin repl reads line-buffered stdin, not GNU readline; the rich completion lives in cozy's repl.)
- [x] Stream events live + render result (path/value) + timings.
- [x] Tests: load-once (model not reloaded between lines), arg parsing, cancel-returns-to-prompt, clean shutdown on exit.

---

# #349: Stable host-integration surface for cozy-local: first-class `describe --json`, capability/protocol handshake, serve control sidecar, documented IPC protocol

**Completed:** yes
**Status:** completed
**Related:** Consumed by cozy-local: function-discovery (cozy #7), warm-serve/invoke (cozy #6), REPL (cozy #20), Current ad-hoc surface to replace: serve --list-functions --json (serve.py:127); cozy --help scraping (cozy internal/cli/serveproc.go gwSupportsSub), Protocol pairs with #352 (cancel) / #344 (streaming) / #347 (tcp transport), Introspection source: gen_worker.discovery, discovery/toml_manifest, endpoint.lock

## Why

cozy-local (~/cozy/cozy-local, the Go host that installs + runs endpoints) already drives gen-worker, but through an AD-HOC, version-skewed surface:
- Introspection is OVERLOADED onto `serve --list-functions --json` (serve.py:127) rather than a dedicated command.
- cozy SCRAPES `gen-worker <sub> --help` to feature-detect flags across versions (cozy internal/cli/serveproc.go `gwSupportsSub`/`gwSupports`; comment: "PyPI 0.8.0 lacks --list-functions --json").
- cozy REIMPLEMENTS serve lifecycle (pidfile/socket/ready/idle) because gen-worker exposes no machine-readable handle.

Make gen-worker expose a STABLE, VERSIONED, machine-readable contract so any host orchestrator (cozy-local first) integrates cleanly and stops guessing. This is the python-gen-worker half of 'make it easy for cozy-local to interact with endpoints'.

## Surface to define

1. **First-class `gen-worker describe --json` (no model load).** Promote introspection off `serve`. Stable document:
   - `protocol_version`, `gen_worker_version`
   - `capabilities`: what the host can rely on (`describe`, `cancel`, `streaming`, `tcp_listen`, `prefetch`, ...) — REPLACES `--help` scraping.
   - `endpoint`: kind, main module, classes.
   - `functions[]`: wire name, class.method, INPUT JSON Schema (from the msgspec Struct: types, defaults, ge/le/multiple_of bounds), output type, `is_generator`, model bindings (param -> {provider, ref, tag, flavor, version pin, allow_override, allow_patterns}), declared build profiles.
   Keep `serve --list-functions --json` as a thin alias of `describe` for back-compat.
2. **Capability / protocol handshake.** `describe --json` (and the serve sidecar) carry `protocol_version` + `capabilities`; cozy keys behavior off that instead of `--help` scraping. Bump `protocol_version` on wire-format changes (#352/#344/#347).
3. **serve control sidecar (machine-readable lifecycle).** On boot+ready, `serve` writes a sidecar JSON next to the socket (e.g. `.gen-worker.serve.json`): `{pid, listen (uds path / tcp host:port), protocol_version, ready_at, functions, idle_timeout}`; removes it on teardown; also emits an explicit `ready` event. cozy reads this instead of reinventing pidfile/socket/ready guesswork.
4. **Documented IPC protocol.** The NDJSON wire contract cozy's client already speaks ad hoc, written down + versioned: request `{request_id, function, payload}`, streamed events (#344), terminal result frame, error-envelope kinds, and the cancel control message (#352). Put it in `docs/host-integration.md` as the supported surface.
5. **TCP transport (#347)** advertised in `capabilities` when present.

## Principle

gen-worker stays the SINGLE-endpoint machinery with a stable, versioned, machine-readable contract; cozy-local is the MULTI-endpoint host that consumes it. Treat the contract (describe schema + protocol + sidecar + capabilities + exit codes) as semi-public API governed by `protocol_version`.

## Tasks
- [x] First-class `gen-worker describe --json` (no model load): protocol_version, gen_worker_version, capabilities, endpoint kind/classes, functions[] with input JSON Schema + bounds + output + is_generator + model bindings + profiles.
- [x] Keep `serve --list-functions --json` as a thin alias of `describe` (back-compat).
- [x] capabilities + protocol_version handshake; coordinate removing cozy's `--help` scraping (cozy serveproc.go gwSupports*).
- [x] serve ready-sidecar `.gen-worker.serve.json` {pid, listen, protocol_version, ready_at, functions, idle_timeout}; write on ready, remove on teardown; explicit `ready` event.
- [x] docs/host-integration.md: describe schema, NDJSON IPC protocol (request_id/function/payload, streamed events, terminal frame, error kinds, cancel msg), sidecar, capabilities, exit codes; versioned by protocol_version.
- [x] Advertise tcp_listen + cancel + streaming in capabilities as #347/#352/#344 land.
- [x] Tests: describe --json schema snapshot; sidecar lifecycle; capability handshake; serve alias parity.

---

# #350: Ergonomic schema-driven payload args for run/invoke: typed `field=value` (httpie-style) instead of raw JSON

**Completed:** yes
**Status:** completed
**Related:** Schema source: #349 describe --json + msgspec Struct (_decode_payload/apply_payload_constraints, run.py), Consumed by cozy invoke key=value (cozy #6) + REPL (cozy #20), Prior art: httpie request-item grammar (=, :=, @)

## Why

Today `run`/`invoke` take payloads only as raw JSON (`--payload '{"positive_prompt":"hello","seed":123}'`) — command-line quoting + brackets are painful. The payload type is a known `msgspec.Struct` (the same one #349's `describe` exposes), so the CLI can accept TYPED key/value tokens and assemble + COERCE the payload itself.

## Proposed grammar (httpie-inspired, schema-coerced)

`gen-worker invoke flux/generate "a cat" seed=123 guidance=3.5 hires=true`

- `field=value` -> set `field`; value COERCED by the schema type (str/int/float/bool). `seed=123` -> int because the Struct says int; `prompt=hi` stays str.
- `field:=rawjson` -> raw JSON value for lists/objects/explicit types: `tags:='["a","b"]'`, `size:=1024`.
- `field@path` -> load the field's value from a FILE (long prompts, images, audio): `image@./cat.png` (bytes/base64 per field type).
- bare positional -> the endpoint's declared PRIMARY field (e.g. the prompt), so `invoke flux/generate "a cat"` works.
- dotted keys for nested structs: `sampler.steps=30`.
- `--payload '<json>'` stays as the escape hatch; `field=` tokens merge OVER it.

### Syntax (DECIDED: httpie-style `field=value`)

Bare `field=value` tokens. Chosen over `--arg:field value`: terser, and leading-`--` stays reserved for the TOOL's own flags (`--socket`, `--timeout`, `--device`) so user fields never collide with them. (`--arg:` rejected as verbose.)

## Coercion / validation

Reuse the existing msgspec decode + payload-constraints path (`_decode_payload` / `apply_payload_constraints`) so clamps, bounds (ge/le/multiple_of) and error messages are IDENTICAL to production. Edge cases: bool = true/false/1/0/yes/no; null via `field:=null` or empty `field=` only if Optional; repeated `field=` builds a list (or `field:=[...]`); unknown field -> error LISTING valid fields (typo catcher).

## Home / reuse

gen-worker owns payload decoding + the msgspec Struct, so the precise coercion lives in `run`/`invoke`. cozy-local's `cozy invoke ... key=value` (cozy #6) and the REPL (cozy #20) reuse THIS one grammar — either by delegating to gen-worker's arg parser or by reproducing it from `describe --json`. One coercion spec, documented once.

## Tasks
- [x] Parse `field=value` / `field:=json` / `field@file` / dotted-nested / bare-primary tokens in run + invoke; merge over optional --payload.
- [x] Coerce against the function's msgspec Struct types (str/int/float/bool/Optional/list/nested); reuse `_decode_payload` + `apply_payload_constraints` for validation.
- [x] Derive the 'primary' field for the bare positional (decorator hint, else first required str field).
- [x] Friendly errors: unknown field -> list valid fields; bad coercion -> field-level message matching production.
- [x] Document the arg grammar in docs/local-dev.md + host-integration.md so cozy invoke (#6) / REPL (#20) reuse one spec.
- [x] Tests: coercion matrix (int/float/bool/list/nested/optional), := raw json, @file, primary positional, unknown-field error.

---

# #351: Civitai ref resolution in `gen-worker run`: model→latest-version mapping, version pinning, fail-loud

**Completed:** yes
**Status:** completed
**Related:** Builds on warm-serve auto-attach #340, src/gen_worker/models/refs.py CivitaiRef, src/gen_worker/api/binding.py CivitaiRepo (.version / _version_id), downloader: conversion/ingest.fetch_civitai_model + download_civitai_model_version_files

Correct Civitai model-ref resolution in the local `gen-worker run` resolver (`_resolve_local_path`, src/gen_worker/cli/run.py). Largely implemented this session on top of a merge that left the path broken; remaining work is tests + commit.

## What was broken

1. **Merge dead code.** A merge kept both branches' trailing "unsupported model ref" raise, leaving a `raise _ModelResolutionError(...)` ABOVE the Civitai branch — so the entire Civitai download path was unreachable; any civitai ref died with "unsupported model ref" instead of fetching.
2. **Model id treated as version id.** `CivitaiRepo`'s ref is a MODEL id by convention (binding.py:367), but the resolver passed it straight to `download_civitai_model_version_files` as a VERSION id.
3. **Version pins ignored.** `CivitaiRepo.version("<id>")` pins (`_version_id`) never flowed through `_resolve_binding_to_ref`/`CivitaiRef`, so a pinned version was silently never honored (always downloaded latest).
4. **Fail-silent fallback.** A `try/except: version_id = ref_id` fallback meant a failed model lookup silently re-interpreted the id as a version id and could download an unrelated model (model-id and version-id integer spaces overlap).

## What changed (done this session, uncommitted)

- Removed the premature catch-all raise; a single catch-all (provider in the message) remains at the end of `_resolve_local_path`.
- Unpinned refs: map model id → latest version via `fetch_civitai_model(model_id).modelVersions[0].id`.
- Pinned refs: thread `binding._version_id` → `_resolve_local_path(civitai_version_id=...)` (mirrors how `_allow_patterns` is threaded), guarded against payload `_models` overrides; when pinned, use the id directly and skip the model lookup.
- Fail loud: removed the silent fallback; raise `_ModelResolutionError` when the model lookup fails or the model has no published versions.

## Remaining

Tests + commit (working tree currently dirty/unstaged).

## Tasks
- [x] Remove the dead-code premature `raise` from the merge; keep one catch-all (provider in message).
- [x] Map Civitai model id → latest version id via `fetch_civitai_model`.
- [x] Honor `CivitaiRepo.version()` pins: thread `_version_id`, guard against `_models` overrides, use the pinned id directly (skip model lookup).
- [x] Fail loud on failed model lookup / no published versions (remove `except: version_id = ref_id`).
- [x] Add `tests/test_run_civitai.py`: pinned-skips-lookup, bad-model-id-raises, no-versions-raises, happy-path-picks-modelVersions[0] (stub `fetch_civitai_model` / `download_civitai_model_version_files`).
- [x] Commit the working-tree changes (landed in 568e72a).

---

# #352: Unified request cancellation in the CLI: reuse the production `ctx.cancel()` + per-request registry so `Ctrl-C` cancels one request and the server keeps running

**Completed:** yes
**Status:** completed
**Related:** Production model to reuse: worker.py _active_requests (504/7304), _handle_interrupt_request (7363), interrupt_job_cmd dispatch (5881), Primitive: RequestContext.cancel()/is_canceled()/raise_if_canceled()/cancel_event (request_context/__init__.py:587), Pairs with #353 (signal mapping) and #344 (streaming), Cross-repo contract: #346 (orchestrator user-facing cancel → interrupt_job_cmd)

## Principle

There is exactly ONE cancellation primitive — `RequestContext.cancel()` (request_context/__init__.py:587) — observed by tenant code via `ctx.is_canceled()` / `ctx.raise_if_canceled()` (the documented canonical idiom) / `ctx.cancel_event` (a `threading.Event`, via `ctx.done()`). Every cancel SOURCE resolves a `request_id`, looks the ctx up in a per-request registry, and calls `ctx.cancel()`. Cancelling a request NEVER tears down the server/endpoint — teardown is a separate lifecycle action (#353).

This already exists in the PRODUCTION worker and is the model to reuse, NOT reinvent:
- Registry: `self._active_requests: Dict[request_id, RequestContext]` (worker.py:504; registered 7304, popped in `finally`).
- Source: orchestrator → `interrupt_job_cmd{request_id}` over the gRPC stream → `_dispatch` (worker.py:5881) → `_handle_interrupt_request` (worker.py:7363) → `ctx.cancel()` (+ `engine.abort(rid)` for BatchedWorker, 7378).

The CLI (`serve`/`run`/`invoke`) does NOT participate today: `_Endpoint.dispatch` builds a fresh ctx (serve.py:436) but never registers it and offers no way to trip it from outside the synchronous handler; there is no `request_id` in the socket protocol; and standalone `run`'s `_SigintHandler` pokes `ctx._canceled`/`_cancel_event` privately instead of calling `ctx.cancel()`.

## Goal

`Ctrl-C` on a `run`/`invoke` client (or any external cancel) cancels THAT request via `ctx.cancel()`; the `serve` process and its loaded endpoint keep running for the next request — exactly mirroring the orchestrator's per-request `interrupt_job_cmd`.

## Design (complete; no legacy/back-compat constraints)

1. **request_id everywhere.** The CLI NDJSON request carries a `request_id` (client-supplied or server-assigned and echoed in the first response frame). It is the cancellation key, matching production.
2. **Per-request registry in serve.** Mirror worker's `_active_requests`: `Dict[request_id, RequestContext]`, registered when dispatch starts, popped in `finally`. Factor a shared helper so worker + CLI serve use ONE implementation.
3. **Run handlers off the accept thread.** Keep request EXECUTION serialized (existing `_dispatch_lock` + GPU semaphore), but run the handler on a dispatch thread so the accept loop stays free to receive an out-of-band cancel while a request is in flight — exactly how production keeps the gRPC stream live for `interrupt_job_cmd` while jobs run on the job-queue thread.
4. **Cancel control message — the CLI analog of `interrupt_job_cmd`.** A `{"cancel": {"request_id": ...}}` request (on a fresh short-lived connection; it does NOT need the original request's conn) → registry lookup → `ctx.cancel()` → return immediately. Expose an `interrupt_request(request_id)` method on the shared endpoint object.
5. **Client `Ctrl-C` → cancel its own request.** Two-stage SIGINT in `run` warm-attach + `invoke`: 1st `Ctrl-C` opens a fresh connection, sends `{"cancel":{request_id}}`, keeps waiting for the (canceled) response to print; 2nd `Ctrl-C` → `os._exit(130)` the CLIENT only (never the server).
6. **Client-disconnect backstop.** If the request connection drops (client crash / 2nd Ctrl-C), serve trips `ctx.cancel()` for that request_id too, so abandoned work never runs on forever (the local analog of the orchestrator auto-cancelling on stream drop — #346).
7. **Standalone `run` uses the public API.** `_SigintHandler` (run.py) calls `ctx.cancel()` instead of setting `_canceled`/`_cancel_event` directly; two-stage behavior preserved (2nd Ctrl-C → `os._exit(130)`).
8. **BatchedWorker parity.** The CLI cancel path also calls `engine.abort(request_id)` like `_handle_interrupt_request` (worker.py:7378) where a batched engine is in play.

## Caveat

Prompt cancellation requires cooperative tenant code (`raise_if_canceled()` in loops, or waiting on `ctx.cancel_event`). A single-shot handler stuck in a tight CUDA/C call won't observe cancel until it returns — same as production. Truly freeing the GPU then needs server teardown (#353), a deliberate separate action.

## Tasks
- [x] Add `request_id` to the CLI serve/invoke NDJSON protocol (client-supplied or server-assigned + echoed in the first response frame).
- [x] Per-request `Dict[request_id, RequestContext]` registry in serve (register on dispatch, pop in finally), MIRRORING the production worker's `_active_requests` pattern. (Scope note: did not extract a literally-shared module — the two registries stay independent; revisit if duplication grows.)
- [x] Run each request handler on a dispatch thread so the accept loop can receive cancels concurrently (keep `_dispatch_lock` + GPU-semaphore serialization).
- [x] `{"cancel":{request_id}}` control message → `endpoint.interrupt_request(request_id)` → `ctx.cancel()` (+ `engine.abort` for batched).
- [x] run warm-attach + invoke: two-stage SIGINT — 1st C sends cancel for its request_id and waits for the canceled response; 2nd C `os._exit(130)` client only.
- [x] Client-disconnect backstop: a dropped request connection trips `ctx.cancel()` for that request_id.
- [x] Standalone run `_SigintHandler`: call public `ctx.cancel()` (not `_canceled`/`_cancel_event`).
- [x] Tests: cancel mid-stream cancels only that request; server serves a subsequent request; disconnect backstop fires; standalone-run two-stage.

---

# #353: Signal → cancellation mapping: SIGINT/SIGTERM funnel through `ctx.cancel()`; separate request-cancel from server teardown (worker + CLI)

**Completed:** yes
**Status:** completed
**Related:** Built on the #352 primitive + registry, Production precedents: worker_drain_cmd dispatch (worker.py:5887), serve teardown (serve.py:823-873), SIGINT precedent: run.py _SigintHandler

OS signals must drive the SAME `ctx.cancel()` mechanism (#352) so endpoint/tenant code stops cleanly — the local analog of the orchestrator's `interrupt_job_cmd` / `worker_drain_cmd`. Killing the server and cancelling a request are DISTINCT actions:

- **Cancel a request** (client `Ctrl-C`, orchestrator interrupt, socket cancel): `ctx.cancel()` for ONE request_id; server + endpoint keep running. (#352.)
- **Stop the server** (server-terminal SIGINT; SIGTERM): cancel ALL in-flight ctx, let them unwind, run `shutdown()`, exit.

## Mapping (both `gen-worker serve` and the production worker)

- **SIGINT to `serve`'s own terminal:** iterate the per-request registry → `ctx.cancel()` for each → bounded drain → `endpoint.shutdown()` → exit. Replaces serve's current `stop`-event + 2s daemon-thread `join` (serve.py:865) which never cancels in-flight work.
- **SIGTERM (orchestrator/k8s graceful stop, to worker or serve):** same as the production `worker_drain_cmd` path (worker.py:5887) — cancel in-flight via `ctx.cancel()`, await unwind, `instance.shutdown()` per class, exit 0. Wire SIGTERM to that drain entry point.
- **SIGKILL:** uncatchable by definition — bypasses all cleanup (no `shutdown()`, no GPU free). Document as the hard-kill of last resort; the two-stage client `Ctrl-C` (#352) and bounded drain timeouts exist precisely so SIGKILL is rarely needed.
- **Standalone `run` SIGINT:** two-stage via the public `ctx.cancel()` (1st = cancel, 2nd within ~2s = `os._exit(130)`) — implemented under #352.

## Tenant contract (single source of truth, documented)

Tenant/endpoint code observes cancellation IDENTICALLY in prod and locally: `ctx.raise_if_canceled()` in loops (canonical), or wait on `ctx.cancel_event` / `ctx.done()`. One idiom, every cancel source.

## Tasks
- [x] serve own-SIGINT: cancel all in-flight ctx via the registry, bounded drain, then `endpoint.shutdown()` (replace the 2s join at serve.py:865).
- [x] Wire SIGTERM (worker + serve) to the existing drain path (`worker_drain_cmd` semantics): cancel in-flight, await, shutdown, exit 0.
- [x] Document SIGKILL as uncatchable / no-cleanup; add bounded drain timeouts so it is rarely needed.
- [x] Document the tenant cancellation idiom (raise_if_canceled / cancel_event) as identical in prod and CLI.
- [x] Tests: serve SIGINT cancels in-flight + shuts down; SIGTERM drains; a single request-cancel (#352) leaves the server running (contrast).

---

# #355: Unblock async concurrency past executor width + exit on persistent auth rejection

**Completed:** yes
**Status:** completed
**Related:** tensorhub #462-T4 (auth give-up), gen-worker #447 (load-test concurrency), #338 (reconnect backoff), #345 Improvement B (async handlers on shared loop)

## Fix 1 (#447 follow-up): async SerialWorker dispatch no longer consumes a job-executor thread for the await

Pre-fix, `async def` handlers ran on the shared asyncio loop BUT each job's dispatcher thread blocked on `run_coroutine_threadsafe(...).result()`, so the ThreadPoolExecutor default width min(32, cpu+4) capped async concurrency at ~32. Now `_execute_serial_class_request` branches async specs to `_execute_serial_async_request`: blocking pre-work (GPU semaphore acquire, lazy setup, payload decode, #337 injection) stays on the pooled dispatcher thread, then the tenant coroutine is scheduled onto the shared loop and the thread returns immediately. Completion (encode + `_send_request_result` + GPU busy/semaphore release) runs on the loop in `_run_serial_async_request` / `_finish_serial_async_request` (precedent: the BatchedWorker path already sends results from the loop thread). `_drain_async_incremental` became loop-native (`async for` instead of per-item cross-thread futures). Sync handlers keep the executor unchanged. Invariants preserved: GPU semaphore acquired before scheduling and released exactly once; ctx.cancel()/interrupt maps to a terminal `canceled` result; one terminal result per request; delta-then-done ordering within the single coroutine.

## Fix 2 (tensorhub #462-T4): exit on persistent capability-token rejection

Reconnect already had bounded full-jitter backoff (#338), but a worker whose token is revoked/expired spun forever (observed zombies). Now UNAUTHENTICATED/PERMISSION_DENIED at connect/register (`_connect_once`) or on the control stream (`_receive_loop`) increments a CONSECUTIVE auth-failure counter; after GEN_WORKER_MAX_AUTH_FAILURES (default 10, 0 disables) the worker logs 'capability token rejected N times; exiting — token is likely revoked/expired' and exits (`_fatal_exit = os._exit`, injectable for tests). Any inbound scheduler message resets the counter; transient network errors neither count nor reset.

## Tasks
- [x] Branch async serial specs out of the blocking dispatch body (_execute_serial_async_request / _run_serial_async_request / _finish_serial_async_request).
- [x] Make _drain_async_incremental loop-native (no per-delta cross-thread future round-trips).
- [x] Auth-failure counter + exit: _is_auth_rejection, _note_auth_failure, reset in _process_message, hooks in _connect_once + _receive_loop.
- [x] Tests: 64 sequential dispatches all simultaneously in flight (>32); failure + canceled terminal results via the callback path; auth counter consecutive/reset/exit + receive-loop wiring + transient codes don't count.
