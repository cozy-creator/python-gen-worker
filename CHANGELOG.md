# Changelog

## 0.38.6 (2026-07-19)

- **gw#593 companion: publish_as_is's zero-cost passthrough never resharded
  an oversized monolithic weight file.** `run_clone`'s `tree = source.dir`
  shortcut (dtype already matches, no cast needed) bypasses
  `build_flavor_tree` entirely — every one of ITS branches ends in
  `_stage_oversize_safetensors` — so a source shipping ONE oversized
  safetensors file with no HF-convention shards to begin with (exactly
  `Lightricks/LTX-2.3`'s 46GB `ltx-2.3-22b-dev.safetensors`) was published
  raw, and Tensorhub's commit API rejected it (`request_too_large: file
  exceeds max_bytes_per_file`). Found live: e2e#185 ltx-firstlight run 8.
  Now hardlinks into a scratch tree and reshards only when something is
  actually oversize — the common case (already-sharded sources) stays the
  zero-cost passthrough.

## 0.38.5 (2026-07-19)

- **gw#592/gw#593 companion: disk preflight didn't know about LTX-2's
  publish_as_is routing.** `run_clone` routes `strategy="aio_singlefile"`
  LTX-2 sources through publish_as_is regardless of the requested output
  layout (gw#592 — no diffusers pipeline exists for the family), but
  `_preflight_disk` only sees the pre-download classification and had no
  equivalent carve-out, so it budgeted a full layout-repack +
  materialized-dtype-tree estimate (388GB for a 43GB source) for a clone
  that only ever needs the source bytes + margin. Preflight now derives the
  same LTX-2 hint from the pre-download file listing
  (`layout.infer_model_family_variant_from_hint` on each path — the
  filename itself carries the "ltx2" token) and applies the
  publish-as-is budget. Found live: e2e#185 ltx-firstlight run 7,
  `CloneDiskSpaceError` on a real 43GB LTX-2.3 dev-checkpoint clone.

## 0.38.4 (2026-07-19)

- **gw#593 item 2: `source_include` — explicit source-file selection on the
  clone request.** `Lightricks/LTX-2.3` bundles dev/distilled/distilled-lora/
  upscaler checkpoints at repo root; even with item 1's fix, the classifier
  groups all of them into one bundle (over the 100GB size gate) because it
  has no way to know which one the caller wants. `source_include` is a new
  optional clone field, dual-form like the rest of the clone request surface:
  a compact single glob string, or a structured list of globs, matched
  against repo-relative paths. When given, only the matching subset ever
  reaches `classify_repo` — every existing strategy branch keeps working
  unchanged on the narrowed listing. Every glob MUST match >=1 file; an
  unmatched glob (typo, stale pattern) is a loud, typed
  `SourceIncludeError` naming the bad glob and what the other globs matched,
  never a silent no-op. HuggingFace-only for now (civitai clones raise if
  given).

## 0.38.3 (2026-07-19)

- **gw#593: classifier._variant_tag must not match embedded version numbers.**
  A repo whose root safetensors filenames carry their own dotted version
  number (`ltx-2.3-22b-dev.safetensors`) had that number misread as a
  diffusers dtype-variant suffix, silently excluding the real checkpoint from
  a multi-file bundle in favor of an unrelated smaller file. Now only a
  recognized dtype token (bf16/fp16/fp32/fp8/...) counts as a variant tag;
  everything else is untagged. Turns a silent wrong-file publish into a loud
  refusal for oversized bundles instead.

## 0.38.2 (2026-07-19)

- **gw#592: LTX-2.3 family detection + native publish routing.**
  `Lightricks/LTX-2.3` (monolith or DiffSynth-Studio repackage layout) has no
  diffusers pipeline manifest, so family detection stamped 'unknown' and a
  requested `{dtype bf16, layout diffusers}` clone died "no publishable
  flavor" trying to repackage into a diffusers layout that doesn't exist for
  this family. Now detects `model_family='ltx2'` (filename hint + a
  repo-structure sentinel for the repackage layout) and routes it through
  the existing publish_as_is/th#901 dtype-cast path — the te#70 trainer
  resolves the native singlefile snapshot directly, so no repackager is
  built. Other aio_singlefile families (sd15/sdxl/flux/zimage) unaffected.

## 0.38.1 (2026-07-19)

- **gw#591: finish boot setup when hub-delivered snapshots arrive.** The
  startup scan parks class+models functions whose tensorhub refs are not yet
  local; the hub's desired-disk plan delivers them seconds later but nothing
  re-ran setup — the worker never advertised the function and the request
  queued forever (cold-boot deadlock, found live in ie#519). A boot watcher
  now completes setup when the awaited refs land and pushes a StateDelta.

## 0.38.1 (2026-07-19)

- **gw#588: reconcile resident low-VRAM prep mode to the cell's traced
  mode.** `off` and `vae_only` are both fully-CUDA-resident preps differing
  only in the vae-slicing/vae-tiling/attention-slicing flag groups. When a
  delivered cell's `low_vram_mode` and the pipeline's current mode are both
  resident and differ, both consumer arm paths (`enable()` + hot adopt) now
  converge the pipeline to the cell's traced mode before the drift check
  instead of refusing — the ie#501 run-18 mandatory-w8a8 starvation
  (producer mints alone → 'off'; multi-lane serve load → 'vae_only').
  Offload-mode drift keeps refusing: genuinely different graphs/residency.

## 0.38.0 (2026-07-18)

- **gw#590: root w8a8 generality — nested multi-set layouts, weight-set
  selector, pipeline-class key_map hook.** Non-diffusers w8a8 sources scan
  the whole tree for weight sets (split-checkpoint layouts nest component
  files); `streaming_w8a8_snapshot(weight_set_patterns=)` selects the
  denoiser set(s) when several exist and the rest pass through
  byte-identical (CAS-dedup against the source mirror, no stray scale
  twins). `detect_w8a8_artifact`/`verify_w8a8_snapshot` recurse the same
  way. `load_w8a8_root_pipeline` forwards a pipeline-class-declared
  `_cozy_w8a8_key_map` (staticmethod) to `swap_w8a8_linears` for
  converter-renamed families. Root-lane produce results now report the
  selected weight-set rel paths in `components` (was `[""]`).

## 0.37.5 (2026-07-18)

- **gw#589/th#901: publish_as_is clone strategies cast an explicitly
  requested dtype instead of silently swallowing it.** `run_clone`'s
  publish_as_is branch (transformers/diffusers_component/peft/
  sentence_transformers/native_lora/gguf/pipeline_tree) special-cased
  `spec.dtype != "bf16"`, making an explicit bf16 request indistinguishable
  from `normalize_outputs`' unspecified-request default — an explicit bf16
  ask against a non-bf16 source silently republished the source untouched,
  no error surfaced. `explicit_outputs` now gates passthrough vs cast: an
  explicit mismatch on a cast-eligible dense-safetensors strategy runs a
  real cast via `build_flavor_tree`; any other mismatch fails loud.

## 0.37.4 (2026-07-18)

- **gw#586: mints trace through the serving pipeline class.** Traced FX graphs
  depend on the pipeline's CALL path, not just the module tree: the serving
  Condition class drives LTX's DiT with per-token timestep/modulation tensors
  while `build()`'s generic load broadcast them — no generic-load cell ever
  served a serving-path lookup (pre-proof fleets silently cold-compiled the
  real graphs at boot). `build(pipeline_class=...)` +
  `resolve_pipeline_class()` load through the named serving class; unknown
  names refuse loudly. Class stamped in metadata for observability; the ck1
  key and graph_signature stay class-agnostic.

## 0.37.3 (2026-07-18)

- **gw#579: reclaim idle on-disk checkpoint cache under host-RAM pressure.**
  When prior rotations have already moved every old pipeline to DISK, admission
  now advises clean pages from the oldest immutable snapshots out of the file
  cache one ref at a time and re-probes exact headroom after each. Incoming,
  loaded, executing, and shared inodes remain protected; model bytes stay local.

## 0.37.2 (2026-07-18)

- **gw#579: reclaim idle checkpoints behind shared-reference pins.** Host-RAM
  admission now tests whether the checkpoint selected for eviction is in use,
  so an incoming job's pin on a shared VAE no longer freezes an unrelated idle
  pipeline. Vacating the idle record still preserves the pinned shared asset.

## 0.37.1 (2026-07-18)

- **gw#584: defer compile-declared endpoints from eager boot setup.** `Lifecycle.startup()`
  raced `transport.run()`'s HelloAck handshake: a `spec.compile`-declared function with
  locally-present weights could reach `ensure_setup` at boot with bare authored refs and
  `snapshots=None`, silently skipping compile-cell selection while a later HelloAck-driven
  setup materialized the resolved w8a8 lane — selection and materialization derived from
  different resolved states, fail-closing `enable()` generically (ie#501 run 17). Compile
  cells now defer the same way `Slot` picks already do (pgw#532): both arrive only via hub
  delivery, never a boot default.

## 0.37.0 (2026-07-18)

- **gw#581 (th#883): worker-owned cell selection.** New `gen_worker.cell_key`
  module — the ONE compatibility brain: the worker computes its exact
  compile-cell identity (`ck1-<56hex>` over the honest axes; `cuda_driver`
  excluded) and requests cells BY KEY. Protocol additions (additive):
  `CompileTarget.requested_cell_key`/`requested_cell_axes`,
  `StateDelta.cell_lookups`. Mints stamp the key their axes describe;
  `build(requested_cell_key=…)` refuses a mismatched mint. A SELF-REQUESTED,
  identity-verified cell that fails to arm surfaces as `cell_selection_bug`
  (loud, wire-visible), never a silent eager fallback; cozy-local verdicts
  share the same brain. Key-flavored cells (`#ck1-…`) ride the existing
  delivery rails.

## 0.36.1 (2026-07-17)

- **gw#567: prompt-corpus / eval-set artifacts drop parquet — jsonl
  shards.** `convert.dataset.Dataset` and `conversion.prompt_corpus`
  (training-endpoints) wrote/read `data/train-*.parquet` via pyarrow;
  no internal consumer needed columnar storage at these row counts.
  Switched to `data/train-*.jsonl` (one JSON object per row; `bytes`
  columns base64-wrapped as `{"__bytes_b64__": ...}`). `Dataset.shards()`
  replaces `parquet_shards()`; `write_jsonl_shard()` is the shared
  writer. Dropped the `[datasets]` pyarrow extra — no lazy pyarrow
  import left in this codepath. Pre-launch: no back-compat reader for
  old parquet corpora, regenerate.

## 0.36.0 (2026-07-17)

- **gw#564: sm_89 W8A8 inference lane — per-tensor fp8 GEMM + per-channel
  epilogue rescale (4090/L40S).** ie#498 measured rowwise-scaled
  `_scaled_mm` W8A8 as NO-GO on sm_89 (+79% compiled): torch's fast
  rowwise kernels are CUTLASS sm_90+ and Ada falls to a ~half-rate
  fallback — a kernel gap, not silicon. `Fp8ScaledLinear` gains a second
  dispatch branch chosen ONCE at load by SKU: `gemm_mode="pertensor"`
  runs a scalar-scaled fp8 GEMM (cuBLASLt's Ada fast path, per-TENSOR
  dynamic activation scale) and applies the SAME per-channel
  `weight_scale` vector as a post-GEMM column-multiply epilogue (bias
  after the rescale; fuses under inductor) — mathematically identical to
  the rowwise lane, ONE weight artifact serves both (no new producer or
  flavor). The capability probe is replaced by `w8a8_gemm_mode()`:
  candidates per SKU class arm only when the kernel call succeeds AND a
  load-time micro-benchmark GEMM beats the bf16 reference (probe-pass ≠
  profitable, the ie#498 lesson — generalizes the gate for every future
  SKU); `scaled_mm_supported()` is gone, loader modes are now
  `rowwise`/`pertensor`/`dequant`. The gw#558 additive LoRA branch rides
  the epilogue lane unchanged; lane stamp stays `w8a8` for both GEMM
  branches (cells are per-SKU keyed). Root-layout swaps
  (`swap_w8a8_linears`) thread `gemm_mode` identically.

## 0.35.2 (2026-07-17)

- **gw#565: publish `/complete` survives edge-masked 5xx during a long
  server-side verify.** tensorhub's `/complete` streams the shard back from
  R2 and hashes it synchronously; on a degraded hub link a 2GB shard verify
  runs 10+ minutes, the tunnel in front (ngrok) times out first and answers
  the pod 503 HTML. The bounded inner retry (5 attempts, ~2 min) then
  RETURNED the 503 and the commit died fatal — while the hub finished the
  verify anyway (found live, te#89: a gate-PASSED flavor lost at the seal).
  A returned >=500 now joins the same patient re-POST clock as a severed
  connection (`_COMPLETE_NETWORK_MAX_WAIT_S`); the idempotent Finalized
  fast path answers the catch-up POST.

## 0.35.1 (2026-07-17)

- **gw#562 follow-up: oversize tensors shard alone instead of failing the
  cast.** `plan_shards` gave up (`tensor_exceeds_max_shard_bytes`) on any
  single tensor over the 2GiB shard target — which killed EVERY fp8/w8a8
  cast of an fp32 tree whose excluded lm_head/embedding exceeds it
  (hidream-o1's fp32 lm_head = 2.49GB, found live ie#480). HF
  `split_torch_state_dict_into_shards` semantics now: the oversize tensor
  rides alone in its own oversized shard.

## 0.35.0 (2026-07-17)

- **gw#562: w8a8 lane for root-layout (DiffSynth/singlefile) families —
  hidream-o1 + anima.** The `#fp8-w8a8` path drops its diffusers-tree
  assumption end to end. Producer: `streaming_w8a8_snapshot` accepts any
  non-diffusers layout with a single root weight set (hidream-o1's
  sharded-transformers root) — same per-channel requant, byte-gate
  unchanged. Detector: `detect_w8a8_artifact` header-sniffs root shard
  sets (index-aware) when no `model_index.json` exists; artifact
  `component=""` marks the root layout. Serve: pipeline classes that
  construct their own model (DiffSynth `from_pretrained` wrappers) call
  `sanitize_w8a8_state_dict` while reading shards — quantized weights
  dequant correctly on ANY host (never an unscaled fp8 upcast) — and
  `load_from_pretrained` then swaps the constructed denoiser's quantized
  Linears onto `Fp8ScaledLinear` in place (`swap_w8a8_linears`, module
  path = tensor key, `key_map` hook for converter-renamed checkpoints
  like anima's `net.` strip). Lane stamps (`w8a8`/`bf16-resident`),
  scale-presence exclusion, and skip patterns are identical to the
  diffusers lane. DiffSynth families have no compile cells yet — the
  root lane serves eager w8a8.

## 0.34.0 (2026-07-17)

- **gw#561 (gw#547 remainder; ie#488 turbo critical path): lora-bucket
  compile cells.** `Compile(lora_bucket=N)` declares a dynamic-LoRA
  endpoint's traced rank bucket: the worker enables canonical zeroed
  rank-N branches after load/placement and BEFORE compile arming, so the
  pipeline traces (and only adopts) the `<lane>-lora<bucket>` graph family;
  staying eager rolls the branches back (canonical zeroed slots measured
  +21-32% eager in gw#547). `compile_cache.build(lora_bucket=N)` produces
  branch-bearing cells the same way — labels/metadata inherit the lane
  (`inductor-<sku>-torch<mm>-w8a8-lora128`), plus a `lora_bucket` metadata
  field. Cell pick at boot is lane-AND-bucket exact (a branchless endpoint
  never fetches a lora cell and vice versa — both would lane_drift and
  shadow the right cell); hub-pushed runtime adoption re-applies the
  declared lane before the drift check and rolls it back on failure.
  TRT engines never serve lora buckets. Discovery carries
  `compile.lora_bucket` for the hub's producer reconciler (th#854/te#88).
  Local cells (cozy-local self-mint) key, mint and adopt lora-bucket cells
  through the same seam.

## 0.33.0 (2026-07-17)

- **gw#558 (ie#388 dynamic-LoRA primary path): lane-general runtime LoRA
  additive branches.** The gw#547 w8a8 side-branch generalizes to every
  serve lane: plain `nn.Linear` denoisers (bf16-resident lane) and
  layerwise-cast denoisers (the fp8-storage `fp8+te` lane) carry the same
  `y += B(A @ x)` compute-dtype branch through an idempotent instance-forward
  wrap — never peft module wrapping (ie#374), never a weight mutation, and
  removal is bit-exact. Branch tensors on cast lanes live in the module
  `__dict__` so the cast hooks can never fp8-round-trip them (verdict: the
  branch COMPOSES with diffusers layerwise casting — the ie#374
  incompatibility was peft-implementation-level, as ruled). Adapter state
  dicts normalize through the pipeline class's own `lora_state_dict`
  converter first (te#81's zero-drift pattern) with the existing
  diffusers/peft/kohya grammar as fallback. Denoiser halves ALWAYS ride the
  branch; text-encoder halves keep peft on uncast TEs and are refused typed
  (`RefCompatibilitySurprise`) on cast TEs; unmappable adapters (conv-
  targeting LoCon-class) on the plain lane fall back to whole-adapter peft.
  Lane stamps compose as `<base>-lora<bucket>` (`w8a8-lora32`,
  `fp8-hooks-lora32`, `lora32`), keeping branch-bearing pipelines and
  branchless compile cells apart under the symmetric `lane_drift` guard;
  compiled pipelines still refuse bucket resizes (no recompile at swap).

## 0.32.2 (2026-07-16)

- **gw#559 / ie#496: Forge captures every declared image CFG regime.**
  `Compile.guidance_scales` is an explicit family-cell contract axis carried
  through producer and local warmup, artifact metadata, and adoption drift
  checks. A compatible family can therefore capture ordinary CFG batch-2 and
  no-CFG batch-1 graphs in one checkpoint-independent cell when both calls
  share the same module graph. LoRA-mutated calls remain a separate cell lane.
- **W8A8 exact-byte verification uses the declared quantization-input dtype.**
  `verify_w8a8_snapshot` can cast immutable source storage to the code-owned
  producer compute dtype before exact sampled byte/scale recomputation and
  reports both storage and compute dtypes. This keeps production BF16 input
  truthful for FP16-stored SDXL checkpoints without weakening exact equality.

## 0.32.1 (2026-07-16)

- **ie#496: W8A8 production requires the exact compatible Forge cell.**
  Workers select the family cell's `-w8a8` lane instead of the first attached
  family artifact, and cell metadata binds the loaded module/tensor graph,
  dynamic-scaled-mm/excluded-layer schema, shape table, GPU SM, CUDA driver,
  compiler stack, and serving-image digest. The signature deliberately omits
  checkpoint refs, digests, and tensor values, so graph-compatible SDXL, Pony,
  and Illustrious weights share one cell. Missing/mismatched cells and runtime
  graph failures are retryable lane failures; they never silently claim W8A8
  while serving eager or dequantized compute.

## 0.32.0 (2026-07-16)

- **gw#557 (ie#494 W8A8 productization core): streaming per-channel-scaled
  fp8 producer + byte-gate + fp8+te TE wiring on the w8a8 lane.**
  `convert.writer` gains `streaming_w8a8_cast` / `streaming_w8a8_snapshot` —
  a data-free requant of repeated-block denoiser Linears from the bf16
  source into the gw#534 `#fp8-w8a8` artifact (fp8-E4M3 weights + F32 [out]
  per-output-channel `weight_scale` twins; dynamic activation scales at
  serve time, no calibration; gate-logit projections and everything outside
  repeated blocks stay at source precision — the ie#494 probe's flip/skip
  spec). Streaming two-pass per tensor, te#81 pattern: the model is never
  materialized. `verify_w8a8_snapshot` byte-gates a produced tree against
  its source (consumer-side detection, sampled recompute-exact quant+scale
  bytes, dequant within the fp8-e4m3 format error bound).
  `load_w8a8_pipeline` honors `storage_dtype="fp8+te"`: the gw#460
  block-window fp8 storage now arms on the TEXT ENCODERS of a w8a8-served
  pipeline (never its scaled-mm denoiser) via the new `components=` scope
  override on `apply_fp8_storage`.


## 0.30.2 (2026-07-16)

- **gw#554: clone disk admission follows the resolved work instead of a
  configurable source-size multiplier.** Pure source mirrors account for
  hardlinks and only the safetensors files that must be resharded; plan-known
  conversions account for materialized output, layout-repack, and intermediate
  GGUF trees. Hugging Face ingest records the observed on-disk dtype, and
  existing HF shard groups are resharded with one valid index. This admits the
  immutable 19.1 GiB Z-Image source mirror on the standard 40 GiB CPU worker
  disk while retaining fail-fast bounds for real conversions. Repack tools may
  still reject provider-fetched base components that were absent from the
  source plan. The `COZY_CONVERT_DISK_HEADROOM` override is removed.


## 0.30.1 (2026-07-16)

- **ie#381 fix 2: the bf16-resident fit check counts fp8 bytes per TENSOR,
  not per component majority label.** Produced fp8 flavors store scales and
  norms in bf16, so a shard is majority-BF16 by tensor count while its
  weight bytes are fp8 (LTX DiT: 247 bf16 vs 137 fp8 tensors, fp8 = 3x the
  bytes) — `detect_on_disk_dtype`'s majority gate counted the upcast as
  ZERO, neutering both the weights-margin rule and 0.28.1's declared-VRAM
  envelope term, and the upgrade proceeded into the activation budget.
  `snapshot_component_fp8_bytes` sums F8_E4M3 tensor bytes from the
  safetensors headers; `bf16_resident_fits` doubles exactly those.


## 0.30.0 (2026-07-16)

- **gw#551: demoted lanes serve instead of crashing — swap-per-request for
  multi-model releases.** te#79's serve proof showed a merged two-lane
  endpoint whose lanes overcommit VRAM (bf16 qwen pair on one H100) demotes
  one lane to host RAM and then CRASHES the next request on it (addmm device
  mismatch / cuda generator vs cpu latents): every declared slot was
  job-pinned (the idle sibling could never be LRU-swapped out), eager
  promotion tried to promote ALL lanes (can never fit), and nothing between
  "demoted" and "the handler calls the pipeline" re-promoted the used lane.
  - `models/lane_gate.py`: every worker-constructed pipeline's `__call__` is
    wrapped (identity/isinstance-preserving) to pin its lane and promote it
    if demoted — LRU-swapping the idle sibling — before executing; a lane is
    NEVER run cpu-resident. When VRAM truly cannot fit, the call queues
    briefly then fails RETRYABLE; monolithic pipelines instead arm a
    coherent CPU-offload rung (`memory.rearm_offload`) and serve degraded.
  - Records holding 2+ worker-constructed pipelines become call-time-owned:
    excluded from the whole-job pin and from eager `_promote_setup_refs`
    (the gate owns exactly the lane a request touches). Swaps log loudly
    (`LANE_SWAP … promote_ms=`) and keep riding the gw#479 ModelEvent
    durations.
  - `models/pinned_swap.py`: tier swaps go through a pinned host-RAM weight
    cache instead of pageable `.to()` — demote of an unchanged weight is a
    pointer swap (host copy already current), promote is one `non_blocking`
    H2D per tensor at full PCIe bandwidth. Fail-soft to `.to()` on any
    unsupported shape; `Residency.demote`'s host-RAM floor counts cached
    bytes as already-resident.
  - `Residency.promote` refuses fast (no doomed multi-GB partial move) when
    free VRAM cannot hold the actual weight bytes after `make_room`.

## 0.29.0 (2026-07-16)

- **gw#549/gw#550: media transfer efficiency + boot host canary.** On-GPU
  uint8 conversion + pinned async D2H staging + zero-copy PyAV handoff for
  video encode; boot host canary (memcpy / pinned PCIe bandwidth / CPU score)
  reported with worker registration. (Shipped in PR #269; entry added
  retroactively.)

## 0.28.1 (2026-07-16)

- **ie#381: the gw#534 rung-2 bf16-resident upgrade now respects the
  function's declared VRAM envelope.** The weights-only fit check upgraded
  LTX-22B's fp8+te lane to bf16-resident on 80 GB cards, silently consuming
  the activation budget the envelope was measured around — every >=10 s
  1080p request then served through the DEGRADED tiled-refine rung (slower
  than the stored-fp8 recipe AND quality-taxed), while the compile-cell
  producer traced the opposite weight lane. `bf16_resident_fits` gains a
  `declared_vram_gb` term (upgrade only when `free >= declared +
  upcast_extra`), plumbed from `Resources.vram_gb` through
  `load_slot`/`load_from_pretrained`, and `compile_cache.build()` accepts
  the same value so producer and serving worker decide the lane from the
  same inputs (gw#391 parity). Declared-unknown loads (local CLI) keep the
  old margin-only rule.


## 0.28.0 (2026-07-16)

- **gw#470 boot warmup default-on.** GPU inference endpoints now warm before
  READY with zero author code: the worker synthesizes one minimal request per
  handler from its typed payload schema (defaults kept; required `str` fields
  fill `"warmup"`; required `ImageAsset`/`AudioAsset` fields get a tiny
  generated PNG/WAV; nested structs/lists synthesize recursively) and runs it
  post-`setup()` under the load lock. Output is discarded (no emitter, no
  capability token, throwaway `local_output_dir`) — never billing/outputs/CAS.
  - Fallback: `@endpoint(warmup={"method": {...}})` declares per-method
    payloads (validated against the schema at decoration/walk time);
    `{"method": None}` skips a method.
  - A class-defined `warmup()` method still wins outright (the LTX path).
  - Opt-out: `@endpoint(warmup=NoWarmup("reason"))` — in code, recorded, no
    env knob.
  - Enforcement: a GPU inference class with no warmable path and no explicit
    declaration fails at decoration/walk time, not at first request.
  - `ctx.boot_warmup` lets a handler cheapen its warmup run (e.g.
    `steps = 1 if ctx.boot_warmup else steps` — the allocator peak is
    shape-driven, not step-driven).
  - A warmup CUDA OOM defers to the gw#521 runtime fit ladder (warn + READY)
    instead of hard-failing the function; other warmup errors remain load
    failures (loud, th#581 rails). Cancel/drain-safe on the existing
    `_to_thread_complete` rails.


## 0.27.0 (2026-07-16)

- **th#826 call-out primitive (workflows-as-endpoints).** Functions declared
  `@endpoint(child_calls=True)` may call other endpoints as attributed,
  bounded, cancellable child requests:
  - `ctx.call_endpoint(endpoint, function, payload, *, tag, wait, timeout_s,
    tier)` — sync-await (returns output items) or `wait=False` for a
    `ChildRequest` handle (`.status()` / `.result()` / `.cancel()`).
  - `ctx.workflow_checkpoint(key, fn)` — step-result memoization under the
    invocation (crash-resume by fast-forward; WORKFLOW-DESIGN.md §4).
  - Typed errors: `ChildCallRefusedError` (depth/cycle/budget/tier/parent
    refusals + `child_calls_not_declared`), `ChildRequestFailedError`,
    `ChildRequestCanceledError`, `ChildCallTimeoutError`.
  - Discovery emits `child_calls = true`; the hub mints the `invoke_child`
    capability grant only for declaring functions. Children bill the parent
    request's payer, inherit its availability tier, and die with the tree on
    parent cancel.


## 0.26.9 (2026-07-15)

- hub_policy: probe `modelopt` in the known optional-libs list (te#79
  regression: `Resources(libraries=("modelopt",))` functions were
  structurally unavailable — the executor's find_spec fallback passed but
  plan_serve re-checked installed_libs, which never probed modelopt).


## 0.26.1 (2026-07-14)

- **NVENC per-request fallback recreates the PyAV output container.** A
  hardware stream that failed during codec open remained attached to the
  original container, so mux startup retried that orphan and failed even
  after adding libx264. The fallback now starts with a clean container.
- **Discovery stubs no longer poison later optional-dependency probes.** A
  missing heavy module remains usable through its returned stub reference,
  but is removed from `sys.modules` immediately so `find_spec()` stays honest.

## 0.26.0 (2026-07-14)

- **Model residency is declarative.** Protocol v3 replaces ordinary
  download/load/unload commands with a full-replace per-worker desired disk
  set and ordered hot runnable instances. Workers report the accepted
  generation separately from actual residency events.
- **Tenant work preempts background reconciliation.** A `RunJob` cancels
  unrelated desired-state work before request setup, then resumes the current
  desired generation when the executor becomes idle.
- Hot dynamic-slot instances reuse the request binding and setup path, so the
  function plus complete slot-to-immutable-ref map identifies exactly one
  runnable instance without a second loader.

## 0.25.2 (2026-07-14)

- **Mixed CPU/GPU releases probe the device owned by their concrete image.**
  A release-level discovery manifest contains both lanes, so the prior
  any-GPU-function check killed the CPU conversion image before worker hello.
  Mixed manifests now use the installed Torch build as the lane signal: CUDA
  images retain the bad-host health probe, while CPU-only images start their
  CPU functions without an environment-variable override. GPU-only manifests
  still fail closed when CUDA is absent.

## 0.25.0 (2026-07-14)

- **cl#27: local-only GGUF fit rung.** A bare Tensorhub binding can select the
  best compatible `#gguf-<qtype>` sibling only after the base, runtime-fp8,
  and compatible native flavors miss. The local resolver composes that
  denoiser with the base diffusers tree in the CAS, and the loader injects it
  with `GGUFQuantizationConfig` while retaining the base encoders, VAE, and
  scheduler. Production precision resolution remains hub-owned and never
  selects this local-only rung.
- **Placement follows code, not an ENV veto.** Removed
  `GEN_WORKER_FORBID_CPU_OFFLOAD` and its test-only overrides; CPU/offload
  decisions now always run through the worker's actual fit and OOM-demotion
  logic. Shared-component lanes also correctly recognize `vae_only` as a
  resident mode.
- **CI/runtime torch baseline is CUDA 13.0.** The locked Linux/Windows uv
  source now resolves `torch==2.13.0+cu130` and
  `torchvision==0.28.0+cu130`, matching the managed endpoint fleet.
- No endpoint-authoring API names changed in this release.

## 0.24.2 (2026-07-14)

- **gw#534: compile-cache cell labels carry the traced weight lane.**
  `flavor_label(sku, torch, weight_lane)` suffixes non-plain lanes
  (`-w8a8` for scaled_mm graphs, `-w8a16` for layerwise-cast-hook graphs);
  plain resident stays unsuffixed. Cells of different lanes are different
  FX graphs — one label per (family, sku, torch) made a W8A8 cell and a
  bf16 cell collide in the family repo. `build()` derives the suffix from
  the loaded pipeline's actual lane. Tensorhub's compilecache.FlavorLabel
  mirrors byte-compatibly (th#786 companion).

## 0.24.1 (2026-07-14)

- **gw#534: Fp8ScaledLinear eager quant without fp32 intermediates.** The
  activation quant now runs in the compute dtype (reciprocal multiply);
  the fp32 division path doubled eager activation traffic and made eager
  w8a8 as slow as the cast hooks it replaces. Measured on H100 SXM
  (qwen-image DiT 20B, 1024² b=1): eager w8a8 306.2 -> 265.4 ms/forward
  (eager w8a16 prod lane: 304.8; bf16: 211.9). COMPILED (regional
  per-block, ie#381 — quant ops fuse, GEMMs run fp8-rate): bf16 163.9 |
  w8a16 250.8 | **w8a8 142.1 ms/forward — 1.77x vs the compiled W8A16 prod
  lane, 2.15x vs today's eager prod path, 1.15x over compiled bf16**. The
  cast tax survives compile on the w8a16 lane (hooks run outside the
  graphs) — W8A8 is the only fix. Quality parity (same-seed
  FLUX.2-klein-4B, 1024², vs bf16 reference): w8a8 PSNR 25.02 dB ==
  w8a16 24.87 dB.

## 0.24.0 (2026-07-13)

- **gw#534: W8A8 fp8-GEMM loader mode — the calibrated-quant serve path.**
  A `#fp8-w8a8` flavor (fp8-E4M3 weights WITH scales; per-Linear
  `weight` / `weight_scale` / optional static `input_scale`; exclusion by
  absence — the gw#534 artifact contract) is detected by header sniff and
  served with quantized Linears swapped for `Fp8ScaledLinear`:
  `torch._scaled_mm` over RESIDENT fp8 weights, per-row dynamic activation
  quant (static calibrated scale when present), bias fused, no per-layer
  upcast. Hosts without usable scaled_mm (pre-sm89 / missing kernels — live
  device probe, never a version table) dequant once at load to bf16-resident:
  same numerics, never a refusal. Scale-FREE fp8 trees (the storage-cast
  `#fp8` flavor) never match — the scales are the distinguisher. Pipelines
  stamp `_cozy_weight_lane="w8a8"`, keying the compile cache (lane_drift):
  W8A8 pipelines never adopt W8A16/bf16-traced graphs and vice versa.
  `models/w8a8.py` also ships the data-free producer (`quantize_tree_w8a8`,
  per-out-channel amax scales) used by tests and `scripts/w8a8_parity.py`
  (same-seed bf16 / w8a16 / w8a8 quality + speed harness); calibrated
  production artifacts come from the conversion side (te#79).

## 0.23.0 (2026-07-13)

- **gw#534: fp8 download, bf16 resident — W8A16 layerwise casting is never
  voluntary.** The measured per-forward cast tax of the fp8-storage lane is
  +44% wall on H100 / +73% on B200 vs bf16-resident (gw#534 profiles). A
  planned fp8 storage lane (stored `#fp8` flavor or resolved `storage_dtype`
  cast) is now UPGRADED at load to plain bf16-resident weights whenever the
  snapshot fits free VRAM with headroom (`bf16_resident_fits`, 4GB activation
  margin; fp8-stored cast targets counted at 2x for the upcast): the fp8
  artifact stays the small download, `from_pretrained`'s torch_dtype upcasts
  once, and the per-layer cast hooks are skipped. Hooks remain only when bf16
  does not fit (involuntary W8A16). Compile-cache: cells record
  `weight_lane` — the lane the built pipeline ACTUALLY traced under
  ("" plain-resident / "fp8-hooks") — and `enable()` + the executor adopt
  path reject lane drift symmetrically (a bf16-resident pipeline must never
  adopt hook-cast-traced graphs and vice versa; both are guaranteed FX-graph
  misses that would serve eager while reporting adopted).
## 0.22.6 (2026-07-13)

- convert/layout: HiDream-O1 family hint — `HiDream-ai/HiDream-O1-Image*`
  repos stamp `model_family="hidream-o1"` at ingest, so the mirror's th#767
  inference-defaults PUT has a family to key on (ie#478).

## 0.22.5 (2026-07-13)

- **packaging: grpcio floor 1.82.1.** The shipped pb stubs are generated
  with grpcio-tools 1.82.1 and refuse to import under older grpcio
  (th#766: conversion image crash-looped on locked 1.81.1).

## 0.22.4 (2026-07-13)

- **gw: fp8 storage flavors for transformers-BACKBONE snapshots (ie#478).**
  `streaming_fp8_snapshot` / the clone `build_flavor_tree` fp8 lane now
  accept a non-diffusers layout when the snapshot is exactly ONE root
  weight set (sharded-transformers backbone — the whole checkpoint IS the
  denoiser, e.g. HiDream-O1's pixel-space UiT). The cast is BLOCK-SCOPED:
  eligible weights must live under a repeated-block container (`.<idx>.`
  path segment) in addition to the existing skip patterns, keeping the
  stored fp8 set a strict subset of the runtime block-window walk
  (`_fp8_block_windows`) so every stored-fp8 tensor is re-armed by any
  consumer. Zero-cast outputs refuse loudly (never a silently-uncast
  "fp8" flavor). Multi-set singlefile bundles still refuse — component
  identity is ambiguous there. New `run_inline_conversion(...,
  fp8_block_scope=)` / `streaming_fp8_storage_cast(..., block_scope=)`
  pass-throughs.

## 0.22.3 (2026-07-13)

- **pgw#516 (settled foundation): LoRA-kind family vocabulary + FIELD-LEVEL
  lora composition (th#767 one level down).** `gen_worker.families` gains a
  KIND axis — `@family(name, kind="checkpoint"|"lora")`, same family name,
  a separate typed vocabulary struct per kind (default `kind="checkpoint"`,
  every existing `@family("sdxl")` call site is unaffected). Ships
  `SdxlLoraDefaults` (`trigger_words`, `recommended_weight`, `steps`,
  `guidance`, `max_guidance`, `scheduler` — every field but
  `trigger_words`/`schema_version` defaults to `None`, "no opinion").
  `family_for`/`family_registry`/`export_json_schema`/`export_all_schemas`
  all take/key on `kind`; `gen-worker families export-schemas` now writes
  `<family>.schema.json` (checkpoint) AND `<family>.lora.schema.json`
  (lora) per registered pair (`export_json_schema` also now round-trips its
  return value through JSON so every caller — not just the CLI — sees
  JSON-safe types, e.g. tuple defaults as arrays).
- **Composition rule**: when a lora rides a pick, its non-`None`
  inference-defaults fields override the resolved checkpoint recipe FIELD
  BY FIELD (not the whole-object repo-metadata-over-fallback precedence),
  in lora-ride order — a distillation lora's `steps=4`/`guidance=0` beats
  the base checkpoint's `28`/`6`. `resolve_slot`/`resolve_slots` gain
  `lora_metadata_json=`; `ctx.slots[slot].defaults` is the merged result.
  Wire: `LoraOverlay` gains `inference_defaults` (proto field 3) — see
  `proto/CONTRACT.md`.
- Endpoint-authoring surface for curated LoRA menus (e.g. `TurboLora`-style
  enums) stays OUT of scope by design — open per pgw#516/th#767.
- **Rebased onto 0.21.0 (pgw#524's `Slot(fallback=)` -> `Slot(default_config=)`
  rename, pgw#532's dynamic slot materialization).** This PR's original diff
  (authored pre-rename) still referenced `slot.fallback` in
  `api/slot.py::resolve_slot`'s no-repo-metadata branch; ported to
  `slot.default_config`, keeping the pgw#516 field-level lora-override
  composition (`_apply_lora_overrides`) wrapped around it — same treatment
  the repo-metadata branch already got. No interaction with #243's dispatch
  materialization: that PR rebinds WHICH checkpoint a Slot resolves to
  per-dispatch (`_effective_spec`); this PR governs what `.defaults` looks
  like once a checkpoint (any checkpoint) is already picked — the two
  compose without overlap.

## 0.22.2 (2026-07-13)

- **pgw#505: selective component download — declare-on-binding.** `Hub`/`HF`
  gain `components=` (mirrors `files=`): restricts a fetch to the named
  pipeline component subfolders (+ root config files, e.g.
  `model_index.json`) instead of the whole repo — the win case is a full
  pipeline repo bound for exactly ONE component, e.g.
  `Hub("owner/sdxl-repo", components=("vae",))`. Civitai/modelscope reject
  it (civitai artifacts aren't component-structured; modelscope already has
  `files=`). `components=` surfaces on the manifest binding block (tensorhub
  + huggingface) so the hub's ModelOp DOWNLOAD scoping can read it once
  built — that platform-side selective CAS resolve is NOT part of this
  release. Worker-side filtering ships now on two paths that fully own
  their own resolve+download+materialize loop: the HF downloader (both the
  production executor and the CLI) narrows `snapshot_download`'s
  `allow_patterns` to the declared subfolders before the existing
  flavor-selection logic runs; the CLI's hub-less tensorhub resolve
  (`cozy run` / `gen-worker run`, th#560) narrows the fetched blob set and
  keys the materialized snapshot directory by `(digest, components)` so a
  component-scoped fetch can never collide with — or be mistaken for — the
  full-repo one. The production executor's orchestrator-resolved snapshot
  path is deliberately left unfiltered: its residency layer digest-verifies
  the materialized tree against the orchestrator's full file list, so
  scoping there is the hub's job, not the worker's.
- **Rebased onto 0.21.0 (post-#233 `allow_lora` eviction, post-#243 dynamic
  slot materialization).** This PR's original diff (authored pre-#233) also
  re-added `allow_lora=` to `ModelRef`/`Hub`/`HF` and an
  `_binding_to_manifest`/discovery emission path for it — all of that is
  dropped on rebase; `allow_lora` stays evicted (overlay permission is a
  slot-policy concern, th#772, not a binding-identity flag). Only the
  `components=` axis (a genuinely new field, disjoint from `allow_lora`)
  survives. `binding.py`, `discover.py`, `test_binding.py`, and
  `test_discovery_and_decorators.py` all had this same shape of conflict;
  resolved identically in each. No interaction with #243's slot
  materialization — `components=` is a fetch-scope hint on a fixed binding,
  orthogonal to which pick gets dispatched.

## 0.22.1 (2026-07-13)

- **pgw#506: discovery-time lazy-import stubs for heavy deps — the
  defer-`import torch`-into-handlers convention is retired.** Build-time
  discovery (`python -m gen_worker.discovery` / `discover_functions`) arms
  `gen_worker.discovery.heavy_deps.stub_missing_heavy_deps`: when an
  allowlisted heavy root (torch, torchvision, torchaudio, triton, xformers,
  flash_attn, bitsandbytes; extend via `[tool.gen_worker]
  discovery_heavy_deps = [...]`) is MISSING from the environment, a stub is
  served for it (and any submodule) so module-top `import torch` succeeds and
  schemas still build — while every attribute TOUCH on the stub raises an
  actionable `HeavyDepStubError` naming the fix (move module-scope use into
  `setup()`/the handler, or install the dep). When the dep IS installed
  (in-image discovery), nothing changes.
- **Discovery hard-fails on endpoint module import errors.** `find_endpoints`
  no longer logs-and-continues when a module or submodule fails to import —
  that silently shipped endpoint.locks (and live route tables) missing
  functions. Any non-heavy-dep ImportError/SyntaxError now raises
  `EndpointImportError` with the original exception chained; the discovery
  CLI prints the full traceback and fails the build.
- docs: endpoint-authoring guide — module-top imports are the rule; the
  defer-imports convention is deleted.
- **Rebased onto 0.21.0 (pgw#523 provider/ref alias retirement, pgw#517
  compile= hard-error, pgw#524 SDK friction batch).** No functional overlap:
  the heavy-dep stub seam and the hard-fail-on-import-error walk both operate
  purely at `find_endpoints`/`discover_functions` time, before any
  Slot/ModelRef emission logic runs; those PRs' changes to manifest/binding
  emission are downstream of a successful discovery walk and untouched here.

## 0.22.0 (2026-07-13)

**pgw#526 + pgw#527 — ctx hierarchy honesty + dead-surface cuts (BREAKING).**

- **Producer state off the base ctx (pgw#526).** `_source_info` /
  `_destination_info` / `_source_path` / `_hf_token` / `_repo_spec` now
  initialize in `_PublisherMixin.__init__` — a plain inference
  `RequestContext` no longer carries state only producer accessors could
  read. `RequestContext.__init__` loses the `source_info=` /
  `destination_info=` / `hf_token=` / `compute=` kwargs (producer kwargs
  move to the mixin; the executor passes them only for producer kinds).
  The upload-budget gate stays on the base: the base `save_file` path
  reserves against it.
- **`ctx.compute` DELETED; `ResolvedCompute.gpu_count`/`vram_gb` cut from
  the wire (pgw#526, audit P5).** The surface was documented in
  proto/CONTRACT + docstrings, plumbed through every dispatch, raised
  `AttributeError` on inference contexts, and had ZERO endpoint readers.
  `gen_worker.api.Compute` is gone; proto field numbers 3/4 are
  `reserved`. `accelerator` + `gpu_index` (GPU-semaphore gating + CUDA
  binding) survive. Coordinated tensorhub PR trims the proto mirror +
  dispatch population — deploy both sides together (protobuf keeps the
  wire compatible either way: unknown/absent fields decode as zeros).
- **Save-path dedupe (pgw#526).** `_save_file_create` folded into
  `save_file(ref, local_path, *, create=False)`; `save_checkpoint` /
  `save_checkpoint_bytes` now share one `_publish_checkpoint` core (the
  bytes variant gains the upload-budget reservation and `attributes=`
  parity it was missing). `publish_dataset_revision`'s raw-requests hub
  plumbing moved next to `HubClient`
  (`gen_worker.convert.hub.publish_dataset_revision`); the
  `DatasetContext` method is a thin delegate.
- **`checkpoint_dir` stops promising persistence (pgw#527, audit §5.5).**
  Documented as JOB-SCOPED SCRATCH under pod-local `/tmp`: gone at pod
  churn; durable resume goes through published checkpoints. Evidence: the
  only production trainer (image_lora_finetuner) wipes the dir at start
  ("Resume v1 = clean restart") and publishes checkpoints for resume;
  RunPod volumes, when mounted, are the model-cache volume — not a
  trainer-resume home. No behavior change.
- **`hub_policy.select_variant` + `VariantChoice` deleted (pgw#527).**
  Zero production callers since `--variant auto` was removed (pgw#226/
  #515); ranking lives hub-side. `variant_fit` (the serve-fit ladder's
  verdict function) is untouched.
- **Rebased onto 0.21.0 (pgw#532 dynamic slot materialization).** No
  functional overlap: pgw#532 rebinds each declared `Slot` to the
  dispatch-resolved pick in `_effective_spec`/`_slot_dispatch_binding`;
  this PR's ctx-hierarchy changes are orthogonal (producer-state
  location, not slot resolution). Both apply cleanly side by side —
  `_effective_spec`'s derived-binding-set flow and the `_PublisherMixin`
  producer-state move touch disjoint concerns in `executor.py`.

## 0.21.0 (2026-07-13)

**BREAKING(-ish) — pgw#532: worker-side dynamic slot materialization (the last th#767
piece).** A hub-connected worker no longer materializes a declared `Slot`'s
`default_checkpoint` from its raw upstream — the fc157 live failure where a Civitai
default hit `civitai_not_found` at boot setup and cascaded `load_failed` onto every
healthy hub binding.

- **Materialization precedence per declared Slot** (executor `_slot_dispatch_binding`):
  the hub-resolved pick from `RunJob.models[slot]` (a tensorhub-CAS ref; snapshots ride
  the dispatch / earlier ModelOps, th#763 re-mint covers cold refs) > the code-declared
  `default_checkpoint` when it is itself a `Hub(...)` CAS ref > **fail RETRYABLE** —
  never a raw Civitai/HF/ModelScope self-fetch (mirror-first, gw#465). Hub-less
  (`cozy run` / `gen-worker run`) resolution of the raw default is unchanged
  (`models/provision.resolve_bindings`).
- **Boot**: `lifecycle.startup()` no longer prefetches Slot seeds from upstream and no
  longer eagerly sets up Slot-declared endpoints with the code seed; dynamic-slot
  functions advertise available once hardware-gated and materialize per dispatch.
- **Instance-per-pick**: `_effective_spec` rebinds every declared Slot to the dispatch's
  resolved pick; the derived binding set derives a new `instance_key`, so `setup()` runs
  once per (class, resolved pick), `self.pipeline`-style setup-held state stays coherent
  per checkpoint, multiple picks stay warm side by side, and the existing residency/LRU
  machinery evicts whole instances. `ModelOp{LOAD}` now also matches per-pick derived
  records (promote/re-set-up a previously-dispatched pick); a LOAD for a never-dispatched
  pick banks bytes+snapshot and reports `load_failed` (pre-warm degrades to a download).
- **`ctx.slots[name].ref` is the resolved pick** (not the code default);
  `.defaults` still merges the wire's `inference_defaults` over the code preset.
- `gen_worker.testing` helpers unchanged (the `ctx.slots` stub shape is identical).

## 0.20.0 (2026-07-13)

**BREAKING — pgw#523: `ModelRef` is pure identity + fetch scope; `.provider`/`.ref` aliases retired.**

- **Part A — evict `allow_lora` (identity != permission).** Deleted `allow_lora` from
  `ModelRef` and the `Hub(...)`/`HF(...)` kwargs. th#772 moved overlay permission to the
  slot-policy `loras` axis; the th#586 architecture gate has always keyed off the declared
  binding/slot FAMILY (`EffectiveBindingFamily`), never this flag, so it never gated
  anything at runtime — only a registration-time co-occurrence check (allow_lora=true
  requires family), which tensorhub also retires this release. `_stamp_lora_family` ->
  `_stamp_family`: family stamping is now unconditional-when-known on every binding
  (top-level `bindings` blocks and `model.choices[].binding` rows alike), not
  allow_lora-triggered.
- **Part B — retire the `.provider`/`.ref` back-compat aliases.** `ModelRef` now exposes
  only `.source`/`.path`. Every in-repo consumer (discovery manifest emission, executor
  prefetch/download plumbing, residency cache-key labeling, the CLI's list/prefetch
  commands) repoints at `.source`/`.path`. The manifest `bindings.<slot>.provider` wire
  field now carries the pgw#511 vocabulary directly (`"huggingface"`/`"modelscope"`, not
  the old `"hf"` short form) — requires tensorhub's widened `provider` DB CHECK deployed
  first (th#523 companion PR). `models/refs.py::parse_model_ref` accepts both `"hf"` and
  `"huggingface"` as input and keeps normalizing to the internal `"hf"` token, so the
  ref-grammar module and every `parsed.provider == "hf"` comparison downstream (download/
  provision) are unaffected.
- Hard cut, no back-compat: constructing `Hub(..., allow_lora=True)` or reading
  `ref.provider`/`ref.ref` now fails immediately (`TypeError`/`AttributeError`).

## 0.19.1 (2026-07-13)

- **pgw#517: `compile=` is no longer silently inert on self-loading
  (str/Path-slot) endpoints.** The executor only ever armed
  `compile=Compile(...)` automatically on a `setup()` slot it loads itself
  (a pipeline-class annotation) — an endpoint that self-loads from a
  `str`/`Path` slot declared `compile=` that seeded the manifest/shape
  contract but never actually armed at request time. Discovery now hard-
  errors on that combination (was silent). Two fixes, both documented in
  the error: annotate the slot with the pipeline class so the worker loads
  it and arms compile automatically, or call the new
  `gen_worker.arm_compile(pipe)` at the end of `setup()` yourself — same
  cache-artifact-gated policy as the automatic path, eager otherwise. The
  arming context (`Compile` spec, cache dir, hub-attached artifact) is
  carried by a `contextvars.ContextVar` the executor scopes to the
  `setup()` call, so `arm_compile` needs no `ctx` parameter and cannot be
  called outside `setup()`. See `docs/compile-cache.md`.

## 0.19.0 (2026-07-13)

**pgw#524: SDK friction batch (first-Slot-consumer findings).**

- **BREAKING: `Slot(default=, fallback=)` -> `Slot(default_checkpoint=,
  default_config=)`.** Manifest wire keys `default_ref`/`fallback_defaults`
  -> `default_checkpoint`/`default_config` in LOCKSTEP with a tensorhub
  companion PR (`manifest_contract.go`, release hydration, slot resolution).
  Hard cut, no back-compat alias — `default_config` still LOSES to repo
  metadata (a recipe of last resort). The `inference-endpoints` sdxl
  endpoint pins `gen-worker<0.19` and keeps working on the old kwargs until
  its own floor bump; it is NOT updated by this release (out of scope).
- **Discovery-time error: a request-branching Slot needs a default.**
  `Slot(selected_by=..., default_checkpoint=None)` now fails at
  registration (`extract_specs`/discovery walk) instead of at hub publish
  — tensorhub already rejected this manifest shape; the SDK now catches it
  at author time.
- **`selected_by` field contract widened.** A payload field named by
  `selected_by` may now type as `str | ModelRef` in addition to plain
  `str` — the wire already accepts a client-supplied structured `ModelRef`
  object (BYOM), which the hub resolves before the worker sees it.
- **`gen_worker.testing.fake_context`/`stub_slots`** — the `ctx.slots`
  test helper every Slot-declared endpoint's unit tests needed, replacing
  hand-rolled `FakeCtx`es.
- **`FamilyDefaults` positional construction confirmed + locked by test.**
  msgspec's `kw_only=True` on the base only affects the base's own field
  (`schema_version`); it does not propagate to a subclass's own fields, so
  `SdxlDefaults("euler_a", 28, 6.0)` already worked — documented loudly on
  the class (positional order follows field declaration order; msgspec
  does not type-check plain construction, so prefer keyword args).
- **CI/publish hardening:** both `ci.yml` and `publish.yml` now run
  `uv sync --locked`, so a green PR actually implies a green publish (the
  0.18.0 silent-publish-failure root cause: publish re-resolved different
  dependency versions than what PR CI validated).
## 0.18.1 (2026-07-13)

- fix(families): normalize docstring-derived schema descriptions with `inspect.cleandoc`
  (msgspec 0.21 emits raw-indented docstrings; broke the golden-file test and the v0.18.0
  PyPI publish). No API change.

## 0.18.0 (2026-07-12)

- **pgw#520: `Slot(pipeline_cls, selected_by=, default=, fallback=)` — the
  SDK half of th#767.** The model SET moves off the endpoint into hub-side
  configuration; code declares a slot's load-time compat, which payload
  field branches it, an optional hub-less/seed default ref, and a typed
  code fallback preset. A bare `ModelRef` value in `models={}`/`model=` is
  still sugar for `Slot(<inferred class>, default=ref)` — fully back-compat
  within this release. `selected_by` is validated at spec-construction time
  against the handler's OWN payload (a plain `str` field; the hub overlays
  the live allowed-value enum, never baked into the SDK schema).
- **`gen_worker.families` — per-family inference-defaults vocabulary.** New
  `FamilyDefaults` msgspec base (frozen, `forbid_unknown_fields=True`) with
  `class SdxlDefaults(FamilyDefaults, family="sdxl")`-style self-
  registration (msgspec's `StructMeta` doesn't forward unrecognized class
  kwargs to `__init_subclass__`, so registration lives in a small metaclass
  wrapping it). Ships the SDXL vocabulary (scheduler/steps/guidance/
  quality_preamble/negative/max_guidance — `max_guidance` is a CLAMP
  constraint, never a wire reshape). `gen-worker families export-schemas
  <dir>` writes `<family>.schema.json` per registered family — standard
  JSON Schema draft 2020-12, `additionalProperties: false` — the contract
  tensorhub validates repo metadata against at PUT time.
- **Discovery emits a `slots` manifest block** for Slot-declared endpoints:
  `{name, pipeline_class, selected_by?, default_ref?{source,path,tag,
  flavor}, family?, fallback_defaults?}` per slot. `model.choices[]`
  (ModelChoice) is untouched and still emitted for existing endpoints —
  Slot endpoints simply never emit it (no first-party curated list,
  th#767). An `allow_lora` binding on a Slot with no `Compile(family=)`
  now resolves its family stamp from the Slot's own fallback-preset
  registration (mirrors/extends pgw#519's `_stamp_lora_family`).
- **`ctx.slots["<name>"]` resolution chain.** Returns a typed
  `ResolvedSlot[D]` (`.ref`, `.defaults`): repo-metadata inference defaults
  (wire: new `ModelBinding.inference_defaults` JSON field, th#767c —
  documented in `proto/CONTRACT.md`) merged over the endpoint's code
  `Slot(fallback=...)` preset, whole-object precedence (a repo either fully
  specifies its family vocabulary, validated by tensorhub at PUT time, or
  it doesn't — no field-level merge). A slot with neither metadata nor a
  fallback raises on FIRST ACCESS, not at dispatch, so an unrelated
  handler never pays for a slot it doesn't read. Explicit payload values
  still win over `.defaults` — that precedence is handler logic.
- **Hub-less resolution (`cozy run` / `gen-worker run` / `serve`).** A
  Slot's `default=` ref is the only resolution source with no hub
  configured; a payload that NAMES a model via `selected_by` in hub-less
  mode fails clearly (`ModelResolutionError`) instead of silently running
  the default. `ctx.slots` resolves the same way locally, against
  `Slot(fallback=...)` only (no repo metadata exists off-hub).

## 0.17.4 (2026-07-12)

- **pgw#515: de-fork `cli/run.py` from the executor.** The local CLI's
  hand-synchronized replica of binding→download→dtype→placement→compile
  (five "Executor parity" comments) is gone. New
  `gen_worker.models.provision` is the ONE load+place core both drive:
  `load_slot` (annotation-typed injection, binding dtype/storage_dtype,
  th#737 pre-load cast gate, gw#491 adaptive-rung outcomes, worker-owned
  placement) and `enable_compiled` (TRT-then-inductor policy — the CLI now
  gets the TRT lane it previously lacked). Executor behavior is unchanged;
  it reports the `SlotLoad` outcomes into ServePlan/FnDegraded exactly as
  before. The CLI's hub-less resolve half (`resolve_local_path` /
  `resolve_bindings`: local CAS, th#560 standalone Hub resolve, direct
  HF/Civitai/ModelScope) also lives in provision; `prefetch` drives it too.
- **Deleted the duplicate precision-ladder walk** —
  `models/ladder.resolve` / `resolve_local_bindings` (+ `LadderModel` /
  `FlavorRow` / `Resolution` and the Py copy of the shared conformance
  vectors). It was reachable only from the CLI and its fp8 VRAM factor
  (0.75 of weight bytes) disagreed with the loader's fit factor (0.55 of
  card size) — two ladders, different physics. One fit ladder remains: the
  loading layer's runtime rungs (fp8 storage → nf4 → offload), same as
  production. The hub-side Go resolver keeps the walk (picks arrive via
  HelloAck); `ladder.py` keeps only the classification + placement halves.
  Local runs no longer pre-rebind bare Hub refs to stored sibling flavors
  (`GEN_WORKER_NO_PRECISION_LADDER` is gone with the feature).
- **`GEN_WORKER_LOCAL_DEVICE` env-to-self channel deleted.** `--device` now
  threads as an explicit argument (`run_setup(..., device=)`, serve's
  `_Endpoint(device=)`); endpoint code should not read that env var.
- **CLI `--variant` removed** (run/invoke) along with
  `select_function_with_variant` and the `variant_auto` capability token —
  `variants={}` no longer exists (pgw#509); base function selection only.
- **CLI behavior now matches production where the fork disagreed:** slot
  compute dtype defaults to the loader's own default instead of a local
  fp16/fp32 guess (binding `dtype=` still wins); unannotated or
  unrecognized setup slots receive the snapshot path (not a force-loaded
  `DiffusionPipeline`); handler-parameter injection passes snapshot paths
  (`str`/`Path`), as the executor always did.

## 0.17.3 (2026-07-12)

- **gw#490: host-resource requirements vocabulary.** `Resources()` gains
  `ram_gb` and `vcpus` — the per-function HOST ask (video-class endpoints:
  pinned TE park + CPU-heavy encode need ~64 GB / 16 vCPU). Discovery emits
  them in `endpoint.lock` (same `msgspec.to_builtins` path as `vram_gb`);
  tensorhub's builder maps them to `min_ram_gb`/`min_vcpus` in the release
  requirement payload and folds them into pod-creation minimums
  (`CreatePodRequest.MinMemoryGB`/`MinVCPUCount`, th#740
  read-back-and-reject). Host asks never imply `gpu=True`.


## 0.17.2 (2026-07-12)

- **pgw#519: `model.choices[].binding` was missing the `family` stamp.**
  `_collect_model_placement_key` (discovery/discover.py) emitted each
  ModelChoice pick's binding without the `family` that top-level `bindings`
  blocks get from `Compile(family=...)` — tensorhub's th#586 architecture
  gate rejects `allow_lora=true` bindings lacking a family, so builder-path
  deploys of ModelChoice endpoints (sdxl) hard-failed. The stamping logic
  is now one shared helper (`_stamp_lora_family`) applied identically to
  both the top-level `bindings` block and every `choices[].binding` row —
  an allow_lora choice binding with no declared `Compile(family=...)` now
  raises at discovery time instead of silently shipping unstamped.

## 0.17.1 (2026-07-12)

- **gw#516: hub-visible finalize.** `StateDelta.finalizing_jobs` (field 5)
  counts jobs past the decode→finalize handoff — GPU slot terminally
  released, encode/upload tail still running, `JobResult` unshipped — so the
  hub's drain/retire/idle logic can treat GPU-idle ≠ work-idle.
  `JobMetrics.slot_held_ms` (11) + `finalize_wall_ms` (12) split
  `runtime_ms` into slot occupancy vs the slotless finalize tail (the
  FINALIZE_OVERLAP log line is now corroborating evidence, not the only
  one). Tensorhub counterpart consumes all three (tensorhub PR #299).


## 0.17.0 (2026-07-12)

- **th#714: C2PA Content Credentials on generated media (EU AI Act Art. 50).**
  New `gen_worker.content_credentials`: every media asset saved through
  `RequestContext.save_bytes` / `save_file` (and therefore `save_image` /
  `save_audio` / `save_video` / `io.write_image` / `io.write_video`) gets a
  signed C2PA manifest — `c2pa.created` action with digitalSourceType
  `trainedAlgorithmicMedia`, generator name/version, model refs, and a
  request-id **hash** (no user PII). Issuer identity comes from the platform
  signing cert. Signing is ON iff `GEN_WORKER_C2PA_CERT_PATH` +
  `GEN_WORKER_C2PA_KEY_PATH` are set (PEM chain + PKCS#8 key, new Settings
  fields incl. `GEN_WORKER_C2PA_ALG` / `GEN_WORKER_C2PA_TA_URL`);
  unconfigured no-ops with a loud startup warning; configured-but-broken
  fails worker startup; a per-request sign failure fails the request rather
  than shipping an unlabeled asset. Non-media payloads (JSON, checkpoints,
  tensors) pass through untouched via content sniffing. New `signing` extra
  (c2pa-python, the official CAI c2pa-rs binding); sign+verify round-trip
  tests (png/webp/jpeg/mp4) run in CI against an openssl-generated test cert.


## 0.16.0 (2026-07-12)

- **pgw#514: dead-surface + protocol-drift sweep (BREAKING, hard cut).**
  Every deletion grep-verified zero-caller across gen-worker,
  inference-endpoints, and training-endpoints post-0.15.0.
  - Dead code: `base_model_families` trimmed to `civitai_to_family` (the
    only mapping with a caller); dead exports `worker_local_model_cache_dir_default`,
    `ensure_local_sync`, `build_function_owned_pipeline`, `InputTooLargeError`,
    `TokenStreamSignal`/`_SIGNAL_TYPES`, `PositivePrompt`/`NegativePrompt`;
    six dead `RequestContext.__init__` params (`required_models`,
    `parent_request_id`, `child_request_id`, `item_id`, `item_index`,
    `item_span`) + the dead `hints["job_id"]` branch; `Executor.__init__`'s
    `on_state_change` kwarg (worker.py assigns the attribute directly);
    stale docstrings (streaming `Done`, `checkpoint_dir` "survives worker
    restart" — it is pod-local /tmp, `Worker._handle_job_request` references,
    ar_tts `sglang_runner` field).
  - Dead config: Settings fields with no producer deleted — `grpc_ca_bundle`
    (its lone transport.py consumer now always uses system roots),
    `worker_git_commit`, the `COZY_HF_*` trio and `attached_lora_max*`
    (now fixed module constants), the `GEN_WORKER_COMPILE_*` trio (now raw
    env reads in `compile_cache.py`), the `HUGGING_FACE_HUB_TOKEN` alias.
    `worker_image_digest` kept with a TODO (tensorhub may start stamping
    it). The false "no module reads os.environ" Settings docstring now
    describes the real library-standalone raw-read exceptions.
  - Protocol: bare `ValueError` now maps **FATAL**, not INVALID (P9) —
    typed `ValidationError` + msgspec decode errors keep INVALID;
    `ensure_local`'s unsupported-ref raise is now typed. `_sanitize`
    additionally redacts absolute filesystem paths from client-visible
    messages (P8; URLs and `owner/repo` refs survive). Deadline expiry
    proto comment fixed to match reality (FATAL, not CANCELED — P6);
    `insufficient_disk` removed from the FnUnavailable reason vocabulary
    (it is transient, RETRYABLE-only — P10). Worker stopped populating
    `ModelEvent.cache_hits/cache_misses/warmup_s` and
    `WorkerResources.git_commit` (zero Go readers — P3/P4); the proto
    fields stay pending a coordinated tensorhub trim.

## 0.15.2 (2026-07-12)

- **th#763: cold tensorhub refs block-and-serve instead of fataling the
  first request.** A snapshot-less tensorhub ref in `ModelStore.ensure_local`
  now emits `missing_snapshot` (the hub's re-mint trigger) and BLOCKS up to
  60s for the re-minted DOWNLOAD to bank a snapshot, then downloads and
  serves — the first user request per unseen ref completes instead of dying
  as the sacrificial cache warmer. When nothing arrives, the typed
  `MissingSnapshotError` now maps to `JOB_STATUS_RETRYABLE` (was FATAL via
  the catch-all): a cold worker mid-resolution never fatals a user request.
  Root cause of the ie#383 fatals is hub-side (tensorhub th#754 fold drift:
  ':prod' elided hub-side while gw#492 workers stamp it — fixed in the
  paired tensorhub PR); this half makes any residual spelling/race miss
  self-heal in place.

## 0.15.1 (2026-07-12)

- **gw#479: canonical config digests hoist child-only scalars to the
  parent.** The qwen pair's remaining split (exact-container repro):
  transformers 4.53 serialized image/video/vision token ids in
  ``text_config`` ONLY, 4.57 at the top level ONLY — same values, mirrored
  paths; parent-duplicate pruning alone could not equate them and each fp8
  lane kept booking its own 9.4GB text encoder. Child scalar duplicating
  the parent drops; child-only scalar hoists; a CONFLICTING child value
  keeps both sides (keys separate). Verified equal inside the serve image
  on the two real fp8 TE configs.


## 0.14.15 (2026-07-12)

- **gw#479: canonical config digests hoist child-only scalars to the
  parent.** The qwen pair's remaining split (live A100 pod, exact-container
  repro): transformers 4.53 serialized image/video/vision token ids in
  ``text_config`` ONLY, 4.57 at the top level ONLY — same values, mirrored
  paths, so parent-duplicate pruning alone could not equate them and each
  lane kept booking its own 9.4GB fp8 text encoder. Canonical form now:
  child scalar duplicating the parent drops; child-only scalar hoists to
  the parent; a CONFLICTING child value keeps both sides (keys separate).
  Verified equal inside the serve image on the two real fp8 TE configs.
## 0.15.0 (2026-07-12)


## 0.14.14 (2026-07-12)

- **gw#407 host-RAM admission sizes multi-slot setups by the LARGEST slot,
  not the sum.** Slots stage sequentially under the load lock and move to
  VRAM before the next slot loads; summing refused two 28GiB fp8 lanes as
  "56.2GiB incoming" on a 61GiB-RAM A100 pod that never stages more than
  one slot at a time (gw#479 J24M run19). Single-slot behavior unchanged;
  the J17 16-variant case (separate records) unchanged.


## 0.14.13 (2026-07-12)

- **ie#468 rung 2: `apply_block_window_offload` — block-window weight offload
  to pinned host RAM.** The gw#460 windows in reverse: per-block weights rest
  in (pinned) host RAM and stream to the device only for that block's
  forward; params outside the windows move to the device. Composes with fp8
  storage windows (fp8 bytes over PCIe, on-device upcast). Guaranteed-
  completion degraded rung for VRAM-constrained cards — quality-preserving,
  known-slow, never a production mode. PRECEDENCE: the
  `GEN_WORKER_FORBID_CPU_OFFLOAD=1` operator veto wins over degraded mode —
  the call raises before parking any weight (same rule as the gw#463
  OOM-demotion path). Plus `block_offload_active()` probe.

- **gw#476 fix: NVENC probe respected the encoder's minimum dimensions.**
  The boot probe encoded a 64x64 frame — below H.264 NVENC's minimum
  (145x49) — so genuinely NVENC-capable cards failed the probe with
  "Frame Dimension less than the minimum supported value" (measured live on
  an L4; the GeForce-in-SECURE-tenancy "OpenEncodeSessionEx: unsupported
  device" refusal is real and unaffected). Probe now encodes 256x256, and
  `StreamingVideoEncoder` opens the codec context eagerly inside `_open()`
  so hardware refusals that FFmpeg defers to the first `encode()` hit the
  per-encode x264 fallback instead of failing the request mid-encode.

## 0.14.12 (2026-07-12)

## 0.14.11 (2026-07-12)

- **gw#476: fast video encode path — NVENC when the silicon has it, streaming
  encode, fast presets.** New `gen_worker.video_encode`: the mp4 backend is
  probed ONCE per process (one tiny real encode — codec presence in the PyAV
  wheel is not enough; H100/A100/B200 ship without the NVENC block) and
  `h264_nvenc` (p4/vbr/cq19) is used when present, else libx264 at
  `veryfast`/CRF 18 instead of the archival default (medium/CRF 23, 5-10x the
  encode CPU for invisible gains on short generated clips). Override with
  `GEN_WORKER_VIDEO_ENCODER=auto|nvenc|x264`. `StreamingVideoEncoder` feeds
  frames to the encoder in chunks as they are produced, and
  `gen_worker.io.write_video` now accepts an iterator/generator of frame
  chunks (VAE framewise-decode seam) so long/4K clips never rebuffer a second
  raw array. Motivation: B200 gauntlet measured one 10s@1080p clip at 179.6s
  x264 encode vs 118s GPU compute; a 5s@4K probe spent ~25min encoding while
  the GPU idle-billed.
- **gw#516 (core): terminal GPU-slot release at the decode->finalize
  handoff.** `write_video` releases the request's GPU slot as soon as frames
  are on the host — the CPU encode + upload tail overlaps the NEXT request's
  denoise instead of idling the GPU. Unlike the gw#382 yield window there is
  no reacquire, so a finishing request never blocks behind its successor's
  denoise just to return; the executor's post-handler release no-ops (lease
  transitions are once-only) and drain/cancel/failure attribution are
  unchanged because the job stays in the handler until finalize completes.
  Buffered encodes take a bounded finalize permit BEFORE the release
  (`GEN_WORKER_VIDEO_ENCODE_CONCURRENCY`, default 2) so back-pressure holds
  the slot rather than stacking raw-frame buffers in host RAM. The executor
  logs `FINALIZE_OVERLAP` with slot-held vs handler-wall ms (overlap evidence
  until JobMetrics grows a slot-held field).

## 0.14.10 (2026-07-12)

- **pgw#511 hotfix: ModelRef.__post_init__ uses force_setattr.**
  `object.__setattr__` on a frozen msgspec Struct raises "can't apply this
  __setattr__" under CPython 3.12 (every serve image) while passing on 3.13
  (dev venvs + CI) — any endpoint import died at decoration time and
  discovery advertised NOTHING (J24M run16 image build gate caught it).
  `msgspec.structs.force_setattr` is the repo convention (Resources,
  Compile) and works on both.

## 0.14.9 (2026-07-12)

- **gw#479: canonical config digests prune sub-config keys duplicating the
  parent.** Live qwen evidence (fp8 lanes, A100): transformers 4.53
  serialized vision_start/end_token_id into BOTH the top-level VL config
  and text_config; 4.57 writes them only at top — materialized top-level
  values identical, but the sub-config duplicate kept two byte-identical
  text encoders on separate content keys (each lane booked its own 9.4GB
  TE; only the vae shared). Sub-config values that DIFFER from the parent
  still separate keys.


## 0.14.8 (2026-07-12)

- **ie#463: `diffusers_step_callback` gains `window=(start, end)`.** Multi-stage
  pipelines (denoise + latent-upsample refine, etc.) now compose two calls,
  each reporting into its own sub-range of the request's 0..1 progress bar,
  instead of the second stage resetting the bar to 0. `step`/`total` on the
  wire still describe progress within the current stage. Default
  `window=(0.0, 1.0)` is unchanged (fully backward compatible) — every
  existing single-stage caller is unaffected. Fixes the gap that led
  ltx-video-2.3 to hand-roll its own step callback, which omitted
  `raise_if_cancelled()` and left long video jobs uncancellable mid-run.

## 0.14.7 (2026-07-12)

- **gw#421: retire the gen-worker-repo GPU CI lane; real-GPU coverage moves
  to the e2e nightly.** Deleted the ephemeral-RunPod-4090 self-hosted-runner
  scaffolding (`.github/workflows/gpu-ci.yml`, `gpu-runner-image.yml`,
  `.github/gpu-runner/`) — it booted a 4090 per master push, the runner
  registration was RunPod-host-flakiness-prone (booted pods that never came
  online, one idle 40 min), and every check it ran either duplicated CPU CI
  or is now covered end-to-end by the e2e repo's nightly `TestJ6` on the REAL
  production path. Removed `tests/test_gpu_generation_smoke.py` (its garbage
  tripwire duplicated `e2e/quality`; its fp8-vs-bf16 SSIM assertion is now the
  J6 fp8 chapter against a real 4090, not a repo-local smoke).
- **examples/flux2-klein-image: add the fp8 lane.** A second `@endpoint`
  (`generate-turbo-fp8`, `storage_dtype="fp8"`) over the same repo so the
  nightly proves fp8-E4M3 denoiser storage matches bf16 at the same seed
  (SSIM gate). Shared components dedupe the text encoder + VAE across lanes.

## 0.14.6 (2026-07-11)

- **gw#479: per-digest inflight lock in the content-addressed blob store.**
  Two refs materializing concurrently share blobs (split-vendor lanes: 9.7GB
  of byte-identical encoder shards) — both tasks streamed into the SAME
  `.part` file, interleaved writes failed size/blake3 verification 3x, and
  the second ref died `download_failed` on every attempt (J24M runs 10-12,
  three A100 pods, ~2.5min in, while every blob verified byte-perfect in
  R2). The first task now downloads under a process-wide per-digest
  asyncio lock; siblings await it, re-check usability, and reuse the
  finished blob. Regression: concurrent two-ref materialization downloads
  a shared blob exactly once (fails without the lock).
- Releases the merged-but-unreleased th#757 forensics change:
  `download_failed` ModelEvents carry the sanitized root cause.


## 0.14.4 (2026-07-11)

- **th#757 (worker side): terminal download failures carry the root cause.**
  The generic `download_failed` ModelEvent now appends the sanitized
  exception (`download_failed: <Type>: <detail>`, 200 chars) — serve pods
  are often unreachable (no SSH/logs), making the hub log the only forensic
  surface; J24M run11's starved request was undiagnosable without it. The
  exact-match vocab strings the hub switches on (`url_expired`,
  `missing_snapshot`, `insufficient_disk`, `digest_mismatch`) are unchanged.


## 0.14.5 (2026-07-11)

- **gw#504: media-output wire contract pinned — save_image on ANY job kind
  rides the media route, renewed token included.** J19 runs 48b–52d
  post-mortem: the worker was wire-correct all along (media create carried
  request_id/job_id; the hub keyed `outputs/<request-id>/<blake3>` and
  stamped `producer_request_id` — verified in the run-51/52d hub logs). The
  runs went red because tensorhub th#724 flipped OUTPUT-OWNER attribution
  (invoked org → invoker org) between runs, which the harness's `?tenant=`
  query didn't follow. New strict stand-in-hub suite
  (`tests/test_media_output_route.py`) pins the worker half so a real
  regression can't hide behind stack-side attribution changes: producer job
  with repo-CAS routing armed + save_image → media create bound to the
  token's request/job claims, parts + complete on the media route, ZERO
  /commits-family calls; same asserts after a real ~80%-TTL capability-token
  renewal against a stand-in renew endpoint; inference-kind parity.
  Checkpoints keep the gw#471 /commits route. No runtime code change.

## 0.14.4 (2026-07-11)

- **gw#497: mypy gates CI at ZERO errors — no baseline.** The #356-era type
  debt (107 errors at audit time, 95 on the 0.14 stack) is fixed outright and
  `uv run mypy src/gen_worker` is a blocking CI step. Seam fixes, not
  suppressions: pb.Snapshot now converts ONCE into the typed
  `WorkerResolvedRepo` (`executor._snapshot_to_resolved`) and threads through
  `ensure_local` / `ensure_snapshot_async` — the dict-or-object `_field`
  duck-type coercion in cozy_snapshot is DELETED along with the legacy
  entries[]/snapshotDigest/camelCase wire tolerances (the 3-representation
  th#736-shaped seam is gone). `_PublisherMixin` declares its host contract
  (TYPE_CHECKING block) instead of 31 attr-defined suppressions;
  `cli/transport.Address` is a NamedTuple (scheme/host/port) killing the
  union-of-tuple-sizes indexing hacks; `__exit__` return types no longer
  claim exception-swallowing; `RUNTIME_FACTORIES` is typed; stale
  `type: ignore`s removed. Zero behavior change intended; suite green.

## 0.14.3 (2026-07-11)

- **gw#479 follow-up: canonical JSON-config digests in content keys.** Live
  qwen fp8 casts proved split-vendor pairs ship byte-identical component
  weights whose tiny JSON sidecars differ only in save-era serialization
  (provenance stamps `_name_or_path`/`transformers_version`/
  `_diffusers_version`, explicit class defaults vs omitted, nulls,
  transformers 4.56 `torch_dtype`->`dtype` rename) — all-file content keys
  never shared. `ModelStore.component_digests(ref, local_path=)` now hashes
  small (<=256KB) JSON sidecars CANONICALLY from the local snapshot
  (`models/config_identity.py`): structural normalization for all configs,
  plus AutoConfig `to_diff_dict()` default-folding for transformers
  `config.json`. Weights keep manifest blake3 (never hashed from disk);
  parse failures fall back to raw digests (conservative no-share). Keys are
  process-local, so folding through the installed transformers version is
  safe by construction.

## 0.14.2 (2026-07-11)

- **gw#494: transactional HelloAck re-resolution — residency re-keys, gates
  re-run.** The literal th#736 mechanic worker-side is closed. ONE pick-fold:
  `rebind_pick` (api/binding.py) is shared by the hub HelloAck path and the
  local ladder (`resolve_local_bindings`) — both now carry the round-trip
  guard. Residency booking and clearing are provably same-space: `_setup_locked`
  derives its wire refs ONCE, books under them, and stamps them as the class
  record's `held_refs`; `_vacate_record` / `_record_holding` / `_record_in_use`
  operate on `held_refs`, never a re-derivation over possibly-rebound
  `spec.models`. A resolution re-pick marks divergent ready records stale and
  vacates them (async revalidate task + vacate-on-next-setup), releasing the
  OLD resolved refs' VRAM — no orphaned bookings, pins/promotes/LoRA targets
  hit the live entry after reload. `gate_functions` is idempotent (gate-owned
  unavailable marks cleared on re-gate; setup failures survive), remembers the
  probe, and re-runs inside `apply_model_resolutions` — closing the
  startup-vs-HelloAck gate race. Regression: `tests/test_resolution_rekey_gw494.py`
  (resolve→book→re-resolve→clear leaves zero orphans; revert-to-declared;
  HF-pick rejection; re-gate idempotence; single-fold contract).

## 0.14.1 (2026-07-11)

- **gw#492: ONE ref normal form.** `gen_worker.models.refs` is now the single
  formatter/parser surface for model-ref strings. Normal form = minimal
  grammar string: `:latest` (the grammar default) elided, every other tag —
  including `prod` — stamped verbatim. New: `format_model_ref`,
  `normalize_model_ref`, `fold_ref` (grammar-correct twin of tensorhub's
  `ModelRefWithTagFlavor`), `flavor_token` (the ONE gw#488 colon-hygiene
  site), `WireRef` NewType. `Hub` default tag is `latest` (was `prod`, which
  silently resolved as latest — an explicit `tag="prod"` now addresses a real
  prod tag); discovery elides the default tag at the manifest boundary so
  hub-minted keep/routing refs stay byte-equal to worker wire refs.
  `Hub(x)` and `Hub(x, tag="latest")` are now ONE residency/GC identity.
  Deleted: `download._binding_canonical_ref`, the provider-index tag-strip
  hack (index re-keyed: exact normal form + repo-identity fallback),
  hand-rolled cell-ref parsing in compile_cache/trt_engine (now
  `parse_cell_ref` via `parse_model_ref`), 3 inline `.replace(":", "-")`
  copies. Shared grammar vectors gain `canonical` normal-form fields +
  `:latest`/`:prod` vectors (tensorhub copy sync = filed follow-up).
  Grep-guard `tests/test_ref_normal_form.py` rejects new ad-hoc grammar
  sites; round-trip vector test pins `format(parse(s))`.
## 0.13.25 (2026-07-11)

- **gw#479: content-keyed shared components + transformer lanes.**
  `LoadedComponentKey` identity is now the component's CONTENT — the sorted
  blake3 digest set of its files (`content_set_digest`) plus load facts
  (dtype, quant/storage_dtype + config digest, device, placement, component
  name, adapter overlay). ref/revision drop out of identity (readable label
  only), so byte-identical components mirrored under different Hub refs
  share ONE in-memory copy. Executor: class records binding 2+ pipeline
  slots get lane loading — the shared set loads once via `acquire_shared`
  (VRAM counted once, refcount-held), later slots inject the same module
  objects into `from_pretrained` and load only exclusive weights; each
  lane's residency entry is its exclusive module set, so the existing
  make_room/LRU ladder swaps ONLY the transformer (dual-resident when the
  budget admits, swap-mode otherwise). New `@endpoint(route=)`:
  `route(payload) -> slot names` makes per-request promote/pin selective —
  swap-mode lanes never thrash both transformers. Telemetry: promote/demote
  are timed + counted per entry (`Residency.transition_stats()`); wire
  `ModelEvent.duration_ms` now carries swap walls; lanes appear as distinct
  refs in `Hello.models`. `apply_fp8_storage` is idempotent per module
  (shared injected modules are never double-hooked).
  `load_from_pretrained(components=)` forwards preloaded modules. Offload
  placement and sharing are mutually exclusive (guarded, monolithic
  fallback). Sharing engages only when wire snapshots carry digests and
  keys collide; single-slot records are byte-for-byte unchanged.

## 0.13.23 (2026-07-11)

- **th#721: adaptive RAM tier — host-RAM probes are cgroup-aware.**
  `get_total_ram_gb` / `get_available_ram_gb` now return
  min(/proc/meminfo, cgroup memory limit) via `probe_host_ram()` (cgroup v2
  `memory.max` walked root→self, v1 `memory.limit_in_bytes` fallback).
  RunPod containers see their real 31GB cgroup cap instead of the host's
  62GB meminfo, so warm-tier admission (`make_room_ram`, size-aware demote
  floor) spills pipelines to disk instead of the kernel SIGKILLing at the
  ceiling (tensorhub ie#357, wan-2.2 VAE decode). A one-time `RAM_BUDGET=`
  boot line names the derived budget, its source (cgroup vs meminfo), and
  the floor.
- **th#721: `memory_gb` removed from `ctx.compute`.** Host RAM is not
  provider-selectable; the endpoint adapts to the RAM the pod delivers.

## 0.13.21 (2026-07-11)

- **gw#468: env-gate sweep — every ambient worker knob reads through the typed
  `Settings` loader.** New fields (same env names): `TENSORHUB_URL/_TOKEN/
  _CACHE_DIR/_CAS_DIR`, `CIVITAI_API_KEY` (alias `CIVITAI_TOKEN`),
  `COZY_HF_DOWNLOAD_STALL_TIMEOUT_S/_MAX_SECONDS`, `COZY_HF_MAX_REPO_BYTES`,
  `GEN_WORKER_ATTACHED_LORA_MAX/_MAX_BYTES`, `GEN_WORKER_COMPILE_CACHE/
  _CACHE_URL/_ALLOW_COLD`, `WORKER_IMAGE_DIGEST`, `WORKER_GIT_COMMIT`;
  `HUGGING_FACE_HUB_TOKEN` resolves as an `HF_TOKEN` alias. Loader gains a
  cached `get_settings()` accessor, alias resolution (primary env wins when
  non-empty), and forgiving source coercion (empty values fall back to struct
  defaults; bools accept `1/true/yes`). `compile_cache.apply()` takes an
  explicit `allow_cold` for the producer path (env self-set kept for spawned
  compile workers). Guard test `tests/test_env_surface.py` fails on any raw
  `os.getenv`/`os.environ` outside `config/` not on the plumbing allowlist;
  survivors documented in `docs/environment.md`.
## 0.13.19 (2026-07-10)

- **gw#471: checkpoint saves publish via tensorhub's real `/commits` API.**
  `ctx.save_checkpoint` / `save_checkpoint_bytes` / `open_checkpoint_stream`
  spoke a phantom `POST /repos/:o/:r/revisions` open-session dialect that
  tensorhub deleted in th#514/#515 (2026-07-03) — every checkpoint save since
  image 0.3.6 died with `upload_session_open failed (404)` (J19 run43), and
  mid-run checkpoint events vanished. The stream finalize now publishes each
  checkpoint as ONE commit through `gen_worker.convert.hub.HubClient` (the
  same client conversion's publish_flavors uses daily): create commit with
  the add operation (blake3 rolled during write) → part PUTs / transfer
  grant → complete → finalize. Each save materializes a real finalized repo
  revision (the old session path never finalized at all). The destination
  repo auto-creates server-side under the job's create_repo grant;
  `set_repo_spec` fields ride the commit body. `_upload_session.py` and the
  ctx session-manager wiring are deleted. `HubClient.commit` grew an optional
  `part_progress` callback so byte-level upload progress events keep flowing.
- **gw#471 scope-add: upload failures are REPORTED, not just logged.**
  Checkpoint and media stream-upload failures emit a typed
  `request.warning` event `{code: artifact_upload_failed, kind:
  checkpoint|sample, ref, step_number?, error(≤500), attempt}` through the
  ctx emitter before the exception propagates — the phantom-route breakage
  was invisible for a dozen runs.
- The gw#453 route test's stand-in hub now serves ONLY tensorhub's real
  route table and 404s everything else (it previously accepted any path —
  which is how the phantom dialect slipped through), asserting the full
  commit sequence for a >256MiB checkpoint plus repo-absent first-publish.

## 0.13.16 (2026-07-10)

- **gw#465: boot-prefetch model-op batches no longer fail systematically.**
  Three worker-side fixes for the paired `download_failed` (variant) +
  `load_failed` (companion vae) signature seen on every J23 GPU worker:
  - `ModelOp{LOAD}` no longer cascades: a LOAD for a shared companion ref
    (one vae bound to every variant of a family) satisfied by a READY
    instance just touches/promotes it — it never cold-sets-up sibling
    variant specs. A cold LOAD sets up exactly ONE spec whose every slot is
    materializable.
  - The store remembers every digest-carrying snapshot per ref, so
    snapshot-less ops (LOAD, companion-slot setups) can materialize refs the
    hub already resolved. Stale URLs self-heal via the url_expired re-mint.
  - A tensorhub ref with no snapshot anywhere is a deterministic local miss:
    typed `MissingSnapshotError`, failed FAST (no DOWNLOADING ghost event, no
    1s+4s retry burn — the observed ~5s failure) with its own contract
    vocabulary `missing_snapshot` instead of a phantom `download_failed`;
    the hub re-mints and re-sends DOWNLOAD (tensorhub-side handler), and the
    function is never disabled by it.
- **gw#469: unavailable ladder rungs are skipped, and no rung renders with
  broken dtype.**
  - The emergency bnb-nf4 rung is gated on bitsandbytes importability: absent
    from the endpoint image -> the rung is SKIPPED with a logged reason (the
    offload ladder carries the load), never attempted into a
    `PackageNotFoundError` setup_failed.
  - A `force_upcast` VAE (SDXL family) is never hook-managed by any offload
    rung (gw#441): group offload excludes it (`exclude_modules`), the
    model/sequential rungs exclude it via diffusers'
    `_exclude_from_cpu_offload`; it stays resident on the execution device so
    the pipeline's own upcast dance works — no more Half/float decode fatals.

## 0.13.15 (2026-07-10)

- **fp8: SM-aware ladder ordering + remove the pre-Ada fp8 refuse-bug.**
  Stored `#fp8` flavors upcast to bf16 at compute (fp8-E4M3 bytes resident,
  per-layer bf16 upcast — no fp8 silicon required), so they serve on ANY CUDA
  card; the old `FP8_FLAVOR_MIN_SM=89` refusal is deleted. Preference is now
  SM-conditional: SM>=89 (Ada/Hopper/Blackwell) prefers fp8 over bf16 (faster
  AND smaller); SM<89 prefers bf16-if-it-fits with fp8 as a fit fallback.
  nvfp4 stays SM-gated (genuine Blackwell-native format, no upcast path).
  - Single-sourced constants: `loading.EMERGENCY_FIT_FACTOR` derives from
    `ladder.EMERGENCY_NF4_VRAM_FACTOR`; the private `FP8_FLAVOR_MIN_SM` is gone.
  - SM-conditional ordering encoded in the shared Go/Py conformance vectors
    (byte-identical with tensorhub).
  - GPU smoke lane: bf16-vs-fp8 same prompt+seed generation asserts
    SSIM >= 0.88 (gated by `GEN_WORKER_GPU_SMOKE`; skips on CPU).

## 0.13.14 (2026-07-10)

- **Remove the CPU-offload serveability veto — a worker runs DEGRADED, never
  refuses (Paul's ruling 2026-07-10).** `plan_serve`/`gate_functions` no
  longer consult `GEN_WORKER_FORBID_CPU_OFFLOAD` to mark a function
  unserveable: a function that only fits via CPU/disk offload (or CPU-only)
  now SERVES degraded and reports `FnDegraded`, instead of gating off with
  `offload_forbidden`/`cuda_unavailable`. Gen workers don't offload because
  we want them to — they do it out of necessity; better to run degraded than
  not run at all. The orchestrator hears every degraded serve and owns moving
  the release to a bigger card (tensorhub th#208 → active reschedule).
  - `Resources(strict_vram=True)` is the sole opt-out (salvaged from gw#139):
    an AUTHOR who would rather refuse than serve slowly (compiled fixed-shape
    graphs, TensorRT engines). It skips only the CPU-touching rungs
    (offload / cpu); the on-GPU runtime rungs (fp8 storage, emergency 4-bit)
    still serve.
  - **Box protection preserved.** `GEN_WORKER_FORBID_CPU_OFFLOAD=1` still
    raises at actual pipeline PLACEMENT time (`memory.place_pipeline` /
    `apply_low_vram_config`) — the dev-box kill-switch that stops a
    directly-invoked local worker from melting this shared box with real
    CPU-offloaded inference. Its meaning narrows from "refuse to serve" to
    "refuse to actually place real weights on CPU". The orchestrated path is
    covered by tensorhub's th#657 local-provider capability gate.
  - Supersedes gw#139 (veto-removal). Complements the merged gw#463
    (0.13.11, reactive OOM demotion): the plan-time veto is gone AND a
    runtime CUDA OOM still demotes down the same ladder.

## 0.13.13 (2026-07-10)

- **gw#464 follow-up: checkpoint-key translation works on transformers 4.x.**
  `te_fp8_castable_keys` now falls back to the 4.x class-attr
  `_checkpoint_conversion_mapping` (regex `re.sub` chain, the
  `from_pretrained` mechanism of that line) when the 5.x
  `conversion_mapping`/`core_model_loading` modules are absent — the
  conversion fleet image locks transformers 4.57. Verified against the
  real LTX-2.3 Gemma3 TE metadata: 498/498 loader-castable weights match
  their stored (old-layout) key names.

## 0.13.12 (2026-07-10)

- **gw#464: storage-side fp8 for text encoders — `streaming_fp8_snapshot(te_components=...)`.**
  The gw#460 loader casts transformers TEs with block-window weight-only
  rules; the writer can now produce the same cast as a STORED flavor with
  zero drift: `te_fp8_castable_keys()` meta-instantiates the component's
  architecture from its config, runs the loader's own
  `_fp8_block_windows` walk, and maps the checkpoint's stored key names
  onto the graph with transformers' own load-path renaming (old-layout
  Gemma3 `language_model.model.*` resolves exactly like `from_pretrained`;
  zero matches is a hard error, never a silent no-op). New
  `streaming_fp8_te_cast()`, `FP8_TE_COMPONENTS` (drift-guarded against
  the loader constant). Embeddings/norms/biases/tied lm_head pass through
  at source precision.

## 0.13.11 (2026-07-10)

- **gw#463: CUDA OOM never fatals — degraded mode is the fit-ladder's formal
  terminal rung, plan-time AND reactive.** One unified ladder: `plan_serve`'s
  offload verdict now drives `place_pipeline`'s starting placement (a plan
  that already says "can't fit resident" never pays the doomed resident
  attempt — ie#369 measured 9-28 min for 70 GB models), and a runtime CUDA
  OOM is a ladder *transition*, not a failure. Two core catch-sites:
  (a) setup/load — an OOM inside placement flushes and demotes one offload
  rung (`model_offload -> group_offload -> sequential`) and retries;
  (b) mid-inference — the executor flushes the CUDA cache, demotes the
  function's resident pipelines one rung, and retries the request ONCE in
  degraded mode; the job only fails (RETRYABLE `out of memory`, never FATAL)
  if degraded also fails. Demotions are sticky per model until reload and
  learned in-process (`Executor.degraded_floor`), so subsequent loads start
  at the learned rung. Every transition logs
  `DEGRADED_MODE=engaged fn=... model=... phase=... rung=a->b needed_gb=..
  free_gb=..`, updates the ServePlan, emits a per-request `ctx.log` event,
  and re-emits `FnDegraded` with `ran="offload:<mode>"` (lifecycle dedupe is
  now per-rung; the emit passes also run on unchanged StateDelta bytes —
  previously a plan change without a delta change was never reported).
  Allocator-flavored `RuntimeError`s ("CUDA error: out of memory",
  CUBLAS/CUDNN alloc failures) now classify as OOM (`is_cuda_oom`) instead of
  falling through to FATAL. `GEN_WORKER_FORBID_CPU_OFFLOAD=1` still vetoes
  every CPU-touching demotion (dev-box guard). Generalizes ltx-video-2.3's
  bespoke OOM fallback (ie PR #22) into the worker core.

## 0.13.10 (2026-07-10)

- **gw#462: conversion worker hardening — disk preflight, scratch hygiene,
  publish resume.** Two live J24 ingest killers fixed. (1) Disk: `run_clone`
  preflights free space against the source plan's known file sizes
  (`COZY_CONVERT_DISK_HEADROOM`=2.5 + 2 GiB margin) and raises typed
  `CloneDiskSpaceError` ("need ~X GiB free, have Y GiB") before any download,
  instead of ENOSPC mid-stream; the workdir is removed after EVERY job —
  success and failure (`COZY_CONVERT_RETAIN_WORKDIR=1` keeps a failed job's
  scratch for debugging) — and each run sweeps stale scratch left by crashed
  predecessors (flock-free + idle past `COZY_CONVERT_SCRATCH_TTL_S`=1h).
  (2) Publish: a `409 staging_object_missing` from `/complete` (th#699: the
  hub lost the staged bytes; retrying complete can never succeed) now
  re-opens that ONE file's upload (`POST .../commits/<rev>/uploads`) and
  re-sends just it, bounded at 2 re-uploads, instead of fataling the whole
  job at the last shard. Part PUTs and hub POSTs split connect (15s) from
  read timeouts (gw#456 parity on the upload side); publish errors name the
  file, attempt count, and last status. Counterpart: tensorhub th#699
  (staging retained on transient verify failures + the re-open endpoint).

## 0.13.9 (2026-07-10)

- **gw#453: training contexts arm repo-CAS checkpoint routing.** The executor
  now populates producer contexts with `kind`, `destination_repo` (reserved
  `payload.destination.ref` struct or flat `payload.destination_repo` scalar)
  and the cap token's `job_id` claim, so `ctx.save_checkpoint` /
  `open_checkpoint_stream` on `kind=training` jobs ride the job-bound repo-CAS
  checkpoint route (multi-GB per-file grant) instead of silently falling back
  to the media route (256 MiB/file cap — J19 run41 trained 500/500 steps then
  died `file_too_large` publishing the final LoRA). Output-stream routing is
  now split by artifact kind: checkpoint streams -> repo-CAS when the
  destination scope is armed; asset streams (sample images, media outputs)
  always -> media route under the `upload_media` grant. A training
  `save_checkpoint` with no destination scope now fails loudly instead of
  riding media.

## 0.13.8 (2026-07-10)

- **th#683 P3: complete the serve-time adaptive-fit ladder + structured degradation events.** `plan_serve` now walks `bf16 -> fp8 -> nvfp4 -> runtime nf4 4-bit -> CPU/disk offload -> CPU-only`, picking the highest-quality lever that fits the actual card. Stored fp8/nvfp4 flavors are HW-gated (fp8 -> SM89 Ada/Hopper, nvfp4 -> SM100 Blackwell); a runtime fp8-E4M3 storage rung needs no fp8 silicon. Never refuses on the recommended-VRAM hint. New `FnDegraded` event (`{function, wanted, ran, reason, est_latency_multiplier, recommended_vram_gb}`) rides the orchestrator transport (cozy-local emits nothing; the honest-guidance advisory is the terminal surface).
- **th#697 P1: precision-class + placement model; publish-time placement stamping.**

## 0.13.7 (2026-07-10)

- **gw#459: `TrainingMetric` validation fields.** `val_loss`, `best_step`, and
  `advice` (all optional, omitted when None) join the typed
  `request.training_metric` payload, and `TrainingContext.training_metric`
  accepts them. Any event carrying `val_loss` bypasses the 5s min-interval
  throttle like first/last — val points are sparse and every one must reach
  the hub. Counterpart: tensorhub th#696 (val series + recommended checkpoint
  in the training-metrics API).

## 0.13.6 (2026-07-10)

- **gw#456: clone downloads can no longer hang forever.** huggingface_hub's
  default HTTP client has no timeout (and `HfApi.repo_info` passes an explicit
  `timeout=None`), so one stalled connection wedged clone-huggingface jobs —
  and the tensorhub demand rows dedup-joined to them — indefinitely (observed
  live: CLOSE-WAIT sockets, empty workdirs, frozen progress). New
  `gen_worker/net.py` installs a process-wide timeout floor via
  `set_client_factory` (`COZY_HTTP_CONNECT_TIMEOUT_S`=15,
  `COZY_HTTP_READ_TIMEOUT_S`=60; explicit numeric timeouts win; the read
  timeout doubles as the per-socket stall detector; requests fallback for
  hub 0.x). Clone ingest now runs `snapshot_download` under the gw#379 stall
  watchdog — real byte progress during clone downloads — with bounded
  resumable retries (`COZY_CLONE_DOWNLOAD_ATTEMPTS`=3; hf_hub Range-resumes
  `.incomplete` files), raising typed `CloneDownloadError` when exhausted.
  Civitai files get a bounded per-file retry (`COZY_CIVITAI_DOWNLOAD_ATTEMPTS`=3).
  Tensorhub-side demand-row TTL sweep is th#694.

## 0.13.5 (2026-07-10)

- **gw#457: `resolve_dataset` rides the DATASET-V2 async snapshot contract
  (th#691).** `GET /datasets/:id/materialize` may now answer
  `202 {status: building, state_version, retry_after}` while the snapshot
  builds in the background; `fetch_materialize_manifest` polls until ready —
  long-polling via `?wait=30` (ignored by pre-v2 hubs), honoring `retry_after`
  with capped exponential backoff, within an overall budget (default 30 min,
  ≥ the hub's 20-min build budget; `resolve_dataset(..., budget_s=)` to
  override) and respecting job cancellation mid-poll. A typed
  `snapshot_build_failed` raises the new `SnapshotBuildFailedError`
  (`gen_worker.api.errors`, carries `error_code`) instead of a generic
  non-2xx RuntimeError; transient transport errors and bare 502/503/504
  retry within the same budget (hub restart mid-build). The executor's
  `payload.datasets` pre-materialization path (gw#425) goes through the same
  helper, so training jobs survive a live 202 window. Backward-compatible:
  today's hub never returns 202. Per-shard download retries unchanged.

## 0.13.4 (2026-07-10)

- **th#683 P3: serve-time adaptive fit — the worker never refuses on the VRAM
  hint.** New `models/serve_fit.plan_serve` decides, per (already
  flavor-resolved) function, HOW it runs on the actual card: native ->
  emergency 4-bit -> CPU/disk offload -> CPU-only, choosing the highest-quality
  lever that fits. `gate_functions` now consults it instead of hard-refusing on
  `insufficient_vram`/`cuda_unavailable`: a model bigger than the card serves
  via emergency-quant or offload (fit over speed, the primary lever at the low
  end), and a GPU function with no GPU falls back to CPU-only — all behind loud
  honest-guidance warnings (realistic latency multiplier + the ideal card),
  never a silent refusal. A function is unserveable only on a genuine
  incompatibility (compute capability / missing quant library) or when the sole
  lever here is a CPU-touching placement this box forbids
  (`GEN_WORKER_FORBID_CPU_OFFLOAD=1` — those runs belong on the GPU lane).
  `run --list` / `serve --list-functions --json` now carry `serveable`,
  `run_mode`, `est_latency_multiplier`, `recommended_vram_gb`, and an
  `advisory` string. Added `memory.cpu_offload_forbidden()` (non-raising
  predicate).

- **ie#455: log functions stuck waiting on hub `ModelOp{DOWNLOAD}` snapshots.**
  `startup()` now WARNs, naming each function still in `loading_functions` and
  the repo refs it awaits, instead of silently advertising `fns=[]`. Surfaces
  the empty-`keep` serve deadlock (a release registered without model
  `bindings`) that previously produced a silent, GPU-independent hang.

## 0.13.3 (2026-07-10)

- **gw#450: `TrainingContext.training_metric` — typed step/loss/lr/it_s/eta
  channel.** Emits a `request.training_metric` event (msgspec `TrainingMetric`
  payload: `{step, total, loss, lr?, it_s?, eta_s?}`, None fields omitted) over
  the ctx emitter / JobProgress envelope from gw#438, so tensorhub can
  downsample-persist a chartable series (th#681). Built-in min-interval
  throttle (`metric_min_interval_s`, default 5s); the first and the final
  (`step >= total`) metric always emit. `ctx.progress` stays the
  human-readable stage-text channel.

## 0.13.2 (2026-07-10)

- **gw#452: media uploads target the capability-token-bound owner.**
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
