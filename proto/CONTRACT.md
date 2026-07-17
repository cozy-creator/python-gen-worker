# Worker <-> Orchestrator Protocol Contract

Proto: `worker_scheduler.proto`, package `cozy.scheduler`, service
`WorkerScheduler`, one RPC: `Connect(stream WorkerMessage) returns (stream SchedulerMessage)`.
This REPLACES the old `scheduler.v1` proto entirely. No compat, no negotiation:
`ProtocolVersion.PROTOCOL_VERSION_CURRENT = 3` is the only accepted version.

Roles: **W** = gen-worker (Python), **O** = orchestrator (tensorhub Go).
Every field below names its producer and consumer; anything without both was deleted.

---

## 1. Connection lifecycle

**Dial.** W dials the orchestrator gRPC address (TLS in prod, plaintext in dev)
and opens `Connect` with metadata `authorization: Bearer <worker JWT>`.
O verifies the JWT before reading any message; claims are authoritative:
`sub` = worker_id, `release_id` claim = release. Auth failure ⇒
`UNAUTHENTICATED`, stream closed. Dev mode (no verifier configured +
allow-unauthenticated flag) takes `Hello.worker_id`/`release_id` at face value.
Identity is per connection — no message carries worker identity.

**Hello / HelloAck.** W's FIRST message MUST be `Hello`. O MUST reply
`HelloAck` before sending anything else. W MUST NOT send any other message
before receiving `HelloAck`. Any protocol-order violation ⇒ O closes with
`FAILED_PRECONDITION`.

Version check: `Hello.protocol_version != PROTOCOL_VERSION_CURRENT` ⇒ O closes
with `FAILED_PRECONDITION` `protocol_version_mismatch`. W likewise validates
`HelloAck.protocol_version` and exits with a fatal log on mismatch (a stale
orchestrator build).

**Hello deadline.** O bounds the wait for the connection-opening `Hello`
(15s default). An authenticated client that never sends Hello gets
`DEADLINE_EXCEEDED` `hello_timeout` and the stream is closed.

**Redirect (`not_leader`).** If this replica does not own the worker's release
partition, O closes the stream with `FAILED_PRECONDITION` and status message
`not_leader:<host:port>`. The redirect address is SCHEMELESS: W inherits the
TLS/plaintext mode of the connection it is currently on (a TLS deployment is
TLS on every replica — W must never downgrade to plaintext on redirect).
W reconnects to that address immediately (no backoff), max 3 consecutive
redirect hops, then falls back to the configured address list with backoff.
This gRPC status is the ONLY routing mechanism — there is no message-level
forwarding or relay.

**Duplicate stream.** A second Connect from the same worker_id SUPERSEDES the
first: O admits the new stream and closes the old one with `ABORTED`
`worker_stream_superseded`. A half-open zombie connection never blocks a live
reconnect, and a superseded stream is never an auth failure.

**Reconnect.** On any stream error/close (except Drain-initiated) W reconnects
with jittered exponential backoff: `delay = rand(0, min(30s, 1s * 2^n))`
(full jitter), reset after a connection survives 60s. Every reconnect starts
from scratch: new `Hello` with full snapshots.

**Worker JWT rotation (kubelet model).** The worker JWT is SHORT-lived (4h
default) and W never handles its expiry: O tracks each connected worker's
token lifetime (recorded at pod create) and pushes a renewed JWT via
`TokenRefresh` at ~80% of the TTL, iff the pod is still tracked (active
admission record) and not draining. W stores the newest pushed token and
dials every reconnect with it; the live connection is unaffected (identity is
per connection). O admits the superseded token alongside the current one
until the old token's own `exp` ends it, so a reconnect racing the push never
locks a worker out. A worker that misses every rotation (e.g. persistent send
failures) eventually fails reconnect auth with `UNAUTHENTICATED` and exits
under the normal strike rules — O's disconnect handling reaps the pod.
`TokenRefresh` is an additive proto3 field: a worker build that predates it
ignores the message (unknown oneof member) and keeps its boot-time token.

**In-flight reconcile on Hello.** `Hello.in_flight` lists every job W still
owns: currently executing PLUS completed-but-result-not-yet-shipped. O
reconciles against its assignment state:

- entry matches O's record (assigned to this worker, same `attempt`) → keep;
  O awaits the (re)sent `JobResult`.
- entry unknown to O or `attempt` stale → O sends `CancelJob{request_id, attempt}`
  so W frees the resources; O drops any late result for that attempt.
- request O has assigned to this worker but MISSING from `in_flight` → the
  worker lost it (crash/restart); O requeues with `attempt+1`.

Immediately after `HelloAck`, W first replays active typed host-RAM failures
and undelivered satisfying progress, then flushes its buffered unsent
`JobResult`s (§4 results are never dropped). A satisfied ref replays only its
self-contained progress event, never its obsolete failure (an older hub safely
ignores the additive progress enum). The finite HelloAck baseline and this
reconnect replay use a nonblocking prepend lane and are idempotent for a
repeated midstream `HelloAck`. Newer logical state replaces every older
same-identity queued copy. Host-capacity evidence uses a latest-generation-
per-ref delivery lane, so an older FAILED cannot remain behind a newer
PROGRESS or a retained result.

**HelloAck re-send.** O re-sends `HelloAck` mid-connection whenever release
config or desired model residency changes. Full-replace semantics: W
overwrites `file_base_url`, resolutions, and desired residency wholesale.
The same residency generation MAY be resent with refreshed snapshot URLs;
only a generation lower than the latest observed generation is stale.

**Send-queue policy (W).** One outbound queue with a bounded event/progress
lane. `JobResult` is NEVER dropped: results persist across reconnects until
written to a live stream and do not consume bounded event/progress capacity.
Under overflow, drop order is: `JobProgress` (oldest first) → never anything
else; if the bounded lane is still full, W blocks the producer (backpressure)
rather than dropping `ModelEvent`/`StateDelta`/`FnUnavailable`. The finite
HelloAck baseline/reconnect prepend described above is also outside that bound
so the pre-send handler cannot deadlock behind preserved results or ordinary
events queued while disconnected. Normal connected producers still use the
bounded lane and its backpressure. Typed host-capacity evidence is finite
state and bypasses that ordinary bound: only the newest undelivered generation
per ref is queued. Successful `stream.write` is its delivery boundary (the
same boundary used by durable `JobResult`); write failure/reset leaves
executor-owned evidence for the next HelloAck replay.

---

## 2. Fencing (`attempt`)

One integer replaces assignment epochs, the duplicate-request handshake, and
resume epochs.

- **Who bumps:** ONLY O. `attempt` starts at 1 for the first dispatch of a
  request_id and is incremented every time O (re)queues the request (worker
  loss, retryable result, reconcile requeue). W never invents or increments
  attempts; it echoes the value from `RunJob`.
- **W side:** job identity is `(request_id, attempt)`. If W receives `RunJob`
  for a pair it already holds (retransmission), it re-sends `JobAccepted` and
  does NOT start a second execution. If W receives `RunJob` for a request_id
  it holds with a DIFFERENT attempt, it aborts the old attempt silently (no
  result for the old attempt) and runs the new one.
- **O side:** for each request O tracks the current attempt. Any
  `JobAccepted` / `JobProgress` / `JobResult` whose `attempt` differs from the
  current one is logged and dropped — a stale worker cannot corrupt state or
  double-complete a request. `CancelJob` always carries the attempt O believes
  current; W ignores a `CancelJob` whose attempt it doesn't hold.
- **Dispatch retry:** after sending `RunJob`, O retransmits the SAME
  `(request_id, attempt)` every 5s until `JobAccepted` or a terminal
  `JobResult` arrives, up to 3 sends on one connection; then O requeues with
  `attempt+1` (possibly to another worker).

---

## 3. Liveness

gRPC HTTP/2 keepalive is the ONLY liveness mechanism. No app-level heartbeats,
no padded keepalive envelopes, no inbound-silence watchdogs, no external
lease refresh.

- **O (server):** `KeepaliveParams{Time: 20s, Timeout: 10s}` (no
  MaxConnectionIdle/Age), `EnforcementPolicy{MinTime: 5s, PermitWithoutStream: true}`.
- **W (client channel options):** `grpc.keepalive_time_ms = 20000`,
  `grpc.keepalive_timeout_ms = 10000`, `grpc.keepalive_permit_without_calls = 1`,
  `grpc.http2.max_pings_without_data = 0`.

A dead peer is detected within ≤30s on both sides; detection surfaces as a
stream error → W reconnects (§1), O treats the worker as gone: it requeues
that worker's assigned requests (attempt+1) unless the worker returns and
re-claims them via `Hello.in_flight` reconcile first. Stream state is the
worker-liveness authority — there is no heartbeat-timestamp pruner.

**Load balancer requirements.** Any proxy/LB between W and O's gRPC port MUST:
- forward HTTP/2 PING frames end-to-end (no PING termination at the proxy —
  keepalive is the ONLY liveness mechanism and must observe the real peer);
- impose no idle/stream timeout below 30s (the keepalive detection window);
  long-lived idle bidi streams are the steady state, not an anomaly;
- not multiplex/round-robin frames of one TCP connection across backends —
  a Connect stream is pinned to one replica for its lifetime (routing across
  replicas is the `not_leader` redirect, never the LB).

---

## 4. Messages

### Hello (W → O)
Sent once per connection, first message.

| field | producer | consumer | semantics |
|---|---|---|---|
| `protocol_version` | W constant | O version gate | must equal `PROTOCOL_VERSION_CURRENT` |
| `worker_id` | W settings | O (dev mode identity; prod cross-check vs JWT `sub`, mismatch ⇒ close) | worker identity |
| `release_id` | W settings | O (dev mode; prod cross-check vs JWT claim) | release this worker serves |
| `resources` | W hardware probe at boot | O placement filters + supply | static snapshot, see below |
| `state` | W lifecycle | O availability/supply | initial dynamic state (same shape as StateDelta) |
| `models` | W model cache | O residency index (cache-aware routing baseline) | full residency snapshot; replaces any prior state O held for this worker |
| `in_flight` | W executor + result buffer | O reconcile (§1) | jobs W still owns |

### WorkerResources (embedded in Hello)
Static per-boot facts. Never re-sent mid-connection.

| field | producer | consumer | semantics |
|---|---|---|---|
| `gpu_count` | W probe | O placement (multi-GPU slotting) | number of GPUs |
| `vram_total_bytes` | W probe | O placement VRAM budgeting | total VRAM |
| `gpu_name` | W probe | O placement/tier display | GPU model name |
| `gpu_sm` | W probe | O placement SM-floor filter | compute capability, e.g. "89" |
| `installed_libs` | W probe | O placement library filters | e.g. "torchao" |
| `image_digest` | W env | O resource-profile keying + triage | worker image sha256 |
| `git_commit` | W env | O triage/display | build provenance |
| `instance_id` | W env | O provider bookkeeping (e.g. runpod pod retire) | cloud instance id; empty for local |
| `torch_version` | W runtime probe | O Inductor-cell compatibility filter | full actual `torch.__version__`; O fails closed when absent or major.minor differs from the selected cell |

### ModelResidency (embedded in Hello)

| field | producer | consumer | semantics |
|---|---|---|---|
| `ref` | W model cache | O residency index | canonical ref string (one grammar; tag/flavor/digest inside the ref) |
| `tier` | W model cache | O cache-aware placement (VRAM > RAM > DISK > cold) | highest current tier |
| `vram_bytes` | W measured at load | O model-size cache / VRAM packing | set when tier=VRAM |
| `snapshot_digest` | W exact materialization result | O immutable residency identity | digest of the snapshot that actually produced this residency; empty means unknown, never "current tag target" |
| `residency_generation` | W desired-state receiver | O per-ref event fence | desired generation under which this exact snapshot was materialized; zero means a legacy/unknown observation |

### InFlightJob (embedded in Hello)

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | W executor | O reconcile | job id |
| `attempt` | W (echo of RunJob) | O reconcile fencing | attempt W holds |

### HelloAck (O → W)
Reply to Hello; re-sent on config change (full replace).

| field | producer | consumer | semantics |
|---|---|---|---|
| `protocol_version` | O constant | W version gate | symmetric validation |
| `file_base_url` | O config (`FileAPIBaseURL`) | W upload stack / blob-ref PUT flow (§8) | tensorhub HTTP base for capability-token calls |
| `keep` | legacy v2 release config | none in v3 | retained field number only; v3 W ignores it |
| `resolutions` | O precision resolver | W endpoint bindings | full-replace declared-ref to worker-specific ref/cast picks |
| `desired_residency` | O scheduler/controller | W model reconciler | full-replace per-worker disk and hot-instance goal (§7) |

### DesiredResidency / DesiredInstance (embedded in HelloAck)

| field | producer | consumer | semantics |
|---|---|---|---|
| `generation` | O desired-state store | W observation fence | monotonic desired revision; observation does not imply convergence |
| `disk_refs` | O locality scheduler | W downloader + disk GC | ordered refs to materialize and protect from eviction while headroom allows |
| `hot` | O locality scheduler | W endpoint setup | ordered runnable instances; each names a function and its complete `ModelBinding` slot map |
| `snapshots` | O resolver | W downloader | snapshots for every disk/hot ref that needs one; same semantics as `RunJob.snapshots` |
| `hot[].function_name` | O release manifest | W registry lookup | endpoint function whose persistent instance should be warm |
| `hot[].models` | O binding resolver | W existing RunJob binding path | complete slot to immutable-ref map; incomplete or mismatched maps are rejected |

### StateDelta (W → O)
Full-replace snapshot of ALL dynamic worker state. Edge-triggered: sent
whenever any field changes (function becomes available, phase advances,
free VRAM shifts materially — quantize to ≥5% of total or a residency change
to avoid chatter). Never periodic. O overwrites its copy wholesale.

| field | producer | consumer | semantics |
|---|---|---|---|
| `phase` | W lifecycle | O supply/boot-progress display, cold-start hold | startup phase; moves forward only until READY; ERROR is terminal |
| `available_functions` | W registry + model gating | O dispatch eligibility + autoscale supply | dispatchable now |
| `loading_functions` | W registry | O supply (counts 0 capacity) + display | present but models still materializing; disjoint from available |
| `free_vram_bytes` | W CUDA probe | O placement (free-VRAM ladder) | measured free VRAM |
| `finalizing_jobs` | W executor (gw#516) | O drain/retire gating + worker status display | jobs past the decode→finalize handoff: GPU slot released, encode/upload tail running, `JobResult` unshipped. GPU-idle alone is NOT work-idle |
| `observed_residency_generation` | W desired-state receiver | O controller status | latest non-stale generation accepted; not a convergence claim — `ModelEvent`/`Hello.models` report actual state |
| `compile_targets` | W executor | O compile-cell planner + dispatch fence | full-replace snapshot of exact READY live pipeline incarnations; omission removes the old address immediately |

### CompileTarget (embedded in StateDelta)

One row addresses one exact live pipeline object. It is session-local: setup
mints `incarnation_id`, and vacate/reload mints another even when checkpoint
bytes are unchanged. O MUST NOT reconstruct or match a target from family
alone.

| field | producer | consumer | semantics |
|---|---|---|---|
| `incarnation_id` | W setup | O adoption + RunJob planner | opaque live-object address; immutable for the row lifetime |
| `family` | W endpoint `Compile` contract | O cell compatibility | exact non-empty Forge family |
| `pipeline_weight_lane` | W loaded object | O lane matching | observed lane such as plain or `w8a8`; never inferred from the requested ref |
| `lora_bucket` | W loaded object/contract | O cell compatibility | exact compiled LoRA bucket |
| `contract_digest` | W graph/lane contract | O adoption + dispatch fence | digest of graph, shapes, targets, guidance regimes, low-VRAM mode, weight lane, activation scaling, and LoRA bucket |
| `active_compile_ref` | W proven boot/adoption path | O dispatch eligibility | exact cell ref, or empty when eager/revoked; ref and digest are both present or both empty |
| `active_compile_snapshot_digest` | W proven boot/adoption path | O immutable cell identity | exact cell snapshot digest paired with `active_compile_ref` |
| `function_names` | W per-handler warmup proof | O applicability | sorted aliases proved on this exact object. The set is immutable for the incarnation; a skipped/unproved sibling is absent even if it shares endpoint config |
| `model_bindings` | W setup ownership | O applicability + RunJob fence | sorted exact slot/ref/snapshot triples owned by this target. It may be a strict subset of RunJob setup bindings (for example SDXL pipeline owned, ancillary VAE omitted) |

Inductor proof separates two causal facts: the exact object wrapper must observe
at least one isolated cache activation hit, and every advertised handler alias
must independently execute that same wrapper successfully. Inductor does not
increment its lookup counter on every execution of an already-loaded graph.
Setup/adoption warmups exclude concurrent GPU work while capturing activation,
and wrapper-local counters prevent a sibling object from supplying it.
Mandatory W8A8 publishes no partial target: every non-skipped compatible alias
must prove the exact active object/cell or setup fails and those handlers are
reported unavailable. Explicitly skipped handlers such as legacy SDXL Turbo do
not inherit another alias's proof.

### RunJob (O → W)

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | O | W executor | unique request id |
| `attempt` | O fencing (§2) | W (echoed on all replies) | dispatch fence |
| `function_name` | O from client submit | W registry lookup | dispatch key; unknown fn ⇒ immediate `JobResult{INVALID}` |
| `input_payload` | O assignment payload derived from `/invoke` | W deserializes, validates against the fn payload type | MessagePack bytes; O keeps canonical stored refs in durable request identity but replaces typed Asset refs only in this ephemeral payload with fresh authorized HTTP(S) URLs. W rejects opaque refs and caller `local_path`, downloads distinct URLs once in payload order, assigns that single fresh attempt-local path to every duplicate occurrence, and cleans it on every terminal path before model/handler work can observe a failure |
| `timeout_ms` | O from endpoint config | W deadline watchdog | 0 = none; on expiry W aborts and sends `JobResult{FATAL, safe_message:"deadline exceeded"}` |
| `tenant` | O request state | W upload scoping + structured logs | invoking owner slug |
| `invoker_id` | O request state | W `ctx.invoker_id` (read-only tenant surface) | optional |
| `capability_token` | O token mint (cached per worker+tenant) | W tensorhub HTTP auth (uploads, blob PUT) | per-job scoped credential |
| `output_mode` | O from `Prefer: bytes=inline\|url` header | W save/output path | INLINE = return small media raw in payload; URL = upload + return refs. UNSPECIFIED = URL |
| `compute` | O scheduler | W GPU-semaphore gating + CUDA binding | see below |
| `models` | O binding resolver (endpoint defaults + `_models` envelope) | W model injection (`ensure_local` + typed path/pipeline) + per-request LoRA overlays | slot → ref (+ optional `loras`) |
| `snapshots` | O resolver | W download stack | presigned snapshots for tensorhub-CAS refs in `models` (including LoRA overlay refs) that O doesn't know to be on this worker's disk; W ignores entries already local (digest match). hf/civitai refs need no snapshot |
| `required_compile` | O unique target selection | W pre-setup + pre-execution fence | exact target/cell/contract required for this attempt; mandatory for W8A8 and fail-closed whenever present |

### RequiredCompileExecution (embedded in RunJob)

O selects exactly one active `CompileTarget` whose `function_names` contains
the RunJob function and whose target-owned binding subset matches the RunJob
refs plus immutable snapshot digests. Zero applicable active targets makes a
W8A8 request wait/retry; multiple applicable targets are ambiguous and fail
closed rather than choosing by map/order. Ancillary RunJob bindings that the
target does not own do not broaden or certify the compiled graph.

| field | producer | consumer | semantics |
|---|---|---|---|
| `target_incarnation_id` | O from current StateDelta | W exact live-object lookup | must still name the READY target selected for this worker session |
| `cell_ref` | O from target active identity | W equality fence | exact immutable compile cell ref |
| `cell_snapshot_digest` | O from target active identity | W equality fence | exact digest paired with `cell_ref` |
| `contract_digest` | O from target | W graph/lane equality fence | must equal the live target contract digest |

W validates this structure before setup/mutation and again after acquiring the
GPU execution permit. Target replacement, binding drift, cell revocation, or a
runtime guard failure therefore returns RETRYABLE without running tenant GPU
work, including a request that queued while the old evidence was still valid.
Hot adoption MUST prove the already-advertised immutable function set; it may
not mutate aliases in place between ADOPTED, StateDelta, and a later causal
runtime-guard failure.

### ResolvedCompute (embedded in RunJob)

| field | producer | consumer | semantics |
|---|---|---|---|
| `accelerator` | O endpoint resources | W GPU-semaphore gating ("cuda" acquires; "none" bypasses) | "cuda" \| "none" |
| `gpu_index` | O per-GPU slot scheduler | W `CUDA_VISIBLE_DEVICES` / `set_device` before handler | 0-based; 0 for single-GPU |

`gpu_count`/`vram_gb` (fields 3/4) were CUT in pgw#526 along with the `ctx.compute`
tenant surface they fed: documented and plumbed through every dispatch, read by zero
endpoints. Field numbers + names are `reserved`. The worker's own VRAM decisions read
the endpoint's declared `Resources` + probed free VRAM, never the wire.

### ModelBinding (embedded in RunJob)

| field | producer | consumer | semantics |
|---|---|---|---|
| `slot` | O from endpoint manifest | W injection: maps to the endpoint's declared model parameter | slot name |
| `ref` | O resolver | W `ensure_local` + injection | canonical ref string |
| `loras` | O `_models` override gate (AllowLora) + BYOM ingest (th#585) | W per-request adapter overlay (gw#393) + adapter residency (gw#399) | LoRA overlays riding this slot's base model; attached as unfused named adapters that stay resident on the pipeline, ACTIVE only for jobs that name them (explicit activation — adapter-free jobs always run with adapters disabled). Purely W-side: O never routes on adapter residency. Empty for adapter-free jobs; workers predating the field ignore it |
| `inference_defaults` | O repo-metadata store (th#767c: PUT-time validated against the family's exported JSON Schema) | W `ctx.slots[slot].defaults` resolution chain (pgw#520) | JSON object, an instance of the slot's family vocabulary struct (`gen_worker.families.FamilyDefaults` subclass). Empty when the resolved repo carries no metadata — W merges by falling back to the endpoint's code-declared `Slot(fallback=...)` preset; empty AND no fallback ⇒ `ctx.slots[slot]` raises when the handler reads it. Additive: workers predating the field ignore it (Slot resolution falls back to the code preset unconditionally) |

### LoraOverlay (embedded in ModelBinding.loras)

Re-added deliberately in gw#393: the recut deleted the old
`ResolvedModelBinding.loras` stack as unreachable (§10); the BYOM pipeline
(th#585/#586) is now its producer. Overlay refs are ordinary tensorhub-CAS
refs — external (hf/civitai) LoRAs are mirrored into CAS at ingest and O
attaches presigned snapshots in `RunJob.snapshots` keyed by the overlay ref,
exactly like base-model refs. W never fetches user LoRA refs upstream.

| field | producer | consumer | semantics |
|---|---|---|---|
| `ref` | O resolver | W `ensure_local` + digest-keyed adapter state-dict cache | canonical tensorhub ref of the adapter snapshot |
| `weight` | O from `_models[slot].loras[].weight` | W `set_adapters` per-adapter scale | hub validates to [-4, 4]; W mirror-checks and fails INVALID out of range |
| `inference_defaults` | O repo metadata (th#767b / pgw#516, `PUT .../metadata/inference-defaults` on a LORA-kind repo, validated against the family's LORA schema at PUT time) | W `ctx.slots[slot].defaults` resolution chain (`gen_worker.api.slot.resolve_slot`) | JSON-encoded family-typed LoRA recipe-opinion object (an instance of the family's `kind="lora"` vocabulary struct, e.g. `SdxlLoraDefaults`); `""` = lora repo has none set. Additive — workers predating the field ignore it |

**Composition rule (pgw#516 settled foundation) — FIELD-LEVEL, not whole-object.**
`ModelBinding.inference_defaults` (the CHECKPOINT's metadata) uses
whole-object precedence over the endpoint's code fallback (§ above: repo
metadata replaces the fallback entirely). `LoraOverlay.inference_defaults`
is different: after the checkpoint's resolved recipe is computed, W applies
each lora's inference_defaults ONE FIELD AT A TIME — only the fields it
shares with the checkpoint's family struct (e.g. SDXL's `scheduler`,
`steps`, `guidance`, `max_guidance`), and only when that field is non-null
("no opinion" fields never touch the result). Loras apply IN ORDER
(`ModelBinding.loras[0]` first); a later lora's non-null field wins over an
earlier one's. Fields with no checkpoint-recipe analog
(`trigger_words`/`recommended_weight`) are NOT merged into
`ctx.slots[slot].defaults` — an endpoint reads a lora's own resolved object
for those (out of this contract's settled scope). Worked example: a
checkpoint resolves `steps=28, guidance=6.0`; a distillation lora rides the
pick with `{"steps":4,"guidance":0}` — the merged
`ctx.slots["pipeline"].defaults` reads `steps=4, guidance=0`, scheduler/
max_guidance untouched (the lora left them null).

### Snapshot / SnapshotFile (embedded in RunJob, DesiredResidency, and ModelOp)

| field | producer | consumer | semantics |
|---|---|---|---|
| `digest` | O resolver | W CAS layout + dedupe | pinned snapshot digest |
| `files[].path` | O | W file placement | repo-relative path |
| `files[].size_bytes` | O | W disk-headroom check + progress totals | file size |
| `files[].blake3` | O | W post-download verification (digest-poisoning guard) | content hash |
| `files[].url` | O presigner | W downloader | presigned GET; expiry ⇒ `ModelEvent{FAILED, error:"url_expired"}` and O re-sends the same desired generation or job with fresh URLs |

Standalone CLIENTS (cozy CLI, `gen-worker prefetch`) pull the SAME shape over
HTTP instead: `GET /api/v1/repos/:tenant/:name/resolve?tag=&flavor=` (#560 —
anonymous for public repos, rate limited per client). Workers under an
orchestrator never call it; their snapshots stay gRPC-pushed.

### JobAccepted (W → O)
Sent once per accepted `(request_id, attempt)`, immediately on admit (before
model materialization / queueing). Re-sent on RunJob retransmission.

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | W | O dispatch retry loop (stop retransmitting) | |
| `attempt` | W echo | O fencing | dropped if stale |

If W cannot accept (draining, unknown function, over limits) it skips
JobAccepted and sends `JobResult` directly with the appropriate status.

### JobResult (W → O)
Exactly one per accepted attempt. Terminal — also ends any JobProgress stream
for the attempt. Never dropped (§1 send-queue policy).

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | W | O request completion | |
| `attempt` | W echo | O fencing | stale ⇒ dropped |
| `status` | W executor | O retry policy + client status mapping (§9) | |
| `output.inline` | W serializer | O → client response | MessagePack output struct, only when ≤ ~64KB |
| `output.blob_ref` | W upload stack (§8) | O → client response (serves/redirects) | opaque tensorhub file ref for payloads > ~64KB |
| `safe_message` | W (sanitized: no secrets/paths/signed URLs) | O → client error body | empty on OK |
| `metrics` | W executor | O resource-profile learner | see below |

### JobMetrics (embedded in JobResult)
The ONE metrics vehicle. Worker-observed values only; O enriches with its own
context (release, function, machine class, provider) from its request state —
those never travel on the wire.

| field | producer | consumer | semantics |
|---|---|---|---|
| `runtime_ms` | W | O runtime EWMA (placement/autoscale ETA) | handler start → completion |
| `queue_ms` | W | O local-queue EWMA | worker-local wait before start |
| `rss_at_end_bytes` | W (psutil sample at job completion) | O memory profile samples | instantaneous RSS, NOT a per-job peak (the OS gives no per-process peak-RSS reset — pgw#513); 0 = unmeasured |
| `peak_vram_bytes` | W (CUDA peak stats, reset at handler start) | O VRAM profile samples | true per-job peak: GPU jobs serialize under W's GPU semaphore, so `reset_peak_memory_stats()` at handler start isolates this job's peak (pgw#513); 0 = unmeasured |
| `concurrency_at_start` | W executor | O observed-parallelism profile | active jobs at admit |
| `output_media_duration_s` | W executor (sums probed `duration_s` of output media assets) | O `per_output_second` settlement (th#572) | MEDIA seconds, never wall-clock; 0 = unreported ⇒ media-unit settlement fails closed |
| `input_tokens` | W (folds streaming `TokenUsage`) | O `per_million_tokens` settlement (pgw#512) | 0 = unreported |
| `input_cached_tokens` | W (folds streaming `TokenUsage`) | O `per_million_tokens` settlement (pgw#512) | 0 = unreported |
| `output_tokens` | W (folds streaming `TokenUsage`) | O `per_million_tokens` settlement (pgw#512) | 0 = unreported |
| `output_count` | W executor (counts output `Asset`s) | O `per_output` settlement (pgw#512) | replaces result-payload scavenging; 0 = unreported |
| `slot_held_ms` | W executor (gw#516) | O GPU-occupancy profile / overlap evidence | GPU-slot acquire → terminal release; < `runtime_ms` when the handler released at the decode→finalize handoff |
| `finalize_wall_ms` | W executor (gw#516) | O overlap evidence | slotless encode/upload tail after the release (overlaps the next request's compute). Both 0 = unmeasured (CPU job / pre-gw#516 worker) |

### JobProgress (W → O)
Streaming output chunks (LLM token deltas, AR-TTS audio, structured partials).

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | W | O per-request pub/sub fan-out to SSE/wait subscribers | |
| `attempt` | W echo | O fencing | stale ⇒ dropped, not fanned out |
| `seq` | W counter | O ordering + client gap detection | starts at 1, strictly increasing per (request_id, attempt) |
| `data` | W streaming handler | SSE edge (routed by content_type: text → JSON stream, audio → binary stream) | chunk bytes |
| `content_type` | W handler | SSE edge routing + client decode | e.g. `text/plain`, `application/json`, `audio/pcm;rate=24000`, `audio/opus;rate=48000` |

**Semantics:** never persisted — O fans out in memory on the owning replica
and drops chunks with no subscriber; only terminal state hits postgres. Within
a connection: ordered, complete. Across reconnect: W MAY drop buffered
progress (§1), so subscribers may observe seq gaps; `JobResult` is the
authoritative output. There is no stream-done/stream-error message: the
terminal `JobResult` (any status) ends the stream.

**The ctx event lane (pgw#508).** `RequestContext` methods (`ctx.progress`,
`ctx.log`, checkpoint/warning/training-metric helpers) don't get their own
wire message — each rides a `JobProgress` chunk whose `content_type` is the
constant `application/x-request-event+json` (`EVENT_CONTENT_TYPE` in
executor.py / `RequestEventContentType` in tensorhub's runtimestore) and
whose `data` is a JSON envelope `{request_id, type, payload, timestamp}`.
This is a CONTENT-TYPE DISCRIMINATOR inside one wire message, not a second
message type — O inspects `content_type` before falling back to the generic
streaming-output path. `type` is one of:

| `type` | audience | O persistence | tolerates shedding |
|---|---|---|---|
| `request.progress` | USER-facing (cozy-art job card) | latest-tick only, in-memory (th#737); never a durable row | yes — a dropped mid-run tick is superseded by the next one; only the final state matters |
| `request.log` | PLATFORM/OPERATOR ONLY, full stop — never user-facing | durable `job.log` row (request_events) | yes — a dropped debug line is an acceptable loss; nothing downstream depends on completeness |
| `request.checkpoint` | USER-facing | durable `job.checkpoint` row | no — every save is individually meaningful (a training run's checkpoint list must be complete) |
| `request.warning` | USER-facing | durable `job.warning` row | no — sparse and actionable (e.g. `artifact_upload_failed`) |
| `request.training_metric` | USER-facing (training job UI) | durable `job.training.metric` row, downsampled to ≥5s apart (final step and any `val_loss` point always persist) | yes between persisted points — the chart only needs the trend, not every tick |

O enforces the `request.log` exception at TWO points, because the live
`JobProgress` chunk reaches O before the durable row exists: (1) the live
hub fan-out (`ProgressHub`/SSE `output.delta` path) drops any chunk whose
decoded `type` is `request.log` outright — it is never wrapped as a generic
delta and shown to a tenant subscriber; (2) the durable read path
(`/v1/requests/:id/events`, `/events.bin`) filters `job.log` rows out of
what it streams to the tenant-scoped SSE routes. There is no operator-facing
read surface for `job.log` yet — it exists to be queried directly (psql /
future admin tooling), not to be served over the tenant API.

Every other `type` keeps the base `JobProgress` shedding contract (§1
send-queue policy: `JobProgress` is the FIRST thing dropped under queue
overflow) — durability past that point is what the table above's
"persistence" column adds, not a wire-level delivery guarantee.

### CancelJob (O → W)
Sent on client cancel, reconcile cleanup (§1), or O-side abort.

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | O | W executor | |
| `attempt` | O current attempt | W (ignores if it doesn't hold this pair) | |

W aborts the handler (cooperative cancel + hard deadline) and replies
`JobResult{CANCELED}`. If the job already finished, the natural result stands
and the cancel is a no-op. Cancel of an unknown pair is silently ignored.
There is no queued-only/item-level cancel.

### ModelOp (O → W)
Producer: compile-cache adoption only. Model download/load/unload lifecycle is
declarative through `HelloAck.desired_residency`; v3 O MUST NOT send the legacy
DOWNLOAD/LOAD/UNLOAD enum values.

| field | producer | consumer | semantics |
|---|---|---|---|
| `op` | O | W model layer | ADOPT_COMPILE_CACHE only |
| `ref` | O | W | canonical ref |
| `snapshot` | O resolver | W downloader | required for ADOPT_COMPILE_CACHE |
| `operation_id` | O adoption controller | W adoption handler | required causal identity for this attempt; retained after ADOPTED for at most one matching runtime-guard failure |
| `target_incarnation_id` | O adoption controller | W adoption handler | exact live object to mutate; family/ref fallback is forbidden |

Every ModelOp is answered by ≥1 `ModelEvent` for the ref (success path emits
ADOPTED; failure path emits FAILED). Every terminal result echoes the exact
`ModelOp.snapshot.digest` and `ModelOp.operation_id` in the matching ModelEvent.
A missing/empty operation ID or digest fails closed as
`adopt_failed:missing_operation_id` or `adopt_failed:missing_snapshot_digest`
before any download, cache seeding, pipeline wrapping, or resident-state mutation.

**ADOPT_COMPILE_CACHE** (hot adoption, #567): `ref` is a compile-cache flavor
ref — `_system/family-<f>#inductor-<sku>-torch<maj.min>`. W downloads the
artifact snapshot, verifies its key (family/SKU/torch/triton/libs/producer
gen-worker version + low-VRAM prep mode, gw#391 — the prep flags are traced
into the graphs) against its own runtime and resident pipelines, seeds the
inductor+triton cache dirs, re-wraps the
already-VRAM-resident modules of endpoints declaring
`compile=Compile(family=<f>)` (module re-wrap only — weights are untouched,
no reload), runs one warmup trace, and answers
`ModelEvent{ADOPTED, duration_ms, cache_hits, cache_misses, warmup_s}`. The
warmup is the proof (gw#391): each inductor object reports ADOPTED only when
that exact object's guarded warm call executes and observes ≥1 FX-graph cache
hit — a process-global hit from a sibling cannot certify it. Zero hits rolls the wrap back to
eager and answers `adopt_failed:cache_miss`; an endpoint without `warmup()`
answers `adopt_failed:no_warmup` (unprovable). ANY failure ⇒ discard, stay
eager, answer `ModelEvent{FAILED, error:"adopt_failed:<reason>"}` — adoption
must never degrade service, and ADOPTED must never mean silently-eager. O policy: send only to job-idle workers (no in-flight jobs),
at most one worker per release at a time, and only when the artifact's
version key matches the pod (SKU + torch); W enforces the same checks
defensively. Workers predating this kind ignore it silently (unknown enum, no
ModelEvent); O treats the absence of a reply as not-adopted, never an error.
W serializes the entire adoption operation worker-wide, including download,
seed, re-wrap, warmup, rollback, and terminal evidence; two commands cannot
concurrently mutate resident modules. O correlates adoption evidence by active
worker session plus `(ref, snapshot_digest, operation_id)`, never by mutable
`ref` alone. Thus a late terminal result from operation A cannot certify or
fail operation B, including when both commands target the same digest.

After ADOPTED, the first compiled-runtime guard failure synchronously clears
the target's active ref/digest before an optional eager fallback and emits one
causal `FAILED{error:"adopt_failed:runtime_guard"}` retaining that adoption's
ref, digest, operation ID, and target incarnation. The same target row remains
visible with empty active fields so event/StateDelta arrival order is harmless.
Mandatory W8A8 fails closed, and the worker's repeated post-GPU-acquisition
fence prevents a sibling that validated while queued from executing. The exact
failed cell identity is quarantined on that incarnation: a repeated command
fails immediately without wrap/warmup until the cell or target changes. This is
state-driven; a desired-state generation, operation ID, or controller ordering
rewrite alone does not re-arm the same target and immutable cell. Re-arming
requires a new target/binding/contract incarnation, a different cell ref or
digest, or a new worker session. Quarantines accumulate for the incarnation;
successfully adopting cell B never clears an earlier failed identity A. There
is no retry timer or timeout knob. A
boot-attached cell has no ModelOp operation ID and never fabricates a causal
failure event. Declarative reconciliation may re-advertise the failed target's
aliases only after a fresh target has an exact active cell and proves those
same aliases; it clears only the old target/record-owned `compile_cell_failed`
marks, never hardware, setup, or another target's disable.

### ModelEvent (W → O)
The single model-residency channel. Emitted for desired-state outcomes,
compile-cache adoption, AND worker-initiated transitions (LRU eviction,
demotion, on-demand load during a job). Replaces load/unload results,
model-ready signals, and the JSON download-event fabric.

| field | producer | consumer | semantics |
|---|---|---|---|
| `ref` | W | O residency index | |
| `state` | W model cache | O: DOWNLOADING/ON_DISK feed pending-download affinity (parked jobs dispatch on ON_DISK); IN_RAM/IN_VRAM/ON_DISK update the placement tier index; EVICTED removes the ref; FAILED fails the pending op; HOST_CAPACITY_PROGRESS clears only the matching older host-RAM capacity block | current state machine position or a typed non-residency outcome |
| `vram_bytes` | W measured (allocator delta across load) | O model-size cache → VRAM packing | set with IN_VRAM |
| `error` | W | O reconcile handling (e.g. OOM ⇒ revise desired hot set; url_expired ⇒ re-mint snapshots and re-send the same generation) + triage log | set with FAILED |
| `bytes_done`/`bytes_total` | W downloader (emit ≤1 per 5s per ref) | O boot/capacity progress display | set with DOWNLOADING |
| `duration_ms` | W adopt handler | O adoption bookkeeping + fleet-adoption-latency metric | set with ADOPTED: wall time of download+seed+re-wrap+warmup |
| `cache_hits`/`cache_misses` | W exact target guard (inductor FX-graph counter delta around that object's warm call) | O adoption observability: hits ≥ 1 on every inductor ADOPTED; misses > 0 = partial shape coverage | set with ADOPTED (gw#391); zero for TRT |
| `warmup_s` | W adopt handler | O adoption bookkeeping | set with ADOPTED: warmup wall seconds |
| `host_ram_required_bytes` | W host-RAM admission | O capacity block / durable request evidence | exact incoming staging requirement including the derived safety floor; set with `FAILED{error:"insufficient_host_ram"}` and HOST_CAPACITY_PROGRESS |
| `host_ram_available_before_bytes` / `host_ram_available_after_bytes` | W cgroup-aware host-RAM probe | O capacity block / durable request evidence | exact measured headroom around failed cleanup, or from the last insufficient observation to the satisfying observation |
| `host_ram_evicted_refs` | W owner-aware record teardown | O capacity audit | FAILED: canonical refs released during that failed admission attempt; PROGRESS: refs released in the exact measured before→after transition that satisfied the requirement. Never inferred from logs, accumulated from unrelated observations, or an opaque worker-local `shared::*` cache key |
| `host_ram_capacity_generation` | W executor | O capacity fencing | process-monotonic observation generation, fenced within the active Connect stream; the consumer resets its numeric fence on each Hello re-baseline, while a same-process reconnect replays only undelivered satisfying progress; progress clears only the same ref's older failed generation |
| `snapshot_digest` | W exact materialization operation | O immutable residency identity | digest operated on by this event; empty is unknown and O MUST NOT fill it from the ref's current tag target |
| `residency_generation` | W desired-state receiver | O per-ref event fence | desired generation captured when the operation began; lower generations and same-generation digest conflicts are stale and do not mutate residency |
| `operation_id` | W adoption handler/guard | O adoption controller | exact ADOPT_COMPILE_CACHE attempt on ADOPTED/adopt-FAILED and at most one later causal runtime FAILED; empty for boot-attached cells and ordinary residency |

**State machine per (worker, ref):** `DOWNLOADING → ON_DISK → IN_RAM ⇄ IN_VRAM`,
demotions emit the new lower tier, `EVICTED` = removed from disk (fully gone).
`FAILED` is not a residency: it reports "the last op failed" via `error`;
residency stays whatever the last non-FAILED event said. HOST_CAPACITY_PROGRESS
is also not residency: it is emitted only when a remembered host-RAM failure's
measured available headroom increases to at least its typed requirement after
an owner or execution pin is released. Active failures replay before pending
results after a stream reconnect; after satisfaction, only the self-contained
progress event remains until successful stream delivery. Elapsed time and unrelated
insufficient demotions do not produce it. O's baseline is `Hello.models`;
events mutate it; reconnect
re-baselines. A positive `residency_generation` plus non-empty
`snapshot_digest` is exact evidence; zero/empty remains legacy-unknown. O never
manufactures observed identity from desired config or a mutable tag target.
`ADOPTED` is also not a residency tier: it reports one-shot success of ADOPT_COMPILE_CACHE for
a compile-cache ref (whose bytes independently report DOWNLOADING/ON_DISK
like any snapshot download). O MUST NOT feed `_system/family-*` compile-cache
refs into ordinary model-failure availability handling. An optional lane may
stay eager; mandatory W8A8 remains unavailable until exact compiled evidence
changes and must never be marked function-ready from the failed cell.

### FnUnavailable (W → O)
Emitted when a hardware-axis gate fails at startup or model-load time, or
when a function's model download / pipeline setup fails terminally
(`reason: setup_failed` — the function is dropped from BOTH
`available_functions` and `loading_functions` instead of sitting in
loading forever under a READY phase). One message per function,
deduplicated. A later desired-state reconciliation or `RunJob` retries setup;
on success the function reappears in `StateDelta.available_functions`.

| field | producer | consumer | semantics |
|---|---|---|---|
| `function_name` | W gate | O availability snapshot: narrows dispatch for that fn away from this worker; requests with no eligible worker fail 424 | |
| `reason` | W gate | O error classification + admin availability route | closed vocabulary (see proto) |
| `detail` | W gate | O admin availability route / logs | human-readable |
| `axes` | W gate | O admin availability route | structured gate data, e.g. `{detected_sm:"8.9", required_sm:"10.0"}` |

Cleared implicitly by the next `Hello` (reconnect wipes all per-worker fn
disables before processing the new snapshot). There is no "available again"
message — a worker that recovers reconnects or simply includes the fn in
`StateDelta.available_functions` (which also clears the disable).

### Drain (O → W)
Producers: autoscaler retirement, deploy rollover, operator action.

| field | producer | consumer | semantics |
|---|---|---|---|
| `deadline_ms` | O policy | W drain loop | `0` waits indefinitely; `>0` is the total grace budget from receipt |

W behavior: stop admitting (`RunJob` after Drain ⇒ `JobResult{RETRYABLE,
safe_message:"worker draining"}` without JobAccepted), finish in-flight jobs,
ship all buffered results, close the stream, exit 0. At deadline expiry W
aborts remaining jobs (`JobResult{RETRYABLE}`), flushes, exits. O behavior:
stop dispatching to the worker the moment Drain is sent; the stream close is
the completion signal (no drain-result message). Jobs that come back
RETRYABLE are requeued with attempt+1. A draining worker's JWT is never
rotated; revocation happens at pod terminate (a draining worker still needs
its token to ship results/blob uploads).

### TokenRefresh (O → W)
Producer: O's rotation sweep (§1 worker JWT rotation), at ~80% of the current
token's TTL. No reply message.

| field | producer | consumer | semantics |
|---|---|---|---|
| `token` | O token mint (same identity, fresh jti/iat/exp) | W stored credential | replaces the worker JWT used for every subsequent reconnect; live connection unaffected |
| `expires_at_unix` | O | W logs/observability | `exp` claim of the replacement |

W MUST apply it in-place (no reconnect, no re-Hello). Additive proto3 field:
older workers ignore it (unknown oneof member is skipped by the proto3
runtime — `WhichOneof` sees nothing) and keep their boot-time token.

---

## 5. Cancellation

Client cancel → O `CancelJob{request_id, current attempt}` → W cooperative
abort (one cancellation signal on ctx; sync handlers polled, async handlers
cancelled) → `JobResult{CANCELED}` → O marks terminal, releases capacity.
If W never answers (hung handler), O's dispatch state is cleaned by the
worker's stream death or by the request `timeout_ms` at the worker; O also
enforces its own request-level timeout independent of the worker.

---

## 6. Ordering guarantees

The stream is HTTP/2: reliable, ordered per direction. Consequences relied on:
- `Hello` precedes everything; `HelloAck` precedes all O→W traffic.
- `JobAccepted` precedes that attempt's `JobProgress`, which precede its `JobResult`.
- A `ModelEvent` sequence per ref is ordered; O never sees IN_VRAM before ON_DISK
  within one connection. Across reconnects or concurrent work, the per-ref
  `residency_generation`/`snapshot_digest` fence rejects late transitions for a
  prior desired snapshot.
- `StateDelta` full-replace makes reordering across reconnects harmless.

---

## 7. Declarative model residency

`HelloAck.desired_residency` is the full per-worker goal. W accepts each
non-stale generation without blocking the receive loop, replaces its disk
keep set from `disk_refs`, cancels obsolete background reconciliation, and
works the ordered disk refs then ordered hot instances. Reconciliation starts
only while the executor is job-idle; an active `RunJob` and its exact bindings
always take precedence over pending background work. Hot instances reuse the
same binding derivation and setup path as `RunJob`; there is no second loader.

`observed_residency_generation` means only that W accepted the desired state.
Actual progress and failures are `ModelEvent`s, with `Hello.models` as the
reconnect baseline. W MAY evict desired refs under real disk/VRAM pressure and
reports that honestly; O then revises or re-sends desired state. Same-generation
re-sends refresh expired snapshots and retry failed work.

W captures the desired generation and exact snapshot digest at the start of a
materialization, reports that identity on every resulting residency event, and
retains it in `Hello.models` across reconnects. A later tag resolution cannot
relabel bytes already in RAM/VRAM. If old loaded bytes coexist with a newer
disk snapshot, W continues reporting the old highest-tier identity, vacates the
stale instance, and only then promotes the newer disk identity.

**Compile-cache snapshots (#569).** When a release's endpoint declares
`compile=Compile(family=...)` and boot-attach is enabled (opt-in; default OFF
— boot-time attach worsens TTFI, hot adoption is the primary path), O MAY add
the resolved `_system/family-<f>#inductor-<sku>-torch<maj.min>` snapshot to
`RunJob.snapshots` keyed by that ref, alongside the model snapshots. W
recognizes the key by the `_system/family-<f>#inductor-` prefix for the
declared family, downloads it like any snapshot, and seeds it before pipeline
load; verification failure ⇒ eager, never an error. The same ref may arrive
via `ModelOp{ADOPT_COMPILE_CACHE}` (hot adoption, §4).

---

## 8. Blob-ref output flow (results > ~64KB)

1. W serializes the output struct; if > 65536 bytes, W requests a presigned
   PUT from tensorhub HTTP (`file_base_url`, existing presigned-upload API,
   auth: `RunJob.capability_token`).
2. W PUTs the payload (existing upload/transfer stack, retries included).
3. W sends `JobResult{blob_ref: <returned file ref>}`.
4. O stores the ref as the request's result; the HTTP edge serves it to the
   client via presigned GET/redirect.

**Capability token renewal (#561).** `RunJob.capability_token` is TTL'd per
workload (inference 1h / training 1h / conversion 24h — th#639: trainers renew hourly instead of holding a week-long credential). A job that outlives
its token renews over HTTP: `POST {file_base_url}/v1/worker/capability/renew`
with `authorization: Bearer <worker JWT>` and body `{request_id, attempt,
capability_token}`. O re-mints the SAME grants with a fresh expiry iff the
job is still running on that worker and the attempt matches (fencing); the
old token is accepted up to 5min past its exp (job state is the real authz).

Inline (`output.inline`) is used at or below the threshold — no S3 round trip
for small results. Media assets produced DURING execution (`ctx.save_image`
etc.) already upload through the same stack and appear as refs inside the
payload; the 64KB rule governs the result envelope itself. Upload failure
after retries ⇒ `JobResult{RETRYABLE, safe_message:"output upload failed"}`.

---

## 9. Error / status taxonomy

`JobStatus` is the complete job-outcome vocabulary:

| status | meaning | O action | HTTP mapping |
|---|---|---|---|
| `OK` | success | persist result | 200 |
| `INVALID` | input failed validation / unknown function | terminal, never retry | 400 |
| `RETRYABLE` | transient (draining, OOM-retry exhausted, upload failure, engine hiccup) | requeue with attempt+1 up to retry cap, then surface as 503 | 503 after cap |
| `FATAL` | permanent execution failure (handler exception, deadline) | terminal | 500 |
| `CANCELED` | cancel ack / deadline abort after cancel | terminal | 499 |

`safe_message` is the only client-visible error text and MUST be sanitized by
W. Internal diagnostics (tracebacks, worker logs) go to the logging pipeline,
never this protocol.

Stream-level (gRPC status) errors:

| code | producer | meaning |
|---|---|---|
| `UNAUTHENTICATED` | O | bad/missing/revoked JWT — an actual auth VERDICT, never a transient condition. W may treat repeated occurrences as fatal |
| `UNAVAILABLE` | O | transient hub-side condition (shared-state store blip, unowned partition window, owner registered without a peer addr) — W retries with backoff |
| `ABORTED` `worker_stream_superseded` | O | a newer stream from the same worker took over (§1) — the OLD stream sees this; not an error for the worker process |
| `DEADLINE_EXCEEDED` `hello_timeout` | O | no Hello within the Hello deadline (§1) |
| `FAILED_PRECONDITION` `not_leader:<addr>` | O | redirect (§1) |
| `FAILED_PRECONDITION` `protocol_version_mismatch` | O | version gate |
| `FAILED_PRECONDITION` (other) | O | protocol-order violation (e.g. first message not Hello) |

Model-op errors travel as `ModelEvent{FAILED, error}` with classified tokens
such as `oom`, `model_in_use`, `url_expired`, `digest_mismatch`,
`insufficient_disk`, `download_failed`, and `load_failed`. The terminal answer
to an initial ADOPT_COMPILE_CACHE attempt uses `adopt_failed:<initial-reason>`;
the suffix is produced by the validation, staging, activation, and warmup rails
(for example `bad_ref`, `missing_operation_id`, `missing_snapshot_digest`,
`target_not_ready`, `target_replaced`, `artifact_invalid`, `key_mismatch`,
`warmup`, `no_warmup`, `cache_miss`, or `guard_unbound`). Consumers MUST
preserve unknown classified suffixes rather than treating this example list as
an enum.

`adopt_failed:runtime_guard` is deliberately not an initial-attempt reason. It
is the later, at-most-once causal failure emitted only after ADOPTED when that
active cell's runtime guard fails; it retains the original operation, cell,
snapshot, and target identities as specified in §4.

---

## 10. Deleted from the old protocol (and why)

- Mega-oneof shared envelope, `WorkerResults`/`WorkerEvents` aux streams +
  handshakes, `supports_split_streams`/`stream_role` — one stream; blob-ref
  outputs remove the head-of-line pressure that motivated the split.
- `WorkerRegistration`-as-heartbeat, 10s resource snapshots, padded keepalive
  envelope, app watchdog, redis stream lease — keepalive is the one liveness
  mechanism (§3).
- `ActiveAssignmentResume`, `duplicate_request_id`, assignment epochs,
  per-assignment seq fencing — single `attempt` int (§2).
- `WorkerEvent` JSON fabric, `WorkerDiagnosticLog`, `RunMetricsV1`
  triple-emission — typed messages only; diagnostics go to logging;
  `JobResult.metrics` is the one metrics vehicle.
- `LoadModelResult`/`UnloadModelResult`/`WorkerModelReadySignal`/
  `worker.model.download.*` — `ModelEvent`.
- `WorkerStartupPhaseSignal` — `StateDelta.phase`.
- `WorkerFunctionCapabilitiesSignal` — written to a map nothing read;
  per-function capability metadata belongs to deploy-time manifests.
- `WorkerKVPrefixCache` — producer never existed.
- `EndpointConfig` (repo allowlists, per-function model keyspaces, disabled
  functions, payload-ref availability) — O gates dispatch itself; W gets its
  model goal from `HelloAck.desired_residency` and request bindings from
  `RunJob`. Nothing else was consumed by the worker it will keep.
- `InterruptJobCommand.item_ids`/`cancel_queued_only` — item batching is dead;
  cancel is whole-request.
- `JobExecutionObservation` fields: identity/context fields O already knows
  (release, fn, worker, status, image, provider, machine class,
  build_profile), never-produced fields (ttft/itl/prefix-hit/kv-blocks/
  scaling_factors), unread `local_queued_count_at_start`.
- `JobExecutionRequest` fields: `job_id` + `execution_hints` (conversion-only;
  clone/conversion leaves the worker package), `required_flavor_refs`
  (bindings carry refs).
- `ResolvedModelBinding` tag/flavor/provider/source/checkpoint_id/
  compatibility_* — the ref grammar already encodes selection; compatibility
  gating is O-side. Its `loras` overlay stack was deleted here as unreachable
  and re-added deliberately as `ModelBinding.loras` once BYOM (gw#393 /
  th#585) gave it a producer.
- `WorkerResources.gpu_is_busy` — derivable from O's own assignment map.
- `DownloadModelCommand.priority` — never read; ordered desired-state lists
  express controller preference without a second priority system.
- `WorkerDrainResult` — stream close is the drain ack.
- Reserved-tag graveyard — messages were renumbered from 1; only removed v3
  `ModelOpKind` values 1–3 stay reserved to prevent accidental wire reuse.
