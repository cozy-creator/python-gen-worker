# Worker <-> Orchestrator Protocol Contract

Proto: `worker_scheduler.proto`, package `cozy.scheduler`, service
`WorkerScheduler`, one RPC: `Connect(stream WorkerMessage) returns (stream SchedulerMessage)`.
This REPLACES the old `scheduler.v1` proto entirely. No compat, no negotiation:
`ProtocolVersion.PROTOCOL_VERSION_CURRENT = 2` is the only accepted version.

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

Immediately after `HelloAck`, W flushes its buffered unsent `JobResult`s
(§4 results are never dropped).

**HelloAck re-send.** O re-sends `HelloAck` mid-connection whenever release
config changes (e.g. the `keep` set). Full-replace semantics: W overwrites
`file_base_url` and its keep set wholesale.

**Send-queue policy (W).** One bounded outbound queue. `JobResult` is NEVER
dropped: results persist in the queue across reconnects until written to a
live stream. Under overflow, drop order is: `JobProgress` (oldest first) →
never anything else; if the queue is still full, W blocks the producer
(backpressure) rather than dropping `ModelEvent`/`StateDelta`/`FnUnavailable`.

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
| `installed_libs` | W probe | O placement library filters | e.g. "flashpack", "torchao" |
| `image_digest` | W env | O resource-profile keying + triage | worker image sha256 |
| `git_commit` | W env | O triage/display | build provenance |
| `instance_id` | W env | O provider bookkeeping (e.g. runpod pod retire) | cloud instance id; empty for local |

### ModelResidency (embedded in Hello)

| field | producer | consumer | semantics |
|---|---|---|---|
| `ref` | W model cache | O residency index | canonical ref string (one grammar; tag/flavor/digest inside the ref) |
| `tier` | W model cache | O cache-aware placement (VRAM > RAM > DISK > cold) | highest current tier |
| `vram_bytes` | W measured at load | O model-size cache / VRAM packing | set when tier=VRAM |

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
| `keep` | O release prefetch plan | W disk-eviction policy | refs the worker must not evict from disk; NOT a download order (see §7) |

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

### RunJob (O → W)

| field | producer | consumer | semantics |
|---|---|---|---|
| `request_id` | O | W executor | unique request id |
| `attempt` | O fencing (§2) | W (echoed on all replies) | dispatch fence |
| `function_name` | O from client submit | W registry lookup | dispatch key; unknown fn ⇒ immediate `JobResult{INVALID}` |
| `input_payload` | O pass-through from `/invoke` | W deserializes, validates against the fn payload type | MessagePack bytes; input file refs already materialized to presigned URLs by O |
| `timeout_ms` | O from endpoint config | W deadline watchdog | 0 = none; on expiry W aborts and sends `JobResult{FATAL, safe_message:"deadline exceeded"}` |
| `tenant` | O request state | W upload scoping + structured logs | invoking owner slug |
| `invoker_id` | O request state | W `ctx.invoker_id` (read-only tenant surface) | optional |
| `capability_token` | O token mint (cached per worker+tenant) | W tensorhub HTTP auth (uploads, blob PUT) | per-job scoped credential |
| `output_mode` | O from `Prefer: bytes=inline\|url` header | W save/output path | INLINE = return small media raw in payload; URL = upload + return refs. UNSPECIFIED = URL |
| `compute` | O scheduler | W CUDA binding + `ctx.compute` | see below |
| `models` | O binding resolver (endpoint defaults + `_models` envelope) | W model injection (`ensure_local` + typed path/pipeline) | slot → ref |
| `snapshots` | O resolver | W download stack | presigned snapshots for tensorhub-CAS refs in `models` that O doesn't know to be on this worker's disk; W ignores entries already local (digest match). hf/civitai refs need no snapshot |

### ResolvedCompute (embedded in RunJob)

| field | producer | consumer | semantics |
|---|---|---|---|
| `accelerator` | O endpoint resources | W GPU-semaphore gating ("cuda" acquires; "none" bypasses) + ctx | "cuda" \| "none" |
| `gpu_index` | O per-GPU slot scheduler | W `CUDA_VISIBLE_DEVICES` / `set_device` before handler | 0-based; 0 for single-GPU |
| `gpu_count` | O endpoint resources | W `ctx.compute` (tenant adaptation, e.g. parallelism) | GPUs granted |
| `vram_gb` | O endpoint resources | W `ctx.compute` (tenant batch-size adaptation) + low-VRAM decider | VRAM granted |

### ModelBinding (embedded in RunJob)

| field | producer | consumer | semantics |
|---|---|---|---|
| `slot` | O from endpoint manifest | W injection: maps to the endpoint's declared model parameter | slot name |
| `ref` | O resolver | W `ensure_local` + injection | canonical ref string |

### Snapshot / SnapshotFile (embedded in RunJob.snapshots and ModelOp)

| field | producer | consumer | semantics |
|---|---|---|---|
| `digest` | O resolver | W CAS layout + dedupe | pinned snapshot digest |
| `files[].path` | O | W file placement | repo-relative path |
| `files[].size_bytes` | O | W disk-headroom check + progress totals | file size |
| `files[].blake3` | O | W post-download verification (digest-poisoning guard) | content hash |
| `files[].url` | O presigner | W downloader | presigned GET; expiry ⇒ `ModelEvent{FAILED, error:"url_expired"}` and O re-sends the op with fresh URLs |

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
| `peak_rss_bytes` | W (psutil sample) | O memory profile samples | 0 = unmeasured |
| `peak_vram_bytes` | W (CUDA peak stats) | O VRAM profile samples | 0 = unmeasured |
| `concurrency_at_start` | W executor | O observed-parallelism profile | active jobs at admit |

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
Producers: prefetch planner (after Hello, for `keep` refs missing from
`Hello.models`), cold-locality download affinity, VRAM packer (LOAD ahead of
predicted demand, UNLOAD for headroom).

| field | producer | consumer | semantics |
|---|---|---|---|
| `op` | O | W model layer | DOWNLOAD = to disk only; LOAD = promote to VRAM (fetching first if needed); UNLOAD = demote out of VRAM (stays on disk/RAM per cache policy) |
| `ref` | O | W | canonical ref |
| `snapshot` | O resolver | W downloader | required for tensorhub-CAS refs not already local; unset for hf/civitai |

Every ModelOp is answered by ≥1 `ModelEvent` for the ref (success path emits
the new state; failure path emits FAILED). W MAY refuse UNLOAD for a model
serving in-flight jobs: `ModelEvent{FAILED, error:"model_in_use"}`.

### ModelEvent (W → O)
The single model-residency channel. Emitted for ModelOp outcomes AND
worker-initiated transitions (LRU eviction, demotion, on-demand load during a
job). Replaces load/unload results, model-ready signals, and the JSON
download-event fabric.

| field | producer | consumer | semantics |
|---|---|---|---|
| `ref` | W | O residency index | |
| `state` | W model cache | O: DOWNLOADING/ON_DISK feed pending-download affinity (parked jobs dispatch on ON_DISK); IN_RAM/IN_VRAM/ON_DISK update the placement tier index; EVICTED removes the ref; FAILED fails the pending op | current state machine position |
| `vram_bytes` | W measured (allocator delta across load) | O model-size cache → VRAM packing | set with IN_VRAM |
| `error` | W | O op-failure handling (e.g. OOM ⇒ pick LRU vram model, send UNLOAD then retry LOAD; url_expired ⇒ re-mint snapshot and re-send) + triage log | set with FAILED |
| `bytes_done`/`bytes_total` | W downloader (emit ≤1 per 5s per ref) | O boot/capacity progress display | set with DOWNLOADING |

**State machine per (worker, ref):** `DOWNLOADING → ON_DISK → IN_RAM ⇄ IN_VRAM`,
demotions emit the new lower tier, `EVICTED` = removed from disk (fully gone).
`FAILED` is not a residency: it reports "the last op failed" via `error`;
residency stays whatever the last non-FAILED event said. O's baseline is
`Hello.models`; events mutate it; reconnect re-baselines.

### FnUnavailable (W → O)
Emitted when a hardware-axis gate fails at startup or model-load time
(one message per function, deduplicated).

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
| `deadline_ms` | O policy | W drain loop | relative grace budget from receipt |

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
  within one connection.
- `StateDelta` full-replace makes reordering across reconnects harmless.

---

## 7. Model prefetch / keep

`HelloAck.keep` is a retention set, not a download order. After HelloAck, O
compares `keep` against its residency view (`Hello.models` + events) and
issues `ModelOp{DOWNLOAD}` (with snapshots where needed) for missing refs —
the ONE prefetch mechanism. W's eviction policy never removes a `keep` ref
from disk while headroom allows; if disk pressure forces it, W emits
`ModelEvent{EVICTED}` so O knows to re-download later.

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
workload (inference 1h / conversion 24h / training 7d). A job that outlives
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

Model-op errors travel as `ModelEvent{FAILED, error}` with a closed-ish
vocabulary: `oom`, `model_in_use`, `url_expired`, `digest_mismatch`,
`insufficient_disk`, `download_failed`, `load_failed`.

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
  functions, payload-ref availability) — O gates dispatch itself; W learns
  its retention set from `HelloAck.keep` and gets snapshots on
  `ModelOp`/`RunJob`. Nothing else was consumed by the worker it will keep.
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
  compatibility_*/loras — the ref grammar already encodes selection; the LoRA
  overlay stack was unreachable; compatibility gating is O-side.
- `WorkerResources.gpu_is_busy` — derivable from O's own assignment map.
- `DownloadModelCommand.priority` — never read; retention lives in `keep`.
- `WorkerDrainResult` — stream close is the drain ack.
- Reserved-tag graveyard — renumbered from 1, no reservations, one version.
