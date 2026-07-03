<!-- python-gen-worker issue tracker — FUTURE / someday issues -->

> One `# #<id>: <name>` section per issue, separated by `---` lines; section anchor for
> tooling is a line starting with `# #`. IDs are stable for an issue's whole lifecycle and
> share ONE per-repo id space across progress.md / future.md / completed.md.
> CONCURRENT EDITS: only ever edit/append your own issue's section with targeted string
> replacement — never rewrite the whole file.

---

# #329: LLM serving — prefill/decode disaggregation (LATER, depends on #273 measurement)

**Completed:** no
**Status:** planned
**Related:** Parent: #273 (BatchedWorker / JoyCaption first user), Dynamo prefill/decode design doc: https://docs.nvidia.com/dynamo/v-0-7-1/design-docs/disaggregated-serving, NIXL primer: https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/

Follow-up to #273. Open only when measurement shows that prefill stalls decode on the same H100 for JoyCaption-class workloads, OR when serving model sizes / context lengths exceed what fits comfortably on one GPU pool.

**Pattern: NVIDIA Dynamo prefill/decode split applied to our BatchedWorker.** Separate GPU pools, KV transfer between them. Pre-Blackwell baseline transport: TCP. Post-Blackwell: NIXL-equivalent RDMA over InfiniBand. The orchestrator router picks a (prefill_worker, decode_worker) pair per request; prefill worker computes the prompt KV cache, returns transfer descriptors; decode worker pulls the KV blocks (RDMA or TCP) and adds the request to its in-flight batch.

**Requires:**
  - New worker pool kinds in the orchestrator (currently one pool per release; this issue introduces 'prefill' vs 'decode' sub-pools)
  - KV transfer protocol on the wire (new typed proto messages for descriptor handshake + transfer-completion ack)
  - Per-engine: vLLM's `--enable-chunked-prefill` and `--enable-prefix-caching` config; SGLang's analogous controls
  - The router needs to track active KV-block utilization on each decode worker to balance

**Reference:** Dynamo's disaggregated-serving design doc — https://docs.nvidia.com/dynamo/v-0-7-1/design-docs/disaggregated-serving — and NIXL primer https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/.

DO NOT pre-build. Open this when prod numbers show prefill-stall is a real bottleneck.

Moved from gen-orchestrator #329 on 2026-05-21 (implementation belongs in python-gen-worker, not gen-orchestrator). Cross-references like #273/#322 still point to gen-orchestrator issues.

## Tasks
- [ ] Measure prefill vs decode latency on JoyCaption in production. If prefill < 20% of total request time, this issue stays parked.
- [ ] Design the (prefill_worker, decode_worker) wire protocol — proto additions for transfer descriptors, transfer-completion ack, and the orchestrator's pool-kind signaling.
- [ ] Implement TCP transport for KV blocks first; benchmark vs colocation.
- [ ] If TCP shows wins, implement NIXL-RDMA transport for InfiniBand-equipped clusters.
- [ ] Router changes: pick prefill+decode worker pair, track decode-side KV utilization.

---

# #330: SLO-driven planner for BatchedWorker pools (LATER, depends on #273 + #322 observability)

**Completed:** no
**Status:** planned
**Related:** Parent: #273 (observability fields shipped in Phase 2), Dynamo 0.4 SLO blog: https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/

Follow-up to #273. Open when there's enough JobExecutionObservation data (TTFT/ITL/prefix_hit_rate/kv_blocks_used) flowing to drive a real planner.

**Pattern: Dynamo's SLA planner.** Each endpoint declares SLOs (TTFT target, ITL target). Inputs: prefill queue depth, decode KV-block utilization, request rate forecast, GPU capacity. Forecasting: ARIMA or Prophet. Decision: scale prefill / decode pools independently to meet SLOs. Proactive, not reactive — pre-scale before bottleneck forms.

**Requires:**
  - Persistent observability of #273 fields (TTFT_ms, ITL_p50_ms, kv_blocks_used, kv_blocks_total) — these now exist on JobExecutionObservation as of the #273 Phase 2 work
  - A planner sub-system in gen-orchestrator that reads recent observations and a load forecast
  - SLO declarations on the endpoint — extend Resources or @inference(...) with `slo_ttft_ms` / `slo_itl_p50_ms`
  - Autoscaler hook so the planner's decisions actually scale pods

**Reference:** Dynamo 0.4 SLO-based autoscaling blog: https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/

Moved from gen-orchestrator #330 on 2026-05-21 (implementation belongs in python-gen-worker, not gen-orchestrator). Cross-references like #273/#322 still point to gen-orchestrator issues.

## Tasks
- [ ] Surface SLO kwargs on @inference: `slo_ttft_ms`, `slo_itl_p50_ms`. Store in _EndpointClassSpec.
- [ ] Sink observation data into a queryable form (orchestrator-side persistence).
- [ ] Forecasting layer: ARIMA or Prophet. Predict request rate + average prompt length 30s out.
- [ ] Decision layer: compute desired prefill + decode pool sizes from forecast + current utilization + SLO.
- [ ] Wire decisions into the existing autoscaler.
- [ ] Validation: synthetic load test that shows the planner pre-scales before SLO breach.

---

# #331: Cross-request stage interleaving scheduler for SerialWorker (LATER, depends on 3D throughput measurement)

**Completed:** no
**Status:** planned
**Related:** Parent: #325 (3D endpoints with @inference.stage annotations), Adjacent: #329 (disaggregation — different GPU pools, different problem)

Follow-up to #325. Open only when measured 3D throughput on TRELLIS-class workloads shows the GPU is idle >15% of total request wall-time during light stages (image_encoder / mesh_extract while another request is in the heavy texture stage).

**Pattern: stage interleaving on a single GPU.** A SerialWorker endpoint annotates its internal stages with `@inference.stage(name=..., gpu_class=...)` (already shipped in #325). The scheduler examines request flight times per stage and admits a second request to a worker when the first is in a stage that doesn't fully use the GPU. Different from disaggregation (#329): same GPU, different time slices, not different pools.

**Requires:**
  - Worker-side scheduler that can interleave stage executions across multiple in-flight requests (currently SerialWorker runs each request end-to-end before admitting the next)
  - Per-stage timing telemetry so the scheduler knows which stages are GPU-light
  - SDK contract: stage methods must be cancellable mid-execution if the scheduler reorders

**Don't pre-build.** Wait for measured idle time.

Moved from gen-orchestrator #331 on 2026-05-21 (implementation belongs in python-gen-worker, not gen-orchestrator). Cross-references like #273/#322 still point to gen-orchestrator issues.

## Tasks
- [ ] Measure 3D stage timings on a TRELLIS endpoint at production load. If GPU-idle time during light stages < 15% of total wall-time, this issue stays parked.
- [ ] If pursuing: prototype an in-process stage scheduler in worker.py that runs request A's heavy stage alongside request B's light stage on the same GPU.
- [ ] Determine whether CUDA stream parallelism is sufficient or whether torch.compile graph + memory layout interactions break it.
