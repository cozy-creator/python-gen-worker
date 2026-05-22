from __future__ import annotations

# Worker<->scheduler gRPC wire protocol version.
# MAJOR: breaking changes, MINOR: additive/backward-compatible changes.
#
# 1.3 (#321): protocol cleanup — removed Realtime{Open,Frame,Close}Cmd,
# RepoURLRefreshRequest/Response, RuntimeBatchingConfigCommand/Result,
# WorkerFunctionAvailabilityClearedSignal; stripped unused fields from
# WorkerResources / JobExecutionRequest / JobExecutionResult / WorkerEvent /
# IncrementalToken* / BatchExecutionItemResult / WorkerDrainResult. The
# old fields are `reserved` in the proto so wire bytes from older workers
# (1.2 and below) will still parse with the removed fields dropped — but
# emitting any of them from a worker is an error at construction time.
#
# 1.4 (#321): typed worker lifecycle signals — added WorkerStartupPhaseSignal
# (worker_startup_phase = 30), WorkerModelReadySignal (worker_model_ready =
# 32), WorkerFunctionCapabilitiesSignal (worker_function_capabilities = 33).
# Replaces the prior string-switched worker_event dispatch on event_type
# values "startup_phase"/"worker.startup.phase"/"function_capabilities"/
# "model.ready"/"model.cached"/"model.download.completed". Fixed a silent bug:
# worker emitted "worker.function_capabilities" while the orchestrator matched
# "function_capabilities" without the prefix, so capabilities had been silently
# dropped. Also unified the proactive `available: false` per-function flag with
# WorkerFunctionUnavailableSignal so the placement filter actually respects
# function capabilities. Additive; older workers continue to use the
# worker_event path (orchestrator no longer routes those, so capabilities and
# startup phase only flow via typed messages now — coordinate the release).
#
# 1.5 (#321): BATCH wire family removed — BatchExecutionRequest /
# BatchExecutionItem / BatchExecutionItemResult / BatchExecutionResult proto
# messages deleted plus oneof tags 21/22 reserved. The "batching" was
# wire-level grouping that the worker unpacked and ran serially — no real
# GPU batching, no orchestrator producer. Real LLM batching needs continuous
# batching with shared KV-cache and a different wire shape. UNLOAD: orchestrator
# now produces UnloadModelCommand on OOM-LRU eviction; worker side was already
# complete.
# 1.6 (#322): SDK foundation hard-cut. The wire protocol itself is unchanged
# — the SDK refactor is internal to gen-worker — but workers running with
# the new class-shape decorator advertise 1.6 so an old orchestrator can tell
# new-shape workers apart. Adds 'warming' startup phase (between
# pipeline_loading and ready) so workers in torch.compile warmup report
# honestly. SerialWorker / BatchedWorker archetype detection is driven by
# the discovered class shape; no proto change.
#
# 1.7 (#327): IncrementalTokenDelta gains `bytes audio_chunk = 12` and
# `string audio_codec = 13` so AR-TTS endpoints (Chatterbox, GPT-SoVITS,
# Bark, MusicGen) can stream raw audio bytes on the same wire as text
# token deltas instead of base64-encoding them inside payload_json. Saves
# ~33% wire size and the JSON CPU tax. Mutually exclusive at the semantic
# level: text deltas set delta_text/payload_json, audio deltas set
# audio_chunk/audio_codec. Additive — text-streaming workers keep working
# unchanged; only the chatterbox-tts endpoint switches over in this cut.
#
# 1.8 (#346): orchestrator-restart recovery. WorkerRegistration gains
# `in_flight_request_ids` (field 30) and `stream_started_unix_ms` (field 31).
# On RE-registration (not heartbeat) the worker reports the request_ids it is
# still processing (snapshot of _active_requests) plus any request_ids whose
# JobExecutionResult is still buffered in the outgoing queue, so an
# orchestrator that lost its in-memory assignment map across a restart can
# reconcile (requeue what the worker dropped, keep what it still holds)
# instead of guessing. Additive — older orchestrators ignore the new fields,
# older workers send them empty.
WIRE_PROTOCOL_MAJOR = 1
WIRE_PROTOCOL_MINOR = 8


def wire_protocol_version_string() -> str:
    return f"{WIRE_PROTOCOL_MAJOR}.{WIRE_PROTOCOL_MINOR}"
