- **The v2 serve path calls `materialize_input_assets` again, and the SDK emits the `moderation`
  block again (pgw#1418).** `cd46c957` deleted `executor.py` entire and the rewrite never re-wired
  either half, so every `@entrypoint` taking an `ImageAsset`/`VideoAsset`/`AudioAsset` failed
  `asset not materialized` — measured on a rented route-2 pod, 3 of `dj-utils`' 4 functions and
  ALL THREE of `music-analysis`'. Materialization now runs at the payload seam in
  `ServeLoop.invoke`, after decode and before any lease, and `discovery.moderation` re-derives the
  hub's `{prompts, media}` field paths from the payload type. The prompt half went with it, which
  was the safety-relevant half nobody had assessed.
- **`receipts.gate_delivered_artifact` fails CLOSED (pgw#1425, security).** It returned `True` when
  nobody had configured it, and the v2 serve path configured nobody — so every fleet worker armed
  hub-delivered native code with no receipt verified at all. Three explicit postures now: `armed`
  (`configure` at HelloAck), `local` (`trust_local_store(reason)`, for cozy-local/CLI/rigs), and the
  DEFAULT `unset`, which refuses with the typed `gate_unconfigured` event and falls through to
  self-mint. `configure("")` raises instead of returning silently.
- **Eight more orphaned wires reconnected.** Per-request capability renewal, the `StageTimer`
  handler span and its `stage_ms` on `JobResult.metrics`, `postmortem.current_inflight_request` on
  the `aot_ingress_refused` event, four `boot_phases.mark_once` milestones (`sdk_ready`, `hello`,
  `first_request_servable`, `eager_ready`), `process_role.declare`/`emit_boot_role` at the transport
  bind, C2PA remote signing at HelloAck, and the `ServePosture` scheduler message.
- **Fifteen rows leave `unreached_surface_baseline.txt`** — thirteen by acquiring a production
  caller, two DELETED with the ruling that kills them (`serving_facts.facts_or_degrade` and
  `models.memory.report_unevidenced_serving_facts` both collapse catalog-stamped serving facts, and
  pgw#1373 deleted the catalog). The baseline shrinks instead of accreting, which is the half of
  pgw#1425 that is about the instrument rather than the wires.
