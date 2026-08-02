# Changelog

## Unreleased

## 0.90.3 (2026-08-01) — **three families that could never be wired to the AOT lane now can** (pgw#853: a declaration that refuses to MINT was refusing to IMPORT), and `prop_s` is measured so the export-reuse change can be decided on its own sign (pgw#847)

- **pgw#853 — a declaration that REFUSES must not be able to refuse the ENDPOINT.**
  ltx-2.3, qwen-image and z-image express their mint blockers by raising
  `MintRefused` at MODULE SCOPE — and module-scope import is the only mechanism
  the platform has for registering a declaration. **So the three families most
  in need of the AOT lane were exactly the three that could not be wired to it:
  doing so would have taken the endpoint DOWN AT BOOT.** A refusal to MINT was
  being expressed as a refusal to IMPORT.
  `register_export_declaration()` now accepts a zero-arg **callable** with an
  explicit `family=`, and `export_declaration()` evaluates it so the refusal
  surfaces where the mint is asking. `has_export_declaration()` /
  `registered_entry()` read the registry WITHOUT evaluating (a blocked family
  is declared; it just refuses to mint), and `register_declared_exports` uses
  the non-evaluating accessor — reading it back through the evaluating one
  would have detonated a thunk inside endpoint COLLECTION, the exact blast
  radius this removes. `fleet_cells.mint_recipe` turns the refusal into a typed
  `self_mint_skipped` under a new `declaration_refused` phase **with the
  blocker text intact** (the evidence is the point; a try/except that swallowed
  it would be a different defect). The mint gate's catch is `Exception`, not
  `BaseException` — swallowing a `KeyboardInterrupt` or a cancellation inside
  the SERVING process would be its own defect; the import boundary keeps
  `BaseException` deliberately, because it runs at BOOT where nothing outranks
  the endpoint coming up.

- **pgw#847 — `prop_s` is measured, once per mint.** `torch.export.export` runs
  once per declared class row, serially, in the parent: sdxl is **36 entries**
  at a banked `export_s` of 37.8 s, so that loop is **~22 min of wall the
  pgw#809 pool never covered**. An exported graph's `graph_module.code` is
  byte-identical across shape rows, and one export plus a per-row
  `FakeTensorProp` reproduces `wrapper.cpp`, `kernel.cpp` and the linked `.so`
  byte for byte (proven with `torch.export.export` monkeypatched to **raise**,
  so the reuse path cannot have silently fallen back). The saving is
  `export_s - prop_s`, and `prop_s` had never been measured on a real family —
  off-pod probes bound it only to 0.25-0.97x, which is not enough to know the
  change's SIGN. Now recorded as `timings['prop_probe_s']` ONCE per mint
  process, on a FRESH `program.module()` so the program is untouched, and
  recorded not at all on any failure. **Telemetry only — no decision reads it.**

- **pgw#847 — export ONCE per module, re-specialize per shape row** — the change
  `prop_probe_s` exists to size. **Behind a fail-closed gate and OFF by
  default**; it does not affect any mint in this release unless explicitly
  enabled.

- **pgw#847 — one export can serve every shape row, behind a gate that must PROVE it (OFF by
  default).** A cell's N entries are one module traced at N shape rows, and an `ExportedProgram`'s
  `graph_module.code` is byte-identical across them — the row lives entirely in node metadata. So
  `gen_worker.aot_export_reuse` re-specializes one exported graph per row (deep copy +
  `FakeTensorProp` + torch's own `_update`) instead of re-running `torch.export.export`, which
  `aot_mint` does once per declared class row, **serially, in the parent** — ~22 minutes of an sdxl
  mint that pgw#809's K-wide pool divides by one. Measured byte-identical on every emitted file
  including the **linked `.so`**, with `torch.export.export` monkeypatched to raise for the whole
  reuse arm. Worth **11-21 minutes of serial mint wall per 36-entry sdxl cell**; the range is that
  wide only because `prop_s` on a real family has never been measured, and the new
  `timings["prop_probe_s"]` (one pass per mint, read by nobody) is there to close it.
  **Gated, because the invariant is a property of the module and not a law:** a family branching on
  a size traces a different graph per row, so the gate requires graph-text equality AND a
  byte-identical artifact from a re-specialization of a witness row, both arms built in the same
  cleared cache dir (`-g1` bakes the source path into the object). Every failure mode — exception,
  missing artifact, empty digest set, unplaceable input — falls back to a full per-row export;
  absence of evidence is never a pass. The verdict is per `(target, adapter arm)` per MINT and is
  never memoised across families. Enable with `GEN_WORKER_AOT_EXPORT_REUSE=1`. No inductor config,
  compiler, flag or library changes, and neither module is in the code closure, so **no cell
  re-keys**.

## 0.90.2 (2026-08-01) — **the mint's VRAM ceiling was its own estimate, not the card**: a whole-graph AOT mint died for 30 MiB with 21.48 GiB free (pgw#848), an OOM-killed entry child is no longer laundered into a never-retried refusal, the pool's host-RAM bound finally sees the compiler, and AOT-regional is deleted (pgw#846)

- **pgw#848 (CRITICAL PATH) — the mint's VRAM cap was the ESTIMATE, not the card.**
  Measured on two pods, two card sizes, one number:

        pod   card total   free at OOM   cap imposed   entries exported
        4090   23.52 GiB      660 MiB     11.09 GiB     1 of 36
        L40S   44.39 GiB    21.48 GiB     11.08 GiB     5 of 36

  The cap moved with neither a 2x card change nor `vram_gb` 12 -> 20, because it
  was a property of neither: `mint_budget.co_residency().need_bytes` was handed
  to the child as a hard `set_per_process_memory_fraction`, and for sdxl that is
  `4.87 x 1.25 + 4 + 1 = 11.09 GiB` — exactly what both pods printed, derived
  from `_UNMEASURED_ACTIVATION_FRACTION`, which nobody ever measured. The mint
  was never out of GPU; it was enforcing a self-imposed ceiling and then
  reporting the result as a deterministic refusal.
  **`need_bytes` now answers only "should this start"; a new `cap_bytes` answers
  "how far may it go"** and is `free - activation` — what the card has, less what
  the tenant needs for its NEXT forward (its weights are already allocated, so
  already outside `free`). On the L40S: **11.08 -> 20.26 GiB**. pgw#784 is not
  weakened: the tenant's next peak is still reserved by construction on every
  card, and a tight card still falls back to the estimate.
  `_UNMEASURED_ACTIVATION_FRACTION` is deliberately **not** replaced with a
  different off-pod constant — substituting one unmeasured number for another is
  a move this program has already paid for; it now bounds only the admission
  estimate and the tenant reserve, never the child. Widen-on-OOM now exists on
  the device half too, so a child that dies inside `torch.export` no longer
  banks nothing and leaves attempt N+1 to re-ask identically.

- **pgw#848 item 4 — an OOM-killed entry child was reported as a DETERMINISTIC
  REFUSAL,** so the one failure a narrower K would fix could never try one.
  Every entry-pool failure converged on `EntryCompileFailed -> MintRefused ->
  EXIT_REFUSED`, which `mint_process` documents as terminal and never retries —
  while the pool's own `_exit_note` has said since pgw#809 that a SIGKILL there
  "is the OOM killer far more often than a compiler bug". `EntryCompileFailed`
  now carries `resource`, `basis` and the dead entry's measured
  `peak_rss_bytes` (a child the OOM killer takes writes no report, so the
  parent's live per-row sample is the only measurement that will ever exist).
  `cgroup_oom_kills()` reads the kernel's counter, with `cgroup` (the counter
  moved — a fact) and `sigkill` (an inference, worth one retry) kept distinct
  and unreadable reported as -1, never a silent 0. New
  `aot_mint.MintResourceExhausted` is deliberately NOT a `MintRefused` subclass.
  Reproduced against a REAL kernel OOM kill of a real entry child under a
  cgroup v2 cap, with `memory.events` `oom_kill` asserted to have moved.

- **pgw#848 — the pool's memory bound had never seen the memory.**
  `aot_compile_child._peak_rss` read `RUSAGE_SELF` (blind to the compiler) and
  `aot_compile_pool._peak_rss_bytes` walked ONE level of `/proc` children — but
  on a real `aoti_compile_and_package` the entry child's direct children are
  `g++` (a driver that allocates nothing) and inductor's async_compile workers;
  **`cc1plus` is at DEPTH 2 and `ld` at depth 3**. Measured off-pod on the real
  sdxl AOTI wrapper TU (6,324,290 bytes, production flags, g++ 13.3): ground
  truth **2.052 GiB**, of which one `cc1plus` is 2.049 — against 0.012 GiB from
  instrument 1 (**171x low**) and 0.015 GiB from instrument 2 (**133x low**).
  Nothing banked the pool's own `peak_child_rss_bytes`, and `aot_mint` never
  passed `peak_rss_bytes` to `entry_workers()` at all, so `mem_workers` divided
  available RAM by a 3 GiB constant on every mint the fleet has ever run and
  `per_entry_rss_basis` read `"default"` permanently.

- **pgw#846 (P0, Paul's ruling) — AOT-regional is DELETED** (full entry under
  Unreleased history below; the exported cell is always WHOLE-GRAPH again and
  whole-graph cell identity does not move).

- **te#148 — svdq low-rank branch can arrive QUANTIZED (int8 | fp8_e4m3).**
  `decode_linear`/`load_svdq_native_denoiser` accept a branch pair stored
  int8 or fp8_e4m3 with fp32 per-block-32 scales along each factor's
  contraction dim (LoRaQ, arXiv 2604.18117), declared by the new
  `__metadata__` key `lowrank_quant` (absent = bf16, historical format —
  byte-identical behavior). Declaration and bytes must agree; either mismatch
  refuses. v1 dequantizes on load (bf16 downstream — SvdqLinear /
  fold_to_dense / split_decoded untouched); `quantize_lowrank` /
  `dequantize_lowrank` are the format's encode/decode pair for the te#148
  producer half. Quantized branch tensors are plain row-major (the 16x16
  lowrank fragment pack is a 16-bit-operand convention; nunchaku cannot read
  a quantized-branch file regardless).

- **pgw#846 (P0, Paul's ruling) — AOT-regional is DELETED.** Regional
  (block-class) export/mint/arm is removed end to end: `aot_regional.py`, the
  regional export fork and shell-digest machinery in `aot_mint`, the
  `regional=`/`regional_shape_strategy` declaration plumbing
  (`aot_declaration`, `Compile`, `export_contract`), the `aot-regional` mint
  recipe, and the pgw#829 regional entry-collapse. The exported cell is
  always WHOLE-GRAPH again; contract-facts stay v3 with `shell_digest`/`mode`
  pinned `""`, so whole-graph cell identity does not move. What survives:
  `Compile.regional` and all of `compile_cache`'s use of it (the ltx-video
  dynamo/JIT per-block OOM workaround, ie#381/gw#472 — a different feature);
  the numerics calibration (`NUMERICS_FLOOR`/`NUMERICS_WARN`/
  `declared_thresholds`), moved to `numerics_ladder` — family-general, the
  gate that should have caught the regional serve regression; pgw#844's
  dispatch-admission gate, re-keyed on `(target, adapter arm)` and proven to
  ADMIT the 36-entry whole-graph sdxl shape; and `provision.arm_route`'s
  decline-by-name — a `mode='regional'` cell now declines by name and stays
  eager (the retirement semantics), with a pinned test.

## 0.90.1 (2026-08-01) — the release that actually CONTAINS the drain fix: a drain no longer drops a COMPLETED job's result (pgw#845 P1), superseding a 0.90.0 published from a pre-fix head

- **Supersedes 0.90.0 — never pin 0.90.0.** The v0.90.0 tag was cut at `9332c0e`,
  BEFORE the pgw#845 P1 drain fix (`09133ca`) landed, so the published 0.90.0
  wheel still drops the result of a job that already completed when the pod
  drains (scale-down window). 0.90.0's section below describes the intended
  release; the drain bullet in it is only TRUE of 0.90.1. Lesson, recorded in
  the tracker: a green CI run proves the tree it ran on, not that the tag
  points at the tree you meant to ship — verify tag contents BY CONTENT.
- **pgw#845 (P1) — a cancelled write cancels the whole gRPC call.** The drain's
  "clean close" cancelled the sender task; grpc.aio answered by cancelling the
  RPC, the RST discarded a `job_result` already retired from the durable
  pending set, and the same cancellation escaped `run()` with no exit code.
  Fixed via `SendQueue.quiesce` / `SenderQuiesced`: the sender is quiesced,
  never cancelled mid-write. Red 12/12 before; green 56/56 after (24
  sequential + 24 across four concurrent lanes on two pinned cores + 8
  top-up). The same defect sat on the pgw#763 supervisor stream cycle, where
  it would have discarded the typed death JobResult. Residual (filed, not
  fixed): on an ABRUPT close a written-but-unflushed result is in neither the
  wire nor the queue; closing that needs a hub-side result ack (proto change)
  — today the hub reconciles it (a gap in the guarantee, not a silent drop).
- **pgw#845 — `test_entry_collapse_pgw829` NameError repaired** (a helper
  rename verified as a single-node run instead of the file: a file-wide change
  verified as a local one), and the sibling stopwatch assertion de-clocked.

## 0.90.0 (2026-08-01) — **an AOT cell can advertise `compiled` for the first time** (pgw#844 P0: the exported lane was never asked for its guard-revocation signal, and a partially dispatchable cell claimed nothing); the bake gate refuses a wheel that omits an endpoint module (pgw#833: the sp086 pod-death P0), the child's stderr rides the post-mortem, the T_BOOT_FATAL ack closes the verdict race, and sdxl's regional mint can derive 8 entries from 72 (pgw#829, per-family opt-in); the gw#640 SIGTERM drain hang is a FIXED lost wakeup (pgw#833 follow-on), a drain no longer drops the result of a job that already succeeded (pgw#845 P1: a cancelled write cancels the whole gRPC call), four runner-flake classes die and the wall-clock guard can finally fail (pgw#845), the seal split rides the phase table (pgw#842), and re-sharding is retired (th#1362)

- **pgw#844 (P0) — an AOT cell could never advertise compiled, and one
  undispatchable aspect bucket cost the pod every other shape.** Attempt
  twelve (L4 `o0legpgj5olhic`) adopted the first cross-pod cell in platform
  history — 72 entries armed, 58 s — and then served 100 % eager, including
  1024x1024, whose entry was armed, correct and unambiguous. Two independent
  defects, both fixed, both red-first on the real boot path:
  - **the exported lane was never asked for its revocation signal.**
    `_bind_compile_guard` probed TRT and dynamo only, and
    `provision.enable_compiled` returns as soon as `arm_aot` succeeds — so an
    AOT-armed pipeline carries no `compile_cache` `failure_signal` marker at
    all, answered *"no runtime guard revocation signal"*, and had its
    `active_compile_ref` cleared on every boot. `aot_serve` has owned
    `set_guard_failure_callback` since pgw#721 and nothing called it. This is
    why no `serving_mode=compiled` row exists anywhere on the release: a
    compiled AOT serve was structurally unreachable regardless of dispatch.
  - **the boot's coverage claim was all-or-nothing.** A transformer block sees
    `(B, H_lat*W_lat, C)` — the token PRODUCT — while entries are keyed on the
    latent H and W separately, so sdxl's 9 aspect buckets collapse to 4 token
    counts and 8 of 9 are `entry_ambiguous`. An alias attributed to an object
    only when EVERY declared graph class proved there, so a partially
    dispatchable cell claimed nothing (hot adopt: `function_alias_unproven` ->
    rollback). On the EXPORTED lane an alias that proves SOME of its classes
    is now attributed, and the classes that stayed eager are NAMED on one
    `compiled_shape_coverage / partial_shape_coverage` event. Dynamo keeps the
    strict rule — there an unproven class is an unannounced recompile, which
    is silent, while an exported refusal is typed, counted and armed-through.
  `boot_ended_uncompiled` now means *nothing is dispatchable*, never
  *something wasn't*.
- **pgw#844 — a refused shape no longer contaminates the compiled
  measurement.** An `aot_serve` / regional-block ingress refusal reports
  through a new `set_ingress_refusal_callback` seam and charges THIS request
  `fallback_reason=ingress_refused` on its own `JobMetrics`, so an eager
  sample from an armed lane stops being counted as `serving_mode=aot_cell`
  with no reason. Only reachable now that a partially dispatchable cell stays
  armed, which is why it lands with the fix rather than before it.
- **pgw#844 part B — the mint's dispatch-ambiguity gate asks about ADMISSION,
  not equality.** pgw#829's gate compared a digest of each entry's placeholder
  shapes, which catches identical contracts (sdxl's 9 static rows) but not the
  case its own remedy introduces: a static row and a collapsed row over the
  same token hull have different digests and both admit the same call. Every
  entry's declared call now runs against every sibling's contract through
  `aot_serve.assert_ingress` itself — the same function `EntryDispatch.select`
  runs on a pod — grouped by (target, block class, adapter arm) exactly as
  dispatch groups. Still pre-compile: seconds to refuse, not a full compile
  bill.
- **pgw#845 (P1) — a drain dropped the result of a job that had already
  SUCCEEDED.** Roughly one drain in six, measured: `job finished r-last
  status=1` -> `drain complete` -> stream closed, and the tenant's completed
  request returned nothing. Every scale-down could swallow a request that
  finished in the drain window; the GPU time was spent and billed either way.
  The cycle, in one line: the clean close did `send_task.cancel()`, and if the
  sender happened to be inside `stream.write()` of the next post-flush event,
  **grpc.aio answers a cancelled write by cancelling the whole RPC**
  (`_call._write` calls `self.cancel()` on CancelledError) — the RST discarded
  the `job_result` that was buffered one message earlier and that
  `mark_result_shipped` had already retired from the durable queue, so the
  half-close had nothing left to flush; `read()` then raised the same
  cancellation (`_raise_for_status`), which escaped
  `except (TimeoutError, ConnectionError) / except Exception`, skipped the
  wait for the peer to end the call, and rode out of `run()` -> `arun()` ->
  `Worker.run()`, killing the process with no exit code at all. Note what
  `wait_empty` proves and what it does not: it proved the result was WRITTEN,
  and the write was then thrown away underneath it.
  The sender now ends BETWEEN writes — `SendQueue.quiesce()` ends the loop
  once nothing is left to write (never with a message queued), bounded by the
  keepalive window; cancelling is the last resort and marks the close abrupt
  (`_clean_close = False`) instead of pretending it was clean. The peer-close
  wait is `asyncio.wait`, which neither cancels the receiver nor re-raises
  what it ended with, so an RPC-level cancellation can no longer escape a
  graceful close; and a cancelled call reaching the generic path is re-raised
  as the `ConnectionError` it actually is, which reconnects instead of ending
  the process. Both close paths — hub drain and the pgw#763 supervisor stream
  cycle, whose comment already named the supervisor's typed death JobResult as
  the thing at stake — share the two helpers. Guard: the pre-existing
  `test_drain_finishes_in_flight_then_closes_and_rejects_new_work`, left red
  on purpose by the previous commit, plus a new assertion that the drained
  worker exits 0. Red 12/12 on the parent commit (5 lost the result outright,
  7 shipped it and died by exception); green 48/48 after — 24 sequential and
  24 across four concurrent lanes, all on a two-core pin.
- **pgw#845 — the wall-clock source guard could not fail, and two more
  "flakes" were the test.** `_LITERAL_DEADLINE` required a DIGIT after
  `time.monotonic() +`, so `deadline = time.monotonic() + _TIMEOUT` was
  invisible and six `tests/harness/` files hid deadlines that way — the harness
  guard was asserting an empty set. Widened to see a name bound to a literal
  duration, the clock call itself, latency/rtt, and BOTH directions (a lower
  bound fails on a FAST runner). Five test files and two harness fixtures were
  exposed and each dispositioned rather than allowlisted. Separately,
  `test_procsplit_pgw763.py::test_signal_death_consumes_the_inflight_marker_*`
  asserted the parent's post-mortem at the instant it observed the durable
  job_result — `_handle_child_death` does attribution first and forensics
  second, so under two-core contention 3 of 5 runs had no dial yet; it now
  waits on the forensics via `await_progress`, giving up only when the parent
  process is gone. And the bounded-shutdown tests now assert the escalation's
  PRODUCT — the post-mortem naming `worker_process_exit` / `SIGKILL` — with the
  wall bounded by the grace the test itself CONFIGURED instead of a literal 45,
  so the gw#640 entry leaves the burndown rather than living there. Tests only;
  no product change.

- **pgw#833 follow-on — forwarding a signal is not draining a pod.** The
  gw#640 SIGTERM test hung CI three runs running. It was not the stderr tee
  (`844f9f6` was CI-3's parent and the hang survived it) and not a blocked
  signal mask: all three live specimens off the failing runs had `SigBlk` and
  `ShdPnd` **zero**, and an instrumented repro caught the child's SIGTERM
  handler firing 1 ms after the parent forwarded it. The deadlock is a lost
  wakeup: the child installed a handler, announced `READY`, then entered
  `signal.pause()` — and on a contended box the forwarded SIGTERM lands in the
  gap, is consumed by the handler, and `pause()` then waits forever for a
  signal that already came, while the parent waits forever in `waitpid`. The
  stand-in now blocks SIGTERM and `sigwait`s it, which has no gap.
  Underneath the flake sat the real P0: `supervise()` forwarded a terminating
  signal and then waited **unboundedly**. Any child that cannot answer — deaf,
  wedged below Python in a CUDA call, or blocked writing into a stderr pipe
  nobody drains — pins PID 1 alive, and a rented GPU keeps billing. Three
  fixes in `supervisor.py`:
  - **TimeoutStopSec for the outer layer.** The first terminating signal arms
    `setitimer(ITIMER_REAL, 180s)`; when it expires the worker is SIGKILLed and
    the post-mortem names the SIGKILL, so shutdown always completes and never
    silently. 180s deliberately outlives the procsplit parent's own 120s
    `_DEFAULT_STOP_TIMEOUT_S` so the inner escalation and its death dial run
    first. Repeated SIGTERMs cannot push the deadline out.
  - **The fork window is closed.** Between `fork()` and the parent installing
    handlers, SIGTERM was still `SIG_DFL` — landing there killed the reporter
    and stranded the worker as an orphan (a shape observed live on the box).
    The contract signals are now blocked across the fork and unblocked only
    *after* the handlers exist; the reverse order delivers a pending signal
    straight into `SIG_DFL`.
  - **The mask is taken back on every path**, including the un-supervised one:
    it survives fork AND exec, so a launcher that blocks SIGTERM otherwise
    decides that the drain is undeliverable. (Mechanism found by the 0.90.0
    cut lane.)
  Guards executed red first: a wedged child (`deaf`, and the exact
  stalled-stderr-consumer hazard) hangs to the timeout without the escalation,
  and the blocked-mask launcher hangs without the unblock — 3 red / 3 green,
  11 passed in 14.6 s where the pre-fix tree burned 211 s in hangs, 6/6 under
  2-core pinning. The shutdown tests also reap their own pair now, so a red run
  stops stranding supervisor orphans on the host.
  Separately, CI's `RuntimeError: Event loop is closed` pair is **not** this
  hang: they are `PytestUnraisableExceptionWarning`s in the warnings summary,
  attributed to two *passing* tests, from `BaseSubprocessTransport.__del__`
  running after its loop closed — newly reachable because pgw#833 gave the
  child a stderr PIPE for `__del__` to tear down. The parent now closes a
  reaped child's transport deterministically, after `_settle_link` has drained
  stderr to EOF.

- **th#1362 — retire safetensors sharding: read-tolerant, write-invariant.**
  Sharding exists upstream to solve resumable transfer, parallel download,
  object-size caps and partial-failure re-upload. Chunked CAS solves all four
  BELOW the file, and uniformly — including for files that were never sharded —
  so the shard planner was a second, coarser answer to solved problems, plus an
  index that can disagree with the bytes it describes (the klein-4b unloadable
  publish). Four changes, split by WHO OWNS THE BYTES:
  - `models/chunk_cas.py` materialises POSITIONALLY: each worker `pwrite`s its
    chunk into its own byte range as blocks arrive, and the whole-file hash runs
    on its own thread over the contiguous VERIFIED prefix. RAM per worker drops
    from a 64 MiB chunk to one read block, so the default window goes 6 -> 16:
    measured 188.0 -> 274.3 MB/s at 3.6x LESS peak RSS. Resume is now per chunk,
    out of order, via a sidecar journal that is re-verified off disk — which
    also makes resume WORK for the first time (the part file used to carry a
    fresh uuid per call). Disk is proven up front with `posix_fallocate`.
  - Mirror ingest DE-shards (`deshard_mirror_tree`,
    `merge_safetensors_by_offset`) instead of re-sharding, including pure
    pass-through mirrors, so the corpus we own has one shape. The merge verifies
    its result against the index it consumed. `_reshard_indexed_safetensors` and
    `_stage_oversize_safetensors` are gone.
  - The planner is DELETED: `MAX_SAFETENSORS_SHARD_BYTES`, `ShardPlan`,
    `plan_shards`, `build_index`, `shard_safetensors_by_offset`, and
    `shard_threshold` from eight signatures. `shard_prefix` is `output_stem`;
    `ConversionResult.index_path` is gone.
  - `assert_one_file_per_component` fails closed at every producer output check
    and at `publish_flavors`, and those `save_pretrained` calls pass
    `NEVER_SHARD_MAX_SIZE` — save_pretrained shards on its own, which is why
    this is checked rather than assumed.
  READING a sharded artifact stays supported PERMANENTLY, and a user's own
  sharded upload is never refused or rewritten — it does not pass through
  `publish_flavors` at all.
  BREAKING for endpoint repos importing the deleted names (training-endpoints
  `conversion/fuse.py` is updated on its chaos branch and needs the pin bump).

- **pgw#842 — the entry pool's width is now explainable, and monotone in the
  box.** Two real L4 mints of the SAME 72-entry sdxl regional cell, back to
  back: attempt ten (0.86.0, 16 vcpu / 62 GB) `entry_workers=5`, `compile_s`
  1314.94, wall **347.94 s**; attempt eleven (0.89.0, 21 vcpu / 83 GB)
  `entry_workers=3`, `compile_s` 1327.23 (+0.9 % — identical work), wall
  **554.78 s (+59 %)**. Pool efficiency was ~97 % in both: the entire
  regression is K, on a bigger host.
  Why nobody could say which bound bound: `_mint_phase_table` has always built
  a `pool` block holding every input (`cpu_workers`/`mem_workers`/
  `device_workers`, the free-VRAM and available-RAM readings, the per-entry
  asks) and `emit_phase_events` **never emitted it** — only the scalar
  `entry_workers` reached a hub row, folded into `totals`. And pgw#830's pool
  ledger IS emitted, from the mint CHILD, which holds no orchestrator session
  (`mint_delegate._emit_aot_phases` exists because of exactly that) — so it
  died in a pod log. Verified against the chaos hub: **zero `phase='pool'`
  rows for either release**, on a stack where both mints are otherwise fully
  recorded. The two pods are gone, so their binding constraint is
  unrecoverable — that is the defect, not a footnote to it.
  Fixed on both halves. (1) The pool's decision is a typed hub event:
  `kind=aot_mint_phases phase=pool`, `duration_ms` = the pool's wall clock,
  detail leading with `entry_workers=`, `binding=` and `underwidth=` (workers
  the pod could have run but didn't, named with the constraint that held
  them), then every reading and the pgw#830 ledger — relayed parent-side like
  the rest of the table, and emitted for aborted mints too. `PoolWidth` now
  carries `CpuFacts`/`MemoryFacts`/`DeviceFacts`: which of quota / affinity /
  host cores the vCPU number came from (RunPod advertises `host_vcpus`; the
  kernel enforces a quota, and they are routinely different), which of
  meminfo / cgroup bounded RAM, every free-VRAM sample, and whether the
  per-entry asks were `measured` or a `default` constant. A narrow pool also
  logs WARNING with its inputs.
  (2) Two readings that were not monotone in the box are corrected. The cgroup
  RAM headroom counted PAGE CACHE against the pool (`memory.max -
  memory.current`), so the bound shrank in proportion to how much I/O the pod
  had already done — a mint reads GBs (weights, the toolchain the seal hashes,
  every staged program). It now subtracts reclaimable file pages, i.e. sizes
  on the working set, as every container runtime does: on a 62 GB pod with
  50 GiB charged and 40 GiB of it cache, 12 GiB -> 52 GiB, K=2 -> K=8. And
  free VRAM was a SINGLE `mem_get_info` taken on a card the pool shares with a
  live tenant by construction (pgw#784): a sample landing inside a tenant
  forward reads that forward's activation set as gone, while
  `DEVICE_RESERVE_BYTES` reserves the tenant's peak anyway — the same bytes
  charged twice, and worth exactly the observed 5-vs-3 (21 GiB steady vs
  13 GiB dipped, at 3 GiB/entry: K=6 vs K=3). `device_facts` now takes the max
  over three samples 50 ms apart and records all of them.
  Red/green on the real path: 11 new tests, all 11 red on the unfixed tip;
  the hub-visibility one drives a REAL 2-entry pool, its real ledger and real
  width through the real parent-side relay into a bound activity sink.
  Projected at a corrected K=5, attempt eleven's own numbers give ~374 s
  against the measured 554.78 s.
- **pgw#842 (second item) — pgw#832's headline was the HASH, not the seal:
  2.76 s/entry on a pod, not "0.10 s".** 0.87.0 recorded "9.8 s -> 0.10 s per
  entry, MEASURED"; that 0.10 s is `seal_libhash_s` alone, measured off-pod.
  Attempt eleven measured the span the pod actually pays, `child_seal_s`, at
  **199.085 s / 72 = 2.76 s/entry** — a ~3x cut against 8.14 s, not ~98x, and
  still 15 % of `compile_s`. The gap is measured, not guessed: with a warm
  memo on this box a child's seal splits into `seal_config_s` 3.62-4.70 s and
  `seal_libhash_s` **0.056-0.080 s** (bare `import torch`: 1.895 s). The memo
  did everything claimed OF THE HASH (~8 s -> ~0.07 s); what remains in
  `child_seal_s` is the child's own `import torch`, which `establish_config`
  owns by design — untouched by pgw#832, and now the largest per-entry fixed
  cost in the pool (~199 s of a 72-entry mint).
  So the overlays travel with the phase table too: per entry under
  `timings.overlays` and summed into the roll-up as `overlays=` beside
  `phases=` (never inside it — that was pgw#830's second attribution bug). A
  reader who sees `child_seal_s` alone re-opens a question the split answers.

- **pgw#840 — the entry-compile child must BE the parent's own gen_worker.**
  pgw#830's attribution invariant went red on a tree nobody had changed: one
  entry's table had no child spans at all, so its entire 19.5 s compile fell
  into `reap_lag_s` and the partition stopped closing. The compile had
  SUCCEEDED and returned files that exist — only the report was from other
  code. Root cause: the pool spawned `sys.executable -m
  gen_worker.aot_compile_child` with the parent's env verbatim and let the
  CHILD's import system pick a `gen_worker` — the cwd first, then any inherited
  `PYTHONPATH`, then site-packages. On a box with more than one checkout that
  is a coin flip, and the child that wins compiles the loose files the cell
  publishes while every gate runs in the parent against the parent's program.
  MEASURED on the box that filed it: of 236 preserved entry reports, **150 were
  written by a child predating pgw#830's span table**, several under a parent
  that had pgw#832 (their pool workdirs hold the `seal-lib-memo.json` only such
  a parent writes) — same venv, same interpreter. Attribution was the symptom;
  the defect was an unpinned compiler.
  `child_env` now prepends the parent's own package root to `PYTHONPATH` and
  sets `PYTHONSAFEPATH=1` (the cwd would otherwise still outrank it), and the
  child stamps every report — success and both refusals — with the digest of
  the parent/child contract source it computed at ITS import, plus where it
  imported from. `_collect` REFUSES the entry by name on any mismatch,
  including the empty digest an old child leaves: an artifact compiled by code
  the parent never ran must not be packed into a cell whose identity claims the
  parent's. Identity-inert: `PYTHONPATH`/`PYTHONSAFEPATH` are in no scrub
  namespace and in nothing `env_seal` reads back, and the pool's own
  cell-identity tests are unchanged and green.
  Deliberately NOT extended to pgw#784's mint child: that child DOES load the
  endpoint, and `PYTHONSAFEPATH` would remove the cwd its module may resolve
  through — the same hole, needing its own evidence, filed rather than guessed.
  Red/green on the real path: on the unfixed tip a child that is not this
  gen_worker is ACCEPTED and reproduces the filed table verbatim
  (`compile_s == reap_lag_s`, no `child_wall_s`, the same violation string);
  with the fix it is refused by name. Post-fix ledger, 4 real entries: **zero
  violations, dark residual 4.9-8.1 %, `child_other_s` <= 0.002 s**, and
  pgw#832's `seal_libhash_s` still 0.06-0.17 s.


- **pgw#833 follow-on — the child-stderr tee writes OFF the event loop.** The
  pump teed each chunk to the parent's stderr with a blocking `flush()` on the
  loop thread; when the parent's own stderr is a pipe with a stalled consumer
  (pytest capture, a throttled log collector), the flush froze the loop —
  signal handling and the shutdown path included (measured:
  `test_sigterm_is_forwarded_to_the_worker` 60 s timeout, CI 2/2 and a 2-core
  local repro). The tee now runs per-chunk in `asyncio.to_thread`; ordering
  within the single pump task is unchanged.
- **th#1303 S1 (worker half) — the v1 (blake3) verify-on-fetch arms die.**
  Every fetch-side check is unconditional v2 (chunked sha256); the test
  harness speaks v2, v1-pinning cases are retired, and each deleted arm's
  guard was EXECUTED red before deletion (two were wrong and are fixed).

- **pgw#829 — a conv-free block class collapses its whole SHAPE axis onto ONE
  entry: sdxl's regional mint goes 72 entries -> 8.** pgw#830 measured that
  attempt nine's 72 entries were not a scheduling problem (the pool was already
  >= 95.5 % efficient) but a per-ENTRY fixed cost — interpreter boot, the env
  seal, the torch import, the staged-program load, the reap lag — multiplied 72
  times. pgw#812 measured the lever: dynamic inner dims are **0.0 %** at serve
  on a conv-FREE region, because `decide_layout_opt` bails only on conv + free
  symbol, and #730's static-rows verdict (+7.2 %) was always a statement about
  CONVS — which regional leaves in the eager shell.
  `Compile.regional_shape_strategy` declares the BLOCK population's strategy
  separately from `shape_strategy` (which keeps governing the conv-bearing
  whole-graph route), and under `dynamic-collapse` a plan's whole class-row set
  becomes one entry whose block dims are DERIVED: the shell is run eagerly once
  per declared row, the block's own input shapes are recorded, and axes that
  move become `torch.export` dims over their observed hull. That derivation is
  the point — the declaration binds `H_lat`/`W_lat` to `sample`, and the block
  is handed a flat `(B, H*W/f**2, C)` hidden state carrying neither name, so
  the varying axis is only observable. Sweeping the rows costs no extra eager
  forwards (one per plan-row either way) and retains no activations: only the
  seed row's tensors survive the probe.
  Guarded, not assumed: the collapse is decided PER BLOCK CLASS off the live
  module (`aot_regional.block_has_conv`) — a conv-bearing block class keeps one
  static entry per class row; an axis whose hull reaches 1 is REFUSED (torch's
  0/1 specialization is not overridable, ie#543) rather than silently
  specialized; slots that move in lockstep across the rows share one symbol
  (pgw#812 D1, one level down); a rank change across rows is a fork, refused by
  name. `_regional_entry_count` derives the pool's width the same way, so
  pgw#812 S7's re-price does not go stale. Dispatch needed no change: a
  collapsed entry is discriminated by its recorded range, and the structural
  adapter/CFG forks stay separate entries. `regional_shape_strategy` without
  `regional=True` is a declaration nothing reads, and is refused.
  Proven off-pod on real `torch.export` + real AOTInductor: 8 per-shape entries
  become 2, every declared coordinate dispatches onto a collapsed entry, and
  every block INSTANCE at every SHAPE agrees with the per-shape cell and with
  eager to 1e-6 while being no FURTHER from eager than the 8-entry cell it
  replaces — stated as no-degradation rather than as bit-equality on purpose,
  since a dynamic kernel is different compiled code from a static one and
  demanding identical bytes would over-specify. Priced on a real pool ledger
  at the A/B's own entry counts: the 6 entries that stopped existing took 6
  whole copies of pgw#830's per-entry constant with them. pgw#831's
  folded-constant refusal still fires under a dynamic dim — asserted, so it
  cannot silently become unreachable for collapsed entries.

- **pgw#829 (found by its own A/B) — a REGIONAL entry traced from a
  NON-CONTIGUOUS captured block feed computes a 16 % WRONG answer, silently.**
  A block's example inputs are CAPTURED from a live forward (pgw#812 S5), never
  constructed, so whatever memory layout the shell happens to hand the block is
  what gets traced — and AOTInductor generates against that layout. Measured
  off-pod, $0, CPU, on a 3-block toy whose shell does diffusers' own
  `permute(0, 2, 3, 1).reshape(b, h*w, c)` (a non-contiguous view):

  | arm | max abs delta vs eager |
  |---|---|
  | traced non-contiguous, served with the pgw#791 realign | 0.1645 |
  | traced non-contiguous, realign DISABLED | 0.1690 |
  | traced CONTIGUOUS, served with the realign | 1.5e-08 |

  So it is not the realign — it is the artifact, and no serve-side layout can
  satisfy it. Nothing refused it either: the ingress contract records shapes and
  dtypes, never strides, so the call is admitted and the answer is quietly off.
  The mint now traces every regional entry from a contiguous feed, which makes
  it agree with `aligned_feeds` (already staging out-of-contract inputs into an
  owned contiguous buffer) by construction. A no-op for a feed that already
  arrives contiguous — diffusers passes sdxl's blocks through `proj_in`, a
  Linear, so the real family is in that case — and eager is untouched.

- **pgw#829 (found by its own A/B) — a cell whose entries cannot be told apart
  at dispatch is REFUSED at mint instead of serving eager forever.**
  `EntryDispatch.select` calls two entries admitting one call
  `entry_ambiguous`, which is a per-REQUEST refusal: the cell arms, reports
  armed, and serves those coordinates 100 % eager while looking healthy. A
  regional entry is exported one block deep — the shell has already flattened
  the latent extents into a token count — so two class rows that are different
  coordinates upstream can hand the block the IDENTICAL shape. What collides
  is the token PRODUCT, of which a transposed aspect pair is only the obvious
  case: sdxl's NINE aspect rows carry just FOUR distinct token counts (15360 =
  1536x640 / 640x1536; 15808 = 1216x832 / 832x1216; 16128 = 1344x768 /
  1152x896 / 896x1152 / 768x1344, a quadruple whose members are not each
  other's transpose since 96*168 == 112*144; and 16384 = 1024x1024, the only
  unique row). So attempt nine's 72-entry cell could have served exactly ONE
  of its nine aspect ratios compiled — the other eight were `entry_ambiguous`
  -> eager, per (CFG arm x adapter arm x block class). The new gate groups
  entries exactly as the serve path does (target x block class x adapter arm)
  and compares the EXPORTED placeholder signature — the thing the packed
  contract is derived from, so it is an exact discriminator that is already
  available before a single kernel is built. It refuses by name and names the
  remedy: the collapse, under which one entry over the hull is unique by
  construction.

- **pgw#833 — the "split cannot boot a hub pod" P0 root-caused: a wheel-omitted
  endpoint module, and the two gates that let it reach a paid pod.** The first
  hub-launched 0.88.0 pod crash-looped its compute child untyped (`exit:1`
  pre-Hello, ×3, `compute_boot_crash_loop`) — reproduced OFF-POD in the real
  wan-2.2 image: `wan_2_2/finish.py` imports package-root `cozy_finish`, which
  the wheel's hatch `only-include` never shipped, so `collect_endpoints` dies
  with `ModuleNotFoundError` at boot. The split executes fine (G=1 and G=2
  boots proven; with the module present the same image boots past the death
  point). Three gen-worker fixes: (1) **bake gate parity** —
  `discover_functions` refuses a walk that imported source-tree-only modules
  when the project is installed (`SourceOnlyModuleError`; the bake previously
  passed because it injects `root`/`root/src` into `sys.path`, a weaker
  predicate than the runtime walker); (2) **child stderr in the post-mortem**
  — the control parent captures each compute child's stderr (teed byte-for-byte
  back to the container log), and a child death dial now carries
  `child_stderr_tail` (and the `compute_boot_crash_loop` give-up names the last
  lines) — a pre-Hello death is diagnosable from `pod_events` alone, no
  container-logs API needed; (3) **T_BOOT_FATAL ack** (the pgw#826 follow-on
  race) — the parent acks after RECORDING a terminal boot verdict and the
  dying child waits (bounded) for it, so the typed verdict can no longer lose
  to the reap on a slow host.

## 0.89.0 (2026-08-01) — the cell self-mint publisher speaks chunked sha256: the v1 (blake3) client is deleted, and the procsplit allowlist stops refusing the publisher's own payload
- **pgw#807 item 3 — the cell self-mint publisher ships over CHUNKED SHA-256,
  and the seam it rides stops refusing its own payloads.** The first AOT mint
  in platform history (attempt ten, a real L4) compiled 72 entries, packaged
  them, adopted the cell into its own runtime — and then lost the artifact at
  `seal_publish`, because `fleet_cells.CellPublisher.publish` was still on the
  frozen v1 (blake3) `commits` route: `410 unsupported_digest_algorithm`. The
  publisher now calls `HubClient.publish_v2` (th#1303's declare -> `{have,
  need}` -> PUT -> complete). Both gates the staged flip waited on are
  discharged upstream — th#1340 gave the v2 route the cell-publish claim (the
  same receipt + `cell_store` + `cell_receipts` writes v1 does), and the
  receipt reader already dispatches on the receipt's OWN algorithm tag, so a
  `sha256:`-bound cell resolves. Digest identity is sha256 end to end; no new
  blake3 anywhere. Proven against a real tensorhub: a 50.4 MB cell tarball
  published, `cell_store` row written with the post-0080 axes (`lane=w8a8-
  lora64 sm=89 sku=l4 gen_worker_version mint_duration_ms minted_by_pod_id
  minted_for_release_id`), a `sha256:`-bound signed receipt minted and
  verified against the local bytes, and a re-publish of the same bytes
  uploading nothing (`resident=1 uploaded=0`).
  - **The delta-1 seam allowlist was refusing the live publish bodies.**
    `cells.publish_intent` permitted `(family, cell_key, axes)` while the
    publisher has sent `identity_axes` + `mint_duration_ms` since th#1355, and
    `cells.publish_complete` permitted `(…, status, detail, axes)` — three
    names no caller and no hub route ever had — while refusing `ok`/`error`.
    An unlisted key is an `ActionRefused`, and the compute child is the process
    that publishes, so under the split (the only execution model since
    pgw#783) the publish-intent would have been refused before the 410 could
    even be reached. The table now enumerates exactly what the publisher
    sends, pinned by a test that drives the real publisher and authorizes its
    real payloads.
  - **pgw#711's `artifact_digest`/`manifest_digest` on publish-complete are
    gone.** The hub's route decodes `family, cell_key, checkpoint_id, ok,
    error` and nothing else, so the blake3 pass over the artifact computed a
    value no reader had — dead weight that the seam then refused.
  - **The publish LEG is on the wire.** `publish_v2` gained an `on_stage`
    callback and the publisher emits `self_mint_publish` `declared` /
    `uploading` / `committing` / `committed` beside the existing `started` /
    `published`, so `worker_activity_events` can say WHERE a publish stopped
    rather than only that it did. `HubPublishError` now carries the hub's own
    `status` / `code` / `retryable`, and a failed publish reports that code as
    its `phase` (`unsupported_digest_algorithm`, `cell_publish_flavor_
    mismatch`, …) instead of a prose string nothing can group by.

- **THE v1 (blake3) PUBLISH CLIENT IS DELETED (Paul's ruling: one transport in
  the tree, ever).** Item 3 survived as "accepted as dark" for a whole release
  cycle precisely because two transports could coexist — a caller of deleted
  code fails CI the same day, a caller of a frozen protocol fails at runtime
  with a 410 and a rented pod. `HubClient.commit` and its whole apparatus
  (`_upload_one`/`_upload_entry_once`/`_reopen_upload`/`_check_complete`/
  `_finalize`, the SDK grant lane, `_StagingLostError`, `BankedBlobGoneError`,
  `blake3_file`, `CommitFile.resolve`/`.blake3` and the by-reference add) are
  gone. `gen_worker.convert.hub` now speaks exactly one protocol.
  - **`ctx.save_checkpoint` publishes v2** (`request_context/_stream.py`, the
    pgw#807 item-4 site). Its fleet-floor blocker closed at 0.79.0. A multi-GB
    adapter now retries a 64 MiB chunk instead of a whole shard, and the
    `Tensors.blob_digest` it returns is `sha256:` — the key the blob is
    actually stored under, where it used to name a `blake3:` key nothing
    could resolve.
  - **The th#592 download-skip bank is DELETED** (`convert/bank.py`,
    `_publish_from_bank`, `lookup_clone_manifests`, `record_clone_manifests`,
    the bank-record leg of `run_clone`). Its adds were BY REFERENCE with a
    caller-asserted blake3 and no local bytes — exactly what a protocol whose
    guarantee is "the digest is proven from the bytes in hand" cannot accept,
    so it was un-migratable by construction and already dead in practice under
    the hub's write freeze (every bank publish 410'd into the full-clone
    fallback). Clones are unaffected in outcome; they lose an optimisation
    that had stopped working. **Follow-up for the coordinator: if download-skip
    is wanted back, it needs a v2-shaped design (bank the sha256 CAS refs and
    replay through `publishes` with `have`), not a revival of this code.**
  - **The receipt reader is sha256-ONLY** (`receipts.py` "phase 4"):
    `ARTIFACT_DIGEST_ALGORITHMS == ("sha256",)`, one hash pass per arm, the
    legacy bare-hex `artifact.blake3` claim and the `?blake3=` fetch param are
    gone (and with them that key on the delta-1 seam allowlist). A
    blake3-bound receipt is now a typed refusal, so its cell is re-minted and
    republished sha256-bound — the designed miss policy.
  - **Self-attested artifact digests are sha256** (`aot_cells`, `fleet_cells`,
    `mint_child`): the digest a pod advertises for the cell it armed now EQUALS
    the digest the hub recorded for those bytes, so the two are joinable.
  - **Sweep result — `blake3` remaining in `src/gen_worker`: 0 in the publish
    and receipt paths.** What remains is deliberately retained and is NOT the
    v1 publish protocol: (a) the model/artifact READ side (`models/cozy_cas`,
    `cozy_snapshot`, `hub_client`, `refs`, `residency`, `volume_verify`,
    `config_identity`, `executor`, `lifecycle`, `provision`, `api/types`) —
    th#1303's own rule is that READS ARE NEVER FROZEN, since freezing them
    would dark every artifact ever published; (b) `presigned_upload` /
    `s3_transfer` / `input_assets` / `request_context/_datasets` — the media,
    dataset-CAS and input-asset manifest protocols, each with its own hub
    routes and its own migration, none of them repo-CAS v1.

## 0.88.0 (2026-08-01) — the author ENVELOPE has an SDK carrier: `Resources(max_gpu_count=, parallel=)` (pgw#748, the SP fast-tier declaration surface)
- **pgw#748/th#1285 — `Resources` gains `max_gpu_count` and `parallel`: the
  author ENVELOPE has an SDK carrier.** The hub's builder has parsed
  `resources["max_gpu_count"]` / `resources["parallel"]` since th#1285
  (`extractStaffingEnvelope`), and the whole tier→degree product (typed
  admission, cohort-exact buys, degree-exact dispatch, the th#1347 flip) keys
  off `requirement_payload["parallel"]` — but the SDK could not declare either
  field, so no real endpoint could opt into the hardware-fast tier through the
  build path. Both fields are elided from the manifest when they carry nothing
  (`omit_defaults`): every existing release's payload is byte-identical.
  Validation mirrors the builder's ingest refusals at declaration time
  (whole-number ceiling >= `gpu_count`, known mechanisms only —
  `sequence`/`cfg`, a mechanism requires device headroom); declaring either
  implies `gpu=True`. The author never writes a degree, a tier, a packing or
  a device id.

## 0.87.0 (2026-08-01) — the process split is the ONLY execution model (the flag is gone), the control parent keeps the SIGUSR2 forensic contract, and the entry compile's dark time is named and stops being re-paid
- **pgw#832 — pooled entry children stop re-paying the toolchain hash: 9.8 s ->
  0.10 s per entry, MEASURED.**
  *(Corrected by pgw#842: those figures are `seal_libhash_s` — the hash pass
  alone, off-pod. The span a pod pays, `child_seal_s`, went 8.14 -> **2.76
  s/entry** (attempt eleven, 199.085 s / 72): a ~3x cut, not ~98x. The
  remainder is the child's `import torch`, which `establish_config` owns.
  Quote 2.76 s/entry.)*
  `env_seal`'s identity manifest SHA-256s every
  toolchain `.so` the image ships (36 files, 3.96 GB); its memo was per
  PROCESS, and the pgw#809 pool's worker is a process that compiles one entry
  and exits — so a 72-entry mint re-paid the pass 72 times, K-wide (~28 % of
  per-entry `compile_s`, pgw#830's measurement). The pool parent now seeds an
  on-disk digest memo (`write_library_memo`, keyed by `(path, mtime_ns,
  size)`) once per pool — near-free when the parent already sealed — and each
  child, pointed at it via `GEN_WORKER_SEAL_LIB_MEMO`, still enumerates and
  stats every file ITSELF, using a memo digest only on an exact triple match
  and falling back to the full rehash on any mismatch. The seal value is
  byte-identical to a full rehash in every detectable case (proven by test);
  the one undetectable case — same-size content rewritten with `mtime_ns`
  restored — is exactly the trust boundary the in-process `lru_cache` (same
  key) always had, asserted as such rather than papered over. Seeding cost is
  named (`seal_seed_s` on the pool ledger, outside the capacity identity);
  seeding failure emits a typed `aot_mint_phases phase=pool` event and children
  rehash in full (the safe path). `seal_libhash_s` is now timed where the pass
  runs instead of inferred from `seal_effective_s`, so the span stays honest
  under a memo.
- **pgw#830 — the dark 44 % of AOT compile time is named, and the attribution
  can no longer rot in silence.** Attempt nine recorded `compile_s=1331.72` with
  five phases summing to 742.6 s (56 %); the other 589 s had no name. The cause
  was structural, not a missing metric key: in the pgw#809 pooled path
  `compile_s` is the parent's Popen-to-reap wall of an entry CHILD PROCESS,
  while `phases` only ever measured the inside of `aot_compile`. New
  `aot_compile_spans` defines three nested partitions, each with an explicit
  residual — `compile_s = child_boot_s + child_wall_s + reap_lag_s`,
  `child_wall_s = child_seal_s + child_torch_import_s + child_devlock_s +
  child_program_load_s + compile_wall_s + child_other_s`, and
  `compile_wall_s = lowering_s + codegen_s + graph_passes_s + host_compile_s +
  compile_other_s` — so a phase that stops being measured shows up as a growing
  residual instead of as time silently vanishing. `triton_s`,
  `device_lock_wait_s` (counted by pgw#809's lock since it existed, never read
  until now) and `inductor_total_s` are reported as OVERLAYS, never summed into
  the partition: they nest inside members above them, and adding them in was
  the second attribution bug.
- **pgw#830 — what the dark time actually was: `env_seal.establish()`, once per
  ENTRY.** MEASURED off-pod on a real 8-entry pooled compile through real
  children: `child_seal_s` is **37 % of one entry's `compile_s`** (18.36 s of
  49.49 s), of which `seal_libhash_s` is 10.47 s — the identity manifest
  SHA-256s **36 toolchain `.so` files, 3.96 GB**, and its memo is an
  `lru_cache`, i.e. per PROCESS. A pool whose worker is a process that exits
  therefore re-pays a multi-GB hashing pass for every entry. `env_seal` now
  publishes `LAST_ESTABLISH_SPANS` (scrub / config / isa / posture / effective /
  libhash) so the cost is a line item rather than an anonymous setup charge.
  Telemetry only — no digest, no decision and no identity depends on it.
- **pgw#830 — POOL IDLE is a separate line item from serial dark time, because
  the two have opposite fixes.** New `PoolLedger` closes
  `capacity = busy + idle` exactly and splits idle by CAUSE
  (`idle_staging_s` — a freed slot waiting on the parent's serial
  `torch.export.save`; `idle_drain_s` — the straggler tail; `idle_spawn_s`;
  `idle_other_s`), charged as free-slot-seconds at the moment they occur rather
  than divided out afterwards. The pool emits its own typed
  `aot_mint_phases phase=pool` event, and per-entry `parent_stage_s` /
  `parent_spawn_s` ride the existing phase channel under a distinct prefix so
  nobody sums parent work into a compile total.
- **pgw#830 — the invariant is a TEST.** `test_compile_attribution_pgw830.py`
  drives the real `EntryCompilePool` through real `aot_compile` children and
  asserts each partition equals the sum of its members, that the residual stays
  under 15 % of `compile_s`, and that the seal split still covers
  `child_seal_s`. On the recorded run the unnamed residual is **4.8 %**, against
  44 % before.

- **pgw#783 — the parent/child process split is now the ONLY worker execution
  model; the `GEN_WORKER_PROCESS_SPLIT` flag is gone.** `entrypoint`
  unconditionally becomes the control parent and execs one compute child per
  execution group — there is no single-process mode to fall back to.
  `split_enabled()`/`ENV_SPLIT` are removed from `procsplit`. G=1 behaviour is
  byte-identical to the previous flag-on path. Re-landed on top of pgw#826
  (the first landing, 79894b4, crash-looped a hardware-unsuitable pod forever
  and was reverted out of the 0.85.0 cut): a compute child's terminal boot
  verdict now ends the pod — T_BOOT_FATAL relays the typed HardwareUnsuitable
  report through the parent's credential and the parent exits 1, no respawn —
  and that machinery is exercised with the split always on. The boot-smoke
  hardware-report suite asserts the split-world contract (child hands the
  verdict to the parent; the parent's relay is what the report budget bounds).
- **pgw#783 follow-on — the control parent honors the pgw#639 SIGUSR2
  contract: dump and forward, never die.** tests_v2's zero-based boot gate
  caught it under the unconditional split: SIGUSR2 to the pod pid (now always
  the control parent) took default action and killed the worker. The parent
  installs the forward-to-children handler first, then faulthandler with
  `chain=True`, so one signal prints parent + children thread stacks to the
  pod log and kills nothing.

## 0.86.0 (2026-08-01) — no mint route could publish a cell: the regional arm gets its caller, the delegated mint child gets its slots, eager serving names its reason, and the zero-based suite becomes a CI gate

- **pgw#827 — a REGIONAL cell is adopted through the REGIONAL arm, so it can be
  PUBLISHED at all.** On a real L4 (0.85.0, sdxl 0.2.105, lane `w8a8-lora64`,
  recipe `aot-regional`, pod `o7y87kfunc3rmm`) the platform's first successful
  AOT mint — `aot_mint_phases phase=minted n_entries=72 total_s=354.45` — was
  then discarded at the mint's own self-adopt verification with `aot_adopt
  constants_constant_unresolved`: all 30 declared constants of every entry
  unresolvable. `models.provision.arm_aot` DETECTED the regional cell (pgw#825
  added that, to skip the lifted install) and handed it to `aot_serve.enable`
  anyway — the whole-graph arm, which builds ONE bind table per TARGET from
  `resident_constants(unet)`. A regional entry's FQNs are block-relative
  (`attn1.to_k.weight`); the denoiser carries them under their full path; and
  `resolve_constants` is a direct FQN lookup by design. Because
  `fleet_cells.adopt_delegated_mint` runs that same `arm_aot`, an unwired
  regional arm did not merely mean regional cells could not serve — it meant
  they could not be published. New `aot_regional.load_and_arm`/`enable` is the
  regional twin of `aot_serve.load_and_wrap`/`enable`: the same gates in the
  same order, differing only in that the bind table is built per block
  INSTANCE. It publishes the SAME format-2 pipeline marker, so `is_armed`,
  `execution_count`, `proven_since`, `set_guard_failure_callback` and `unwrap`
  need no regional variant — without which the executor's adoption proof
  (pgw#735) could never pass for a regional cell and the mint would publish and
  then be rolled back as unproven.
- **pgw#827 — the arm is asked BEFORE the mint spends.** New
  `models.provision.arm_route(mode)` is one registry, consulted by the arm when
  it arms and by `fleet_cells.mint_recipe` before the child is spawned. A cell
  whose mode this runtime has no arm for is refused by name rather than routed
  to whichever arm is the default, and a regional recipe with no wired arm
  declines `regional_arm_unwired` at `self_mint_started` instead of after
  354 s of L4.
- **pgw#827 — the regional lane gets the whole-graph lane's fail-soft
  contract.** `aot_regional.BlockShim` had no guard: an artifact fault was a
  failed REQUEST. It now serves eager on an ingress refusal (named, counted,
  still armed) and on any other artifact fault marks the instance failed,
  revokes scheduler-visible compiled proof, and serves eager for the rest of
  the process.
- **pgw#827 — the adapter fork routes by module STATE for a regional cell.**
  pgw#790 discriminates the two arms by ingress, which works because the lift
  wraps the DENOISER. A regional entry is exported one block deep from the
  block's own signature, which never carries the lifted pair, so both arms
  declare the same contract, both admit every call, and `EntryDispatch`
  correctly refuses `entry_ambiguous` on every forward — the cell arms, reports
  armed, and serves 100% eager. New `aot_regional.BlockDispatch` picks the arm
  from the denoiser's live adapter state (`lora_lifted.adapter_active`, the
  same fact `aot_serve.adapter_call_kwargs` already reads), and ingress then
  discriminates cfg/shape within the arm.
- **pgw#831 (gated here, fixed on its own train) — a folded constant bakes the
  PROTOTYPE block's weights into every other instance.**
  `eliminated_constants`' "routine compiler fusion, recorded, never fatal" is
  right for a whole-graph cell and FATAL for a regional one, which reuses one
  artifact across instances that do not share weights. Measured off-pod with
  `ff.bias` folded: instance 0 reproduces eager at 0.0, instance 1 is wrong by
  0.53, with nothing unbound and nothing refused. A regional entry that folded
  any `state_dict` constant is now REFUSED before packing. The remedy —
  `aot_inductor.use_runtime_constant_folding=True`, verified to keep them
  bindable — re-keys every cell and needs `_FOLDED_CONST_*` handling, so it
  rides pgw#831.
- **pgw#828 — the delegated mint child runs the endpoint's warm job with the
  SAME context the serving path builds.** On a real L4 the child loaded the
  pipeline in 16.45 s (pgw#816 holding) and then died at
  `ctx.slots["pipeline"]` with `KeyError: 'pipeline'`: it hand-rolled a
  `RequestContext` with no slots, no models and no root slot. Nothing was
  missing from the wire — warm-shape slot resolution needs only the spec, which
  the child builds itself; it simply never asked. New `warmup.warm_context`
  (plus `warmup.resolved_slots_kwargs` / `warmup.spec_root_slot`, moved out of
  `executor`) is now the ONE construction, called by the boot warm path, the
  in-process mint seed and the child; `executor._resolve_slots_kwargs` and
  `executor._spec_root_slot` are aliases of it. It lives in `warmup` so the
  child never imports the executor. The regression asserts the two contexts are
  the SAME construction, slot by slot — a child that resolved different slot
  defaults would trace different shapes and the parent's proof would miss.
- **pgw#824 — the fleet-wide silent-failure audit (SDK half): eager serving names its reason,
  a silent mint phase ticks, and two swallowed failures that were corrupting DECISIONS.**
  Ordered by Paul after the five-silent-blockers retrospective: five defects hid for weeks
  because failures were log-only, success events did not exist, refusals carried empty reasons,
  and eager-first masked every symptom. pgw#760 swept `except` handlers; this closes the three
  pattern classes that sweep structurally missed, plus the files it deferred and everything
  written since.

  **Eager serving is an EVENT, not a default.** `serving_mode`'s four fallback classes
  (`guard_miss`/`ingress_refused`/`healing`/`volatile`) all presuppose an ARMED cell, so the
  commonest eager case by far — nothing armed at all — reported `serving_mode=eager,
  fallback_reason=""`. That empty string could not distinguish a release that declares no
  compile target from a pod still minting from a pod that declined for cause, so "why is this
  fleet eager right now" had no query. One level down, the declines that DID emit shared one
  constant phase: `_fail_closed`'s nine distinct exits all sent
  `self_mint_skipped phase=mint_unavailable` with the cause in free text only — the th#1250
  lesson (kind-only coalescing erases the reason) repeating. Now: `ArmOutcome.eager_reason`
  carries the arming brain's token out of its own decision; the nine exits carry nine tokens
  (`no_family`, `no_cuda`, `no_toolchain`, `no_compile_target`, `delivered_cell_seeded`,
  `key_computation_failed`, `capture_conflict`, `multi_group_in_process`,
  `capture_arm_failed`); the in-process cell-QUARANTINE exit — the one eager exit that returned
  before `_fail_closed` and only `logger.error`'d, on a pod that then serves eager for the rest
  of its life — is typed (`cell_quarantined`); the delegated arm reports `mint_in_progress`,
  because eager with an END must never read the same as eager forever; `serving_mode.POSTURE_*`
  + `resolve(eager_posture=)` reports it per request, applying ONLY when the mode is already
  eager and never setting `served_eager_fallback` (nothing fell back — there was nothing to
  fall back FROM), so every existing compiled-vs-eager comparison keeps its meaning; and five
  `_install_compile_targets` omission branches that were `logger.warning`+`continue` type
  themselves. `fallback_reason` on the request row is now the SAME string as `phase` on the
  worker's activity event, so the question is one `GROUP BY` joining the two on a token.

  **A multi-entry AOT mint reports every entry.** `aot_mint.mint` was ONE opaque call spanning
  the family's whole declared graph-class set (sdxl declares 18), so `mint_child` framed
  `trace_graph` once and said nothing until `seal_publish` — a real export measured ~5 minutes
  of complete wire silence, with the pod's only liveness evidence being that its CPU was warm.
  That proves ALIVE, not PROGRESSING, and the distinction is the content of the
  no-magic-timeouts doctrine. `mint(on_progress=)` reports per class row BEFORE the row runs (a
  row that never returns is the one a reader most needs named);
  `EntryCompilePool.compile(on_entry=)` fires as each entry lands, covering the longest
  wire-silent stretch of a mint; `mint_child` frames both through the protocol that already
  exists, so the parent's `_on_frame` lands them on the same `self_mint_compile` activity with
  step/total. `mint_delegate` also finally passes `on_evidence` — `run_mint` has accepted it
  since pgw#784 and nobody ever did, so the child's measured progress (tree CPU + capture-dir
  growth) existed only to decide whether to KILL it, never to prove it was working.

  **A cell-discovery MISS says why every candidate lost, as counts.** `_candidates` dropped
  rows on `logger.debug`/bare `continue` and the `miss` event reported only "no matching cell
  among N checkpoint(s)" — so a family with 12 published cells that rejects all 12 read
  identically to a family with none, and those are different bugs with different owners. The
  rejections are counted by class and ride the miss detail.

  **Two swallowed failures that corrupt decisions, fixed at the decision.** (1) An emergency
  nf4 quant landing on ZERO modules was SELF-SUPPRESSING: the failure did `adaptive_rung = ""`,
  and the `if adaptive_rung:` stamp below it is the very mechanism that reports rung outcomes to
  placement — so the worst outcome the fit ladder can produce (serving full precision over the
  budgeted VRAM, on a host already too tight for stored precision) was the only one that
  reported nothing, while every sibling rung reported itself. Now `RUNG_NF4_UNLANDED`, routed
  through `SlotLoad.rung` -> `_record_adaptive_rung` -> ServePlan/FnDegraded like every other
  rung. (2) A failed residency eviction booked `tier=RAM, vram_bytes=0` while
  `_move_verified`'s own rollback had just put the object back on CUDA: the registry believed
  the entry held ZERO VRAM, `make_room` handed out headroom that does not exist, and the OOM
  landed on an unrelated `promote()` later with nothing tying it back. "Book the truth" now
  means the truth in both branches.

  Also: the block-window offload engagement (every forward now streams weights over PCIe from
  host RAM — the biggest per-request latency change the loader can make) is typed, the sibling
  the pgw#760 `apply_fp8_storage` fix missed; and an unparseable `destination_repo` confesses
  once per context instead of silently redirecting a job's outputs off the repo-CAS path.

  The endpoint half is **ie#589** — the sweep's root finding was that `activity.emit_event` has
  ZERO adopters across all 27 endpoint families.

  **Reconciled with pgw#825, into ONE mechanism.** 0.85.0's cut merged this branch textually
  clean and got 16 failed / 6 errored: pgw#825 had split `mint()` into
  `mint` / `_attach_partial_phases` / `_mint_cell`, so this branch's `on_progress` parameter
  landed on one function and its `_beat` helper inside another —
  `NameError: name 'on_progress' is not defined`. Both issues had arrived at the same question
  ("where is this mint?") from opposite ends: pgw#825 accumulates the partial state so an
  ABORTED mint reports the seconds it spent, pgw#824 pushes those positions out LIVE. Rather
  than thread a second parameter past the first mechanism, both now ride one
  `aot_mint.MintProgress` — it owns `timings`/`minted`/`width`/`t_mint` (pgw#825) and
  `beat()` (pgw#824), and `beat` RECORDS the position before reporting it, so the two halves
  cannot disagree. The dividend is pgw#825's one remaining blind spot closed: its per-entry
  rows name the entries that FINISHED, and an aborted table now also carries `at` — the entry
  the mint died ON, which is the row a reader is actually asking about.
- **pgw#808 — the zero-based suite is a CI gate.** `tests_v2/` shipped in 0.85.0 and `ci.yml`
  ran `tests/` only, so its coverage was imaginary: nothing would have reddened if a v2
  scenario broke. CI (and `task test`) now run both directories as two independent steps.
  Both, not the flip: `tests_v2/` has 2 of its 14 planned suites, and nine behavior domains
  have zero v2 coverage today, so `tests/` is still the net. The flip deletes the v1 step.

  Wiring it found the drift on the first run, which is the argument in one line: pgw#810 (the
  unknown-function refusal that sent an empty `safe_message`) was FIXED on chaos, and its
  `xfail(strict=True)` — written to "fail loudly the moment it starts passing" — was never
  deleted, because nothing ran the file it lived in. A strict xfail that XPASSes is a FAILURE,
  so `tests_v2/` was already red and nobody could tell. Marker deleted; the refusal now rides
  the names-its-cause loop with the rest of the matrix, and pgw#810 is closed.
## 0.85.0 (2026-07-31) — a regional cell's LoRA branch pair is BINDABLE: the ONE bind template, the mismatch refused before the compile is paid for, an aborted mint that reports where it spent, and the process split as the only execution model

- **pgw#825 — a regional cell's LoRA branch pair is BINDABLE, the mismatch is
  refused BEFORE the compile is paid for, and an aborted mint still reports
  where it spent.** On a real L4 (0.84.0, sdxl 0.2.104, lane `w8a8-lora64`,
  recipe `aot-regional`) the mint compiled and was refused at 351.73 s, per
  entry and AFTER that entry's compile: `20 declared state_dict constant(s)
  are absent from the resident module's state_dict, e.g.
  ['attn1.to_q.lora_a', ...]`. Mechanism, reproduced off-pod with a real
  `torch.export` + AOTI pack: `w8a8_lora.alloc_branch_buffers` registers
  `lora_a`/`lora_b` **non-persistent** (a checkpoint must not carry a zeroed
  adapter), `module.state_dict()` omits non-persistent buffers, and
  `torch.export` still lifts them as BUFFER inputs that AOTInductor declares
  `ConstantType::Buffer` under their real FQN — i.e. `source=state_dict`. The
  gate was right and the bind TEMPLATE was wrong, on the mint side and at both
  arm sites. New `aot_serve.resident_constants` is the ONE definition of what a
  `state_dict`-sourced constant may bind to (parameters + ALL buffers), used by
  the mint's bindability gate, `aot_serve`'s whole-graph arm and
  `aot_regional.arm_blocks`. Input lifting is NOT the fix and was never the
  defect: it wraps the DENOISER's forward, and a regional entry is exported one
  block deep — a block's branch pair stays module-resident and binds per
  instance by reference (`user_managed=True`), which gives regional the same
  property lifting gives the family graph (an adapter swap is an in-place
  buffer write, never a rebind and never a recompile). Regional block entries
  therefore no longer inherit the family's `lifted_inputs`, and
  `models.provision.arm_aot` refuses to install the lifted forward under a
  regional cell, whose leaves it would reassign out from under the arm.
- **pgw#825 — the same question, asked before the compile.** New
  `aot_package.unbindable_program_constants` runs the bindability check on the
  ExportedProgram, in both the whole-graph and regional export paths, so a
  declaration/module mismatch costs milliseconds and a typed refusal instead of
  4-6 minutes of GPU per entry. The packed gate stays: it reads the artifact's
  own generated wrapper, which is the only proof of what shipped.
- **pgw#825 — `aot_mint_phases` reports on EVERY terminus.** An aborted mint
  emitted only `total_s`, so `compile_s`/`export_s`/`n_entries` parsed to `-`
  and a run that paid for real compiles produced no measurement. `aot_mint.mint`
  now attaches the partial phase table to whatever it fails with, the child
  carries it into a `refused`/`failed` report, and the parent emits the table's
  roll-up under `phase=aborted` (never `minted`) plus the per-entry
  `entry:<name>` rows.

- **th#1355 (worker half) — a published cell states its own identity and
  cost.** The publish payload now carries `identity_axes` (the axes the cell
  was keyed on) and `mint_duration_ms` (what the mint actually cost), so the
  hub's cell inventory records what a cell IS and what it took to make
  instead of inferring both. `CellPublisher.publish` grows a
  `mint_duration_ms` parameter and `fleet_cells.publish_self_mint` threads
  the mint's measured duration through to it.

- **pgw#823 (SDK half) — ask for a C++ compiler before spending 336s
  discovering there isn't one.** An AOT mint on an image with no `g++` (or no
  CUDA crt/nv headers) ran the full load-and-trace and only failed once
  inductor tried to build a kernel — 336 s of rented GPU for a question
  answerable at boot. `compile_cache` now probes the C++/CUDA toolchain and
  `fleet_cells.mint_recipe` declines by name before the pod is spent;
  `mint_child` refuses typed on the same predicate.

- **pgw#818 — the worker's fabric gate adopts the hub's full predicate.** The
  Hello-time demote read `interconnect` alone while the hub's grew a measured
  bandwidth floor, so in the band `nvlink AND peer_gbps < 200` a 2x2 pod refused
  half of every dispatch RETRYABLE forever and a 1x4 pod overstated capacity 4x.
  `delivered_topology` now demotes unless `sp_admits(interconnect, peer_gbps)` —
  `nvlink AND >= SP_MIN_PEER_GBPS (200.0)`, the hub's `topology.SPAdmits`
  verbatim — and a WEDGED fabric (`peer_access AND peer_gbps == 0.0`, the
  collective that hangs with no error) raises
  `topology_fabric_wedged_peer_access_zero_bandwidth` typed at boot for any
  multi-GPU topology, closing the race against the hub's quarantine drain.
  Deliberately still no HelloAck demote field: two independent gates over one
  measurement is the design; th#1285 interpretation 4's "agree by construction"
  holds only while the predicates match, which they now again do.

- **pgw#808 — `tests_v2` scaffold: the declarative endpoint catalog and the
  first two scenario suites.** A new `tests_v2/` tree describes endpoints as
  DATA (`catalog.py`) and drives boot and dispatch scenarios from that
  description rather than from hand-built fixtures per test. Additive only —
  no existing suite or shipped module changes.

## 0.84.0 (2026-07-31) — the AOT mint exports the module it DECLARED: one lifted arm for the export, a declaration/module check before a pod is rented, and the process split becomes an N-group AUTHORIZATION boundary (dark)

- **pgw#822 — the AOT mint exports the module it DECLARED: ONE lifted arm, plus a
  declaration/module check that runs BEFORE a pod is rented.** On a real L4 (0.82.0, sdxl
  0.2.100, lane `w8a8`/`lora64`) the delegated child loaded the pipeline and reached
  `trace_graph`, then refused: `declared input(s) ['lora_a', 'lora_b'] are not parameters of
  'forward' on UNet2DConditionModel`. The declaration was right — pgw#725 option 2 lifts the
  rank-bucket adapter to graph INPUTS so it can never be baked — and the OBJECT was wrong:
  `mint_child` armed the DYNAMO lane's preparation (`compile_cache.apply_lora_lane`, branch
  containers only) and then ran the AOT recipe, so `torch.export` got the bare denoiser. Three
  copies of "prepare a bucket-bearing pipeline for export" existed and only the SERVING one
  (`models.provision.arm_aot`) was right; the operator path (`aot_inputs.compose`) had the
  mirror defect, installing the lifted forward over containers that were never allocated. Now
  there is one — `models.lora_lifted.arm_lifted_lora_lanes` (containers, then lifted signature,
  in the serving arm's order; idempotent; a no-op at bucket 0) — and `aot_mint.mint` owns
  calling it, because that function already owned the other end of pgw#790's fork
  (`_disarm_branches`). Per-class preparation is preserved and proven: `adapter=true` exports
  the lifted forward, `adapter=false` the plain module after one disarm. `_export_entry` now
  refuses an unarmed lifted class by naming the PREPARATION rather than the declaration. New
  `aot_mint.declaration_module_gaps` compares every declared class's input names against its
  target module's own `forward` signature on the PARENT, and `fleet_cells.mint_recipe` declines
  by name (`self_mint_skipped`, `phase=declaration_module_mismatch`) — per class, admitting the
  lifted pair on a lift-CAPABLE target so it predicts the mint, and abstaining rather than
  declining when it cannot read the composed pipeline. Serving is never affected: a decline
  falls back to the dynamo recipe like every other named decline.

- **pgw#783 — the process split generalises to N execution groups: the parent supervises
  one compute child PER GPU (flag `GEN_WORKER_PROCESS_SPLIT` still default-OFF).** pgw#782
  measured that one CPython process cannot multiplex N GPUs (four groups in one interpreter
  served 0.94x of serial at 21% per card; four PROCESSES one group each served **4.00x** at
  91-93%), so the child of pgw#763's split is the EXECUTION GROUP. `ParentControl` now holds
  one `_ChildSlot` per group; each child is a single-group worker scoped to its own cards
  (`CUDA_VISIBLE_DEVICES`), with its own CUDA context, inductor cache and mint. The parent
  routes each dispatch to the group owning the hub-picked rank-0 device (rewriting `gpu_index`
  to the child-local 0; a mis-dispatch is refused, never floored), and aggregates N children
  into ONE worker the hub sees: `available_functions`/residency UNION, `state_delta` merged,
  `activity_update`/`fn_unavailable`/`fn_degraded` reconciled to a single worker-level truth,
  one parent-originated beat, per-group liveness/watchdog so a single child's death is
  attributed to ITS request and respawns ITS group while siblings serve. Per-group resource
  correctness: the CPU intra-op divisor is `groups x host_siblings()`, the host-RAM guard/floor
  divide the cgroup by the sibling count, each child gets its own inductor/triton dir, a
  cross-process `flock` dedups a CAS fetch across children, and `PR_SET_PDEATHSIG` makes every
  child die with the parent so a crashed group cannot strand VRAM. `worker_session_id` is now
  parent-minted (survives child respawns; a latent defect even at G=1). **At G=1 the whole thing
  is byte-identical to the pgw#763 single-child parent** — every worker-level aggregation point
  takes an explicit `groups==1` fast path, and the pgw#763 procsplit integration suite passes
  unchanged. The 4x is proven in the two/four-process arms of pgw#782; a live 4xGPU
  demonstration of the split driving them is the remaining acceptance. Merges dark; the
  default-on decision is deferred.

- **pgw#763 driver 3 — the process split becomes an AUTHORIZATION boundary, not just a
  fault boundary (flag still default-OFF).** Stages 1-4 built the parent/child seam for
  resilience; the premise of the whole design is that tenant endpoint code is imported into
  the worker process, so the JWT, capability tokens, hardware/canary measurement and billing
  were all forgeable by that code. Six deltas move each of them to the parent side:

  - the split switch itself is platform-reserved and pod-launch-injected (hub half:
    tensorhub `e4016fe9`) — a boundary the contained code can decline is not a boundary;
  - the compute child holds **no worker JWT**: `T_TOKEN` is deleted AND `WORKER_JWT` is
    stripped from the child's environment, so identity-bearing hub calls become narrow,
    allowlisted, audited **parent actions** (`procsplit/actions.py`) with the base URL chosen
    by the parent. Identity travels as a claim (`WORKER_ID`, new `WORKER_RELEASE_ID`), never
    as a credential;
  - `Hello.resources` — the hardware and the boot canary, i.e. the fleet-wide verdict keys —
    are measured by the PARENT in a subprocess that imports no endpoint module, and stamped
    onto every relayed Hello;
  - billable quantities the parent can observe (wall clock, dispatch concurrency, child RSS)
    are attested by the parent; the ones it cannot see without pulling the data plane through
    its interpreter stay child-reported and are NAMED in a durable record;
  - the parent DECIDES on each per-job capability token — forward, or withhold and refuse the
    job — instead of relaying whatever arrives;
  - C2PA signing is a `sign(hash)` ask over the seam; neither the key nor the credential that
    reaches the oracle exists in the child.

  Nothing changes with `GEN_WORKER_PROCESS_SPLIT` unset, which is every pod today.

- **pgw#820 — an SP follower now dies with rank 0, ABORT included.** The split's
  `PR_SET_PDEATHSIG` covers parent -> compute child and does not cascade; a group's D-1
  follower ranks are `daemon=True` grandchildren whose only exit path was a clean
  interpreter exit — and the measured rank-0 death is `rc=-6` (NCCL abort), where atexit
  never runs. The orphans would hold a full weight replica on cards 1..D-1 for their own
  300 s collective timeout while the parent respawned the group onto those same cards in
  ~1 s: a crash loop seeded by its own orphans, masked today only because rank 0 IS the
  worker process and the container restart reaps everything. Every follower now sets
  `PR_SET_PDEATHSIG(SIGKILL)` in its own bootstrap (`_follower_main`), with the spawn-race
  re-parent check, so a rank-0 death of ANY kind frees the followers' VRAM immediately —
  also the correct answer for a SIGKILLed worker, split or no split. Proven by a real
  spawn-path reap test that goes red when the bootstrap is reverted.

- **pgw#780 items 1/2/4 + pgw#776/DPA-6 — the per-group bookkeeping pgw#748 CLAIMED is now
  wired.** Four "wire it or delete the claim" gaps between the DP commits' claims and what
  ran: (1) `PinnedPool.set_group_count` was called nowhere in src/ — the cap/G pinned-host
  fair share was dead code and group 0 could claim the whole 50% of host RAM on a G=4
  degraded pod; `bind_topology` wires it now. (2) Registries were created lazily on first
  dispatch, so the boot disk re-track (a union over `all_residencies()`) was a no-op for
  groups 1..G-1; `bind_topology` creates every group's registry eagerly. (3=item 4) The
  "dedicated H2D copy stream" was a device-0 singleton — a promote onto cuda:3 queued its
  copies under card 0's stream context and synchronized card 0, silently losing the
  overlap-with-compute property for every group but 0; one stream per device now, keyed by
  the promote's target. (4=DPA-6) `residency_snapshot()` read the CURRENT group's registry,
  which on the event-loop thread is always group 0 — at G=4 the hub saw 1/4 of the resident
  set and every cache-aware decision (victims, keep-warm, warm routing) ran against a
  quarter of the truth; it unions across groups now, one row per ref at its best tier,
  vram summed. Items 3 (per-group preloader/boot warm) and 5 (per-(function,group)
  disables) are recorded as dissolved by the pgw#783 split — each child IS one group — and
  stay open only for the in-process multi-group interim (see the tracker).

- **pgw#777/DPA-8 — the in-process capture is REFUSED at G>1, never arbitrated.**
  `capture_env` moves the process-global `TORCHINDUCTOR_CACHE_DIR` and clears inductor's
  latch for the whole interpreter; under G in-process groups that lands mid-compile or
  mid-serve on G-1 sibling cards (a mint published from bytes another group produced, or a
  sibling's seeded FX entries going invisible). The delegated mint (pgw#784) dissolves this
  — its capture lives in the mint child's own process — so the residual is exactly the
  typed refusal the dpharden ruling called for: when delegation is refused on a multi-group
  in-process worker, the mint declines through the ordinary miss policy (plain lanes serve
  eager, mandatory lanes keep their typed refusal) instead of pretending a process-global
  control plane is per-group. G==1 — every pod today — keeps the exact in-process fallback
  path. The mint-once-adopt-N story belongs to the pgw#783 split world.

- **pgw#763 follow-on (found by this release's CI gate) — the parent's four report throttles
  swallowed their FIRST report on every freshly-booted pod.** `last_*_report_at` was seeded to
  `0.0` and compared against `time.monotonic()`, which on Linux is time since BOOT — so the
  sentinel did not mean "never reported", it meant "reported at boot", and every throttle stayed
  closed for the host's first 300 seconds. That silently dropped the first crash-loop report,
  action refusal, billing-attestation divergence and capability withholding of a pod's life,
  which is precisely the window in which a child that cannot boot or is probing the action
  allowlist most needs to reach the hub — the pgw#763 security deltas are only a boundary if the
  refusal is BANKED, and this made the first one log-only. Now `_NEVER_REPORTED = float("-inf")`,
  so the first report of each class always goes out and the interval throttles only the ones
  after it. **This box could never have caught it** (uptime in days); a CI runner boots minutes
  before the suite, which is why three procsplit rows went red on the release tree and green
  locally. A sweep test pins all four fields and fails on any new `_report_at = 0.0`.

- **pgw#821 (th#1303 empty-guard class) — component sharing was silently OFF fleet-wide on
  every manifest-v2 snapshot.** `component_digests` read `f.blake3`, which is EMPTY on every
  v2 entry, so every file of a v2 snapshot was skipped and gw#479 component sharing never
  engaged — the fail-CLOSED half of the empty-guard class, invisible because a skipped file
  looks exactly like a file with nothing to share. Now a dual-read (the tagged `digest`
  first, then the legacy `blake3` mirror) until th#1303 S1 retires the v1 arm. Extracted
  from the blake3-removal branch so the live defect does not wait on the v1 write-freeze
  gate.

## 0.83.0 (2026-07-31) — REGIONAL CELLS: the minutes-scale mint. A cell's entries become BLOCK CLASSES, and flux2 can mint at all again

> ### ⚠ TWO THINGS TO READ BEFORE UPGRADING
>
> **1. The contract-facts bump v2 -> v3 RE-KEYS EVERY PUBLISHED `aot-inductor` CELL.**
> `shell_digest` is mandatory from this release, and a mandatory fact is part of the
> key — so every cell published under v2 is unreachable to a 0.83.0 worker and every
> family re-mints once. This is deliberate and there is no smaller way to do it: a v2
> key describes the PARTS and does not bind the ASSEMBLY, which is exactly the
> cache-poisoning class regional would otherwise introduce. **The cost is being paid
> now on purpose**: the fleet holds ~zero published `aot-inductor` cells today (leg 4
> has never completed a mint — `aot_mint_phases` has been empty on both stacks since
> th#1322), so the re-key is free at this instant and would not be at any later one.
> Recorded here as a decision, not left to be discovered by a re-minting fleet.
>
> **2. sdxl's regional opt-in is deliberately NOT in this train, and the ORDER is
> forced.** `Compile(regional=True)` is honoured by the export lane only from 0.83.0;
> on any earlier SDK the same declaration reaches `fleet_cells.delegation_refusal`,
> which declines `aot_regional_targets`, and **sdxl stops AOT-minting altogether**. So
> the flag and its `gen-worker==0.79.0 -> ==0.83.0` pin bump are ONE commit on
> inference-endpoints branch `agent/817-sdxl-regional-optin` (`75712ba`), which merges
> only AFTER this release is on PyPI. The sequence is: publish 0.83.0 -> merge the
> opt-in with the pin bump -> only then can a pod mint a regional cell. Until the opt-in
> lands a pod mints whole-graph, correctly.

- **pgw#817 — REGIONAL CELLS: a cell whose entries are BLOCK CLASSES, so the sdxl
  w8a8 mint is 19.4 s of graph compile instead of 274.7 s.** A DiT — and sdxl's
  UNet — is one block repeated N times, and a whole-graph mint traces, lowers,
  codegens and g++-compiles all N. pgw#812 measured compiling ONE block per class
  and reusing it on our own path: **14.2x on the real sdxl w8a8 mint** with serve
  parity **+0.24%**, artifact 4.7 MB instead of 18.2 MB, and numerics CLEANER than
  whole-graph on fp8 (cos 0.989-0.993 against the pgw#814 whole-graph degradation).
  bf16 is 7.7x, and there regional serves 5.7% FASTER than the whole-graph
  artifact. This is the adoption.

  **It is still ONE `.pt2` and the entry grammar is unchanged** —
  `unet/block=BasicTransformerBlock#0,cfg=true/B=2`. What inverts is the entry
  AXIS: entries enumerate block classes of a target instead of shape coordinates
  of its whole forward. No new artifact class, no hub change, no `cell_store`
  change. The shell stays EAGER (exporting it with the blocks elided is not
  expressible in `torch.export` today), so the compiled fraction of the model
  equals the repeated-block fraction — stated here rather than discovered later.

  **`shell_digest` is now a mandatory contract fact (v2 -> v3), and it is the
  load-bearing part.** Regionally `combined_graph_hash` describes only the PARTS,
  so two models with identical blocks and a different shell — a different
  `num_layers`, a different rope construction, a diffusers minor that rewrites the
  outer forward — would key identically while serving different math. Without it
  regional trades compile time for a cache-poisoning class we do not have today.
  This re-keys every published `aot-inductor` cell; correct and expected, since a
  v2 key does not bind the assembly and there is no way to add the binding without
  moving the key. `cell_identity`'s `"mode": ""` hardcode goes with it, along with
  its comment that "regional is a dynamo partitioning strategy with no export
  counterpart" — falsified by measurement.

  **Binding is now BY REFERENCE, and it had to be.** `user_managed` appeared
  nowhere in the SDK: `ArtifactRunner.bind` copied every constant. Whole-graph
  that is a one-off duplicate; regionally it is N copies of the block weights in
  VRAM — for flux2, a second whole model. `bind(..., user_managed=True)` binds by
  reference (the resident pipeline keeps the tensors alive by construction); the
  whole-graph call shape is byte-identical to what pgw#721/#723 measured on a pod,
  and a torch whose `load_constants` lacks the parameter is a NAMED refusal rather
  than a silent copy that would OOM the card N blocks later. Arming is
  per-INSTANCE and **all-or-nothing per target** — a model with 24 of 25 blocks
  armed is a silently half-eager model — and the unbound-call gate runs before the
  FIRST call of EVERY instance, because that segfault surface multiplies by N.

  **A cell that degrades its output now REFUSES to arm, typed.** pgw#800's verdict
  ladder is lifted into a shared `numerics_ladder` primitive — the rungs, the
  norm-weighted aggregate rule (never a per-row median: a few destroyed high-norm
  outputs must not hide behind many intact ones), the evidence formatting and the
  fail-closed gate shape — with `adapter_fidelity` as one caller. The
  output-comparison population gets its own O(n) evaluator and its own DERIVED
  band, because pgw#814 says in as many words that the adapter floors are
  calibrated for adapter deltas and must not be inherited: floor **0.98** =
  sqrt(0.9890 x 0.9730), worst accepted being flux2 w8a8 REGIONAL vs eager and
  best refused being the flux2 w8a8 whole-graph artifact pgw#814 ruled
  unadoptable; warn **0.999**, since everything anyone has called healthy measures
  0.9998+. A magnitude bound (**0.95**, symmetric in the log) is new and not
  decoration — cosine is scale-invariant, so an artifact reproducing eager's
  direction exactly at 0.9x the magnitude scores a PERFECT cosine while serving a
  systematically dimmer image. Families declare their own tolerance
  (`Compile(numerics_floor=, numerics_warn=)`). New `cell_numerics` activity kind,
  `phase=refused|degraded|armed`.

  **`Compile(regional=True)` + `dynamic=(...)` is admitted**, and the old refusal's
  content survives where it is still true: the DYNAMO regional branch calls
  `compile_repeated_blocks(dynamic=None)` and cannot honour the marks, so it
  declines by name and the target takes the whole-forward branch, which does mark.
  The export lane implements it directly — measured FREE on a conv-free region
  (+0.2% bf16 / 0.0% w8a8, against pgw#730's +7.2% for the same axis on sdxl's
  conv lane).

  **`mint_recipe` gains `aot-regional`**, selected from the family's own export
  declaration — per-family, never a fleet default, because on a small-table DiT
  regional is a 2x that costs a serve-path change while pgw#811 buys a comparable
  win with none. The blanket `regional_targets` delegation refusal is discharged
  and deleted: regional is the shape that most wants delegating, being the one
  that finishes in minutes. pgw#809's pool is RE-PRICED rather than assumed —
  regional moves the entry count UP (one per plan x block class; sdxl's 18 become
  36) and the per-entry DEVICE ask DOWN by the measured block fraction, and VRAM
  is the bound that actually binds K, so multiplying the two levers would have
  sized the pool for a whole-model child regional never runs.

  Two defects found while building, both of which would have been silent: the
  delegation tail chose its reporting kind with a string-literal `recipe == "aot"`
  and would have sent every regional mint's phase table down `jit_compile` — the
  one channel the minutes-scale claim is measured on — and a block entry's
  bindability template was the target rather than the block, which refused every
  correct regional mint by name and only a real-path mint could catch.

  A third was caught by the release gate's full suite, which the lane's 351-test
  regression ring did not cover: `test_regional_dynamic_refusal_pgw746.py` still
  asserted the DECLARATION-side refusal D4 relocated, so the tree was two tests
  red. Relocated rather than deleted — the file now pins what pgw#746 was right
  about (the dynamo branch cannot honour the marks, so it declines BY NAME and
  the target falls through to the whole-forward branch, which does mark; a
  decline that read as a skip would be an uncompiled target) and drops the two
  premises pgw#812 measured away. Red-verified against the v0.82.0 source: 4 of
  the 8 fail there.

- **pgw#812 D1 + D2 — the two defects that make flux2 unmintable, and neither is
  about regional compilation.** D1: `dynamic_shapes_spec` minted one torch symbol per
  (input, axis), so a declared `Dim` with several carriers became several INDEPENDENT
  symbols and strict export refused the declaration —
  `Constraints violated (img_ids_1)!`. flux2 binds `T_img` to BOTH `hidden_states[1]`
  and `img_ids[1]` deliberately, so the edit lane cannot let `img_ids` specialize and
  silently pin the artifact to generate; the most careful declaration in the fleet was
  the one that could not mint, and ie#571 recorded it "READY — no open mint blockers".
  `DynamicDim` now carries the declared dim NAME and every carrier of one dim shares
  one symbol; rows with no declared name keep a symbol each, which the hand-registered
  builder path requires (latent H and W are two independent axes of one input).

  D2: the ie#566 G3 range gate then refused the mint on
  `Eq(Mod(3072*s + 1572864, 48*s + 24576), 0)` — which is `Mod(64*X, X) == 0`,
  identically true, pinning nothing. The gate matched "an Eq guard mentioning a
  declared symbol" without asking whether the guard was satisfiable-for-all, so a
  vacuous divisibility fact read as a specialization. It now admits a guard sympy can
  PROVE is a tautology and keeps refusing everything else, so it still fails closed on
  a guard it cannot reduce.

## 0.82.0 (2026-07-31) — the delegated mint child loads the composition the parent SERVES: the `phase=load` crash that closed BOTH mint routes on 0.81.0 is gone, and a crash the child classified is no longer retried

> ### ⚠ 0.82.0 IS THE FIRST SDK ON WHICH A DELEGATED MINT CAN GET PAST `child:load`
>
> 0.81.0 made the delegated route reachable (pgw#813) and the first `aot_mint_phases`
> rows in platform history were written on it — then the child died at `phase=load` in
> ~8.5 s, twice, on a tree the serving process in the same pod was loading and answering
> requests from. The dynamo/JIT mint is delegated on 0.81.0 too and died identically in
> 13 s, so that release has NO working mint route. This one fixes the boundary both
> routes cross: a directory path does not describe a composition.

- **pgw#816 — the delegated mint child could not load the pipeline the serving process
  was serving, so the first AOT mint in platform history crashed at `phase=load`.** On
  the first production run of the delegated route (0.81.0, real L4, sdxl `w8a8`,
  `cyberrealistic-xl:fp8-linearonly-review`) the child died twice at ~8.5 s with
  `OSError: Error no file named config.json found in directory
  …/cas/snapshots/sha256:32fa2ba6…__x76b2ae62d32f`, while the serving process in the same
  pod loaded that exact directory and answered requests from it throughout.

  The `__x76b2ae62d32f` suffix is the diagnosis, and our own `snapshot_dir_key` wrote it:
  `__x` marks a tree materialized with an overridden component EXCLUDED from the fetch
  (th#1330 B2), and `sha1("vae")[:12] == 76b2ae62d32f`. That tree has no `vae/` by
  construction — it is loadable only TOGETHER with the override tree it was narrowed for.
  The parent had resolved that override and injected it through
  `from_pretrained(components=…)`; the child was handed `Dict[slot, path]` and nothing
  else, so it re-composed a pipeline with a component missing, and diffusers reported it
  as a missing `config.json` at the tree's ROOT — naming neither the component nor the
  cause. The boundary was the bug: **a directory path does not describe a composition.**

  `MintRequest`/`MintTask`/`_BackgroundMint` now carry the parent's resolved
  `component_paths` (slot -> component -> local tree), and `cli.run.run_setup` grew the
  same `components=` seam the executor uses, so the child loads through the identical
  `load_component_override` -> `load_slot` -> `from_pretrained` path. Nothing is
  re-fetched (minimum-fetch is intact) and the B2 exclusion stands. New
  `mint_child.assert_composable` refuses a request whose slot is override-narrowed but
  carries no override — by name, before discovery, the toolchain probe or a single weight
  read. This unblocks BOTH mint routes: on 0.81.0 the dynamo/JIT mint is delegated too and
  died in the same child load.

- **pgw#754/pgw#811 follow-on — the host-ISA clamp was THREAD-LOCAL, so every host
  compile off the boot thread was built `-march=native`.** Found by this release's CI
  gate and root-caused to torch's own config semantics: `inductor_config.cpp.march = x`
  writes the `user_override` layer, which torch documents as thread-local (it is a
  `ContextVar`). `env_seal.establish` imposes on the BOOT thread, so the clamp reached
  nothing else — and two threads that host-compile are squarely on the production path:
  `hot_swap`'s process-global background shape-warm/heal worker, and pgw#811's K-way
  `run_impl` splitter pool (which is exactly what a serving-pod AOT mint drives).

  Before pgw#811 that silently produced unclamped, unportable objects — the pgw#754
  SIGILL class, on the background-warm path specifically. Since pgw#811's
  `assert_command_is_clamped` landed in 0.81.0 it is louder and worse: those compiles
  RAISE `HostIsaError`, so on 0.81.0 every background shape-warm compile fails, and per
  pgw#680's doctrine two failed heals mark the signature permanently `volatile` (eager
  forever). `impose()` now writes the process-wide `default` layer as well, and — the
  part that was actually missing — **verifies the read-back on a FOREIGN thread**, which
  is the only place the defect was ever visible. A torch internals change that puts the
  process-wide layer out of reach now refuses loudly at boot instead of silently
  reverting to per-thread clamping.

- **pgw#816 — a crash the child CLASSIFIED is not retried.** Those two identical 8.5 s
  attempts bought nothing. A `status="failed"` report means the child caught its own
  exception and named it, from the same request file against the same on-disk inputs, so
  attempt 2 re-runs it exactly; `MintOutcome.retryable` now holds only for a death the
  child could NOT classify (no report: signal, OOM-killer, the parent's stall kill) and
  for `EXIT_RESOURCE`, which was always the case a retry exists for. The abort event says
  `deterministic` or `retryable` so the wire states why there was no second attempt.

## 0.81.0 (2026-07-31) — the w8a8 lane can mint again and a mint can no longer publish nothing silently: the two refusals that kept `aot_mint_phases` empty platform-wide are gone, every publish terminus is typed, and `run_impl` splits K ways for a 12.6x host compile

> ### ⚠ 0.81.0 IS THE FIRST SDK ON WHICH A `prefer_aot` POD CAN ACTUALLY REACH A MINT
>
> 0.80.0 wired the serving-pod mint (pgw#805) but nothing could travel it: the plain lane
> is held on dynamo by pgw#730, and the w8a8 lane refused `aot_requires_delegation` for
> reasons that were false on the pod — while the delegated route it named had never once
> run, because `_eager_first_eligible` demanded a hot-swap router that a delegated pending
> never has. Both are fixed here (pgw#813). pgw#815 then makes the far end honest: a mint
> that publishes nothing now says so. An EMPTY `aot_mint_phases` on a 0.81.0 w8a8 pod is a
> real finding; on anything earlier it only meant the SDK could not get there.

- **pgw#813 — the w8a8 lane can mint AOT again: "executes quantized activations"
  stops meaning "cannot serve eager".** Measured on a real 0.80.0 L4: the plain lane
  declined `aot_lane_regressed` (correct, #730's hold) and the w8a8 lane declined
  `aot_requires_delegation` naming two causes that were both FALSE on that pod — no
  env was set. The operative refusal was `fleet_cells.delegatable` reading
  `mandatory_serving(pipe)` as a serveability answer. It is not one:
  `_Fp8ScaledLinear.forward` / `_W4A4Linear.forward` are complete `torch._scaled_mm`
  eager forwards, the fleet's cold-boot ladder measures w8a8 eager serving all day,
  and pgw#672/#673 already retired the "mandatory lanes raise instead of degrade"
  posture inside `_guard`. With the plain lane held, that left NO lane on which a
  serving pod could mint an AOT cell — the reason `aot_mint_phases` has zero rows
  platform-wide. New `compile_cache.eager_tier_available` answers the honest question
  ("can this object answer a forward with nothing armed?") and is false only when an
  AOTI export or TRT engine has REPLACED the callable; `mandatory_serving` keeps its
  real job (router fail-closed: the compiled tier is the intended production tier).

  A second, independent blocker went with it: `_eager_first_eligible` demanded a
  hot-swap ROUTER on every pending pipe, and a DELEGATED pending never has one
  because nothing is armed on its pipe by construction — so every delegated mint
  failed the predicate and was discarded, and pgw#784's out-of-process route could
  not run on ANY lane. The delegated arm now asks for an eager tier, not a router;
  the in-process arm is unchanged. The stamp-based `_mandatory_lane_of_bound`
  early-out (a model ref's `#fp8-w8a8` STORAGE flavor read as serveability — the
  same proxy pgw#677's reopen removed one layer down) is gone.

  Every delegation refusal is now typed and named on the wire from its true cause:
  `aot_mint_forced_in_process`, `aot_eager_first_disabled`, `aot_regional_targets`,
  `aot_no_eager_tier` — instead of one phase carrying a hand-written either/or.

- **pgw#815 — a self-mint can no longer reach `finalize completed` having published
  nothing.** A real 24m22s L4 mint walked `seal_publish -> finalize completed` and
  left zero cells, zero receipts, no local arm, no `self_mint_publish`, no abort and
  no error. Three structural silences made that indistinguishable from success and
  all three are closed: (1) NO success event existed at any publish terminus, so a
  completed publish and a publish thread killed mid-upload when the pod retired were
  the same observation — `self_mint_publish` now fires at `sealed`, `started` and
  `published`, carrying the cell key and the byte count, and in-flight publishes are
  an observable fact (`fleet_cells.publishes_in_flight`); (2) `publish_self_mint` and
  `withhold_self_mint_publish` both returned BARE when nothing was packed — both now
  emit `self_mint_publish_withheld phase=nothing_to_publish`; (3) the executor's
  whole publish gate lived inside `if proves_inductor or proves_exported:`, so a boot
  that answered "nothing proves by FX or export" walked past every terminus in
  silence — `_assert_mint_termini` now runs OUTSIDE that block and on every
  background-driver exit, confessing `self_mint_abort phase=no_terminus` and
  discarding the phantom capture so the next pod re-mints. A delegated child that
  produces no adoptable cell while a sibling succeeds is resolved too, instead of
  being dropped by a bare `continue`.

- **pgw#811 — `run_impl` is split across K translation units, the largest measured
  compile-speed win on the board.** Two independent compiler profiles of the real
  banked SDXL w8a8 wrapper agree that parse is 3-5% of the compile and that
  `AOTInductorModel::run_impl` ALONE is 68% of it, because the cost is superlinear
  (measured n^1.57) in ONE function's size: 10,032 declarations and 4,231 dispatch
  calls in a single body, burning in stack-slot conflict colouring and CFG-wide block
  placement. No flag fixes that (best -19%, most NEGATIVE) and no bigger machine
  fixes it (32-core i9: 140 s vs a 4-vCPU pod's 180.6 s — one process, one core,
  either way). Measured here on the real TU with torch's own flags and the production
  march clamp: **120.0 s -> 38.7 s of total CPU (3.1x, so it wins even serially) and
  121.8 s -> 9.7 s wall at K=8 (12.6x)**, with peak RSS 2.02 GB -> 0.63 GB. The
  4-vCPU pod emulation is 16.1 s.

  It is a continuation CHAIN, not K calls from `run_impl`: each part ends by calling
  the next with the live set by reference, so **no declaration is rewritten and every
  statement is byte-identical to what inductor emitted**, and every local's lifetime
  and destruction order is unchanged. Each part is a member function of a generated
  `_pgw811_run_ctx` carrying `constants_`/`kernels_`/`device_idx_`/`cubin_dir_` under
  their original names, which is what buys zero body rewriting. Real liveness, not
  the research prototype's fresh defaults: on the real TU only 386 of 8,590 compute
  declarations (4.5%) cross a K=8 boundary; the rest stay part-local, and the ones
  that are pure re-derivations (`constants_->at(n)` bindings, const `int_array_N`)
  are copied rather than threaded.

  FAIL-CLOSED, as v1 is, and by the same architecture: a self-contained mechanical
  inverse reads ONLY the split output and must reproduce the input byte for byte.
  It consumes no side table, so the transform cannot fool it — RED-verified against
  seven corruption classes (statement deleted / reordered / inserted / argument
  changed / disguised as generated / disguised as a re-derivation / re-derived
  binding altered). The last one found a real hole: re-derivations were being dropped
  blindly, so a mutated `constants_->at(n)` survived; they now carry their own marker
  and must be found verbatim in the reconstruction they claim to copy. Equivalence is
  also proved by EXECUTION — a shape-faithful wrapper is compiled both ways, partial-
  linked exactly as the mint does it, run, and required to print the same bytes.
  Anything unrecognised, unbalanced or untypable declines; if a part will not build,
  the whole wrapper is recompiled unmodified. A slow mint beats a wrong artifact.

  No cell is re-keyed: all three digests (inductor config, env seal, toolchain) are
  asserted unmoved, and neither module is reachable from the static code closure. The
  split adds compiler INVOCATIONS, not a compiler or a flag. Kill switches:
  `GEN_WORKER_AOT_RUN_IMPL_SPLIT_OFF` for v2 alone, `GEN_WORKER_AOT_WRAPPER_SPLIT_OFF`
  for both. Concurrency comes from `GEN_WORKER_AOT_HOST_COMPILE_JOBS` so pgw#809's
  pool owns the pod budget across entries while this fans out within one.

- **pgw#754's march clamp is now asserted at the ARGV level, not just the config
  level.** `host_isa.impose()` verified `config.cpp.march` and nothing ever read the
  command line torch actually built — and pgw#793's research harness is a live
  example of a path that mints `-march=native` objects because it never booted
  through `env_seal.establish`. `host_isa.assert_command_is_clamped` now runs on
  torch's single host-compile funnel and RAISES: an unclamped object is not slower,
  it is unportable, and pgw#754 is a SIGILL-class defect. (The harness itself was
  fixed to boot through `env_seal.establish` and record `mint_march`.)

## 0.80.0 (2026-07-30) — a serving pod can finally MINT an AOT cell: a discovery miss starts a real out-of-process mint instead of nothing, every decline names itself, and the boot-span ladder runs on the real path

> ### ⚠ 0.80.0 IS THE FIRST SDK ON WHICH A `prefer_aot` POD CAN PRODUCE A CELL
>
> Every release up to and including 0.79.0 was a pure AOT *consumer*: discovery
> missed, nothing was minted, and the next pod missed identically — with no
> refusal on the wire. That wire is connected here. Two consequences for
> whoever runs the first proof:
>
> - **The lane matters.** pgw#730 holds the PLAIN lane on dynamo, so a
>   plain-lane release (e.g. `d9d9bf2691d0e1f89e23999d`, `sdxl` 0.2.93) now
>   correctly declines with `self_mint_skipped phase=aot_lane_regressed`
>   instead of minting. A serving-pod AOT mint proof needs the **w8a8** lane.
> - **`aot_mint_phases` was empty on both stacks since th#1322 shipped**, because
>   the child process that fills it holds no orchestrator session. The parent
>   now re-emits the child's phase table, so that column starts carrying rows
>   from this version — an empty column on 0.80.0 means no mint ran, which is
>   information the previous releases could not give.

- **th#1303 phase 3, producer class 2: the CONVERSION producer publishes v2.**
  `publish_flavors` — the surface every quantize / fuse / cast / distil / produce
  job in the conversion endpoint calls, and the highest-volume publisher after the
  mirror — still called `commit()`, so the corpus repoint had a second tap filling
  the blake3 CAS behind it. It now calls `HubClient.publish_v2`: chunked sha256,
  each object's digest signed into its presigned PUT so R2 refuses bytes that do
  not hash to the key. Sixteen of training-endpoints' seventeen publish entry
  points reach the hub through this one call, so they flip with it.

  Safe because every file is a real local file (`_flavor_files` walks the produced
  tree) — v2's guarantee is that the digest is PROVEN from bytes in hand, which is
  exactly why the mirror's by-reference BANK arm (`clone.py:556`) is deliberately
  NOT flipped and stays on `commit()` until the phase-2 backfill.

  No auto-select and no env knob: the protocol is named at the call site, so
  "which producers are on v2 today?" is answerable by reading the code.

- **th#1303/pgw#807 — cell receipts are ALGORITHM-AGNOSTIC, which is what the cell
  self-mint flip was actually blocked on.** Arming compares `receipt.artifact_blake3`
  inside a signed JWS that can never be edited, and the receipt is fetched by
  `?blake3=<hex>`. Publish a cell over v2 and the lookup answers "no receipt", every
  worker refuses to arm, and the fleet re-mints — silently, looking exactly like a
  cold cache. The canonical binding is now `artifact.digest`, always algorithm-tagged,
  and verification DISPATCHES on that tag: an untagged bare-hex digest is refused
  rather than read as some assumed algorithm, and a receipt binding no digest at all
  is refused rather than compared against nothing.

  Dual-read, deliberately: `crv` stays `cell-receipt-v1` (the deployed fleet refuses
  any other version outright, so a bump would make every hub-minted receipt
  unverifiable on first deploy), and a legacy bare-hex `blake3` claim still verifies —
  every cell the fleet holds today was published over v1. Both arms die at phase 4.
  `fleet_cells.py`'s self-mint stays on `commit()` with its two remaining gates named
  in code: the hub's v2 route mints no cell receipt at all (th#1340), and the serving
  fleet must be past the release carrying this reader.

- **pgw#809 — a cell's entries compile K-wide, out of process.** `aot_mint` exported
  and compiled a pgw#758 cell's graph classes one at a time; an sdxl cell is 18 entries
  at ~420 s, so a mint was ~2 h of which almost all was independent, embarrassingly
  parallel work. Export stays serial (one pipeline, one card, one branch-arm toggle),
  and the compiles now run in a bounded pool of `gen_worker.aot_compile_child`
  processes.

  **Processes, because threads are wrong here — measured, not assumed.** Four
  concurrent `aot_compile` calls in one process returned one usable result and three
  distinct internal failures (`CURRENT_PATCHER is None`, `KeyError: 'custom'` in
  `fx.traceback.annotate`, a fake-tensor crash): inductor's compile path keeps
  process-global mutable state. The exported program travels on disk, and that
  roundtrip is byte-exact — a compile after `torch.export.save`/`load` produces a
  `wrapper.cpp` identical to the in-process compile, under the same inductor cache
  hash.

  **K is derived, never configured**, as the min of three bounds the pod actually
  has: free VRAM over one entry child's device footprint (`mint_budget.co_residency`,
  the bound that binds — an AOTI compile benchmarks kernels ON THE CARD),
  `effective_cpu_count()` minus serving headroom, and available host RAM; ceiling 8.
  A 4-vCPU pod gets K=1, which is the previous serial path exactly.

  **The safety interlock is the point, not the speed.** Kernel configs are chosen by
  timing kernels on the device, and the cell key does not move when a config changes —
  so two entries benchmarking at once could publish a slower cell under a good cell's
  identity. `aot_device_lock` registers a cross-process `flock` on torch's own
  `set_gpu_benchmark_lock_context` hook, so no two entries ever time a kernel at the
  same moment; a torch without that hook forces K=1 rather than benchmarking against
  itself. Entry children carry `PR_SET_PDEATHSIG` so an abandoned mint cannot orphan
  a compile onto a serving pod.

  Cell identity is untouched: `env_seal.inductor_config_digest()` is unmoved across K
  and across the pool's cache dir, and assembly is ordered by entry NAME, never by
  completion. `mint_phases.pool` records the K a mint ran at and every input that
  chose it.


- **pgw#805 — an AOT cell-discovery MISS never started a mint. The AOT lane was a
  pure consumer.** `aot_mint.mint()` was reachable only from
  `python -m gen_worker.aot_mint`: no module on the serving path imported it, so a
  `prefer_aot` pod's miss fell through to the DYNAMO self-mint, whose artifact kind
  `aot_cells._candidates` rejects — every pod missed, "re-minted" the wrong kind, and
  the next pod missed identically. Measured on five real 0.78.0 L4 pods with every
  precondition present: discovery missed honestly and then nothing happened, for the
  rest of each pod's life, with **no refusal of any kind** on the wire. A miss now
  chooses a RECIPE (`fleet_cells.mint_recipe`) and an AOT miss mints out of process
  (pgw#784's route — an AOTI export has no router to yield through, so in-process
  would break eager-first outright), exporting the family's whole declared class set
  as one multi-graph cell against the pipeline the child already loaded through the
  endpoint's own `setup()`. A self-minted `.pt2` adopts through the AOT gates
  (`provision.arm_aot`, extracted), not the inductor seed path.

  Two more wires were missing behind it. **The declaration**: `export_declaration`
  registered only when a mint REQUEST named a module, which a serving pod has no
  access to — a `compile=` block carrying graph classes is now registered as its
  family's export declaration at endpoint-collection time. **The telemetry**:
  `aot_mint` emits `aot_mint_phases` from the mint CHILD, which holds no orchestrator
  session, so th#1322's column has been empty on both stacks since it shipped; the
  parent now re-emits the child's phase table (`phase=minted` roll-up,
  `phase=entry:<class>` rows, `phase=aborted` when no cell came out).

  And the silence is gone. `_fail_closed`'s plain-lane eager degrade was a bare
  `logger.info` on a pod that exposes no logs (pgw#760); every decline now names
  itself — `self_mint_started phase=<recipe>`, `self_mint_skipped` with
  `no_export_declaration` / `aot_lane_regressed` / `aot_lifted_torch_gap` /
  `aot_requires_delegation` / `mint_unavailable`, and a
  `boot_ended_uncompiled` backstop when a declared compile target ends a boot
  unarmed with no mint in flight. Note the five measured pods were on the PLAIN
  lane, which #730 holds on dynamo: they must decline. The hold was right; being
  unable to say so was the defect.

- **pgw#797 — the boot spans pgw#789 shipped never ran on the real path.** Three real
  hub-spawned L4 boots on 0.78.0 recorded two rows each, `first_request_servable` at
  ordinal 1 and `hello` at ordinal 2 — the boot closing seconds before the stream to
  the hub existed, and `in_boot()` then suppressing `weights_fetch`, `pipeline_load`
  and `warm_complete` for the rest of the process. One cause: `Worker.arun` runs
  `Lifecycle.startup()` concurrently with the transport, and every real release
  declares `Compile`/`Slot`, so its specs go to `dynamic`, `awaiting_hub` is empty,
  and pgw#789's guarded mark fired anyway. The milestone now has ONE owner —
  `maybe_send_state_delta`, marking on the fact that a StateDelta advertising a
  function went out — and the recorder HOLDS a close that arrives before `hello`, so
  the inversion is unrepresentable rather than unlikely. Cumulative rows can no
  longer be a span's child, and the span stack is a ContextVar so interleaved
  setup/mint/adopt tasks nest against their own creator.

- **pgw#797 — warmup is its own phase.** `warmup` split out of `pipeline_load` as a
  nested span tagged `armed=0|1` (so "what does a cell save on warmup" is a GROUP BY,
  not an estimate), one `warmup_iteration` row per forward because the first
  dominates, the eager custom-`warmup()` duration ungated from `spec.compile is not
  None` and put on `ActivityUpdate.duration_ms` instead of a logger no hub-spawned pod
  can read, and the post-arm warm recorded as a boot row matching th#1329's
  `warmup_ms` by construction. `pipeline_load` now means weights->VRAM.

- **pgw#797 — the test that would have caught it.** `test_boot_span_ladder_pgw797.py`
  drives the REAL entrypoint sequence (real Worker, real gRPC socket, hub-stamped
  DesiredInstance, rows read off the wire) instead of calling the emitters, which is
  what pgw#789's suite did while production emitted nothing.

- **pgw#793 — AOT mint host compile is 7-9 % cheaper, with no cell re-key.** pgw#793
  measured that an AOTI mint's whole host cost is ONE `g++ -O1 -c wrapper.cpp` (46 % of
  the AOTI compile; linking is 0.5 %), and that the largest function in that TU is
  inductor's `constants_info_` table written as 26,642 straight-line statements — data
  compiled as code, executed once at model construction. `gen_worker.aot_wrapper_split`
  regroups exactly that run into chunked `noinline`/`optimize("O0")` helpers before the
  compiler sees the source: **−20 % on the wrapper object on an idle host** (−14 % under
  heavy load), which is ~8-11 minutes off an 18-entry sdxl cell. It verifies itself by
  re-inlining its output to the original byte for byte and declines unmodified on any
  wrapper shape it does not recognise, emitting a typed `aot_wrapper_split` event either
  way. `GEN_WORKER_AOT_WRAPPER_SPLIT_OFF=1` disables it. No compiler, flag, inductor
  config or library changes, so no cell is re-keyed.

- **th#1335 — cell discovery names an authorization refusal.** th#1310 made the
  platform's `root/family-*` cell repos private, and the worker's checkpoint
  listing then answered 404: indistinguishable from "this family has no cells",
  so one boot's discovery was abandoned and the pod served eager for life. The
  worker JWT now carries a hub-issued `read_repo` grant for exactly the families
  its release declares, 401/403 surface as a NAMED `not_authorized` phase on the
  `aot_cell_discovery` event (distinct from `list_failed`), and the anonymous
  retry on 401/403 is deleted — it was th#1310's own recorded hazard, one
  visibility flip away from unauthenticated cross-tenant enumeration.

- **pgw#795 (round 4) — the fix's own defect: a flake traded for a 13-minute hang.**
  `Cadence` shared its slowest sample session-wide with a `10x` window, so one slow
  advance anywhere multiplied every later wait and a wait with NO progress could sit
  for many minutes where the old code flaked at 15s. That is a worse release gate,
  not a better one: a flake costs one re-run and names itself; a hang costs the whole
  job and says nothing. A cadence is now scoped to ONE wait, so the bound is
  `max(floor, headroom x this wait's own slowest advance)` — a wait that never
  advances dies at the floor, always, and only a wait that IS advancing may extend
  itself. `test_residency_republish_pgw628` went 62.1s -> 2.5s with the same
  correction (its `_settle()` scaled a quiet period to `10x` a measured latency;
  a runaway re-announce loop re-announces immediately or not at all, so waiting
  longer catches nothing).

  Also: `WorkerHarness.stop()` discarded its 15s join result, so a worker that never
  exited passed teardown in silence — the one outcome nothing asserted. It is now a
  loud failure (red-verified: a wedged worker is caught, a clean exit returns in
  0.2s). `run_entrypoint`'s silence floor is a FLOOR on a caller's number, never a
  cap, and its expiry now says "produced no output for Ns (a SILENCE window, not a
  time budget)" instead of `TimeoutExpired`'s misleading stock message.
  `hardware_report_hub.wait_for_message()` is progress-gated only when the caller
  passes no bound — widening every call site at once was measured breaking six
  unrelated tests, so a must-happen wait opts in.

- **pgw#795 / th#1314 — the publish gate asserts PROVENANCE instead of re-deriving it.**
  `publish.yml` no longer re-runs the suite `ci.yml` already ran green on the same
  tree with the same `--locked` resolution; a second identical run cannot produce new
  information, only flake, and it produced three in a row for v0.78.0. It now asserts
  that some successful CI run carries this exact TREE (trees, not commit SHAs — a
  release branch, a squash merge and a cherry-pick all give different commits for
  identical content), then runs only the artifact-facing steps. STRICT: an unproven
  tree refuses to publish, with a message naming the two one-run ways to satisfy it.
  It also gains `timeout-minutes: 20` — it had none, so a hang ran to GitHub's 6-hour
  cap. Validated against the live API on both paths. **Finding it surfaced: v0.78.0's
  published tree was never itself CI-green** — the promotion PR tested master plus
  cherry-picks, a different tree from the chaos commit the tag pointed at.

- **pgw#680 follow-on — a background warm/heal now compiles under the REQUESTING thread's
  intra-op count, so the entry it produces is servable by the thread that asked for it.**
  Dynamo's `GLOBAL_STATE` guard snapshots `torch.get_num_threads()` on the thread that
  COMPILES, and the OpenMP intra-op ICV is per-thread and sticky once initialized.
  `hot_swap` runs ONE process-global shape-warm thread, so if it is created before anything
  narrows the serving thread's count, every entry it compiles carries a guard the serving
  thread can never satisfy — `GLOBAL_STATE changed: num_threads`. The pgw#622 hot-swap path
  then never swaps, and a pgw#680 guard-miss heal marks the signature warm while its entry is
  unservable, so the next request misses again and `_GUARD_MISS_HEAL_LIMIT` makes the
  signature permanently `volatile` — the outcome the doctrine exists to prevent.

  **Latent rather than live on today's ordering, and stated that way deliberately:** both
  `cpu_budget.impose_intra_op_threads()` call sites run at boot, before any warm job exists
  (`entrypoint._impose_group_host_policy` and `Executor.__init__`), so the warm thread
  inherits the imposed value and no shipped pod is known to have hit this. It goes live the
  moment anything imposes AFTER serving starts — a second `Executor` in one process, a
  topology re-assert, a harness path. The warm job already carried the requesting thread's
  grad-mode and autocast state; the intra-op count is the same class of state and was simply
  missed, so it is now carried and imposed the same way. Found by the release gate rather
  than by a pod, and red/green-verified through the real `Router` and the real
  process-global warm thread.

- **pgw#806 / pgw#802 / pgw#738+#764 — suite minimization and test health (no library
  behaviour change).** The publish gate is the suite, so a test that fails on the
  runner's mood is a release-blocking defect. Source-text handcuffs (tests asserting the
  presence or ABSENCE of a string in `src/`) are deleted in favour of behavioural
  assertions; four issue-labelled satellite files that were four views of one seam
  collapse into the e2e file that already drives it; pgw#802's postmortem carriers are
  redirected in `conftest.py` on the SUBPROCESS half too, so a child that records a
  native crash can no longer poison the host's machine-global crash registry and fail
  six of every other lane's tests; and four untracked tests that were red for every lane
  are adjudicated against a clean archive rather than the shared worktree — none of them
  was actually broken. One production-visible line rides along: pgw#797's warm-iteration
  counter advances only inside the boot window, so a steady-state warm cannot misreport
  a later boot span's iteration total.

- **Correction to the 0.79.0 entry (the tag cannot change, so it is recorded here).**
  0.79.0's `publish_v2` bullet says the route REFUSES a provenance stamp and that "the
  mirror flip is blocked on the route (th#1331)". `f50dae2` landed in that same release
  and deleted the guard: as shipped in 0.79.0, `publish_v2` WRITES `body["provenance"]`
  and the mirror publishes v2. The 0.79.0 bullet describes an intermediate state no
  published artifact has.

## 0.79.0 (2026-07-30) — the v2 serving path actually works: the vendored proto is no longer stale, the chunked manifest parses against the REAL hub shape, and a publish refusal stops being retried as an outage

> ### ⚠ 0.79.0 IS THE FLOOR FOR SERVING ANY CHUNKED (v2) CHECKPOINT — 0.78.0 CANNOT
>
> **0.78.0 shipped two defects that made a v2 artifact unresolvable. Both are
> fixed here. `0.79.0` is the minimum version for any worker that may be handed a
> v2 snapshot.**
>
> **(1) The vendored proto was STALE, so the PRODUCTION gRPC path could not carry
> a v2 snapshot at all.** th#1303 checkpoint 3 added `digest`, `chunk_size_bytes`
> and `chunks` to `worker_scheduler.proto` and regenerated only the Go side;
> 0.78.0's `SnapshotFile` stub has fields `['path','size_bytes','blake3','url']`
> and no `ChunkRef` message. Production workers receive snapshots over gRPC, so
> every v2 snapshot arrived with no digest and no chunks and the worker refused
> the model. Fail-closed, but wholly dark.
>
> **(2)** `models/hub_client._parse_chunks` in 0.78.0 required a `url` INSIDE each chunk
> object. The hub cannot put one there: `chunks: [{digest, len}]` IS the
> content-addressed manifest identity, so resolve-time URLs ride in a SEPARATE
> index-aligned `chunk_urls: []`. Against a real hub, EVERY chunked (>64 MiB)
> checkpoint therefore failed to parse with `missing digest/url/len` and
> `resolve_repo` refused it. Impact was bounded only because no v2 artifact is in
> production yet. Whole files and every v1/blake3 checkpoint were unaffected.
>
> **The rollout order this implies:** producing v2 chunks is safe at any version
> (a v2 artifact nobody resolves harms nothing), but **no tag may be repointed at
> a v2 manifest, and no v2 artifact may be resolved, until the SERVING FLEET is on
> 0.79.0.** The gate lifts on fleet version, not on this upload.

- **pgw#781 / th#1303 — the vendored proto is synced and the stubs regenerated.**
  `SnapshotFile` now matches the hub byte-for-byte and `ChunkRef{sha256,url,len}`
  exists. ADDITIVE (fields 5/6/7 on an existing message; `blake3`=3 and `url`=4
  untouched), so this is not a protocol break and needs no th#1282 floor bump.
  `_snapshot_to_resolved` drops its `getattr(f, "digest", "")` defaults for
  direct field access: reading a possibly-absent proto field with a default is
  how "the stub does not have this field" became "the hub sent an empty value",
  which is the same absent-becomes-empty class as guarding a digest check on a
  legacy field's truthiness. A missing field is now an AttributeError at the
  wire boundary. Guarded by tests that assert the STUB's fields (naming
  `task proto` in the failure) and one that drives the real conversion over a
  real protobuf message with three chunks — the previous tests all built the
  dataclass directly and so never crossed the wire, which is why this survived a
  green suite.

- **pgw#781 / th#1303 — the live acceptance, and the three defects it found.**
  `v2 publish -> promoted -> resolve -> worker fill -> byte-identical` now runs
  end to end against a real hub and real R2 (evidence:
  `~/cozy/samples/ingestv2-accept/`). Everything below was found by that run and
  by nothing else; each had passing unit tests over it.

  - **The chunked-manifest wire shape** (above). The parser was written from a
    design document instead of from the hub's serializer, and the unit tests
    encoded the same assumption — a self-consistent suite proving the client
    agreed with itself. For a WIRE contract the fixture must come from the
    producer's serializer or a captured real response. `chunk_urls` is now read
    index-aligned, a nested `url` still works (the proto's `ChunkRef` carries
    `sha256`/`url`/`len` together, so one library serves both transports), and a
    MISALIGNED url list is fatal rather than "fetch fewer chunks" — index
    position is the only thing binding a URL to its digest.

  - **A typed refusal was retried into a different error.** A v2 completion does
    not answer with an error envelope; it answers with the th#1301 PROJECTION,
    whose refusal carries an explicit `status.failure.retryable`.
    `_send_with_retries` guesses definiteness from body SHAPE, so it read
    `retryable: false` as a proxy non-answer and retried — and the retry, finding
    the session terminal, returned 409 `publish_repudiated`. Measured twice: the
    caller was told a consequence while the cause
    (`invalid_manifest_for_kind: missing_diffusers_single_file_safetensors`) was
    discarded. A publish is not idempotent to re-complete, so a blind retry there
    can only destroy the diagnosis. `_send_with_retries` gains an optional
    `definite` predicate; `/complete` supplies one and the error now leads with
    the hub's own code, retryable bit and failing stage.

  - **The publish error envelope is a SECOND hub shape.** tensorhub's
    `publishError.body()` emits `{"error": "<code>", "message": ...}` — the code
    as a STRING, not an object — which `response_is_from_hub` did not recognise.
    So every publish refusal, on **v1 as well as v2**, has been classified
    proxy-shaped and retried under the silence window. It stayed invisible
    because a retried v1 refusal usually fails again identically. pgw#743's bias
    is preserved and re-pinned: an unrecognised body is still proxy-shaped,
    because mis-terminating an outage throws away paid-for work while
    mis-retrying a refusal costs a bounded backoff.

- **pgw#781 / th#1303 — `publish_v2` REFUSES a provenance stamp rather than
  dropping it.** The v2 declare route writes `Provenance: []byte("{}")`
  unconditionally, while every mirror publish carries
  `{"upstream_revision": …}`. Flipping the mirror path to v2 would blank,
  silently, the field naming which upstream commit we copied — the field
  th#1301's strongest check (our sha256 vs upstream's published sha256, the
  interop the whole re-key was for) is computed against. Filed as th#1331; the
  guard is deleted in the same commit the route learns the field.


- **th#1330 B2 — a component OVERRIDE no longer also fetches the component it overrides.**
  pgw#617 is load-then-substitute: the override's tree is materialized separately and handed
  to `from_pretrained` as a constructed object, so the base composition's copy of that
  subfolder was downloaded and then never read — ~1.64 GB per SDXL text-encoder override, per
  pod. `_materialize_local` now derives an exclusion from the binding's own
  `component_overrides` (only for subfolders the snapshot actually carries) and passes it
  through `ensure_local` -> `cozy_snapshot.ensure_snapshot`; every byte figure on the path
  (disk-headroom gate, DOWNLOADING totals, the pgw#789 boot weights span) counts what is
  actually fetched, and a typed `component_fetch_skipped` activity event names the bytes
  skipped. The narrowed tree keys as `<digest>__x<fp>` (pgw#505's mechanism, negative side),
  so it can never occupy the name reserved for a complete snapshot: a later FLAT dispatch of
  the same base ref still materializes the full tree. Proven RED at HEAD / GREEN with the fix
  on the real worker+CAS boundary in `tests/test_component_override_fetch_th1330.py` — the
  evidence is the CAS itself, the base's `vae/` blob is absent from `blobs/`.

- **th#1330 B4 (worker half) — `reclaimable_bytes` stopped over-promising, and disk GC stopped
  deleting a tree a sibling ref still points at.** The reported figure summed each evictable
  ref's whole indexed tree, so a blob two evictable refs shared was counted twice AND a blob
  an evictable ref shares with a RETAINED one was counted as reclaimable even though
  `sweep_orphan_blobs` only unlinks at `st_nlink == 1` — deleting that tree frees nothing.
  The hub sizes every capacity decision off this number. Reclaimable is now the deduped set of
  digests no retained ref holds; a ref with no banked manifest keeps its full indexed size.
  Separately, `_evict_disk_ref` now refuses to `rmtree` a snapshot directory another
  still-resident ref is materialized at (two refs at one digest share one directory).

- **th#1330 B5 (worker half) — the banked snapshot map is no longer append-only.**
  `ModelStore._snapshots` / `_snapshot_generations` / `_verified` kept a manifest forever, so a
  ref dropped from DesiredResidency could still be materialized from OBSOLETE bytes by a later
  bare `ensure_local(ref)`, with no hub prompting. `replace_desired_snapshots` now drops
  manifests for refs that are neither desired, resident, in the preserve set, in use, nor mid
  materialization; a ref wanted again goes through `_await_hub_snapshot`, which is the correct
  path (the hub re-mints with LIVE presigned URLs).

- **pgw#791 — the AOT serve path now satisfies the artifact's ALIGNED-input contract at
  ingress, once, instead of letting AOTInductor copy per call.** Inductor compiles its fast
  path for 16-byte-aligned inputs and, when a pointer is not, its generated wrapper clones the
  tensor on EVERY call and reports it with a C++ `TORCH_WARN` — i.e. on the worker's stderr,
  which hub-spawned pods do not expose. diffusers hands the denoiser `timesteps[i]`, a scalar
  VIEW at an odd element offset, so an armed SDXL cell paid it 28+ times per request:
  measured on an RTX 4090 (production `w8a8-lora64`, gw 0.76.8), the request residual over
  28x(per-forward) was 196 ms for the `.pt2` against 77 ms for the equivalent dynamo cell —
  the whole AOT advantage, spent on a check the serve path never performed.
  `aot_serve` now checks alignment and contiguity at ingress and realigns into an owned,
  pointer-stable buffer allocated ONCE per input, and the residual fallback is a typed,
  hub-visible `aot_input_realigned` event naming the input (coalesced to one per
  (entry, input, reason), with every occurrence counted and surfaced by
  `aot_serve.realigned_inputs(pipeline)`).

  **Measured under control (RTX 4090, one artifact, one process, only the ingress differs):
  the defect reproduces exactly — `timestep` is unaligned on 126 of 168 denoiser calls and the
  pre-fix path logs 126 runner-side clone warnings against the fixed path's zero — but the COST
  is +0.8 ms on a 3,373 ms request (+0.02%), not the ~119 ms the issue inferred from an
  uncontrolled residual comparison.** So this is a contract fix and an OBSERVABILITY fix, not a
  latency win: the warnings are written by C++ to fd 2, where even an in-process
  `redirect_stderr` cannot see them, and the typed event is the only surface that reaches the
  hub. WARM-INFERENCE-MATRIX §2c's 196-vs-77 ms residual gap needs another explanation.

- **pgw#790 — a LoRA-bucket family now mints BOTH graph classes into one cell, and the serve
  path routes adapter-free traffic to the branchless one.** gw#627's canonical zeroed
  rank-bucket branch predicted "a small constant overhead" on non-LoRA requests; measured it is
  **+31.8% of the compiled per-forward on a 4090 and +44.9-45.9% on a 5090**, with kernel
  launches +54%/+114% — paid to compute zeros by the 95% of sdxl denoiser forwards that name no
  adapter. The mint fans every branch-capable target's plan into `<target>/adapter=true/...`
  and `<target>/adapter=false/...`; the branchless class declares `excluded_inputs` (the
  NEGATIVE half of the ingress contract, without which both classes admit an attach call and
  the dispatch refuses `entry_ambiguous`); and `aot_serve` omits the lifted pair from an
  adapter-free call so only the branchless class admits it. Both classes are exported,
  compiled and KEYED at mint — the arm is a fork coordinate like any other — so no program's
  structure varies with runtime state, and the eager fallback still always receives the pair.
  `range_digest` keys `excluded` only when non-empty, so published cells are not re-keyed.

- **pgw#790 (P0 found while proving it) — a FLATTENED container argument shifted every
  later declared input name, so an armed sdxl cell refused EVERY request.**
  `aot_package.input_contract` zipped the caller-side parameter names against the exported
  program's user inputs positionally, but a container argument occupies one parameter slot
  and produces N placeholders. Measured on a real SDXL UNet: `added_cond_kwargs`
  ({text_embeds, time_ids}) shifted everything after it, and the recorded contract came out
  as `added_cond_kwargs` [2, 1280] at position 7 and `down_block_additional_residuals`
  [2, 6] at position 8 — the two flattened leaves wearing the names of the parameters that
  follow them. At serve time `bind_call_inputs` binds the pipeline's `added_cond_kwargs`
  DICT to a declared tensor input and cannot find `down_block_additional_residuals` at all,
  so every request refuses by name and the cell serves eager for life. `aot_mint.
  flat_input_names` now derives one name per exported user input, mapping leaves taking
  their bare key in sorted order (what torch's pytree does, and what the serve-side nested
  resolution looks for).

- **pgw#790 (prerequisite) — the lifted-LoRA call convention is now STATED, which is what makes
  the branch-bearing class exportable at all.** `install_lifted_lora_forward` declared
  `lora_a`/`lora_b` keyword-only, so the declared-input path refused them by name
  (`_positionalize` rejects declared keyword-only inputs) while feeding them positionally — the
  pgw#723 mint obligation — landed them in `*args` and died on `the lifted LoRA argument
  'lora_a' is missing`. The wrapper now appends the pair to the denoiser's own positional
  parameters via `__signature__` and accepts it either way.

- **pgw#795 (round 3) — three more load-induced failures, found by running the
  suite against a clean `chaos` tree on a loaded box.** None was caused by any
  code change; all three asserted the machine. `subprocess_runner.run_entrypoint`
  turned its `subprocess.run(timeout=25.0)` TOTAL budget into a SILENCE window —
  the entrypoint emits `worker.startup.phase` lines throughout boot, so a boot
  that keeps talking now runs as long as it needs and only silence gives up
  (red-verified: a chatty child survives 3.6 s through a 1.0 s window; a wedged
  one is caught in 1.3 s and killed, output preserved). `test_mint_liveness_pgw784`
  asserted a beat-gap margin three times tighter than the hub's own kill line and
  failed at 0.51 s against 0.50 s while the kill line sat at 1.50 s — it now
  asserts the property (the hub would never kill this worker) and reports the
  margin. `test_p2_residency_reconcile` sampled `resident_identity` the instant an
  event arrived and read `('snap-a', 0)`; it now waits for the store to rebase on
  progress.

- **pgw#795 (sweep + guard) — the anti-pattern is now detected, not remembered.**
  `test_magic_timeouts_gw666`'s guard pinned five NAMED constants in five NAMED
  `src/` files, so it missed this twice over: it never looked at `tests/`, and a
  freshly-invented wall clock walks past a regression pin even in `src/`. "Tests
  are exempt" does not survive the evidence — the publish workflow runs
  `pytest tests/` as the gate on every PyPI upload, so a test that fails on the
  runner's mood is a release-blocking defect (three publish jobs, each ~4.6 h
  queued and ~22 min run, three lanes blocked). Three kinds of fixed duration in
  tests remain legitimate and are NOT flagged: induced durations (the fake
  handler's sleep IS the stimulus), absence probes (`timeout=2.0` inside
  `except TimeoutError`, whose expiry makes the test PASS), and hang bounds with
  an order of magnitude of headroom. The flagged shape is the fourth: a fixed
  duration whose EXPIRY FAILS the test.

  Guards added: `tests/harness/` may hold no fixed deadline at all (one there is
  one in ~37 files), and the remaining sites in test modules — 6 deadline waits,
  5 elapsed-vs-constant assertions — are pinned as burn-down inventories that may
  shrink but not grow. Fixed in this pass: `disk_usage_report` proves it skipped
  the stalled statvfs structurally (`refresh_task` still pending) instead of by a
  0.1s budget; `_close_sequence_group` proves it returned before the 0.5s close
  finished instead of by a 0.4s budget; pgw#677's tenant-latency budgets anchor to
  the harness's own `seed_forward_s` / `compile_delay_s` instead of 0.45s / 300ms
  / 400ms; and the auth-retry and permit-holder waits count progress through
  `await_count`.

- **pgw#794 — a fail-closed adapter-fidelity gate: an adapter the serving dtype
  DESTROYS is now refused instead of served silently.** Fusing a shipped adapter
  into fp8-E4M3 weights does not erase the delta, it SUBSTITUTES for it
  (measured: surviving-delta cosine 0.074 on qwen Lightning, 15x the true
  delta's norm, 99.7% orthogonal to it). `_reject_zero_delta` catches an EMPTY
  adapter, never an INERT one, and te#86's produce-time detector diffs in the
  SOURCE dtype, so the fp8 cast — the whole hazard — was invisible to it.
  `models/adapter_fidelity.py` scores the norm-weighted whole-adapter cosine
  against the grid read off the REAL destination; refusal is the typed,
  hub-visible `AdapterFidelityRefused` plus a `lora_fidelity` activity event.
  Refused ONLY on the path that destroys it — the resident branch keeps A and B
  separate and rides at 0.999999.

## 0.78.0 (2026-07-30) — cross-SKU adoption becomes real, the benchmark telemetry is connected, and the suite stops asserting the runner

- **th#1330 (th#1316 worker half) — a `disk_ref` the hub's own resolutions have
  already replaced is no longer materialized.** The reconcile pass executed
  `DesiredResidency.disk_refs` verbatim and serially, so a desired set carrying
  BOTH `<ref>` and `<ref>#fp8` — with this worker's own `resolutions` mapping the
  first onto the second — pulled the bf16 base ahead of the fp8 variant it was
  meant to replace. Measured on prod (`tensorhub/sdxl` 0.2.23, L4): 6.94 GB at
  +178 s ahead of 4.38 GB at +230 s, 144 s of a 270 s cold boot, for weights the
  pod never loaded (`lane=fp8-w8a16`). The declared spelling is now skipped —
  exactly when its resolved twin is desired in the SAME generation, so a
  canonical-spelling lane override (th#913) is never skipped — and it is dropped
  from `store.keep` so it cannot outrank cold refs in the GC preserve set. Each
  skip emits a typed `residency_ref_superseded` activity event.

- **pgw#795 — the v0.78.0 publish blockers: two tests that asserted the RUNNER,
  not the code.** Three consecutive publish jobs died on them, and no code under
  test had changed.

  `test_residency_republish_pgw628` waited on a fixed 15 s deadline for its third
  ON_DISK re-report and reported "saw 2 of 3". The clock was the messenger, and it
  lied about the cause: the real defect is that `harness/residency-tiny` is a ref a
  toy endpoint DECLARES, so ~2 s after boot the eager first boot promotes it to RAM
  and every later re-sent plan re-announces the held identity as IN_RAM instead of
  ON_DISK. The test needed three ON_DISK re-reports inside a ~2 s window and was
  racing a worker-side timer; a loaded runner loses that race. Measured directly:
  with the declared ref the re-announce stops at t~=2.03 s regardless of ack
  spacing; with an UNDECLARED ref it holds at 0.5 s and 5 s spacing alike. A wall
  clock cannot tell "slow" from "this can never happen now", which is why raising
  the timeout would have bought a fourth failed release rather than a fix.

  `test_th1130_deferred_tail` asserted `encode_ms >= 10` — a claim about the
  runner's spare CPU. It had already been lowered from `>= 100` (failed at 83 ms)
  and from `>= 0.5 * total.tail` (failed at 76 vs 77). Lowering the constant each
  time is the anti-pattern with a smaller number: the quantity is the machine's
  speed. It now proves the encode by its PRODUCT — a 1024^2 WEBP frame on the wire
  plus an attributed `image_encode` stage — and the slot-exclusion inequality takes
  its slack from the stage map's own unattributed residual instead of a hard-coded
  15 ms, so the tolerance grows with the noise it exists to absorb.

  New `tests/harness/progress_wait.py` gives the suite the give-up rule
  `gen_worker.stall` already gives production code: a wait ends when the thing
  happens, when the peer is provably gone (definitive, no clock), or when the
  awaited observable has not advanced for a staleness window this run MEASURED
  (`Cadence`: 10x the slowest advance seen, session-shared, floored). Only the
  awaited observable counts as progress — peer liveness and unrelated chatter do
  not reset the window, because a window that resets on them never closes (both
  hangs measured while authoring this). `hub_double`'s `wait_for` /
  `wait_for_count` / `wait_connection` drop their 15 s wall clocks onto the same
  rule for all ~37 files that use them, while an EXPLICIT `timeout=` keeps its old
  meaning for the callers that probe for absence. Expiry now names what it waited
  on, what it last saw, and how the window was derived.

- **pgw#781 / th#1303 — the chunked sha256 CAS is WIRED, in both directions.** The
  primitives landed with nothing calling them: `models/chunk_upload.py` had zero
  production callers, and `models/cozy_snapshot.py` — the real fill path — never
  imported `models/chunk_cas.py` at all. So a v2 publish was unconsumable and a v2
  publish was unproducible, which made "publish v2" untestable as an outcome
  rather than merely unproven.

  **Download.** `WorkerResolvedRepoFile` carries `digest` / `chunks[]` /
  `chunk_size_bytes`, and `cas_ref()` is the ONE place the v1/v2 dual-read is
  expressed for a resolved entry — it RAISES on an entry with no readable digest
  instead of returning `""`, because every caller of it is an integrity check.
  Both wire boundaries (gRPC `pb.Snapshot`, HTTP resolve) carry the new fields,
  and neither requires a whole-file URL any more — a chunked entry legitimately
  has none. Local CAS becomes `blobs/<algo>/aa/bb/<hex>`, so legacy trees keep
  their exact paths while sha256 blobs cannot collide with blake3 ones. Chunked
  files reassemble through `chunk_cas.download_chunked_file` (bounded
  out-of-order fetch, in-order commit, fused whole-file hash, per-chunk progress
  floor — pgw#786's lemon-host pathology now solved on the real path, at chunk
  granularity, because the retry unit is 64 MiB and completed chunks are
  durable).

  **Upload.** `convert.hub.HubClient.publish_v2()` implements the hub's v2
  contract: declare -> `{have, need}` -> PUT the granted objects with the hub's
  headers VERBATIM -> complete. The checksum is inside the presigned signature,
  so R2 refuses wrong bytes (400, and the object does not exist afterwards) and
  refuses a substituted claim (403) — a claimed digest stops being assertable,
  which kills th#1305's inherit/overwrite class structurally. Resume needs no
  client state: re-plan and the need set comes back smaller. There is NO protocol
  auto-select and no env knob — the caller names the protocol, so flipping a
  producer class is a code change plus a deploy, and a hub without the v2 routes
  FAILS rather than silently downgrading to blake3.

  **Two silent defects fixed on the way, both of the same family as the executor
  gate.** `_verify_materialized_tree` read `f.blake3` and guarded on it being
  non-empty, so a reused v2 tree was reported CLEAN having hashed ZERO bytes; and
  quarantine/`delete_blobs` STRIPPED the algorithm tag, aiming the unlink at
  `blobs/blake3/<sha256hex>` — at nothing — leaving the corrupt blob to be
  re-linked by the very next fill. Separately, `NetworkBytesScope`'s sink was read
  through a contextvar per call, and chunk fetches run on a `ThreadPoolExecutor`
  that does not propagate contextvars, so every chunked transfer would have
  reported ZERO network bytes — th#850's "volume-attached boot => ~0 network
  bytes" assertion reads that counter, so a cold boot would have looked warm.

  23 new tests, all over real localhost HTTP servers with real sockets, threads
  and files. The download side asserts BYTES HASHED, never merely "ok" — "ok" is
  exactly what the old code got right while doing nothing. The upload side runs
  against a stub that ENFORCES LIKE R2, because that enforcement is the design.

- **test hygiene (release-train blocker)** — `test_th1130_deferred_tail` asserted
  that the deferred encode occupies at least half the request tail. That is a
  claim about the RUNNER's spare capacity, not about the code: a busy machine
  inflates the unattributed residual and the ratio moves with nothing under test
  changing. It failed a release attempt at `76 >= 0.5 * 155`, having already been
  weakened once from an absolute `>= 100` that failed the same way. Removed; the
  property it was reaching for (the encode runs slotless, entirely after the
  permit is released) is asserted directly and remains true at any speed.

- **pgw#788 — a TORCHLESS worker could not boot. torch is a CAPABILITY now, and its
  absence is a sealed FACT.** `entrypoint.py` calls `env_seal.establish()` on every boot
  regardless of `accelerator`, and from **0.70.3** onward that chain bare-imported torch at
  three call sites with no guard — `env_seal.establish_config()` (and `effective_config()`),
  `host_isa.impose()`, `guard_closure.establish_posture()` — so a torchless image died at
  `phase=env_seal` before advertising a single function, with no env knob to skip it. The
  window is 0.70.3 / 0.70.4 / 0.70.5 / 0.75.x / 0.76.x / 0.77.0 / **0.78.0**; the last safe
  line is 0.70.2. `task e2e`'s marco-polo J2/J3/DelegatedSpend have been unbootable that
  whole time, and ie#578 has already relocked `dj-utils`, `quality-benchmark` and
  `dj-pipeline` onto pins inside it — they break on their next BUILD, not on the relock.

  New `torch_capability.torch_or_none()` is the one probe all three use. When torch is
  absent the seal RECORDS it: `config` carries `torch: "absent"` (plus the torch-free
  interpreter facts), `inductor` is `"absent"`, `posture` is `{"torch": "absent"}`, and the
  ISA clamp and guard posture no-op, logged once. That is a *stronger* seal than one that
  cannot be computed — "this pod had no torch" is a real, keyable environment fact, so the
  digest stays meaningful for CPU cells. **Adding torch to the CPU images was rejected**: a
  ~2-3 GB dependency on pods whose entire value is being cheap inverts the point and encodes
  an SDK defect as a fleet requirement (`accelerator='none'` is a first-class shape, th#721).
  A DECLARED config knob on a torchless worker still refuses by name — every canonical knob
  is a torch flag, and honouring one silently would fork cell identity.

  **With torch present the seal is byte-identical**, verified rather than asserted: the same
  `seal_digest` (`6eda4772...`) before and after on this box. A changed seal shape would have
  stranded every published cell.

  `tests/test_torchless_boot_pgw788.py` is the guard that did not exist — it import-blocks
  torch with a `sys.meta_path` finder (no second venv) and asserts the boot chain completes,
  and `publish.yml` now imports the public entrypoint and runs the seal **in the torch-free
  wheel-contract venv**, which is a real environment rather than a simulation.

- **th#1299 (worker half of the visibility defect)** — an abandoned self-mint now names
  its own cause. The abort event reported `phase="abandoned"` with the detail
  `"(adopt-on-arm / vacate / shutdown)"` — three unrelated causes in one string on the
  only wire record the hub keeps, so 41 such rows on the master stack could not be
  triaged from worker evidence at all (the real cause, the hub retiring the pod
  mid-mint, was found only by joining `worker_activity_events` to
  `worker_pods.retire_reason` by hand). `abandon_background_mint` now takes a `code`
  and the terminal handler reports `phase=abandoned_<code>`
  (`adopt_on_arm` / `vacate` / `shutdown` / `tenant_oom`, `unspecified` for a caller
  that omits it — a legible gap, never a plausible-looking wrong cause).

- **pgw#773 residual / pgw#748 — multi-group sequence parallelism is SERVED, and proven on
  four real H100s.** A group's cards now come from the delivered topology instead of the group
  ordinal (at degree D group `g` owns `[g*D, (g+1)*D)`, so on a `2x2` pod group 1 owns cards 2-3
  while the old helper pinned card 1 — group 0's follower card, silently, on every load-thread
  hop). With that, the `G>1 ∧ D>1` boot refusal is lifted for `sequence`; it stays typed and
  reachable for a degree>1 group whose sharding nothing here installs (`cfg`).

  Live acceptance on 4xH100-80GB-HBM3 SXM (NVLink `NV18`, real Wan2.2-A14B transformer,
  32,760 tokens / 40 heads): degree 2 **1.80x**, degree 4 **3.42x**, and two concurrent degree-2
  groups at **1.71x / 1.70x** each — every arm **bit-identical** to degree 1 at the same seed
  (`max|Δ| = 0.0` in fp64, 0 of 2,096,640 elements differ). Group 1's weights landed on card 2.

  Found live and fixed on the way: NCCL enables NVLink SHARP (NVLS) multicast by default on
  NVSwitch hosts and our containers cannot bind it, so EVERY SP collective died with
  `ncclUnhandledCudaError ... CUDA error 401`. `init_rank` now decides `NCCL_NVLS_ENABLE=0`
  before any communicator exists; Ulysses is all-to-all, which NVLS does not accelerate.

  Found live and NOT fixed (pgw#792, filed): killing a rank mid-call does not fail the group
  typed — the collective runs to the full 300 s ceiling and NCCL then takes the worker process
  down. Evidence and the re-runnable probe: `~/cozy/samples/spaccept/`.

- **th#1322** — compile duration is a NUMBER on the wire, for both mint routes.
  `ActivityUpdate.duration_ms` (proto field 17) carries the worker's own
  monotonic span, and a new typed `jit_compile` event gives the JIT path the same
  event shape `aot_mint_phases` has: `phase=minted` for the roll-up,
  `phase=entry:<graph class>` / `shape:<WxH>` / `child:<phase>` for the spans
  inside it. `aot_mint._emit_phase_event` now stamps `total_s` numerically and
  emits a per-graph-class event; `compile_cache.emit_jit_compile_event` retires
  the log-only `"compiled %s in %.0fs"` line at `compile_cache.py:3803`;
  `mint_child` measures per-phase spans through its own `frame()` funnel and
  `mint_delegate` turns the child's report into hub events (`phase=aborted` when
  no cell came out, so a failed mint's real seconds are recorded without
  polluting an AOT-vs-JIT comparison). Before this, AOT durations were only
  parseable out of the free-text `detail` and JIT duration existed nowhere off
  the pod. Hub half: tensorhub migration 0070 + `GET /v1/admin/compile-duration`.

### 0.78.0 detail — the cross-SKU adoption trio (pgw#765 + pgw#772 + pgw#789)

Everything on chaos since 0.77.0. The headline trio, cut together because none of
them is worth much alone:

- **pgw#765** — AOT adoption is pinned to `sm`, never to the GPU SKU. `verify`
  refuses on `("sm", "torch", "cuda")` + host ISA + family/contract hashes; `sku`
  is recorded and never refused on, and `_candidates` turned the old SKU FILTER
  into a SELECTION PREFERENCE (same-SKU first for autotune affinity, then stamped
  `sm`, then newest, key digest as tie-break) so a cross-SKU same-arch cell is
  always in the list, just behind. A second tier reads torch's own
  `AOTI_COMPUTE_CAPABILITY` out of the `.pt2` and refuses `sm_mismatch` before
  dlopen. An AST sweep over every adoption-path module fails the suite on a
  live-code comparison against `"sku"` unless allow-listed with a reason (two
  entries, both named).
- **pgw#772** — the serving lane is deterministic; the voluntary bf16-resident
  upcast is REMOVED. Detail section below.
- **pgw#789** — the benchmark telemetry that was built and never connected.
  `serving_mode` reached 0 of 416 request rows because the module was imported by
  nothing but its own test; the `weights_fetch` / `pipeline_load` /
  `warm_complete` boot spans were never emitted; and `first_request_servable` was
  measuring "startup() returned", recording 4.2-12.3s for pods whose real cold
  boots took minutes. Wired for real, so compile / cold-boot / inference numbers
  reach the hub.

pgw#765 + pgw#772 are the two halves of the same defect and only work together:
pgw#765 removed the `sku` pin, pgw#772 removed the `lane` fork, and cross-SKU
adoption is end-to-end only with both. pgw#789 is what makes the result
measurable. **Consumers relocking for the AOT/JIT/eager benchmark matrix want
this version or later** — `c87ea3d` (pgw#789) existed in no published tag before
now.

Also aboard, by issue (see each entry below where one exists): pgw#770 +
follow-up (nunchaku svdq fragment packing; `qweight` k % 128), pgw#773 / #774 /
#775 / #776 / #778 (per-group process groups and failure domains, rank-symmetric
forwards, the pod's hub-facing VRAM truth, never advertising what dispatch will
refuse), pgw#781 / th#1303 (chunked sha256 reassembly + the mandatory volume
check), pgw#782 / th#1313 (the width-4 DP collapse root-caused to the shared
interpreter; a worker refuses to materialize a snapshot containing pickle
weights), pgw#784 (a mint runs in its own OS process, now wired), and **th#1307**
(the C2PA private key never enters a pod — its detail section further down is
still headed "Unreleased"; it ships HERE).

One test-only fix was made by this release lane to get the train green:
th#1307's `signer_configured` fixture reset the C2PA module globals with
`monkeypatch.setattr` in its post-yield teardown, so monkeypatch's own finalizer
restored the ARMED values right back and leaked a configured signer — pointed at
an already-shut-down fake hub — into every later test that saves an image. Seven
tests across `test_image_encoding_default`, `test_th1111_stage_timing`, and
`test_th1130_deferred_tail` went `JOB_STATUS_FATAL`. Teardown now assigns plainly.
No product path touched.

- **pgw#784 — a mint runs in its OWN OS process; the worker serves eager and
  reports throughout.** th#1299 killed a live sd15 pod mid-mint and the hub was
  right to. The mint's compile driver ran INSIDE the serving process, and
  inductor's orchestration layer is long-running GIL-holding Python, so it
  starved the one asyncio task carrying BOTH the 10s heartbeat and eager
  serving: 72s of app-level silence, an activity counter frozen 126s, and —
  read at source — a pod that resumed reporting 6 seconds after being declared
  hung, with an evidence counter advancing at 500/s. It was STARVED, not hung.
  The hub-side patience fix was REJECTED and reverted (`ef890253` ->
  `755834a5`): a worker whose reporting its own compute can mute is a broken
  worker, and this is the worker-side fix (WORKER-CONTRACTS.md §1-2).

  **The shape:** on a cache miss, SPAWN a mint process — spawn, never fork,
  because a CUDA context cannot survive `fork()` — which loads what it needs
  itself, builds the cell, writes the artifact and exits. The serving process
  never stops serving eager or beating; on artifact-ready it swaps through the
  ordinary delivered-cell path. No keepalive, no held verdicts, no hub
  tolerance, no magic numbers.

  - `mint_process` — the loop-native supervisor. The whole boundary is one JSON
    request in, one JSON report + one artifact out, so a failed mint is
    reproducible by hand (`python -m gen_worker.mint_child request.json`).
    Nothing live crosses, which is why `multiprocessing` is not used either.
    Every child death is CLASSIFIED and the class drives the retry: a named
    refusal is terminal (re-running it buys a second billed compile for the
    same sentence), a resource shortfall or unclassified crash gets exactly one
    more attempt. **No wall-clock cap anywhere** — liveness is MEASURED from the
    child's process-tree CPU plus the bytes in its capture dir, so a 9.5-minute
    mint is never killed and a wedge dies quickly. Abandonment reaps the process
    GROUP, because inductor forks its own compile workers.
  - `mint_child` — seals the environment as a boot does (a differently-sealed
    child would stamp a cell the parent's own `verify()` rejects), caps its
    VRAM, loads the endpoint's pipeline through the real standalone loader,
    arms COLD and drives the endpoint's OWN derived warm plan. Never
    `mint_artifact`'s producer-style warm call: that is gw#586/gw#587's whole
    lesson, and a cell packed from the wrong graphs bricks every adopting boot.
  - **VRAM co-residency** (`mint_budget.co_residency`) — the child holds its own
    copy of what it compiles, so the ask is `resident weights + ONE activation
    set + inductor workspace + one CUDA context`; in exchange the serving
    process loses the entire in-process capture (`2 * activation + workspace`,
    plus the retained dummy batches the tenant's next peak had to fit around).
    Delegation is therefore CHEAPER for activation-heavy families and dearer for
    weight-heavy ones, which DECLINE — pgw#737's policy, unchanged: eager
    serving, cell absent, a roomier pod mints it. The ask is handed to the child
    as a hard `set_per_process_memory_fraction` cap, so an under-estimate is the
    CHILD's OOM and never the tenant's (the wan-2.2 failure); measured child
    peaks are banked monotonically, so the second ask on a pod is a fact.
  - **Failure inversion** — a dead mint process is a FAILED MINT reported by a
    LIVE worker that keeps serving eager. Nothing raises out of the mint path;
    the classification, the phase it died in and the child's own last words all
    ride the wire as typed `self_mint_abort` / `self_mint_skipped` events,
    because a serve pod exposes no logs.
  - **Two restrictions lift**, both of which existed only because the capture
    was in-process: one live capture per process, and gw#608's seeded-cell gate,
    which had been leaving a slot whose own cell is missing eager for life
    because an unrelated sibling got a delivered cell first.
  - **Cell swap unchanged.** `fleet_cells.adopt_delegated_mint` runs exactly the
    delivered-cell adoption the cache-HIT path runs; `verify()` semantics
    untouched (th#1098 exact identity). In-process capture made the warmup proof
    tautological — the artifact was byte-derived from the execution the proof
    observed. A child-built cell must EARN adoption through the same gates a
    hub-delivered cell does, so a parity gap degrades to eager-with-no-cell
    instead of poisoning the store.
  - Progress reporting comes free: the child's phase frames land on the SAME
    `self_mint_compile` activity the hub already reads, and `activity.watchdog`
    already sums live children's CPU recursively, so
    `evidence:self_mint_compile` advances by itself. No protocol change either
    side of the wire.

  **Proof** — `tests/test_mint_liveness_pgw784.py` drives a real worker (real
  `Lifecycle._heartbeat_loop`, real `Executor`, real gRPC socket) through a mint
  longer than the hub's kill window, sampling beats from the HUB's vantage
  point: the green arm keeps every gap inside 2 intervals against a 6-interval
  kill line with eager jobs completing throughout; the red arm — the same work
  on the loop — blows through the window and stops serving with it, which is
  what makes the green arm believable. Scaled cadence by default, the literal
  10s/140s numbers under `PGW784_REAL_CADENCE=1`.

  A measurement worth keeping: on a 32-core box a pure-Python GIL-holding burn
  in ONE thread stretches a 0.25s nominal beat to 0.256s, and needs ~16
  contending threads to reach 1.17s (4.7x). The incident lost 72-126s. The live
  mechanism is strictly worse than any pure-Python synthetic reproduces, which
  is why the fix moves the compile out of the interpreter rather than trying to
  make it yield.

  **Wired into the executor**, which is what makes any of the above run: a
  compile-cell miss on a delegating boot now reaches `mint_delegate.build_cell`
  instead of the in-process capture. The wiring exposed a defect that would have
  made the feature a no-op — the pending-self-mint recording gate was `if armed
  and selection is not None`, and a delegated arm reports `armed=False`
  truthfully (nothing IS armed; the pipe serves eager), so the mint obligation
  was silently dropped. Delegated pendings are recorded on their own merits and
  deliberately kept OUT of `active_compile_artifacts` — that pipe serves eager,
  and claiming an active artifact for it would advertise bytes it does not serve
  (gw#586) — while still advertising the claimed key ref for th#910's
  self-attested fence. Sibling pipes sharing a key mint their union in ONE child.
  Delegation is eager-first by construction (the two switches move together) and
  `fleet_cells.delegatable()` keeps mandatory quantized lanes and regional
  targets on the in-process capture, since those cannot serve eager meanwhile.
  Delegation is a POLICY of the arming brain, not a caller argument: threading a
  `delegate=` flag from the executor broke every arming double in the suite.

- **pgw#770 — native svdq decoded five of seven tensors in the wrong order; the
  official nunchaku qwen-image artifact rendered noise.** In a nunchaku v1
  checkpoint EVERY tensor is warp-fragment-permuted, not just
  `qweight`/`wscales`. We read `proj_down`, `proj_up`, `smooth_factor`, `bias`
  and `wcscales` verbatim, and `unpack_wscales` split a 128-channel tile
  `(4, 8, 4)` where deepcompressor's `pack_micro_scale` splits it `(4, 4, 8)`.
  Fixed: `unpack_vector`/`pack_vector` (deepcompressor `pack_scale`,
  `group_size=-1`) and `unpack_lowrank`/`pack_lowrank` (`pack_lowrank_weight`)
  added, the micro-scale row split corrected, `decode_linear` applies all of
  them. Measured on the real artifact vs the true bf16 weight: `to_out.0`
  rel-err **1.145 -> 0.070**, `img_mlp.net.0.proj` **1.251 -> 0.056**, fused
  `to_qkv` **1.181 -> 0.063**, bias now exact. Both the blockwise buffers and
  the dense fold reconstruct bit-identically. Evidence: te#137 Run 1b measured
  the official artifact at lpips 0.8215 / psnr 4.64 dB / CLIP 15.26 through the
  old decoder (th#1094's rig scored the same file 0.1050 / 25.20 dB).
  `pack_wscales` also changes, so any artifact written by the te#137 producer
  before this commit decodes wrong — the producer must swizzle the other five
  on write before its next run.
  New `tests/test_svdq_official_layout_pgw770.py` asserts every inverse against
  deepcompressor's forward packers transcribed verbatim, never against our own
  encode side: pgw#685's suite round-tripped our packers against our unpackers,
  which is exactly why a shared wrong convention survived it.

### 0.78.0 detail — pgw#772: the serving lane is deterministic; the voluntary bf16-resident upcast is REMOVED

The gw#534 "rung 2" free-VRAM upgrade (`bf16_resident_fits` /
`BF16_RESIDENT_MARGIN_GB`, and `load_from_pretrained`'s
`allow_bf16_resident_upgrade=` parameter) is deleted — REMOVED outright, not
default-off: any knob that can flip it back per-pod re-forks cell identity,
and a release that genuinely wants plain bf16 residency declares it (bind the
bf16 flavor / `storage_dtype=""`) instead of probing for it. A declared fp8
storage lane is now served as fp8 storage on every card, so the serving lane
— and with it the ck5 `lane` axis, the ONLY GPU-dependent axis in the key —
is a pure function of (release × declared config), never of the individual
card's free VRAM.

Why (th#1198 CP-D wire evidence + pgw#727 numbers): on the same release,
image, and sm_89, an RTX 4090's ~1.5 GiB `mem_get_info` surplus over an L4
passed the probe and silently moved it to base lane `""` — a lane nothing
mints for — so it missed all 144 published checkpoints INCLUDING its own
same-SKU cell and served eager for life (the L4 armed and banked −21%
request-level). The cast tax the upgrade dodged re-measured **+1.9%** for the
structural storage lane (pgw#727; the +44–73% that justified it measured the
retired hook form), so VRAM-rich cards were paying ~2x weight VRAM and
forfeiting the compiled win for a 1.9% saving.

Unblocks BOTH halves th#1198 found broken: JIT cross-SKU key convergence
(pull-by-key can now match across cards) and production-path AOT adoption on
higher-VRAM cards (pgw#765 removed the `sku` pin; this removes the `lane`
fork — together they make cross-SKU adoption real end-to-end). Lane
populations converge; NO key-scheme change (identity axes untouched — cells
minted at the declared lane become adoptable by every card of that config).

Involuntary transitions are PRESERVED and red-tested: the fit ladder's
can't-fit rungs (runtime fp8-E4M3, emergency nf4) and the w8a8/w4a4
dequant-on-unsupported-host lanes are declared rungs, structurally reported —
degradation is fine; a probe deciding identity was the bug.
`tests/test_lane_determinism_pgw772.py` carries the standing guard: same
declared config under 4 GiB vs 999 GiB of mocked free VRAM must compute
identical requested cell keys (red on the pre-fix tree), and no adoption or
identity path may take a live device measurement as a hash input.

## 0.77.0 (2026-07-29) — everything since 0.76.8: the 0.77–0.83 chaos span ships as ONE release

MAPPING: the chaos labels 0.77.0–0.83.0 below (entries re-marked "stamp — SHIPPED IN
0.77.0") were version claims that never reached PyPI — nothing installable ever existed at
any of them. Any reference elsewhere to "landed as 0.79.0" / "0.83.0 carries th#1283"
means: first installable in THIS release. Previous published version: 0.76.8.

Aboard, by chaos label:
- 0.77.0 th#1257 serving-task declaration; 0.77.1 pgw#738 upload-park fix + publish phase
- 0.78.0 th#1276/pgw#753 ref grammar defaults to `prod`
- 0.79.0 pgw#755 forward AWQ W4A16 encoders (unblocks deleting the conversion
  endpoint's vendored `awq_forward.py` at its next pin bump)
- 0.80.0 pgw#756 guard-closure classifier veto removal
- 0.81.0 pgw#758 multi-graph cells; 0.81.1 pgw#733 typed adopt/arm wire events
- 0.82.0 pgw#748 phase 0 multi-GPU bookkeeping; 0.82.1 pgw#760 typed fail-soft events;
  0.82.2 pre-clamp AOT cell retirement; 0.82.3 nested added_cond bind; 0.82.4
  aot_forward envelope parity (also backported to 0.76.8)
- 0.83.0 th#1283 worker half, per-intent fail-closed

Plus unlabelled chaos work since those stamps: pgw#748 phase 1 (sequence-parallel
runtime), pgw#761 standalone-component subfolder ingest, pgw#763 host-move guard +
typed boot-phase errors, pgw#764 boot-milestone dedup, and the pgw#740 registry
contract gate — this is the first release published through that gate (clean-venv
install of the built wheel, bare-state + consumer-declaration contract asserted
before upload).

## 0.83.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — th#1283 worker half: per-intent fail-closed

The hub now declares `mandatory` per intent instead of a blanket command flag,
and `apply_command` matches. Errors on mandatory intents still reject and latch
(`protocol_rejected`). A command-level error latches only when rejecting would
abandon mandatory work not already registered under the same identity — so a
hub-side `COMMAND_SEQ_CONFLICT` over already-registered work rejects the resend
without bricking the process. Errors scoped to advisory intents no longer
reject the command at all: exactly those intents are declined (typed FAILED
IntentState + `rejections` on an ACCEPTED receipt) and the rest applies, so a
bad preposition or an unknown-function binding leaves the worker SERVING.
HOLD RESOLVED (2026-07-29, pgw#740 determination): the empty registries that held
this tag are the DESIGN — vocabularies are endpoint declarations (pgw#740/#739),
verified per-registry with stated-intent commits and merged adoption halves
(te#130/ie#567). Nothing was defective; ships in 0.77.0 through the new
registry-contract publish gate.

Live-named twice on the 0.76.7 canary: with bind fixed, in-contract calls EXECUTED the
adopted artifact and the caller crashed downstream (4-vs-2 broadcast) because diffusers
calls `unet(..., return_dict=False)[0]` on the raw-tensor return. The wrap restores the
declared envelope. Shipped on the rerun line as 0.76.8.

## 0.82.3 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — nested added_cond input resolution at bind (arm-events lane)

Live-named on the 0.76.6 canary (pod ae2uc81yub0gyq): the FIRST successful cross-pod AOT
arm (`aot_adopt armed`) then refused every real call with `aot_ingress_refused
input_missing: text_embeds` — the export flattens `added_cond_kwargs` entries into declared
inputs, but every diffusers caller passes them nested in one dict kwarg, and
`bind_call_inputs` never looked inside. Resolution: keyword -> position -> inside any
mapping-valued kwarg -> optional -> named refusal. Shipped on the rerun line as 0.76.7.

## 0.82.2 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — discovery retires pre-clamp AOT cells (arm-events lane follow-up)

Live-proven on the 0.76.6 canary (pod 3cjmd3ohuk98a5, first `aot_adopt` wire rows): a cell
minted before the pgw#754 host-ISA stamp carries no metadata requirement, passes every
discovery gate, gets downloaded, then refuses at stage with `host_isa_unsupported` (the
.pt2's own torch package stamp — built x86-64-v4). Unstamped cells are structurally
unadoptable across a mixed fleet: `aot_cells._candidates` now retires them by name instead
of shipping doomed candidates. Also corrects the verdict lane's CP4 "unstable rows[0]"
finding: the pick variance was the ISA filter reacting to per-host CPU capability
(AVX-512 vs not), not ordering instability — ordering was already deterministic.

## 0.82.1 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#760: swallowed-error audit — important fail-soft outcomes ride typed events

Doctrine (Paul, verbatim in spirit): errors should be exposed to the orchestrator so it
can report on them. The pgw#733 incident class — a classified reason reduced to a local
`logger.warning`, structurally invisible on hub-spawned workers — was audited across the
whole SDK (840 swallow-shaped except handlers reviewed). Nine seams affected serving,
placement, health, or hub decisions with nothing on the wire; each now emits a typed
event through the existing th#1250-persisted activity pipe. One shape everywhere:
`emit_event(kind, detail, phase=reason)` — `phase` a stable countable reason token,
`detail` the identifiers + exception. **No control flow changed anywhere** — every
fail-soft stays fail-soft; this release is visibility only.

- `trt_engine.enable` — classified `AdoptError` refusals and successful arms ride
  `trt_adopt` (phase = the reason, mirroring pgw#733's `aot_adopt`); a mid-serve engine
  failure that permanently reroutes to eager rides `serve_degrade/trt_runtime_failed`.
- `hot_swap` — signature-vocabulary explosion (permanently disables concurrent routing),
  background warm/heal compile failure (the outcome the `guard_miss` event's
  `heal=healing` promise never reported), warm-worker crashes, and failed cell-republish
  callbacks (fleet re-compiles that shape forever) all ride `serve_degrade`.
- `utils/lora` (`lora_hygiene`) — failed deactivate/branch-clear (possible cross-request
  adapter bleed), failed detach-on-demote, failed LRU eviction (VRAM creep).
- `models/residency` (`residency_fault`) — a move whose rollback also fails leaves a
  mixed-device unusable pipeline (the next forward fatals mid-denoise): named with the
  ref at fault time, not discovered via the downstream job error. A failed residency
  event callback (hub view silently diverging) confesses on the independent channel.
- `preload` (`rotation_preload`) — a stage failure abandons the hub's desired-hot plan
  for the whole generation; a driver crash parks the entire subsystem.
- `capability_renewal` (`capability_renewal`) — terminal denial and silent retry
  exhaustion, previously surfacing minutes later as a bare expired-token upload error.
- `models/lane_gate` — serve-time CPU-offload engagement and a failed gate wrap (silent
  loss of the te#79 promote-on-use protection) ride `serve_degrade`.
- `models/loading` — a PARTIAL fp8 cast failure (one component of N) returned
  `applied=True` and evaded the th#737 structural report entirely; per-component
  failures now ride `serve_degrade/fp8_cast_failed`.
- `runtimes/server` — an engine that boots on a degraded rung (fewer GPU layers /
  CPU-only) rides `serve_degrade/engine_boot_degraded`.

Red-verified in `tests/test_error_visibility_pgw760.py`: forced failure at each seam →
captured `ActivityUpdate` naming the reason class and identifiers, with the fail-soft
behavior asserted unchanged.

## 0.82.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#748 phase 0: multi-GPU bookkeeping that is wrong today

Sequence parallelism's phase 0 ships alone, before any parallelism code, because both
halves are latent bugs on the fleet we already run.

- **`DeviceGroup` gains a placement mode; replicated groups budget against MIN, not SUM.**
  `free_vram_bytes()` summed unconditionally across a group's devices — correct only when
  the WEIGHTS are split. Context/sequence parallelism replicates weights and shards
  activations, so a 2x24GB replicated group reported 48 GB free and would admit a 30 GB
  model that fits on neither card: pgw#648's original all-device-sum bug reproduced one
  level up. `DeviceGroup(devices=(0,1))` now defaults to `placement_mode="replicated"` and
  reports `min` over members (a device the host does not have contributes 0, which
  correctly makes the group unusable); `placement_mode="sharded"` keeps the sum for a
  future TP mesh. `fits()` and `make_room()` both read through it, so admission changes
  with it. Single-device groups are identical under both modes.
- **`host_canary` gains a 2-GPU leg** (`interconnect`, `peer_gbps`, `peer_access`,
  `topo_link` on `HostCanary`, tags 10-13). The hub can tell SXM from PCIe by SKU
  identity; only the pod can say whether ITS two cards have peer access, and that is what
  decides whether a sequence-parallel release meets its latency SLO. The leg classifies
  the fabric (`nvlink | pcie-p2p | host-staged`) from `nvidia-smi topo -m` +
  `can_device_access_peer` and times a device-to-device copy of the same 256 MiB buffer
  the other legs use. Two classification rules, both measured on real pods rather than
  assumed: **peer access overrules good-looking wiring** (2x RTX 4090 report `NODE` and no
  P2P — 1.96 GB/s, identical with `NCCL_P2P_DISABLE=1`), and **`SYS` overrules peer
  access** (2x H100 PCIe report `SYS` and peer access TRUE, yet achieve 14.5 GB/s
  bit-identically with P2P disabled — the flag buys nothing across CPU sockets, so the
  class is host-staged, not pcie-p2p). Inert on 1-GPU pods: it never runs, and the fields
  stay empty.
- **`measure_peer_collective()`** is the deep leg, never run at boot: a real NCCL
  `all_to_all_single` across N spawned ranks on the production activation shape
  `[1, 40, 37800, 128]` bf16, optionally under `NCCL_P2P_DISABLE=1`.
  `python -m gen_worker.host_canary` runs everything and prints JSON.
  Measured 2xH100-80GB-HBM3 (RunPod Secure, NV18): 0.71 ms/call, 272.6 GB/s achieved
  over-link, 113.6 ms per model call's worth of collectives; the same pod with P2P
  disabled: 8.55 ms/call, 22.6 GB/s. The cheap boot leg's `peer_gbps` reads ~1.42x the
  achieved collective bandwidth — an upper bound, with the classification exact.

## 0.81.1 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#733 arm half: every AOT adopt/arm outcome is a typed wire event

The AOT verdict lane's blocker: cross-pod adopts fail inside stage/bind/arm with a
classified `AdoptError` reason that `aot_serve.enable` / `fleet_cells.enable_compiled`
reduced to `logger.warning` — invisible from hub-spawned workers (no stdout). One event
class, `aot_adopt` (`aot_serve.ADOPT_EVENT`), now carries every outcome on the th#1250
persisted-event pipe: `phase` = `armed` on success or the classified refusal reason
(`host_isa_unsupported`/`key_mismatch`/`artifact_invalid`/`constants_*`/...), `detail`
names the candidate cell (family + key, best-effort even on unreadable artifacts). The
F1 consumer additionally binds fall-through outcomes to the DISCOVERED candidate's
identity: `did_not_arm`, `armed_other_path`, `lane_unavailable` rows carry key + ref.

## 0.81.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#758: multi-graph cells — every declared graph class in ONE .pt2

Paul's ruling: "generate and generate_turbo are separate functions, they have separate
graphs, but they are COMBINED TOGETHER INTO ONE FILE." One mint invocation now produces ONE
cell per (family x lane x contract) carrying the family's WHOLE declared class set as named
AOTI models (`data/aotinductor/<entry>/` — mechanism verified on the 2.13 pin: per-entry
constant tables, per-entry bind, per-entry B1 segfault). Removes the pilot runbook's
one-artifact-per-pod serving ceiling: sdxl's 18 declared classes (9 aspects x cfg/no-cfg)
collapse from 6-18 artifacts to one, and a single resident cell serves generate AND
generate-turbo compiled.

- **Envelope format 2** (`aot_serve`): metadata carries an `entries` map — per-entry
  target/fork/class-dim coordinate, ingress contract, constant manifest, graph block,
  `range_digest`, `class_hash`. Literals namespace as `<entry>::<fqn>`. Entry names derive
  from declaration coordinates (`unet/cfg=true/B=2,...`) — stable across mints.
- **The pgw#716 key formula, implemented as anticipated**: `combined_graph_hash` = first 16
  hex of sha256 over the newline-joined SORTED per-class hashes; per-class hashes ride
  metadata so a mismatch NAMES the class; each class hash folds its entry's range digest
  (the measured node-only-collision fix). Contract facts v1 -> v2 — **this re-keys and
  RETIRES every published format-1 aot-inductor cell** (ck5 exact identity: correct and
  expected; dynamo cells untouched). TRAIN CAVEAT: do not run mixed-SDK-version double-mint
  experiments across this release — per-release fleet pinning already prevents it.
- **Mint** (`aot_mint.mint(pipeline, spec, out)`): enumerates `cell_plans` across ALL
  declared targets; every gate (declared-range, lifted-input, no-baked-adapter, B1
  code-only, bindability, constant-set drift) runs PER ENTRY and refuses naming the entry
  AND the cause. Requests name a family only — coordinate-shaped requests are refused.
  LoRA buckets scope per target by composed truth (`branch_targets`): wan's vae.decode
  entry mints bucket-0 beside its bucket-128 transformer.
- **WARM CANON EXECUTED**: `warm_changes_key=True` families get their declared pre-warm RUN
  before export (previously keyed but never acted on); a failed warm is a named refusal.
- **Serve** (`aot_serve.load_and_wrap`): one staged artifact, EVERY entry bound before ANY
  wrap, per-module `EntryDispatch` routes each call to the entry whose declared ingress
  contract admits it (0 admitting = named refusal + eager; >1 = `entry_ambiguous`). Dotted
  targets (`vae.decode`) wrap their owner method. Multi-target markers behind the same
  `is_armed`/`execution_count`/`proven_since`/`unwrap` surface.
- **Mint-phase telemetry** (#757's instrument-first doctrine): every mint records
  `mint_phases` — per-entry export/compile/warm seconds plus labeled inductor phases
  (lowering / codegen / triton / HOST C++ COMPILE+LINK — measured dominant on CPU tiny
  models at ~9.5s of 11s, the 3.9x suspect), graph-class count, and the autotune posture —
  and emits a typed `aot_mint_phases` event.
- **#724 REJECTED fallout** (Paul): no dedicated mint fleet — serving pods background-mint
  under the pgw#677 eager-first machinery; `python -m gen_worker.aot_mint` stays for
  ops/testing. Docstrings no longer claim "serving pods never compile".
- Removed: `aot_mint.mint_target`, `compile_package`, `identity_blocks` (split into
  `compile_entry_files`/`package_cell` + `shared_identity_blocks`/`entry_graph_block`),
  `aot_declaration.apply_declaration` (the CLI derives whole-cell plans now).
- Mint default `compile_threads=4` (`MINT_COMPILE_THREADS`; #757 MEASURED: 32 -> 4 is FREE,
  -2% wall clock — same speed, less CPU contention for background mints on serving pods;
  identity-inert per #757's re-key pre-verification, caller override wins). The ONE resolved
  inductor config every entry compiles under is recorded verbatim in `mint_phases`
  (`inductor_configs`) — #757's open per-call seal-bypass concern is auditable there.

## 0.80.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#756: the guard-closure classifier loses its veto

Paul's ruling. `guard_closure` extracted dynamo's guard tree post-mint, classified every
guard against the declared contract, and **refused the mint** on anything it could not
classify. It is now ADVISORY: it extracts, classifies, records, emits a countable
`guard_leak` event, and the mint CONTINUES.

**Why.** The gate protects against WASTED REUSE, not incorrectness. Dynamo re-evaluates
these very guards on every call at the consumer, so a cell depending on unpinned state
fails its guards THERE — and with pgw#680 fail-on-recompile armed, that raises, serves
eager, and reports the reason. A missed leak degrades gracefully and loudly; it can never
produce a wrong result. A CLASSIFIER BUG, by contrast, refuses every mint on every family
— which happened twice fleet-wide (pgw#691's NO_TENSOR_ALIASING root dispatch, pgw#733's
`_source_root` prefix match) while the gate caught zero real leaks in production. The risk
profile was inverted.

- **`assert_closure` -> `closure_manifest`** (no alias — the name lied). Same extraction,
  same classifier, same manifest, same `consolidate` fleet audit, same
  `python -m gen_worker.guard_closure` CLI and exit codes. Only the veto is gone.
- **`activity.KIND_GUARD_LEAK`** — one countable event per minting pod naming the suspected
  variables, so a real leak class surfaces as a hub-side trend instead of a fleet outage.
- **Manifest v3**: `+ "unproven"` rows (a cache entry whose guards could not be walked — a
  fact about the torch guard debug surface, not about the mint) and `+ "gate": "advisory"`.
- **Refusals RETAINED**, each proving a defect rather than inferring one: *no compiled
  graphs at all* (nothing compiled — the cell would be empty) and *non-canonical process
  posture* (pgw#695 — a measurement against exact canonical values). `canonical_ingress`'s
  stride/dtype boundary errors (ie#544) are a different mechanism and are untouched.

**Never rebuild this classifier.** torch's own `guard_filter_fn` / `GuardFilterEntry` is the
supported structured hook — typed entries instead of the C++ guard-manager walk plus repr
parsing where BOTH fleet-wide bugs lived — and upstream's precompile work already maintains
a versioned `UNSUPPORTED_SERIALIZATION_GUARD_TYPES` classification. If a JIT-resident family
ever needs the check, it is a thin call into those, never a hand-rolled grammar.

**Step 2 (ratified, gated on the AOT migration):** when the last family migrates to AOT,
`guard_closure.py` is DELETED outright — exported artifacts carry no dynamo and no guards at
serve time, so the module becomes dead code. Tracked as pgw#756's second checklist.

**Rollout note:** manifest v2 -> v3 changes `manifest_digest`, so two pods minting the SAME
cell key on DIFFERENT SDK versions will report unequal manifest digests to the pgw#711
confirmation gate. Do not straddle this version while double-minting a key.

## 0.79.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#755: forward AWQ W4A16 encoders (te#137 producer)

`svdq_awq` gains the production FORWARD path its decoders invert: `pack_w4x16`
(tinychat pack_w4), `apply_adanorm_splits` (adaLN row interleave + bias+1), and
`encode_awq_linear` (asymmetric per-group minmax int4, zeros stored pre-scaled
AND pre-negated with upstream's exact double-rounding, grids padded to
ceil_num_groups). Adapted from deepcompressor (Apache-2.0); bit-exactness vs
the vendored upstream exporter asserted in tests. Consumed by conversion's
`svdq_produce` (te#137) to emit our own nunchaku-v1 checkpoints.

## 0.76.5 — pgw#754 host-ISA portability (the AOT SIGILL fix) + pgw#752 + th#1259 SDK half

One patch train, AOT-flip-critical. Ships as 0.76.5: th#1259's stamp was the
highest aboard; pgw#752's 0.76.4 claim is ABSORBED into this release (its
section below) — no version hole.

- **pgw#754 (the headline)**: AOT cells no longer SIGILL foreign hosts. Root
  cause: the mint host compiled its .pt2 wrapper code with -march=native
  (AVX-512 on the mint host) while torch hashes cpp.march=None identically
  everywhere — the seal was blind to real machine-code ISA. Host compiles now
  CLAMP to x86-64-v3 (measured perf-neutral: the wrapper has zero vector
  instructions), the clamp is SEAL-VISIBLE (SEAL_VERSION 5 -> 6), discovery
  filters unexecutable cells BEFORE download, and serve refuses by name before
  dlopen. CONSEQUENCE: seal_v6 retires ALL pre-clamp cells (JIT and AOT,
  including the prod-recipe AOT cells and canary cells) — remint required, by
  design (the recipe changed).
- **pgw#752**: clean page cache is RAM the next load can have (capability
  probe + residency accounting; see its absorbed section below).
- **th#1259 SDK half**: a ref the PAYLOAD named fails the REQUEST
  (typed payload-ref provenance), never the release — the worker-side half of
  the breaker-poisoning fix (the hub half rides a hub train).

## 0.78.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — th#1276/pgw#753: the ref grammar's default tag is `prod`, not `latest`

Paul's ruling. A bare `owner/repo` now means `owner/repo:prod`. `prod` is the STABLE
SERVING pointer, which only moves on an explicit promote; `latest` is the MOVING PUBLISH
pointer that the finalize path auto-binds on every publish, and it is now an ordinary tag
that must always be written explicitly. The normal form elides `:prod` and stamps every
other tag verbatim, `:latest` included. `latest` is not deprecated — it is de-defaulted.

- **`gen_worker.models.refs.DEFAULT_REF_TAG`** is the one literal, the twin of tensorhub's
  `refgrammar.DefaultTag`. Every grammar-coupled site references it — parse default,
  `canonical()` elision, `ModelRef` coercion and `.label`, the `Hub()` `tag=` default,
  `wire_ref`, discovery-manifest elision, the CAS ref-map path — so grammar sites stay
  greppable and stay distinct from code that means the `latest` publish tag.
- **The shared conformance fixture was decorative.** `tests/testdata/ref_grammar_vectors.json`
  is vendored byte-identically in tensorhub and was referenced only in comments: no test in
  either repo loaded it, so the th#597 C5 contract had been unenforced since it was written
  and the two parsers could have drifted silently. `tests/test_ref_grammar_conformance_th1276.py`
  (and its Go twin) now assert every vector's fields, canonical form, and that the canonical
  form is a fixed point.
- **Bug this surfaced**: `Hub("owner/repo", tag="latest")` was silently converted to the
  default — `wire_ref` dropped the tag when it equalled the old default and `fold_ref`
  re-applied the parser default, so an explicit `latest` pin never survived. It is now
  stamped verbatim, with a round-trip test locking it in.

**Rollout**: do NOT upgrade workers independently of the hub. Bare refs cross the wire and
the receiver re-parses them with its own default, so a mixed-version fleet can disagree on
what a bare ref means — silently (wrong checkpoint served) or loudly (snapshot-key miss).
See th#1276 for the scenarios and the durable fix.

## 0.77.0 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — th#1257: handlers declare the SERVING TASK they perform

The hub's human quality sign-off for a quantization lane is no longer keyed by model family
alone — it is keyed `(family, task, variant, lane, quant_method)`. A fp8 approval earned on
text-to-image says nothing about image-edit on the SAME weights: an edit path must preserve
an input image, and flux.2-klein-9b serves both from one checkpoint. The task therefore
cannot be inferred from the model; the handler declares it.

- **`@worker_function(tasks=(...))`** beside the existing `objectives=` / `distilled=`,
  riding the same path into the requirement payload (`fn["tasks"]`). Vocabulary in
  `gen_worker.api.slot.TASKS`, mirroring tensorhub's `internal/modeltask` exactly.
- Several tasks means one handler INPUT-ROUTES them (a merged `generate()` that does
  text-to-image with no image and image-edit with one). The hub requires every declared
  task to be approved before the lane is auto-eligible.
- Declaration is strict — canonical spellings only, separators folded (`image-edit` ->
  `image_edit`), unknown/empty/duplicate rejected at decoration time. The hub still folds
  legacy aliases on read; authors write the canonical token.
- **Omitting `tasks=` is meaningful, not neutral**: an undeclared handler resolves NO quant
  approval and serves at base precision. That is fail-closed by design — it degrades, it
  never strands — but every serving endpoint should declare.
## 0.76.3 — the reuse wave: adopt-without-mint unblocked + AOT flip seams

The four structural bugs that held cell adoption at zero in the sdxl 0.2.14
four-leg proof (verdict: "airtight cells, proven; adoption economics: not
yet"), plus the flip-critical AOT serve seams.

- **pgw#745**: host driver libs (libcuda.so.*, libnvidia-*) excluded from the
  env_seal lib manifest — cell keys stop fracturing per RunPod driver cohort
  (gw#577: driver is never identity). seal_v4.
- **pgw#749**: the identity lib manifest is the userspace toolchain ON DISK
  (torch/triton/nvidia package roots, content-digested), never the
  phase-dependent /proc/self/maps snapshot — cold-boot candidate keys now
  equal published keys, so boot-attach adoption can fire. seal_v5. The maps
  probe stays as the live substitution-refusal surface.
- **pgw#751**: adoption keeps LOCAL bytes on same-key byte-divergent cache
  members (bytes are not the identity — #699/#711 respec); warm pods can
  install cells. Structural conflicts still refuse typed.
- **pgw#750 (task 1)**: the off-vs-vae_only resident refinement keys on TOTAL
  card capacity (per-SKU constant), never marginal live free VRAM — the mint's
  traced graph class and object set are deterministic per SKU.
- **pgw#722 pilot seams F1/F2/F3**: flag-gated AOT serve flip
  (`GEN_WORKER_PREFER_AOT`, default OFF) — discovery/adopt of exported cells,
  lifted-LoRA arm order, binding-routed adapter swaps. Seal-digest invariant.
- **pgw#747**: a bare-typed auxiliary slot emits `family=""` — family-agnostic
  artifacts (RIFE, upscalers) become bindable.
- **pgw#743**: a proxy-shaped answer is not a hub verdict at ANY status
  (+ convert keepalive). [Relabelled from 0.76.2 — it missed that wheel.]
- `equivalence.py` deleted (deliberate re-land of the chaos deletion; Paul's
  ck5/ck6 exact-identity ruling).
- NOT aboard: pgw#752 (rides 0.76.4); pgw#735 boot-proof gap fix (flip-prep
  lane; the flip smoke notes the gap until it ships).


## Unreleased — C2PA signing moves hub-side (th#1307)

- **The C2PA private key never enters a pod (th#1307).** The hub used to inject
  `GEN_WORKER_C2PA_KEY_PEM` into every tenant pod and this process signed
  in-process — tenant code shares the process, so one `print(os.environ[...])`
  leaked the platform-wide leaf signing key. Now the worker holds only the PUBLIC
  chain and signs through the hub: `content_credentials` installs a c2pa-rs
  callback signer that POSTs the claim's COSE to-be-signed octets to
  `/v1/worker/c2pa/sign` (worker-JWT authenticated, armed at HelloAck alongside
  the cell-receipt gate). No media leaves the pod; a claim is ~1 KiB regardless
  of asset size. `GEN_WORKER_C2PA_KEY_PEM` / `_KEY_PATH` are now REFUSED at
  configure() and no longer exist as Settings fields, so a hub regression kills
  the pod loudly instead of quietly re-creating the leak. Every failure mode
  (unarmed signer, hub refusal, hub unreachable) RAISES: the request fails
  rather than shipping media that looks signed and isn't.

## Unreleased (ck5 interim -> ck6 design) — exact recipe identity; equivalence machinery deleted

Paul's exact-identity ruling chain (design of record: tracker pgw#716). This ships
the ck5 INTERIM scheme; ck6 (graph-hash identity) follows per pgw#716-#720.

- **KEY_SCHEME ck5**: the key is a recipe digest — format/kind/family/lane/mode/sm/
  contract/env_seal/**toolchain**/**code_closure**. Every VERSION axis left the key
  (torch/triton/cuda/gen_worker/diffusers/transformers/image_digest -> metadata
  observability; content rides `toolchain` — dist-info RECORDs incl. diffusers/
  transformers/peft + bundled ptxas/nvdisasm binary hashes — and
  `code_closure` — the static import-graph closure from the compile entrypoints,
  AST-resolved, sound under the root-imports convention). ck2/ck3/ck4 keys are dead.
- **Equivalence adoption DELETED** (`gen_worker/equivalence.py`, its tests, the
  designated-axes machinery): exact identity needs none of it. Kept as TRUST:
  pgw#711 publish-complete digests (blake3 artifact + sha256 manifest), pgw#712
  no-republish fence (`fleet_cells.ADOPTION_MARK`, defense-in-depth), toolchain
  digests, `guard_closure.manifest_digest`.
- `static_code_closure()` + `closure_completeness_gap()` retarget as the pgw#717
  tier-2 recipe HANDLE and pgw#720 deferred-memo diagnostics (the completeness gate
  was briefly a mint gate; removed — it false-fired on executor-side models/*
  modules, recorded on pgw#720).
- `cell_key.compute` drops `image_digest`, gains `closure_roots` (endpoint modules;
  executor wiring rides the train lane).

### Erase-and-impose env contract + seal v3 (pgw#718/#719)

The worker OWNS its process environment. We no longer audit the world's env vars and
refuse on surprises (the superseded pgw#696 allowlist — it bit a 0.70.3 boot on an
informational base-image var); we ERASE and IMPOSE.

- **`env_seal.scrub_env()`** — unconditionally deletes every var in the behavior
  namespaces (`TORCH*`/`PYTORCH*`/`TRITON*`/`CUBLAS*`/`CUDNN*`/`NVIDIA_TF32*`/`OMP_*`/
  `MKL_*`), known or unknown, BEFORE torch imports; logs the erased names; never fails.
  Plumbing (CUDA_VISIBLE_DEVICES, paths, credentials) untouched.
- **Typed knobs only** — `establish_config(overrides=...)`; a scrubbed var that turns
  out to be needed becomes a knob, never an unscrub.
- **CANONICAL_CONFIG is now the pgw#654 SERVING posture (TF32 on / precision "high")**.
  This fixed a REAL latent bug the drift check surfaced: mints sealed TF32-OFF while
  serving ran TF32-ON, diverging the inner FX key — **every sealed cell was unhittable
  in serving**.
- **Seal v3** — loaded-library manifest (`/proc/self/maps` -> the native `.so` set,
  content-digested; closes the LD_PRELOAD/LD_LIBRARY_PATH hole), FROZEN AT BOOT so lazy
  `dlopen` growth cannot re-key mints while substitution is still caught at point-of-use;
  hash-seed facts recorded.
- **Boot-vs-point-of-use** — `assert_seal_unchanged()` before every mint (all three mint
  paths) and the config half of the arm-time `artifact_drift` check: endpoint code
  mutating config behind our back is a NAMED refusal, never a silently different graph.

### ck6 canonical graph identity — the hashing half (pgw#716)

- **`gen_worker/graph_hash.py`** (new): one canonicalizer, two ingest paths — dynamo
  `fx.GraphModule`s and `torch.export.ExportedProgram`s. Scrubs node/arg names, all
  provenance meta, symbol names and device index; hashes ops, connectivity, literal
  args, tensor meta, the export signature and pytree specs. **Weight VALUES never
  key** (a fine-tune must share the graph).
- **Symbolic-dim RANGES are in the hash** — the pgw#704 S8 soundness fix: three sdxl
  exports differing only in declared range produced ONE node-only digest
  (`9dd33abbc7617d98`), which would let a worker adopt a cell that refuses the traffic
  its key promised. Covers the dynamo path (`ShapeEnv.var_to_range`) as well, since the
  defect is identical for declared dynamic dims (pgw#702).
- `combined_graph_hash()` — first 16 hex of sha256 over the newline-joined SORTED
  per-class hashes; order-independent by construction.

### fp8 storage is module STRUCTURE, not a cast hook (pgw#727, + pgw#726 slots)

The w8a16 lane's weights still RESIDE in fp8 and still compute in bf16 — but the
upcast now happens at the use site inside `forward` instead of by mutating
`Parameter.data` at the forward boundary. Measured (pgw#704 S12-c, L4, real SDXL
UNet): the hook form is compile-hostile (dynamo 386.5 vs eager 385.9 ms — a 0.2%
regression for a 38.9s mint), the structural form is **14.8% faster under dynamo**
and `torch.export` accepts it. This lands independent of the AOT migration.

- New `models/fp8_storage.py`. `apply_fp8_storage` no longer calls diffusers'
  `enable_layerwise_casting` for denoisers; it restructures the cast-eligible
  leaves (class pun: `nn.Linear` -> `Fp8StorageLinear(nn.Linear)`, same object,
  same FQNs, same state_dict keys, `isinstance` intact). Transformers text
  encoders KEEP the `_Fp8WeightWindow` block hooks — they read weight dtype
  outside the owning leaf (gw#460) and are not on any compiled path.
- **Coverage is upstream's rule, mirrored and asserted** — same leaf set as
  `_apply_layerwise_casting` (supported layers, default + model-declared skip
  patterns, peft adapter names), with a set-equality tape against the installed
  diffusers so upstream drift fails in CI, and a by-name refusal for any leaf
  kind we cannot restructure. Narrower coverage would be a silent VRAM AND
  numerics change.
- **The `fp8-hooks` lane VALUE is unchanged** (wire-compatible; tensorhub maps it
  to `w8a16`). The traced graph DOES change, and it shows up where it should:
  module types and hook counts in `execution_contract` — new cell keys, no
  cross-lane adoption in either direction.
- fp8 weights are BUFFERS now, not parameters — diffusers resolves
  `ModelMixin.dtype` from the first floating-point parameter and special-cases an
  armed cast hook; drop the hook with fp8 parameters left behind and `model.dtype`
  answers fp8, which breaks every denoiser that casts to `self.dtype` in forward
  (`UNet2DModel.get_time_embed`, measured). Buffers restore the correct answer
  with no hook and no property override.
- Measured on CPU against the hook lane, real tiny UNets: identical coverage set,
  **bitwise-equal outputs**, coverage parity on resident bytes. pgw#704's "+11.6%
  VRAM" is not usable either way — that prototype swapped `nn.Linear` only,
  leaving convs bf16-resident.
- **The pun must rebind the outgoing Parameter, or VRAM goes UP ~50%.** A class
  pun replaces the weight tensor OBJECT, so anything still holding the original
  `Parameter` — accelerate device hooks, `low_cpu_mem_usage` bookkeeping, any
  earlier `list(model.parameters())` — keeps the bf16 storage alive next to the
  fp8 copy. An L4 measured fp8-storage 7.35 GB vs plain bf16 4.89 GB (**+50.3%**,
  both copies resident); reproduced on CPU at +49.9%. `_to_storage_buffer` now
  rebinds the outgoing Parameter onto the fp8 storage, restoring the hook lane's
  property that every holder follows the cast. A module-only residency walk
  cannot see this failure — the tape now holds the parameters the way a pod does.
- `restructure_fp8_storage` releases the freed bf16 blocks to the driver
  (`empty_cache`, never initializing CUDA): the fit ladder reads driver-level
  free VRAM, so ~half a denoiser sitting in torch's caching allocator would make
  the rung decision moments later see free VRAM as taken.
- Latent bug fixed on the way: `apply_block_window_offload` re-moved just-parked
  weights back onto the device whenever they were buffers (i.e. the w8a8 lane
  today), so the degraded rung silently saved nothing.
- **pgw#726**: `_Fp8ScaledLinear.lora_a`/`lora_b` are DECLARED `None` buffer slots
  instead of plain attributes — `register_buffer` no longer has to pop `__dict__`
  to get its tensors in, the FQNs are structural from construction, and a
  branch-disable cycle keeps the slots declared.

## 0.77.1 stamp — SHIPPED IN 0.77.0 (chaos label; kept for the chaos record) — pgw#738: an upload never parks a job forever; the publish phase becomes visible

Root cause of te#125's silent deaths (62922680/d0cbf910: one evidence blob persisted, then
3h51m of heartbeating silence on a billing H100): admission took GPU permit -> instance
run_lock, but `save_bytes` yields the permit mid-handler while HOLDING run_lock — a second
job packed on the worker took the freed permit, blocked on run_lock, and the uploader
blocked forever in `reacquire()`. Classic ABBA inversion; no HTTP error ever happened.
(The transport half of this class — origin discrimination + silence-bounded retries in
`convert/hub.py` — shipped from this lane's WIP via pgw#743 in 0.76.2.)

- **Lock order flipped: run_lock -> GPU permit -> turn_mutex** (tenant path and the
  pgw#677 background-turn path). A job parked behind a busy instance holds NO permit, so
  a yielded permit always comes back. Bonus: a parked job no longer pins the scarce GPU
  permit while it waits.
- **`reacquire()` is bounded and typed** (`GpuSlotReacquireTimeout` -> RETRYABLE, 30 min):
  any future inversion fails the job loudly instead of parking it while the GPU bills.
- **Never-silent guarantee**: a done-callback reaper on the job task emits a terminal
  RETRYABLE JobResult if the task ends without reporting — the pgw#738 face where a job
  goes back to `queued`/`assigned` with a full verdict banked (3318d70e) instead of
  reaching terminal.
- **The publish phase emits**: `publish_flavors` logs phase transitions + throttled part
  progress through ctx.log (the same channel watchdogs read) and feeds the activity
  proof-of-life beat; the R2 SDK-grant boto3 lane forwards byte progress (it was
  progress-blind); botocore gets explicit connect/read timeouts. te#125's edit run was
  killed on the dead-signature ~10 min into a publish that was silent BY CONSTRUCTION —
  a live 20 GB publish is now distinguishable from a dead one.

## 0.76.6 stamp — SHIPPED IN 0.76.3 (relabelled by the train; kept for the chaos record) — pgw#722 finding 2: a pure-AOT arm proves itself at boot (the #735 gap)

The shipped #735 kind-aware boot proof ran only under `proves_inductor` — a worker whose
ONLY arm is an adopted exported cell (the prod-flip shape: F1 adopts, the delivered dynamo
artifact is skipped) skipped the boot warmup proof entirely and stayed armed UNPROVEN: an
artifact that never executed, or executed and revoked, kept advertising itself. Hot-adopt's
proof was already complete; boot now matches it.

- **The proof loop runs for exported arms too** (`proves_inductor or proves_exported`);
  per-object scoring is unchanged (dynamo by FX hits, exported by `aot_serve.proven_since`).
- **Exported disarm goes through its own lane**: a disproven/unexercised exported arm gets
  `aot_serve.unwrap` (restores the forward it captured — under the F2 flip, the lifted
  LoRA forward) then `remove_lifted_lora_lanes`, landing back on the exact pre-flip eager
  shape; identity quarantined in-process as before. Mandatory (w8a8) lanes keep the
  pgw#672 degrade-to-eager posture.
- FX-key forensics no longer fire on a pure-exported disproof (they would describe the
  SKIPPED dynamo artifact's cache state — pure noise).
- Red-verified both directions through the REAL executor setup path
  (`tests/test_aot_boot_proof_gap_pgw735.py`): exercised arm => recorded proven;
  unexercised arm => disarmed to true eager + quarantined.
- Test-isolation fix on the way (pre-existing): `aot_serve.note_aot_key` learns into a
  process-global set; the pgw#722 discovery suite's learned `ck5-999…` collided with the
  adopt suite's stubbed mint digest and silently flipped its whole proof lane in combined
  runs. Autouse conftest fixture restores the learned-key set per test.

## 0.76.5 (2026-07-27) — th#1259: a bad address in the payload fails the REQUEST, not the release

A `score_benchmark` invoke passed the ref-STEM of a two-address image where the content
digest belonged. `ctx.materialize_blob` raised a bare `RuntimeError: blob fetch 404`, the
executor mapped that to `JOB_STATUS_FATAL`, and the hub counted a fatal as evidence the
RELEASE was unhealthy — `503 release_broken / model_load_failure_streak` for every caller
of release `866eaaefa7b868289aa65855`. One wrong field, no special privilege, shared release
down until a new one was cut.

**PROVENANCE decides the class** — not the status code, not the message text. A resolve
boundary now says where the address came from, and only that answers "whose fault":

- `REF_ORIGIN_PAYLOAD` (default) — the caller named it. A terminal miss raises the typed
  `PayloadRefError` family (`BlobNotFoundError` / `BlobForbiddenError` /
  `DatasetNotFoundError`, all `ValidationError`), so `_map_exception` returns
  `JOB_STATUS_INVALID`: the REQUEST fails 4xx with a machine-readable code and the hub
  books no health signal at all.
- `REF_ORIGIN_PLATFORM` — the hub produced the address (dataset manifest blobs, via the
  new `_fetch_platform_blob`). Unchanged: still fatal, still real breaker evidence.

`str(exc)` is `"<code>: <detail>"` so the code survives the `safe_message` hop into the
request's `error.code`. `resolve_dataset` classifies at the same boundary — the `_datasets`
helpers see an opaque id and raise the internal `DatasetRefNotFound` marker; the caller,
which knows the ref came from the payload, converts. Downstream-of-resolution faults
(empty manifest, silent hub, exhausted download) stay platform faults.

Hub half in tensorhub th#1259 (breaker input allowlist). **Version claim: 0.76.5** is the
first worker that classifies payload-ref misses; older workers still report these FATAL.
## 0.76.4 (2026-07-27) — pgw#752: clean page cache is RAM the next load can have
## 0.76.4 stamp — ABSORBED INTO 0.76.5 (never separately published) — pgw#752: clean page cache is RAM the next load can have

ie#535's last wan-2.2 blocker. `text_to_video_turbo` was refused on an H100 pod with
**251 GB of host RAM** for "~64.3GiB incoming + 8.0GiB safety floor = 72.3GiB required;
71.5GiB available" — then bounced 5 attempts across 2 identically-sized pods (th#1228).

- **Root cause: the model was charged twice.** `probe_host_ram` credited only the
  *inactive* file LRU back out of `memory.current`. Pages read or written seconds ago sit
  on the ACTIVE file LRU, so the pipeline's own freshly-downloaded 64.3 GiB snapshot cache
  counted as consumed memory in the very decision about whether there was room to load
  that snapshot. ~180 GiB of a 251 GB cgroup read as unavailable. The turbo tier tipped
  over first only because its two LoRA halves added ~1.8 GiB of fresh cache that the base
  tier did not read — base cleared the same bar by luck, and bounced once itself.
- **Fix**: the working set is `memory.current` minus every reclaimable clean page (both
  file LRUs), excluding what the kernel genuinely cannot drop on demand — shmem/tmpfs,
  dirty and writeback pages. Anonymous memory is still fully charged, so an over-admit
  cannot trade a false refusal for an OOM kill. `HostRam.reclaimable_file_gb` reports it.
- **A structural shortfall stops re-selling the same pod**: when the requirement exceeds
  the host's TOTAL RAM, no eviction and no identically-sized pod can ever satisfy it. That
  verdict is now `HostRamCapacityError` (`reason=host_ram_capacity`, a HardwareUnmetError
  carrying required-vs-total axes) — the function self-disables on this worker and the
  orchestrator gets a placement fact instead of another dispatch. Genuine local pressure
  stays `InsufficientHostRamError`/RETRYABLE.

## 0.76.3 (2026-07-27) — pgw#747: an auxiliary slot stops claiming the function's family

Discovery stamped **every** slot of a function with that function's architecture family —
including a slot the endpoint declared as a bare type with no ref. A frame interpolator and an
upscaler are family-agnostic by construction (they consume decoded RGB frames and know nothing
about the model that produced them, which is why ONE mirror is meant to serve every consumer),
so the hub's th#586 gate read `family = "wan-2.2-i2v-a14b"` off the slot, could not classify the
artifact as anything, and failed closed:

```
binding_incompatible: image-to-video/interpolator (tensorhub "tensorhub/rife-4.25"):
  slot declares family "wan-2.2-i2v-a14b" but the artifact's family is undeterminable
```

The gate is **manifest-wide**, so this blocked every wan-2.2 and ltx-video-2.3 release carrying
the `fps` or `resolution` preset — including the functions that do not use them. Both features
were written, tested and merged, and neither had ever been deployable.

No catalog-side stamp fixes it honestly: `Compatible` compares architecture ROOTS, so stamping
`rife-4.25` as `wan22` would be FALSE and would break the moment ltx-video-2.3 shares the same
mirror, which is the design. The gate already no-ops on an empty slot family, so emitting `""` is
the whole fix and nothing is needed hub-side.

The fix lands in `registry.py`, where `slot_family` is derived, so discovery, the executor,
`cozy run` and provisioning all see one answer; `discovery/discover.py` stops re-defaulting the
slot family back to `Compile(family=...)` after the fact, which would have put the bug straight
back on exactly the emptied slots.

**Two conditions, not one.** "No ref ⇒ no family" alone would also empty a deliberately
DEFAULTLESS **root** slot — a real model slot bound at deploy time through `?bindings=`, which is
the shape `krea-2` ships — silently dropping the LoRA-overlay policing pgw#523 added it for. Only
a **non-root, ref-less** slot is family-agnostic. Tests cover all three shapes; red-verified
against the pre-fix registry, where the auxiliary slot comes out carrying the function's family.

## 0.76.2 (2026-07-27) — pgw#743: a proxy-shaped answer is not a verdict, at ANY status

pgw#715 taught the publisher that "a 404 from a PROXY is not a 404 from the hub". It taught it
about **404 only**. Two independent clones then downloaded 53 GiB over ~58 minutes each and died
byte-identically at the **first upload**:

```
HubPublishError: upload complete failed (503) for
'transformer_2/diffusion_pytorch_model-00001-of-00014.safetensors'
after 5 attempts: <!DOCTYPE html>
```

Same proxy, same HTML page, different number — and the different number was enough to throw away
a fully-paid download. `tensorhub/wan22-i2v-a14b` has been a half-present MoE ever since, and the
`krea-2-{raw,turbo}` mirrors have zero checkpoints.

**The question is never "was it a 404?"** — it is "did the hub ITSELF answer?".
`http_origin.is_definite_hub_answer(resp)` now asks exactly that, status-agnostically: a 2xx/3xx
(nothing fabricates a success for a route it cannot reach) or a 4xx carrying the hub's
`{"error": {"code": ...}}` envelope is a VERDICT and ends a retry loop. Everything else — 5xx,
and any status whose body is HTML/empty/enveloped-wrong — is the peer failing to answer, and is
retried under a silence window. `convert/hub.py::_send_with_retries` is gated on that single
predicate instead of `code == 404 and is_proxy_outage(...)`, so an ngrok 403 or 502 rides out
exactly like the 503 did not.

Hub-origin refusals are **unchanged and still terminal on the first attempt** — biasing toward
retry must not convert real refusals into retry loops, and there is a test that counts the POSTs.

**Sizing, on measured evidence.** `_COMPLETE_SILENCE_WINDOW_S` went from 2 to 6 verify-lengths
(20 -> 60 min). The recorded outages OUTLIVED the 20-minute window: the chaos hub's container was
being rebuilt under the running clones. A window is a claim about how long the channel may
plausibly be gone, and a container rebuild is tens of minutes. The arithmetic agrees — an hour
parked on the CPU rig these jobs run on costs about what re-downloading costs, and unlike the
re-download it cannot fail the same way again. The retry loop beats `activity.note_progress()`
every pass so an hour of legitimate waiting is not the dead-job signature (pgw#738).

**`convert/keepalive.py` (new) — the hub-silent hour that set the trap.** A clone spends ~58
minutes downloading from HuggingFace and makes not one hub request; both losses landed on the
first request after such a gap. `HubKeepalive` probes one cheap repo GET every 120s for the whole
of `run_clone` (download AND cast), so there is never an hour-old idle path for a multi-GB upload
to discover, and — whatever the idle-tunnel hypothesis turns out to be worth — the log now DATES
the outage instead of leaving its corpse. The probe is deliberately toothless: it never retries,
never raises, never fails a job. Deciding what an outage means belongs to the retry loop that
knows what work is at stake.

Sub-chunking uploads to start them earlier was the alternative and does not fit clone: no output
file exists until the whole snapshot has been downloaded and run through repackage/cast
(`build_flavor_tree` reshards across the complete file set), so "upload sooner" means rebuilding
the pipeline as a streaming one, not adding a call.

Landed together with the `_send_with_retries` silence-window rewrite authored by the pgw#738 lane
in the shared chaos worktree — pgw#743 IS that rewrite plus the generalisation above, and a
second implementation would only have collided with it. pgw#738's other halves (executor lock
order, publish observability) remain that lane's to land under its own version claim; this bump
takes 0.76.2 and leaves 0.76.1 to it.

## 0.75.2 (2026-07-27) — pgw#737: the self-mint never takes the tenant request down again

The ie#535 wan-2.2 1.3.1 go-live spent $2.61 and rendered zero frames. On both tiers and both
80 GiB H100s the gw#587 fleet self-mint ran `inductor_compile` against a **~54.2 GiB resident
bf16 MoE** (two 14B experts plus LoRA branch containers), OOMed its warm plan three times,
and the **tenant request died with it** — 78.07 GiB peak, 26 of 40 denoise steps banked and
lost, `JOB_STATUS_RETRYABLE`. The hub then re-dispatched that deterministic failure 5 times
and bought a second H100 for it (th#1228 class, now priced).

gw#587's premise — the serving worker's boot warmup IS a perfect mint by construction — holds
only while the capture FITS. Nothing checked. Three fixes:

- **A VRAM pre-budget, before anything is armed** (`gen_worker/mint_budget.py`). The gate sits
  at the eager-first arm decision, not just in the background driver: enabling the routers is
  already the first allocation of a capture (the boot warm's own forwards enqueue background
  compiles). It is a MEASUREMENT, not a model of the graph — the CUDA peak high-water minus
  the resident set is the largest transient the process has actually sustained (this family's
  activation working set at serving shapes, once a forward has run; the driver re-checks after
  the boot warm), floored by a quarter of the resident set. A mint needs two of those working
  sets — its own seed forwards, plus what the capture retains for the tenant's next peak to
  fit around — on top of a flat inductor working-set floor. Not fitting is not an error:
  the targets go back to true eager, the branch lane is dropped, the allocator is emptied, the
  cell stays ABSENT, and one structured line —
  `mint_skipped reason=insufficient_vram headroom=24.99GiB needed~=31.10GiB resident=54.20GiB
  activation=13.55GiB(measured)` — is logged and put on the wire as a typed
  `self_mint_skipped` event. No env knob: a roomier config, or a smaller-resident flavor,
  mints the same cell later.
- **A survivable abort.** The mint is architecturally OFF the request (pgw#671 background
  driver), so nothing banked is lost by an abort — what killed the tenant was the capture's
  RESIDENT cost. Every mint terminal (declined, failed, OOM-aborted) now unwraps its targets,
  closes their routers so no queued warm job can still compile onto the card, drops the branch
  buffers and empties the allocator. An OOM'd seed pass re-budgets on the allocator state the
  OOM just measured and declines instead of retrying into the tenant three more times. And
  from the tenant's side (`_evict_mint_for_oom`): a request that OOMs with a mint in flight
  EVICTS the mint — the one co-resident consumer this worker put on the card itself — and
  re-runs on the clean allocator.
- **Eager serving is a SUCCESS path.** The re-run request returns `JOB_STATUS_OK`, so there is
  nothing for the hub's ladder to re-dispatch or buy a pod for, and a declined mint terminates
  its `self_mint_compile` activity COMPLETED (a mint we declined is an outcome, not a worker
  failure).

Fence: `tests/test_mint_vram_budget_pgw737.py` — the wan-2.2 card declines and an sdxl-class
residency on the same rig still mints; stubbing either fix turns the tapes red with the exact
live symptoms (a capture attempted anyway / `JOB_STATUS_RETRYABLE: out of memory`).

Proven on a real card (one L4, 22 GiB, `cozy-creator-tracker/scripts/pgw737/`, $0.27): with
14.49 GiB resident the gate declines — `mint_skipped reason=insufficient_vram headroom=7.28GiB
needed~=11.24GiB resident=14.49GiB activation=3.62GiB(estimated)` — zero captures are
attempted, capture residue is 0.01 GiB and the request completes in 7.9s at an 18.5 GiB peak
(the pre-forward estimate, 3.62 GiB, called the measured tenant activation of 4.0 GiB within
10%). The same arm with the gate stubbed out mints, charges the tenant 1.01 GiB of retained
capture and takes 17.3s; one rung tighter (16.04 GiB resident, 5.80 GiB free) the stubbed arm's
request DIES `JOB_STATUS_RETRYABLE: out of memory` — the live wan-2.2 symptom — while the gated
arm declines cleanly. A roomier card (8.5-11.6 GiB resident) mints in every arm: no false
declines.

## 0.75.1 (2026-07-27) — pgw#715: a 404 from a PROXY is not a 404 from the hub (te#125/th#1238)

A hub restart lasting seconds destroyed a **116-minute H100 producer run** at its last step.
The job finished quantizing and rendering, then its in-flight media upload hit ngrok while the
backend was briefly gone, received ngrok's HTML 404 page, and
`presigned_upload.py` classified **every** 404 as `retryable=False` — so the worker reported
`JOB_STATUS_FATAL` and two hours of GPU work was thrown away with no recourse.

The gRPC worker stream reconnects across a hub restart. **In-flight HTTP calls did not**, and
that asymmetry is why "hub-only restarts are safe" held for serving and quietly did not hold
for long conversion producers.

**`gen_worker/http_origin.py`** (new) separates the two cases, on measured evidence rather
than assumption — the hub answers `application/json` carrying its `{"error": {...}}` envelope,
ngrok's offline page answers `text/html`:

- `response_is_from_hub(resp)` — parses the body, not just the Content-Type, since a proxy may
  mislabel HTML as JSON but will not synthesise our error envelope.
- `is_proxy_outage(resp)` — the inverse, used at the call sites.

Deliberately **biased toward retrying**: an unrecognised body counts as proxy-origin. Retrying
a genuinely missing route costs a bounded backoff and then fails anyway; treating an outage as
fatal destroys hours of work. The asymmetry of those two mistakes is not close.

Applied at every site that conflated the two **for hub calls** — the point was to fix the
class, not the one instance that bit us:

- `presigned_upload.py` — the P0. Proxy 404 during upload create is now `retryable=True`.
- `models/hub_client.py` — a proxy 404 was reported as `HubRepoNotFoundError`, sending the
  reader to hunt a catalog problem that does not exist. Now `HubResolveError` (transient).
- `callout.py::checkpoint_get` — **the worst of the three**, because it did not crash: it
  returned `(None, False)`, i.e. "no saved progress", so an outage made a resumable job
  silently restart from scratch. Now raises instead.

**Deliberately NOT changed:** `models/download.py`'s civitai 404s. Civitai is a third-party
host with no proxy of ours in the path, so there a 404 really does mean not-found. Widening
the helper to non-hub hosts would trade a real bug for a fake one.

Still open: `request_context/__init__.py:1494` carries the same pattern, but that file holds
another agent's uncommitted work in the shared chaos worktree, so it is left alone rather than
entangled. Recorded in pgw#715 for its owner.

Also riding in 0.75.1 (landed after the 0.75.0 stamp, folded here by the train): the
**pgw#700 arc** — equivalence adoption SDK half (`gen_worker/equivalence.py`: code-closure
fast tier + manifest/fingerprint slow tier per the th#1229 ruling), pgw#710 toolchain
digests in cell metadata, pgw#711 publish digests (`artifact_digest`/`manifest_digest` on
publish-complete), pgw#712 no-republish fencing + unicity refusal. All behind
`GEN_WORKER_EQUIVALENCE_ADOPTION`, **default OFF** — the flag flip is gated on th#1229/
th#1239 hub halves. No behavior change with the flag unset.

## 0.75.0 (2026-07-26) — pgw#660: the hard GPU-architecture floor has a declared carrier again

`Resources(compute_capability=8.9)` is restored. The v2 API freeze (pgw#647) deleted it on
the reasoning that "precision-per-card is the fit ladder's call, never a placement gate".
That is right about precision SELECTION and wrong about INCAPABILITY: a producer whose
kernel is `torch._scaled_mm` cannot run below sm_89 at any precision, on any rung, ever.

Tensorhub's builder never stopped reading the key, so a v2 endpoint emitted no floor,
`endpoint_function_schemas.compute_capability_min` went NULL, and the scheduler placed the
fp8 modelopt producer on **sm_80 A100s** — th#1155 six times in ten minutes, and again in
te#125 on 2026-07-26, where conversion 0.6.1's in-pod envelope guard refused in 4 ms after
the pod was already rented. Design 1 of the three in pgw#660, ruled by Paul.

- **Not a hint.** `vram_gb_hint` / `ram_gb_hint` are allocation-time asks the platform may
  miss; this one is filtered on. It carries no `_hint` suffix and has no second spelling.
- **Dual-form input, one canonical value.** `8.9`, `"8.9"`, and `"sm_89"` all normalize to
  `8.9`. A bare SM code (`89`) is REFUSED with the two correct spellings named — 89 and 8.9
  are a silent factor of ten apart.
- **Implies `gpu=True`**, like `vram_gb_hint`.
- **Wire name is `compute_capability`** — already what `internal/builder/
  function_requirements.go` parses into `FunctionRequirements.ComputeCapabilityMin`. Note
  v1's author-facing `min_compute_capability` stays typed-REJECTED by the builder
  (th#1015 `ErrMinComputeCapabilityRemoved`) and must never reach the wire.
- **Undeclared is unchanged**: no key, no column value, no gate. Declare it only for a
  genuine incapability, never because a function merely runs better on newer silicon.

## 0.74.2 (2026-07-26) — pgw#714: background-compile crashes tell the truth and degrade to eager

From the th#1226 post-mortem (qwen-image v0.2.1 pinned pre-pgw#677 0.67.1: the
ungated background compile SIGSEGV'd live serving, and the death was recorded
against `fn=generate`, condemning H200 + B200 in the hub's SKU-compat table
for a software race):

- The hot-swap warm thread stamps a `compile` inflight marker
  (`compile:<label>`) around every background compile; a signal death with a
  compile marker present records the streak against the COMPILE, never the
  tenant request that happened to be in flight.
- Boot gate: crash-registry rows of compile kind disable process compiles
  (`compile_cache.disable_process_compiles`) — the pod reboots into
  eager-only serving instead of re-running the native crash, and the serving
  function is NOT refused (degrade-never-die at process-death scope).
- `native_crash_streak` refusal axes carry `last_kind` so the hub can spare
  the SKU table for non-serving deaths (hub half: th#1236).
- **Operator kill switch:** `ModelResolution.lane_pinned` (proto, additive) —
  when the hub marks the resolved lane as an endpoint-pin and its execution
  axis is `+eager`, `compile_cache.apply()` refuses to arm at all: no router,
  no background/foreground self-mint, pure eager serving. Auto-resolved
  `+eager` lanes keep today's eager-first + background-mint behavior.

## 0.74.1 (2026-07-26) — pgw#692: `WanDefaults` carries the hub's recipe wire name

**P0, every wan-2.2 request of every tier.** th#1174's migration
`0046_recipe_steps_rename_hidream_wan22` renamed the recipe field `steps` ->
`num_inference_steps` in the hub's `wan22.schema.json` and in every stored
`repo_inference_defaults` / `release_slot_recipe` row (chaos, 2026-07-26
00:12:20Z) — but the SDK half never shipped, so the hub-stamped recipe hit
`GenerationDefaults`' `forbid_unknown_fields` and every request died at slot
resolution, before handler code:

```
ValueError: slot 'pipeline': catalog inference-defaults metadata failed
WanDefaults validation: Object contains unknown field `num_inference_steps`
```

- `WanDefaults.steps` -> `WanDefaults.num_inference_steps`. The hub is the
  half that is right: `RuntimeFormula` resolves its terms by same-named
  lookup across payload and `ctx.defaults`, and the
  `PUT .../metadata/inference-defaults` route refuses a `steps` spelling
  outright (`additionalProperties: false`).
- Full audit of the registered vocabularies against the hub's family schemas:
  `SdxlDefaults` and `SdxlLoraDefaults` keep `steps` (0040/0046 deliberately
  left sdxl and qwen-image alone) and match field-for-field; wan22 was the
  only skew in this repo. `HiDreamO1Defaults` has the identical exposure but
  lives in `inference-endpoints/hidream-o1-image` — cross-repo, tracked on
  pgw#692.
- Guard (`tests/test_family_wire_names_pgw692.py`): every registered family's
  `__struct_fields__` is asserted against a recorded snapshot of its hub
  schema `properties`, so a one-sided rename can never be silent again.

## 0.74.0 (2026-07-26) — pgw#685 S2c: the native svdq engine actually serves

Wires the native engine into the load path. `load_svdq_pipeline` now chooses an
ENGINE (`select_svdq_engine`) instead of assuming nunchaku, so `loading.py` needs
no change at all — the dispatch lives behind the entry point the loading layer
already called.

- **`load_svdq_native_denoiser`** materializes a nunchaku-format checkpoint as a
  STOCK diffusers module: skeleton on meta, W4A4 linears swapped for
  `SvdqLinear` (fused `to_qkv` / `add_qkv_proj` split across the diffusers
  projections), AWQ modulation layers decoded to bf16 Linears, everything else
  assigned verbatim, then a STRICT check that nothing is left on the meta device —
  a checkpoint that does not cover the module fails loud instead of serving a
  half-initialized denoiser.
- **`adanorm_splits_for`** is a table keyed on (diffusers class, module suffix),
  not an inference from `out_features // in_features`. An unknown modulation layer
  REFUSES; the exporter's adaLN transform is unrecoverable from the tensors and a
  wrong split count corrupts output silently.
- **Verified against the REAL 13 GB `svdq-fp4_r128-qwen-image` artifact**, pod-side
  (weights never touch the dev box). Every one of its 4573 tensors is accounted
  for: 480 svdq W4A4 prefixes (360 direct + 120 fused-qkv splits), 120 AWQ
  modulation layers, 247 plain — **zero unmapped in any category** against the
  stock `QwenImageTransformer2DModel`. Both decoders produce finite,
  structurally-correct values (rank 128 throughout; `per_channel` second-level
  scale exactly on the two fused qkv layers and `per_tensor` elsewhere, as the
  layout predicted). The assembled loader then ran end to end: 2-block truncation
  in 7.0s / 3.65 GB RSS and the **full 60 blocks in 163s / 55.7 GB RSS**, nothing
  left on meta, `to_q`/`to_k`/`to_v` present as three working 3072-wide Linears,
  modulation decoded to `Linear`, and a live forward finite.
- mypy: fixed 7 `Optional`-narrowing errors this lane introduced (5 in
  `svdq_native.py`, 2 in `svdq_layout.py`) by narrowing through locals rather than
  silencing them; `mypy src/gen_worker` is green across 155 files.

Admission is still NOT widened: `ladder.py`'s svdq placement keeps
`sm_allowed=(120,121)`, `engines=("nunchaku",)`. Native now SERVES where it is
selected, but the hub does not yet schedule svdq-fp4 onto sm_100 — that flip wants
the fp4 (`blockwise`) full-artifact load and a numerical A/B, neither of which is
measured yet (the verified load ran in the `dense` fold, and no nunchaku wheel was
present to difference against).

## 0.73.1 (2026-07-26)

- **pgw#684: a fourth reserved repo field, `candidate`, for producer payloads
  (te#121 two-ref quality eval).** The executor's reserved repo names were
  `source`/`destination`/`text_encoder`; a producer payload can now also declare
  `candidate: SourceRepo | None = None` and get it materialized the same way as
  `source` — into `ctx.candidate_path` (`ctx.candidate` for the raw dict), fully
  independent of `ctx.source_path`. This is the arm a two-ref eval COMPARES
  against `source` rather than a component it builds from, which is what lets a
  quality gate point at one of OUR OWN hub artifacts (a mirror, or a flavor the
  quant ladder just produced) instead of only a public HF/Civitai coordinate.
  Generic mechanism: gen-worker has no eval awareness. Absent field (every
  existing endpoint) is byte-for-byte unchanged — no extra `ensure_local` call,
  `ctx.candidate_path` stays `None`. The reserved-name set is still a hardcoded
  literal list; pgw#690 tracks making it declarative.
- Also re-covers reserved-repo materialization in `tests/`, which had NO coverage
  after th#960/pgw#609 Phase 3b (`0b437aa`) swept pgw#594's two test files.
- Corrects a `uv.lock` drift: 0.73.0 bumped `pyproject.toml` but left the lock's
  `gen-worker` entry at 0.72.0, so `uv lock --check` failed on that commit.

## 0.73.0 (2026-07-26) — pgw#687: a cancel that never unwinds no longer absorbs the next job

Cancelling a job mid-compute could wedge a worker permanently while every
hub-side signal read healthy: connected, heartbeating, still advertising its
functions. Live (th#1165): job A cancelled mid-modelopt-calibration, job B
assigned to the same pod 61 s later, then ZERO events of any kind for 46
minutes. The only symptom was absence.

Mechanism: cancelling a SYNC handler is cooperative. `handle_cancel` sets
`ctx.cancelled` and (for async handlers only) cancels the task; a thread
running `asyncio.to_thread` cannot be cancelled at all. A handler that never
polls the flag — a modelopt calibration loop — keeps running, so `_run_job`
never returns, the GPU permit and the per-instance run gate are never
released, and the next job parks in `_gpu_semaphore.acquire()`, a wait that
emits nothing. Nothing watched the cancel -> terminal edge.

- **The cancel -> terminal edge is now watched.** Past
  `_CANCEL_UNWIND_GRACE_S` (45 s) the executor is presumed unable to return
  to idle and FAILS CLOSED: every function goes `unavailable` with reason
  `cancel_unwind_stuck` (a real `FnUnavailable` on the wire, not merely an
  empty function set), and any job still parked pre-execution is failed
  RETRYABLE so the hub replans it NOW instead of letting it sit eventless.
  Reversible: a late unwind re-advertises exactly the functions we took.
- **A thread that never honours the cancel replaces the pod** — process
  recycle after a further `_CANCEL_UNWIND_RECYCLE_S` (300 s), the only way to
  reclaim a wedged thread. Routed through the same injectable exit seam as
  the deadline reaper.
- The bound is on cancel -> terminal latency, never on handler progress, so
  it does not re-introduce the wall-clock bound gw#666/th#1157/th#1160
  forbid: a 51-minute silent source download is not a cancelled job and is
  untouched.
- A PRODUCER (conversion/training) handler cancelled mid-run now marks its
  instance stale, so the next dispatch reloads clean — modelopt installs
  module-level quantizer hooks that the next `setup()` would otherwise
  inherit. Inference cancels are excluded on purpose: a cancelled forward
  mutates nothing, and discarding a warm serving pipeline on every user
  cancel would be its own regression.
- `tests/test_cancel_unwind_pgw687.py` drives the real executor over the
  hub-double with a handler that ignores cancellation. The red row
  (`test_wedge_shape_without_the_guard_is_silent_absorption`) is kept
  permanently: with the grace pushed out of reach it reproduces the pre-fix
  silence, and the four fix rows fail without the watchdog.

## 0.72.0 (2026-07-26) — pgw#685 S2b: the AWQ W4A16 modulation decoder

An svdq artifact does not quantize everything the same way. Its DiT Linears are
W4A4 nvfp4 with a low-rank branch; its adaLN MODULATION layers (`img_mod` /
`txt_mod`, which consume the timestep embedding rather than the token stream) are
AWQ **W4A16** — a completely different layout. `models/svdq_awq.py` decodes them,
which was the one thing blocking a real artifact from loading natively.

- The layout is the inverse of deepcompressor's
  `convert_to_nunchaku_w4x16_linear_weight` -> `convert_to_tinychat_w4x16y16_linear_weight`
  -> `pack_w4` chain, and the tests invert that UPSTREAM code bit-exactly rather
  than a paraphrase of it: within each run of 32 input elements output nibble `j`
  packs elements `{j, 8+j, 16+j, 24+j}`, then the int16 grid is shuffled
  `[oc/4, 4, ic/64, 16] -> permute(0, 2, 1, 3)` and stored as int32 pairs.
  Confirmed against the real artifact's geometry (`qweight I32 [4608, 1536]`,
  `wscales`/`wzeros BF16 [48, 18432]`, group 64).
- Dequant is an **ADD**, not a subtract — `W = codes * wscales + wzeros` — because
  the exporter stores the zero point already scaled AND negated.
- `ceil_num_groups` padding is handled: trailing all-zero scale rows are the pad,
  not groups. Reading 16 stored rows as 16 groups where only 2 are live would
  rescale every weight in the layer.
- **The trap this decoder exists to get right (`adanorm_splits`):** for modulation
  layers the exporter ALSO interleaves output channels (stored row `j*splits + s`
  is original row `s*(oc/splits) + j`) and ADDS 1 to the bias of splits `1` and
  `splits-2` — adaLN's `1 + scale` folded into the artifact. Both are undone. The
  split count is a REQUIRED argument defaulting to 1, never inferred: a wrong
  count still yields a full-rank, plausible-looking weight and silently wrong
  images, which `test_decoding_with_the_wrong_split_count_is_visibly_wrong` pins
  at >0.5 relative error against <0.15 for the correct count.

Also recorded from the S1 card verification, because it will otherwise look like a
bug in the fused quantizer: compiled-vs-eager output for a `W4A4Linear` is NOT
bit-identical, and that is inductor reassociating the bf16 second-level epilogue,
not the kernel. Measured on a 5090 with the fused kernel disabled, the PURE-TORCH
chain drifts **7.4x MORE** (5.9e-3) than the fused path (7.9e-4); the custom op
itself is bit-exact under `fullgraph=True`.

## 0.71.0 (2026-07-26) — pgw#685: a NATIVE svdq engine — SVDQuant checkpoints without nunchaku

The layout converter, the serving module, and engine selection for serving
svdq-fp4 on **every** Blackwell part through stock `torch._scaled_mm` instead of
nunchaku's `sm_120a`-only kernels. Not yet wired into the default load path —
see the named gap below.

- **`models/svdq_layout.py`** inverts nunchaku's v1 single-file layout: the
  `qweight` `mma.sync m16n8k64` FRAGMENT interleave and the `wscales` transpose
  + 8-lane/stride-4/4-pack swizzle, both by replaying the packer's permutes
  backwards (guaranteed bijective) rather than re-deriving index math. Decoding
  lands in a LOGICAL domain first, which is what makes nunchaku's fused
  `to_qkv` splittable into diffusers' separate `to_q`/`to_k`/`to_v` — exact
  along the output dim, partitioning `proj_up` while SHARING `proj_down`.
- **`models/svdq_native.py`** — `SvdqLinear`: `W4A4Linear` plus the three things
  an SVDQuant checkpoint needs. A per-OUTPUT-CHANNEL second-level weight scale
  (`wcscales`) as well as the scalar `wtscale`; the low-rank branch
  `y += (x @ proj_down) @ proj_up.T`, which is what makes 4-bit survive
  qwen-class outliers (plain nvfp4 PTQ measured lpips 0.63-0.69 vs the official
  svdq artifact's 0.105); and `smooth_factor`, which DIVIDES the activation
  feeding the 4-bit branch **only** — the low-rank branch consumes RAW x,
  because deepcompressor pre-divides `proj_down` at export. Activation scaling
  is always DYNAMIC: an svdq checkpoint carries no `input_scale` at all.
- **Degrade, never refuse**: `fold_to_dense` collapses the 4-bit weight, the
  smoothing vector and the low-rank branch into ONE plain bf16 Linear
  (`W_eff = W_q / smooth + proj_up @ proj_down.T`, exact in the dequant limit),
  so an svdq artifact stays servable on hardware with no fp4 tensor cores.
- **Engine selection** (`svdq.svdq_engine_candidates` / `select_svdq_engine`):
  `"native"` is preferred for fp4 — no nunchaku wheel, no diffusers signature
  window, no pin-matrix row, no torch downgrade (the gw#405 / th#1211 coupling
  class), and it covers sm_100/103 which nunchaku never will. int4 stays
  nunchaku-only (a different single-level group-64 scale path). An explicit
  override (`GEN_WORKER_SVDQ_ENGINE`, or the `override=` argument) is honored
  STRICTLY — the other engine is never silently substituted.
- **The native SM window is Blackwell only** (sm_100/103/120/121). torch's own
  nvfp4 gate is `major >= 9 || (8,9)`, which admits sm_89/sm_90, but neither Ada
  nor Hopper has fp4 tensor cores; below Blackwell the honest degrade is fp8
  rowwise, which we already ship. Nothing emulates fp4.
- **Named gap — deliberately NOT the default load path yet.** A real qwen svdq
  artifact also carries its modulation layers as AWQ W4A16 (`wzeros`, group 64,
  int32-packed, ~33% of the parameters but negligible FLOPs). That decoder is not
  written, and shipping it unverified would be worse than declaring it, so
  `loading.py` routing and the `ladder.py` svdq placement are UNCHANGED:
  widening admission before a real artifact can fully load would strand
  requests.

## 0.70.3 (2026-07-26) — pgw#694 determinism hardening + cache-review fixes (ck4 keys, env-seal boot wiring, inner-FX sm shim)

One train: the pgw#694 hardening set (chaos a73e6c8), its executor-side boot wiring (`entrypoint._establish_env_seal`), and the ML-cache-review fixes (chaos 23a34bd — the P0 inner-inductor-cache portability shim, B200-verified). ck3 -> ck4 is the second and final planned key-scheme bump; expect one `cell_exchange_key_split` alarm per (endpoint, family) and a one-time re-mint wave.

### pgw#694 (#695-#698): execution-environment determinism hardening

Four of the pgw#694 umbrella's five measures (the fifth, pgw#699, is a tracker-side
harness). All red-verified against real torch state / real file trees; CPU only.

- **#695 process-posture seal**: ONE canonical serving posture (grad, autocast,
  torch-function stack, default device, deterministic-algos);
  `guard_closure.establish_posture()` at boot, sealed into the guard manifest at mint
  (manifest v2), re-asserted by `artifact_drift` before every arm — drift refuses the
  arm NAMED, never a downstream guard miss. A mint in a non-canonical posture fails
  red. `consolidate` flags cross-pod posture divergence.
- **#696 config-surface freeze + ck4**: new `env_seal` module — canonical flag table
  set explicitly at boot (`cudnn.allow_tf32` default-True pinned False,
  float32_matmul_precision, TF32 matmul, cudnn.benchmark), unknown `TORCH*` env vars
  refuse boot naming the var, portable inductor-config digest. Posture+config+inductor
  fold into ONE versioned `env_seal` dict recorded verbatim in metadata; its digest is
  a REQUIRED ck4 key axis recomputed from the recorded facts. KEY_SCHEME ck3 -> ck4
  (final planned bump: seal_v versions the dict internally).
- **#697 composition fingerprint**: module rows now carry hook presence; new
  `composition_fingerprint()` stores per-module digests in metadata so a graph-
  signature mismatch at adoption names the exact drifted module
  (`transformer:lin2: cell ... != consumer ...`) — the pgw#683 bf16/Half class.
  Fine-tunes still share cells (no tensor values in any row).
- **#698 cubin-completeness gate**: `pack()` refuses (named kernels) when any kernel
  ships PTX without an sm-exact cubin — closing the one path where the deliberately
  unkeyed driver (gw#577) could re-enter behavior via PTX JIT.
- **ck3-completion bug fix**: `verify()` still hard-pinned `sku` after the pgw#691
  collapse — a same-sm cell minted on a different SKU refused to arm. sku is now
  observability-only in verify; sm/cuda/torch/triton carry the hardware identity.

### Cache-design review fixes — inner FX key portability + strict verify

From the ML-systems + build-systems cache reviews (tracker, both in
python-gen-worker/progress.md). All red-verified (7/8 new tests fail pre-fix).

- **P0 — inner FX key hashed the GPU marketing name** (VERIFIED on a real B200 cell:
  `system_info[device] = {'name': 'NVIDIA B200'}`): inductor's `CacheBase.get_system()`
  files every fxgraph entry under the minting pod's SKU string, so the ck3/ck4 sku
  collapse delivered ZERO cross-SKU hits — same-sm adoption passed every gate then
  missed inside torch's own lookup. New version-pinned shim
  (`compile_cache._install_fx_system_shim`, installed symmetrically via `apply()`)
  normalizes the device name to the `sm_XX` token with the hash recomputed by torch's
  own strategy (upstream precedent: `AOTI_COMPUTE_CAPABILITY`, codecache.py:260;
  upstream ask tracked on pgw#708). A source-shape pin test fails loudly on a torch
  bump (pgw#705 doctrine).
- **P1 — `verify()` fail-open retired** (the JAX PR #27814 wrong-hit shape): silent
  axes (`sm`/`cuda`/`image_digest`/libs/family) were accepted via `if want and ...`;
  now absent axis = named refusal, no legacy path (pre-launch, per the no-legacy
  doctrine).
- `cache_key_tag` bound to the semantic cell identity (format|kind|family|lane|mode|
  contract, environment axes deliberately excluded) — a foreign semantic identity can
  never consume delivered inner entries; equivalence adoption (pgw#700) survives.
- `content_keys` (torch_key/triton_key digests) recorded in metadata as the pgw#700
  equivalence precondition (a patched wheel under an unchanged version string is now
  visible; full toolchain closure is pgw#710).
- **env_seal v2**: R7 defect fixed — the env gate matched `TORCH*` only, so every
  `PYTORCH_*` var (incl. the live `PYTORCH_CUDA_ALLOC_CONF`) evaded it; gate now
  covers `TORCH*`/`PYTORCH*`/`TRITON*` with both allocator spellings + the SDK's
  `TRITON_CACHE_DIR` allowlisted, and `TRITON_PTXAS_PATH`-class toggles refused.
  Recorded-env set extended (CUDA_LAUNCH_BLOCKING, CUDA_MODULE_LOADING,
  NVIDIA_TF32_OVERRIDE, PYTHONHASHSEED). R2: operator `epoch` salt (`COZY_CELL_EPOCH`)
  sealed as a fact — disowning a poisoned mint generation is one config change, never
  a scheme bump (Bazel Action.salt / ccache HASH_PREFIX precedent).

## 0.70.2 (2026-07-26) — pgw#691: guard-closure classifier fixes + sku key collapse (ck2 -> ck3)

The offline audit (546 real torch-2.13 guard rows) proved the pgw#681 mint
gate would refuse 100% of sdxl mints. Four fixes, all red-verified against
real dynamo guard trees (CPU, `backend="eager"`):

- **P0**: the RelationalGuard family (`NO_TENSOR_ALIASING`,
  `OBJECT_ALIASING`, `STORAGE_OVERLAPPING`) is judged by TYPE before root
  dispatch — torch attaches it to the INPUT managers, so the old root-first
  dispatch leaked on every graph with >=2 tensor inputs.
- Vocabulary rebuilt on the C++ leaf-class names the guard walk yields:
  `SYMBOLIC_SHAPE_GUARD` (2.13's shape-env facts, incl. its synthetic tuple
  sources) rides declared dynamic dims; input-rooted
  `ID_MATCH`/`DICT_VERSION` classify as the object identity of a call-path
  constant; `DUAL_LEVEL_MATCH`/`FAKE_SCRIPT_TYPE_MATCH` named; phantoms
  (`AUTOCAST_STATE`, `KEYS_MATCH`) dropped.
- `EQUALS_MATCH` against torch singletons (`L['dt'] == torch.float32`,
  memory formats, device literals) is covered by weight_lane + the ingress
  dtype memo instead of leaking "unparseable".
- **ck2 → ck3**: `sku` left the cell-key identity axes (two byte-identical
  cell pairs split only by sku in the audited corpus; no guard observes a
  SKU — sm/cuda/torch/triton pin the hardware facts). `sku` stays in cell
  metadata (publish-intent attestation unchanged); same-sku preference among
  same-key candidates is a hub-side selection follow-up. Old ck2 keys fail
  `is_key` outright — clean MISS, never a half-match. Hub-side confirmed
  scheme-agnostic (`IsCellKey` shape check only, th#1183).

## 0.70.1 (2026-07-26) — pgw#677 REOPEN: the fix that did not hold live — one serveability brain, compile-steal sizing, and every mint refusal on the wire; plus pgw#689

The ie#546 final verification cycle reproduced the 0.70.0 starvation
verbatim on three cold L4 pods with the gate armed: one tenant `generate`
held 26m25s behind 4-7-minute mint units, finalize at unit 8/18, and not
one publish-intent from any of six workers. Root causes (each red-taped in
`tests/test_mint_reopen_pgw677.py`):

### pgw#677 reopen

- **One serveability brain (`compile_cache.mandatory_serving`)** — THE live
  break. sdxl's mixed `#fp8-w8a8`-storage checkpoint stamps
  `_cozy_weight_lane` `w8a8*` (cell identity, pgw#686) while the hub serves
  it `fp8-w8a16+eager`. `_eager_first_eligible` and the router's
  `fail_closed` read the weight-lane PREFIX, classified the boot
  mandatory-quantized, and silently fell back to the FOREGROUND
  compile-then-serve mint — the tenant sat inside `ensure_setup` for the
  whole inline-compile plan and none of the 0.70.0 gate/preemption
  machinery ever ran. Serveability now follows the hub-resolved th#913
  execution lane (only real w8a8/w4a4 ACTIVATIONS forbid eager), stamped on
  every arm path via a setup-scoped window; without lane evidence the
  weight-lane stamp remains the fail-closed fallback (the qwen real-w8a8
  shape keeps its foreground proof).
- **Stolen-compile sizing (the ~100x correction)**: a stolen turn is not
  preemptible and a real inductor compile is 4-7 unabortable minutes, not
  the advertised 30-90s. Compile turns now steal only after
  `_BG_COMPILE_STEAL_FLOOR_S` (600s) of continuous tenant demand — real
  traffic has idle gaps between completions, and that is where compiles
  run — and a granted compile steal announces itself as a typed
  `bg_turn_steal` event. Seed turns keep the 30s floor.
- **The mint→publish break is typed on the wire, every door**: the cycle
  lost its root cause to unreachable pod logs; never again. `self_mint_abort`
  (pack/closure-gate refusal with the pgw#681 leak named verbatim, warmup
  OOM, proof-failed degrade, abandon, driver error),
  `self_mint_publish_withheld` (gw#612 sharer gap, missing publish sink),
  `self_mint_publish_failed` (hub refusal / upload error).
- **A truncated warm plan can never finalize a partial capture**: an
  OOM-cut seed pass no longer satisfies the convergence check (bounded
  retries, then a loud abort); the foreground path withholds publish on a
  cut-short plan; `Router.route`'s seed-window holes (`_MAX_SIGS` overflow,
  dummy-batch failure) route EAGER and count `seed_dropped` so the driver
  refuses an incomplete capture instead of inline-compiling under the run
  gate.

### pgw#689: the swap benchmark loads what SERVING loads; a broken diagnostic is not a broken release

The SDK's own swap-latency benchmark could not load a QUANTIZED component
tree — i.e. every flavor the fleet actually serves — so every `load` /
`demote` / `stage` / `swap` row was unreachable on a real serving pod. It
owned a second loader (`cls.from_pretrained` per component), and a
modelopt-produced tree carries a `quantization_config` block diffusers
rebuilds into `NVIDIAModelOptConfig`, whose constructor requires a
`quant_type` the block does not supply.

- **One component-load path (`models.loading.load_component`)**: everything
  that loads a single component now goes through it — the executor's pgw#617
  substitution, the pgw#674 rotation preloader (`load_component_override` is
  a wrapper), and the benchmark. Quantized artifacts take their own lane
  exactly as `load_from_pretrained` routes a whole pipeline; svdq/gguf, which
  have no component-level loader, refuse by name
  (`ComponentLaneUnsupported`) rather than measuring something serving never
  runs.
- **No more identical-retry `except TypeError`**: it caught any
  construction-time TypeError (including one from deep inside quant-config
  reconstruction) and retried a path that failed identically, destroying the
  evidence. Whether a loader takes `torch_dtype` is now answered by
  signature inspection.
- **A diagnostic failure is its own outcome**: `SwapLatencyDiagnostics`
  returns `status="refused"|"failed"` with the cause on a SUCCEEDING job
  instead of raising. Raising fed the hub's fast-failure breaker — two
  invokes tripped `model_load_failure_streak` and recycled a warm pod that
  had just spent 26 minutes minting. A function that never serves tenant
  traffic cannot cost a serving pod.
- **`SwapLatencyInput.checkpoint`/`to` default to `""`**: a family-less
  `Slot(str)` cannot carry a curated policy, a fixed slot rejects every
  supplied value, and a required msgspec field cannot be omitted — without
  the defaults no payload exists that the hub accepts.
- **Discovery's out-of-package skip is LOUD**: `discovery/walk.py` dropped
  decorated classes defined outside the walked package at INFO, which is how
  an SDK diagnostics endpoint, re-exported exactly as its own docstring
  instructed, vanished from a release silently. Now a WARNING naming the
  class and the fix — subclass it locally, which is what the docstring says.

## 0.70.0 (2026-07-26) — pgw#677: tenant work always wins the GPU — the background mint yields

th#1187's promise was "serve at eager speed while the compile mint runs in
background". Measured reality (ie#546 retag cycle): the mint's 18 warm units
monopolized the per-instance run gate — 0 renders in 19 min on a fresh L4
release, every concurrent request degraded ~8.6x (completions landing exactly
as mint units freed the gate), and on sm_86 the ungated warm-thread compile
racing a tenant LoRA-branch forward SIGSEGV'd the worker process (pgw#676's
confirmed frame: `w8a8_lora._forward_with_branch` concurrent with
`hot_swap._run_warm`'s `compile_wrapper`). Two defects in one lock: the run
gate was too WIDE for latency (a "seed" unit could inline-compile for minutes
while holding it) and too NARROW for safety (the real compiles ran outside it,
against live tenant forwards).

- **Background-turn gate (executor)**: mint seed units AND shape-warm/heal
  compiles now run only inside a granted background TURN — single-flight with
  each other, holding the GPU permit + the instance's run gate + a new
  per-instance `turn_mutex`, granted only when the worker is tenant-idle
  (compile turns additionally wait a short quiescence window). A tenant
  request can wait at most ONE bounded background unit, ever.
- **Preemption**: a tenant admission cooperatively cancels the in-flight
  idle-granted seed forward (the driver re-queues the unit), so the boundary
  a tenant waits on is the handler's next cancel poll — not the unit.
- **Seeds never compile inline**: inside the new `hot_swap.mint_seed_window`
  a novel signature always routes EAGER + background enqueue, even on a
  degraded router. The VRAM-headroom degrade-to-inline-compile is retired for
  turn-gated routers (the warm thread ensures headroom inside its exclusive
  turn); ungated legacy routers keep it.
- **Race exclusion (the pgw#676 class)**: the shape-warm thread's compile
  executes the shared modules only while holding the instance `turn_mutex`,
  which the tenant path holds across adapter mutation + handler — concurrent
  branch-forward/compile execution is structurally impossible. The gate is
  loop-free (threading primitives) so the warm thread needs no event loop.
- **Minimum progress under sustained load**: a background lane blocked by
  continuous tenant demand STEALS one bounded turn per debt window
  (`_BG_STEAL_FLOOR_S`, duty cycle capped by `_BG_STEAL_DEBT_FACTOR`); stolen
  turns are not preemptible, so an 18-unit mint always finishes.
- **Honest metrics**: time a tenant request spends queued behind the instance
  gate is attributed to a new `instance_gate_wait` stage and excluded from
  `runtime_ms` — mint contention no longer bills as tenant compute (measured:
  16.9s reported for 1.95s of real work).
- Kill switch `GEN_WORKER_BG_YIELD=0` restores the pre-fix shape
  (red-verification and emergencies only).

Tapes: `tests/test_mint_gate_pgw677.py` — starvation shape red-verified via
the kill switch (tenant queued behind an inline-compiling mint unit), the
pgw#676 overlap red-verified impossible post-fix with the bounded wait
attributed, mint completion under a sustained tenant stream, and the
seed-window routing contract.

## 0.69.0 (2026-07-26) — pgw#685: one fused triton kernel for the nvfp4 activation quantizer

The `#nvfp4-w4a4` lane's per-call activation quantization was ~8 pure-torch
passes (fp32 upcast, per-16-block amax, e4m3 scale, divide, `searchsorted`
e2m1 cast, nibble pack, then a pad/permute swizzle of the scales into the
cuBLAS blocked layout). gw#540 measured that chain — not the fp4 GEMM — as the
whole reason the lane LOST to bf16. It is now ONE triton kernel.

- **`gen_worker.models.nvfp4_quant`** owns the nvfp4 format primitives (moved
  out of `w4a4.py`, which re-exports them unchanged) plus the fused kernel:
  per-16 amax → e4m3 block scale → e2m1 RTN cast → nibble pack → block-scale
  store **directly at its cuBLAS blocked-layout address**, in one pass.
  Measured on RTX 5090 (sm_120) and B200 (sm_100): the quant step 6-36x
  faster, taking the lane from 0.807x → 2.043x bf16 eager and 2.045x → 2.472x
  compiled on sm_120, and 1.03x → 1.24x on sm_100 (pgw#682).
- **Arch-portable by construction**: triton JITs per arch, so one source covers
  sm_100/103 (B200/B300) and sm_120/121 (RTX 50xx / PRO 6000) — no nvcc
  extension build, no per-family port. Launch config from the measured sweep:
  128 blocks/program on both arches, 8 warps on sm_120/121 vs 4 on sm_100/103.
- **`tl.math.div_rn`, deliberately.** Triton's `/` lowers to an approximate
  divide; a 1-ulp drift lands values exactly on the 1.75 e2m1 tie boundary and
  the ties-round-UP rule then flips a whole 4-bit code (0.16% of nibbles, 0.6%
  output drift — small enough to read as noise).
- **Arming is gated on BIT-IDENTITY against the pure-torch chain, not on a
  tolerance**, on three probe shapes at load. The existing load-time numerics
  self-check cannot see that class of bug (its threshold is 2e-2). A triton
  release that changes rounding, or an arch we have not measured, falls back to
  the reference chain rather than serving drifted numerics — never a refusal.
- **Compile-safe**: registered as a `torch.library.custom_op` with an explicit
  schema and a `register_fake`, so it is traced as an opaque call instead of
  graph-breaking. Registration happens at load, never mid-trace.

## 0.68.0 (2026-07-26) — pgw#681 guard-closure gate + boundary canonicalization; gw#679 per-expert MoE LoRA branch routing

Dynamo's guard set IS the exhaustive dependency list of a compiled graph, so
cell portability is now proven by construction instead of discovered one
guard-miss at a time. SDK-generic, parameterized only by the declared
contract — zero per-endpoint/per-family code (Paul's mandate).

- **Guard-closure gate** (`gen_worker.guard_closure`): every mint path
  (`finish_fleet_mint`, `mint_artifact`, `build`, the pgw#622 shape-warm
  republish) now extracts the complete live guard set per compiled graph
  (torch 2.13 structured `GuardManager` walk via
  `_debug_get_cache_entry_list` → `guard_manager.root` child/leaf traversal
  + `verbose_code_parts()`, repr-parse fallback) and classifies each guard
  by source root: module-rooted = weights/structure identity, global-rooted
  = code identity (both ck2-pinned), input-rooted + ambient = the CLOSED
  WORLD every guard type must be contract-covered in. An out-of-contract
  guard (e.g. an undeclared scalar baked in as `EQUALS_MATCH`) fails the
  mint RED naming the exact variable; an unprovable closure (no extractable
  graphs) also refuses. Downstream posture unchanged: pgw#672 degrades to
  explicit eager, never a pod death.
- **Boundary canonicalization** (`guard_closure.canonical_ingress`,
  installed by `compile_cache.apply` at the single compiled-graph ingress,
  mint and consumer symmetrically): tensor strides are pinned to the
  canonical contiguous layout — including the ie#544 trap where a size-1
  dim keeps `is_contiguous()` true under an arbitrary stride and
  `.contiguous()` is a no-op (residue rebuilt via `as_strided`); a
  stride-perturbed serving input now HITS the minted graph instead of
  paying a recompile (red-verified both directions). Per-path dtype drift
  raises a NAMED `GuardBoundaryError` instead of a silent recompile.
- **Full guard reporting**: the complete classified guard dump rides every
  cell as `metadata.json`'s `guard_manifest` (deterministic: comments
  stripped, ASLR ids scrubbed, rows sorted) — into CAS and the publish
  metadata. Audit surfaces: `guard_closure.audit_armed(pipeline, cfg)` for
  a live armed cell, and `python -m gen_worker.guard_closure <cells...>`
  as the runnable N-cold-pod closure/zero-miss check (exit 0 closed +
  consistent, 2 leaks, 3 divergence; per-host ambient state compared by
  presence, content kept in the dump).

**gw#679: a denoiser is a SET — per-expert LoRA branch routing for MoE pipelines.**

Wan 2.2 A14B is a dual-expert MoE (`transformer` high-noise + `transformer_2`
low-noise) and its Lightning distillation is two adapters. `branch_target()`
returned ONE denoiser, and diffusers' Wan converters rewrite every
non-diffusers key onto the `transformer.` prefix whatever expert the file was
trained for — so both halves landed rank-concatenated on the HIGH expert,
`map_adapter` succeeded, and the LOW expert ran undistilled weights on a
4-step distilled ladder. A wrong picture with a clean log. (Same failure
class as ie#522's `fuse_lora(components=["transformer"])`, at runtime attach.)

- **`branch_targets(pipe) -> {component: module}`** replaces `branch_target`:
  every branch operation runs over the whole denoiser set
  (`transformer`/`transformer_2`/`unet`). `enable_branch_lanes` /
  `clear_branch_lanes` / `disable_branch_lanes` / `apply_branch_adapter_set` /
  `pipeline_branch_bucket` / `stamp_lane(pipe)` are the set-level surface;
  the per-module primitives are unchanged.
- **Routing is DATA, not a wire field.** An adapter half declares its expert
  in its own keys (`transformer.` / `transformer_2.`), which is how diffusers
  already namespaces multi-denoiser pipelines. Per-expert mirrors carry the
  prefix; one repo carrying both prefixes works identically. A `component`
  field on the overlay was deliberately rejected — the fact is in the weights.
- **Fail-closed, three ways.** On a multi-expert pipeline an adapter that
  names no component is refused as ambiguous (checked on the RAW keys, before
  the converter can synthesize a `transformer.` prefix); a key naming a
  component the pipeline does not carry is refused on any topology; and a
  compiled pipeline whose experts are not all armed refuses instead of
  copying a half into nothing. Every refusal happens BEFORE any buffer is
  touched — a half-attached MoE (distilled expert beside an undistilled one)
  is never an outcome. The peft fallback is refused on multi-expert
  pipelines: diffusers reaches the second expert only through a
  `load_into_transformer_2=True` kwarg the fallback cannot pass.
- **The bucket container is per component.** `Compile(lora_bucket=N)` /
  `apply_lora_lane` arm every denoiser, and the whole set always shares ONE
  bucket, so the pipeline carries one coherent graph family. The lane STRING
  is unchanged (`<base>-lora<bucket>`): how many experts a family has is a
  property of the pipeline class, not of the lane, so published cell keys
  keep their meaning.
- **Single-denoiser lanes (LTX/sdxl/qwen) are untouched by construction**:
  with one target every denoiser key routes to it, prefixed or not, kohya-flat
  `lora_unet_` included (never read as a component declaration — sd-scripts
  emits it for transformer denoisers too), and no declaration is required.
- Endpoint note: a Wan MoE family should declare BOTH experts as compile
  targets (`Compile(targets=("transformer", "transformer_2", "vae.decode"))`)
  — the container is armed on both either way, and canonical branches on an
  uncompiled expert pay the eager branch tax.

## 0.67.4 (2026-07-26) — pgw#680: guard-miss doctrine — fail-on-recompile at serve time

The 187s incident class (ie#546 retag cycle): a tenant request whose inputs
missed every cached guard set paid dynamo's INLINE recompile inside the
request — and, single-flight, stalled every request queued behind it.
Doctrine (Paul, verbatim intent): "instead of compiling [inline], it should
throw an error, which we catch, then run in eager mode + compile [in
background] and note the mismatch."

- **Serve-window stance**: tenant request execution (the executor's
  `tenant_serve_window`, entered around the handler call only) runs guarded
  compiled targets under `torch._dynamo.config.error_on_recompile`, scoped
  per call via `config.patch`. Chosen over
  `torch.compiler.set_stance("fail_on_recompile")` deliberately: on torch
  2.13 ConfigModule user overrides are thread-local ContextVars, so the
  stance arms exactly the serving thread while the concurrent shape-warm
  thread and background mint keep compiling; and `error_on_recompile` fires
  only on a genuine recompile, composing with the multi-graph cache (warm
  entries serve under it; a first compile never trips it). `set_stance` is
  process-global and raises for any tensor frame. Warm/mint/adopt/boot
  windows never enter the window — they exist to compile.
- **The catch** (`compile_cache._guarded` / `_guarded_regional`): dynamo's
  `RecompileError` serves THIS request eager immediately (regional: original
  runs once under thread-scoped `config.patch(disable=True)` — block impls
  untouched), never the permanent-degrade path: no revocation, no tier flip,
  no quarantine — the lane is healthy for its known input classes.
- **The confession is data**: every miss emits a typed `guard_miss`
  activity event (`activity.emit_event` — one self-contained COMPLETED
  update that never displaces an open activity's `_current` beat): phase =
  the guard-reason class token (`compile_cache.guard_miss_reason_class`),
  detail = torch's verbatim reason + signature identity + cell key + request
  id (`postmortem.current_inflight_request`) + heal verdict — hub-countable
  per (release, SKU, guard-reason); top-N reasons are one grep away.
- **Background heal** (`hot_swap.Router.record_guard_miss`): the exact input
  class recompiles through the existing shape-warm driver (nice +10, own
  CUDA stream, zero-filled dummies of the failing request's args, dedup by
  signature) so the SECOND request of the shape is compiled. Signatures
  missing past `_GUARD_MISS_HEAL_LIMIT` (2) heals are per-request-volatile:
  routed eager permanently on every router (including never-concurrent
  mandatory lanes) instead of thrashing compile churn.

Red-verified end to end (`tests/test_guard_miss_pgw680.py`): real-torch
tapes (real dynamo guards, real RecompileError, real warm thread) + the
full `handle_run_job` tenant path; with the window neutralized (the pre-fix
tree) the same input pays a silent inline recompile with zero confession.

## 0.67.3 (2026-07-26) — th#1211: the svdq 5 GB guard cited a cap that does not exist

`MAX_SVDQ_FILE_BYTES` was `5 * 1000**3`, justified in-comment as an "R2
single-PUT cap". There is no such cap in this stack: the hub's
checkpoint-commit grant allows **64 GiB/file**
(`checkpointGrantMaxBytesPerFile`), uploads leave as presigned **multipart**
parts (the hub issues `part_urls`/`part_size`), and a 46 GB monolithic
`ltx-2.3-22b-dev.safetensors` has already been published raw through this
platform. The guard therefore refused **every** nunchaku-supported family
except z-image-turbo — qwen-image at 11.5–13.1 GB and flux.1 at 6.8–7.0 GB
were both blocked, even though the comment claimed flux fit.

- `MAX_SVDQ_FILE_BYTES` is now `64 * 1024**3`, aligned with the hub's real
  per-file ceiling, so it still refuses a genuinely absurd file. Comment
  cites `checkpointGrantMaxBytesPerFile` so the next reader does not inherit
  the myth.
- The real constraint the guard was groping for is unchanged and still
  honored: a nunchaku checkpoint must publish **whole**, because resharding
  strips the `__metadata__` its loader reads. Verified the svdq lane never
  reaches `clone.py`'s resharder — `build_svdq_flavor_tree` hardlinks the
  single file itself, and `publish` → `publish_flavors` → `HubClient` goes
  straight to the hub's part plan.
- Same myth corrected in `models/loading.py`'s `_single_file_checkpoint`
  docstring.

Unblocks: mirroring the official nunchaku qwen-image fp4 artifact (th#1211's
speed benchmark) and flux.1's 7 GB artifacts.

### th#1211 G4: `_PIN_MATRIX` gains nunchaku 1.3 <-> diffusers 0.39 (UNVERIFIED)

The matrix had ONE row (nunchaku 1.2 <-> diffusers 0.36), so
`check_svdq_stack_versions` raised for every other nunchaku minor. That left no
legal svdq stack for qwen-image, which serves diffusers 0.39: nunchaku 1.2.1 is
in the matrix but demands 0.36 and caps at torch 2.11, while nunchaku 1.3.0dev
has the torch-2.12/cu13.0 wheel but was refused outright.

- New row `SvdqPin((1, 3), (0, 39), (0, 40))`, marked **UNVERIFIED** in-comment:
  admitted on a STATIC signature check, with the live A/B still owed (closes
  during th#1211's P2). Add-then-verify follows gw#405's own precedent, and the
  row is inert until a nunchaku 1.3 wheel is actually installed — today only
  th#1211's benchmark-only serve variant.
- The static basis, recorded so the live check knows what to confirm: nunchaku's
  `forward` and diffusers 0.39's have drifted positionally (nunchaku keeps
  `txt_seq_lens`, 0.39 dropped it and added `additional_t_cond`, so position 6
  onward misbinds on a positional call) — but diffusers 0.39's
  `QwenImagePipeline` calls the transformer with **keyword arguments only** and
  passes neither drifted param, so the gw#405 positional hazard is void and no
  unexpected-kwarg error can fire on the t2i path. **Numerical equivalence is
  NOT established by this.** t2i only; `QwenImageEditPlusPipeline` unchecked.
- The 1.2 row is untouched and still rejects diffusers 0.39; the wheel-tag torch
  guard still fires (a torch2.12 wheel on torch 2.13 raises, as before).

## 0.67.1 (2026-07-26) — pgw#675 override dtype + pgw#676 native-crash attribution (the sdxl retag blockers)

### pgw#675: a component override now loads at the dtype the base tree's LOAD LANE computes at

`composition_compute_dtype` (pgw#647 gap #2) picked the compute dtype by
MAJORITY on-disk sniff. A produced `#fp8-w8a8` flavor quantizes only the
repeated-block Linears; every other tensor passes through at SOURCE
precision — so an fp16-mirrored fine-tune sniffs majority-fp16 (measured on
the live sdxl snapshot: 1902 F16 / 739 F8_E4M3 / 739 F32) while
`load_w8a8_pipeline` loads the composition at its bf16 compute default. The
fp16-fix VAE override therefore loaded Half into a bf16-activation pipeline
and EVERY forward through it — foreground eager warm, background-mint seed,
serve — died with `RuntimeError: Input type (c10::BFloat16) and bias type
(c10::Half) should be the same` (ie#546 sdxl finale: 3/3 workers, release
unprovable; the defect was latent since 0.64.0 and surfaced when the hub's
th#1134 fix + the finale's full-binding deploy first actually delivered a
`components.vae` override to a worker).

- `composition_compute_dtype` is now LANE-aware: a quantized-artifact tree
  (svdq / w8a8 / w4a4) answers the lane's bf16 compute default
  (`QUANT_LANE_COMPUTE_DEFAULT`, test-guarded against the loaders'
  `compute_dtype or torch.bfloat16`); the majority sniff only decides for
  plain trees. Binding-declared dtype still wins outright; pgw#667
  per-component facts still take precedence for fragile parts.
- Because every path (warm, eager, serve, background mint) forwards through
  the ONE instance built at setup, fixing the load fixes them all — there
  was no per-path cast to re-apply.
- Tapes (`tests/test_override_dtype_pgw675.py`, red-verified on the pre-fix
  tree): the REAL producer (`streaming_w8a8_cast`) builds a majority-fp16
  w8a8 tree, the REAL `load_component_override` loads a REAL tiny diffusers
  `AutoencoderKL`, and bf16 latents decode through it on CPU — pre-fix the
  override lands `torch.float16` (the exact crash precondition), post-fix
  `torch.bfloat16` end to end.

### pgw#676: native crashes are named, attributed, and stop crash-looping the pod

gen-worker 0.66.0 SIGSEGV'd (`exit_code=139`) on the 28-step CFG-on
`generate` shape on RTX A4500 (sm_86) — 6x across two pods, every request
burned 5 attempts deep, pod billed until th#878's wedge terminate — and the
hub saw NOTHING but the exit code. Degrade-never-die now extends below
Python:

- **faulthandler dump file** (`postmortem.enable_fault_dump`, on from the
  entrypoint): SIGSEGV/SIGABRT/SIGBUS/SIGFPE write every thread's Python
  stack to a file the surviving supervisor attaches — exit 139 carries
  frames.
- **In-flight markers** (`postmortem.note_inflight`): real requests and
  warm forwards (foreground AND background mint) stamp what is executing;
  a signal death attributes to the exact function + request id. Token
  stack, so overlapping executions all stay visible.
- **Per-pod crash-streak refusal**: the supervisor (and the next-boot
  container-death path) records the attributed crash; after
  `NATIVE_CRASH_REFUSE_STREAK` (2) signal deaths mid-flight on one
  function, `gate_functions` refuses THAT function on THIS pod —
  `native_crash_streak`, loud and typed — while siblings (turbo completed
  on the same A4500s) keep serving. The registry lives on the pod's fs:
  per-SKU-instance by construction.
- Hub half filed as th#1205 (placement rented the sm_86 card KNOWING it was
  below the preferred arch): respect the arch floor or gate on th#1198
  benchmark rows, and feed `native_crash_streak` into the fence.
- Tapes (`tests/test_native_crash_streak_pgw676.py`, red-verified): a real
  fork + real NULL-deref SIGSEGV produces an attributed post-mortem with
  frames; the gate refuses `generate` after 2 recorded deaths and leaves
  `generate-turbo` serving.

## 0.67.0 (2026-07-25) — rotation preload + the three recorded SDK gaps

Two lanes ride this train: pgw#674 (rotation preload) and the
pgw#667/#669/#670 SDK-gap sweep (+ gw#668, th#1174's SDK half).

### pgw#674: rotation preload — the NEXT checkpoint stages while the current job computes

Worker half of WORKER-RESIDENCY-DESIGN's Paul-ratified "Rotating
double-buffer serving" (north star: load model-B while model-A runs
inference; rotate on completion; GPU hot the entire time). Until now the
only path acting on the hub's desired plan was lifecycle's reconcile —
tenant-idle-gated and cancelled by every run_job — so every checkpoint hop
paid the full visible swap (ie#546 measured ~14s: 11s repo-cas pull + 3s
VRAM load).

- **`Preloader`** (`gen_worker/preload.py`, executor-owned): level-triggered
  background driver fed by the HelloAck desired set and poked at job
  admit/finish. NOT idle-gated, NOT cancelled by run_job; stops on drain.
  Stage ladder per desired instance: (1) bytes to local NVMe CAS
  (`ensure_local` — kills the download term); (2) `fits()` with the
  resident set protected -> full background `ensure_setup` = TRUE
  DOUBLE-BUFFER, dispatch finds a ready record and the visible swap is ~0;
  (3) otherwise COMPONENT-FIRST host staging: exclusive components (by
  content digest; resident shared TE/VAE stay put by construction) load on
  CPU on a dedicated nice+10 thread, get eagerly pinned, and are seeded
  into the shared-component cache — the existing content-keyed injection
  consumes them at dispatch (from_pretrained skips those disk loads).
  pgw#638 fence intact: refs resident under a moved identity are left to
  the idle-gated reconcile. Quantized flavors (fp8/svdq/w8a8/nf4) stop at
  the disk tier.
- **Copy-stream promotes** (`models/staging.py` + `pinned_swap`): weight
  H2D rides a dedicated CUDA copy stream — copy engines are separate
  hardware from the SMs, so a background promote overlaps the serving
  job's compute; only the copy stream is synchronized, never the device.
- **Bounded pinned pool** (`PinnedPool`): pinned host RAM is budget-gated
  (measured available minus the residency floor, capped at half of total;
  no knobs); refusal degrades to pageable. `prestage_module` builds the
  pinned cache eagerly for CPU-staged modules so their FIRST promote is
  full-PCIe.
- **Benchmark vehicle** (closes the ie#546 "no delivery path" gap): the
  swap-latency harness moved INTO the wheel
  (`python -m gen_worker.benchmarks.swap_latency`; the repo-root script is
  now a shim) and gained a `stage` case (disk -> CPU -> pin -> H2D on the
  copy stream — the exact rotation path). `gen_worker.diagnostics.
  SwapLatencyDiagnostics` exposes it as an ordinary worker function
  (str-slot snapshot trees, NoWarmup), dispatchable through the normal
  request path — th#1198's admin benchmark-run machinery.

### pgw#667: per-COMPONENT load dtype — wan's last migration blocker

A component's dtype is part of its resident identity and is decided AT
LOAD: upcasting a bf16-loaded tensor recovers no precision, it only hides
the truncation. wan-2.2's VAE must come off disk fp32 while the transformer
loads bf16, which the v2 SDK could not express — so wan kept its legacy
`Slot(str)` self-loading shape and forfeited the derived component tree,
the only wave-2 endpoint that could not migrate.

- **`gen_worker.families.facts.COMPONENT_DTYPES`** is the ONE home for that
  knowledge, keyed by the diffusers component CLASS name
  (`AutoencoderKLWan -> fp32`, with its reason carried as data so the loader
  logs WHY it widened a part). Keying on the class means every wan pipeline
  class (T2V/I2V/V2V/TI2V) and every fine-tune inherits the fact with zero
  endpoint declaration — the `SAMPLERS` posture: endpoint-private tables
  must not exist. Deliberately not catalog recipe data (a stability floor is
  an architecture fact, identical for every checkpoint) and deliberately not
  a `Slot(component_dtypes=...)` declaration (the sibling-as-part shape v2
  deleted).
- **The derived tree carries it** — `api.tree.component_dtypes()` resolves
  per-part classes from the pipeline class's `__init__` annotations at build
  time and from the snapshot's own `model_index.json` at load time (the
  latter authoritative: a fine-tune may substitute a class).
  `components_manifest()` publishes `dtype` beside `kind`, so the hub gets
  the precision in the same path vocabulary it already has for policy.
- **Applied at materialize** — `load_from_pretrained` passes diffusers' own
  per-component `torch_dtype` map (`{"default": bf16, "vae": fp32}`) instead
  of one scalar; a fact agreeing with the composition default stays out of
  the kwargs, so uniform trees are byte-identical. A loader that rejects the
  map collapses back to the scalar with a loud warning.
  `load_component_override` honors the fact for SUBSTITUTED parts too.

### pgw#669: `CompileAxis` can express a SPARSE legal set; `for_request` owns the per-request-state enumeration

`CompileAxis` partitions fields INDEPENDENTLY, so the derived warm plan is
the cross-product of those partitions. For an endpoint with real inter-field
constraints (ltx-video-2.3: **29 legal combinations of 120**) most of that
product is not a servable request, and every illegal row failed the WORKER
LOAD while the plan built its payloads — so declaring axes at all was a boot
hazard.

- **`IllegalCombination`** (new, exported): raise it from a payload
  `__post_init__` for an inter-field constraint. The plan drops exactly
  those rows and records a counted coverage claim ("5 of 12 axis
  combinations are declared IllegalCombination; 7 legal graph classes
  planned"); every OTHER exception still fails the load, so a sparse legal
  set and a synthesis bug are no longer the same event. It subclasses both
  `ValidationError` and `ValueError`, so the wire contract is unchanged (a
  request carrying such a combination is still a msgspec `ValidationError`).
  Axes admitting no legal combination, or neutral defaults that are
  themselves illegal, are declaration bugs and raise at the walk (CI /
  discovery), never at first request.
- **`view.for_request` clones EVERY sampler**, not just the attribute named
  `scheduler` (`discover_schedulers`, or an explicit `schedulers=(...)`). A
  pipeline with a second stateful sampler — ltx's `audio_scheduler`, which
  diffusers' own `__call__` drives with `retrieve_timesteps` and a `.step()`
  per denoise step — kept SHARING it across concurrent requests: a
  half-fixed pipeline that read as fully fixed. `sampler`/`objective`/
  `scheduler_config` still apply to the primary scheduler only (they are the
  denoiser's decision); secondaries are cloned faithfully from their own
  config, which is the privacy fix they need.

### pgw#670: `Resources(ram_gb_hint=)` — a measured host-RAM floor has a carrier again

The v2 cut deleted `ram_gb` on the reasoning that host RAM is an
opportunistic latency tier. For ltx-video-2.3 it was neither opportunistic
nor a guess: ie#484 measured 179-301 s mp4-encode and 147 s VAE-decode tails
on host-starved allocations at IDENTICAL GPU step-ms, and ie#492 sized the
floor at 64 GB from that failure. `vcpus` survived the same cut and covers
the CPU side of the very same encode tail, which made the deletion read as
accidental. `ram_gb_hint` is an ALLOCATION-time ask (never a runtime gate)
and does NOT imply `gpu=True`. `Resources.manifest_dict()` now owns the one
declaration -> wire-name mapping, projecting it under the builder's existing
`ram_gb` key (-> scheduler `min_ram_gb`: pod-create minimum plus th#740's
read-back-and-reject), so the floor actually reaches placement.

### gw#668: "no boot config was ever injected" is not "generation 0"

`WORKER_CONFIG_GENERATION` is injected only into pod-launch env, so a
host-process worker (the e2e local-worker shape, the dev loop, any BYO
fleet) never receives one. `min(gen, 0)` conflated the two facts and every
such worker reported `BOOT_STALE` for its whole life — with a remedy (pod
replacement) that does not exist for a pod-less worker (th#1172: every
local-worker e2e journey starved). `Settings.boot_config_generation` now
defaults to the sentinel `-1` = "never injected", and the boot config class
converges as NOT APPLICABLE for such a process. A pod-launched worker with a
genuinely old stamp still reports `BOOT_STALE` so the th#1087 rollout
replaces it.

**BREAKING (internal):** `Settings.boot_config_generation` /
`IntentRegistry(boot_config_generation=)` default to `-1`, not `0`. Anything
constructing an `IntentRegistry` for a pod-launched shape must pass the
stamp explicitly.

### th#1174 (SDK half): `ddim_trailing`

`gen_worker.view.SAMPLERS` defines `"ddim_trailing" -> ("DDIMScheduler",
{"timestep_spacing": "trailing"})` — Hyper-SD's published recipe, which the
hub already recognized in both sdxl schemas with no SDK definition to serve
it. Added to the `SdxlScheduler` literal as well (widening the enum is safe
on its own; no catalog row may be authored with the value until an endpoint
pins a wheel that has it).

## 0.66.0 (2026-07-25) — pgw#672: the minted cell serves itself; broken compiled lanes degrade, never die

Live defect closed (ie#546 burst rerun #2, 0.64.0, L4): a worker minted and
armed its ck2 cell, then failed its own finalize with
`CompiledLaneUnavailableError: ... did not serve their own warmup graph
(warmups=18, calls=18, cache_hits=0, cache_misses=0, compile_seconds=0.0)`
-> cell quarantined -> both functions disabled -> pod retired -> the
replacement re-minted the SAME key. 5 cycles, 4 dead workers, 6/36 requests
served. Root cause: dynamo's in-memory code cache (keyed on the
class-shared `__code__`; torch 2.13 inlined-module guards match any
same-class instance) serves the proof warmup of a LATER same-family arm in
a warm process — the pending capture stays empty (`finish_fleet_mint:
captured nothing`), and warm re-proofs of a published cell read 0 hits / 0
misses past the pgw#637 escape.

- **Honest proof windows** (`compile_cache.reset_target_code`): a scoped
  per-code dynamo reset runs immediately before every proof window — the
  foreground exclusive-GPU warmup, the pgw#671 background-mint seed, and
  hot adoption (`arm_staged_artifact`) — so the warmup MUST go through the
  real lookup path: a mint truly compiles into its capture, an adoption
  truly HITS its seeded FX entries. Sibling-safe: the live cache root is
  an additive union, so a sibling whose shared code is dropped re-traces
  into an FX hit (seconds), never a recompile.
- **In-process finalized-mint reuse** (`fleet_cells._FINALIZED`): a cell
  key this process already minted + folded re-arms `cache_ready` from the
  live root (proven by a real FX hit) instead of opening a doomed second
  capture.
- **Degrade, never die (Paul's doctrine; also pgw#673):** a failed
  serve/finalize proof, a failed compiled call (sm120 `InductorError:
  CantSplit` class), or a tripped runtime guard now DEGRADES the object to
  explicit eager serving — mandatory (w8a8/w4a4) lanes included. The
  compiled identity is revoked (serving_tier flips to `"eager"` on the
  wire, th#1187 field), the confession rides the activity stream
  (`Activity.note`), the functions STAY dispatchable, and the pod stays
  up. `compile_cell_failed` self-disables and the
  `all_declared_functions_disabled` retire no longer exist on these paths.
- **In-process cell quarantine** (`record_cell_quarantined`, key-normalized
  refs — also fixes the th#1166 exact-string false-negative class in the
  proven-cell registry): a proof-failed identity is never re-selected or
  re-minted by the same process, breaking the fail/re-mint churn loop.
  Cross-worker (hub-side) quarantine visibility remains a tensorhub
  follow-up.
- Red-verified over the REAL `ensure_setup` + fleet_cells miss policy in
  `tests/test_serve_finalize_pgw672.py`: the pre-fix tree reproduces the
  exact live signature (`cache_hits=0, cache_misses=0,
  compile_seconds=0.0` + empty capture); the fixed tree mints, hits, and
  finalizes.

## 0.65.0 (2026-07-25) — eager-first boot: READY in ~2 min, the mint compiles in the background

pgw#671 (worker half of th#1187, Paul's ruling): the startup ladder no
longer serializes `pipeline_loading -> self_mint_compile -> ready`. On an
eager-compatible lane whose boot would self-mint, the worker goes READY at
the **EAGER tier** as soon as weights load and the derived warm plan's
eager pass completes, serves real requests at eager speed, and runs the
whole mint as a background task through the pgw#622 hot-swap routers —
then hot-swaps to compiled when the cell arms (its own mint finishing, or
a peer's upload adopted opportunistically). The pre-0.65 behavior — first
render blocked 15–30 min behind the full trace/proof — is gone.

- **Wire contract (th#1187):** `FunctionCapability` gains field 9,
  `serving_tier` (`"" | "eager" | "compiled"`), carried on READY only. A
  tier field, deliberately NOT a new state enum value — an old hub ignores
  it and keeps dispatching (fail-closed to today's behavior). The tier
  flips `eager -> compiled` on arm with no capability-state flap.
  "Serving eager while minting" needs no new signal: it is the hub-derived
  pair (READY + `self_mint_compile` activity running); the activity now
  outlives READY and terminates from the background driver.
- **Background driver:** seeds the FULL derived plan through the routers
  (each novel signature serves eager and compiles on the router's warm
  thread — nice +10, dedicated CUDA stream, VRAM-headroom guarded), waits
  for the compiles, proves with the standard cold-proof (a successful
  compiled call on a fresh capture), packs and publishes via the existing
  attested gate, and activates the identity on the live targets.
  Interference bound: tenant work is preferred (idle-gated units,
  single-flight with real requests) — at most one eager forward.
- **Adopt-on-arm, clean abandonment:** a peer cell arriving mid-build
  (ModelOp ADOPT) abandons the local build cleanly — finish the current
  unit or discard wholesale; nothing half-packed is ever advertised or
  published. Vacate/shutdown abandon the same way.
- **A mint failure never un-serves:** the function stays READY(eager) for
  the process with a typed activity failure on the wire.
- **Unchanged on purpose:** mandatory-quantized lanes (w8a8/w4a4) keep the
  sequential fail-closed boot (eager is not a production lane there,
  gw#586); delivered-cell boots keep their ~0-cost sequential proof;
  0.64.0's warm-inheritance refusal while a self-mint capture is pending
  stays (purely local sequencing — the capture must trace every graph).
- Kill switch: `GEN_WORKER_EAGER_FIRST_BOOT=0` restores the 0.64.0
  ladder.

## 0.64.0 (2026-07-25) — the warm tax dies: contract-keyed warm runs; wall clocks become liveness

The P0 for the sdxl multi-checkpoint stress test (pgw#654 warm-tax fix,
measured by the ie#546 canary on 0.61.0), plus gw#665/gw#666 (the last
fixed wall clocks become liveness checks — ONE breaking deletion) and
pgw#647 gap #2.

### pgw#654 — warm RUNS are contract-keyed, never instance-keyed

The 0.61.0 derived warm plan re-ran per CHECKPOINT INSTANCE: first boots
took ~28–32 min (18 real eager denoises at 1MP+; was ~3.3 min on 0.60.x)
and every juggle swap paid a ~9-min warm re-run on top of its genuine
~74s transfer+load. Multi-checkpoint juggling — the v2 flagship — now
works as designed:

- **Warm-run memory is keyed by CONTRACT** (endpoint class + per-slot
  precision-lane facts + component overrides — never the checkpoint ref).
  The plan executes once per contract per process; a further checkpoint
  instance of the same family runs ONE clamped verification job (proves
  the fresh weights compose and forward pre-READY; supplies the calls>0
  the compile proof needs). Swap cost is transfer + VRAM load, bounded by
  the residency path, never a warm-plan re-run. Allocator pool,
  cuBLAS/cuDNN heuristics and dynamo's in-memory compiled code are
  process-global — that is what makes the inheritance sound. Inheritance
  is refused while a self-mint capture is pending or an armed cell is not
  yet proven in-process (those boots keep the full plan and full proof).
- **Eager lanes stop paying the cross-product.** When nothing is armed or
  minting, the boot warm runs ONE shape representative per (function,
  guidance class) — allocator peak and kernel-heuristic warm are
  shape-driven, not coverage-driven; nothing is being traced. Numeric
  shape axes (megapixels-style) keep their max-area bucket; enum axes
  keep the first declared class. The full class x bucket product still
  runs whenever a compile artifact is armed or minting.
- **Synthesized warm payloads clamp step fields** (`num_inference_steps`
  / `steps`) to their declared floor (`msgspec.Meta` ge/gt honored) —
  the traced graph and the allocator peak are step-count independent, so
  a warm run never pays a full recipe's steps even on an endpoint that
  forgot `ctx.boot_warmup`. A `@worker_function(warm=...)` override still
  wins.
- New pod-side benchmark: `benchmarks/swap_latency.py` — per-component
  disk->VRAM load, VRAM->host-RAM demote, resident re-pick promote,
  component-first swap by content address (in-place copy vs replace; a
  DMD-distilled sibling is the same case), H2D copy-stream overlap with
  compute-interference measurement. Refuses to run off-pod.

### pgw#647 gap #2 — component overrides inherit the composition's dtype

`load_component_override` resolves dtype as: base binding's declared
dtype, else the base COMPOSITION's compute dtype
(`composition_compute_dtype`: fp8-stored bases map to their bf16 compute
default), else — last resort — the override's own on-disk dtype. The old
override-on-disk fallback loaded the fp32-stored `sdxl-vae-fp16-fix` into
a bf16 pipeline and setup died on the first latent (ie#546 canary, 2/2
pods). The VAE-override deploy path is unblocked.

### `ctx.adjustments` — the public read side of the adjustment ledger

`RequestContext.adjustments` returns an immutable tuple of the
`ctx.adjusted`/`ctx.clamp` rows. Endpoint test suites migrate off the
private `ctx._adjustments` read at their next relock.

### gw#666 — BREAKING: `boot_timeout_s` is DELETED (fixed durations -> liveness)

The last five gen-worker wall clocks became liveness checks
(`gen_worker.stall`: `SilenceWindow` over the engine's own output +
`ProgressFloor`). `ServerProcess`/`vllm_server`/`llama_server` no longer
accept `boot_timeout_s` — pass `stall_window_s` (silence window, not a
boot budget) if you must tune it; an engine that keeps talking boots for
as long as it needs. **Migration (ie#558): both qwen3.6 endpoints on ie
master (`qwen3.6-35b-a3b`, `qwen3.6-27b-mtp-gguf`, currently pinned
0.61.0) pass `boot_timeout_s=1800` — delete the argument in the same
commit that bumps their pin, or the import dies at decoration time.**

### gw#665 — the conversion toolchain is bounded by silence, not a 2-hour wall

Conversion subprocesses (`subproc.LineTail`) are killed on output
SILENCE, not on a fixed wall clock — a talking 3-hour GGUF conversion
finishes; a silent hang dies fast.

## 0.63.0 (2026-07-25) — debt sweep: a worker never advertises a model it does not have

Four fleet-debt issues from Paul's audit (pgw#655 / #656 / #657) plus the SDK half of
ie#554 (pgw#663). One new public API; three behaviour changes worth reading before
upgrading.

### pgw#663 — one guarded URL fetch for endpoint code (NEW PUBLIC API; ie#554)

`gen_worker.fetch_bytes` / `fetch_image` (and `url_fetch.open_guarded_stream`
for streaming callers) replace the hand-rolled `urlopen(url).read()` that
endpoints accepting caller URLs were each writing for themselves:

- scheme, destination (private/loopback/link-local/metadata addresses refused)
  and an optional host allowlist — caller-level plus a deployment-wide outer
  bound in `GEN_WORKER_URL_FETCH_ALLOWED_HOSTS`;
- **redirects are followed manually, with the policy re-applied to every hop.**
  `urlopen` follows them silently, so a pre-flight check on the caller's URL
  said nothing about where the bytes came from — a public URL 302-ing to the
  cloud metadata service was the one-line bypass. `input_assets` had the same
  hole on its caller-transport path and now goes through the same opener
  (resolver-minted units keep their internal-object-host exemption);
- a streamed byte cap that does not depend on the server declaring a size;
- MIME matched against the SNIFFED type.

Refusals are `ValidationError` (caller-caused), transport failures
`RetryableError`. Documented limit: per-hop validation does not close DNS
rebinding — see the module docstring.

### pgw#655 — READY-without-the-model, and a download bound that was off (P0)

- **Boot prefetch failure now GATES the function.** A failed startup prefetch of
  an hf/civitai-source ref used to log "failed terminally" and walk on to READY.
  Function-shaped (`cls=None`) and non-inference functions are advertised
  unconditionally, so the hub then dispatched paid GPU jobs that each
  re-discovered the missing model as a per-request load failure. Those functions
  now go `FnUnavailable(reason="model_unavailable", axes={"ref": …})` instead.
  Per function, never the process: a sibling whose model landed keeps serving.
  Hub-resolved slot refs are unaffected — they arrive by delivery, not prefetch.
- **The HF download bound is a progress-RATE floor, not a wall clock.**
  `_HF_DOWNLOAD_MAX_SECONDS` was `0.0` (off) for its entire life, and the stall
  watchdog reset its window on ANY byte — so a trickle pinned
  `DOWNLOADING_MODELS` forever. A transfer must now put
  `_HF_DOWNLOAD_MIN_WINDOW_BYTES` (8 MiB) on disk within the 180s window or it
  raises `DownloadStalledError`. That is ~46 KiB/s: three orders of magnitude
  under any real pod link, and still 60+ hours for a 10GB checkpoint. The dead
  wall-clock knob is deleted, not defaulted.

### pgw#656 — dataset ingest never creates a duplicate it could have found

`publish_dataset_revision`'s existing-dataset lookup had three ways to come back
empty, and empty means CREATE:

- a swallowed JSON-parse error **and** a non-2xx response that simply fell
  through — both now raise (`AuthError` on 401/403, `RuntimeError` otherwise);
- one unpaginated page (tensorhub defaults `limit` to 50) — now walks
  `next_cursor` at the API's 200 cap, with a no-movement/page-ceiling guard;
- `?tenant=`, which the hub's `listDatasets` never reads — now `?org=`.

The `__cozy_kind__` / `__cozy_dataset_info__` / `__cozy_snapshot_manifest__`
features_json squat needs hub columns first: tracked as th#1162.

### pgw#657 — fail-loud hardening

- `install_hf_http_timeouts()` now **proves** the huggingface_hub timeout floor
  on the client hf will actually build, and raises `HfHttpFloorError` at boot if
  it cannot. The patch reaches into hf's private backend (already reshaped once,
  requests -> httpx); a silent revert puts the fleet back on infinite HTTP
  timeouts (gw#456) with nothing to say so.
- The gw#640 boot record picks a **durable** carrier
  (`GEN_WORKER_BOOT_RECORD` > the model-cache volume > `/tmp`), detects a
  tmpfs/ramfs carrier, records `carrier_volatile` in the record itself, and warns
  loudly — a record on RAM is freed by the very OOM it exists to report.
- `Source.iter_hf_components()` on a diffusers-singlefile source is now a typed
  `ValidationError` naming the supported layouts, replacing a
  `NotImplementedError` that the docstring promised worked.
- Compile-evidence probes say why they failed: a silent
  `has_inmemory_compiled_code` false negative kills a healthy compiled lane, and
  a silently-degraded `runtime_key()` computes a different cell key than every
  healthy pod (a guaranteed miss + re-mint).
- Three inline `gc.collect` + `empty_cache` copies in the executor fold into
  `models.memory.aflush_memory` (`reset_peak=False` — pgw#652's activation
  learning reads `max_memory_allocated`).

## 0.62.1 (2026-07-25) — gw#661: a retrying worker no longer reports itself failed

A setup/warmup loss whose contract is "will be re-attempted" now reports
`ACTIVITY_STATE_RUNNING` with a `retrying (attempt N/5)` detail instead of the
`ACTIVITY_STATE_FAILED` terminal. Exhausting the budget reports FAILED and
disables the function (`reason=retry_exhausted`), so the hub still gets the
terminal truth.

- **Premise correction.** gw#661 was filed as "`RetryableError` is mapped onto
  startup `phase=error`". It is not: `WORKER_PHASE_ERROR` has exactly five
  sources (mandatory-command rejection, config-snapshot write failure,
  `UnreportedIntentWait` out of residency reconcile, config-snapshot failure,
  drain failure) and none of them is the self-mint compile path. The signal the
  incident actually quoted (`activity failed kind=self_mint_compile
  phase=warmup_forward error=RetryableError: ... retrying`) is the
  **ActivityUpdate** carrier, and that is the one that was lying.
- **Why it condemned pods anyway.** The hub's th#1160 progress-aware verdict
  reads `lastWorkerProgressLocked`, which *excludes* activities in state
  `failed`. Declaring a will-retry attempt FAILED therefore erased exactly the
  progress evidence that keeps a working pod alive — the worker was deleting its
  own defense. Measured 2026-07-25: 4 condemnations, 4 self-mint compiles that
  then COMPLETED, one finishing 53s after its pod was condemned.
- `CompiledLaneUnavailableError` subclasses `RetryableError` but the worker does
  give up on it (it disables every handler needing the unproven lane), so it
  stays on the FAILED rung. Job-path `RetryableError` -> `JOB_STATUS_RETRYABLE`
  is unchanged; `ConfigSnapshotWriteError` -> `WORKER_PHASE_ERROR` is unchanged
  (nothing re-attempts a failed snapshot write worker-side).
- No proto change: `WorkerPhase` and `ActivityState` are untouched, so this
  needs no hub lockstep. A distinct `retrying` wire rung remains available as
  follow-up if the hub ever wants to distinguish it from ordinary running.
- The hub-side defenses (th#1157 debounce, th#1160 progress-aware verdict) stay.
  They are correct independent of this fix — defense in depth.

## 0.62.0 (2026-07-25) — th#1130: the image encode+upload tail overlaps the next request

**No endpoint changes required.** `ctx.save_image` now DEFERS its encode +
C2PA stamp + upload to the finalize tail: the executor releases the GPU permit
at handler return (as it always did), then drains the deferred outputs, so the
webp encode (~250ms for a 1024^2 frame, ~1.1s for 1328^2 at q95) and the upload
run while the next request is already denoising. Previously they ran inside the
handler, holding the permit — `finalize_wall_ms` was 0 on every image endpoint.

- Why deferral and not a terminal marker: th#1107's
  `_release_gpu_slot_for_finalize` is terminal and once-only, so it could not be
  copied onto `save_image` — endpoints save mid-pipeline and in N-image loops,
  and the first of N saves would have freed the permit with GPU work still to
  come. The terminality signal already existed: the handler's RETURN. `save_image`
  releases nothing; the executor's existing post-handler release is the seam.
- Safety, by construction: pixels are SNAPSHOTTED at save (`image.copy()`, ~2ms
  against a ~250ms encode) so mutate-after-save cannot change the upload;
  reading a bytes field (`size_bytes`, `sha256`, `inline_bytes`, ...) inside the
  handler forces the encode inline (correct, just not overlapped, and logged);
  the drain runs before the OK result is built, so a failing encode fails the
  request instead of shipping a hollow asset. Format validation and the ref's
  extension stay EAGER (a bad `format=` still raises at the call).
- Not armed for streaming handlers (they serialize items mid-handler), CLI runs
  or endpoint unit tests — those stay exactly as they were.
- th#1111: the stage window now covers the tail, so `image_encode`/
  `credential_stamp`/`upload` land in `total.tail` and the map still closes
  exactly against `runtime_ms`. `FINALIZE_OVERLAP` gained `handoff=handler|
  executor`; `runtime_ms` is unchanged in meaning (the same work, reordered).
- New: `gen_worker.io.image_format(format) -> (PIL format, extension)`.

## 0.61.0 (2026-07-25) — pgw#654: the objective/distilled vocabulary train (v2.1)

**HARD CUT on the v2 vocabulary — the wave-1 endpoints (sdxl, z-image,
qwen-image, ernie, anima) relock to `gen-worker==0.61.0` in the same
train; wave 2 (ie#546) builds on this release.** No aliases, no shims:
old declarations fail loudly at import/decoration time. What every
endpoint author must change:

- **The 3-value inference "regime" enum is SPLIT into two orthogonal
  checkpoint facts, and the word "regime" is retired.**
  `objective: "epsilon" | "v_prediction" | "flow"` is what the network
  predicts — it drives scheduler math at VIEW construction
  (prediction_type + zero-terminal-SNR rescale for v-pred, folded in so
  no endpoint can forget it; flow-match scheduler classes required for
  flow — a diffusion sampler on a flow checkpoint raises).
  `distilled: bool` is a post-training property — it drives the recipe
  (few steps, CFG off), the graph choice (cfg_off/batch-1 class) and
  adapter policy (never stack a distillation LoRA on already-distilled
  weights). `ResolvedSlot.regime` -> `.objective` + `.distilled`;
  `resolve_slot(inference_regime=, allowed_regimes=)` ->
  `(objective=, distilled=, allowed_objectives=, allowed_distilled=)`;
  `RegimeMismatchError` -> `ObjectiveMismatchError`; `REGIMES` ->
  `OBJECTIVES`; `for_request(regime=)` -> `for_request(objective=)`.
  The wire adds `ModelBinding.objective`/`.distilled` (hub-stamped; ""
  = unstamped, and the declaration backstop only fires on stamped facts).
  Conversion payloads rename `inference_regime` -> `objective` +
  `distilled` (`apply_regime_scheduler_config` ->
  `apply_objective_scheduler_config`).
- **Per-function declarations replace the class-level name-keyed dicts.**
  `@endpoint(regimes={...})` and `@endpoint(warmup={...})` payload dicts
  are decoration-time errors. Declare per handler, at the definition
  site: `@worker_function(objectives=("epsilon", "v_prediction"),
  distilled=False)` — omit `objectives` for unrestricted handlers, omit
  `distilled` for either. Works identically on class methods and bare
  `@endpoint` functions. The manifest emits `objectives`/`distilled` per
  function (both omitted = unrestricted).
- **The boot warm plan is DERIVED — delete your `warmup=` payloads.**
  Defaulted fields keep their schema defaults (post-th#1116 = the
  neutral recipe), `CompileAxis` fields cross-product their classes'
  `warm=` representatives (that product IS the graph set, so "fully
  warmed?" is computable), required no-default fields synthesize neutral
  values (short string for prompts, tiny generated image/audio for
  media). One run per (function, graph class) — every alias proves its
  own code path causally; re-executions of an already-traced graph are
  cache hits, never second traces. Cheapen non-graph work on
  `ctx.boot_warmup` (`steps = 1 if ctx.boot_warmup else steps`).
  `@worker_function(warm={...}, warm_reason=...)` survives ONLY for a
  non-axis field that genuinely changes tracing (mandatory recorded
  reason; axis fields are rejected). `warmup=NoWarmup(reason)` and a
  custom `warmup()` method remain; the custom-warmup proof now
  attributes to every contract-compatible sibling (the
  `warmup={...: None}` per-alias opt-out died with the dict — an
  incompatible sibling is a separate `@endpoint` class).
- **The class cell contract is the UNION across sibling functions
  (pgw#647 gap #1 — the sdxl/z-image/qwen strict-xfails flip green).**
  `CompileCell.guidance_scales` derives as the union of the class's
  payload-axis warm sets, so a distilled sibling with no guidance field
  shares the family cell instead of failing closed on w8a8 lanes (or
  silently serving eager on w8a16). Union is per CLASS — sibling
  `@endpoint` classes keep their own contracts. The ck2/execution
  contract digests changed (v3): existing cells re-mint on first build.
- **Per-lane text pins (gap #6).** `@worker_function(text_len=)`
  overrides the class `Compile.text_len` for that handler's lane; the
  contract digests the sorted UNION of the class's pins (`text_lens`),
  so a dual-lane class (qwen: 512 t2i / 1024 edit) describes both and
  the forge warms every pin. Delete module-constant pin workarounds.
- **Multi-slot classes declare their root (gap #7).** Two or more model
  slots with no slot named `pipeline` is now a decoration-time error
  unless exactly one is marked `Slot(root=True)`. `ctx.defaults` /
  `ctx.for_request` resolve the root; non-root lanes pass
  `ctx.for_request(pipe, slot="edit")` / read
  `ctx.slots["edit"].defaults`. The old silent standard-regime fallback
  on ambiguity is GONE — ambiguity raises.
- **`RuntimeFormula` evaluates on RESOLVED EFFECTIVE values (gap #4).**
  A referenced field may be `Optional[...] = None` when the handler's
  derived defaults schema carries the same-named field: validation
  accepts it and the worker evaluates payload-over-`ctx.defaults`.
  Endpoints that dropped their steps terms (z-image, ernie, anima)
  can restore them.
- **Non-diffusers `lora_bucket=` roots must declare `lora_state_dict`
  (gap #9)** — decoration-time error otherwise (adapters previously
  failed closed at first live attach). `Path`-typed modifier slots are
  now rejected like `str` ones (gap #8).
- **`ctx.adjusted(field, requested, applied, reason)` / `ctx.clamp(...)`
  — caller-visible adjustment warnings.** Whenever the serve path
  modifies a requested value (guidance clamped by `max_guidance`,
  neutral defaults substituted for an unconfigured checkpoint, ...),
  record it; rows ride `JobResult.adjustments` and the hub persists them
  on the request record + events stream. Adjustments WARN-AND-SERVE;
  catalog-LOCKED recipe fields refuse typed upstream instead.
- **`gen_worker.view.SAMPLERS` is the ONE sampler definition (absorbs
  gap #2).** Added `euler_trailing` (Lightning recipe); the `dpmpp_2m*`
  entries now carry `solver_order=2` + `final_sigmas_type="zero"` as
  part of the sampler's definition. Endpoint-private sampler maps are
  deleted in the wave-1 relock.
- **`ModelRef.label`** — the one `model_used`/logging label (wire form
  per source registry); endpoints delete their `_ref_label` copies.
- **Runtime owns generic tuning.** The executor sets TF32 once at
  bootstrap and disables per-request progress bars at materialization —
  endpoints delete their `_tune_pipeline` copies.
- **Convention: imports at module top.** No function-body imports unless
  breaking a genuine cycle or deferring an optional extra.

## 0.60.1 (2026-07-25) — pgw#654: the false fleet-wide "model load failure"

**Fixes the 0.56.0–0.60.0 defect that broke every real model release on the
new floor (ie#544: qwen-image, z-image, ernie — 3/3 `release_broken` /
`model_load_failure_streak`).** The worker loaded its models fine and went
READY; the first real jobs then killed it: the post-RunJob residency re-pass
asked `ensure_intent` for already-converged (terminal SUCCEEDED) command
work, got `""` back by design, and `reported_await("")` armed `guard_await`'s
2.0s unreported-wait fail-closed while `wait_idle()` blocked on the in-flight
job — `UnreportedIntentWait` → `WORKER_PHASE_ERROR` exactly 2s after first
dispatch → the hub counted a startup failure on a healthy worker and tripped
the release breaker. Toy/CPU gates never caught it because their jobs finish
inside the grace period.

- `IntentRegistry.ensure_intent` now mints a worker-local compat carrier for
  re-verified command work instead of returning `""` — every long await
  stays reportable; the fail-closed guard remains for waits that genuinely
  carry no intent id.
- `apply_command` supersedes COMMAND-BORN intents only. Worker-local
  job/setup/materialize obligations survive a generation bump — previously a
  bump terminalized a live job's intent mid-flight.
- Observability (the reason this took a fleet outage to see): every
  `WORKER_PHASE_ERROR` entry now dials its cause through the gw#640
  worker-fatal carrier (`Lifecycle._enter_error_phase` →
  `report_worker_error_async`), so the hub persists a durable, queryable
  `pod_events` row (class `hardware_unsuitable`, reason `worker_fatal`) with
  the exception + traceback. The hub previously stored only
  `detail="worker phase reported error"`, and the worker's lifecycle
  snapshot carrying the failed intent was itself dropped by hub shadow
  validation — double blindness.

## 0.60.0 (2026-07-25) — SDK v2: THE breaking cut (pgw#647)

**HARD CUT, no compatibility window.** Design of record:
`SDK-API-V2-DESIGN.md` (tracker root); the campaign that consumes it is
ie#546 — every endpoint is rewritten against this release, and this release
IS the API freeze. Old declarations fail loudly at import/decoration time
(there are no shims and no deprecation aliases). What every endpoint author
must change:

- **The component tree is DERIVED from the pipeline class — sibling-as-part
  slots are DELETED, not migrated.** Declare only the root:
  `models={"pipeline": Slot(StableDiffusionXLPipeline, selected_by="model")}`.
  `pipeline.unet` / `pipeline.vae` / `pipeline.text_encoder(_2)` /
  `pipeline.scheduler` are addressable automatically; the derived tree
  (classified weights vs config) is published into the release manifest
  (`functions[].slots[].components`) for per-path policy
  (fixed | curated | open) and component-level routing. A sibling slot that
  names a component of another slot's tree (`"vae": Slot(AutoencoderKL)`)
  or a `str` modifier slot next to a pipeline slot (z-image's
  `turbo_lora: Slot(str)`) is now a decoration-time error. The SDXL VAE-fix
  override becomes CATALOG DATA (th#1116), not endpoint code — and with it,
  `setup()` loses its wiring role (`pipeline.vae = vae.to(...)` is gone).
  Explicit multi-slot declaration survives ONLY for non-introspectable
  runtimes (llama/gguf, custom engines).
- **Handlers own their instance: `self.pipeline`, not per-handler model
  args.** `setup(self, pipeline, ...)` runs once per instance and stores
  the pipeline; handlers are exactly `(self, ctx, payload)` — extra
  injected model params on class handlers now raise, and a class with
  `models=` requires `setup()`. One live instance == one binding set: the
  "which checkpoint am I?" question does not exist, the runtime routed the
  request here because the bindings match. `ctx` keeps only request-scoped
  facts (resolved refs/digests, typed defaults, progress/cancel, request
  id).
- **Per-request VIEWS replace instance mutation.** `ctx.for_request(
  self.pipeline, sampler=p.sampler, seed=p.seed)` returns a container copy
  sharing every module by reference (the compiled graph stays bound — the
  view WRAPS, never swaps) with its OWN scheduler cloned from config. The
  SDK owns the sampler table (`gen_worker.view.SAMPLERS`) and applies the
  resolved checkpoint's regime (v_prediction) at view construction.
  Assigning `self.pipeline.scheduler` per request (sdxl's old
  `_ensure_scheduler`) was a live concurrency-corruption bug — diffusers
  schedulers are stateful and shared — and must be deleted, along with
  hand-rolled handler locks (`@endpoint` single-flight, shipped in 0.58.0,
  owns that) and `_scheduler_kind` / `_scheduler_config` maps.
- **`Resources` declares only what the endpoint CANNOT RUN WITHOUT.**
  `vram_gb`, `ram_gb` and `compute_capability` are DELETED (th#683
  profiling measures the real VRAM per lane/shape; host RAM is an
  opportunistic tier; precision-per-card is the fit ladder's call — the
  worker no longer refuses pods on a declared compute capability).
  Surviving fields: `gpu`, `gpu_count`, `libraries`, `strict_vram`,
  `vcpus`, plus `vram_gb_hint` — an optional FIRST-BUILD placement hint
  only, never a gate or reservation. sdxl's
  `Resources(vram_gb=12, ram_gb=48, compute_capability=8.0)` becomes
  `Resources(gpu=True)`.
- **Compile axes moved onto PAYLOAD FIELDS.** `Compile(guidance_scales=)`
  is DELETED; annotate the field with its equivalence classes::

      guidance_scale: Annotated[float, CompileAxis(classes=(
          AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
          AxisClass("cfg_on",  match=lambda v: v != 0, warm=5.0),
      ))] = 5.0
      aspect_ratio: Annotated[AspectRatio, CompileAxis(classes="enum")]

  Each class carries a WARM representative, so the warm plan is DERIVABLE
  (classes x shape buckets) and catalog recipes validate against the
  declared class names at publish time. The manifest carries the projection
  (`compile_axes`).
- **The shape contract is DECLARED to the compiler, never discovered.**
  The compile encoding is now `automatic_dynamic_shapes=False` +
  `assume_static_by_default=True` + `torch.compile(dynamic=None)` + explicit
  marks (ie#543: `dynamic=False` + `mark_dynamic` raises
  ConstraintViolationError and is not expressible). NEW REQUIRED AXIS —
  every inference `compile=` endpoint must declare its text-sequence axis
  or fail at decoration (the ie#544 lint): `Compile(text_len=<pinned token
  length>)` (pad embeddings with `gen_worker.pad_text_sequence`, which
  produces canonically-strided allocations — dynamo guards on strides, and
  a pin that fixes only the size is not a pin), `text_len=0` for an
  explicitly unconditioned model, or a declared range
  `Compile(dynamic=(DynamicDim("sequence", min=.., max=..),))`. BATCH stays
  a supported declarable axis (`DynamicDim("batch", min>=2, max=..)`) even
  though no endpoint uses it today (ie#542/ie#543: ~1.0-1.2x at our
  operating point; min>=2 because torch 0/1 specialization is not
  overridable).
- **`lora_bucket` moved to the decorator:** `@endpoint(lora_bucket=64,
  compile=Compile(...))` — it shapes the resident branch and the graph
  family whether or not compile is armed. `Compile(lora_bucket=)` is gone.
- **Cell keys are `ck2` and carry a shape-contract digest.** The declared
  contract (shapes, targets, text_len, dynamic dims, regional, lora bucket,
  warm guidance classes) is digested into a new REQUIRED `contract` axis
  and recorded in artifact metadata (`shape_contract`), so a worker on a
  newer contract can never consume an older incompatible cell. All ck1
  cells are naturally invalidated (this release's `gen_worker` version axis
  changed anyway).
- **The config type is renamed and DERIVED.** `FamilyDefaults` is now
  `GenerationDefaults`, and the schema comes from the handler's context
  annotation — `def generate(self, ctx: RequestContext[SdxlDefaults], p)` —
  never from a decorator/Slot kwarg. `Slot(default_config=...)` is DELETED:
  the catalog owns recipe VALUES (th#1116 stamps one resolved recipe per
  slot); code owns the schema only. Read the resolved recipe as
  `ctx.defaults` (typed as your annotation). With no catalog metadata
  (hub-less runs) the neutral schema defaults apply — identical to the
  hub's neutral stamp. Code-side lineage recipe constants are deleted with
  it.
- **Component sharing is AUTOMATIC by content address.**
  `Slot(share_components=...)` is DELETED — every component of a
  Slot-declared pipeline slot is a content-keyed share candidate; equal
  bytes alias, unequal bytes stay exclusive. An endpoint can no longer
  forget to share.
- **Output contract, stated:** RETURN the output object (`ImageOutput`
  etc.); the SDK owns encode + upload (gw#516 handoff). A handler that
  hand-uploads inside its body opts out of the encode/upload tail overlap.
  Async handlers deliberately did NOT ride this cut — `async def` is
  already a per-handler opt-in (dual dispatch exists today), and the
  blocking `ctx.save_*` methods will gain `async` twins in a later
  non-breaking pass (pgw#652 Phase 1). No new sync ctx I/O was added.

## 0.58.1 (2026-07-25)

- **gw#640 (secondary half): a WAITING lifecycle intent always carries a
  blocker, a retry time, or a deadline.** The hub's shadow validator requires
  one of the three. Compat-synthesized intents (`ensure_local_intent` ->
  `compat-materialize-*` / `compat-setup-*`) are minted OUTSIDE a
  `DesiredStateCommand`, so they never received the command's `first_action`
  deadline and carried none of the three. That malformed state is what the hub
  rejected on every reconnect for twelve consecutive th#1085 cold-boot runs —
  and it used to reject it by ENDING THE RPC, which is the real gw#640 root
  cause and is fixed hub-side (tensorhub `257743b4`: shadow validation may
  never terminate a live worker's RPC). Guaranteed now at
  `IntentRegistry.transition`, the single choke point, rather than at each call
  site; an explicitly supplied blocker/retry/deadline is never overwritten.
  Without this the hub correctly DROPS the snapshot, so the shadow projection
  never converges even though the worker and its jobs are healthy.

## 0.58.0 (2026-07-24)

- **pgw#648: VRAM is accounted PER DEVICE-GROUP, never summed across cards.**
  `models/residency.py::_default_free_vram_bytes` used to sum free VRAM over
  every CUDA device, so a 3x24GB pod reported 72GB free and would admit a 30GB
  model that fits on no single card. `DeviceGroup(devices=(...))` now owns the
  probe and one `Residency` accounts for exactly one group's pool. A group that
  SPANS devices (a future tensor-parallel mesh) sums its own members only —
  that is one placement unit by definition. No endpoint-visible change.
- **pgw#641 Stage 2: admission LEASES replace the whole-job pin.**
  `Residency.admit(sizes) -> Lease` is taken before a job starts and held for
  its whole lifetime. From admission on, no eviction/demotion/`release_to_disk`
  path may victim a leased ref — INCLUDING refs whose entries do not exist yet,
  which is the structural gap the old `executing()` pin no-op'd on (a freshly
  created entry was demotable between its `track_vram` and the execution-time
  pin, gw#409). Bytes for not-yet-loaded refs are RESERVED, so two concurrent
  admissions can no longer book the same free VRAM and OOM each other mid-load;
  two leases on one ref claim the MAX, not the sum (one future load serves
  both), and a claim is consumed the moment `track_vram` books real bytes.
  `make_room(for_refs=)` excludes the caller's own reservation — it IS the
  demand being satisfied. `Residency.fits(sizes)` is a new read-only "could
  this worker serve this now?" query. Admission never refuses here: the
  adaptive-fit ladder still owns genuine overcommit.
- **pgw#647: handlers on one live instance are SINGLE-FLIGHT by default.**
  One live instance == one binding set == one materialized graph with MUTABLE
  buffers (the resident LoRA branch, adapter enable state), so two concurrent
  requests on it corrupt each other. One-job-per-GPU masked this;
  `BoundedSemaphore(gpu_count)` on a multi-GPU pod does not. A per-instance run
  gate now serializes adapter attach + handler + detach; jobs on DIFFERENT
  instances (different checkpoint picks) still run concurrently, so this costs
  nothing under multi-residency. **New endpoint surface:**
  `@endpoint(reentrant=True)` is the explicit opt-out for classes whose
  handlers mutate no instance state; it is CLASS-ONLY (a loose function holds
  no instance state to serialize) and raises if declared on a function.
  Endpoints that hand-rolled their own handler lock can delete it.
- **pgw#652: admission reserves ACTIVATION VRAM, learned from measured peaks.**
  Weights are not the whole cost of admitting a request — a concurrent 1024^2
  diffusion request also holds GBs of latents and attention workspace, so
  interleaving would OOM the moment it started working. Leases now carry an
  activation claim that SUMS across concurrent leases (each request allocates
  its own) where weight claims MAX per ref. It is LEARNED, never declared: the
  executor already measured `peak_vram_bytes` and threw the useful part away,
  so `record_activation` now takes peak minus what was already allocated when
  the handler took the GPU and keeps a decaying high-water — up instantly in
  full (under-reserving is what OOMs), bleeding 12.5% per subsequent request so
  one 4096^2 outlier does not permanently tax residency depth. Keyed by
  FUNCTION, not by checkpoint pick, so a never-seen checkpoint inherits a real
  measurement instead of reserving nothing on its first request. An unmeasured
  function claims 0, so this is inert on CPU workers and at first boot. No
  endpoint-visible change and no knob — residency depth vs concurrency headroom
  is one runtime decision.
- **th#1130: WebP is THE image-encoding default, on one shared encode core.**
  Paul's ruling: "the default image-encoding should be webp, always, with png
  or jpg as optional alternatives." Both encode surfaces had independently
  reimplemented the encode — and had already drifted to different default
  qualities (`io.write_image` q90 vs `ctx.save_image` q95). They now route
  through one core, `io.encode_image(image, *, format, quality, lossless,
  method, **encode_kwargs) -> (bytes, extension)`, so the defaults cannot
  drift apart again: `webp` / q95 everywhere (`DEFAULT_IMAGE_FORMAT`,
  `DEFAULT_IMAGE_QUALITY`).
  Two real gaps closed on the `write_image` side: `format="jpg"` used to
  reach PIL as `"JPG"` and raise (PIL's name is `JPEG`), and a transparent
  image encoded to JPEG raised instead of converting — both paths now
  normalize `jpg`/`jpeg` and convert `RGBA`/`LA`/`P` to `RGB` first. An
  unrecognized format raises `ValidationError` naming the supported set
  instead of surfacing a PIL traceback (`ctx.save_image` previously raised a
  bare `ValueError`). `write_image` also appends the format's extension when
  the given ref has none — a webp payload stored as `image` defeats
  downstream mime inference.
  `ctx.save_image` gained `**encode_kwargs` passthrough (e.g. `method=6`).
  Prefer `ctx.save_image`: it is what the whole endpoint fleet calls and it
  returns a typed `ImageAsset`. `io.write_image` survives only for its
  terminal decode->finalize handoff (th#1107), which `ctx.save_image`
  deliberately does not perform — endpoints call that one mid-pipeline and in
  N-image loops, where a terminal GPU-slot release would be wrong.

- **pgw#636: hot-GPU mandate — pack VRAM with checkpoints.**
  `Resources.vram_gb` is now purely a placement minimum, never a per-load
  reservation: `_make_room_for` estimates a never-seen pick from its wire
  snapshot's real byte total (prior measured `vram_hint` still wins) and
  falls back to the declaration only when NO byte facts exist. A 24 GB card
  therefore packs several ~5 GB checkpoint picks hot instead of evicting the
  resident pipeline on every hop. New `Slot(share_components=(...))` opts a
  pipeline slot into CROSS-PICK content-keyed component sharing (gw#479
  machinery generalized): declared components become independent residency
  entries — equal bytes alias across picks, unequal bytes stay exclusive —
  and the per-pick denoiser lane LRU-swaps on its own. Residency shared
  entries move from refcount-holds to `holders` semantics: referenced
  components are demotable while idle (owners re-promote before executing —
  jobs pin + promote their record's shared entries), never evictable while
  referenced, and 2+-holder shared entries sort LAST in LRU victim order;
  record vacate no longer drains unreferenced shared entries eagerly.
- **pgw#638 serve-while-downloading: attempted, REVERTED, no behavior change
  in this release.** Letting hub-staged DISK materializations run concurrently
  with tenant jobs (dropping their tenant-idle gate and the run_job
  cancellation) is the obvious fix for the measured starvation — staged 4.7 GB
  downloads sat at 0%% for 4+ minutes on busy workers while the same blobs
  landed in 5-15s on idle ones — and it is WRONG. The reconcile pass's first
  act for a ref is to fence a stale resident identity, and a tenant job may
  legitimately be re-materializing OLDER bytes for that ref at the same time;
  unserialized, the fence loses the race and the worker keeps reporting the
  stale identity until the next HelloAck (the th#1066 hub/worker drift class).
  Measured non-convergence over 10s, caught by
  `test_mutable_tag_move_fences_events_by_digest_and_generation`. Re-verifying
  after the job does not fix it either, because `handle_run_job` returns before
  the job's load completes. The correct fix is to make transfers first-class
  tasks with their own lifecycle instead of steps inside a serialized
  reconcile pass (pgw#641 Stage 4).
- **pgw#637: dynamo's in-memory code cache is a legitimate serving
  surface.** Cell keys are checkpoint-free, so the 2nd checkpoint of an
  already-proven family serves its warmup from dynamo's in-memory compiled
  code with zero FX/AOT counter movement — the finalize proof now credits
  that signature (calls>0, hits=0, misses=0) when the exact cell was already
  proven in this process AND dynamo confirms live compiled code for that
  object's compile targets, instead of deterministically bricking the
  compiled lane (`compile_cell_failed`) on every multi-checkpoint session.
  Both conditions are load-bearing: the registry alone would let a sibling
  object's cache hit certify this object's silence (gw#603/gw#611 forbid
  exactly that), the dynamo probe alone would credit a cell never proven
  anywhere. The disproof cleanup no longer fires the global
  `torch._dynamo.reset()` while a healthy sibling pipeline is still armed.
- **pgw#639: SIGUSR2 dumps every thread's stack.** The worker's asyncio loop
  owns the heartbeat while model work runs on threads, so a wedged worker
  looks perfectly healthy from the hub. `kill -USR2 <pid>` now prints all
  thread stacks to stderr (`faulthandler`, allocation-free, always armed) —
  the pod-side forensic surface that was missing during the 2026-07-24
  incident. Getting a shell into the pod to send it is still open.

## 0.56.3 (2026-07-24)

- **gw#640: a message-handler exception is no longer indistinguishable from a
  dropped socket — and this is the bug the last two instrument releases were
  hunting.** `transport._recv_loop` awaits the handlers inline, so a raise while
  handling (say) a `RunJob` propagated into `Transport.run()`'s catch-all and was
  logged as `connection to <addr> failed`. The worker then reconnected with
  backoff, forever; the hub, whose only death signal is a closed stream, reported
  `young worker death lifetime=1s` and `requeue_exhausted: workers kept dying
  mid-job`. The process was alive the entire time — which is why 0.56.1's
  `worker_fatal`/`UnexpectedWorkerExit` and 0.56.2's post-mortem supervisor all
  stayed silent across ten live th#1085 cold-boot runs: nothing ever escaped to
  them and nothing ever exited. Proof from run 10: one unchanging
  `worker_session_id`, `state_seq` climbing 14 -> 146, and six byte-identical
  (process-cached) boot-canary payloads.
  `HandlerError` now wraps handler raises with the offending message kind,
  `run()` catches it as its own class, and `_report_handler_failure` dials the
  existing `worker_fatal` carrier with `phase=message_handler:<kind>` plus the
  traceback — a durable `pod_events` row on every hub pin already deployed, no
  proto change and no hub redeploy. Deduped per (message kind, exception class)
  so the reconnect loop cannot re-dial the same fault every cycle. Reconnect
  behaviour is deliberately unchanged: this release unmasks the fault, it does
  not change liveness policy.

## 0.56.2 (2026-07-24)

- **gw#640: a post-mortem supervisor names the death that happens BELOW
  Python.** 0.56.1 instruments every death Python can observe; th#1085 run 9
  produced six process restarts and ZERO `worker_fatal` rows, which leaves
  exactly one class — a signal (cgroup OOM `SIGKILL`, `SIGSEGV` in a C
  extension, an external kill). No `except` catches that and no in-process
  reporter can dial out after it, so the reporter is now the NEXT process.
  `python -m gen_worker.entrypoint` forks a supervisor before its heavy
  imports: the parent stays a bare interpreter (the OOM killer picks the fat
  child, not the reporter), forwards signals, and on an abnormal exit reports
  `WIFSIGNALED`/`WTERMSIG`/`WCOREDUMP` plus `memory.max` vs `memory.current`
  vs `memory.peak`, `cpu.max`, and the `memory.events` `oom_kill` counter
  delta through the same `worker_fatal` carrier — a durable `pod_events` row
  on any already-deployed hub, queryable with `class='hardware_unsuitable'
  AND reason='worker_fatal'`. A boot record on the container filesystem covers
  the case where the whole cgroup goes (`memory.oom.group`) or the container
  is restarted: the next boot finds the unfinished record and reports it.
  Exit status is propagated unchanged, so container-restart semantics are
  identical; `GEN_WORKER_SUPERVISOR=0` opts out.
- **gw#640: the host canary reports the cores this container actually owns.**
  It shipped `os.cpu_count()` — the HOST's count, 32 on a pod that owns 4 —
  next to a cgroup-derived `ram_total_gb` of 14.9 GB, a "32 vCPUs / 14.9 GB"
  report that misdirected the th#1085 investigation. `vcpus` is now
  `min(host cores, sched_getaffinity, cpu.max quota)`, and the multi-core
  throughput probe runs that many threads instead of oversubscribing a quota.

## 0.56.1 (2026-07-24)

- **gw#640/th#1077: a worker fatal now reaches the HUB, not just pod stdout.**
  `entrypoint._log_worker_fatal` wrote the exception class, message and
  traceback to stdout only; RunPod exposes no container-logs API, so every
  cloud-only worker death was unobservable by construction (six live th#1085
  runs burned on a crash whose traceback existed and could not be read). The
  fatal is now also dialed to the orchestrator over a fresh Connect stream,
  reusing the `HardwareUnsuitable` carrier with `reason_class="worker_fatal"`
  — the hub already persists that as a durable `pod_events` row, so this
  needs NO proto change and NO hub redeploy and works against every hub pin
  already deployed. Additionally, the run loop ending WITHOUT a hub Drain or
  a shutdown signal is now itself a reported fatal (`UnexpectedWorkerExit`)
  instead of a clean, silent `exit 0` — that silent exit was exactly the
  gw#640 signature the hub could only see as a young-worker death.

## 0.56.0 (2026-07-24)

- **th#1085 Slice 5: exact mutable-config convergence.** Protocol-v5 desired
  state now carries a full MessagePack parameter snapshot and the config
  classes changed at one target generation. The worker separately proves
  receipt, atomic parameter persistence, binding convergence, and boot
  generation; boot-only changes report `BOOT_STALE`, and exact function
  capabilities remain non-ready until the release/config/binding tuple
  converges. Boot generation now comes only from Tensorhub's pod-launch
  `WORKER_CONFIG_GENERATION` stamp; the first desired-state receipt can no
  longer falsely certify an older environment.

## 0.55.0 (2026-07-24)

- **gw#496: make checkpoint metadata honest.** `save_checkpoint` and
  `open_checkpoint_stream` no longer accept `produced_by_kind`,
  `target_dtype`, `flavor`, or `attributes`, which the repo-commit route
  never persisted. Step, epoch, and checkpoint kind remain on their live
  provenance/event paths. The private stream's write-only uploader metadata
  and unused elapsed-time property are removed. The zero-consumer
  `Dataset.as_hf_dataset` and `Dataset.is_eval_set` conveniences are removed;
  callers use `iter_examples`/`as_dataloader` and `Dataset.kind`.

## 0.54.0 (2026-07-24)

- **th#1087 stages B+D: declared mutable config + worker reconcile.**
  `@endpoint(config=[ConfigParam(name, type, default, choices/ge/le/regex)])`
  declares deployer-settable knobs; `@endpoint(env=[...])` declares the env
  names the code reads (D2). Both emit into the discovery manifest
  (`config_params` / `env`) so the hub 422s config writes outside the
  declared surface. Handlers read effective values via `ctx.config`
  (declared defaults <- worker config store <- RunJob-stamped values).
  New `gen_worker.runtime_config`: on a config-generation push the worker
  updates memory AND atomically rewrites a local snapshot file
  (`GEN_WORKER_CONFIG_SNAPSHOT_PATH`, default
  `/app/.tensorhub/runtime_config.msgpack`); per-invoke subprocesses read it
  via `read_snapshot()` (`run_process` forwards the path into explicit
  child envs). `run_process(ctx=...)` receives an immutable invocation
  snapshot, so an older dispatched job cannot read a newer generation that
  arrived before its child started. Bindings keep riding the existing
  HelloAck desired-residency reconcile (gw#614/623, pod-churn-free);
  env-class changes stay boot-only (the hub drain-rolls). Wire adapter
  follows the A+C tracker contract —
  DesiredResidency `release_id`/`config_generation` observed on HelloAck,
  RunJob `config_generation`+`config_params` (msgpack) stamped per dispatch,
  and StateDelta `observed_config_generation` echo.

## 0.53.0 (2026-07-24)

- **th#1084 (pgw side): input-caused refusals raise ValidationError ->
  INVALID.** RepoRefusal + SourceIncludeError are ValidationErrors; HF
  repo-access errors wrap as typed input refusals at the convert ingest
  boundary with the class name preserved; an all-input-rejection "no
  publishable flavor" aggregate is INVALID. Bad user repos fail only their
  request — never the release.

## 0.52.2 (2026-07-24)

- **gw#627 live fix 2: normalization rejects legacy attn-processor
  converter output.** diffusers' non-diffusers converter turns kohya-flat
  sdxl attention keys into legacy `…attn1.processor.to_q_lora.down.weight`
  names that match no real module, failing the whole curated adapter typed.
  `normalize_adapter_state_dict` now falls back to the raw keys (which
  resolve directly against module paths) whenever the converted dict
  carries `.processor.` names.

## 0.52.1 (2026-07-24)

- **gw#627 live fix: `enable_compiled` skips the branch lane on slot
  objects with no compile target.** The arming path runs for every
  worker-loaded setup slot; a bare component slot (sdxl's standalone
  AutoencoderKL vae) resolves none of `cfg.targets`, and `apply_lora_lane`'s
  no-denoiser error broke the whole model load (release-broken streak on
  the first gw#627 sdxl deploy). The loud misconfig error remains for real
  compile targets.

## 0.52.0 (2026-07-24)

- **pgw#628: residency reporting v2 — content-addressed idempotent
  observations (worker half of th#1070).** Every applied HelloAck opens a
  republish epoch: the reconcile pass re-announces verified cached
  identities (ON_DISK/IN_RAM/IN_VRAM with exact ref+digest) once per epoch
  even when unchanged — a re-received plan (hub redrive, overdue resend,
  reconnect) is the hub asking for a resync, and success observations are
  now safe to emit late, twice, or across plan revisions (the v2 hub
  accepts them by digest, generations survive only for cancel/evict/
  failure attribution). Job-path cache hits within one epoch stay deduped
  (no event spam). gw#614's no-cancel-on-same-set reconcile behavior is
  unchanged — under the v2 hub it is simply correct instead of a trap.
  Version floor note: endpoint images need no forced rebuild — v2 hubs
  accept 0.44–0.51 success reports fine; images pick up >=0.52.0 on their
  next routine rebuild for the lost-observation resync hardening.

## 0.51.0 (2026-07-24)

- **gw#627: Conv2d additive-branch support in the runtime LoRA overlay.**
  The curated sdxl distill adapters (Lightning/DMD2) carry 49 conv LoRA
  pairs each; the gw#547/558 branch was Linear-only, so on the w8a8 lane
  (`_Fp8ScaledLinear`, no peft fallback) the only route for them was raw
  peft injection — which rejects the quantized module class fatally (8/8
  live sdxl generate-turbo failures on L4, th#1037 addendum). Plain
  ``nn.Conv2d`` modules in the denoiser are now branch-capable: canonical
  zeroed conv branches ride the same rank bucket (A [bucket, in, kh, kw]
  at the base conv's stride/padding, B [out, bucket, 1, 1]), swap is the
  same staged buffer copy, and the eager instance-forward wrap adds
  ``conv1x1(conv(x, A), B)``. Convs are never quantized, so every lane
  takes the wrap path; graph stability and cell lane naming
  (``w8a8-lora<bucket>``) are unchanged.

## 0.50.3 (2026-07-24)

- **th#1063: loud boot log when a datacenter pod has no CAS fill source.**

## 0.50.2 (2026-07-23)

- **th#1055: desired-hot warm works on slot-only endpoints; failures are
  loud.** `ensure_desired_instance` demanded the instance's binding set
  equal `spec.models`, but deploy-bound Slots (ie#524/th#980) carry no code
  default so `spec.models` is empty on every fleet endpoint — every hub hot
  intent (gw#587 self-mint prewarm, th#912 slot-default seeding, #567
  compile-cell reload) was refused with a ValidationError swallowed as one
  pod-local warning: no warmup, no self-mint, w8a8 fence never opened
  (qwen/sdxl/ltx serving deadlocks), precompiled cells never armed.
  Validation now accepts exactly the declared slots (code defaults may fill
  their own), declared-space bindings remap through the HelloAck precision
  picks (th#697 contract), and every desired-instance failure — including
  pre-setup refusals — emits MODEL_STATE_FAILED for the instance refs so a
  stalled warm is fleet-visible.

## 0.50.1 (2026-07-23)

- **pgw#626 / th#1059 twin: mandatory-lane admission follows the hub-resolved
  EXECUTION lane, not the flavor token.** The `#fp8-w8a8` flavor names the
  STORAGE format; SDXL's mixed variant serves the w8a16 upcast lane (plain
  graphs). `_validate_required_compile` (and every other mandatory-lane
  derivation) refused hub dispatches for the mixed lane with
  `required_compile_missing` — the worker half of the 2026-07-23 sdxl P0
  (Paul's live jobs failed at 21:54Z). Mandatory-ness now derives from the
  HelloAck resolution lane when known (w8a8/w4a4 activations stay
  fail-closed; plain activations admit and JIT-warm like bf16); the flavor
  token remains the fallback without lane evidence, and conflicting
  evidence fails closed.

## 0.48.2 (2026-07-23)

- **th#1043: the forced group-fit fp8 downgrade reports structurally.**
  A joint shared-lane fit that forces fp8 storage now stamps the same
  rung outcome as the adaptive ladder (`FnDegraded` wanted=bf16
  ran=fp8_storage) instead of serving a silent precision downgrade the
  hub believes is native bf16.

## 0.48.1 (2026-07-23)

- **th#1043 second layer: a forced group-fit fp8 survives the gw#534
  bf16-resident upcast.** Found live on the first 0.48.0 pod: the joint
  group decision forced fp8, but `load_from_pretrained`'s single-lane
  resident-upcast check saw the FIRST lane fitting current free VRAM and
  silently upgraded it back to bf16 residency — re-starving the sibling
  lane into the refused offload placement. `force_storage_dtype` now
  disables the local upgrade (`allow_bf16_resident_upgrade=False`): the
  headroom belongs to the group, not the first lane to load.

## 0.48.0 (2026-07-23)

- **th#1043: joint precision fit for shared-component multi-lane loads.**
  A `gw#479` shared-component multi-lane record (e.g. qwen-image's t2i/edit
  lanes: a shared text encoder + VAE, an exclusive transformer per lane)
  decided each lane's resident precision reactively, one lane at a time,
  against whatever free VRAM happened to be measured at that moment. The
  first lane to load could consume all headroom at native precision,
  starving a sibling shared lane into an offload placement the shared-
  component invariant then refused outright (`RetryableError:
  shared-component lanes require resident placement`). Precision for the
  whole shared-component group is now decided jointly, against its
  combined footprint (shared components counted once), before any lane
  loads — every lane in a starved group forces fp8 storage together
  instead of one lane greedily grabbing headroom another lane needs.

## 0.47.0 (2026-07-23)

- **th#1031: `cell_selection_bug` recovers via self-mint instead of
  retry-blocking every request.** A self-requested compile cell whose graph
  signature drifts from this runtime's own (`cell_key` has no graph-shape
  axis, so structurally different graphs can collide on one key) used to
  raise `CellSelectionBugError` straight out of `fleet_cells.enable_compiled`
  — fatal on a mandatory w8a8/w4a4 lane, so setup failed and retried from
  scratch against the identical stale cell forever, paying a full
  `self_mint_compile` cycle on every request. It now falls through to
  self-mint (the ordinary MISS recovery) while still reporting the th#883
  invariant loudly (unchanged `cell_selection_bug` ModelEvent/pod_event);
  a genuine mint impossibility on a mandatory lane still fails closed.

## 0.46.0 (2026-07-22)

- **th#1017: inference regimes — checkpoints whose weights demand a specific
  inference configuration.** New per-method `regimes=` on `@endpoint`
  ("standard" | "v_prediction" | "distilled"; per-method dict on class
  handlers mirroring `warmup=`, bare tuple on the function form; absent =
  ("standard",)), exported per-function in the discovery manifest (key
  omitted for the default). `ResolvedSlot` gains `.regime` (from the hub
  resolve response's `inference_regime`; "standard" on older hubs) so
  handlers can branch — e.g. a dual-mode turbo lane skips its distillation
  LoRA for an already-distilled (fused DMD/Lightning/LCM) checkpoint. The
  executor enforces a `RegimeMismatchError` backstop (hub gates enforce
  upstream at deploy + request time; the wire carries no real regime yet,
  so every dispatch resolves "standard" — never declare a tuple excluding
  it). `SdxlScheduler` vocab gains "lcm" and "euler_trailing"; the
  converter stamps regime-correct scheduler config into produced diffusers
  snapshots (v_prediction -> prediction_type + rescale_betas_zero_snr,
  distilled -> trailing timestep spacing) via an `inference_regime` hint on
  clone/convert.

## 0.45.0 (2026-07-22)

- **pgw#622: eager-while-compiling with hot-swap — a novel request shape
  serves immediately.** A compiled target's first call at an unseen input
  signature no longer stalls 30-60s behind Dynamo+Inductor: the consumer
  guard serves the request (and followups at that signature) through the
  EAGER original, one background thread warms the compiled callable with a
  zero-filled dummy batch of the same signature (separate CUDA stream,
  thread nice +10), and a successful warm atomically hot-swaps the
  signature onto the compiled path. Each completed warm repacks the live
  inductor/triton cache root and republishes the cell under the same key
  (mode=replace) so no other worker ever compiles that (shape, GPU, lane)
  again. Sequential compile-then-serve is preserved for: the boot
  warmup-proof window, mandatory quantized lanes (w8a8/w4a4 — eager is not
  a production lane there), tight VRAM headroom (degrade, never OOM),
  regional mode, and signature-vocabulary explosions (per-request scalar
  leaking into signatures disables concurrent routing loudly).

## 0.44.0 (2026-07-21)

- **pgw#617: hierarchical slot bindings (th#980 companion).**
  `RunJob.ModelBinding` gains `components` (field 5, component name ->
  canonical tensorhub ref). The worker loads the base composition and
  substitutes each named component from its OWN materialized snapshot via
  the gw#479 `components=` from_pretrained injection (load-then-substitute).
  The composition (base + sorted component refs) is the instance/residency
  identity — a component-only rebind derives a new instance and reconcile
  reloads it; flat bindings (empty map) are byte-identical to 0.43.x.
  Unknown component names, non-CAS override refs, and overrides on
  self-loading (str/Path) slots refuse typed (`ComponentSubstitutionError`)
  at setup, never mid-denoise. Override refs join job pins, held refs/
  digests, and compile-cell binding facts.
- **pgw#617: `selected_by=` slots may omit `default_checkpoint`.** Deploy-
  time bindings (th#980) seed the hub mapping now, so the registry's
  author-time requirement is dropped (mirror of tensorhub's relaxed
  registration rule). Unblocks the ie#524 de-hardcode sweep of
  request-branching endpoints (wan-2.2, sdxl slot-model, z-image).

## 0.43.1 (2026-07-21)

- **gw#608 CLOSED: self-mint arm is transactional over the process-global
  cache env — cells are cross-host portable end to end.** The store-served
  8/8-miss root cause: a no-target sibling slot (or a delivered-cell-seeded
  process) could open a fleet self-mint capture and repoint
  TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR away from the seeded cache
  before the real warmup traced. Now: no-target siblings decline BEFORE
  any process-global env mutation, `begin_fleet_mint` restores the prior
  env on arm failure, and a delivered-cell-seeded process never opens a
  capture (mandatory lanes keep the typed refusal). Four
  revert-turns-red tests. Live-verified: first end-to-end store-served
  LTX boot (release 587…970) — consumer warmup served from the delivered
  cell, ~0s compile, no re-publish.
- **gw#608: FX-cache failure forensics.** A store-served proof failure now
  carries `fx_cache_failure_report` in the CompiledLaneUnavailableError
  detail: hit/miss/bypass counts, compile_seconds, per-object proof
  counts, a component-level FxGraphHashDetails key diff against the
  delivered cell, and same-key re-save/load probes — clamped under the
  2000-char activity error cap. Failure-path only; no serving overhead.

## 0.43.0 (2026-07-21)

- **gw#585: tensorhub v4 private-input manifests — gRPC protocol v3 -> v4
  HARD CUT (th#886).** The hub no longer rewrites canonical payload bytes
  with presigned URLs. `RunJob.input_assets` (field 15) carries the ordered
  credential-free manifest; the worker resolves fresh transport URLs itself
  with one strict `POST /worker/input-assets/resolve` under its
  attempt-scoped capability, verifies exact size/BLAKE3/MIME/kind, preserves
  opaque `Asset.ref`, sets only `local_path`, and cleans attempt-owned temp
  files on every terminal path. Endpoint build now rejects Asset-bearing
  `set`/`frozenset` and non-string-keyed mappings (unordered containers have
  no manifest order); base `Asset`/`MediaAsset` discover as `kind=media`.
  A v0.43 worker cannot serve a v3 hub and vice versa — deploy in lockstep
  with the tensorhub v4 release.
- **`GEN_WORKER_INTERNAL_OBJECT_HOSTS`**: exact-host allowlist that exempts
  resolver-minted private-input URLs from the private-IP SSRF gate for
  deployments whose object store lives on an internal network. Caller public
  transports always face the full SSRF policy.
- marco-polo example: `marco-polo-attach` private-input echo probe (e2e).

## 0.42.0 (2026-07-21)

- **gw#615: disk telemetry can no longer freeze the event loop (0.40.7
  post-seal_publish hang).** `_state_delta()` now reads only ModelStore's
  cached `disk_usage_report()`; the actual statvfs/ref-index measurement
  runs as a fire-and-forget `asyncio.to_thread` refresh gated to the
  report TTL. A stalled provider volume mount leaves telemetry stale
  instead of blocking StateDeltas, the th#965 heartbeat, and serving —
  the 0.40.7 LTX boots that sealed+published then never served.
- **th#767: `gen_worker.families.wan` — WanDefaults registered under
  `wan22`** (wan-2.2 slot migration surface for inference-endpoints).
- **gw#614: synthesized media-modality warmup coverage — multi-lane family
  cells mint complete.** gw#612's publish gate left any endpoint whose
  input-routed sibling lane needs media (qwen edit: an input image) unable
  to ever publish its family cell — the declared/synthesized warmup fills
  only required payload fields, the edit lane records calls=0, publish is
  withheld, and every second boot re-mints (~24 min). The synthesized
  warmup now runs a coverage pass: when a compile-target object is still
  unexercised after the planned jobs, media VARIANTS of the same base
  payloads (base = declared warmup payload when present, else the
  synthesized default; exactly ONE optional image/audio field filled with
  a generated asset, nothing else drifts) exercise the remaining lanes.
  Driven by payload schema + compile-object coverage, no endpoint-name
  switch; applies to mint (union cell publishes) and adopt (the sibling
  lane proves against the cell instead of arming unproven). New
  `warmup.media_variants`; real-inductor fresh-subprocess proof that a
  two-lane union cell serves BOTH lanes as FX hits
  (tests/test_cell_portability_gw611.py).
- **gw#614: on_hello_ack model-set-diff cancel (th#961 defense in
  depth).** Every HelloAck used to cancel + restart the residency-
  reconcile task, killing any in-flight self_mint_compile at phase=load
  (th#961: 4,602 cancels in 19 min). The worker now diffs the ack's
  semantic model set (resolutions + disk_refs + snapshots + hot) against
  the running reconcile's target: identical set → keep the task, apply
  non-model deltas only; changed set → cancel as before.
- **gw#612: multi-lane self-mint — publish gated on full capture coverage;
  post-proof activity phase.** ie#501 run 26's "post-seal_publish hang" is
  DISPROVEN on evidence: the qwen 2-lane minting worker completed setup,
  advertised readiness (`newly_available=[generate]` hub-side 20:00:50),
  and idled; the 2.5h wedge was hub-side — the singular compile fence saw
  the record's TWO same-identity self-attested targets (t2i + edit riding
  one family cell) as ambiguous and starved dispatch forever (tensorhub
  lockstep fix: same-identity siblings collapse to one deterministic
  pick). Worker-side real defect fixed here: the shared capture packs
  only the graphs the warmup compiled, so a mandatory sibling lane the
  warmup never exercised (qwen edit — no warmup modality) left the
  published "family cell" lane-1-only, bricking every adopting boot at
  the gw#607 per-object proof (gw#611 qwen variant, hits=1/misses=1 →
  compile_cell_failed → release broken). `finalize_self_mint` now only
  packs; the executor decides after the whole proof pass:
  `publish_self_mint` when every capture-sharing object proved into the
  cell, `withhold_self_mint_publish` (typed, loud:
  `SELF_MINT_PUBLISH_WITHHELD`) when any sharer went unexercised — the
  boot still serves compiled locally and re-mints next boot instead of
  poisoning the store. New `finalize` activity phase covers the
  post-proof tail (sibling resolution, publish decision, bookkeeping
  through readiness) so completed activities stop reporting a stale
  `seal_publish`.
- **gw#611: adopt-proof counter blindness fixed; portability contract
  pinned.** Measured (torch 2.13): with the AOT autograd cache in BUNDLED
  mode an AOT hit serves the compiled artifact with the fxgraph counters
  fully silent — a healthy serving adopt read `cache_hits=0,
  cache_misses=0` to the warmup proof and fail-closed BRICKED the release
  (th#954 SDXL second boot). `inductor_counters` now reports
  `aot_cache_hit`/`aot_cache_miss` and the guard wrapper credits AOT-layer
  hits as serving evidence (production pins the AOT layer off per gw#608,
  so these stay 0 unless a config regression re-enables it — which now
  degrades to a proven boot instead of a bricked release). The fail-closed
  detail gains `calls=` so an orphaned/never-invoked wrapper (calls=0) is
  distinguishable from counter-blind serving (calls>0) on the wire. New
  real-codepath repro (tests/test_cell_portability_gw611.py): mint ->
  pack -> fresh-process adopt -> warmup counts >=1 FX hit (CPU inductor,
  real subprocesses), plus the bundled-AOT 0/0 mechanism pin. Hub lockstep
  (tensorhub chaos 9e7dca8e): cells that fail their own adopt proof are
  QUARANTINED (unattachable; next boot self-mints) instead of bricking
  the release via model_load_failure_streak.

## 0.41.0 (2026-07-21)

- **gw#613/th#965: universal app-level heartbeat (liveness layer 2).**
  ie#501 run 26 proved transport keepalive validates the gRPC library's
  threads, not the application: a worker answered HTTP/2 pings through
  2.5h of app-level silence, indistinguishable from a hung one (it was in
  fact healthy-idle with no idle beat, starved by a hub fence bug — the
  beat makes that diagnosis instant). The worker now declares
  `Hello.heartbeat_interval_ms=10000` and force-re-sends the full
  StateDelta every 10s from the asyncio event loop (the pgw#610
  disk-report task, promoted to the beat — never a detached thread), in
  every state including drain; the hub declares the worker dead after 6
  consecutive misses (~60s, `worker_heartbeat_lost`) and recycles it on
  the worker_disappeared enforcement path. 10s x 6 keeps a single missed
  beat at 10s of slack so a transient stall never reads as death. Disk
  stats ride every beat but are measured at most every 30s. NEW contract
  clause (§3 event-loop discipline): worker code must never block the
  event loop longer than the miss window — long synchronous work
  (torch.compile, model loads, CUDA sync) runs in executor threads, as
  the executor already does (`_to_thread_complete`: setup, warmup/
  compile, residency promote/demote, GC scans). A stuck coroutine that
  leaves the loop beating is caught hub-side by layer 3 (loading
  function with no open activity for 10min). Lockstep with tensorhub;
  contract §3 rewritten in both repos.

## 0.40.4 (2026-07-21)

- **gw#608: compiled-cell cross-pod portability.** The AOTAutogradCache key
  hashes the decomposition-table function's repr — including its process
  memory address — so AOT entries can never hit across pods, and an AOT
  miss recompiled without consulting the (byte-portable, proven-identical)
  on-disk FX-graph entries: store-served boots failed closed 8/8 on
  perfectly good cells. The worker now disables the AOT autograd cache
  symmetrically at capture/seed/apply time; the FX cache is the lookup
  surface. Hardening: `finish_fleet_mint` refuses to publish a capture with
  zero (or fewer-than-proven) FX-graph entries — a minting boot now proves
  its artifact, not just its execution.

## 0.40.3 (2026-07-20)

- **Release cut for gw#607/gw#587:** first wheel carrying pgw#606/th#938
  ("Slot fns never boot-setup from the image-baked code default"). Beyond
  the Slot-default fix, this closes gw#607's store-served gap: the gw#591
  boot watcher used to run `ensure_setup(spec)` with NO snapshots the
  moment weights landed, so a compile-declared fn always booted cell-blind
  and re-minted even when the hub had attached the stored cell to the hot
  DesiredInstance (live: four consecutive boots). Slot/compile fns now set
  up ONLY on hub delivery (Hot DesiredInstance / RunJob), which carries the
  attached cell — store-served boots engage.

## 0.40.1 (2026-07-20)

- **gw#604: a hub redelivery of the worker's OWN just-published self-mint
  cell is a no-op re-arm.** Cell identity is the key (gw#581); the store's
  snapshot digest and the self-attested tar blake3 are two transport forms
  of the same bytes, so a same-key desired cell no longer vacates a proven
  serving record (whose warm-process re-proof could never pass honestly —
  the live fail-closed re-arm loop). The worker aligns its advertised
  digest to the store's form instead. Delivered/label republish
  (vacate/rebuild) and hot-adopt convergence are unchanged. Hub half:
  tensorhub PR #512 (attach skip + th#930 prewarm demand conversion).

## 0.40.0 (2026-07-20)

- **gw#596 instructed-lane contract (th#913 pair).** A lane is the FULL
  execution descriptor `<weights>-<activation>[-<scale>]+<execution>`
  (shared vocabulary `gen_worker.models.lanes`, twin of tensorhub's
  `internal/orchestrator/precision/lane.go`). The worker now honors
  hub-resolved lanes: `HelloAck.ModelResolution.lane` records each pick's
  concrete lane; a per-request `RunJob.lane` (family `bf16|fp8|4bit` or
  full descriptor) rebinds the job's laddered models to the instructed
  lane on a derived instance key (warm workers keep both variants
  resident and cycle them via gw#551), an unserveable lane refuses TYPED
  (`lane_unavailable: <lane>` — never a silent fallback; w8a8 stays
  compiled-only), and `JobMetrics.lane` reports the CONCRETE lane that
  served every request. `gen-worker run --lane` gives cozy-local the same
  dual-form choice (bf16 = declared base, fp8 = cast lane or the
  `#fp8-w8a8` fold for the full descriptor). Absent instructions leave
  today's behavior untouched.

## 0.39.4 (2026-07-20)

- **gw#603: custom-warmup proof attributes to contract-compatible sibling
  functions; `warmup={...: None}` aliases stay fail-closed.** Live-found by
  the gw#587 proof rerun: the previous single-name attribution made any
  custom-warmup class with 2+ warmable mandatory-lane aliases (LTX
  generate/edit/extend) unbootable compiled on >=0.38.8 — delivered cells
  included. Proof is now a property of the warmed OBJECT and the graph set
  actually exercised: a proven object certifies every sibling alias sharing
  its family, lora bucket, execution-contract digest, and bindings, except
  explicitly opted-out aliases (the legacy-Turbo carve-out, revert-turns-red
  pinned). Non-custom (decorator/synthesized) warmup semantics unchanged;
  per-call guarded degradation + hit/miss telemetry remain the runtime
  backstop.

## 0.39.3 (2026-07-20)

- **gw#587 prove-produces-the-mint: the self-mint captures the executor's
  REAL warmup and publishes only after the proof passes.** 0.39.2's
  self-mint minted via the producer synthetic warm loop
  (`mint_artifact`/`_warm_call`) before the proof ran; live forensics
  showed the cell differing from the known-good forge cell in 20/490 FX
  graph keys (broadcast vs per-token modulation — the gw#586 defect class
  inside self-mint), so the proof correctly failed closed. Now a cell miss
  arms COLD into a capture dir (`compile_cache.begin_fleet_mint`, no
  synthetic call; `fleet_cells.enable_compiled` returns `PendingSelfMint`
  with the key ref known from static axes), the endpoint's own warmup
  performs the only compile the mint sees, and the proof loop finalizes
  (`finish_fleet_mint` pack -> real blake3 digest -> background publish)
  ONLY proven captures — unproven ones are abandoned, mandatory lanes
  still fail closed, and the publish-before-proof window is closed. The
  published artifact is byte-derived from the same execution the proof
  observed; no second code path re-creates serving's execution. Delivered
  (store-served) proof semantics, `cell_key`, `cell_selection_bug`, and
  the `STORE_SERVED_BOOT_COMPILED` alarm are unchanged.

## 0.39.2 (2026-07-20)

- **gw#587 serving bootstrap: self-mint boots advertise their own key and
  run the warmup proof** (pairs with tensorhub PR #488's hot-prewarm +
  self-attested dispatch fence; closes the live-proof deadlock). A minting
  worker now records its mint into `active_compile_artifacts` at BOTH
  arming sites (`fleet_cells.enable_compiled` returns `ArmOutcome` with the
  `SelfMint` identity: own key ref `_system/family-<f>#<key>` + blake3
  self-attested artifact digest; `ArmingScope` gains an executor-routed
  `enable` seam for self-loaded pipelines), so `_install_compile_targets`
  advertises the worker's own key instead of raising
  `CompiledLaneUnavailable`, and `proves_inductor` runs the SAME warmup
  proof for self-mint and store-served boots — zero-hit self-mints fail
  closed, never eager. `STORE_SERVED_BOOT_COMPILED` stays delivered-only:
  a minting boot legitimately compiles and is exempt.

## 0.39.1 (2026-07-19)

- **gw#595: qwen compiled serve — producer guidance-kwarg parity +
  per-object warmup provability (ie#501 run 19).** Two coupled defects:
  (a) `compile_cache._warm_call` warmed every declared guidance scale
  through `guidance_scale=` — on classes exposing `true_cfg_scale` (qwen)
  that is the distilled-guidance embed no-op and classic CFG rides
  `true_cfg_scale` + a non-None `negative_prompt`, so the minted "cfg 4.0"
  pass traced the SAME unconditioned graph as the 1.0 pass and the serving
  CFG lookup could never hit (the gw#586 class, call-KWARG axis). The warm
  call now uses the true-CFG convention when the signature declares it
  (also inherited by 0.39.0's shared mint brain `mint_artifact`, so fleet
  self-mints trace the real CFG graphs too).
  (b) The post-arm warmup proof required EVERY armed compile object to
  serve its own cache hit, but a merged multi-lane endpoint (qwen t2i +
  edit on one family cell) has objects the declared warmup structurally
  cannot exercise (edit needs an input image) — mandatory-W8A8 setup
  fail-closed forever. The proof now scopes per object: an EXERCISED
  object (calls>0) must hit or the cell is disproven and fails closed
  exactly as before; an unexercised object (calls=0) with a proven sibling
  stays armed unproven on mandatory lanes (logged:
  "armed unproven: no warmup modality") or unwraps to eager on optional
  lanes; zero proven objects still fails closed (the gw#586 hole stays
  shut). Hot adopt gains the distinct `no_warmup_modality` refusal so an
  unprovable target is named as such instead of `cache_miss`.

## 0.39.0 (2026-07-19)

- **gw#587: fleet worker self-mint — the serving worker compiles its own
  cell on a store miss, serves compiled immediately, and publishes it
  through the hub's attested gate (th#910).** The boot warmup IS the mint
  (right SKU/image/lane/pipeline-class/shapes by construction — the gw#586
  parity-treadmill class of producer defects becomes unrepresentable).
  Eager fallback and the fail-closed cell WAIT are retired: a mandatory
  (W8A8/W4A4) miss now proceeds to load and self-mints instead of refusing
  before load; the typed quantized refusal remains only at genuine mint
  impossibilities (no CUDA, no toolchain, mint failure), and plain lanes
  keep eager there. The `cell_selection_bug` receipt invariant and the
  post-arm warmup proof are unchanged. New: `fleet_cells` (arming policy +
  `CellPublisher`: publish-intent → the standard repo-commit flow →
  publish-complete; refusals are terminal and NEVER affect serving — the
  triggering request is always served from the local mint), shared mint
  brain `compile_cache.mint_artifact` (cozy-local `local_cells` delegates;
  its store stays local-only — cozy-local never publishes to the fleet),
  additive Hello field `WorkerResources.gen_worker_version` (the hub's
  attestation basis; absent ⇒ publish refused, harmless for old workers
  which never call the route). A hub delivery without a cell no longer
  tears down a worker's own armed, proven target (worker-owned cells).

## 0.38.7 (2026-07-19)

- **pgw#594: second reserved model-input field, `text_encoder`, for producer
  payloads (te#70 Gemma-TE video-LoRA training).** `source`/`destination`
  were reserved-name `SourceRepo`/`DestinationRepo` fields special-cased by
  literal name in the executor; any producer (training/conversion) payload
  can now also declare `text_encoder: SourceRepo | None = None` and get it
  materialized the same way as `source` — into `ctx.text_encoder_path`
  (`ctx.text_encoder` for the raw dict), fully independent of
  `ctx.source_path`. Generic mechanism: gen-worker has no ltx2/Gemma
  awareness. Absent field (every existing endpoint) is byte-for-byte
  unchanged — no extra `ensure_local` call, `ctx.text_encoder_path` stays
  `None`.

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
