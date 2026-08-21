- **pgw#1584: `ctx.boot_warmup` has a WRITER again, so the first-call tax is
  paid by the pod's boot instead of by a paying request.** The property, its
  constructor kwarg, its docstring (*"a handler MAY cheapen the run — `steps = 1
  if ctx.boot_warmup else steps`"*) and its reader in
  `output_integrity.judged` all shipped; `grep -rn "boot_warmup=True" src/`
  returned nothing, so every `if ctx.boot_warmup:` arm in the fleet was
  unreachable code and every endpoint that wrote one had written it for a flag
  that was permanently `False`. v1 had a functioning warm pass and no
  `v1_deleted.py` tombstone row was ever written for it, so by this campaign's
  own rule — *every field with a tombstone row was a decision, every one without
  is an accident* — it is restored, not retired. `ServeLoop.boot_warmup` runs
  one synthetic invocation per entrypoint through the SAME `invoke` a dispatch
  calls: real envelope decode, real residency lease, real author body.

- **pgw#1584: the payload is the entrypoint's own schema at its NEUTRAL
  DEFAULTS**, which is v1's warm-plan shape (the int32 incident's record:
  *"a single run at the schema's neutral defaults, at BOOT, before any request
  exists"*). `gen_worker.warm_payload` synthesizes it: every defaulted field
  keeps its declared default, a required `str` takes `WARMUP_TEXT`, a required
  `ImageAsset`/`AudioAsset` gets a stdlib-generated flat mid-gray PNG / silence
  WAV, and a schema that cannot synthesize honestly — a required `VideoAsset` —
  is SKIPPED with the reason, never faked. There is no `warmup=` declaration and
  no `NoWarmup`: both are tombstoned (*"warmup is not an author declaration"*),
  and an endpoint cheapens its warm run through `ctx.boot_warmup` in the body.

- **pgw#1584: the placement is the deliverable.** The pass runs after the last
  weight lands and BEFORE the state flips to ready — while the worker still
  advertises `loading_functions` and `first_request_servable` cannot be stamped
  — so that milestone stops meaning "the process is up" and starts meaning "a
  real forward has completed on this pod". th#2233's false-servable fix gets an
  actual readiness probe to hang on. `boot_phases.PHASE_WARMUP`, declared in the
  vocabulary and emitted by nothing since the hardcut, gets its producer.

- **pgw#1584: a warm-pass failure degrades LOUDLY and does not brick the pod.**
  v1 propagated one as a load failure; that turns a defect in an optimization
  into an unservable pod. Now: a `serve_degrade` event naming the function and
  the exception (pgw#760 — a hub-spawned worker exposes no stdout, so a log line
  is invisible), the boot continues, and the first real request takes the cold
  cost it would have taken anyway. Never worse than the state before this.

- **pgw#1583: `RequestContext.for_request`'s docstring promised what its body
  ruled out, in the same function.** It stated twice that the resolved
  checkpoint's OBJECTIVE was applied to the per-request view and that ambiguity
  raised *"never a silent objective-less fallback"* — while the body opened
  `objective = ""` and never read `slot=`, so the body WAS that fallback and the
  raise was unreachable. Settled toward the CODE, because there is no fact to
  apply: `ModelBinding.objective` has no reader anywhere in the SDK. The
  docstring now states the measured behaviour and names the surface that does
  honour one (`gen_worker.view.for_request(pipeline, objective=…)`), and `slot=`
  is GONE from both the serving and the trace context — it was accepted and
  discarded on each, and a kwarg that changes nothing is a third way to be
  silently wrong.
