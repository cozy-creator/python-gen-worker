- **pgw#1573: a LoRA on a compiled-armed module serves EAGER and says so —
  on the path that actually runs.** pgw#1571 measured the defect exactly (peft
  wraps a denoiser's SUBMODULES; an armed graph replaces the PARENT's forward
  and never enters them, so an adapter attached after arming does not execute
  and the base model is served bit-identically with no refusal and no log —
  eager `max|delta| = 2.2e-02` against armed `0.0` with 32 wrappers attached)
  and fixed it inside `aot_serve.wrap_module`. That function has **zero
  non-test callers**: the production arm is `torchcg.adopt.AdoptSession` →
  `torchcg.serve.aoti_loader`, which none of `PEFT_MARKER_ATTR`,
  `_say_adapter_ops_once`, `rearm_constants` or the `wrap_module` guard
  touches. So the P0 was open on the serving path, masked only by adoption
  itself being broken — and pgw#1573 fixed adoption, which arms it.

  `serving/adapter_guard.py` is the same guard on the live dispatcher. One
  `getattr` per call; a module carrying a live `peft_config` routes to its own
  eager forward and states the degradation ONCE, on the wire as well as the
  log (a serve pod's stdout goes nowhere, pgw#760). Both `ctx.compile` hosts —
  `EndpointHost.setup` and `ServeAdoption.sink_for` — hand out the guarded
  sink, and that wiring has its own red arm: restore the bare `session.adopt`
  and a LoRA request serves the compiled base model again.

- **pgw#1573: `lora_fold._compiled_armed` can see a v2 arm.** It asked
  `aot_serve.serves_compiled`, whose `_cozy_compile` marker no pod has carried
  since pgw#1373 — so on a real worker it answered False for every armed
  module and every compiled-aware branch behind it was dead code. It now reads
  the live dispatcher through `adapter_guard.compiled_armed` first.
  `adapter_guard.rearm_constants` is the v2 `Rebind` the fold path needs: AOTI
  folds its constants once on the first `run()` and never re-folds on a bare
  in-place weight write, and a runner exposing no bound table is a typed
  refusal rather than a silently stale weight.
