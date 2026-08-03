# torch.compile cache artifacts (#384)

**The measured justification for the whole producer/consumer split:** compile
wins **15-34 % warm latency** on flux-class models, but costs **20-46 s per
(model, shape)** and needs a C toolchain prod worker images don't ship. That
one sentence is the entire reason compilation happens in a separate
first-party job and arrives at serving as an artifact.

## The two execution rulings

- **W8A8 is MANDATORY compiled execution.** A missing, mismatched, or unproven
  cell fails **retryably before any GPU/handler work**. It never dequantizes
  and never runs eager — a silently-eager w8a8 path is a wrong-numerics
  outcome, not a slow one.
- **A plain compiled call is never a failed request.** If it still needs a
  fresh compile (undeclared shape, no toolchain) it permanently unwraps to
  eager. The two rules are deliberately opposite; do not unify them.

Trust: compiled artifacts are CODE. Only platform jobs may publish to
`root/*` (invoke-time destination-write preflight + cap-token repo+owner
gate + `root` is a platform-reserved slug tenants cannot claim). Tenant
custom-code endpoints
get per-release private caches (same-principal rule) — not implemented yet.

Family keying: caches key on the traced graph + shapes, not weights — one
artifact serves every fine-tune of a family. Add a boot `warmup()` that
renders each declared shape (see examples/flux2-klein-image) so requests
never see the (cache-served) compile.

## Self-loading (str/Path-slot) endpoints — pgw#517

Compile arming only happens for a `setup()` slot
the worker loads itself — a slot annotated with the pipeline class (e.g.
`pipeline: StableDiffusionXLPipeline`). A `str`/`Path`-annotated slot is
**self-loading**: the endpoint constructs (and places) the pipeline inside
its own `setup()`, so the executor never sees the object and has nothing to
arm compile on. Declaring `compile=Compile(...)` on such an endpoint used to
be silently inert — the manifest/shape contract still got seeded, but
nothing ever compiled. Discovery now hard-errors on this combination.

Fix one of:

1. **Annotate the slot with the pipeline class** instead of `str`/`Path` —
   the worker loads it and arms compile automatically, same as any other
   endpoint.
2. **Keep the self-load and arm explicitly.** Call `gen_worker.arm_compile(pipe)`
   once per pipeline object at the end of `setup()`, after placement:

   ```python
   def setup(self, pipeline: str) -> None:
       pipe = _load_pipeline(pipeline, WanPipeline)
       pipe = _place(pipe)
       gen_worker.arm_compile(pipe)   # same cache-artifact-gated policy
       self.pipeline = pipe
   ```

   `arm_compile` reads the endpoint's own `Compile` spec, cache dir, and any
   hub-attached artifact from a scope the executor holds open for the
   duration of `setup()` — no `ctx` parameter needed, and it raises if
   called anywhere else (compile is a setup-time-only concern). An endpoint
   with several self-loaded pipelines sharing weights (e.g. one class
   assembling `self.t2i`/`self.i2v`/`self.v2v`) calls it once per object.
