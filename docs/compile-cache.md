# torch.compile cache artifacts (#384)

Compile wins 15-34% warm latency on flux-class models but costs 20-46s per
(model, shape) and needs a C toolchain prod worker images don't ship. The
split:

- **Producer** — the platform's first-party compile job (training-endpoints
  `produce-inductor-cache`) runs on the target GPU SKU with a toolchain,
  compiles the declared shape set, and publishes the captured
  `TORCHINDUCTOR_CACHE_DIR` + `TRITON_CACHE_DIR` as ONE deterministic
  `.tar.gz` flavor `#inductor-<sku>-torch<maj.min>` of the family system repo
  `_system/family-<family>`.
- **Consumer** — an endpoint opts in with
  `@endpoint(compile=Compile(family="flux2-klein-4b", shapes=((768,768),(1024,1024))))`.
  At load the worker seeds a VERIFIED artifact (exact-match on family, SKU,
  torch, triton, diffusers/transformers), then arms guarded `torch.compile`
  (dynamic=False) on `Compile.targets`. Plain optional lanes fall back to eager
  on a miss or mismatch. W8A8 is mandatory compiled execution: a missing,
  mismatched, or unproven cell fails retryably before GPU/handler work and
  never dequantizes or runs eager. A plain compiled call that still needs a
  fresh compile (undeclared shape, no toolchain) permanently unwraps to eager
  — never a failed request.

Serving artifacts are immutable per-(SKU, torch) snapshots attached by
Tensorhub. They are verified against the exact live pipeline contract before
the worker activates their cache files. Local tooling passes artifact paths
explicitly or uses `gen_worker.local_cells`; the compile producer opts into
cold compilation through an explicit library argument. There is no serving
environment fallback that can bypass scheduler attachment or W8A8 fencing.

Trust: compiled artifacts are CODE. Only platform jobs may publish to
`_system/*` (invoke-time destination-write preflight + cap-token repo+owner
gate + the tenant slug grammar bars `_system`). Tenant custom-code endpoints
get per-release private caches (same-principal rule) — not implemented yet.

Family keying: caches key on the traced graph + shapes, not weights — one
artifact serves every fine-tune of a family. Add a boot `warmup()` that
renders each declared shape (see examples/flux2-klein-image) so requests
never see the (cache-served) compile.

## Self-loading (str/Path-slot) endpoints — pgw#517

The arming described above ("At load the worker seeds a VERIFIED artifact
... then arms guarded `torch.compile`") only happens for a `setup()` slot
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
