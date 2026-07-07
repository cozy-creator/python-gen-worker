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
  (dynamic=False) on `Compile.targets`. Any miss/mismatch => eager. A
  compiled call that still needs a fresh compile (undeclared shape, no
  toolchain) permanently unwraps to eager — never a failed request.

Artifact sources today: `GEN_WORKER_COMPILE_CACHE` (local tar) or
`GEN_WORKER_COMPILE_CACHE_URL` (presigned GET). Hub-side per-(SKU, torch)
snapshot attach is tensorhub #569. `GEN_WORKER_COMPILE_ALLOW_COLD=1` permits
cold compilation (compile job / dev only; requires a toolchain).

Trust: compiled artifacts are CODE. Only platform jobs may publish to
`_system/*` (invoke-time destination-write preflight + cap-token repo+owner
gate + the tenant slug grammar bars `_system`). Tenant custom-code endpoints
get per-release private caches (same-principal rule) — not implemented yet.

Family keying: caches key on the traced graph + shapes, not weights — one
artifact serves every fine-tune of a family. Add a boot `warmup()` that
renders each declared shape (see examples/flux2-klein-image) so requests
never see the (cache-served) compile.
