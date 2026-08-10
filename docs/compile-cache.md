# torch.compile cache artifacts (#384)

> **pgw#1010 — "cell" means AOT cell.** The only artifact class the platform
> publishes, delivers or adopts is `kind="aot-inductor"` (an exported `.pt2`,
> `aot_mint` -> `aot_cells.discover` -> `aot_serve`). The
> `kind="torch-inductor-cache"` artifact this document was written about is
> RETIRED: `aot_cells` rejects it by name, so it had no consumer, and nothing
> mints, seals or publishes one any more. JIT/dynamo compilation survives as
> **intake** — a family with no export declaration arms
> `compile_cache.arm_jit_intake`, compiles on the pod that needs it, serves
> compiled for that pod's life, and produces no artifact. Honest cold boots are
> the contract there, not a gap. Sections below that describe capturing,
> packing or publishing an inductor cache are history; the keying, verification
> and lane material still applies to the cell that survived.

## Vocabulary: the graph digest is a fact, the envelope is a promise

A cell states two different kinds of thing about itself, and keeping them apart
is what makes its refusals readable:

- **The graph digest** — the digest of the traced graph. A MEASURED FACT about
  the program that was exported: same computation, same digest, on any pod. One
  derivation, used identically when the cell is stamped, looked up and admitted.
- **The envelope** — the DECLARED serving region: which resolutions, text
  lengths, guidance values and batch sizes the cell promises to serve. As in a
  flight envelope: a declared region of operation, with graceful fallback
  outside it. A request outside the envelope is not an error — it is served
  eager and named (`request out of declared envelope, serving eager`).
- **`input_contract`** — DERIVED, not a third fact. It is a projection of the
  packaged program's own placeholder list and container arities, read off the
  artifact rather than declared beside it. A label carried alongside an artifact
  can drift from it, and a label that can drift is not an identity (pgw#1058).

Two further contracts share the word and are neither of the above. Both are
about the WEIGHTS, not about the compiled program:

- **The tensor-layout contract** — how tensors exist ON DISK: byte packing,
  scale layout, swizzle, key-naming convention, file topology. Named by a
  registered descriptor handle `<producer>.<format>@<major>`
  (`cozy.fp8-rowwise@1`, `cozy.svdq-nvfp4-lr8@1`, …); a decoder declares the
  handles it implements with `@implements_contract`
  (`gen_worker.models.tensor_layout_contract`), and the vocabulary itself lives
  in tensorhub. It says what the bytes ARE and nothing about compilation.
  (th#1580 / th#1721; was called "the artifact contract".)
- **The tensor-binding contract** — the artifact's LINKING rule for tensors:
  bound by name at load (DYNAMIC — an opaque slot the compiler must never
  value-specialize, which is what makes a cell checkpoint-agnostic) versus a
  baked literal (STATIC — the value folds into cell identity; driven to zero).
  GB-scale derived data is neither and becomes a named CAS component. The
  classification derives from `state_dict` membership at trace time: the author
  configures the compiler by how the code is written, never out of band.
  **Its DYNAMIC half has a named enforcement point since pgw#1097**: mints
  compile under `aot_mint.CONSTANT_BINDING_CONFIGS`
  (`aot_inductor.use_runtime_constant_folding=True`, which defers the fold to
  load so inductor cannot inline a 0-dim or `<=8`-element tensor's values into
  the kernel) and `aot_package.folded_weights` refuses, per entry, any lifted
  weight the compiled artifact does not declare. `constant_folding_fenced` is a
  declared axis, so a pre-fence cell is refused before a byte moves.
  Authoring rules in `docs/endpoint-authoring.md`. (pgw#857; was
  "weight-binding".)

`tensor-` and not `weight-` in both names, deliberately — they govern scales,
buffers and computed tables, not just trained weights.

Widening the envelope moves the promise; it does not move the fact. The cell
key says exactly that since pgw#1059 (the ck1 REDEFINITION, landed pre-launch
with the disposable corpus purged): the key is `graph` x `envelope` x `sm` x
`toolchain` — the traced computation, the declared serving region, the GPU
architecture, and the compiler stack as we configure it (binaries + the
settings-declaration digest; the env seal's declaration folds in there, its
boot/pre-trace GATES unchanged). `kind`/`format` are single-valued metadata,
`family`/`lane` store metadata + discovery scoping, and the membership axiom
(pgw#1059 amendment 6: "don't key on parameters that don't require us to
recompile") is enforced by `tests/test_cell_key_pgw1059.py`.

Compile wins 15-34% warm latency on flux-class models but costs 20-46s per
(model, shape) and needs a C toolchain prod worker images don't ship. The
split:

- **Producer** — the platform's first-party compile job (training-endpoints
  `produce-inductor-cache`) runs on the target GPU SKU with a toolchain,
  compiles the declared shape set, and publishes the captured
  `TORCHINDUCTOR_CACHE_DIR` + `TRITON_CACHE_DIR` as ONE deterministic
  `.tar.gz` flavor `#inductor-<sku>-torch<maj.min>` of the family system repo
  `root/family-<family>`.
- **Consumer** — an endpoint opts in with
  `@endpoint(compile=Compile(family="flux2-klein-4b", shapes=((768,768),(1024,1024)), text_len=512))`.
  Every `compile=` endpoint MUST state `text_len` (ie#544): a positive value
  pins the token length, `0` declares "no text conditioning". Omit it and a
  prompt-length-dependent sequence dim mints a new graph per distinct prompt
  length — unbounded and un-warmable.
  At load the worker seeds a VERIFIED artifact (exact-match on family, SKU,
  torch, triton, diffusers/transformers), then arms guarded `torch.compile`
  (static by declaration: `dynamic=None` + `assume_static_by_default` + explicit marks, SDK v2) on `Compile.targets`. Plain optional lanes fall back to eager
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
`root/*` (invoke-time destination-write preflight + cap-token repo+owner
gate + `root` is a platform-reserved slug tenants cannot claim). Tenant
custom-code endpoints
get per-release private caches (same-principal rule) — not implemented yet.

Family keying: caches key on the graph digest + the declared envelope, not on
weights — one artifact serves every fine-tune of a family. Add a boot `warmup()`
that renders each shape the envelope declares (see examples/flux2-klein-image)
so requests never see the (cache-served) compile.

## Self-loading (str/Path-slot) endpoints — pgw#517

The arming described above ("At load the worker seeds a VERIFIED artifact
... then arms guarded `torch.compile`") only happens for a `setup()` slot
the worker loads itself — a slot annotated with the pipeline class (e.g.
`pipeline: StableDiffusionXLPipeline`). A `str`/`Path`-annotated slot is
**self-loading**: the endpoint constructs (and places) the pipeline inside
its own `setup()`, so the executor never sees the object and has nothing to
arm compile on. Declaring `compile=Compile(...)` on such an endpoint used to
be silently inert — the manifest's compile block still got seeded, but
nothing ever compiled. Registration (`registry.py` `_validate_compile_arms`,
at decoration time, not discovery) now raises on this combination. It is a
best-effort source scan, so it stays silent when `inspect.getsource(setup)`
fails or when the string `arm_compile` appears anywhere in the setup body —
do not treat a green build as proof that compile is armed.

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
   duration of `setup()` — no `ctx` parameter needed. With no active scope
   (an eager release that declared no `compile=`) it logs once at info and
   returns `False`; it never raises, so a self-loading `setup()` may call it
   unconditionally (ie#522). An endpoint
   with several self-loaded pipelines sharing weights (e.g. one class
   assembling `self.t2i`/`self.i2v`/`self.v2v`) calls it once per object.

## The serving-kernel lane is MEASURED at mint, not gated by SM (pgw#863, pgw#947)

The svdq path has two independent kernel choices:

| axis | armed | degraded | what it buys |
|---|---|---|---|
| `linear` | `fused` | `baseline` | throughput (W4A4 matmuls) |
| `modulation` | `packed` | `dense` | residency (W4A16 AdaLN, 22.8 → 13.3 GB on B200) |

Which one is faster is a per-card fact — a custom op is opaque to inductor,
so on sm_120 our fusion beats what inductor does with the open chain and on
sm_100 it loses to it — and it used to be two hand-maintained SM tuples, one
$12 benchmark campaign and one human edit per new card class. While they were
*one* tuple, sm_100 had to give up either the 9.5 GB or 19% of its step time,
because a single switch cannot say "baseline linears, packed modulation".

A lane is therefore the **combination**, written `<linear>+<modulation>`
(`baseline+packed`, `fused+dense`, …). It is measured on the card the cell is
minted for and recorded in the cell (`gen_worker/kernel_path.py`):

- **Mint.** `mint_child.lane_verdict_for` loads the endpoint once per
  candidate combination (the swap happens at model load, so comparing lanes
  means loading once each), runs `aot_mint.bench_step` — one forward of the
  family's dominant declared graph class, on its own declared example feed,
  under `torch.compile` — and times it with a fixed warmup/median protocol.
  The winner's pipeline is the one that gets exported. Candidates come from
  `kernel_path.candidate_axes`, which asks only capability questions
  (Blackwell block-scaled MMA for the fused linear; triton plus a numerics
  self-check for the packed modulation, which has no SM term at all). An axis
  with one buildable value contributes no candidates, so a non-Blackwell card
  has exactly one combination and pays for no benchmark.
- **The rule: fit-constrained speed maximization.** Among lanes whose
  measured peak plus a stated allowance (`+20%` for activation spikes and
  resolution variance, `+1 GiB` for fragmentation) fits the card, the
  FASTEST wins. VRAM is a constraint, not an objective; it breaks a tie only
  inside the 5% margin. A B200 therefore takes the baseline linears
  (228 ms/step) over the fused ones (350 ms/step) — the card has the room —
  while on a 24 GB card the same fit constraint excludes a lane outright.
  Ranking combinations is what makes this one rule enough for both axes: the
  packed modulation is speed-neutral, so it can only win on the VRAM
  tiebreak, and it does — which is how sm_100 arrives at `baseline+packed`
  without anyone hand-editing a tuple to say so.
- **Determinism.** The 5% margin means measurement noise cannot flip a
  recorded verdict between two mints on one card, and every number behind a
  verdict is recorded as its evidence.
- **Where it lands.** `metadata.json` inside the packed cell carries the
  DISCRETE verdict (`kernel_lane`: winner, rule, binding term, margin,
  candidates) plus a `fit` block — each measured candidate's peak, QUANTIZED
  up to 256 MiB, and the fallback order. No wall clocks, because the #699
  double-mint byte-compare requires a reproducible artifact; peak BYTES are
  admissible precisely because quantizing them makes them reproducible the
  way the 5% margin makes the winner reproducible. The timings ride the
  published checkpoint metadata as `kernel_lane_evidence`, beside
  `mint_phases`.
- **Serving re-applies the fit rule locally.** Cell keys are keyed on SM and
  the lane is deliberately NOT a key axis, so one key spans very different
  cards — a 96 GB RTX PRO 6000 and a 32 GB RTX 5090 are both sm_120. A
  recorded verdict is therefore EVIDENCE, not an instruction. The executor
  reads it off the delivered cell and pins it BEFORE `setup()` runs, but only
  after re-checking the recorded winner's peak against THIS device's honestly
  detected total: it fits, the verdict stands (`kernel_lane_verdict_adopted`);
  it does not, the fastest recorded candidate that DOES fit here is pinned
  (`kernel_lane_refit_local`); nothing fits, the smallest recorded peak is
  pinned (`kernel_lane_refit_no_fit`) — never the declared default, which
  carries the larger DENSE modulation and would be the bigger ask. A cell with
  no recorded peaks is adopted and marked `kernel_lane_fit_unverified`. Each
  axis then projects its own value out of the pin
  (`native_kernels.svdq_linear_lane` / `svdq_modulation_lane`). No cell, a
  pre-pgw#947 cell, or an unreadable envelope is the declared conservative
  default (`baseline+dense`) with a typed reason
  (`kernel_lane_verdict_absent` / `_unreadable` / `_unknown_lane` /
  `kernel_lane_no_cell`) — never a silent fall-through. An armed axis still
  has to pass its OWN numerics self-check on the box; a gap degrades that
  axis alone, same artifact, with the reason logged. The numerics checks are
  CORRECTNESS checks and say nothing about memory, which is why the fit has
  to be re-applied rather than left to them.
- **Known limitation (speed axis).** The re-fit covers MEMORY only. The
  ranking itself can also differ inside an SM class (1189 ms/step on a 5090
  vs 1063 on a PRO 6000 for the same work), and detecting that would need
  re-benchmarking, which serving must not do. Tracked on pgw#947.

`GEN_WORKER_NATIVE_KERNELS` survives only as the pgw#859 G0 rollout gate and
kill switch (unset = off, `=0` = forced off). It gates the ROLLOUT and never
picks a lane; flipping it is pgw#865's call. It is on th#1445's elimination
list and must not grow new meanings.
