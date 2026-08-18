- **`gen-worker release derive` runs the real sdxl contract file, and the graph document now
  orders the DEFAULT-parameter class first (pgw#1384).** The serving hole list inherits document
  order and the miner mints in it, so the first published graph is the one an all-defaults
  request needs — the e2e#1892 run-9 finding (`aot_ingress_refused | no_entry_admits` on a
  fully-adopted pod, because the one packaged entry was `cfg=false/B=1` and a default request is
  CFG-on `B=2`) is an ORDERING defect, and the ordering is now a stated preference.

- **The derive reads the whole ratified author surface.** `DistillationAdapter` is a real type
  (a distillation SLOT is a type, not a flag; style LoRAs ride `list[Adapter]`), the synthesized
  trace adapter is built as the SLOT'S OWN class, every `Model`-annotated parameter is a SLOT
  (multi-model entrypoints — an auxiliary model with its own checkpoint loads under the same
  hollow session), and the ruled ctx-FIRST order `(ctx, payload, model(s), adapter(s))` is a
  TYPED REFUSAL at derive, naming the parameters it read.

- **The release document publishes each entrypoint's request-envelope JSON Schema, derived from
  the signature.** Parameter name IS slot name: `turbo: DistillationAdapter | None` publishes
  `adapters.turbo`, `video_ref: H3Model` publishes `models.video_ref`. The hub serves these as
  auto-generated API docs, which is exactly why a parameter rename is a visible API break.

- **A trace pass costs ONE denoise step, not the checkpoint's twenty-eight.** Every step of a
  diffusion loop runs the same shapes, so the derive budgets the author's own
  `callback_on_step_end` and treats the budget signal as a completed pass; when a marked module
  has still not been reached (it runs after the loop) it re-drives unbudgeted before calling it
  unobserved. Measured on the real sdxl contract file: 72 enumerated combinations in ~2 minutes
  instead of ~35.

- **The derive stores THE WHOLE TRACED GRAPH, not just its hash** (Paul, 2026-08-20).
  `gen-worker release derive --graph-cas <root>` serializes each discovered graph class's
  `ExportedProgram` into a tensorfs `LocalCAS` and carries the blob digest in the release
  document beside the `cg-graph-v1` hash and the ingress spec (torchcg tcg#49). *"We only ever
  need to run trace() once"* now holds literally: the runtime miner downloads the graph and runs
  inductor — it never re-traces and never executes author code at mint time. Proven by execution:
  save → CAS → `torch.export.load` → `aoti_compile_and_package` → `aoti_load_package` → ran.
  Weights-locality holds by construction — a fake-tensor `ExportedProgram` serializes parameters
  as metadata, so a 4096x4096 linear (67 MB of real weights) stores as a 7,312-byte blob.

- 🔻 **A stored graph did not load back, and the fix is the honest serialization.** `torch.export.save`
  writes a FAKE tensor's phantom storage, so the archive claimed shapes whose bytes were not
  there and `torch.export.load` died — measured on the real SDXL UNet, a `[1280, 320]` bf16
  parameter wanting 819,200 bytes over a 23,040-byte storage. The derive now demotes every fake
  tensor in the program's `state_dict`/`constants` to META before saving: shape and dtype only,
  which is exactly what a graph artifact should carry, since the miner binds the checkpoint's
  real weights before it compiles. Buffers `hollow_session` computed for real are left alone.
