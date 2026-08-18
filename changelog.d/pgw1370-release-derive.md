- **pgw#1370: `gen-worker release derive` — the publish-time instrumented
  derive over author code as-is.** The ship-code-as-is surface (pgw#1367,
  ratified 2026-08-19): the author writes a `Model[<ModelType>]` subclass with
  `lanes=(<tensorfs layout-contract references>)` class kwargs, marks compile
  targets imperatively in `load(ctx)` (`self.pipe.unet =
  ctx.compile(self.pipe.unet)`), and exposes free `@entrypoint` functions
  `(payload, model, ctx, *injected facts)` bound by annotation role. The
  derive runs INSIDE the release env, hollow-instantiates the CONFIG-ONLY
  checkpoint tree (torchcg `hollow_session`; fake parameters, no weights, no
  GPU), AUTO-ENUMERATES trace payloads from the entrypoints' payload schemas
  (one pass per enum value, cross-product capped at 64 with a deterministic
  prefix), and emits the byte-canonical `gen-worker.release-metadata@1`
  document: `cg-graph-v1` hashes + ingress specs per lane, per-lane contract
  stamps/documents, lockfile-closure env identity, and the model type's
  `checkpoint_defaults_schema` (the successor of the hub-embedded per-family
  defaults registry). New surface: `gen_worker.Model` / `LoadContext` /
  `Adapter` / `@entrypoint` (interim primitives — pgw#1382 owns the runtime
  classes), `gen_worker.models.Knob`/`SDXL` seam (pgw#1376/#1377), and the
  callable `gen_worker.entrypoint` module shim (name-collision interim,
  rename decision with pgw#1372/#1373). torchcg rides as a dev-only PEP 735
  dependency group — the published wheel still carries no torchcg
  requirement.
