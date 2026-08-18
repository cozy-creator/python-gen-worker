- **pgw#1377: `gen_worker.models` grows the model-typed serving-defaults
  vocabulary (the pgw#1376 program's gen-worker leg).** `ModelType` + the launch
  set (`SDXL`, `SD15`, `SD2`, `HiDreamO1`, `Wan22`, plus `SDXL.Lora` /
  `SD15.Lora` overlays) — each a name + frozen `Defaults` struct whose field
  defaults ARE the platform fallback values (zero-arg = servable trace fixture)
  + a tensorfs contract-pattern ingest fingerprint. `SDXL.Recipe`/`SD15.Recipe`
  carry the five independent serving axes (cfg, steps, guidance, scheduler,
  timesteps — cfg and few-step are separate facts) and BOTH defaults types
  inherit them, so an endpoint annotates `recipe: SDXL.Recipe` — one nominal
  type, no union. `SchedulerName` is the platform-wide scheduler vocabulary
  (checkpoint metadata layer of the request > metadata > tree > endpoint-default
  chain; None = trust the tree's shipped scheduler; endpoints own their served
  subset and constructor tables). `SDXL.Defaults.step_distilled` is the
  checkpoint-level fused-merge fact, decoupled from `cfg` (the guidance axis) —
  it is what makes stacking a distillation adapter harmful (endpoints warn and
  ignore). Each model type also carries `canonical_scheduler_config` — the
  architecture's training-time scheduler config (SDXL: scaled_linear
  0.00085/0.012 epsilon) — the data ingest synthesizes into a classified tree
  that ships no scheduler_config.json, replacing any endpoint-side backstop. `Knob` is the one reusable
  min/max/default triple: `d.steps.resolve(value, ctx)` defaults on None and
  clamps caller-visibly through `RequestContext.clamp` — it never rejects (the
  endpoint's API Meta bounds rejected upstream). `decode_model_defaults` is the
  read-side authority over the hub's two-column row: partial JSONB overlays
  field-by-field onto platform values, knob ranges merge to the narrowest
  layer, ill-typed rows are typed refusals naming the field, an unclassified
  checkpoint serves fallbacks under the `checkpoint_defaults_unclassified`
  warning, and a mistyped one is a `ModelTypeMismatch` refusal. `gen-worker
  models export` emits the mechanical `{names, schemas}` document (draft
  2020-12, generated from the structs) that th#2140's recognized-name guard and
  th#2141's write-time validation consume. Seams left open by design: the
  `canonical_contract` objects are `PendingContract` placeholders until the
  vendored tensorfs ships `contracts` (tensorfs#111), and state-dict sniffing
  rides tensorfs's matcher — this side only maps recorded stamps to names.
