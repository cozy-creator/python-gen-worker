# Changelog

## 0.8.0

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
  0.8.0 to emit the new field.

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
