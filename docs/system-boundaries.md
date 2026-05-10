# System Boundaries

`python-gen-worker` is a reusable worker library. It is not a published
endpoint, an operator CLI, or the owner of product-specific conversion jobs.

## What This Package Owns

- Worker authoring APIs: decorators, `RequestContext`, typed payload helpers,
  streaming helpers, model injection, and trainer/conversion contract types.
- Build-time discovery: `python -m gen_worker.discovery` scans the endpoint
  project and writes `/app/.tensorhub/endpoint.lock`.
- Runtime startup: `python -m gen_worker.entrypoint` loads the baked lock file,
  connects to the scheduler, and runs advertised functions.
- Platform protocol assumptions: Tensorhub model refs, checkpoint attributes,
  cache/CAS layout, upload/download APIs, and the worker scheduler gRPC protocol.
- Function capability metadata: resource requirements, optional library
  requirements, supported precisions/profiles, and worker-local availability.
- Generic conversion primitives and metadata: dtype vocabulary, file/layout
  helpers, calibration policy helpers, dataset descriptors, source/destination
  descriptors, lower-level safetensors/repackage/streaming helpers, and inline
  conversions that are safe to treat as local library utilities.

## What This Package Must Not Own

- Published endpoint names, endpoint catalogs, or endpoint function names.
- Operator commands or CLI flows for launching conversion work.
- Hardcoded paths to sibling Cozy repos or local developer workspaces.
- Product routing decisions such as "run this other endpoint/function".
- Endpoint-specific Docker image dependencies or build-profile decisions.
- Calibrated quantization workflows that require product-owned datasets,
  runtime image choices, hardware placement, or lineage semantics.

Those responsibilities belong to endpoint repos and control-plane services.
For the canonical conversion endpoint, the product conversion functions live in
`~/cozy/training-endpoints/conversion`.

## Quantization Boundary

The split is:

- `python-gen-worker`: generic primitives and metadata.
- `training-endpoints/conversion`: product conversion functions and calibrated
  quantization workflows.

`python-gen-worker` may expose primitives that workers can call directly:

- dtype casts such as `bf16`, `fp16`, and `fp32`
- weight-only torchao conversions
- weight-only bitsandbytes conversions
- GGUF conversion/quant encodings
- calibration policy resolution and structured "deferred conversion"
  requirements

`python-gen-worker` does not implement modelopt execution today. Modelopt
recipes such as `nvfp4`, `int4_awq`, and `w4a8_awq` are calibrated,
hardware-specific workflows. The library can describe that such a request
requires calibration, GPU placement, and a `modelopt` runtime, but the endpoint
function owns the actual modelopt import, calibration loop, artifact publishing,
and lineage metadata.

If a future shared modelopt helper is added here, it must be an optional helper
API with no core dependency, no published endpoint names, and no operator
command rendering.

## Command Surface Boundary

This package intentionally does not publish console scripts.

The only supported package module invocations are:

- `python -m gen_worker.discovery` for build-time `endpoint.lock` generation.
- `python -m gen_worker.entrypoint` for the worker process runtime.

Testing helpers under `gen_worker.testing` are import-only harness pieces. They
must not grow `argparse`, `__main__` entrypoints, or docs that present them as
general-purpose commands.

## Worker-Facing Import Surfaces

The top-level `gen_worker` package is intentionally small. It is for common
inference/realtime worker code:

- decorators and runtime context: `inference_function`, `realtime_function`,
  `ResourceRequirements`, `RequestContext`, `RealtimeSocket`
- payload/model binding helpers: `ModelRef`, `ModelRefSource`, `Clamp`
- common output/input types: `Asset`, `Tensors`, `LoraSpec`, `Compute`
- common errors: `ValidationError`, `RetryableError`, `ResourceError`,
  `AuthError`, `CanceledError`, `FatalError`, `OutputTooLargeError`
- common inference helpers: `iter_transformers_text_deltas`, `load_loras`,
  `apply_low_vram_config`, `with_oom_retry`
- `clone` as a lazy submodule for endpoint wrappers that delegate clone flows

Training hook types live under `gen_worker.trainer`.
Conversion hook types live under `gen_worker.conversion`.
Hardware self-disable helpers live under `gen_worker.capability`.
Model cache, downloader, pipeline loader, discovery validation, and sharding
internals live in explicit submodules and are not re-exported from
`gen_worker`.

`gen_worker.conversion` exposes only the conversion endpoint authoring
contract: `training_function`, `ConversionContext`, `Source`, `Dataset`,
`ProducedFlavor`, `StreamingWriter`, `Component`, `FileLayout`,
`TrainingFunctionSpec`, and calibration policy helpers. Lower-level ingest,
layout, safetensors, classifier, and clone plumbing remains in explicit
submodules.

## Endpoint Lock Boundary

`endpoint.lock` is the handoff between endpoint source code and the platform.
The library owns extracting the manifest shape from decorators and
`endpoint.toml`. It does not own the product meaning of a specific endpoint or
function name.

Endpoint repos own:

- which functions exist
- which optional libraries their image installs
- which conversion recipes are offered publicly
- which build profiles can run each function
- how user-facing requests are mapped onto worker functions

The control plane owns:

- publishing and storing the lock file
- request validation and routing
- worker placement based on function requirements and availability
- capability tokens and Tensorhub authorization
