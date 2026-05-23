# Shared Model Components

Issue: `agents/progress.json` #334.

## Goal

`Repo` / `HFRepo` bindings declare model identity. The worker should use that
identity to share immutable base model components across compatible functions
inside one process, instead of letting each endpoint class independently call
`from_pretrained()` and allocate duplicate CUDA storage.

The worker owns sharing. Endpoint code should not manually coordinate Python
globals just to avoid duplicate weights.

## Cache Key

Loaded component cache entries are process-local and GPU-local. A cache key must
include every field that can change loaded tensor identity or layout:

- provider: `tensorhub`, `hf`, or `civitai`
- resolved ref: canonical repo/checkpoint identity after tag/flavor/revision
- revision or snapshot digest when available
- dtype
- quantization mode and quantization config
- pipeline class or component-set identity
- device id
- placement/offload mode

CUDA storage is not shared across GPU devices. A two-GPU worker normally has
one loaded component entry per `device_id`.

## Ownership Model

The cache owns immutable base components only. Function instances own mutable
state:

- pipeline wrappers
- schedulers
- LoRA/adapters
- compiled modules/wrappers
- request-local state

The worker may inject the exact same object only when a binding marks that as
safe. The default Diffusers shape should be:

1. Load one canonical base pipeline/component set into the component cache.
2. Create function-owned pipeline objects from shared components with
   `from_pipe()` / `pipeline.components` / ComponentsManager where available.
3. Refcount shared component entries.
4. Unload only when every function/request using the entry has released it.

## Worker Insertion Points

Current SerialWorker flow:

1. Discovery serializes class-level `models={...}` bindings into
   `endpoint.lock`.
2. Worker boot parses function/class binding metadata from the manifest.
3. `_ensure_serial_class_started()` calls `_resolve_serial_model_paths()` and
   passes resolved values into class `setup()`.
4. Endpoint `setup()` commonly calls `DiffusersPipeline.from_pretrained(...)`.

The component cache belongs between steps 3 and 4. Instead of passing only a
local path/ref for known Diffusers bindings, the worker should be able to pass
a cached component handle or a pipeline wrapper built from cached components.

The API should stay narrow:

```python
key = LoadedComponentKey.from_binding(binding, resolved_model, device_id=device_id)
handle = component_cache.acquire(key, loader=load_base_components)
try:
    pipeline = handle.pipeline_for(function_name, mutable=True)
    instance.setup(pipeline=pipeline)
finally:
    component_cache.release(handle)
```

The concrete loader may still call Diffusers `from_pretrained()`, but it should
do it once per compatible key.

## LoRA And Compile Safety

LoRA overlays mutate pipeline/module state in most Diffusers paths. Torch
compiled graphs are also bound to concrete modules and shapes. These states must
not share mutable objects.

Compiled functions must opt out of:

- `_models.<binding>.ref` model overrides
- `_models.<binding>.loras[]` LoRA overlays

If a future product needs compiled custom models or compiled LoRA variants, each
variant needs its own explicit compile cache key and startup benchmark. Do not
silently reuse the base compiled graph for a mutated model.

## Measurement Gate

Before building the full component cache, measure whether the compiled FLUX.2
Klein path is worth keeping:

- compile warmup time for each of the seven aspect-ratio buckets
- total startup delay caused by compile
- warm worker-side inference time for `generate` vs `generate_compiled`
- complete round-trip time for the same requests
- CUDA allocated/reserved for base-only, compiled-only, and both resident
- storage-sharing diagnostics proving whether tensors are shared

Keep the compiled path only if it shows at least a 25-30% warm worker-side
inference speedup, or a clearly better realistic p95 round-trip.
