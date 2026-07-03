# `Resources` cost-shape fields

The cost-shape fields on `Resources` tell the orchestrator how a function's
VRAM and runtime grow with payload inputs. They're used for **admission**
(reject jobs that won't fit even on the largest GPU class) and **scheduling**
(prefer faster GPU classes for jobs that will benefit).

`Resources` also carries the static placement envelope (`min_vram_gb`,
`min_compute_capability`, …). The placement floor is a hard gate; the cost-shape
fields are predictions, with coefficients learned from observed runs.

## The fields

```python
from gen_worker import Resources

Resources(
    requires_gpu=True,
    min_vram_gb=14.0,                # hard placement floor

    # Cost-shape fields:
    vram_must_fit="full_model",
    vram_base=500 * 1024 * 1024,
    vram_size_multiplier=1.10,
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("num_inference_steps", "num_images_per_prompt"),
)
```

### `vram_must_fit`

`"full_model"` or `"largest_component"`. Selects which `source.size_facts`
entry the orchestrator multiplies by `vram_size_multiplier` when computing
required VRAM at submit time.

- `"full_model"` — the whole pipeline must be resident at once.
- `"largest_component"` — only the largest sub-module needs to fit (the
  orchestrator and worker know how to swap components in and out).

### `vram_base`

Constant VRAM overhead in bytes — runtime buffers, optimizer state,
intermediate activations that don't scale with payload size. Add this to the
size-fact multiplied component to get the static-envelope VRAM prediction.

### `vram_size_multiplier`

Multiplier applied to `source.size_facts[vram_must_fit]`. Accounts for the
gap between model bytes on disk and model bytes resident in VRAM (typically
1.05–1.20 because of CUDA workspace + activation buffers).

### `vram_scales_with`

Tuple of payload field names. The orchestrator learns a per-field coefficient
from observed runs — bigger field values predict more VRAM. Use this for the
inputs that genuinely move memory: image dimensions, batch size, sequence
length, video frame count.

### `runtime_scales_with`

Tuple of payload field names. The orchestrator learns a per-field, per-GPU
class coefficient. Use this for inputs that move wall-clock time: number of
inference steps, number of images, audio length.

---

## What the orchestrator does with them

**At admission** — when a job arrives, the orchestrator computes a predicted
VRAM envelope:

```
predicted_vram = vram_base
               + vram_size_multiplier * size_facts[vram_must_fit]
               + Σ coeff(field) * payload[field]   for field in vram_scales_with
```

If `predicted_vram > max(VRAM across all GPU classes)`, the job is rejected
before any worker spins up.

**At scheduling** — the orchestrator computes predicted runtime per GPU
class:

```
predicted_runtime[gpu] = Σ coeff(field, gpu) * payload[field]   for field in runtime_scales_with
```

Jobs are routed to the cheapest GPU class that meets the latency target.

**Learning** — the coefficients are learned from observed run telemetry
across the fleet, per `(function, gpu_class, field)`. The tenant declares
*which* fields drive cost; the platform measures *how much*.

---

## scales_with fields must name real payload fields

Every name in `vram_scales_with` and `runtime_scales_with` must reference a
real field on the function's payload struct (the first segment, before any
`.` or `[`). Keep them in sync with the struct — a name that doesn't match a
payload field is a silent bug: the cost-shape coefficient never fires.

```python
class Input(msgspec.Struct):
    prompt: str
    num_images: int = 1

# OK — `num_images` is a real field:
Resources(runtime_scales_with=("num_images",))

# Bug — payload struct has no `width` field, so this hint never applies:
Resources(vram_scales_with=("width",))
```

Dotted paths are allowed (only the head is validated):

```python
Resources(vram_scales_with=("specs[0].scheme",))   # validates `specs` exists
```

---

## Cost-shape vs. placement floor

`Resources.min_vram_gb` is the **hard placement floor**. A host with less
VRAM than this is never considered for this function — the worker marks the
function unavailable at boot.

The cost-shape fields are **predictions** the orchestrator uses for
admission and scheduling. They can over- or under-estimate; coefficients
are learned over time.

For dispatch picks where the table can resolve to multiple checkpoints,
set `min_vram_gb` to the **largest** pick's requirement. The cost-shape
fields then tell the orchestrator how to bias scheduling per payload.

```python
_flux_dispatch = Resources(
    requires_gpu=True,
    min_vram_gb=14.0,                # largest pick: int8 at 14GB
    vram_must_fit="full_model",
    vram_base=500 * 1024 * 1024,
    vram_size_multiplier=1.10,
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("num_inference_steps", "num_images_per_prompt"),
)

@inference(
    resources=_flux_dispatch,
    models={"pipeline": dispatch(
        field="variant",
        table={
            "nf4":  flux.flavor("nf4"),     # ~6GB
            "int8": flux.flavor("int8"),    # ~14GB
        },
    )},
)
class GenerateBnb:
    def setup(self, pipeline) -> None:
        self.pipeline = pipeline

    @invocable(name="generate_bnb")
    def generate_bnb(self, ctx, payload: BnbInput) -> Output: ...
```

A request for `variant="nf4"` with small dimensions can still admit on a
14GB host because the placement floor is satisfied; the predicted VRAM via
the cost-shape fields would actually fit on a smaller GPU, but the floor
gate stays where the largest pick demands.

When you need finer-grained per-variant VRAM floors, split the function:
one `@inference` class per variant, each with its own `Resources` and a
fixed pick.
