# `Src.PAYLOAD_REF` — caller-supplied checkpoint refs

Lets a caller pass their own checkpoint ref at invoke time against an endpoint
that was NOT pre-configured for that ref. Opt-in per function; default posture
is still to pin via `FIXED` / `PAYLOAD`.

## When to use

Use `Src.PAYLOAD_REF` when the endpoint is deliberately open to
arbitrary fine-tunes of a known family:

- `cozy/sdxl-inference` accepts any SDXL derivative
- `cozy/llama-chat` accepts any Llama-3 family causal LM
- `cozy/flux-lora-inference` accepts any Flux.2 base + caller LoRA

Do NOT use it for:

- One-function-one-checkpoint endpoints (stay on `FIXED`)
- Curated shortlists where the endpoint author wants to audit every
  supported checkpoint (stay on `PAYLOAD` with a per-function keyspace)
- Latency-SLA endpoints (stay on `FIXED`; every allowed checkpoint
  can be warmed)

## Basic usage

```python
from typing import Annotated
from diffusers import StableDiffusionXLPipeline
from gen_worker import inference_function, RequestContext
from gen_worker.api.injection import ModelRef, ModelRefSource as Src

class GenerateInput(msgspec.Struct):
    prompt: str
    num_inference_steps: int = 8

@inference_function
def generate(
    ctx: RequestContext,
    pipeline: Annotated[
        StableDiffusionXLPipeline,                    # ← auto-derived compat gate
        ModelRef(Src.PAYLOAD_REF, key="model_ref"),
    ],
    payload: GenerateInput,
):
    return pipeline(payload.prompt, num_inference_steps=payload.num_inference_steps).images[0]
```

Caller at invoke time:

```json
{
  "prompt": "a watercolor cat",
  "num_inference_steps": 20,
  "model_ref": "alice/my-sdxl-ft@sha256:abc..."
}
```

**No `[models]` section in endpoint.toml required.** The function
signature carries the expected pipeline class, and discovery pulls
`StableDiffusionXLPipeline.__name__` as the derived compat gate.

## Compat validation (what gen-orchestrator checks before dispatch)

Every caller ref goes through layered validation BEFORE the request
reaches a worker. Cheapest checks first.

### Axis 1 — access control
Caller's JWT must have read permission on the ref. Rejects as
`ref_access_denied` (404 — identical to "doesn't exist" for
visibility-opaque orgs).

### Axis 2 — file layout
`_file_layout` on the caller's ref must match the endpoint's expected
layout. Rejects as `ref_file_layout_mismatch`. Cheapest because it's
a single attribute compare.

### Axis 3 — pipeline class / architectures
- Diffusers: `_pipeline_class` (from `model_index.json._class_name` at
  commit time) must match or be in the function's allowlist. Rejects
  as `ref_pipeline_class_mismatch`.
- Transformers: `_architectures` (from `config.json.architectures`) must
  intersect the derived/allowed set. Rejects as
  `ref_architectures_mismatch`.

### Axis 4 — per-component class map
Diffusers repos carry per-component library + class info in
`model_index.json` (`unet: ["diffusers", "UNet2DConditionModel"]`).
When the endpoint declares `required_components`, each component
class must match. Rejects as `ref_component_class_mismatch` with
the offending component name echoed.

### Axis 5 — lineage
When the function declares `require_lineage_descendant_of=X`,
tensorhub's `ListAncestors` DAG walk must find X in the caller's
ref's ancestor chain (max depth 10). Rejects as
`ref_not_lineage_descendant`. When `require_lineage_verified=True`
filters to `verification_status='verified'` edges only.

### Axis 6 — attribute scope
When the function declares `required_attributes={k: [v1, v2]}`, the
caller ref's attributes must subset-contain it. Rejects as
`ref_attribute_mismatch`.

## Auto-dispatch base classes are non-restricting

Some diffusers / transformers classes are deliberately broad — they
exist precisely to dispatch to whatever concrete class the checkpoint
carries. Discovery recognizes these and emits NO derived class gate;
remaining axes (file_layout, lineage, attributes) still apply.

Recognized classes:

- **diffusers**: `DiffusionPipeline`, `AutoPipelineForText2Image`,
  `AutoPipelineForImage2Image`, `AutoPipelineForInpainting`
- **transformers**: `AutoModel`, `AutoModelForCausalLM`,
  `AutoModelForSeq2SeqLM`, `AutoModelForImageClassification`,
  `AutoModelForVision2Seq`, `PreTrainedModel`

The list lives in
`python-gen-worker/src/gen_worker/discovery/known_pipelines.py` with a
matching Go mirror in
`gen-orchestrator/internal/release/compat_classes.go`. Both files
carry a `KNOWN_PIPELINES_REVISION` token that must match — discovery
embeds the revision in each compat_spec so orchestrator can WARN on
drift between the two lists.

## Explicit overrides — `allow_pipeline_classes` / `allow_architectures`

Tenant authors with non-standard subclasses that should still accept
any family-compatible checkpoint:

```python
pipeline: Annotated[
    CustomSDXLSubclass,                    # your local subclass
    ModelRef(
        Src.PAYLOAD_REF, key="model_ref",
        # Accept any of these pipeline classes at invoke time.
        allow_pipeline_classes=["StableDiffusionXLPipeline", "CustomSDXLSubclass"],
    ),
]
```

`allow_*` fields replace the signature-derived gate entirely. Tenants
rarely need these — use only when the signature's class name doesn't
reflect the full set of acceptable checkpoint classes.

## Other scoping knobs

```python
ModelRef(
    Src.PAYLOAD_REF, key="model_ref",
    allow_pipeline_classes=["StableDiffusionXLPipeline"],
    required_file_layout="diffusers",                   # axis 2
    required_components={                                # axis 4
        "unet": "UNet2DConditionModel",
        "vae": "AutoencoderKL",
    },
    require_lineage_descendant_of="stabilityai/stable-diffusion-xl-base-1.0",  # axis 5
    require_lineage_verified=True,                       # axis 5 tightening
    required_attributes={"dtype": ["bf16", "fp16"]},     # axis 6
)
```

All fields are optional. Omitted fields → that axis is not gated.

## Mixing binding modes

Three modes can coexist on one function. A common pattern — fallback
base model + caller override:

```python
@inference_function
def generate(
    ctx: RequestContext,
    base: Annotated[
        StableDiffusionXLPipeline,
        ModelRef(Src.FIXED, "sdxl_base"),       # fallback; endpoint.toml [models] pre-declares
    ],
    pipeline: Annotated[
        StableDiffusionXLPipeline,
        ModelRef(Src.PAYLOAD_REF, key="model_ref"),  # caller override
    ],
    payload: GenerateInput,
): ...
```

Tenant logic: prefer `pipeline` when `model_ref` was supplied; fall
back to `base` otherwise. Downstream gen-orchestrator resolves both
refs, but only the one populated is materialized.

## What compat validation does NOT catch

Post-download surprises the pre-dispatch check can't see:

- Merged LoRAs that added tensor names the base pipeline doesn't expect
- Fine-tunes that renamed a tensor block
- VAE swaps (compatible pipeline class, different latent channel count)
- Subtle scheduler config drift

These reach the worker and fail at `from_pretrained` time. Worker
classifies them as `error_type=ref_compatibility_surprise` so the
caller's error response distinguishes "incompatible ref" from "infra
flake." Not prevented — just labeled.

## Client Payload

```json
{
  "prompt": "a watercolor cat",
  "model_ref": "alice/my-sdxl-ft@sha256:abc..."
}
```

`model_ref` is the payload field declared by the function. Multi-ref functions
should declare one payload field per ref, or accept an explicit structured
payload that contains the refs.

## Placement implications (pairs with #47)

When #47's per-request hardware projection sees a `PAYLOAD_REF`-resolved
checkpoint, it pulls that ref's `_vram_bytes_loaded` +
`_min_compute_capability` and routes the request to the smallest
sufficient worker. A caller's 24 GB fine-tune ref routes to an A100
automatically; a 6 GB base routes to an L4. Without projection, the
caller's fine-tune might submit against the endpoint's default tier
and 424 at the worker-level VRAM gate post-dispatch.

## Admin introspection

```bash
curl -H "Authorization: Bearer $TOKEN" \
  $ORCH_URL/v1/admin/endpoints/cozy/sdxl-inference/functions/generate/compat-probe
```

Returns the function's compat_spec: derived pipeline class, any
explicit allowlists, file_layout / lineage / component / attribute
scoping. Read-only introspection for operators.

## Related

- Depends on tensorhub's `repo_checkpoint_lineage` DAG (shipped in #12).
- Pairs with #47 (per-request hardware projection) for good placement
  on caller-supplied refs.
- Pairs with #48 (inference compute override) — fine-tune callers
  often want to pick their own GPU too.
