# Model-selectable endpoints (SharedBase + per-request variants)

Issue: `agents/progress.json` #337. Builds on #336 (GPU mutex + VRAM↔disk
`ModelCache`) and #21 (per-model readiness).

## When to use this

Your endpoint exposes a **set** of selectable models — e.g. SDXL fine-tunes
(`illustrious` / `animagine` / `dreamshaper` / `juggernaut`) — and there may be
**far more models than fit in RAM or VRAM**. The models share most of their
weights (the two CLIP text encoders + the VAE) and differ only in one swappable
component (the UNet/transformer).

Loading N full pipelines is wasteful: memory = N × (shared stack + UNet). With
this contract memory = (shared stack, **once**) + {LRU set of variant UNets} —
so hundreds of fine-tunes become feasible (~2.5 GB UNet vs ~5 GB full pipeline;
shared stack ~1.8 GB once).

## The contract in one sentence

`setup()` builds what is **shared and load-once**; `dispatch`/variant slots are
resolved **per request** and injected into the **handler**, VRAM-readied by the
SDK.

## Authoring API

```python
from typing import Literal
import msgspec
from diffusers import StableDiffusionXLPipeline
from gen_worker import HFRepo, RequestContext, Resources, SharedBase, dispatch, inference

# Frozen stack — loaded ONCE, pinned in VRAM, shared BY REFERENCE across variants.
sdxl_base = SharedBase(
    StableDiffusionXLPipeline,
    text_encoder   = HFRepo("stabilityai/stable-diffusion-xl-base-1.0").subfolder("text_encoder"),
    text_encoder_2 = HFRepo("stabilityai/stable-diffusion-xl-base-1.0").subfolder("text_encoder_2"),
    vae            = HFRepo("madebyollin/sdxl-vae-fp16-fix"),
)

# Each selectable model = shared base + its own fine-tuned UNet (the swap unit).
MODELS = {
    "illustrious": sdxl_base.variant(unet=HFRepo("OnomaAIResearch/Illustrious-XL-v1.0")),
    "animagine":   sdxl_base.variant(unet=HFRepo("cagliostrolab/animagine-xl-3.1")),
    "dreamshaper": sdxl_base.variant(unet=HFRepo("Lykon/dreamshaper-xl-1-0")),
    "juggernaut":  sdxl_base.variant(unet=HFRepo("RunDiffusion/Juggernaut-XL-v9")),
}

class GenInput(msgspec.Struct):
    prompt: str
    model: Literal["illustrious", "animagine", "dreamshaper", "juggernaut"] = "illustrious"

@inference(models={"pipeline": dispatch(field="model", table=MODELS)})
class SDXL:
    def setup(self):                       # NO model arg — SDK pins the shared base.
        ...                                # stateless init only

    @inference.function(name="generate")
    def generate(self, ctx: RequestContext, payload: GenInput,
                 pipeline: StableDiffusionXLPipeline):
        # `pipeline` = the model named by payload.model (shared base + its UNet),
        # already swapped into VRAM by the SDK. Just run it.
        img = pipeline(prompt=payload.prompt, num_inference_steps=25).images[0]
        ...
```

### Rules the SDK enforces at build time

- **A dispatch slot must NOT be a `setup()` parameter.** `setup(self, pipeline)`
  on a per-request slot is a clear discovery-time error (it was the motivating
  runtime crash: `setup() missing 1 required positional argument: 'pipeline'`).
  Accept the slot on the **handler** instead.
- **A fixed `Repo` slot must be a `setup()` parameter** (the historical contract
  for load-once models).
- Handler parameters after `payload` must name a dispatch slot from
  `models={...}`; an unknown name is a signature error.

## What the SDK does per request

1. **Resolve** the discriminator field (`payload.model`) → the variant.
2. **Ensure VRAM-ready** via a 3-tier residency cache (below).
3. **Assemble shared-by-reference**: load the shared components once, pin them,
   and construct each variant pipeline pointing at the **same** component
   objects — only the UNet is per-model.
4. **Inject** the assembled pipeline as the handler argument.

All weight loads / swaps run under the worker GPU mutex (#336), so loads,
evictions, and inference never thrash on the same device.

## 3-tier residency cache

```
VRAM  (GPU)      active variant UNet(s) + PINNED shared base
  ↑ demote .to("cpu")     ↓ promote .to("cuda")
CPU   (host RAM) warm UNets, fast PCIe swap-in (~0.1–0.5s for ~2.5GB)
  ↑ drop bytes            ↓ re-load from disk
DISK  (CAS)      all downloaded variant weights, re-pullable from source
```

- Per-tier LRU. VRAM budget auto-sizes from torch VRAM; the CPU-RAM tier from
  `psutil` host RAM — both minus a safety margin.
- **The shared base is PINNED** — never evicted.
- Eviction **demotes** the LRU variant VRAM→CPU (not straight to disk) so a
  re-selected model swaps back over PCIe instead of re-loading from disk.

## Partial readiness

A variant's function is serveable the moment **that** variant downloads and the
shared base is up — the worker never blocks startup on the full set. A request
for a not-yet-resident variant triggers an on-demand load (and emits a
`DOWNLOADING → VRAM` residency transition) rather than failing.

## Tier-aware availability

On every load/evict/download transition the worker emits a **debounced**
`WorkerModelReadySignal` carrying the model's residency tier
(`VRAM | RAM | DISK | DOWNLOADING | ABSENT`). It is mapped onto the existing
availability-kind enum (VRAM→READY, RAM/DISK→CACHED) so legacy orchestrators
keep working; set `COZY_EMIT_RESIDENCY_TIER=1` to additionally carry the raw
tier string for a residency-aware orchestrator. The orchestrator companion
(separate repo) consumes these to route requests to the **hottest** worker for a
given model (`VRAM > RAM > DISK > ABSENT`).

## The spectrum (one cache path)

`dispatch(field=, table={...})` table values may be:

- a **`SharedBase.variant(...)`** — shared base + a full UNet/transformer (this
  guide);
- a **plain `Repo`/`HFRepo`** — a standalone full pipeline (use this as the
  escape hatch when a fine-tune retrained its own text encoder/VAE — see below);
- a **`.allow_lora()` Repo** — a tiny LoRA adapter on the shared base.

All three share the same per-request resolution + 3-tier residency path.

## Component-compatibility safety

Sharing is **by declaration**: the SDK pulls only the variant slot (`unet=`)
from each variant repo and pairs it with the tenant-declared shared components.
If a variant repo ships its **own** `text_encoder`/`vae` that differs from the
declared base (some anime SDXL fine-tunes retrain CLIP), pairing the shared CLIP
with that UNet yields subtly-wrong output — so the SDK logs a **warning at load
and ignores** the bundled component. If that model genuinely needs its own
encoder/VAE, declare it as a standalone full pipeline (a plain `HFRepo` in the
dispatch table) instead of a `.variant(...)`.
