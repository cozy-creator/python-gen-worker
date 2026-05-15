# Cookbook: Image Diffusion Endpoints

One page to ship a production-quality image-diffusion endpoint on the SDK
foundation (#322 class shape) with the canonical acceleration stack
(#324: torch.compile + ParaAttention FBCache + NVFP4).

Audience: you know Python and PyTorch, you've written a Diffusers
pipeline before, and you have a model in mind. You don't yet know the
gen-worker conventions.

Cross-links:
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference
  (bindings, Resources, payload structs, RequestContext).
- [cookbook-acceleration.md](cookbook-acceleration.md) — the
  `gen_worker.accel` five-call surface in isolation.
- [cookbook-video-diffusion.md](cookbook-video-diffusion.md) — video DiTs.
- [cookbook-stages.md](cookbook-stages.md) — `@inference.stage` for
  multi-stage pipelines (mostly 3D and very large DiTs).

---

## The class shape

Every inference endpoint is a class with four lifecycle hooks plus one
or more invocable methods. There are no function-shape endpoints
anymore.

```python
from gen_worker import RequestContext, Repo, Resources, inference

flux = Repo("black-forest-labs/flux.2-klein-4b-base")

@inference(
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=22.0,
        min_compute_capability=8.0,
    ),
    models={"pipe": flux.flavor("bf16")},
    allowed_shapes=((1024, 1024),),
)
class FluxGenerate:
    def setup(self, pipe):
        """Cheap. Load weights, apply compile/cache/quant wrappers."""
        self.pipe = pipe

    def warmup(self):
        """Expensive. Run a dummy forward at each allowed_shape so
        torch.compile and CUDA graphs are captured before the worker
        reports `ready`."""
        ...

    @inference.function
    def generate(self, ctx: RequestContext, payload):
        ...

    def shutdown(self):
        """Release engines / CUDA graph caches."""
        pass
```

The five contract points the SDK depends on:

| Hook                  | When called                              | Required? |
|-----------------------|------------------------------------------|-----------|
| `__init__(self)`      | Worker process boot, before `setup`      | optional  |
| `setup(self, **models)` | After model bindings resolve            | yes       |
| `warmup(self)`        | After setup, before traffic              | optional  |
| `@inference.function` | Each request                             | yes (≥1)  |
| `shutdown(self)`      | Worker drain (graceful) or termination   | yes       |

`models={"pipe": ...}` keys MUST match `setup`'s kwargs by name. The SDK
constructs the right kwargs for you from the binding resolution.

`allowed_shapes` is a tuple of `(H, W)` pairs that `warmup()` should
sweep through. Declaring shapes here lets the orchestrator route only
matching requests to this worker and tells `torch.compile` how many
graphs to cache.

---

## The canonical acceleration stack

Three layers, applied in `setup()`, in this order:

1. **`torch.compile`** — the foundation. Inductor + CUDA graphs.
   ~1.5-2× on most DiTs.
2. **A per-request cache wrapper** — `FBCache` for DiTs (Flux family),
   `DeepCache` for UNets (SDXL), `TeaCache` for video DiTs. Stacks
   multiplicatively with compile.
3. **Quantization** — `NVFP4` on Blackwell (B100/B200/B300, SM 10+),
   `FP8` on Hopper (H100, SM 9+), `INT8` everywhere else.

Realistic stacked speedup: ~3-5× on H100, ~8-10× on B200. Numbers come
from WaveSpeed.ai's production stack (the author of ParaAttention runs
this in production).

```python
from gen_worker import compile as gw_compile, cache as gw_cache, quant as gw_quant

def setup(self, pipe):
    # 1. Compile the heavy module (transformer for DiTs, unet for UNets).
    pipe.transformer = gw_compile.torch_compile(
        pipe.transformer, mode="reduce-overhead"
    )
    # 2. Per-request feature cache.
    gw_cache.FBCache(threshold=0.12).apply(pipe)
    # 3. Quantize on Blackwell; auto-fallback to FP8 on Hopper.
    pipe.transformer = gw_quant.nvfp4(pipe.transformer, fallback="fp8")
    self.pipe = pipe
```

### `gen_worker.compile`

`torch_compile(module, mode="reduce-overhead")` — the default. Wraps
`torch.compile` with CUDA graph capture on top of Inductor codegen.
`mode="max-autotune"` trades startup time for steady-state throughput;
use it when you can afford long warmup.

`nexfort_compile(module)` — OneDiff's torch.compile backend. Typically
10-30% over stock Inductor on Flux/SD3-class DiTs. Requires
`pip install nexfort onediff`.

`tensorrt(module, precision="bf16")` — Model-Optimizer or
`torch_tensorrt`. On pre-Hopper hardware, no-ops with a warning and
falls back to `torch_compile`. Optional.

### `gen_worker.cache`

| Wrapper       | Architecture    | Speedup        | Stack notes                              |
|---------------|-----------------|----------------|------------------------------------------|
| `FBCache`     | DiT (Flux, SD3, Qwen-Image) | ~1.5-2×        | Cheap install (`pip install para-attn`)  |
| `DeepCache`   | UNet (SDXL family)          | ~2.5-3×        | Per CVPR 2024 paper                      |
| `TeaCache`    | DiT, video                  | ~1.6-2×        | `breaks_cross_request_batching = True`   |

`TeaCache` cannot be combined with cross-request micro-batching
(nunchaku #597 — `batch > 1` breaks the cache key). The SDK reads the
flag and auto-disables the batching aggregator on workers using
TeaCache. Stay aware: enabling TeaCache trades throughput-via-batching
for per-request latency.

### `gen_worker.quant`

| Helper           | Hardware       | Notes                                        |
|------------------|----------------|----------------------------------------------|
| `nvfp4(module)`  | Blackwell SM 10+ | ~3-6× on Flux.2 per NVIDIA. Falls back to FP8 on Hopper. |
| `fp8(module)`    | Hopper SM 9+   | ~40% VRAM reduction on H100. Falls back to INT8 on older. |
| `int8(module)`   | Turing+        | bitsandbytes LLM.int8(). No calibration step. |

Calibration artifacts are content-addressed-cached at
`$TORCHINDUCTOR_CACHE_DIR/../gen_worker_quant_cache` (sibling of the
compile cache), keyed on `(method, module class, state_dict shape
fingerprint, torch version, CUDA cc, extra kwargs)`. Re-deploys hit the
cache; cold starts pay the calibration cost once per
`(checkpoint, hardware)` pair.

### `CompileUnavailableError` / `CacheUnavailableError`

Every helper lazy-imports its third-party dependency inside `.apply()`
or the function call. The wheel doesn't pull in `para-attn`,
`DeepCache`, `teacache`, `nexfort`, `nvidia-modelopt`, etc. You install
whatever you need in your endpoint image.

If you call a helper whose dep is missing, you get a clear typed error
with the install hint:

```python
try:
    gw_cache.FBCache().apply(self.pipe)
except gw_cache.CacheUnavailableError as e:
    # Logged with the missing package and install line.
    raise
```

Practically: pin the helpers you actually use in your endpoint's
`pyproject.toml`. The error is the safety net for misconfigured
images, not a runtime fallback.

---

## Per-model recommended config

These are the configurations the helper library was designed around.
Pick the row that matches your model.

| Model class                          | Compile      | Cache           | Quant                  | Micro-batching | Notes                                             |
|--------------------------------------|--------------|-----------------|------------------------|----------------|---------------------------------------------------|
| **SDXL** (Lightning, Juggernaut, …)  | `torch_compile` | `DeepCache(3)`  | `int8` opt.            | only if step-distilled to <300ms | UNet model. DeepCache is the unique-to-UNets win.  |
| **Flux.2 / SD3 / Qwen-Image**        | `torch_compile` | `FBCache(0.12)` | `nvfp4` / `fp8`        | conflicts with FBCache+TeaCache; OK with FBCache off | The DiT canon.                                    |
| **Sana-Sprint**                      | `torch_compile` (only) | —               | optional               | yes — opt in       | Step-distilled to 1024² in 0.1s; cache wrappers add little. |
| **HiDream-I1**                       | `torch_compile` | `FBCache`       | `nvfp4` recommended    | no             | Same family as Flux DiTs.                         |
| **FLUX.2-klein-4B**                  | `torch_compile` | `FBCache`       | `nvfp4` (B200) / `fp8` (H100) | no       | Reference deployment for the canonical stack.      |

**SDXL specifics.** `DeepCache(cache_interval=3, cache_branch_id=0)` is
the typical sweet spot — paper measures ~2.5-3×. UNet model, so
`pipe.unet` is the compile target.

**Flux / SD3 / Qwen-Image specifics.** `FBCache(threshold=0.12)` per
ParaAttention defaults. DiT model, so `pipe.transformer` is the compile
target. NVFP4 on Blackwell delivers the canonical 8-10× total stack.

**Sana-Sprint specifics.** Distilled to <300ms per 1024² image — that
puts it in the cross-request micro-batching window. Add
`max_concurrent_per_worker=4` on `@inference.function` and the SDK's
aggregator will pack concurrent requests into one forward. Cache
wrappers add little because the forward is already short.

---

## What NOT to do

**Don't mutate `self` inside `generate()`.** The class is shared across
all requests this worker handles. Assigning `self.foo = payload.x`
inside `generate()` leaks state between requests.

```python
# WRONG — request 2 sees request 1's seed.
@inference.function
def generate(self, ctx, payload):
    self.seed = payload.seed              # leaks across requests
    return self._render(...)

# RIGHT — payload data stays local.
@inference.function
def generate(self, ctx, payload):
    seed = payload.seed                   # local, ephemeral
    return self._render(seed=seed, ...)
```

The only legitimate `self.x =` sites are inside `setup()` (and
`shutdown()`'s teardown). If you find yourself wanting `self.x =`
inside `generate()`, it's almost always either (a) a payload field
that should be a local variable, or (b) a per-request cache that
should be a `dict[request_id, T]` keyed on `ctx.request_id`.

**Don't apply TeaCache and cross-request micro-batching together.**
TeaCache's `breaks_cross_request_batching = True` flag tells the SDK
to auto-disable the batching aggregator. But if you also explicitly
set `max_concurrent_per_worker > 1`, the SDK can't reconcile the
intent. Pick one:

- For latency-sensitive endpoints (Flux/SD3/HunyuanVideo at high
  step counts): TeaCache, single-request.
- For throughput on short-forward models (Sana-Sprint, step-distilled
  variants): micro-batching, no TeaCache.

**Don't compile in `generate()`.** `torch.compile` is a setup-time
operation. Calling it on the request path means every cold request
pays the compile cost. If you need per-request shape variability, set
`dynamic=True` on `torch_compile`, but prefer declaring
`allowed_shapes` on the class and letting `warmup()` populate the
compile cache.

**Don't return PIL objects directly.** Use
`gen_worker.io.write_image(ctx, "out", img)` to encode and persist.
The Asset returned carries the storage ref the orchestrator uses to
hand the file back to the caller.

---

## Complete working example: Flux.2-klein-4B with the full stack

A production-shaped endpoint with compile + FBCache + NVFP4, falling
back gracefully on non-Blackwell hardware:

```python
"""Flux.2-klein-4B image generation, full acceleration stack."""

from __future__ import annotations

import msgspec
from diffusers import Flux2KleinPipeline

from gen_worker import (
    Asset,
    Repo,
    RequestContext,
    Resources,
    cache as gw_cache,
    compile as gw_compile,
    io as gw_io,
    inference,
    quant as gw_quant,
)


flux = Repo("black-forest-labs/flux.2-klein-4b-base")


class GenerateInput(msgspec.Struct):
    prompt: str
    num_inference_steps: int = 4    # klein is step-distilled to 4
    guidance_scale: float = 1.0     # CFG-free; folded into the weights
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    num_images_per_prompt: int = 1


class GenerateOutput(msgspec.Struct):
    image: Asset


@inference(
    label="Flux.2-klein-4B (compile + FBCache + NVFP4)",
    description=(
        "Flux.2-klein-4B turbo with the canonical SerialWorker "
        "acceleration stack: torch.compile + ParaAttention FBCache + "
        "NVFP4 weight quant. NVFP4 falls back to FP8 on Hopper and "
        "INT8 on older hardware."
    ),
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=10.0,            # NVFP4 footprint
        min_compute_capability=8.0,        # FP8 path needs Hopper+; NVFP4 needs Blackwell
        peak_vram_per_request_gb=12.0,
        vram_must_fit="full_model",
        vram_base=500 * 1024 * 1024,
        vram_size_multiplier=1.10,
        vram_scales_with=("width", "height", "num_images_per_prompt"),
        runtime_scales_with=("num_inference_steps", "num_images_per_prompt"),
    ),
    models={"pipe": flux.flavor("bf16")},
    allowed_shapes=((1024, 1024),),
)
class FluxKleinGenerate:
    def setup(self, pipe: Flux2KleinPipeline) -> None:
        # Compile the heavy DiT. mode='reduce-overhead' captures CUDA
        # graphs on top of Inductor codegen; the right default for
        # static-shape inference.
        pipe.transformer = gw_compile.torch_compile(
            pipe.transformer, mode="reduce-overhead"
        )
        # Per-request feature cache. threshold=0.12 is the
        # ParaAttention upstream default.
        gw_cache.FBCache(threshold=0.12).apply(pipe)
        # Quantize. On Blackwell: NVFP4. On Hopper: FP8 (auto-fallback).
        # On Ampere or older: INT8 (auto-fallback via fp8).
        pipe.transformer = gw_quant.nvfp4(
            pipe.transformer, fallback="fp8"
        )
        self.pipe = pipe

    def warmup(self) -> None:
        # One dummy forward at the declared shape so torch.compile +
        # FBCache state are warm before the worker reports ready.
        _ = self.pipe(
            prompt="warmup",
            num_inference_steps=2,
            width=1024,
            height=1024,
        )

    @inference.function(
        timeout_ms=60_000,
        description="Generate one or more 1024² images from a prompt.",
    )
    def generate(self, ctx: RequestContext, payload: GenerateInput) -> GenerateOutput:
        import torch

        ctx.raise_if_canceled()

        # Local state ONLY — never assign to self inside generate().
        steps = max(4, min(8, int(payload.num_inference_steps)))
        gen = None
        if payload.seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(payload.seed))

        ctx.progress(0.1, stage="denoise")
        result = self.pipe(
            prompt=payload.prompt,
            num_inference_steps=steps,
            guidance_scale=payload.guidance_scale,
            width=payload.width,
            height=payload.height,
            num_images_per_prompt=payload.num_images_per_prompt,
            generator=gen,
        )
        ctx.raise_if_canceled()

        ctx.progress(0.95, stage="encode")
        out = gw_io.write_image(ctx, "out", result.images[0], format="webp", quality=92)
        ctx.progress(1.0, stage="done")
        return GenerateOutput(image=out)

    def shutdown(self) -> None:
        # Drop references; the SDK takes care of CUDA memory release
        # on worker termination.
        self.pipe = None
```

What this gives you on B200:
- Cold start: ~30-60s (model load + compile + NVFP4 calibration on
  first run; cached after).
- TTFT warm: <2s for a 1024² 4-step generation.
- Steady-state: ~10× over the bf16 baseline per NVIDIA's measurements.

What this gives you on H100:
- NVFP4 falls back to FP8 (`fallback="fp8"`).
- Stacked speedup ~3-5× over bf16.

What this gives you on a 4090:
- FP8 falls back to INT8 via bitsandbytes.
- Compile + FBCache still apply; INT8 weight footprint fits the
  24GB envelope comfortably.

---

## Next steps

- **More than one variant?** Pin per-quantization classes (one class
  per fixed pick) or use `dispatch(field, table)` to route on a
  payload discriminator. See
  [endpoint-authoring.md](endpoint-authoring.md#dispatch-pick--payload-driven-selection).
- **Two-stage pipelines** (SDXL base + refiner, IP-Adapter chains)?
  Inject multiple bindings in `models={}` and consume them as separate
  `setup` kwargs. See the multi-param injection section in
  [endpoint-authoring.md](endpoint-authoring.md#multi-param-injection--each-binding-is-independent).
- **Multi-stage pipelines where future disaggregation will pay off**
  (very large DiTs where the text encoder and VAE could batch on
  small GPUs)? Use `@inference.stage(name=..., gpu_class=...)`. See
  [cookbook-stages.md](cookbook-stages.md).
- **Video generation?** See [cookbook-video-diffusion.md](cookbook-video-diffusion.md)
  — different acceleration tradeoffs (TeaCache, FP8/NVFP4, xDiT
  sequence parallelism on multi-GPU).
