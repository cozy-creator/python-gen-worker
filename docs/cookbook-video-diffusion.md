# Cookbook: Video Diffusion Endpoints

One page to ship a production video-diffusion endpoint. Read
[cookbook-image-diffusion.md](cookbook-image-diffusion.md) first — the
class shape and acceleration helpers are the same. This page covers
the video-specific tradeoffs: TeaCache, step-distilled models, FP8/NVFP4,
and xDiT sequence parallelism for multi-GPU single-request inference.

Cross-links:
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference.
- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) — image
  DiTs (Flux, SD3, SDXL).

---

## The class shape (recap)

Same as image diffusion. Class with `setup` / `warmup` /
`@inference.function` / `shutdown`. Models injected via
`@inference(models={...})`. `Resources` declared per class.

```python
from gen_worker import Repo, Resources, RequestContext, inference

hunyuan = Repo("tencent/HunyuanVideo")

@inference(
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=60.0,            # FP8 footprint
        min_compute_capability=9.0,        # FP8 needs Hopper
        peak_vram_per_request_gb=70.0,
    ),
    models={"pipe": hunyuan.flavor("bf16")},
    allowed_shapes=((720, 1280),),
)
class HunyuanVideoGenerate:
    def setup(self, pipe): ...
    def warmup(self): ...
    @inference.function
    def generate(self, ctx, payload): ...
    def shutdown(self): ...
```

The video-specific delta is what goes inside `setup()` and what knobs
on the payload matter.

---

## Why video is different from image

- **No cross-request batching.** A single 5-second 1280×720 video at
  HunyuanVideo's native settings fills a 80GB H100 by itself.
  `batch > 1` is dead on arrival for any big video DiT.
- **TeaCache is the highest-impact knob.** Per-step caching pays for
  itself across the long denoising loop. 1.5-3× speedup on
  HunyuanVideo / Wan / CogVideoX.
- **Step distillation matters more.** A non-distilled 50-step Wan2.2
  run is 5-10× a distilled 8-step LTX-Video run before any other
  acceleration.
- **Quantization is mandatory at scale.** FP8 saves ~10GB on
  HunyuanVideo; NVFP4 on B200 unlocks Wan-class on a single GPU.
- **Multi-GPU per request becomes practical.** xDiT's Unified Sequence
  Parallel hits 3.54× on Mochi across 6×L40. Image diffusion almost
  never needs this; video DiTs at 1080p × 5s do.

---

## The video acceleration stack

Three layers, applied in `setup()`:

1. **`torch.compile`** on the transformer (`pipe.transformer`).
2. **`TeaCache(threshold=0.6)`** — the video DiT cache. Conflicts
   with `batch > 1` (auto-disabled by the SDK).
3. **`fp8(pipe.transformer)`** or **`nvfp4(pipe.transformer)`** for
   weight quantization.

For 28B-class MoE models (Wan2.2): add a fourth layer —
**`SequenceParallel(gpus=N)`** to split the transformer's sequence
dimension across multiple GPUs in one request.

```python
from gen_worker import (
    cache as gw_cache,
    compile as gw_compile,
    parallelism as gw_par,
    quant as gw_quant,
)

def setup(self, pipe):
    # 1. Compile the DiT.
    pipe.transformer = gw_compile.torch_compile(
        pipe.transformer, mode="reduce-overhead"
    )
    # 2. Video-aware cache. AUTO-DISABLES cross-request batching.
    gw_cache.TeaCache(threshold=0.6).apply(pipe)
    # 3. Quantize. FP8 on H100; NVFP4 on B200 (auto-fallback to FP8).
    pipe.transformer = gw_quant.fp8(pipe.transformer)
    # 4. Optional: multi-GPU per-request for Wan-class.
    pipe = gw_par.SequenceParallel(gpus=4).apply(pipe)
    self.pipe = pipe
```

### `TeaCache` — batch>1 conflict

The cache key is invalidated when the engine packs multiple requests
into one forward (nunchaku #597). `TeaCache.breaks_cross_request_batching
= True` and the SDK's micro-batching aggregator reads this flag and
auto-disables itself for any worker using TeaCache.

This is almost never a problem for video — you weren't going to batch
across requests anyway because each request fills the GPU. But: if
you're on a short-forward distilled video model (LTX-fast with <300ms
per forward) you have a choice. TeaCache and micro-batching are
mutually exclusive; pick the one that matches your traffic.

### `SequenceParallel` — when multi-GPU per request makes sense

xDiT's Unified Sequence Parallel (USP) splits the DiT's sequence
dimension across `gpus` devices. Best speedup at large frame counts.
Use it when:

- One request alone fills a GPU and you have more GPUs idle, AND
- Your traffic isn't aggregate-bound (i.e. throughput is already
  saturated by a small number of in-flight requests).

```python
pipe = gw_par.SequenceParallel(gpus=4).apply(pipe)
```

Sub-linear scaling — xDiT reports 3.54× on 6×L40 for Mochi (~59%
efficiency at 6 GPUs). The remaining throughput comes from running
multiple replicas. Use this only for the highest-end DiTs (Wan2.2
28B-MoE, HunyuanVideo at long durations) where one replica per worker
isn't fast enough on a single GPU.

If only one GPU is visible, `SequenceParallel(gpus=N)` is a no-op
(with `gpus=1`) or falls back to passthrough with a warning. Safe to
include unconditionally as long as `gpus` matches the worker's GPU
count.

---

## Per-model recommended config

| Model                  | Compile  | Cache             | Quant         | Sequence parallelism | Notes                                          |
|------------------------|----------|-------------------|---------------|----------------------|------------------------------------------------|
| **LTX-Video distilled (8-step)** | `torch_compile` | (skip — forward too short) | FP8 optional   | no                   | Real-time on H100. ~50% wins from distillation alone. |
| **HunyuanVideo**       | `torch_compile` | `TeaCache(0.6)`   | `fp8` required (saves 10GB) | no | Reference video DiT. Forward >1s per step.       |
| **CogVideoX-5B**       | `torch_compile` | `TeaCache(0.6)`   | `fp8`         | no                   | TeaCache 1.6-2× per arXiv 2411.19108.          |
| **Wan2.2 (28B MoE)**   | `torch_compile` | `TeaCache(0.6)`   | `nvfp4` or `fp8` | YES (4-8 GPUs)    | MoE-class. xDiT mandatory for sub-minute generations. |
| **Mochi-1**            | `torch_compile` | `TeaCache(0.6)`   | `fp8`         | YES (4-6 GPUs)       | The xDiT paper's reference (3.54× on 6×L40).   |

**LTX-Video specifics.** Lightricks ships an 8-step distilled
variant (`lightricks/ltx-video-0.9.7-distilled`). 8 steps × ~50ms per
step on H100 ≈ real-time generation. Compile the DiT, skip TeaCache
(forward is too short for the cache amortization to pay off), and
add FP8 only if you're VRAM-bound.

**HunyuanVideo specifics.** Per Tencent's DeepWiki, FP8 saves ~10GB
relative to bf16. That brings the model from ~70GB resident on a
80GB H100 to ~60GB — leaving headroom for activations and KV-style
state. TeaCache on the 50-step denoise saves another ~40% of compute.

**Wan2.2 specifics.** 28B parameters in a MoE shape. Single-GPU is
slow; production deployments split across 4-8 GPUs via xDiT. NVFP4 on
Blackwell + xDiT(gpus=4) on a 4×B200 box is the canonical setup.

---

## VRAM budgets — 80GB H100 = batch=1 only

For any of the big video DiTs (HunyuanVideo, Wan, CogVideoX-5B,
Mochi), on an 80GB H100 the answer to "how many concurrent requests"
is one. Resources should declare:

```python
resources=Resources(
    accelerator="cuda",
    requires_gpu=True,
    min_vram_gb=60.0,                    # FP8 footprint
    min_compute_capability=9.0,                # FP8 needs Hopper
    peak_vram_per_request_gb=70.0,       # SerialWorker — one request owns it
    vram_must_fit="full_model",
    runtime_scales_with=("num_frames", "num_inference_steps", "height", "width"),
)
```

`peak_vram_per_request_gb` tells the scheduler that two concurrent
requests don't fit on one GPU. SerialWorker enforces it at the
runtime layer too.

For step-distilled short-forward models (LTX-Video-distilled at
8 steps, ~5s on H100), the math can permit `batch > 1` if the
denoiser doesn't blow up the activation footprint. Measure before
opting in. Most production tenants stick with batch=1 even on the
distilled variants and run more replicas instead.

---

## What NOT to do

**Don't enable TeaCache and micro-batching together.** TeaCache's
`breaks_cross_request_batching = True` flag tells the SDK to
auto-disable the aggregator. But if you also explicitly set
`max_concurrent_per_worker > 1`, the SDK can't reconcile the intent.
Pick one. For video, this is almost always "stick with TeaCache".

**Don't compile inside `generate()`.** Same rule as image diffusion.
`torch.compile` is a setup-time operation; calling it on the request
path makes every cold request pay the compile cost.

**Don't claim more GPUs than the worker has.** `SequenceParallel(gpus=8)`
on a 4-GPU worker passes through with a warning and runs single-GPU.
Match `gpus` to the worker's actual visible-device count, or pass it
as a `Resources(required_libraries=("xfuser",))` declaration the
orchestrator can route on.

**Don't ship without `peak_vram_per_request_gb` for big DiTs.** Without
it, the scheduler thinks two requests fit on one GPU and you'll OOM
in production. Set it equal to or slightly above your actual measured
peak.

**Don't return raw frame lists.** Use `ctx.save_file(...)` to save
the encoded MP4 / WebM bytes, or
`ctx.open_output_stream(...)` for chunked uploads. The Asset returned
carries the storage ref.

---

## Complete working example: HunyuanVideo with TeaCache + FP8

```python
"""HunyuanVideo generation with the canonical video stack."""

from __future__ import annotations

import tempfile
from pathlib import Path

import msgspec
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

from gen_worker import (
    Asset,
    Repo,
    RequestContext,
    Resources,
    cache as gw_cache,
    compile as gw_compile,
    inference,
    quant as gw_quant,
)


hunyuan = Repo("tencent/HunyuanVideo")


class GenerateInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 129              # 5.4s at 24fps
    height: int = 720
    width: int = 1280
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: int | None = None
    fps: int = 24


class GenerateOutput(msgspec.Struct):
    video: Asset
    num_frames: int
    fps: int


@inference(
    label="HunyuanVideo (compile + TeaCache + FP8)",
    description=(
        "Tencent HunyuanVideo with the canonical video acceleration "
        "stack: torch.compile + TeaCache + FP8 weight quant. Saves "
        "~10GB vs bf16 on H100."
    ),
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=60.0,                # FP8 footprint
        min_compute_capability=9.0,            # FP8 path needs Hopper
        peak_vram_per_request_gb=70.0,   # one request owns this GPU
        required_libraries=("torch", "diffusers"),
        vram_must_fit="full_model",
        vram_base=2 * 1024 * 1024 * 1024,
        vram_size_multiplier=1.10,
        vram_scales_with=("num_frames", "height", "width"),
        runtime_scales_with=("num_frames", "num_inference_steps", "height", "width"),
    ),
    models={"pipe": hunyuan.flavor("bf16")},
    allowed_shapes=((720, 1280),),
)
class HunyuanVideoGenerate:
    def setup(self, pipe: HunyuanVideoPipeline) -> None:
        # 1. Compile the DiT.
        pipe.transformer = gw_compile.torch_compile(
            pipe.transformer, mode="reduce-overhead"
        )
        # 2. Video DiT cache. Auto-disables cross-request batching.
        gw_cache.TeaCache(threshold=0.6).apply(pipe)
        # 3. FP8 weight quant — saves ~10GB on Hopper.
        pipe.transformer = gw_quant.fp8(pipe.transformer)
        self.pipe = pipe

    def warmup(self) -> None:
        # Short warmup: 16 frames, 2 steps, declared resolution. Just
        # enough to capture compile graphs and prime TeaCache.
        _ = self.pipe(
            prompt="warmup",
            num_frames=16,
            num_inference_steps=2,
            height=720,
            width=1280,
        )

    @inference.function(
        timeout_ms=600_000,    # 10 min — videos take a while
        description="Generate a video from a text prompt (HunyuanVideo).",
    )
    def generate(
        self,
        ctx: RequestContext,
        payload: GenerateInput,
    ) -> GenerateOutput:
        import torch

        ctx.raise_if_canceled()

        # Local state only — never assign to self inside generate().
        num_frames = max(1, min(257, int(payload.num_frames)))
        steps = max(1, min(100, int(payload.num_inference_steps)))
        gen = None
        if payload.seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(payload.seed))

        ctx.progress(0.05, stage="denoise")
        result = self.pipe(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt or None,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=payload.guidance_scale,
            height=payload.height,
            width=payload.width,
            generator=gen,
        )
        ctx.raise_if_canceled()

        # diffusers returns a list of PIL frames; encode to MP4 then save.
        ctx.progress(0.92, stage="encode")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4_path = Path(f.name)
        try:
            export_to_video(result.frames[0], str(mp4_path), fps=payload.fps)
            video_asset = ctx.save_file(
                f"jobs/{ctx.request_id}/outputs/video.mp4",
                str(mp4_path),
            )
        finally:
            try:
                mp4_path.unlink()
            except OSError:
                pass

        ctx.progress(1.0, stage="done")
        return GenerateOutput(
            video=video_asset,
            num_frames=num_frames,
            fps=payload.fps,
        )

    def shutdown(self) -> None:
        self.pipe = None
```

What this gives you on H100 SXM:
- FP8 brings the model resident size from ~70GB (bf16) to ~60GB.
- TeaCache cuts the 50-step denoise compute by ~40%.
- torch.compile + CUDA graphs amortize the launch overhead.
- Total: ~2-3× over the bf16 baseline.

What this gives you on B200:
- FP8 → NVFP4 path further reduces the resident footprint.
- Stacked speedup ~3-4× over the H100 FP8 numbers.

---

## Next steps

- **Multi-GPU per request** (Wan2.2-class, Mochi at 1080p×5s)? Add
  `SequenceParallel(gpus=N).apply(pipe)` after the other accelerators.
  Declare matching `Resources(required_libraries=("xfuser",))` so the
  orchestrator only places this endpoint on multi-GPU workers.
- **Step-distilled models** (LTX-Video distilled, Wan2.2-fast)? Skip
  TeaCache (forward too short to amortize) and rely on compile + FP8.
  Optionally opt into micro-batching if the forward fits under 300ms.
- **Image-to-video / video-to-video pipelines**? Inject the
  conditioning model as a second binding in `models={}` and consume it
  as a second `setup` kwarg. See the multi-param injection section in
  [endpoint-authoring.md](endpoint-authoring.md#multi-param-injection--each-binding-is-independent).
