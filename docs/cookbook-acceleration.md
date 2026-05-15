# Cookbook: `gen_worker.accel` — Diffusion Acceleration

One page on the canonical five-call surface for accelerating
SerialWorker diffusion endpoints. Put these in your `setup()` so the
per-request fast path stays clean.

Cross-links:
- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) — the
  acceleration stack in context (Flux.2-klein-4B end-to-end).
- [cookbook-video-diffusion.md](cookbook-video-diffusion.md) — video
  DiTs (same acceleration story, larger working sets).
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference.

---

## TL;DR

```python
from gen_worker import accel, inference, Repo, Resources

flux = Repo("black-forest-labs/flux.2-klein-4b-base")

@inference(
    models={"pipe": flux.flavor("bf16")},
    resources=Resources(
        accelerator="cuda",
        min_vram_gb=12.0,
        min_compute_capability=8.0,
    ),
    allowed_shapes=((1024, 1024),),
)
class FluxKleinGenerate:
    def setup(self, pipe):
        caps = accel.gpu_capability()

        # 1. Compile the heavy DiT.
        pipe.transformer = accel.compile_diffusion(
            pipe.transformer, mode="reduce-overhead",
        )
        # 2. ParaAttention First-Block Cache (~1.5-2x on Flux/SD3/Qwen-Image).
        accel.apply_fbcache(pipe, residual_diff_threshold=0.12)
        # 3. ParaAttention general adapter (no-op on single-GPU workers).
        accel.apply_para_attn(pipe)
        # 4. NVFP4 weight quant (Blackwell only; no-op + warn elsewhere).
        if caps.has_nvfp4:
            pipe.transformer = accel.apply_nvfp4(pipe.transformer)

        self.pipe = pipe
    ...
```

Five entry points cover the canonical stack:

| Call                         | Purpose                                   | Safe-on-CPU? |
|------------------------------|-------------------------------------------|--------------|
| `accel.gpu_capability()`     | Cached hardware probe                     | yes          |
| `accel.compile_diffusion(m)` | `torch.compile` for the heavy DiT module  | yes (no-op)  |
| `accel.apply_fbcache(pipe)`  | ParaAttention First-Block Cache           | raises*      |
| `accel.apply_para_attn(pipe)`| ParaAttention general (sequence-parallel) | raises*      |
| `accel.apply_nvfp4(model)`   | NVFP4 weight quantization                 | yes (no-op)  |

`*` These two require `pip install para-attention` in the endpoint
image. The other three either operate without a third-party dep or
gracefully no-op when the hardware isn't a match.

The lower-level modules — `gen_worker.cache`, `gen_worker.compile_helpers`,
`gen_worker.quant`, `gen_worker.parallelism` — remain available for
advanced cases (multiple cache backends, multi-precision quant
fallbacks, sequence parallelism with custom placement). For new
endpoints, reach for `accel.*` first.

---

## `accel.gpu_capability()` — probe once, branch cleanly

```python
caps = accel.gpu_capability()
# caps.arch:                 'blackwell' | 'hopper' | 'ampere' |
#                            'lovelace' | 'turing'  | 'unknown' |
#                            'none'  (no CUDA visible)
# caps.compute_capability:   '10.0' on B200, '9.0' on H100, ''  on CPU
# caps.device_name:          'NVIDIA H100 80GB HBM3' (or similar)
# caps.vram_gb_total:        80.0
# caps.gpu_count:            1
# caps.has_fp8:              True on Hopper+ (SM 9.0+)
# caps.has_nvfp4:            True on Blackwell+ (SM 10.0+)
# caps.torch_version:        '2.11.0+cu128' (empty if torch missing)
```

Probing happens once at first call; the result is cached for the
lifetime of the worker process. Repeated reads in `setup()` are free.

**When to gate on `caps.arch == 'blackwell'`** — NVFP4 weight
quantization needs Blackwell-class tensor cores (SM 10.0+, B100 /
B200 / B300 / RTX 50-series). Gating keeps the call out of the
non-Blackwell path entirely — `accel.apply_nvfp4` will no-op + warn on
non-Blackwell hosts, but gating in `setup()` makes the deployment
matrix explicit:

```python
if caps.has_nvfp4:
    pipe.transformer = accel.apply_nvfp4(pipe.transformer)
elif caps.has_fp8:
    # Fall back to FP8 via the lower-level gen_worker.quant module.
    pipe.transformer = gen_worker.quant.fp8(pipe.transformer)
```

**When to gate on `caps.has_fp8`** — Hopper-class FP8 tensor cores
(H100 / H200 / H800 / 4090 / L40). The five-call surface doesn't ship
an `apply_fp8` helper — it's a frequent enough pattern that the
fallback story is per-endpoint, so reach for `gen_worker.quant.fp8`
directly when you need it.

**On CPU-only hosts** (`caps.arch == 'none'`) — every helper either
no-ops (the safe-on-CPU calls in the table above) or raises with an
install hint (the two `para_attn` helpers). For a true CPU-only
endpoint, declare `Resources(accelerator='none')` and skip the
`accel.*` calls entirely.

---

## `accel.compile_diffusion(model, *, mode='reduce-overhead')`

Thin wrapper around `torch.compile` for the DiT module:

```python
pipe.transformer = accel.compile_diffusion(
    pipe.transformer, mode="reduce-overhead",
)
```

`mode="reduce-overhead"` is the recommended default for inference —
Inductor codegen plus CUDA graph capture. `mode="max-autotune"` trades
warmup time for steady-state throughput; pick it when your endpoint
can afford long warmup and benefits from the deeper autotune.

The wrapper is **safe to call unconditionally** in setup():

- No CUDA visible? Returns the module unchanged + stderr warning.
- torch < 2.5? Returns the module unchanged + stderr warning.
- torch missing entirely? Returns the module unchanged + stderr warning.

Compilation is lazy — the first forward pass at a new shape pays the
compile cost. Pre-warm in `warmup()`:

```python
def warmup(self):
    _ = self.pipe(prompt="warmup", num_inference_steps=2,
                  width=1024, height=1024)
```

The SDK sets `TORCHINDUCTOR_CACHE_DIR` before `setup()` runs so compile
artifacts persist across worker restarts (#322).

---

## `accel.apply_fbcache(pipe, *, residual_diff_threshold=0.12)`

ParaAttention's First-Block Cache skips redundant DiT block evaluations
across denoising steps when the first block's residual is unchanged
within `residual_diff_threshold`. ParaAttention's published benchmarks
show ~1.5-2× on Flux and similar gains on SD3 / Qwen-Image. The
wrapper patches the pipeline in-place and also returns it for
chaining.

```python
accel.apply_fbcache(pipe, residual_diff_threshold=0.12)
```

`residual_diff_threshold=0.12` is the ParaAttention-recommended
default. Lower thresholds invalidate more aggressively (higher
fidelity, smaller speedup); higher thresholds invalidate less often
(more speedup, more quality drift).

**Stacks multiplicatively with `compile_diffusion`** — both layers
operate on different axes (compile = lower per-step cost, FBCache =
fewer steps in effect).

**Does NOT stack with `TeaCache`** — TeaCache wraps the same axis with
a different cache key shape. Pick FBCache for Flux / SD3 / Qwen-Image
single-request; reach into `gen_worker.cache.TeaCache` only when you
need its specific video-DiT properties.

**Requires `pip install para-attention`** in the endpoint image. The
import is lazy — `from gen_worker import accel` is free; the install
is only needed when `apply_fbcache` is called.

---

## `accel.apply_para_attn(pipe)`

ParaAttention's general adapter rewrites the pipeline's attention
layers for **sequence-parallel inference** — splitting the sequence
dimension across multiple GPUs to accelerate a single request.

```python
accel.apply_para_attn(pipe)
```

- On multi-GPU workers: parallelism degree = visible CUDA device count;
  expect a per-request speedup roughly proportional to the device
  count on large DiTs.
- On single-GPU workers: parallelism degree = 1; the adapter installs
  cleanly but the speedup is zero. **Safe to call unconditionally.**

The wrapper tries the modern ParaAttention adapter entry point first
(`para_attn.diffusers_adapters.apply_adapter_on_pipe`) and falls back
to the legacy `apply_para_attn_on_pipe` shape so already-pinned
endpoint images don't break when they upgrade gen-worker.

Distinct from `apply_fbcache`: FBCache is a per-request cross-timestep
cache (same GPU, multiple steps); `apply_para_attn` is multi-GPU
sequence parallelism (single step, multiple GPUs). They stack — call
both in `setup()`.

**Requires `pip install para-attention`** in the endpoint image.

---

## `accel.apply_nvfp4(model)`

NVIDIA's 4-bit floating-point weight quantization. Per NVIDIA's
measurements, ~3-6× on Flux.2 at B200. The tensor-core path requires
Blackwell-class GPUs (SM 10.0+, B100 / B200 / B300 / RTX 50-series).

```python
pipe.transformer = accel.apply_nvfp4(pipe.transformer)
```

**Safe to call unconditionally.** On non-Blackwell hosts the helper
logs a warning and returns the module unchanged — leave the call in
`setup()` and the same endpoint deploys cleanly on H100 / A100 / older
without branching.

Calibration is weight-only — no calibration dataset required at
`setup()` time. The wrapper resolves the NVFP4 config name across
modelopt releases (`NVFP4_DEFAULT_CFG` / `NVFP4_KV_CFG` /
`CONFIG_CHOICES['NVFP4_DEFAULT_CFG']`) so endpoints stay portable
across modelopt versions.

**Requires `pip install nvidia-modelopt`** in the endpoint image when
the host is Blackwell. Non-Blackwell hosts skip the import entirely
(no ImportError, just a warning).

---

## Canonical Flux.2-klein-4B example

Complete `setup()` for the canonical reference deployment, gating the
NVFP4 path on capability so the same endpoint runs on B200 / H100 /
A100 without branching at the call site:

```python
@inference(
    label="Flux.2-klein-4B (accel five-call stack)",
    models={"pipe": flux.flavor("bf16")},
    resources=Resources(
        accelerator="cuda",
        min_vram_gb=12.0,
        min_compute_capability=8.0,
        peak_vram_per_request_gb=12.0,
        vram_must_fit="full_model",
    ),
    allowed_shapes=((1024, 1024),),
)
class FluxKleinGenerate:
    def setup(self, pipe):
        caps = accel.gpu_capability()

        # 1. Compile the heavy DiT — works on every CUDA-capable arch.
        pipe.transformer = accel.compile_diffusion(
            pipe.transformer, mode="reduce-overhead",
        )

        # 2. ParaAttention First-Block Cache (~1.5-2x).
        accel.apply_fbcache(pipe, residual_diff_threshold=0.12)

        # 3. Multi-GPU sequence parallelism — no-op on single-GPU workers.
        accel.apply_para_attn(pipe)

        # 4. NVFP4 weight quant — only on Blackwell.
        if caps.has_nvfp4:
            pipe.transformer = accel.apply_nvfp4(pipe.transformer)

        self.pipe = pipe

    def warmup(self):
        # Trigger torch.compile + FBCache state population at the
        # declared shape so the first request sees the warm path.
        _ = self.pipe(
            prompt="warmup",
            num_inference_steps=4,
            width=1024,
            height=1024,
        )

    @inference.function(timeout_ms=60_000)
    def generate(self, ctx, payload):
        ...

    def shutdown(self):
        self.pipe = None
```

What this gives you on B200:
- Compile + FBCache + NVFP4 stacked: ~8-10× over the bf16 baseline.
- Cold start: ~30-60s for model load + compile + quant calibration
  (cached on subsequent boots via `TORCHINDUCTOR_CACHE_DIR` and the
  modelopt cache).
- Steady-state TTFT for 1024² 4-step: <2s warm.

What this gives you on H100:
- Compile + FBCache stacked, NVFP4 skipped (warning logged, module
  unchanged). For FP8 weight quant on Hopper, reach into the lower-
  level `gen_worker.quant.fp8` directly — the `accel` five-call surface
  intentionally keeps the most-common Blackwell path narrow.

What this gives you on a CPU-only host:
- The endpoint shouldn't be deployed there in the first place
  (`Resources(accelerator='cuda')` will gate placement away from CPU
  pools at the orchestrator). Locally, the safe-on-CPU helpers
  (`compile_diffusion`, `apply_nvfp4`) no-op + warn; the two
  ParaAttention helpers raise `ImportError` with the install hint.

---

## When to drop down to the lower-level modules

The five-call surface covers the canonical Apache-2.0 stack. Reach for
the lower-level modules when you need:

- **Multiple cache backends per endpoint** — e.g. swapping between
  FBCache and TeaCache by payload field. Use `gen_worker.cache.*`
  directly.
- **Multi-precision quant fallbacks** — e.g. NVFP4 on Blackwell, FP8
  on Hopper, INT8 elsewhere. Use `gen_worker.quant.*` with each
  precision's helper.
- **Sequence parallelism with custom placement** — e.g. xDiT's USP
  with a non-default world size or device map. Use
  `gen_worker.parallelism.*`.
- **TensorRT / Nexfort / OneDiff** — alternate compile backends
  outside the `torch.compile` line. Use `gen_worker.compile_helpers.*`.

These modules pre-date `gen_worker.accel` and carry richer parameter
spaces. They are NOT deprecated — they remain the right tool for the
advanced-case endpoints that need them. For new endpoints, prefer
`accel.*` and drop down only when you hit a concrete need.

---

## Compatibility notes

- `accel.gpu_capability()` reads from `torch.cuda` if available; on
  CPU-only hosts it returns the `arch='none'` report without raising.
  Decoration + discovery for `Resources(accelerator='none')` endpoints
  do not touch `torch.cuda` (see `tests/test_no_accelerator.py`).
- `accel.compile_diffusion` silently no-ops on torch < 2.5 with a
  stderr warning. The minimum torch version for production-quality
  compile of diffusion DiTs is 2.5; the default endpoint image pins
  torch >= 2.11.
- Both ParaAttention helpers require an opt-in `pip install
  para-attention` in the endpoint image's `pyproject.toml`. The wheel
  does NOT pull `para_attn` as a hard dependency.
- The NVFP4 helper requires `pip install nvidia-modelopt` only on
  Blackwell hosts. Non-Blackwell hosts skip the import (no
  ImportError; warning instead).
