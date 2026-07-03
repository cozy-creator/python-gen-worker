# Cookbook: BatchedWorker LLM Endpoints

One page on shipping LLM-class endpoints — multimodal captioners,
chat / instruction models, autoregressive audio models — with the
`@batched_inference` class shape (#273). Continuous batching, typed
streaming signals, cooperative cancellation.

Audience: you've shipped a SerialWorker diffusion endpoint and want
to deploy an LLM-class workload that shares the GPU across many
concurrent requests.

Cross-links:
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference.
- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) — image
  DiTs (SerialWorker contrast).
- [cookbook-audio.md](cookbook-audio.md) — autoregressive audio.

---

## SerialWorker vs BatchedWorker — pick the right archetype

There's exactly one rule:

> **Is the workload many requests sharing each forward pass through a
> long-lived autoregressive engine?**
> - **Yes** → BatchedWorker. `@batched_inference`, async generator
>   methods, continuous batching / KV-cache reuse / token streaming.
> - **No** → SerialWorker. `@inference`, sync methods, one request
>   fully owns the GPU until done.

| Family                              | Examples                               | Archetype       |
|-------------------------------------|----------------------------------------|-----------------|
| Image / video / 3D diffusion        | Flux.2, HunyuanVideo, TRELLIS.2        | SerialWorker    |
| Flow-matching audio                 | Stable Audio Open, small TTS DiTs      | SerialWorker    |
| LLM (text)                          | Llama-class chat / instruct            | BatchedWorker   |
| Multimodal LLM                      | JoyCaption, Qwen2.5-VL                 | BatchedWorker   |
| Autoregressive audio                | Chatterbox, GPT-SoVITS, MusicGen       | BatchedWorker   |

BatchedWorker is the right archetype when your engine can interleave
multiple requests through a single forward step (vLLM continuous
batching, SGLang RadixAttention prefix-cache reuse). SerialWorker is
the right archetype when the per-request working set saturates the GPU
on its own.

---

## The class shape

```python
from typing import AsyncIterator

from gen_worker import (
    Done,
    Error,
    IncrementalTokenDelta,
    Repo,
    RequestContext,
    Resources,
    batched_inference,
)

@batched_inference(
    models={"llm": Repo("org/llama-3b").flavor("bf16")},
    resources=Resources(
        accelerator="cuda",
        min_vram_gb=24.0,
        min_compute_capability=8.0,
    ),
)
class MyChat:
    def setup(self, llm):
        """Construct the engine. Required — tenant owns engine choice."""
        ...

    def warmup(self):
        """Optional. Cheap warmup forward to populate KV / compile caches."""
        ...

    @batched_inference.function
    async def chat(self, ctx: RequestContext, payload) -> AsyncIterator:
        """Async generator. Yields IncrementalTokenDelta / Done / Error."""
        ...

    def shutdown(self):
        """Required. Tear down the engine."""
        ...
```

Five contract points:

| Hook                              | When called                              | Required? |
|-----------------------------------|------------------------------------------|-----------|
| `__init__(self)`                  | Worker process boot                      | optional  |
| `setup(self, **models)`           | After model bindings resolve             | yes       |
| `warmup(self)`                    | After setup, before traffic              | optional  |
| `@batched_inference.function`     | Each request                             | yes (≥1)  |
| `shutdown(self)`                  | Worker drain or termination              | yes       |

Differences from `@inference`:

- **Async-generator method required.** The `@batched_inference.function`
  method MUST satisfy `inspect.isasyncgenfunction(method)` — an
  `async def` with at least one `yield`. Plain coroutines (no yield)
  and sync methods are rejected at decoration time with a clear error.
- **Yields typed signals.** The method yields
  `IncrementalTokenDelta(text=...)` per delta, `Done()` at clean
  end, `Error(message=...)` on engine failure. The wire dispatcher
  maps these onto the existing `IncrementalTokenDelta` /
  `IncrementalTokenStreamDone` / `IncrementalTokenStreamError` proto
  messages.
- **Tenant owns the engine.** The SDK does NOT pick or construct the
  engine for you. Build `AsyncLLMEngine.from_engine_args(...)` (or
  `sgl.Engine(...)`) inside `setup()` and store it on `self`. The
  wire shape only requires that you stream typed signals back through
  the async generator.

---

## vLLM example — JoyCaption-shaped multimodal LLM

JoyCaption (`fancyfeast/joy-caption-alpha-two`) is a multimodal
captioner: a SigLIP image tower plus a Llama-class decoder. Same
shape as text-only LLMs, with the image-tower features prepended to
the prompt.

```python
"""JoyCaption multimodal LLM via vLLM AsyncLLMEngine."""

from __future__ import annotations

from typing import AsyncIterator

import msgspec

from gen_worker import (
    Asset,
    Done,
    Error,
    IncrementalTokenDelta,
    Repo,
    RequestContext,
    Resources,
    batched_inference,
)


joy = Repo("fancyfeast/joy-caption-alpha-two")


class CaptionInput(msgspec.Struct):
    image: Asset
    system_prompt: str = (
        "You are a precise image captioner. Describe the image in "
        "detail without speculation."
    )
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9


@batched_inference(
    label="JoyCaption (vLLM continuous batching)",
    description=(
        "JoyCaption multimodal captioner via vLLM continuous batching. "
        "SigLIP image tower + Llama-class decoder. Many concurrent "
        "captioning requests share each forward pass through the "
        "engine."
    ),
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=24.0,
        min_compute_capability=8.0,
        # BatchedWorker: per-request peak is small relative to the
        # static engine footprint. The engine manages KV cache + scratch
        # from its shared pool.
        peak_vram_per_request_gb=2.0,
    ),
    models={"joy": joy.flavor("bf16")},
)
class JoyCaptionGenerate:
    def setup(self, joy) -> None:
        """Construct the vLLM AsyncLLMEngine. The tenant owns engine
        choice — the SDK does not pick or wrap the engine."""
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=joy.local_path,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            # Multimodal models need the image tower wired in. vLLM
            # auto-detects the architecture from the model config.
            limit_mm_per_prompt={"image": 1},
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @batched_inference.function(
        timeout_ms=120_000,
        description="Caption one image with a system-prompt-anchored decoder.",
    )
    async def caption(
        self,
        ctx: RequestContext,
        payload: CaptionInput,
    ) -> AsyncIterator:
        from vllm import SamplingParams

        # The wire-side request id surfaces here. vLLM uses it to
        # disambiguate concurrent generations in its scheduler — and we
        # need it for the abort() call on cancellation.
        request_id = str(ctx.request_id)

        sampling_params = SamplingParams(
            max_tokens=int(payload.max_tokens),
            temperature=float(payload.temperature),
            top_p=float(payload.top_p),
        )

        # Build the multimodal prompt. vLLM accepts a dict with
        # 'prompt' (system + user text) and 'multi_modal_data' (the
        # image bytes / PIL image).
        prompt = {
            "prompt": (
                f"<|system|>{payload.system_prompt}<|end|>"
                f"<|user|><image><|end|><|assistant|>"
            ),
            "multi_modal_data": {"image": _load_image(payload.image)},
        }

        last_text = ""
        try:
            async for output in self.engine.generate(
                prompt, sampling_params, request_id=request_id,
            ):
                # Cooperative cancellation: client disconnected, or the
                # caller explicitly canceled. Tell vLLM to release the
                # slot — otherwise the engine holds the request open
                # until natural completion.
                if ctx.cancelled():
                    await self.engine.abort(request_id)
                    return

                full_text = output.outputs[0].text
                delta = full_text[len(last_text):]
                last_text = full_text
                if delta:
                    yield IncrementalTokenDelta(text=delta)

        except Exception as e:
            yield Error(message=f"engine error: {e}")
            return

        yield Done()

    def shutdown(self) -> None:
        # vLLM AsyncLLMEngine doesn't expose a public shutdown(); drop
        # the reference and let GC + the process exit clean up. The SDK
        # gives shutdown() up to 30s before forcing a process kill.
        self.engine = None


def _load_image(image_asset):
    """Load the image into the shape vLLM expects."""
    from PIL import Image
    return Image.open(image_asset.local_path).convert("RGB")
```

What this gives you:

- One engine, many concurrent requests. vLLM's continuous batching
  packs multiple in-flight generations into each forward step.
- KV-cache reuse across requests that share a system prompt (vLLM's
  automatic prefix caching).
- Streaming deltas — clients see tokens as they're generated.
- Cooperative cancellation — `ctx.cancelled()` triggers
  `engine.abort()`, releasing the engine slot.

---

## SGLang note

SGLang's `sgl.Engine` works analogously: construct the engine in
`setup()`, iterate its streaming API in the function method, call its
abort path on cancellation.

- Pick **SGLang** when you need RadixAttention's prefix cache to land
  high prefix-hit-rate workloads. JoyCaption is the canonical case —
  a long system prompt plus image-tower features that repeat across
  every request. RadixAttention shares the prefix KV across requests
  at the tree level; vLLM's automatic prefix caching is hash-based
  and less effective when the shared prefix has small per-request
  variations.
- Pick **vLLM** when you need the broadest model coverage (most new
  architectures land on vLLM first) or when the workload is
  dominated by short suffixes with little prefix overlap.

For JoyCaption specifically, both engines work; SGLang's prefix cache
gives a meaningful throughput win at production scale. For chat /
instruct models with per-user system prompts, vLLM's coverage is
usually the deciding factor.

---

## Cancellation pattern

Stream-EOF (the client disconnects mid-generation) flips
`ctx.cancelled()` to `True`. The async-generator loop should check it
between deltas and call the engine's abort path so the engine slot is
released immediately rather than at natural EOS:

```python
async for output in self.engine.generate(prompt, params, request_id=request_id):
    if ctx.cancelled():
        await self.engine.abort(request_id)
        return
    delta = _extract_delta(output)
    if delta:
        yield IncrementalTokenDelta(text=delta)
yield Done()
```

The cancellation check is cheap — `ctx.cancelled()` reads a local
atomic. Place it once per loop iteration, immediately after the
engine yields. Two failure modes the pattern guards against:

1. **Slow generations holding engine slots.** Without `engine.abort`,
   a canceled long-form generation continues to occupy a KV-cache
   slot until natural EOS. With abort, the slot is freed within the
   next vLLM scheduler step.
2. **Token leakage to a disconnected client.** Yielding deltas after
   cancellation is a no-op on the wire (the SDK drops them), but the
   work is still being done. Early-exit on cancellation reclaims the
   compute too.

---

## Yielding signals

Three signal types, all importable from `gen_worker`:

| Signal                          | When to yield                                       |
|---------------------------------|-----------------------------------------------------|
| `IncrementalTokenDelta(text=…)` | Per delta. `text` is the new substring since last delta. |
| `Done()`                        | Exactly once at clean end. Terminates the stream.   |
| `Error(message=…)`              | Terminal error. Prefer raising a typed `gen_worker` exception when you have one; `Error()` is the inline fallback. |

The wire dispatcher checks `isinstance(item, _SIGNAL_TYPES)` first and
falls through to a legacy duck-typed path otherwise, so endpoints that
need richer per-delta payloads (image bytes, structured tool calls,
audio chunks with metadata) can still define their own
`msgspec.Struct` shapes. For the LLM-text case, the three signals
above are enough.

Stream lifecycle:

1. `yield IncrementalTokenDelta(text="Hello")` → wire emits one
   `IncrementalTokenDelta` proto.
2. `yield IncrementalTokenDelta(text=" world")` → wire emits another.
3. `yield Done()` → wire emits `IncrementalTokenStreamDone` plus the
   terminal `JobExecutionResult(success=True)`.

If the generator raises an unhandled exception, the dispatcher catches
it, emits an `IncrementalTokenStreamError` carrying the exception's
str, and the terminal result is `success=False`. Yielding `Error()`
explicitly produces the same wire shape — pick the explicit form when
the engine returns a soft error you want to report inline rather than
unwind.

**Stream-EOF (client disconnect) → SDK auto-cancels.** Your loop sees
`ctx.cancelled() == True` and exits early per the cancellation
pattern above.

---

## AR TTS extension (#327)

Autoregressive TTS (Chatterbox, GPT-SoVITS) lives on the same
BatchedWorker shape as JoyCaption — the only difference is what the
deltas carry. Instead of text, you stream audio-token deltas through
`IncrementalTokenDelta` (the tokens are opaque to the SDK; the wire
treats them as bytes that the client decodes through the model's
codec).

Reference: [randombk/chatterbox-vllm](https://github.com/randombk/chatterbox-vllm)
is a port of Resemble AI's Chatterbox onto vLLM. Same engine
construction in `setup()`, same async-generator loop, same
cancellation contract.

```python
@batched_inference(
    label="Chatterbox AR-TTS (BatchedWorker)",
    models={"chatterbox": Repo("ResembleAI/chatterbox").flavor("bf16")},
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=16.0,
        min_compute_capability=8.0,
        peak_vram_per_request_gb=2.0,
    ),
)
class ChatterboxGenerate:
    def setup(self, chatterbox):
        # Tenant owns the engine. chatterbox-vllm exposes an
        # AsyncLLMEngine-shaped interface; alternate engines (raw
        # transformers, GPT-SoVITS' built-in stack) wire the same way.
        from chatterbox_vllm import AsyncChatterboxEngine
        self.engine = AsyncChatterboxEngine.from_pretrained(
            chatterbox.local_path,
            dtype="bfloat16",
        )

    @batched_inference.function(timeout_ms=120_000)
    async def synthesize(self, ctx, payload):
        import base64

        request_id = str(ctx.request_id)
        try:
            async for audio_chunk in self.engine.generate(
                text=payload.text,
                voice_prompt=payload.voice_prompt_audio_ref,
                request_id=request_id,
            ):
                if ctx.cancelled():
                    await self.engine.abort(request_id)
                    return

                # Audio bytes ride inside IncrementalTokenDelta.text as
                # base64 for now — same envelope as text streaming.
                yield IncrementalTokenDelta(
                    text=base64.b64encode(audio_chunk.bytes).decode("ascii"),
                )

        except Exception as e:
            yield Error(message=f"chatterbox engine error: {e}")
            return

        yield Done()

    def shutdown(self):
        self.engine = None
```

Same cancellation semantics — client disconnect flips `ctx.cancelled()`,
the loop calls `engine.abort()`, the engine releases the slot.

The audio-token-as-`text` envelope is a transitional shape: production
audio endpoints carry richer per-chunk metadata (sample rate, item
index, processed count) by defining their own delta struct. See
[cookbook-audio.md](cookbook-audio.md) for the established AR audio
pattern that uses a custom `AudioDelta` struct alongside (or in place
of) `IncrementalTokenDelta`. The signal type to reach for first is
`IncrementalTokenDelta`; drop down to a custom struct when the
metadata is necessary for the client decoder.

---

## What the SDK does NOT do for you

The SDK foundation (#273) is the wire-side surface — class detection,
async-generator validation, signal-type plumbing, dispatch routing,
ctx-cancellation. The SDK does **not** ship an engine integration:

- **No vLLM auto-construction.** You construct
  `AsyncLLMEngine.from_engine_args(...)` (or `sgl.Engine(...)`) inside
  your `setup()`. This is deliberate — engine version pinning,
  precision, tensor-parallel size, gpu_memory_utilization, and the
  myriad other engine knobs are per-endpoint decisions the SDK has no
  right to make for you.
- **No engine-side prefix-cache configuration.** vLLM's automatic
  prefix caching and SGLang's RadixAttention are configured via the
  engine's own constructor kwargs. The SDK is opaque to those choices.
- **No multimodal preprocessing helpers.** JoyCaption needs image
  loading + SigLIP-shaped feature feeding; that's a tenant-owned
  helper inside `setup` / `caption`.
- **No request batching across engines.** One worker process hosts
  one engine. Scale-out across more requests = scale-out across more
  worker replicas (each running its own engine).

The engine-integration story (`runtime="vllm"` auto-wiring with
opinionated defaults) is a follow-up; today's shape gives you the
class boundary, the signal types, and the cancellation hook, and
lets you wire the engine however the production endpoint needs.
