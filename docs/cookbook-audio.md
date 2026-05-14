# Cookbook: Audio Endpoints

One page covering both audio archetypes: **flow-matching / diffusion**
audio models (F5-TTS, Kokoro, Stable Audio Open) shipped as
SerialWorker, and **autoregressive** audio models (Chatterbox,
GPT-SoVITS, Bark, MusicGen) shipped as BatchedWorker with vLLM
continuous batching.

Audience: you know Python + PyTorch and want to publish a TTS or
audio-generation endpoint.

Cross-links:
- [endpoint-authoring.md](endpoint-authoring.md) — base SDK reference.
- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) — image DiTs.
- [cookbook-video-diffusion.md](cookbook-video-diffusion.md) — video DiTs.
- [cookbook-stages.md](cookbook-stages.md) — `@inference.stage`.

---

## The decision: SerialWorker or BatchedWorker?

There's exactly one rule:

> **Is the model autoregressive (a Llama-class decoder emitting
> tokens)?**
> - **Yes** → BatchedWorker. Async class, `runtime="vllm"` (or
>   `"sglang"`), continuous batching shares the GPU across requests.
> - **No** → SerialWorker. Sync class, one request fully owns the
>   GPU until done.

The axis isn't "audio vs text" or "small vs large" — it's the
parallelization model the architecture demands. AR decoders pack
across requests via KV-cache continuous batching; flow-matching and
diffusion models fill the GPU with one request's worth of work.

| Family                              | Examples                               | Archetype       | Engine        |
|-------------------------------------|----------------------------------------|-----------------|---------------|
| Flow-matching DiT                   | F5-TTS, MaskGCT                        | SerialWorker    | direct PyTorch |
| Small flow-matching (CPU-capable)   | Kokoro-82M                             | SerialWorker    | direct PyTorch |
| Latent diffusion (audio)            | Stable Audio Open, MusicLDM            | SerialWorker    | direct PyTorch |
| AR Llama-class decoder              | Chatterbox, GPT-SoVITS, Bark, MusicGen | BatchedWorker   | vLLM          |

When in doubt, ask: "does the forward pass emit one token at a time?"
If yes, you're looking at AR. If no, SerialWorker.

---

## Part 1: Flow-matching audio (SerialWorker)

The class shape is identical to image and video diffusion. `setup` /
optional `warmup` / `@inference.function` (sync, not async) /
`shutdown`. Compile the heavy module in `setup()`. One request per
GPU at a time.

### What goes in `setup()`

```python
import torch
from gen_worker import compile as gw_compile

def setup(self, dit, vocoder):
    self.dit = dit
    self.vocoder = vocoder
    # Compile the flow-matching DiT. mode="max-autotune" trades
    # startup time for steady-state throughput; right call for audio
    # where the steady-state RTF is the metric tenants care about.
    self.dit = gw_compile.torch_compile(self.dit, mode="max-autotune")
```

Optional: compile the vocoder too. Most vocoders (BigVGAN, HiFi-GAN,
the F5-TTS native vocoder) compile cleanly and the win is ~10-20%.

### RTF — the metric that matters

For speech: **Real-Time Factor** = `compute_seconds / audio_seconds`.
RTF < 1.0 means the model can synthesize faster than real-time.

| Model                  | Reference RTF (H100)        | Notes                                             |
|------------------------|------------------------------|---------------------------------------------------|
| F5-TTS                 | 0.15 (≈10s speech in 1.5s)   | Per F5-TTS-ONNX measurements (arXiv 2410.06885)   |
| Kokoro-82M             | ~0.5 on CPU, <0.1 on GPU     | Small enough to deploy on CPU workers              |
| Stable Audio Open 1.0  | ~47s for stereo audio        | 20 steps/s on H100 — measure as `samples/second`   |

How to measure inside your endpoint, for telemetry:

```python
import time

t0 = time.time()
audio_samples = self.synthesize(text, ...)
compute_s = time.time() - t0

duration_s = len(audio_samples) / sample_rate
rtf = compute_s / duration_s if duration_s > 0 else float("inf")

ctx.emit("synthesis_metrics", {
    "compute_seconds": compute_s,
    "audio_seconds": duration_s,
    "rtf": rtf,
})
```

Emit RTF on every request and the SDK's metrics pipeline will track
it as a per-endpoint dimension. Use it to validate compile / quant
gains and to spot regressions across worker hardware.

### Cross-request batching: usually no

The 300ms rule from image diffusion still applies: cross-request
micro-batching pays off only when the forward is short enough that
the SDK's admission window (~50ms) doesn't dominate. For F5-TTS at
~1.5s per call, this is solidly past the threshold. For Kokoro-82M
short utterances, it can be in the window — measure before opting in.

### Kokoro special case: CPU deployment

Kokoro-82M is small enough that CPU-only deployment beats waiting in
a GPU queue for typical traffic. Declare it with `accelerator="none"`:

```python
@inference(
    resources=Resources(
        accelerator="none",        # or use "cpu" — both normalize
        requires_gpu=False,
    ),
    models={"model": kokoro_repo},
)
class KokoroSynth:
    ...
```

Now the orchestrator will place this endpoint on cheap CPU-only
workers. You can still ship a `cu128` build profile in
`endpoint.toml` and the orchestrator will pick whichever worker
becomes free first.

### Complete working example: F5-TTS SerialWorker

```python
"""F5-TTS — non-autoregressive flow-matching DiT for zero-shot speech."""

from __future__ import annotations

import time

import msgspec

from gen_worker import (
    Asset,
    Repo,
    RequestContext,
    Resources,
    compile as gw_compile,
    inference,
    io as gw_io,
)


f5_tts = Repo("SWivid/F5-TTS")


class F5TTSInput(msgspec.Struct):
    gen_text: str
    ref_audio: Asset
    ref_text: str = ""
    nfe_step: int = 32           # flow-matching ODE steps; quality plateaus around 32
    cfg_strength: float = 2.0
    speed: float = 1.0
    target_sample_rate: int = 24000
    seed: int = -1


class F5TTSOutput(msgspec.Struct):
    audio: Asset
    sample_rate: int
    duration_s: float
    rtf: float                   # compute_s / audio_s


@inference(
    label="F5-TTS",
    description="Non-AR flow-matching DiT for zero-shot voice cloning TTS.",
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=12.0,
        min_compute_capability=8.0,
        required_libraries=("torch", "torchaudio"),
        vram_must_fit="full_model",
        peak_vram_per_request_gb=14.0,
        vram_base=512 * 1024 * 1024,
        vram_size_multiplier=1.15,
        runtime_scales_with=("nfe_step", "gen_text"),
    ),
    models={
        "dit": f5_tts.flavor("bf16"),
        "vocoder": f5_tts.flavor("vocoder"),
    },
)
class F5TTSSynth:
    def setup(self, dit, vocoder) -> None:
        self.dit = dit
        self.vocoder = vocoder
        # Compile the heavy DiT. max-autotune trades startup time
        # for steady-state RTF — right call for audio.
        try:
            self.dit = gw_compile.torch_compile(self.dit, mode="max-autotune")
        except gw_compile.CompileUnavailableError:
            # Best-effort: fall back to eager if compile not viable.
            pass

    def warmup(self) -> None:
        # Synthesize a 1-second fixed reference at the nominal step
        # count so the compile cache and CUDA workspace are warm.
        pass    # replace with your actual short-form forward

    @inference.function(
        timeout_ms=120_000,
        description="Synthesize speech from text + a 3-30s reference clip.",
    )
    def synthesize(
        self,
        ctx: RequestContext,
        payload: F5TTSInput,
    ) -> F5TTSOutput:
        import torch

        ctx.raise_if_canceled()

        gen_text = payload.gen_text.strip()
        if not gen_text:
            raise ValueError("gen_text must be non-empty")

        nfe_step = max(1, min(64, int(payload.nfe_step)))
        speed = max(0.25, min(4.0, float(payload.speed)))
        sample_rate = max(1, int(payload.target_sample_rate))

        if payload.seed >= 0:
            torch.manual_seed(int(payload.seed))

        # Load the reference clip.
        ref_audio, ref_sr = gw_io.read_audio(
            payload.ref_audio, target_sample_rate=sample_rate
        )

        t0 = time.time()
        ctx.progress(0.05, stage="prepare")

        # Run the flow-matching ODE for nfe_step iterations, then
        # decode mel via the vocoder. Replace with your real impl.
        for step in range(nfe_step):
            ctx.raise_if_canceled()
            # ... one ODE step ...
            ctx.progress(0.05 + 0.85 * (step + 1) / nfe_step, stage="flow_matching")

        ctx.progress(0.92, stage="vocoder")
        # ... vocoder forward, returning a 1D float32 numpy array ...
        audio_array = ...    # shape: (N,)

        compute_s = time.time() - t0
        duration_s = len(audio_array) / sample_rate
        rtf = compute_s / duration_s if duration_s > 0 else float("inf")

        # Save the waveform. Use ctx.save_bytes for the in-memory shape
        # or stream via ctx.open_output_stream for very long outputs.
        audio_ref = f"jobs/{ctx.request_id}/outputs/synth.wav"
        # ... encode to WAV bytes via soundfile.write to a BytesIO ...
        audio_asset = ctx.save_bytes(audio_ref, wav_bytes)

        # Telemetry: emit RTF for the metrics pipeline.
        ctx.emit("synthesis_metrics", {
            "compute_seconds": compute_s,
            "audio_seconds": duration_s,
            "rtf": rtf,
        })

        ctx.progress(1.0, stage="done")
        return F5TTSOutput(
            audio=audio_asset,
            sample_rate=sample_rate,
            duration_s=duration_s,
            rtf=rtf,
        )

    def shutdown(self) -> None:
        import gc
        self.dit = None
        self.vocoder = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
```

---

## Part 2: Autoregressive audio (BatchedWorker)

The class shape is the same FOUR hooks — but `setup` is `async`, and
`@inference.function` methods are `async` generators that yield
deltas. The decorator carries `runtime="vllm"` to tell the SDK to
wire the vLLM continuous-batching engine adapter.

### What goes in `setup()` for an AR-TTS endpoint

```python
async def setup(self, model_dir):
    from gen_worker.runtimes.ar_tts import lookup as ar_tts_lookup
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    # Look up the engine + codec metadata for this model class.
    spec = ar_tts_lookup("Chatterbox")
    if spec is None or not spec.vllm_arch:
        raise FatalError("ar_tts registry missing vLLM port for Chatterbox")

    # vLLM engine config. trust_remote_code is required because the
    # community vLLM port registers a custom model architecture.
    engine_args = AsyncEngineArgs(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    # The codec runs on the GPU after vLLM emits audio tokens.
    from chatterbox.models.s3gen import S3Gen
    self._codec = S3Gen.from_pretrained(str(model_dir)).to("cuda").eval()
    self._spec = spec
```

`vllm` and the codec library should be lazy-imported INSIDE `setup()`
so the module imports cleanly on a CPU-only machine for discovery and
unit testing. The SDK's discovery pass runs on every build (image
build time); requiring CUDA-bound packages at module import breaks
the CI loop.

### The `ar_tts` model-class registry

The SDK ships a model-class → engine wiring registry at
`gen_worker.runtimes.ar_tts`:

```python
from gen_worker.runtimes.ar_tts import lookup, ARTTSModelSpec

spec = lookup("Chatterbox")          # case-insensitive
# spec.vllm_arch              -> "LlamaChatterboxModel"
# spec.audio_codec_decoder    -> "chatterbox.models.s3gen.S3Gen"
# spec.sample_rate_hz         -> 24000
# spec.audio_token_vocab_size -> 8192
# spec.supports_streaming     -> True
```

Built-in entries: `Chatterbox`, `GPT-SoVITS-v2`, `Bark`, `MusicGen`.
The registry holds engine/codec metadata only — no engine imports
happen at lookup time, so discovery on a CPU-only laptop succeeds.

For out-of-tree models, register at module-import time:

```python
from gen_worker.runtimes.ar_tts import register, ARTTSModelSpec

register(ARTTSModelSpec(
    model_class="MyCustomAR",
    vllm_arch="MyCustomARModel",
    audio_codec_decoder="my_pkg.codec.MyCodec",
    sample_rate_hz=24000,
    audio_token_vocab_size=4096,
    supports_streaming=True,
))
```

### Audio token streaming

Continuous batching gives you per-token deltas as the AR decoder
emits. The pattern is to buffer N audio tokens, run the codec on
that chunk, and yield the resulting WAV bytes as an `AudioDelta`.

For now, audio bytes ride the existing wire envelope's `payload_json`
field as base64-encoded chunks. A typed `IncrementalAudioChunk` envelope
is a planned follow-up once production endpoints have telemetry on
the JSON path's CPU cost — until then, base64 is the bridge that
works without a proto change.

```python
audio_b64 = base64.b64encode(wav_chunk_bytes).decode("ascii")
yield AudioDelta(
    index=item_index,
    total=total_items,
    processed=processed_count,
    audio_b64=audio_b64,
    mime="audio/wav",
    sample_rate_hz=spec.sample_rate_hz,
    audio_token_count=len(audio_token_ids),
)
```

`stream_chunk_every` = how many audio tokens to accumulate before
yielding. 25 is a reasonable default — short enough that the client
sees responsive output, long enough that the codec amortizes its
fixed cost.

### Cancellation in async endpoints

Same idiom as sync endpoints: `ctx.is_canceled()` returns a bool,
`ctx.raise_if_canceled()` raises `CanceledError`. For vLLM
specifically, you need to call `engine.abort(request_id)` to release
the engine's slot:

```python
async for vllm_out in stream:
    if ctx.is_canceled():
        await self._engine.abort(request_id)
        return
    ...
```

### Complete working example: Chatterbox BatchedWorker

```python
"""Chatterbox autoregressive TTS — BatchedWorker via vLLM."""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, AsyncIterator, Optional

import msgspec

from gen_worker import (
    FatalError,
    Repo,
    RequestContext,
    Resources,
    inference,
)
from gen_worker.runtimes.ar_tts import lookup as ar_tts_lookup


_MODEL_CLASS = "Chatterbox"
chatterbox = Repo("ResembleAI/chatterbox")


class SynthesisItem(msgspec.Struct):
    text: str
    voice_prompt_audio_ref: Optional[str] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None


class SynthesisInput(msgspec.Struct):
    items: list[SynthesisItem]
    temperature: float = 0.7
    top_p: float = 0.9
    max_audio_tokens: int = 1500
    output_sample_rate_hz: int = 24000
    stream: bool = True


class AudioDelta(msgspec.Struct):
    index: int
    total: int
    processed: int
    item_id: str | msgspec.UnsetType = msgspec.UNSET
    audio_b64: str | msgspec.UnsetType = msgspec.UNSET
    mime: str | msgspec.UnsetType = msgspec.UNSET
    sample_rate_hz: int | msgspec.UnsetType = msgspec.UNSET
    audio_token_count: int | msgspec.UnsetType = msgspec.UNSET
    finished: bool | msgspec.UnsetType = msgspec.UNSET
    error: str | msgspec.UnsetType = msgspec.UNSET


@inference(
    runtime="vllm",                  # tells the SDK to use vLLM continuous batching
    label="Chatterbox AR-TTS (BatchedWorker)",
    description=(
        "Resemble AI Chatterbox autoregressive TTS via vLLM continuous "
        "batching. Llama-class decoder emits S3 audio tokens; S3Gen "
        "codec turns tokens into a 24 kHz waveform."
    ),
    resources=Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=16.0,
        min_compute_capability=8.0,
        # BatchedWorker: many requests share the GPU. Per-request peak
        # is much smaller than the static model footprint (engine
        # manages KV cache + activations from a shared pool).
        peak_vram_per_request_gb=2.0,
        vram_must_fit="full_model",
        vram_base=2 * 1024 * 1024 * 1024,
        vram_size_multiplier=1.15,
        vram_scales_with=("items",),
        runtime_scales_with=("items", "max_audio_tokens"),
    ),
    models={"chatterbox_model": chatterbox.flavor("bf16")},
)
class ChatterboxTTS:
    def __init__(self) -> None:
        self._engine: Any = None
        self._codec: Any = None
        self._spec: Any = None

    async def setup(self, chatterbox_model: Any) -> None:
        # 1. Look up engine + codec metadata.
        self._spec = ar_tts_lookup(_MODEL_CLASS)
        if self._spec is None or not self._spec.vllm_arch:
            raise FatalError(f"ar_tts registry has no vLLM port for {_MODEL_CLASS!r}")

        # 2. Lazy imports — keep module imports CPU-clean.
        import torch                                                   # noqa: F401
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import chatterbox_vllm                                         # noqa: F401 — registers arch with vLLM
        from chatterbox.models.s3gen import S3Gen

        model_path = str(chatterbox_model)    # SDK gives os.PathLike or path str

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            trust_remote_code=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._codec = S3Gen.from_pretrained(model_path).to("cuda").eval()

    @inference.function(
        timeout_ms=180_000,
        max_concurrent_per_worker=64,       # AR engine handles many concurrent requests
        description="Synthesize speech for one or more text items.",
    )
    async def synthesize(
        self,
        ctx: RequestContext,
        payload: SynthesisInput,
    ) -> AsyncIterator[AudioDelta]:
        from vllm import SamplingParams

        total = len(payload.items)
        if total == 0:
            return

        sampling = SamplingParams(
            temperature=float(payload.temperature),
            top_p=float(payload.top_p),
            max_tokens=int(payload.max_audio_tokens),
        )

        processed = 0
        for index, item in enumerate(payload.items):
            if ctx.is_canceled():
                return

            request_id = f"{ctx.request_id}:{index}:{int(time.time() * 1000)}"
            audio_token_ids: list[int] = []
            last_emit = 0
            chunk_every = 25

            stream = self._engine.generate(
                prompt=item.text,
                sampling_params=sampling,
                request_id=request_id,
            )

            async for vllm_out in stream:
                if ctx.is_canceled():
                    await self._engine.abort(request_id)
                    return
                if not vllm_out.outputs:
                    continue
                primary = vllm_out.outputs[0]
                audio_token_ids = list(primary.token_ids)

                # Stream a chunk every N tokens.
                if payload.stream and (len(audio_token_ids) - last_emit) >= chunk_every:
                    chunk_ids = audio_token_ids[last_emit:]
                    last_emit = len(audio_token_ids)
                    wav_bytes = await asyncio.to_thread(
                        self._decode_audio_chunk,
                        chunk_ids, item, payload.output_sample_rate_hz,
                    )
                    if wav_bytes:
                        yield AudioDelta(
                            index=index, total=total, processed=processed,
                            audio_b64=base64.b64encode(wav_bytes).decode("ascii"),
                            mime="audio/wav",
                            sample_rate_hz=payload.output_sample_rate_hz,
                            audio_token_count=len(audio_token_ids),
                        )

            # Tail tokens + finished marker.
            tail = audio_token_ids[last_emit:]
            if tail:
                wav_bytes = await asyncio.to_thread(
                    self._decode_audio_chunk,
                    tail, item, payload.output_sample_rate_hz,
                )
                if wav_bytes:
                    yield AudioDelta(
                        index=index, total=total, processed=processed,
                        audio_b64=base64.b64encode(wav_bytes).decode("ascii"),
                        mime="audio/wav",
                        sample_rate_hz=payload.output_sample_rate_hz,
                        audio_token_count=len(audio_token_ids),
                    )

            processed += 1
            ctx.progress(processed / total, stage="synthesizing")
            yield AudioDelta(
                index=index, total=total, processed=processed,
                audio_token_count=len(audio_token_ids),
                sample_rate_hz=payload.output_sample_rate_hz,
                finished=True,
            )

    def _decode_audio_chunk(self, token_ids, item, sample_rate_hz) -> bytes:
        """Synchronous codec call. Run via asyncio.to_thread."""
        import numpy as np
        import soundfile as sf
        import torch

        if not token_ids:
            return b""
        with torch.inference_mode():
            tokens = torch.tensor([token_ids], dtype=torch.long).to("cuda")
            voice = (item.voice_prompt_audio_ref or "").strip() or None
            waveform = self._codec.decode(tokens, voice_prompt=voice)
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().float().numpy()
            waveform = np.asarray(waveform).reshape(-1)
        buf = io.BytesIO()
        sf.write(buf, waveform, sample_rate_hz, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    async def shutdown(self) -> None:
        try:
            if self._engine is not None:
                shutdown_fn = getattr(self._engine, "shutdown_background_loop", None)
                if callable(shutdown_fn):
                    try:
                        await shutdown_fn()
                    except TypeError:
                        shutdown_fn()
        finally:
            self._engine = None
            self._codec = None
```

What this gives you on H100:
- `chatterbox-vllm` (the community port) reports ~4× over the
  reference HF implementation from kernel optimizations alone at
  batch=1.
- Continuous batching adds ~10× over that for traffic mixes with
  multiple concurrent requests.
- The codec runs on GPU after the engine emits tokens; codec cost
  is fixed per chunk and ~5% of total request time.

---

## When to pick which — quick reference

| Scenario                                            | Archetype       | Why                                                |
|-----------------------------------------------------|-----------------|----------------------------------------------------|
| F5-TTS, MaskGCT (flow-matching)                     | SerialWorker    | One request fills the GPU; no token batching.      |
| Kokoro on CPU                                       | SerialWorker    | Small enough to ship CPU-only; no GPU coordination. |
| Stable Audio Open (latent diffusion)                | SerialWorker    | Diffusion forward; no AR sharing.                  |
| Chatterbox, GPT-SoVITS (Llama-class decoder + codec)| BatchedWorker   | AR decoder fits vLLM continuous batching directly. |
| Bark, MusicGen                                      | BatchedWorker   | Three-stage AR decoders; each stage benefits from batching. |
| Whisper (ASR transcription)                         | SerialWorker    | Encoder-decoder; not the AR-token pattern.         |

If your model emits one audio token at a time from a Llama-class
decoder, it's BatchedWorker. Everything else is SerialWorker.

---

## What NOT to do

**Don't mix archetypes in one class.** Either every
`@inference.function` is async (BatchedWorker) or every one is sync
(SerialWorker). The SDK detects async-ness on `setup` or any
function method and routes accordingly. Mixing them is a decoration-
time error.

**Don't import vLLM / chatterbox-vllm at module import.** Lazy-import
inside `setup()`. Module-import-time GPU imports break CPU-only
discovery and CI.

**Don't ship without `runtime="vllm"` if you want continuous batching.**
An async class without `runtime=` runs as a one-off async worker
(useful for streaming text generation with no engine, but loses the
batching gains for AR-TTS).

**Don't assign payload data to `self`.** The class is shared across
requests; `self.text = payload.text` inside `synthesize()` leaks
across concurrent requests. Per-request state is local variables,
or a `dict[request_id, T]` keyed on `ctx.request_id` if you must
carry state across yields.

**Don't forget to abort vLLM on cancel.** `await self._engine.abort(request_id)`
releases the engine slot. Without it, canceled requests keep
occupying engine slots until they hit their token budget.

---

## Next steps

- **Two-stage AR + diffusion hybrids** (e.g., AR text-to-codec
  followed by diffusion super-resolution)? Use
  `@inference.stage(name=..., gpu_class=...)` on the diffusion
  stage so future disaggregation can split it onto a separate GPU
  pool. See [cookbook-stages.md](cookbook-stages.md).
- **Custom AR-TTS model**? Register an `ARTTSModelSpec` at module
  import time via `gen_worker.runtimes.ar_tts.register(...)` and use
  the same BatchedWorker shape.
- **Whisper / ASR**? SerialWorker, no special engine. Standard
  encoder-decoder; see the `medasr-transcribe` example in
  `examples/`.
