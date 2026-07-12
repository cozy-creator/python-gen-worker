"""Autoregressive-TTS model-class registry (progress.json #327).

Autoregressive TTS models (Chatterbox, Bark, MusicGen, …) are
architecturally Llama-class decoders that emit *audio* tokens instead of text
tokens. The decoder side fits cleanly onto a continuous-batching engine
(vLLM today; SGLang has no community AR-TTS runners yet); the post-decoder
side runs a learned audio codec (S3 token decoder, SoundStream, EnCodec, …)
to turn token ids back into a waveform.

The BatchedWorker SDK shape (``@endpoint(runtime="vllm")`` on an async class,
see ``api/decorators.py``) needs two pieces of model-specific metadata to
wire an endpoint:

  1. ``vllm_arch`` — the vLLM model class name to load with
     ``LLM(model=..., model_class=vllm_arch)``. Maps to a registered vLLM
     architecture; chatterbox-vllm publishes ``LlamaChatterboxModel`` as
     a community vLLM model. ``None`` if the model has no vLLM port yet.
  2. ``audio_codec_decoder`` — the decoder import path that turns the
     audio-token sequence into a waveform. The BatchedWorker calls this
     after the engine emits a token-id batch. Different models use
     different codecs; the registry abstracts that detail so endpoints
     can share one engine-driver harness.

The registry is intentionally a plain in-memory dict keyed by the human
model-class name (e.g. ``"Chatterbox"``). Endpoints look up their entry at
``setup()`` time:

    from gen_worker.runtimes.ar_tts import lookup
    spec = lookup("Chatterbox")
    if spec is None:
        raise FatalError("ar_tts model class is not registered")
    # spec.vllm_arch == "LlamaChatterboxModel"

This module performs *no* engine imports. ``vllm`` / codec weights are
loaded inside the endpoint's ``setup()`` so endpoint code can import on a
CPU-only machine for discovery + unit testing.

References:
  - chatterbox-vllm port: https://github.com/randombk/chatterbox-vllm
  - Resemble AI Chatterbox: https://github.com/resemble-ai/chatterbox
  - vLLM AR-TTS tracking: https://github.com/vllm-project/vllm/issues/21989
"""

from __future__ import annotations

from typing import Optional

import msgspec


class ARTTSModelSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Engine + codec wiring for one AR-TTS model class.

    Attributes:
      model_class: Human name (`"Chatterbox"`, `"MusicGen"`, etc.).
      vllm_arch: vLLM model class name (registered via
        ``ModelRegistry.register_model``). ``None`` if the model has no
        vLLM port yet.
      audio_codec_decoder: Import path or registry key for the codec
        decoder that converts audio-token ids → waveform samples.
      sample_rate_hz: Native sample rate of the codec output, in hertz.
      audio_token_vocab_size: Size of the audio-token vocabulary (NOT
        text — engine config needs this for output-token clamping).
      supports_streaming: True if the codec can decode partial windows
        for incremental audio chunk streaming. False forces end-of-stream
        decoding.
      notes: Free-form prose describing quirks (model-card link, known
        batching quirks, eos-token-id, etc.).
    """

    model_class: str
    vllm_arch: Optional[str] = None
    audio_codec_decoder: Optional[str] = None
    sample_rate_hz: int = 24000
    audio_token_vocab_size: Optional[int] = None
    supports_streaming: bool = False
    notes: Optional[str] = None


# ----------------------------------------------------------------------
# Registry — keyed by human model-class name. Lowercased on lookup so
# tenants can write "Chatterbox" / "chatterbox" / "CHATTERBOX" and get
# the same entry.
#
# All engine + codec libraries are referenced by string ONLY — this file
# stays import-light so discovery on a CPU-only laptop succeeds.
# ----------------------------------------------------------------------


_REGISTRY: dict[str, ARTTSModelSpec] = {
    # Chatterbox (Resemble AI). chatterbox-vllm port reports >10x batching
    # gains vs the reference HF implementation. Codec is Resemble's S3
    # token-to-waveform decoder shipped with the model checkpoint.
    "chatterbox": ARTTSModelSpec(
        model_class="Chatterbox",
        vllm_arch="LlamaChatterboxModel",
        audio_codec_decoder="chatterbox.models.s3gen.S3Gen",
        sample_rate_hz=24000,
        audio_token_vocab_size=8192,
        supports_streaming=True,
        notes=(
            "Llama-class decoder emitting S3 audio tokens. "
            "chatterbox-vllm registers `LlamaChatterboxModel` with vLLM "
            "via `ModelRegistry.register_model` at import time."
        ),
    ),
    # MusicGen (Meta). T5 encoder + transformer decoder emitting EnCodec
    # tokens. HF transformers ships native batching. Continuous batching
    # via vLLM is being explored in vllm#21989.
    "musicgen": ARTTSModelSpec(
        model_class="MusicGen",
        vllm_arch=None,
        audio_codec_decoder="encodec.EncodecModel",
        sample_rate_hz=32000,
        audio_token_vocab_size=2048,
        supports_streaming=False,
        notes=(
            "T5 encoder + AR transformer decoder. EnCodec waveform "
            "decoder. Targeted for vLLM continuous batching via "
            "vllm-project/vllm#21989."
        ),
    ),
}


def lookup(model_class: str) -> Optional[ARTTSModelSpec]:
    """Return the registry entry for a model class, or ``None`` if absent.

    Lookup is case-insensitive. Whitespace is stripped.
    """
    if not model_class:
        return None
    key = str(model_class).strip().lower()
    if not key:
        return None
    return _REGISTRY.get(key)



__all__ = [
    "ARTTSModelSpec",
    "lookup",
]
