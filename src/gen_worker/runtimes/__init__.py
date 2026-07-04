"""Runtime engine helpers for engine-hosted endpoints.

Submodules:
  - ``server`` — boot/health-wait/abort/shutdown for server subprocesses
    (vLLM, llama-server); backs ``@endpoint(runtime=...)``.
  - ``ar_tts`` — model-class registry for autoregressive TTS endpoints
    (Chatterbox, GPT-SoVITS, Bark, MusicGen). Maps a public model class
    name to the vLLM/SGLang architecture + audio-codec decoder hooks
    that the BatchedWorker shape needs to wire continuous batching.

Heavy engine imports (``vllm``, ``sglang``, ``torch``, codec libs) are
lazy — importing ``gen_worker.runtimes`` does NOT import vLLM. Endpoint
``setup()`` performs the import inside the worker process.
"""

from __future__ import annotations

__all__ = ["ar_tts"]
