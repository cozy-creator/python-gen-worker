"""Runtime engine helpers for engine-hosted endpoints.

Submodules:
  - ``server`` — boot/health-wait/abort/shutdown for server subprocesses
    (vLLM, llama-server); backs ``@endpoint(runtime=...)``.

Heavy engine imports (``vllm``, ``sglang``, ``torch``, codec libs) are
lazy — importing ``gen_worker.runtimes`` does NOT import vLLM. Endpoint
``setup()`` performs the import inside the worker process.
"""

from __future__ import annotations

__all__: list[str] = []
