"""Continuous-batching engine adapters for BatchedWorker (#273).

This module provides thin Python wrappers around vLLM and SGLang so the
worker can host a long-lived inference engine inside the BatchedWorker SDK
shape. The wrappers expose four operations:

  - ``await engine.generate(request_id, payload) -> AsyncIterator[dict]``
    Stream the engine's token outputs back to the worker as plain dicts
    (``{"delta_text": str, "finished": bool, ...}``). The worker translates
    these into IncrementalTokenDelta typed messages on the wire.

  - ``await engine.abort(request_id) -> None``
    Tell the engine to drop the named request from its in-flight batch
    and free its KV blocks at the next iteration boundary. Idempotent.

  - ``await engine.shutdown() -> None``
    Stop the engine, releasing GPU memory and worker subprocesses.

  - ``engine.is_available()`` (classmethod) — True iff the underlying
    Python package can be imported.

**Import-without-GPU is supported.** Constructing the wrapper class itself
imports nothing heavy — vllm / sglang are imported lazily inside
``start()`` / ``generate()``. ``from gen_worker.engines import SGLangEngine``
works without a GPU on the box.

The real engine init lives in ``SGLangEngine.start(model_path)`` /
``VLLMEngine.start(model_path)``. Tenant code calls this implicitly via the
SDK lifecycle: worker calls ``await engine.start(model_path)``, then
``await batched_worker.setup(engine=engine)``, then warmup/serve loop.

JoyCaption is multimodal Llama (vision-language). Both SGLang and vLLM
support multimodal Llama variants directly — the wrappers pass through the
``image_url`` field on the payload as the engine's documented multimodal
input format.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


class EngineUnavailableError(RuntimeError):
    """Raised when an engine's underlying package is not installed.

    Tenants opt into the engine via ``@inference(runtime="sglang")`` or
    ``runtime="vllm"`` on the class. The corresponding package must be
    pinned in the endpoint image. Missing-import surfaces here so the
    worker boot fails loud rather than running with a non-batched fallback.
    """


class EngineBase:
    """Common interface every continuous-batching engine wrapper implements.

    BatchedWorker tenant code receives an EngineBase instance from the
    SDK via ``setup(engine=...)`` and calls ``engine.generate(...)`` from
    inside its ``@inference.function`` method. The wrapper handles request
    submission, async iteration over output tokens, cancellation, and
    shutdown.
    """

    name: str = "engine"

    @classmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    async def start(self, model_path: str, **kwargs: Any) -> None:
        raise NotImplementedError

    async def generate(
        self,
        request_id: str,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield streaming output chunks for the request.

        Each yielded dict carries at minimum ``{"delta_text": str,
        "finished": bool}``. Tenant code wraps each chunk into the
        endpoint's typed delta struct before yielding back to the SDK.
        """
        raise NotImplementedError
        yield  # pragma: no cover — type hint as async generator

    async def abort(self, request_id: str) -> None:
        """Drop the named request from the engine's in-flight batch.

        Idempotent — calling abort on a request the engine has never
        seen, or has already completed, must be a no-op.
        """
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Tear down the engine, releasing GPU memory and subprocesses.

        Called once when the worker drains. Must be safe to call when
        no requests are in flight (the drain path waits for active
        requests to finish first).
        """
        raise NotImplementedError


class SGLangEngine(EngineBase):
    """SGLang ``sgl.Engine`` wrapper.

    SGLang's RadixAttention is the strongest a-priori fit for prefix-heavy
    multimodal workloads (JoyCaption ships a long system prompt + image
    tower output that all requests share). See progress.json #273 Phase 0
    runtime-selection spike for details.

    Requires (lazy-imported in ``start()``):
        pip install sglang  # Apache 2.0
    """

    name = "sglang"

    def __init__(self, **engine_kwargs: Any) -> None:
        self._engine_kwargs = dict(engine_kwargs or {})
        self._engine: Any = None  # set in start()
        self._lock = asyncio.Lock()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import importlib

            importlib.import_module("sglang")
            return True
        except ImportError:
            return False

    async def start(self, model_path: str, **kwargs: Any) -> None:
        try:
            import sglang as sgl  # type: ignore
        except ImportError as e:
            raise EngineUnavailableError(
                "SGLangEngine requires `sglang`. Install with `pip install "
                "sglang` (Apache 2.0) and rebuild the endpoint image."
            ) from e

        merged = {**self._engine_kwargs, **(kwargs or {})}
        merged.setdefault("model_path", model_path)
        # SGLang's offline engine entry point. The class name differs
        # slightly across SGLang versions (`sgl.Engine` is canonical
        # post-0.4); we accept either path.
        EngineCls = getattr(sgl, "Engine", None) or getattr(sgl, "engine", None)
        if EngineCls is None:
            raise EngineUnavailableError(
                "sglang installed but no `sgl.Engine` class found. "
                "This wrapper expects SGLang >= 0.4."
            )
        logger.info("Starting SGLang engine with model_path=%s", model_path)
        # Engine init is sync but blocking — run in executor so we don't
        # stall the asyncio loop the worker uses for the receive stream.
        loop = asyncio.get_running_loop()
        self._engine = await loop.run_in_executor(
            None, lambda: EngineCls(**merged)
        )
        logger.info("SGLang engine ready")

    async def generate(
        self,
        request_id: str,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        if self._engine is None:
            raise RuntimeError("SGLangEngine.generate() called before start()")
        # SGLang Engine.async_generate yields dicts with `text` and `meta_info`.
        kwargs = dict(payload)
        # Map our common payload keys to SGLang's expected names.
        prompt = kwargs.pop("prompt", None) or kwargs.pop("text", None) or ""
        image = kwargs.pop("image_url", None) or kwargs.pop("image", None)
        sampling_params = kwargs.pop("sampling_params", None) or {}
        # Common sampling-knob aliases tenant code may use.
        for src, dst in (
            ("max_new_tokens", "max_new_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
        ):
            if src in kwargs:
                sampling_params.setdefault(dst, kwargs.pop(src))

        gen_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "sampling_params": sampling_params,
            "stream": True,
            "rid": request_id,
        }
        if image is not None:
            gen_kwargs["image_data"] = image

        last_text = ""
        async for chunk in self._engine.async_generate(**gen_kwargs):
            # SGLang streaming chunk shape: {"text": str, "meta_info":
            # {"finish_reason": dict | None, "id": str, ...}}.
            text = str(chunk.get("text") or "")
            meta = chunk.get("meta_info") or {}
            finish_reason = meta.get("finish_reason")
            delta = text[len(last_text) :] if text.startswith(last_text) else text
            last_text = text
            yield {
                "delta_text": delta,
                "finished": finish_reason is not None,
                "finish_reason": finish_reason,
                "meta": meta,
            }
            if finish_reason is not None:
                break

    async def abort(self, request_id: str) -> None:
        if self._engine is None:
            return
        # SGLang exposes abort_request(rid) on the Engine; older versions
        # name it abort(...). Accept either.
        abort_fn = getattr(self._engine, "abort_request", None) or getattr(
            self._engine, "abort", None
        )
        if abort_fn is None:
            logger.warning(
                "SGLang engine has no abort_request method; KV blocks for "
                "rid=%s will free when generation completes",
                request_id,
            )
            return
        try:
            if asyncio.iscoroutinefunction(abort_fn):
                await abort_fn(request_id)
            else:
                # Some SGLang versions expose abort as sync.
                abort_fn(request_id)
        except Exception:
            logger.exception(
                "SGLang abort failed for rid=%s (idempotent; swallowing)",
                request_id,
            )

    async def shutdown(self) -> None:
        if self._engine is None:
            return
        try:
            shutdown_fn = getattr(self._engine, "shutdown", None)
            if shutdown_fn is None:
                return
            if asyncio.iscoroutinefunction(shutdown_fn):
                await shutdown_fn()
            else:
                shutdown_fn()
        except Exception:
            logger.exception("SGLang shutdown raised; continuing drain")
        finally:
            self._engine = None


class VLLMEngine(EngineBase):
    """vLLM ``AsyncLLMEngine`` wrapper.

    vLLM 0.5+ implements automatic prefix caching analogous to SGLang's
    RadixAttention and has broader model coverage. Picked as the runtime
    when ``@inference(runtime="vllm")`` is declared.

    Requires (lazy-imported in ``start()``):
        pip install vllm  # Apache 2.0
    """

    name = "vllm"

    def __init__(self, **engine_kwargs: Any) -> None:
        self._engine_kwargs = dict(engine_kwargs or {})
        self._engine: Any = None
        self._lock = asyncio.Lock()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import importlib

            importlib.import_module("vllm")
            return True
        except ImportError:
            return False

    async def start(self, model_path: str, **kwargs: Any) -> None:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore
        except ImportError as e:
            raise EngineUnavailableError(
                "VLLMEngine requires `vllm`. Install with `pip install vllm` "
                "(Apache 2.0) and rebuild the endpoint image."
            ) from e

        merged = {**self._engine_kwargs, **(kwargs or {})}
        merged.setdefault("model", model_path)
        logger.info("Starting vLLM engine with model=%s", model_path)
        engine_args = AsyncEngineArgs(**merged)
        loop = asyncio.get_running_loop()
        self._engine = await loop.run_in_executor(
            None, lambda: AsyncLLMEngine.from_engine_args(engine_args)
        )
        logger.info("vLLM engine ready")

    async def generate(
        self,
        request_id: str,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        if self._engine is None:
            raise RuntimeError("VLLMEngine.generate() called before start()")
        try:
            from vllm import SamplingParams  # type: ignore
        except ImportError as e:
            raise EngineUnavailableError(
                "vllm installed but `SamplingParams` import failed."
            ) from e

        kwargs = dict(payload)
        prompt = kwargs.pop("prompt", None) or kwargs.pop("text", None) or ""
        image = kwargs.pop("image_url", None) or kwargs.pop("image", None)
        sp_kwargs: Dict[str, Any] = {}
        for k in ("max_new_tokens", "temperature", "top_p", "top_k"):
            if k in kwargs:
                # vLLM names it max_tokens, not max_new_tokens.
                dst = "max_tokens" if k == "max_new_tokens" else k
                sp_kwargs[dst] = kwargs.pop(k)
        sampling = SamplingParams(**sp_kwargs) if sp_kwargs else SamplingParams()

        # Multimodal payload shape: vLLM accepts {"prompt": str,
        # "multi_modal_data": {"image": <PIL or URL>}}.
        prompt_obj: Any = prompt
        if image is not None:
            prompt_obj = {"prompt": prompt, "multi_modal_data": {"image": image}}

        last_text = ""
        async for output in self._engine.generate(
            prompt_obj, sampling, request_id=request_id
        ):
            # vLLM streaming output shape: RequestOutput with outputs[0].text
            # accumulating the running generation.
            try:
                gen = output.outputs[0]
            except (AttributeError, IndexError):
                continue
            text = str(getattr(gen, "text", "") or "")
            finished = bool(getattr(output, "finished", False))
            finish_reason = getattr(gen, "finish_reason", None)
            delta = text[len(last_text) :] if text.startswith(last_text) else text
            last_text = text
            yield {
                "delta_text": delta,
                "finished": finished,
                "finish_reason": finish_reason,
            }
            if finished:
                break

    async def abort(self, request_id: str) -> None:
        if self._engine is None:
            return
        abort_fn = getattr(self._engine, "abort", None)
        if abort_fn is None:
            logger.warning(
                "vLLM engine has no abort method; KV blocks for rid=%s will "
                "free when generation completes",
                request_id,
            )
            return
        try:
            if asyncio.iscoroutinefunction(abort_fn):
                await abort_fn(request_id)
            else:
                abort_fn(request_id)
        except Exception:
            logger.exception(
                "vLLM abort failed for rid=%s (idempotent; swallowing)",
                request_id,
            )

    async def shutdown(self) -> None:
        if self._engine is None:
            return
        try:
            shutdown_fn = getattr(self._engine, "shutdown_background_loop", None) or getattr(
                self._engine, "shutdown", None
            )
            if shutdown_fn is None:
                return
            if asyncio.iscoroutinefunction(shutdown_fn):
                await shutdown_fn()
            else:
                shutdown_fn()
        except Exception:
            logger.exception("vLLM shutdown raised; continuing drain")
        finally:
            self._engine = None


def make_engine(runtime: str, **engine_kwargs: Any) -> EngineBase:
    """Factory used by the worker startup path.

    Caller passes the ``runtime=`` declared on ``@inference(runtime=...)``;
    we return a wrapper instance. The wrapper does NOT call into the
    underlying package at construction — that happens in ``start()``.
    """
    r = (runtime or "").strip().lower()
    if r == "sglang":
        return SGLangEngine(**engine_kwargs)
    if r == "vllm":
        return VLLMEngine(**engine_kwargs)
    raise ValueError(
        f"Unknown runtime {runtime!r}. Supported: 'sglang' | 'vllm'. "
        "See progress.json #273 for the BatchedWorker shape."
    )


__all__ = [
    "EngineBase",
    "SGLangEngine",
    "VLLMEngine",
    "EngineUnavailableError",
    "make_engine",
]
