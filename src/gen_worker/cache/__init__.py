"""Per-request feature caches for diffusion models (#324).

.. note::
   For the common case, prefer :func:`gen_worker.accel.apply_fbcache` —
   the canonical five-call surface in :mod:`gen_worker.accel` covers most
   SerialWorker endpoints. This module remains available for advanced
   cases that need additional cache backends (DeepCache, TeaCache) or
   finer-grained control.

These wrap a HuggingFace Diffusers pipeline to skip redundant computation
across timesteps. They are per-request acceleration — multiple concurrent
requests each get their own cache state (the cache wrapper holds no
cross-request state). Apply in `setup()`:

    @inference(models={"pipe": flux_klein})
    class FluxKleinGenerate:
        def setup(self, pipe):
            pipe.transformer = torch.compile(pipe.transformer, mode='reduce-overhead')
            gen_worker.cache.FBCache(threshold=0.12).apply(pipe)
            self.pipe = pipe

The actual third-party integrations (ParaAttention, DeepCache, TeaCache)
are imported lazily inside `apply()` so they don't force install at
gen-worker boot when the tenant doesn't use them. Importing any wrapper
class (e.g. ``from gen_worker.cache import FBCache``) without calling
``.apply()`` does NOT require the third-party library to be installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CacheUnavailableError(RuntimeError):
    """Raised when a cache helper's third-party dependency is not installed.

    Tenants control which acceleration packages get installed in their
    endpoint image. If the worker code calls ``FBCache().apply(pipe)``
    but ``para-attn`` is not present in the image, we surface a clear
    message so the build/deploy can be fixed.
    """


class _CacheBase:
    """Common interface every cache wrapper implements.

    Tenant calls `.apply(pipe)` in `setup()`. The wrapper patches the
    pipeline in-place and returns the patched pipeline (also stored on
    `self.pipe = pipe` by the tenant for clarity).

    Set `breaks_cross_request_batching = True` for any cache that
    cannot be used together with cross-request micro-batching (e.g.
    TeaCache — see nunchaku #597). The SDK reads this flag to
    auto-disable micro-batching when this cache is in use.
    """

    breaks_cross_request_batching: bool = False

    def apply(self, pipe: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__}.apply() not implemented."
        )


class FBCache(_CacheBase):
    """First-Block Cache for DiT-class models (Flux, SD3, Qwen-Image).

    Wraps `para_attn.first_block_cache.diffusers_adapters.apply_cache_on_pipe`.
    ~1.5-2x speedup on Flux per ParaAttention benchmarks.

    Args:
        threshold: residual-diff threshold for cache invalidation. 0.12 is
            the default sweet spot per ParaAttention docs. Lower = more
            cache invalidations = higher fidelity / less speedup. Higher
            = fewer invalidations = more speedup / more quality drift.

    Requires (lazy-imported when ``apply()`` is called):
        pip install para-attn  # Apache 2.0
    """

    def __init__(self, threshold: float = 0.12) -> None:
        self.threshold = threshold

    def apply(self, pipe: Any) -> Any:
        try:
            from para_attn.first_block_cache.diffusers_adapters import (
                apply_cache_on_pipe,
            )
        except ImportError as e:
            raise CacheUnavailableError(
                "FBCache requires `para-attn`. Install with `pip install "
                "para-attn` (Apache 2.0) and rebuild the endpoint image."
            ) from e

        apply_cache_on_pipe(pipe, residual_diff_threshold=self.threshold)
        logger.info(
            "FBCache applied to %s (threshold=%s)",
            type(pipe).__name__,
            self.threshold,
        )
        return pipe


class DeepCache(_CacheBase):
    """Deep cache for UNet-class models (SDXL family).

    Wraps DeepCacheSDHelper. ~2.5-3x on SDXL per CVPR 2024 paper.

    Args:
        cache_interval: skip-then-cache interval across denoising steps.
            3 is the typical sweet spot.
        cache_branch_id: which UNet up-block to start caching from.
            0 is the deepest branch (most reuse, most speedup); higher
            values are more conservative.

    Requires (lazy-imported when ``apply()`` is called):
        pip install DeepCache  # Apache 2.0
    """

    def __init__(
        self,
        cache_interval: int = 3,
        cache_branch_id: int = 0,
    ) -> None:
        self.cache_interval = cache_interval
        self.cache_branch_id = cache_branch_id
        self._helper: Any = None

    def apply(self, pipe: Any) -> Any:
        try:
            from DeepCache import DeepCacheSDHelper
        except ImportError as e:
            raise CacheUnavailableError(
                "DeepCache requires `DeepCache`. Install with "
                "`pip install DeepCache` and rebuild the endpoint image."
            ) from e

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=self.cache_interval,
            cache_branch_id=self.cache_branch_id,
        )
        helper.enable()
        # Hold a reference so the helper survives for the lifetime of the
        # pipe; DeepCache patches pipe.unet.forward via a closure, but the
        # helper object owns the cleanup hooks.
        pipe._gen_worker_deepcache_helper = helper
        self._helper = helper
        logger.info(
            "DeepCache applied to %s (cache_interval=%s, cache_branch_id=%s)",
            type(pipe).__name__,
            self.cache_interval,
            self.cache_branch_id,
        )
        return pipe


class TeaCache(_CacheBase):
    """Timestep-Embedding-Aware cache (DiT / video).

    1.6-2x on Flux/HunyuanVideo per arXiv 2411.19108. Conflicts with
    batch>1 (nunchaku #597) — the SerialWorker micro-batching aggregator
    auto-disables when this is enabled.

    Args:
        threshold: relative L1 distance threshold for cache invalidation.
            0.6 is the upstream recommended default for DiT models.

    Requires (lazy-imported when ``apply()`` is called):
        pip install teacache  # MIT, ali-vilab/TeaCache
    """

    breaks_cross_request_batching = True

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def apply(self, pipe: Any) -> Any:
        # Upstream ali-vilab/TeaCache ships per-model patches rather than a
        # single uniform `apply_on_pipe(...)` entry point. We try the most
        # common adapter shapes in order; if none load, surface a clear
        # error pointing at the canonical install + integration docs.
        candidates = (
            ("teacache.diffusers_adapters", "apply_teacache_on_pipe"),
            ("teacache", "apply_teacache_on_pipe"),
            ("teacache", "apply_on_pipe"),
        )
        last_err: BaseException | None = None
        for module_name, attr in candidates:
            try:
                module = __import__(module_name, fromlist=[attr])
            except ImportError as e:
                last_err = e
                continue
            apply_fn = getattr(module, attr, None)
            if apply_fn is None:
                continue
            apply_fn(pipe, rel_l1_thresh=self.threshold)
            logger.info(
                "TeaCache applied to %s via %s.%s (threshold=%s)",
                type(pipe).__name__,
                module_name,
                attr,
                self.threshold,
            )
            return pipe

        raise CacheUnavailableError(
            "TeaCache requires the `teacache` package from ali-vilab/TeaCache "
            "(https://github.com/ali-vilab/TeaCache). Install with "
            "`pip install teacache` and rebuild the endpoint image. "
            f"Last import error: {last_err!r}"
        )


__all__ = ["FBCache", "DeepCache", "TeaCache", "CacheUnavailableError"]
