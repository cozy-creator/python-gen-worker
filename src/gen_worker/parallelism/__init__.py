"""Multi-GPU parallelism helpers for SerialWorker endpoints (#324).

.. note::
   For the common case, prefer :func:`gen_worker.accel.apply_para_attn` —
   the canonical five-call surface in :mod:`gen_worker.accel` covers most
   SerialWorker endpoints. This module remains available for advanced
   cases (xDiT Unified Sequence Parallel, tensor parallelism with
   non-default placement strategies).

Wraps xDiT's Unified Sequence Parallel (USP) for single-request multi-GPU
inference on large video/image DiTs. The xDiT paper measured 3.54x on Mochi
across 6xL40 — wins are per-request; throughput still scales horizontally
across replicas.

Tenant usage:

    @inference(models={"pipe": mochi})
    class MochiGenerate:
        def setup(self, pipe):
            pipe = gen_worker.parallelism.xdit_sequence_parallel(pipe, gpus=4)
            self.pipe = pipe

The helper is a no-op on single-GPU workers (gpus=1) and falls back with a
warning on multi-GPU workers when xDiT isn't installed.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)


class ParallelismUnavailableError(RuntimeError):
    """Raised when xDiT (or another required parallel runtime) is missing."""


def _detect_gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def xdit_sequence_parallel(
    pipe: Any,
    *,
    gpus: int,
    ring_degree: int | None = None,
    ulysses_degree: int | None = None,
    use_cfg_parallel: bool = False,
    fallback: str = "passthrough",
) -> Any:
    """Wrap a Diffusers pipeline in xDiT's Unified Sequence Parallel.

    USP splits the sequence dimension of the DiT forward across `gpus`
    devices. Best speedup on Mochi / Wan / HunyuanVideo at large frame
    counts. Sub-linear scaling — xDiT reports 3.54x on 6xL40 for Mochi
    (i.e. ~59% efficiency at 6 GPUs).

    Args:
        pipe: the Diffusers pipeline (must own a `transformer` or `dit`
            module compatible with xDiT's wrappers).
        gpus: total degree of parallelism. ``gpus=1`` is a no-op.
        ring_degree: ring-attention degree. If None, defaults to ``gpus``
            when ulysses_degree is None, else ``gpus // ulysses_degree``.
        ulysses_degree: ulysses (DeepSpeed) degree. ``ring * ulysses ==
            gpus`` must hold.
        use_cfg_parallel: also split the classifier-free-guidance pass
            (extra 2x potential when guidance is enabled, but requires
            even-numbered ``gpus``).
        fallback: behaviour when the multi-GPU worker doesn't have xDiT
            installed or has fewer visible CUDA devices than ``gpus``.
            ``"passthrough"`` returns ``pipe`` unchanged with a warning;
            ``"raise"`` raises :class:`ParallelismUnavailableError`.

    Requires (lazy-imported here):
        pip install xfuser  # Apache 2.0 — xDiT's PyPI name
    """
    if gpus < 1:
        raise ValueError(f"gpus must be >= 1, got {gpus}")
    if gpus == 1:
        return pipe

    visible = _detect_gpu_count()
    if visible and visible < gpus:
        msg = (
            f"xdit_sequence_parallel: requested gpus={gpus} but only "
            f"{visible} CUDA devices visible. Falling back to '{fallback}'."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        if fallback == "raise":
            raise ParallelismUnavailableError(msg)
        return pipe

    # Resolve ring / ulysses split.
    if ring_degree is None and ulysses_degree is None:
        # Default to pure ring parallelism — easiest to drive.
        ring_degree = gpus
        ulysses_degree = 1
    elif ring_degree is None:
        assert ulysses_degree is not None
        ring_degree = gpus // ulysses_degree
    elif ulysses_degree is None:
        ulysses_degree = gpus // ring_degree
    assert ring_degree is not None and ulysses_degree is not None
    if ring_degree * ulysses_degree != gpus:
        raise ValueError(
            f"ring_degree * ulysses_degree must equal gpus, got "
            f"{ring_degree} * {ulysses_degree} != {gpus}"
        )

    try:
        from xfuser import (
            xFuserArgs,
            xFuserPipelineWrapper,
        )
    except ImportError:
        try:
            # Older API surface.
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            import torch.distributed as dist
        except ImportError as e:
            msg = (
                "xdit_sequence_parallel requires `xfuser` (xDiT). Install "
                "with `pip install xfuser` (Apache 2.0) and rebuild the "
                "endpoint image."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning("%s (%r)", msg, e)
            if fallback == "raise":
                raise ParallelismUnavailableError(msg) from e
            return pipe

        # Bootstrap distributed if not already initialized.
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("WORLD_SIZE", str(gpus))
            os.environ.setdefault("RANK", "0")
            init_distributed_environment()
        initialize_model_parallel(
            sequence_parallel_degree=gpus,
            ring_degree=ring_degree,
            ulysses_degree=ulysses_degree,
        )
        logger.info(
            "xdit_sequence_parallel: initialized USP (gpus=%s, ring=%s, ulysses=%s)",
            gpus,
            ring_degree,
            ulysses_degree,
        )
        # Older xfuser monkey-patches the pipe in-place at this point via
        # per-model adapters loaded elsewhere; we return the pipe so callers
        # can keep their reference.
        return pipe

    # New-style xFuser: build args + wrap.
    args = xFuserArgs(
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
        use_cfg_parallel=use_cfg_parallel,
    )
    wrapped = xFuserPipelineWrapper(pipe, args)
    logger.info(
        "xdit_sequence_parallel: wrapped pipeline (gpus=%s, ring=%s, ulysses=%s, cfg_parallel=%s)",
        gpus,
        ring_degree,
        ulysses_degree,
        use_cfg_parallel,
    )
    return wrapped


# Class-shaped alias for code that prefers explicit construction.
class SequenceParallel:
    """Class-shaped wrapper around :func:`xdit_sequence_parallel`.

    Mirrors the cache helpers' ``X(...).apply(pipe)`` style for consistency.
    """

    def __init__(
        self,
        gpus: int,
        *,
        ring_degree: int | None = None,
        ulysses_degree: int | None = None,
        use_cfg_parallel: bool = False,
        fallback: str = "passthrough",
    ) -> None:
        self.gpus = gpus
        self.ring_degree = ring_degree
        self.ulysses_degree = ulysses_degree
        self.use_cfg_parallel = use_cfg_parallel
        self.fallback = fallback

    def apply(self, pipe: Any) -> Any:
        return xdit_sequence_parallel(
            pipe,
            gpus=self.gpus,
            ring_degree=self.ring_degree,
            ulysses_degree=self.ulysses_degree,
            use_cfg_parallel=self.use_cfg_parallel,
            fallback=self.fallback,
        )


__all__ = [
    "xdit_sequence_parallel",
    "SequenceParallel",
    "ParallelismUnavailableError",
]
