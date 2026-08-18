"""The native store->VRAM serving loader (pgw#1380).

``ctx.load(StableDiffusionXLPipeline)`` — one spelling, no ``torch_dtype=``
(the lane contract IS the dtype), no ``.to("cuda")`` (weights land on device),
and no file: a serving pytorch endpoint never materializes tensor bytes.

* :mod:`.skeleton` — the pipeline built from configs, parameters on meta.
* :mod:`.source` — the tensorfs byte-source seam (the torch boundary).
* :mod:`.staging` — the pinned ring and the copy stream.
* :mod:`.engine` — the file-order walk that fills the skeleton.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .engine import LoadError, LoadReport, NameMismatch, StreamingLoader
from .skeleton import Skeleton, SkeletonError
from .source import (
    BridgeWeightStore,
    NativeWeightStore,
    StreamedTensor,
    TensorStream,
    WeightStore,
    WeightStoreUnavailable,
    native_available,
    store_for,
)
from .staging import StagingPool

logger = logging.getLogger(__name__)


def engine_for(
    checkpoint_dir: Path | str,
    *,
    device: Any = "cuda",
    io: str = "buffered",
) -> Optional[StreamingLoader]:
    """The loader engine for a projected checkpoint tree, or ``None``.

    ``None`` is the honest answer for a tree with no chunk store behind it
    (a bare download, a fixture): there is nothing to stream from, so the
    caller keeps whatever path it had rather than being handed an engine
    that would refuse on first use.

    Binding one ARMS the no-fill defect signal: from here on a
    ``materialized_view`` call in this process is a bug, not a burn-down row
    (Paul's 2026-08-19 ruling narrowed tier 3 away from serving pytorch).
    """
    store = store_for(checkpoint_dir)
    if store is None:
        logger.info(
            "ctx.load: %s is not a projected snapshot tree — no chunk store "
            "to stream from, so no streaming engine is bound",
            checkpoint_dir,
        )
        return None
    from ...models import materialized_view

    materialized_view.no_fill_serving(True)
    logger.info(
        "ctx.load: streaming %s through the %s byte source",
        checkpoint_dir,
        getattr(store, "KIND", type(store).__name__),
    )
    return StreamingLoader(store, device=device, io=io)


__all__ = [
    "engine_for",
    "BridgeWeightStore",
    "LoadError",
    "LoadReport",
    "NameMismatch",
    "NativeWeightStore",
    "Skeleton",
    "SkeletonError",
    "StagingPool",
    "StreamedTensor",
    "StreamingLoader",
    "TensorStream",
    "WeightStore",
    "WeightStoreUnavailable",
    "native_available",
    "store_for",
]
