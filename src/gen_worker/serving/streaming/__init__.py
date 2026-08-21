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
    """The loader engine for a projected checkpoint tree, or ``None``."""
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
