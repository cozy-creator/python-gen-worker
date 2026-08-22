from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .census import (
    CENSUS_KIND,
    Census,
    CensusError,
    CensusMismatch,
    ComponentCensus,
    TensorRow,
)
from .engine import LoadError, LoadReport, NameMismatch, StreamingLoader
from .fill_client import Destination
from .skeleton import Skeleton, SkeletonError
from .source import (
    NativeWeightStore,
    StreamedTensor,
    TensorStream,
    WeightStore,
    WeightStoreUnavailable,
    store_for,
)

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
    "CENSUS_KIND",
    "Census",
    "CensusError",
    "CensusMismatch",
    "ComponentCensus",
    "Destination",
    "TensorRow",
    "LoadError",
    "LoadReport",
    "NameMismatch",
    "NativeWeightStore",
    "Skeleton",
    "SkeletonError",
    "StreamedTensor",
    "StreamingLoader",
    "TensorStream",
    "WeightStore",
    "WeightStoreUnavailable",
    "store_for",
]
