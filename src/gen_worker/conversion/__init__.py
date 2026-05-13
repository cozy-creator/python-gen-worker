"""gen_worker.conversion — tenant API for training-kind endpoints.

Tenant authors write functions like:

    from gen_worker import (
        training_function, ConversionContext, Source, Dataset, ProducedFlavor,
    )

    @training_function
    def convert_weights(ctx: ConversionContext, source: Source, specs: list[DtypeCastSpec]):
        ...

Contract summary:
  - Reserved names (ctx, source, datasets) bound to library-injected types.
  - Everything else decoded from wire payload by name via msgspec.
  - Secondary-model loads via the training-side `_PayloadRef` annotation
    (private; declared on extra Source-typed parameters).
  - Return list[ProducedFlavor]; library handles upload + destination.tags.

Boundary summary:
  - This package owns generic primitives and metadata.
  - Endpoint repos own product conversion functions and calibrated
    quantization workflows such as modelopt.
"""

from __future__ import annotations

from .calibration import (
    CalibrationAction,
    CalibrationPolicy,
    resolve_calibration_action,
)
from .component import Component
from .context import ConversionContext
from .dataset import Dataset
from .dispatch import (
    TrainingFunctionSpec,
    training_function,
)
from .produced import ProducedFlavor
from .source import FileLayout, Source
from .writer import StreamingWriter

__all__ = [
    "Component",
    "ConversionContext",
    "TrainingFunctionSpec",
    "Dataset",
    "FileLayout",
    "ProducedFlavor",
    "Source",
    "StreamingWriter",
    "training_function",
    "CalibrationAction",
    "CalibrationPolicy",
    "resolve_calibration_action",
]
