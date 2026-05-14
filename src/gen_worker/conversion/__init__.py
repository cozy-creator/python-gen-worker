"""gen_worker.conversion — tenant API for transform-kind endpoints.

Tenant authors write a class:

    from gen_worker import conversion
    from gen_worker.conversion import ConversionContext, Source, ProducedFlavor

    @conversion(sub_kind="format-conversion", models={"pipe": Repo(...)})
    class CastDtype:
        def setup(self): ...

        @conversion.function
        def cast_dtype(self, ctx: ConversionContext, payload: CastDtypeInput)
            -> Iterator[ProducedFlavor]:
            ...

        def shutdown(self): ...

Contract summary:
  - Method signature is ``(self, ctx, payload)``; the worker decodes the
    wire payload into the method's msgspec.Struct type.
  - Source materialization is reserved-name: the worker resolves
    ``payload.source`` to a local snapshot dir on ``ctx.source_path``,
    and the body constructs a ``Source(...)`` handle from it.
  - Return ``Iterator[ProducedFlavor]`` or ``list[ProducedFlavor]``; the
    library handles upload + destination.tags + revision publish.

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
from .produced import ProducedFlavor
from .source import FileLayout, Source
from .writer import StreamingWriter

__all__ = [
    "Component",
    "ConversionContext",
    "Dataset",
    "FileLayout",
    "ProducedFlavor",
    "Source",
    "StreamingWriter",
    "CalibrationAction",
    "CalibrationPolicy",
    "resolve_calibration_action",
]
