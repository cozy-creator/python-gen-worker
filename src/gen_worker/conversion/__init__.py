"""gen_worker.conversion — tenant API for transform-kind endpoints.

Tenant authors write functions like:

    from gen_worker.conversion import (
        conversion_function, ConversionContext, Source, Dataset, ProducedVariant,
    )

    @conversion_function
    def convert_dtype(ctx: ConversionContext, source: Source, specs: list[DTypeSpec]):
        ...

See e2e progress.json issue #5 for the full contract:
  - Reserved names (ctx, source, datasets) bound to library-injected types.
  - Everything else decoded from wire payload by name via msgspec.
  - Secondary-model loads via Annotated[Source, ModelRef(Src.PAYLOAD, '...')].
  - Return list[ProducedVariant]; library handles upload + destination.tags.
"""

from __future__ import annotations

from .component import Component
from .context import ConversionContext
from .core_types import (
    ConversionArtifact,
    ConversionOutput,
    IngestResult,
    tensors_with,
)
from .dataset import Dataset
from .dispatch import (
    DEFAULT_KIND,
    RECOMMENDED_KINDS,
    ConversionFunctionSpec,
    conversion_function,
)
from .produced import ProducedVariant
from .safetensors_io import (
    materialize_safetensors_input,
    persist_safetensors_output,
)
from .source import FileLayout, Source
from .validation import (
    ValidationReport,
    ValidationViolation,
    format_report,
    validate_transform_module,
)
from .writer import StreamingWriter

__all__ = [
    # Tenant-facing contract types
    "Component",
    "ConversionContext",
    "ConversionFunctionSpec",
    "DEFAULT_KIND",
    "RECOMMENDED_KINDS",
    "Dataset",
    "FileLayout",
    "ProducedVariant",
    "Source",
    "StreamingWriter",
    "conversion_function",
    # Publish-time validation
    "ValidationReport",
    "ValidationViolation",
    "format_report",
    "validate_transform_module",
    # Legacy / shared core types (used by clone_pipeline + any tenant that
    # wants richer multi-artifact outputs or dict-metadata returns)
    "ConversionArtifact",
    "ConversionOutput",
    "IngestResult",
    "tensors_with",
    # Path-in-path-out safetensors primitives (for non-@conversion_function
    # flows like clone_pipeline's external-URL ingest)
    "materialize_safetensors_input",
    "persist_safetensors_output",
    # Library submodules (heavy imports — available via gen_worker.conversion.<module>)
    #   ingest, layout, dtype_utils, gguf_utils, repackage, streaming_primitives,
    #   safetensors_io — not re-exported here to avoid eager-importing torch etc.
]
