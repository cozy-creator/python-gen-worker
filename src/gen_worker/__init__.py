"""Worker-author API for gen-worker.

Keep this surface intentionally small. Endpoint code that needs advanced
subsystems should import their explicit modules, for example
``gen_worker.trainer`` or ``gen_worker.conversion``.

Context types (issue #1: slim-request-context):
  - ``RequestContext`` — inference handlers (the base type).
  - ``ConversionContext`` — ``@conversion`` handlers that produce a
    new repo revision (format-conversion, quantization, fine-tuning, …).
    Adds ``publish_repo_revision`` / ``materialize_blob`` /
    ``read_repo_metadata`` / ``write_repo_metadata`` plus the conversion
    helper API (``mktemp``, ``open_output_writer``, …).
  - ``DatasetContext`` — ``@dataset``
    handlers. Adds ``publish_dataset_revision`` / ``resolve_dataset``.
  - ``TrainingContext`` — trainer-class endpoints. Adds repo-metadata RPCs.

All three subclass ``RequestContext``; the kind-specific subclass is
constructed by the worker before dispatch based on the endpoint kind.

Bindings (issue #9: decorator-table-model-bindings, #10: typed-provider-repo):
  - ``Repo(name, default_ref)`` — named tensorhub model slot with an initial ref.
  - ``HFRepo(name, default_ref)`` — named Hugging Face model slot.
  - ``CivitaiRepo(name, default_ref)`` — named Civitai model slot.
  - Legacy ``Repo(ref)`` / ``HFRepo(ref)`` / ``CivitaiRepo(ref)`` still works;
    discovery falls back to the model parameter name as the slot key.
  - ``dispatch(field, table)`` — payload-driven dispatch binding.
  - ``Resources(...)`` — per-function hardware envelope + cost shape.
  - All bindings support ``.allow_override(*classes)`` to permit
    caller-supplied substitutions inside an explicit pipeline class allowlist.
"""

from . import io
from .api.binding import (
    CivitaiRepo,
    Dispatch,
    HFRepo,
    ModelScopeRepo,
    Repo,
    SharedBase,
    Variant,
    dispatch,
)
from .api.decorators import (
    Case,
    Resources,
    conversion,
    dataset,
    inference,
    invocable,
    training,
)
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)
from .api.errors import (
    CanceledError,
    FatalError,
    RetryableError,
    ValidationError,
)
from .api.types import (
    Asset,
    AudioAsset,
    ExpectedOutput,
    ImageAsset,
    StringEnum,
    VideoAsset,
)
from .api.streaming import iter_transformers_text_deltas
from .models.memory import apply_low_vram_config
from .diagnostics import emit_diagnostic_log


def __getattr__(name: str):
    if name == "clone":
        import importlib

        return importlib.import_module(".clone", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Decorators + binding model (#322 class-only).
    "invocable",
    "inference",
    "training",
    "dataset",
    "conversion",
    "Resources",
    "Case",
    "Repo",
    "HFRepo",
    "CivitaiRepo",
    "ModelScopeRepo",
    "Dispatch",
    "dispatch",
    "SharedBase",
    "Variant",
    # Context types.
    "RequestContext",
    "ConversionContext",
    "DatasetContext",
    "TrainingContext",
    # Errors.
    "CanceledError",
    "RetryableError",
    "ValidationError",
    "FatalError",
    # Payload + media helpers.
    "Asset",
    "AudioAsset",
    "ExpectedOutput",
    "ImageAsset",
    "StringEnum",
    "VideoAsset",
    "iter_transformers_text_deltas",
    "apply_low_vram_config",
    "emit_diagnostic_log",
    "clone",
    "io",
]
