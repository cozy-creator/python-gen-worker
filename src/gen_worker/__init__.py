"""Worker-author API for gen-worker.

Keep this surface intentionally small. Endpoint code that needs advanced
subsystems should import their explicit modules, for example
``gen_worker.trainer`` or ``gen_worker.conversion``.

Context types (issue #1: slim-request-context):
  - ``RequestContext`` — inference handlers (the base type).
  - ``ConversionContext`` — ``@training_function`` handlers that produce a
    new repo revision (format-conversion, quantization, fine-tuning, …).
    Adds ``publish_repo_revision`` / ``materialize_blob`` /
    ``read_repo_metadata`` / ``write_repo_metadata`` plus the conversion
    helper API (``mktemp``, ``open_output_writer``, …).
  - ``DatasetContext`` — ``@training_function(kind="dataset-generation")``
    handlers. Adds ``publish_dataset_revision`` / ``resolve_dataset``.
  - ``TrainingContext`` — trainer-class endpoints. Adds repo-metadata RPCs.

All three subclass ``RequestContext``; the kind-specific subclass is
constructed by the worker before dispatch based on the endpoint kind.

Bindings (issue #9: decorator-table-model-bindings):
  - ``Repo(ref)`` — module-level repo handle and fixed-pick binding.
  - ``dispatch(field, table)`` — payload-driven dispatch binding.
  - ``Resources(...)`` — per-function hardware envelope + cost shape.
  - Both ``Repo`` and ``Dispatch`` support ``.allow_override(*classes)``
    to permit caller-supplied substitutions inside an explicit pipeline
    class allowlist.
"""

from . import io
from .api.binding import Binding, Dispatch, Repo, dispatch
from .api.decorators import (
    Resources,
    inference_function,
)
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)
from .api.errors import (
    AuthError,
    CanceledError,
    FatalError,
    InputTooLargeError,
    OutputTooLargeError,
    RefCompatibilitySurprise,
    ResourceError,
    RetryableError,
    ValidationError,
    WorkerError,
)
from .api.types import Asset, Compute, LoraSpec, Tensors
from .api.payload_constraints import Clamp
from .api.streaming import iter_transformers_text_deltas
from .utils.lora import load_loras
from .inference_memory import apply_low_vram_config, with_oom_retry


_REMOVED_PUBLIC_SYMBOLS = {
    "ModelRef": (
        "gen_worker.ModelRef was removed in gen-worker 0.7.0. Use Repo / dispatch / "
        "the models={...} kwarg on @inference_function instead. See `progress.json` "
        "issue #9 (decorator-table-model-bindings)."
    ),
    "ModelRefSource": (
        "gen_worker.ModelRefSource was removed in gen-worker 0.7.0. The ModelRef "
        "concept was replaced by Repo / Dispatch bindings on the decorator's "
        "models={...} kwarg. See `progress.json` issue #9."
    ),
    "Src": (
        "gen_worker.Src was removed in gen-worker 0.7.0. The Src.FIXED / Src.PAYLOAD / "
        "Src.PAYLOAD_REF discriminators are gone — fixed bindings use Repo(...), "
        "payload-driven dispatch uses dispatch(field, table), and caller overrides "
        "use .allow_override(*classes)."
    ),
    "ResourceRequirements": (
        "gen_worker.ResourceRequirements was renamed to gen_worker.Resources in "
        "gen-worker 0.7.0. The five ScalingHints fields (vram_must_fit, vram_base, "
        "vram_size_multiplier, vram_scales_with, runtime_scales_with) are now part "
        "of the same struct. The `kind` field was dropped."
    ),
    "ScalingHints": (
        "gen_worker.ScalingHints was merged into gen_worker.Resources in gen-worker "
        "0.7.0. Pass vram_must_fit / vram_base / vram_size_multiplier / "
        "vram_scales_with / runtime_scales_with directly on Resources(...)."
    ),
}


def __getattr__(name: str):
    if name == "clone":
        import importlib

        return importlib.import_module(".clone", __name__)
    if name in _REMOVED_PUBLIC_SYMBOLS:
        raise ImportError(_REMOVED_PUBLIC_SYMBOLS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Decorators + binding model.
    "inference_function",
    "Resources",
    "Repo",
    "Dispatch",
    "Binding",
    "dispatch",
    # Context types.
    "RequestContext",
    "ConversionContext",
    "DatasetContext",
    "TrainingContext",
    # Errors.
    "AuthError",
    "CanceledError",
    "RetryableError",
    "ResourceError",
    "ValidationError",
    "FatalError",
    "InputTooLargeError",
    "OutputTooLargeError",
    "RefCompatibilitySurprise",
    "WorkerError",
    # Payload + media helpers.
    "Asset",
    "Compute",
    "Tensors",
    "LoraSpec",
    "Clamp",
    "iter_transformers_text_deltas",
    "load_loras",
    "apply_low_vram_config",
    "with_oom_retry",
    "clone",
    "io",
]
