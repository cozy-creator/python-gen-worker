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
  - ``Repo(ref)`` — module-level handle for a **tensorhub** ref (default).
  - ``HFRepo(ref)`` — explicit huggingface ref. Use this instead of relying
    on a string prefix.
  - ``CivitaiRepo(ref)`` — explicit civitai ref by model id.
  - ``dispatch(field, table)`` — payload-driven dispatch binding.
  - ``Resources(...)`` — per-function hardware envelope + cost shape.
  - All bindings support ``.allow_override(*classes)`` to permit
    caller-supplied substitutions inside an explicit pipeline class allowlist.
"""

from . import io
from . import accel  # noqa: F401 — exposed as gen_worker.accel (#324)
from . import cache  # noqa: F401 — exposed as gen_worker.cache
from . import compile_helpers as compile  # noqa: F401 — exposed as gen_worker.compile
from . import parallelism  # noqa: F401 — exposed as gen_worker.parallelism
from . import quant  # noqa: F401 — exposed as gen_worker.quant
from .api.binding import Binding, CivitaiRepo, Dispatch, HFRepo, Repo, dispatch
from .api.decorators import (
    Resources,
    batched_inference,
    conversion,
    dataset,
    inference,
    inference_function,  # hard-cut migration stub (raises ImportError if called)
    realtime_function,   # hard-cut migration stub
    training,
    training_function,   # hard-cut migration stub
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
from .api.streaming import (
    Done,
    Error,
    IncrementalTokenDelta,
    TokenStreamSignal,
    iter_transformers_text_deltas,
)
from .utils.lora import load_loras
from .inference_memory import apply_low_vram_config, with_oom_retry


_REMOVED_PUBLIC_SYMBOLS = {
    "ModelRef": (
        "gen_worker.ModelRef was removed in gen-worker 0.7.0. Use Repo / dispatch / "
        "the models={...} kwarg on @inference instead. See `progress.json` "
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
    # Decorators + binding model (#322 class-only).
    "inference",
    "batched_inference",
    "training",
    "dataset",
    "conversion",
    "Resources",
    "Repo",
    "HFRepo",
    "CivitaiRepo",
    "Dispatch",
    "Binding",
    "dispatch",
    # Hard-cut migration stubs (raise ImportError if used).
    "inference_function",
    "training_function",
    "realtime_function",
    # Acceleration helpers (#324):
    "accel",
    "cache",
    "compile",
    "quant",
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
    "Done",
    "Error",
    "IncrementalTokenDelta",
    "TokenStreamSignal",
    "iter_transformers_text_deltas",
    "load_loras",
    "apply_low_vram_config",
    "with_oom_retry",
    "clone",
    "io",
]
