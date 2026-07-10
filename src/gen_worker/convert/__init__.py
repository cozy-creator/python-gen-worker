"""gen_worker.convert — Cozy Creator's model ETL (hub ingest, dtype cast / quant, repackage, publish).

Tenant SDK (conversion endpoints)::

    from gen_worker.convert import Source, ProducedFlavor, Dataset

Clone / mirror::

    from gen_worker.convert import clone
    result = clone.from_huggingface(ctx, payload)

Heavy deps (torch/safetensors) are imported lazily by the modules that need
them; ``import gen_worker.convert`` stays cheap.
"""

from __future__ import annotations

from .calibration import CalibrationAction, CalibrationPolicy, resolve_calibration_action
from .classifier import RepoClassification, RepoRefusal, classify_repo
from .clone import (
    CloneResult,
    OutputSpec,
    from_civitai,
    from_huggingface,
    run_clone,
)
from .component import Component
from .dataset import Dataset
from .hub import CommitFile, CommitResult, HubClient, HubPublishError
from .ingest import IngestedSource, ingest_civitai, ingest_huggingface
from .loaded_component import LoadedComponent
from .produced import ProducedFlavor
from .publish import publish_flavors
from .source import FileLayout, Source
from .svdq import build_svdq_flavor_tree, fetch_svdq_checkpoint, svdq_flavor_label
from .writer import (
    FP8_TE_COMPONENTS,
    streaming_cast_snapshot,
    streaming_dtype_cast,
    streaming_fp8_snapshot,
    streaming_fp8_storage_cast,
    streaming_fp8_te_cast,
    te_fp8_castable_keys,
)

# `gen_worker.convert.clone` module alias (clone.from_huggingface style).
from . import clone

__all__ = [
    "build_svdq_flavor_tree",
    "fetch_svdq_checkpoint",
    "svdq_flavor_label",
    # Tenant SDK
    "Source",
    "Component",
    "LoadedComponent",
    "FileLayout",
    "Dataset",
    "ProducedFlavor",
    "streaming_cast_snapshot",
    "streaming_dtype_cast",
    "streaming_fp8_snapshot",
    "streaming_fp8_storage_cast",
    "streaming_fp8_te_cast",
    "te_fp8_castable_keys",
    "FP8_TE_COMPONENTS",
    "CalibrationAction",
    "CalibrationPolicy",
    "resolve_calibration_action",
    # Ingest + classify
    "IngestedSource",
    "ingest_huggingface",
    "ingest_civitai",
    "RepoClassification",
    "RepoRefusal",
    "classify_repo",
    # Clone + publish
    "clone",
    "CloneResult",
    "OutputSpec",
    "run_clone",
    "from_huggingface",
    "from_civitai",
    "publish_flavors",
    "HubClient",
    "HubPublishError",
    "CommitFile",
    "CommitResult",
]
