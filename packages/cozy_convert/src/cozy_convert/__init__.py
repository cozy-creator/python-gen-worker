"""cozy_convert — Cozy Creator's model ETL (split out of gen_worker, #367).

Tenant SDK (conversion endpoints)::

    from cozy_convert import Source, StreamingWriter, ProducedFlavor, Dataset

Clone / mirror::

    from cozy_convert import clone
    result = clone.from_huggingface(ctx, payload)

Heavy deps (torch/safetensors) are imported lazily by the modules that need
them; ``import cozy_convert`` stays cheap.
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
from .writer import StreamingWriter

# `cozy_convert.clone` module alias (clone.from_huggingface style).
from . import clone

__all__ = [
    # Tenant SDK
    "Source",
    "Component",
    "LoadedComponent",
    "FileLayout",
    "Dataset",
    "ProducedFlavor",
    "StreamingWriter",
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
