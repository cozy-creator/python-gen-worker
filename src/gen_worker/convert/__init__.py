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

from .base_model_families import (
    CIVITAI,
    civitai_to_family,
    declare_foreign_family_map,
    foreign_to_family,
)
from .calibration import CalibrationAction, CalibrationPolicy, resolve_calibration_action
from .classifier import (
    RepoClassification,
    RepoRefusal,
    SourceIncludeError,
    apply_source_include,
    classify_repo,
)
from .clone import (
    CloneResult,
    OutputSpec,
    from_civitai,
    from_huggingface,
    normalize_source_include,
    run_clone,
)
from .component import Component
from .dataset import Dataset, write_jsonl_shard
from .hub import CommitFile, CommitResult, HubClient, HubPublishError
from .ingest import IngestedSource, ingest_civitai, ingest_huggingface
from .layout_spec import DirMatch, HintMatch, LayoutDeclaration
from .loaded_component import LoadedComponent
from .produced import ProducedFlavor
from .registry import (
    UnknownFamilyError,
    load_declaration_module,
    normalize_family,
    register_layout,
    register_repackage_family,
    registered_layouts,
    registered_repackage_families,
    repackage_family,
    require_repackage_family,
)
from .repack_spec import (
    ComponentRepack,
    DeclarationError,
    DtypePolicy,
    LayoutSignature,
    RenameRule,
    RepackVariant,
    RepackageFamily,
    SinglefileTarget,
)
from .publish import publish_flavors
from .source import FileLayout, Source
from .svdq import build_svdq_flavor_tree, fetch_svdq_checkpoint, svdq_flavor_label
from .writer import (
    fp8_te_components,
    streaming_cast_snapshot,
    streaming_dtype_cast,
    streaming_fp8_snapshot,
    streaming_fp8_storage_cast,
    streaming_fp8_te_cast,
    streaming_w8a8_cast,
    streaming_w8a8_snapshot,
    te_fp8_castable_keys,
    verify_w8a8_snapshot,
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
    "write_jsonl_shard",
    "ProducedFlavor",
    "streaming_cast_snapshot",
    "streaming_dtype_cast",
    "streaming_fp8_snapshot",
    "streaming_fp8_storage_cast",
    "streaming_fp8_te_cast",
    "te_fp8_castable_keys",
    "streaming_w8a8_cast",
    "streaming_w8a8_snapshot",
    "verify_w8a8_snapshot",
    "fp8_te_components",
    "CalibrationAction",
    "CalibrationPolicy",
    "resolve_calibration_action",
    # Ingest + classify
    "IngestedSource",
    "ingest_huggingface",
    "ingest_civitai",
    "RepoClassification",
    "RepoRefusal",
    "SourceIncludeError",
    "apply_source_include",
    "classify_repo",
    # Clone + publish
    "clone",
    "CloneResult",
    "OutputSpec",
    "run_clone",
    "from_huggingface",
    "from_civitai",
    "normalize_source_include",
    "publish_flavors",
    "HubClient",
    "HubPublishError",
    "CommitFile",
    "CommitResult",
    # Family declarations (pgw#740): the endpoint declares, the SDK executes.
    "CIVITAI",
    "ComponentRepack",
    "DeclarationError",
    "DirMatch",
    "DtypePolicy",
    "HintMatch",
    "LayoutDeclaration",
    "LayoutSignature",
    "RenameRule",
    "RepackVariant",
    "RepackageFamily",
    "SinglefileTarget",
    "UnknownFamilyError",
    "civitai_to_family",
    "declare_foreign_family_map",
    "foreign_to_family",
    "load_declaration_module",
    "normalize_family",
    "register_layout",
    "register_repackage_family",
    "registered_layouts",
    "registered_repackage_families",
    "repackage_family",
    "require_repackage_family",
]
