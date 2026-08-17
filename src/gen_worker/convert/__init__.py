"""gen_worker.convert — Cozy Creator's model ETL (hub ingest, dtype cast / quant, repackage, publish).

Tenant SDK (conversion endpoints)::

    from gen_worker.convert import Source, ProducedFlavor, Dataset

Clone / mirror::

    from gen_worker.convert import clone
    result = clone.from_huggingface(ctx, payload)

Heavy deps (torch/safetensors) are imported lazily by the modules that need
them; ``import gen_worker.convert`` stays cheap. Anything not re-exported
here is still importable from its defining submodule.
"""

from __future__ import annotations

from .base_model_families import CIVITAI, civitai_to_family, declare_foreign_family_map
from .calibration import resolve_calibration_action
from .clone import CloneResult
from .dataset import Dataset, write_jsonl_shard
from .ingest import ingest_civitai, ingest_huggingface
from .layout_converters import (
    ConversionCase,
    ConversionHop,
    ConversionIO,
    ConversionPlan,
    ConversionProofError,
    ConversionResult,
    CorpusTensor,
    LayoutProduction,
    LayoutRung,
    LayoutVerdict,
    QuantRepack,
    TopologyConversion,
    classify_layout,
    conversion_provenance,
    derived_artifact_identity,
    plan_layout_conversions,
    register_layout_conversion,
    register_layout_production,
    registered_layout_conversions,
    registered_layout_productions,
    run_layout_conversion,
)
from .layout_spec import DirMatch, HintMatch, LayoutDeclaration
from .produced import ProducedFlavor
from .registry import (
    UnknownFamilyError,
    load_declaration_module,
    normalize_family,
    register_layout,
    register_repackage_family,
    registered_layouts,
    registered_repackage_families,
    require_repackage_family,
)
from .repack_spec import (
    ComponentRepack,
    LayoutSignature,
    RenameRule,
    RepackVariant,
    RepackageFamily,
    SinglefileTarget,
)
from .publish import PrecisionClassRefusal, publish_flavors
from .source import Source
from .svdq import build_svdq_flavor_tree, fetch_svdq_checkpoint
from .writer import (
    fp8_te_components,
    streaming_cast_snapshot,
    streaming_fp8_snapshot,
    streaming_w8a8_snapshot,
    verify_w8a8_snapshot,
)

# `gen_worker.convert.clone` module alias (clone.from_huggingface style).
from . import clone

__all__ = [
    "build_svdq_flavor_tree",
    "fetch_svdq_checkpoint",
    # Tenant SDK
    "Source",
    "Dataset",
    "write_jsonl_shard",
    "ProducedFlavor",
    "streaming_cast_snapshot",
    "streaming_fp8_snapshot",
    "streaming_w8a8_snapshot",
    "verify_w8a8_snapshot",
    "fp8_te_components",
    "resolve_calibration_action",
    # Ingest + clone + publish
    "ingest_huggingface",
    "ingest_civitai",
    "clone",
    "CloneResult",
    "PrecisionClassRefusal",
    "publish_flavors",
    # Family declarations: the endpoint declares, the SDK executes.
    "CIVITAI",
    "ComponentRepack",
    "DirMatch",
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
    "load_declaration_module",
    "normalize_family",
    "register_layout",
    "register_repackage_family",
    "registered_layouts",
    "registered_repackage_families",
    "require_repackage_family",
    # §1.33: the layout converter registry — the CONVERTIBLE rung.
    "ConversionCase",
    "ConversionHop",
    "ConversionIO",
    "ConversionPlan",
    "ConversionProofError",
    "ConversionResult",
    "CorpusTensor",
    "LayoutProduction",
    "LayoutRung",
    "LayoutVerdict",
    "QuantRepack",
    "TopologyConversion",
    "classify_layout",
    "conversion_provenance",
    "derived_artifact_identity",
    "plan_layout_conversions",
    "register_layout_conversion",
    "register_layout_production",
    "registered_layout_conversions",
    "registered_layout_productions",
    "run_layout_conversion",
]
