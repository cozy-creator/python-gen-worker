"""Mint and reuse verified PyTorch AOTInductor graphs through tensorfs."""

from .adopt import (
    AdoptError,
    AdoptSession,
    ArtifactLoader,
    Hole,
)
from .artifact import ARTIFACT_METADATA_FIELDS, COMPILED_GRAPH_FORMAT, ArtifactError
from .compiler import CompileError
from .declaration import (
    DeclarationError,
    GraphClassDeclaration,
    GraphClassSpec,
    RuntimeCompatibility,
)
from .discovery import DiscoveryError, discover_lane, discover_modules
from .document import (
    DOCUMENT_FORMAT,
    DocumentError,
    GraphRecord,
    GraphSetDocument,
    LaneGraphs,
)
from .engine import (
    AdmissionError,
    Engine,
    EnsureOutcome,
    EnsureResult,
)

# `torchcg.hollow` (the weights-free derive session, tcg#45) is DELIBERATELY
# not re-exported here: it is the PUBLISH-time surface and it names model
# libraries (diffusers/transformers loader interception), so importing it from
# the package root would put those libraries on every consumer's serve-role
# import closure (pgw#1331's fence measured exactly that). The derive imports
# `torchcg.hollow` by its own name.
from .graph_identity import (
    ENV_SCHEME,
    GRAPH_SCHEME,
    EnvIdentity,
    GraphIdentityError,
    closure_hash,
    graph_hash,
    installed_closure,
    is_graph_hash,
)
from .identity import (
    ARTIFACT_KIND,
    GRAPH_CLASS_BLOCK,
    REQUIRED_AXES,
    CompiledGraphKey,
    IdentityError,
    is_compiled_graph_key,
)
from .ingress import (
    CallIngress,
    CallInput,
    IngressError,
    build_call_ingress,
    exported_input_name,
)
from .lane import (
    LaneError,
    LaneRef,
    require_contract_ref,
    require_targets,
    resolve_target,
)
from .requirements import (
    ArtifactCandidate,
    EnvironmentMismatch,
    RequirementsError,
    RequirementsManifest,
    assert_exact_env,
    rank,
)
from .runner import CompiledGraphRunner, ConstantBindingError
from .selection import (
    ClassReport,
    FeedNormalization,
    GraphClassCandidate,
    IngressMiss,
    MissReason,
    NormalizationKind,
    PresentedCall,
    PresentedValue,
    RealignReason,
    Selection,
    SelectionError,
    SelectionOutcome,
    describe_call,
    select,
)
from .storage import (
    QuarantinedArtifact,
    StorageError,
    StoredCompiledGraph,
    StoreOutcome,
    StoreResult,
)
from .store import GraphStore, LocalGraphStore, PublishOutcome, StoreError, holes

__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_METADATA_FIELDS",
    "AdmissionError",
    "AdoptError",
    "AdoptSession",
    "ArtifactCandidate",
    "ArtifactError",
    "ArtifactLoader",
    "CompileError",
    "COMPILED_GRAPH_FORMAT",
    "DOCUMENT_FORMAT",
    "DiscoveryError",
    "DocumentError",
    "ENV_SCHEME",
    "EnvIdentity",
    "EnvironmentMismatch",
    "LaneRef",
    "GRAPH_SCHEME",
    "GraphIdentityError",
    "GraphRecord",
    "GraphSetDocument",
    "GraphStore",
    "Hole",
    "LaneError",
    "LaneGraphs",
    "LocalGraphStore",
    "PublishOutcome",
    "RequirementsError",
    "RequirementsManifest",
    "StoreError",
    "assert_exact_env",
    "closure_hash",
    "discover_lane",
    "discover_modules",
    "graph_hash",
    "holes",
    "installed_closure",
    "is_graph_hash",
    "rank",
    "require_contract_ref",
    "require_targets",
    "resolve_target",
    "CompiledGraphKey",
    "CompiledGraphRunner",
    "CallIngress",
    "CallInput",
    "ClassReport",
    "ConstantBindingError",
    "DeclarationError",
    "Engine",
    "EnsureOutcome",
    "EnsureResult",
    "FeedNormalization",
    "GRAPH_CLASS_BLOCK",
    "GraphClassCandidate",
    "GraphClassDeclaration",
    "GraphClassSpec",
    "IdentityError",
    "REQUIRED_AXES",
    "IngressError",
    "IngressMiss",
    "MissReason",
    "NormalizationKind",
    "PresentedCall",
    "PresentedValue",
    "QuarantinedArtifact",
    "RealignReason",
    "RuntimeCompatibility",
    "Selection",
    "SelectionError",
    "SelectionOutcome",
    "StorageError",
    "StoreOutcome",
    "StoreResult",
    "StoredCompiledGraph",
    "build_call_ingress",
    "describe_call",
    "exported_input_name",
    "is_compiled_graph_key",
    "select",
]
