"""Mint, store, and adopt PyTorch AOTInductor graphs.

Five modules, and each answers one question:

* :mod:`torchcg.identity` -- what a compiled graph IS. One graph hash, one key.
* :mod:`torchcg.mint`     -- making the artifact: bind, compile, package.
* :mod:`torchcg.store`    -- where it lives: content-addressed, mint-once.
* :mod:`torchcg.adopt`    -- serving it: load, verify, dispatch-or-eager.
* :mod:`torchcg.refuse`   -- every way this can say no.

Torch does the compiling. What this library owns is identity, storage and
adoption -- and the refusals that keep an artifact from claiming to be
something it is not.
"""

from __future__ import annotations

from . import adopt, identity, mint, refuse, store  # noqa: F401
from .adopt import Dispatcher, Recast, Record, fit
from .identity import (
    GRAPH_SCHEME,
    KEY_SCHEME,
    ArtifactKey,
    CallIngress,
    CallInput,
    LayoutMorphism,
    artifact_key,
    build_call_ingress,
    canonical_graph,
    compile_stack,
    contiguous_handle,
    graph_hash,
    is_artifact_key,
    is_graph_hash,
    require_morphism,
)
from .mint import (
    GraphSpec,
    Minted,
    bind_static_spec,
    compile_policy,
    strip_diagnostics,
)
from .refuse import (
    AdoptError,
    BindError,
    DroppedOptimization,
    IdentityError,
    IngressError,
    KeyAlreadyMinted,
    KeyMismatch,
    LayoutError,
    LayoutUndeliverableError,
    MintError,
    RangeNarrowed,
    StoreError,
    TorchCGError,
)
from .store import Store, StoredArtifact

__all__ = [
    "AdoptError",
    "ArtifactKey",
    "BindError",
    "CallIngress",
    "CallInput",
    "Dispatcher",
    "KeyAlreadyMinted",
    "DroppedOptimization",
    "GRAPH_SCHEME",
    "GraphSpec",
    "IdentityError",
    "IngressError",
    "KEY_SCHEME",
    "KeyMismatch",
    "LayoutError",
    "LayoutMorphism",
    "LayoutUndeliverableError",
    "MintError",
    "Minted",
    "RangeNarrowed",
    "Recast",
    "Record",
    "Store",
    "StoreError",
    "StoredArtifact",
    "TorchCGError",
    "adopt",
    "artifact_key",
    "bind_static_spec",
    "build_call_ingress",
    "canonical_graph",
    "compile_stack",
    "compile_policy",
    "contiguous_handle",
    "fit",
    "graph_hash",
    "identity",
    "is_artifact_key",
    "is_graph_hash",
    "mint",
    "refuse",
    "require_morphism",
    "store",
    "strip_diagnostics",
]
