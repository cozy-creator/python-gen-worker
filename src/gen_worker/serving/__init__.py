"""The ship-code-as-is serving layer (pgw#1382 split; program pgw#1367).

Authors ship code AS-IS, split along state (pgw#1382):

* ``Model[MT]`` — the STATEFUL class (weights, lanes, defaults, caches,
  lifecycle, single-flight). The class header is the declaration surface:
  model type via the generic parameter, weight-format lanes via ``lanes=``.
* ``@entrypoint`` functions — STATELESS module-level request handlers
  declaring their models (and adapters) by parameter annotation
  (ctx-first: ``(ctx, payload, model, ...)``).
* "endpoint" survives as the deployable-unit NOUN: the module named by
  ``endpoint.toml``'s ``main =`` — its entrypoints plus the Model classes
  they reference. No catalog, no ModelSpec, no codegen.

The ctx splits with the state: ``LoadContext`` (checkpoint tree, lane,
``load``/``compile``/``defaults`` seams) rides ``Model.load``/``unload``;
``RequestContext`` (request facts + the salvaged base surface) rides
entrypoints. The EAGER path works standalone and first; adoption of
compiled graphs swaps module forwards on the author's own objects (torchcg
``AdoptSession`` behind ``ctx.compile``) and hands the ordered hole list to
the background mint (pgw#1371).
"""

from .context import (
    Adapter,
    DefaultsError,
    DistillationAdapter,
    DeployBinding,
    LoadContext,
    LoaderEngine,
    RequestContext,
)
from .engine_runtime import (
    EngineBootError,
    EngineCommand,
    EngineHandle,
    EngineSpec,
    LlamaServer,
    VllmServer,
    boot_engine,
)
from .envelope import (
    AdapterRow,
    DecodedRequest,
    EnvelopeError,
    decode_envelope,
)
from .entrypoints import (
    ENTRYPOINT_ATTR,
    EntrypointDeclarationError,
    EntrypointSpec,
    SlotSpec,
    entrypoint,
)
from .host import EndpointHost, ModelInstance, ServeDispatchError
from .mint import (
    BackgroundMint,
    MintCondemned,
    MintFailure,
    MintOutcome,
    MintProgress,
    MintedHole,
    MissingProgram,
    mint_holes,
)
from .loader import (
    EndpointLoadError,
    LoadedEndpoint,
    load_endpoint,
    load_endpoint_module,
)
from .self_mint import SelfMint, SelfMintStatus
from .serve_adoption import ServeAdoption
from .residency import (
    Charge,
    InstanceSizer,
    Lease,
    ModelBackend,
    NeverFits,
    ResidencyError,
    ResidencyManager,
    admission_charge,
)
from .serve_loop import (
    BindingResolver,
    InvokeOutcome,
    ServeLoop,
    manifest_sizer,
)
from .lane_spec import (
    DYNAMIC,
    STATIC,
    DeclaredLane,
    LaneDeclarationError,
    LaneSpec,
    Structural,
    lane,
)
from .model import (
    LaneContract,
    Model,
    ModelDeclarationError,
    lane_handle,
    model_declared_lanes,
    model_lane_spec,
    model_lanes,
    model_marks_compile,
    model_requires,
    model_shapes,
    model_structural,
    model_type,
)
from .placement import (
    DeviceFacts,
    Shortfall,
    device_facts,
    shortfalls,
    warn_if_degraded,
)

__all__ = [
    "boot_engine",
    "VllmServer",
    "LlamaServer",
    "EngineSpec",
    "EngineHandle",
    "EngineCommand",
    "EngineBootError",
    "DeviceFacts",
    "Shortfall",
    "device_facts",
    "shortfalls",
    "warn_if_degraded",
    "manifest_sizer",
    "decode_envelope",
    "ServeLoop",
    "ResidencyManager",
    "ResidencyError",
    "Charge",
    "NeverFits",
    "ModelBackend",
    "Lease",
    "InvokeOutcome",
    "InstanceSizer",
    "EnvelopeError",
    "DecodedRequest",
    "BindingResolver",
    "AdapterRow",
    "ENTRYPOINT_ATTR",
    "Adapter",
    "BackgroundMint",
    "DefaultsError",
    "DeployBinding",
    "DistillationAdapter",
    "EndpointHost",
    "EndpointLoadError",
    "EntrypointDeclarationError",
    "EntrypointSpec",
    "LaneContract",
    "LoadContext",
    "LoadedEndpoint",
    "LoaderEngine",
    "MintCondemned",
    "MintFailure",
    "MintOutcome",
    "MintProgress",
    "MintedHole",
    "MissingProgram",
    "Model",
    "ModelDeclarationError",
    "ModelInstance",
    "RequestContext",
    "ServeDispatchError",
    "SlotSpec",
    "entrypoint",
    "lane_handle",
    "load_endpoint",
    "load_endpoint_module",
    "mint_holes",
    "SelfMint",
    "SelfMintStatus",
    "ServeAdoption",
    "model_declared_lanes",
    "model_lane_spec",
    "model_lanes",
    "admission_charge",
    "model_marks_compile",
    "model_requires",
    "model_shapes",
    "model_structural",
    "model_type",
    "DYNAMIC",
    "STATIC",
    "DeclaredLane",
    "LaneDeclarationError",
    "LaneSpec",
    "Structural",
    "lane",
]
