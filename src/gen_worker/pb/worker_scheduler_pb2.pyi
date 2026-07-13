from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProtocolVersion(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROTOCOL_VERSION_UNSPECIFIED: _ClassVar[ProtocolVersion]
    PROTOCOL_VERSION_CURRENT: _ClassVar[ProtocolVersion]

class ResidencyTier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESIDENCY_TIER_UNSPECIFIED: _ClassVar[ResidencyTier]
    RESIDENCY_TIER_DISK: _ClassVar[ResidencyTier]
    RESIDENCY_TIER_RAM: _ClassVar[ResidencyTier]
    RESIDENCY_TIER_VRAM: _ClassVar[ResidencyTier]

class WorkerPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_PHASE_UNSPECIFIED: _ClassVar[WorkerPhase]
    WORKER_PHASE_BOOTING: _ClassVar[WorkerPhase]
    WORKER_PHASE_DOWNLOADING_MODELS: _ClassVar[WorkerPhase]
    WORKER_PHASE_LOADING_PIPELINES: _ClassVar[WorkerPhase]
    WORKER_PHASE_WARMING: _ClassVar[WorkerPhase]
    WORKER_PHASE_READY: _ClassVar[WorkerPhase]
    WORKER_PHASE_ERROR: _ClassVar[WorkerPhase]

class OutputMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OUTPUT_MODE_UNSPECIFIED: _ClassVar[OutputMode]
    OUTPUT_MODE_URL: _ClassVar[OutputMode]
    OUTPUT_MODE_INLINE: _ClassVar[OutputMode]

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATUS_UNSPECIFIED: _ClassVar[JobStatus]
    JOB_STATUS_OK: _ClassVar[JobStatus]
    JOB_STATUS_INVALID: _ClassVar[JobStatus]
    JOB_STATUS_RETRYABLE: _ClassVar[JobStatus]
    JOB_STATUS_FATAL: _ClassVar[JobStatus]
    JOB_STATUS_CANCELED: _ClassVar[JobStatus]

class ModelOpKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_OP_KIND_UNSPECIFIED: _ClassVar[ModelOpKind]
    MODEL_OP_KIND_DOWNLOAD: _ClassVar[ModelOpKind]
    MODEL_OP_KIND_LOAD: _ClassVar[ModelOpKind]
    MODEL_OP_KIND_UNLOAD: _ClassVar[ModelOpKind]
    MODEL_OP_KIND_ADOPT_COMPILE_CACHE: _ClassVar[ModelOpKind]

class ModelState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_STATE_UNSPECIFIED: _ClassVar[ModelState]
    MODEL_STATE_DOWNLOADING: _ClassVar[ModelState]
    MODEL_STATE_ON_DISK: _ClassVar[ModelState]
    MODEL_STATE_IN_RAM: _ClassVar[ModelState]
    MODEL_STATE_IN_VRAM: _ClassVar[ModelState]
    MODEL_STATE_EVICTED: _ClassVar[ModelState]
    MODEL_STATE_FAILED: _ClassVar[ModelState]
    MODEL_STATE_ADOPTED: _ClassVar[ModelState]
PROTOCOL_VERSION_UNSPECIFIED: ProtocolVersion
PROTOCOL_VERSION_CURRENT: ProtocolVersion
RESIDENCY_TIER_UNSPECIFIED: ResidencyTier
RESIDENCY_TIER_DISK: ResidencyTier
RESIDENCY_TIER_RAM: ResidencyTier
RESIDENCY_TIER_VRAM: ResidencyTier
WORKER_PHASE_UNSPECIFIED: WorkerPhase
WORKER_PHASE_BOOTING: WorkerPhase
WORKER_PHASE_DOWNLOADING_MODELS: WorkerPhase
WORKER_PHASE_LOADING_PIPELINES: WorkerPhase
WORKER_PHASE_WARMING: WorkerPhase
WORKER_PHASE_READY: WorkerPhase
WORKER_PHASE_ERROR: WorkerPhase
OUTPUT_MODE_UNSPECIFIED: OutputMode
OUTPUT_MODE_URL: OutputMode
OUTPUT_MODE_INLINE: OutputMode
JOB_STATUS_UNSPECIFIED: JobStatus
JOB_STATUS_OK: JobStatus
JOB_STATUS_INVALID: JobStatus
JOB_STATUS_RETRYABLE: JobStatus
JOB_STATUS_FATAL: JobStatus
JOB_STATUS_CANCELED: JobStatus
MODEL_OP_KIND_UNSPECIFIED: ModelOpKind
MODEL_OP_KIND_DOWNLOAD: ModelOpKind
MODEL_OP_KIND_LOAD: ModelOpKind
MODEL_OP_KIND_UNLOAD: ModelOpKind
MODEL_OP_KIND_ADOPT_COMPILE_CACHE: ModelOpKind
MODEL_STATE_UNSPECIFIED: ModelState
MODEL_STATE_DOWNLOADING: ModelState
MODEL_STATE_ON_DISK: ModelState
MODEL_STATE_IN_RAM: ModelState
MODEL_STATE_IN_VRAM: ModelState
MODEL_STATE_EVICTED: ModelState
MODEL_STATE_FAILED: ModelState
MODEL_STATE_ADOPTED: ModelState

class WorkerMessage(_message.Message):
    __slots__ = ("hello", "state_delta", "job_accepted", "job_result", "job_progress", "model_event", "fn_unavailable", "fn_degraded")
    HELLO_FIELD_NUMBER: _ClassVar[int]
    STATE_DELTA_FIELD_NUMBER: _ClassVar[int]
    JOB_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    JOB_RESULT_FIELD_NUMBER: _ClassVar[int]
    JOB_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_EVENT_FIELD_NUMBER: _ClassVar[int]
    FN_UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    FN_DEGRADED_FIELD_NUMBER: _ClassVar[int]
    hello: Hello
    state_delta: StateDelta
    job_accepted: JobAccepted
    job_result: JobResult
    job_progress: JobProgress
    model_event: ModelEvent
    fn_unavailable: FnUnavailable
    fn_degraded: FnDegraded
    def __init__(self, hello: _Optional[_Union[Hello, _Mapping]] = ..., state_delta: _Optional[_Union[StateDelta, _Mapping]] = ..., job_accepted: _Optional[_Union[JobAccepted, _Mapping]] = ..., job_result: _Optional[_Union[JobResult, _Mapping]] = ..., job_progress: _Optional[_Union[JobProgress, _Mapping]] = ..., model_event: _Optional[_Union[ModelEvent, _Mapping]] = ..., fn_unavailable: _Optional[_Union[FnUnavailable, _Mapping]] = ..., fn_degraded: _Optional[_Union[FnDegraded, _Mapping]] = ...) -> None: ...

class SchedulerMessage(_message.Message):
    __slots__ = ("hello_ack", "run_job", "cancel_job", "model_op", "drain", "token_refresh")
    HELLO_ACK_FIELD_NUMBER: _ClassVar[int]
    RUN_JOB_FIELD_NUMBER: _ClassVar[int]
    CANCEL_JOB_FIELD_NUMBER: _ClassVar[int]
    MODEL_OP_FIELD_NUMBER: _ClassVar[int]
    DRAIN_FIELD_NUMBER: _ClassVar[int]
    TOKEN_REFRESH_FIELD_NUMBER: _ClassVar[int]
    hello_ack: HelloAck
    run_job: RunJob
    cancel_job: CancelJob
    model_op: ModelOp
    drain: Drain
    token_refresh: TokenRefresh
    def __init__(self, hello_ack: _Optional[_Union[HelloAck, _Mapping]] = ..., run_job: _Optional[_Union[RunJob, _Mapping]] = ..., cancel_job: _Optional[_Union[CancelJob, _Mapping]] = ..., model_op: _Optional[_Union[ModelOp, _Mapping]] = ..., drain: _Optional[_Union[Drain, _Mapping]] = ..., token_refresh: _Optional[_Union[TokenRefresh, _Mapping]] = ...) -> None: ...

class Hello(_message.Message):
    __slots__ = ("protocol_version", "worker_id", "release_id", "resources", "state", "models", "in_flight")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    protocol_version: ProtocolVersion
    worker_id: str
    release_id: str
    resources: WorkerResources
    state: StateDelta
    models: _containers.RepeatedCompositeFieldContainer[ModelResidency]
    in_flight: _containers.RepeatedCompositeFieldContainer[InFlightJob]
    def __init__(self, protocol_version: _Optional[_Union[ProtocolVersion, str]] = ..., worker_id: _Optional[str] = ..., release_id: _Optional[str] = ..., resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., state: _Optional[_Union[StateDelta, _Mapping]] = ..., models: _Optional[_Iterable[_Union[ModelResidency, _Mapping]]] = ..., in_flight: _Optional[_Iterable[_Union[InFlightJob, _Mapping]]] = ...) -> None: ...

class WorkerResources(_message.Message):
    __slots__ = ("gpu_count", "vram_total_bytes", "gpu_name", "gpu_sm", "installed_libs", "image_digest", "git_commit", "instance_id")
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    VRAM_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_SM_FIELD_NUMBER: _ClassVar[int]
    INSTALLED_LIBS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    gpu_count: int
    vram_total_bytes: int
    gpu_name: str
    gpu_sm: str
    installed_libs: _containers.RepeatedScalarFieldContainer[str]
    image_digest: str
    git_commit: str
    instance_id: str
    def __init__(self, gpu_count: _Optional[int] = ..., vram_total_bytes: _Optional[int] = ..., gpu_name: _Optional[str] = ..., gpu_sm: _Optional[str] = ..., installed_libs: _Optional[_Iterable[str]] = ..., image_digest: _Optional[str] = ..., git_commit: _Optional[str] = ..., instance_id: _Optional[str] = ...) -> None: ...

class ModelResidency(_message.Message):
    __slots__ = ("ref", "tier", "vram_bytes")
    REF_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    ref: str
    tier: ResidencyTier
    vram_bytes: int
    def __init__(self, ref: _Optional[str] = ..., tier: _Optional[_Union[ResidencyTier, str]] = ..., vram_bytes: _Optional[int] = ...) -> None: ...

class InFlightJob(_message.Message):
    __slots__ = ("request_id", "attempt")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ...) -> None: ...

class HelloAck(_message.Message):
    __slots__ = ("protocol_version", "file_base_url", "keep", "resolutions")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    FILE_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    KEEP_FIELD_NUMBER: _ClassVar[int]
    RESOLUTIONS_FIELD_NUMBER: _ClassVar[int]
    protocol_version: ProtocolVersion
    file_base_url: str
    keep: _containers.RepeatedScalarFieldContainer[str]
    resolutions: _containers.RepeatedCompositeFieldContainer[ModelResolution]
    def __init__(self, protocol_version: _Optional[_Union[ProtocolVersion, str]] = ..., file_base_url: _Optional[str] = ..., keep: _Optional[_Iterable[str]] = ..., resolutions: _Optional[_Iterable[_Union[ModelResolution, _Mapping]]] = ...) -> None: ...

class ModelResolution(_message.Message):
    __slots__ = ("ref", "resolved_ref", "cast")
    REF_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_REF_FIELD_NUMBER: _ClassVar[int]
    CAST_FIELD_NUMBER: _ClassVar[int]
    ref: str
    resolved_ref: str
    cast: str
    def __init__(self, ref: _Optional[str] = ..., resolved_ref: _Optional[str] = ..., cast: _Optional[str] = ...) -> None: ...

class StateDelta(_message.Message):
    __slots__ = ("phase", "available_functions", "loading_functions", "free_vram_bytes", "finalizing_jobs")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    LOADING_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    FINALIZING_JOBS_FIELD_NUMBER: _ClassVar[int]
    phase: WorkerPhase
    available_functions: _containers.RepeatedScalarFieldContainer[str]
    loading_functions: _containers.RepeatedScalarFieldContainer[str]
    free_vram_bytes: int
    finalizing_jobs: int
    def __init__(self, phase: _Optional[_Union[WorkerPhase, str]] = ..., available_functions: _Optional[_Iterable[str]] = ..., loading_functions: _Optional[_Iterable[str]] = ..., free_vram_bytes: _Optional[int] = ..., finalizing_jobs: _Optional[int] = ...) -> None: ...

class RunJob(_message.Message):
    __slots__ = ("request_id", "attempt", "function_name", "input_payload", "timeout_ms", "tenant", "invoker_id", "capability_token", "output_mode", "compute", "models", "snapshots")
    class SnapshotsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Snapshot
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Snapshot, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    INVOKER_ID_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_TOKEN_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MODE_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    function_name: str
    input_payload: bytes
    timeout_ms: int
    tenant: str
    invoker_id: str
    capability_token: str
    output_mode: OutputMode
    compute: ResolvedCompute
    models: _containers.RepeatedCompositeFieldContainer[ModelBinding]
    snapshots: _containers.MessageMap[str, Snapshot]
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., function_name: _Optional[str] = ..., input_payload: _Optional[bytes] = ..., timeout_ms: _Optional[int] = ..., tenant: _Optional[str] = ..., invoker_id: _Optional[str] = ..., capability_token: _Optional[str] = ..., output_mode: _Optional[_Union[OutputMode, str]] = ..., compute: _Optional[_Union[ResolvedCompute, _Mapping]] = ..., models: _Optional[_Iterable[_Union[ModelBinding, _Mapping]]] = ..., snapshots: _Optional[_Mapping[str, Snapshot]] = ...) -> None: ...

class ResolvedCompute(_message.Message):
    __slots__ = ("accelerator", "gpu_index")
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    GPU_INDEX_FIELD_NUMBER: _ClassVar[int]
    accelerator: str
    gpu_index: int
    def __init__(self, accelerator: _Optional[str] = ..., gpu_index: _Optional[int] = ...) -> None: ...

class ModelBinding(_message.Message):
    __slots__ = ("slot", "ref", "loras", "inference_defaults")
    SLOT_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    LORAS_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    slot: str
    ref: str
    loras: _containers.RepeatedCompositeFieldContainer[LoraOverlay]
    inference_defaults: str
    def __init__(self, slot: _Optional[str] = ..., ref: _Optional[str] = ..., loras: _Optional[_Iterable[_Union[LoraOverlay, _Mapping]]] = ..., inference_defaults: _Optional[str] = ...) -> None: ...

class LoraOverlay(_message.Message):
    __slots__ = ("ref", "weight", "inference_defaults")
    REF_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    ref: str
    weight: float
    inference_defaults: str
    def __init__(self, ref: _Optional[str] = ..., weight: _Optional[float] = ..., inference_defaults: _Optional[str] = ...) -> None: ...

class Snapshot(_message.Message):
    __slots__ = ("digest", "files")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    digest: str
    files: _containers.RepeatedCompositeFieldContainer[SnapshotFile]
    def __init__(self, digest: _Optional[str] = ..., files: _Optional[_Iterable[_Union[SnapshotFile, _Mapping]]] = ...) -> None: ...

class SnapshotFile(_message.Message):
    __slots__ = ("path", "size_bytes", "blake3", "url")
    PATH_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    BLAKE3_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    path: str
    size_bytes: int
    blake3: str
    url: str
    def __init__(self, path: _Optional[str] = ..., size_bytes: _Optional[int] = ..., blake3: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

class JobAccepted(_message.Message):
    __slots__ = ("request_id", "attempt")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ...) -> None: ...

class JobResult(_message.Message):
    __slots__ = ("request_id", "attempt", "status", "inline", "blob_ref", "safe_message", "metrics")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    INLINE_FIELD_NUMBER: _ClassVar[int]
    BLOB_REF_FIELD_NUMBER: _ClassVar[int]
    SAFE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    status: JobStatus
    inline: bytes
    blob_ref: str
    safe_message: str
    metrics: JobMetrics
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., status: _Optional[_Union[JobStatus, str]] = ..., inline: _Optional[bytes] = ..., blob_ref: _Optional[str] = ..., safe_message: _Optional[str] = ..., metrics: _Optional[_Union[JobMetrics, _Mapping]] = ...) -> None: ...

class JobMetrics(_message.Message):
    __slots__ = ("runtime_ms", "queue_ms", "rss_at_end_bytes", "peak_vram_bytes", "concurrency_at_start", "output_media_duration_s", "input_tokens", "input_cached_tokens", "output_tokens", "output_count", "slot_held_ms", "finalize_wall_ms")
    RUNTIME_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    RSS_AT_END_BYTES_FIELD_NUMBER: _ClassVar[int]
    PEAK_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    CONCURRENCY_AT_START_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MEDIA_DURATION_S_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_COUNT_FIELD_NUMBER: _ClassVar[int]
    SLOT_HELD_MS_FIELD_NUMBER: _ClassVar[int]
    FINALIZE_WALL_MS_FIELD_NUMBER: _ClassVar[int]
    runtime_ms: int
    queue_ms: int
    rss_at_end_bytes: int
    peak_vram_bytes: int
    concurrency_at_start: int
    output_media_duration_s: float
    input_tokens: int
    input_cached_tokens: int
    output_tokens: int
    output_count: int
    slot_held_ms: int
    finalize_wall_ms: int
    def __init__(self, runtime_ms: _Optional[int] = ..., queue_ms: _Optional[int] = ..., rss_at_end_bytes: _Optional[int] = ..., peak_vram_bytes: _Optional[int] = ..., concurrency_at_start: _Optional[int] = ..., output_media_duration_s: _Optional[float] = ..., input_tokens: _Optional[int] = ..., input_cached_tokens: _Optional[int] = ..., output_tokens: _Optional[int] = ..., output_count: _Optional[int] = ..., slot_held_ms: _Optional[int] = ..., finalize_wall_ms: _Optional[int] = ...) -> None: ...

class JobProgress(_message.Message):
    __slots__ = ("request_id", "attempt", "seq", "data", "content_type")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    seq: int
    data: bytes
    content_type: str
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ..., seq: _Optional[int] = ..., data: _Optional[bytes] = ..., content_type: _Optional[str] = ...) -> None: ...

class CancelJob(_message.Message):
    __slots__ = ("request_id", "attempt")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    attempt: int
    def __init__(self, request_id: _Optional[str] = ..., attempt: _Optional[int] = ...) -> None: ...

class ModelOp(_message.Message):
    __slots__ = ("op", "ref", "snapshot")
    OP_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    op: ModelOpKind
    ref: str
    snapshot: Snapshot
    def __init__(self, op: _Optional[_Union[ModelOpKind, str]] = ..., ref: _Optional[str] = ..., snapshot: _Optional[_Union[Snapshot, _Mapping]] = ...) -> None: ...

class ModelEvent(_message.Message):
    __slots__ = ("ref", "state", "vram_bytes", "error", "bytes_done", "bytes_total", "duration_ms", "cache_hits", "cache_misses", "warmup_s")
    REF_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    BYTES_DONE_FIELD_NUMBER: _ClassVar[int]
    BYTES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    CACHE_HITS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MISSES_FIELD_NUMBER: _ClassVar[int]
    WARMUP_S_FIELD_NUMBER: _ClassVar[int]
    ref: str
    state: ModelState
    vram_bytes: int
    error: str
    bytes_done: int
    bytes_total: int
    duration_ms: int
    cache_hits: int
    cache_misses: int
    warmup_s: float
    def __init__(self, ref: _Optional[str] = ..., state: _Optional[_Union[ModelState, str]] = ..., vram_bytes: _Optional[int] = ..., error: _Optional[str] = ..., bytes_done: _Optional[int] = ..., bytes_total: _Optional[int] = ..., duration_ms: _Optional[int] = ..., cache_hits: _Optional[int] = ..., cache_misses: _Optional[int] = ..., warmup_s: _Optional[float] = ...) -> None: ...

class FnUnavailable(_message.Message):
    __slots__ = ("function_name", "reason", "detail", "axes")
    class AxesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    AXES_FIELD_NUMBER: _ClassVar[int]
    function_name: str
    reason: str
    detail: str
    axes: _containers.ScalarMap[str, str]
    def __init__(self, function_name: _Optional[str] = ..., reason: _Optional[str] = ..., detail: _Optional[str] = ..., axes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class FnDegraded(_message.Message):
    __slots__ = ("function_name", "wanted", "ran", "reason", "est_latency_multiplier", "recommended_vram_gb")
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    WANTED_FIELD_NUMBER: _ClassVar[int]
    RAN_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    EST_LATENCY_MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDED_VRAM_GB_FIELD_NUMBER: _ClassVar[int]
    function_name: str
    wanted: str
    ran: str
    reason: str
    est_latency_multiplier: float
    recommended_vram_gb: float
    def __init__(self, function_name: _Optional[str] = ..., wanted: _Optional[str] = ..., ran: _Optional[str] = ..., reason: _Optional[str] = ..., est_latency_multiplier: _Optional[float] = ..., recommended_vram_gb: _Optional[float] = ...) -> None: ...

class Drain(_message.Message):
    __slots__ = ("deadline_ms",)
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    deadline_ms: int
    def __init__(self, deadline_ms: _Optional[int] = ...) -> None: ...

class TokenRefresh(_message.Message):
    __slots__ = ("token", "expires_at_unix")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    token: str
    expires_at_unix: int
    def __init__(self, token: _Optional[str] = ..., expires_at_unix: _Optional[int] = ...) -> None: ...
