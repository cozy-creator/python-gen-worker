from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkerStartupPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_STARTUP_PHASE_UNSPECIFIED: _ClassVar[WorkerStartupPhase]
    WORKER_STARTUP_PHASE_BOOTING: _ClassVar[WorkerStartupPhase]
    WORKER_STARTUP_PHASE_MODELS_DOWNLOADING: _ClassVar[WorkerStartupPhase]
    WORKER_STARTUP_PHASE_PIPELINE_LOADING: _ClassVar[WorkerStartupPhase]
    WORKER_STARTUP_PHASE_READY: _ClassVar[WorkerStartupPhase]
    WORKER_STARTUP_PHASE_ERROR: _ClassVar[WorkerStartupPhase]
    WORKER_STARTUP_PHASE_WARMING: _ClassVar[WorkerStartupPhase]

class ModelAvailabilityKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_AVAILABILITY_UNSPECIFIED: _ClassVar[ModelAvailabilityKind]
    MODEL_AVAILABILITY_READY: _ClassVar[ModelAvailabilityKind]
    MODEL_AVAILABILITY_DOWNLOAD_COMPLETED: _ClassVar[ModelAvailabilityKind]
    MODEL_AVAILABILITY_CACHED: _ClassVar[ModelAvailabilityKind]

class KVPrefixCacheEvent(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    KV_PREFIX_CACHE_EVENT_UNSPECIFIED: _ClassVar[KVPrefixCacheEvent]
    KV_PREFIX_CACHE_EVENT_ADDED: _ClassVar[KVPrefixCacheEvent]
    KV_PREFIX_CACHE_EVENT_REMOVED: _ClassVar[KVPrefixCacheEvent]
WORKER_STARTUP_PHASE_UNSPECIFIED: WorkerStartupPhase
WORKER_STARTUP_PHASE_BOOTING: WorkerStartupPhase
WORKER_STARTUP_PHASE_MODELS_DOWNLOADING: WorkerStartupPhase
WORKER_STARTUP_PHASE_PIPELINE_LOADING: WorkerStartupPhase
WORKER_STARTUP_PHASE_READY: WorkerStartupPhase
WORKER_STARTUP_PHASE_ERROR: WorkerStartupPhase
WORKER_STARTUP_PHASE_WARMING: WorkerStartupPhase
MODEL_AVAILABILITY_UNSPECIFIED: ModelAvailabilityKind
MODEL_AVAILABILITY_READY: ModelAvailabilityKind
MODEL_AVAILABILITY_DOWNLOAD_COMPLETED: ModelAvailabilityKind
MODEL_AVAILABILITY_CACHED: ModelAvailabilityKind
KV_PREFIX_CACHE_EVENT_UNSPECIFIED: KVPrefixCacheEvent
KV_PREFIX_CACHE_EVENT_ADDED: KVPrefixCacheEvent
KV_PREFIX_CACHE_EVENT_REMOVED: KVPrefixCacheEvent

class WorkerResources(_message.Message):
    __slots__ = ("worker_id", "gpu_count", "gpu_memory_bytes", "available_functions", "vram_models", "release_id", "runpod_pod_id", "gpu_is_busy", "gpu_name", "gpu_memory_free_bytes", "gpu_sm", "disk_models", "installed_libs", "image_digest", "git_commit", "loading_functions", "ram_models")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    VRAM_MODELS_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    RUNPOD_POD_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_IS_BUSY_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_FREE_BYTES_FIELD_NUMBER: _ClassVar[int]
    GPU_SM_FIELD_NUMBER: _ClassVar[int]
    DISK_MODELS_FIELD_NUMBER: _ClassVar[int]
    INSTALLED_LIBS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    LOADING_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    RAM_MODELS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    gpu_count: int
    gpu_memory_bytes: int
    available_functions: _containers.RepeatedScalarFieldContainer[str]
    vram_models: _containers.RepeatedScalarFieldContainer[str]
    release_id: str
    runpod_pod_id: str
    gpu_is_busy: bool
    gpu_name: str
    gpu_memory_free_bytes: int
    gpu_sm: str
    disk_models: _containers.RepeatedScalarFieldContainer[str]
    installed_libs: _containers.RepeatedScalarFieldContainer[str]
    image_digest: str
    git_commit: str
    loading_functions: _containers.RepeatedScalarFieldContainer[str]
    ram_models: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, worker_id: _Optional[str] = ..., gpu_count: _Optional[int] = ..., gpu_memory_bytes: _Optional[int] = ..., available_functions: _Optional[_Iterable[str]] = ..., vram_models: _Optional[_Iterable[str]] = ..., release_id: _Optional[str] = ..., runpod_pod_id: _Optional[str] = ..., gpu_is_busy: bool = ..., gpu_name: _Optional[str] = ..., gpu_memory_free_bytes: _Optional[int] = ..., gpu_sm: _Optional[str] = ..., disk_models: _Optional[_Iterable[str]] = ..., installed_libs: _Optional[_Iterable[str]] = ..., image_digest: _Optional[str] = ..., git_commit: _Optional[str] = ..., loading_functions: _Optional[_Iterable[str]] = ..., ram_models: _Optional[_Iterable[str]] = ...) -> None: ...

class ActiveAssignmentResume(_message.Message):
    __slots__ = ("request_id", "item_id", "assignment_attempt_epoch", "last_job_result_seq", "last_worker_event_seq", "last_incremental_seq")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENT_ATTEMPT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    LAST_JOB_RESULT_SEQ_FIELD_NUMBER: _ClassVar[int]
    LAST_WORKER_EVENT_SEQ_FIELD_NUMBER: _ClassVar[int]
    LAST_INCREMENTAL_SEQ_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    item_id: str
    assignment_attempt_epoch: int
    last_job_result_seq: int
    last_worker_event_seq: int
    last_incremental_seq: int
    def __init__(self, request_id: _Optional[str] = ..., item_id: _Optional[str] = ..., assignment_attempt_epoch: _Optional[int] = ..., last_job_result_seq: _Optional[int] = ..., last_worker_event_seq: _Optional[int] = ..., last_incremental_seq: _Optional[int] = ...) -> None: ...

class WorkerRegistration(_message.Message):
    __slots__ = ("resources", "is_heartbeat", "protocol_major", "protocol_minor", "active_assignments", "in_flight_request_ids", "stream_started_unix_ms", "supports_split_streams", "stream_role")
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    IS_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_MAJOR_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_MINOR_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    IN_FLIGHT_REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    STREAM_STARTED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_SPLIT_STREAMS_FIELD_NUMBER: _ClassVar[int]
    STREAM_ROLE_FIELD_NUMBER: _ClassVar[int]
    resources: WorkerResources
    is_heartbeat: bool
    protocol_major: int
    protocol_minor: int
    active_assignments: _containers.RepeatedCompositeFieldContainer[ActiveAssignmentResume]
    in_flight_request_ids: _containers.RepeatedScalarFieldContainer[str]
    stream_started_unix_ms: int
    supports_split_streams: bool
    stream_role: str
    def __init__(self, resources: _Optional[_Union[WorkerResources, _Mapping]] = ..., is_heartbeat: bool = ..., protocol_major: _Optional[int] = ..., protocol_minor: _Optional[int] = ..., active_assignments: _Optional[_Iterable[_Union[ActiveAssignmentResume, _Mapping]]] = ..., in_flight_request_ids: _Optional[_Iterable[str]] = ..., stream_started_unix_ms: _Optional[int] = ..., supports_split_streams: bool = ..., stream_role: _Optional[str] = ...) -> None: ...

class ResolvedCompute(_message.Message):
    __slots__ = ("accelerator", "min_compute_capability", "vram_gb", "gpu_count", "gpu_tier", "memory_gb", "cpu_cores", "disk_gb", "gpu_index")
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    MIN_COMPUTE_CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    VRAM_GB_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GPU_TIER_FIELD_NUMBER: _ClassVar[int]
    MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    CPU_CORES_FIELD_NUMBER: _ClassVar[int]
    DISK_GB_FIELD_NUMBER: _ClassVar[int]
    GPU_INDEX_FIELD_NUMBER: _ClassVar[int]
    accelerator: str
    min_compute_capability: str
    vram_gb: int
    gpu_count: int
    gpu_tier: str
    memory_gb: int
    cpu_cores: int
    disk_gb: int
    gpu_index: int
    def __init__(self, accelerator: _Optional[str] = ..., min_compute_capability: _Optional[str] = ..., vram_gb: _Optional[int] = ..., gpu_count: _Optional[int] = ..., gpu_tier: _Optional[str] = ..., memory_gb: _Optional[int] = ..., cpu_cores: _Optional[int] = ..., disk_gb: _Optional[int] = ..., gpu_index: _Optional[int] = ...) -> None: ...

class ResolvedLoraBinding(_message.Message):
    __slots__ = ("ref", "tag", "flavor", "provider", "weight", "compatibility_status", "compatibility_detail")
    REF_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    FLAVOR_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_DETAIL_FIELD_NUMBER: _ClassVar[int]
    ref: str
    tag: str
    flavor: str
    provider: str
    weight: float
    compatibility_status: str
    compatibility_detail: str
    def __init__(self, ref: _Optional[str] = ..., tag: _Optional[str] = ..., flavor: _Optional[str] = ..., provider: _Optional[str] = ..., weight: _Optional[float] = ..., compatibility_status: _Optional[str] = ..., compatibility_detail: _Optional[str] = ...) -> None: ...

class ResolvedModelBinding(_message.Message):
    __slots__ = ("slot_name", "ref", "tag", "flavor", "provider", "source", "checkpoint_id", "compatibility_status", "compatibility_detail", "loras")
    SLOT_NAME_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    FLAVOR_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPATIBILITY_DETAIL_FIELD_NUMBER: _ClassVar[int]
    LORAS_FIELD_NUMBER: _ClassVar[int]
    slot_name: str
    ref: str
    tag: str
    flavor: str
    provider: str
    source: str
    checkpoint_id: str
    compatibility_status: str
    compatibility_detail: str
    loras: _containers.RepeatedCompositeFieldContainer[ResolvedLoraBinding]
    def __init__(self, slot_name: _Optional[str] = ..., ref: _Optional[str] = ..., tag: _Optional[str] = ..., flavor: _Optional[str] = ..., provider: _Optional[str] = ..., source: _Optional[str] = ..., checkpoint_id: _Optional[str] = ..., compatibility_status: _Optional[str] = ..., compatibility_detail: _Optional[str] = ..., loras: _Optional[_Iterable[_Union[ResolvedLoraBinding, _Mapping]]] = ...) -> None: ...

class LoadModelCommand(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class DownloadModelCommand(_message.Message):
    __slots__ = ("model_id", "ref", "priority")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    ref: str
    priority: int
    def __init__(self, model_id: _Optional[str] = ..., ref: _Optional[str] = ..., priority: _Optional[int] = ...) -> None: ...

class UnloadModelCommand(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class InterruptJobCommand(_message.Message):
    __slots__ = ("request_id", "item_ids", "cancel_queued_only", "assignment_attempt_epoch")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    CANCEL_QUEUED_ONLY_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENT_ATTEMPT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    item_ids: _containers.RepeatedScalarFieldContainer[str]
    cancel_queued_only: bool
    assignment_attempt_epoch: int
    def __init__(self, request_id: _Optional[str] = ..., item_ids: _Optional[_Iterable[str]] = ..., cancel_queued_only: bool = ..., assignment_attempt_epoch: _Optional[int] = ...) -> None: ...

class ResolvedRepoFile(_message.Message):
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

class ResolvedRepo(_message.Message):
    __slots__ = ("snapshot_digest", "files")
    SNAPSHOT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    snapshot_digest: str
    files: _containers.RepeatedCompositeFieldContainer[ResolvedRepoFile]
    def __init__(self, snapshot_digest: _Optional[str] = ..., files: _Optional[_Iterable[_Union[ResolvedRepoFile, _Mapping]]] = ...) -> None: ...

class JobExecutionRequest(_message.Message):
    __slots__ = ("request_id", "function_name", "input_payload", "timeout_ms", "tenant", "invoker_id", "file_base_url", "resolved_repos_by_id", "job_id", "worker_capability_token", "resolved_compute", "execution_hints", "resolved_models")
    class ResolvedReposByIdEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ResolvedRepo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ResolvedRepo, _Mapping]] = ...) -> None: ...
    class ExecutionHintsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ResolvedModelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ResolvedModelBinding
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ResolvedModelBinding, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    INVOKER_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_BASE_URL_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_REPOS_BY_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_CAPABILITY_TOKEN_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_COMPUTE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_HINTS_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_MODELS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    function_name: str
    input_payload: bytes
    timeout_ms: int
    tenant: str
    invoker_id: str
    file_base_url: str
    resolved_repos_by_id: _containers.MessageMap[str, ResolvedRepo]
    job_id: str
    worker_capability_token: str
    resolved_compute: ResolvedCompute
    execution_hints: _containers.ScalarMap[str, str]
    resolved_models: _containers.MessageMap[str, ResolvedModelBinding]
    def __init__(self, request_id: _Optional[str] = ..., function_name: _Optional[str] = ..., input_payload: _Optional[bytes] = ..., timeout_ms: _Optional[int] = ..., tenant: _Optional[str] = ..., invoker_id: _Optional[str] = ..., file_base_url: _Optional[str] = ..., resolved_repos_by_id: _Optional[_Mapping[str, ResolvedRepo]] = ..., job_id: _Optional[str] = ..., worker_capability_token: _Optional[str] = ..., resolved_compute: _Optional[_Union[ResolvedCompute, _Mapping]] = ..., execution_hints: _Optional[_Mapping[str, str]] = ..., resolved_models: _Optional[_Mapping[str, ResolvedModelBinding]] = ...) -> None: ...

class JobExecutionResult(_message.Message):
    __slots__ = ("request_id", "success", "output_payload", "error_type", "retryable", "safe_message", "observation")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    SAFE_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    output_payload: bytes
    error_type: str
    retryable: bool
    safe_message: str
    observation: JobExecutionObservation
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., output_payload: _Optional[bytes] = ..., error_type: _Optional[str] = ..., retryable: bool = ..., safe_message: _Optional[str] = ..., observation: _Optional[_Union[JobExecutionObservation, _Mapping]] = ...) -> None: ...

class JobExecutionObservation(_message.Message):
    __slots__ = ("release_id", "function_name", "build_profile", "image_digest", "provider", "worker_id", "machine_class", "status", "error_type", "runtime_ms", "local_queue_ms", "peak_memory_bytes", "peak_vram_bytes", "active_count_at_start", "local_queued_count_at_start", "ttft_ms", "itl_p50_ms", "prefix_hit_rate_pct", "kv_blocks_used", "kv_blocks_total", "scaling_factors")
    class ScalingFactorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    BUILD_PROFILE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    MACHINE_CLASS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_MS_FIELD_NUMBER: _ClassVar[int]
    LOCAL_QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    PEAK_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    PEAK_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_COUNT_AT_START_FIELD_NUMBER: _ClassVar[int]
    LOCAL_QUEUED_COUNT_AT_START_FIELD_NUMBER: _ClassVar[int]
    TTFT_MS_FIELD_NUMBER: _ClassVar[int]
    ITL_P50_MS_FIELD_NUMBER: _ClassVar[int]
    PREFIX_HIT_RATE_PCT_FIELD_NUMBER: _ClassVar[int]
    KV_BLOCKS_USED_FIELD_NUMBER: _ClassVar[int]
    KV_BLOCKS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    SCALING_FACTORS_FIELD_NUMBER: _ClassVar[int]
    release_id: str
    function_name: str
    build_profile: str
    image_digest: str
    provider: str
    worker_id: str
    machine_class: str
    status: str
    error_type: str
    runtime_ms: int
    local_queue_ms: int
    peak_memory_bytes: int
    peak_vram_bytes: int
    active_count_at_start: int
    local_queued_count_at_start: int
    ttft_ms: int
    itl_p50_ms: int
    prefix_hit_rate_pct: int
    kv_blocks_used: int
    kv_blocks_total: int
    scaling_factors: _containers.ScalarMap[str, float]
    def __init__(self, release_id: _Optional[str] = ..., function_name: _Optional[str] = ..., build_profile: _Optional[str] = ..., image_digest: _Optional[str] = ..., provider: _Optional[str] = ..., worker_id: _Optional[str] = ..., machine_class: _Optional[str] = ..., status: _Optional[str] = ..., error_type: _Optional[str] = ..., runtime_ms: _Optional[int] = ..., local_queue_ms: _Optional[int] = ..., peak_memory_bytes: _Optional[int] = ..., peak_vram_bytes: _Optional[int] = ..., active_count_at_start: _Optional[int] = ..., local_queued_count_at_start: _Optional[int] = ..., ttft_ms: _Optional[int] = ..., itl_p50_ms: _Optional[int] = ..., prefix_hit_rate_pct: _Optional[int] = ..., kv_blocks_used: _Optional[int] = ..., kv_blocks_total: _Optional[int] = ..., scaling_factors: _Optional[_Mapping[str, float]] = ...) -> None: ...

class WorkerEvent(_message.Message):
    __slots__ = ("request_id", "event_type", "payload_json")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    event_type: str
    payload_json: bytes
    def __init__(self, request_id: _Optional[str] = ..., event_type: _Optional[str] = ..., payload_json: _Optional[bytes] = ...) -> None: ...

class WorkerDiagnosticLog(_message.Message):
    __slots__ = ("worker_id", "release_id", "runpod_pod_id", "category", "severity", "message", "payload_json", "emitted_at_unix_ms")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    RUNPOD_POD_ID_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    EMITTED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    release_id: str
    runpod_pod_id: str
    category: str
    severity: str
    message: str
    payload_json: bytes
    emitted_at_unix_ms: int
    def __init__(self, worker_id: _Optional[str] = ..., release_id: _Optional[str] = ..., runpod_pod_id: _Optional[str] = ..., category: _Optional[str] = ..., severity: _Optional[str] = ..., message: _Optional[str] = ..., payload_json: _Optional[bytes] = ..., emitted_at_unix_ms: _Optional[int] = ...) -> None: ...

class IncrementalTokenDelta(_message.Message):
    __slots__ = ("request_id", "item_id", "function_name", "sequence", "timestamp_unix_ms", "delta_text", "payload_json", "audio_chunk", "audio_codec")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    DELTA_TEXT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CODEC_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    item_id: str
    function_name: str
    sequence: int
    timestamp_unix_ms: int
    delta_text: str
    payload_json: bytes
    audio_chunk: bytes
    audio_codec: str
    def __init__(self, request_id: _Optional[str] = ..., item_id: _Optional[str] = ..., function_name: _Optional[str] = ..., sequence: _Optional[int] = ..., timestamp_unix_ms: _Optional[int] = ..., delta_text: _Optional[str] = ..., payload_json: _Optional[bytes] = ..., audio_chunk: _Optional[bytes] = ..., audio_codec: _Optional[str] = ...) -> None: ...

class IncrementalTokenStreamDone(_message.Message):
    __slots__ = ("request_id", "item_id", "function_name", "sequence", "timestamp_unix_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    item_id: str
    function_name: str
    sequence: int
    timestamp_unix_ms: int
    def __init__(self, request_id: _Optional[str] = ..., item_id: _Optional[str] = ..., function_name: _Optional[str] = ..., sequence: _Optional[int] = ..., timestamp_unix_ms: _Optional[int] = ...) -> None: ...

class IncrementalTokenStreamError(_message.Message):
    __slots__ = ("request_id", "item_id", "function_name", "sequence", "timestamp_unix_ms", "error_message")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    item_id: str
    function_name: str
    sequence: int
    timestamp_unix_ms: int
    error_message: str
    def __init__(self, request_id: _Optional[str] = ..., item_id: _Optional[str] = ..., function_name: _Optional[str] = ..., sequence: _Optional[int] = ..., timestamp_unix_ms: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class LoadModelResult(_message.Message):
    __slots__ = ("model_id", "success", "error_message", "size_bytes")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    success: bool
    error_message: str
    size_bytes: int
    def __init__(self, model_id: _Optional[str] = ..., success: bool = ..., error_message: _Optional[str] = ..., size_bytes: _Optional[int] = ...) -> None: ...

class UnloadModelResult(_message.Message):
    __slots__ = ("model_id", "success", "error_message")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    success: bool
    error_message: str
    def __init__(self, model_id: _Optional[str] = ..., success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class ModelSpec(_message.Message):
    __slots__ = ("ref",)
    REF_FIELD_NUMBER: _ClassVar[int]
    ref: str
    def __init__(self, ref: _Optional[str] = ...) -> None: ...

class ModelsByKey(_message.Message):
    __slots__ = ("models",)
    class ModelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelSpec
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelSpec, _Mapping]] = ...) -> None: ...
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.MessageMap[str, ModelSpec]
    def __init__(self, models: _Optional[_Mapping[str, ModelSpec]] = ...) -> None: ...

class DisabledFunction(_message.Message):
    __slots__ = ("function_name", "model_key", "ref", "reason", "detail", "detected_at_unix")
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    DETECTED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    function_name: str
    model_key: str
    ref: str
    reason: str
    detail: str
    detected_at_unix: int
    def __init__(self, function_name: _Optional[str] = ..., model_key: _Optional[str] = ..., ref: _Optional[str] = ..., reason: _Optional[str] = ..., detail: _Optional[str] = ..., detected_at_unix: _Optional[int] = ...) -> None: ...

class RefStatus(_message.Message):
    __slots__ = ("ref", "status", "reason", "detail", "last_checked_unix")
    REF_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECKED_UNIX_FIELD_NUMBER: _ClassVar[int]
    ref: str
    status: str
    reason: str
    detail: str
    last_checked_unix: int
    def __init__(self, ref: _Optional[str] = ..., status: _Optional[str] = ..., reason: _Optional[str] = ..., detail: _Optional[str] = ..., last_checked_unix: _Optional[int] = ...) -> None: ...

class FunctionRefAvailability(_message.Message):
    __slots__ = ("by_model_key",)
    class ByModelKeyEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: RefStatus
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[RefStatus, _Mapping]] = ...) -> None: ...
    BY_MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    by_model_key: _containers.MessageMap[str, RefStatus]
    def __init__(self, by_model_key: _Optional[_Mapping[str, RefStatus]] = ...) -> None: ...

class WorkerFunctionUnavailableSignal(_message.Message):
    __slots__ = ("worker_id", "release_id", "function_name", "reason", "detail", "detected_at_unix", "axes")
    class AxesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    DETECTED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    AXES_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    release_id: str
    function_name: str
    reason: str
    detail: str
    detected_at_unix: int
    axes: _containers.ScalarMap[str, str]
    def __init__(self, worker_id: _Optional[str] = ..., release_id: _Optional[str] = ..., function_name: _Optional[str] = ..., reason: _Optional[str] = ..., detail: _Optional[str] = ..., detected_at_unix: _Optional[int] = ..., axes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class WorkerStartupPhaseSignal(_message.Message):
    __slots__ = ("phase", "status", "scheduler_addr", "elapsed_ms", "detail")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_ADDR_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    phase: WorkerStartupPhase
    status: str
    scheduler_addr: str
    elapsed_ms: int
    detail: str
    def __init__(self, phase: _Optional[_Union[WorkerStartupPhase, str]] = ..., status: _Optional[str] = ..., scheduler_addr: _Optional[str] = ..., elapsed_ms: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class WorkerModelReadySignal(_message.Message):
    __slots__ = ("model_id", "kind")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    kind: ModelAvailabilityKind
    def __init__(self, model_id: _Optional[str] = ..., kind: _Optional[_Union[ModelAvailabilityKind, str]] = ...) -> None: ...

class WorkerFunctionCapabilitiesSignal(_message.Message):
    __slots__ = ("capabilities_json", "signature")
    CAPABILITIES_JSON_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    capabilities_json: bytes
    signature: str
    def __init__(self, capabilities_json: _Optional[bytes] = ..., signature: _Optional[str] = ...) -> None: ...

class WorkerKVPrefixCache(_message.Message):
    __slots__ = ("block_hashes", "block_size", "model_id", "event")
    BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    block_hashes: _containers.RepeatedScalarFieldContainer[bytes]
    block_size: int
    model_id: str
    event: KVPrefixCacheEvent
    def __init__(self, block_hashes: _Optional[_Iterable[bytes]] = ..., block_size: _Optional[int] = ..., model_id: _Optional[str] = ..., event: _Optional[_Union[KVPrefixCacheEvent, str]] = ...) -> None: ...

class WorkerDrainCommand(_message.Message):
    __slots__ = ("reason", "deadline_unix_ms", "terminate_after_deadline")
    REASON_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    TERMINATE_AFTER_DEADLINE_FIELD_NUMBER: _ClassVar[int]
    reason: str
    deadline_unix_ms: int
    terminate_after_deadline: bool
    def __init__(self, reason: _Optional[str] = ..., deadline_unix_ms: _Optional[int] = ..., terminate_after_deadline: bool = ...) -> None: ...

class WorkerDrainResult(_message.Message):
    __slots__ = ("reason", "status", "active_requests", "emitted_at_unix_ms")
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    EMITTED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    reason: str
    status: str
    active_requests: int
    emitted_at_unix_ms: int
    def __init__(self, reason: _Optional[str] = ..., status: _Optional[str] = ..., active_requests: _Optional[int] = ..., emitted_at_unix_ms: _Optional[int] = ...) -> None: ...

class EndpointConfig(_message.Message):
    __slots__ = ("supported_repo_refs", "repo_ref_by_key", "resolved_repos_by_ref", "required_flavor_refs", "models_by_function", "disabled_functions", "ref_availability_by_function")
    class RepoRefByKeyEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ResolvedReposByRefEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ResolvedRepo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ResolvedRepo, _Mapping]] = ...) -> None: ...
    class ModelsByFunctionEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ModelsByKey
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ModelsByKey, _Mapping]] = ...) -> None: ...
    class RefAvailabilityByFunctionEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: FunctionRefAvailability
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[FunctionRefAvailability, _Mapping]] = ...) -> None: ...
    SUPPORTED_REPO_REFS_FIELD_NUMBER: _ClassVar[int]
    REPO_REF_BY_KEY_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_REPOS_BY_REF_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FLAVOR_REFS_FIELD_NUMBER: _ClassVar[int]
    MODELS_BY_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    DISABLED_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    REF_AVAILABILITY_BY_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    supported_repo_refs: _containers.RepeatedScalarFieldContainer[str]
    repo_ref_by_key: _containers.ScalarMap[str, str]
    resolved_repos_by_ref: _containers.MessageMap[str, ResolvedRepo]
    required_flavor_refs: _containers.RepeatedScalarFieldContainer[str]
    models_by_function: _containers.MessageMap[str, ModelsByKey]
    disabled_functions: _containers.RepeatedCompositeFieldContainer[DisabledFunction]
    ref_availability_by_function: _containers.MessageMap[str, FunctionRefAvailability]
    def __init__(self, supported_repo_refs: _Optional[_Iterable[str]] = ..., repo_ref_by_key: _Optional[_Mapping[str, str]] = ..., resolved_repos_by_ref: _Optional[_Mapping[str, ResolvedRepo]] = ..., required_flavor_refs: _Optional[_Iterable[str]] = ..., models_by_function: _Optional[_Mapping[str, ModelsByKey]] = ..., disabled_functions: _Optional[_Iterable[_Union[DisabledFunction, _Mapping]]] = ..., ref_availability_by_function: _Optional[_Mapping[str, FunctionRefAvailability]] = ...) -> None: ...

class WorkerSchedulerMessage(_message.Message):
    __slots__ = ("worker_registration", "job_result", "load_model_result", "unload_model_result", "worker_event", "worker_diagnostic_log", "incremental_token_delta", "incremental_token_stream_done", "incremental_token_stream_error", "worker_function_unavailable", "worker_drain_result", "worker_startup_phase", "worker_model_ready", "worker_function_capabilities", "worker_kv_prefix_cache", "job_request", "load_model_cmd", "unload_model_cmd", "interrupt_job_cmd", "endpoint_config", "worker_drain_cmd", "download_model_cmd")
    WORKER_REGISTRATION_FIELD_NUMBER: _ClassVar[int]
    JOB_RESULT_FIELD_NUMBER: _ClassVar[int]
    LOAD_MODEL_RESULT_FIELD_NUMBER: _ClassVar[int]
    UNLOAD_MODEL_RESULT_FIELD_NUMBER: _ClassVar[int]
    WORKER_EVENT_FIELD_NUMBER: _ClassVar[int]
    WORKER_DIAGNOSTIC_LOG_FIELD_NUMBER: _ClassVar[int]
    INCREMENTAL_TOKEN_DELTA_FIELD_NUMBER: _ClassVar[int]
    INCREMENTAL_TOKEN_STREAM_DONE_FIELD_NUMBER: _ClassVar[int]
    INCREMENTAL_TOKEN_STREAM_ERROR_FIELD_NUMBER: _ClassVar[int]
    WORKER_FUNCTION_UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    WORKER_DRAIN_RESULT_FIELD_NUMBER: _ClassVar[int]
    WORKER_STARTUP_PHASE_FIELD_NUMBER: _ClassVar[int]
    WORKER_MODEL_READY_FIELD_NUMBER: _ClassVar[int]
    WORKER_FUNCTION_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    WORKER_KV_PREFIX_CACHE_FIELD_NUMBER: _ClassVar[int]
    JOB_REQUEST_FIELD_NUMBER: _ClassVar[int]
    LOAD_MODEL_CMD_FIELD_NUMBER: _ClassVar[int]
    UNLOAD_MODEL_CMD_FIELD_NUMBER: _ClassVar[int]
    INTERRUPT_JOB_CMD_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    WORKER_DRAIN_CMD_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_MODEL_CMD_FIELD_NUMBER: _ClassVar[int]
    worker_registration: WorkerRegistration
    job_result: JobExecutionResult
    load_model_result: LoadModelResult
    unload_model_result: UnloadModelResult
    worker_event: WorkerEvent
    worker_diagnostic_log: WorkerDiagnosticLog
    incremental_token_delta: IncrementalTokenDelta
    incremental_token_stream_done: IncrementalTokenStreamDone
    incremental_token_stream_error: IncrementalTokenStreamError
    worker_function_unavailable: WorkerFunctionUnavailableSignal
    worker_drain_result: WorkerDrainResult
    worker_startup_phase: WorkerStartupPhaseSignal
    worker_model_ready: WorkerModelReadySignal
    worker_function_capabilities: WorkerFunctionCapabilitiesSignal
    worker_kv_prefix_cache: WorkerKVPrefixCache
    job_request: JobExecutionRequest
    load_model_cmd: LoadModelCommand
    unload_model_cmd: UnloadModelCommand
    interrupt_job_cmd: InterruptJobCommand
    endpoint_config: EndpointConfig
    worker_drain_cmd: WorkerDrainCommand
    download_model_cmd: DownloadModelCommand
    def __init__(self, worker_registration: _Optional[_Union[WorkerRegistration, _Mapping]] = ..., job_result: _Optional[_Union[JobExecutionResult, _Mapping]] = ..., load_model_result: _Optional[_Union[LoadModelResult, _Mapping]] = ..., unload_model_result: _Optional[_Union[UnloadModelResult, _Mapping]] = ..., worker_event: _Optional[_Union[WorkerEvent, _Mapping]] = ..., worker_diagnostic_log: _Optional[_Union[WorkerDiagnosticLog, _Mapping]] = ..., incremental_token_delta: _Optional[_Union[IncrementalTokenDelta, _Mapping]] = ..., incremental_token_stream_done: _Optional[_Union[IncrementalTokenStreamDone, _Mapping]] = ..., incremental_token_stream_error: _Optional[_Union[IncrementalTokenStreamError, _Mapping]] = ..., worker_function_unavailable: _Optional[_Union[WorkerFunctionUnavailableSignal, _Mapping]] = ..., worker_drain_result: _Optional[_Union[WorkerDrainResult, _Mapping]] = ..., worker_startup_phase: _Optional[_Union[WorkerStartupPhaseSignal, _Mapping]] = ..., worker_model_ready: _Optional[_Union[WorkerModelReadySignal, _Mapping]] = ..., worker_function_capabilities: _Optional[_Union[WorkerFunctionCapabilitiesSignal, _Mapping]] = ..., worker_kv_prefix_cache: _Optional[_Union[WorkerKVPrefixCache, _Mapping]] = ..., job_request: _Optional[_Union[JobExecutionRequest, _Mapping]] = ..., load_model_cmd: _Optional[_Union[LoadModelCommand, _Mapping]] = ..., unload_model_cmd: _Optional[_Union[UnloadModelCommand, _Mapping]] = ..., interrupt_job_cmd: _Optional[_Union[InterruptJobCommand, _Mapping]] = ..., endpoint_config: _Optional[_Union[EndpointConfig, _Mapping]] = ..., worker_drain_cmd: _Optional[_Union[WorkerDrainCommand, _Mapping]] = ..., download_model_cmd: _Optional[_Union[DownloadModelCommand, _Mapping]] = ...) -> None: ...
