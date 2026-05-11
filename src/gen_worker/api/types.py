from __future__ import annotations

import os
from typing import IO, Optional

import msgspec


class Asset(msgspec.Struct):
    """Reference to a file in the invoking owner's file store.

    The worker runtime should populate `local_path` before invoking tenant code
    so the function can open/read the file efficiently.

    An Asset can carry its payload one of two ways on the way out
    (see tensorhub/docs/api-conventions.md):

      - URL: ``url`` + ``url_expires_at`` point at a presigned download.
        The worker uploaded the bytes to tensorhub/MinIO and got back a
        signed receipt (``receipt_jws``).
      - Inline ``inline_bytes``: the worker SKIPPED the tensorhub upload
        and returned the raw bytes directly. Used when the client asked
        for ``Prefer: bytes=inline`` and the payload fits under the
        worker's inline threshold. Faster and cheaper for small
        responses; no S3 round-trip. The bytes flow through worker →
        orchestrator → client as msgpack ``bin`` when the client
        accepts msgpack, or base64-encoded when the client accepts
        JSON.

    Exactly one of ``url`` and ``inline_bytes`` is set on a successful
    output Asset. The field is named ``inline_bytes`` rather than
    ``bytes`` to avoid the name collision with Python's builtin
    ``bytes`` (which breaks msgspec.Struct decoding).
    """

    ref: str
    owner: Optional[str] = None
    local_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    blake3: Optional[str] = None
    media_id: Optional[str] = None
    url: Optional[str] = None
    url_expires_at: Optional[str] = None
    receipt_jws: Optional[str] = None
    download_token: Optional[str] = None
    stream_mode: Optional[str] = None
    inline_bytes: Optional[bytes] = None

    def __fspath__(self) -> str:
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        return self.local_path

    def open(self, mode: str = "rb") -> IO[bytes]:
        if "b" not in mode:
            raise ValueError("Asset.open only supports binary modes")
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        return open(self.local_path, mode)

    def exists(self) -> bool:
        if self.local_path is None:
            return False
        return os.path.exists(self.local_path)

    def read_bytes(self, max_bytes: Optional[int] = None) -> bytes:
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        with open(self.local_path, "rb") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("asset too large to read into memory")
        return data


class Tensors(msgspec.Struct):
    """Reference to checkpoint/model-weight artifacts.

    This mirrors `Asset` behavior but gives tensor/checkpoint payloads a
    first-class type for training/conversion code paths.
    """

    ref: str
    owner: Optional[str] = None
    local_path: Optional[str] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    blake3: Optional[str] = None
    blob_digest: Optional[str] = None
    blob_domain: Optional[str] = None
    blob_path: Optional[str] = None
    snapshot_digest: Optional[str] = None
    download_token: Optional[str] = None
    stream_mode: Optional[str] = None

    def __fspath__(self) -> str:
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        return self.local_path

    def open(self, mode: str = "rb") -> IO[bytes]:
        if "b" not in mode:
            raise ValueError("Tensors.open only supports binary modes")
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        return open(self.local_path, mode)

    def exists(self) -> bool:
        if self.local_path is None:
            return False
        return os.path.exists(self.local_path)

    def read_bytes(self, max_bytes: Optional[int] = None) -> bytes:
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        with open(self.local_path, "rb") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("tensors file too large to read into memory")
        return data


class LoraSpec(msgspec.Struct):
    """A LoRA adapter to load for a single inference request.

    ``file`` is materialized by the worker before the function runs, so
    ``file.local_path`` is guaranteed to be set when your function executes.
    ``weight`` controls the adapter scale (fuse strength).
    ``adapter_name`` is optional; if omitted the worker assigns ``lora_0``,
    ``lora_1``, ... based on list position.
    """

    file: Asset
    weight: float = 1.0
    adapter_name: Optional[str] = None


class SourceRepo(msgspec.Struct):
    """Reserved-name source descriptor for conversion/training job payloads.

    gen-orchestrator inspects ``payload.source`` by name before dispatching a
    conversion or training job. Endpoints embed this type in their Input struct
    as the reserved-name field ``source``.

    Fields:
      - ref: "owner/repo" | "owner/repo:tag[#flavor...]" | "owner/repo@<checkpoint-id>"
      - checkpoint_id: explicit content-addressed checkpoint id; highest-priority selector
      - attributes: subset-containment selector against the checkpoint flavor's attributes
        map. Well-known keys include dtype, file_layout, file_type, quant_library,
        plus family-specific keys (quant_bits, quant_group_size, quant_sym,
        quant_desc_act, quant_block_size, quant_double_quant, quant_compute_dtype,
        quant_layout, quant_granularity, quant_activation_scheme, etc.). See
        tensorhub ``docs/checkpoint-flavors.md`` for the namespace.
    """

    ref: str
    checkpoint_id: Optional[str] = None
    attributes: dict = msgspec.field(default_factory=dict)


class DestinationRepo(msgspec.Struct):
    """Reserved-name destination descriptor for conversion/training job payloads.

    gen-orchestrator inspects ``payload.destination`` by name before dispatch.
    The library applies ``tags`` against the newly-produced checkpoint after the
    tenant function returns success (atomic per-tag move; empty list = no-op).
    """

    ref: str
    tags: list = msgspec.field(default_factory=list)


class DatasetRef(msgspec.Struct):
    """Reserved-name dataset descriptor for transform-kind job payloads.

    Each entry in ``payload.datasets`` materializes into a :class:`Dataset`
    object the tenant function receives. Used by calibration-based quant,
    pruning with gradient scoring, distillation, and fine-tuning.

    Capability-token scope: orchestrator adds each dataset's repo_id to the
    token's ``reads`` claim alongside the primary source.

    Fields:
      - ref: "owner/dataset" | "owner/dataset:tag" | "owner/dataset@<checkpoint-id>"
      - checkpoint_id: explicit dataset checkpoint id; highest-priority selector
      - attributes: subset-containment selector against the dataset checkpoint's
        attributes map (tensorhub #229).
      - split: "train" | "validation" | "test" | "calibration" | ... — the
        dataset split the tenant wants. Library materializes only this split.
    """

    ref: str
    checkpoint_id: Optional[str] = None
    attributes: dict = msgspec.field(default_factory=dict)
    split: str = "train"


class OutputSpec(msgspec.Struct):
    """Describes one variant a conversion endpoint will emit into the destination checkpoint.

    Every entry in ``payload.outputs`` produces one variant on the destination
    repo's new checkpoint. Tenant code may augment ``attributes`` at upload time
    (e.g. record runtime-discovered provenance like quant_library_version);
    the final stored attribute bag is OutputSpec.attributes ∪ tenant-augmented.

    The attribute map must be complete enough to pass tensorhub's per-family
    variant-attribute validation — missing required keys for the declared
    quant_library (bitsandbytes, gptqmodel, autoawq, torchao, modelopt, gguf)
    are rejected at commit time. See tensorhub ``docs/variant_attributes.md``.
    """

    attributes: dict = msgspec.field(default_factory=dict)


class Compute(msgspec.Struct, frozen=True):
    """The resolved hardware specification for one invocation.

    Populated by gen-orchestrator at dispatch:
      - For inference functions: equals the endpoint's ``[resources]``.
      - For training functions: endpoint ``[resources]`` merged with the
        invoker's ``compute`` overrides from the wire payload.

    Surfaced to tenant code read-only via ``RequestContext.compute``.
    Architecture axes (accelerator, cuda_compute_min) are always pinned by
    the endpoint image; invoker overrides on those are rejected at submit.

    See tensorhub issue #232 for the full contract.
    """

    accelerator: str = ""
    cuda_compute_min: str = ""
    vram_gb: int = 0
    gpu_count: int = 0
    gpu_tier: Optional[str] = None
    memory_gb: int = 0
    cpu_cores: int = 0
    disk_gb: int = 0
