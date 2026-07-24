from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import IO, Literal, Optional

import msgspec


class Asset(msgspec.Struct):
    """Reference to a file in the invoking owner's file store.

    The worker runtime should populate `local_path` before invoking tenant code
    so the function can open/read the file efficiently.

    An Asset can carry its payload one of two ways on the way out
    (see tensorhub/docs/api-conventions.md):

      - Stored: the worker uploaded the bytes to tensorhub/MinIO and
        returns just ``ref`` + the immutable bytes-attestation fields
        (``sha256``, ``blake3``, ``size_bytes``, ``mime_type``,
        ``media_id``). gen-orchestrator re-presigns the download URL on
        every client read via tensorhub's ``POST /api/v1/media/urls``.
        gen-orchestrator issue #318 deleted the upload-time URL caching
        and the asset-receipt JWT.
      - Inline ``inline_bytes``: the worker SKIPPED the tensorhub upload
        and returned the raw bytes directly. Used when the client asked
        for ``Prefer: bytes=inline`` and the payload fits under the
        worker's inline threshold. Faster and cheaper for small
        responses; no S3 round-trip. The bytes flow through worker →
        orchestrator → client as msgpack ``bin`` when the client
        accepts msgpack, or base64-encoded when the client accepts
        JSON.

    Exactly one of ``ref`` and ``inline_bytes`` is set on a successful
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
    download_token: Optional[str] = None
    stream_mode: Optional[str] = None
    inline_bytes: Optional[bytes] = None
    url_max_bytes: Optional[int] = None
    url_allowed_mime_types: tuple[str, ...] = ()
    url_max_width: Optional[int] = None
    url_max_height: Optional[int] = None
    url_max_pixels: Optional[int] = None
    url_validation_context: Optional[str] = None

    def __fspath__(self) -> str:
        # Kept so `open(asset)`, `Path(asset)`, and the wider os.PathLike
        # protocol still work. All other I/O is now free functions in
        # ``gen_worker.io`` — see `gen_worker.io.read_bytes`,
        # `gen_worker.io.open`, `gen_worker.io.exists`,
        # `gen_worker.io.read_image`, `gen_worker.io.read_audio`,
        # `gen_worker.io.write_image`.
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        return self.local_path


class MediaAsset(Asset):
    """Reference to user-supplied media bytes."""


class ImageAsset(MediaAsset):
    """Reference to image media bytes."""


class VideoAsset(MediaAsset):
    """Reference to video media bytes, plus probed container metadata.

    ``ctx.save_video`` fills the media fields by probing the container
    (PyAV, best-effort): ``duration_s``/``fps``/``width``/``height`` from the
    video stream, ``has_audio``/``sample_rate`` from the audio stream. They
    stay ``None`` when the probe is unavailable. ``duration_s`` is MEDIA
    seconds (what per_output_second billing should meter), not wall-clock.
    """

    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_audio: Optional[bool] = None
    sample_rate: Optional[int] = None


class AudioAsset(MediaAsset):
    """Reference to audio media bytes."""


# Named, documented base for endpoint string-enum payload fields. It is a
# direct alias for enum.StrEnum, NOT a subclass: subclassing an empty enum base
# makes members of a tenant's `class Size(StringEnum)` resolve as `str` to mypy
# (an empty-enum-base gap), which forced `# type: ignore[assignment]` on every
# preset default `size: Size = Size.SQUARE`. The alias checks clean (ie#345).
StringEnum = StrEnum


@dataclass(frozen=True)
class ExpectedOutput:
    """Planning metadata for an output media field.

    Use with ``typing.Annotated`` on an output struct field:

        images: Annotated[
            list[ImageAsset],
            ExpectedOutput(
                count="input.num_images",
                width="input.width",
                height="input.height",
                mime_type="image/png",
            ),
        ]

    Values are deliberately small: literals or ``input.<field>`` refs.
    Discovery validates refs and emits plain JSON for Tensorhub /
    gen-orchestrator to evaluate at request submit time.
    """

    count: int | str = 1
    width: int | str | None = None
    height: int | str | None = None
    aspect_ratio: str | None = None
    mime_type: str | None = None
    media_type: Literal["image", "video", "audio", "file", "other"] | None = None
    # Expected MEDIA duration in seconds (video/audio): a literal or an
    # ``input.<field>`` ref to a payload field already denominated in seconds.
    # Derived expressions (num_frames / fps) are not supported — frame-based
    # endpoints leave this unset; settlement uses the probed
    # ``VideoAsset.duration_s`` instead (tensorhub counterpart th#572).
    duration_s: int | str | None = None


@dataclass(frozen=True)
class PromptRole:
    role: Literal["positive", "negative"]

    def __post_init__(self) -> None:
        if self.role not in ("positive", "negative"):
            raise ValueError("PromptRole.role must be 'positive' or 'negative'")


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


class SourceRepo(msgspec.Struct):
    """Reserved-name source descriptor for conversion/training job payloads.

    gen-orchestrator inspects ``payload.source`` by name before dispatching a
    conversion or training job. Endpoints embed this type in their Input struct
    as the reserved-name field ``source``.

    Also reused verbatim (pgw#594) for the second, independent reserved
    field ``text_encoder`` — e.g. a text-encoder repo wholly separate from
    the primary ``source`` model. Materialized the same way, into
    ``ctx.text_encoder_path``.

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


class DatasetRef(msgspec.Struct):
    """Reserved-name dataset descriptor for transform-kind job payloads.

    The executor materializes each ``payload.datasets`` entry into a local
    dataset snapshot before the handler runs (``ctx.dataset_paths[ref]`` /
    ``ctx.resolve_dataset``). Used by calibration-based quant, pruning with
    gradient scoring, distillation, and fine-tuning.

    Capability-token scope: orchestrator adds each dataset's repo_id to the
    token's ``reads`` claim alongside the primary source.

    Fields:
      - ref: "owner/dataset" | "owner/dataset:tag" | "owner/dataset@<checkpoint-id>"
      - checkpoint_id: explicit dataset checkpoint id; highest-priority selector
      - attributes: subset-containment selector against the dataset checkpoint's
        attributes map (tensorhub #229).
      - split: "train" | "validation" | "test" | "calibration" | ... — the
        dataset split the tenant wants.
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


