from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .._vendor.tensorfs import LocalCAS, RepositoryManifest, TensorWriter, open_tensors
from .._vendor.tensorfs import gguf as tfs_gguf
from .._vendor.tensorfs.manifest import FileEntry
from .._vendor.tensorfs.tensors import TensorReader, TensorView

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkSharing",
    "GGUFConversion",
    "GGUFUnsupported",
    "TensorDisposition",
    "convert_snapshot_to_gguf",
    "hyperparameter_metadata",
    "plan_tensors",
]


class GGUFUnsupported(ValueError):
    """This converter will not produce a GGUF for this input."""


DIRECT_ENCODINGS: frozenset[str] = frozenset({"f32", "f16", "bf16"})

_SHARED_DTYPES: dict[str, str] = {"F32": "F32", "F16": "F16", "BF16": "BF16"}

_ENCODING_DTYPE: dict[str, str] = {"f32": "F32", "f16": "F16", "bf16": "BF16"}

_ARCHITECTURES: dict[str, str] = {
    "llama": "LLAMA",
    "mistral": "LLAMA",
    "qwen2": "QWEN2",
}

_PERMUTED_SUFFIXES = ("attn_q.weight", "attn_k.weight")


@dataclass(frozen=True, slots=True)
class TensorDisposition:
    """What happened to one tensor, and therefore what it cost the store."""

    source_name: str
    gguf_name: str
    dtype: str
    shape: tuple[int, ...]
    transform: str
    objects: tuple[str, ...]

    @property
    def passthrough(self) -> bool:
        return self.transform == "passthrough"


@dataclass(frozen=True, slots=True)
class ChunkSharing:
    """The dedup measurement, per tensor rather than in aggregate."""

    shared: tuple[str, ...]
    transformed: tuple[str, ...]
    unshareable: tuple[str, ...]
    shared_digests: frozenset[str]
    new_digests: frozenset[str]

    def describe(self) -> str:
        return (
            f"shared={len(self.shared)} tensors / {len(self.shared_digests)} objects; "
            f"transformed={len(self.transformed)}; "
            f"unshareable={len(self.unshareable)}; "
            f"new objects={len(self.new_digests)}"
        )


@dataclass(frozen=True, slots=True)
class GGUFConversion:
    entry: FileEntry
    dispositions: tuple[TensorDisposition, ...]
    sharing: ChunkSharing


def _architecture(config: Mapping[str, object]) -> tuple[str, str]:

    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type in _ARCHITECTURES:
        return model_type, _ARCHITECTURES[model_type]
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        for raw in architectures:
            cleaned = str(raw or "").strip().lower()
            for prefix, arch in _ARCHITECTURES.items():
                if cleaned.startswith(prefix):
                    return prefix, arch
    raise GGUFUnsupported(
        f"gguf_unsupported_architecture:{model_type or architectures!r}; this "
        f"converter maps {sorted(_ARCHITECTURES)} by name map plus rope permute "
        f"alone. An architecture that rewrites tensor VALUES needs its rule "
        f"written down before it can be converted here."
    )


def hyperparameter_metadata(
    config: Mapping[str, object],
    *,
    encoding: str,
    vocabulary: bytes = b"",
    vocabulary_count: int = 0,
) -> tuple[bytes, int, str]:
    """Encode the GGUF key/value block for one model, minus its vocabulary."""

    import gguf as gguf_py

    _model_type, arch_name = _architecture(config)
    arch = getattr(gguf_py.MODEL_ARCH, arch_name)
    writer = gguf_py.GGUFWriter(path=None, arch=gguf_py.MODEL_ARCH_NAMES[arch])

    block_count = _positive_int(config, "num_hidden_layers")
    writer.add_block_count(block_count)
    writer.add_context_length(_positive_int(config, "max_position_embeddings"))
    writer.add_embedding_length(_positive_int(config, "hidden_size"))
    writer.add_feed_forward_length(_positive_int(config, "intermediate_size"))
    writer.add_head_count(_positive_int(config, "num_attention_heads"))
    writer.add_head_count_kv(_head_count_kv(config))
    writer.add_layer_norm_rms_eps(_number(config, "rms_norm_eps", 1e-5))
    if config.get("rope_theta") is not None:
        writer.add_rope_freq_base(_number(config, "rope_theta", 10000.0))
    writer.add_vocab_size(_positive_int(config, "vocab_size"))
    writer.add_file_type(_file_type(encoding))

    count = len(writer.kv_data[0])
    encoded = bytearray()
    for key, value in writer.kv_data[0].items():
        encoded += writer._pack_val(key, gguf_py.GGUFValueType.STRING, add_vtype=False)
        encoded += writer._pack_val(
            value.value, value.type, add_vtype=True, sub_type=value.sub_type
        )
    if vocabulary:
        if vocabulary_count <= 0:
            raise ValueError("a vocabulary block must declare its key count")
        encoded += vocabulary
        count += vocabulary_count
    return bytes(encoded), count, arch_name


def _head_count_kv(config: Mapping[str, object]) -> int:

    raw = config.get("num_key_value_heads")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return _positive_int(config, "num_attention_heads")


def _number(config: Mapping[str, object], key: str, default: float) -> float:
    raw = config.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    return float(raw)


def _positive_int(config: Mapping[str, object], key: str) -> int:
    raw = config.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise GGUFUnsupported(f"gguf_config_missing:{key}")
    return raw


def _file_type(encoding: str) -> int:
    import gguf as gguf_py

    match encoding:
        case "f32":
            return int(gguf_py.LlamaFileType.ALL_F32)
        case "f16":
            return int(gguf_py.LlamaFileType.MOSTLY_F16)
        case "bf16":
            return int(gguf_py.LlamaFileType.MOSTLY_BF16)
    raise GGUFUnsupported(
        f"gguf_unsupported_encoding:{encoding}; direct encodings are "
        f"{sorted(DIRECT_ENCODINGS)} (k-quants run over this converter's "
        f"output, not over a model tree)"
    )


@dataclass(frozen=True, slots=True)
class _Planned:
    view: TensorView
    gguf_name: str
    dtype: str
    ne: tuple[int, ...]
    transform: str
    permute_heads: int = 0


def plan_tensors(
    reader: TensorReader,
    config: Mapping[str, object],
    *,
    encoding: str,
) -> tuple[_Planned, ...]:
    """Decide, per tensor, its GGUF name, dtype and whether its bytes change."""

    import gguf as gguf_py

    _model_type, arch_name = _architecture(config)
    arch = getattr(gguf_py.MODEL_ARCH, arch_name)
    block_count = _positive_int(config, "num_hidden_layers")
    name_map = gguf_py.TensorNameMap(arch, block_count)
    target = _ENCODING_DTYPE.get(encoding)
    if target is None:
        raise GGUFUnsupported(f"gguf_unsupported_encoding:{encoding}")

    head_count = _positive_int(config, "num_attention_heads")
    head_count_kv = _head_count_kv(config)

    planned: list[_Planned] = []
    for name in reader:
        view = reader[name]
        if name.endswith((".attn_bias", ".rotary_emb.inv_freq")):
            continue
        gguf_name = name_map.get_name(name, try_suffixes=(".weight", ".bias"))
        if gguf_name is None:
            raise GGUFUnsupported(
                f"gguf_unmapped_tensor:{name}; gguf-py's name map for "
                f"{arch_name} has no GGUF name for it"
            )

        dtype = _target_dtype(gguf_name, view, target)
        permute = arch_name == "LLAMA" and gguf_name.endswith(_PERMUTED_SUFFIXES)
        heads = 0
        if permute:
            transform = "permute"
            heads = head_count_kv if gguf_name.endswith("attn_k.weight") else head_count
        elif dtype != _SHARED_DTYPES.get(view.dtype, view.dtype):
            transform = f"cast:{view.dtype}->{dtype}"
        else:
            transform = "passthrough"
        planned.append(
            _Planned(
                view=view,
                gguf_name=gguf_name,
                dtype=dtype,
                ne=tuple(reversed(view.shape)),
                transform=transform,
                permute_heads=heads,
            )
        )
    if not planned:
        raise GGUFUnsupported("gguf_no_tensors: the snapshot declares no tensors")
    return tuple(planned)


def _target_dtype(gguf_name: str, view: TensorView, target: str) -> str:

    if len(view.shape) <= 1 or gguf_name.endswith("_norm.weight"):
        return "F32"
    return target


def convert_snapshot_to_gguf(
    cas: LocalCAS,
    manifest: RepositoryManifest,
    *,
    encoding: str,
    metadata: bytes,
    metadata_count: int,
    path: str = "model.gguf",
    alignment: int = 32,
    config: Mapping[str, object] | None = None,
) -> GGUFConversion:
    """Compose a GGUF from a snapshot, admitting only what actually changed."""

    reader = open_tensors(cas, manifest)
    try:
        if config is None:
            config = json.loads(reader.read_file("config.json").decode("utf-8"))
        assert config is not None
        planned = plan_tensors(reader, config, encoding=encoding)

        header = tfs_gguf.GGUFHeader(
            version=3,
            alignment=alignment,
            metadata_count=metadata_count,
            metadata=metadata,
            directory_start=0,
            directory_end=0,
            data_start=0,
            tensors=(),
        )
        writer = TensorWriter(cas, path, gguf_header=header)

        source_digests = {
            str(chunk.digest) for entry in manifest.files for chunk in entry.chunks
        }
        dispositions: list[TensorDisposition] = []
        for item in planned:
            writer.add(item.gguf_name, item.dtype, item.ne, _bytes_for(item))
            objects = tuple(str(ref) for ref, _length in writer._pending[-1].chunks or ())
            dispositions.append(
                TensorDisposition(
                    source_name=item.view.name,
                    gguf_name=item.gguf_name,
                    dtype=item.dtype,
                    shape=item.ne,
                    transform=item.transform,
                    objects=objects,
                )
            )
        entry = writer.finish()
    finally:
        reader.close()

    sharing = _measure(dispositions, source_digests)
    logger.info(
        "gguf.native path=%s encoding=%s tensors=%d %s",
        path, encoding, len(dispositions), sharing.describe(),
    )
    return GGUFConversion(entry=entry, dispositions=tuple(dispositions), sharing=sharing)


def _bytes_for(item: _Planned) -> Iterable[bytes]:

    if item.transform == "passthrough":
        return cast("Iterable[bytes]", item.view.pieces())
    return (_transformed(item),)


def _transformed(item: _Planned) -> bytes:

    import numpy as np

    view = item.view
    data = view.tobytes()
    if item.permute_heads:
        raw = np.frombuffer(data, dtype=_RAW_DTYPE[_ITEMSIZE[view.dtype]])
        array = raw.reshape(view.shape)
        heads = item.permute_heads
        array = (
            array.reshape(heads, 2, array.shape[0] // heads // 2, *array.shape[1:])
            .swapaxes(1, 2)
            .reshape(array.shape)
        )
        data = np.ascontiguousarray(array).tobytes()
    if item.dtype != _SHARED_DTYPES.get(view.dtype, view.dtype):
        data = _cast(data, view.dtype, item.dtype)
    return data


_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}
_RAW_DTYPE = {4: "<u4", 2: "<u2", 1: "u1"}


def _cast(data: bytes, source: str, target: str) -> bytes:

    import torch

    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    tensor = torch.frombuffer(bytearray(data), dtype=dtypes[source])
    result = tensor.to(dtypes[target]).contiguous()
    integral = {4: torch.int32, 2: torch.int16}[_ITEMSIZE[target]]
    return bytes(result.view(integral).numpy().tobytes())


def _measure(
    dispositions: Sequence[TensorDisposition], source_digests: set[str]
) -> ChunkSharing:
    shared: list[str] = []
    transformed: list[str] = []
    unshareable: list[str] = []
    shared_digests: set[str] = set()
    new_digests: set[str] = set()
    for item in dispositions:
        hit = {digest for digest in item.objects if digest in source_digests}
        miss = {digest for digest in item.objects if digest not in source_digests}
        shared_digests |= hit
        new_digests |= miss
        if not item.passthrough:
            transformed.append(item.gguf_name)
        elif miss:
            unshareable.append(item.gguf_name)
        else:
            shared.append(item.gguf_name)
    return ChunkSharing(
        shared=tuple(shared),
        transformed=tuple(transformed),
        unshareable=tuple(unshareable),
        shared_digests=frozenset(shared_digests),
        new_digests=frozenset(new_digests),
    )
