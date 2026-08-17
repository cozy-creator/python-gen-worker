"""HF safetensors -> GGUF, read and written through tensorfs (pgw#1344).

The property this exists for is NOT that it is cheaper than handing a
materialized model tree to ``convert_hf_to_gguf.py``. It is that **the GGUF
composes through the store**: a tensor whose GGUF bytes equal its safetensors
bytes is admitted under the digest it already has, so a GGUF flavor of a model
the store already holds costs only the tensors that genuinely change.

Two facts make that work, and both are load-bearing:

* the CAS object grid is **per tensor** -- a tensor of at least
  ``MAX_CHUNK_SIZE`` gets objects cut from its own start in both containers, so
  its objects are named by its own bytes and nothing else;
* GGUF's per-tensor alignment padding is a **separate** object
  (``TensorWriter._gguf_domains``), so padding never contaminates a tensor's
  digest.

Cross-container inheritance-by-reference is refused by
:meth:`TensorWriter.inherit` -- a safetensors ``TensorView`` cannot be handed
into a GGUF file, because the two containers disagree about dimension order.
Sharing is therefore obtained the other way, and it is not a weaker result:
:meth:`TensorWriter.add` puts the bytes through the content-addressed store,
which returns the *existing* object for bytes it already holds. The saving is
the same one -- nothing is rewritten and the hub has nothing new to fetch --
and it is measured here rather than assumed, by
:class:`ChunkSharing`.

**What this module does not do.** It does not invent a model's metadata block.
GGUF metadata is a model's vocabulary, its chat template and its
hyperparameters, and reproducing a tokenizer faithfully is
``convert_hf_to_gguf.py``'s several thousand lines, not this file's. The
hyperparameter half IS mechanical and is built here from the natively-read
``config.json``; the vocabulary half must be supplied by a caller that has a
faithful source for it. A converter that guessed a vocabulary would be exactly
the silent change this lane exists to avoid.
"""

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
    """This converter will not produce a GGUF for this input.

    Raised rather than guessed. Every message names the exact thing that is
    missing, because the failure mode this class prevents is a GGUF that loads
    and is quietly wrong.
    """


# The GGUF encodings this converter emits directly. Everything below is a pure
# container/dtype decision; the k-quants are `llama-quantize`'s, and they run
# over a GGUF this converter produced rather than over a model tree.
DIRECT_ENCODINGS: frozenset[str] = frozenset({"f32", "f16", "bf16"})

# safetensors dtype -> ggml type name, for the dtypes that mean the same bytes
# in both containers. Anything absent has to be cast rather than carried.
_SHARED_DTYPES: dict[str, str] = {"F32": "F32", "F16": "F16", "BF16": "BF16"}

_ENCODING_DTYPE: dict[str, str] = {"f32": "F32", "f16": "F16", "bf16": "BF16"}

# `config.json` model_type -> the gguf-py architecture whose tensor name map
# and metadata keys apply. Deliberately short: an architecture is on this list
# when its HF->GGUF mapping is exactly the name map plus the rope permute
# below, and NOT when it merely has a `MODEL_ARCH` entry. Gemma rewrites norm
# weights, and the qwen MoE families restack experts; neither is a name map,
# so neither is here -- see this module's docstring on guessing.
_ARCHITECTURES: dict[str, str] = {
    "llama": "LLAMA",
    "mistral": "LLAMA",
    "qwen2": "QWEN2",
}

# The tensors the llama-family rope permute applies to, by their GGUF names.
# `convert_hf_to_gguf.py::LlamaModel.modify_tensors` permutes exactly these.
_PERMUTED_SUFFIXES = ("attn_q.weight", "attn_k.weight")


@dataclass(frozen=True, slots=True)
class TensorDisposition:
    """What happened to one tensor, and therefore what it cost the store."""

    source_name: str
    gguf_name: str
    dtype: str
    shape: tuple[int, ...]
    # "passthrough" when the bytes are the source's, verbatim; otherwise the
    # transform that changed them. Only "passthrough" can share objects.
    transform: str
    objects: tuple[str, ...]

    @property
    def passthrough(self) -> bool:
        return self.transform == "passthrough"


@dataclass(frozen=True, slots=True)
class ChunkSharing:
    """The dedup measurement, per tensor rather than in aggregate.

    An aggregate count cannot tell "shared because the bytes are identical"
    from "new because the tensor was transformed", and the second is not a
    defect. This keeps them apart by name, so a regression reads as a NAMED
    tensor that stopped sharing.
    """

    shared: tuple[str, ...]
    transformed: tuple[str, ...]
    # A passthrough tensor whose objects are NOT in the source: the source
    # packed it with its neighbours, so it has no digest of its own to share.
    # Not a defect -- a fact about the source's grid -- but never silent.
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


# ---------------------------------------------------------------------------
# metadata (the hyperparameter half only -- see the module docstring)
# ---------------------------------------------------------------------------


def _architecture(config: Mapping[str, object]) -> tuple[str, str]:
    """(model_type, gguf-py MODEL_ARCH name) for a natively-read config.json."""

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
    """Encode the GGUF key/value block for one model, minus its vocabulary.

    Returns ``(metadata bytes, key count, MODEL_ARCH name)``. The bytes are
    produced by ``gguf-py``'s own packer -- the same encoder
    ``convert_hf_to_gguf.py`` writes with -- so this is a re-drive of the
    reference encoding rather than a second implementation of it.

    ``vocabulary`` is an already-encoded key/value run appended verbatim, for a
    caller holding a faithful tokenizer block. It is NOT synthesized here; a
    GGUF whose vocabulary this converter guessed would load and be wrong.
    """

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
        # `_pack_val` IS gguf-py's key/value encoder; `write_kv_data_to_file`
        # is the same two calls with a file on the end. Reaching for it keeps
        # one encoder in play instead of a second one written here.
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
    """GQA's KV head count, defaulting to full multi-head when absent."""

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


# ---------------------------------------------------------------------------
# the tensor plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Planned:
    view: TensorView
    gguf_name: str
    dtype: str
    ne: tuple[int, ...]
    transform: str
    # Non-zero only for a rope-permuted tensor: the head count its permute is
    # cut by (q takes the attention heads, k takes the KV heads).
    permute_heads: int = 0


def plan_tensors(
    reader: TensorReader,
    config: Mapping[str, object],
    *,
    encoding: str,
) -> tuple[_Planned, ...]:
    """Decide, per tensor, its GGUF name, dtype and whether its bytes change.

    Nothing is read here. The plan is what makes the dedup measurement legible
    afterwards: a tensor's ``transform`` says up front whether its objects are
    allowed to be the source's.
    """

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
                # GGUF's `ne` is the reverse of safetensors' row-major shape.
                ne=tuple(reversed(view.shape)),
                transform=transform,
                permute_heads=heads,
            )
        )
    if not planned:
        raise GGUFUnsupported("gguf_no_tensors: the snapshot declares no tensors")
    return tuple(planned)


def _target_dtype(gguf_name: str, view: TensorView, target: str) -> str:
    """`convert_hf_to_gguf.py::prepare_tensors`' dtype rule, verbatim.

    One dimension, or a norm weight, stays F32 whatever was asked for -- these
    are the tensors whose precision llama.cpp will not trade. Everything else
    takes the requested encoding.
    """

    if len(view.shape) <= 1 or gguf_name.endswith("_norm.weight"):
        return "F32"
    return target


# ---------------------------------------------------------------------------
# the conversion
# ---------------------------------------------------------------------------


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
    """Compose a GGUF from a snapshot, admitting only what actually changed.

    Reads tensors through :class:`TensorReader` and writes through
    :class:`TensorWriter`. No directory is projected, no file is written, and
    peak resident memory is one tensor -- the streaming property
    ``convert/writer.py`` already has, preserved because
    :meth:`TensorWriter.add` takes an iterable of buffers.
    """

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
            # The writer recomputes the directory from what it was handed; only
            # version/alignment/metadata are read off this.
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
            # The writer records each tensor's admitted objects as it goes, so
            # the dedup measurement is read off the real admission rather than
            # recomputed from the bytes a second time.
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
    """The tensor's GGUF bytes, streamed when nothing has to change.

    A passthrough tensor is handed to the writer as the source's own buffers,
    so it is never contiguous in memory and the store recognises its objects by
    their digests. A transformed one is materialized once -- the transform
    needs the whole tensor -- and released as soon as it is admitted.
    """

    if item.transform == "passthrough":
        # `pieces()` yields `memoryview`s -- buffers, which is what the writer
        # documents it takes; the annotation is `bytes` because that is the
        # common case, not because a copy is wanted here.
        return cast("Iterable[bytes]", item.view.pieces())
    return (_transformed(item),)


def _transformed(item: _Planned) -> bytes:
    """The tensor's bytes after its declared transform, and nothing else.

    The permute is done on RAW ELEMENTS -- it is an axis swap, exact at any
    width, so it needs no float semantics and works on bf16 without a library
    that knows what bf16 is. The dtype cast is done by torch, because numpy has
    no bfloat16 and hand-rolling round-to-nearest-even is exactly the silent
    numerics change this lane exists to avoid.
    """

    import numpy as np

    view = item.view
    data = view.tobytes()
    if item.permute_heads:
        raw = np.frombuffer(data, dtype=_RAW_DTYPE[_ITEMSIZE[view.dtype]])
        array = raw.reshape(view.shape)
        heads = item.permute_heads
        # `convert_hf_to_gguf.py::LlamaModel.permute`, verbatim.
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
    """A float cast, done by torch so the rounding is the fleet's own."""

    import torch

    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    # `frombuffer` wants a writable buffer; the copy is one tensor, which is
    # this function's whole working set either way.
    tensor = torch.frombuffer(bytearray(data), dtype=dtypes[source])
    result = tensor.to(dtypes[target]).contiguous()
    # bf16 has no numpy counterpart, so the bytes come out through an integer
    # view of the same width rather than through a dtype numpy must understand.
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
