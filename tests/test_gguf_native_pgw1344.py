from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from gen_worker._vendor.tensorfs import LocalCAS, RepositoryManifest
from cas_fixture import ingest_repository
from gen_worker._vendor.tensorfs.manifest import MAX_CHUNK_SIZE
from gen_worker.convert.gguf_native import (
    GGUFUnsupported,
    convert_snapshot_to_gguf,
    hyperparameter_metadata,
)

HIDDEN = 4096
KV_ROWS = 512
VOCAB = 256

CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 1,
    "hidden_size": HIDDEN,
    "intermediate_size": HIDDEN,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "vocab_size": VOCAB,
}

FIXTURE: dict[str, tuple[str, tuple[int, ...], int]] = {
    "model.embed_tokens.weight": ("F32", (VOCAB, HIDDEN), 0x11),
    "model.layers.0.input_layernorm.weight": ("F32", (HIDDEN,), 0x22),
    "model.layers.0.self_attn.q_proj.weight": ("F32", (HIDDEN, HIDDEN), 0x33),
    "model.layers.0.self_attn.k_proj.weight": ("F32", (KV_ROWS, HIDDEN), 0x44),
    "model.layers.0.self_attn.v_proj.weight": ("F32", (HIDDEN, HIDDEN), 0x55),
    "model.layers.0.self_attn.o_proj.weight": ("F32", (HIDDEN, HIDDEN), 0x66),
    "model.layers.0.post_attention_layernorm.weight": ("F32", (HIDDEN,), 0x77),
    "model.norm.weight": ("F32", (HIDDEN,), 0x88),
}

_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


def _content(shape: tuple[int, ...], seed: int) -> bytes:

    import numpy as np

    elements = 1
    for dimension in shape:
        elements *= dimension
    values = (np.arange(elements, dtype=np.float32) + seed) % 997.0
    return values.tobytes()


def _safetensors_bytes(tensors: dict[str, tuple[str, tuple[int, ...], int]]) -> bytes:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    cursor = 0
    for name, (dtype, shape, seed) in tensors.items():
        length = _ITEMSIZE[dtype]
        for dimension in shape:
            length *= dimension
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + length],
        }
        cursor += length
        blobs.append(_content(shape, seed))
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(blobs)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> tuple[LocalCAS, RepositoryManifest]:
    """A synthetic HF snapshot, resident in a real CAS."""

    root = tmp_path_factory.mktemp("pgw1344")
    cas = LocalCAS(root / "cas")
    staged = root / "staged"
    staged.mkdir()
    (staged / "model.safetensors").write_bytes(_safetensors_bytes(FIXTURE))
    (staged / "config.json").write_text(json.dumps(CONFIG))
    manifest = ingest_repository(cas, staged)
    for path in sorted(staged.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    staged.rmdir()
    return cas, manifest


def _metadata(encoding: str) -> tuple[bytes, int]:
    blob, count, arch = hyperparameter_metadata(CONFIG, encoding=encoding)
    assert arch == "LLAMA"
    return blob, count


def _source_digests(manifest: RepositoryManifest) -> set[str]:
    return {str(chunk.digest) for entry in manifest.files for chunk in entry.chunks}


def test_the_source_grid_gives_the_large_tensors_objects_of_their_own(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """The precondition the whole property rests on, asserted rather than hoped."""

    _cas, manifest = snapshot
    entry = next(e for e in manifest.files if e.path == "model.safetensors")
    lengths = [chunk.length for chunk in entry.chunks]
    assert lengths.count(MAX_CHUNK_SIZE) >= 2, (
        f"the fixture's 64 MiB tensors did not get objects of their own: {lengths}"
    )


def test_an_unpermuted_tensor_shares_its_objects_with_its_safetensors_source(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:

    cas, manifest = snapshot
    blob, count = _metadata("f32")
    result = convert_snapshot_to_gguf(
        cas, manifest, encoding="f32", metadata=blob, metadata_count=count
    )
    sharing = result.sharing

    assert set(sharing.shared) == {
        "token_embd.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
    }
    assert set(sharing.transformed) == {"blk.0.attn_q.weight", "blk.0.attn_k.weight"}

    assert set(sharing.unshareable) == set(), sharing.describe()

    source = _source_digests(manifest)
    passthrough = [d for d in result.dispositions if d.passthrough and d.objects]
    assert passthrough, "the plan transformed every tensor; nothing could share"
    for item in passthrough:
        if item.gguf_name in sharing.unshareable:
            continue
        assert set(item.objects) <= source, (
            f"{item.gguf_name} passes through unchanged but was admitted as new "
            f"objects -- the GGUF stopped composing through the store"
        )

    for item in result.dispositions:
        if item.transform == "permute":
            assert not (set(item.objects) & source), (
                f"{item.gguf_name} was permuted but kept its source objects"
            )


def test_a_dtype_cast_legitimately_shares_nothing(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """f16 from an f32 source rewrites every 2-D tensor, and says so."""

    cas, manifest = snapshot
    blob, count = _metadata("f16")
    result = convert_snapshot_to_gguf(
        cas, manifest, encoding="f16", metadata=blob, metadata_count=count
    )
    two_dimensional = [d for d in result.dispositions if len(d.shape) > 1]
    assert two_dimensional
    assert all(not d.passthrough for d in two_dimensional), (
        "a cast to f16 left a 2-D tensor claiming passthrough"
    )
    assert any(d.passthrough for d in result.dispositions if len(d.shape) == 1)


def test_the_norms_keep_f32_whatever_encoding_was_asked_for(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """`convert_hf_to_gguf.py::prepare_tensors`' rule, not this file's opinion."""

    cas, manifest = snapshot
    blob, count = _metadata("bf16")
    result = convert_snapshot_to_gguf(
        cas, manifest, encoding="bf16", metadata=blob, metadata_count=count
    )
    by_name = {d.gguf_name: d for d in result.dispositions}
    assert by_name["blk.0.attn_norm.weight"].dtype == "F32"
    assert by_name["output_norm.weight"].dtype == "F32"
    assert by_name["blk.0.attn_v.weight"].dtype == "BF16"


def test_the_conversion_reads_no_directory(
    snapshot: tuple[LocalCAS, RepositoryManifest], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam is not merely unused -- it is unreachable from this path."""

    import gen_worker.models.materialized_view as materialized_view

    def forbidden(path: object, *, why: str) -> Path:
        raise AssertionError(f"the native GGUF conversion projected a tree: {why}")

    monkeypatch.setattr(materialized_view, "third_party_dir", forbidden)
    cas, manifest = snapshot
    blob, count = _metadata("f32")
    result = convert_snapshot_to_gguf(
        cas, manifest, encoding="f32", metadata=blob, metadata_count=count
    )
    assert result.entry.path == "model.gguf"
    assert result.entry.size_bytes > 0


def test_the_output_parses_as_a_gguf_the_store_can_read_back(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """A composed file is only real if the reader that plans GGUF can walk it."""

    from gen_worker._vendor.tensorfs import open_tensors

    cas, manifest = snapshot
    blob, count = _metadata("f32")
    result = convert_snapshot_to_gguf(
        cas, manifest, encoding="f32", metadata=blob, metadata_count=count
    )
    reader = open_tensors(cas, RepositoryManifest((result.entry,)))
    try:
        header = reader.gguf_header("model.gguf")
        assert header.metadata_count == count
        names = {tensor.name for tensor in header.tensors}
        assert "blk.0.attn_q.weight" in names
        assert "output_norm.weight" in names
        v = next(t for t in header.tensors if t.name == "blk.0.attn_v.weight")
        assert v.shape == (HIDDEN, HIDDEN)
        assert reader["blk.0.attn_v.weight"].nbytes == MAX_CHUNK_SIZE
    finally:
        reader.close()


def test_an_architecture_without_a_written_down_rule_is_refused(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """Gemma rewrites norm VALUES; a name map cannot express that."""

    cas, manifest = snapshot
    gemma = dict(CONFIG, model_type="gemma", architectures=["GemmaForCausalLM"])
    with pytest.raises(GGUFUnsupported, match="gguf_unsupported_architecture"):
        hyperparameter_metadata(gemma, encoding="f32")
    blob, count = _metadata("f32")
    with pytest.raises(GGUFUnsupported, match="gguf_unsupported_architecture"):
        convert_snapshot_to_gguf(
            cas, manifest, encoding="f32", metadata=blob,
            metadata_count=count, config=gemma,
        )


def test_a_k_quant_encoding_is_refused_by_name(
    snapshot: tuple[LocalCAS, RepositoryManifest],
) -> None:
    """k-quants are `llama-quantize`'s, and it runs over this output."""

    with pytest.raises(GGUFUnsupported, match="gguf_unsupported_encoding"):
        hyperparameter_metadata(CONFIG, encoding="q4_k_m")


def test_the_remaining_gguf_hatch_is_named_and_is_the_only_one() -> None:

    convert = Path(__file__).resolve().parents[1] / "src" / "gen_worker" / "convert"

    native = (convert / "gguf_native.py").read_text()
    for forbidden in ("third_party_dir(", "run_process(", "import subprocess"):
        assert forbidden not in native, (
            f"gen_worker/convert/gguf_native.py reaches {forbidden!r}. The native "
            f"path composes through the store; a hatch there is the whole defect."
        )

    census = {
        "gguf_tools.py": "pgw#1344",
        "source.py": "pgw#1335",
        "ingest.py": "pgw#1335",
    }
    callers = {
        path.name
        for path in convert.glob("*.py")
        if "third_party_dir(" in path.read_text()
    }
    assert callers == set(census), (
        f"the conversion package's seam census moved: on disk {sorted(callers)}, "
        f"declared {sorted(census)}. Every projection in this package is owned "
        f"by an issue; an undeclared one is a new hatch, not a detail."
    )
    tools = (convert / "gguf_tools.py").read_text()
    assert "pgw#1344" in tools, (
        "the surviving hatch must name the issue that scopes it, so a reader "
        "meets the reason rather than the call"
    )


def test_a_two_dimensional_norm_weight_is_pinned_to_f32_too() -> None:
    """The half of llama.cpp's dtype rule a llama fixture cannot reach."""

    from gen_worker.convert.gguf_native import _target_dtype

    class _View:
        shape = (16, 16)

    view = cast("object", _View())
    assert _target_dtype("blk.0.attn_norm.weight", view, "F16") == "F32"  # type: ignore[arg-type]
    assert _target_dtype("blk.0.attn_v.weight", view, "F16") == "F16"  # type: ignore[arg-type]
