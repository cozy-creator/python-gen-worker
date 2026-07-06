"""convert_safetensors_to_flashpack packs non-float buffers (int64 position
ids, bool masks) alongside float weights and round-trips exactly (te#42)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flashpack")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from cozy_convert.flashpack import convert_safetensors_to_flashpack  # noqa: E402
from cozy_convert.writer import ConversionImplementationError  # noqa: E402


def _roundtrip(path):
    from flashpack.deserialization import (
        iterate_from_flash_tensor,
        read_flashpack_file,
    )

    storage, meta = read_flashpack_file(str(path))
    return dict(iterate_from_flash_tensor(storage, meta))


def test_pack_mixed_dtypes_roundtrip(tmp_path):
    tensors = {
        "unet.conv.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "te.embeddings.position_ids": torch.arange(77, dtype=torch.int64).unsqueeze(0),
        "te.mask": torch.tensor([[True, False], [False, True]]),
        "vae.scale": torch.randn(8, dtype=torch.float16),
    }
    src = tmp_path / "in.safetensors"
    save_file(tensors, str(src))
    out = tmp_path / "out.flashpack"

    result = convert_safetensors_to_flashpack(src, out, target_dtype="preserve")

    assert out.is_file() and result["output_path"] == str(out)
    got = _roundtrip(out)
    assert set(got) == set(tensors)
    for name, want in tensors.items():
        assert got[name].dtype == want.dtype, name
        assert torch.equal(got[name].cpu(), want), name


def test_pack_missing_input_is_deterministic_error(tmp_path):
    with pytest.raises(ConversionImplementationError, match="input_missing"):
        convert_safetensors_to_flashpack(
            tmp_path / "absent.safetensors", tmp_path / "out.flashpack",
            target_dtype="preserve",
        )


def test_pack_rejects_unwired_dtype_mode(tmp_path):
    with pytest.raises(ConversionImplementationError, match="dtype_mode_not_wired"):
        convert_safetensors_to_flashpack(
            tmp_path / "x.safetensors", tmp_path / "out.flashpack",
            target_dtype="bf16",
        )
