"""The torch GGML dequant kernels are BIT-EXACT against the `gguf` package."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")

from gen_worker.models import gguf_dequant

QTYPES = [
    ("BF16", 512),
    ("Q8_0", 512),
    ("Q5_1", 512),
    ("Q5_0", 512),
    ("Q4_1", 512),
    ("Q4_0", 512),
    ("Q6_K", 512),
    ("Q5_K", 512),
    ("Q4_K", 512),
    ("Q3_K", 512),
    ("Q2_K", 512),
    ("IQ4_NL", 512),
    ("IQ4_XS", 512),
]

QUANTIZABLE = {"BF16", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0"}

ROWS = 6


def _qtype(name: str) -> int:
    return int(getattr(gguf.GGMLQuantizationType, name))


def _row_bytes(name: str, cols: int) -> int:
    block_size, type_size = gguf_dequant.block_geometry(_qtype(name))
    assert cols % block_size == 0, f"{name}: {cols} not a multiple of {block_size}"
    return cols // block_size * type_size


def _ours(raw: np.ndarray, name: str, shape: tuple[int, ...]) -> np.ndarray:
    out = gguf_dequant.dequantize(
        torch.from_numpy(raw), _qtype(name), torch.Size(shape),
        dtype=torch.float32)
    assert out.dtype is torch.float32
    return out.numpy()


def _assert_bit_exact(ours: np.ndarray, ref: np.ndarray, name: str) -> None:
    ref = ref.astype(np.float32)
    assert ours.shape == ref.shape, f"{name}: shape {ours.shape} != {ref.shape}"
    ours_bits = ours.view(np.uint32)
    ref_bits = ref.view(np.uint32)
    finite = np.isfinite(ref)
    mismatch = (ours_bits != ref_bits) & finite
    if mismatch.any():
        i = np.argmax(mismatch.ravel())
        raise AssertionError(
            f"{name}: {int(mismatch.sum())}/{mismatch.size} finite values differ; "
            f"first at flat {i}: ours={ours.ravel()[i]!r} ref={ref.ravel()[i]!r}")
    assert np.array_equal(np.isnan(ours), np.isnan(ref)), f"{name}: NaN mask differs"
    assert np.array_equal(np.isinf(ours), np.isinf(ref)), f"{name}: Inf mask differs"


@pytest.mark.parametrize("name,cols", QTYPES)
def test_random_block_bytes_match_gguf_reference(name: str, cols: int) -> None:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    raw = rng.integers(0, 256, size=(ROWS, _row_bytes(name, cols)), dtype=np.uint8)

    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType[name])
    _assert_bit_exact(_ours(raw, name, (ROWS, cols)), ref, name)


@pytest.mark.parametrize("name,cols",
                         [(n, c) for n, c in QTYPES if n in QUANTIZABLE])
def test_round_tripped_weights_match_gguf_reference(name: str, cols: int) -> None:
    rng = np.random.default_rng(1498)
    values = (rng.standard_normal((ROWS, cols)) * 0.02).astype(np.float32)
    raw = gguf.quants.quantize(values, gguf.GGMLQuantizationType[name])

    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType[name])
    _assert_bit_exact(_ours(raw, name, (ROWS, cols)), ref, name)
    assert np.abs(ref - values).max() < 0.02


@pytest.mark.parametrize("name,cols", QTYPES)
def test_multi_dim_logical_shape_is_restored(name: str, cols: int) -> None:
    """Conv weights are 4-D; the block stream is flat."""
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, size=(4, _row_bytes(name, cols)), dtype=np.uint8)
    flat = _ours(raw, name, (4 * cols,))
    shaped = _ours(raw, name, (4, 2, cols // 2))
    assert shaped.shape == (4, 2, cols // 2)
    assert np.array_equal(shaped.reshape(-1).view(np.uint32),
                          flat.view(np.uint32))


def test_unsupported_qtype_refuses_loudly() -> None:
    raw = np.zeros((2, 128), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="IQ2_XXS"):
        gguf_dequant.dequantize(
            torch.from_numpy(raw), _qtype("IQ2_XXS"), torch.Size((2, 512)),
            dtype=torch.float32)


def test_supported_set_is_exactly_the_thirteen_plus_native() -> None:
    names = {gguf_dequant.qtype_name(q) for q in gguf_dequant.supported_qtypes()}
    assert names == {n for n, _ in QTYPES} | {"F32", "F16"}
    for name in ("IQ1_S", "IQ2_XXS", "IQ3_S"):
        assert not gguf_dequant.is_supported(_qtype(name))


@pytest.mark.parametrize("name", ["F32", "F16"])
def test_native_types_pass_through(name: str) -> None:
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    src = torch.from_numpy(values).to(
        torch.float32 if name == "F32" else torch.float16)
    out = gguf_dequant.dequantize(src, _qtype(name), torch.Size((8, 8)),
                                  dtype=torch.float32)
    assert torch.equal(out, torch.from_numpy(values))
